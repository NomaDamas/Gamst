"""Asynchronous perception runtime with bounded subscriptions and local memory."""
from __future__ import annotations

import asyncio
import inspect
import math
import time
from functools import wraps
import re
from collections import Counter
from copy import deepcopy

from .schema import envelope, hypothesis, timestamp, validate_output
from .storage import Store
from .plugins import PerceptionPlugin


def _window_ms(value):
    if isinstance(value, (int, float)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("window must be a positive finite duration")
        return int(value)
    if not isinstance(value, str):
        raise ValueError("window must be a duration string or milliseconds")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", value)
    if not match:
        raise ValueError("window must use ms, s, m, h, or d")
    duration = int(float(match[1]) * {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}[match[2]])
    if duration <= 0:
        raise ValueError("window must be positive")
    return duration


def _timed(stage):
    def decorate(method):
        @wraps(method)
        async def wrapped(self, *args, **kwargs):
            with self.latency.measure(stage):
                return await method(self, *args, **kwargs)
        return wrapped
    return decorate


class HumanSense:
    def __init__(self, *, consent=False, store=":memory:", retention_hours=24,
                 window_ms=30_000, queue_size=256, save_raw_media=False,
                 capabilities=None):
        if not isinstance(consent, bool):
            raise TypeError("consent must be an explicit boolean")
        if save_raw_media:
            raise ValueError("raw media storage is intentionally unsupported")
        if window_ms <= 0 or queue_size <= 0:
            raise ValueError("window_ms and queue_size must be positive")
        supported = {"conversation", "conversational", "interaction", "behavior", "consistency"}
        if capabilities is not None and (isinstance(capabilities, str) or set(capabilities)-supported):
            raise ValueError("supported capabilities: conversation, behavior, consistency")
        self.capabilities = tuple(capabilities or ("conversation", "behavior", "consistency"))
        from .metrics import LatencyMonitor
        self.latency = LatencyMonitor()
        self.consent = bool(consent)
        self.paused = False
        self.closed = False
        self.window_ms = int(window_ms)
        self.queue_size = int(queue_size)
        self.store = Store(store, retention_hours)
        self.prediction_definitions = {}
        self.analysis_backend = None
        self._analysis_results = {}
        self.plugins = []
        self._subscribers = set()
        self._last_ts = {}
        self._generation = 0
        self._metric_directions = {}
        self._speech_stops = {}
        self.dropped_records = 0
        self.plugin_errors = []

    async def __aenter__(self):
        if self.closed:
            raise RuntimeError("runtime is closed")
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _check(self):
        if self.closed:
            raise RuntimeError("runtime is closed")
        if not self.consent:
            raise PermissionError("explicit consent=True is required before ingesting input")
        if self.paused:
            raise RuntimeError("observation is paused")

    def _prune(self):
        now = timestamp()
        self.store.prune(now)
        cutoff = now-self.store.retention_ms
        self._speech_stops = {subject: value for subject,value in self._speech_stops.items() if value[2] >= cutoff}
        self._analysis_results = {subject: record for subject, record in self._analysis_results.items() if record["timestamp_ms"] >= cutoff}
        self._last_ts = {subject: ts for subject, ts in self._last_ts.items() if ts >= cutoff}
        self._metric_directions = {key: value for key, value in self._metric_directions.items() if value[1] >= cutoff}

    def _emit(self, record):
        validate_output(record)
        self.store.add(record)
        self._last_ts[record["subject_id"]] = max(record["timestamp_ms"], self._last_ts.get(record["subject_id"], 0))
        for queue, kinds in self._subscribers:
            if record["type"] not in kinds:
                continue
            if queue.full():
                queue.get_nowait()
                self.dropped_records += 1
            queue.put_nowait(deepcopy(record))
        return record

    def _record(self, kind, subject, ts, **kwargs):
        if ts > timestamp()+5000:
            raise ValueError("timestamp_ms exceeds wall clock by more than 5000ms; provide aligned Unix timestamps")
        self._prune()
        result = envelope(kind, subject, ts, window_ms=self.window_ms, **kwargs)
        if ts < self._last_ts.get(subject, 0):
            result["quality"]["flags"].append("out_of_order")
        return result

    @_timed("sdk.push_text")
    async def push_text(self, text, subject_id="local-user", timestamp_ms=None, role="user"):
        self._check()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        if role not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        ts = timestamp(timestamp_ms)
        normalized = " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())
        # Text is an explicitly provided input, unlike raw microphone recordings.
        signal = self._emit(self._record("observation.signal", subject_id, ts,
            observation={"name": "text.utterance", "value": text, "normalized": normalized, "role": role, "modality": "text"}))
        if role == "user":
            normalized = " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())
            history = self.store.query(subject_id, max(0, ts-self.window_ms), ts, "observation.signal", name="text.utterance")
            previous = [r for r in history if r["id"] != signal["id"] and r["observation"].get("name") == "text.utterance"]
            request = re.search(r"다시\s*(설명|말)|설명.*(해\s*줘|해주세요|해\s*주|해줄)|이해.*(안\s*돼|안\s*되|못)|무슨\s*(뜻|말)|잘\s*모르|\b(explain again|clarify|don't understand|do not understand|help me|repeat that)\b", text, re.I)
            if request:
                self._event("interaction.clarification_requested", subject_id, ts,
                            {"explicit_request": True}, [signal["id"]])
            repeated_count, repeated_ids = self.store.repeated_text(subject_id, max(0, ts-self.window_ms), ts, normalized)
            if normalized and repeated_count > 1:
                self._event("interaction.utterance_repeated", subject_id, ts,
                            {"count": repeated_count, "question_mark_present": "?" in text,
                             "evidence_truncated": repeated_count > len(repeated_ids)}, repeated_ids)
            assistant = [r for r in previous if r["observation"].get("role") == "assistant"]
            if assistant:
                anchor = assistant[-1]
                users_after = [r for r in previous if r["observation"].get("role") == "user" and r["timestamp_ms"] >= anchor["timestamp_ms"]]
                if not users_after:
                    self._event("interaction.response_received", subject_id, ts,
                        {"response_delay_ms": ts-anchor["timestamp_ms"], "anchor": "assistant_text_received", "interpretation": "timing_only"}, [anchor["id"], signal["id"]])
        return await self._infer(subject_id, ts)

    def _event(self, name, subject, ts, observation, evidence):
        sources = self.store.get_many(evidence)
        models = {(m["id"], m["version"]): m for record in sources for m in record["models"]}
        result = self._record("observation.event", subject, ts,
            observation=observation, evidence=[{"record_id": item} for item in evidence])
        for model in result["models"]:
            models[(model["id"], model["version"])] = model
        result["models"] = list(models.values())
        result["event"] = name
        result["name"] = name
        return self._emit(result)

    @_timed("sdk.push_signal")
    async def push_signal(self, name, value, modality, subject_id="local-user", timestamp_ms=None,
                          model_id="external", model_version="unspecified", quality=None):
        self._check()
        if modality not in ("video", "audio", "text"):
            raise ValueError("modality must be video, audio, or text")
        if not name or not model_id or not model_version:
            raise ValueError("signal name and model provenance are required")
        ts = timestamp(timestamp_ms)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("signal must be finite")
        if isinstance(quality, str):
            quality = {"status": quality, "flags": []}
        if quality is not None:
            if not isinstance(quality, dict) or quality.get("status") not in ("sufficient", "insufficient", "unavailable"):
                raise ValueError("quality status must be sufficient, insufficient, or unavailable")
            quality = {"status": quality["status"], "flags": list(quality.get("flags", []))}
        record = self._record("observation.signal", subject_id, ts,
            observation={"name": name, "value": value, "modality": modality},
            models=[{"id": model_id, "version": model_version, "backend": "plugin"}], quality=quality)
        self._emit(record)
        if name == "speech.activity" and isinstance(value, bool) and record["quality"]["status"] == "sufficient":
            activity = [r for r in self.store.query(subject_id, end=ts, kind="observation.signal", name=name)
                        if r["id"] != record["id"] and r["quality"]["status"] == "sufficient"]
            prior = activity[-1] if activity else None
            if value and (prior is None or prior["observation"]["value"] is False):
                self._speech_stops.pop(subject_id,None)
                self._event("speech.started", subject_id, ts, {"speaking": True}, [record["id"]])
            elif not value and prior and prior["observation"]["value"] is True:
                event = self._event("speech.stopped", subject_id, ts, {"speaking": False}, [prior["id"], record["id"]])
                self._speech_stops[subject_id] = (event["id"], time.perf_counter_ns(), ts)
        trends = await self.get_trends(subject_id, self.window_ms, end_ms=ts)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and record["quality"]["status"] == "sufficient":
            metric = next((m for m in trends["metrics"].values() if m["metric"] == name and m["model_id"] == model_id and m["model_version"] == model_version and m["modality"] == modality), {})
            change = metric.get("relative_change")
            direction = (1 if change > 0 else -1) if change is not None and abs(change) >= .2 else 0
            key = (subject_id, name, modality, model_id, model_version)
            previous_direction, previous_ts = self._metric_directions.get(key, (0, ts))
            current = ts >= previous_ts
            if current:
                self._metric_directions[key] = (direction, ts)
            if current and direction and direction != previous_direction:
                self._event("behavior.metric_changed", subject_id, ts,
                    {"metric": name, **metric, "detection_threshold": .2}, [record["id"]])
        await self._infer(subject_id, ts, trends=trends)
        return record

    @_timed("sdk.push_claim")
    async def push_claim(self, key, value, subject_id="local-user", timestamp_ms=None,
                         scope="session", source="user"):
        self._check()
        if not isinstance(key, str) or not key.strip() or not scope or not source:
            raise ValueError("key, scope, source are required")
        ts = timestamp(timestamp_ms)
        previous = self.store.query(subject_id, end=ts, kind="observation.signal", name="statement.claim", limit=100_000)
        matching = [r for r in previous if r["observation"].get("name") == "statement.claim" and r["observation"].get("key") == key and r["observation"].get("scope") == scope]
        record = self._emit(self._record("observation.signal", subject_id, ts,
            observation={"name": "statement.claim", "key": key, "value": value, "scope": scope,
                         "source": source, "modality": "text"}))
        if matching and matching[-1]["observation"]["value"] != value:
            self._event("statement.conflict_detected", subject_id, ts,
                {"key": key, "scope": scope, "previous_value": matching[-1]["observation"]["value"],
                 "current_value": value, "verification_status": "not_checked", "meaning": "structured_assertions_differ"},
                [matching[-1]["id"], record["id"]])
        return await self._infer(subject_id, ts)

    async def _infer(self, subject, ts, trends=None):
        signals = self.store.latest_signals(subject, max(0, ts-self.window_ms), ts)
        event_counts, events = self.store.event_summary(subject, max(0, ts-self.window_ms), ts, per_kind=8)
        modalities = {m: "available" if any(r["observation"].get("modality") == m for r in signals) else "unavailable" for m in ("video", "audio", "text")}
        clarification = [r for r in events if r.get("event") == "interaction.clarification_requested"]
        repeated = [r for r in events if r.get("event") == "interaction.utterance_repeated"]
        conflicts = [r for r in events if r.get("event") == "statement.conflict_detected"]
        interaction_evidence = clarification or repeated
        result = self._record("human.inference", subject, ts,
            evidence=[{"record_id": r["id"], "type": r.get("event", r["type"])} for r in events],
            quality={"status": "sufficient" if any(r["quality"].get("status") == "sufficient" for r in signals) else "insufficient",
                     "flags": sorted({flag for r in signals for flag in r["quality"].get("flags", [])})})
        models = {(m["id"], m["version"]): m for r in signals for m in r["models"]}
        for model in result["models"]:
            models[(model["id"], model["version"])] = model
        result["models"] = list(models.values())
        result["modalities"] = modalities
        result["event_summary"] = {"counts": event_counts, "total_count": sum(event_counts.values()),
            "evidence_count": len(events), "max_evidence_per_event_type": 8,
            "evidence_truncated": sum(event_counts.values()) > len(events)}
        result["predictions"] = {
            "conversation": {"status": "available" if modalities["text"] == "available" else "unavailable",
                "needs_clarification": hypothesis("user_requests_clarification", [r["id"] for r in interaction_evidence],
                    description="User explicitly requested clarification" if clarification else "Repeated utterance may merit a clarifying question",
                    status="observed" if clarification else "hypothesis" if repeated else "insufficient"),
                "repeated_utterance_count": max([r["observation"]["count"] for r in repeated], default=0),
                "expressions": {"status": "unavailable", "reason": "no_expression_model"}},
            "behavior": trends if trends is not None else await self.get_trends(subject_id=subject, window=self.window_ms, end_ms=ts),
            "consistency": {"status": "available" if any(r["observation"].get("name") == "statement.claim" for r in signals) else "unavailable",
                "contradiction_detected": bool(conflicts), "verification_status": "not_checked",
                "method": "explicit_structured_claim_comparison", "conflicts": [{"event_id": r["id"], **r["observation"], "evidence": r["evidence"]} for r in conflicts],
                "deception_assessed": False}}
        visual = next((r for r in signals if r["observation"].get("name") == "scene.model_description"), None)
        if visual is not None:
            value = visual["observation"].get("value", {})
            if isinstance(value, dict) and value.get("interpretation") == "unverified_model_hypotheses":
                result["predictions"]["conversation"]["expressions"] = {
                    "status": "model_hypotheses", "items": deepcopy(value.get("expressions", [])),
                    "evidence_ids": [visual["id"]], "probability": None,
                    "calibrated": False, "verified": False}
                result["predictions"]["behavior"]["visual_hypotheses"] = {
                    "observations": deepcopy(value.get("observations", [])),
                    "evidence_ids": [visual["id"]], "verified": False,
                    "interpretation": "model_interpretation_not_baseline_measurement"}
                result["evidence"].append({"record_id": visual["id"], "type": "visual_model_interpretation"})
        result["recommended_action"] = ({"type": "ask_clarifying_question", "requires_confirmation": False,
            "text": "핵심을 짧게 다시 설명해 드릴까요?", "evidence_ids": [r["id"] for r in interaction_evidence]}
            if interaction_evidence else None)
        cached = self._analysis_results.get(subject)
        if cached and cached["timestamp_ms"] <= ts <= cached["timestamp_ms"]+self.window_ms:
            source_ids = {item["record_id"] for item in cached["evidence"]}
            newer_turn = any(r["observation"].get("name") == "text.utterance" and r["id"] not in source_ids
                             for r in signals)
            analysis = deepcopy(cached["predictions"])
            provenance = {"record_id": cached["id"], "timestamp_ms": cached["timestamp_ms"],
                          "based_on_prior_turn": newer_turn, "verified": False}
            result["predictions"]["conversation"]["model_analysis"] = {**analysis["conversation"], **provenance}
            result["predictions"]["consistency"]["model_analysis"] = {**analysis["consistency"], **provenance}
            result["evidence"].append({"record_id": cached["id"], "type": "model_assisted_analysis"})
            result["models"] = self._models_for([cached], result["models"])
            if not newer_turn and result["recommended_action"] is None:
                result["recommended_action"] = deepcopy(cached.get("recommended_action"))
        if self.prediction_definitions:
            result["predictions"]["custom"] = self._custom_predictions(signals)
        return self._emit(result)

    def define_prediction(self, name, *, target, score_name, features=None, calibration=None,
                          scorer_model_id=None, scorer_model_version=None):
        """Register a measurable target and upstream score; never train from a prompt."""
        if not all(isinstance(v, str) and v.strip() for v in (name, target, score_name)):
            raise ValueError("name, target and score_name must be non-empty strings")
        if calibration is not None:
            if calibration.target != target:
                raise ValueError("calibration target does not match prediction target")
            scorer_model_id = calibration.scorer_model_id
            scorer_model_version = calibration.scorer_model_version
        self.prediction_definitions[name] = {
            "target": target, "score_name": score_name, "features": list(features or []),
            "calibration": calibration, "model_id": scorer_model_id, "model_version": scorer_model_version}
        return self

    def _custom_predictions(self, signals):
        output = {}
        for name, definition in self.prediction_definitions.items():
            candidates = [r for r in signals if r["observation"].get("name") == definition["score_name"]
                          and r["quality"].get("status") == "sufficient"
                          and (definition["model_id"] is None or r["models"][0]["id"] == definition["model_id"])
                          and (definition["model_version"] is None or r["models"][0]["version"] == definition["model_version"])]
            prediction = hypothesis(definition["target"], [], description="Registered prediction lacks a valid upstream score", status="unavailable")
            if candidates:
                record = candidates[-1]
                score = record["observation"].get("value")
                if type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1:
                    prediction.update(status="uncalibrated", model_score=score, evidence_ids=[record["id"]],
                                      scorer={"id": record["models"][0]["id"], "version": record["models"][0]["version"]})
                    if definition["calibration"] is not None:
                        prediction.update(definition["calibration"].predict(score))
            output[name] = prediction
        return output

    async def record_feedback(self, inference_id, outcome, *, subject_id="local-user", prediction=None):
        self._check()
        if type(outcome) is not bool:
            raise ValueError("outcome must be a boolean observed result")
        records = self.store.get_many([inference_id])
        if not records or records[0]["subject_id"] != subject_id or records[0]["type"] != "human.inference":
            raise ValueError("inference must be a retained record for this subject")
        source = records[0]
        selected = source["predictions"].get("custom", {}).get(prediction) if prediction else None
        if prediction is not None and selected is None:
            raise ValueError("prediction not found in inference")
        return self._event("interaction.feedback_received", subject_id, timestamp(),
            {"outcome": outcome, "inference_id": inference_id, "prediction": prediction,
             "prediction_snapshot": deepcopy(selected), "source": "explicit_application_feedback"}, [inference_id])

    def set_analysis_backend(self, backend):
        if not callable(getattr(backend, "analyze", None)):
            raise TypeError("backend must implement analyze(records)")
        self.analysis_backend = backend
        return self

    @_timed("model.analysis")
    async def analyze(self, subject_id="local-user", window="5m"):
        """Explicit model pass over retained text/measurements; does not block capture scheduling."""
        self._check()
        if self.analysis_backend is None:
            raise RuntimeError("No reasoning backend configured")
        self._prune()
        now = timestamp()
        start = now-_window_ms(window)
        # Keep conversation turns when high-rate scalar signals would otherwise
        # fill the history tail. Send only latest scalar measurements per metric.
        texts = []
        for text_name in ("text.utterance", "text.transcript", "speech.transcript"):
            texts.extend(self.store.query(subject_id, start, now, kind="observation.signal", limit=80, name=text_name))
        latest = self.store.latest_signals(subject_id, start, now)
        records = sorted({r["id"]: r for r in texts+latest}.values(), key=lambda r: r["timestamp_ms"])[-100:]
        generation = self._generation
        result = await self.analysis_backend.analyze(deepcopy(records))
        if generation != self._generation or self.paused or self.closed:
            raise RuntimeError("Analysis discarded after pause/deletion/close")
        # The backend must verify all evidence references and keep hypotheses uncalibrated.
        from .reasoning import validate_analysis
        result = validate_analysis(result, records)
        output = self._record("human.inference", subject_id, now,
            observation={"method": "model_assisted_analysis"},
            models=[{"id": self.analysis_backend.model_id, "version": self.analysis_backend.model_version,
                     "backend": "reasoning"}],
            evidence=[{"record_id": r["id"]} for r in records])
        output["models"] = self._models_for(records, output["models"])
        output["predictions"] = result["predictions"]
        output["predictions"]["behavior"] = await self.get_trends(subject_id, window)
        output["recommended_action"] = result.get("recommended_action")
        output["modalities"] = {m: "available" if any(r["observation"].get("modality") == m for r in records) else "unavailable" for m in ("audio", "video", "text")}
        emitted = self._emit(output)
        self._analysis_results[subject_id] = deepcopy(emitted)
        return emitted

    def register(self, plugin):
        if not callable(getattr(plugin, "process", None)):
            raise TypeError("plugin must implement process(InputFrame)")
        self.plugins.append(plugin)
        return self

    async def _push_media(self, data, modality, subject_id, timestamp_ms, sample_rate=None):
        self._check()
        generation = self._generation
        from .plugins import InputFrame
        ts = timestamp(timestamp_ms)
        if ts > timestamp()+5000:
            raise ValueError("timestamp_ms exceeds wall clock by more than 5000ms")
        frame = InputFrame(data=data, modality=modality, timestamp_ms=ts, subject_id=subject_id, sample_rate=sample_rate)
        results = []
        for plugin in self.plugins:
            modalities = getattr(plugin, "modalities", (getattr(plugin, "modality", modality),))
            if modality not in modalities:
                continue
            try:
                with self.latency.measure("plugin."+str(getattr(plugin, "name", type(plugin).__name__))):
                    output = plugin.process(frame)
                    if inspect.isawaitable(output):
                        output = await output
                if generation != self._generation or self.paused or self.closed:
                    return []
                for signal in output or []:
                    signal = dict(signal)
                    signal.setdefault("subject_id", subject_id)
                    signal.setdefault("timestamp_ms", ts)
                    signal.setdefault("modality", modality)
                    signal.setdefault("model_id", getattr(plugin, "name", type(plugin).__name__))
                    signal.setdefault("model_version", getattr(plugin, "version", "unspecified"))
                    if signal.pop("type", None) == "text" or signal.get("name") in ("text.transcript", "speech.transcript"):
                        await self.push_signal(**signal)
                        results.append(await self.push_text(str(signal["value"]), subject_id=signal["subject_id"], timestamp_ms=signal["timestamp_ms"]))
                    else:
                        results.append(await self.push_signal(**signal))
            except Exception as exc:
                self.plugin_errors.append({"plugin": getattr(plugin, "name", type(plugin).__name__), "error": str(exc), "timestamp_ms": ts})
                self.plugin_errors = self.plugin_errors[-100:]
                raise
        return results

    @_timed("sdk.push_video")
    async def push_video(self, frame, subject_id="local-user", timestamp_ms=None):
        return await self._push_media(frame, "video", subject_id, timestamp_ms)

    @_timed("sdk.push_audio")
    async def push_audio(self, samples, subject_id="local-user", timestamp_ms=None, sample_rate=16000):
        prior_stop = self._speech_stops.get(subject_id)
        records = await self._push_media(samples, "audio", subject_id, timestamp_ms, sample_rate)
        stop = self._speech_stops.get(subject_id)
        if stop is not None and stop != prior_stop:
            # Process this entire audio block before flushing: ASR may follow VAD
            # in the plugin list, and needs the just-ended speech samples first.
            records.extend(await self.flush_audio(subject_id=subject_id))
        return records

    @_timed("sdk.flush_audio")
    async def flush_audio(self, subject_id="local-user", timestamp_ms=None):
        """Finalize residual ASR windows; no artificial transcript on silence."""
        self._check()
        generation = self._generation
        records = []
        for plugin in self.plugins:
            flush = getattr(plugin, "flush", None)
            if flush is None or "audio" not in getattr(plugin, "modalities", ()):
                continue
            with self.latency.measure("plugin."+str(getattr(plugin,"name",type(plugin).__name__))+".flush"):
                output = flush(subject_id=subject_id, timestamp_ms=timestamp_ms)
                if inspect.isawaitable(output):
                    output = await output
            if generation != self._generation or self.paused or self.closed:
                return []
            for item in output or []:
                signal = dict(item)
                signal.setdefault("modality", "audio")
                signal.setdefault("subject_id", subject_id)
                signal.setdefault("model_id", getattr(plugin,"name",type(plugin).__name__))
                signal.setdefault("model_version", getattr(plugin,"version","unspecified"))
                measured = await self.push_signal(**signal)
                records.append(measured)
                if signal.get("name") in ("text.transcript", "speech.transcript"):
                    records.append(await self.push_text(str(signal["value"]),subject_id=subject_id,
                                                       timestamp_ms=signal.get("timestamp_ms",timestamp_ms)))
                    stop = self._speech_stops.get(subject_id)
                    if stop is not None:
                        self.latency.record("pipeline.speech_stop_decision_to_flushed_transcript",
                                            (time.perf_counter_ns()-stop[1])/1_000_000)
                        self._speech_stops.pop(subject_id,None)
        return records

    async def _stream(self, kinds):
        if self.closed:
            return
        queue = asyncio.Queue(self.queue_size)
        subscriber = (queue, frozenset(kinds))
        self._subscribers.add(subscriber)
        try:
            while True:
                record = await queue.get()
                if record is None:
                    break
                yield record
        finally:
            self._subscribers.discard(subscriber)

    def observe(self):
        return self._stream(("human.inference",))

    def events(self):
        return self._stream(("observation.event",))

    def signals(self):
        return self._stream(("observation.signal",))

    async def get_metrics(self):
        return {"latency": self.latency.snapshot(), "dropped_records": self.dropped_records,
                "clock": "perf_counter_monotonic", "interpretation": "runtime_measurements_not_model_accuracy"}

    async def get_history(self, subject_id="local-user", hours=24, limit=100, start_ms=None, end_ms=None, type=None):
        self._prune()
        return self.store.query(subject_id, start_ms if start_ms is not None else timestamp()-int(hours*3_600_000), end_ms, type, limit)

    @_timed("sdk.get_state")
    async def get_state(self, subject_id="local-user"):
        self._prune()
        now = timestamp()
        records = self.store.latest_signals(subject_id, max(0, now-self.window_ms), now)
        latest = {}
        for record in records:
            latest[record["observation"]["name"]] = record
        result = envelope("observation.state", subject_id, now, window_ms=self.window_ms)
        result["models"] = self._models_for(records, result["models"])
        result.update({"paused": self.paused, "consent": self.consent,
            "modalities": {m: "available" if any(r["observation"].get("modality") == m for r in records) else "unavailable" for m in ("video", "audio", "text")},
            "signals": latest, "dropped_records": self.dropped_records, "plugin_errors": deepcopy(self.plugin_errors)})
        return result

    @staticmethod
    def _models_for(records, defaults):
        models = {(m["id"], m["version"]): m for record in records for m in record["models"]}
        for model in defaults:
            models[(model["id"], model["version"])] = model
        return list(models.values())

    async def get_trends(self, subject_id="local-user", window="1h", end_ms=None):
        self._prune()
        end = timestamp(end_ms)
        duration = _window_ms(window)
        rows = self.store.numeric_summary(subject_id, end-duration, end)
        metrics = {}
        names = Counter(row[0] for row in rows)
        for name, modality, model_id, model_version, count, mean, prior_count, prior_mean, prior_min, prior_max in rows:
            available = prior_count >= 5
            base = prior_mean if available else None
            label = name if names[name] == 1 else f"{name}@{model_id}:{model_version}"
            metrics[label] = {"metric": name, "modality": modality, "model_id": model_id, "model_version": model_version,
                "current_mean": mean, "sample_count": count, "baseline_available": available,
                "baseline_mean": base, "baseline_sample_count": prior_count, "baseline_duration_ms": prior_max-prior_min if prior_count else 0,
                "method": "arithmetic_mean_previous_retained_samples", "relative_change": (mean-base)/abs(base) if base else None}
        return {"status": "available" if metrics else "insufficient", "metrics": metrics,
                "window_ms": duration, "interpretation": "observed_patterns_only"}

    @_timed("sdk.get_context")
    async def get_context(self, subject_id="local-user", window="1h", include=None):
        self._prune()
        now = timestamp()
        start = now-_window_ms(window)
        counts, events = self.store.event_summary(subject_id, start, now, per_kind=8)
        total_events = sum(counts.values())
        records = self.store.latest_signals(subject_id, start, now)+events
        result = envelope("observation.context", subject_id, now, window_ms=_window_ms(window),
            evidence=[{"record_id": r["id"]} for r in events])
        result["models"] = self._models_for(records, result["models"])
        result.update({"summary": {"event_counts": counts, "event_count": total_events,
                                   "evidence_count": len(events), "max_evidence_per_event_type": 8,
                                   "evidence_truncated": total_events > len(events),
                                   "record_count": self.store.count(subject_id, start, now), "method": "deterministic_aggregate"},
                       "trends": await self.get_trends(subject_id, window),
                       "include": include or ["speech", "movement", "interaction"]})
        return result

    async def summarize(self, **kwargs):
        return await self.get_context(**kwargs)

    async def inspect_interval(self, start_ms, end_ms, subject_id="local-user"):
        if start_ms > end_ms:
            raise ValueError("start_ms must not exceed end_ms")
        return {"subject_id": subject_id, "start_ms": start_ms, "end_ms": end_ms,
                "records": await self.get_history(subject_id, start_ms=start_ms, end_ms=end_ms, limit=100_000)}

    async def pause(self):
        self.paused = True
        self._analysis_results.clear()
        self._speech_stops.clear()
        self._generation += 1
        await self._reset_plugins()
        return {"paused": True}

    async def resume(self):
        if not self.consent:
            raise PermissionError("explicit consent required")
        if self.closed:
            raise RuntimeError("runtime is closed")
        self.paused = False
        return {"paused": False}

    async def delete_data(self, subject_id=None):
        self._generation += 1
        deleted = self.store.delete(subject_id)
        await self._reset_plugins()
        self.latency.reset()
        # Imported aggregate artifacts cannot be untrained for one subject. Drop
        # active calibration for every deletion rather than retain unknown contributions.
        for definition in self.prediction_definitions.values():
            definition["calibration"] = None
        if subject_id is None:
            self._analysis_results.clear()
            self._speech_stops.clear()
            self._last_ts.clear()
            self._metric_directions.clear()
        else:
            self._speech_stops.pop(subject_id,None)
            self._analysis_results.pop(subject_id, None)
            self._last_ts.pop(subject_id, None)
            self._metric_directions = {key: value for key, value in self._metric_directions.items() if key[0] != subject_id}
        # Pending subscriber records are derived data as well.
        for queue, _ in self._subscribers:
            retained = []
            while not queue.empty():
                record = queue.get_nowait()
                if record is None or subject_id is not None and record["subject_id"] != subject_id:
                    retained.append(record)
            for record in retained:
                queue.put_nowait(record)
        return {"deleted_records": deleted, "subject_id": subject_id}

    async def _reset_plugins(self):
        for plugin in self.plugins:
            reset = getattr(plugin, "reset", None)
            if reset:
                result = reset()
                if inspect.isawaitable(result):
                    await result

    async def reset_processing(self):
        """Discard pending media results and model buffers without pausing text."""
        self._generation += 1
        self._speech_stops.clear()
        await self._reset_plugins()
        return {"processing_reset": True}

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self._generation += 1
        for queue, _ in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)
        errors = []
        try:
            for plugin in self.plugins:
                close = getattr(plugin, "close", None)
                if close:
                    try:
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        errors.append(exc)
        finally:
            self.store.close()
        if errors:
            raise RuntimeError(f"{len(errors)} plugin cleanup operation(s) failed") from errors[0]
