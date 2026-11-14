"""Asynchronous perception runtime with bounded subscriptions and local memory."""
from __future__ import annotations

import asyncio
import inspect
import math
import re
import uuid
from collections import Counter
from copy import deepcopy

from .schema import envelope, hypothesis, timestamp, validate_output
from .storage import Store
from .plugins import PerceptionPlugin
from .capabilities import CapabilityRegistry, SensorMetadata, identifier
from .memory import ObjectMemory
from .transcripts import TranscriptMemory, TRANSCRIPT_OUTPUT
from dataclasses import asdict


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


class HumanSense:
    def __init__(self, *, consent=False, store=":memory:", retention_hours=24,
                 window_ms=30_000, queue_size=256, save_raw_media=False,
                 capabilities=None, sensor_inference_interval_ms=0,
                 max_records=5000, max_payload_bytes=32*1024*1024, evaluation_observer=None):
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
        self._evaluation_observers=[]
        if evaluation_observer is not None:self._register_evaluation_observer(evaluation_observer)
        self.consent = bool(consent)
        self.paused = False
        self.closed = False
        self.window_ms = int(window_ms)
        self.queue_size = int(queue_size)
        self.store = Store(store, retention_hours, max_records=max_records, max_payload_bytes=max_payload_bytes)
        self.prediction_definitions = {}
        self.analysis_backend = None
        self._analysis_results = {}
        self.plugins = []
        self.perception_capabilities = CapabilityRegistry()
        self.object_memory = ObjectMemory()
        self.transcript_memory = TranscriptMemory(retention_ms=min(86400000, self.store.retention_ms))
        self._subscribers = set()
        self._last_ts = {}
        self._generation = 0
        self._session_id = uuid.uuid4().hex
        self._metric_directions = {}
        self.dropped_records = 0
        self.plugin_errors = []
        if not isinstance(sensor_inference_interval_ms, (int, float)) or not 0 <= sensor_inference_interval_ms <= 5000:
            raise ValueError('sensor_inference_interval_ms must be between 0 and 5000')
        self.sensor_inference_interval_ms = sensor_inference_interval_ms
        self._sensor_refresh = {}
        self._call_controllers = {}
        self._separation_media = None

    def _register_evaluation_observer(self, observer):
        if not callable(getattr(observer,"invalidate",None)):
            raise TypeError("evaluation observer requires callable invalidate")
        if any(value is observer for value in self._evaluation_observers):return
        if len(self._evaluation_observers)>=32:
            raise ValueError("at most 32 distinct evaluation observers per runtime")
        self._evaluation_observers.append(observer)

    def _invalidate_evaluation(self, reason, subject_id=None):
        # Privacy cleanup must run even when the external accounting sink fails.
        for observer in self._evaluation_observers:
            try:observer.invalidate(reason, subject_id=subject_id)
            except BaseException:pass

    def enable_separation_media(self, *, subject_id='local-user', source_id, sensor_id,
                                ttl_ms=30000, capacity=8, max_bytes=16*1024*1024):
        """Grant an explicit local media owner. No public API/MCP audio exposure.

        Call within this runtime's event loop. Retention settings are shared by
        all grants and cannot silently change when another stream is enabled.
        """
        self._check()
        from .separation_media import SeparationMediaStore
        if any(type(value) is not int for value in (ttl_ms, capacity, max_bytes)):
            raise ValueError('separation retention settings require integer bounds')
        if self._separation_media is None:
            self._separation_media = SeparationMediaStore(self, ttl_ms=ttl_ms,
                capacity=capacity, max_bytes=max_bytes)
        store = self._separation_media
        if (ttl_ms, capacity, max_bytes) != (store.ttl_ms, store.capacity, store.max_bytes):
            raise ValueError('separation retention settings differ from existing runtime owner')
        return store.grant(subject_id, source_id, sensor_id)

    def _clear_separation_media(self):
        if self._separation_media is not None:
            self._separation_media.clear()

    @property
    def consent(self):
        return self._consent

    @consent.setter
    def consent(self, value):
        if type(value) is not bool:
            raise TypeError('consent must be an explicit boolean')
        prior = getattr(self, '_consent', False)
        self._consent = value
        if prior and not value:
            self._invalidate_evaluation("consent_revoked")
            self._generation += 1
            self._clear_separation_media()
            for controller in getattr(self, '_call_controllers', {}).values():
                controller.reset_sources(now_ms=timestamp())

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
        self._analysis_results = {subject: record for subject, record in self._analysis_results.items() if record["timestamp_ms"] >= cutoff}
        self._last_ts = {subject: ts for subject, ts in self._last_ts.items() if ts >= cutoff}
        self._metric_directions = {key: value for key, value in self._metric_directions.items() if value[1] >= cutoff}

    def _emit(self, record):
        validate_output(record)
        transcript = None
        if record.get("observation", {}).get("name") == TRANSCRIPT_OUTPUT:
            transcript = self.transcript_memory.prepare(record)
        stored = self.store.add(record)
        if transcript is not None and not stored:
            raise ValueError('transcript record exceeds storage capacity')
        if transcript is not None:
            self.transcript_memory.commit(transcript)
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

    def register_capability(self, spec):
        """Register metadata without loading a provider or activating a sensor."""
        if self.closed:
            raise RuntimeError('runtime is closed')
        self.perception_capabilities.register(spec)

    async def push_observation(self, capability_id, value, *, sensor,
                               subject_id='local-user', timestamp_ms=None, quality=None):
        spec = self.perception_capabilities.get(capability_id)
        return await self.push_signal(spec.output_name, value, spec.modality,
            subject_id=subject_id, timestamp_ms=timestamp_ms, quality=quality,
            model_id=spec.provider, model_version=spec.version,
            capability_id=capability_id, sensor=sensor)

    async def push_signal(self, name, value, modality, subject_id="local-user", timestamp_ms=None,
                          model_id="external", model_version="unspecified", quality=None,
                          capability_id=None, sensor=None):
        self._check()
        contract = None
        if capability_id is not None:
            spec = self.perception_capabilities.get(capability_id)
            if (name, modality, model_id, model_version) != (spec.output_name, spec.modality, spec.provider, spec.version):
                raise ValueError('signal provenance must match registered capability')
            if not isinstance(sensor, SensorMetadata):
                raise ValueError('registered observations require SensorMetadata')
            value = spec.validate(value)
            contract = dict(id=spec.id, streaming_mode=spec.streaming_mode,
                window_ms=spec.window_ms, lookahead_ms=spec.lookahead_ms,
                freshness_ms=spec.freshness_ms, interpretation=spec.interpretation,
                units={f.name: f.unit for f in spec.fields if f.unit is not None})
        elif sensor is not None:
            raise ValueError('sensor metadata requires a registered capability')
        elif modality not in ("video", "audio", "text"):
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
        observation = {"name": name, "value": value, "modality": modality}
        if contract is not None:
            sensor_data = asdict(sensor)
            sensor_data['channels'] = list(sensor.channels)
            observation.update(capability=contract, sensor=sensor_data,
                               processing_epoch=f'{self._session_id}:{self._generation}')
        record = self._record("observation.signal", subject_id, ts,
            observation=observation,
            models=[{"id": model_id, "version": model_version, "backend": "plugin"}], quality=quality)
        self._emit(record)
        if name == "speech.activity" and isinstance(value, bool) and record["quality"]["status"] == "sufficient":
            activity = [r for r in self.store.query(subject_id, end=ts, kind="observation.signal", name=name)
                        if r["id"] != record["id"] and r["quality"]["status"] == "sufficient"]
            prior = activity[-1] if activity else None
            if value and (prior is None or prior["observation"]["value"] is False):
                self._event("speech.started", subject_id, ts, {"speaking": True}, [record["id"]])
            elif not value and prior and prior["observation"]["value"] is True:
                self._event("speech.stopped", subject_id, ts, {"speaking": False}, [prior["id"], record["id"]])
        # Preserve every signal and speech transition. Full SQL aggregation and
        # inference are sampled for live sensor streams, never for text input.
        now = timestamp()
        if self.sensor_inference_interval_ms and now - self._sensor_refresh.get(subject_id, 0) < self.sensor_inference_interval_ms:
            return record
        self._sensor_refresh[subject_id] = now
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
            if not getattr(plugin, "runtime_enabled", True) or modality not in modalities:
                continue
            try:
                plugin_generation = getattr(plugin, "runtime_generation", 0)
                output = plugin.process(frame)
                if inspect.isawaitable(output):
                    output = await output
                if generation != self._generation or self.paused or self.closed:
                    return []
                if not getattr(plugin, "runtime_enabled", True) or plugin_generation != getattr(plugin, "runtime_generation", 0):
                    continue
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

    async def ingest_plugin_output(self, output, frame, plugin):
        """Commit output from a bounded background perception worker."""
        for signal in output or []:
            signal = dict(signal)
            signal.setdefault('subject_id', frame.subject_id)
            signal.setdefault('timestamp_ms', frame.timestamp_ms)
            signal.setdefault('modality', frame.modality)
            signal.setdefault('model_id', plugin.name)
            signal.setdefault('model_version', getattr(plugin, 'version', 'unspecified'))
            is_text = signal.pop('type', None) == 'text' or signal.get('name') in ('text.transcript', 'speech.transcript')
            await self.push_signal(**signal)
            if is_text:
                await self.push_text(str(signal['value']), subject_id=signal['subject_id'], timestamp_ms=signal['timestamp_ms'])

    async def push_video(self, frame, subject_id="local-user", timestamp_ms=None):
        return await self._push_media(frame, "video", subject_id, timestamp_ms)

    async def push_audio(self, samples, subject_id="local-user", timestamp_ms=None, sample_rate=16000):
        return await self._push_media(samples, "audio", subject_id, timestamp_ms, sample_rate)

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

    async def contexts(self, subject_id="local-user", window="5m", interval=.25):
        """Bounded-cadence derived context snapshots; never raw media."""
        if type(interval) not in (int, float) or not math.isfinite(interval) or not .05 <= interval <= 60:
            raise ValueError("context interval must be .05..60 seconds")
        while not self.closed:
            yield await self.get_context(subject_id=subject_id, window=window)
            await asyncio.sleep(interval)

    def observe(self):
        return self._stream(("human.inference",))

    def events(self):
        return self._stream(("observation.event",))

    def signals(self):
        return self._stream(("observation.signal",))

    async def get_history(self, subject_id="local-user", hours=24, limit=100, start_ms=None, end_ms=None, type=None):
        self._prune()
        return self.store.query(subject_id, start_ms if start_ms is not None else timestamp()-int(hours*3_600_000), end_ms, type, limit)

    async def list_capabilities(self, offset=0, limit=100, max_bytes=65536, revision=None):
        """Page registered contracts without loading providers; revision pins pagination."""
        import hashlib
        import json
        if self.closed:
            raise RuntimeError('runtime is closed')
        self._bounded_query([], limit, max_bytes)
        if type(offset) is not int or not 0 <= offset <= 256:
            raise ValueError('offset must be 0..256')
        if revision is not None and (not isinstance(revision, str) or len(revision) != 64):
            raise ValueError('revision must be a SHA256 registry revision')
        contracts = sorted(self.perception_capabilities.describe(), key=lambda item: item['id'])
        digest = hashlib.sha256(json.dumps(contracts, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        if revision is not None and revision != digest:
            raise ValueError('registry changed; restart pagination at offset 0')
        if offset and revision is None:
            raise ValueError('revision is required for continuation pages')
        if offset > len(contracts):
            raise ValueError('offset exceeds registry size')
        page = self._bounded_query(contracts[offset:], limit, max_bytes)
        if not page['items'] and offset < len(contracts):
            required = len(json.dumps(contracts[offset], ensure_ascii=True).encode())
            raise ValueError(f'next contract requires max_bytes >= {required}')
        next_offset = offset + page['returned_count'] if page['truncated'] else None
        return {'capabilities': page['items'], 'revision': digest, 'next_offset': next_offset,
                'total_count': len(contracts), 'returned_count': page['returned_count'],
                'truncated': page['truncated'], 'entry_byte_budget': max_bytes,
                'meaning': 'registered_contracts_only', 'activation': 'explicit_application_setup'}

    def register_call_controller(self, controller_id, controller):
        """Bind a fresh opt-in controller; never starts external calls."""
        from .call_controller import CallController
        self._check(); identifier(controller_id)
        if type(controller) is not CallController:
            raise TypeError('CallController required')
        if controller_id in self._call_controllers or len(self._call_controllers) >= 8:
            raise ValueError('duplicate controller ID or controller capacity')
        def validate_record(record):
            observation = record.get('observation', {})
            if observation.get('processing_epoch') != f'{self._session_id}:{self._generation}':
                raise ValueError('observation is from an obsolete or foreign runtime epoch')
            spec = self.perception_capabilities.get(observation.get('capability', {}).get('id'))
            if record.get('models') != [{'id': spec.provider, 'version': spec.version, 'backend': 'plugin'}]:
                raise ValueError('observation provenance differs from registered capability')
        controller.bind_runtime(guard=self._check, clock=timestamp, validate_record=validate_record,
                                epoch=lambda: f'{self._session_id}:{self._generation}')
        self._call_controllers[controller_id] = controller

    async def get_call_status(self, controller_id=None, limit=100, max_bytes=65536):
        """Read-only bounded diagnostics; totals describe whole controller scope."""
        self._bounded_query([], limit, max_bytes)
        if controller_id is not None:
            identifier(controller_id)
            if controller_id not in self._call_controllers:
                raise ValueError('unknown call controller')
        rows = []
        for name, controller in sorted(self._call_controllers.items()):
            if controller_id is not None and name != controller_id:
                continue
            state = controller.snapshot()
            budget = {k:v for k,v in state['budget'].items() if k not in ('active', 'recent')}
            rows.append(dict(kind='controller', controller_id=name, closed=state['closed'],
                source_count=len(state['sources']), request_count=len(state['requests']),
                detached_financial_count=len(state['detached_financial_requests']),
                overflow_rejections=state['overflow_rejections'], omitted_receipts=state['omitted_receipts'],
                budget=budget, admission_denials=state['admission_denials'],
                cadence_scope='independent_sensor_and_context_delivery; see get_sensor_status/get_context',
                accounting_scope='whole_controller_all_subjects'))
            for request in state['requests']:
                decision = request['decision']
                rows.append(dict(kind='request', controller_id=name, request_id=request['id'],
                    source=request['source'], state=request['state'], priority=request['priority'],
                    desired_route=decision['desired_route'], reason=decision['reason'], blockers=request['blockers'],
                    evidence=decision['evidence'], expires_ms=request['expires_ms'],
                    finished=request['finished'], charged=request['charged'], discard=request['discard'],
                    allowed_route=request['allowed_route'], admission_blockers=request['admission_blockers'],
                    admission_advisory=True, dispatched_route=request['dispatched_route'],
                    last_admission=request['last_admission'], execution=request['execution'],
                    semantic_audit_due=request['semantic_audit_due'],
                    last_semantic_capture_ms=request['last_semantic_capture_ms'],
                    unknown_candidate_unresolved=request['unknown_candidate_unresolved']))
            for token in state['detached_financial_requests']:
                rows.append(dict(kind='detached_liability', controller_id=name, request_id=token,
                    meaning='sensory_associations_deleted_billing_unresolved'))
            for receipt in reversed(state['recent']):
                rows.append(dict(kind='receipt', controller_id=name, request_id=receipt['id'],
                    reason=receipt['reason'], trigger_reason=receipt['trigger_reason'],
                    unknown_candidate_unresolved=receipt['unknown_candidate_unresolved'], evidence=receipt['evidence'],
                    last_admission=receipt['last_admission'], execution=receipt['execution']))
            for handle, source in state['sources'].items():
                rows.append(dict(kind='source', controller_id=name, source=handle,
                    identity=list(source['base']), owner_revision=source['owner'],
                    last_dispatch_ms=source['last_dispatch'], last_semantic_capture_ms=source['last_semantic'],
                    modality=source['modality'], semantic_audit_due=source['semantic_audit_due']))
        page = self._bounded_query(rows, limit, max_bytes)
        return dict(entries=page['items'], total_count=len(rows), returned_count=page['returned_count'],
            omitted_count=len(rows)-page['returned_count'], truncated=page['truncated'], entry_byte_budget=max_bytes,
            paused=self.paused, consent=self.consent, closed=self.closed,
            executes_model=False, changes_sensor_cadence=False, quality_preservation_proven=False)

    async def get_sensor_status(self):
        if self.closed:
            raise RuntimeError('runtime is closed')
        return {'managers': [manager.status() for manager in getattr(self, '_sensor_streams', ())],
                'paused': self.paused, 'consent': self.consent}

    async def finish_sensor_stream(self, manager_id, source_id, sensor_id):
        """Finalize an explicitly selected existing typed stream; never opens devices."""
        self._check()
        identifier(manager_id); identifier(source_id); identifier(sensor_id)
        manager = next((m for m in getattr(self, '_sensor_streams', ())
                        if m.manager_id == manager_id), None)
        if manager is None:
            raise ValueError('unknown sensor manager')
        return await manager.finish(source_id, sensor_id)

    async def endpoint_sensor_stream(self, manager_id, source_id, sensor_id, capability_id,
                                     stream_epoch, after_sequence=None):
        """Finalize an utterance at an accepted-input barrier without closing capture."""
        self._check()
        identifier(manager_id)
        manager = next((m for m in getattr(self, '_sensor_streams', ())
                        if m.manager_id == manager_id), None)
        if manager is None:
            raise ValueError('unknown sensor manager')
        return await manager.endpoint(source_id, sensor_id, capability_id,
            stream_epoch=stream_epoch, after_sequence=after_sequence)

    @staticmethod
    def _bounded_query(items, limit, max_bytes):
        import json
        if type(limit) is not int or not 1 <= limit <= 100 or type(max_bytes) is not int or not 1024 <= max_bytes <= 262144:
            raise ValueError('limit must be 1..100 and max_bytes 1024..262144')
        result, size = [], 0
        # Input is newest first. Keep whole records, never truncate evidence text.
        for item in items:
            cost = len(json.dumps(item, ensure_ascii=True).encode())
            if len(result) >= limit or size+cost > max_bytes:
                break
            result.append(item)
            size += cost
        return {'items': result, 'returned_count': len(result), 'truncated': len(result) < len(items),
                'entry_byte_budget': max_bytes}

    async def get_observations(self, subject_id='local-user', window='1h', limit=20,
                               max_bytes=65536, capability_id=None, source_id=None, sensor_id=None):
        if self.closed:
            raise RuntimeError('runtime is closed')
        self._bounded_query([], limit, max_bytes)
        for value in (capability_id, source_id, sensor_id):
            if value is not None:
                identifier(value)
        duration = _window_ms(window)
        self._prune()
        now = timestamp()
        rows = self.store.query(subject_id, max(0, now-duration), now, 'observation.signal',
            limit=limit+1, registered_only=True, capability_id=capability_id,
            source_id=source_id, sensor_id=sensor_id)
        result = self._bounded_query(list(reversed(rows)), limit, max_bytes)
        result['order'] = 'newest_first'
        result['scope'] = 'retained_registered_observations'
        return result

    async def get_observation_page(self, subject_id='local-user', cursor=None, limit=20,
                                   max_bytes=65536, capability_id=None, source_id=None,
                                   sensor_id=None, events_only=False):
        if self.closed:
            raise RuntimeError('runtime is closed')
        from .observation_pages import observation_page
        self._prune()
        return observation_page(self.store, subject_id=subject_id, cursor=cursor, limit=limit,
            max_bytes=max_bytes, capability_id=capability_id, source_id=source_id,
            sensor_id=sensor_id, events_only=events_only)

    async def get_transcripts(self, subject_id='local-user', window='1h', limit=16, max_bytes=32768):
        if self.closed:
            raise RuntimeError('runtime is closed')
        self._bounded_query([], limit, max_bytes)
        now = timestamp()
        return self.transcript_memory.snapshot(subject_id, now_ms=now,
            start_ms=max(0, now-_window_ms(window)), limit=limit, max_bytes=max_bytes)

    async def get_object_memory(self, subject_id='local-user', text=None, entity_id=None,
                                limit=20, max_bytes=65536):
        if self.closed:
            raise RuntimeError('runtime is closed')
        self._bounded_query([], limit, max_bytes)
        # ObjectMemory has a 100-result query bound. The extra candidate makes
        # count truncation explicit except at the API's maximum, handled below.
        items = self.object_memory.find(subject_id=subject_id, text=text, entity_id=entity_id, limit=100)
        result = self._bounded_query(items, limit, max_bytes)
        result['possibly_more'] = len(items) == 100
        result['scope'] = 'retained_object_evidence'
        return result

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
            "storage": self.store.status(), "signals": latest, "dropped_records": self.dropped_records, "plugin_errors": deepcopy(self.plugin_errors)})
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

    async def get_context(self, subject_id="local-user", window="1h", include=None):
        self._prune()
        now = timestamp()
        start = now-_window_ms(window)
        counts, events = self.store.event_summary(subject_id, start, now, per_kind=8)
        total_events = sum(counts.values())
        records = [r for r in self.store.latest_signals(subject_id, start, now)+events
                   if r.get("observation", {}).get("name") != TRANSCRIPT_OUTPUT]
        result = envelope("observation.context", subject_id, now, window_ms=_window_ms(window),
            evidence=[{"record_id": r["id"]} for r in events])
        result["models"] = self._models_for(records, result["models"])
        from .context import observation_packet
        result.update({"observations": observation_packet(records, now),
                       "transcripts": self.transcript_memory.snapshot(subject_id, now_ms=now, start_ms=start),
                       "context_policy": {"raw_media_included": False, "hypotheses_verified": False, "freshness_ms": 5000},
                       "summary": {"event_counts": counts, "event_count": total_events,
                                   "evidence_count": len(events), "max_evidence_per_event_type": 8,
                                   "evidence_truncated": total_events > len(events),
                                   "record_count": self.store.count(subject_id, start, now), "method": "deterministic_aggregate"},
                       "trends": await self.get_trends(subject_id, window),
                       "include": include or ["speech", "movement", "interaction"]})
        return result

    async def summarize(self, **kwargs):
        return await self.get_context(**kwargs)

    async def remember_objects(self, record_id, *, objects_field='objects', complete_snapshot=True):
        """Remember only a stored registered observation, retaining its evidence."""
        self._check()
        if not isinstance(record_id, str) or len(record_id) > 128:
            raise ValueError('invalid evidence ID')
        records = self.store.get_many([record_id])
        if not records:
            raise ValueError('observation evidence is not retained in this runtime')
        observation = records[0]['observation']
        live = observation.get('processing_epoch') == f'{self._session_id}:{self._generation}'
        epoch = observation.get('sensor', {}).get('stream_epoch')
        if epoch is not None:
            live = live and any(lane.get('epoch') == epoch and not s.get('closing')
                for manager in getattr(self, '_sensor_streams', ()) for s in manager.streams.values()
                for lane in s['lanes'])
        return self.object_memory.observe(records[0],
            objects_field=objects_field, complete_snapshot=complete_snapshot,
            live=live)

    async def find_objects(self, *, subject_id='local-user', text=None, entity_id=None, limit=20):
        if self.closed:
            raise RuntimeError('runtime is closed')
        return self.object_memory.find(subject_id=subject_id, text=text, entity_id=entity_id, limit=limit)

    async def inspect_interval(self, start_ms, end_ms, subject_id="local-user"):
        if start_ms > end_ms:
            raise ValueError("start_ms must not exceed end_ms")
        return {"subject_id": subject_id, "start_ms": start_ms, "end_ms": end_ms,
                "records": await self.get_history(subject_id, start_ms=start_ms, end_ms=end_ms, limit=100_000)}

    async def pause(self):
        self._invalidate_evaluation("pause")
        self.paused = True
        self._analysis_results.clear()
        self._generation += 1
        self._clear_separation_media()
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
        self._invalidate_evaluation("delete", subject_id)
        from .streams import _finish
        cleanup = asyncio.create_task(self._delete_data(subject_id))
        try:
            return await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await _finish(cleanup)
            raise

    async def _delete_data(self, subject_id=None):
        self._generation += 1
        self._clear_separation_media()
        for controller in self._call_controllers.values():
            controller.forget_subject(subject_id, now_ms=timestamp())
        deleted = self.store.delete(subject_id)
        self.object_memory.clear(subject_id)
        self.transcript_memory.clear(subject_id)
        reset_error = None
        try:
            await self._reset_plugins()
        except Exception as exc:
            reset_error = exc
        if subject_id is None:
            self._sensor_refresh.clear()
        else:
            self._sensor_refresh.pop(subject_id, None)
        # Imported aggregate artifacts cannot be untrained for one subject. Drop
        # active calibration for every deletion rather than retain unknown contributions.
        for definition in self.prediction_definitions.values():
            definition["calibration"] = None
        if subject_id is None:
            self._analysis_results.clear()
            self._last_ts.clear()
            self._metric_directions.clear()
        else:
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
        if reset_error is not None:
            raise reset_error
        return {"deleted_records": deleted, "subject_id": subject_id}

    async def _reset_plugins(self):
        from .streams import _finish
        cleanup = asyncio.create_task(self._reset_all_plugins())
        try:
            return await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await _finish(cleanup)
            raise

    async def _reset_all_plugins(self):
        for controller in self._call_controllers.values():
            controller.reset_sources(now_ms=timestamp())
        self.object_memory.invalidate_visibility()
        self.transcript_memory.interrupt()
        errors = []
        for streams in getattr(self, '_sensor_streams', ()):
            try:
                await streams.clear()
            except Exception as exc:
                errors.append(exc)
        if hasattr(self, "_media_router"):
            try:
                await self._media_router.clear()
            except Exception as exc:
                errors.append(exc)
        for plugin in self.plugins:
            reset = getattr(plugin, "reset", None)
            if reset:
                try:
                    result = reset()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError(f'{len(errors)} processing reset operation(s) failed') from errors[0]

    async def reset_processing(self):
        """Discard pending media results and model buffers without pausing text."""
        self._invalidate_evaluation("reset")
        self._generation += 1
        self._clear_separation_media()
        await self._reset_plugins()
        return {"processing_reset": True}

    async def close(self):
        from .streams import _finish
        task = getattr(self, '_close_task', None)
        if task is None:
            self._invalidate_evaluation("close")
            task = self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish(task)
            raise

    async def _close(self):
        if self.closed:
            return
        self.closed = True
        for controller in self._call_controllers.values():
            controller.forget_subject(now_ms=timestamp())
            controller.close(now_ms=timestamp())
        self.object_memory.clear()
        self.transcript_memory.clear()
        self._generation += 1
        self._clear_separation_media()
        if hasattr(self, "_media_router"):
            await self._media_router.clear()
        for queue, _ in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)
        errors = []
        try:
            for streams in getattr(self, '_sensor_streams', ()):
                try:
                    await streams.close()
                except Exception as exc:
                    errors.append(exc)
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
