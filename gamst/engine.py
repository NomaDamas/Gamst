"""Asynchronous perception runtime with bounded subscriptions and local memory."""
from __future__ import annotations

import asyncio
import inspect
import json
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
from .semantic_fusion import SemanticStateFusion
from .semantic_state import SemanticStateChain
from .context_memory import ContextMemory, sync_records
from .transcripts import TranscriptMemory, TRANSCRIPT_OUTPUT
from .watch import WatchManager
from .counting import INPUT_CAPABILITY, INSTRUCTION_CAPABILITY
from .revisions import RETRACTION, identity, revision_fields
from .temporal import TemporalAligner
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


class Gamst:
    def __init__(self, *, consent=False, store=":memory:", retention_hours=24,
                 window_ms=30_000, queue_size=256, save_raw_media=False,
                 capabilities=None, sensor_inference_interval_ms=0,
                 max_records=5000, max_payload_bytes=32*1024*1024):
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
        self._capture_owners: set[str] = set()
        self._paused_before_capture = False
        self._session_id = uuid.uuid4().hex
        self.semantic_states = SemanticStateChain(retention_ms=min(self.store.retention_ms, 300_000))
        self.semantic_fusion = SemanticStateFusion()
        # Portable temporal/context memory mirrors normalized envelopes only;
        # raw media and provider payloads remain owned by Store/provider runtimes.
        self.context_memory = ContextMemory(session_id=self._session_id,
                                            max_bytes=min(max_payload_bytes, 4 * 1024 * 1024))
        self.context_memory_drops = 0
        self._metric_directions = {}
        self.instructions = None        # of a running recipe (gamst.recipes.instructions)
        self._instruction_log = None
        self.dropped_records = 0
        self.plugin_errors = []
        if not isinstance(sensor_inference_interval_ms, (int, float)) or not 0 <= sensor_inference_interval_ms <= 5000:
            raise ValueError('sensor_inference_interval_ms must be between 0 and 5000')
        self.sensor_inference_interval_ms = sensor_inference_interval_ms
        self._sensor_refresh = {}
        self._call_controllers = {}
        self._separation_media = None
        # The multimodal decision provider is an internal perception module.
        # It is registered lazily so discovery of unrelated capabilities stays
        # backward compatible and never loads a model or starts a sensor.
        self._multimodal_decision = None
        from .voice_backend import RealtimeVoiceConfig
        self.voice_backend = RealtimeVoiceConfig()
        self.pipeline = None  # the running gamst.pipeline.Pipeline, if any
        # Clocks of sources that do not use this runtime's capture clock (add_clock_sample), and what the
        # Pipeline said about its completeness over time: both are read by get_timeline only.
        from .timeline import History
        self.clocks = TemporalAligner()
        self._timeline_history = History()
        self._timeline_lock = asyncio.Lock()
        # Owner-declared Watches are the only alarm path; there is no hidden trigger.
        self.watches = WatchManager(
            lambda subject, start, end, kind, limit: self.store.query(subject, start, end, kind, limit=limit),
            timestamp)

    def enable_separation_media(self, *, subject_id='local-user', source_id, sensor_id,
                                ttl_ms=30000, capacity=8, max_bytes=16*1024*1024,
                                visual_source=None):
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
        return store.grant(subject_id, source_id, sensor_id, visual_source=visual_source)

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
        try:
            if stored:
                sync_records(self.context_memory, (record,))
        except (TypeError, ValueError, OverflowError):
            # Memory is a bounded derived view. A malformed/oversized provider
            # envelope must not roll back the canonical Store commit.
            self.context_memory_drops += 1
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
        if stored:
            observation = record.get('observation', {})
            if (observation.get('capability') or {}).get('id') == INSTRUCTION_CAPABILITY and self._instruction_log is not None:
                self._instruction_log += [(self.store.known_now(), event) for event in observation['value']['events']]
                del self._instruction_log[:-256]
            self.watches.observe(record)
            self._timeline_history.sample(self.pipeline, self.clocks, self.store.known_now())
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

    def _ensure_multimodal_decision(self):
        """Create/register the internal multimodal decision perception module."""
        if self._multimodal_decision is None:
            from .providers.multimodal_decision import MultimodalDecisionProvider
            provider = MultimodalDecisionProvider()
            self.register_capability(provider.spec)
            self._multimodal_decision = provider
        return self._multimodal_decision

    async def list_decision_modules(self):
        """Describe decision perception modules without activating sensors."""
        provider = self._ensure_multimodal_decision()
        return {"modules": [provider.descriptor], "activation": "explicit_decide_multimodal"}

    async def decide_multimodal(self, subject_id='local-user', window='5s',
                                requested_capabilities=None, limit=32,
                                timestamp_ms=None):
        """Run the internal multimodal perception decision over retained evidence.

        The result is itself a registered ``decision.multimodal`` observation,
        so it becomes part of Context/Memory and is available through MCP.
        """
        self._check()
        identifier(subject_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be 1..100')
        duration = _window_ms(window)
        now = timestamp(timestamp_ms)
        provider = self._ensure_multimodal_decision()
        records = self.store.query(subject_id, max(0, now-duration), now,
                                   'observation.signal', limit=limit * 4,
                                   registered_only=True)
        value = provider.decide(records, now_ms=now,
                                requested_capabilities=requested_capabilities,
                                input_window_ms=duration)
        record = await self.push_observation(provider.spec.id, value,
            sensor=provider.next_sensor(), subject_id=subject_id,
            timestamp_ms=now)
        record['decision_input'] = {
            'window_ms': duration,
            'requested_capabilities': list(requested_capabilities or []),
            'evidence_count': len(value['evidence_ids']),
            'raw_media_included': False,
        }
        return record

    async def push_observation(self, capability_id, value, *, sensor,
                               subject_id='local-user', timestamp_ms=None, quality=None, supersedes=None):
        spec = self.perception_capabilities.get(capability_id)
        return await self.push_signal(spec.output_name, value, spec.modality,
            subject_id=subject_id, timestamp_ms=timestamp_ms, quality=quality,
            model_id=spec.provider, model_version=spec.version,
            capability_id=capability_id, sensor=sensor, supersedes=supersedes)

    async def push_signal(self, name, value, modality, subject_id="local-user", timestamp_ms=None,
                          model_id="external", model_version="unspecified", quality=None,
                          capability_id=None, sensor=None, supersedes=None):
        """Commit one signal.

        ``supersedes`` is the ID of a retained record this one corrects: it adds a revision, no stored record
        changes (:mod:`gamst.revisions`). ``quality`` may carry ``level``, which is kept and shown as given.
        """
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
        else:
            from .private_payload import refuse
            refuse(value)
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
            if "level" in quality:
                identifier(quality["level"], "quality level")
            quality = {"status": quality["status"], "flags": list(quality.get("flags", [])),
                       **({"level": quality["level"]} if "level" in quality else {})}
        observation = {"name": name, "value": value, "modality": modality}
        if contract is not None:
            sensor_data = asdict(sensor)
            sensor_data['channels'] = list(sensor.channels)
            observation.update(capability=contract, sensor=sensor_data,
                               processing_epoch=f'{self._session_id}:{self._generation}')
        record = self._record("observation.signal", subject_id, ts,
            observation=observation,
            models=[{"id": model_id, "version": model_version, "backend": "plugin"}], quality=quality)
        if supersedes is not None:
            record.update(revision_fields(self.store, record, supersedes))
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

    async def retract_observation(self, record_id, reason=''):
        """Record that a retained observation turned out not to be there; nothing stored is changed."""
        self._check()
        if not isinstance(reason, str) or len(reason) > 500:
            raise ValueError('reason must be a string of at most 500 characters')
        target = next(iter(self.store.get_many([record_id])), None) if isinstance(record_id, str) else None
        if target is None:
            raise ValueError('only a retained record can be retracted')
        observation = target['observation']
        record = self._record(RETRACTION, target['subject_id'], target['timestamp_ms'], observation=dict(
            name=observation.get('name'), capability_id=(observation.get('capability') or {}).get('id'), reason=reason),
            evidence=[{'record_id': record_id}])
        record.update(revision_fields(self.store, record, record_id))
        return self._emit(record)

    def add_clock_sample(self, clock_id, source_ms, local_ms, *, clock_epoch=None):
        """One pair of simultaneous readings: a source's own clock and this runtime's capture clock (ms).

        Not needed when a server echoes the capture time it was sent. A source whose records carry
        ``SensorMetadata.clock_id`` and has too few samples stays out of the timeline rows and is named there.
        """
        from .timeline import clock_key
        identifier(clock_id, 'clock_id')
        if clock_epoch is not None:
            identifier(clock_epoch, 'clock_epoch')
        self.clocks.add_clock_sample(clock_key(clock_id, clock_epoch), source_ms, local_ms)
        self._timeline_history.sample(self.pipeline, self.clocks, self.store.known_now(), force=True)
        return self.clocks.clock_status(clock_key(clock_id, clock_epoch))

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
        if getattr(plugin, "private", False):
            raise PermissionError("a private worker answers its owning module only")
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
            rows.append(dict(kind='controller', controller_id=name, closed=state['closed'],
                source_count=len(state['sources']), request_count=len(state['requests']),
                overflow_rejections=state['overflow_rejections'], omitted_receipts=state['omitted_receipts'],
                admission_denials=state['admission_denials'],
                cadence_scope='independent_sensor_and_context_delivery; see get_sensor_status/get_context'))
            for request in state['requests']:
                decision = request['decision']
                rows.append(dict(kind='request', controller_id=name, request_id=request['id'],
                    source=request['source'], state=request['state'], priority=request['priority'],
                    desired_route=decision['desired_route'], reason=decision['reason'], blockers=request['blockers'],
                    evidence=decision['evidence'], expires_ms=request['expires_ms'],
                    finished=request['finished'], discard=request['discard'],
                    allowed_route=request['allowed_route'], admission_blockers=request['admission_blockers'],
                    admission_advisory=True, dispatched_route=request['dispatched_route'],
                    last_admission=request['last_admission'], execution=request['execution'],
                    semantic_audit_due=request['semantic_audit_due'],
                    last_semantic_capture_ms=request['last_semantic_capture_ms'],
                    unknown_candidate_unresolved=request['unknown_candidate_unresolved']))
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

    def _who_intervals(self, subject_id, as_of_ms=None, key='who'):
        """``{track_id: [(from_ms, to_ms, who)]}`` of the retained ``vision.person_binding`` records (person_binding.py);
        ``key='person_id'``: the person ids (P1, P2, ...; named or not)."""
        from .counting import MAX_SCAN
        from .providers.person_binding import CAPABILITY, binding_events, intervals_of
        rows = self.store.query(subject_id, None, None, 'observation.signal', limit=MAX_SCAN, registered_only=True,
                                capability_id=CAPABILITY, known_until=as_of_ms)
        return intervals_of(binding_events(rows), key=key)

    @staticmethod
    def _with_who(rows, intervals, persons=None):
        """Stored rows whose events that name a track carry the person bound to it (retroactive within the track). One
        action of a person counted on two tracks (one body seen twice) within ``SAME_EVENT_MS`` is kept once
        (:class:`gamst.providers.person_binding.PersonOnce`, the earlier one). ``persons``: person-id intervals; the
        events then carry ``person_id`` too."""
        from .providers.person_binding import PersonOnce, resolve
        mine = lambda item: isinstance(item, dict) and item.get('track_id') and \
            str(item.get('event', '')).startswith('exemplar.')
        found = []
        for index, record in enumerate(rows):
            value = (record.get('observation') or {}).get('value')
            for position, item in enumerate((value.get('events') if isinstance(value, dict) else None) or ()):
                if mine(item):
                    end_ms = item.get('end_ms')
                    found.append((end_ms if type(end_ms) in (int, float) else 0, index, position))
        once, dropped = PersonOnce(), set()
        for end_ms, index, position in sorted(found):
            item = rows[index]['observation']['value']['events'][position]
            if not once.fresh(resolve(item, intervals), item['track_id'], item.get('end_ms'),
                              (item.get('event'), item.get('label'))):
                dropped.add((index, position))
        shown = []
        for index, record in enumerate(rows):
            value = (record.get('observation') or {}).get('value')
            events = value.get('events') if isinstance(value, dict) else None
            if isinstance(events, list) and any(isinstance(item, dict) and item.get('track_id') for item in events):
                events = [dict(item, who=resolve(item, intervals), **({'who_when_counted': item['who']} if item.get('who')
                               and item.get('who') != resolve(item, intervals) else {}),
                               **({'person_id': resolve(item, persons, 'person_id')} if persons is not None else {}))
                          if mine(item) else item
                          for position, item in enumerate(events) if (index, position) not in dropped]
                record = dict(record, observation=dict(record['observation'], value=dict(value, events=events)))
            shown.append(record)
        return shown

    async def count_events(self, capability_id, subject_id='local-user', window='1h', event=None,
                           where=None, group_by=None, max_evidence=20, as_of_ms=None, who=None, person_id=None):
        """Count events of retained records of one capability, with bounded evidence record IDs.

        ``complete`` is False when the Pipeline lost frames this capability depends on in the window (``lost``
        says how much), or when not every record was scanned: the true count may differ. Events of a record
        that a revision or retraction replaced are not counted. Counts of an instruction and of the one it replaced are
        never added up (:func:`gamst.counting.count_events`). ``as_of_ms`` counts what was known at that
        knowledge time, over the window that ended then.
        """
        if self.closed:
            raise RuntimeError('runtime is closed')
        identifier(capability_id)
        if where is not None and (type(where) is not dict or len(where) > 16):
            raise ValueError('where must be an object of at most 16 path: value pairs')
        if type(max_evidence) is not int or not 0 <= max_evidence <= 100:
            raise ValueError('max_evidence must be 0..100')
        from .counting import GAP_CAPABILITY, MAX_SCAN, count_events, current_instructions, gaps_of, lost
        if self.pipeline is not None and as_of_ms is None:
            await self.pipeline.commit_gaps(force=True)
        self._prune()
        now = self._as_of(as_of_ms)
        start = max(0, now - _window_ms(window))
        rows = self.store.query(subject_id, start, now, 'observation.signal', limit=MAX_SCAN + 1,
                                registered_only=True, capability_id=capability_id, known_until=as_of_ms)
        truncated = len(rows) > MAX_SCAN
        gaps = gaps_of(self.store.query(subject_id, start, now, 'observation.signal', limit=MAX_SCAN, registered_only=True,
                                        capability_id=GAP_CAPABILITY, known_until=as_of_ms), capability_id)
        rows = rows[-MAX_SCAN:]
        replaced, scanned = self.store.superseded(subject_id, as_of_ms), len(rows)
        rows = [record for record in rows if record['id'] not in replaced]
        changes = [change for _, change in self.instruction_changes(as_of_ms) if capability_id in change.get('capabilities', ())]
        if who is not None or person_id is not None:
            # a person's events: who / person_id are resolved from the bindings, also for events counted before
            if who is not None and (not isinstance(who, str) or not 1 <= len(who) <= 128):
                raise ValueError('who is the name of an enrolled person (1..128 characters)')
            if person_id is not None and (not isinstance(person_id, str) or not 1 <= len(person_id) <= 32):
                raise ValueError('person_id is a person of the run (P1, P2, ...)')
            rows = self._with_who(rows, self._who_intervals(subject_id, as_of_ms),
                                  self._who_intervals(subject_id, as_of_ms, 'person_id') if person_id is not None else None)
            where = dict(where or {}, **({'who': who} if who is not None else {}),
                         **({'person_id': person_id} if person_id is not None else {}))
        result = count_events(rows, event=event, where=where, group_by=group_by, max_evidence=max_evidence,
                              current=current_instructions(changes))
        if changes:     # since when what is counted has been looked for
            result['counted_from_ms'] = {name: from_ms for name, (_, from_ms) in current_instructions(changes).items()}
        if who is not None:
            result['who'] = who
        if person_id is not None:
            result['person_id'] = person_id
        return dict(result, capability_id=capability_id, window_start_ms=start, window_end_ms=now,
                    scanned_records=scanned, scan_truncated=truncated,
                    complete=not gaps and not truncated, lost=lost(gaps),
                    **({} if as_of_ms is None else dict(as_of_ms=as_of_ms, removed_since=bool(
                        self.store.lost_until_ms is not None and self.store.lost_until_ms >= start))))

    # ------------------------------------------------------------------ instructions of a running recipe

    def instruction_changes(self, known_until=None, subject_id='local-user'):
        """``[(known_ms, pipeline.instructions event)]`` in the order they were learned; from the Store once, then kept."""
        if self._instruction_log is None:
            self._instruction_log = [(known, event) for record, known in self.store.query(
                subject_id, kind='observation.signal', capability_id=INSTRUCTION_CAPABILITY, registered_only=True, limit=256,
                with_known=True) for event in record['observation']['value']['events']]
        return [item for item in self._instruction_log if known_until is None or (item[0] or 0) <= known_until]

    def input_starts(self, subject_id='local-user', known_until=None, limit=8):
        """The newest ``limit`` ``pipeline.input_start`` events (oldest first): where each input of a run began, so a
        capture time t is ``position_ms + t - start_ms`` into its recording (:data:`gamst.pipeline.INPUTS`)."""
        found = self.store.query(subject_id, kind='observation.signal', capability_id=INPUT_CAPABILITY,
                                 registered_only=True, limit=limit, known_until=known_until)
        return [dict(event) for record in found for event in record['observation']['value']['events']][-limit:]

    def _instructed(self):
        if self.instructions is None:
            raise ValueError('no recipe is running here; instructions belong to a running recipe (gamst recipe run ... --serve)')
        return self.instructions

    async def list_instructions(self):
        """What the running recipe looks for now, since when, who set it, and what can be set (:mod:`gamst.recipes.instructions`)."""
        return await self._instructed().list()

    async def set_instruction(self, kind, **options):
        """Set or replace the words or the movement the running recipe looks for, without restarting it."""
        return await self._instructed().set(kind, **options)

    async def enroll_person(self, **options):
        """Enroll a person (few-shot, consent required) into the running recipe, so instructions can name them (``who``)."""
        return await self._instructed().enroll_person(**options)

    async def remove_instruction(self, reference, actor='user'):
        return await self._instructed().remove(reference, actor=actor)

    async def search_retained(self, reference, window='1h', max_events=50):
        """Look for the current words or movement in records retained from before it was set; results are ``retrospective``."""
        return await self._instructed().search(reference, window=window, max_events=max_events)

    def _as_of(self, as_of_ms):
        if as_of_ms is None:
            return timestamp()
        if type(as_of_ms) is not int or not 0 <= as_of_ms <= self.store.known_now():
            raise ValueError('as_of_ms must be a knowledge time (ms) that has passed')
        return as_of_ms

    async def get_timeline(self, subject_id='local-user', window=None, start_ms=None, end_ms=None, step_ms=200,
                           capabilities=None, settled_only=False, as_of_ms=None, cursor=None, max_bytes=65536, who=None,
                           person_id=None):
        """Retained records as rows of capture time, with an explicit status per capability (:mod:`gamst.timeline`).

        Give ``window`` (default 10 s, ending now) or ``start_ms`` and ``end_ms``. ``as_of_ms``: what was known at
        that time. ``cursor``: only what changed since the answer it came from, with old and new values.
        """
        if self.closed:
            raise RuntimeError('runtime is closed')
        from . import timeline
        if not isinstance(subject_id, str) or not 1 <= len(subject_id) <= 128:
            raise ValueError('subject_id must be a string of 1..128 characters')
        if type(step_ms) is not int or not timeline.MIN_STEP_MS <= step_ms <= timeline.MAX_STEP_MS:
            raise ValueError(f'step_ms must be {timeline.MIN_STEP_MS}..{timeline.MAX_STEP_MS}')
        if type(settled_only) is not bool:
            raise ValueError('settled_only must be boolean')
        if type(max_bytes) is not int or not 4096 <= max_bytes <= 262144:
            raise ValueError('max_bytes must be 4096..262144')
        if capabilities is not None:
            if type(capabilities) not in (list, tuple) or not 1 <= len(capabilities) <= timeline.MAX_COLUMNS:
                raise ValueError(f'capabilities must be a list of 1..{timeline.MAX_COLUMNS} names')
            for name in capabilities:
                identifier(name, 'capability')
        explicit = start_ms is not None or end_ms is not None
        if explicit and (window is not None or type(start_ms) is not int or type(end_ms) is not int
                         or not 0 <= start_ms <= end_ms):
            raise ValueError('give window, or integer start_ms and end_ms in order')
        if cursor is not None and as_of_ms is not None:
            raise ValueError('a cursor already names what was known; give cursor or as_of_ms')
        self._prune()
        self._as_of(as_of_ms)
        known = self.store.read_mark()
        self._timeline_history.sample(self.pipeline, self.clocks, known, force=True)
        now = min(timestamp(), known) if as_of_ms is None else as_of_ms
        if not explicit:
            start_ms, end_ms = max(0, now - _window_ms('10s' if window is None else window)), now
        async with self._timeline_lock:     # one read at a time: its records are held in memory
            key = self.store._cursor_key
            try:
                answer = await timeline.read(
                    self.store, self._timeline_history, subject_id=subject_id, start_ms=start_ms,
                    end_ms=max(start_ms, min(end_ms, now)), known_ms=known, as_of_ms=as_of_ms, now_ms=now, step_ms=step_ms,
                    capabilities=capabilities, settled_only=settled_only, cursor=cursor, max_bytes=max_bytes,
                    contracts=self.perception_capabilities._specs, changes=self.instruction_changes(as_of_ms or known))
            except Exception:
                if not self.closed:
                    raise
            if self.closed or self.store._cursor_key != key:
                raise RuntimeError('timeline discarded after deletion or close')
            if who is not None or person_id is not None or any(
                    isinstance(entry.get('value'), dict) and entry['value'].get('track_id')
                    for entry in answer.get('evidence', {}).values()):
                answer = self._timeline_who(answer, self._who_intervals(subject_id, as_of_ms), who,
                                            self._who_intervals(subject_id, as_of_ms, 'person_id'), person_id)
            if isinstance(answer, dict) and not answer.get('restart'):
                starts = self.input_starts(subject_id, known_until=as_of_ms)
                if starts:
                    answer['input_starts'] = starts
            return answer

    @staticmethod
    def _timeline_who(answer, intervals, who, persons=None, person_id=None):
        """Events of a track show the person bound to it (retroactive): ``who`` and ``person_id``; with ``who`` (or
        ``person_id``), only that person's events are left and ``who`` says when the person was on screen. The cursor
        stays the unfiltered read's."""
        from .providers.person_binding import resolve, seen_intervals
        dropped = set()
        for ref, entry in answer.get('evidence', {}).items():
            value = entry.get('value')
            if not isinstance(value, dict) or not value.get('track_id') or not str(value.get('event', '')).startswith('exemplar.'):
                continue
            found = resolve(value, intervals)
            person = resolve(value, persons, 'person_id') if persons is not None else None
            entry['value'] = dict(value, who=found, **({'person_id': person} if person else {}))
            if who is not None and found != who or person_id is not None and person != person_id:
                dropped.add(ref)
        if who is None and person_id is None:
            return answer
        if who is not None and (not isinstance(who, str) or not 1 <= len(who) <= 128):
            raise ValueError('who is the name of an enrolled person (1..128 characters)')
        if person_id is not None and (not isinstance(person_id, str) or not 1 <= len(person_id) <= 32):
            raise ValueError('person_id is a person of the run (P1, P2, ...)')
        for ref in dropped:
            del answer['evidence'][ref]
        for row in answer.get('rows', ()):
            for cell in row.get('cells', {}).values():
                if cell.get('ref') in dropped:
                    del cell['ref']
                if 'intervals' in cell:
                    cell['intervals'] = [ref for ref in cell['intervals'] if ref not in dropped]
        answer['who'] = dict(name=who, **({'person_id': person_id} if person_id is not None else {}),
                             filtered_events=len(dropped), seen=seen_intervals(
                                 intervals if person_id is None else persons, who if person_id is None else person_id,
                                 answer.get('start_ms'), answer.get('end_ms'))[:64],
                             note='events of other people and of unknown tracks are left out; next_cursor is of the '
                                  'unfiltered read')
        return answer

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

    async def get_context_memory(self, *, evidence_id=None, entity_id=None,
                                 start_ms=None, end_ms=None, query_cutoff_ms=None,
                                 limit=50, max_bytes=65536):
        """Read the bounded normalized context mirror without exposing raw media."""
        if self.closed:
            raise RuntimeError('runtime is closed')
        self._bounded_query([], limit, max_bytes)
        items = self.context_memory.retrieve(evidence_id=evidence_id,
            entity_id=entity_id, start_ms=start_ms, end_ms=end_ms,
            query_cutoff_ms=query_cutoff_ms, limit=limit)
        result = self._bounded_query(items, limit, max_bytes)
        result.update(scope='normalized_context_memory',
                      dropped=self.context_memory_drops,
                      session_id=self.context_memory.session_id)
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

    async def push_semantic_state(self, state, *, subject_id="local-user",
                                  timestamp_ms=None, events=None, evidence_ids=None,
                                  sources=None, model_id="semantic_state.rules",
                                  model_version="1"):
        """Append a bounded state-chain update; never calls a heavy model."""
        self._check()
        item = self.semantic_states.append(subject_id, timestamp_ms=timestamp_ms,
            state=state, events=events, evidence_ids=evidence_ids, sources=sources,
            model_id=model_id, model_version=model_version)
        return item

    async def get_semantic_state(self, subject_id="local-user", *, window="5m",
                                 limit=32):
        """Read short-term semantic state updates without triggering inference."""
        self._check()
        now = timestamp()
        return self.semantic_states.get(subject_id,
            start_ms=max(0, now - _window_ms(window)), end_ms=now, limit=limit)

    async def observe_semantic_video_state(self, state, *, subject_id="local-user",
                                           timestamp_ms=None, evidence_ids=None,
                                           model_id="streaming_vlm", model_version="unverified"):
        """Record a streaming-video semantic state; model execution is external."""
        self._check()
        ts = timestamp(timestamp_ms)
        self.semantic_fusion.observe_video_state(state, timestamp_ms=ts,
            evidence_ids=evidence_ids or [], model_id=model_id, model_version=model_version)
        return await self.push_semantic_state(
            {"video_state": state, "fusion": self.semantic_fusion.snapshot(timestamp_ms=ts)},
            subject_id=subject_id, timestamp_ms=ts, evidence_ids=evidence_ids,
            sources=["video"], model_id=model_id, model_version=model_version)

    async def observe_perception_signal(self, signal):
        """Merge a validated plugin signal into fusion and the state chain."""
        self._check()
        self.semantic_fusion.observe_signal(signal)
        ts = signal.get("timestamp_ms", timestamp())
        fusion = self.semantic_fusion.snapshot(timestamp_ms=ts)
        return await self.push_semantic_state(
            {"fusion": fusion}, subject_id=signal.get("subject_id", "local-user"),
            timestamp_ms=ts, evidence_ids=fusion["evidence_ids"],
            sources=fusion["modalities"], model_id="gamst.semantic_fusion", model_version="1")

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

    async def build_temporal_context(self, subject_id="local-user", window="1h", *, query_cutoff_ms=None, max_bytes=131072):
        """Return a bounded timestamped event packet for temporal consumers.

        This is additive; ``get_context`` remains the stable legacy snapshot API.
        Records after the query cutoff are never eligible.
        """
        from .temporal import TemporalEvent
        from .contradictions import detect_contradictions
        self._prune()
        cutoff = timestamp(query_cutoff_ms)
        start = cutoff - _window_ms(window)
        # Context snapshots may collapse same-name observations to their latest
        # value. Contradiction detection needs both cross-modal source records,
        # so query the bounded signal history directly with the same cutoff.
        records = [r for r in self.store.query(subject_id, start, cutoff,
                                               "observation.signal", limit=64)
                   if r.get("timestamp_ms", 0) <= cutoff]
        aligner = TemporalAligner()
        # A correction is a revision of one event; the event's identity is its first record.
        retracted = {row[0] for row in self.store.revisions(subject_id) if row[3]}
        events = {}
        for record in records[:64]:
            event_id = identity(record)
            if event_id in retracted:
                continue
            observation = record.get("observation", {})
            ts = record.get("timestamp_ms")
            if type(ts) is not int or not observation.get("name"):
                continue
            sensor = observation.get("sensor", {})
            models = record.get("models", [])
            model = models[0] if models and isinstance(models[0], dict) else {}
            event = TemporalEvent(
                event_id=event_id, stream_id=str(sensor.get("source_id", "runtime")),
                subject_id=subject_id, modality=str(observation.get("modality", "unknown")),
                event_type=str(observation["name"]), source_clock=str(sensor.get("source_clock", "wall_ms")),
                clock_id=str(sensor.get("clock_id", sensor.get("source_clock", "wall_ms"))),
                clock_epoch=str(sensor.get("clock_epoch", "legacy")),
                capture_timestamp_ms=ts, start_ms=int(observation.get("start_ms", ts)),
                end_ms=int(observation.get("end_ms", ts)), sequence=record.get("sequence", 0),
                provider=str(model.get("id", "unknown")), model_revision=str(model.get("version", "unknown")),
                value=observation.get("value"), evidence_ids=(record["id"],),
                uncertainty=record.get("quality"),
                monotonic_timestamp_ms=sensor.get("monotonic_timestamp_ms"),
                receive_timestamp_ms=record.get("received_timestamp_ms"),
                processing_start_ms=record.get("processing_start_ms"),
                processing_end_ms=record.get("processing_end_ms"),
                gap=bool(record.get("gap", False)),
                revision=int(record.get("revision", 0)) if event_id in events else 0,
                supersedes=event_id if event_id in events and record.get("revision") else None)
            if aligner.accept(event) is not None:
                events[event_id] = event
        events = list(events.values())
        detection = detect_contradictions(events, query_cutoff_ms=cutoff,
                                          now_ms=cutoff, aligner=aligner)
        # A rejected relation is an explicit abstention: it keeps both source
        # records inspectable without presenting unresolved evidence as a
        # contradiction. TemporalPacket validates all relation provenance.
        relations = list(detection["candidates"])
        for item in detection["abstentions"]:
            if len(relations) >= 64 or len(item.get("event_ids", [])) != 2:
                continue
            try:
                left = next(event for event in events if event.event_id == item["event_ids"][0])
                right = next(event for event in events if event.event_id == item["event_ids"][1])
                rejected = aligner.materialize_relation(left, right, "contradiction",
                                                        status="rejected")
                rejected.update({"contradiction_status": "abstained",
                                 "verification_status": "not_checked",
                                 "deception_assessed": False,
                                 "abstention_reason": item.get("reason", "unresolved"),
                                 "uncertainty": item.get("uncertainty", {"status": "unresolved"})})
                relations.append(rejected)
            except (StopIteration, ValueError):
                continue
        packet = aligner.packet(cutoff, now_ms=cutoff, relations=relations[:64])
        result = packet.as_dict()
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > max_bytes:
            raise ValueError("temporal context exceeds byte budget")
        return result

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
        from .context import observation_packet
        result.update({"observations": observation_packet(records, now),
                       "transcripts": self.transcript_memory.snapshot(subject_id, now_ms=now, start_ms=start),
                       "semantic_state_chain": self.semantic_states.get(subject_id,
                           start_ms=start, end_ms=now, limit=32),
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

    async def acquire_capture(self, owner: str):
        """Register a transport that consented to observation; resume on the first owner.

        Transports share one runtime, so one client must not override another's
        deliberate pause after it leaves. The last owner restores the pause state
        that existed before the first owner arrived.
        """
        if not self._capture_owners:
            self._paused_before_capture = self.paused
            await self.resume()
        self._capture_owners.add(owner)
        return {"paused": self.paused, "owners": len(self._capture_owners)}

    async def release_capture(self, owner: str, *, pause_when_idle: bool = False):
        """Drop a transport's ownership.

        When the last owner leaves, restore the pause state from before the first
        owner arrived; ``pause_when_idle`` pauses regardless (an explicit user Stop).
        """
        if owner not in self._capture_owners:
            return {"paused": self.paused, "owners": len(self._capture_owners)}
        self._capture_owners.discard(owner)
        if not self._capture_owners and (pause_when_idle or self._paused_before_capture) and not self.closed:
            await self.pause()
        return {"paused": self.paused, "owners": len(self._capture_owners)}

    async def delete_data(self, subject_id=None):
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
        self._instruction_log = None
        self.object_memory.clear(subject_id)
        self.transcript_memory.clear(subject_id)
        self.watches.clear_subject(subject_id)
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
        self._generation += 1
        self._clear_separation_media()
        await self._reset_plugins()
        return {"processing_reset": True}

    async def close(self):
        from .streams import _finish
        task = getattr(self, '_close_task', None)
        if task is None:
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
