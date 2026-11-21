"""Structured audio-interaction Context from caller-owned audio workers.

This provider does not capture audio, open a device, call a commercial API, or
implement full-duplex playback. An application-owned bridge supplies bounded
measurements from AEC/VAD/ASR/CED/diarization/prosody/DOA workers. GAMST
combines them into one time-aligned Context observation for MCP and LLM/VLM
consumers.

Unavailable inputs remain null. In particular, barge-in, double-talk, AEC
residual and DOA must not be invented when playback reference or a microphone
array is absent.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame

_FIELDS = (
    "speech_active", "silence_ms", "assistant_playing", "barge_in", "double_talk",
    "aec_residual_dbfs", "aec_erle_db", "echo_reduction_verified", "sound_events",
    "speaker_turn", "speaker_slots", "doa_deg", "doa_confidence", "prosody", "transcript",
)


def _finite(value, name, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} is out of range")
    return float(value)


class AudioContext:
    """One bounded, stateful audio Context reducer; no raw samples leave the caller."""

    def __init__(self, *, max_gap_ms=2000, source_revision="local-audio-context-v1"):
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 60000:
            raise ValueError("max_gap_ms must be 1..60000")
        identifier(source_revision, "source_revision")
        self.max_gap_ms = max_gap_ms
        self.source_revision = source_revision
        digest = hashlib.sha256(json.dumps([max_gap_ms, source_revision]).encode()).hexdigest()
        self.spec = CapabilitySpec(
            "audio.context", "audio.interaction_context", "audio", "gamst.audio_context",
            "1." + digest, (OutputField("context", "object"),), "native", window_ms=100,
            freshness_ms=max_gap_ms, stateful=True, reset_on_gap=True, interpretation="measurement",
            devices=("cpu",), sensor_requirements=("structured-audio-worker-context",), license="Apache-2.0")
        self._lock = threading.RLock()
        self.closed = False
        self.reset()

    def reset(self):
        self.owner = None
        self.last_ts = None
        self.last_speech_ts = None
        self.sequence = None

    def close(self):
        with self._lock:
            self.reset()
            self.closed = True

    @staticmethod
    def _events(value):
        if value is None:
            return None
        if type(value) is not list or len(value) > 32:
            raise ValueError("sound_events must be a list of at most 32 entries")
        out = []
        for event in value:
            if not isinstance(event, dict) or set(event) - {"label", "score", "start_ms", "end_ms", "evidence_id"}:
                raise ValueError("sound event has unsupported fields")
            if not isinstance(event.get("label"), str) or not 1 <= len(event["label"]) <= 128:
                raise ValueError("sound event label is required")
            item = {"label": event["label"]}
            if "score" in event:
                item["score"] = _finite(event["score"], "sound event score", 0, 1)
            for key in ("start_ms", "end_ms"):
                if key in event:
                    if type(event[key]) is not int or event[key] < 0:
                        raise ValueError(f"{key} must be a nonnegative integer")
                    item[key] = event[key]
            if "evidence_id" in event:
                identifier(event["evidence_id"], "evidence_id")
                item["evidence_id"] = event["evidence_id"]
            out.append(item)
        return out

    def process(self, frame: InputFrame):
        with self._lock:
            if self.closed:
                raise RuntimeError("audio Context provider is closed")
            if not isinstance(frame, InputFrame) or frame.modality != "audio" or not isinstance(frame.sensor, SensorMetadata):
                raise ValueError("structured audio InputFrame with sensor metadata required")
            if type(frame.timestamp_ms) is not int or frame.timestamp_ms < 0:
                raise ValueError("integer capture timestamp required")
            if not isinstance(frame.data, dict) or set(frame.data) - set(_FIELDS):
                raise ValueError("audio Context contains unsupported fields")
            data = frame.data
            owner = (frame.subject_id, frame.sensor.source_id, frame.sensor.sensor_id, frame.sensor.stream_epoch)
            sequence = frame.sensor.capture_sequence if frame.sensor.capture_sequence is not None else frame.sensor.sequence
            if self.owner == owner and self.last_ts is not None:
                if frame.timestamp_ms <= self.last_ts or sequence <= self.sequence:
                    raise ValueError("audio Context timestamps and sequence must increase")
                if frame.timestamp_ms - self.last_ts > self.max_gap_ms:
                    self.reset()
            self.owner, self.last_ts, self.sequence = owner, frame.timestamp_ms, sequence

            speech = data.get("speech_active")
            if speech is not None and type(speech) is not bool:
                raise ValueError("speech_active must be boolean or null")
            playing = data.get("assistant_playing")
            if playing is not None and type(playing) is not bool:
                raise ValueError("assistant_playing must be boolean or null")
            if speech:
                self.last_speech_ts = frame.timestamp_ms
            silence = data.get("silence_ms")
            if silence is None and self.last_speech_ts is not None and speech is False:
                silence = frame.timestamp_ms - self.last_speech_ts
            if silence is not None:
                silence = int(_finite(silence, "silence_ms", 0, 86_400_000))

            reference = data.get("aec_residual_dbfs") is not None or data.get("aec_erle_db") is not None
            barge = data.get("barge_in")
            if barge is None and speech is not None and playing is not None:
                barge = bool(speech and playing)
            double_talk = data.get("double_talk")
            if double_talk is None and speech is not None and playing is not None:
                double_talk = bool(speech and playing)
            if (barge is not None and type(barge) is not bool) or (double_talk is not None and type(double_talk) is not bool):
                raise ValueError("barge_in/double_talk must be boolean or null")

            context = {"speech_active": speech, "silence_ms": silence, "assistant_playing": playing,
                       "barge_in": barge, "double_talk": double_talk,
                       "aec_residual_dbfs": data.get("aec_residual_dbfs"),
                       "aec_erle_db": data.get("aec_erle_db"),
                       "echo_reduction_verified": data.get("echo_reduction_verified"),
                       "sound_events": self._events(data.get("sound_events")),
                       "speaker_turn": bounded_json(data.get("speaker_turn"), max_bytes=4096) if data.get("speaker_turn") is not None else None,
                       "speaker_slots": bounded_json(data.get("speaker_slots"), max_bytes=4096) if data.get("speaker_slots") is not None else None,
                       "doa_deg": data.get("doa_deg"), "doa_confidence": data.get("doa_confidence"),
                       "prosody": bounded_json(data.get("prosody"), max_bytes=4096) if data.get("prosody") is not None else None,
                       "transcript": bounded_json(data.get("transcript"), max_bytes=8192) if data.get("transcript") is not None else None,
                       "reference_available": reference,
                       "source_revision": self.source_revision,
                       "raw_audio_included": False}
            if context["aec_residual_dbfs"] is not None:
                _finite(context["aec_residual_dbfs"], "aec_residual_dbfs", -200, 20)
            if context["aec_erle_db"] is not None:
                _finite(context["aec_erle_db"], "aec_erle_db", -10, 200)
            if context["doa_deg"] is not None:
                _finite(context["doa_deg"], "doa_deg", -360, 360)
            if context["doa_confidence"] is not None:
                _finite(context["doa_confidence"], "doa_confidence", 0, 1)
            return self.spec.validate({"context": context})
