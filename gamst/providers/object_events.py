"""Generic object-event reducer for tosses and visible face changes.

This is intentionally object-agnostic. An upstream tracker/flow provider
supplies one tracked object's support/contact state, vertical velocity and an
optional semantic face label (for example heads/tails). The reducer turns
those measurements into stable events:

released -> airborne -> caught = one toss
heads -> tails or tails -> heads = one flip

It never guesses when evidence is unavailable: missing support or face labels
keep the state unchanged and confidence remains explicit.
"""
from __future__ import annotations

import hashlib
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame


class ObjectEventReducer:
    def __init__(self, *, max_gap_ms=500, min_release_speed=40.0, min_airborne_ms=80,
                 face_confidence=0.65, stable_face_frames=2):
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError("max_gap_ms must be 1..10000")
        if type(min_release_speed) not in (int, float) or not math.isfinite(min_release_speed) or min_release_speed <= 0:
            raise ValueError("min_release_speed must be positive")
        if type(min_airborne_ms) is not int or not 1 <= min_airborne_ms <= 10000:
            raise ValueError("min_airborne_ms must be 1..10000")
        if type(face_confidence) not in (int, float) or not 0 <= face_confidence <= 1:
            raise ValueError("face_confidence must be 0..1")
        if type(stable_face_frames) is not int or not 1 <= stable_face_frames <= 10:
            raise ValueError("stable_face_frames must be 1..10")
        self.max_gap_ms = max_gap_ms
        self.min_release_speed = float(min_release_speed)
        self.min_airborne_ms = min_airborne_ms
        self.face_confidence = float(face_confidence)
        self.stable_face_frames = stable_face_frames
        version = hashlib.sha256(repr((max_gap_ms, min_release_speed, min_airborne_ms,
                                       face_confidence, stable_face_frames)).encode()).hexdigest()[:16]
        self.spec = self.capability(version=version, max_gap_ms=max_gap_ms)
        self._lock = threading.RLock()
        self.closed = False
        self.reset()

    @staticmethod
    def capability(*, version="installed", max_gap_ms=500):
        identifier(version, "model version")
        return CapabilitySpec(
            "vision.object_events", "vision.object_events", "video", "gamst.object_events", version,
            (OutputField("track_id", "string"), OutputField("state", "string"),
             OutputField("toss_count", "integer"), OutputField("flip_count", "integer"),
             OutputField("face", "string"), OutputField("events", "array")), "frame",
            stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms, interpretation="measurement",
            devices=("cpu",), sensor_requirements=("tracked-object-state",), license="Apache-2.0")

    def reset(self):
        self.owner = self.track_id = None
        self.last_ts = self.last_sequence = None
        self.state = "held"
        self.release_ts = None
        self.toss_count = self.flip_count = 0
        self.face = "unknown"
        self.face_candidate = None
        self.face_candidate_frames = 0

    def close(self):
        with self._lock:
            self.reset()
            self.closed = True

    @staticmethod
    def _bool_or_none(value, name):
        if value is not None and type(value) is not bool:
            raise ValueError(f"{name} must be boolean or null")
        return value

    def process(self, frame: InputFrame):
        with self._lock:
            if self.closed:
                raise RuntimeError("object event reducer is closed")
            if not isinstance(frame, InputFrame) or frame.modality != "video" or not isinstance(frame.sensor, SensorMetadata):
                raise ValueError("structured video InputFrame with sensor metadata required")
            if type(frame.timestamp_ms) is not int or frame.timestamp_ms < 0:
                raise ValueError("integer capture timestamp required")
            data = bounded_json(frame.data, max_bytes=16384)
            allowed = {"track_id", "support_contact", "vertical_velocity_px_s", "face", "face_confidence", "observed"}
            if not isinstance(data, dict) or set(data) - allowed:
                raise ValueError("object event data contains unsupported fields")
            track_id = data.get("track_id")
            if not isinstance(track_id, str) or not track_id:
                raise ValueError("track_id is required")
            observed = data.get("observed", True)
            if type(observed) is not bool:
                raise ValueError("observed must be boolean")
            support = self._bool_or_none(data.get("support_contact"), "support_contact")
            velocity = data.get("vertical_velocity_px_s")
            if velocity is not None and (type(velocity) not in (int, float) or not math.isfinite(velocity)):
                raise ValueError("vertical_velocity_px_s must be finite or null")
            label = data.get("face", "unknown")
            if not isinstance(label, str) or label not in ("heads", "tails", "edge", "unknown"):
                raise ValueError("face must be heads, tails, edge, or unknown")
            confidence = data.get("face_confidence", 0.0)
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError("face_confidence must be 0..1")
            owner = (frame.subject_id, frame.sensor.source_id, frame.sensor.sensor_id, frame.sensor.stream_epoch)
            sequence = frame.sensor.capture_sequence if frame.sensor.capture_sequence is not None else frame.sensor.sequence
            reset = (owner != self.owner or track_id != self.track_id or self.last_ts is None or
                     frame.timestamp_ms - self.last_ts > self.max_gap_ms or
                     self.last_sequence is not None and sequence <= self.last_sequence)
            if reset:
                self.reset()
                self.owner, self.track_id = owner, track_id
            self.last_ts, self.last_sequence = frame.timestamp_ms, sequence
            events = []
            if not observed:
                self.state = "unknown"
            elif support is False and velocity is not None and abs(velocity) >= self.min_release_speed:
                if self.state in ("held", "caught", "unknown"):
                    self.state = "released"
                    self.release_ts = frame.timestamp_ms
                    events.append({"event": "object.released", "track_id": track_id, "timestamp_ms": frame.timestamp_ms})
                elif self.state == "released" and self.release_ts is not None and frame.timestamp_ms - self.release_ts >= self.min_airborne_ms:
                    self.state = "airborne"
            elif support is True and self.state in ("released", "airborne") and self.release_ts is not None:
                if frame.timestamp_ms - self.release_ts >= self.min_airborne_ms:
                    self.state = "caught"
                    self.toss_count += 1
                    events.append({"event": "object.toss.completed", "index": self.toss_count,
                                   "timestamp_ms": frame.timestamp_ms, "duration_ms": frame.timestamp_ms - self.release_ts})
            if label in ("heads", "tails") and confidence >= self.face_confidence:
                if self.face_candidate == label:
                    self.face_candidate_frames += 1
                else:
                    self.face_candidate, self.face_candidate_frames = label, 1
                if self.face_candidate_frames >= self.stable_face_frames and self.face != label:
                    previous = self.face
                    self.face = label
                    if previous in ("heads", "tails"):
                        self.flip_count += 1
                        events.append({"event": "object.face_changed", "index": self.flip_count,
                                       "from": previous, "to": label, "timestamp_ms": frame.timestamp_ms})
            return self.spec.validate({"track_id": track_id, "state": self.state,
                "toss_count": self.toss_count, "flip_count": self.flip_count, "face": self.face,
                "events": events})
