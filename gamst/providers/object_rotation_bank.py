"""Per-track fan-out for the generic AngularMotionCounter."""
from __future__ import annotations

import hashlib
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from .angular_motion import AngularMotionCounter


class ObjectRotationBank:
    """Run one AngularMotionCounter per externally tracked object."""

    def __init__(self, *, max_tracks=32, max_gap_ms=500, min_radius=2.0,
                 max_points=128, min_quality=.25, history_s=10):
        if type(max_tracks) is not int or not 1 <= max_tracks <= 128:
            raise ValueError("max_tracks must be 1..128")
        self.max_tracks = max_tracks
        self.options = dict(max_gap_ms=max_gap_ms, min_radius=min_radius, max_points=max_points,
                            min_quality=min_quality, history_s=history_s)
        digest = hashlib.sha256(repr((max_tracks, self.options)).encode()).hexdigest()[:16]
        self.spec = self.capability(version="1." + digest)
        self._lock = threading.RLock()
        self.closed = False
        self.counters = {}

    @staticmethod
    def capability(*, version="installed"):
        identifier(version, "model version")
        return CapabilitySpec("vision.object_rotations", "vision.object_rotations", "video",
            "gamst.angular_motion.bank", version, (OutputField("objects", "array"),), "frame",
            stateful=True, reset_on_gap=True, freshness_ms=500, interpretation="measurement",
            devices=("cpu",), sensor_requirements=("multi-object-tracks", "corresponding-feature-points"),
            license="Apache-2.0")

    def reset(self):
        for counter in self.counters.values():
            counter.reset()

    def close(self):
        with self._lock:
            self.counters.clear()
            self.closed = True

    def process(self, frame: InputFrame):
        with self._lock:
            if self.closed:
                raise RuntimeError("object rotation bank is closed")
            if not isinstance(frame, InputFrame) or frame.modality != "video" or not isinstance(frame.sensor, SensorMetadata):
                raise ValueError("structured video InputFrame with sensor metadata required")
            data = bounded_json(frame.data, max_bytes=256 * 1024)
            if not isinstance(data, dict) or set(data) != {"objects"} or type(data["objects"]) is not list:
                raise ValueError("data must contain only an objects list")
            if len(data["objects"]) > self.max_tracks:
                raise ValueError("too many tracked objects")
            live = set()
            output = []
            for item in data["objects"]:
                if not isinstance(item, dict) or set(item) - {"track_id", "center", "points", "feature_scores"}:
                    raise ValueError("object contains unsupported fields")
                track_id = item.get("track_id")
                if not isinstance(track_id, str) or not track_id:
                    raise ValueError("object track_id is required")
                live.add(track_id)
                counter = self.counters.setdefault(track_id, AngularMotionCounter(**self.options))
                value = counter.process(InputFrame({"track_id": track_id, "center": item["center"],
                    "points": item["points"], "feature_scores": item.get("feature_scores")},
                    "video", frame.timestamp_ms, subject_id=frame.subject_id, sensor=frame.sensor))
                output.append(value)
            for track_id in set(self.counters) - live:
                self.counters[track_id].reset()
            return self.spec.validate({"objects": output})
