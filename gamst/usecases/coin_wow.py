"""Coin WOW application composition over existing GAMST boundaries.

This is not a new capture architecture and does not change Engine/MCP/Store
contracts. The application owns camera access, a light detector/tracker and a
Qwen3.5-VL semantic perception client. It passes their bounded outputs into
the additive GAMST perception capabilities:

    semantic object list + tracked feature points
      -> ObjectRotationBank (all tracks)
      -> ObjectEventReducer (one reducer per track)
      -> engine.push_observation / Context / Watch

Qwen is not called here: it is an application-owned semantic perception input.
That keeps network/model policy outside the GAMST core while making the full
UseCase executable and testable with real or recorded frames.
"""
from __future__ import annotations

import threading

from ..capabilities import SensorMetadata
from ..plugins import InputFrame
from ..providers.object_events import ObjectEventReducer
from ..providers.object_rotation_bank import ObjectRotationBank


class CoinWowUseCase:
    """End-to-end object-rotation/toss/flip composition over a Gamst runtime."""

    def __init__(self, human, *, rotation_options=None, event_options=None):
        self.human = human
        self.rotation = ObjectRotationBank(**(rotation_options or {}))
        self.event_options = dict(event_options or {})
        self.events: dict[str, ObjectEventReducer] = {}
        self.lock = threading.RLock()
        self.closed = False
        human.register_capability(self.rotation.spec)

    def _event_reducer(self, track_id: str) -> ObjectEventReducer:
        reducer = self.events.get(track_id)
        if reducer is None:
            reducer = ObjectEventReducer(**self.event_options)
            self.events[track_id] = reducer
            self.human.register_capability(reducer.spec)
        return reducer

    @staticmethod
    def _objects(semantic_objects):
        if not isinstance(semantic_objects, list) or len(semantic_objects) > 128:
            raise ValueError("semantic_objects must contain at most 128 objects")
        output = []
        for item in semantic_objects:
            if not isinstance(item, dict):
                raise ValueError("semantic object must be an object")
            required = {"track_id", "center", "points"}
            if not required <= set(item):
                raise ValueError("semantic object requires track_id, center and points")
            track_id = item["track_id"]
            if not isinstance(track_id, str) or not track_id:
                raise ValueError("semantic object track_id is required")
            # Semantic fields are evidence supplied by the application/Qwen;
            # the numeric providers only consume their declared measurements.
            output.append(item)
        return output

    async def process(self, *, subject_id: str, timestamp_ms: int, sensor: SensorMetadata,
                      semantic_objects: list[dict]) -> dict:
        """Process one camera timestamp and commit all derived Context observations.

        ``semantic_objects`` is the application boundary: it may contain labels,
        rotating-state hints and heads/tails from Qwen3.5-VL plus detector/tracker
        measurements. No raw frame is accepted or persisted here.
        """
        with self.lock:
            if self.closed:
                raise RuntimeError("CoinWowUseCase is closed")
            objects = self._objects(semantic_objects)
        rotation_objects = [{key: item[key] for key in ("track_id", "center", "points", "feature_scores") if key in item}
                            for item in objects]
        frame = InputFrame({"objects": rotation_objects}, "video", timestamp_ms, subject_id=subject_id, sensor=sensor)
        rotation = self.rotation.process(frame)
        await self.human.push_observation(self.rotation.spec.id, rotation, sensor=sensor,
                                          subject_id=subject_id, timestamp_ms=timestamp_ms)
        event_values = {}
        for item in objects:
            measurements = {key: item[key] for key in
                            ("support_contact", "vertical_velocity_px_s", "face", "face_confidence", "observed")
                            if key in item}
            if not measurements:
                continue
            measurements["track_id"] = item["track_id"]
            reducer = self._event_reducer(item["track_id"])
            event_frame = InputFrame(measurements, "video", timestamp_ms,
                                     subject_id=subject_id, sensor=sensor)
            value = reducer.process(event_frame)
            event_values[item["track_id"]] = value
            await self.human.push_observation(reducer.spec.id, value, sensor=sensor,
                                              subject_id=subject_id, timestamp_ms=timestamp_ms)
        return {"rotation": rotation, "events": event_values,
                "semantic_objects": len(objects), "track_count": len(self.events)}

    def status(self):
        return {"closed": self.closed, "rotation_capability": self.rotation.spec.id,
                "event_tracks": sorted(self.events), "semantic_input_only": True}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.rotation.close()
            for reducer in self.events.values():
                reducer.close()
            self.events.clear()
