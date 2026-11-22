"""Coin WOW application composition over existing GAMST boundaries.

The UI owns capture and presentation. process_frame accepts BGR pixels and
uses SDK perception to discover/associate regions and measure image-plane
turns. process accepts explicitly tracked measurements from another provider.
Both use the same registered rotation capability, Store and Watch path. A
general semantic scene stream may run alongside them; it does not gate counts.
"""
from __future__ import annotations

import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata
from ..plugins import InputFrame
from ..providers.object_events import ObjectEventReducer
from ..providers.object_rotation_bank import ObjectRotationBank
from ..providers.rotation_tracks import RotationTracks


class CoinWowUseCase:
    """End-to-end object-rotation/toss/flip composition over a Gamst runtime."""

    def __init__(self, human, *, rotation_options=None, event_options=None, references=None):
        self.human = human
        self.rotation = ObjectRotationBank(**(rotation_options or {}))
        self.references = references
        self.rotation_tracks = None  # structured-input callers do not need OpenCV
        self.event_options = dict(event_options or {})
        self.events: dict[str, ObjectEventReducer] = {}
        self.lock = threading.RLock()
        self.closed = False
        human.register_capability(self.rotation.spec)
        self.track_spec = CapabilitySpec('vision.rotation_tracks', 'vision.rotation_tracks', 'video',
            'gamst.rotation_tracks', '1', (OutputField('regions', 'array'),), 'frame',
            interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        human.register_capability(self.track_spec)

    async def process_frame(self, image, *, subject_id, timestamp_ms, sensor):
        """Pixels in, derived GAMST records out. No raw media in either record."""
        self.human._check()
        if self.closed:
            raise RuntimeError('CoinWowUseCase is closed')
        if self.rotation_tracks is None:
            self.rotation_tracks = RotationTracks(self.references, max_tracks=self.rotation.max_tracks)
        owner = (subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch)
        objects = self.rotation_tracks.observe(image, timestamp_ms, owner)
        regions = [{k: item[k] for k in ('track_id', 'label', 'box', 'method')} for item in objects]
        tracks = await self.human.push_observation(self.track_spec.id, {'regions': regions},
            sensor=sensor, subject_id=subject_id, timestamp_ms=timestamp_ms)
        result = await self.process(subject_id=subject_id, timestamp_ms=timestamp_ms, sensor=sensor,
                                    semantic_objects=objects)
        return dict(result, tracks_record=tracks)

    def reset_capture(self):
        if self.rotation_tracks is not None:
            self.rotation_tracks.reset()
        self.rotation.reset()
        for reducer in self.events.values():
            reducer.close()
        self.events.clear()

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
        record = await self.human.push_observation(self.rotation.spec.id, rotation, sensor=sensor,
                                                   subject_id=subject_id, timestamp_ms=timestamp_ms)
        event_values, event_records = {}, {}
        for track_id in self.events.keys() - {item['track_id'] for item in objects}:
            self.events.pop(track_id).close()
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
            event_records[item['track_id']] = await self.human.push_observation(
                reducer.spec.id, value, sensor=sensor, subject_id=subject_id, timestamp_ms=timestamp_ms)
        return {"rotation": rotation, "rotation_record": record, "events": event_values,
                "event_records": event_records, "semantic_objects": len(objects), "track_count": len(objects)}

    def status(self):
        return {"closed": self.closed, "rotation_capability": self.rotation.spec.id,
                "event_tracks": sorted(self.events), "image_perception_active": self.rotation_tracks is not None}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.rotation.close()
            if self.rotation_tracks is not None:
                self.rotation_tracks.reset()
            for reducer in self.events.values():
                reducer.close()
            self.events.clear()
