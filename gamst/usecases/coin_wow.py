"""Coin Halo: object spin and rotation counting composed over existing GAMST boundaries.

The application owns capture, the detector process and presentation. GAMST
owns the high-rate perception and its evidence:

* ``detections`` accepts labelled boxes from a semantic detector (YOLOE through
  ProcessProvider) at the detector's own cadence. Labels come only from a model;
  the geometric fallback is called ``round candidate``.
* ``process_frame`` re-finds every detected object in each capture, fits its
  apparent ellipse and feature points, and commits three observations:
  ``vision.rotation_tracks`` (regions), ``vision.object_spin`` (out-of-plane
  half turns: a flipped or table-spun coin) and ``vision.object_rotations``
  (in-plane turns: a wheel facing the camera).
* ``process`` accepts already-tracked measurements from another provider and
  drives the same rotation/event capabilities.

Counts are withheld as ``undersampled`` when the capture rate cannot resolve
the rotation; GAMST never fills that gap with a guess. A VLM/LLM is consulted
only by an owner's Watch when a spin episode ends.
"""
from __future__ import annotations

import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata
from ..plugins import InputFrame
from ..providers.object_events import ObjectEventReducer
from ..providers.object_rotation_bank import ObjectRotationBank
from ..providers.object_spin import ObjectSpinCounter


class CoinWowUseCase:
    """End-to-end object spin/rotation/flip composition over a Gamst runtime."""

    def __init__(self, human, *, rotation_options=None, spin_options=None, event_options=None, tracks_options=None):
        self.human = human
        self.rotation = ObjectRotationBank(**(rotation_options or {}))
        self.spin = ObjectSpinCounter(**(spin_options or {}))
        self.tracks_options = dict(tracks_options or {})
        self.tracks = None  # structured-input callers do not need OpenCV
        self.event_options = dict(event_options or {})
        self.events: dict[str, ObjectEventReducer] = {}
        self.lock = threading.RLock()
        self.closed = False
        human.register_capability(self.rotation.spec)
        human.register_capability(self.spin.spec)
        self.track_spec = CapabilitySpec('vision.rotation_tracks', 'vision.rotation_tracks', 'video',
            'gamst.shape_tracks', '2', (OutputField('regions', 'array'),), 'frame',
            interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        human.register_capability(self.track_spec)

    def _shape_tracks(self):
        if self.tracks is None:
            from ..providers.shape_tracks import ShapeTracks
            self.tracks = ShapeTracks(**self.tracks_options)
        return self.tracks

    def detections(self, items, *, image_shape, timestamp_ms):
        """Feed one detector result captured at ``timestamp_ms`` (normalized boxes)."""
        with self.lock:
            if self.closed:
                raise RuntimeError('CoinWowUseCase is closed')
            self._shape_tracks().detections(items, image_shape, timestamp_ms)

    async def process_frame(self, image, *, subject_id, timestamp_ms, sensor):
        """Pixels in, derived GAMST records out. No raw media in any record."""
        self.human._check()
        with self.lock:
            if self.closed:
                raise RuntimeError('CoinWowUseCase is closed')
            owner = (subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch)
            objects = self._shape_tracks().observe(image, timestamp_ms, owner)
        regions = [{k: item[k] for k in ('track_id', 'label', 'score', 'box', 'observed')} for item in objects]
        tracks = await self.human.push_observation(self.track_spec.id, {'regions': regions},
            sensor=sensor, subject_id=subject_id, timestamp_ms=timestamp_ms)
        spin = self.spin.process(InputFrame({'objects': [
            {'track_id': o['track_id'], 'axes': o['axes'], 'observed': o['observed']} for o in objects]},
            'video', timestamp_ms, subject_id=subject_id, sensor=sensor))
        spin_record = await self.human.push_observation(self.spin.spec.id, spin, sensor=sensor,
            subject_id=subject_id, timestamp_ms=timestamp_ms)
        result = await self.process(subject_id=subject_id, timestamp_ms=timestamp_ms, sensor=sensor,
            semantic_objects=[{k: o[k] for k in ('track_id', 'center', 'points', 'points_epoch')}
                              for o in objects if o['observed'] and len(o['points']) >= 4])
        return dict(result, objects=objects, tracks_record=tracks, spin=spin, spin_record=spin_record)

    def reset_capture(self):
        with self.lock:
            if self.tracks is not None:
                self.tracks.reset()
            self.rotation.reset()
            self.spin.reset()
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
            # Semantic fields are evidence supplied by the application/VLM;
            # the numeric providers only consume their declared measurements.
            output.append(item)
        return output

    async def process(self, *, subject_id: str, timestamp_ms: int, sensor: SensorMetadata,
                      semantic_objects: list[dict]) -> dict:
        """Process one camera timestamp of tracked measurements and commit Context observations.

        ``semantic_objects`` is the application boundary: detector/tracker
        measurements plus optional VLM hints such as heads/tails. No raw frame
        is accepted or persisted here.
        """
        with self.lock:
            if self.closed:
                raise RuntimeError("CoinWowUseCase is closed")
            objects = self._objects(semantic_objects)
        rotation_objects = [{key: item[key] for key in ("track_id", "center", "points", "feature_scores", "points_epoch")
                             if key in item} for item in objects]
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
            value = reducer.process(InputFrame(measurements, "video", timestamp_ms, subject_id=subject_id, sensor=sensor))
            event_values[item["track_id"]] = value
            event_records[item['track_id']] = await self.human.push_observation(
                reducer.spec.id, value, sensor=sensor, subject_id=subject_id, timestamp_ms=timestamp_ms)
        return {"rotation": rotation, "rotation_record": record, "events": event_values,
                "event_records": event_records, "semantic_objects": len(objects), "track_count": len(objects)}

    def status(self):
        return {"closed": self.closed, "rotation_capability": self.rotation.spec.id,
                "spin_capability": self.spin.spec.id, "event_tracks": sorted(self.events),
                "image_perception_active": self.tracks is not None}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.rotation.close()
            self.spin.close()
            if self.tracks is not None:
                self.tracks.reset()
            for reducer in self.events.values():
                reducer.close()
            self.events.clear()
