"""Bounded observed image trajectories; association and camera motion are caller-owned."""
import hashlib
import json
import math
import re
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame


class ImageTrajectories:
    """Process complete, already-associated normalized-image track snapshots locally."""

    def __init__(self, *, max_gap_ms=1000, max_samples=8):
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError('max_gap_ms must be 1..10000')
        if type(max_samples) is not int or not 2 <= max_samples <= 8:
            raise ValueError('max_samples must be 2..8')
        self._gap, self._samples = max_gap_ms, max_samples
        version = '1.' + hashlib.sha256(json.dumps([max_gap_ms, max_samples]).encode()).hexdigest()
        self.spec = CapabilitySpec('vision.image_trajectories', 'vision.trajectories', 'video',
            'gamst.image_trajectories', version,
            (OutputField('objects', 'array'), OutputField('geometry', 'object')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms,
            interpretation='estimate', devices=('cpu',),
            sensor_requirements=('associated-image-tracks', 'declared-camera-motion'), license='Apache-2.0')
        self._lock = threading.RLock()
        self._closed = False
        self._clear()

    def _clear(self):
        self._owner = self._last = None
        self._tracks = {}

    def reset(self):
        with self._lock:
            self._clear()

    def close(self):
        with self._lock:
            self._clear()
            self._closed = True

    def process(self, frame):
        with self._lock:
            if self._closed:
                raise RuntimeError('trajectory provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _process(self, frame):
        if (not isinstance(frame, InputFrame) or frame.modality != 'video'
                or not isinstance(frame.sensor, SensorMetadata)):
            raise ValueError('structured video InputFrame with sensor required')
        identifier(frame.subject_id)
        sensor = frame.sensor
        if sensor.coordinate_frame is None or sensor.stream_epoch is None:
            raise ValueError('explicit image coordinate frame and stream epoch required')
        ts = frame.timestamp_ms
        if type(ts) is not int or not 0 <= ts <= 2**53-1:
            raise ValueError('integer capture timestamp required')
        data = bounded_json(frame.data, max_bytes=16384)
        if type(data) is not dict or set(data) != {
                'units', 'image_size', 'tracker_id', 'tracker_version', 'camera_motion', 'objects'}:
            raise ValueError('exact complete snapshot schema required')
        if data['units'] != 'normalized_image' or data['camera_motion'] not in ('stationary', 'moving', 'cut', 'unknown'):
            raise ValueError('normalized image units and declared camera motion required')
        size = data['image_size']
        if type(size) is not list or len(size) != 2 or any(type(x) is not int or not 1 <= x <= 16384 for x in size):
            raise ValueError('image_size must be [width,height]')
        identifier(data['tracker_id']); identifier(data['tracker_version'])
        objects = data['objects']
        if type(objects) is not list or len(objects) > 8:
            raise ValueError('complete snapshot supports at most eight tracks')
        samples = {}
        for item in objects:
            if type(item) is not dict or set(item) != {'track_id', 'centroid', 'association_confidence', 'evidence_sha256'}:
                raise ValueError('exact observed track schema required')
            track = identifier(item['track_id'])
            if track in samples:
                raise ValueError('duplicate track')
            point = item['centroid']
            confidence = item['association_confidence']
            if type(point) is not list or len(point) != 2 or any(
                    type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in point):
                raise ValueError('centroid must be normalized [x,y]')
            if confidence is not None and (type(confidence) not in (int, float)
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError('association confidence must be null or 0..1')
            digest = item['evidence_sha256']
            if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
                raise ValueError('original evidence SHA256 required')
            samples[track] = dict(timestamp_ms=ts, centroid=point, association_confidence=confidence,
                                  evidence_sha256=digest, kind='observed')
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch,
                 sensor.coordinate_frame, sensor.calibration_id, sensor.channels, tuple(size),
                 data['tracker_id'], data['tracker_version'], sensor.capture_sequence is not None)
        seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        same = owner == self._owner
        if same and self._last and (ts <= self._last[0] or seq <= self._last[1]):
            raise ValueError('capture time and sequence must strictly increase')
        continuous = bool(same and self._last and seq == self._last[1]+1
                          and ts-self._last[0] <= self._gap
                          and self._last[2] == data['camera_motion'] == 'stationary')
        old = self._tracks if continuous else {}
        pending, output = {}, []
        for track, sample in samples.items():
            previous = old.get(track)
            segment = previous['segment_id'] if previous else uuid.uuid4().hex
            history = (previous['observations'] + [sample])[-self._samples:] if previous else [sample]
            velocity = interval = None
            if previous:
                last = previous['observations'][-1]
                dt = (ts-last['timestamp_ms'])/1000
                velocity = [(x-y)/dt for x, y in zip(sample['centroid'], last['centroid'])]
                interval = [last['timestamp_ms'], ts]
            pending[track] = dict(segment_id=segment, observations=history)
            output.append(dict(track_id=track, segment_id=segment, observations=history,
                velocity_image_s=velocity, velocity_interval_ms=interval,
                identity_verified=False, visibility='observed', predictions=[]))
        result = self.spec.validate(dict(objects=output, geometry=dict(
            units='normalized_image', image_size=size, coordinate_frame=sensor.coordinate_frame,
            timestamp_ms=ts, stream_epoch=sensor.stream_epoch, tracker_id=data['tracker_id'],
            tracker_version=data['tracker_version'], camera_motion=data['camera_motion'],
            continuity=continuous, complete_snapshot=True, metric_geometry=False,
            camera_motion_compensated=False, evidence_verified=False)))
        self._owner, self._last, self._tracks = owner, (ts, seq, data['camera_motion']), pending
        return result
