"""Metric track finite differences; no depth reconstruction or identity inference."""
import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame

_IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _vector(value, bound):
    if type(value) not in (list, tuple) or len(value) != 3 or any(
            type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound for x in value):
        raise ValueError('expected three finite bounded coordinates')
    return tuple(float(x) for x in value)


class MetricKinematics:
    """Consume complete snapshots of caller-associated 3D tracks in meters.

    process() accepts a local video InputFrame with structured data, not pixels.
    Publish the returned context using this instance's spec. Not a pixel worker.
    """
    def __init__(self, *, calibration_id, input_frame, output_frame,
                 rotation=_IDENTITY, translation_m=(0, 0, 0), max_gap_ms=1000):
        for name in (calibration_id, input_frame, output_frame):
            identifier(name)
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError('max_gap_ms must be 1..10000')
        if type(rotation) not in (list, tuple) or len(rotation) != 3:
            raise ValueError('rotation must be 3x3')
        rotation = tuple(_vector(row, 1) for row in rotation)
        for i in range(3):
            for j in range(3):
                if abs(sum(rotation[i][k]*rotation[j][k] for k in range(3)) - (i == j)) > 1e-6:
                    raise ValueError('rotation must be orthonormal')
        a, b, c = rotation
        det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(det-1) > 1e-6:
            raise ValueError('rotation must preserve handedness')
        translation = _vector(translation_m, 10000)
        if input_frame == output_frame and (rotation != _IDENTITY or any(translation)):
            raise ValueError('nonidentity transform requires different frame names')
        self.calibration_id, self.input_frame, self.output_frame = calibration_id, input_frame, output_frame
        self.rotation, self.translation, self.max_gap_ms = rotation, translation, max_gap_ms
        options = [calibration_id, input_frame, output_frame, rotation, translation, max_gap_ms]
        version = '1.' + hashlib.sha256(json.dumps(options).encode()).hexdigest()
        self.spec = CapabilitySpec('vision.metric_kinematics', 'vision.kinematics', 'video',
            'gamst.metric_kinematics', version, (OutputField('objects', 'array'), OutputField('geometry', 'object')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms,
            interpretation='estimate', devices=('cpu',),
            sensor_requirements=('metric-3d-tracks', 'fixed-rigid-transform', 'declared-calibration'), license='Apache-2.0')
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
                raise RuntimeError('kinematics provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _process(self, frame):
        if not isinstance(frame, InputFrame) or frame.modality != 'video' or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError('structured video InputFrame with sensor metadata required')
        identifier(frame.subject_id)
        sensor = frame.sensor
        if (sensor.coordinate_frame, sensor.calibration_id) != (self.input_frame, self.calibration_id):
            raise ValueError('coordinate frame and calibration must match configured metric geometry')
        ts = frame.timestamp_ms
        if type(ts) is not int or not 0 <= ts <= 2**53-1:
            raise ValueError('integer capture timestamp required')
        data = bounded_json(frame.data, max_bytes=16384)
        if type(data) is not dict or set(data) != {'units', 'tracker_id', 'tracker_version', 'objects'} or data['units'] != 'm':
            raise ValueError('require units=m, tracker_id, tracker_version and complete objects snapshot')
        identifier(data['tracker_id']); identifier(data['tracker_version'])
        objects = data['objects']
        if type(objects) is not list or len(objects) > 32:
            raise ValueError('complete snapshot must contain at most 32 tracks')
        positions = {}
        for item in objects:
            if type(item) is not dict or set(item) != {'track_id', 'position_m'}:
                raise ValueError('each track requires only track_id and position_m')
            track = identifier(item['track_id'])
            if track in positions:
                raise ValueError('duplicate track ID')
            xyz = _vector(item['position_m'], 10000)
            positions[track] = tuple(sum(row[k]*xyz[k] for k in range(3))+offset
                                    for row, offset in zip(self.rotation, self.translation))
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch,
                 sensor.coordinate_frame, sensor.calibration_id, data['tracker_id'], data['tracker_version'],
                 sensor.capture_sequence is not None)
        sequence = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        same = owner == self._owner
        if same and self._last and (ts <= self._last[0] or sequence <= self._last[1]):
            raise ValueError('capture timestamps and sequence must strictly increase')
        continuous = bool(same and self._last and sequence == self._last[1]+1 and ts-self._last[0] <= self.max_gap_ms)
        previous = self._tracks if continuous else {}
        output, pending = [], {}
        for track, position in positions.items():
            distance = math.sqrt(sum(x*x for x in position))
            old = previous.get(track)
            velocity = acceleration = approach = midpoint = interval = None
            if old is not None:
                dt = (ts-self._last[0])/1000
                velocity = tuple((x-y)/dt for x, y in zip(position, old['position']))
                interval = [self._last[0], ts]
                # Relative time avoids losing sub-ms midpoint precision for large timestamps.
                midpoint = self._last[0] + (ts-self._last[0])/2
                approach = (old['range']-distance)/dt
                if old['velocity'] is not None:
                    # Velocity estimates belong to interval midpoints; unequal intervals matter.
                    midpoint_dt = (ts-old['interval_start'])/2000
                    acceleration = [(x-y)/midpoint_dt for x,y in zip(velocity,old['velocity'])]
            output.append(dict(track_id=track, position_m=list(position), range_to_output_origin_m=distance,
                velocity_m_s=None if velocity is None else list(velocity),
                speed_m_s=None if velocity is None else math.sqrt(sum(x*x for x in velocity)),
                acceleration_m_s2=acceleration, approach_to_output_origin_m_s=approach,
                velocity_interval_ms=interval, velocity_midpoint_ms=midpoint,
                status='warmup' if velocity is None else 'estimated', identity_verified=False))
            pending[track] = dict(position=position, range=distance, velocity=velocity,
                                  interval_start=None if interval is None else interval[0])
        result = self.spec.validate(dict(objects=output, geometry=dict(
            coordinate_frame=self.output_frame, input_coordinate_frame=self.input_frame,
            calibration_id=self.calibration_id, rotation=[list(r) for r in self.rotation],
            translation_m=list(self.translation), tracker_id=data['tracker_id'], tracker_version=data['tracker_version'],
            timestamp_ms=ts, complete_snapshot=True, continuity=continuous,
            calibration_verified=False, stationary_output_frame_required=True,
            contact_inferred=False, force_measured=False, smoothing_applied=False)))
        self._owner, self._last, self._tracks = owner, (ts, sequence), pending
        return result
