"""Reduced-order, uncertainty-carrying metric physics; no learned world rollout."""
import hashlib
import json
import math
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from .kinematics import MetricKinematics

_IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
_STATE_ORDER = ('x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
                'ax_m_s2', 'ay_m_s2', 'az_m_s2')
_MAX_TRACKS = 8
_MAX_PLANES = 4
_MAX_MODAL_EVENTS = 8
_MAX_CONTACTS = 12
_MAX_RELATIONS = 16
_MAX_ALIGNED_EVENTS = 4


def _number(value, bound, label, *, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > bound:
        raise ValueError(label + ' must be finite and bounded')
    if minimum is not None and value < minimum:
        raise ValueError(label + ' is below its supported minimum')
    return float(value)


def _vector(value, length=3, bound=10000, label='vector'):
    if (type(value) not in (list, tuple) or len(value) != length
            or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound for x in value)):
        raise ValueError(label + ' must contain finite bounded coordinates')
    return tuple(float(x) for x in value)


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _scale(a, scale):
    return tuple(x*scale for x in a)


def _timestamp(value, label='timestamp'):
    if type(value) is not int or not 0 <= value <= 2**53-1:
        raise ValueError(label + ' must be an integer capture timestamp')
    return value


def _predict_axis(x, p, dt, spectral_density):
    """One position/velocity/acceleration axis driven by continuous white jerk."""
    half = .5*dt*dt
    nx = (x[0]+dt*x[1]+half*x[2], x[1]+dt*x[2], x[2])
    f = ((1., dt, half), (0., 1., dt), (0., 0., 1.))
    fp = [[sum(f[i][k]*p[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    result = [[sum(fp[i][k]*f[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    if dt and spectral_density:
        dt2, dt3, dt4, dt5 = dt*dt, dt**3, dt**4, dt**5
        q = ((dt5/20, dt4/8, dt3/6), (dt4/8, dt3/3, dt2/2), (dt3/6, dt2/2, dt))
        for i in range(3):
            for j in range(3):
                result[i][j] += spectral_density*q[i][j]
    return nx, result


def _kalman_predict(x, covariance, dt, spectral_density):
    nx, np = list(x), [row[:] for row in covariance]
    for axis in range(3):
        indices = (axis, axis+3, axis+6)
        old_x = [x[i] for i in indices]
        old_p = [[covariance[i][j] for j in indices] for i in indices]
        new_x, new_p = _predict_axis(old_x, old_p, dt, spectral_density)
        for i, index in enumerate(indices):
            nx[index] = new_x[i]
            for j, other in enumerate(indices):
                np[index][other] = new_p[i][j]
    return nx, np


def _initial_filter(position, variances, velocity_variance, acceleration_variance):
    x = [0.]*9
    covariance = [[0.]*9 for _ in range(9)]
    for axis in range(3):
        x[axis] = position[axis]
        covariance[axis][axis] = variances[axis]
        covariance[axis+3][axis+3] = velocity_variance
        covariance[axis+6][axis+6] = acceleration_variance
    return x, covariance


def _kalman_update(x, covariance, position, variances):
    """Joseph scalar position updates; input axes have independent noise."""
    nx, np = list(x), [row[:] for row in covariance]
    for axis in range(3):
        indices = (axis, axis+3, axis+6)
        old_x = [nx[i] for i in indices]
        old_p = [[np[i][j] for j in indices] for i in indices]
        innovation_variance = old_p[0][0]+variances[axis]
        if innovation_variance <= 0 or not math.isfinite(innovation_variance):
            raise ValueError('position covariance is not positive definite')
        gain = [row[0]/innovation_variance for row in old_p]
        residual = position[axis]-old_x[0]
        new_x = [old_x[i]+gain[i]*residual for i in range(3)]
        new_p = [[old_p[i][j]-gain[i]*old_p[0][j]-old_p[i][0]*gain[j]
                  + gain[i]*innovation_variance*gain[j] for j in range(3)] for i in range(3)]
        for i in range(3):
            new_p[i][i] = max(0., new_p[i][i])
            for j in range(i):
                value = (new_p[i][j]+new_p[j][i])/2
                new_p[i][j] = new_p[j][i] = value
        for i, index in enumerate(indices):
            nx[index] = new_x[i]
            for j, other in enumerate(indices):
                np[index][other] = new_p[i][j]
    return nx, np


def _position_covariance(covariance):
    return [[covariance[i][j] for j in range(3)] for i in range(3)]


def _support_radius(shape, normal):
    if shape['kind'] == 'sphere':
        return shape['radius_m']
    return sum(half*abs(n) for half, n in zip(shape['half_extents_m'], normal))


def _pair_geometry(a, b):
    """Return gap and a normal from a to b for bounded sphere/AABB shapes."""
    delta = _sub(b['position'], a['position'])
    sa, sb = a['shape'], b['shape']
    if sa['kind'] == sb['kind'] == 'sphere':
        distance = _norm(delta)
        if distance <= 1e-12:
            return None
        return distance-sa['radius_m']-sb['radius_m'], _scale(delta, 1/distance)
    if sa['kind'] == sb['kind'] == 'aabb':
        gaps = [abs(delta[i])-sa['half_extents_m'][i]-sb['half_extents_m'][i] for i in range(3)]
        # The largest separating/least penetration axis supplies the candidate normal.
        axis = max(range(3), key=lambda i: gaps[i])
        normal = [0., 0., 0.]
        normal[axis] = 1. if delta[axis] >= 0 else -1.
        return gaps[axis], tuple(normal)
    # Mixed shapes use a separating-axis candidate. Curved/AABB contact is
    # deliberately conservative; it is never promoted to measured contact.
    sphere, box = (a, b) if sa['kind'] == 'sphere' else (b, a)
    direction = 1 if sphere is a else -1
    local = _sub(sphere['position'], box['position'])
    closest = tuple(max(-h, min(h, x)) for x, h in zip(local, box['shape']['half_extents_m']))
    difference = _sub(local, closest)
    distance = _norm(difference)
    radius = sphere['shape']['radius_m']
    if distance > 1e-12:
        # Convert the sphere-to-box direction back to a-to-b.
        normal = _scale(difference, -direction/distance)
        return distance-radius, normal
    gaps = [abs(local[i])-box['shape']['half_extents_m'][i]-radius for i in range(3)]
    axis = max(range(3), key=lambda i: gaps[i])
    normal = [0., 0., 0.]
    normal[axis] = direction * (1. if local[axis] >= 0 else -1.)
    return gaps[axis], tuple(normal)


def _containment_clearance(container, contained):
    outer, inner = container['shape'], contained['shape']
    delta = _sub(contained['position'], container['position'])
    if outer['kind'] == inner['kind'] == 'sphere':
        return outer['radius_m']-_norm(delta)-inner['radius_m']
    if outer['kind'] == 'aabb' and inner['kind'] == 'sphere':
        return min(outer['half_extents_m'][i]-abs(delta[i])-inner['radius_m'] for i in range(3))
    if outer['kind'] == inner['kind'] == 'aabb':
        return min(outer['half_extents_m'][i]-abs(delta[i])-inner['half_extents_m'][i] for i in range(3))
    # AABB inside sphere: check the furthest box corner.
    corner_distance = max(_norm((delta[0]+sx*inner['half_extents_m'][0],
                                 delta[1]+sy*inner['half_extents_m'][1],
                                 delta[2]+sz*inner['half_extents_m'][2]))
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
    return outer['radius_m']-corner_distance


def _free_fall(position, velocity, elapsed, gravity, drag):
    if drag == 0:
        return (_add(position, _add(_scale(velocity, elapsed), _scale(gravity, .5*elapsed*elapsed))),
                _add(velocity, _scale(gravity, elapsed)))
    terminal = _scale(gravity, 1/drag)
    decay = math.exp(-drag*elapsed)
    displacement = _add(_scale(terminal, elapsed),
                        _scale(_sub(velocity, terminal), (1-decay)/drag))
    return _add(position, displacement), _add(terminal, _scale(_sub(velocity, terminal), decay))


def _ground_acceleration(velocity, gravity, normal, friction):
    normal_load = max(0., -_dot(gravity, normal))
    tangent_gravity = _sub(gravity, _scale(normal, _dot(gravity, normal)))
    speed = _norm(velocity)
    if speed > 1e-9:
        return _sub(tangent_gravity, _scale(velocity, friction*normal_load/speed))
    gravity_speed = _norm(tangent_gravity)
    if gravity_speed <= friction*normal_load or gravity_speed == 0:
        return (0., 0., 0.)
    return _scale(tangent_gravity, 1-friction*normal_load/gravity_speed)


def _grounded_step(position, velocity, dt, gravity, normal, friction):
    acceleration = _ground_acceleration(velocity, gravity, normal, friction)
    candidate = _add(velocity, _scale(acceleration, dt))
    speed = _norm(velocity)
    if speed > 1e-9 and _dot(candidate, velocity) < 0:
        direction = _scale(velocity, 1/speed)
        tangent_gravity = _sub(gravity, _scale(normal, _dot(gravity, normal)))
        stopping = friction*max(0., -_dot(gravity, normal))-_dot(tangent_gravity, direction)
        if stopping > 0:
            elapsed = min(dt, speed/stopping)
            return (_add(position, _add(_scale(velocity, elapsed),
                                       _scale(acceleration, .5*elapsed*elapsed))), (0., 0., 0.))
    return (_add(position, _add(_scale(velocity, dt), _scale(acceleration, .5*dt*dt))), candidate)


class PhysicalState:
    """Filter complete calibrated metric snapshots and emit physical candidates.

    The input is structured video metadata, not pixels. Audio and tactile items
    are already timestamped on the declared common clock. All geometry,
    dynamics and calibration assertions remain caller-owned and unverified.
    """
    def __init__(self, *, calibration_id, input_frame, output_frame,
                 rotation=_IDENTITY, translation_m=(0, 0, 0), clock_id,
                 gravity_m_s2=(0., 0., -9.80665), max_gap_ms=1000,
                 horizon_ms=600, prediction_steps=3, prediction_substep_ms=20,
                 jerk_spectral_density_m2_s5=10., initial_velocity_std_m_s=100.,
                 initial_acceleration_std_m_s2=100., restitution=0.,
                 contact_tolerance_m=.01, contact_speed_tolerance_m_s=.1,
                 event_window_ms=100, deceleration_threshold_m_s2=.5):
        identifier(clock_id, 'clock_id')
        gravity = _vector(gravity_m_s2, bound=100, label='gravity_m_s2')
        if _norm(gravity) == 0:
            raise ValueError('gravity must be nonzero')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError('max_gap_ms must be 1..10000')
        if type(horizon_ms) is not int or not 10 <= horizon_ms <= 2000:
            raise ValueError('horizon_ms must be 10..2000')
        if type(prediction_steps) is not int or not 1 <= prediction_steps <= 4:
            raise ValueError('prediction_steps must be 1..4')
        if type(prediction_substep_ms) is not int or not 1 <= prediction_substep_ms <= 100:
            raise ValueError('prediction_substep_ms must be 1..100')
        if type(event_window_ms) is not int or not 1 <= event_window_ms <= 1000:
            raise ValueError('event_window_ms must be 1..1000')
        self.gravity = gravity
        self.horizon_ms, self.prediction_steps = horizon_ms, prediction_steps
        self.substep_s = prediction_substep_ms/1000
        self.jerk_density = _number(jerk_spectral_density_m2_s5, 1e6,
                                    'jerk spectral density', minimum=0)
        self.initial_velocity_variance = _number(initial_velocity_std_m_s, 1e4,
            'initial velocity standard deviation', minimum=1e-9)**2
        self.initial_acceleration_variance = _number(initial_acceleration_std_m_s2, 1e4,
            'initial acceleration standard deviation', minimum=1e-9)**2
        self.restitution = _number(restitution, 1, 'restitution', minimum=0)
        self.contact_tolerance = _number(contact_tolerance_m, 1, 'contact tolerance', minimum=0)
        self.contact_speed_tolerance = _number(contact_speed_tolerance_m_s, 100,
                                               'contact speed tolerance', minimum=0)
        self.event_window_ms = event_window_ms
        self.deceleration_threshold = _number(deceleration_threshold_m_s2, 1000,
                                              'deceleration threshold', minimum=0)
        self.clock_id = clock_id
        self._kinematics = MetricKinematics(calibration_id=calibration_id,
            input_frame=input_frame, output_frame=output_frame, rotation=rotation,
            translation_m=translation_m, max_gap_ms=max_gap_ms)
        if self._kinematics.rotation != _IDENTITY:
            raise ValueError('physical state requires identity rotation for diagonal covariance and AABB semantics')
        options = [self._kinematics.spec.version, clock_id, gravity, horizon_ms,
                   prediction_steps, prediction_substep_ms, self.jerk_density,
                   self.initial_velocity_variance, self.initial_acceleration_variance,
                   restitution, self.contact_tolerance, self.contact_speed_tolerance,
                   event_window_ms, self.deceleration_threshold]
        version = '1.'+hashlib.sha256(json.dumps(options).encode()).hexdigest()
        self.spec = CapabilitySpec('physics.physical_state', 'physics.state', 'video',
            'gamst.physical_state', version,
            (OutputField('objects', 'array'), OutputField('contacts', 'array'),
             OutputField('relations', 'array'), OutputField('events', 'array'),
             OutputField('geometry', 'object'), OutputField('summary', 'object')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms,
            interpretation='estimate', devices=('cpu',),
            sensor_requirements=('calibrated-metric-3d-tracks', 'position-variance',
                                 'explicit-shape-and-dynamics', 'declared-common-clock'),
            license='Apache-2.0')
        self._lock = threading.RLock()
        self._closed = False
        self._clear()

    def _clear(self):
        self._tracks = {}
        self._visual_events = {}
        self._kinematics.reset()

    def reset(self):
        with self._lock:
            self._clear()

    def close(self):
        with self._lock:
            if not self._closed:
                self._clear()
                self._kinematics.close()
                self._closed = True

    def process(self, frame):
        with self._lock:
            if self._closed:
                raise RuntimeError('physical-state provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _parse_object(self, item):
        required = {'track_id', 'position_m', 'position_variance_m2', 'input_dimensionality',
                    'calibrated', 'shape', 'linear_drag_coefficient_s_inv', 'mass_kg',
                    'mass_calibrated', 'is_container'}
        if type(item) is not dict or set(item) != required:
            raise ValueError('exact physical track schema required')
        track = identifier(item['track_id'], 'track_id')
        if type(item['calibrated']) is not bool or type(item['mass_calibrated']) is not bool \
                or type(item['is_container']) is not bool or item['input_dimensionality'] not in (2, 3) \
                or type(item['input_dimensionality']) is not int:
            raise ValueError('explicit dimensionality and calibration flags required')
        if not item['calibrated']:
            if (item['position_m'] is not None or item['position_variance_m2'] is not None
                    or item['shape'] is not None or item['linear_drag_coefficient_s_inv'] is not None
                    or item['mass_kg'] is not None or item['mass_calibrated'] or item['is_container']):
                raise ValueError('uncalibrated tracks must abstain without metric physics fields')
            return dict(track_id=track, calibrated=False)
        if item['input_dimensionality'] != 3:
            raise ValueError('calibrated physical tracks must be 3D')
        position = _vector(item['position_m'], label='position_m')
        variance = _vector(item['position_variance_m2'], bound=1e6, label='position_variance_m2')
        if any(x <= 0 for x in variance):
            raise ValueError('position variance must be strictly positive')
        shape = item['shape']
        if type(shape) is not dict or shape.get('kind') not in ('sphere', 'aabb'):
            raise ValueError('sphere or axis-aligned box shape required')
        if shape['kind'] == 'sphere':
            if set(shape) != {'kind', 'radius_m'}:
                raise ValueError('exact sphere shape required')
            radius = _number(shape['radius_m'], 1000, 'sphere radius', minimum=1e-6)
            shape = dict(kind='sphere', radius_m=radius)
        else:
            if set(shape) != {'kind', 'half_extents_m'}:
                raise ValueError('exact AABB shape required')
            half = _vector(shape['half_extents_m'], bound=1000, label='half extents')
            if any(x <= 0 for x in half):
                raise ValueError('half extents must be positive')
            shape = dict(kind='aabb', half_extents_m=half)
        drag = _number(item['linear_drag_coefficient_s_inv'], 100,
                       'linear drag coefficient', minimum=0)
        mass = item['mass_kg']
        if item['mass_calibrated']:
            mass = _number(mass, 1e6, 'calibrated mass', minimum=1e-9)
        elif mass is not None:
            raise ValueError('uncalibrated mass must be null')
        return dict(track_id=track, calibrated=True, position=position, variance=variance,
                    shape=shape, drag=drag, mass_kg=mass,
                    mass_calibrated=item['mass_calibrated'], is_container=item['is_container'])

    def _parse_modal_events(self, data, timestamp):
        audio, tactile = data['audio_onsets'], data['tactile_events']
        if type(audio) is not list or len(audio) > _MAX_MODAL_EVENTS \
                or type(tactile) is not list or len(tactile) > _MAX_MODAL_EVENTS:
            raise ValueError('bounded audio/tactile event lists required')
        parsed_audio, parsed_tactile = [], []
        ids = set()
        for event in audio:
            required = {'event_id', 'timestamp_ms', 'impact_like', 'confidence', 'track_id'}
            if type(event) is not dict or set(event) != required or type(event['impact_like']) is not bool:
                raise ValueError('exact audio onset schema required')
            event_id = identifier(event['event_id'], 'audio event_id')
            if event_id in ids:
                raise ValueError('duplicate cross-modal event ID')
            ids.add(event_id)
            ts = _timestamp(event['timestamp_ms'], 'audio timestamp')
            if abs(ts-timestamp) > self.event_window_ms:
                raise ValueError('audio onset is outside the active alignment window')
            confidence = event['confidence']
            if confidence is not None:
                confidence = _number(confidence, 1, 'audio confidence', minimum=0)
            track = event['track_id']
            if track is not None:
                identifier(track, 'audio track_id')
            parsed_audio.append(dict(event_id=event_id, timestamp_ms=ts,
                impact_like=event['impact_like'], confidence=confidence, track_id=track))
        for event in tactile:
            required = {'event_id', 'timestamp_ms', 'track_id', 'force_spike', 'slip',
                        'force_n', 'force_calibrated'}
            if type(event) is not dict or set(event) != required \
                    or type(event['force_spike']) is not bool or type(event['force_calibrated']) is not bool \
                    or (event['slip'] is not None and type(event['slip']) is not bool):
                raise ValueError('exact tactile event schema required')
            event_id = identifier(event['event_id'], 'tactile event_id')
            if event_id in ids:
                raise ValueError('duplicate cross-modal event ID')
            ids.add(event_id)
            ts = _timestamp(event['timestamp_ms'], 'tactile timestamp')
            if abs(ts-timestamp) > self.event_window_ms:
                raise ValueError('tactile event is outside the active alignment window')
            force = event['force_n']
            if event['force_calibrated']:
                force = _number(force, 1e6, 'calibrated tactile force', minimum=0)
            elif force is not None:
                raise ValueError('uncalibrated tactile force must be null')
            parsed_tactile.append(dict(event_id=event_id, timestamp_ms=ts,
                track_id=identifier(event['track_id'], 'tactile track_id'),
                force_spike=event['force_spike'], slip=event['slip'],
                force_n=force, force_calibrated=event['force_calibrated']))
        return parsed_audio, parsed_tactile

    def _parse(self, frame):
        if (not isinstance(frame, InputFrame) or frame.modality != 'video' or frame.sample_rate is not None
                or not isinstance(frame.sensor, SensorMetadata)):
            raise ValueError('structured video InputFrame with sensor metadata required')
        timestamp = _timestamp(frame.timestamp_ms)
        if timestamp > 2**53-1-self.horizon_ms:
            raise ValueError('prediction timestamps exceed the supported range')
        data = bounded_json(frame.data, max_bytes=32768)
        required = {'units', 'tracker_id', 'tracker_version', 'clock_id', 'objects',
                    'planes', 'audio_onsets', 'tactile_events'}
        if type(data) is not dict or set(data) != required or data['units'] != 'm':
            raise ValueError('exact metric physical snapshot schema required')
        if data['clock_id'] != self.clock_id:
            raise ValueError('all modalities must use the configured common clock')
        identifier(data['tracker_id']); identifier(data['tracker_version'])
        objects = data['objects']
        if type(objects) is not list or len(objects) > _MAX_TRACKS:
            raise ValueError('complete physical snapshot supports at most eight tracks')
        parsed, ids = [], set()
        for item in objects:
            track = self._parse_object(item)
            if track['track_id'] in ids:
                raise ValueError('duplicate track ID')
            ids.add(track['track_id'])
            parsed.append(track)
        planes = data['planes']
        if type(planes) is not list or len(planes) > _MAX_PLANES:
            raise ValueError('at most four supporting planes are supported')
        parsed_planes, plane_ids = [], set()
        for plane in planes:
            required = {'plane_id', 'point_m', 'normal_unit', 'friction_coefficient', 'restitution'}
            if type(plane) is not dict or set(plane) != required:
                raise ValueError('exact supporting plane schema required')
            plane_id = identifier(plane['plane_id'], 'plane_id')
            if plane_id in plane_ids:
                raise ValueError('duplicate plane ID')
            plane_ids.add(plane_id)
            normal = _vector(plane['normal_unit'], label='plane normal')
            if abs(_norm(normal)-1) > 1e-6:
                raise ValueError('plane normal must be unit length')
            friction = plane['friction_coefficient']
            if friction is not None:
                friction = _number(friction, 10, 'friction coefficient', minimum=0)
            restitution = plane['restitution']
            if restitution is not None:
                restitution = _number(restitution, 1, 'plane restitution', minimum=0)
            parsed_planes.append(dict(plane_id=plane_id,
                point=_vector(plane['point_m'], label='plane point'), normal=normal,
                friction=friction, restitution=restitution))
        audio, tactile = self._parse_modal_events(data, timestamp)
        return data, parsed, parsed_planes, audio, tactile

    def _contacts(self, states, planes, tactile, timestamp):
        records = list(states.values())
        contacts = []
        for index, a in enumerate(records):
            for b in records[index+1:]:
                geometry = _pair_geometry(a, b)
                if geometry is None:
                    continue
                gap, normal = geometry
                if normal is None or gap > self.contact_tolerance:
                    continue
                relative = None
                if a['samples'] >= 2 and b['samples'] >= 2:
                    relative = _dot(_sub(b['velocity'], a['velocity']), normal)
                if relative is not None and relative > self.contact_speed_tolerance:
                    continue
                calibrated = a['mass_calibrated'] and b['mass_calibrated'] and relative is not None
                impulse = None
                if calibrated and relative < 0:
                    impulse = (-(1+self.restitution)*relative
                               / (1/a['mass_kg']+1/b['mass_kg']))
                contacts.append(self._contact_output('tracks', a['track_id'], b['track_id'],
                    gap, normal, relative, calibrated, impulse, tactile, timestamp, self.restitution))
        for track in records:
            for plane in planes:
                signed = _dot(_sub(track['position'], plane['point']), plane['normal'])
                gap = signed-_support_radius(track['shape'], plane['normal'])
                if gap > self.contact_tolerance:
                    continue
                relative = None
                if track['samples'] >= 2:
                    relative = _dot(track['velocity'], plane['normal'])
                if relative is not None and relative > self.contact_speed_tolerance:
                    continue
                calibrated = track['mass_calibrated'] and relative is not None
                restitution = self.restitution if plane['restitution'] is None else plane['restitution']
                impulse = None
                if calibrated and relative < 0:
                    impulse = -(1+restitution)*relative*track['mass_kg']
                contact = self._contact_output('plane', track['track_id'], plane['plane_id'],
                    gap, plane['normal'], relative, calibrated, impulse, tactile, timestamp, restitution)
                contact['friction_coefficient'] = plane['friction']
                contacts.append(contact)
        return contacts

    def _contact_output(self, kind, track_id, other_id, gap, normal, relative,
                        calibrated, impulse, tactile, timestamp, restitution):
        related = [event for event in tactile if event['track_id'] in (track_id, other_id)]
        impulse_status = 'model_estimate' if impulse is not None else (
            'abstained_mass_or_velocity_not_calibrated' if not calibrated
            else 'abstained_not_an_approaching_collision')
        return dict(kind='contact_estimate' if calibrated else 'contact_candidate',
            status='estimated' if relative is not None else 'geometry_only_velocity_warmup',
            geometry_kind=kind, track_id=track_id,
            other_track_id=other_id if kind == 'tracks' else None,
            plane_id=other_id if kind == 'plane' else None, timestamp_ms=timestamp,
            contact_time_status='frame_aligned_not_interpolated', gap_m=gap,
            penetration_m=max(0., -gap), normal_unit=list(normal),
            relative_normal_velocity_m_s=relative,
            approaching=None if relative is None else relative < 0,
            normal_impulse_n_s=impulse, impulse_status=impulse_status,
            normal_force_n=None, force_status='not_estimated_contact_duration_unknown',
            restitution=restitution, tactile_event_ids=[event['event_id'] for event in related],
            tactile_slip_observed=any(event['slip'] is True for event in related),
            contact_verified=False, physical_parameters_verified=False)

    def _relations(self, states, planes, contacts):
        relations = []
        for contact in contacts:
            if contact['relative_normal_velocity_m_s'] is None:
                continue
            normal = contact['normal_unit']
            if contact['geometry_kind'] == 'plane':
                if _dot(self.gravity, normal) < 0:
                    relations.append(dict(relation='support', supporter_type='plane',
                        supporter_id=contact['plane_id'], supported_track_id=contact['track_id'],
                        timestamp_ms=contact['timestamp_ms'], gap_m=contact['gap_m'],
                        status='geometric_contact_candidate', normal_force_n=None,
                        support_verified=False))
            elif _dot(self.gravity, normal) < 0:
                relations.append(dict(relation='support', supporter_type='track',
                    supporter_id=contact['track_id'], supported_track_id=contact['other_track_id'],
                    timestamp_ms=contact['timestamp_ms'], gap_m=contact['gap_m'],
                    status='geometric_contact_candidate', normal_force_n=None,
                    support_verified=False))
            elif _dot(self.gravity, normal) > 0:
                relations.append(dict(relation='support', supporter_type='track',
                    supporter_id=contact['other_track_id'], supported_track_id=contact['track_id'],
                    timestamp_ms=contact['timestamp_ms'], gap_m=contact['gap_m'],
                    status='geometric_contact_candidate', normal_force_n=None,
                    support_verified=False))
        records = list(states.values())
        for container in records:
            if not container['is_container']:
                continue
            for contained in records:
                if container is contained:
                    continue
                clearance = _containment_clearance(container, contained)
                if clearance >= 0:
                    relations.append(dict(relation='containment',
                        container_track_id=container['track_id'], contained_track_id=contained['track_id'],
                        timestamp_ms=contained['last_observed_ms'], clearance_m=clearance,
                        status='geometric_candidate', containment_verified=False))
        return relations

    def _prediction_envelopes(self, covariance):
        envelopes, current = [], covariance
        previous = 0
        for step in range(1, self.prediction_steps+1):
            horizon = self.horizon_ms*step//self.prediction_steps
            dt = (horizon-previous)/1000
            _, current = _kalman_predict([0.]*9, current, dt, self.jerk_density)
            position = _position_covariance(current)
            radius = 3*math.sqrt(max(0., sum(position[i][i] for i in range(3))))
            envelopes.append((horizon, position, radius))
            previous = horizon
        return envelopes

    def _hypotheses(self, track, contacts, planes, timestamp):
        position, velocity = track['position'], track['velocity']
        envelopes = self._prediction_envelopes(track['covariance'])
        free_samples = []
        for horizon, covariance, radius in envelopes:
            elapsed = horizon/1000
            p, v = _free_fall(position, velocity, elapsed, self.gravity, track['drag'])
            free_samples.append(self._prediction_sample(timestamp, horizon, p, v, covariance, radius))
        hypotheses = [dict(hypothesis='free_fall_gravity_linear_drag',
            status='estimated' if track['samples'] >= 2 else 'warming_up_prior_velocity',
            parameters=dict(gravity_m_s2=list(self.gravity),
                            linear_drag_coefficient_s_inv=track['drag']),
            parameters_verified=False, predictions=free_samples)]
        supports = [c for c in contacts if c['geometry_kind'] == 'plane'
                    and c['track_id'] == track['track_id']
                    and c['relative_normal_velocity_m_s'] is not None
                    and _dot(self.gravity, c['normal_unit']) < 0]
        supports.sort(key=lambda c: (c['gap_m'], c['plane_id']))
        grounded = dict(hypothesis='grounded_coulomb_friction',
            status='abstained_no_supporting_plane', support_plane_id=None,
            parameters=None, parameters_verified=False, predictions=[])
        if supports:
            contact = supports[0]
            plane = next(p for p in planes if p['plane_id'] == contact['plane_id'])
            grounded['support_plane_id'] = plane['plane_id']
            if plane['friction'] is None:
                grounded['status'] = 'abstained_friction_unknown'
            else:
                grounded['status'] = 'estimated' if track['samples'] >= 2 else 'warming_up_prior_velocity'
                grounded['parameters'] = dict(gravity_m_s2=list(self.gravity),
                    friction_coefficient=plane['friction'], normal_unit=list(plane['normal']),
                    integration_substep_ms=int(self.substep_s*1000))
                p, v = position, velocity
                # Remove any tiny numerical normal component at constraint entry.
                v = _sub(v, _scale(plane['normal'], _dot(v, plane['normal'])))
                previous = 0
                samples = []
                for horizon, covariance, radius in envelopes:
                    elapsed = previous/1000
                    target = horizon/1000
                    while elapsed < target-1e-12:
                        dt = min(self.substep_s, target-elapsed)
                        p, v = _grounded_step(p, v, dt, self.gravity,
                                             plane['normal'], plane['friction'])
                        elapsed += dt
                    samples.append(self._prediction_sample(timestamp, horizon, p, v, covariance, radius))
                    previous = horizon
                grounded['predictions'] = samples
        hypotheses.append(grounded)
        return hypotheses

    @staticmethod
    def _prediction_sample(timestamp, horizon, position, velocity, covariance, radius):
        return dict(timestamp_ms=timestamp+horizon, horizon_ms=horizon,
            position_m=list(position), velocity_m_s=list(velocity),
            position_covariance_m2=covariance, envelope_radius_m=radius,
            envelope_semantics='three_times_root_trace_position_covariance',
            covariance_model='constant_acceleration_white_jerk_linearized',
            coverage_calibrated=False)

    def _object_output(self, track, hypotheses):
        state = dict(position_m=list(track['position']), velocity_m_s=list(track['velocity']),
            acceleration_m_s2=list(track['acceleration']), orientation_quaternion_wxyz=None,
            angular_velocity_rad_s=None, angular_acceleration_rad_s2=None,
            orientation_status='abstained_not_estimated_from_translation')
        return dict(track_id=track['track_id'], segment_id=track['segment_id'],
            status='estimated' if track['samples'] >= 3 else 'initializing',
            sample_count=track['samples'], last_observed_ms=track['last_observed_ms'],
            state=state, covariance=dict(state_order=list(_STATE_ORDER),
                matrix=track['covariance'], covariance_semantics='symmetric_metric_state_covariance'),
            hypotheses=hypotheses, identity_verified=False, calibration_verified=False)

    @staticmethod
    def _abstained_output(track, timestamp):
        return dict(track_id=track['track_id'], segment_id=None, status='abstained',
            reason='calibrated_metric_3d_position_and_variance_required', sample_count=0,
            last_observed_ms=timestamp, state=None, covariance=None, hypotheses=[],
            identity_verified=False, calibration_verified=False)

    def _visual_event(self, track):
        if track['samples'] < 3:
            return None
        speed = _norm(track['velocity'])
        if speed <= 1e-9:
            return None
        deceleration = -_dot(track['acceleration'], track['velocity'])/speed
        if deceleration < self.deceleration_threshold:
            return None
        return dict(track_id=track['track_id'], timestamp_ms=track['last_observed_ms'],
            deceleration_m_s2=deceleration, speed_m_s=speed,
            acceleration_m_s2=list(track['acceleration']))

    def _aligned_events(self, audio, tactile):
        candidates = []
        for sound in audio:
            if not sound['impact_like']:
                continue
            for touch in tactile:
                if not touch['force_spike'] or sound['track_id'] not in (None, touch['track_id']):
                    continue
                for visual in self._visual_events.get(touch['track_id'], []):
                    times = [sound['timestamp_ms'], touch['timestamp_ms'], visual['timestamp_ms']]
                    span = max(times)-min(times)
                    if span > self.event_window_ms:
                        continue
                    candidates.append((span, sound['event_id'], touch['event_id'],
                                       visual['timestamp_ms'], sound, touch, visual))
        candidates.sort(key=lambda item: item[:4])
        events = []
        for span, _, _, _, sound, touch, visual in candidates[:_MAX_ALIGNED_EVENTS]:
            events.append(dict(kind='impact_candidate', track_id=touch['track_id'],
                evidence=dict(audio_event_id=sound['event_id'], tactile_event_id=touch['event_id'],
                    visual_timestamp_ms=visual['timestamp_ms']),
                timestamps_ms=dict(audio_onset=sound['timestamp_ms'], tactile=touch['timestamp_ms'],
                                   visual=visual['timestamp_ms']),
                alignment_span_ms=span, alignment_window_ms=self.event_window_ms,
                clock_id=self.clock_id, clock_alignment_verified=False,
                audio_impact_confidence=sound['confidence'],
                visible_deceleration_m_s2=visual['deceleration_m_s2'],
                tactile_force_n=touch['force_n'],
                tactile_force_status='calibrated_measurement' if touch['force_calibrated']
                                      else 'abstained_force_not_calibrated',
                tactile_slip=touch['slip'], event_verified=False,
                fused_probability=None))
        return events, len(candidates) > _MAX_ALIGNED_EVENTS

    def _process(self, frame):
        data, tracks, planes, audio, tactile = self._parse(frame)
        timestamp = frame.timestamp_ms
        metric_objects = [dict(track_id=t['track_id'], position_m=list(t['position']))
                          for t in tracks if t['calibrated']]
        metric_data = dict(units='m', tracker_id=data['tracker_id'],
                           tracker_version=data['tracker_version'], objects=metric_objects)
        metric_frame = InputFrame(metric_data, 'video', timestamp, frame.subject_id,
                                  sensor=frame.sensor)
        kinematics = self._kinematics.process(metric_frame)
        continuous = kinematics['geometry']['continuity']
        if not continuous:
            self._tracks = {}
            self._visual_events = {}
        transformed = {item['track_id']: item['position_m'] for item in kinematics['objects']}
        pending = {}
        for track in tracks:
            if not track['calibrated']:
                continue
            position = tuple(transformed[track['track_id']])
            previous = self._tracks.get(track['track_id'])
            if previous is None:
                x, covariance = _initial_filter(position, track['variance'],
                    self.initial_velocity_variance, self.initial_acceleration_variance)
                samples, segment = 1, uuid.uuid4().hex
            else:
                dt = (timestamp-previous['last_observed_ms'])/1000
                x, covariance = _kalman_predict(previous['state'], previous['covariance'],
                                                dt, self.jerk_density)
                x, covariance = _kalman_update(x, covariance, position, track['variance'])
                samples, segment = previous['samples']+1, previous['segment_id']
            pending[track['track_id']] = dict(track, position=position, velocity=tuple(x[3:6]),
                acceleration=tuple(x[6:9]), state=x, covariance=covariance,
                samples=samples, segment_id=segment, last_observed_ms=timestamp)
        self._tracks = pending
        visual_pending = {}
        for track_id, track in pending.items():
            history = list(self._visual_events.get(track_id, []))
            event = self._visual_event(track)
            if event is not None:
                history.append(event)
            visual_pending[track_id] = [item for item in history
                                        if timestamp-item['timestamp_ms'] <= self.event_window_ms][-4:]
        self._visual_events = visual_pending
        all_contacts = self._contacts(pending, planes, tactile, timestamp)
        all_relations = self._relations(pending, planes, all_contacts)
        objects = []
        for track in tracks:
            if not track['calibrated']:
                objects.append(self._abstained_output(track, timestamp))
            else:
                current = pending[track['track_id']]
                objects.append(self._object_output(current,
                    self._hypotheses(current, all_contacts, planes, timestamp)))
        events, events_truncated = self._aligned_events(audio, tactile)
        contact_key = lambda c: (c['track_id'], c['plane_id'] or '', c['other_track_id'] or '')
        relation_key = lambda r: json.dumps(r, sort_keys=True)
        contacts = sorted(all_contacts, key=contact_key)[:_MAX_CONTACTS]
        relations = sorted(all_relations, key=relation_key)[:_MAX_RELATIONS]
        geometry = dict(kinematics['geometry'], clock_id=self.clock_id,
            smoothing_applied=True, filter='constant_acceleration_kalman',
            state_order=list(_STATE_ORDER), jerk_spectral_density_m2_s5=self.jerk_density,
            gravity_m_s2=list(self.gravity), prediction_horizon_ms=self.horizon_ms,
            prediction_steps=self.prediction_steps, physical_parameters_verified=False,
            clock_alignment_verified=False, metric_geometry=True,
            orientation_estimated=False, force_estimated=False,
            cross_modal_input='caller_supplied_timestamps_not_raw_media')
        summary = dict(track_count=len(tracks), calibrated_track_count=len(pending),
            abstained_track_count=len(tracks)-len(pending), contact_count=len(all_contacts),
            relation_count=len(all_relations), aligned_event_count=len(events),
            contacts_truncated=len(all_contacts) > len(contacts),
            relations_truncated=len(all_relations) > len(relations),
            events_truncated=events_truncated, complete_snapshot=True,
            physical_event_verified=False, learned_world_model_used=False)
        return self.spec.validate(dict(objects=objects, contacts=contacts,
            relations=relations, events=events, geometry=geometry, summary=summary))
