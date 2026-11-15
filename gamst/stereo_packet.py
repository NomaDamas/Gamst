"""Bounded paired BGR frames with explicit capture-clock declarations."""
from dataclasses import asdict, replace
import hashlib
import json
import struct

from .capabilities import SensorMetadata, bounded_json, identifier
from .plugins import InputFrame

MAGIC = b'GSTEREO1'
HEADER_LIMIT = 16384
PIXEL_LIMIT = 1280*720*3


def _interval(value):
    if (type(value) is not list or len(value) != 2 or any(
            type(t) is not int or not 0 <= t <= 2**53-1 for t in value)
            or not 0 <= value[1]-value[0] <= 1000):
        raise ValueError('bounded capture interval required')


def _meta(value):
    if type(value) is not dict or set(value) != {'subject_id','sensor','interval_ms','sha256'}:
        raise ValueError('exact stereo side metadata required')
    identifier(value['subject_id'])
    _interval(value['interval_ms'])
    sensor = value['sensor']
    if type(sensor) is not dict or set(sensor) != set(SensorMetadata.__dataclass_fields__):
        raise ValueError('exact stereo SensorMetadata required')
    if type(sensor['channels']) is not list:
        raise ValueError('sensor channel list required')
    sensor = SensorMetadata(**dict(sensor, channels=tuple(sensor['channels'])))
    if any(getattr(sensor, k) is None for k in ('stream_epoch','coordinate_frame','calibration_id')):
        raise ValueError('stereo source epoch, optical coordinate frame and calibration required')
    digest = value['sha256']
    if (type(digest) is not str or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('lowercase stereo pixel SHA256 required')
    return sensor


def _validate(header):
    if type(header) is not dict or set(header) != {'width','height','clock_id','alignment_id','left','right'}:
        raise ValueError('exact stereo header required')
    w, h = header['width'], header['height']
    if type(w) is not int or type(h) is not int or not (96 <= w <= 1280 and 64 <= h <= 720):
        raise ValueError('calibrated stereo shape outside bounds')
    identifier(header['clock_id']); identifier(header['alignment_id'])
    ls, rs = _meta(header['left']), _meta(header['right'])
    if ((ls.source_id,ls.sensor_id) == (rs.source_id,rs.sensor_id)
            or header['left']['subject_id'] != header['right']['subject_id']
            or ls.calibration_id != rs.calibration_id):
        raise ValueError('distinct stereo sensors with shared subject/calibration required')
    return w*h*3


def stereo_frame(left, right, *, left_interval_ms, right_interval_ms, clock_id, alignment_id):
    """Copy caller frames into one immutable packet; never infer clock alignment.

    Capture intervals and a shared clock/alignment identity are caller assertions.
    The configured consumer separately checks time skew and source calibration.
    """
    import numpy as np
    owned, side, shapes = [], [], []
    for frame, interval in ((left,left_interval_ms),(right,right_interval_ms)):
        if (not isinstance(frame,InputFrame) or frame.modality != 'video'
                or frame.sample_rate is not None or not isinstance(frame.sensor,SensorMetadata)):
            raise ValueError('stereo video frames with metadata required')
        a = frame.data
        if (type(a) is not np.ndarray or a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3
                or not 64 <= a.shape[0] <= 720 or not 96 <= a.shape[1] <= 1280):
            raise ValueError('bounded plain BGR uint8 stereo image required')
        interval = bounded_json(interval, max_bytes=128)
        _interval(interval)
        if type(frame.timestamp_ms) is not int or frame.timestamp_ms != interval[1]:
            raise ValueError('frame timestamp must equal capture interval end')
        snapshot = a.copy(order='C')
        if (snapshot.dtype != np.uint8 or snapshot.ndim != 3 or snapshot.shape[2] != 3
                or not 64 <= snapshot.shape[0] <= 720 or not 96 <= snapshot.shape[1] <= 1280):
            raise ValueError('stereo input shape changed during snapshot')
        shapes.append(snapshot.shape)
        raw = snapshot.tobytes()
        owned.append(raw)
        sensor = asdict(frame.sensor); sensor['channels'] = list(sensor['channels'])
        side.append(dict(subject_id=frame.subject_id,sensor=sensor,interval_ms=interval,
                         sha256=hashlib.sha256(raw).hexdigest()))
    if shapes[0] != shapes[1]:
        raise ValueError('equal stereo image shapes required')
    header = dict(width=shapes[0][1],height=shapes[0][0],clock_id=clock_id,
                  alignment_id=alignment_id,left=side[0],right=side[1])
    _validate(header)
    encoded = json.dumps(header,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(encoded) > HEADER_LIMIT:
        raise ValueError('stereo header too large')
    return replace(left, data=MAGIC+struct.pack('>I',len(encoded))+encoded+b''.join(owned))


def unpack_stereo(data):
    """Validate immutable packet completely before exposing pixel slices."""
    if type(data) is not bytes or not 12 < len(data) <= 12+HEADER_LIMIT+2*PIXEL_LIMIT or data[:8] != MAGIC:
        raise ValueError('bounded immutable stereo packet required')
    n, = struct.unpack('>I',data[8:12])
    if not 0 < n <= HEADER_LIMIT or 12+n >= len(data):
        raise ValueError('stereo header length mismatch')
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('duplicate stereo field')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite stereo JSON')
    try:
        header = json.loads(data[12:12+n].decode('utf-8'),object_pairs_hook=pairs,parse_constant=reject)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid stereo JSON') from exc
    count = _validate(header)
    if len(data) != 12+n+2*count:
        raise ValueError('stereo pixel layout mismatch')
    left, right = data[12+n:12+n+count],data[12+n+count:]
    for name,raw in (('left',left),('right',right)):
        if hashlib.sha256(raw).hexdigest() != header[name]['sha256']:
            raise ValueError('stereo pixels differ from declared hash')
    return header,left,right
