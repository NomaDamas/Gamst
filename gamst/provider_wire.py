"""Bounded local pipe framing. This is not the public context/MCP format."""
from dataclasses import asdict
import json
import math
import struct

from .capabilities import SensorMetadata, bounded_json
from .plugins import InputFrame

HEADER_LIMIT = 131072
PAYLOAD_LIMIT = 16*1024*1024
DTYPES = {'|u1': 1, '<i2': 2, '>i2': 2, '<f4': 4, '>f4': 4, '<f8': 8, '>f8': 8}
# Large frames (video) may travel through one shared memory segment the owner creates and reuses, instead of the
# pipe: a 1280x720 frame takes about 8 ms through the pipe on an M-series Mac, a copy into shared memory under 1 ms.
# The owner asks for it at start; a worker that does not answer with the same name keeps the pipe.
FRAME_MEMORY = 'gamst.frame_memory.v1'
FRAME_MEMORY_MIN = 256 * 1024


def spec_value(spec):
    value = asdict(spec)
    for key in ('fields', 'devices', 'sensor_requirements'):
        value[key] = list(value[key])
    return bounded_json(value)


def envelope(header, payload=b''):
    body = json.dumps(bounded_json(header, max_bytes=HEADER_LIMIT), allow_nan=False,
                      ensure_ascii=True, separators=(',', ':')).encode()
    if len(body) > HEADER_LIMIT or len(payload) > PAYLOAD_LIMIT:
        raise ValueError('provider packet exceeds wire limits')
    return struct.pack('!II', len(body), len(payload))+body+payload


def sizes(prefix):
    head, payload = struct.unpack('!II', prefix)
    if not 1 <= head <= HEADER_LIMIT or payload > PAYLOAD_LIMIT:
        raise ValueError('provider packet exceeds wire limits')
    return head, payload


def parse_header(data):
    def no_constant(value):
        raise ValueError('nonfinite protocol number')
    return bounded_json(json.loads(data, parse_constant=no_constant), max_bytes=HEADER_LIMIT)


def encode_frame(frame):
    if not isinstance(frame, InputFrame) or not isinstance(frame.sensor, SensorMetadata):
        raise ValueError('InputFrame with SensorMetadata required')
    data = frame.data
    if isinstance(data, (bytes, bytearray, memoryview)):
        size = data.nbytes if isinstance(data, memoryview) else len(data)
        if size > PAYLOAD_LIMIT:
            raise ValueError('sensor payload too large')
        layout, payload = {'kind': 'bytes'}, bytes(data)
    elif hasattr(data, 'dtype') and hasattr(data, 'shape') and hasattr(data, 'tobytes'):
        import numpy as np
        if type(data) is not np.ndarray:
            raise ValueError('plain ndarray required; masks/subclass semantics are not serialized')
        dtype, shape = data.dtype.str, list(data.shape)
        count = array_size(dtype, shape)
        if data.nbytes != count:
            raise ValueError('invalid sensor array layout')
        payload = data.tobytes(order='C')
        layout = dict(kind='ndarray', dtype=dtype, shape=shape)
    else:
        layout, payload = dict(kind='json', value=bounded_json(data)), b''
    metadata = dict(modality=frame.modality, timestamp_ms=frame.timestamp_ms,
        subject_id=frame.subject_id, sample_rate=frame.sample_rate, sensor=asdict(frame.sensor))
    metadata['sensor']['channels'] = list(metadata['sensor']['channels'])
    return dict(metadata=metadata, layout=layout), payload


def array_size(dtype, shape):
    if dtype not in DTYPES or not isinstance(shape, list) or not 1 <= len(shape) <= 3:
        raise ValueError('unsupported sensor array dtype/dimensions')
    if any(type(dim) is not int or not 1 <= dim <= 192000 for dim in shape):
        raise ValueError('invalid sensor array dimensions')
    size = math.prod(shape)*DTYPES[dtype]
    if size > PAYLOAD_LIMIT:
        raise ValueError('sensor array exceeds byte limit')
    return size


class FrameMemory:
    """The owner's side: one shared memory segment, grown when a frame does not fit; only this process unlinks it."""

    def __init__(self):
        self.segment = None

    def put(self, frame):
        """(header, b'') of a frame whose array is written into the segment, or None when it goes by the pipe."""
        import numpy as np
        data = frame.data
        if type(data) is not np.ndarray or data.nbytes < FRAME_MEMORY_MIN or data.nbytes > PAYLOAD_LIMIT:
            return None
        header, _ = encode_frame(replace_data(frame, None))
        dtype, shape = data.dtype.str, list(data.shape)
        if array_size(dtype, shape) != data.nbytes:
            raise ValueError('invalid sensor array layout')
        if self.segment is None or self.segment.size < data.nbytes:
            from multiprocessing import shared_memory
            self.close()
            self.segment = shared_memory.SharedMemory(create=True, size=data.nbytes)
            _OWNED.add(self.segment._name)
        np.ndarray(data.shape, dtype=data.dtype, buffer=self.segment.buf)[...] = data
        header['layout'] = dict(kind='shared', name=self.segment.name, dtype=dtype, shape=shape)
        return header, b''

    def close(self):
        if self.segment is not None:
            segment, self.segment = self.segment, None
            _OWNED.discard(segment._name)
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


def replace_data(frame, data):
    from dataclasses import replace
    return replace(frame, data=data)


_ATTACHED = {}    # the worker's side: segments by name (the owner's, never unlinked here)
_OWNED = set()    # segments this process created (its own resource tracker removes them)


def _attached(name, size):
    from multiprocessing import resource_tracker, shared_memory
    segment = _ATTACHED.get(name)
    if segment is None:
        for other in _ATTACHED.values():   # the owner grew it: the old one is gone
            other.close()
        _ATTACHED.clear()
        segment = shared_memory.SharedMemory(name=name)
        if segment._name not in _OWNED:
            try:   # attaching must not make this process remove the owner's segment when it exits
                resource_tracker.unregister(segment._name, 'shared_memory')
            except Exception:
                pass
        _ATTACHED[name] = segment
    if segment.size < size:
        raise ValueError('shared frame is larger than its segment')
    return segment


def decode_frame(header, payload):
    metadata, layout = header['metadata'], header['layout']
    kind = layout['kind']
    if kind == 'shared' and not payload:
        if type(layout.get('name')) is not str or len(layout['name']) > 64:
            raise ValueError('invalid shared frame name')
        size = array_size(layout['dtype'], layout['shape'])
        import numpy as np
        segment = _attached(layout['name'], size)
        data = np.ndarray(layout['shape'], dtype=layout['dtype'], buffer=segment.buf).copy()
        kind, layout = 'decoded', None
    if kind == 'decoded':
        pass
    elif kind == 'bytes':
        data = payload
    elif kind == 'ndarray':
        if array_size(layout['dtype'], layout['shape']) != len(payload):
            raise ValueError('sensor array payload length mismatch')
        import numpy as np
        data = np.frombuffer(payload, dtype=layout['dtype']).reshape(layout['shape']).copy()
    elif kind == 'json' and not payload:
        data = bounded_json(layout['value'])
    else:
        raise ValueError('invalid sensor packet kind')
    sensor = dict(metadata['sensor'])
    sensor['channels'] = tuple(sensor.get('channels', ()))
    return InputFrame(data=data, modality=metadata['modality'], timestamp_ms=metadata['timestamp_ms'],
        subject_id=metadata['subject_id'], sample_rate=metadata['sample_rate'], sensor=SensorMetadata(**sensor))
