"""Bounded tactile acquisition evidence, not contact/slip/force inference."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame

MAGIC = b'GTACT01\n'
HEADER_LIMIT = 8192
PIXEL_LIMIT = 640 * 480 * 3
PACKET_LIMIT = len(MAGIC) + 4 + HEADER_LIMIT + PIXEL_LIMIT


def _integer(value, maximum, label):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('invalid ' + label)
    return value


def _shape(width, height):
    if type(width) is not int or type(height) is not int or not 1 <= width <= 640 or not 1 <= height <= 480:
        raise ValueError('tactile geometry must fit 640x480')
    return width * height * 3


def _settings(fps, intensity):
    if fps is not None and (type(fps) is not int or fps not in (15, 30, 60)):
        raise ValueError('invalid declared fps')
    if intensity is not None and (type(intensity) is not int or not 0 <= intensity <= 15):
        raise ValueError('invalid declared illumination intensity')


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('SHA256 required')


def _strict_json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(blob, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('invalid tactile JSON') from exc


@dataclass(frozen=True)
class TactileSource:
    """Caller-declared sensor/reference binding; no physical calibration claim."""
    source_id: str
    sensor_id: str
    sensor_model: str
    serial: str
    source_epoch: str
    calibration_id: str
    illumination_id: str
    coordinate_frame: str
    clock_id: str
    width: int
    height: int
    reference: bytes
    color_order: str = 'BGR'
    subject_id: str = 'local-user'
    configured_fps: int | None = None
    configured_led_intensity: int | None = None

    def __post_init__(self):
        for name in ('source_id', 'sensor_id', 'serial', 'source_epoch', 'calibration_id',
                     'illumination_id', 'coordinate_frame', 'clock_id', 'subject_id'):
            identifier(getattr(self, name), name)
        if self.sensor_model not in ('digit', 'gelsight_mini') or self.color_order not in ('BGR', 'RGB'):
            raise ValueError('explicit supported tactile model and color order required')
        _settings(self.configured_fps, self.configured_led_intensity)
        if type(self.reference) is not bytes or len(self.reference) != _shape(self.width, self.height):
            raise ValueError('reference must be bounded raw uint8 bytes matching geometry/color order')

    def metadata(self):
        result = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != 'reference'}
        result['reference_sha256'] = hashlib.sha256(self.reference).hexdigest()
        return result


def tactile_frame(raw, source, *, sequence, timestamp_ms, read_start_ms, read_end_ms,
                  acquisition, backend, backend_version, backend_revision):
    """Pack already declared raw uint8 pixels; never resize/flip/subtract a reference."""
    if not isinstance(source, TactileSource): raise ValueError('TactileSource required')
    if type(raw) is not bytes or len(raw) != _shape(source.width, source.height):
        raise ValueError('raw tactile byte length mismatch')
    meta = source.metadata()
    meta.update(sequence=sequence, timestamp_ms=timestamp_ms, read_start_ms=read_start_ms,
                read_end_ms=read_end_ms, acquisition=acquisition, backend=backend,
                backend_version=backend_version, backend_revision=backend_revision,
                pixels_sha256=hashlib.sha256(raw).hexdigest() if type(raw) is bytes else '')
    _validate(meta, raw)
    header = json.dumps(meta, separators=(',', ':'), sort_keys=True).encode()
    if len(header) > HEADER_LIMIT: raise ValueError('tactile header too large')
    return _frame(MAGIC + struct.pack('!I', len(header)) + header + raw, meta)


def _validate(meta, raw):
    source_keys = set(TactileSource.__dataclass_fields__) - {'reference'}
    extra = {'reference_sha256', 'sequence', 'timestamp_ms', 'read_start_ms', 'read_end_ms',
             'acquisition', 'backend', 'backend_version', 'backend_revision', 'pixels_sha256'}
    if type(meta) is not dict or set(meta) != source_keys | extra:
        raise ValueError('exact tactile header fields required')
    for name in source_keys - {'width', 'height', 'sensor_model', 'color_order', 'configured_fps', 'configured_led_intensity'}:
        identifier(meta[name], name)
    _settings(meta['configured_fps'], meta['configured_led_intensity'])
    if meta['sensor_model'] not in ('digit', 'gelsight_mini') or meta['color_order'] not in ('BGR', 'RGB'):
        raise ValueError('unsupported tactile model/color order')
    if type(raw) is not bytes or len(raw) != _shape(meta['width'], meta['height']):
        raise ValueError('raw tactile byte length mismatch')
    for name in ('reference_sha256', 'pixels_sha256'): _sha(meta[name])
    if hashlib.sha256(raw).hexdigest() != meta['pixels_sha256']: raise ValueError('tactile pixels hash mismatch')
    _integer(meta['sequence'], 2**63-1, 'sequence')
    for name in ('timestamp_ms', 'read_start_ms', 'read_end_ms'):
        _integer(meta[name], 2**53-1, name)
    if not meta['read_start_ms'] <= meta['timestamp_ms'] <= meta['read_end_ms']:
        raise ValueError('timestamp must be inside declared acquisition interval')
    if meta['read_end_ms'] - meta['read_start_ms'] > 60000:
        raise ValueError('acquisition interval exceeds 60 seconds')
    if meta['acquisition'] not in ('host_read_end', 'recorded_capture'):
        raise ValueError('explicit acquisition clock semantics required')
    if meta['acquisition'] == 'host_read_end' and meta['timestamp_ms'] != meta['read_end_ms']:
        raise ValueError('host read timestamp must be read completion, not invented exposure time')
    for name in ('backend', 'backend_version', 'backend_revision'): identifier(meta[name], name)


def _frame(packet, meta):
    sensor = SensorMetadata(meta['source_id'], meta['sensor_id'], meta['sequence'],
        coordinate_frame=meta['coordinate_frame'], channels=tuple(meta['color_order'].lower()),
        calibration_id=meta['calibration_id'])
    return InputFrame(packet, 'tactile', meta['timestamp_ms'], meta['subject_id'], sensor=sensor)


def decode_tactile(packet):
    if type(packet) is not bytes or not len(MAGIC)+4 < len(packet) <= PACKET_LIMIT or not packet.startswith(MAGIC):
        raise ValueError('bounded tactile packet required')
    size = struct.unpack('!I', packet[len(MAGIC):len(MAGIC)+4])[0]
    if not 1 <= size <= HEADER_LIMIT or len(packet) < len(MAGIC)+4+size:
        raise ValueError('invalid tactile header size')
    split = len(MAGIC)+4+size
    meta, raw = _strict_json(packet[len(MAGIC)+4:split]), packet[split:]
    _validate(meta, raw)
    return meta, raw


def _bound_frame(frame):
    if not isinstance(frame, InputFrame) or frame.modality != 'tactile' or not isinstance(frame.sensor, SensorMetadata):
        raise ValueError('tactile InputFrame with sensor metadata required')
    meta, raw = decode_tactile(frame.data)
    expected = _frame(frame.data, meta)
    for name in ('source_id', 'sensor_id', 'sequence', 'coordinate_frame', 'channels', 'calibration_id'):
        if getattr(frame.sensor, name) != getattr(expected.sensor, name):
            raise ValueError('tactile packet/sensor provenance mismatch')
    if frame.sensor.capture_sequence not in (None, meta['sequence']):
        raise ValueError('tactile capture sequence mismatch')
    if type(frame.timestamp_ms) is not int or frame.timestamp_ms != expected.timestamp_ms or frame.subject_id != expected.subject_id or frame.sample_rate is not None:
        raise ValueError('tactile packet/frame provenance mismatch')
    return meta, raw


class TactileWindow:
    """Six consecutive captures, yielding current/t-5 without invented samples."""
    def __init__(self, *, max_gap_ms=100):
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 1000: raise ValueError('gap must be 1..1000ms')
        self.max_gap_ms = max_gap_ms
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=6)
        self.identity = None

    def push(self, frame):
        try: meta, _ = _bound_frame(frame)
        except Exception:
            self.reset()
            raise
        identity = tuple(meta[name] for name in TactileSource.__dataclass_fields__ if name != 'reference') + tuple(meta[name] for name in ('reference_sha256', 'backend', 'backend_version', 'backend_revision', 'acquisition'))
        reason = 'initial' if self.identity is None else None
        if self.identity is not None and identity != self.identity:
            self.reset(); reason = 'source_or_reference_change'
        if self.frames:
            previous, _ = _bound_frame(self.frames[-1])
            if meta['sequence'] <= previous['sequence'] or meta['timestamp_ms'] <= previous['timestamp_ms']:
                self.reset()
                raise ValueError('tactile sequence/time must increase; reset after discontinuity')
            if meta['sequence'] != previous['sequence']+1 or meta['timestamp_ms']-previous['timestamp_ms'] > self.max_gap_ms:
                self.reset(); reason = 'gap'
        self.identity = identity
        self.frames.append(frame)
        pair = (self.frames[-1], self.frames[0]) if len(self.frames) == 6 else None
        return meta, pair, reason


class TactileInputContext:
    """Input validity/provenance only. Task-trained perception remains separate."""
    def __init__(self, *, max_gap_ms=100):
        self.window = TactileWindow(max_gap_ms=max_gap_ms)
        self.closed = False
        self.spec = self.capability()

    @staticmethod
    def capability():
        return CapabilitySpec('tactile.input', 'tactile.input', 'tactile', 'gamst.tactile', '1',
            (OutputField('capture', 'object'),), 'frame', stateful=True, reset_on_gap=True,
            interpretation='measurement', devices=('cpu',), license='Apache-2.0',
            sensor_requirements=('explicit-tactile-sensor', 'declared-reference'))

    def reset(self): self.window.reset()
    def close(self): self.closed = True; self.reset()

    def process(self, frame):
        if self.closed: raise RuntimeError('tactile input provider closed')
        meta, pair, reason = self.window.push(frame)
        result = dict(meta, native_shape=[meta['height'], meta['width'], 3], dtype='uint8',
                      preprocessing='none', reference_verified_as_no_contact=False,
                      physical_calibration_verified=False, task_inference_performed=False,
                      window_ready=pair is not None, buffered_captures=len(self.window.frames),
                      reset_reason=reason, max_gap_ms=self.window.max_gap_ms)
        result['pair_timestamps_ms'] = [item.timestamp_ms for item in pair] if pair else []
        return {'capture': result}


def recording_frames(manifest):
    """Yield at most1024 hash-bound packets; no decoder, pickle, network or devices."""
    path = Path(manifest)
    if not path.is_absolute(): raise ValueError('absolute recording manifest required')
    with path.open('rb') as handle: payload = handle.read(262145)
    if len(payload) > 262144: raise ValueError('recording manifest too large')
    document = _strict_json(payload)
    if type(document) is not dict or set(document) != {'schema_version', 'packets'} or type(document['schema_version']) is not int or document['schema_version'] != 1:
        raise ValueError('invalid tactile recording manifest')
    entries = document['packets']
    if type(entries) is not list or not 1 <= len(entries) <= 1024: raise ValueError('recording needs 1..1024 packets')
    seen = set()
    root = path.resolve().parent
    for entry in entries:
        if type(entry) is not dict or set(entry) != {'file', 'sha256'}:
            raise ValueError('invalid packet entry')
        name = entry['file']
        if type(name) is not str or not re.fullmatch('[a-zA-Z0-9_-]{1,100}\\.gtact', name) or name in seen:
            raise ValueError('unique local packet basename required')
        seen.add(name); _sha(entry['sha256'])
        target = (root/name).resolve()
        if target.parent != root: raise ValueError('packet escapes recording directory')
        with target.open('rb') as handle: packet = handle.read(PACKET_LIMIT+1)
        if len(packet) > PACKET_LIMIT or hashlib.sha256(packet).hexdigest() != entry['sha256']:
            raise ValueError('recording packet size/hash mismatch')
        meta, _ = decode_tactile(packet)
        yield _frame(packet, meta)
