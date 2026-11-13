"""Dependency-free sensory contracts. Registration never imports a provider.

OutputField is a bounded structural contract, not a JSON Schema implementation.
Provider-specific semantic validation (for example calibrated temperature) remains
the provider's responsibility. Raw sensor payloads never enter this registry.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import re

MODALITIES = frozenset(('video', 'audio', 'text', 'thermal', 'radar', 'csi',
                        'imu', 'tactile', 'environment', 'events'))
STREAMING_MODES = frozenset(('native', 'window', 'frame', 'offline'))
RAW_KEYS = frozenset(('raw', 'frame', 'samples', 'jpeg', 'image', 'audio', 'landmarks'))
_KINDS = {'string': str, 'number': (int, float), 'integer': int,
          'boolean': bool, 'object': dict, 'array': list}


def identifier(value, label='identifier'):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}', value):
        raise ValueError(f'{label} must be a 1..128 character identifier')
    return value


def bounded_json(value, *, max_bytes=65536, forbid_raw=False):
    """Copy strict JSON with bounded work before encoding; reject NaN and cycles."""
    remaining = 8192

    def walk(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 12:
            raise ValueError('observation exceeds structural bounds')
        if item is None or type(item) is bool:
            return item
        if type(item) in (int, float):
            if abs(item) > 1e100 or not math.isfinite(item):
                raise ValueError('observation numbers must be finite and bounded')
            return item
        if type(item) is str:
            if len(item) > 8192:
                raise ValueError('observation string too long')
            return item
        if type(item) is list:
            if len(item) > 2048:
                raise ValueError('observation array too long')
            return [walk(child, depth+1) for child in item]
        if type(item) is dict:
            if len(item) > 128:
                raise ValueError('observation object too large')
            result = {}
            for key, child in item.items():
                if not isinstance(key, str) or not 1 <= len(key) <= 128:
                    raise ValueError('observation keys must be bounded strings')
                if forbid_raw and key.lower() in RAW_KEYS:
                    raise ValueError('raw media fields are not context observations')
                result[key] = walk(child, depth+1)
            return result
        raise ValueError('observation must contain only JSON values')

    copied = walk(value)
    if len(json.dumps(copied, allow_nan=False, ensure_ascii=True).encode()) > max_bytes:
        raise ValueError('observation exceeds byte budget')
    return copied


@dataclass(frozen=True)
class OutputField:
    name: str
    kind: str
    unit: str | None = None
    required: bool = True

    def __post_init__(self):
        identifier(self.name, 'field name')
        if self.name.lower() in RAW_KEYS or self.kind not in _KINDS:
            raise ValueError('invalid context field or field kind')
        if type(self.required) is not bool:
            raise ValueError('required must be boolean')
        if self.unit is not None and (not isinstance(self.unit, str) or not 1 <= len(self.unit) <= 64):
            raise ValueError('unit must be a bounded nonempty string')


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    output_name: str
    modality: str
    provider: str
    version: str
    fields: tuple[OutputField, ...]
    streaming_mode: str
    window_ms: int = 0
    lookahead_ms: int = 0
    freshness_ms: int = 5000
    stateful: bool = False
    reset_on_gap: bool = False
    interpretation: str = 'estimate'
    devices: tuple[str, ...] = ()
    sensor_requirements: tuple[str, ...] = ()
    license: str = 'unspecified'

    def __post_init__(self):
        for key in ('id', 'output_name', 'provider', 'version'):
            identifier(getattr(self, key), key)
        if self.modality not in MODALITIES or self.streaming_mode not in STREAMING_MODES:
            raise ValueError('unsupported modality or streaming mode')
        if type(self.fields) is not tuple or not 1 <= len(self.fields) <= 32 or any(not isinstance(f, OutputField) for f in self.fields):
            raise ValueError('fields must be a tuple of 1..32 OutputField values')
        if len({f.name for f in self.fields}) != len(self.fields):
            raise ValueError('duplicate output fields')
        for key in ('window_ms', 'lookahead_ms', 'freshness_ms'):
            if type(getattr(self, key)) is not int or not 0 <= getattr(self, key) <= 86400000:
                raise ValueError(f'{key} must be integer milliseconds within one day')
        if self.streaming_mode == 'window' and not self.window_ms:
            raise ValueError('window mode requires a nonzero window')
        if type(self.stateful) is not bool or type(self.reset_on_gap) is not bool:
            raise ValueError('state flags must be boolean')
        if self.reset_on_gap and not self.stateful:
            raise ValueError('reset_on_gap requires stateful=True')
        if self.interpretation not in ('measurement', 'estimate', 'classification', 'semantic_description'):
            raise ValueError('invalid interpretation')
        for key in ('devices', 'sensor_requirements'):
            values = getattr(self, key)
            if type(values) is not tuple or len(values) > 16 or any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in values):
                raise ValueError(f'{key} must be a bounded tuple of strings')
        if not isinstance(self.license, str) or not 1 <= len(self.license) <= 256:
            raise ValueError('license must be a bounded string')

    def validate(self, value):
        value = bounded_json(value, forbid_raw=True)
        if type(value) is not dict or set(value)-{f.name for f in self.fields}:
            raise ValueError('output must be an object containing only declared fields')
        for field in self.fields:
            if field.name not in value:
                if field.required:
                    raise ValueError(f'missing required field: {field.name}')
                continue
            item = value[field.name]
            expected = _KINDS[field.kind]
            if not isinstance(item, expected) or (field.kind in ('number', 'integer') and type(item) is bool):
                raise ValueError(f'invalid type for {field.name}: expected {field.kind}')
        return value

    def describe(self):
        return asdict(self)


@dataclass(frozen=True)
class SensorMetadata:
    source_id: str
    sensor_id: str
    sequence: int
    coordinate_frame: str | None = None
    channels: tuple[str, ...] = ()
    calibration_id: str | None = None
    stream_epoch: str | None = None

    def __post_init__(self):
        identifier(self.source_id, 'source_id')
        identifier(self.sensor_id, 'sensor_id')
        if type(self.sequence) is not int or not 0 <= self.sequence <= 2**63-1:
            raise ValueError('sequence must be a nonnegative 64-bit integer')
        for key in ('coordinate_frame', 'calibration_id', 'stream_epoch'):
            if getattr(self, key) is not None:
                identifier(getattr(self, key), key)
        if type(self.channels) is not tuple or len(self.channels) > 64:
            raise ValueError('channels must be a tuple of at most 64 names')
        for channel in self.channels:
            identifier(channel, 'channel')
        if len(set(self.channels)) != len(self.channels):
            raise ValueError('channel names must be unique')


class CapabilityRegistry:
    def __init__(self):
        self._specs = {}

    def register(self, spec):
        if not isinstance(spec, CapabilitySpec):
            raise TypeError('expected CapabilitySpec')
        if spec.id in self._specs:
            if self._specs[spec.id] == spec:
                return
            raise ValueError('capability already registered with a different contract')
        if len(self._specs) >= 256:
            raise ValueError('capability registry is full')
        self._specs[spec.id] = spec

    def get(self, capability_id):
        try:
            return self._specs[capability_id]
        except (KeyError, TypeError) as exc:
            raise ValueError('capability is not registered') from exc

    def describe(self):
        return deepcopy([spec.describe() for spec in self._specs.values()])
