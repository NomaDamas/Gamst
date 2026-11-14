"""Private SAM separation pairs and explicitly owned, ephemeral local retrieval.

This module implements no model, generic media transport or public download API.
The owner object is a local application capability, never a Context reference.
"""
import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import time
import uuid

from .capabilities import CapabilitySpec, OutputField, bounded_json, identifier

PROFILE = 'sam_audio.text_temporal.local'
PROTOCOL = 'sam-audio-pair-v1'
CAPABILITY = 'audio.sam_audio.separation'
MAX_SAMPLES = 192000
MAX_PAIR_BYTES = MAX_SAMPLES * 8
MEDIA_HEADER_LIMIT = 16384
_FIELDS = ('separation', 'window', 'prompt', 'model', 'media')


class SeparationCapability(CapabilitySpec):
    def validate(self, value):
        from .separation_metadata import validate_metadata
        return validate_metadata(super().validate(value))


def capability(config):
    """Transport identity only; #201 must validate native assets before loading."""
    config = bounded_json(config)
    if type(config) is not dict:
        raise ValueError('SAM separation configuration must be an object')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return SeparationCapability(CAPABILITY, 'audio.source_separation', 'audio', PROFILE,
        'transport1.' + digest, tuple(OutputField(k, 'object') for k in _FIELDS),
        'window', window_ms=4000, freshness_ms=5000, stateful=True,
        reset_on_gap=True, interpretation='estimate',
        sensor_requirements=('mono-float32-48000hz-whole-window',), license='SAM-License')


def frame_binding(header, payload, *, generation, config_id, owner_revision=0):
    """Bind media to the actual copied request, never child-supplied ownership."""
    metadata, layout = header['metadata'], header['layout']
    sensor = metadata['sensor']
    shape = layout.get('shape')
    if (metadata['modality'] != 'audio' or type(metadata['sample_rate']) is not int or
            metadata['sample_rate'] != 48000 or layout.get('kind') != 'ndarray' or
            layout.get('dtype') != '<f4' or type(shape) is not list or len(shape) != 1 or
            type(shape[0]) is not int or not 48000 <= shape[0] <= MAX_SAMPLES or
            len(payload) != shape[0] * 4 or len(sensor['channels']) != 1):
        raise ValueError('separation requires a bounded mono little-endian float32 48kHz window')
    end = metadata['timestamp_ms']
    if type(end) is not int or not shape[0] / 48 <= end <= 2**53-1:
        raise ValueError('separation requires an aligned capture-end timestamp')
    for key in ('source_id', 'sensor_id', 'stream_epoch'):
        identifier(sensor[key], key)
    identifier(metadata['subject_id'], 'subject_id')
    if any(not math.isfinite(v[0]) or abs(v[0]) > 1 for v in struct.iter_unpack('<f', payload)):
        raise ValueError('separation PCM must be finite and normalized')
    return bounded_json(dict(subject_id=metadata['subject_id'], sensor=sensor,
        capture_end_ms=end, capture_start_ms=end-shape[0]/48, sample_rate=48000,
        sample_count=shape[0], generation=generation, owner_revision=owner_revision, capability_id=CAPABILITY,
        config_id=config_id, input_sha256=hashlib.sha256(payload).hexdigest()))


def request_binding(header, payload, *, config, generation, config_id, owner_revision=0):
    if 'visual_policy' in config:
        from .sam_visual_packet import visual_request_binding
        return visual_request_binding(header,payload,policy=config['visual_policy'],
            generation=generation,config_id=config_id,owner_revision=owner_revision)
    return frame_binding(header,payload,generation=generation,config_id=config_id,
                         owner_revision=owner_revision)


def _wave(data, samples):
    if type(data) is not bytes or len(data) != samples * 4:
        raise ValueError('separation waveform must be exact mono float32 bytes')
    # Preserve model values, including values outside normalized PCM. A closed
    # safety bound rejects overflow; no implicit clipping or normalization.
    if any(not math.isfinite(v[0]) or abs(v[0]) > 16 for v in struct.iter_unpack('<f', data)):
        raise ValueError('separation waveform exceeds finite amplitude bounds')


@dataclass(frozen=True)
class SeparationPair:
    observation: dict
    target: bytes
    residual: bytes
    padded_samples: int


def pack_pair(pair, binding, spec):
    if type(pair) is not SeparationPair:
        raise ValueError('native separation must return SeparationPair')
    n = binding['sample_count']
    if type(pair.padded_samples) is not int or not n <= pair.padded_samples <= n + 48000:
        raise ValueError('invalid bounded codec padding')
    observation = spec.validate(pair.observation)
    if observation['media']:
        raise ValueError('only the parent may create media references')
    parts = []
    for index, (name, data) in enumerate((('target', pair.target), ('residual', pair.residual))):
        _wave(data, n)
        parts.append(dict(name=name, offset=index*n*4, bytes=n*4,
                          sha256=hashlib.sha256(data).hexdigest()))
    header = dict(protocol=PROTOCOL, binding=binding, encoding='<f4',
        padded_samples=pair.padded_samples, components=parts)
    bounded_json(header, max_bytes=MEDIA_HEADER_LIMIT)
    return observation, header, pair.target + pair.residual


def unpack_pair(observation, header, payload, binding, spec):
    """Validate the closed response layout and return privately owned pair bytes."""
    if type(header) is not dict or set(header) != {'protocol', 'binding', 'encoding', 'padded_samples', 'components'}:
        raise ValueError('invalid separation response fields')
    if (header['protocol'] != PROTOCOL or header['encoding'] != '<f4' or
            json.dumps(header['binding'], sort_keys=True) != json.dumps(binding, sort_keys=True)):
        raise ValueError('separation response does not match its request')
    n = binding['sample_count']
    if type(payload) is not bytes or len(payload) != n*8 or len(payload) > MAX_PAIR_BYTES:
        raise ValueError('separation response length mismatch')
    parts = header['components']
    if type(parts) is not list or len(parts) != 2:
        raise ValueError('exactly target and residual required')
    waves = []
    for index, name in enumerate(('target', 'residual')):
        data = payload[index*n*4:(index+1)*n*4]
        expected = dict(name=name, offset=index*n*4, bytes=n*4,
                        sha256=hashlib.sha256(data).hexdigest())
        part = parts[index]
        if (type(part) is not dict or set(part) != set(expected) or
                type(part.get('offset')) is not int or type(part.get('bytes')) is not int or part != expected):
            raise ValueError('invalid separation component layout or hash')
        waves.append(data)
    pair = SeparationPair(observation, *waves, header['padded_samples'])
    pack_pair(pair, binding, spec)
    return pair


class SeparationMediaStore:
    """Loop-bound memory owner; limits cover retained pairs across all grants."""
    def __init__(self, human, *, ttl_ms=30000, capacity=8, max_bytes=16*1024*1024):
        for value, upper in ((ttl_ms, 60000), (capacity, 32), (max_bytes, 64*1024*1024)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError('invalid separation retention bound')
        if max_bytes < MAX_PAIR_BYTES:
            raise ValueError('retention must accommodate one maximum separation pair')
        self.human, self.loop = human, asyncio.get_running_loop()
        self.ttl_ms, self.capacity, self.max_bytes = ttl_ms, capacity, max_bytes
        self.entries, self.owners = OrderedDict(), set()
        self.reservations = {}
        self.bytes = 0
        self.timer = None

    def _loop(self):
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError('separation ownership is bound to its runtime event loop')

    def grant(self, subject_id, source_id, sensor_id):
        self._loop()
        self.human._check()
        for value in (subject_id, source_id, sensor_id):
            identifier(value)
        if len(self.owners) >= 32:
            raise ValueError('separation owner limit reached')
        owner = SeparationMediaOwner(self, subject_id, source_id, sensor_id)
        self.owners.add(owner)
        return owner

    def _drop(self, ref):
        self.bytes -= self.entries.pop(ref)['size']

    def _prune(self):
        now = time.monotonic()
        for ref, entry in list(self.entries.items()):
            if entry['deadline'] <= now or entry['generation'] != self.human._generation:
                self._drop(ref)
        self._schedule()

    def _schedule(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if self.entries:
            delay = max(0, min(e['deadline'] for e in self.entries.values())-time.monotonic())
            self.timer = self.loop.call_later(delay, self._prune)

    def clear(self):
        # Also used by the synchronous consent setter. Revocation must not
        # require a running loop or await a stalled worker before dropping bytes.
        for owner in self.owners:
            owner.closed = True
        self.owners.clear()
        self.entries.clear()
        self.bytes = 0
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


class SeparationMediaOwner:
    """Application-only authorization. Never serialize this object or expose it remotely."""
    def __init__(self, store, subject, source, sensor):
        self._store = store
        self.subject_id, self.source_id, self.sensor_id = subject, source, sensor
        self.generation, self.epoch = store.human._generation, None
        self.closed, self.claimed = False, False
        self.revision, self.reset_pending = 0, False

    def _check(self):
        store = self._store
        store._loop()
        store.human._check()
        if self.closed or self not in store.owners or self.generation != store.human._generation:
            raise PermissionError('separation owner has been revoked')

    def _claim(self):
        self._check()
        if self.claimed:
            raise ValueError('separation owner requires a fresh isolated worker')
        self.claimed = True

    def _bind(self, binding):
        self._check()
        if self.reset_pending or binding['owner_revision'] != self.revision:
            raise PermissionError('separation request invalidated by provider reset')
        sensor = binding['sensor']
        if (binding['subject_id'], sensor['source_id'], sensor['sensor_id'], binding['generation']) != (
                self.subject_id, self.source_id, self.sensor_id, self.generation):
            raise PermissionError('separation request differs from authorized owner')
        epoch = sensor['stream_epoch']
        if self.epoch is not None and self.epoch != epoch:
            raise PermissionError('reopen separation ownership after an epoch change')
        self.epoch = epoch

    def _admit(self, pair, binding):
        self._bind(binding)
        store = self._store
        store._prune()
        size = len(pair.target) + len(pair.residual)
        if not 0 < size <= store.max_bytes:
            raise ValueError('separation pair exceeds retention budget')
        others = sum(store.reservations.values()) - store.reservations.get(self, 0)
        while store.entries and (len(store.entries) >= store.capacity or store.bytes+size+others > store.max_bytes):
            store._drop(next(iter(store.entries)))
        ref, now = uuid.uuid4().hex, time.monotonic()
        deadline = now + store.ttl_ms/1000
        metadata = dict(reference=ref, available_at_publication=True,
            availability='explicit-owner-lookup-required', ttl_ms=store.ttl_ms,
            retention_capacity=store.capacity, retention_max_bytes=store.max_bytes,
            expires_at_ms=time.time_ns()//1000000+store.ttl_ms,
            sample_rate=48000, sample_count=binding['sample_count'], encoding='<f4',
            components={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                                   samples_outside_normalized_range=sum(abs(v[0]) > 1 for v in struct.iter_unpack('<f', data)))
                        for name, data in (('target', pair.target), ('residual', pair.residual))})
        store.entries[ref] = dict(owner=self, binding=deepcopy(binding), generation=self.generation,
            deadline=deadline, size=size, target=pair.target, residual=pair.residual,
            metadata=metadata)
        store.bytes += size
        store._schedule()
        return deepcopy(metadata)

    def _reserve(self, samples):
        self._check()
        store = self._store
        store._prune()
        # Conservative parent pipe working-set reservation: copied request,
        # response, decoded components and validation concatenation. SensorStreams
        # queue snapshots and child/model memory have separate explicit budgets.
        size = samples * 32
        if self in store.reservations or sum(store.reservations.values())+size > store.max_bytes:
            raise ValueError('separation in-flight byte budget exhausted')
        while store.entries and store.bytes+sum(store.reservations.values())+size > store.max_bytes:
            store._drop(next(iter(store.entries)))
        store.reservations[self] = size

    def _release(self):
        self._store.reservations.pop(self, None)

    def retrieve(self, reference, component, *, stream_epoch):
        """Explicit local API; reference possession alone grants no access."""
        self._check()
        if type(reference) is not str or component not in ('target', 'residual'):
            raise ValueError('select an opaque reference and target or residual')
        self._store._prune()
        entry = self._store.entries.get(reference)
        if entry is None or entry['owner'] is not self or stream_epoch != self.epoch:
            raise PermissionError('separation media unavailable to this owner/epoch')
        return entry[component]  # Immutable bytes; no writable backing array.

    def lookup(self, reference, *, stream_epoch):
        self.retrieve(reference, 'target', stream_epoch=stream_epoch)
        return deepcopy(self._store.entries[reference]['metadata'])

    def discard(self, reference):
        """Internal publication rollback; never removes another owner's pair."""
        entry = self._store.entries.get(reference)
        if entry is not None and entry['owner'] is self:
            self._store._drop(reference)
            self._store._schedule()

    def _reset(self):
        self._check()
        if self.reset_pending:
            raise RuntimeError('separation reset is already pending')
        # Intent is synchronous: an old response cannot restore cleared bytes
        # while reset waits behind its in-flight pipe exchange.
        self.revision += 1
        self.reset_pending = True
        for ref, entry in list(self._store.entries.items()):
            if entry['owner'] is self:
                self._store._drop(ref)
        self.epoch = None
        self._store._schedule()

    def _reset_complete(self):
        self._check()
        self.reset_pending = False

    def close(self):
        self._store._loop()
        self.closed = True
        for ref, entry in list(self._store.entries.items()):
            if entry['owner'] is self:
                self._store._drop(ref)
        self._store.owners.discard(self)
        self._store._schedule()
