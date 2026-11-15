"""Owned immutable acquisition, separate from SDK/provider delivery lifetimes.

Only trusted built-in drivers implement CaptureSink. JSON never selects code.
Reading, clock validation and acquisition remain charged to the recording even
when a downstream mailbox refuses the acquired packet.
"""
import asyncio
from dataclasses import dataclass
import hashlib
import os
import stat
import time

from gamst.streams import _finish
from .recorded_source import validate_manifest, digest, _integer
from .replay_clock import ReplayClock


@dataclass(frozen=True, slots=True)
class CaptureCapacity:
    source_buffer_bytes: int
    max_lateness_ns: int = 50_000_000

    def __post_init__(self):
        _integer(self.source_buffer_bytes, 1, 256*1024**2)
        _integer(self.max_lateness_ns, 1, 1_000_000_000)


@dataclass(frozen=True, slots=True)
class CapturePacket:
    source_manifest_sha256: str
    record_sha256: str
    record_ordinal: int
    source_index: int
    source_id: str
    sensor_id: str
    original_stream_epoch: str
    capture_sequence: int
    original_capture_ns: int
    offset_ns: int
    mapping_sha256: str
    scheduled_monotonic_ns: int
    acquired_monotonic_ns: int
    sdk_timestamp_ms: int
    submillisecond_remainder_ns: int
    format: str
    shape: tuple | None
    sample_rate: int | None
    sample_count: int | None
    payload_sha256: str
    payload: bytes

    def __post_init__(self):
        # The constructor is internal to acquisition, but a sink never accepts a
        # mutable bytearray/array masquerading as a frozen packet.
        if type(self.payload) is not bytes or hashlib.sha256(self.payload).hexdigest() != self.payload_sha256:
            raise ValueError('owned immutable capture bytes required')
        if self.format == 'bgr8_hwc':
            if (type(self.shape) is not tuple or len(self.shape) != 3 or self.shape[2] != 3
                    or self.sample_rate is not None or self.sample_count is not None
                    or len(self.payload) != self.shape[0]*self.shape[1]*3):
                raise ValueError('exact BGR capture shape required')
        elif self.format == 'pcm_s16le_mono':
            if self.shape is not None or self.sample_rate is None or self.sample_count is None or len(self.payload) != self.sample_count*2:
                raise ValueError('exact PCM capture layout required')
        else:
            raise ValueError('unsupported owned capture format')


@dataclass(frozen=True, slots=True)
class DeliveryAdmission:
    admitted: bool
    reason: str
    delivery_ordinal: int | None

    def __post_init__(self):
        if type(self.admitted) is not bool or self.reason not in ('accepted', 'closed', 'capacity', 'unsupported'):
            raise ValueError('exact delivery admission required')
        if self.admitted != (self.reason == 'accepted'):
            raise ValueError('delivery acceptance/reason mismatch')
        if self.admitted:
            _integer(self.delivery_ordinal, 0, 2**63-1)
        elif self.delivery_ordinal is not None:
            raise ValueError('refused delivery has no ordinal')


class CaptureSink:
    """Synchronous bounded mailbox seam. Never await/model/execute in offer."""
    def offer(self, packet):
        raise NotImplementedError


def _read_owned(record, root_fd):
    directory = os.dup(root_fd)
    try:
        parts = record['payload_path'].split('/')
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        size = record['payload_bytes']; info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError('exact regular capture payload required')
        blocks = []; remaining = size
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                raise ValueError('capture truncated during owned read')
            blocks.append(block); remaining -= len(block)
        if os.read(fd, 1) or os.fstat(fd).st_size != size:
            raise ValueError('capture grew during owned read')
        payload = b''.join(blocks)
        if hashlib.sha256(payload).hexdigest() != record['payload_sha256']:
            raise ValueError('actual capture bytes differ from frozen hash')
        return payload
    finally:
        os.close(fd)


class OwnedCaptureSource:
    """Single-use source owner; never closes a shared SDK/provider/writer."""
    @classmethod
    async def preflight(cls, validated_source, owned_payload_root_fd, capacity, *, journal):
        if type(capacity) is not CaptureCapacity:
            raise TypeError('typed joint source capacity reservation required')
        manifest = validate_manifest(validated_source)
        if type(owned_payload_root_fd) is not int or not stat.S_ISDIR(os.fstat(owned_payload_root_fd).st_mode):
            raise ValueError('owned payload directory FD required')
        maxima = {s['source_id']: 0 for s in manifest['sources']}
        for record in manifest['records']:
            maxima[record['source_id']] = max(maxima[record['source_id']], record['payload_bytes'])
        # Two retained mailbox packets plus one read's blocks and joined bytes.
        # Active calls and decoded/PNG/base64/request copies reserve separately.
        reserved = sum(size*4 for size in maxima.values())
        if reserved > capacity.source_buffer_bytes:
            raise ValueError('joint source buffer reservation cannot fit')
        self = cls(); self._manifest = manifest; self.manifest_sha256 = digest(manifest)
        self._fd = os.dup(owned_payload_root_fd); self.capacity = capacity
        self.reserved_source_bytes = reserved; self._journal = journal
        self._reads = set(); self._jobs = set(); self._run_task = self._close_task = None
        self._stopped = False; self._closed = False; self._clock = self._sink = None
        self.stop_reason = None; self.terminal_reached_ns = None
        self._by_source = [[] for _ in manifest['sources']]
        indices = {s['source_id']: i for i, s in enumerate(manifest['sources'])}
        for ordinal, record in enumerate(manifest['records']):
            self._by_source[indices[record['source_id']]].append((ordinal, record))
        self._counters = [dict(declared=len(rows), attempted=0, captured=0, dropped=0,
                               verified_reads=0, offered=0, admitted=0, refused=0)
                          for rows in self._by_source]
        try:
            verified = set()
            for record in manifest['records']:
                if record['payload_path'] not in verified:
                    await self._read(record); verified.add(record['payload_path'])
        except BaseException:
            await self.close()
            raise
        return self

    async def _read(self, record):
        task = asyncio.create_task(asyncio.to_thread(_read_owned, record, self._fd))
        self._reads.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish(asyncio.create_task(_drain(task)))
            raise
        finally:
            self._reads.discard(task)

    def start(self, clock, sink):
        if self._closed or self._stopped or self._run_task is not None:
            raise RuntimeError('owned source is single-use')
        if type(clock) is not ReplayClock or clock.mapping['source_manifest_sha256'] != self.manifest_sha256:
            raise ValueError('exact original replay clock required')
        if not isinstance(sink, CaptureSink):
            raise TypeError('trusted built-in capture sink required')
        clock.check(); self._clock = clock; self._sink = sink
        self._run_task = asyncio.create_task(self._run())
        return self._run_task

    def stop_admission(self, reason):
        if type(reason) is not str or not 1 <= len(reason) <= 128:
            raise ValueError('bounded source stop reason required')
        self._stopped = True
        if self.stop_reason is None: self.stop_reason = reason
        for task in self._jobs:
            if not task.done(): task.cancel()

    async def _until(self, offset):
        delay = self._clock.monotonic_ns + offset - time.monotonic_ns()
        if delay > 0: await asyncio.sleep(delay/10**9)
        if self._stopped: raise asyncio.CancelledError()
        return self._clock.check()

    async def _source(self, index):
        source = self._manifest['sources'][index]; counters = self._counters[index]
        for ordinal, record in self._by_source[index]:
            payload = None; failure = None
            try:
                payload = await self._read(record); counters['verified_reads'] += 1
            except (ValueError, OSError):
                failure = 'payload_verification_failed'
            now = await self._until(record['offset_ns'])
            counters['attempted'] += 1
            timing = self._clock.capture(record['offset_ns'])
            if failure or now - timing['scheduled_monotonic_ns'] > self.capacity.max_lateness_ns:
                counters['dropped'] += 1
                self._journal.emit('capture_drop', dict(source_id=source['source_id'], ordinal=ordinal,
                    sequence=record['sequence'], reason=failure or 'acquisition_late', **timing))
                if failure: raise ValueError('recording source changed or unavailable')
                continue
            packet = CapturePacket(self.manifest_sha256, digest(record), ordinal, index,
                source['source_id'], source['sensor_id'], source['stream_epoch'], record['sequence'],
                timing['original_capture_ns'], record['offset_ns'], timing['mapping_sha256'],
                timing['scheduled_monotonic_ns'], now, timing['sdk_timestamp_ms'],
                timing['submillisecond_remainder_ns'], record['format'],
                tuple(record['shape']) if 'shape' in record else None,
                record.get('sample_rate'), record.get('sample_count'), record['payload_sha256'], payload)
            counters['captured'] += 1
            self._journal.emit('capture_acquired', dict(source_id=source['source_id'], ordinal=ordinal,
                sequence=record['sequence'], payload_sha256=record['payload_sha256'],
                acquired_monotonic_ns=now, **timing))
            counters['offered'] += 1
            admission = self._sink.offer(packet)
            if type(admission) is not DeliveryAdmission:
                raise TypeError('exact immutable delivery admission required')
            counters['admitted' if admission.admitted else 'refused'] += 1
            self._journal.emit('delivery_admitted' if admission.admitted else 'delivery_refused',
                dict(source_id=source['source_id'], ordinal=ordinal, sequence=record['sequence'],
                     reason=admission.reason, delivery_ordinal=admission.delivery_ordinal))
            payload = packet = None
            await asyncio.sleep(0)

    async def _run(self):
        try:
            self._jobs = {asyncio.create_task(self._source(i)) for i in range(len(self._by_source))}
            await asyncio.gather(*self._jobs)
            self.terminal_reached_ns = await self._until(self._manifest['duration_ns'])
        except BaseException:
            self.stop_admission('source_failed_or_cancelled')
            await _finish(asyncio.create_task(_drain(*self._jobs)))
            raise

    async def join_reads(self):
        if self._reads:
            await _finish(asyncio.create_task(_drain(*tuple(self._reads))))

    def counters(self):
        return [dict(source_index=i, source_id=self._manifest['sources'][i]['source_id'],
                     **row, uninspected_suffix=row['declared']-row['attempted'])
                for i, row in enumerate(self._counters)]

    async def close(self):
        if asyncio.current_task() is self._run_task:
            raise RuntimeError('source cannot join its own scheduler')
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try: await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise

    async def _close(self):
        self.stop_admission('closed')
        if self._run_task is not None and not self._run_task.done(): self._run_task.cancel()
        if self._run_task is not None: await _drain(self._run_task)
        await self.join_reads()
        if not self._closed:
            os.close(self._fd); self._closed = True


async def _drain(*tasks):
    await asyncio.gather(*tasks, return_exceptions=True)
