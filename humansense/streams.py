"""Explicit per-sensor/per-provider lanes with bounded asynchronous processing."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import inspect
import json
import math
import time
import uuid

from .capabilities import CapabilitySpec, SensorMetadata, bounded_json, identifier
from .plugins import InputFrame
from .schema import timestamp


async def _finish(task):
    """Finish owned work despite repeated cancellation of its waiter."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _call(function, *args, cleanup_on_cancel=None):
    """Never reset/close a synchronous backend while its worker is still running."""
    if inspect.iscoroutinefunction(function):
        return await function(*args)
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            result = await _finish(task)
            if cleanup_on_cancel:
                await cleanup_on_cancel(result)
        finally:
            raise
    if inspect.isawaitable(result):
        return await result
    return result


def _snapshot(data, limit):
    """Bound raw input before copying. Arrays retain dimensions and channels."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        size = data.nbytes if isinstance(data, memoryview) else len(data)
        if size > limit:
            raise ValueError('sensor packet exceeds byte limit')
        return bytes(data), size
    if hasattr(data, 'nbytes') and hasattr(data, 'copy'):
        size = int(data.nbytes)
        if not 0 <= size <= limit:
            raise ValueError('sensor array exceeds byte limit')
        if getattr(getattr(data, 'dtype', None), 'hasobject', False):
            raise ValueError('object arrays are not sensor data')
        return data.copy(), size
    if isinstance(data, list):
        # PCM lists may exceed observation JSON bounds, but remain bounded here.
        if len(data) > 192000:
            raise ValueError('sensor sample list too long')
        result, count = [], 0
        for sample in data:
            values = sample if isinstance(sample, list) else [sample]
            if not 1 <= len(values) <= 64:
                raise ValueError('invalid sample channels')
            count += len(values)
            if count*32 > limit:
                raise ValueError('sensor samples exceed byte limit')
            if any(type(v) not in (int, float) or abs(v) > 1e100 or not math.isfinite(v) for v in values):
                raise ValueError('sensor samples must be finite numbers')
            result.append(list(values) if isinstance(sample, list) else sample)
        return result, count*32
    result = bounded_json(data, max_bytes=limit)
    return result, len(json.dumps(result).encode())


class SensorStreams:
    """Factories create isolated provider instances only on explicit open().

Each provider implements process(InputFrame) -> declared output dict or None,
and reset()/close() as needed. Synchronous methods run off the event loop.
This is an additional typed-input path; legacy MediaRouter remains compatible.
"""
    def __init__(self, human, *, capacity=2, max_streams=16,
                 max_packet_bytes=4*1024*1024, max_buffer_bytes=64*1024*1024):
        for value, maximum in ((capacity, 64), (max_streams, 64),
                               (max_packet_bytes, 16*1024*1024), (max_buffer_bytes, 256*1024*1024)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('invalid stream resource bound')
        self.manager_id = uuid.uuid4().hex
        self.human, self.capacity, self.max_streams = human, capacity, max_streams
        self.max_packet_bytes, self.max_buffer_bytes = max_packet_bytes, max_buffer_bytes
        self.factories, self.streams = {}, {}
        self.buffer_bytes = 0
        self.closed = False
        self._lock = asyncio.Lock()
        if not hasattr(human, '_sensor_streams'):
            human._sensor_streams = []
            human._sensor_provider_owners = {}
        if len(human._sensor_streams) >= 4 or human.closed:
            raise RuntimeError('runtime closed or sensor managers limit reached')
        human._sensor_streams.append(self)

    def register(self, spec, factory):
        if self.closed or not callable(factory):
            raise ValueError('open manager and callable provider factory required')
        if spec.id in self.factories or len(self.factories) >= 32:
            raise ValueError('provider already registered or provider limit reached')
        self.human.register_capability(spec)
        self.factories[spec.id] = factory

    async def _discard_unowned(self, provider):
        # A cancelled factory must not dispose a singleton already used elsewhere.
        if id(provider) not in self.human._sensor_provider_owners:
            close = getattr(provider, 'close', None)
            if close:
                await _call(close)

    async def _dispose(self, provider):
        try:
            close = getattr(provider, 'close', None)
            if close:
                await _call(close)
        finally:
            self.human._sensor_provider_owners.pop(id(provider), None)

    async def open(self, source_id, sensor_id, capabilities, *, subject_id='local-user'):
        """Explicitly construct providers; discovery and submit never load them."""
        self.human._check()
        identifier(source_id); identifier(sensor_id); identifier(subject_id)
        ids = tuple(capabilities)
        if not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids):
            raise ValueError('select 1..8 distinct capabilities')
        specs = [self.human.perception_capabilities.get(i) for i in ids]
        if any(i not in self.factories for i in ids) or len({s.modality for s in specs}) != 1:
            raise ValueError('select registered factories with one input modality')
        key = (source_id, sensor_id)
        async with self._lock:
            if self.closed or key in self.streams or len(self.streams) >= self.max_streams:
                raise ValueError('stream exists, manager closed or stream limit reached')
            lanes = []
            try:
                for spec in specs:
                    provider = await _call(self.factories[spec.id], cleanup_on_cancel=self._discard_unowned)
                    owners = self.human._sensor_provider_owners
                    if id(provider) in owners:
                        raise ValueError('factory must return an isolated provider instance')
                    # Reserve before awaiting the next factory, including across managers.
                    owners[id(provider)] = provider
                    lane = dict(spec=spec, provider=provider, queue=asyncio.Queue(self.capacity),
                                epoch=uuid.uuid4().hex,
                                task=None, gap=False, accepted=0, processed=0, dropped=0,
                                errors=0, resets=0, last_latency_ms=None, last_error=None)
                    lanes.append(lane)
                    declared = getattr(provider, 'spec', None)
                    if isinstance(declared, CapabilitySpec) and declared != spec:
                        raise ValueError('provider contract differs from registered capability')
                    if not callable(getattr(provider, 'process', None)):
                        raise ValueError('provider must implement process')
                    if spec.stateful and not callable(getattr(provider, 'reset', None)):
                        raise ValueError('stateful provider must implement reset')
                self.human._check()
                self.streams[key] = dict(subject=subject_id, modality=specs[0].modality,
                                        sequence=-1, timestamp=-1, lanes=lanes, epoch=uuid.uuid4().hex)
            except BaseException:
                errors = []
                for lane in lanes:
                    try:
                        await self._dispose(lane['provider'])
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    raise RuntimeError('provider construction cleanup failed') from errors[0]
                raise

    async def submit(self, frame):
        self.human._check()
        if self.closed or not isinstance(frame, InputFrame) or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError('open manager and InputFrame with SensorMetadata required')
        key = (frame.sensor.source_id, frame.sensor.sensor_id)
        # No awaits while validating/enqueuing: lifecycle mutations cannot interleave.
        stream = self.streams.get(key)
        if stream is None or stream.get('closing'):
            raise ValueError('sensor stream is not open')
        if frame.subject_id != stream['subject'] or frame.modality != stream['modality']:
            raise ValueError('frame does not match stream owner/modality')
        if frame.sensor.stream_epoch not in (None, stream['epoch']):
            raise ValueError('frame belongs to another stream epoch')
        frame = replace(frame, sensor=replace(frame.sensor, stream_epoch=stream['epoch']))
        ts = timestamp(frame.timestamp_ms)
        if ts < stream['timestamp'] or ts > timestamp()+5000 or frame.sensor.sequence <= stream['sequence']:
            raise ValueError('capture time and sequence must advance within the stream')
        if frame.modality == 'audio' and (type(frame.sample_rate) is not int or not 8000 <= frame.sample_rate <= 192000):
            raise ValueError('audio sample rate must be 8000..192000')
        data, size = _snapshot(frame.data, self.max_packet_bytes)
        lanes = stream['lanes']
        reclaimed = sum(l['queue']._queue[0][1] for l in lanes if l['queue'].full())
        if self.buffer_bytes-reclaimed+size*len(lanes) > self.max_buffer_bytes:
            raise ValueError('sensor buffer budget exhausted')
        gap = stream['sequence'] >= 0 and frame.sensor.sequence != stream['sequence']+1
        stream['sequence'], stream['timestamp'] = frame.sensor.sequence, ts
        for lane in lanes:
            if lane['queue'].full():
                _, old_size, _, _ = lane['queue'].get_nowait()
                lane['queue'].task_done()
                self.buffer_bytes -= old_size
                lane['dropped'] += 1
                lane['gap'] = True
            # Each provider receives its own snapshot to prevent cross-provider mutation.
            owned, _ = _snapshot(data, self.max_packet_bytes)
            lane['queue'].put_nowait((replace(frame, data=owned, timestamp_ms=ts), size, gap, self.human._generation))
            self.buffer_bytes += size
            lane['accepted'] += 1
            if lane['task'] is None or lane['task'].done():
                lane['task'] = asyncio.create_task(self._run(stream, lane))
        return {'accepted': True, 'sequence': frame.sensor.sequence, 'providers': len(lanes)}

    async def _run(self, stream, lane):
        queue = lane['queue']
        while not queue.empty():
            frame, size, gap, generation = queue.get_nowait()
            started = time.monotonic()
            try:
                if generation != self.human._generation or stream.get('closing'):
                    continue
                if gap or lane['gap']:
                    lane['gap'] = False
                    if lane['spec'].stateful:
                        await _call(lane['provider'].reset)
                        self.human.object_memory.invalidate_visibility(frame.sensor.source_id,
                            frame.sensor.sensor_id, lane['spec'].id)
                        self.human.transcript_memory.interrupt(frame.sensor.source_id,
                            frame.sensor.sensor_id, lane['spec'].id,
                            subject_id=frame.subject_id, epoch=lane['epoch'])
                        lane['epoch'] = uuid.uuid4().hex
                        lane['resets'] += 1
                frame = replace(frame, sensor=replace(frame.sensor, stream_epoch=lane['epoch']))
                result = await _call(lane['provider'].process, frame)
                if generation != self.human._generation or stream.get('closing') or self.human.paused or self.human.closed:
                    continue
                if result is not None:
                    await self.human.push_observation(lane['spec'].id, result, sensor=frame.sensor,
                        subject_id=frame.subject_id, timestamp_ms=frame.timestamp_ms)
                lane['processed'] += 1
                lane['last_latency_ms'] = round((time.monotonic()-started)*1000, 3)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                lane['errors'] += 1
                lane['last_error'] = type(exc).__name__
                lane['gap'] = True
                self.human.object_memory.invalidate_visibility(frame.sensor.source_id,
                    frame.sensor.sensor_id, lane['spec'].id)
                self.human.transcript_memory.interrupt(frame.sensor.source_id,
                    frame.sensor.sensor_id, lane['spec'].id,
                    subject_id=frame.subject_id, epoch=lane['epoch'])
            finally:
                self.buffer_bytes -= size
                queue.task_done()

    async def drain(self):
        await asyncio.gather(*(l['queue'].join() for s in self.streams.values() for l in s['lanes']))

    async def clear(self, source_id=None, sensor_id=None):
        cleanup = asyncio.create_task(self._clear(source_id, sensor_id))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await _finish(cleanup)
            raise

    async def _clear(self, source_id=None, sensor_id=None):
        async with self._lock:
            keys = [k for k in self.streams if (source_id is None or k[0] == source_id)
                    and (sensor_id is None or k[1] == sensor_id)]
            errors = []
            for key in keys:
                stream = self.streams[key]
                stream['closing'] = True
                self.human.object_memory.invalidate_visibility(*key)
                for lane in stream['lanes']:
                    if lane['task']:
                        lane['task'].cancel()
                        await asyncio.gather(lane['task'], return_exceptions=True)
                    self.human.transcript_memory.interrupt(*key, lane['spec'].id,
                        subject_id=stream['subject'], epoch=lane['epoch'])
                    while not lane['queue'].empty():
                        _, size, _, _ = lane['queue'].get_nowait()
                        self.buffer_bytes -= size
                        lane['queue'].task_done()
                    try:
                        await self._dispose(lane['provider'])
                    except Exception as exc:
                        errors.append(exc)
                del self.streams[key]
            if errors:
                raise RuntimeError('sensor provider cleanup failed') from errors[0]

    async def close(self):
        self.closed = True
        await self.clear()

    def status(self):
        return dict(manager_id=self.manager_id, closed=self.closed,
            registered_capabilities=list(self.factories),
            buffer_bytes=self.buffer_bytes, max_buffer_bytes=self.max_buffer_bytes,
            streams=[dict(source_id=key[0], sensor_id=key[1], subject_id=s['subject'],
                modality=s['modality'], last_sequence=s['sequence'],
                providers=[dict(capability_id=l['spec'].id, queued=l['queue'].qsize(),
                    **{k: l[k] for k in ('accepted', 'processed', 'dropped', 'errors', 'resets',
                                        'last_latency_ms', 'last_error')}) for l in s['lanes']])
                for key, s in self.streams.items()])
