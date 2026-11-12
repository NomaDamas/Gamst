"""Transport-neutral, bounded live media routing. No raw media persistence."""
from __future__ import annotations
import asyncio
import base64
import inspect
import time
from dataclasses import dataclass
from .plugins import _pcm_floats
from .schema import timestamp


def decode_media(body):
    """Shared browser/WebSocket wire format: normalized PCM or base64 JPEG."""
    modality = body.get('modality', body.get('op'))
    if modality == 'audio':
        samples, rate = body.get('samples'), body.get('sample_rate', 16000)
        if not isinstance(samples, list) or len(samples) > 96000:
            raise ValueError('Audio requires at most 96000 normalized mono samples')
        if type(rate) is not int or not 8000 <= rate <= 96000:
            raise ValueError('sample_rate must be an integer from 8000 to 96000')
        return modality, _pcm_floats(samples), rate
    if modality == 'video':
        encoded = body.get('jpeg')
        if not isinstance(encoded, str) or len(encoded) > 1_000_000:
            raise ValueError('Invalid video payload')
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] > 1080 or frame.shape[1] > 1920:
            raise ValueError('Invalid video dimensions')
        return modality, frame, None
    raise ValueError('modality must be audio or video')


@dataclass
class MediaPacket:
    modality: str
    data: object
    timestamp_ms: int
    source: str
    subject_id: str
    sequence: int
    sample_rate: int | None = None
    generation: int = 0
    discontinuity: bool = False


class MediaRouter:
    """One bounded lane per modality. A lane has one source/subject owner.

    Video drops oldest pending frames. Audio drops oldest pending blocks and
    resets audio model buffers before processing the next contiguous segment.
    Separate lane tasks prevent vision work from blocking audio ingestion.
    """
    def __init__(self, human, audio_capacity=4, video_capacity=1):
        if any(type(n) is not int or not 1 <= n <= 64 for n in (audio_capacity, video_capacity)):
            raise ValueError('lane capacity must be 1..64')
        self.human = human
        self.queues = {m: asyncio.Queue(n) for m,n in [('audio', audio_capacity), ('video', video_capacity)]}
        self.tasks, self.owners, self.sequences = {}, {}, {}
        self.gaps = set()
        self.clearing = set()
        self._clear_lock = asyncio.Lock()
        self.metrics = {m: dict(accepted=0, processed=0, dropped=0, discontinuities=0, errors=0,
                               last_latency_ms=None, last_timestamp_ms=None) for m in self.queues}

    async def submit(self, modality, data, *, source, subject_id='local-user', sequence=None,
                     timestamp_ms=None, sample_rate=None):
        self.human._check()
        if modality not in self.queues:
            raise ValueError('modality must be audio or video')
        if any(not isinstance(s,str) or not s or len(s)>128 for s in (source,subject_id)):
            raise ValueError('source and subject_id must be nonempty strings of at most 128 characters')
        if modality in self.clearing:
            raise RuntimeError('Media lane is resetting; retry after lifecycle completion')
        owner = (source,subject_id)
        if modality in self.owners and self.owners[modality] != owner:
            raise ValueError(f'{modality} lane already belongs to another source/subject')
        previous = self.sequences.get(modality, -1)
        seq = previous+1 if sequence is None else sequence
        if type(seq) is not int or seq <= previous:
            raise ValueError('sequence must increase strictly within a lane')
        ts = timestamp(timestamp_ms)
        if ts > timestamp()+5000:
            raise ValueError('timestamp exceeds wall clock')
        if modality == 'audio':
            if type(sample_rate) is not int or not 8000 <= sample_rate <= 96000:
                raise ValueError('Invalid sample_rate')
            data = _pcm_floats(data)
            if len(data)>96000:
                raise ValueError('Audio block exceeds 96000 samples')
        gap = previous >= 0 and seq != previous+1
        self.owners[modality], self.sequences[modality] = owner, seq
        queue, stats = self.queues[modality], self.metrics[modality]
        if queue.full():
            queue.get_nowait(); queue.task_done()
            stats['dropped'] += 1
            self.gaps.add(modality)
        queue.put_nowait(MediaPacket(modality,data,ts,source,subject_id,seq,sample_rate,self.human._generation,gap))
        stats['accepted'] += 1
        if modality not in self.tasks or self.tasks[modality].done():
            self.tasks[modality] = asyncio.create_task(self._run(modality))
        return dict(accepted=True, modality=modality, sequence=seq, queued=queue.qsize(), dropped=stats['dropped'])

    async def _reset_modality(self, modality):
        for plugin in self.human.plugins:
            if modality in getattr(plugin,'modalities', ()):
                reset = getattr(plugin,'reset',None)
                if reset:
                    result = reset()
                    if inspect.isawaitable(result):
                        await result

    async def _run(self, modality):
        queue, stats = self.queues[modality], self.metrics[modality]
        while not queue.empty():
            packet = queue.get_nowait()
            started = time.monotonic()
            try:
                if packet.generation != self.human._generation or self.human.paused or self.human.closed:
                    continue
                if modality in self.gaps or packet.discontinuity:
                    self.gaps.discard(modality)
                    stats['discontinuities'] += 1
                    if modality == 'audio':
                        await self._reset_modality(modality)
                if modality == 'audio':
                    await self.human.push_audio(packet.data, subject_id=packet.subject_id,
                        timestamp_ms=packet.timestamp_ms, sample_rate=packet.sample_rate)
                else:
                    await self.human.push_video(packet.data, subject_id=packet.subject_id, timestamp_ms=packet.timestamp_ms)
                stats['processed'] += 1
                stats['last_timestamp_ms'] = packet.timestamp_ms
                stats['last_latency_ms'] = round((time.monotonic()-started)*1000,1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                stats['errors'] += 1
                self.human.plugin_errors.append(dict(plugin='media_router', error=str(exc), timestamp_ms=packet.timestamp_ms))
                self.human.plugin_errors = self.human.plugin_errors[-100:]
            finally:
                queue.task_done()

    def status(self):
        return {m:dict(**self.metrics[m], queued=q.qsize(), capacity=q.maxsize,
                       source=self.owners.get(m, (None,None))[0],
                       subject_id=self.owners.get(m, (None,None))[1],
                       last_sequence=self.sequences.get(m)) for m,q in self.queues.items()}

    async def drain(self):
        await asyncio.gather(*(q.join() for q in self.queues.values()))

    async def clear(self, source=None):
        async with self._clear_lock:
            selected = {m for m in self.queues if source is None or self.owners.get(m,(None,))[0] == source}
            self.clearing.update(selected)
            try:
                await self._clear(source)
            finally:
                self.clearing.difference_update(selected)

    async def _clear(self, source=None):
        for modality, queue in self.queues.items():
            if source is not None and self.owners.get(modality,(None,))[0] != source:
                continue
            task = self.tasks.pop(modality,None)
            if task:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
            while not queue.empty():
                queue.get_nowait(); queue.task_done()
            self.owners.pop(modality,None)
            self.sequences.pop(modality,None)
            self.gaps.discard(modality)
            if source is not None:
                await self._reset_modality(modality)


def media_router(human):
    if not hasattr(human,'_media_router'):
        human._media_router = MediaRouter(human)
    return human._media_router
