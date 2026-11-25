"""Single-producer shared-memory media rings with one media clock.

One producer decodes the input once; every model process maps the same
segments read-only. Frames carry the media timestamp (``pts_ms``) and the
producer wall time, so every worker shares one clock and latency is measured
against the moment the frame became available, not when a worker got to it.

Readers use a sequence check (seqlock style): a slot is copied, then its
sequence is re-read; a changed sequence means the producer overwrote it and
the frame is reported as dropped instead of returning torn data.
"""
from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

_HEADER = 8  # int64 words


def _attach(name: str) -> shared_memory.SharedMemory:
    """Map an existing segment without registering it for cleanup.

    Before Python 3.13 every attaching process registers the segment with its
    resource tracker, which unlinks it when that reader exits and breaks the
    producer's own cleanup. Only the creating process owns the segment.
    """
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        from multiprocessing import resource_tracker
        segment = shared_memory.SharedMemory(name=name)
        resource_tracker.unregister(segment._name, "shared_memory")
        return segment


class FrameRing:
    """Fixed-shape uint8 BGR frames. Header: write_seq, height, width, slots."""

    SLOT_META = 3  # seq, pts_ms, wall_ns

    def __init__(self, segment: shared_memory.SharedMemory, owner: bool):
        self.segment, self.owner = segment, owner
        self.header = np.ndarray((_HEADER,), np.int64, segment.buf, 0)
        self.height, self.width, self.slots = (int(v) for v in self.header[1:4])
        self.frame_bytes = self.height * self.width * 3
        self.slot_bytes = self.SLOT_META * 8 + self.frame_bytes
        self.base = _HEADER * 8

    @classmethod
    def create(cls, name: str, height: int, width: int, slots: int = 64) -> "FrameRing":
        size = _HEADER * 8 + slots * (cls.SLOT_META * 8 + height * width * 3)
        segment = shared_memory.SharedMemory(name=name, create=True, size=size)
        header = np.ndarray((_HEADER,), np.int64, segment.buf, 0)
        header[:] = 0
        header[0] = -1
        header[1:4] = (height, width, slots)
        del header
        ring = cls(segment, owner=True)
        for slot in range(slots):
            ring._meta(slot)[:] = (-1, 0, 0)
        return ring

    @classmethod
    def attach(cls, name: str) -> "FrameRing":
        return cls(_attach(name), owner=False)

    def _meta(self, slot: int) -> np.ndarray:
        return np.ndarray((self.SLOT_META,), np.int64, self.segment.buf, self.base + slot * self.slot_bytes)

    def _pixels(self, slot: int) -> np.ndarray:
        offset = self.base + slot * self.slot_bytes + self.SLOT_META * 8
        return np.ndarray((self.height, self.width, 3), np.uint8, self.segment.buf, offset)

    @property
    def write_seq(self) -> int:
        return int(self.header[0])

    def write(self, frame: np.ndarray, pts_ms: int) -> int:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("frame shape/dtype does not match the ring")
        seq = self.write_seq + 1
        slot = seq % self.slots
        meta = self._meta(slot)
        meta[0] = -1  # invalidate while writing
        self._pixels(slot)[:] = frame
        meta[1], meta[2] = pts_ms, time.time_ns()
        meta[0] = seq
        self.header[0] = seq
        return seq

    def read(self, seq: int):
        """Return (pts_ms, wall_ns, frame copy) or None if not written yet / overwritten."""
        if seq < 0 or seq > self.write_seq:
            return None
        slot = seq % self.slots
        meta = self._meta(slot)
        if int(meta[0]) != seq:
            return None
        pts_ms, wall_ns = int(meta[1]), int(meta[2])
        frame = self._pixels(slot).copy()
        if int(meta[0]) != seq:
            return None
        return pts_ms, wall_ns, frame

    def oldest_available(self) -> int:
        return max(0, self.write_seq - self.slots + 2)

    def close(self) -> None:
        del self.header
        self.segment.close()
        if self.owner:
            self.segment.unlink()


class AudioRing:
    """Mono int16 PCM. Header: total_samples_written, sample_rate, capacity, -, wall_ns_of_last_write."""

    def __init__(self, segment: shared_memory.SharedMemory, owner: bool):
        self.segment, self.owner = segment, owner
        self.header = np.ndarray((_HEADER,), np.int64, segment.buf, 0)
        self.sample_rate, self.capacity = int(self.header[1]), int(self.header[2])
        self.samples = np.ndarray((self.capacity,), np.int16, segment.buf, _HEADER * 8)

    @classmethod
    def create(cls, name: str, sample_rate: int = 16000, seconds: int = 120) -> "AudioRing":
        capacity = sample_rate * seconds
        segment = shared_memory.SharedMemory(name=name, create=True, size=_HEADER * 8 + capacity * 2)
        header = np.ndarray((_HEADER,), np.int64, segment.buf, 0)
        header[:] = 0
        header[1:3] = (sample_rate, capacity)
        del header
        return cls(segment, owner=True)

    @classmethod
    def attach(cls, name: str) -> "AudioRing":
        return cls(_attach(name), owner=False)

    @property
    def written(self) -> int:
        return int(self.header[0])

    def write(self, pcm: np.ndarray) -> int:
        pcm = np.asarray(pcm, np.int16).reshape(-1)
        start = self.written
        for offset in range(0, len(pcm), self.capacity):
            part = pcm[offset:offset + self.capacity]
            index = (start + offset) % self.capacity
            first = min(len(part), self.capacity - index)
            self.samples[index:index + first] = part[:first]
            self.samples[:len(part) - first] = part[first:]
        self.header[4] = time.time_ns()
        self.header[0] = start + len(pcm)
        return start + len(pcm)

    def read(self, start: int, count: int):
        """Return samples [start, start+count) or None if not yet written / overwritten."""
        end = start + count
        if start < 0 or end > self.written:
            return None
        if self.written - start > self.capacity:
            return None
        index = start % self.capacity
        first = min(count, self.capacity - index)
        out = np.concatenate([self.samples[index:index + first], self.samples[:count - first]])
        if self.written - start > self.capacity:
            return None
        return out

    def close(self) -> None:
        del self.header, self.samples
        self.segment.close()
        if self.owner:
            self.segment.unlink()
