"""Every audio frame of one input, kept until nobody needs it any more: in memory, then in a spill file.

The Pipeline keeps one :class:`AudioMemory` per audio input that a catch-up lane or a verification lane reads
(:mod:`gamst.pipeline`). Frames are kept as they were captured (mono float32, each with the capture time of its last
sample, the Pipeline's convention). Nothing is ever dropped for lack of room: above ``memory_bytes`` the oldest frames
move to an unnamed temporary file (deleted by the system when it is closed), and they are read back from there.

A frame is let go only when it is older than ``keep_ms`` before the newest one *and* older than every hold
(:meth:`hold`): a catch-up range that is not done yet, or the next window a verification lane will read. ``dropped``
counts frames that could not be kept (a failing disk); it is 0 in every run that kept everything.

Raw audio stays inside the process (and the spill file): it never becomes a record or enters any context.
"""
from __future__ import annotations

from collections import deque
import tempfile

from .audio_ring import mono_float


class AudioMemory:
    def __init__(self, *, memory_bytes=64 * 1024 * 1024, keep_ms=30000, spill_dir=None):
        if int(memory_bytes) < 0 or int(keep_ms) < 0:
            raise ValueError('memory_bytes and keep_ms must not be negative')
        self.memory_bytes, self.keep_ms, self.spill_dir = int(memory_bytes), int(keep_ms), spill_dir
        self._chunks = deque()      # [start_ms, end_ms, rate, subject, samples | None, offset | None, count]
        self._holds = {}
        self._file = None
        self.in_memory = self.spilled = self.max_in_memory = self.max_spilled = 0
        self.frames = self.released = self.dropped = 0

    # ------------------------------------------------------------------ keeping

    def append(self, samples, end_ms, sample_rate, subject_id='local-user'):
        import numpy as np
        try:
            values = np.array(mono_float(samples), dtype=np.float32, copy=True)
        except ValueError:
            self.dropped += 1
            return
        if not len(values) or not sample_rate:
            return
        start = float(end_ms) - 1000. * len(values) / sample_rate
        self._chunks.append([start, float(end_ms), int(sample_rate), subject_id, values, None, len(values)])
        self.frames += 1
        self.in_memory += values.nbytes
        self.max_in_memory = max(self.max_in_memory, self.in_memory)
        self._spill()
        self.release()

    def _spill(self):
        for chunk in self._chunks:
            if self.in_memory <= self.memory_bytes:
                return
            if chunk[4] is None:
                continue
            try:
                if self._file is None:
                    self._file = tempfile.TemporaryFile(dir=self.spill_dir)
                self._file.seek(0, 2)
                offset = self._file.tell()
                self._file.write(chunk[4].tobytes())
            except OSError:
                self.dropped += 1      # kept in memory after all: nothing is lost, the budget is exceeded
                return
            self.in_memory -= chunk[4].nbytes
            self.spilled += chunk[4].nbytes
            self.max_spilled = max(self.max_spilled, self.spilled)
            chunk[4], chunk[5] = None, offset

    def hold(self, name, from_ms):
        """Keep everything captured from ``from_ms`` on for ``name`` (None: ``name`` needs nothing any more)."""
        if from_ms is None:
            self._holds.pop(name, None)
        else:
            self._holds[name] = float(from_ms)
        self.release()

    def holds(self):
        return dict(self._holds)

    def release(self):
        if not self._chunks:
            return
        floor = self._chunks[-1][1] - self.keep_ms
        if self._holds:
            floor = min(floor, min(self._holds.values()))
        while self._chunks and self._chunks[0][1] < floor:
            chunk = self._chunks.popleft()
            self.released += 1
            if chunk[4] is not None:
                self.in_memory -= chunk[4].nbytes
            else:
                self.spilled -= 4 * chunk[6]
        if self._file is not None and not self.spilled:
            self._file.seek(0)
            self._file.truncate()

    # ------------------------------------------------------------------ reading

    @property
    def start_ms(self):
        return self._chunks[0][0] if self._chunks else None

    @property
    def end_ms(self):
        return self._chunks[-1][1] if self._chunks else None

    def _samples(self, chunk):
        if chunk[4] is not None:
            return chunk[4]
        import numpy as np
        self._file.seek(chunk[5])
        return np.frombuffer(self._file.read(4 * chunk[6]), dtype=np.float32)

    def chunks(self, start_ms, end_ms):
        """``[(end_ms, samples, rate, subject_id)]`` of the kept frames that overlap [start_ms, end_ms], in order."""
        return [(int(round(chunk[1])), self._samples(chunk), chunk[2], chunk[3]) for chunk in list(self._chunks)
                if chunk[1] > start_ms and chunk[0] < end_ms]

    def after(self, end_ms):
        """The first kept frame that ends after ``end_ms`` (None: nothing newer is kept yet)."""
        for chunk in self._chunks:
            if chunk[1] > end_ms:
                return int(round(chunk[1])), self._samples(chunk), chunk[2], chunk[3]
        return None

    def span(self, start_ms, end_ms, sample_rate=16000):
        """Samples captured in [start_ms, end_ms) that are kept (at ``sample_rate``; other rates are left out)."""
        import numpy as np
        parts = []
        for chunk in list(self._chunks):
            if chunk[1] <= start_ms or chunk[0] >= end_ms or chunk[2] != sample_rate:
                continue
            samples = self._samples(chunk)
            first = max(0, int(round((start_ms - chunk[0]) * sample_rate / 1000)))
            last = min(len(samples), int(round((end_ms - chunk[0]) * sample_rate / 1000)))
            if last > first:
                parts.append(samples[first:last])
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def covers(self, start_ms, end_ms):
        return bool(self._chunks) and self._chunks[0][0] <= start_ms + 1 and self._chunks[-1][1] >= end_ms - 1

    def status(self):
        return dict(frames=self.frames, kept_from_ms=None if self.start_ms is None else int(self.start_ms),
                    kept_until_ms=None if self.end_ms is None else int(self.end_ms), bytes_in_memory=self.in_memory,
                    bytes_spilled=self.spilled, max_bytes_in_memory=self.max_in_memory, max_bytes_spilled=self.max_spilled,
                    memory_bytes=self.memory_bytes, released=self.released, dropped=self.dropped,
                    holds={key: int(value) for key, value in self._holds.items()})

    def close(self):
        self._chunks.clear()
        self.in_memory = self.spilled = 0
        if self._file is not None:
            self._file.close()
            self._file = None
