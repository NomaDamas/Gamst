"""Every video frame a lane has not processed yet, kept until it is processed: in memory, then in a spill file.

The video counterpart of :mod:`gamst.providers.audio_memory`. A lane declared ``backlog="spill"``
(:meth:`gamst.pipeline.Pipeline.add_module`) may fall far behind its live input (a verification lane that runs a
slower, more accurate model over every frame): its waiting frames are not thinned or given up. Up to
``memory_bytes`` of them stay in memory (as the arrays the camera gave, not copied); the rest are written to an
unnamed temporary file (deleted by the system when it is closed) and read back when the lane reaches them. The place
of a frame the lane is done with is written again by the next frame of the same size, so the file is as large as the
most frames spilled at once, not as all frames ever spilled (a lane that always has a few spilled frames, as one that
keeps a pre-roll in memory, never empties it: before, the file grew by every frame until the disk bound dropped frames);
it is emptied whenever nothing is spilled. ``disk_bytes`` bounds the file: a frame beyond it is not kept (``dropped``,
and the Pipeline declares it a ``backlog`` gap), so a lane that can never catch up does not fill the disk. The disk is
part of the budget: the bound is at most what is free beyond ``disk_floor_bytes`` (10 GB) when the memory is made
(``disk_limited`` says it was smaller than asked), and every write checks the floor again.

Frames stay inside the process and the spill file: they never become records or enter any context.
"""
from __future__ import annotations

import tempfile

MEMORY_BYTES = 256 * 1024 * 1024      # about 90 frames of 1280x720 BGR
DISK_BYTES = 8 * 1024 ** 3            # about 2,900 such frames (48 s of 60 fps)
DISK_FLOOR_BYTES = 10 * 1024 ** 3     # the spill file never takes the disk below this much free space
FLOOR_ENV = 'GAMST_SPILL_FLOOR_GB'    # an owner's override of that floor for one run (GB), e.g. a short test clip


class Kept:
    """One frame in the memory: its array, or where it is in the spill file."""
    __slots__ = ('array', 'offset', 'shape', 'dtype', 'nbytes')

    def __bool__(self):
        return True

    def __init__(self, array=None, offset=None, shape=None, dtype=None, nbytes=0):
        self.array, self.offset, self.shape, self.dtype, self.nbytes = array, offset, shape, dtype, nbytes


class VideoMemory:
    def __init__(self, *, memory_bytes=MEMORY_BYTES, disk_bytes=DISK_BYTES, spill_dir=None,
                 disk_floor_bytes=None):
        if disk_floor_bytes is None:
            import os
            given = os.environ.get(FLOOR_ENV)
            disk_floor_bytes = DISK_FLOOR_BYTES if not given else int(float(given) * 1024 ** 3)
        if int(memory_bytes) < 0 or int(disk_bytes) < 0 or int(disk_floor_bytes) < 0:
            raise ValueError('memory_bytes, disk_bytes and disk_floor_bytes must not be negative')
        self.memory_bytes, self.spill_dir, self.disk_floor_bytes = int(memory_bytes), spill_dir, int(disk_floor_bytes)
        self.asked_disk_bytes = int(disk_bytes)
        free = self._free()
        self.disk_bytes = self.asked_disk_bytes if free is None else max(0, min(self.asked_disk_bytes,
                                                                               free - self.disk_floor_bytes))
        self.disk_limited = self.disk_bytes < self.asked_disk_bytes
        self._file, self._end, self._outstanding = None, 0, 0
        self._free_slots = {}      # nbytes -> offsets in the file whose frames were released
        self.frames = self.spilled_frames = self.dropped = 0
        self.in_memory = self.spilled = self.max_in_memory = self.max_spilled = 0

    def keep(self, array):
        """A :class:`Kept` for ``array`` (a numpy array), or None when it could not be kept (disk bound or failure)."""
        import numpy as np
        array = np.asarray(array)
        size = int(array.nbytes)
        self.frames += 1
        if self.in_memory + size <= self.memory_bytes:
            self.in_memory += size
            self.max_in_memory = max(self.max_in_memory, self.in_memory)
            return Kept(array=array, nbytes=size)
        slots = self._free_slots.get(size)
        offset = slots.pop() if slots else None
        if offset is None:      # the file grows: within its bound and the disk's floor
            free = self._free()
            if self._end + size > self.disk_bytes or free is not None and free - size < self.disk_floor_bytes:
                self.dropped += 1
                return None
        try:
            if self._file is None:
                self._file = tempfile.TemporaryFile(dir=self.spill_dir)
            self._file.seek(self._end if offset is None else offset)
            self._file.write(np.ascontiguousarray(array).tobytes())
        except OSError:
            if offset is not None:
                slots.append(offset)
            self.dropped += 1
            return None
        kept = Kept(offset=self._end if offset is None else offset, shape=array.shape, dtype=array.dtype.str, nbytes=size)
        if offset is None:
            self._end += size
        self._outstanding += 1
        self.spilled += size
        self.spilled_frames += 1
        self.max_spilled = max(self.max_spilled, self.spilled)
        return kept

    def take(self, kept):
        """The frame's array (read back from the spill file if it was written there)."""
        if kept.array is not None:
            return kept.array
        import numpy as np
        self._file.seek(kept.offset)
        data = self._file.read(kept.nbytes)
        return np.frombuffer(data, dtype=np.dtype(kept.dtype)).reshape(kept.shape)

    def release(self, kept):
        """The lane is done with the frame."""
        if kept.array is not None:
            kept.array = None
            self.in_memory -= kept.nbytes
            return
        if kept.offset is None:
            return
        self._free_slots.setdefault(kept.nbytes, []).append(kept.offset)
        kept.offset = None
        self.spilled -= kept.nbytes
        self._outstanding -= 1
        if not self._outstanding and self._file is not None:   # nothing spilled any more: start the file over
            self._file.truncate(0)
            self._end = 0
            self._free_slots.clear()

    def _free(self):
        """Free bytes on the spill file's disk (None if unknown)."""
        import shutil
        import tempfile
        try:
            return shutil.disk_usage(self.spill_dir or tempfile.gettempdir()).free
        except OSError:
            return None

    def status(self):
        return dict(disk_limited=self.disk_limited, asked_disk_bytes=self.asked_disk_bytes,
                    disk_floor_bytes=self.disk_floor_bytes, frames=self.frames, in_memory_bytes=self.in_memory, spilled_bytes=self.spilled,
                    max_in_memory_bytes=self.max_in_memory, max_spilled_bytes=self.max_spilled,
                    spilled_frames=self.spilled_frames, dropped=self.dropped, file_bytes=self._end,
                    memory_bytes=self.memory_bytes, disk_bytes=self.disk_bytes)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
