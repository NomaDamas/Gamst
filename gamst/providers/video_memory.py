"""Every video frame a lane has not processed yet, kept until it is processed, losslessly and as compactly as it can be.

The video counterpart of :mod:`gamst.providers.audio_memory`. A lane declared ``backlog="spill"``
(:meth:`gamst.pipeline.Pipeline.add_module`) may fall far behind its live input (a verification lane that runs a
slower, more accurate model over every frame; an L2 chain's intervals in realtime): its waiting frames are not thinned or
given up. Where a frame is kept, in this order:

1. **memory**: up to ``memory_bytes`` of them stay as the arrays the input gave (not copied);
2. **its place in the file** (``origin``: a re-readable recording, ``--realtime`` replay of a file): only the frame's
   index and a CRC-32 of its pixels are kept, no pixel; :meth:`take` decodes it again from the file
   (:class:`gamst.inputs.FileFrames`) and checks the CRC. On a mismatch it decodes again from the file's start; a frame
   that still differs is not given (``take`` returns None, the Pipeline declares it): never a different frame;
3. **compressed in a spill file** (a live camera): losslessly, PNG (level 1, OpenCV) or zlib, encoded by a worker
   thread beside the event loop, in unnamed temporary files (deleted by the system when closed) of ``CHUNK_BYTES``
   each; a chunk whose frames were all taken back is deleted, so the files hold about what is waiting, not all that
   ever waited. ``codec="raw"`` keeps the uncompressed spill file of earlier versions (a place given back is written
   again by the next frame of the same size).

``disk_bytes`` bounds the spill: a frame beyond it is not kept (``dropped``, and the Pipeline declares it a ``backlog``
gap), so a lane that can never catch up does not fill the disk. The disk is part of the budget: the bound is at most
what is free beyond ``disk_floor_bytes`` (5 GB; ``GAMST_SPILL_FLOOR_GB`` overrides it) when the memory is made
(``disk_limited`` says it was smaller than asked), and every write checks the floor again (a frame being compressed when
the floor is reached stays in memory instead: a frame that was taken is never lost).

Frames stay inside the process and its files: they never become records or enter any context.
"""
from __future__ import annotations

import tempfile
import threading
import time
import zlib

MEMORY_BYTES = 256 * 1024 * 1024      # about 90 frames of 1280x720 BGR
DISK_BYTES = 8 * 1024 ** 3            # the spill files' bound (compressed frames: PNG keeps 1280x720 at 2-6x smaller)
DISK_FLOOR_BYTES = 5 * 1024 ** 3      # the spill files never take the disk below this much free space
FLOOR_ENV = 'GAMST_SPILL_FLOOR_GB'    # an owner's override of that floor for one run (GB), e.g. a short test clip
CODECS = ('png', 'zlib', 'raw')
CHUNK_BYTES = 64 * 1024 * 1024        # one spill file of compressed frames; deleted once all its frames are taken
MAX_PENDING = 64                      # frames waiting for the encoder; beyond this, keep() encodes the frame itself
PNG_LEVEL = 1                         # the fastest PNG level; level 3 is ~6 % smaller and ~40 % slower (measured)
ZLIB_LEVEL = 1


class Kept:
    """One frame in the memory: its array, where it is in a spill file, or where it is in its recording."""
    __slots__ = ('array', 'offset', 'shape', 'dtype', 'nbytes', 'origin', 'digest', 'chunk', 'length', 'future',
                 'codec', 'released', 'estimate')

    def __bool__(self):
        return True

    def __init__(self, array=None, offset=None, shape=None, dtype=None, nbytes=0, origin=None, digest=None):
        self.array, self.offset, self.shape, self.dtype, self.nbytes = array, offset, shape, dtype, nbytes
        self.origin, self.digest = origin, digest
        self.chunk = self.length = self.future = self.codec = None
        self.released, self.estimate = False, 0


class _Chunk:
    __slots__ = ('file', 'end', 'outstanding')

    def __init__(self, file):
        self.file, self.end, self.outstanding = file, 0, 0


def _digest(array):
    import numpy as np
    return zlib.crc32(memoryview(np.ascontiguousarray(array)).cast('B'))


def _encode(array, codec):
    """``(codec used, bytes)``: PNG for 8/16-bit images of 1, 3 or 4 channels, else zlib of the raw bytes."""
    import numpy as np
    if codec == 'png' and array.dtype in (np.uint8, np.uint16) and (
            array.ndim == 2 or array.ndim == 3 and array.shape[2] in (1, 3, 4)):
        try:
            import cv2
            ok, found = cv2.imencode('.png', array, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
            if ok:
                return 'png', found.tobytes()
        except ImportError:
            pass
    return 'zlib', zlib.compress(memoryview(np.ascontiguousarray(array)).cast('B'), ZLIB_LEVEL)


def _decode(data, codec, shape, dtype):
    import numpy as np
    if codec == 'png':
        import cv2
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        return image.reshape(shape)
    return np.frombuffer(zlib.decompress(data), dtype=np.dtype(dtype)).reshape(shape)


def default_codec():
    try:
        import cv2  # noqa: F401
        return 'png'
    except ImportError:
        return 'zlib'


class VideoMemory:
    def __init__(self, *, memory_bytes=MEMORY_BYTES, disk_bytes=DISK_BYTES, spill_dir=None,
                 disk_floor_bytes=None, codec=None, chunk_bytes=CHUNK_BYTES, workers=2):
        if disk_floor_bytes is None:
            import os
            given = os.environ.get(FLOOR_ENV)
            disk_floor_bytes = DISK_FLOOR_BYTES if not given else int(float(given) * 1024 ** 3)
        if int(memory_bytes) < 0 or int(disk_bytes) < 0 or int(disk_floor_bytes) < 0:
            raise ValueError('memory_bytes, disk_bytes and disk_floor_bytes must not be negative')
        codec = default_codec() if codec is None else codec
        if codec not in CODECS:
            raise ValueError(f'codec is one of {CODECS}')
        self.memory_bytes, self.spill_dir, self.disk_floor_bytes = int(memory_bytes), spill_dir, int(disk_floor_bytes)
        self.codec, self.chunk_bytes, self.workers = codec, max(1, int(chunk_bytes)), max(1, int(workers))
        self.asked_disk_bytes = int(disk_bytes)
        free = self._free()
        self.disk_bytes = self.asked_disk_bytes if free is None else max(0, min(self.asked_disk_bytes,
                                                                               free - self.disk_floor_bytes))
        self.disk_limited = self.disk_bytes < self.asked_disk_bytes
        self._file, self._end, self._outstanding = None, 0, 0     # the raw spill file
        self._free_slots = {}      # nbytes -> offsets in the raw file whose frames were released
        self.frames = self.spilled_frames = self.dropped = 0
        self.in_memory = self.spilled = self.max_in_memory = self.max_spilled = 0
        # frames kept by their place in a recording
        self.reread_frames = self.rereading = self.max_rereading = self.reread_bytes = 0
        self.rereads = self.reread_mismatches = self.reread_failed = 0
        self.reread_s = 0.
        # compressed spill
        self._lock = threading.Lock()
        self._chunks, self._chunk = [], None
        self._executor, self._pending, self._pending_bytes = None, 0, 0
        self.compressed_frames = self.compressed_raw_bytes = self.compressed_bytes = 0
        self.encode_s = self.decode_s = 0.
        self.kept_after_error = 0

    # ------------------------------------------------------------------ keep

    def keep(self, array, origin=None):
        """A :class:`Kept` for ``array`` (a numpy array), or None when it could not be kept (disk bound or failure).

        ``origin``: ``(reader, index)`` where the frame is in its recording (``reader.read(index)`` gives it again,
        :class:`gamst.inputs.FileFrames`); beyond the memory, only that is kept."""
        import numpy as np
        array = np.asarray(array)
        size = int(array.nbytes)
        self.frames += 1
        if self.in_memory + size <= self.memory_bytes:
            self.in_memory += size
            self.max_in_memory = max(self.max_in_memory, self.in_memory)
            return Kept(array=array, nbytes=size)
        if origin is not None:
            self.reread_frames += 1
            self.rereading += 1
            self.max_rereading = max(self.max_rereading, self.rereading)
            self.reread_bytes += size
            return Kept(origin=origin, digest=_digest(array), shape=array.shape, dtype=array.dtype.str, nbytes=size)
        if self.codec == 'raw':
            return self._keep_raw(array, size)
        return self._keep_compressed(array, size)

    def _keep_raw(self, array, size):
        import numpy as np
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

    def _ratio(self):
        """The compression measured so far (raw / stored), 1 before the first frame: what a frame is expected to take."""
        return self.compressed_raw_bytes / self.compressed_bytes if self.compressed_bytes else 1.

    def _keep_compressed(self, array, size):
        estimate = int(size / self._ratio()) + 1
        free = self._free()
        with self._lock:
            used = self._disk_used() + self._pending_bytes
        if used + estimate > self.disk_bytes or free is not None and free - estimate - self._pending_bytes < \
                self.disk_floor_bytes:
            self.dropped += 1
            return None
        kept = Kept(array=array, shape=array.shape, dtype=array.dtype.str, nbytes=size)
        kept.estimate = estimate
        with self._lock:
            self._pending += 1
            self._pending_bytes += estimate
        self.spilled_frames += 1
        if self._pending > MAX_PENDING:      # the encoder is behind: this frame is encoded here (backpressure)
            self._write(kept)
        else:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(self.workers, thread_name_prefix='gamst-video-memory')
            kept.future = self._executor.submit(self._write, kept)
        return kept

    def _disk_used(self):
        return sum(chunk.end for chunk in self._chunks)

    def _write(self, kept):
        """Encode a frame and append it to the current chunk (a worker thread; or keep() itself when it is behind)."""
        array = kept.array
        started = time.perf_counter()
        codec, data = _encode(array, self.codec)
        took = time.perf_counter() - started
        with self._lock:
            self._pending -= 1
            self._pending_bytes -= kept.estimate
            self.encode_s += took
            if kept.released:
                return
            free = self._free()
            try:
                if free is not None and free - len(data) < self.disk_floor_bytes:
                    raise OSError('the disk floor')
                chunk = self._chunk
                if chunk is None or chunk.end + len(data) > self.chunk_bytes and chunk.end:
                    chunk = self._chunk = _Chunk(tempfile.TemporaryFile(dir=self.spill_dir))
                    self._chunks.append(chunk)
                chunk.file.seek(chunk.end)
                chunk.file.write(data)
            except OSError:      # it stays in memory: a frame that was taken is never lost
                self.kept_after_error += 1
                self.in_memory += kept.nbytes
                self.max_in_memory = max(self.max_in_memory, self.in_memory)
                kept.codec = 'memory'
                return
            kept.chunk, kept.offset, kept.length, kept.codec = chunk, chunk.end, len(data), codec
            chunk.end += len(data)
            chunk.outstanding += 1
            kept.array = None
            self.compressed_frames += 1
            self.compressed_raw_bytes += kept.nbytes
            self.compressed_bytes += len(data)
            self.spilled += len(data)
            self.max_spilled = max(self.max_spilled, self.spilled)

    # ------------------------------------------------------------------ take

    def take(self, kept):
        """The frame's array (read back from a spill file, or decoded again from its recording), or None when its
        recording no longer gives the same frame (it is not replaced by another)."""
        if kept.future is not None:
            kept.future.result()
            kept.future = None
        if kept.array is not None:
            return kept.array
        if kept.origin is not None:
            return self._reread(kept)
        if kept.chunk is not None:
            started = time.perf_counter()
            with self._lock:
                kept.chunk.file.seek(kept.offset)
                data = kept.chunk.file.read(kept.length)
            found = _decode(data, kept.codec, kept.shape, kept.dtype)
            self.decode_s += time.perf_counter() - started
            return found
        import numpy as np
        self._file.seek(kept.offset)
        data = self._file.read(kept.nbytes)
        return np.frombuffer(data, dtype=np.dtype(kept.dtype)).reshape(kept.shape)

    def _reread(self, kept):
        reader, index = kept.origin
        started = time.perf_counter()
        self.rereads += 1
        try:
            found = reader.read(index)
            if found is None or tuple(found.shape) != tuple(kept.shape) or _digest(found) != kept.digest:
                self.reread_mismatches += 1
                found = reader.read(index, fresh=True)
                if found is None or tuple(found.shape) != tuple(kept.shape) or _digest(found) != kept.digest:
                    self.reread_failed += 1
                    return None
            return found
        except Exception:
            self.reread_failed += 1
            return None
        finally:
            self.reread_s += time.perf_counter() - started

    # ------------------------------------------------------------------ release

    def release(self, kept):
        """The lane is done with the frame."""
        if kept.released:
            return
        if kept.origin is not None:
            kept.released, kept.origin = True, None
            self.rereading -= 1
            return
        if kept.codec is not None or kept.future is not None or kept.chunk is not None:     # the compressed spill
            self._release_compressed(kept)
            return
        if kept.array is not None:
            kept.array = None
            self.in_memory -= kept.nbytes
            kept.released = True
            return
        if kept.offset is None:
            return
        kept.released = True
        self._free_slots.setdefault(kept.nbytes, []).append(kept.offset)
        kept.offset = None
        self.spilled -= kept.nbytes
        self._outstanding -= 1
        if not self._outstanding and self._file is not None:   # nothing spilled any more: start the file over
            self._file.truncate(0)
            self._end = 0
            self._free_slots.clear()

    def _release_compressed(self, kept):
        with self._lock:
            kept.released = True
            if kept.codec == 'memory':          # kept in memory after a failed write
                self.in_memory -= kept.nbytes
                kept.array = None
                return
            chunk = kept.chunk
            kept.array = None
            if chunk is None:                  # still with the encoder: it sees ``released`` and drops it
                return
            kept.chunk = None
            self.spilled -= kept.length
            chunk.outstanding -= 1
            if chunk.outstanding:
                return
            if chunk is self._chunk:            # the current chunk, empty: written again from its start
                chunk.file.seek(0)
                chunk.file.truncate(0)
                chunk.end = 0
            else:                               # a full chunk whose frames were all taken: the file goes
                self._chunks.remove(chunk)
                chunk.file.close()

    # ------------------------------------------------------------------ the rest

    def _free(self):
        """Free bytes on the spill file's disk (None if unknown)."""
        import shutil
        import tempfile
        try:
            return shutil.disk_usage(self.spill_dir or tempfile.gettempdir()).free
        except OSError:
            return None

    def status(self):
        with self._lock:
            files = self._end + self._disk_used()
        return dict(disk_limited=self.disk_limited, asked_disk_bytes=self.asked_disk_bytes,
                    disk_floor_bytes=self.disk_floor_bytes, frames=self.frames, in_memory_bytes=self.in_memory,
                    spilled_bytes=self.spilled, max_in_memory_bytes=self.max_in_memory, max_spilled_bytes=self.max_spilled,
                    spilled_frames=self.spilled_frames, dropped=self.dropped, file_bytes=files,
                    memory_bytes=self.memory_bytes, disk_bytes=self.disk_bytes, codec=self.codec,
                    reread=dict(frames=self.reread_frames, waiting=self.rereading, max_waiting=self.max_rereading,
                                bytes_not_stored=self.reread_bytes, reads=self.rereads, mismatches=self.reread_mismatches,
                                failed=self.reread_failed, read_ms=round(1000 * self.reread_s, 1)),
                    compressed=dict(frames=self.compressed_frames, raw_bytes=self.compressed_raw_bytes,
                                    stored_bytes=self.compressed_bytes, ratio=round(self._ratio(), 3),
                                    encode_ms=round(1000 * self.encode_s, 1), decode_ms=round(1000 * self.decode_s, 1),
                                    files=len(self._chunks), kept_in_memory_after_error=self.kept_after_error))

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._file is not None:
            self._file.close()
            self._file = None
        with self._lock:
            for chunk in self._chunks:
                chunk.file.close()
            self._chunks, self._chunk = [], None
