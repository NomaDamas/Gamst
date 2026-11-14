"""Owned bounded recording artifacts; hashes do not attest physical measurements."""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

MAX_EVENT_BYTES = 2048
MAX_QUEUE_FRAMES = 64
MAX_QUEUE_BYTES = 1024 * 1024
MAX_JOURNAL_BYTES = 1024 ** 3
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSON_TOTAL = 80 * 1024 * 1024
MAX_JSON_FILES = 128
MAX_TERMINAL_BYTES = 1024 * 1024
ZERO = '0' * 64


def preflight(predicted_events):
    if type(predicted_events) is not int or not 0 <= predicted_events <= MAX_JOURNAL_BYTES // MAX_EVENT_BYTES:
        raise ValueError('predicted recording event inventory exceeds journal capacity')
    return predicted_events * MAX_EVENT_BYTES


def _json_shape(value, budget, depth=0):
    if depth > 64:
        raise ValueError('JSON nesting exceeds recording artifact bound')
    kind = type(value)
    if value is None or kind in (bool, float, int, str):
        if kind is float and not math.isfinite(value):
            raise ValueError('finite JSON numbers required')
        if kind is int and value.bit_length() > budget[0] * 4:
            raise ValueError('JSON integer exceeds artifact bound')
        budget[0] -= len(value) + 2 if kind is str else len(str(value))
    elif kind is list:
        budget[0] -= 2
        for item in value:
            _json_shape(item, budget, depth + 1)
    elif kind is dict:
        budget[0] -= 2
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('JSON string keys required')
            budget[0] -= len(key) + 3
            _json_shape(item, budget, depth + 1)
    else:
        raise ValueError('strict finite JSON values required')
    if budget[0] < 0:
        raise ValueError('recording JSON exceeds byte bound')


def _encode(value, limit):
    _json_shape(value, [limit])
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    data = bytearray()
    for text in encoder.iterencode(value):
        chunk = text.encode('ascii')
        if len(data) + len(chunk) > limit:
            raise ValueError('recording JSON exceeds byte bound')
        data.extend(chunk)
    return bytes(data)


def _name(name):
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,122}\.json', name) or name == 'terminal.json':
        raise ValueError('exclusive simple JSON artifact filename required')
    return name


async def _finish(task):
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError('recording artifact write made no progress')
        view = view[count:]


class RecordingArtifacts:
    """Single-event-loop owner; root cleanup precedes immutable terminal creation."""
    @classmethod
    async def open(cls, output, *, declarations, predicted_events):
        preflight(predicted_events)
        if type(declarations) is not dict or len(declarations) >= MAX_JSON_FILES:
            raise ValueError('bounded declaration file inventory required')
        for name in declarations:
            _name(name)
        self = cls(); self.output = Path(output).absolute()
        self.predicted_events = predicted_events
        self._queue = asyncio.Queue(MAX_QUEUE_FRAMES); self._wake = asyncio.Event()
        self._pending_frames = self._pending_bytes = 0
        self._admitted = self._admitted_bytes = self._written = self._written_bytes = 0
        self._last = self._durable_last = ZERO
        self._failure = None; self._accepting = True; self._files_accepting = True; self._stopping = False
        self._fd = None; self._directory_fd = None; self._writer = None; self._close_task = None; self._abort_task = None
        self._threads = set(); self._writes = set(); self._files = {}; self._json_reserved = 0
        self._aborted = False
        self._file_slots = asyncio.Semaphore(4); self._terminal = None
        try:
            await self._thread(self._create)
            for name, value in declarations.items():
                data = await self._thread(_encode, value, MAX_JSON_BYTES)
                if self._json_reserved + len(data) > MAX_JSON_TOTAL - MAX_TERMINAL_BYTES:
                    raise ValueError('aggregate declaration bytes exceed artifact capacity')
                self._json_reserved += len(data); self._files[name] = None
                self._files[name] = await self._thread(self._save_bytes, name, data)
            self._writer = asyncio.create_task(self._write())
            return self
        except BaseException:
            self._accepting = False
            cleanup = asyncio.create_task(self._cleanup_open())
            await _finish(cleanup)
            raise

    async def _cleanup_open(self):
        try:
            await self._thread(self._close_fd)
        finally:
            await self._thread(self._close_directory)

    def _create(self):
        self.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(self.output, 0o700)
        self._directory_fd = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._fd = os.open('journal.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self._directory_fd)
        os.fchmod(self._fd, 0o600)

    async def _thread(self, fn, *args):
        task = asyncio.create_task(asyncio.to_thread(fn, *args))
        self._threads.add(task)
        try:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _finish(task)
                raise
        finally:
            self._threads.discard(task)

    def _fail(self, reason):
        if self._failure is None:
            self._failure = reason
        self._accepting = False

    def emit(self, event, fields):
        if not self._accepting or self._failure is not None:
            raise RuntimeError('recording journal admission is closed or failed')
        if type(event) is not str or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', event) or type(fields) is not dict:
            self._fail('invalid_event')
            raise ValueError('bounded structural event and JSON fields required')
        try:
            body = dict(schema_version=1, sequence=self._admitted, previous_sha256=self._last,
                        monotonic_ns=time.monotonic_ns(), event=event, fields=fields)
            digest = hashlib.sha256(_encode(body, MAX_EVENT_BYTES)).hexdigest()
            data = _encode(dict(body, sha256=digest), MAX_EVENT_BYTES - 1) + b'\n'
        except BaseException:
            self._fail('event_encoding_failed')
            raise
        size = len(data)
        if (self._admitted >= self.predicted_events or self._admitted_bytes + size > MAX_JOURNAL_BYTES or
                self._pending_frames >= MAX_QUEUE_FRAMES or self._pending_bytes + size > MAX_QUEUE_BYTES):
            self._fail('journal_capacity_exhausted')
            raise RuntimeError('recording journal capacity exhausted before admission')
        self._queue.put_nowait((data, digest))
        self._pending_frames += 1; self._pending_bytes += size
        self._admitted += 1; self._admitted_bytes += size; self._last = digest
        self._wake.set()
        return dict(sequence=self._admitted - 1, sha256=digest)

    def _append(self, rows, offset):
        try:
            for data, _ in rows:
                _write_all(self._fd, data)
            os.fsync(self._fd)
        except BaseException:
            # Only our current append is rolled back. Previously fsynced rows
            # are never evicted. If rollback fails, valid_prefix_bytes still
            # identifies the last confirmed prefix; the terminal is failed.
            try:
                os.ftruncate(self._fd, offset); os.lseek(self._fd, offset, os.SEEK_SET)
                os.fsync(self._fd)
            except OSError:
                pass
            raise

    async def _write(self):
        try:
            while True:
                if self._queue.empty():
                    if self._stopping:
                        return
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                rows = []
                while not self._queue.empty() and len(rows) < MAX_QUEUE_FRAMES:
                    rows.append(self._queue.get_nowait())
                await self._thread(self._append, rows, self._written_bytes)
                size = sum(len(data) for data, _ in rows)
                self._written += len(rows); self._written_bytes += size; self._durable_last = rows[-1][1]
                self._pending_frames -= len(rows); self._pending_bytes -= size
        except BaseException:
            self._fail('journal_write_failed')
        finally:
            if self._failure is not None:
                while not self._queue.empty():
                    self._queue.get_nowait()
                self._pending_frames = self._pending_bytes = 0

    def _save_json(self, name, value, limit):
        return self._save_bytes(name, _encode(value, limit))

    def _save_bytes(self, name, data):
        current = os.stat(self.output, follow_symlinks=False)
        owned = os.fstat(self._directory_fd)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise ValueError('owned recording output directory was replaced')
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self._directory_fd)
        try:
            os.fchmod(fd, 0o600); _write_all(fd, data); os.fsync(fd)
        finally:
            os.close(fd)
        return dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    async def write_json(self, name, value, *, max_bytes=65536):
        _name(name)
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_JSON_BYTES:
            raise ValueError('bounded JSON file bytes required')
        if not self._files_accepting or name in self._files or len(self._files) >= MAX_JSON_FILES - 1:
            raise RuntimeError('recording JSON inventory closed, duplicate or exhausted')
        if self._json_reserved + max_bytes > MAX_JSON_TOTAL - MAX_TERMINAL_BYTES:
            self._fail('json_capacity_exhausted')
            raise RuntimeError('recording JSON aggregate reservation exhausted')
        self._json_reserved += max_bytes; self._files[name] = None
        task = asyncio.create_task(self._write_json(name, value, max_bytes)); self._writes.add(task)
        try:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _finish(task)
                raise
        finally:
            self._writes.discard(task)

    async def _write_json(self, name, value, max_bytes):
        try:
            async with self._file_slots:
                result = await self._thread(self._save_json, name, value, max_bytes)
            self._files[name] = result
            self._json_reserved -= max_bytes - result['bytes']
            return dict(result)
        except BaseException:
            self._fail('json_write_failed')
            raise

    def status(self):
        return dict(schema_version=1, admitted_events=self._admitted, written_events=self._written,
            admitted_bytes=self._admitted_bytes, valid_prefix_bytes=self._written_bytes,
            last_admitted_sha256=self._last, valid_prefix_sha256=self._durable_last,
            unwritten_events=self._admitted-self._written, queued_frames=self._pending_frames,
            queued_bytes=self._pending_bytes, predicted_events=self.predicted_events,
            failure=self._failure, accepting=self._accepting, writer_done=self._writer is None or self._writer.done(),
            owned_write_jobs=len(self._threads), json_reserved_bytes=self._json_reserved,
            actual_acceptance='incomplete', measurement_truth_verified=False, instrumentation_coverage_verified=False,
            receipt_authenticity_verified=False, isolation_verified=False, whole_pipeline_cost_verified=False)

    def _close_fd(self):
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)

    def _close_directory(self):
        fd, self._directory_fd = self._directory_fd, None
        if fd is not None:
            os.close(fd)

    async def _drain(self):
        self._accepting = False; self._files_accepting = False; self._stopping = True; self._wake.set()
        if self._writer is not None:
            await _finish(self._writer)
        for task in tuple(self._writes):
            try:
                await _finish(task)
            except Exception:
                self._fail('json_write_failed')
        for task in tuple(self._threads):
            await _finish(task)
        await self._thread(self._close_fd)

    async def close(self, summary, *, cleanup_finished):
        if cleanup_finished is not True:
            raise ValueError('caller-owned runtime cleanup must finish before terminal publication')
        if self._abort_task is not None:
            await _finish(self._abort_task)
            raise RuntimeError('aborted recording artifacts cannot publish a terminal')
        self._accepting = False; self._files_accepting = False
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(summary))
        try:
            return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise

    async def _close(self, summary):
        try:
            await self._drain()
            if self._aborted:
                raise RuntimeError('aborted recording cannot publish a terminal')
            state = 'failed' if (self._failure is not None or type(summary) is not dict or
                summary.get('state') not in ('completed', 'success', 'replayed_unscored') or
                summary.get('failure_category') is not None) else summary['state']
            value = dict(schema_version=1, state=state, journal=self.status(), summary=summary,
                files=[value for value in self._files.values() if value is not None],
                cleanup_finished=True, actual_acceptance='incomplete', measurement_truth_verified=False,
                receipt_authenticity_verified=False, instrumentation_coverage_verified=False,
                isolation_verified=False, whole_pipeline_cost_verified=False)
            self._terminal = await self._thread(self._save_json, 'terminal.json', value, MAX_TERMINAL_BYTES)
            if self._failure is not None:
                raise RuntimeError('recording artifacts incomplete: '+self._failure)
            return dict(self._terminal)
        finally:
            await self._thread(self._close_directory)

    async def abort(self):
        if self._terminal is not None:
            return
        if self._abort_task is None:
            self._aborted = True; self._files_accepting = False; self._fail('aborted')
            self._abort_task = asyncio.create_task(self._abort())
        try:
            await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            await _finish(self._abort_task)
            raise

    async def _abort(self):
        if self._close_task is not None:
            try:
                await _finish(self._close_task)
            except Exception:
                pass
        try:
            await self._drain()
        finally:
            await self._thread(self._close_directory)
