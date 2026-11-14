"""Bounded native4 files. Integrity of these files is not billing authenticity."""
import asyncio
from collections import deque
import hashlib
import os
from pathlib import Path
import re
import threading
import time

from .recording_artifacts import _encode, _write_all, _finish

EVENT_BYTES = 2048
CHUNK_BYTES = 1024 * 1024
HEADER_BYTES = 4096
FOOTER_BYTES = 2048
PAGE_BYTES = 65536
QUEUE_FRAMES = 64
QUEUE_BYTES = 1024 * 1024
SUMMARY_KEYS = frozenset(('mapping_sha256', 'reservation_count', 'released_count',
    'outstanding_tickets', 'outstanding_entries', 'outstanding_work',
    'registered_workers', 'unclosed_workers', 'unsettled_entries', 'counter_pages',
    'incomplete_counts'))
CONTEXT_KEYS = frozenset(('schema_version', 'kind', 'nonce', 'invocation_ordinal',
    'collection_sha256', 'scope_protocol_sha256', 'instrumentation_protocol_sha256'))


def _int(value, maximum, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('bounded exact integer required')
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lowercase SHA256 required')
    return value


def _id(value):
    if type(value) is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value) is None:
        raise ValueError('bounded original identifier required')
    return value


def _line(value, limit):
    return _encode(value, limit - 1) + b'\n'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


class ArtifactBudget:
    """One shared byte admission budget, including precharged source artifacts.

    Reservations are consumed permanently even on failed/partial exclusive writes;
    failure never creates extra capacity by hiding retained files.
    """
    def __init__(self, limit=4*1024**3, initial_bytes=0):
        self.limit = _int(limit, 4*1024**3, 1)
        self._used = _int(initial_bytes, self.limit)
        self._lock = threading.Lock()
        self._financial = dict(source_bytes=0, attachment_bytes=0, attachments=0, declared_lines=0)
        self._claims = set(); self._documents = {}

    def bind_collection(self, collection_sha256, precharge):
        """Charge original source/staging maxima once for all parent run copies."""
        _sha(collection_sha256);_int(precharge,self.limit)
        with self._lock:
            previous=getattr(self,'_collection_binding',None)
            if previous is not None:
                if previous!=(collection_sha256,precharge):raise ValueError('artifact budget belongs to another frozen collection')
                return
            if self._used+precharge>self.limit:raise RuntimeError('collection source/staging precharge exceeds artifact budget')
            self._used+=precharge;self._collection_binding=(collection_sha256,precharge)

    def reserve(self, size):
        _int(size, self.limit)
        with self._lock:
            if self._used + size > self.limit:
                raise RuntimeError('shared collection artifact byte capacity exhausted')
            self._used += size

    def admit_source(self, size):
        with self._lock:
            if self._financial['source_bytes'] + size > 64*1024**2:
                raise RuntimeError('collection receipt source capacity exhausted')
            self._financial['source_bytes'] += size

    def admit_attachment(self, size, document_key, document_binding, line_count=1):
        _int(line_count, 64, 1)
        with self._lock:
            previous = self._documents.get(document_key)
            if previous is not None and previous != document_binding:
                raise ValueError('original invoice revision/document changed')
            if self._financial['declared_lines'] + line_count > 262144:
                raise RuntimeError('collection declared line inventory exhausted before storage')
            if self._financial['attachments'] >= 16384 or self._financial['attachment_bytes'] + size > 64*1024**2:
                raise RuntimeError('collection receipt attachment capacity exhausted')
            self._documents[document_key] = document_binding
            self._financial['attachments'] += 1; self._financial['attachment_bytes'] += size
            self._financial['declared_lines'] += line_count

    def admit_claims(self, hashes):
        values = [bytes.fromhex(_sha(value)) for value in hashes]
        with self._lock:
            if len(set(values)) != len(values) or any(value in self._claims for value in values):
                raise ValueError('duplicate global original invoice line claim')
            if len(self._claims) + len(values) > 262144:
                raise RuntimeError('collection invoice claim capacity exhausted')
            self._claims.update(values)

    def release_unused(self, size):
        _int(size, self.limit)
        with self._lock:
            if size > self._used:
                raise ValueError('unreserved collection byte release')
            self._used -= size

    @property
    def used(self):
        with self._lock:
            return self._used


class _Reservation:
    def __init__(self, owner, events, size):
        self.owner = owner
        self.events = events
        self.bytes = size
        self.released = False


class RecordingNativeWriter:
    @classmethod
    async def open(cls, output, context, *, max_events=262144, max_chunks=1024,
                   max_bytes=4*1024**3, budget=None, retain_financial_documents=False):
        if type(context) is not dict or set(context) != CONTEXT_KEYS:
            raise ValueError('exact native4 run context required')
        if type(context['schema_version']) is not int or context['schema_version'] != 4 or context['kind'] != 'recording_native':
            raise ValueError('native4 run context required')
        for key in CONTEXT_KEYS - {'schema_version', 'kind', 'invocation_ordinal'}:
            _sha(context[key])
        _int(context['invocation_ordinal'], 95)
        if type(retain_financial_documents) is not bool:
            raise ValueError('explicit financial retention boolean required')
        self = cls()
        self.context = dict(context)
        self.output = Path(output).absolute()
        self.max_events = _int(max_events, 262144, 1)
        self.max_chunks = _int(max_chunks, 1024, 1)
        self.max_bytes = _int(max_bytes, 4*1024**3, 1)
        self.budget = budget if budget is not None else ArtifactBudget(self.max_bytes)
        if type(self.budget) is not ArtifactBudget:
            raise TypeError('shared ArtifactBudget required')
        self.retain_financial_documents = retain_financial_documents
        self._loop = asyncio.get_running_loop()
        self._lock = threading.RLock()
        self._queue = deque(); self._wake = asyncio.Event(); self._sealed_wake = asyncio.Event(); self._drained_wake = asyncio.Event()
        self._pending_frames = self._pending_bytes = self._reserved_events = self._reserved_bytes = 0
        self._events = self._entries = self._written_events = self._run_bytes = 0
        self._last = _hash(_encode(['gamst.recording.events.v1', self.context], PAGE_BYTES))
        self._last_time = 0
        self._planned_chunk = 0; self._planned_events = self._planned_entries = self._planned_bytes = 0
        self._failure = None; self._accepting = True; self._stopping = False; self._aborted = False
        self._directory_fd = self._fd = None
        self._chunk = None; self._chunks = []; self._pages = []; self._page_pending = []
        self._writer = self._close_task = self._abort_task = None
        self._threads = set(); self._file_jobs = set(); self._file_lock = asyncio.Lock()
        self._terminal = None; self._sources = {}; self._receipts = {}; self._claims = {}; self._counters = []
        self._source_bytes = self._receipt_bytes = 0; self._counter_pairs = set()
        self._framing_credit = 0; self._settlement_refs = set()
        self._metadata_credit = PAGE_BYTES * (1 + (self.max_chunks + 127)//128)
        if self._metadata_credit > self.max_bytes:
            raise ValueError("run bound cannot reserve index and terminal capacity")
        self.budget.reserve(self._metadata_credit)
        try:
            await self._thread(self._create)
            self._writer = asyncio.create_task(self._write())
            return self
        except BaseException:
            await _finish(asyncio.create_task(self._cleanup()))
            raise

    def _create(self):
        self.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(self.output, 0o700)
        self._directory_fd = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def _check_directory(self):
        a = os.stat(self.output, follow_symlinks=False)
        b = os.fstat(self._directory_fd)
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise RuntimeError('owned artifact directory was replaced')

    async def _thread(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
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
        with self._lock:
            self._failure = self._failure or reason
            self._accepting = False

    def bind_capacity(self, capacity):
        """Narrow an unused writer to its frozen run limits before activation."""
        with self._lock:
            if self._events or self._reserved_events or self._failure:
                raise RuntimeError('writer capacity must bind before native events')
            self.max_events=min(self.max_events,_int(capacity['max_events_per_run'],262144,1))
            self.max_chunks=min(self.max_chunks,_int(capacity['max_chunks_per_run'],1024,1))
            self.max_bytes=min(self.max_bytes,_int(capacity['max_collection_bytes'],4*1024**3,1))

    def reserve(self, events, bytes=None):
        events = _int(events, self.max_events, 1)
        size = events * EVENT_BYTES if bytes is None else _int(bytes, self.max_bytes, 1)
        if size < events:
            raise ValueError('reservation bytes insufficient for events')
        with self._lock:
            if not self._accepting or self._failure:
                raise RuntimeError('native writer admission closed')
            if self._events + self._reserved_events + events > self.max_events:
                raise RuntimeError('native event reservation capacity exhausted')
            # Exact file bytes are charged separately; reservation protects event
            # capacity and the worst-case run output before invocation admission.
            if self._run_bytes + self._reserved_bytes + size + HEADER_BYTES + FOOTER_BYTES + PAGE_BYTES > self.max_bytes:
                raise RuntimeError('native byte reservation capacity exhausted')
            self.budget.reserve(size)
            self._reserved_events += events; self._reserved_bytes += size
            return _Reservation(self, events, size)

    def release(self, reservation):
        with self._lock:
            if type(reservation) is not _Reservation or reservation.owner is not self:
                raise ValueError('owned writer reservation required')
            if not reservation.released:
                self._reserved_events -= reservation.events; self._reserved_bytes -= reservation.bytes
                self.budget.release_unused(reservation.bytes)
                reservation.events = reservation.bytes = 0; reservation.released = True

    def emit(self, event, fields, *, reservation=None):
        if type(event) is not str or not re.fullmatch('[a-z][a-z0-9_]{0,63}', event) or type(fields) is not dict:
            raise ValueError('bounded native event fields required')
        with self._lock:
            if not self._accepting or self._failure:
                raise RuntimeError('native writer admission closed')
            if reservation is not None and (type(reservation) is not _Reservation or reservation.owner is not self or reservation.released):
                raise ValueError('live owned reservation required')
            now = max(time.monotonic_ns(), self._last_time)
            body = dict(sequence=self._events, monotonic_ns=now, event=event,
                        fields=fields, previous_sha256=self._last)
            digest = _hash(_encode(body, EVENT_BYTES))
            data = _line(dict(body, sha256=digest), EVENT_BYTES)
            if reservation is not None:
                if reservation.events < 1 or reservation.bytes < len(data):
                    raise RuntimeError('reserved native event capacity exhausted')
            elif self._events + self._reserved_events >= self.max_events:
                raise RuntimeError('unreserved native event capacity exhausted')
            if self._events >= self.max_events or self._pending_frames >= QUEUE_FRAMES or self._pending_bytes + len(data) > QUEUE_BYTES:
                self._fail('capacity_exhausted')
                raise RuntimeError('native writer live/event capacity exhausted')
            if event == 'settlement_ref':
                if set(fields) != {'entry_ordinal','receipt_sha256','claims_page_sha256'}:
                    raise ValueError('exact declared settlement reference required')
                entry_ordinal = _int(fields['entry_ordinal'], 16383)
                claim = self._claims.get(fields['claims_page_sha256'])
                instance = _hash(_encode(['gamst.recording.instance.v1', self.context['nonce'], entry_ordinal], PAGE_BYTES))
                if (claim is None or claim['receipt'] != fields['receipt_sha256'] or claim['owner'] != instance
                        or fields['claims_page_sha256'] in self._settlement_refs):
                    raise ValueError('unique sealed owned declared claim reference required')
            entry = event == 'entered'
            ordinal = self._planned_chunk
            count, entries, size = self._planned_events, self._planned_entries, self._planned_bytes
            if count and (count >= 256 or entries + entry > 128 or size + len(data) + HEADER_BYTES + FOOTER_BYTES > CHUNK_BYTES):
                ordinal += 1; count = entries = size = 0
            if ordinal >= self.max_chunks:
                self._fail('capacity_exhausted')
                raise RuntimeError('native chunk capacity exhausted')
            # Admission includes queued bytes, not just the off-loop writer's
            # latest durable counters. Reserve worst-case framing per new chunk.
            charge = len(data) + (HEADER_BYTES + FOOTER_BYTES if not count else 0)
            if self._run_bytes + self._pending_bytes + charge > self.max_bytes:
                self._fail('capacity_exhausted')
                raise RuntimeError('native run artifact bytes exhausted')
            framing = HEADER_BYTES + FOOTER_BYTES if not count else 0
            self.budget.reserve(framing + (0 if reservation is not None else len(data)))
            self._framing_credit += framing
            if reservation is not None:
                reservation.events -= 1; reservation.bytes -= len(data)
                self._reserved_events -= 1; self._reserved_bytes -= len(data)
            if event == 'settlement_ref': self._settlement_refs.add(fields['claims_page_sha256'])
            self._queue.append((ordinal, self._events, data, fields.get('entry_ordinal') if entry else None))
            self._pending_frames += 1; self._pending_bytes += len(data)
            self._events += 1; self._entries += int(entry); self._last = digest; self._last_time = now
            self._planned_chunk = ordinal; self._planned_events = count + 1
            self._planned_entries = entries + int(entry); self._planned_bytes = size + len(data)
            self._loop.call_soon_threadsafe(self._wake.set)
            return dict(sequence=self._events - 1, sha256=digest, chunk_ordinal=ordinal)

    def _charge(self, size, *, prepaid=False):
        with self._lock:
            if self._run_bytes + size > self.max_bytes:
                raise RuntimeError('native run artifact bytes exhausted')
            if not prepaid: self.budget.reserve(size)
            self._run_bytes += size

    def _open_chunk(self, ordinal, sequence):
        self._check_directory()
        header = dict(context=self.context, chunk_ordinal=ordinal,
            previous_chunk_sha256=self._chunks[-1]['sha256'] if self._chunks else None,
            first_sequence=sequence)
        data = _line(header, HEADER_BYTES)
        self._charge(len(data), prepaid=True)
        self._framing_credit -= len(data)
        self._fd = os.open(f'native-{ordinal:06d}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                           0o600, dir_fd=self._directory_fd)
        os.fchmod(self._fd, 0o600)
        self._chunk = dict(ordinal=ordinal, first_sequence=sequence, events=0, entries=0,
            first_entry=None, last_entry=None, bytes=0, hash=hashlib.sha256())
        _write_all(self._fd, data); os.fsync(self._fd)
        with self._lock:
            self._chunk['bytes'] = len(data); self._chunk['hash'].update(data)

    def _append(self, rows):
        if self._chunk is None:
            self._open_chunk(rows[0][0], rows[0][1])
        self._check_directory()
        old = self._chunk['bytes']
        try:
            for _, _, data, _ in rows:
                self._charge(len(data), prepaid=True); _write_all(self._fd, data)
            os.fsync(self._fd)
        except BaseException:
            try:
                os.ftruncate(self._fd, old); os.lseek(self._fd, old, os.SEEK_SET); os.fsync(self._fd)
            except OSError:
                pass
            raise
        with self._lock:
            for _, _, data, entry in rows:
                self._chunk['hash'].update(data); self._chunk['bytes'] += len(data); self._chunk['events'] += 1
                if entry is not None:
                    if self._chunk['first_entry'] is None: self._chunk['first_entry'] = entry
                    self._chunk['last_entry'] = entry; self._chunk['entries'] += 1

    def _save_bytes(self, name, data):
        self._check_directory()
        metadata = name == 'terminal.json' or name.startswith('index-')
        if metadata:
            if len(data) > self._metadata_credit:
                raise RuntimeError('reserved native metadata capacity exhausted')
            self._charge(len(data), prepaid=True)
            self._metadata_credit -= len(data)
        else:
            self._charge(len(data))
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self._directory_fd)
        try:
            os.fchmod(fd, 0o600); _write_all(fd, data); os.fsync(fd)
        finally:
            os.close(fd)
        return dict(path=name, sha256=_hash(data), bytes=len(data))

    def _save_json(self, name, value, limit):
        return self._save_bytes(name, _line(value, limit))

    def _seal_chunk(self):
        if self._chunk is None:
            return
        c = self._chunk
        footer = dict(kind='chunk_footer', last_sequence=c['first_sequence'] + c['events'] - 1,
            event_count=c['events'], entry_count=c['entries'], first_entry_ordinal=c['first_entry'],
            last_entry_ordinal=c['last_entry'], prefix_bytes=c['bytes'], prefix_sha256=c['hash'].hexdigest())
        data = _line(footer, FOOTER_BYTES)
        if c['bytes'] + len(data) > CHUNK_BYTES or not c['events']:
            raise ValueError('invalid native chunk envelope')
        try:
            self._charge(len(data), prepaid=True); _write_all(self._fd, data); os.fsync(self._fd)
        except BaseException:
            try:
                os.ftruncate(self._fd, c['bytes']); os.lseek(self._fd, c['bytes'], os.SEEK_SET)
            except OSError:
                pass
            raise
        self._framing_credit -= len(data)
        digest = c['hash'].copy(); digest.update(data)
        ref = dict(ordinal=c['ordinal'], path=f"native-{c['ordinal']:06d}.jsonl",
            sha256=digest.hexdigest(), bytes=c['bytes'] + len(data), first_sequence=c['first_sequence'],
            event_count=c['events'], entry_count=c['entries'])
        os.close(self._fd); self._fd = None
        with self._lock:
            self._chunks.append(ref); self._page_pending.append(ref); self._chunk = None
        self._loop.call_soon_threadsafe(self._sealed_wake.set)
        if len(self._page_pending) == 128:
            self._seal_page()

    def _seal_page(self):
        if not self._page_pending:
            return
        if len(self._pages) >= 8:
            raise RuntimeError('native index page bound exceeded')
        ordinal = len(self._pages)
        page = dict(schema_version=1, kind='recording_native_index', nonce=self.context['nonce'],
            page_ordinal=ordinal, previous_page_sha256=self._pages[-1]['sha256'] if self._pages else None,
            chunks=self._page_pending)
        ref = self._save_bytes(f'index-{ordinal:06d}.json', _line(page, PAGE_BYTES))
        with self._lock:
            self._pages.append(ref); self._page_pending = []

    async def _write(self):
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        if self._stopping: break
                        self._wake.clear(); rows = None
                    else:
                        ordinal = self._queue[0][0]; rows = []
                        while self._queue and self._queue[0][0] == ordinal:
                            rows.append(self._queue.popleft())
                if rows is None:
                    await self._wake.wait(); continue
                if self._chunk is not None and self._chunk['ordinal'] != rows[0][0]:
                    await self._thread(self._seal_chunk)
                await self._thread(self._append, rows)
                with self._lock:
                    self._written_events += len(rows)
                    self._pending_frames -= len(rows); self._pending_bytes -= sum(len(r[2]) for r in rows)
                self._drained_wake.set()
            if self._failure is None and not self._aborted:
                await self._thread(self._seal_chunk)
                await self._thread(self._seal_page)
        except BaseException:
            self._fail('journal_failed')
        finally:
            with self._lock:
                self._queue.clear()
            self._sealed_wake.set()
            self._drained_wake.set()

    async def wait_sealed_chunks(self, after):
        """Owned channel follower: no polling or event-history snapshots."""
        _int(after, self.max_chunks)
        while True:
            with self._lock:
                refs = [dict(ref) for ref in self._chunks[after:]]
                done = self._writer.done()
                if refs or done: return refs, done
                self._sealed_wake.clear()
            await self._sealed_wake.wait()

    async def drain(self):
        while True:
            with self._lock:
                pending = self._pending_frames; failed = self._failure
                if failed: raise RuntimeError(failed)
                if not pending: return
                if self._writer.done(): raise RuntimeError("writer stopped before drain")
                self._drained_wake.clear()
            await self._drained_wake.wait()

    def status(self):
        with self._lock:
            prefix = None
            if self._chunk is not None:
                c = self._chunk
                prefix = dict(chunk_ordinal=c['ordinal'], valid_bytes=c['bytes'],
                    valid_events=c['events'], sha256=c['hash'].hexdigest())
            return dict(admitted_events=self._events, written_events=self._written_events,
                entry_count=self._entries, pending_frames=self._pending_frames,
                pending_bytes=self._pending_bytes, reserved_events=self._reserved_events,
                failure=self._failure, chunks=[dict(r) for r in self._chunks],
                pages=[dict(r) for r in self._pages], prefix=prefix, bytes=self._run_bytes)

    async def _file(self, function, *args):
        with self._lock:
            if not self._accepting or self._stopping:
                raise RuntimeError('native metadata admission closed')
            if len(self._file_jobs) >= 4:
                raise RuntimeError('native metadata inflight bound exceeded')
        task = asyncio.create_task(self._file_owned(function, *args))
        self._file_jobs.add(task)
        try:
            try: return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _finish(task); raise
        finally:
            self._file_jobs.discard(task)

    async def _file_owned(self, function, *args):
        async with self._file_lock:
            try: return await self._thread(function, *args)
            except BaseException:
                self._fail('journal_failed'); raise

    async def write_counter_page(self, page):
        def save():
            if type(page) is not dict or set(page) != {'schema_version','kind','nonce','ordinal','counters'}:
                raise ValueError('exact counter page required')
            if type(page['schema_version']) is not int or page['schema_version'] != 1 or page['kind'] != 'recording_counters' or page['nonce'] != self.context['nonce']:
                raise ValueError('bound counter page required')
            ordinal = _int(page['ordinal'], 15)
            if ordinal != len(self._counters): raise ValueError('contiguous counter pages required')
            counters = page['counters']
            if type(counters) is not list or not 1 <= len(counters) <= 4160:
                raise ValueError('bounded nonempty counter inventory required')
            pairs = set()
            for row in counters:
                if type(row) is not dict or set(row) != {'scope','binding','entered','finished','published','unsettled'}:
                    raise ValueError('exact provider counter required')
                pair = (_int(row['scope'],64), _int(row['binding'],63))
                if pair in pairs or pair in self._counter_pairs: raise ValueError('duplicate scope/binding counter')
                pairs.add(pair)
                for key in ('entered','finished','published','unsettled'): _int(row[key],16384)
                if not row['published'] <= row['finished'] <= row['entered'] or row['unsettled'] != row['entered']:
                    raise ValueError('counter conservation or unknown liability changed')
            ref = self._save_bytes(f'counters-{ordinal:06d}.json', _line(page, PAGE_BYTES))
            self._counter_pairs.update(pairs)
            self._counters.append(dict(ordinal=ordinal, sha256=ref['sha256'], bytes=ref['bytes']))
            return dict(self._counters[-1])
        return await self._file(save)

    async def write_receipt_source(self, raw):
        if not self.retain_financial_documents: raise PermissionError('financial document retention not consented')
        if type(raw) is not bytes or not 1 <= len(raw) <= CHUNK_BYTES:
            raise ValueError('bounded original receipt bytes required')
        def save():
            digest = _hash(raw)
            if digest in self._sources: return dict(self._sources[digest])
            if self._source_bytes + len(raw) > 64*1024**2: raise ValueError('receipt source byte bound exceeded')
            self.budget.admit_source(len(raw))
            ref = self._save_bytes(f'receipt-source-{digest}.bin', raw)
            self._source_bytes += len(raw); self._sources[digest] = dict(ref)
            return ref
        return await self._file(save)

    async def write_receipt_attachment(self, receipt):
        if not self.retain_financial_documents: raise PermissionError('financial document retention not consented')
        def save():
            keys = {'schema_version','kind','origin','money_basis','provider','revision','account_scope',
                'invoice_id','source_document_sha256','source_document_bytes','lines'}
            if type(receipt) is not dict or set(receipt) != keys:
                raise ValueError('exact declared receipt attachment required')
            if (type(receipt['schema_version']) is not int or receipt['schema_version'] != 1 or
                    receipt['kind'] != 'recording_declared_receipt' or
                    receipt['origin'] not in ('application_declared','synthetic_fixture') or receipt['money_basis'] != 'unknown'):
                raise ValueError('declared unknown-money receipt required')
            for key in ('provider','revision','account_scope','invoice_id'): _id(receipt[key])
            lines = receipt['lines']
            if type(lines) is not list or not 1 <= len(lines) <= 64:
                raise ValueError('bounded declared receipt lines required')
            hashes = []; line_ids = set()
            for line in lines:
                if type(line) is not dict or set(line) != {'line_id','quantities','claim_sha256'}:
                    raise ValueError('exact declared line required')
                _id(line['line_id'])
                if line['line_id'] in line_ids: raise ValueError('duplicate original invoice line')
                line_ids.add(line['line_id'])
                quantities = line['quantities']
                if type(quantities) is not dict or not quantities:
                    raise ValueError('complete declared quantities required')
                for key, value in quantities.items(): _id(key); _int(value, 10**12)
                digest = _hash(_encode(['gamst.recording.claim.v1', receipt['provider'], receipt['account_scope'],
                    receipt['invoice_id'], line['line_id']], PAGE_BYTES))
                if line['claim_sha256'] != digest: raise ValueError('original claim identity mismatch')
                hashes.append(digest)
            _sha(receipt['source_document_sha256']); _int(receipt['source_document_bytes'], CHUNK_BYTES, 1)
            data = _line(receipt, PAGE_BYTES); digest = _hash(data)
            source = receipt.get('source_document_sha256') if type(receipt) is dict else None
            if source not in self._sources or receipt.get('source_document_bytes') != self._sources[source]['bytes']:
                raise ValueError('sealed original source required before attachment')
            if len(self._receipts) >= 16384 or self._receipt_bytes + len(data) > 64*1024**2:
                raise ValueError('declared receipt inventory exhausted')
            self.budget.admit_attachment(len(data), (receipt['provider'],receipt['account_scope'],receipt['invoice_id']),
                (receipt['revision'], source), len(hashes))
            ref = self._save_bytes(f'receipt-{digest}.json', data)
            self._receipts[digest] = dict(source=source, claimed=False, claims=tuple(bytes.fromhex(h) for h in hashes))
            self._receipt_bytes += len(data)
            return ref
        return await self._file(save)

    async def write_claims_page(self, page):
        if not self.retain_financial_documents: raise PermissionError('financial document retention not consented')
        def save():
            if (type(page) is not dict or set(page) != {'schema_version','kind','receipt_sha256','owner_instance','claims'}
                    or type(page['schema_version']) is not int or page['schema_version'] != 1 or page['kind'] != 'recording_declared_claims'):
                raise ValueError('exact declared claim page required')
            data = _line(page, PAGE_BYTES); digest = _hash(data)
            receipt = page.get('receipt_sha256') if type(page) is dict else None
            if receipt not in self._receipts or self._receipts[receipt]['claimed']:
                raise ValueError('unique sealed receipt required before claim page')
            _sha(page.get('owner_instance'))
            rows = page['claims']; expected = self._receipts[receipt]['claims']
            if type(rows) is not list or len(rows) != len(expected):
                raise ValueError('every receipt line must be claimed once')
            seen = set()
            for row in rows:
                if type(row) is not dict or set(row) != {'claim_sha256','line_index'}:
                    raise ValueError('exact declared line claim required')
                index = _int(row['line_index'], len(expected)-1)
                if index in seen or bytes.fromhex(_sha(row['claim_sha256'])) != expected[index]:
                    raise ValueError('duplicate or changed declared line claim')
                seen.add(index)
            self.budget.admit_claims(value.hex() for value in expected)
            ref = self._save_bytes(f'claims-{digest}.json', data)
            self._claims[digest] = dict(receipt=receipt, owner=page['owner_instance'])
            self._receipts[receipt]['claimed'] = True
            return ref
        return await self._file(save)

    async def close(self, summary, *, cleanup_finished=True):
        if self._close_task is None:
            # Snapshot and encode before an await, but cleanup still happens if
            # malformed terminal input makes serialization fail.
            import json
            summary_error = None
            try:
                detached = json.loads(_line(summary, PAGE_BYTES))
            except BaseException as exc:
                summary_error = exc
                detached = None  # _close still owns writer/FD cleanup before error.
            self._close_task = asyncio.create_task(self._close(detached, cleanup_finished, summary_error))
        try: return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task); raise

    async def _close(self, summary, cleanup_finished, summary_error=None):
        with self._lock:
            self._accepting = False; self._stopping = True
            self._wake.set()
        try:
            if self._file_jobs:
                await asyncio.gather(*tuple(self._file_jobs))
            await self._writer
            if summary_error is not None: raise summary_error
            if type(summary) is not dict or set(summary) != SUMMARY_KEYS:
                raise ValueError('exact disjoint semantic terminal summary required')
            if type(cleanup_finished) is not bool:
                raise ValueError('explicit cleanup completion boolean required')
            for key in SUMMARY_KEYS - {'mapping_sha256','counter_pages','incomplete_counts'}:
                _int(summary[key], 262144)
            if summary['mapping_sha256'] is not None: _sha(summary['mapping_sha256'])
            if summary['counter_pages'] != self._counters:
                raise ValueError('terminal counter inventory differs from sealed pages')
            reasons = {'journal_failed','capacity_exhausted','unobserved_boundary','cleanup_incomplete',
                'source_incomplete','financial_admission_refused','settlement_unknown'}
            if type(summary['incomplete_counts']) is not dict or set(summary['incomplete_counts']) - reasons:
                raise ValueError('bounded known incomplete categories required')
            for count in summary['incomplete_counts'].values(): _int(count,262144)
            snapshot = self.status()
            if not self._failure and (summary['reservation_count'] != self._entries + summary['released_count'] + summary['outstanding_tickets']
                    or summary['unsettled_entries'] != self._entries):
                raise ValueError('terminal reservation/unknown liability conservation mismatch')
            incomplete = dict(summary['incomplete_counts'])
            if self._failure: incomplete[self._failure] = incomplete.get(self._failure, 0) + 1
            used_sources = {v['source'] for v in self._receipts.values()}
            if (used_sources != set(self._sources) or any(not r['claimed'] for r in self._receipts.values())
                    or self._settlement_refs != set(self._claims)):
                incomplete['journal_failed'] = incomplete.get('journal_failed', 0) + 1
                self._fail('journal_failed')
            semantic = _line(summary, PAGE_BYTES)
            import json
            terminal = json.loads(semantic)
            closed = cleanup_finished and not self._failure and not self._aborted and not any(
                summary[k] for k in ('outstanding_tickets','outstanding_entries','outstanding_work','unclosed_workers'))
            if 'journal_failed' in incomplete or 'cleanup_incomplete' in incomplete: closed = False
            terminal.update(schema_version=4, kind='recording_native_terminal', context=self.context,
                state='closed' if closed else 'incomplete', index_tail_sha256=self._pages[-1]['sha256'] if self._pages else None,
                page_count=len(self._pages), chunk_count=len(self._chunks), event_count=self._written_events,
                entry_count=self._entries, prefix=snapshot['prefix'], incomplete_counts=incomplete,
                child_cleanup_finished=cleanup_finished, actual_acceptance='incomplete',
                receipt_authenticity_verified=False, worker_backend_coverage_verified=False)
            self._terminal = await self._thread(self._save_json, 'terminal.json', terminal, PAGE_BYTES)
            if self._failure: raise RuntimeError(self._failure)
            return dict(self._terminal)
        finally:
            await _finish(asyncio.create_task(self._cleanup()))

    async def abort(self):
        if self._abort_task is None:
            self._abort_task = asyncio.create_task(self._abort())
        try: return await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            await _finish(self._abort_task); raise

    async def _abort(self):
        with self._lock:
            self._accepting = False; self._stopping = True; self._aborted = True
            self._wake.set()
        if self._close_task is not None:
            await _finish(asyncio.gather(self._close_task, return_exceptions=True))
        else:
            try:
                if self._file_jobs: await asyncio.gather(*tuple(self._file_jobs), return_exceptions=True)
                if self._writer: await self._writer
            finally:
                await self._cleanup()

    async def _cleanup(self):
        def close_fds():
            # Release only conservatively reserved framing that was never used.
            # Failed actual writes remain charged, even if only partially retained.
            with self._lock:
                if self._metadata_credit:
                    self.budget.release_unused(self._metadata_credit)
                    self._metadata_credit = 0
                if self._framing_credit:
                    self.budget.release_unused(self._framing_credit)
                    self._framing_credit = 0
            if self._fd is not None:
                os.close(self._fd); self._fd = None
            if self._directory_fd is not None:
                os.close(self._directory_fd); self._directory_fd = None
        await self._thread(close_fds)
