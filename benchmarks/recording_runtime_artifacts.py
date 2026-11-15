"""Bounded immutable runtime artifacts, distinct from native calls and scoring.

Synchronous admission copies exact bounded bytes before handing ownership back
to an SDK publisher. All filesystem writes belong to one joined worker. The
runtime join, not this storage layer, establishes original-source provenance.
"""
import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import threading
import time

from .recording_artifacts import _encode, _finish, _write_all
from .recording_native_schema import decode, encoded, digest, keys, integer, sha, identifier
from .recording_native_writer import ArtifactBudget

PAGE_BYTES = 65536
ROW_BYTES = 2048
MAX_PAGES = 2048
MAX_METADATA_BYTES = 16 * 1024**2
MAX_QUEUE_FRAMES = 64
MAX_QUEUE_BYTES = 1024**2
KINDS = {
    'ticks': 'recording_tick_rows', 'adapters': 'recording_adapter_rows',
    'deliveries': 'recording_delivery_rows', 'delivery-summaries': 'recording_delivery_summaries',
    'channels': 'recording_channel_rows', 'mapping-updates': 'recording_mapping_update_rows',
    'transport': 'recording_transport_rows', 'publications': 'recording_publication_rows',
}
LIMITS = dict(ticks=16384, adapters=8, deliveries=262144,
    **{'delivery-summaries':1024, 'channels':4, 'mapping-updates':1024,
       'transport':16384, 'publications':262144})
REASONS = ('completed','not_executed','snapshot_failed','oversize','busy','stale',
    'admission_refused','timeout','cancelled','backend_failed','source_failed',
    'clock_discontinuity','transport_failed','writer_failed')
_REF_PATH = re.compile(r'(?:handshake|mapping|source-mappings|capture-inventory|child-summary|run-inventory|native-inventory|execution|(?:ticks|adapters|deliveries|delivery-summaries|channels|mapping-updates|publications|transport)-index)\.json|(?:ticks|adapters|deliveries|delivery-summaries|channels|mapping-updates|publications|transport)/(?:page|index|row)-[0-9]{6}\.json|tasks/task-[0-9]{4}-(?:context|envelope|record|evidence|result|receipt)\.json|fill/case-[0-9]{6}\.json|records/record-[0-9]{6}\.json|requests/request-[0-9]{6}\.json')
_ORIGINAL_REF_PATH = re.compile(r'source/(?:journal\.jsonl|(?:terminal|mapping|mapping-records|controller-settlements|source-mappings|source-manifest|tasks|profiles|execution)\.json)|native/(?:terminal\.json|native-[0-9]{6}\.jsonl|(?:index|counters)-[0-9]{6}\.json|(?:receipt|claims)-[0-9a-f]{64}\.json)')


def _line(value, limit):
    return _encode(value, limit-1)+b'\n'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def preflight_runtime_artifacts(outcome_rows, *, retained_record_bytes=0, task_artifact_bytes=0):
    """Conservative joint storage admission, before any provider activation."""
    keys(outcome_rows,KINDS)
    total_pages=total_bytes=0
    for kind,count in outcome_rows.items():
        integer(count,0,LIMITS[kind])
        # 31 maximum-size rows plus framing fit a64KiB data page.
        pages=(count+30)//31; indexes=(pages+127)//128
        total_pages+=pages+indexes
        total_bytes+=count*(ROW_BYTES+1024)+PAGE_BYTES
    integer(retained_record_bytes,0,MAX_METADATA_BYTES);integer(task_artifact_bytes,0,4*1024**2)
    total_bytes+=retained_record_bytes+task_artifact_bytes
    if total_pages>MAX_PAGES or total_bytes>MAX_METADATA_BYTES:
        raise ValueError('joint runtime artifact declarations exceed pages/metadata bounds')
    return dict(maximum_bytes=total_bytes,maximum_pages=total_pages)


def validate_ref(value):
    keys(value, ('path','file_sha256','bytes'))
    if type(value['path']) is not str or not (_REF_PATH.fullmatch(value['path']) or _ORIGINAL_REF_PATH.fullmatch(value['path'])):
        raise ValueError('fixed runtime artifact locator required')
    sha(value['file_sha256']); integer(value['bytes'],1,4*1024**3)
    return value


def _nullable(fn, value):
    if value is not None: fn(value)


def _enum(value, choices):
    if type(value) is not str or value not in choices: raise ValueError('known runtime enum required')


def _boolean(value):
    if type(value) is not bool: raise ValueError('exact boolean required')


def _text(value, limit):
    if type(value) is not str or not value or len(value.encode('utf-8'))>limit:
        raise ValueError('bounded nonempty UTF-8 text required')


def validate_execution(value):
    keys(value, ('schema_version','kind','runtime_policy_sha256','collection_sha256',
        'scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal',
        'recording_index','arm','stream_id','execution_namespace','cache_namespace','asset_inventory_sha256'))
    if type(value['schema_version']) is not int or value['schema_version']!=1 or value['kind']!='recording_execution':
        raise ValueError('recording execution1 required')
    for name in ('runtime_policy_sha256','collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','asset_inventory_sha256'): sha(value[name])
    integer(value['invocation_ordinal'],0,95); integer(value['recording_index'],0,31)
    _enum(value['arm'],('A','B','C'))
    for name in ('stream_id','execution_namespace','cache_namespace'): identifier(value[name])
    return value


def _capture(value):
    keys(value,('kind','source_index','capture_ordinal','record_sha256','mapping_sha256'))
    if value['kind']!='capture': raise ValueError('original native capture locator required')
    sha(value['mapping_sha256']); sha(value['record_sha256'])
    integer(value['source_index'],0,15);integer(value['capture_ordinal'])


def validate_outcome(kind, row):
    """Exact structural row grammar; original inventory joins remain separate."""
    if kind=='ticks':
        keys(row,('tick_index','count','first_due_offset_ns','period_ns','policy','source_index',
            'observed_begin_ns','observed_end_ns','outcome','capture_ref','request_id','operation_instance','case'))
        for n in ('tick_index','first_due_offset_ns','source_index','observed_begin_ns','observed_end_ns'): integer(row[n])
        integer(row['count'],1,16384); integer(row['period_ns'],1)
        _enum(row['policy'],('periodic','fill_chain'))
        _enum(row['outcome'],('selected','no_frame','busy','late','stale','not_retained','financial_refused','observer_refused','closed'))
        _nullable(_capture,row['capture_ref']); _nullable(identifier,row['request_id'])
        _nullable(sha,row['operation_instance']); _nullable(validate_ref,row['case'])
        if row['count']>1 and (row['outcome']=='selected' or any(row[n] is not None for n in ('capture_ref','request_id','operation_instance','case'))):
            raise ValueError('only empty skipped consecutive ticks may batch')
        if row['outcome']=='selected' and row['capture_ref'] is None: raise ValueError('selected tick needs original capture')
    elif kind=='adapters':
        keys(row,('lane_id','adapter_id','binding_index','controller_id','state','opened_ns','closed_ns','entry_count','finished_count','published_count','unknown_liability_count','worker_ordinal','attachment','external_backend_closed'))
        for n in ('lane_id','adapter_id','controller_id'): identifier(row[n])
        for n in ('binding_index','entry_count','finished_count','published_count','unknown_liability_count'): integer(row[n])
        for n in ('opened_ns','closed_ns','worker_ordinal'): _nullable(integer,row[n])
        _enum(row['state'],('not_started','closed','incomplete'))
        _enum(row['attachment'],('owned_process','owned_in_process','external_async_callback'))
        _enum(row['external_backend_closed'],('verified','unknown','not_applicable'))
        if row['finished_count']>row['entry_count'] or row['published_count']>row['finished_count']: raise ValueError('invalid adapter count ordering')
        if row['attachment']=='external_async_callback' and row['external_backend_closed']=='verified':
            raise ValueError('callback detach cannot verify external backend closure')
    elif kind=='deliveries':
        keys(row,('delivery_ordinal','event','observed_ns','source_index','capture_ordinal','binding_index','reason','payload_bytes','operation_instance','publication_ordinal'))
        for n in ('delivery_ordinal','observed_ns','source_index','capture_ordinal','payload_bytes'): integer(row[n])
        for n in ('binding_index','publication_ordinal'): _nullable(integer,row[n])
        _nullable(sha,row['operation_instance'])
        _enum(row['event'],('mailbox_accepted','mailbox_replaced','mailbox_refused','provider_submitted','provider_refused','queue_evicted','provider_processed','provider_error','publication_accepted','publication_discarded','scrubbed'))
        _enum(row['reason'],('completed','capacity','closed','generation_changed','backend_error','unknown'))
    elif kind=='delivery-summaries':
        names=('source_index','binding_index','accepted','preclose_queued','preclose_unfinished','preclose_active','postclose_processed','postclose_errors','postclose_evicted','postclose_queued','postclose_unfinished','worker_task_cancelled','residual_unresolved','disposition_truth_verified')
        keys(row,names)
        for n in names:
            (_boolean if n in ('worker_task_cancelled','disposition_truth_verified') else integer)(row[n])
        if row['preclose_active']!=row['preclose_unfinished']-row['preclose_queued']: raise ValueError('exact preclose active count required')
        if row['disposition_truth_verified'] and row['residual_unresolved']!=row['accepted']-row['postclose_processed']-row['postclose_errors']-row['postclose_evicted']:
            raise ValueError('exact residual delivery partition required')
    elif kind=='channels':
        keys(row,('channel','direction','frames','bytes','last_sequence','terminal_seen','eof_seen','state','reason','observed_begin_ns','observed_end_ns'))
        _enum(row['channel'],('native2','coordinator1')); _enum(row['direction'],('parent_to_child','child_to_parent'))
        for n in ('frames','bytes','observed_begin_ns','observed_end_ns'): integer(row[n])
        _nullable(integer,row['last_sequence']); _boolean(row['terminal_seen']); _boolean(row['eof_seen'])
        _enum(row['state'],('closed','failed'))
        _enum(row['reason'],('completed','timeout','eof_before_terminal','malformed','overflow','write_failed','read_failed','trailing_data','cancelled'))
        if row['state']=='closed' and row['direction']=='child_to_parent' and not (row['terminal_seen'] and row['eof_seen']): raise ValueError('closed child direction requires terminal and EOF')
    elif kind=='mapping-updates':
        keys(row,('update_ordinal','source_index','binding_index','previous_delivery_epoch','new_delivery_epoch','observed_ns','capture_ordinal','reason','mapping_sha256'))
        for n in ('update_ordinal','source_index','binding_index','observed_ns'): integer(row[n])
        for n in ('previous_delivery_epoch','new_delivery_epoch'): identifier(row[n])
        _nullable(integer,row['capture_ordinal']); sha(row['mapping_sha256'])
        if row['reason']!='provider_reset' or row['previous_delivery_epoch']==row['new_delivery_epoch']: raise ValueError('actual changed provider reset epoch required')
    elif kind=='transport':
        keys(row,('binding_index','prompt_sha256','wire_request_sha256','response_body_sha256','provider_request_id','http_status','started_ns','finished_ns','outcome','request_bytes','response_bytes','attempts'))
        for n in ('binding_index','started_ns','finished_ns','request_bytes','response_bytes'): integer(row[n])
        for n in ('prompt_sha256','wire_request_sha256'): sha(row[n])
        _nullable(sha,row['response_body_sha256']); _nullable(lambda v:integer(v,100,599),row['http_status'])
        if row['provider_request_id'] is not None:
            _text(row['provider_request_id'],128)
            if any(not 32<=ord(c)<=126 for c in row['provider_request_id']): raise ValueError('printable provider request ID required')
        _enum(row['outcome'],('ok','timeout','cancelled','http_error','invalid_response','transport_error'))
        if type(row['attempts']) is not int or row['attempts']!=1: raise ValueError('one actual transport attempt required')
        if row['finished_ns']<row['started_ns']: raise ValueError('ordered transport interval required')
    elif kind=='publications': validate_ref(row)
    else: raise ValueError('known outcome kind required')
    if 'observed_begin_ns' in row and row['observed_end_ns']<row['observed_begin_ns']: raise ValueError('ordered observed interval required')
    _line(row,ROW_BYTES)
    return row


@dataclass(frozen=True)
class PendingPublication:
    _owner: object
    _future: object
    publication_ordinal: int
    record_id: str
    record_sha256: str


class RecordingRuntimeArtifacts:
    @classmethod
    async def open(cls, output, execution, *, budget, retain_evaluation_artifacts=False):
        """Create only this output directory; explicit external retention required."""
        execution=decode(_line(execution,PAGE_BYTES)); validate_execution(execution)
        if type(budget) is not ArtifactBudget: raise TypeError('shared ArtifactBudget required')
        if retain_evaluation_artifacts is not True: raise ValueError('explicit external runtime artifact retention consent required')
        self=cls(); self._execution_bytes=_line(execution,PAGE_BYTES); self._execution_sha256=digest(execution)
        self.output=Path(output).absolute(); self.budget=budget
        self._lock=threading.RLock(); self._loop=asyncio.get_running_loop()
        self._queue=deque(); self._wake=asyncio.Event(); self._pending_bytes=0; self._pending_frames=0
        self._failure=None; self._accepting=True; self._stopping=False; self._aborted=False
        self._directory=None; self._subdirs={}; self._writer=None; self._close_task=None; self._abort_task=None
        self._run_bytes=0; self._pages=0; self._paths=set(); self._files={}; self._threads=set()
        self._rows={k:[] for k in KINDS}; self._data_refs={k:[] for k in KINDS};self._durable_counts={k:0 for k in KINDS}
        self._counts={k:0 for k in KINDS}; self._logical_ticks=0; self._publications=0; self._record_ids=set()
        self._unique={k:set() for k in ('adapters','channels','transport','mapping-updates','delivery-summaries')}
        self._snapshot_bytes=0; self._sealed=None; self._seal_task=None;self._final_writes=set();self._final_closed=False
        self._imported_indices={};self._parent_imported=False
        try:
            await self._thread(self._create)
            self._writer=asyncio.create_task(self._write())
            return self
        except BaseException:
            await _finish(asyncio.create_task(self._thread(self._cleanup))); raise

    @property
    def execution(self):return decode(self._execution_bytes)

    @property
    def execution_sha256(self):return self._execution_sha256

    def _create(self):
        self.output.mkdir(mode=0o700,exist_ok=False); os.chmod(self.output,0o700)
        self._directory=os.open(self.output,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        for name in (*KINDS,'tasks','records','requests','fill'):
            os.mkdir(name,mode=0o700,dir_fd=self._directory)
            self._subdirs[name]=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self._directory)

    async def _thread(self, fn, *args):
        task=asyncio.create_task(asyncio.to_thread(fn,*args)); self._threads.add(task)
        try:
            try: return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _finish(task); raise
        finally: self._threads.discard(task)

    def _fail(self, reason):
        with self._lock:
            if self._failure is None:self._failure=reason
            self._accepting=False

    def _charge(self, size):
        if self._run_bytes+size>MAX_METADATA_BYTES: raise RuntimeError('runtime metadata exceeds16MiB')
        self.budget.reserve(size); self._run_bytes+=size

    def _global_refs(self, count, publications=0):
        # The same budget object is already shared across all run writers. Its
        # lock protects collection-wide counters without resetting per run.
        with self.budget._lock:
            refs,rows=getattr(self.budget,'_runtime_reference_counts',(0,0))
            if refs+count>1048576 or rows+publications>1048576:
                raise RuntimeError('global runtime publication/reference capacity exhausted')
            self.budget._runtime_reference_counts=(refs+count,rows+publications)

    def _store(self, path, data):
        current=os.stat(self.output,follow_symlinks=False); owned=os.fstat(self._directory)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=(owned.st_dev,owned.st_ino): raise ValueError('owned runtime directory replaced')
        parts=path.split('/'); directory=self._directory
        if len(parts)==2:
            directory=self._subdirs[parts[0]]
            actual=os.stat(parts[0],dir_fd=self._directory,follow_symlinks=False); expected=os.fstat(directory)
            if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino) or not stat.S_ISDIR(actual.st_mode): raise ValueError('owned runtime subdirectory replaced')
        fd=os.open(parts[-1],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        try: os.fchmod(fd,0o600); _write_all(fd,data); os.fsync(fd)
        finally: os.close(fd)
        ref=dict(path=path,file_sha256=_hash(data),bytes=len(data)); self._files[path]=ref
        return ref

    def _admit(self, job, size, *, paths=()):
        with self._lock:
            if not self._accepting or self._failure: raise RuntimeError('runtime artifact admission closed')
            if self._pending_frames>=MAX_QUEUE_FRAMES or self._pending_bytes+size>MAX_QUEUE_BYTES:
                self._fail('queue_capacity'); raise RuntimeError('runtime artifact queue exhausted before admission')
            if any(p in self._paths for p in paths): raise ValueError('immutable runtime artifact already admitted')
            # Include amortized bounded page/header/index capacity before handoff.
            self._charge(size+1024)
            self._global_refs(1)
            self._paths.update(paths); self._queue.append((job,size))
            self._pending_frames+=1; self._pending_bytes+=size
            self._loop.call_soon_threadsafe(self._wake.set)

    def _record(self, kind, row):
        if self._parent_imported and kind!='channels':raise RuntimeError('parent may only append its channel outcomes after child import')
        data=_line(row,ROW_BYTES); row=decode(data); validate_outcome(kind,row)
        with self._lock:
            if self._counts[kind]>=LIMITS[kind]: raise RuntimeError('runtime outcome count exhausted')
            key=None
            if kind=='adapters':key=row['lane_id']
            elif kind=='channels':key=(row['channel'],row['direction'])
            elif kind=='transport':key=(row['binding_index'],row['prompt_sha256'])
            elif kind=='mapping-updates':key=row['update_ordinal']
            elif kind=='delivery-summaries':key=(row['source_index'],row['binding_index'])
            if key is not None and key in self._unique[kind]:raise ValueError('duplicate runtime outcome identity')
            if kind=='ticks' and self._logical_ticks+row['count']>16384:raise RuntimeError('logical tick inventory exhausted')
            self._admit(('row',kind,row),len(data))
            self._counts[kind]+=1
            if key is not None:self._unique[kind].add(key)
            if kind=='ticks':self._logical_ticks+=row['count']

    def record_tick(self,row): self._record('ticks',row)
    def record_adapter(self,row): self._record('adapters',row)
    def record_delivery(self,row): self._record('deliveries',row)
    def record_delivery_summary(self,row): self._record('delivery-summaries',row)
    def record_channel(self,row): self._record('channels',row)
    def record_transport(self,row): self._record('transport',row)
    def record_mapping_update(self,row): self._record('mapping-updates',row)

    def record_publication(self,record,*,binding_index,builder_sha256,operation_reference,
                           capture_refs,role,origin_result_sha256,mapping_sha256):
        from gamst.native_operation import NativeOperationRef
        from gamst.call_controller import _digest
        if asyncio.get_running_loop() is not self._loop:raise RuntimeError('accepted publication handoff belongs to writer event loop')
        data=_line(record,128*1024); original=decode(data)
        if type(original) is not dict or original.get('type')!='observation.signal': raise ValueError('actual accepted observation.signal record required')
        record_id=identifier(original.get('id')); observation=original.get('observation')
        if type(observation) is not dict:raise ValueError('actual observation body required')
        epoch=identifier(observation.get('processing_epoch')); sha(mapping_sha256)
        _enum(role,('specialist_result','raw_capture_descriptor','semantic_description','fill_measurement','fill_descriptor','fill_comparison','task_snapshot'))
        if type(capture_refs) is not list or len(capture_refs)>16: raise ValueError('bounded direct capture references required')
        captures=decode(_line(capture_refs,ROW_BYTES))
        for ref in captures:_capture(ref)
        if len({digest(v) for v in captures})!=len(captures):raise ValueError('duplicate capture reference')
        deterministic=role in ('raw_capture_descriptor','task_snapshot')
        instance=None
        if deterministic:
            if operation_reference is not None or binding_index is not None:raise ValueError('deterministic publication is not backend entry')
            sha(builder_sha256)
        else:
            if type(operation_reference) is not NativeOperationRef: raise TypeError('actual observer-issued operation reference required')
            value=operation_reference.snapshot(); integer(binding_index)
            ctx=value['context']
            if value['binding_index']!=binding_index or any(ctx[k]!=self.execution[k] for k in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal')):
                raise ValueError('publication actual operation belongs to another execution/binding')
            if builder_sha256 is not None:raise ValueError('backend publication cannot claim deterministic builder')
            instance=value['instance_sha256']
        _nullable(sha,origin_result_sha256)
        with self._lock:
            if record_id in self._record_ids:raise ValueError('duplicate accepted record alias')
            ordinal=self._publications
            if ordinal>=262144:raise RuntimeError('publication inventory exhausted')
            record_path='records/record-%06d.json'%ordinal
            record_ref=dict(path=record_path,file_sha256=_hash(data),bytes=len(data))
            row=dict(schema_version=1,kind='recording_publication',execution_sha256=self.execution_sha256,
                publication_ordinal=ordinal,binding_index=binding_index,builder_sha256=builder_sha256,
                operation_instance=instance,role=role,capture_refs=captures,record=record_ref,
                record_sha256=_digest(original),record_id=record_id,processing_epoch=epoch,
                published_monotonic_ns=time.monotonic_ns(),origin_result_sha256=origin_result_sha256,
                source_mapping_sha256=mapping_sha256)
            rowdata=_line(row,ROW_BYTES); path='publications/row-%06d.json'%ordinal
            future=self._loop.create_future()
            # Retrieve internal errors even if the caller only preserves prefix.
            future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
            self._global_refs(len(captures)+2,1)
            self._admit(('publication',record_path,data,path,rowdata,future),len(data)+len(rowdata),paths=(record_path,path))
            self._record_ids.add(record_id);self._publications+=1;self._counts['publications']+=1
            return PendingPublication(self,future,ordinal,record_id,row['record_sha256'])

    async def publication_ready(self,handle):
        if type(handle) is not PendingPublication or handle._owner is not self:raise ValueError('publication handle belongs to another writer')
        return dict(await asyncio.shield(handle._future))

    async def _write_named(self,path,value,maximum):
        data=_line(value,maximum); future=self._loop.create_future()
        future.add_done_callback(lambda f:None if f.cancelled() else f.exception())
        self._admit(('file',path,data,future),len(data),paths=(path,))
        try:return dict(await asyncio.shield(future))
        except asyncio.CancelledError:
            await _finish(future);raise

    async def write_task(self,task_ordinal,kind,value):
        integer(task_ordinal,0,63); _enum(kind,('context','envelope','record','evidence','result','receipt'))
        if self._parent_imported and kind!='receipt':raise RuntimeError('parent cannot manufacture child task artifacts')
        maximum=32768 if kind=='context' else 128*1024 if kind in ('record','envelope') else PAGE_BYTES
        if kind in ('evidence','result','receipt'):validate_task_artifact(kind,value,self.execution_sha256)
        if kind in ('context','envelope','record'):
            size=len(_line(value,maximum))
            with self._lock:
                if self._snapshot_bytes+size>4*1024**2:raise RuntimeError('snapshot artifact inventory exceeds4MiB')
                self._snapshot_bytes+=size
        return await self._write_named('tasks/task-%04d-%s.json'%(task_ordinal,kind),value,maximum)

    async def write_request(self,ordinal,value):
        integer(ordinal,0,16383)
        return await self._write_named('requests/request-%06d.json'%ordinal,value,256*1024)

    async def write_native_inventory(self,inventory):
        from .recording_native_channel import NativeRunInventory
        from .recording_native_schema import snapshot
        if type(inventory) is not NativeRunInventory:raise TypeError('trusted parent-issued native inventory required')
        context=inventory['context']
        if any(context[k]!=self.execution[k] for k in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal')):
            raise ValueError('parent native inventory original execution mismatch')
        return await self._write_named('native-inventory.json',decode(encoded(inventory),PAGE_BYTES),PAGE_BYTES)

    async def write_fill_case(self,ordinal,value):
        integer(ordinal,0,16383);validate_fill_case(value,self.execution_sha256)
        return await self._write_named('fill/case-%06d.json'%ordinal,value,16*1024)

    async def write_metadata(self,kind,value):
        _enum(kind,('handshake','mapping','source-mappings','capture-inventory','child-summary','run-inventory','execution'))
        validate_metadata(kind,value,self.execution_sha256)
        if kind=='capture-inventory':
            if self._sealed is None or value['delivery_outcomes']!=self._sealed['deliveries']:
                raise ValueError('capture inventory follows and binds sealed delivery outcomes')
        if kind=='child-summary':
            if self._sealed is None:raise ValueError('child summary follows sealed runtime inventories')
            expected={'tick_inventory':'ticks','adapter_outcomes':'adapters','delivery_outcomes':'deliveries',
                'publication_inventory':'publications','mapping_updates':'mapping-updates'}
            if any(value[field]!=self._sealed[name] for field,name in expected.items()):raise ValueError('child summary must retain exact sealed runtime indices')
            actual=[ref for path,ref in sorted(self._files.items()) if path.startswith('tasks/task-') and path.endswith('-result.json')]
            if value['task_results']!=actual:raise ValueError('child summary cannot omit/reorder written original task results')
        if self._sealed is not None:
            if kind not in ('capture-inventory','child-summary','run-inventory') or self._directory is None or self._aborted or self._final_closed:
                raise RuntimeError('only final summaries may follow sealed outcome inventory')
            if self._failure and value.get('state')=='closed':raise ValueError('failed prefix cannot publish successful summary')
            data=_line(value,PAGE_BYTES);path=kind+'.json'
            with self._lock:
                if path in self._paths:raise ValueError('immutable final summary already admitted')
                self._charge(len(data));self._paths.add(path)
            task=asyncio.create_task(self._thread(self._store,path,data));self._final_writes.add(task)
            try:
                try:return await asyncio.shield(task)
                except asyncio.CancelledError:
                    await _finish(task);raise
            finally:self._final_writes.discard(task)
        return await self._write_named(kind+'.json',value,PAGE_BYTES)

    def _page(self,kind,rows):
        refs=self._data_refs[kind]
        return dict(schema_version=1,kind=KINDS[kind],execution_sha256=self.execution_sha256,
            page_ordinal=len(refs),previous_page_sha256=refs[-1]['file_sha256'] if refs else None,rows=rows)

    async def _flush(self,kind):
        rows=self._rows[kind]
        if not rows:return
        projected={k:len(v)+(1 if k==kind else 0) for k,v in self._data_refs.items()}
        if sum(projected.values())+sum((n+127)//128 for n in projected.values())>MAX_PAGES:
            raise RuntimeError('combined runtime data/index page inventory exhausted')
        data=_line(self._page(kind,rows),PAGE_BYTES)
        ref=await self._thread(self._store,kind+'/page-%06d.json'%len(self._data_refs[kind]),data)
        self._data_refs[kind].append(ref);self._rows[kind]=[];self._pages+=1
        self._durable_counts[kind]+=len(rows)

    async def _append_row(self,kind,row):
        rows=self._rows[kind]
        try:_line(self._page(kind,rows+[row]),PAGE_BYTES)
        except ValueError:await self._flush(kind)
        self._rows[kind].append(row)
        if len(self._rows[kind])==128:await self._flush(kind)

    async def _write(self):
        try:
            while True:
                with self._lock:
                    queued=self._queue.popleft() if self._queue else None
                    stop=self._stopping
                    if queued is None:self._wake.clear()
                if queued is None:
                    if stop:
                        for kind in KINDS:await self._flush(kind)
                        return
                    await self._wake.wait();continue
                job,size=queued
                try:
                    if job[0]=='row':await self._append_row(job[1],job[2])
                    elif job[0]=='file':
                        ref=await self._thread(self._store,job[1],job[2]);job[3].set_result(ref)
                    else:
                        await self._thread(self._store,job[1],job[2])
                        ref=await self._thread(self._store,job[3],job[4]);await self._append_row('publications',ref)
                        job[5].set_result(ref)
                except BaseException as exc:
                    if job[0]!='row' and not job[-1].done():job[-1].set_exception(exc)
                    raise
                finally:
                    with self._lock:self._pending_frames-=1;self._pending_bytes-=size
        except BaseException as exc:
            self._fail('owned_write_failed')
            with self._lock:
                while self._queue:
                    job,size=self._queue.popleft()
                    if job[0]!='row' and not job[-1].done():job[-1].set_exception(RuntimeError('runtime writer failed before durable artifact'))
                self._pending_frames=self._pending_bytes=0

    async def seal_inventory(self):
        with self._lock:
            self._accepting=False;self._stopping=True;self._wake.set()
        if self._seal_task is None:self._seal_task=asyncio.create_task(self._seal())
        try:return dict(await asyncio.shield(self._seal_task))
        except asyncio.CancelledError:
            await _finish(self._seal_task);raise

    async def _seal(self):
        await _finish(self._writer)
        inventories={}
        for kind in KINDS:
            if kind in self._imported_indices:
                inventories[kind]=self._imported_indices[kind];continue
            refs=self._data_refs[kind];tail=None;index_count=0
            for start in range(0,len(refs),128):
                if self._pages>=MAX_PAGES:raise RuntimeError('combined runtime index page cap exhausted')
                page=dict(schema_version=1,kind='recording_outcome_index_page',execution_sha256=self.execution_sha256,
                    page_kind=KINDS[kind],page_ordinal=index_count,previous_page_sha256=tail['file_sha256'] if tail else None,pages=refs[start:start+128])
                data=_line(page,PAGE_BYTES)
                tail=await self._thread(self._store,kind+'/index-%06d.json'%index_count,data)
                self._pages+=1;index_count+=1
            count=self._durable_counts[kind]
            index=dict(schema_version=1,kind='recording_outcome_index',execution_sha256=self.execution_sha256,
                page_kind=KINDS[kind],page_count=len(refs),row_count=count,index_page_count=index_count,
                index_tail=tail,state='incomplete' if self._failure or self._aborted or count!=self._counts[kind] else 'complete')
            data=_line(index,PAGE_BYTES);self._charge(len(data))
            inventories[kind]=await self._thread(self._store,kind+'-index.json',data)
        self._sealed=inventories
        return inventories

    async def ingest_child(self,directory,*,reaped_inventory,expected_refs=(),allow_financial=False):
        """Exclusive byte-preserving import after the actual owned child reap.

        Native files already arrived through parent native2 and are never
        overwritten. The parent alone writes receipts, handshake, channel rows
        and its final inventory. Unknown files and aliases fail closed.
        """
        from .recording_native_channel import NativeRunInventory
        if type(reaped_inventory) is not NativeRunInventory or not reaped_inventory['parent_reaped']:
            raise TypeError('actual parent-issued reaped native inventory required')
        context=reaped_inventory['context']
        if any(context[k]!=self.execution[k] for k in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal')):
            raise ValueError('reaped child belongs to another original execution')
        if self._parent_imported or self._sealed is not None:raise RuntimeError('one original child import before parent seal required')
        expected={}
        for ref in expected_refs:
            validate_ref(ref)
            if ref['path'] in expected and expected[ref['path']]!=ref:raise ValueError('conflicting original accepted artifact reference')
            expected[ref['path']]=dict(ref)
        if type(allow_financial) is not bool:raise TypeError('explicit original financial policy required')
        result=await self._thread(self._import_files,directory,expected,allow_financial)
        for kind in KINDS:
            if kind=='channels':continue
            path=kind+'-index.json'
            if path in result:
                ref=result[path];index=read_ref(self.output,ref)
                for _ in iter_outcome_rows(self.output,ref,execution_sha256=self.execution_sha256,kind=kind):pass
                self._imported_indices[kind]=ref;self._counts[kind]=index['row_count']
        self._parent_imported=True
        return dict(result)

    def _import_files(self,directory,expected,allow_financial):
        from .recording_runtime_inputs import _absolute_directory,_open_file,_identity
        root=_absolute_directory(str(Path(directory).absolute()));files=[];total_runtime=0;file_count=0;financial_count=0
        try:
            def scan(fd,prefix=''):
                nonlocal total_runtime,file_count,financial_count
                with os.scandir(fd) as entries:
                    for entry in entries:
                        path=prefix+entry.name;info=entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            financial_dir=allow_financial and (not prefix and entry.name=='financial' or prefix=='financial/' and entry.name in ('source','evidence','settlements'))
                            if not financial_dir and (prefix or entry.name not in (*KINDS,'tasks','records','requests','fill','source','native')):
                                raise ValueError('unlisted child artifact directory')
                            child=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
                            try:scan(child,path+'/')
                            finally:os.close(child)
                        elif stat.S_ISREG(info.st_mode):
                            financial=allow_financial and re.fullmatch(r'financial/(?:source/source-[0-9a-f]{64}|evidence/evidence-[0-9]{6}|settlements/receipt-[0-9]{6})\.json',path)
                            if not (financial or _REF_PATH.fullmatch(path) or _ORIGINAL_REF_PATH.fullmatch(path)):
                                raise ValueError('unlisted child artifact file')
                            if path in ('handshake.json','run-inventory.json','native-inventory.json') or path.endswith('-receipt.json'):
                                raise ValueError('child cannot author parent receipts or closure')
                            file_count+=1
                            if file_count>32768:raise ValueError('bounded child artifact file inventory required')
                            if path.startswith('native/'):continue
                            if path=='source/mapping-records.json' and info.st_size>65536:raise ValueError('bounded original mapping records required')
                            if path=='source/controller-settlements.json' and (not allow_financial or info.st_size>1024**2):raise ValueError('U2 bounded original controller settlements required')
                            if financial:
                                financial_count+=1
                                if financial_count>768 or not 1<=info.st_size<=65536:raise ValueError('bounded original financial files required')
                                financial_fd=_open_file(root,path)
                                try:raw=os.read(financial_fd,65537)
                                finally:os.close(financial_fd)
                                if _line(decode(raw,65536),65536)!=raw:raise ValueError('original financial canonical bytes required')
                                if path.startswith('financial/source/') and path[len('financial/source/source-'):-5]!=_hash(raw):raise ValueError('original financial source filename mismatch')
                            if path=='channels-index.json':
                                # Child's empty channel inventory is not parent
                                # observation. It must be empty, never relabeled.
                                childfd=_open_file(root,path)
                                try:data=os.read(childfd,PAGE_BYTES+1)
                                finally:os.close(childfd)
                                value=decode(data,PAGE_BYTES)
                                if (value!={'schema_version':1,'kind':'recording_outcome_index','execution_sha256':self.execution_sha256,'page_kind':KINDS['channels'],'page_count':0,'row_count':0,'index_page_count':0,'index_tail':None,'state':'complete'}):
                                    raise ValueError('child cannot assert parent channel outcomes')
                                continue
                            if path.startswith('channels/'):raise ValueError('child channel observations cannot replace parent evidence')
                            if not path.startswith(('source/','financial/')):
                                total_runtime+=info.st_size
                                if total_runtime>MAX_METADATA_BYTES:raise ValueError('child runtime bytes exceed16MiB')
                            elif info.st_size>1024**3:raise ValueError('original source artifact exceeds bound')
                            files.append((path,_identity(info)))
                        else:raise ValueError('child artifacts must be regular nofollow files')
            scan(root)
            paths={path for path,_ in files}
            if not set(expected)<=paths:raise ValueError('previously accepted original artifact missing after reap')
            if any(path in self._paths for path in paths):raise ValueError('child import would replace parent artifact')
            for kind in KINDS:
                path=kind+'-index.json'
                if kind!='channels' and path in paths:
                    fd=_open_file(root,path)
                    try:data=os.read(fd,PAGE_BYTES+1)
                    finally:os.close(fd)
                    ref=dict(path=path,file_sha256=_hash(data),bytes=len(data))
                    for _ in iter_outcome_rows(directory,ref,execution_sha256=self.execution_sha256,kind=kind):pass
            for ref in expected.values():
                if not ref['path'].startswith(('source/','native/')):read_ref(directory,ref)
            if any(path.startswith('source/') for path in paths) and 'source' not in self._subdirs:
                os.mkdir('source',mode=0o700,dir_fd=self._directory)
                self._subdirs['source']=os.open('source',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self._directory)
            result={}
            for path,identity in sorted(files):
                source=_open_file(root,path)
                try:
                    if _identity(os.fstat(source))!=identity:raise ValueError('child artifact changed before import')
                    size=identity[3]
                    if path.startswith('financial/'):
                        with self.budget._lock:
                            source_bytes,metadata_bytes=getattr(self.budget,'_runtime_financial_import_bytes',(0,0))
                            if path.startswith('financial/source/'):source_bytes+=size
                            else:metadata_bytes+=size
                            if max(source_bytes,metadata_bytes)>64*1024**2:raise ValueError('collection original financial import bound exhausted')
                            self.budget._runtime_financial_import_bytes=(source_bytes,metadata_bytes)
                    if path.startswith(('source/','financial/')):self.budget.reserve(size)
                    else:self._charge(size)
                    self._global_refs(1);self._paths.add(path)
                    parts=path.split('/')
                    if parts[0]=='financial':
                        if 'financial' not in self._subdirs:
                            os.mkdir('financial',mode=0o700,dir_fd=self._directory)
                            self._subdirs['financial']=os.open('financial',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self._directory)
                        key='/'.join(parts[:-1])
                        if key not in self._subdirs:
                            os.mkdir(parts[1],mode=0o700,dir_fd=self._subdirs['financial'])
                            self._subdirs[key]=os.open(parts[1],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self._subdirs['financial'])
                    parent=self._directory if len(parts)==1 else self._subdirs['/'.join(parts[:-1])]
                    target=os.open(parts[-1],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
                    hashed=hashlib.sha256();copied=0
                    try:
                        os.fchmod(target,0o600)
                        while copied<=size:
                            block=os.read(source,min(65536,size+1-copied))
                            if not block:break
                            copied+=len(block)
                            if copied>size:raise ValueError('child artifact grew during import')
                            hashed.update(block);_write_all(target,block)
                        os.fsync(target)
                    finally:os.close(target)
                    if copied!=size or _identity(os.fstat(source))!=identity:raise ValueError('child original changed during import')
                    ref=dict(path=path,file_sha256=hashed.hexdigest(),bytes=size)
                    if path in expected and ref!=expected[path]:raise ValueError('accepted original artifact changed before import')
                    self._files[path]=ref;result[path]=ref
                finally:os.close(source)
            return result
        finally:os.close(root)

    async def close(self,*,cleanup_finished):
        if cleanup_finished is not True:raise ValueError('owned runtime cleanup must precede artifact close')
        if self._abort_task is not None:
            await _finish(self._abort_task);raise RuntimeError('aborted runtime artifacts')
        if self._close_task is None:
            self._final_closed=True;self._close_task=asyncio.create_task(self._close())
        try:return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task);raise

    async def _close(self):
        try:
            result=await self.seal_inventory()
            for task in tuple(self._final_writes):await _finish(task)
            if self._failure or self._aborted:raise RuntimeError('runtime artifacts retain incomplete prefix')
            return result
        finally:await self._thread(self._cleanup)

    def status(self):
        with self._lock:
            return dict(failure=self._failure,queued_frames=self._pending_frames,queued_bytes=self._pending_bytes,
                admitted_rows=dict(self._counts),durable_files=len(self._files),reserved_bytes=self._run_bytes,
                state='incomplete' if self._failure or self._aborted else 'open' if self._accepting else 'sealed',
                actual_acceptance='incomplete',worker_backend_coverage_verified=False,receipt_authenticity_verified=False)

    async def abort(self):
        if self._abort_task is None:
            self._aborted=True;self._fail('aborted');self._stopping=True;self._wake.set()
            self._final_closed=True
            self._abort_task=asyncio.create_task(self._abort())
        try:await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            await _finish(self._abort_task);raise

    async def _abort(self):
        try:
            if self._close_task is not None:
                try:await _finish(self._close_task)
                except Exception:pass
            elif self._seal_task is not None:
                try:await _finish(self._seal_task)
                except Exception:pass
            elif self._writer is not None:await _finish(self._writer)
        finally:
            for task in tuple(self._final_writes):
                try:await _finish(task)
                except Exception:pass
            await self._thread(self._cleanup)

    def _cleanup(self):
        for fd in self._subdirs.values():os.close(fd)
        self._subdirs.clear()
        if self._directory is not None:os.close(self._directory);self._directory=None


def validate_task_artifact(kind,value,execution_sha256):
    common=('schema_version','kind','execution_sha256','task_id','input_sha256')
    fields={
        'result':('scope_index','status','reason','evidence','answer_text','answer_sha256','operation_instance','child_finished_ns','cleanup_finished'),
        'evidence':('scope_index','mapping_sha256','context','context_object_sha256','snapshot_envelope','snapshot_record','snapshot_record_sha256','snapshot_observed_begin_ns','snapshot_observed_end_ns','controller_request_id','request_evidence_sha256','prompt_sha256','operation_instance'),
        'receipt':('result_sha256','received_ns','failed_observed_ns','status','failure_reason','answer_sha256','latency_ns'),
    }
    keys(value,common+fields[kind])
    if type(value['schema_version']) is not int or value['schema_version']!=1 or value['kind']!={'result':'recording_task_result','evidence':'recording_task_evidence','receipt':'recording_parent_task_receipt'}[kind]:raise ValueError('exact task artifact schema required')
    if value['execution_sha256']!=execution_sha256:raise ValueError('task artifact execution mismatch')
    identifier(value['task_id']);sha(value['input_sha256'])
    if kind!='receipt':integer(value['scope_index'],1,64)
    if kind=='result':
        _enum(value['status'],('answer','failed'));_enum(value['reason'],REASONS)
        _nullable(validate_ref,value['evidence']);_nullable(lambda v:_text(v,8192),value['answer_text'])
        _nullable(sha,value['answer_sha256']);_nullable(sha,value['operation_instance'])
        integer(value['child_finished_ns']);_boolean(value['cleanup_finished'])
        if value['status']=='answer':
            if value['reason']!='completed' or any(value[n] is None for n in ('evidence','answer_text','answer_sha256','operation_instance')):raise ValueError('answer requires completed actual operation/evidence')
            if _hash(value['answer_text'].encode('utf-8'))!=value['answer_sha256']:raise ValueError('exact answer text digest required')
        elif value['answer_text'] is not None or value['answer_sha256'] is not None or value['reason']=='completed':raise ValueError('failed task cannot contain accepted answer')
    elif kind=='evidence':
        for name in ('mapping_sha256','context_object_sha256','snapshot_record_sha256'):sha(value[name])
        for name in ('context','snapshot_envelope','snapshot_record'):validate_ref(value[name])
        for name in ('snapshot_observed_begin_ns','snapshot_observed_end_ns'):integer(value[name])
        if value['snapshot_observed_end_ns']<value['snapshot_observed_begin_ns']:raise ValueError('ordered snapshot interval required')
        _nullable(identifier,value['controller_request_id'])
        for name in ('request_evidence_sha256','prompt_sha256','operation_instance'):_nullable(sha,value[name])
    else:
        for name in ('result_sha256','answer_sha256'):_nullable(sha,value[name])
        for name in ('received_ns','failed_observed_ns'):_nullable(integer,value[name])
        integer(value['latency_ns']);_enum(value['status'],('answer','late_answer','failed','not_executed'))
        _enum(value['failure_reason'],REASONS+('no_receipt','child_failed','protocol_failed'))
        if value['status']=='answer' and (value['failure_reason']!='completed' or any(value[k] is None for k in ('received_ns','result_sha256','answer_sha256'))):
            raise ValueError('accepted answer receipt needs validated original result and actual receipt clock')
    _line(value,PAGE_BYTES)
    return value


def validate_metadata(kind,value,execution_sha256):
    if kind=='execution':
        validate_execution(value)
        if digest(value)!=execution_sha256:raise ValueError('original execution changed')
        return
    if kind=='mapping':
        names=('schema_version','source_manifest_sha256','recording_id','origin_capture_ns','duration_ns','sdk_anchor_ms',
            'wall_anchor_ns','monotonic_anchor_ns','scale_numerator','scale_denominator','rounding','max_wall_monotonic_drift_ns')
        keys(value,names);sha(value['source_manifest_sha256']);identifier(value['recording_id'])
        for name in set(names)-{'source_manifest_sha256','recording_id','rounding'}:integer(value[name])
        if (value['schema_version'],value['scale_numerator'],value['scale_denominator'],value['rounding'])!=(1,1,1,'floor_to_sdk_milliseconds'):raise ValueError('unchanged one-to-one original replay mapping required')
        integer(value['duration_ns'],1);integer(value['max_wall_monotonic_drift_ns'],1,1000000000)
        return
    if kind=='source-mappings':
        if type(value) is not list or len(value)>16:raise ValueError('bounded original source mappings required')
        for row in value:
            if type(row) is not dict:raise ValueError('original source mapping object required')
            extra={'mapping_sha256','original_source','provider_epochs'}
            if not extra<=set(row):raise ValueError('original source/provider epoch mapping required')
            mapping={k:v for k,v in row.items() if k not in extra};validate_metadata('mapping',mapping,execution_sha256)
            if row['mapping_sha256']!=digest(mapping):raise ValueError('original source clock mapping digest mismatch')
            if type(row['original_source']) is not dict or type(row['provider_epochs']) is not list or len(row['provider_epochs'])>64:raise ValueError('bounded original source/provider epoch inventory required')
            for epoch in row['provider_epochs']:
                keys(epoch,('capability_id','stream_epoch'));identifier(epoch['capability_id']);identifier(epoch['stream_epoch'])
        return
    if kind=='handshake':
        keys(value,('schema_version','kind','execution_sha256','nonce','clock_domain','clock_implementation','ready_received_ns',
            'start_send_begin_ns','start_send_complete_ns','playback_received_ns','anchor_monotonic_ns','state','reason'))
        sha(value['nonce']);identifier(value['clock_domain']);_text(value['clock_implementation'],256)
        if any(not 32<=ord(c)<=126 for c in value['clock_implementation']):raise ValueError('printable clock implementation required')
        names=('ready_received_ns','start_send_begin_ns','start_send_complete_ns','playback_received_ns','anchor_monotonic_ns')
        for n in names:_nullable(integer,value[n])
        _enum(value['state'],('started','failed'));_enum(value['reason'],('completed','startup_timeout','start_write_failed','invalid_mapping','clock_mismatch','protocol_failed'))
        if value['state']=='started':
            if any(value[n] is None for n in names) or value['reason']!='completed':raise ValueError('started handshake needs all actual milestones')
            if not value['ready_received_ns']<=value['start_send_begin_ns']<=value['anchor_monotonic_ns']<=value['playback_received_ns'] or value['start_send_complete_ns']<value['start_send_begin_ns']:raise ValueError('original START-begin mapping enclosure required')
        expected='recording_handshake'
    elif kind=='capture-inventory':
        keys(value,('schema_version','kind','execution_sha256','source_manifest_sha256','mapping_sha256','source_mappings',
            'source_terminal','source_events','source_counters','delivery_outcomes'))
        sha(value['source_manifest_sha256']);sha(value['mapping_sha256'])
        for n in ('source_mappings','source_terminal','source_counters','delivery_outcomes'):validate_ref(value[n])
        if type(value['source_events']) is not list or len(value['source_events'])>2048:raise ValueError('bounded original source events required')
        for ref in value['source_events']:validate_ref(ref)
        expected='recording_capture_inventory'
    elif kind=='child-summary':
        keys(value,('schema_version','kind','execution_sha256','mapping','capture_inventory','task_results','tick_inventory',
            'adapter_outcomes','delivery_outcomes','publication_inventory','mapping_updates','child_cleanup_finished','clock_state','state','failure_counts'))
        for n in ('mapping','capture_inventory'):_nullable(validate_ref,value[n])
        for n in ('tick_inventory','adapter_outcomes','delivery_outcomes','publication_inventory','mapping_updates'):validate_ref(value[n])
        if type(value['task_results']) is not list or len(value['task_results'])>64:raise ValueError('bounded original task result inventory required')
        for ref in value['task_results']:validate_ref(ref)
        _boolean(value['child_cleanup_finished']);_enum(value['clock_state'],('consistent','discontinuous','unknown'))
        _enum(value['state'],('closed','failed'));_failure_counts(value['failure_counts'])
        if value['state']=='closed' and (not value['child_cleanup_finished'] or value['clock_state']!='consistent' or value['mapping'] is None or value['capture_inventory'] is None or value['failure_counts']):raise ValueError('closed child requires known cleanup/clock/source without failures')
        expected='recording_child_summary'
    elif kind=='run-inventory':
        keys(value,('schema_version','kind','execution_sha256','invocation_ordinal','nonce','spawn_started_ns','process_reaped_ns',
            'parent_resources_closed_ns','child_summary','native_inventory','source_inventory','receipts','channel_outcomes',
            'handshake','transport_outcomes','exit_code','state','failure_counts'))
        integer(value['invocation_ordinal'],0,95);sha(value['nonce']);integer(value['spawn_started_ns'])
        for n in ('process_reaped_ns','parent_resources_closed_ns'):_nullable(integer,value[n])
        for n in ('child_summary','source_inventory'):_nullable(validate_ref,value[n])
        for n in ('native_inventory','channel_outcomes','handshake','transport_outcomes'):validate_ref(value[n])
        if type(value['receipts']) is not list or len(value['receipts'])>64:raise ValueError('bounded all-task parent receipts required')
        for ref in value['receipts']:validate_ref(ref)
        _nullable(lambda v:integer(v,-2**31,2**31-1),value['exit_code']);_enum(value['state'],('closed','failed'));_failure_counts(value['failure_counts'])
        if value['state']=='closed' and (any(value[n] is None for n in ('process_reaped_ns','parent_resources_closed_ns','child_summary')) or value['exit_code']!=0 or value['failure_counts']):raise ValueError('closed parent requires owned reap/resources and complete child')
        expected='recording_run_inventory'
    else:raise ValueError('known metadata artifact required')
    if type(value['schema_version']) is not int or value['schema_version']!=1 or value['kind']!=expected or value['execution_sha256']!=execution_sha256:raise ValueError('original runtime metadata binding required')
    _line(value,PAGE_BYTES)


def _failure_counts(value):
    if type(value) is not dict or any(k not in REASONS+('no_receipt','child_failed','protocol_failed','cleanup_incomplete','capacity_exhausted') for k in value):raise ValueError('finite runtime failure categories required')
    for count in value.values():integer(count,1)


def read_ref(directory,ref,*,maximum=PAGE_BYTES):
    """Strict canonical immutable bytes, regular files, every component nofollow."""
    validate_ref(ref)
    if _ORIGINAL_REF_PATH.fullmatch(ref['path']):raise ValueError('original source/native references require their unchanged format reader')
    if ref['bytes']>maximum:raise ValueError('runtime reference exceeds selected byte bound')
    path=Path(directory).absolute();fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]+tuple(ref['path'].split('/')[:-1]):
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        filefd=os.open(ref['path'].split('/')[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        try:
            info=os.fstat(filefd)
            if not stat.S_ISREG(info.st_mode) or info.st_size!=ref['bytes']:raise ValueError('exact regular runtime artifact required')
            data=bytearray()
            while len(data)<=maximum:
                piece=os.read(filefd,min(65536,maximum+1-len(data)))
                if not piece:break
                data.extend(piece)
            after=os.fstat(filefd)
            if (info.st_size,info.st_mtime_ns,info.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('runtime artifact changed while reading')
        finally:os.close(filefd)
    finally:os.close(fd)
    raw=bytes(data)
    if len(raw)!=ref['bytes'] or _hash(raw)!=ref['file_sha256']:raise ValueError('runtime artifact digest/length mismatch')
    value=decode(raw,maximum)
    if _line(value,maximum)!=raw:raise ValueError('exact canonical runtime artifact bytes required')
    return value


def iter_outcome_rows(directory,index_ref,*,execution_sha256,kind):
    """Validate bounded full hash chains before exposing immutable row copies.

    Structural decoding is not a source/native join or a completeness waiver.
    The caller must require index.state='complete' for a complete-run claim.
    """
    sha(execution_sha256)
    if kind not in KINDS or index_ref['path']!=kind+'-index.json':raise ValueError('fixed outcome index identity required')
    index=read_ref(directory,index_ref)
    keys(index,('schema_version','kind','execution_sha256','page_kind','page_count','row_count','index_page_count','index_tail','state'))
    if type(index['schema_version']) is not int or index['schema_version']!=1 or index['kind']!='recording_outcome_index' or index['execution_sha256']!=execution_sha256 or index['page_kind']!=KINDS[kind]:raise ValueError('original outcome index binding required')
    count=integer(index['page_count'],0,MAX_PAGES);n=integer(index['index_page_count'],0,16)
    integer(index['row_count'],0,LIMITS[kind]);_enum(index['state'],('complete','incomplete'))
    if n!=(count+127)//128 or count+n>MAX_PAGES:raise ValueError('bounded complete index-page inventory required')
    if not count:
        if index['index_tail'] is not None or index['row_count']!=0:raise ValueError('empty outcome inventory must be explicit')
        return
    ref=index['index_tail'];groups=[]
    for ordinal in range(n-1,-1,-1):
        validate_ref(ref)
        if ref['path']!=kind+'/index-%06d.json'%ordinal:raise ValueError('fixed contiguous index locator required')
        page=read_ref(directory,ref)
        keys(page,('schema_version','kind','execution_sha256','page_kind','page_ordinal','previous_page_sha256','pages'))
        if (type(page['schema_version']) is not int or page['schema_version']!=1 or page['kind']!='recording_outcome_index_page' or page['execution_sha256']!=execution_sha256 or page['page_kind']!=KINDS[kind] or type(page['page_ordinal']) is not int or page['page_ordinal']!=ordinal):raise ValueError('original ordered index page required')
        expected=128 if ordinal<n-1 else count-128*ordinal
        if type(page['pages']) is not list or len(page['pages'])!=expected:raise ValueError('nonfinal index pages must be full')
        for value in page['pages']:validate_ref(value)
        groups.append(page['pages'])
        if ordinal:
            sha(page['previous_page_sha256'])
            # Previous files have fixed names; bounded read to recover exact
            # size is followed by the same nofollow/hash verifier.
            path=kind+'/index-%06d.json'%(ordinal-1)
            ref=_fixed_ref(directory,path,page['previous_page_sha256'])
        elif page['previous_page_sha256'] is not None:raise ValueError('index genesis previous hash must be null')
    refs=[ref for group in reversed(groups) for ref in group]
    if sum(r['bytes'] for r in refs)>MAX_METADATA_BYTES:raise ValueError('runtime outcome metadata exceeds16MiB')
    previous=None;row_count=0;logical=0
    for ordinal,ref in enumerate(refs):
        if ref['path']!=kind+'/page-%06d.json'%ordinal:raise ValueError('duplicate/reordered outcome page')
        page=read_ref(directory,ref)
        keys(page,('schema_version','kind','execution_sha256','page_ordinal','previous_page_sha256','rows'))
        if (type(page['schema_version']) is not int or page['schema_version']!=1 or page['kind']!=KINDS[kind] or page['execution_sha256']!=execution_sha256 or type(page['page_ordinal']) is not int or page['page_ordinal']!=ordinal or page['previous_page_sha256']!=previous):raise ValueError('original outcome page chain mismatch')
        if type(page['rows']) is not list or not 1<=len(page['rows'])<=128:raise ValueError('nonempty bounded outcome page required')
        for row in page['rows']:
            validate_outcome(kind,row)
            if kind=='ticks':logical+=row['count']
            row_count+=1
            if row_count>index['row_count'] or logical>16384:raise ValueError('outcome row/logical opportunity bound exceeded')
        previous=ref['file_sha256']
    if row_count!=index['row_count']:raise ValueError('outcome row count mismatch')
    # Re-read exact pinned bytes after the complete bounded first pass. Do not
    # retain up to262144 decoded rows just to yield an iterator.
    for ref in refs:yield from read_ref(directory,ref)['rows']


def _fixed_ref(directory,path,expected_hash):
    # Use descriptor-relative nofollow components even for this bounded stat.
    base=Path(directory).absolute();fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in base.parts[1:]+tuple(path.split('/')[:-1]):
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        info=os.stat(path.split('/')[-1],dir_fd=fd,follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or not 1<=info.st_size<=PAGE_BYTES:raise ValueError('bounded original index page required')
    finally:os.close(fd)
    return dict(path=path,file_sha256=expected_hash,bytes=info.st_size)


FILL_OUTCOMES=('indeterminate_fill','indeterminate_claim','indeterminate_boundary','indeterminate_guard_band',
    'conditional_image_model_conflict','consistent_under_rule','resolved_under_rule','strong_opt_out',
    'blocked_stale_evidence','blocked_case_deadline','blocked_admission','stage_failed','blocked_stage',
    'indeterminate_invalid_claim','cancelled','invalidated')


def validate_fill_case(value,execution_sha256):
    keys(value,('schema_version','kind','execution_sha256','case_id','source_manifest_sha256','mapping_sha256',
        'capture_ref','image_sha256','configuration_sha256','rule_sha256','measurement_publication',
        'descriptor_publication','stages','comparison_publication','outcome'))
    if type(value['schema_version']) is not int or value['schema_version']!=1 or value['kind']!='recording_fill_case' or value['execution_sha256']!=execution_sha256:
        raise ValueError('original recording fill case required')
    identifier(value['case_id']);_capture(value['capture_ref'])
    for name in ('source_manifest_sha256','mapping_sha256','image_sha256','configuration_sha256','rule_sha256'):sha(value[name])
    for name in ('measurement_publication','descriptor_publication'):validate_ref(value[name])
    _nullable(validate_ref,value['comparison_publication']);_enum(value['outcome'],FILL_OUTCOMES)
    if type(value['stages']) is not list or len(value['stages'])>3:raise ValueError('bounded actual fill stages required')
    previous=None
    for i,stage in enumerate(value['stages']):
        keys(stage,('stage','binding_index','controller_request_id','operation_instance','prompt_sha256','raw_response_sha256','parsed_claim_sha256','rule_outcome','predecessor_instance','status'))
        if stage['stage']!=('initial','reinspection','strong')[i]:raise ValueError('original fill stage order required')
        integer(stage['binding_index']);identifier(stage['controller_request_id']);sha(stage['prompt_sha256'])
        for name in ('operation_instance','raw_response_sha256','parsed_claim_sha256','predecessor_instance'):_nullable(sha,stage[name])
        _enum(stage['rule_outcome'],FILL_OUTCOMES);_enum(stage['status'],('admitted','blocked','failed','completed'))
        if stage['predecessor_instance']!=previous:raise ValueError('explicit actual prior fill stage required')
        if stage['status']=='completed' and stage['operation_instance'] is None:raise ValueError('completed fill stage needs actual operation')
        previous=stage['operation_instance']
    _line(value,16*1024)
    return value


def read_publication(directory,ref,*,execution_sha256):
    """Rebind retained accepted-record bytes; actual acceptance is SDK-owned.

    The caller additionally joins the issuer-provided operation and acquired
    capture inventory. A stored record alone cannot authenticate that boundary.
    """
    from gamst.call_controller import _digest
    row=read_ref(directory,ref,maximum=ROW_BYTES)
    keys(row,('schema_version','kind','execution_sha256','publication_ordinal','binding_index','builder_sha256',
        'operation_instance','role','capture_refs','record','record_sha256','record_id','processing_epoch',
        'published_monotonic_ns','origin_result_sha256','source_mapping_sha256'))
    if type(row['schema_version']) is not int or row['schema_version']!=1 or row['kind']!='recording_publication' or row['execution_sha256']!=execution_sha256:raise ValueError('original publication binding required')
    ordinal=integer(row['publication_ordinal'],0,262143)
    if ref['path']!='publications/row-%06d.json'%ordinal or row['record']['path']!='records/record-%06d.json'%ordinal:raise ValueError('original fixed publication ordinal/record required')
    identifier(row['record_id']);identifier(row['processing_epoch']);integer(row['published_monotonic_ns'])
    sha(row['record_sha256']);sha(row['source_mapping_sha256']);_nullable(sha,row['origin_result_sha256'])
    _enum(row['role'],('specialist_result','raw_capture_descriptor','semantic_description','fill_measurement','fill_descriptor','fill_comparison','task_snapshot'))
    if row['role'] in ('raw_capture_descriptor','task_snapshot'):
        sha(row['builder_sha256'])
        if row['binding_index'] is not None or row['operation_instance'] is not None:raise ValueError('deterministic publication cannot invent native operation')
    else:
        integer(row['binding_index']);sha(row['operation_instance'])
        if row['builder_sha256'] is not None:raise ValueError('backend publication cannot use deterministic builder')
    if type(row['capture_refs']) is not list or len(row['capture_refs'])>16:raise ValueError('bounded original capture inventory required')
    for capture in row['capture_refs']:_capture(capture)
    if len({digest(v) for v in row['capture_refs']})!=len(row['capture_refs']):raise ValueError('duplicate publication capture')
    record=read_ref(directory,row['record'],maximum=128*1024)
    if (type(record) is not dict or record.get('type')!='observation.signal' or record.get('id')!=row['record_id'] or
            record.get('observation',{}).get('processing_epoch')!=row['processing_epoch'] or _digest(record)!=row['record_sha256']):raise ValueError('accepted original record bytes/epoch/digest differ')
    return row,record
