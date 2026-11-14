"""Owned channel2 transfer of immutable native4 artifacts, not process attestation.

Processing is capped at262144 post-hello frames/run. Every received byte,
including invalid/truncated frames, consumes a separate wire cap of twice the
shared artifact allowance plus16MiB. Actual persisted bytes independently debit
ArtifactBudget; wire volume is never a monetary or compute measurement.
"""
import asyncio
import base64
from collections import deque
import hashlib
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time

from .recording_native_schema import (decode, encoded, digest, keys, integer, sha,
    run_context, validate_recording_scope_protocol)
from .recording_native_writer import ArtifactBudget
from .recording_native_validation import _FrozenDict, _freeze
from .recording_artifacts import _finish, _write_all

FD_ENVIRONMENT_KEY = 'GAMST_RECORDING_FD_V2'
MAX_FRAME = 512*1024
MAX_BATCH = 256*1024
MAX_DECLARATIONS = 16*1024*1024
MAX_FRAMES = 262144
MAX_WIRE_MULTIPLIER = 2
MAX_WIRE_OVERHEAD = 16*1024*1024
MAX_CHUNK = 1024*1024
REQUIRED = {'plan','collection','scope_protocol','instrumentation_protocol','source_manifest','tasks','profiles'}
OPTIONAL = {'runtime_policy','stream_protocol','allocation_protocol','cost_card'}
FRAME_FIELDS = {
    'chunk_data': {'chunk_ordinal','offset','data_base64'},
    'chunk_seal': {'chunk_ordinal','bytes','sha256'},
    'index_page': {'page_ordinal','page'},
    'receipt_source_data': {'source_sha256','offset','data_base64'},
    'receipt_source_seal': {'source_sha256','bytes'},
    'receipt_attachment': {'receipt_sha256','receipt'},
    'claims_page': {'page_sha256','page'},
    'counter_page': {'ordinal','page'},
    'terminal': {'terminal'},
}
BOOTSTRAP_KEYS = {'schema_version','type','nonce','invocation_ordinal','collection_sha256',
    'scope_protocol_sha256','instrumentation_protocol_sha256','deadline_ns','paths'}
HELLO_KEYS = BOOTSTRAP_KEYS - {'deadline_ns','paths'}
EVENTS = {'mapping_bound','reserved','evidence_ref','dispatched','entered','released','finished',
    'published','worker_registered','worker_milestone','work_started','work_finished','settlement_ref',
    'private_invalidated','incomplete'}


class NativeChannelError(RuntimeError):
    """Fatal protocol error: the process-owning caller must promptly reap."""


def _native_numbers(value, key=None):
    if type(value) is bool and key not in ('child_cleanup_finished','receipt_authenticity_verified','worker_backend_coverage_verified'):
        raise ValueError('boolean cannot stand in for native integer')
    if type(value) is float:
        raise ValueError('exact native integers required')
    if type(value) is dict:
        for name,item in value.items():_native_numbers(item,name)
    elif type(value) is list:
        for item in value:_native_numbers(item)


def _line(value):
    return encoded(value)+b'\n'


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def frame(value):
    data=encoded(value)
    if not 1<=len(data)<=MAX_FRAME: raise ValueError('channel2 frame exceeds512KiB')
    return struct.pack('!I',len(data))+data


async def _deadline(awaitable, deadline_ns):
    remaining=(deadline_ns-time.monotonic_ns())/1e9
    if remaining<=0:
        if hasattr(awaitable,'close'): awaitable.close()
        raise asyncio.TimeoutError()
    task=asyncio.ensure_future(awaitable)
    try:
        done,_=await asyncio.wait({task},timeout=remaining)
        if not done: raise asyncio.TimeoutError()
        return task.result()
    except BaseException:
        task.cancel()
        await _finish(asyncio.gather(task,return_exceptions=True))
        raise


async def receive(sock, *, consume=None):
    async def exact(count, eof=False):
        result=bytearray()
        while len(result)<count:
            part=await asyncio.get_running_loop().sock_recv(sock,count-len(result))
            if not part:
                if not result and eof:return None
                raise NativeChannelError('partial channel frame at EOF')
            if consume:consume(len(part))
            result.extend(part)
        return bytes(result)
    header=await exact(4,True)
    if header is None:return None
    count=struct.unpack('!I',header)[0]
    if not 1<=count<=MAX_FRAME:raise NativeChannelError('invalid channel frame size')
    return decode(await exact(count),MAX_FRAME)


def validate_bootstrap(bootstrap):
    keys(bootstrap,BOOTSTRAP_KEYS)
    if type(bootstrap['schema_version']) is not int or bootstrap['schema_version']!=2 or bootstrap['type']!='bootstrap':
        raise ValueError('channel2 bootstrap required')
    for name in ('nonce','collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256'):sha(bootstrap[name])
    integer(bootstrap['invocation_ordinal'],0,95);integer(bootstrap['deadline_ns'],1)
    paths=bootstrap['paths']
    if type(paths) is not dict or not REQUIRED<=set(paths) or set(paths)-REQUIRED-OPTIONAL:
        raise ValueError('exact declaration paths required')
    for value in paths.values():
        if type(value) is not str or not Path(value).is_absolute() or len(value.encode())>4096 or '\0' in value:
            raise ValueError('bounded absolute declaration path required')
    return bootstrap


def load_declarations(bootstrap):
    validate_bootstrap(bootstrap)
    result={};total=0
    for name,path in bootstrap['paths'].items():
        fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=MAX_DECLARATIONS-total:
                raise ValueError('bounded regular immutable declaration required')
            data=bytearray()
            while len(data)<=info.st_size:
                part=os.read(fd,min(65536,info.st_size+1-len(data)))
                if not part:break
                data.extend(part)
            if len(data)!=info.st_size:raise ValueError('declaration changed while reading')
            total+=len(data)
            result[name]=decode(bytes(data),MAX_DECLARATIONS)
        finally:os.close(fd)
    result=validate_recording_scope_protocol(result)
    context=run_context(result,bootstrap['nonce'],bootstrap['invocation_ordinal'])
    for name in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256'):
        if bootstrap[name]!=context[name]:raise ValueError('bootstrap declaration identity mismatch')
    return result


def _hello(bootstrap):
    return dict((key,'hello' if key=='type' else bootstrap[key]) for key in HELLO_KEYS)


def _decoded_batch(fields):
    value=fields['data_base64']
    if type(value) is not str or not 1<=len(value)<=((MAX_BATCH+2)//3)*4:
        raise ValueError('bounded encoded artifact batch required')
    data=base64.b64decode(value,validate=True)
    if not 1<=len(data)<=MAX_BATCH or base64.b64encode(data).decode('ascii')!=value:
        raise ValueError('canonical bounded artifact batch required')
    return data


def _read_owned(directory,name,maximum):
    from .recording_native_validation import open_original_directory
    dfd=open_original_directory(directory)
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=dfd)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0<=info.st_size<=maximum:
                raise ValueError('bounded owned artifact required')
            data=bytearray()
            while len(data)<=info.st_size:
                part=os.read(fd,min(65536,info.st_size+1-len(data)))
                if not part:break
                data.extend(part)
            if len(data)!=info.st_size:raise ValueError('owned artifact changed')
            return bytes(data)
        finally:os.close(fd)
    finally:os.close(dfd)


_RUN_INVENTORY_TOKEN=object()


class NativeRunInventory(_FrozenDict):
    """Detached immutable snapshot of the parent's own transport/reap state."""
    def __init__(self,value,*,_token=None):
        if _token is not _RUN_INVENTORY_TOKEN:raise TypeError('only the owning parent issues run inventory')
        dict.__init__(self,_freeze(value))


class ParentNativeChannel:
    def __init__(self,context,declarations,*,paths,output,deadline_ns,budget=None,retain_financial_documents=False,collection_declarations=None):
        self.declarations=_freeze(validate_recording_scope_protocol(declarations))
        from .recording_native_schema import preflight_recording_collection
        bundles=[self.declarations] if collection_declarations is None else collection_declarations
        self.preflight=preflight_recording_collection(bundles)
        if self.preflight['collection_sha256']!=digest(self.declarations['collection']) or not any(digest(v)==digest(self.declarations) for v in bundles):
            raise ValueError('parent preflight requires exact current original recording')
        if type(retain_financial_documents) is not bool or retain_financial_documents is not self.declarations['scope_protocol']['retain_financial_documents']:
            raise PermissionError('explicit financial retention must equal frozen policy')
        if context!=run_context(self.declarations,context['nonce'],context['invocation_ordinal']):
            raise ValueError('exact original run context required')
        self.context=_freeze(dict(context));self.output=Path(output).absolute()
        self.bootstrap=dict(schema_version=2,type='bootstrap',nonce=context['nonce'],invocation_ordinal=context['invocation_ordinal'],
            collection_sha256=context['collection_sha256'],scope_protocol_sha256=context['scope_protocol_sha256'],
            instrumentation_protocol_sha256=context['instrumentation_protocol_sha256'],deadline_ns=deadline_ns,
            paths=dict(paths))
        validate_bootstrap(self.bootstrap)
        if set(paths)!=set(self.declarations):raise ValueError('exact frozen snapshot path inventory required')
        self.budget=budget if budget is not None else ArtifactBudget()
        if type(self.budget) is not ArtifactBudget:raise TypeError('shared artifact budget required')
        self.budget.bind_collection(self.preflight['collection_sha256'],self.preflight['parent_budget_precharge'])
        self.parent,self.child=socket.socketpair();self.parent.setblocking(False);self.child.set_inheritable(False)
        self.hello=False;self.eof=False;self.reason=None;self.terminal=None;self.validated_index=None
        self.parent_reaped=False;self.returncode=None;self.sequence=0;self.received=0;self._process=None
        self._directory_fd=None;self._fd=None;self._source_fd=None;self._chunk=None;self._source=None
        self._chunks=[];self._pages=[];self._counters=[];self._sources={};self._receipts={};self._claims={}
        self._native_sequence=0;self._native_previous=digest(['gamst.recording.events.v1',self.context]);self._native_time=0
        self._native_entries=0;self._collect_task=None;self._collect_active=False;self._abort_task=None;self._closed=False
        self._threads=set();self._terminal_digest=None

    def prepare(self,environment,remaining_seconds=None):
        if FD_ENVIRONMENT_KEY in environment:raise ValueError('channel FD must be sole explicit transport delta')
        if remaining_seconds is not None:
            if type(remaining_seconds) not in (int,float) or not 0<remaining_seconds<=3600:
                raise ValueError('positive bounded remaining duration required')
            self.bootstrap['deadline_ns']=min(self.bootstrap['deadline_ns'],time.monotonic_ns()+int(remaining_seconds*1e9))
        return dict(env=dict(environment,**{FD_ENVIRONMENT_KEY:str(self.child.fileno())}),
            pass_fds=(self.child.fileno(),),close_fds=True)

    def spawned(self,process=None):
        if process is not None:
            if not isinstance(process,asyncio.subprocess.Process) or self._process is not None:
                raise ValueError('one actual owned spawned process required')
            self._process=process
        self.child.close()

    async def _thread(self,function,*args):
        task=asyncio.create_task(asyncio.to_thread(function,*args));self._threads.add(task)
        try:
            try:return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _finish(task);raise
        finally:self._threads.discard(task)

    def _create(self):
        self.output.mkdir(mode=0o700,parents=False,exist_ok=False);os.chmod(self.output,0o700)
        self._directory_fd=os.open(self.output,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)

    def _check_directory(self):
        a=os.stat(self.output,follow_symlinks=False);b=os.fstat(self._directory_fd)
        if (a.st_ino,a.st_dev)!=(b.st_ino,b.st_dev):raise ValueError('owned channel output replaced')

    def _consume(self,size):
        self.received+=size
        if self.received>MAX_WIRE_MULTIPLIER*self.budget.limit+MAX_WIRE_OVERHEAD:raise ValueError('channel wire amplification bound exceeded')

    def _save(self,name,data):
        self._check_directory();self.budget.reserve(len(data))
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=self._directory_fd)
        try:os.fchmod(fd,0o600);_write_all(fd,data);os.fsync(fd)
        finally:os.close(fd)
        return dict(path=name,sha256=_hash(data),bytes=len(data))

    def _new_chunk(self,ordinal):
        if ordinal!=len(self._chunks) or ordinal>=self.declarations['collection']['capacity']['max_chunks_per_run']:
            raise ValueError('contiguous bounded native chunk required')
        return dict(ordinal=ordinal,bytes=0,hash=hashlib.sha256(),prefix=hashlib.sha256(),line=bytearray(),
            header=False,footer=False,events=0,entries=0,first_entry=None,last_entry=None,
            sequence=self._native_sequence,previous=self._native_previous,time=self._native_time,
            global_entries=self._native_entries,valid_bytes=0,valid_hash=hashlib.sha256())

    def _chunk_data(self,fields):
        ordinal=integer(fields['chunk_ordinal'],0,1023);offset=integer(fields['offset'],0,MAX_CHUNK)
        data=_decoded_batch(fields)
        original=self._chunk or self._new_chunk(ordinal)
        if ordinal!=original['ordinal'] or offset!=original['bytes'] or offset+len(data)>MAX_CHUNK or original['footer']:
            raise ValueError('native chunk data order/size differs')
        c=dict(original)
        for name in ('hash','prefix','valid_hash'):c[name]=original[name].copy()
        c['line']=bytearray(original['line']);c['hash'].update(data);c['bytes']+=len(data)
        for byte in data:
            c['line'].append(byte)
            bound=4096 if not c['header'] else 2048
            if len(c['line'])>bound:raise ValueError('native line byte bound exceeded')
            if byte!=10:continue
            if c['footer']:raise ValueError('trailing line after native footer')
            line=bytes(c['line']);c['line'].clear();value=decode(line,bound);_native_numbers(value)
            if _line(value)!=line:raise ValueError('native artifact must use exact canonical bytes')
            if not c['header']:
                expected=dict(context=self.context,chunk_ordinal=ordinal,
                    previous_chunk_sha256=self._chunks[-1]['sha256'] if self._chunks else None,first_sequence=self._native_sequence)
                if value!=expected:raise ValueError('native chunk header binding differs')
                c['header']=True;c['prefix'].update(line)
            elif value.get('kind')=='chunk_footer':
                keys(value,('kind','last_sequence','event_count','entry_count','first_entry_ordinal','last_entry_ordinal','prefix_bytes','prefix_sha256'))
                if value!=dict(kind='chunk_footer',last_sequence=c['sequence']-1,event_count=c['events'],entry_count=c['entries'],
                        first_entry_ordinal=c['first_entry'],last_entry_ordinal=c['last_entry'],prefix_bytes=c['valid_bytes'],prefix_sha256=c['prefix'].hexdigest()):
                    raise ValueError('native footer inventory/hash differs')
                if not c['events']:raise ValueError('empty native chunk forbidden')
                c['footer']=True
            else:
                if c['footer']:raise ValueError('trailing data after native footer')
                keys(value,('sequence','monotonic_ns','event','fields','previous_sha256','sha256'))
                if (type(value['sequence']) is not int or value['sequence']!=c['sequence'] or value['previous_sha256']!=c['previous']
                        or value['sha256']!=digest({k:v for k,v in value.items() if k!='sha256'}) or value['event'] not in EVENTS
                        or type(value['fields']) is not dict):raise ValueError('native event sequence/hash/type differs')
                integer(value['monotonic_ns'],c['time']);c['time']=value['monotonic_ns']
                c['sequence']+=1;c['previous']=value['sha256'];c['events']+=1
                if c['events']>256 or c['sequence']>self.declarations['collection']['capacity']['max_events_per_run']:
                    raise ValueError('native event inventory bound exceeded')
                if value['event']=='entered':
                    entry=integer(value['fields'].get('entry_ordinal'),0,16383)
                    if entry!=c['global_entries']:raise ValueError('native entry ordinal differs')
                    c['global_entries']+=1;c['entries']+=1
                    if c['entries']>128:raise ValueError('native chunk entry bound exceeded')
                    if c['first_entry'] is None:c['first_entry']=entry
                    c['last_entry']=entry
                c['prefix'].update(line)
            c['valid_hash'].update(line);c['valid_bytes']+=len(line)
        if c['footer'] and c['line']:raise ValueError('bytes after native footer')
        self._check_directory();self.budget.reserve(len(data))
        if self._fd is None:
            self._fd=os.open(f'native-{ordinal:06d}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=self._directory_fd)
            os.fchmod(self._fd,0o600)
        try:_write_all(self._fd,data);os.fsync(self._fd)
        except BaseException:
            try:os.ftruncate(self._fd,offset);os.lseek(self._fd,offset,os.SEEK_SET)
            except OSError:pass
            raise
        self._chunk=c

    def _chunk_seal(self,fields):
        c=self._chunk
        if c is None or fields!=dict(chunk_ordinal=c['ordinal'],bytes=c['bytes'],sha256=c['hash'].hexdigest()) or not c['footer'] or c['line']:
            raise ValueError('complete exact native chunk seal required')
        os.fsync(self._fd);os.close(self._fd);self._fd=None
        ref=dict(ordinal=c['ordinal'],path=f"native-{c['ordinal']:06d}.jsonl",sha256=c['hash'].hexdigest(),bytes=c['bytes'],
            first_sequence=self._native_sequence,event_count=c['events'],entry_count=c['entries'])
        self._chunks.append(ref);self._native_sequence=c['sequence'];self._native_previous=c['previous']
        self._native_time=c['time'];self._native_entries=c['global_entries'];self._chunk=None

    def _metadata(self,kind,fields):
        if kind=='index_page':
            ordinal=integer(fields['page_ordinal'],0,7);page=fields['page']
            expected=dict(schema_version=1,kind='recording_native_index',nonce=self.context['nonce'],page_ordinal=ordinal,
                previous_page_sha256=self._pages[-1]['sha256'] if self._pages else None,chunks=self._chunks[ordinal*128:(ordinal+1)*128])
            if ordinal!=len(self._pages) or page!=expected or not expected['chunks'] or (self._pages and self._pages[-1]['count']!=128):
                raise ValueError('index page differs from received sealed chunks')
            data=_line(page)
            if len(data)>65536:raise ValueError('index page byte bound exceeded')
            ref=self._save(f'index-{ordinal:06d}.json',data);ref['count']=len(expected['chunks']);self._pages.append(ref)
        elif kind=='counter_page':
            ordinal=integer(fields['ordinal'],0,15);page=fields['page']
            keys(page,('schema_version','kind','nonce','ordinal','counters'))
            if ordinal!=len(self._counters) or page['schema_version']!=1 or type(page['schema_version']) is not int or page['kind']!='recording_counters' or page['nonce']!=self.context['nonce'] or page['ordinal']!=ordinal:
                raise ValueError('counter page identity/order differs')
            data=_line(page)
            if len(data)>65536:raise ValueError('counter page byte bound exceeded')
            ref=self._save(f'counters-{ordinal:06d}.json',data)
            self._counters.append(dict(ordinal=ordinal,sha256=ref['sha256'],bytes=ref['bytes']))
        elif kind=='receipt_attachment':
            receipt=fields['receipt'];keys(receipt,('schema_version','kind','origin','money_basis','provider','revision',
                'account_scope','invoice_id','source_document_sha256','source_document_bytes','lines'))
            data=_line(receipt);identity=sha(fields['receipt_sha256'])
            if len(data)>65536 or _hash(data)!=identity or identity in self._receipts:
                raise ValueError('bounded unique original declared receipt required')
            source=receipt['source_document_sha256'];sha(source)
            if source not in self._sources or receipt['source_document_bytes']!=self._sources[source]['bytes']:
                raise ValueError('receipt requires sealed exact source document')
            if (type(receipt['schema_version']) is not int or receipt['schema_version']!=1 or
                    receipt['kind']!='recording_declared_receipt' or receipt['origin'] not in ('application_declared','synthetic_fixture')
                    or receipt['money_basis']!='unknown' or type(receipt['lines']) is not list or not 1<=len(receipt['lines'])<=64):
                raise ValueError('bounded unknown-money declared receipt required')
            claims=[]
            for line in receipt['lines']:
                keys(line,('line_id','quantities','claim_sha256'));sha(line['claim_sha256'])
                if line['claim_sha256'] in claims:raise ValueError('duplicate declared claim')
                claims.append(line['claim_sha256'])
            self.budget.admit_attachment(len(data),(receipt['provider'],receipt['account_scope'],receipt['invoice_id']),
                (receipt['revision'],source),len(claims))
            self._save(f'receipt-{identity}.json',data);self._receipts[identity]=dict(source=source,claims=tuple(bytes.fromhex(h) for h in claims),owner=None)
        elif kind=='claims_page':
            page=fields['page'];keys(page,('schema_version','kind','receipt_sha256','owner_instance','claims'))
            data=_line(page);identity=sha(fields['page_sha256']);sha(page['owner_instance'])
            receipt=self._receipts.get(page['receipt_sha256'])
            if (len(data)>65536 or _hash(data)!=identity or identity in self._claims or receipt is None or receipt['owner'] is not None
                    or type(page['schema_version']) is not int or page['schema_version']!=1 or page['kind']!='recording_declared_claims'
                    or type(page['claims']) is not list or len(page['claims'])!=len(receipt['claims'])):
                raise ValueError('bounded sealed original declared claim page required')
            seen=set()
            for row in page['claims']:
                keys(row,('line_index','claim_sha256'));n=integer(row['line_index'],0,len(receipt['claims'])-1)
                if n in seen or bytes.fromhex(sha(row['claim_sha256']))!=receipt['claims'][n]:raise ValueError('duplicate/changed claim page line')
                seen.add(n)
            self.budget.admit_claims(value.hex() for value in receipt['claims'])
            self._save(f'claims-{identity}.json',data);receipt['owner']=page['owner_instance']
            self._claims[identity]=dict(receipt=page['receipt_sha256'],owner=page['owner_instance'])
        elif kind=='terminal':
            term=fields['terminal']
            if type(term) is not dict or len(_line(term))>65536 or term.get('context')!=self.context:
                raise ValueError('bounded original native terminal required')
            if self._source is not None:raise ValueError('unsealed financial source at native terminal')
            if self._chunk is not None:
                c=self._chunk
                expected_prefix=dict(chunk_ordinal=c['ordinal'],valid_bytes=c['valid_bytes'],valid_events=c['events'],sha256=c['valid_hash'].hexdigest())
                if term.get('state')!='incomplete' or term.get('prefix')!=expected_prefix or c['line'] or c['footer']:
                    raise ValueError('native failed prefix does not match received canonical prefix')
            elif term.get('prefix') is not None:
                expected_empty=dict(chunk_ordinal=len(self._chunks),valid_bytes=0,valid_events=0,sha256=_hash(b''))
                if term.get('state')!='incomplete' or term['prefix']!=expected_empty:
                    raise ValueError('native terminal claims an unreceived partial prefix')
                self._save(f'native-{len(self._chunks):06d}.jsonl',b'')
            if term.get('counter_pages')!=self._counters or term.get('chunk_count')!=len(self._chunks) or term.get('page_count')!=len(self._pages):
                raise ValueError('native terminal received inventory differs')
            self.terminal=term
        else:
            raise ValueError('unknown metadata frame')

    def _source_data(self,fields):
        identity=sha(fields['source_sha256']);offset=integer(fields['offset'],0,MAX_CHUNK);data=_decoded_batch(fields)
        if identity in self._sources:raise ValueError('duplicate sealed receipt source')
        if self._source is None:
            if offset:raise ValueError('receipt source must start at zero')
            self._check_directory()
            self._source_fd=os.open(f'receipt-source-{identity}.bin',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=self._directory_fd)
            os.fchmod(self._source_fd,0o600)
            self._source=dict(sha256=identity,bytes=0,hash=hashlib.sha256())
        source=self._source
        if identity!=source['sha256'] or offset!=source['bytes'] or offset+len(data)>MAX_CHUNK:
            raise ValueError('receipt source order/size differs')
        self._check_directory();self.budget.reserve(len(data))
        try:_write_all(self._source_fd,data);os.fsync(self._source_fd)
        except BaseException:
            try:os.ftruncate(self._source_fd,offset);os.lseek(self._source_fd,offset,os.SEEK_SET)
            except OSError:pass
            raise
        source['hash'].update(data);source['bytes']+=len(data)

    def _source_seal(self,fields):
        source=self._source
        if source is None or fields!=dict(source_sha256=source['sha256'],bytes=source['bytes']) or source['hash'].hexdigest()!=source['sha256']:
            raise ValueError('exact original source receipt seal required')
        self.budget.admit_source(source['bytes']);os.fsync(self._source_fd);os.close(self._source_fd);self._source_fd=None
        self._sources[source['sha256']]=dict(bytes=source['bytes']);self._source=None

    async def collect(self):
        if self._collect_task is not None:raise RuntimeError('channel collection already started')
        self._collect_task=asyncio.current_task();self._collect_active=True
        try:
            await self._thread(self._create)
            # Read parent snapshots too: caller-supplied dict cannot hide a changed
            # on-disk bootstrap declaration consumed by the child.
            loaded=await self._thread(load_declarations,self.bootstrap)
            if encoded(loaded)!=encoded(self.declarations):raise ValueError('parent frozen snapshot differs')
            limit=min(self.bootstrap['deadline_ns'],time.monotonic_ns()+self.declarations['instrumentation_protocol']['hello_timeout_ms']*1000000)
            await _deadline(asyncio.get_running_loop().sock_sendall(self.parent,frame(self.bootstrap)),limit)
            hello=await _deadline(receive(self.parent,consume=self._consume),limit)
            if hello!=_hello(self.bootstrap) or type(hello.get('schema_version')) is not int:
                raise ValueError('native hello identity mismatch')
            self.hello=True
            while True:
                value=await _deadline(receive(self.parent,consume=self._consume),self.bootstrap['deadline_ns'])
                if value is None:
                    self.eof=True
                    if self.terminal is None:raise ValueError('missing native terminal')
                    break
                if self.terminal is not None:raise ValueError('trailing channel frame after terminal')
                _native_numbers(value)
                keys(value,('schema_version','type','nonce','frame_sequence','fields'))
                kind=value['type']
                if (type(value['schema_version']) is not int or value['schema_version']!=2 or value['nonce']!=self.context['nonce']
                        or type(value['frame_sequence']) is not int or value['frame_sequence']!=self.sequence or kind not in FRAME_FIELDS
                        or self.sequence>=MAX_FRAMES):raise ValueError('channel frame identity/sequence differs')
                fields=value['fields'];keys(fields,FRAME_FIELDS[kind])
                if kind.startswith('receipt_') or kind=='claims_page':
                    if not self.declarations['scope_protocol']['retain_financial_documents']:
                        raise ValueError('financial document retention not consented')
                if kind=='chunk_data':await self._thread(self._chunk_data,fields)
                elif kind=='chunk_seal':await self._thread(self._chunk_seal,fields)
                elif kind=='receipt_source_data':await self._thread(self._source_data,fields)
                elif kind=='receipt_source_seal':await self._thread(self._source_seal,fields)
                else:await self._thread(self._metadata,kind,fields)
                self.sequence+=1
            ref=await self._thread(self._save,'terminal.json',_line(self.terminal));self._terminal_digest=ref['sha256']
            from .recording_native_validation import validate_recording_native_inventory
            self.validated_index=await self._thread(lambda:validate_recording_native_inventory(self.declarations,self.output,
                source_inventory=self.declarations['source_manifest']))
            if self.terminal['state']!='closed':self.reason='native_incomplete'
            return self.snapshot()
        except asyncio.CancelledError:
            self.reason='cancelled';raise
        except Exception as exc:
            self.reason='deadline_exceeded' if isinstance(exc,asyncio.TimeoutError) else 'invalid_channel'
            if isinstance(exc,asyncio.TimeoutError):raise
            raise NativeChannelError('invalid native recording channel') from exc
        finally:
            self.parent.close();self.child.close()
            try:await _finish(asyncio.create_task(self._close_files()))
            finally:self._collect_active=False

    async def _close_files(self):
        def close():
            if self._fd is not None:os.close(self._fd);self._fd=None
            if self._source_fd is not None:os.close(self._source_fd);self._source_fd=None
            if self._directory_fd is not None:os.close(self._directory_fd);self._directory_fd=None
        await self._thread(close)

    def close(self):
        self._closed=True;self.parent.close();self.child.close()

    async def abort(self):
        if self._abort_task is None:self._abort_task=asyncio.create_task(self._abort())
        try:return await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:await _finish(self._abort_task);raise

    async def _abort(self):
        self.close()
        if self._collect_active and self._collect_task is not None and not self._collect_task.done() and self._collect_task is not asyncio.current_task():
            self._collect_task.cancel();await _finish(asyncio.gather(self._collect_task,return_exceptions=True))
        await self._close_files()

    async def record_reaped(self,process):
        if not isinstance(process,asyncio.subprocess.Process) or process is not self._process or process.returncode is None:
            raise ValueError('same actual completed owned subprocess required')
        self.returncode=await process.wait();self.parent_reaped=True

    def snapshot(self):
        prefix=None
        if self._chunk is not None:
            c=self._chunk;prefix=dict(chunk_ordinal=c['ordinal'],valid_bytes=c['valid_bytes'],valid_events=c['events'],sha256=c['valid_hash'].hexdigest())
        return NativeRunInventory(_freeze(dict(context=dict(self.context),terminal_sha256=self._terminal_digest,
            validated_prefix=dict(chunks=len(self._chunks),events=self._native_sequence,partial=prefix),
            parent_reaped=self.parent_reaped,returncode=self.returncode,failure_category=self.reason,
            hello_verified=self.hello,eof=self.eof,actual_acceptance='incomplete')),_token=_RUN_INVENTORY_TOKEN)


class ChildNativeChannel:
    def __init__(self,sock,bootstrap,declarations):
        self.socket=sock;self.socket.setblocking(False);self.socket.set_inheritable(False)
        self.bootstrap=bootstrap;self.declarations=declarations;self.context=run_context(declarations,bootstrap['nonce'],bootstrap['invocation_ordinal'])
        self._loop=asyncio.get_running_loop();self._lock=threading.RLock();self._queue=deque()
        self._wake=asyncio.Event();self._space=asyncio.Event();self._space.set()
        self._pending_frames=self._pending_bytes=self._sequence=0;self._failure=None
        self._stopping=False;self._revoked=False;self._terminal=False
        self._writer=asyncio.create_task(self._write());self._close_task=self._abort_task=None
        self._follow_task=None;self._native_writer=None;self._sent_chunks=0;self._threads=set()
        self._sent_sources=set();self._sent_receipts=set();self._sent_claims=set()

    @classmethod
    async def from_environment(cls):
        value=os.environ.get(FD_ENVIRONMENT_KEY)
        if type(value) is not str or not value.isascii() or not value.isdecimal() or value!=str(int(value)) or int(value)<3:
            raise ValueError('explicit inherited native socket FD required')
        fd=int(value);os.set_inheritable(fd,False);sock=socket.socket(fileno=fd);sock.setblocking(False)
        try:
            bootstrap=validate_bootstrap(await _deadline(receive(sock),time.monotonic_ns()+10_000_000_000))
            task=asyncio.create_task(asyncio.to_thread(load_declarations,bootstrap))
            try:declarations=await asyncio.shield(task)
            except asyncio.CancelledError:await _finish(task);raise
            limit=min(bootstrap['deadline_ns'],time.monotonic_ns()+declarations['instrumentation_protocol']['hello_timeout_ms']*1000000)
            await _deadline(asyncio.get_running_loop().sock_sendall(sock,frame(_hello(bootstrap))),limit)
            return cls(sock,bootstrap,declarations)
        except BaseException:sock.close();raise

    def _packet(self,kind,fields):
        if kind not in FRAME_FIELDS:raise ValueError('known native channel2 frame required')
        keys(fields,FRAME_FIELDS[kind]);_native_numbers(fields)
        if kind.startswith('receipt_') or kind=='claims_page':
            if not self.declarations['scope_protocol']['retain_financial_documents']:
                raise PermissionError('financial retention not consented')
        return frame(dict(schema_version=2,type=kind,nonce=self.context['nonce'],frame_sequence=self._sequence,fields=fields))

    def _enqueue(self,kind,fields,wait=False):
        with self._lock:
            if self._revoked or self._failure or self._stopping or self._terminal:raise NativeChannelError('native child channel admission closed')
            if self._sequence>=MAX_FRAMES:raise NativeChannelError('native frame inventory exhausted')
            data=self._packet(kind,fields)
            if self._pending_frames>=64 or self._pending_bytes+len(data)>1024*1024:
                if wait:self._space.clear();return False
                raise NativeChannelError('native channel queue capacity exhausted')
            self._queue.append(data);self._pending_frames+=1;self._pending_bytes+=len(data);self._sequence+=1
            if kind=='terminal':self._terminal=True
            self._loop.call_soon_threadsafe(self._wake.set)
            return True

    def send(self,kind,fields):
        self._enqueue(kind,fields)

    async def _send_wait(self,kind,fields):
        while not self._enqueue(kind,fields,True):
            await _deadline(self._space.wait(),self.bootstrap['deadline_ns'])

    async def _write(self):
        try:
            while True:
                with self._lock:
                    if self._revoked or self.socket.fileno()<0:return
                    if not self._queue:
                        if self._stopping:return
                        self._wake.clear();data=None
                    else:data=self._queue.popleft()
                if data is None:await self._wake.wait();continue
                await _deadline(self._loop.sock_sendall(self.socket,data),self.bootstrap['deadline_ns'])
                with self._lock:
                    self._pending_frames-=1;self._pending_bytes-=len(data);self._space.set()
        except BaseException:
            self._failure='native_channel_write_failed';self._space.set()
            raise

    async def drain(self):
        while self._pending_frames:
            if self._failure:raise NativeChannelError(self._failure)
            if self._writer.done():raise NativeChannelError('native channel writer stopped before drain')
            self._space.clear()
            if not self._pending_frames:return
            await _deadline(self._space.wait(),self.bootstrap['deadline_ns'])

    async def _read(self,directory,name,maximum):
        task=asyncio.create_task(asyncio.to_thread(_read_owned,directory,name,maximum));self._threads.add(task)
        try:
            try:return await asyncio.shield(task)
            except asyncio.CancelledError:await _finish(task);raise
        finally:self._threads.discard(task)

    async def send_chunk(self,directory,ref):
        data=await self._read(directory,ref['path'],MAX_CHUNK)
        if len(data)!=ref['bytes'] or _hash(data)!=ref['sha256']:raise ValueError('sealed child chunk changed')
        import io
        for line in io.BytesIO(data):
            value=decode(line,4096)
            if value.get('event')=='settlement_ref':
                await self._send_declared_receipt(directory,value['fields'])
        for offset in range(0,len(data),MAX_BATCH):
            await self._send_wait('chunk_data',dict(chunk_ordinal=ref['ordinal'],offset=offset,
                data_base64=base64.b64encode(data[offset:offset+MAX_BATCH]).decode('ascii')))
        await self._send_wait('chunk_seal',dict(chunk_ordinal=ref['ordinal'],bytes=len(data),sha256=ref['sha256']))

    async def _send_declared_receipt(self,directory,reference):
        claim_id=sha(reference['claims_page_sha256']);receipt_id=sha(reference['receipt_sha256'])
        if claim_id in self._sent_claims:return
        claim_data=await self._read(directory,f'claims-{claim_id}.json',65536)
        if _hash(claim_data)!=claim_id:raise ValueError('child declared claims changed')
        claim=decode(claim_data,65536)
        if claim['receipt_sha256']!=receipt_id:raise ValueError('child claim receipt identity differs')
        if receipt_id not in self._sent_receipts:
            receipt_data=await self._read(directory,f'receipt-{receipt_id}.json',65536)
            if _hash(receipt_data)!=receipt_id:raise ValueError('child declared receipt changed')
            receipt=decode(receipt_data,65536);source_id=sha(receipt['source_document_sha256'])
            if source_id not in self._sent_sources:
                source=await self._read(directory,f'receipt-source-{source_id}.bin',MAX_CHUNK)
                if _hash(source)!=source_id or len(source)!=receipt['source_document_bytes']:
                    raise ValueError('child original receipt source changed')
                for offset in range(0,len(source),MAX_BATCH):
                    await self._send_wait('receipt_source_data',dict(source_sha256=source_id,offset=offset,
                        data_base64=base64.b64encode(source[offset:offset+MAX_BATCH]).decode('ascii')))
                await self._send_wait('receipt_source_seal',dict(source_sha256=source_id,bytes=len(source)))
                self._sent_sources.add(source_id)
            await self._send_wait('receipt_attachment',dict(receipt_sha256=receipt_id,receipt=receipt));self._sent_receipts.add(receipt_id)
        await self._send_wait('claims_page',dict(page_sha256=claim_id,page=claim));self._sent_claims.add(claim_id)

    def follow_writer(self,writer):
        if self._follow_task is not None or writer.context!=self.context:raise ValueError('one original native writer required')
        self._native_writer=writer;self._follow_task=asyncio.create_task(self._follow(writer));return self._follow_task

    async def _follow(self,writer):
        while True:
            refs,done=await writer.wait_sealed_chunks(self._sent_chunks)
            for ref in refs:
                await self.send_chunk(writer.output,ref);self._sent_chunks+=1
            if done:return

    async def export_run(self,directory):
        terminal=decode(await self._read(directory,'terminal.json',65536),65536)
        if terminal.get('context')!=self.context:raise ValueError('child terminal belongs to another native run')
        if self._follow_task is not None:await self._follow_task
        for ordinal in range(integer(terminal['page_count'],0,8)):
            page=decode(await self._read(directory,f'index-{ordinal:06d}.json',65536),65536)
            for ref in page['chunks']:
                if ref['ordinal']>=self._sent_chunks:
                    await self.send_chunk(directory,ref);self._sent_chunks+=1
            await self._send_wait('index_page',dict(page_ordinal=ordinal,page=page))
        if terminal.get('prefix') is not None:
            prefix=terminal['prefix'];ordinal=integer(prefix['chunk_ordinal'],0,1023)
            data=await self._read(directory,f'native-{ordinal:06d}.jsonl',MAX_CHUNK)
            count=integer(prefix['valid_bytes'],0,MAX_CHUNK);data=data[:count]
            if len(data)!=count or _hash(data)!=prefix['sha256']:
                raise ValueError('child incomplete prefix changed')
            for offset in range(0,len(data),MAX_BATCH):
                await self._send_wait('chunk_data',dict(chunk_ordinal=ordinal,offset=offset,
                    data_base64=base64.b64encode(data[offset:offset+MAX_BATCH]).decode('ascii')))
        for ref in terminal['counter_pages']:
            ordinal=ref['ordinal'];data=await self._read(directory,f'counters-{ordinal:06d}.json',65536)
            if _hash(data)!=ref['sha256'] or len(data)!=ref['bytes']:raise ValueError('child counter page changed')
            await self._send_wait('counter_page',dict(ordinal=ordinal,page=decode(data,65536)))
        return terminal

    async def close(self,terminal=None):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close(terminal))
        try:return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise

    async def _close(self,terminal):
        limit=min(self.bootstrap['deadline_ns'],time.monotonic_ns()+self.declarations['instrumentation_protocol']['close_timeout_ms']*1000000)
        try:
            if self._native_writer is not None:
                terminal=await _deadline(self.export_run(self._native_writer.output),limit)
            if terminal is None:raise ValueError('immutable native terminal required')
            await self._send_wait('terminal',dict(terminal=terminal))
            with self._lock:self._stopping=True;self._wake.set()
            await _deadline(asyncio.shield(self._writer),limit)
            self.socket.shutdown(socket.SHUT_WR)
        finally:
            self._revoked=True;self.socket.close()
            self._writer.cancel();await _finish(asyncio.gather(self._writer,return_exceptions=True))
            if self._follow_task is not None and not self._follow_task.done():
                self._follow_task.cancel();await _finish(asyncio.gather(self._follow_task,return_exceptions=True))

    async def abort(self):
        if self._abort_task is None:self._abort_task=asyncio.create_task(self._abort())
        try:await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:await _finish(self._abort_task);raise

    async def _abort(self):
        self._revoked=True;self.socket.close();self._wake.set();self._space.set()
        jobs=[self._writer]
        if self._follow_task is not None:jobs.append(self._follow_task)
        if self._close_task is not None:jobs.append(self._close_task)
        for task in jobs:
            if not task.done():task.cancel()
        await _finish(asyncio.gather(*jobs,return_exceptions=True))
