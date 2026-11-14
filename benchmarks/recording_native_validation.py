"""Bounded native4 structural validation; never a monetary or quality gate."""
from array import array
from collections import Counter
import hashlib
import os
from pathlib import Path
import stat
import struct

from .recording_native_schema import (decode,encoded,digest,keys,integer,sha,
    validate_recording_scope_protocol,run_context,REASONS)

ENTRY=struct.Struct('<IIIIQQQQ32s32s16x')
_INDEX_TOKEN=object()


from .recording_native_schema import _FrozenDict, _FrozenList, _freeze


class ValidatedRecordingIndex:
    """Packed verified inventory; source bodies never become mutable native rows."""
    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('validated original inventory is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,directory,declarations,terminal,chunk_refs,entries,flags,count,hashes,settlements,*,_token=None):
        if _token is not _INDEX_TOKEN:raise TypeError('only the original artifact reader issues validated indices')
        self.directory=Path(directory);self.declarations=_freeze(declarations);self.terminal=_freeze(terminal)
        self.chunk_refs=tuple(_freeze(v) for v in chunk_refs);self.entries=bytes(entries);self.flags=bytes(flags);self.count=count;self.hashes=bytes(hashes)
        self.settlements=bytes(settlements)
        self.actual_acceptance='incomplete';self.money_verified=False


def open_original_directory(directory):
    """Walk every directory component without symlinks, never blocking on FIFOs."""
    path=os.path.abspath(os.fspath(directory));parts=Path(path).parts[1:]
    if len(path)>4096 or len(parts)>128:raise ValueError('bounded original directory path required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NONBLOCK)
    try:
        for part in parts:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            os.close(fd);fd=child
        result=fd;fd=None;return result
    finally:
        if fd is not None:os.close(fd)


def _read(directory,name,maximum,expected=None):
    if '/' in name or name in ('.','..'):raise ValueError('owned fixed artifact filename required')
    dfd=open_original_directory(directory)
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=dfd)
        try:
            st=os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or not 0<=st.st_size<=maximum:raise ValueError('bounded regular artifact required')
            data=b''
            while len(data)<=maximum:
                part=os.read(fd,min(65536,maximum+1-len(data)))
                if not part:break
                data+=part
            if len(data)!=st.st_size:raise ValueError('artifact changed or exceeded byte bound')
        finally:os.close(fd)
    finally:os.close(dfd)
    if expected is not None and hashlib.sha256(data).hexdigest()!=sha(expected):raise ValueError('artifact hash mismatch')
    return data


def _json_file(directory,name,maximum,expected=None):
    data=_read(directory,name,maximum,expected);value=decode(data,maximum)
    if encoded(value)+b'\n'!=data:raise ValueError('exact canonical immutable artifact bytes required')
    return value,data


def _chunk_data(directory,ref):
    if not ref.get('partial'):
        return _read(directory,ref['path'],1024**2,ref['sha256'])
    data=_read(directory,ref['path'],1024**2)
    if len(data)<ref['bytes']:raise ValueError('failed prefix truncated')
    data=data[:ref['bytes']]
    if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('failed prefix hash differs')
    return data


def _chunks(directory,refs,context):
    previous=None;sequence=0
    for ordinal,ref in enumerate(refs):
        data=_chunk_data(directory,ref)
        if len(data)!=ref['bytes']:raise ValueError('chunk byte inventory differs')
        lines=data.splitlines(keepends=True)
        if ref.get('partial'):
            if not 1<=len(lines)<=257:raise ValueError('bounded failed chunk prefix required')
            header=decode(lines[0],4096)
            if encoded(header)+b'\n'!=lines[0] or encoded(header)!=encoded(dict(context=context,chunk_ordinal=ordinal,previous_chunk_sha256=previous,first_sequence=sequence)):
                raise ValueError('original failed prefix header differs')
            if len(lines)-1!=ref['event_count']:raise ValueError('failed prefix event inventory differs')
            for line in lines[1:]:
                item=decode(line,2048)
                if encoded(item)+b'\n'!=line:raise ValueError('canonical complete failed prefix line required')
                yield ordinal,item
            if ordinal!=len(refs)-1:raise ValueError('failed prefix must be last')
            continue
        if not 3<=len(lines)<=258:raise ValueError('bounded nonempty chunk framing required')
        header=decode(lines[0],4096);footer=decode(lines[-1],2048)
        if encoded(header)+b'\n'!=lines[0] or encoded(footer)+b'\n'!=lines[-1]:raise ValueError('canonical chunk framing required')
        keys(header,('context','chunk_ordinal','previous_chunk_sha256','first_sequence'))
        if encoded(header)!=encoded(dict(context=context,chunk_ordinal=ordinal,previous_chunk_sha256=previous,first_sequence=sequence)):
            raise ValueError('original chunk context/order/predecessor mismatch')
        keys(footer,('kind','last_sequence','event_count','entry_count','first_entry_ordinal','last_entry_ordinal','prefix_bytes','prefix_sha256'))
        for name in ('last_sequence','event_count','entry_count','prefix_bytes'):integer(footer[name])
        for name in ('first_entry_ordinal','last_entry_ordinal'):
            if footer[name] is not None:integer(footer[name])
        prefix=b''.join(lines[:-1]);n=len(lines)-2
        if (footer['kind']!='chunk_footer' or footer['event_count']!=n or footer['last_sequence']!=sequence+n-1
                or footer['prefix_bytes']!=len(prefix) or footer['prefix_sha256']!=hashlib.sha256(prefix).hexdigest()):
            raise ValueError('chunk footer prefix/count differs')
        found=[]
        for line in lines[1:-1]:
            item=decode(line,2048)
            if encoded(item)+b'\n'!=line:raise ValueError('canonical event line required')
            if item.get('event')=='entered':found.append(item['fields']['entry_ordinal'])
            yield ordinal,item
        if (len(found)>128 or footer['entry_count']!=len(found) or ref['entry_count']!=len(found)
                or footer['first_entry_ordinal']!=(found[0] if found else None)
                or footer['last_entry_ordinal']!=(found[-1] if found else None)
                or ref['event_count']!=n or ref['first_sequence']!=sequence):
            raise ValueError('chunk entry/index inventory differs')
        sequence+=n;previous=ref['sha256']


def validate_recording_native_inventory(declarations,index,terminal=None,*,source_inventory=None):
    """Read a final owned artifact directory using the approved declaration bundle.

    `index` is its directory path, not a caller-supplied parsed substitute. The
    optional terminal argument is an expected file digest. `source_inventory`
    must retain the exact original manifest; mapping remains separately observed.
    """
    d=validate_recording_scope_protocol(declarations);directory=Path(index)
    term,_=_json_file(directory,'terminal.json',65536,terminal)
    required=('schema_version','kind','context','state','mapping_sha256','index_tail_sha256','page_count',
        'chunk_count','event_count','entry_count','reservation_count','released_count','outstanding_tickets',
        'outstanding_entries','outstanding_work','registered_workers','unclosed_workers','unsettled_entries',
        'prefix','counter_pages','incomplete_counts','child_cleanup_finished','actual_acceptance',
        'receipt_authenticity_verified','worker_backend_coverage_verified')
    keys(term,required)
    if term['schema_version']!=4 or type(term['schema_version']) is not int or term['kind']!='recording_native_terminal':
        raise ValueError('original native4 terminal required')
    ctx=term['context'];keys(ctx,('schema_version','kind','nonce','invocation_ordinal','collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256'))
    if ctx!=run_context(d,ctx['nonce'],ctx['invocation_ordinal']):raise ValueError('native context differs from frozen declarations')
    if source_inventory is None or digest(source_inventory)!=digest(d['source_manifest']):raise ValueError('exact original source inventory required')
    if term['actual_acceptance']!='incomplete' or term['receipt_authenticity_verified'] is not False or term['worker_backend_coverage_verified'] is not False:
        raise ValueError('native structural evidence cannot certify actual truth')
    if term['state'] not in ('closed','incomplete') or type(term['child_cleanup_finished']) is not bool:raise ValueError('explicit native terminal state required')
    caps=d['collection']['capacity'];pages=integer(term['page_count'],0,8);chunks=integer(term['chunk_count'],0,caps['max_chunks_per_run'])
    if pages!=(chunks+127)//128:raise ValueError('minimal bounded nonempty index pages required')
    refs=[];previous=None
    for n in range(pages):
        page,data=_json_file(directory,'index-%06d.json'%n,65536)
        keys(page,('schema_version','kind','nonce','page_ordinal','previous_page_sha256','chunks'))
        if (page['schema_version']!=1 or type(page['schema_version']) is not int or page['kind']!='recording_native_index'
                or page['nonce']!=ctx['nonce'] or page['page_ordinal']!=n or page['previous_page_sha256']!=previous):
            raise ValueError('native index identity/order differs')
        integer(page['page_ordinal'],0,7)
        expected=128 if n<pages-1 else chunks-128*n
        if type(page['chunks']) is not list or len(page['chunks'])!=expected:raise ValueError('complete bounded chunk page required')
        for ref in page['chunks']:
            keys(ref,('ordinal','path','sha256','bytes','first_sequence','event_count','entry_count'))
            ordinal=len(refs);integer(ref['ordinal']);integer(ref['first_sequence'])
            if ref['ordinal']!=ordinal or ref['path']!='native-%06d.jsonl'%ordinal:raise ValueError('fixed original chunk locator required')
            sha(ref['sha256']);integer(ref['bytes'],1,1024**2);integer(ref['event_count'],1,256);integer(ref['entry_count'],0,128)
            refs.append(ref)
        previous=hashlib.sha256(data).hexdigest()
    if term['index_tail_sha256']!=previous:raise ValueError('native terminal index inventory differs')
    if term['prefix'] is not None:
        prefix=term['prefix'];keys(prefix,('chunk_ordinal','valid_bytes','valid_events','sha256'))
        if term['state']!='incomplete' or integer(prefix['chunk_ordinal'])!=chunks or chunks>=caps['max_chunks_per_run']:
            raise ValueError('failed prefix must follow sealed original chunks')
        refs.append(dict(ordinal=chunks,path='native-%06d.jsonl'%chunks,sha256=sha(prefix['sha256']),
            bytes=integer(prefix['valid_bytes'],1,1024**2),event_count=integer(prefix['valid_events'],0,256),partial=True))
    count=integer(term['entry_count'],0,caps['max_entries_per_run'])
    entries=bytearray(count*ENTRY.size);flags=bytearray(count);hashes=bytearray(count*96);settlements=bytearray(count*64)
    ticket_state=bytearray(caps['max_reservations_per_run']);ticket_entry=array('i',[-1])*len(ticket_state)
    pending={};workers={};work={};instances={};counters=Counter();reservation=entered=released=finished=published=events=0
    last=digest(['gamst.recording.events.v1',ctx]);last_time=0;mapping=None;active_entries=0;next_work=0;evidence_refs=0
    financial=dict(claims=set(),documents={},files=set(),receipts=set(),sources=set(),source_bytes=0,receipt_bytes=0)
    bindings=next(a['provider_bindings'] for a in d['collection']['arms'] if a['arm']==d['collection']['invocations'][ctx['invocation_ordinal']]['arm'])
    scopes=[s for s in d['scope_protocol']['scopes'] if s['invocation_ordinal']==ctx['invocation_ordinal']]
    for chunk,item in _chunks(directory,refs,ctx):
        keys(item,('sequence','monotonic_ns','event','fields','previous_sha256','sha256'))
        if integer(item['sequence'])!=events or item['previous_sha256']!=last:raise ValueError('native event order/hash chain differs')
        now=integer(item['monotonic_ns'],last_time)
        if sha(item['sha256'])!=digest({k:v for k,v in item.items() if k!='sha256'}):raise ValueError('native event body digest differs')
        last=item['sha256'];last_time=now;events+=1;f=item['fields'];kind=item['event']
        if kind=='mapping_bound':
            keys(f,('mapping_sha256',));sha(f['mapping_sha256'])
            if mapping is not None:raise ValueError('mapping rebound')
            mapping=f['mapping_sha256']
        elif kind=='reserved':
            keys(f,('ticket','scope','binding','operation_kind','phase','relationship','predecessor','request_sha256','evidence_sha256','prompt_sha256'))
            if mapping is None or integer(f['ticket'])!=reservation or reservation>=len(ticket_state):raise ValueError('original ordered reservation required')
            si=integer(f['scope'],0,len(scopes)-1);bi=integer(f['binding'],0,len(bindings)-1)
            if f['operation_kind'] not in bindings[bi]['operation_kinds'] or f['phase'] not in ('cold','warm','unknown') or f['relationship'] not in ('root','retry','escalation'):
                raise ValueError('frozen native operation kind/phase/relationship required')
            for name in ('request_sha256','evidence_sha256'):sha(f[name])
            if f['prompt_sha256'] is not None:sha(f['prompt_sha256'])
            if f['relationship']=='root' and f['predecessor'] is not None:raise ValueError('root has no predecessor')
            if f['relationship']!='root':
                prior=instances.get(sha(f['predecessor']))
                if prior is None:raise ValueError('retry requires a strictly earlier original instance')
                row=ENTRY.unpack_from(entries,prior*ENTRY.size)
                if row[1]!=si or hashes[prior*96+64:(prior+1)*96].hex()!=f['evidence_sha256']:
                    raise ValueError('retry/escalation crosses original scope or evidence')
            counters[si,bi,'reserved']+=1
            pending[reservation]=dict(f,refs=[]);ticket_state[reservation]=1;reservation+=1
        elif kind=='evidence_ref':
            keys(f,('ticket','reference'));t=integer(f['ticket'],0,reservation-1)
            if ticket_state[t]!=1 or len(pending[t]['refs'])>=16:raise ValueError('bounded pre-dispatch evidence required')
            ref=f['reference']
            if ref.get('kind')=='capture':
                keys(ref,('kind','source_index','capture_ordinal','record_sha256','mapping_sha256'))
                source=d['source_manifest'];si=integer(ref['source_index'],0,len(source['sources'])-1);ri=integer(ref['capture_ordinal'],0,len(source['records'])-1)
                if (ref['mapping_sha256']!=mapping or ref['record_sha256']!=digest(source['records'][ri]) or source['records'][ri]['source_id']!=source['sources'][si]['source_id']
                        or source['records'][ri]['source_id'] not in bindings[pending[t]['binding']]['source_ids']):
                    raise ValueError('original capture evidence differs')
            elif ref.get('kind')=='result':
                keys(ref,('kind','entry_ordinal','result_sha256'));ei=integer(ref['entry_ordinal'],0,entered-1)
                if not flags[ei]&1 or ENTRY.unpack_from(entries,ei*ENTRY.size)[8].hex()!=sha(ref['result_sha256']):raise ValueError('prior raw result evidence differs')
            else:raise ValueError('unknown evidence locator')
            pending[t]['refs'].append(ref);evidence_refs+=1
            if evidence_refs>caps['max_collection_evidence_refs']:raise ValueError('original evidence reference capacity exceeded')
        elif kind=='dispatched':
            keys(f,('ticket',));t=integer(f['ticket'],0,reservation-1)
            if ticket_state[t]!=1:raise ValueError('reservation dispatched twice or after release')
            p=pending[t]
            if p['operation_kind']!='semantic':
                if not p['refs'] or any(r['kind']!='capture' for r in p['refs']) or p['request_sha256']!=digest(dict(binding=p['binding'],operation_kind=p['operation_kind'],capture_refs=p['refs'])):
                    raise ValueError('original source request descriptor differs')
            counters[p['scope'],p['binding'],'dispatched']+=1
            ticket_state[t]=2
        elif kind=='entered':
            keys(f,('ticket','entry_ordinal','instance','worker_ordinal'));t=integer(f['ticket'],0,reservation-1)
            if ticket_state[t]!=2 or integer(f['entry_ordinal'])!=entered or entered>=count:raise ValueError('unique ordered actual entry required')
            if f['instance']!=digest(['gamst.recording.instance.v1',ctx['nonce'],entered]):raise ValueError('canonical instance rebound')
            w=workers.get(integer(f['worker_ordinal'],0,1023));p=pending[t]
            if w is None or not w['ready'] or w['closed'] or w['binding']!=p['binding']:raise ValueError('entry worker lifetime differs')
            ENTRY.pack_into(entries,entered*ENTRY.size,t,p['scope'],p['binding'],f['worker_ordinal'],item['sequence'],now,0,chunk,b'\0'*32,bytes.fromhex(item['sha256']))
            hashes[entered*96:(entered+1)*96]=bytes.fromhex(p['request_sha256'])+(b'\0'*32 if p['prompt_sha256'] is None else bytes.fromhex(p['prompt_sha256']))+bytes.fromhex(p['evidence_sha256'])
            instances[f['instance']]=entered
            counters[p['scope'],p['binding'],'entered']+=1;ticket_state[t]=3;ticket_entry[t]=entered;entered+=1;active_entries+=1
            # Header bodies are discarded immediately; terminal joins use packed records.
            pending.pop(t)
        elif kind=='released':
            keys(f,('ticket','reason'));t=integer(f['ticket'],0,reservation-1)
            if ticket_state[t] not in (1,2) or f['reason'] not in ('admission_refused','cancelled_before_entry','observer_failed_before_entry','revoked_before_entry'):
                raise ValueError('unique no-entry release required')
            p=pending[t];counters[p['scope'],p['binding'],'released']+=1
            counters[p['scope'],p['binding'],'released_dispatched' if ticket_state[t]==2 else 'released_undispatched']+=1
            ticket_state[t]=4;pending.pop(t);released+=1
        elif kind=='finished':
            keys(f,('entry_ordinal','outcome','result_sha256'));ei=integer(f['entry_ordinal'],0,entered-1)
            if flags[ei]&1 or f['outcome'] not in ('ok','error','cancelled','timeout'):raise ValueError('unique callback terminal required')
            row=list(ENTRY.unpack_from(entries,ei*ENTRY.size));row[6]=now;row[8]=b'\0'*32 if f['result_sha256'] is None else bytes.fromhex(sha(f['result_sha256']))
            ENTRY.pack_into(entries,ei*ENTRY.size,*row);flags[ei]|=1+4*('ok','error','cancelled','timeout').index(f['outcome']);finished+=1;counters[row[1],row[2],'finished']+=1
        elif kind=='published':
            keys(f,('entry_ordinal','outcome'));ei=integer(f['entry_ordinal'],0,entered-1)
            if not flags[ei]&1 or flags[ei]&2 or f['outcome'] not in ('accepted','discarded','not_applicable'):raise ValueError('unique post-callback publication required')
            flags[ei]|=2+16*('accepted','discarded','not_applicable').index(f['outcome']);active_entries-=1;row=ENTRY.unpack_from(entries,ei*ENTRY.size);published+=1;counters[row[1],row[2],'published']+=1
        elif kind=='worker_registered':
            keys(f,('worker_ordinal','binding','kind'));wi=integer(f['worker_ordinal'],0,1023);bi=integer(f['binding'],0,len(bindings)-1)
            if wi!=len(workers) or f['kind'] not in ('process','in_process','callback'):raise ValueError('unique bounded actual worker lifetime required')
            if f['kind']!={'async_client_callback':'callback','in_process_backend':'in_process','process_adapter':'process'}[bindings[bi]['entry_boundary']]:
                raise ValueError('registered worker kind differs from frozen entry boundary')
            workers[wi]=dict(binding=bi,kind=f['kind'],ready=False,closed=False,marks=set())
        elif kind=='worker_milestone':
            keys(f,('worker_ordinal','milestone'));w=workers.get(integer(f['worker_ordinal'],0,1023));m=f['milestone']
            if w is None or m in w['marks']:raise ValueError('unique registered worker milestone required')
            allowed={'process':{'spawn_started','handle_acquired','ready','startup_failed','reap_confirmed'},'callback':{'callback_attached','ready','callback_detached'},'in_process':{'ready','startup_failed','provider_closed','thread_joined'}}[w['kind']]
            if m not in allowed or w['closed']:raise ValueError('worker milestone claims unsupported ownership')
            if m=='handle_acquired' and 'spawn_started' not in w['marks']:raise ValueError('handle before spawn')
            if m=='reap_confirmed' and 'handle_acquired' not in w['marks']:raise ValueError('reap without acquired process')
            if m=='ready' and ('startup_failed' in w['marks'] or w['kind']=='process' and 'handle_acquired' not in w['marks'] or w['kind']=='callback' and 'callback_attached' not in w['marks']):raise ValueError('worker readiness lacks actual lifetime')
            w['marks'].add(m)
            if m=='ready':w['ready']=True
            if m in ('provider_closed','callback_detached','reap_confirmed') or m=='startup_failed' and 'handle_acquired' not in w['marks']:w['closed']=True
        elif kind=='work_started':
            keys(f,('work_ordinal','worker_ordinal','scope','parent_entry_ordinal','binding','kind'));wi=integer(f['worker_ordinal'],0,1023);w=workers.get(wi)
            integer(f['scope'],0,0);integer(f['binding'],0,len(bindings)-1)
            ordinal=integer(f['work_ordinal'])
            if ordinal!=next_work or w is None or w['binding']!=f['binding'] or f['scope']!=0 or f['kind'] not in ('startup','reset','residency','idle','close','transport'):raise ValueError('original worker interval ownership required')
            parent=f['parent_entry_ordinal']
            if parent is not None:
                ei=integer(parent,0,entered-1);row=ENTRY.unpack_from(entries,ei*ENTRY.size)
                if row[2]!=f['binding'] or row[3]!=wi or flags[ei]&1:raise ValueError('delegated worker parent differs')
            work[ordinal]=(wi,now);next_work+=1
        elif kind=='work_finished':
            keys(f,('work_ordinal','outcome'))
            integer(f['work_ordinal'])
            if f['work_ordinal'] not in work or f['outcome'] not in ('ok','error','cancelled','timeout'):raise ValueError('unique original work interval terminal required')
            work.pop(f['work_ordinal'])
        elif kind=='private_invalidated':
            keys(f,('reason','affected'));integer(f['affected'],0,256)
            if f['reason'] not in ('pause','reset','delete','consent_revoked','close'):raise ValueError('known privacy lifecycle required')
        elif kind=='incomplete':
            keys(f,('reason','count'));integer(f['count'],0,caps['max_events_per_run'])
            if f['reason'] not in REASONS:raise ValueError('known compact incomplete category required')
        elif kind=='settlement_ref':
            keys(f,('entry_ordinal','receipt_sha256','claims_page_sha256'));ei=integer(f['entry_ordinal'],0,entered-1)
            if flags[ei]&64:raise ValueError('duplicate declared settlement')
            row=ENTRY.unpack_from(entries,ei*ENTRY.size)
            _validate_receipt(directory,d,ctx,ei,bindings[row[2]],f,financial)
            settlements[ei*64:(ei+1)*64]=bytes.fromhex(f['receipt_sha256'])+bytes.fromhex(f['claims_page_sha256'])
            flags[ei]|=64
        else:raise ValueError('unknown native event variant')
        if len(pending)+len(work)+active_entries>caps['max_active_tickets']:
            raise ValueError('native active inventory exceeds frozen bound')
    observed=dict(event_count=events,entry_count=entered,reservation_count=reservation,released_count=released,
        outstanding_tickets=len(pending),outstanding_entries=entered-finished,outstanding_work=len(work),
        registered_workers=len(workers),unclosed_workers=sum(not w['closed'] for w in workers.values()),unsettled_entries=entered)
    if term['state']=='closed':
        if term['prefix'] is not None or term['child_cleanup_finished'] is not True or mapping!=term['mapping_sha256']:
            raise ValueError('closed native inventory has incomplete source/cleanup prefix')
        for key,value in observed.items():
            if integer(term[key])!=value:raise ValueError('terminal canonical inventory differs')
        if pending or work or entered!=finished or entered!=published or observed['unclosed_workers']:
            raise ValueError('closed native inventory has outstanding owned work')
    else:
        for key in observed:integer(term[key],0,caps['max_events_per_run'])
        if term['event_count']!=events or term['entry_count']<entered or term['reservation_count']<reservation:
            raise ValueError('failed terminal hides preserved original events')
    if type(term['incomplete_counts']) is not dict or set(term['incomplete_counts'])-set(REASONS):raise ValueError('bounded incomplete diagnostics required')
    for value in term['incomplete_counts'].values():integer(value,0,caps['max_events_per_run'])
    _validate_counters(directory,term,counters,scopes,bindings,ctx)
    if term['state']=='closed':
        names=set();n=0
        with os.scandir(directory) as listing:
            for item in listing:
                n+=1
                if n>65536:raise ValueError('native file inventory exceeds bound')
                if item.name.startswith(('receipt-','receipt-source-','claims-')):names.add(item.name)
        if names!=financial['files']:raise ValueError('orphan or omitted financial artifacts')
    result=ValidatedRecordingIndex(directory,d,term,refs,entries,flags,entered,hashes,settlements,_token=_INDEX_TOKEN)
    for ticket,p in pending.items():
        counters[p['scope'],p['binding'],'outstanding_tickets']+=1
        counters[p['scope'],p['binding'],'pending_dispatched' if ticket_state[ticket]==2 else 'pending_undispatched']+=1
    result._lane_counts=_freeze(dict(counters))
    result.claim_keys=b''.join(bytes.fromhex(v) for v in sorted(financial['claims']))
    result.financial_documents=tuple(financial['documents'].items())
    result.financial_bytes=financial['source_bytes']+financial['receipt_bytes']
    result.source_bytes=financial['source_bytes'];result.receipt_bytes=financial['receipt_bytes']
    result.evidence_ref_count=evidence_refs;result.receipt_count=len(financial['receipts'])
    result._sealed=True
    return result


def _validate_counters(directory,terminal,counters,scopes,bindings,context):
    refs=terminal['counter_pages']
    if type(refs) is not list or len(refs)>16:raise ValueError('bounded counter pages required')
    seen=set()
    for n,ref in enumerate(refs):
        keys(ref,('ordinal','sha256','bytes'))
        integer(ref['ordinal'],0,15);integer(ref['bytes'],1,65536)
        if ref['ordinal']!=n:raise ValueError('ordered counter pages required')
        page,data=_json_file(directory,'counters-%06d.json'%n,65536,ref['sha256'])
        keys(page,('schema_version','kind','nonce','ordinal','counters'))
        integer(page['schema_version'],1,1);integer(page['ordinal'],0,15)
        if len(data)!=ref['bytes'] or page['schema_version']!=1 or page['kind']!='recording_counters' or page['nonce']!=context['nonce'] or page['ordinal']!=n:raise ValueError('counter page identity mismatch')
        if type(page['counters']) is not list or not page['counters']:raise ValueError('nonempty counter page required')
        for row in page['counters']:
            keys(row,('scope','binding','entered','finished','published','unsettled'))
            scope=integer(row['scope'],0,len(scopes)-1);binding=integer(row['binding'],0,len(bindings)-1);key=(scope,binding)
            if key in seen:raise ValueError('duplicate provider counter group')
            seen.add(key)
            for field in ('entered','finished','published','unsettled'):
                expected=counters[scope,binding,'entered' if field=='unsettled' else field]
                if integer(row[field])!=expected and terminal['state']=='closed':raise ValueError('counter differs from canonical entries')
    if seen!={(s['scope_index'],b) for s in scopes for b in range(len(bindings))}:raise ValueError('explicit zero-inclusive counter inventory required')


def iter_canonical_inventory(index,*,scope_index=None,binding_index=None):
    if type(index) is not ValidatedRecordingIndex or not getattr(index,'_sealed',False):raise TypeError('validated native index required')
    pending={};work={};workers={};ctx=index.terminal['context']
    inv=index.declarations['collection']['invocations'][ctx['invocation_ordinal']]
    bindings=next(a['provider_bindings'] for a in index.declarations['collection']['arms'] if a['arm']==inv['arm'])
    scopes=[s for s in index.declarations['scope_protocol']['scopes'] if s['invocation_ordinal']==ctx['invocation_ordinal']]
    if scope_index is not None:integer(scope_index,0,len(scopes)-1)
    if binding_index is not None:integer(binding_index,0,len(bindings)-1)
    for chunk,event in _chunks(index.directory,index.chunk_refs,ctx):
        f=event['fields'];kind=event['event']
        if kind=='reserved':pending[f['ticket']]=dict(f,refs=[])
        elif kind=='evidence_ref':pending[f['ticket']]['refs'].append(f['reference'])
        elif kind=='released':pending.pop(f['ticket'])
        elif kind=='entered':
            original=pending.pop(f['ticket']);ei=f['entry_ordinal'];row=ENTRY.unpack_from(index.entries,ei*ENTRY.size)
            if scope_index is not None and row[1]!=scope_index or binding_index is not None and row[2]!=binding_index:continue
            binding=bindings[row[2]]
            yield dict(kind='operation',owner_kind='recording_native4',owner=dict(context=dict(ctx),scope_index=row[1],binding_index=row[2]),
                entry=dict(ticket_ordinal=row[0],entry_ordinal=ei,instance_sha256=f['instance'],
                    request_decision_sha256=original['request_sha256'],entered_event_sha256=row[9].hex()),
                provider=binding['provider'],model_revision=binding['model_revision'],role=binding['role'],
                prompt_sha256=original['prompt_sha256'],evidence_sha256=original['evidence_sha256'],
                source_chunk=dict(index.chunk_refs[chunk]),source_inventory_locator=dict(
                    recording_id=index.declarations['source_manifest']['recording_id'],
                    source_manifest_sha256=index.declarations['collection']['recordings'][inv['recording_index']]['source_manifest_sha256']),
                instance=f['instance'],invocation_ordinal=ctx['invocation_ordinal'],scope_index=row[1],
                binding_index=row[2],entry_ordinal=ei,entry_locator=dict(chunk_ordinal=chunk,event_sequence=row[4],entry_digest=row[9].hex()),
                outcome=('ok','error','cancelled','timeout')[(index.flags[ei]>>2)&3] if index.flags[ei]&1 else 'outstanding',result_sha256=None if row[8]==b'\0'*32 else row[8].hex(),
                evidence_refs=original['refs'],declared_settlement_ref=None if not index.flags[ei]&64 else dict(
                    receipt_sha256=index.settlements[ei*64:ei*64+32].hex(),
                    claims_page_sha256=index.settlements[ei*64+32:(ei+1)*64].hex()),unknown_liability=True)


        elif kind=='worker_registered':workers[f['worker_ordinal']]=dict(f)
        elif kind=='work_started':
            work[f['work_ordinal']]=dict(f,started_ns=event['monotonic_ns'],start_event_sha256=event['sha256'],
                start_chunk=dict(index.chunk_refs[chunk]))
        elif kind=='work_finished':
            original=work.pop(f['work_ordinal']);original['work_kind']=original.pop('kind')
            if (scope_index is None or scope_index==0) and (binding_index is None or binding_index==original['binding']):
                yield dict(kind='worker_interval',context=dict(ctx),**original,
                    worker_kind=workers[original['worker_ordinal']]['kind'],ended_ns=event['monotonic_ns'],
                    outcome=f['outcome'],end_event_sha256=event['sha256'],end_chunk=dict(index.chunk_refs[chunk]),
                    clock_domain='host_monotonic_ns',money_verified=False,actual_acceptance='incomplete')
    for original in work.values():
        original['work_kind']=original.pop('kind')
        if (scope_index is None or scope_index==0) and (binding_index is None or binding_index==original['binding']):
            yield dict(kind='worker_interval',context=dict(ctx),**original,
                worker_kind=workers[original['worker_ordinal']]['kind'],ended_ns=None,outcome='outstanding',
                end_event_sha256=None,end_chunk=None,clock_domain='host_monotonic_ns',money_verified=False,actual_acceptance='incomplete')


def iter_canonical_operations(index,*,scope_index=None,binding_index=None):
    """Second-pass actual entries only; use unified inventory when work is needed."""
    for row in iter_canonical_inventory(index,scope_index=scope_index,binding_index=binding_index):
        if row['kind']=='operation':
            row.pop('kind');yield row


def canonical_lane_inventory(index,scope_index,binding_index):
    """Complete represented lane inventory before transport/receipt matching.

    An incomplete prefix cannot attest that an unobserved invocation did not run.
    No-entry reservations remain separate from actual callback/provider counts.
    """
    if type(index) is not ValidatedRecordingIndex or not getattr(index,'_sealed',False):raise TypeError('validated native index required')
    ctx=index.terminal['context'];inv=index.declarations['collection']['invocations'][ctx['invocation_ordinal']]
    bindings=next(a['provider_bindings'] for a in index.declarations['collection']['arms'] if a['arm']==inv['arm'])
    scopes=[s for s in index.declarations['scope_protocol']['scopes'] if s['invocation_ordinal']==ctx['invocation_ordinal']]
    si=integer(scope_index,0,len(scopes)-1);bi=integer(binding_index,0,len(bindings)-1)
    counts={key:index._lane_counts.get((si,bi,key),0) for key in ('reserved','dispatched','released','released_dispatched','released_undispatched','outstanding_tickets','pending_dispatched','pending_undispatched','entered','finished','published')}
    return dict(owner_kind='recording_native4',owner=dict(context=dict(ctx),scope_index=si,binding_index=bi),
        provider=bindings[bi]['provider'],model_revision=bindings[bi]['model_revision'],role=bindings[bi]['role'],
        counts=counts,outstanding_entries=counts['entered']-counts['finished'],
        unknown_liability_count=counts['entered'],unknown_dispatch_liability_count=counts['dispatched']-counts['entered'],
        unobserved_entry_inventory_possible=index.terminal['state']!='closed',
        source_inventory_locator=dict(recording_id=index.declarations['source_manifest']['recording_id'],
            source_manifest_sha256=index.declarations['collection']['recordings'][inv['recording_index']]['source_manifest_sha256']),actual_acceptance='incomplete')


def lookup_canonical_operation(index,entry_ordinal):
    """One verified source chunk plus packed original reservation digests, O(1) lookup.

    No caller-supplied identity cache or latest-attempt heuristic participates.
    All auxiliary hashes were captured from original validated reservation events.
    """
    if type(index) is not ValidatedRecordingIndex or not getattr(index,'_sealed',False):raise TypeError('validated recording index required')
    ei=integer(entry_ordinal,0,index.count-1);row=ENTRY.unpack_from(index.entries,ei*ENTRY.size)
    ref=index.chunk_refs[row[7]]
    data=_chunk_data(index.directory,ref)
    found=None
    for line in data.splitlines()[1:None if ref.get('partial') else -1]:
        item=decode(line,2048)
        if item['sequence']==row[4]:found=item;break
    if (found is None or found['event']!='entered' or found['sha256']!=row[9].hex()
            or found['fields']['entry_ordinal']!=ei):raise ValueError('original indexed entry locator differs')
    ctx=index.terminal['context'];inv=index.declarations['collection']['invocations'][ctx['invocation_ordinal']]
    binding=next(a['provider_bindings'] for a in index.declarations['collection']['arms'] if a['arm']==inv['arm'])[row[2]]
    raw=index.hashes[ei*96:(ei+1)*96];prompt=bytes(raw[32:64])
    return dict(owner_kind='recording_native4',owner=dict(context=dict(ctx),scope_index=row[1],binding_index=row[2]),
        entry=dict(ticket_ordinal=row[0],entry_ordinal=ei,instance_sha256=found['fields']['instance'],
            request_decision_sha256=bytes(raw[:32]).hex(),entered_event_sha256=row[9].hex()),
        provider=binding['provider'],model_revision=binding['model_revision'],role=binding['role'],
        prompt_sha256=None if prompt==b'\0'*32 else prompt.hex(),evidence_sha256=bytes(raw[64:]).hex(),
        source_chunk=dict(ref),unknown_liability=True,actual_acceptance='incomplete')


def _validate_receipt(directory,d,context,entry,binding,reference,state):
    if d['scope_protocol']['retain_financial_documents'] is not True:
        raise ValueError('financial retention was not consented')
    card=d.get('cost_card')
    original=d['plan']['comparison']['cost']
    if original.get('mode')!='itemized' or card!=original.get('rate_card'):
        raise ValueError('complete original itemized cost card required for declared quantities')
    page_hash=sha(reference['claims_page_sha256']);receipt_hash=sha(reference['receipt_sha256'])
    page_name='claims-'+page_hash+'.json';receipt_name='receipt-'+receipt_hash+'.json'
    page,_=_json_file(directory,page_name,65536,page_hash)
    keys(page,('schema_version','kind','receipt_sha256','owner_instance','claims'))
    owner=digest(['gamst.recording.instance.v1',context['nonce'],entry])
    if page['schema_version']!=1 or type(page['schema_version']) is not int or page['kind']!='recording_declared_claims' or page['receipt_sha256']!=receipt_hash or page['owner_instance']!=owner:
        raise ValueError('original declared claim ownership differs')
    receipt,body=_json_file(directory,receipt_name,65536,receipt_hash)
    keys(receipt,('schema_version','kind','origin','money_basis','provider','revision','account_scope','invoice_id',
        'source_document_sha256','source_document_bytes','lines'))
    if (receipt['schema_version']!=1 or type(receipt['schema_version']) is not int or receipt['kind']!='recording_declared_receipt'
            or receipt['origin'] not in ('application_declared','synthetic_fixture') or receipt['money_basis']!='unknown'
            or (receipt['provider'],receipt['revision'])!=(binding['provider'],binding['model_revision'])):
        raise ValueError('explicit declared unknown-money provider evidence required')
    from .recording_native_schema import identifier
    for field in ('provider','revision','account_scope','invoice_id'):identifier(receipt[field])
    source_hash=sha(receipt['source_document_sha256']);size=integer(receipt['source_document_bytes'],1,1024**2)
    source_name='receipt-source-'+source_hash+'.bin'
    source=_read(directory,source_name,1024**2,source_hash)
    if len(source)!=size:raise ValueError('original opaque source length differs')
    if receipt_hash in state['receipts']:raise ValueError('declared receipt reused across owners')
    state['receipts'].add(receipt_hash);state['receipt_bytes']+=len(body)
    if source_hash not in state['sources']:state['source_bytes']+=size;state['sources'].add(source_hash)
    if len(state['receipts'])>16384 or max(state['source_bytes'],state['receipt_bytes'])>64*1024**2:
        raise ValueError('financial artifact capacity exceeded')
    doc=(receipt['provider'],receipt['account_scope'],receipt['invoice_id']);identity=(receipt['revision'],source_hash)
    if doc in state['documents'] and state['documents'][doc]!=identity:raise ValueError('same original invoice changed revision/document')
    state['documents'][doc]=identity
    lines=receipt['lines'];claims=page['claims']
    if type(lines) is not list or not 1<=len(lines)<=64 or type(claims) is not list or len(claims)!=len(lines):
        raise ValueError('every bounded original line must be claimed once')
    rates={rid for rid,rate in card.items() if (rate['provider'],rate['revision'])==(binding['provider'],binding['model_revision'])}
    if not rates:raise ValueError('declared original provider rates absent')
    line_hashes=[]
    for line in lines:
        keys(line,('line_id','quantities','claim_sha256'));identifier(line['line_id'])
        expected=digest(['gamst.recording.claim.v1',receipt['provider'],receipt['account_scope'],receipt['invoice_id'],line['line_id']])
        if line['claim_sha256']!=expected or expected in state['claims']:raise ValueError('duplicate original invoice line across owners')
        if type(line['quantities']) is not dict or set(line['quantities'])!=rates:raise ValueError('complete original per-line quantity inventory required')
        for value in line['quantities'].values():integer(value,0,10**12)
        state['claims'].add(expected);line_hashes.append(expected)
    if len(state['claims'])>d['collection']['capacity']['max_collection_claims']:raise ValueError('global declared claim capacity exceeded')
    seen=set()
    for claim in claims:
        keys(claim,('claim_sha256','line_index'));n=integer(claim['line_index'],0,len(lines)-1)
        if n in seen or claim['claim_sha256']!=line_hashes[n]:raise ValueError('duplicate or changed claim index')
        seen.add(n)
    state['files'].update((source_name,receipt_name,page_name))


def validate_recording_native_collection(runs):
    """Sequentially validate all frozen original invocations and global ownership.

    No collection of live SDKs or duplicate per-task cost rows is constructed.
    `runs` contains exact declarations/directory/terminal_sha256/source_inventory.
    """
    if type(runs) is not list or not 3<=len(runs)<=96:raise ValueError('bounded original ABC run inventory required')
    identity=None;nonce_set=set();seen=set();claims=bytearray();entries=chunks=bytes_used=evidence_refs=source_bytes=receipt_bytes=receipts=0;items=[];cap=None;documents={}
    for run in runs:
        keys(run,('declarations','directory','terminal_sha256','source_inventory'))
        sha(run['terminal_sha256'])
        index=validate_recording_native_inventory(run['declarations'],run['directory'],run['terminal_sha256'],source_inventory=run['source_inventory'])
        c=index.declarations['collection'];q=index.declarations['scope_protocol'];i=index.declarations['instrumentation_protocol']
        current=(digest(c),digest(q),digest(i))
        if identity is None:identity=current;cap=c['capacity'];expected=len(c['invocations'])
        if current!=identity or len(runs)!=expected:raise ValueError('complete one frozen collection required')
        context=index.terminal['context'];n=context['invocation_ordinal'];nonce=context['nonce']
        if n in seen or nonce in nonce_set:raise ValueError('original invocation/nonce reused across arms')
        seen.add(n);nonce_set.add(nonce);entries+=index.count;chunks+=len(index.chunk_refs)
        if entries>cap['max_collection_entries'] or chunks>cap['max_collection_chunks']:raise ValueError('joint native inventory capacity exceeded')
        evidence_refs+=index.evidence_ref_count;source_bytes+=index.source_bytes;receipt_bytes+=index.receipt_bytes;receipts+=index.receipt_count
        if (evidence_refs>cap['max_collection_evidence_refs'] or source_bytes>64*1024**2 or receipt_bytes>64*1024**2
                or receipts>cap['max_late_settlements'] or receipts*1024**2>cap['max_late_lookup_bytes']):
            raise ValueError('joint original reference/financial/lookup inventory exceeded')
        for document,identity_value in index.financial_documents:
            if document in documents and documents[document]!=identity_value:
                raise ValueError('same original invoice changed revision/document across runs')
            documents[document]=identity_value
            if len(documents)>cap['max_late_settlements']:raise ValueError('global original invoice inventory exceeded')
        claims.extend(index.claim_keys)
        if len(claims)//32>cap['max_collection_claims']:raise ValueError('joint original claim inventory exceeded')
        # Exact regular artifact bytes, including all metadata and failed prefixes.
        count=0
        with os.scandir(index.directory) as scan:
            for file in scan:
                count+=1
                if count>65536 or not file.is_file(follow_symlinks=False):raise ValueError('bounded regular native artifact directory required')
                bytes_used+=file.stat(follow_symlinks=False).st_size
        if bytes_used>cap['max_collection_bytes']:raise ValueError('joint native artifact bytes exceeded')
        items.append(dict(invocation_ordinal=n,nonce=nonce,terminal_sha256=run['terminal_sha256'],
            entry_count=index.count,state=index.terminal['state']))
        # Release each run's declaration copies/packed records before reading next.
        del index
    if seen!=set(range(expected)):raise ValueError('missing original recording invocation')
    _sort_packed_hashes(claims)
    for offset in range(32,len(claims),32):
        if claims[offset-32:offset]==claims[offset:offset+32]:raise ValueError('original invoice line reused across arms or recordings')
    return dict(schema_version=1,kind='recording_native_collection_inventory',collection_sha256=identity[0],
        scope_protocol_sha256=identity[1],instrumentation_protocol_sha256=identity[2],
        runs=sorted(items,key=lambda v:v['invocation_ordinal']),entry_count=entries,chunk_count=chunks,
        unique_claim_count=len(claims)//32,artifact_bytes=bytes_used,actual_acceptance='incomplete',
        receipt_authenticity_verified=False,worker_backend_coverage_verified=False)


def _sort_packed_hashes(data):
    """In-place heapsort: fixed32-byte working slices, no per-claim Python objects."""
    size=len(data)//32
    def swap(a,b):
        x=data[a*32:(a+1)*32];data[a*32:(a+1)*32]=data[b*32:(b+1)*32];data[b*32:(b+1)*32]=x
    def down(root,end):
        while 2*root+1<end:
            child=2*root+1
            if child+1<end and data[child*32:(child+1)*32]<data[(child+1)*32:(child+2)*32]:child+=1
            if data[root*32:(root+1)*32]>=data[child*32:(child+1)*32]:return
            swap(root,child);root=child
    for root in range(size//2-1,-1,-1):down(root,size)
    for end in range(size-1,0,-1):swap(0,end);down(0,end)
