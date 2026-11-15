"""Original recording source/native/task join; hash integrity is not model truth.

Only this complete finite join issues ValidatedRecordingRun. Failed original
arms remain typed incomplete observations rather than disappearing from finance.
"""
from array import array
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from gamst.call_controller import _digest
from gamst.evaluation_snapshot import (CAPABILITY_ID,VIEW_NAME,_spec,
    evaluation_snapshot_implementation_sha256)
from .recording_native_channel import NativeRunInventory
from .recording_native_schema import digest,encoded,decode,keys,integer,sha
from .recording_native_validation import open_original_directory,iter_canonical_inventory,canonical_lane_inventory
from .recording_runtime_native import validate_runtime_scope_protocol,validate_runtime_native_inventory
from .recording_runtime_artifacts import (read_ref,iter_outcome_rows,read_publication,
    validate_execution,validate_metadata,validate_task_artifact,validate_fill_case)

_ISSUER=object()
_MAX_JOIN_BYTES=128*1024**2


class _JoinBudget:
    def __init__(self,source_records):self.used=source_records*9
    def retain(self,value):
        # Account decoded containers and hash-table slots conservatively, rather
        # than treating the JSON length as the actual Python working set.
        self.used+=8*len(encoded(value))+1024
        if self.used>_MAX_JOIN_BYTES:raise ValueError('actual retained run join exceeds128MiB joint workspace')


def preflight_recording_run_validation(declarations,execution,*,publication_rows):
    """Pure conservative initial-driver join admission, before opening outputs.

    The parent holds its input workspace plus the largest active run peak and
    the SUM of retained prior runs. Independent maxima cannot reset that sum.
    """
    from .recording_native_validation import ENTRY
    validate_execution(execution)
    cap=declarations['collection']['capacity'];entries=cap['max_entries_per_run'];events=cap['max_events_per_run']
    records=len(declarations['source_manifest']['records']);tasks=len(declarations['tasks'])
    bindings=len(next(a['provider_bindings'] for a in declarations['collection']['arms'] if a['arm']==execution['arm']))
    integer(publication_rows,0,262144)
    declarations_bytes=8*len(encoded(declarations))+16384
    packed=entries*(ENTRY.size+1+96+64)+cap['max_chunks_per_run']*8192
    held=declarations_bytes+packed+tasks*8192+min(entries,256)*4096+65536
    temporary=(records*9+cap['max_reservations_per_run']*5+entries*(8*8192+1024)+
        (events//2)*(8*2048+1024)+publication_rows*(8*2048+1024)+
        (tasks+1)*bindings*(8*2048+1024)+4*1024**2)
    peak=held+temporary
    if peak>_MAX_JOIN_BYTES:raise ValueError('declared joint run validation workspace exceeds128MiB')
    return dict(peak_workspace_bytes=peak,retained_workspace_bytes=held)


def _retained_workspace(index,values):
    total=sum(len(encoded(value))+1024 for value in values)
    if index is not None:
        total+=sum(len(getattr(index,name)) for name in ('entries','flags','hashes','settlements'))
        total+=8*len(encoded(dict(declarations=index.declarations,terminal=index.terminal,chunks=index.chunk_refs)))+16384
        total+=8*index.financial_bytes
    return total


def _copy(value):return decode(encoded(value),_MAX_JOIN_BYTES)


@dataclass(frozen=True,slots=True,init=False)
class ValidatedRecordingRun:
    """Immutable original joins, not a caller JSON success attestation."""
    _execution:bytes
    _run_inventory:bytes
    _task_receipts:bytes
    _financial_values:bytes
    _source_coverage:bytes
    _task_rows:bytes
    _native_index:object
    _controller_settlements:bytes
    _retained_workspace_bytes:int

    def __init__(self,execution,run_inventory,task_receipts,financial_values,source_coverage,task_rows,native_index,controller_settlements=(),*,_issuer=None):
        if _issuer is not _ISSUER:raise TypeError('complete original recording join must issue this value')
        for name,value in (('execution',execution),('run_inventory',run_inventory),('task_receipts',task_receipts),
                ('financial_values',financial_values),('source_coverage',source_coverage),('task_rows',task_rows)):
            object.__setattr__(self,'_'+name,encoded(value))
        object.__setattr__(self,'_native_index',native_index)
        object.__setattr__(self,'_controller_settlements',encoded(list(controller_settlements)))
        object.__setattr__(self,'_retained_workspace_bytes',_retained_workspace(native_index,
            (execution,run_inventory,task_receipts,financial_values,source_coverage,task_rows,list(controller_settlements))))

    @property
    def execution(self):return decode(self._execution)
    @property
    def run_inventory(self):return decode(self._run_inventory)
    @property
    def task_receipts(self):return decode(self._task_receipts,4*1024**2)
    @property
    def financial_values(self):return decode(self._financial_values)
    @property
    def source_coverage(self):return decode(self._source_coverage,4*1024**2)
    @property
    def task_rows(self):return decode(self._task_rows,4*1024**2)
    @property
    def native_index(self):return self._native_index
    @property
    def controller_settlements(self):return decode(self._controller_settlements,1024**2)
    @property
    def controller_settlement_inventory_sha256(self):return hashlib.sha256(self._controller_settlements).hexdigest()
    @property
    def retained_workspace_bytes(self):return self._retained_workspace_bytes
    @property
    def runtime_completeness(self):
        v=self.financial_values
        return 'complete' if (v['source_comparable'] and v['clock_consistent'] and v['operational_closed'] and
            all(r['status']=='answer' for r in self.task_receipts)) else 'incomplete'


def _open(directory,path):
    if type(path) is not str or not path or path.startswith('/') or any(p in ('','.','..') for p in path.split('/')):
        raise ValueError('fixed original relative path required')
    fd=open_original_directory(directory)
    try:
        for part in path.split('/')[:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd);os.close(fd);fd=new
        result=os.open(path.split('/')[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        info=os.fstat(result)
        if not stat.S_ISREG(info.st_mode):os.close(result);raise ValueError('regular immutable original evidence required')
        return result
    finally:os.close(fd)


def _fixed(directory,path,maximum=65536,*,minimum=1):
    fd=_open(directory,path)
    try:
        info=os.fstat(fd)
        if not minimum<=info.st_size<=maximum:raise ValueError('bounded original evidence required')
        result=dict(path=path,file_sha256=None,bytes=info.st_size)
        h=hashlib.sha256();count=0
        while True:
            block=os.read(fd,65536)
            if not block:break
            count+=len(block);h.update(block)
            if count>maximum:raise ValueError('original evidence exceeded byte admission')
        after=os.fstat(fd)
        if count!=info.st_size or (info.st_size,info.st_mtime_ns,info.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('original evidence changed while hashing')
        result['file_sha256']=h.hexdigest();return result
    finally:os.close(fd)


def _json(directory,ref,*,maximum=65536,original=False):
    keys(ref,('path','file_sha256','bytes'));sha(ref['file_sha256']);integer(ref['bytes'],1,maximum)
    fd=_open(directory,ref['path'])
    try:
        before=os.fstat(fd)
        if before.st_size!=ref['bytes']:raise ValueError('original evidence byte count changed')
        chunks=[];size=0
        while size<=maximum:
            block=os.read(fd,min(65536,maximum+1-size))
            if not block:break
            chunks.append(block);size+=len(block)
        after=os.fstat(fd)
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('original evidence changed while reading')
    finally:os.close(fd)
    data=b''.join(chunks)
    if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['file_sha256']:raise ValueError('original evidence digest/length mismatch')
    value=decode(data,maximum)
    if encoded(value)+(b'' if original else b'\n')!=data:raise ValueError('unchanged canonical original encoding required')
    return value


def _declarations(documents,bundles,invocation):
    c=documents['runtime/collection.json'];inv=c['invocations'][invocation];original=c['recordings'][inv['recording_index']]
    bundle=bundles[original['recording_id']]
    result=dict(plan=documents['runtime/plan.json'],collection=c,scope_protocol=documents['runtime/scopes.json'],
        instrumentation_protocol=documents['runtime/instrumentation.json'],runtime_policy=documents['runtime/policy.json'],
        source_manifest=bundle['manifest'],tasks=bundle['tasks'],profiles=bundle['profiles'])
    numeric=None
    if result['runtime_policy']['schema_version']==2:
        result['stream_protocol']=documents['runtime/accounting.json']
        numeric=dict(opportunities=documents['runtime/opportunities.json'],exposure_policy=documents['runtime/exposure.json'],
            financial_adapter_inventory=documents['runtime/financial-adapters.json'])
    d=validate_runtime_scope_protocol(result,latency_policy=documents['runtime/latency.json'],source_bundles=bundles,numeric_inputs=numeric)
    return d,numeric,inv,original


def _capture(ref,manifest,mapping,acquired,acquired_ns,*,before=None):
    return _capture_prefix(ref,manifest,digest(mapping),acquired,acquired_ns,before=before)


def _capture_prefix(ref,manifest,mapping_sha,acquired,acquired_ns,*,before=None):
    keys(ref,('kind','source_index','capture_ordinal','record_sha256','mapping_sha256'))
    si=integer(ref['source_index'],0,len(manifest['sources'])-1);ri=integer(ref['capture_ordinal'],0,len(manifest['records'])-1)
    record=manifest['records'][ri]
    if (ref['kind']!='capture' or ref['record_sha256']!=digest(record) or ref['mapping_sha256']!=mapping_sha or
            record['source_id']!=manifest['sources'][si]['source_id'] or acquired[ri]!=2):
        raise ValueError('evidence requires an actually acquired original capture')
    if before is not None and acquired_ns[ri]>before:raise ValueError('future acquired capture cannot be evidence for earlier work')
    return ri


def _operation_captures(op,publications):
    direct=[r for r in op['evidence_refs'] if r['kind']=='capture']
    if direct or op['role'] not in ('small_vlm','strong_vlm','text_llm'):return direct
    # SDK semantic calls bind the complete accepted observation's hash, not a
    # duplicated capture ref. Follow that original record into its publication.
    matches=[p for p in publications.values() if p['record_sha256']==op['evidence_sha256'] and p['published_monotonic_ns']<=op['entered_monotonic_ns']]
    if len(matches)!=1:raise ValueError('semantic entry has no unique prior accepted original source record')
    allowed={'text_llm':('task_snapshot',),'small_vlm':('raw_capture_descriptor','fill_descriptor'),'strong_vlm':('fill_descriptor',)}
    if matches[0]['role'] not in allowed[op['role']]:raise ValueError('semantic entry consumes an unsupported source descriptor role')
    return matches[0]['capture_refs']


def _publication_source(row,record,d,execution,mapping,source_mappings,*,mapping_sha=None):
    recording=next(r for r in d['runtime_policy']['recordings'] if r['recording_id']==d['source_manifest']['recording_id'])
    if record.get('subject_id')!=recording['subject_id']:raise ValueError('accepted publication changed frozen subject')
    capability=record['observation'].get('capability',{}).get('id')
    fixed={'raw_capture_descriptor':'evaluation.original_visual.v1','task_snapshot':CAPABILITY_ID,
        'fill_descriptor':'vision.fill_inspection_input','fill_comparison':'vision.fill_comparison'}
    if row['role'] in fixed and capability!=fixed[row['role']]:raise ValueError('publication role differs from actual SDK capability')
    if row['role']=='task_snapshot':
        if row['capture_refs']:raise ValueError('Context envelope is not a direct sensor capture')
        return
    if len(row['capture_refs'])!=1:raise ValueError('initial recorded provider output requires one exact original capture')
    ref=row['capture_refs'][0];source=d['source_manifest']['sources'][ref['source_index']];capture=d['source_manifest']['records'][ref['capture_ordinal']]
    sensor=record['observation'].get('sensor',{});binding=next(b for b in recording['source_bindings'] if b['source_id']==source['source_id'])
    integer(record['timestamp_ms'])
    if (mapping is not None and record['timestamp_ms']!=mapping['sdk_anchor_ms']+capture['offset_ns']//1_000_000 or
            any(sensor.get(k)!=source[k] for k in ('source_id','sensor_id','calibration_id')) or
            sensor.get('sequence')!=capture['sequence'] or sensor.get('capture_sequence')!=capture['sequence'] or
            sensor.get('coordinate_frame')!=binding['coordinate_frame'] or sensor.get('channels')!=([] if binding['channels'] is None else binding['channels'])):
        raise ValueError('accepted publication changed original capture time/sensor/coordinates')
    expected_epoch=source['stream_epoch']
    if row['role']=='specialist_result' and source_mappings:
        epochs=[e['stream_epoch'] for e in source_mappings[ref['source_index']]['provider_epochs'] if e['capability_id']==capability]
        if len(epochs)!=1:raise ValueError('specialist publication has no frozen original provider epoch')
        expected_epoch=epochs[0]
    if (row['role']!='specialist_result' or source_mappings) and sensor.get('stream_epoch')!=expected_epoch:
        raise ValueError('accepted publication changed original delivery/capture epoch')
    if row['role']=='raw_capture_descriptor':
        expected=dict(source_manifest_sha256=digest(d['source_manifest']),record_sha256=digest(capture),record_ordinal=ref['capture_ordinal'],
            payload_sha256=capture['payload_sha256'],original_capture_ns=capture['capture_ns'],mapping_sha256=digest(mapping) if mapping is not None else mapping_sha)
        if record['observation']['value']!={'capture':expected}:raise ValueError('visual descriptor changed original captured image identity')


def _source(directory,capture,manifest,mapping,max_lateness):
    validate_metadata('capture-inventory',capture,capture['execution_sha256'])
    if capture['source_manifest_sha256']!=digest(manifest) or capture['mapping_sha256']!=digest(mapping):
        raise ValueError('source capture inventory changed original manifest/mapping')
    terminal=_json(directory,capture['source_terminal'],maximum=1024**2,original=True)
    if capture['source_counters']!=capture['source_terminal']:raise ValueError('source counters must retain original terminal')
    if len(capture['source_events'])!=1 or capture['source_events'][0]['path']!='source/journal.jsonl':
        raise ValueError('one original source journal required')
    return _source_prefix(directory,capture['source_terminal'],capture['source_events'][0],terminal,manifest,mapping,max_lateness)


def _source_prefix(directory,terminal_ref,ref,terminal,manifest,mapping,max_lateness):
    """Validate retained acquisition even when its mapping write never finished.

    Missing replay coordinates remain unknown. Original event identities,
    capture payloads, counters and source-owned financial files still bind.
    """
    keys(ref,('path','file_sha256','bytes'));sha(ref['file_sha256']);integer(ref['bytes'],0,1024**3)
    if ref['path']!='source/journal.jsonl':raise ValueError('one original source journal required')
    mapping_sha=terminal['summary']['mapping_sha256']
    if mapping_sha is not None:sha(mapping_sha)
    if mapping is not None and mapping_sha!=digest(mapping):raise ValueError('source terminal changed original replay mapping')
    flags=bytearray(len(manifest['records']));times=array('Q',[0])*len(flags)
    counters=[dict(attempted=0,captured=0,dropped=0) for _ in manifest['sources']]
    mapping_events=[];settlement_events=[]
    sources={s['source_id']:i for i,s in enumerate(manifest['sources'])};last='0'*64;sequence=0;last_time=0;size=0;h=hashlib.sha256();late=False
    fd=_open(directory,ref['path'])
    try:
        with os.fdopen(fd,'rb') as stream:
            before=os.fstat(stream.fileno())
            if before.st_size!=ref['bytes']:raise ValueError('source valid prefix differs from retained original journal')
            while True:
                line=stream.readline(2049)
                if not line:break
                size+=len(line);h.update(line)
                if size>ref['bytes'] or len(line)>2048 or not line.endswith(b'\n'):raise ValueError('bounded complete source journal events required')
                event=decode(line,2048);keys(event,('schema_version','sequence','previous_sha256','monotonic_ns','event','fields','sha256'))
                if (event['schema_version']!=1 or event['sequence']!=sequence or event['previous_sha256']!=last or
                        event['sha256']!=digest({k:v for k,v in event.items() if k!='sha256'}) or encoded(event)+b'\n'!=line):
                    raise ValueError('original source event identity/hash chain differs')
                now=integer(event['monotonic_ns'],last_time);last_time=now;last=event['sha256'];sequence+=1
                if event['event']=='mapping_published':mapping_events.append(event)
                if event['event']=='controller_settlement_inventory':settlement_events.append(event)
                if len(mapping_events)>16 or len(settlement_events)>1:raise ValueError('bounded singleton source authority inventories required')
                if event['event'] not in ('capture_acquired','capture_drop'):continue
                f=event['fields'];ri=integer(f['ordinal'],0,len(flags)-1);record=manifest['records'][ri];si=sources[record['source_id']]
                if flags[ri] or f['source_id']!=record['source_id'] or f['sequence']!=record['sequence']:raise ValueError('source capture duplicate/rebound')
                offset=record['offset_ns'];scheduled=integer(f['scheduled_monotonic_ns'])
                integer(f['sdk_timestamp_ms'])
                expected=dict(mapping_sha256=mapping_sha,original_capture_ns=manifest['origin_capture_ns']+offset,
                    submillisecond_remainder_ns=offset%1000000)
                if mapping is not None:
                    expected.update(scheduled_monotonic_ns=mapping['monotonic_anchor_ns']+offset,
                        sdk_timestamp_ms=mapping['sdk_anchor_ms']+offset//1000000)
                if mapping_sha is None:raise ValueError('captured source prefix requires its original bound mapping hash')
                if any(f[k]!=v for k,v in expected.items()) or now<scheduled:raise ValueError('source capture changed original replay time')
                counters[si]['attempted']+=1
                if event['event']=='capture_acquired':
                    if f['payload_sha256']!=record['payload_sha256']:raise ValueError('source acquired another payload')
                    acquired=integer(f['acquired_monotonic_ns'],scheduled,now);late|=acquired-scheduled>max_lateness
                    flags[ri]=2;times[ri]=acquired;counters[si]['captured']+=1
                else:flags[ri]=1;counters[si]['dropped']+=1
            after=os.fstat(stream.fileno())
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source journal changed while reading')
    except BaseException:raise
    if size!=ref['bytes'] or h.hexdigest()!=ref['file_sha256']:raise ValueError('original source journal digest differs')
    j=terminal['journal'];summary=terminal['summary']
    for name,expected in (('written_events',sequence),('valid_prefix_bytes',size),('valid_prefix_sha256',last)):
        if j[name]!=expected:raise ValueError('source durable prefix counters differ from actual journal')
    declared=summary['counters']
    if len(declared)!=len(counters):raise ValueError('all original source counters required')
    for si,(actual,row) in enumerate(zip(counters,declared)):
        count=sum(r['source_id']==manifest['sources'][si]['source_id'] for r in manifest['records'])
        if (row['source_index']!=si or row['source_id']!=manifest['sources'][si]['source_id'] or row['declared']!=count or
                row['uninspected_suffix']!=count-actual['attempted'] or any(row[k]!=v for k,v in actual.items()) or
                row['attempted']!=row['captured']+row['dropped'] or row['offered']!=row['captured'] or
                row['admitted']+row['refused']!=row['offered']):raise ValueError('original source/delivery counters do not reconcile')
    reached=summary.get('terminal_reached_ns');finished=summary.get('cleanup_finished_ns')
    if reached is not None:integer(reached,0 if mapping is None else mapping['monotonic_anchor_ns']+manifest['duration_ns'])
    if finished is not None:integer(finished,last_time)
    complete=(mapping is not None and all(x==2 for x in flags) and not late and reached is not None and terminal['cleanup_finished'] is True and
        j['failure'] is None and j['unwritten_events']==0 and terminal['state']=='replayed_unscored' and summary['mapping_sha256']==digest(mapping))
    coverage=dict(schema_version=1,kind='recording_validated_source_coverage',source_manifest_sha256=digest(manifest),mapping_sha256=mapping_sha,
        source_terminal=terminal_ref,source_events=ref,source_counters=declared,availability_sha256=digest(manifest['availability']),
        record_inventory_sha256=digest(manifest['records']),capture_partition_sha256=hashlib.sha256(flags).hexdigest(),
        actual_capture_count=sum(x==2 for x in flags),drop_count=sum(x==1 for x in flags),uninspected_count=sum(x==0 for x in flags),
        max_acquisition_lateness_ns=max_lateness,source_comparable=complete)
    return flags,times,terminal,coverage,complete,reached,finished,mapping_events,settlement_events


def _failed_source(directory,ref,d,execution,mapping,max_lateness):
    """Read the real failed terminal independently of optional capture metadata."""
    if ref['path']!='source/terminal.json':raise ValueError('fixed original source terminal required')
    terminal=_json(directory,ref,maximum=1024**2,original=True)
    if terminal.get('schema_version')!=1 or terminal.get('state') not in ('failed','replayed_unscored') or terminal.get('cleanup_finished') is not True:
        raise ValueError('unconfirmed source inventory requires an original cleaned terminal')
    allowed={'source-manifest.json','tasks.json','profiles.json','execution.json','mapping.json',
        'source-mappings.json','mapping-records.json','controller-settlements.json'}
    retained={};names=set()
    for original in terminal['files']:
        keys(original,('path','sha256','bytes'))
        name=original['path']
        if name not in allowed or name in names:raise ValueError('unlisted or duplicate original failed source file')
        names.add(name)
        # Check every retained file digest, including empty financial inventories.
        maximum=65536 if name=='mapping-records.json' else 1024**2 if name=='controller-settlements.json' else 64*1024**2
        value=_json(directory,dict(path='source/'+name,file_sha256=original['sha256'],bytes=original['bytes']),maximum=maximum,original=True)
        if name in ('mapping.json','source-mappings.json'):retained[name]=value
        value=None
    if 'mapping.json' in retained:
        actual=retained['mapping.json'];validate_metadata('mapping',actual,digest(execution))
        if mapping is not None and actual!=mapping:raise ValueError('retained source replay mapping differs')
        mapping=actual
    mappings=retained.get('source-mappings.json',[])
    if 'source-mappings.json' in retained:
        validate_metadata('source-mappings',mappings,digest(execution))
        if mapping is None or len(mappings)!=len(d['source_manifest']['sources']):raise ValueError('source prefix requires its original mapping')
        for source,mapped in zip(d['source_manifest']['sources'],mappings):
            if mapped['original_source']!=source or mapped['mapping_sha256']!=digest(mapping) or any(mapped[k]!=v for k,v in mapping.items()):
                raise ValueError('source prefix changed original reversible mapping')
    if mapping is not None:
        manifest=d['source_manifest']
        if mapping['source_manifest_sha256']!=digest(manifest) or any(mapping[k]!=manifest[k] for k in ('recording_id','origin_capture_ns','duration_ns')):
            raise ValueError('source prefix changed original manifest mapping')
    journal=_fixed(directory,'source/journal.jsonl',1024**3,minimum=0)
    result=_source_prefix(directory,ref,journal,terminal,d['source_manifest'],mapping,max_lateness)
    events=result[-2]
    files,records=_source_files(directory,terminal,d,execution,mapping,mappings,events)
    return result,files,records,mapping,mappings


def _outcomes(directory,ref,execution,kind,violations):
    index=read_ref(directory,ref)
    if index['state']!='complete':violations.add('operational_incomplete')
    return iter_outcome_rows(directory,ref,execution_sha256=execution,kind=kind)


def _source_files(directory,terminal,d,execution,mapping,mappings,events):
    """Rebind the source writer's originals, including actual SDK map records."""
    files={}
    for original in terminal['files']:
        keys(original,('path','sha256','bytes'))
        if original['path'] in files:raise ValueError('duplicate original source file')
        files[original['path']]=dict(path='source/'+original['path'],file_sha256=original['sha256'],bytes=original['bytes'])
    expected={'source-manifest.json':d['source_manifest'],'tasks.json':d['tasks'],
        'profiles.json':d['profiles'],'execution.json':execution,'mapping.json':mapping,'source-mappings.json':mappings}
    for path,value in expected.items():
        # A failed child may stop between the two mapping writes. Validate
        # every retained original, but never infer that absent writes happened.
        if path not in files and terminal['state']=='failed' and path in ('mapping.json','source-mappings.json'):continue
        if path not in files or _json(directory,files[path],maximum=64*1024**2,original=True)!=value:
            raise ValueError('source original declaration/mapping changed')
    ref=files.get('mapping-records.json')
    if ref is None:
        if terminal['state']=='failed' and not events:return files,[]
        raise ValueError('registered original mapping record inventory required')
    records=_json(directory,ref,maximum=65536,original=True)
    if (type(records) is not list or len(records)!=len(mappings) or len(events)>len(records) or
            terminal['state']!='failed' and len(events)!=len(records)):
        raise ValueError('complete registered source mapping inventory required')
    ids=set()
    for ordinal,(record,mapped) in enumerate(zip(records,mappings)):
        source=mapped['original_source'];observation=record.get('observation',{});sensor=observation.get('sensor',{})
        if (record.get('type')!='observation.signal' or record.get('id') in ids or
                observation.get('capability',{}).get('id')!='replay.mapping.v1' or observation.get('value')!={'mapping':mapped} or
                sensor.get('source_id')!=source['source_id'] or sensor.get('sensor_id')!=source['sensor_id'] or
                sensor.get('stream_epoch')!=source['stream_epoch'] or record.get('timestamp_ms')!=mapping['sdk_anchor_ms']):
            raise ValueError('registered original SDK mapping record/event differs')
        ids.add(record['id'])
        if ordinal<len(events):
            fields=events[ordinal]['fields'];keys(fields,('source_id','mapping_sha256','source_mapping_sha256','record_sha256','record_ordinal','mapping_records_sha256'))
            if fields!=dict(source_id=source['source_id'],mapping_sha256=digest(mapping),source_mapping_sha256=digest(mapped),
                    record_sha256=digest(record),record_ordinal=ordinal,mapping_records_sha256=ref['file_sha256']):
                raise ValueError('registered original SDK mapping record/event differs')
    return files,records


def _context_records(directory,context,publications,mapping_records,observed_end):
    """Every Context observation retains its accepted original record and time."""
    records={r['id']:r for r in mapping_records}
    for rid,pub in publications.items():
        if pub['published_monotonic_ns']<=observed_end and pub['role']!='task_snapshot':
            records[rid]=read_ref(directory,pub['record'],maximum=131072)
    observations=context.get('observations')
    if type(observations) is not dict or len(observations)>32:raise ValueError('bounded original SDK Context observations required')
    from gamst.context import observation_packet
    for name,value in observations.items():
        record=records.get(value.get('evidence_id'))
        if record is None or record.get('subject_id')!=context.get('subject_id'):
            raise ValueError('Context observation has no original accepted source evidence')
        if record.get('observation',{}).get('capability',{}).get('id')==CAPABILITY_ID:
            raise ValueError('evaluation snapshots cannot reenter source Context')
        if observation_packet([record],context['timestamp_ms'])!={name:value}:
            raise ValueError('Context observation changed original value, age, freshness or sensor')
    for item in context.get('evidence',[]):
        if item.get('record_id') not in records:raise ValueError('Context event evidence is not an accepted original record')


def _runtime_outcomes(directory,summary,d,execution,mapping,flags,times,operations,publications,counts,violations):
    """Join every frozen lane and original opportunity, independently of source drops."""
    if summary is None:return
    arm=next(a for a in d['runtime_policy']['arms'] if a['arm']==execution['arm'])
    native_arm=next(a for a in d['collection']['arms'] if a['arm']==execution['arm'])
    bindings=native_arm['provider_bindings'];by_binding={b['binding_id']:i for i,b in enumerate(bindings)}
    policy=next(p for p in arm['recording_policies'] if p['recording_id']==d['source_manifest']['recording_id'])
    lanes={l['lane_id']:l for l in arm['lanes']};seen=set()
    for row in _outcomes(directory,summary['adapter_outcomes'],digest(execution),'adapters',violations):
        lane=lanes.get(row['lane_id'])
        if lane is None or row['lane_id'] in seen or any(row[k]!=lane[k] for k in ('adapter_id','controller_id')) or row['binding_index']!=by_binding[lane['binding_id']]:
            raise ValueError('adapter outcome changed frozen lane/binding/controller')
        seen.add(row['lane_id']);bi=row['binding_index']
        lane_counts=[v for (scope,b),v in counts.items() if b==bi]
        for field,native in (('entry_count','entered'),('finished_count','finished'),('published_count','published')):
            if row[field]!=sum(v['counts'][native] for v in lane_counts):raise ValueError('adapter counters conceal original native work')
        if row['unknown_liability_count']!=sum(v['unknown_liability_count']+v['unknown_dispatch_liability_count'] for v in lane_counts):
            raise ValueError('adapter counters conceal original unknown liability')
        if row['state']!='closed':violations.add('operational_incomplete')
    if summary['state']=='closed' and seen!=set(lanes):raise ValueError('closed child omits frozen semantic adapter')
    submitted={};ordinal=0
    for row in _outcomes(directory,summary['delivery_outcomes'],digest(execution),'deliveries',violations):
        if row['delivery_ordinal']!=ordinal:raise ValueError('delivery event original order differs')
        ordinal+=1;si=integer(row['source_index'],0,len(d['source_manifest']['sources'])-1)
        ri=integer(row['capture_ordinal'],0,len(flags)-1);source=d['source_manifest']['sources'][si];record=d['source_manifest']['records'][ri]
        if record['source_id']!=source['source_id'] or flags[ri]!=2 or row['observed_ns']<times[ri] or row['payload_bytes']!=record['payload_bytes']:
            raise ValueError('downstream delivery changed acquired original capture')
        bi=row['binding_index']
        if bi is not None and (not 0<=bi<len(bindings) or source['source_id'] not in bindings[bi]['source_ids']):
            raise ValueError('delivery crosses frozen source binding')
        if row['event']=='provider_submitted':
            if bi is None:raise ValueError('provider submission must name actual binding')
            submitted[si,bi]=submitted.get((si,bi),0)+1
        if row['operation_instance'] is not None and row['operation_instance'] not in operations:raise ValueError('delivery operation has no original native entry')
        if row['publication_ordinal'] is not None and row['publication_ordinal'] not in {v['publication_ordinal'] for v in publications.values()}:
            raise ValueError('delivery has no actual publication')
    expected={(next(i for i,s in enumerate(d['source_manifest']['sources']) if s['source_id']==delivery['source_id']),by_binding[delivery['binding_id']])
        for delivery in policy['deliveries'] if delivery['profile']!='acquire_only'}
    delivery_summaries=_fixed(directory,'delivery-summaries-index.json');seen=set()
    for row in _outcomes(directory,delivery_summaries,digest(execution),'delivery-summaries',violations):
        key=row['source_index'],row['binding_index']
        if key not in expected or key in seen or row['accepted']!=submitted.get(key,0):raise ValueError('SDK delivery admission counter differs from original submissions')
        seen.add(key)
        if row['residual_unresolved']!=row['accepted']-row['postclose_processed']-row['postclose_errors']-row['postclose_evicted']:
            raise ValueError('SDK delivery residual conceals unfinished original work')
        if row['postclose_queued'] or row['postclose_unfinished']:violations.add('operational_incomplete')
    if summary['state']=='closed' and seen!=expected:raise ValueError('closed child omits actual SDK delivery disposition')
    for row in _outcomes(directory,summary['mapping_updates'],digest(execution),'mapping-updates',violations):
        # The fixed initial driver never resets its source/SDK delivery lanes.
        # Supporting resets requires a frozen driver and a complete old/new epoch join.
        raise ValueError('initial recording driver cannot invent provider resets')
    schedules={};manifest=d['source_manifest']
    recording=next(r for r in d['runtime_policy']['recordings'] if r['recording_id']==manifest['recording_id'])
    from .recorded_ticks import RecordedTickSchedule
    for kind in ('periodic','fill_chain'):
        definition=policy[kind]
        if definition is None:continue
        source_id=recording['primary_visual_source_id'] if kind=='periodic' else definition['source_id']
        si=next(i for i,s in enumerate(manifest['sources']) if s['source_id']==source_id)
        schedule=RecordedTickSchedule(duration_ns=manifest['duration_ns'],period_ns=definition['period_ns'] if kind=='periodic' else definition['selection_period_ns'],
            phase_ns=definition['phase_ns'],time_origin_policy=definition.get('time_origin_policy','recording_start'),origin_capture_ns=manifest['origin_capture_ns'],
            max_age_ms=definition.get('max_age_ms',1000),deadline_tolerance_ns=definition.get('deadline_tolerance_ns',definition.get('max_lateness_ns')))
        # Backend completion order is not schedule order. Retain only bounded
        # opportunity coverage and selected ordinals, not all result objects.
        schedules[kind]=dict(definition=definition,source_index=si,schedule=schedule,count=0,
            seen=bytearray(schedule.opportunity_count),selected=array('q',[-1])*schedule.opportunity_count)
    for row in _outcomes(directory,summary['tick_inventory'],digest(execution),'ticks',violations):
        selected=schedules.get(row['policy'])
        if selected is None or mapping is None:raise ValueError('tick activates an unfrozen original perception policy')
        schedule=selected['schedule'];n=row['tick_index'];due=schedule.first_due_ns+n*schedule.period_ns
        if (row['first_due_offset_ns']!=due or row['period_ns']!=schedule.period_ns or row['source_index']!=selected['source_index'] or
                n+row['count']>schedule.opportunity_count or row['observed_begin_ns']<mapping['monotonic_anchor_ns']+due):
            raise ValueError('tick changed or omitted original frozen opportunities: '
                f"policy={row['policy']}, index={row['tick_index']}/{n}, due={row['first_due_offset_ns']}/{due}, "
                f"period={row['period_ns']}/{schedule.period_ns}, source={row['source_index']}/{selected['source_index']}, "
                f"count={row['count']}, opportunities={schedule.opportunity_count}, "
                f"observed_offset={row['observed_begin_ns']-mapping['monotonic_anchor_ns']}")
        for tick in range(n,n+row['count']):
            if selected['seen'][tick]:raise ValueError('duplicate or overlapping original tick opportunity')
            selected['seen'][tick]=1
        selected['count']+=row['count']
        if row['capture_ref'] is not None:
            ri=_capture(row['capture_ref'],manifest,mapping,flags,times,before=row['observed_begin_ns']);original=manifest['records'][ri]
            actual_due=mapping['monotonic_anchor_ns']+due
            if original['offset_ns']>due or times[ri]>actual_due:raise ValueError('tick selected future original capture')
            if row['outcome']!='stale':
                if row['observed_begin_ns']-actual_due>schedule.tolerance_ns or row['observed_begin_ns']-(mapping['monotonic_anchor_ns']+original['offset_ns'])>schedule.max_age_ns:
                    raise ValueError('tick silently changed frozen lateness/freshness')
                selected['selected'][n]=ri
        if row['operation_instance'] is not None:
            op=operations.get(row['operation_instance'])
            if op is None or op['scope_index']!=0 or row['capture_ref'] not in _operation_captures(op,publications):raise ValueError('tick has no actual stream-owned native capture entry')
        if row['case'] is not None:
            _fill_join(directory,row,d,execution,operations,publications,selected['definition'])
    for selected in schedules.values():
        last=-1
        for ri in selected['selected']:
            if ri<0:continue
            if ri<=last:raise ValueError('tick reused or reversed original selected capture')
            last=ri
    if summary['state']=='closed' and any(v['count']!=v['schedule'].opportunity_count for v in schedules.values()):
        raise ValueError('closed child omits original perception opportunities')


def _fill_join(directory,tick,d,execution,operations,publications,definition):
    case=read_ref(directory,tick['case'],maximum=16384);validate_fill_case(case,digest(execution))
    if (case['capture_ref']!=tick['capture_ref'] or case['source_manifest_sha256']!=digest(d['source_manifest']) or
            case['mapping_sha256']!=tick['capture_ref']['mapping_sha256'] or case['configuration_sha256']!=definition['fill_config_sha256'] or case['rule_sha256']!=definition['rule_sha256']):
        raise ValueError('conditional fill case changed original capture/configuration/rule')
    descriptor=None
    for name,role in (('measurement_publication','fill_measurement'),('descriptor_publication','fill_descriptor'),('comparison_publication','fill_comparison')):
        ref=case[name]
        if ref is None:continue
        row,record=read_publication(directory,ref,execution_sha256=digest(execution))
        if row!=publications.get(record['id']) or row['role']!=role or row['capture_refs']!=[case['capture_ref']]:
            raise ValueError('conditional fill case lacks actual original publication')
        if role=='fill_descriptor':descriptor=record
    original=d['source_manifest']['records'][case['capture_ref']['capture_ordinal']]
    descriptor_capture=descriptor['observation']['value']['capture']
    if (descriptor_capture['source_sha256']!=case['image_sha256'] or descriptor_capture['shape']!=original['shape'] or
            descriptor_capture['config_sha256']!=case['configuration_sha256']):
        raise ValueError('conditional fill descriptor changed actual image/shape/configuration')
    for stage in case['stages']:
        if stage['operation_instance'] is None:continue
        op=operations.get(stage['operation_instance'])
        if (op is None or op['binding_index']!=stage['binding_index'] or op['scope_index']!=0 or
                op['prompt_sha256']!=stage['prompt_sha256'] or stage['raw_response_sha256'] is not None and op['result_sha256']!=stage['raw_response_sha256'] or
                op['role']!=('strong_vlm' if stage['stage']=='strong' else 'small_vlm') or case['capture_ref'] not in _operation_captures(op,publications) or
                op['relationship']!={'initial':'root','reinspection':'retry','strong':'escalation'}[stage['stage']] or op['predecessor']!=stage['predecessor_instance'] or
                op['evidence_sha256']!=digest(descriptor)):
            raise ValueError('conditional fill stage has no original actual small/strong entry')


def _settlements(directory,terminal,files,events,documents,execution,operations,native_index):
    """Only the original SDK-owned source inventory supplies receipt authority."""
    if terminal is None:return []
    summary=terminal['summary'];h=summary.get('controller_settlement_inventory_sha256')
    count=summary.get('controller_settlement_count',0);references=summary.get('financial_references',[])
    if documents['runtime/policy.json']['schema_version']==1:
        if h is not None or count or references or events or 'controller-settlements.json' in files:
            raise ValueError('structural U1 cannot activate synthetic financial receipts')
        return []
    ref=files.get('controller-settlements.json')
    if (terminal['state']=='failed' and summary['mapping_sha256'] is None and not operations and
            ref is None and not events and h is None and not count and not references):
        # Startup can fail before financial controllers exist. The failed run
        # remains ineligible; absence here does not close any monetary ledger.
        return []
    if ref is None or len(events)!=1:raise ValueError('U2 requires original actual SDK settlement inventory, including empty')
    rows=_json(directory,ref,maximum=1024**2,original=True)
    if (type(rows) is not list or len(rows)>256 or count!=len(rows) or h!=digest(rows) or
            rows!=sorted(rows,key=lambda row:row['instance'])):
        raise ValueError('actual SDK settlement inventory count/order/digest differs')
    fields=events[0]['fields'];keys(fields,('inventory_sha256','inventory_file_sha256','count'))
    if fields!=dict(inventory_sha256=h,inventory_file_sha256=ref['file_sha256'],count=count):raise ValueError('original source event differs from actual SDK settlement file')
    if type(references) is not list or len(references)!=len(rows):raise ValueError('all original actual SDK charge references required')
    from .recording_financial import validate_adapters,validate_financial_evidence,validate_invoice
    from .recording_lifetime_ledger import validate_controller_settlement
    from .recording_runtime_finance import validate_runtime_charge_scenarios
    from types import SimpleNamespace
    policy=documents['runtime/accounting.json'];adapters=validate_adapters(documents['runtime/financial-adapters.json'],expected_sha256=policy['financial_adapter_inventory_sha256'])
    scenarios=validate_runtime_charge_scenarios(documents['runtime/charge-scenarios.json'],runtime_policy=documents['runtime/policy.json'],
        collection=documents['runtime/collection.json'],responses=[v for p,v in documents.items() if p.startswith('responses/')],
        accounting_policy=policy,financial_adapter_inventory=documents['runtime/financial-adapters.json'])
    scenarios={tuple(v['operation_key'][k] for k in ('invocation_ordinal','binding_index','call_ordinal')):v for v in map(decode,scenarios)}
    by_instance={};claims=set();operation_ordinals={}
    for op in sorted(operations.values(),key=lambda row:row['entry_ordinal']):
        bi=op['binding_index'];ordinal=operation_ordinals.get(bi,0);operation_ordinals[bi]=ordinal+1
        if op['role'] in ('small_vlm','strong_vlm','text_llm'):by_instance[op['instance']]=(op,ordinal)
    ref_rows={}
    for r in references:
        keys(r,('instance','source_document','evidence','controller_settlement'))
        if r['instance'] in ref_rows:raise ValueError('actual SDK charge reference duplicated')
        ref_rows[r['instance']]=r
    if set(ref_rows)!={r['instance'] for r in rows}:raise ValueError('SDK settlement references omit or invent charged instance')
    for row in rows:
        actual=ref_rows[row['instance']]
        invoice=validate_invoice(_json(directory,actual['source_document']))
        evidence=validate_financial_evidence(_json(directory,actual['evidence']),adapters=adapters,invoice=invoice,source_reference=actual['source_document'])
        retained=_json(directory,actual['controller_settlement'])
        if retained!=row:raise ValueError('attachment differs from actual source-owned SDK settlement')
        found=by_instance.get(row['instance'])
        if native_index is not None and found is None:raise ValueError('SDK settlement has no original actual semantic entry')
        op,ordinal=(dict(instance=row['instance']),None) if found is None else found
        validate_controller_settlement(row,run=SimpleNamespace(execution_sha256=digest(execution)),operation=op,evidence=evidence,policy=policy)
        if evidence['owner_instance']!=row['instance'] or len(invoice['lines'])!=1:raise ValueError('SDK charge changed actual original instance or invoice lines')
        line=invoice['lines'][0];operation=line['operation_key']
        if operation is None:raise ValueError('semantic SDK invoice needs original operation key')
        key=tuple(operation[k] for k in ('invocation_ordinal','binding_index','call_ordinal'));scenario=scenarios.get(key)
        if scenario is None or operation['invocation_ordinal']!=execution['invocation_ordinal']:
            raise ValueError('actual SDK invoice has no original frozen response scenario')
        if found is not None and (operation['binding_index']!=op['binding_index'] or operation['call_ordinal']!=ordinal or row['observed_ns']<op['entered_monotonic_ns']):
            raise ValueError('SDK charge crosses original actual entry/call ordinal')
        for name in ('provider','revision','account_scope','invoice_id','currency','currency_unit'):
            if invoice[name]!=scenario[name]:raise ValueError('original synthetic invoice changed frozen financial identity')
        for name in line:
            if name not in ('service_start_ns','service_end_ns','service_event_ns') and line[name]!=scenario[name]:
                raise ValueError('original synthetic invoice line changed frozen scenario')
        if row['components']!=scenario['components'] or line['service_start_ns'] is not None or line['service_end_ns'] is not None or line['service_event_ns']>row['observed_ns']:
            raise ValueError('SDK settlement changed actual synthetic charge/time')
        if evidence['claim_sha256'] in claims:raise ValueError('original charge cannot acquire two owners')
        claims.add(evidence['claim_sha256'])
    return rows


def _native(directory,d,documents,bundles,numeric,parent,run,violations,budget):
    if parent['terminal_sha256'] is None:
        violations.add('missing_native_prefix');return None,{},[],{}
    index=validate_runtime_native_inventory(d,Path(directory)/'native',source_inventory=d['source_manifest'],
        latency_policy=documents['runtime/latency.json'],source_bundles=bundles,numeric_inputs=numeric)
    ref=_fixed(directory,'native/terminal.json',1024**2)
    if ref['file_sha256']!=parent['terminal_sha256'] or dict(index.terminal['context'])!=dict(parent['context']):
        raise ValueError('parent native terminal/context differs from actual original inventory')
    operations={};work=[];counts={}
    for row in iter_canonical_inventory(index):
        budget.retain(row)
        if row['kind']=='operation':
            if row['instance'] in operations:raise ValueError('original native operation duplicated')
            operations[row['instance']]=row
        else:work.append(row)
    for scope in range(len(d['tasks'])+1):
        arm=next(a for a in d['collection']['arms'] if a['arm']==run['arm'])
        for binding in range(len(arm['provider_bindings'])):
            counts[scope,binding]=canonical_lane_inventory(index,scope,binding)
            budget.retain(counts[scope,binding])
    if index.terminal['state']!='closed':violations.add('operational_incomplete')
    return index,operations,sorted(work,key=lambda r:(r['work_ordinal'],r['worker_ordinal'])),counts


def _task_join(directory,summary,tasks,receipts,execution,manifest,mapping,operations,publications,clock_known,mapping_records,require_native=True,source_known=True,*,parent_anchor):
    results={};rows=[];receipt_values=[];used_operations=set()
    if summary is not None:
        for ref in summary['task_results']:
            value=read_ref(directory,ref);validate_task_artifact('result',value,execution)
            tid=value['task_id']
            if tid in results:raise ValueError('original task result duplicated')
            results[tid]=(ref,value)
    original={t['task_id']:i for i,t in enumerate(tasks)}
    if set(results)-set(original):raise ValueError('result names an unknown original task')
    if len(receipts)!=len(tasks):raise ValueError('all original tasks need explicit parent receipts, including not-executed')
    by_task={}
    for ref in receipts:
        receipt=read_ref(directory,ref);validate_task_artifact('receipt',receipt,execution)
        if receipt['task_id'] not in original or receipt['task_id'] in by_task:raise ValueError('parent receipt task duplicated or changed')
        by_task[receipt['task_id']]=(ref,receipt)
    if set(by_task)!=set(original):raise ValueError('all original parent task receipt slots required')
    anchor=None if mapping is None else mapping['monotonic_anchor_ns']
    for n,task in enumerate(tasks):
        tid=task['task_id'];ref,receipt=by_task[tid]
        if ref['path']!='tasks/task-%04d-receipt.json'%n or receipt['input_sha256']!=task['input_sha256']:
            raise ValueError('parent receipt changed original task ordinal/input')
        result_ref,result=results.get(tid,(None,None));evidence_ref=None
        if receipt['result_sha256'] is not None:
            if result is None:
                # The parent may have validated an answer immediately before
                # the child crashed, without a final child-summary frame.
                result_ref=_fixed(directory,'tasks/task-%04d-result.json'%n)
                result=read_ref(directory,result_ref);validate_task_artifact('result',result,execution)
            if result is None or result_ref['file_sha256']!=receipt['result_sha256']:raise ValueError('parent receipt changed original child result')
        if result is not None:
            if result_ref['path']!='tasks/task-%04d-result.json'%n or result['scope_index']!=n+1 or result['input_sha256']!=task['input_sha256']:
                raise ValueError('task result changed original scope/input')
            evidence_ref=result['evidence']
            if evidence_ref is not None:
                evidence=read_ref(directory,evidence_ref);validate_task_artifact('evidence',evidence,execution)
                if evidence_ref['path']!='tasks/task-%04d-evidence.json'%n or any(evidence[k]!=result[k] for k in ('task_id','input_sha256','scope_index')):
                    raise ValueError('task evidence crosses original task/scope')
                context=read_ref(directory,evidence['context'],maximum=32768)
                envelope=read_ref(directory,evidence['snapshot_envelope'])
                record=read_ref(directory,evidence['snapshot_record'],maximum=131072)
                for kind,name in (('context','context'),('envelope','snapshot_envelope'),('record','snapshot_record')):
                    if evidence[name]['path']!='tasks/task-%04d-%s.json'%(n,kind):raise ValueError('snapshot original task artifact ordinal changed')
                if mapping is None or evidence['mapping_sha256']!=digest(mapping):raise ValueError('task snapshot requires original replay mapping')
                expected=dict(kind='frozen_context_bundle_not_new_sensor_measurement',task_id=tid,input_sha256=task['input_sha256'],
                    task_sha256=digest(task),source_manifest_sha256=digest(manifest),mapping_sha256=digest(mapping),context_sha256=digest(context),
                    snapshot_timestamp_ms=context['timestamp_ms'],scheduled_offset_ns=task['snapshot_ns'],
                    observed_begin_ns=evidence['snapshot_observed_begin_ns'],observed_end_ns=evidence['snapshot_observed_end_ns'])
                if any(envelope.get(k)!=v for k,v in expected.items()) or envelope.get('context')!=context:
                    raise ValueError('frozen Context envelope changed original task/time/context')
                policy=context.get('context_policy',{})
                if context.get('type')!='observation.context' or policy.get('view')!=VIEW_NAME or policy.get('excluded_capability_sha256')!=_digest(_spec(evaluation_snapshot_implementation_sha256()).describe()):
                    raise ValueError('Context must retain actual SDK evaluation-wrapper exclusion')
                if context['timestamp_ms']!=record.get('timestamp_ms') or record.get('observation',{}).get('value')!=envelope:
                    raise ValueError('snapshot record cannot republish changed or retimed Context')
                if record.get('observation',{}).get('capability',{}).get('id')!=CAPABILITY_ID:
                    raise ValueError('actual SDK snapshot capability required')
                if source_known:_context_records(directory,context,publications,mapping_records,evidence['snapshot_observed_end_ns'])
                if evidence['context_object_sha256']!=digest(context) or evidence['snapshot_record_sha256']!=_digest(record):raise ValueError('original task Context/record digest differs')
                pub=publications.get(record.get('id'))
                if source_known and (pub is None or pub['record_sha256']!=_digest(record) or pub['role']!='task_snapshot'):raise ValueError('snapshot missing actual accepted publication')
                if not anchor+task['snapshot_ns']<=evidence['snapshot_observed_begin_ns']<=evidence['snapshot_observed_end_ns']:
                    raise ValueError('snapshot observed before original frozen schedule')
                op=operations.get(evidence['operation_instance'])
                if evidence['operation_instance'] is not None and require_native:
                    if (op is None or op['scope_index']!=n+1 or op['role']!='text_llm' or op['evidence_sha256']!=_digest(record) or
                            op['prompt_sha256']!=evidence['prompt_sha256'] or evidence['request_evidence_sha256']!=_digest(record)):
                        raise ValueError('snapshot/controller/native answer entry differs')
                    if op['instance'] in used_operations:raise ValueError('task native answer reused')
                    used_operations.add(op['instance'])
                    # The exact TextContextCalls request is deterministic from
                    # original question, complete record and frozen native ID.
                    from gamst.text_context_call import _SYSTEM
                    request=dict(provider=op['provider'],model_revision=op['model_revision'],messages=[dict(role='system',content=_SYSTEM),
                        dict(role='user',content=json.dumps(dict(question=task['input']['question'],observation_record=record),ensure_ascii=False,allow_nan=False,separators=(',',':')))])
                    if _digest(request)!=op['prompt_sha256']:raise ValueError('answer prompt changed original frozen Context/question')
                if require_native and result['status']=='answer' and (op is None or result['operation_instance']!=op['instance'] or op['outcome']!='ok'):
                    raise ValueError('accepted task answer requires its actual successful original entry')
                response_path='requests/request-%06d.json'%n
                try:response_ref=_fixed(directory,response_path)
                except FileNotFoundError:
                    if result['status']=='answer':raise ValueError('accepted answer omits original actual SDK response')
                else:
                    response=read_ref(directory,response_ref)
                    keys(response,('schema_version','kind','execution_sha256','task_id','input_sha256','scope_index','operation_instance','native_raw_result_sha256','sdk_result'))
                    if (response['schema_version']!=1 or response['kind']!='recording_semantic_response' or
                            response['execution_sha256']!=execution or any(response[k]!=evidence[k] for k in ('task_id','input_sha256','scope_index','operation_instance')) or require_native and op is None):
                        raise ValueError('actual SDK response crosses original task/native operation')
                    sdk=response['sdk_result']
                    keys(sdk,('status','accepted','text','usage','actual_cost','source_record_id','source_record_sha256','prompt_sha256','provider','model_revision','interpretation','usage_verified','quality_validated'))
                    from gamst.text_context_call import TextContextCalls
                    raw=TextContextCalls._response(dict(text=sdk['text'],usage=sdk['usage']))
                    if (sdk['status']!='completed' or sdk['accepted'] is not True or sdk['actual_cost'] is not None or
                            sdk['source_record_id']!=record['id'] or sdk['source_record_sha256']!=digest(record) or
                            require_native and any(sdk[k]!=op[k] for k in ('provider','model_revision','prompt_sha256')) or
                            sdk['interpretation']!='untrusted_text_response' or sdk['usage_verified'] is not False or sdk['quality_validated'] is not False or
                            response['native_raw_result_sha256']!=digest(raw) or require_native and op['result_sha256']!=digest(raw) or
                            result['status']=='answer' and result['answer_text']!=raw['text']):
                        raise ValueError('accepted answer changed original SDK/native raw result')
            elif result['status']=='answer':raise ValueError('answer without actual snapshot evidence')
        floor=task['deadline_ns']-task['opens_ns']
        if receipt['status']=='answer':
            if (not clock_known or result is None or result['status']!='answer' or receipt['answer_sha256']!=result['answer_sha256'] or
                    receipt['received_ns']>anchor+task['deadline_ns'] or receipt['received_ns']<result['child_finished_ns'] or
                    receipt['latency_ns']!=receipt['received_ns']-(anchor+task['opens_ns'])):
                raise ValueError('answer receipt excludes original deadline/validation delay/result')
        else:
            observed=receipt['received_ns'] if receipt['received_ns'] is not None else receipt['failed_observed_ns']
            if parent_anchor is None and (receipt['status']!='not_executed' or receipt['received_ns'] is not None or receipt['result_sha256'] is not None):
                raise ValueError('unconfirmed playback cannot supply a parent task receipt')
            expected=floor if parent_anchor is None or observed is None else max(floor,observed-parent_anchor-task['opens_ns'])
            if receipt['latency_ns']!=expected:raise ValueError('failed original task latency floor/later observation changed')
        receipt_values.append(receipt)
        rows.append(dict(task_id=tid,input_sha256=task['input_sha256'],scope_index=n+1,result=result_ref,evidence=evidence_ref,parent_receipt=ref))
    if summary is not None and summary['state']=='closed' and set(results)!=set(original):raise ValueError('closed child omits original task result')
    return receipt_values,rows


def validate_recording_run(directory,*,documents,source_bundles,invocation_ordinal,parent_native_inventory,activation_started_ns):
    """Join exact original outputs after actual owned parent import and reap.

    `activation_started_ns` is the actual parent's pre-spawn observation. A
    token-issued NativeRunInventory is required even for a failed empty prefix.
    Structural completeness never certifies model quality or authentic money.
    """
    if type(parent_native_inventory) is not NativeRunInventory:raise TypeError('actual owning parent NativeRunInventory required')
    invocation=integer(invocation_ordinal,0,95);activation=integer(activation_started_ns)
    d,numeric,inv,original=_declarations(documents,source_bundles,invocation)
    budget=_JoinBudget(len(d['source_manifest']['records']))
    execution=documents['invocations/%02d/execution.json'%invocation];validate_execution(execution);execution_sha=digest(execution)
    if any(execution[k]!=inv[k] for k in ('arm','stream_id','recording_index')) or execution['invocation_ordinal']!=invocation:
        raise ValueError('execution changed original invocation')
    for name,doc in (('runtime_policy_sha256',d['runtime_policy']),('collection_sha256',d['collection']),
            ('scope_protocol_sha256',d['scope_protocol']),('instrumentation_protocol_sha256',d['instrumentation_protocol'])):
        if execution[name]!=digest(doc):raise ValueError('execution crosses original frozen declarations')
    run_ref=_fixed(directory,'run-inventory.json');run=_json(directory,run_ref);validate_metadata('run-inventory',run,execution_sha)
    retained_execution=_json(directory,_fixed(directory,'execution.json'))
    if retained_execution!=execution:raise ValueError('retained original execution differs from owned input')
    if run['invocation_ordinal']!=invocation:raise ValueError('parent run crosses original invocation')
    parent=_copy(dict(parent_native_inventory));ctx=parent['context']
    if (ctx['invocation_ordinal']!=invocation or ctx['nonce']!=run['nonce'] or any(ctx[k]!=execution[k] for k in
            ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256'))):raise ValueError('parent native inventory crosses original run')
    if _json(directory,run['native_inventory'])!=parent:raise ValueError('retained parent native inventory differs from actual owner token')
    if activation>run['spawn_started_ns']:raise ValueError('activation cannot begin after owned child spawn')
    violations=set();summary=None;mapping=None;capture=None;terminal=None;source_files={};source_mappings=[];mapping_events=[];settlement_events=[];mapping_records=[];coverage=dict(schema_version=1,kind='recording_validated_source_coverage',
        source_manifest_sha256=digest(d['source_manifest']),source_comparable=False,reason='missing_original_source_inventory')
    flags=bytearray(len(d['source_manifest']['records']));times=array('Q',[0])*len(flags);source_finished=cleanup_finished=None;source_complete=False
    handshake=_json(directory,run['handshake']);validate_metadata('handshake',handshake,execution_sha)
    if handshake['nonce']!=run['nonce']:raise ValueError('handshake changed original owned nonce')
    clock_known=handshake['state']=='started' and handshake['clock_implementation']==time.get_clock_info('monotonic').implementation
    if run['child_summary'] is not None:
        summary=read_ref(directory,run['child_summary']);validate_metadata('child-summary',summary,execution_sha)
        if summary['mapping'] is not None:
            mapping=read_ref(directory,summary['mapping']);validate_metadata('mapping',mapping,execution_sha)
            m=d['source_manifest']
            if any(mapping[k]!=m[k] for k in ('recording_id','origin_capture_ns','duration_ns')) or mapping['source_manifest_sha256']!=digest(m):
                raise ValueError('replay clock differs from original source')
            if handshake['anchor_monotonic_ns'] is not None and mapping['monotonic_anchor_ns']!=handshake['anchor_monotonic_ns']:
                raise ValueError('replay anchor differs from actual parent START handshake')
            if handshake['anchor_monotonic_ns'] is None and (run['state']!='failed' or summary['state']!='failed'):
                raise ValueError('unconfirmed playback requires an explicitly failed run and child')
        if summary['capture_inventory'] is not None:
            capture=read_ref(directory,summary['capture_inventory'])
            if run['source_inventory']!=capture['source_terminal'] or mapping is None:raise ValueError('parent/source child inventory mismatch')
            max_lateness=50_000_000 if numeric is None else numeric['exposure_policy']['source_policy']['max_acquisition_lateness_ns']
            flags,times,terminal,coverage,source_complete,source_finished,cleanup_finished,mapping_events,settlement_events=_source(directory,capture,d['source_manifest'],mapping,max_lateness)
            source_mappings=read_ref(directory,capture['source_mappings']);validate_metadata('source-mappings',source_mappings,execution_sha)
            if len(source_mappings)!=len(m['sources']):raise ValueError('one reversible mapping per original source required')
            for source,mapped in zip(m['sources'],source_mappings):
                if mapped['original_source']!=source or mapped['mapping_sha256']!=digest(mapping) or any(mapped[k]!=v for k,v in mapping.items()):
                    raise ValueError('source epoch/clock bridge changed original source')
            source_files,mapping_records=_source_files(directory,terminal,d,execution,mapping,source_mappings,mapping_events)
        clock_known=clock_known and summary['clock_state']=='consistent'
    elif handshake['anchor_monotonic_ns'] is not None:
        # A child can die after a validated PLAYBACK/answer prefix. Keep that
        # original mapping and task receipt; no terminal/source closure follows.
        mapping=read_ref(directory,_fixed(directory,'mapping.json'))
        validate_metadata('mapping',mapping,execution_sha)
        m=d['source_manifest']
        if (mapping['source_manifest_sha256']!=digest(m) or any(mapping[k]!=m[k] for k in ('recording_id','origin_capture_ns','duration_ns')) or
                mapping['monotonic_anchor_ns']!=handshake['anchor_monotonic_ns']):
            raise ValueError('failed original mapping prefix differs from actual START handshake')
    if capture is None and run['source_inventory'] is not None:
        if run['state']!='failed':raise ValueError('closed run requires its complete original capture inventory')
        max_lateness=50_000_000 if numeric is None else numeric['exposure_policy']['source_policy']['max_acquisition_lateness_ns']
        prefix,source_files,mapping_records,mapping,source_mappings=_failed_source(directory,run['source_inventory'],d,execution,mapping,max_lateness)
        flags,times,terminal,coverage,_,source_finished,cleanup_finished,mapping_events,settlement_events=prefix
        if mapping is not None and handshake['anchor_monotonic_ns'] is not None and mapping['monotonic_anchor_ns']!=handshake['anchor_monotonic_ns']:
            raise ValueError('retained source prefix differs from actual parent START handshake')
        coverage['source_comparable']=False
        violations.add('source_mapping_incomplete')
    if not clock_known:violations.add('clock_discontinuity')
    index,operations,work,counts=_native(directory,d,documents,source_bundles,numeric,parent,execution,violations,budget)
    if index is None:source_complete=False;coverage['source_comparable']=False
    mapping_sha=None if mapping is None else digest(mapping)
    if index is not None:
        native_mapping=index.terminal['mapping_sha256']
        if mapping_sha is not None and mapping_sha!=native_mapping:raise ValueError('native terminal changed original replay mapping')
        mapping_sha=native_mapping
    if terminal is not None:
        if (index is not None or mapping_sha is not None) and terminal['summary']['mapping_sha256']!=mapping_sha:
            raise ValueError('retained source/native mapping hashes differ')
        mapping_sha=terminal['summary']['mapping_sha256']
    for op in operations.values():
        if (op['role']=='text_llm' and not 1<=op['scope_index']<=len(d['tasks']) or
                op['role']!='text_llm' and op['scope_index']!=0):
            raise ValueError('actual operation crosses frozen stream/task ownership')
        for ref in op['evidence_refs']:
            if ref['kind']=='capture':
                if mapping is None and (summary is None or summary['state']!='failed'):
                    raise ValueError('native capture requires original replay mapping')
                if terminal is not None:_capture_prefix(ref,d['source_manifest'],mapping_sha,flags,times,before=op['entered_monotonic_ns'])
    publications={};ordinal=0
    if summary is not None:
        for ref in _outcomes(directory,summary['publication_inventory'],execution_sha,'publications',violations):
            row,record=read_publication(directory,ref,execution_sha256=execution_sha)
            budget.retain(row)
            if row['publication_ordinal']!=ordinal or row['record_id'] in publications:raise ValueError('publication original ordinal/record duplicated')
            ordinal+=1
            if mapping_sha is not None and row['source_mapping_sha256']!=mapping_sha:raise ValueError('accepted record changed original source mapping')
            if terminal is not None:
                for cap in row['capture_refs']:_capture_prefix(cap,d['source_manifest'],mapping_sha,flags,times,before=row['published_monotonic_ns'])
            if terminal is not None:_publication_source(row,record,d,execution,mapping,source_mappings,mapping_sha=mapping_sha)
            if row['operation_instance'] is not None:
                op=operations.get(row['operation_instance'])
                if index is not None and (op is None or op['binding_index']!=row['binding_index'] or op['outcome']!='ok' or
                        row['origin_result_sha256'] is not None and op['result_sha256']!=row['origin_result_sha256']):
                    raise ValueError('accepted publication differs from actual native result/owner')
                if index is not None and sorted(row['capture_refs'],key=digest)!=sorted(_operation_captures(op,publications),key=digest):
                    raise ValueError('publication capture provenance differs from actual operation')
            publications[row['record_id']]=row
    if index is not None and capture is not None:
        _runtime_outcomes(directory,summary,d,execution,mapping,flags,times,operations,publications,counts,violations)
    elif summary is not None:
        # A failed native prefix is not an empty operation inventory. Preserve
        # the original indexes but do not certify their asserted counters.
        for kind,name in (('adapters','adapter_outcomes'),('deliveries','delivery_outcomes'),('ticks','tick_inventory'),('mapping-updates','mapping_updates')):
            for _ in _outcomes(directory,summary[name],execution_sha,kind,violations):pass
    receipts,task_rows=_task_join(directory,summary,d['tasks'],run['receipts'],execution_sha,d['source_manifest'],mapping,operations,publications,clock_known,mapping_records,index is not None,capture is not None,parent_anchor=handshake['anchor_monotonic_ns'])
    controller_settlements=_settlements(directory,terminal,source_files,settlement_events,documents,execution,operations,index)
    channel_keys=set()
    for row in _outcomes(directory,run['channel_outcomes'],execution_sha,'channels',violations):
        key=row['channel'],row['direction']
        if key in channel_keys:raise ValueError('parent channel observation duplicated')
        channel_keys.add(key)
        if row['state']!='closed':violations.add('operational_incomplete')
    if run['state']=='closed' and channel_keys!={(c,direction) for c in ('native2','coordinator1') for direction in ('parent_to_child','child_to_parent')}:
        raise ValueError('closed run omits actual parent channel observation')
    for row in _outcomes(directory,run['transport_outcomes'],execution_sha,'transport',violations):
        if row['binding_index']>=len(next(a['provider_bindings'] for a in d['collection']['arms'] if a['arm']==inv['arm'])):
            raise ValueError('transport observation changed original binding')
    # External clients remain unknown even after the owned callback detaches.
    arm=next(a for a in d['runtime_policy']['arms'] if a['arm']==inv['arm']);adapters={a['adapter_id']:a for a in d['runtime_policy']['adapters']}
    external_closed=all(adapters[l['adapter_id']]['implementation'] in ('fixture_completion_v1','smolvlm_process_v1') for l in arm['lanes'])
    if any(w['ended_ns'] is None for w in work):external_closed=False
    if not external_closed:violations.add('external_work_unknown')
    operational=(run['state']=='closed' and summary is not None and summary['state']=='closed' and summary['child_cleanup_finished'] and
        cleanup_finished is not None and parent['parent_reaped'] is True and parent['eof'] is True and parent['hello_verified'] is True and
        parent['failure_category'] is None and parent['returncode']==run['exit_code'] and run['process_reaped_ns'] is not None and
        run['parent_resources_closed_ns'] is not None and index is not None and 'operational_incomplete' not in violations)
    if not operational:violations.add('operational_incomplete')
    source_complete=source_complete and clock_known
    coverage['source_comparable']=source_complete
    if not source_complete:violations.add('source_incomplete')
    if index is None:
        external_closed=False;violations.add('external_work_unknown')
    values=dict(execution_sha256=execution_sha,run_inventory_sha256=digest(run),invocation_ordinal=invocation,stream_id=inv['stream_id'],
        recording_id=original['recording_id'],arm=inv['arm'],source_manifest_sha256=digest(d['source_manifest']),source_coverage_sha256=digest(coverage),
        source_comparable=source_complete,task_row_inventory_sha256=digest(task_rows),clock_domain=handshake['clock_domain'],clock_consistent=clock_known,
        activation_started_ns=activation,child_spawn_started_ns=run['spawn_started_ns'],playback_anchor_ns=handshake['anchor_monotonic_ns'],
        source_finished_ns=source_finished,child_cleanup_finished_ns=cleanup_finished,process_reaped_ns=run['process_reaped_ns'],
        parent_resources_closed_ns=run['parent_resources_closed_ns'],operational_closed=bool(operational),external_work_closed=external_closed,
        worker_inventory_sha256=digest(work),native_terminal_sha256=parent['terminal_sha256'],parent_receipt_inventory_sha256=digest(sorted(receipts,key=lambda r:r['task_id'])),
        semantic_unknown_instances=tuple(sorted(op['instance'] for op in operations.values() if op['unknown_liability'] and op['role'] in ('small_vlm','strong_vlm','text_llm'))),
        violations=tuple(sorted(violations)))
    previous=activation
    for field in ('child_spawn_started_ns','playback_anchor_ns','source_finished_ns','child_cleanup_finished_ns','process_reaped_ns','parent_resources_closed_ns'):
        current=values[field]
        if current is not None:
            integer(current,previous);previous=current
    return ValidatedRecordingRun(execution,run,receipts,values,coverage,task_rows,index,controller_settlements,_issuer=_ISSUER)
