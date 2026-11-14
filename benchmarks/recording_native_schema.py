"""Strict frozen recording declarations. Structural identity is not numeric proof."""
import hashlib
import json
import re
from .recorded_source import validate_manifest
from .replay_clock import validate_tasks
from .three_arm_comparison import validate_three_arm_plan

MAX_DECLARATIONS = 16*1024**2
CAPS = dict(max_entries_per_run=16384, max_reservations_per_run=131072,
    max_events_per_run=262144, max_chunks_per_run=1024, max_active_tickets=256,
    max_collection_entries=131072, max_collection_chunks=8192,
    max_collection_claims=262144, max_collection_evidence_refs=1048576,
    max_collection_bytes=4*1024**3, max_late_settlements=16384,
    max_late_lookup_bytes=16*1024**3)
KINDS = ('process', 'endpoint', 'finish', 'semantic')
BOUNDARIES = ('in_process_backend', 'async_client_callback', 'process_adapter')
REASONS = ('journal_failed','capacity_exhausted','unobserved_boundary',
    'cleanup_incomplete','source_incomplete','financial_admission_refused','settlement_unknown')


class _FrozenDict(dict):
    def _deny(self,*args,**kwargs):raise TypeError('validated original inventory is immutable')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_deny


class _FrozenList(list):
    def _deny(self,*args,**kwargs):raise TypeError('validated original inventory is immutable')
    __setitem__=__delitem__=append=extend=insert=pop=remove=clear=reverse=sort=__iadd__=__imul__=_deny


def _freeze(value):
    if isinstance(value,dict):return _FrozenDict({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,list):return _FrozenList(_freeze(v) for v in value)
    return value



def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
        allow_nan=False, separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def keys(value, required):
    if type(value) is not dict or set(value) != set(required):
        raise ValueError('exact recording fields required')
    return value


def integer(value, low=0, high=2**63-1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('bounded exact recording integer required')
    return value


def sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('canonical SHA256 required')
    return value


def identifier(value):
    if type(value) is not str or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}',value):
        raise ValueError('bounded unchanged recording identifier required')
    return value


def sequence(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError('bounded recording inventory required')
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode(data, maximum=MAX_DECLARATIONS):
    if type(data) is not bytes or len(data)>maximum:
        raise ValueError('recording bytes exceed bound')
    def invalid(value): raise ValueError('nonfinite JSON value')
    return json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
        parse_constant=invalid)


def snapshot(value, maximum=MAX_DECLARATIONS):
    # Check primitive shape before encoding to reject non-string keys, cycles,
    # overly deep input and enormous container graphs without duplicating them.
    stack=[(value,0)]; count=0
    while stack:
        item, depth=stack.pop(); count+=1
        if count>1048576 or depth>32: raise ValueError('bounded JSON tree required')
        if type(item) in (dict,_FrozenDict):
            if any(type(k) is not str for k in item): raise ValueError('string JSON keys required')
            stack.extend((v,depth+1) for v in item.values())
        elif type(item) in (list,_FrozenList): stack.extend((v,depth+1) for v in item)
        elif type(item) not in (str,int,float,bool,type(None)):
            raise ValueError('plain JSON values required')
    data=encoded(value)
    return decode(data,maximum)


def validate_collection(collection, plan):
    c=snapshot(collection)
    keys(c,('schema_version','kind','comparison_plan_sha256','runtime_policy_sha256',
        'recordings','arms','invocations','capacity'))
    if type(c['schema_version']) is not int or c['schema_version']!=2 or c['kind']!='recording_collection':
        raise ValueError('recording collection2 required')
    plan=validate_three_arm_plan(plan,expected_plan_sha256=sha(c['comparison_plan_sha256']))
    if c['runtime_policy_sha256'] is not None: sha(c['runtime_policy_sha256'])
    tasks={t['id']:t for t in plan['comparison']['tasks']}
    assigned=set(); recordings=sequence(c['recordings'],32,1); recording_ids=set()
    for record in recordings:
        keys(record,('recording_id','source_manifest_sha256','tasks_sha256','profiles_sha256','task_ids','duration_ns'))
        rid=identifier(record['recording_id'])
        if rid in recording_ids: raise ValueError('duplicate original recording')
        recording_ids.add(rid)
        for field in ('source_manifest_sha256','tasks_sha256','profiles_sha256'): sha(record[field])
        integer(record['duration_ns'],1,3600*10**9)
        for tid in sequence(record['task_ids'],64,1):
            identifier(tid)
            if tid not in tasks or tid in assigned: raise ValueError('original task must belong once to recording')
            assigned.add(tid)
    if assigned!=set(tasks) or len(assigned)>1000: raise ValueError('complete original task inventory required')
    arms=sequence(c['arms'],3,3)
    if [a.get('arm') for a in arms if type(a) is dict] != ['A','B','C']:
        raise ValueError('original ordered ABC arms required')
    for arm in arms:
        keys(arm,('arm','config_sha256','provider_bindings'))
        if sha(arm['config_sha256'])!=digest(plan['systems'][arm['arm']]):
            raise ValueError('original arm configuration mismatch')
        ids=set()
        for binding in sequence(arm['provider_bindings'],64,1):
            keys(binding,('binding_id','provider','model_revision','role','config_sha256',
                'source_ids','entry_boundary','operation_kinds','semantic_lane'))
            for field in ('binding_id','provider','model_revision','role'): identifier(binding[field])
            if binding['binding_id'] in ids: raise ValueError('duplicate binding ID')
            ids.add(binding['binding_id']); sha(binding['config_sha256'])
            sources=sequence(binding['source_ids'],16)
            if len(set(map(identifier,sources)))!=len(sources): raise ValueError('duplicate binding source')
            if binding['entry_boundary'] not in BOUNDARIES: raise ValueError('unsupported backend observation boundary')
            kinds=sequence(binding['operation_kinds'],4,1)
            if len(set(kinds))!=len(kinds) or any(k not in KINDS for k in kinds):
                raise ValueError('unique supported operation kinds required')
            if 'semantic' in kinds:
                if kinds!=['semantic']: raise ValueError('semantic owner cannot duplicate source observer')
                integer(binding['semantic_lane'],0,7)
            elif binding['semantic_lane'] is not None: raise ValueError('source binding has no semantic lane')
    seen=set(); streams=set(); invocations=sequence(c['invocations'],96,3)
    if len(invocations)!=3*len(recordings): raise ValueError('exact ABC invocation inventory required')
    for ordinal, invocation in enumerate(invocations):
        keys(invocation,('ordinal','recording_index','arm','stream_id'))
        if integer(invocation['ordinal'])!=ordinal: raise ValueError('contiguous invocation ordinal required')
        ri=integer(invocation['recording_index'],0,len(recordings)-1)
        if invocation['arm'] not in ('A','B','C'): raise ValueError('original arm required')
        sid=identifier(invocation['stream_id']); pair=(ri,invocation['arm'])
        if sid in streams or pair in seen: raise ValueError('duplicate stream or recording-arm execution')
        streams.add(sid);seen.add(pair)
    keys(c['capacity'],CAPS)
    for key, maximum in CAPS.items(): integer(c['capacity'][key],1,maximum)
    if c['capacity']['max_late_settlements']*1024**2>c['capacity']['max_late_lookup_bytes']:
        raise ValueError('late settlement read budget insufficient')
    return c,plan


def validate_recording_scope_protocol(declarations):
    d=snapshot(declarations)
    required={'plan','collection','scope_protocol','instrumentation_protocol','source_manifest','tasks','profiles'}
    optional={'runtime_policy','stream_protocol','allocation_protocol','cost_card'}
    if not required<=set(d) or set(d)-required-optional:
        raise ValueError('exact recording declaration bundle required')
    c,plan=validate_collection(d['collection'],d['plan']); q=d['scope_protocol'];i=d['instrumentation_protocol']
    keys(q,('schema_version','kind','collection_sha256','comparison_plan_sha256','stream_protocol_sha256',
        'allocation_protocol_sha256','recording_reconciliation_sha256','cost_card_sha256',
        'retention_policy','retain_financial_documents','scopes'))
    if type(q['schema_version']) is not int or q['schema_version']!=1 or q['kind']!='recording_scope_protocol':
        raise ValueError('recording scope protocol1 required')
    if q['collection_sha256']!=digest(c) or q['comparison_plan_sha256']!=digest(plan):
        raise ValueError('original declaration binding mismatch')
    if q['recording_reconciliation_sha256'] is not None:
        raise ValueError('recording numeric reconciliation is unsupported')
    if q['retention_policy']!='explicit_external_evaluation_hashes' or type(q['retain_financial_documents']) is not bool:
        raise ValueError('explicit recording retention policy required')
    for name in optional:
        expected=c['runtime_policy_sha256'] if name=='runtime_policy' else q[name+'_sha256']
        if expected is None:
            if name in d: raise ValueError('unreferenced optional declaration')
        elif name not in d or sha(expected)!=digest(d[name]):
            raise ValueError('optional declaration differs from frozen hash')
    tasks={t['id']:t for t in plan['comparison']['tasks']}; expected=[]
    for inv in c['invocations']:
        expected.append(dict(invocation_ordinal=inv['ordinal'],scope_index=0,scope_kind='stream',task_id=None,input_sha256=None))
        for n,tid in enumerate(c['recordings'][inv['recording_index']]['task_ids'],1):
            expected.append(dict(invocation_ordinal=inv['ordinal'],scope_index=n,scope_kind='task',task_id=tid,input_sha256=tasks[tid]['input_sha256']))
    sequence(q['scopes'],3096,1)
    for scope in q['scopes']:
        keys(scope,('invocation_ordinal','scope_index','scope_kind','task_id','input_sha256'))
        integer(scope['invocation_ordinal'],0,len(c['invocations'])-1)
        integer(scope['scope_index'],0,64)
    if q['scopes']!=expected: raise ValueError('exact immutable original scope inventory required')
    keys(i,('schema_version','kind','scope_protocol_sha256','collection_sha256','hello_timeout_ms','close_timeout_ms','fd_environment_key'))
    if type(i['schema_version']) is not int or i['schema_version']!=2 or i['kind']!='recording_instrumentation':
        raise ValueError('recording instrumentation2 required')
    if i['scope_protocol_sha256']!=digest(q) or i['collection_sha256']!=digest(c):
        raise ValueError('instrumentation differs from original scopes')
    integer(i['hello_timeout_ms'],1,10000);integer(i['close_timeout_ms'],1,5000)
    if i['fd_environment_key']!='GAMST_RECORDING_FD_V2': raise ValueError('recording channel2 FD required')
    source=validate_manifest(d['source_manifest']); original_tasks=validate_tasks(d['tasks'],source['duration_ns'])
    sequence(source['sources'],16,1)
    validate_frozen_profiles(source,d['profiles'])
    matches=[r for r in c['recordings'] if r['recording_id']==source['recording_id']]
    if len(matches)!=1: raise ValueError('source must belong to frozen recording')
    recording=matches[0]
    if (recording['duration_ns']!=source['duration_ns'] or recording['source_manifest_sha256']!=digest(source)
            or recording['tasks_sha256']!=digest(original_tasks) or recording['profiles_sha256']!=digest(d['profiles'])
            or recording['task_ids']!=[t['task_id'] for t in original_tasks]):
        raise ValueError('original source/task/profile identity mismatch')
    if any(tasks[t['task_id']]['input_sha256']!=t['input_sha256'] for t in original_tasks):
        raise ValueError('task input differs from original comparison')
    sources={s['source_id'] for s in source['sources']}
    for arm in c['arms']:
        for binding in arm['provider_bindings']:
            if not set(binding['source_ids'])<=sources: raise ValueError('binding source is not in original recording')
    d.update(collection=c,plan=plan,source_manifest=source,tasks=original_tasks)
    return d


def validate_frozen_profiles(source,profiles):
    """Portable declaration audit; the unchanged collector still checks live assets."""
    sensors={s['source_id']:s for s in source['sources']}
    keys(profiles,sensors)
    for sid,profile in profiles.items():
        keys(profile,('profile','delay_ms'));delay=integer(profile['delay_ms'],0,500)
        form=sensors[sid]['format'];name=profile['profile']
        if form=='pcm_s16le_mono' and name=='pcm_level' and delay==0:continue
        if form=='bgr8_hwc' and name in ('fixture_bgr','opencv_lk'):
            if name=='opencv_lk' and (delay!=0 or any(min(r['shape'][:2])<16 for r in source['records'] if r['source_id']==sid)):
                raise ValueError('original OpenCV declaration requires dimensions16 and no fixture delay')
            continue
        raise ValueError('unsupported frozen original source profile')


def run_context(declarations, nonce, invocation_ordinal):
    d=declarations;c=d['collection'];sha(nonce)
    n=integer(invocation_ordinal,0,len(c['invocations'])-1)
    record=c['recordings'][c['invocations'][n]['recording_index']]
    if record['recording_id']!=d['source_manifest']['recording_id']:
        raise ValueError('invocation selected a different original recording')
    return dict(schema_version=4,kind='recording_native',nonce=nonce,invocation_ordinal=n,
        collection_sha256=digest(c),scope_protocol_sha256=digest(d['scope_protocol']),
        instrumentation_protocol_sha256=digest(d['instrumentation_protocol']))


def preflight_recording_collection(recording_declarations):
    """Reserve all native/source/output maxima jointly before activation.

    One verified bundle per original recording is required. Repeated frozen
    declarations count once toward declaration bytes, but their emitted copies
    count once per invocation in the artifact prediction.
    """
    sequence(recording_declarations,32,1)
    collection=None;seen=set();unique={};source_bytes=0;metadata_bytes=0
    for value in recording_declarations:
        d=validate_recording_scope_protocol(value);c=d['collection']
        if collection is None:collection=c;identity=(digest(c),digest(d['scope_protocol']),digest(d['instrumentation_protocol']))
        if identity!=(digest(c),digest(d['scope_protocol']),digest(d['instrumentation_protocol'])):
            raise ValueError('one frozen collection preflight required')
        source=d['source_manifest'];rid=source['recording_id']
        if rid in seen:raise ValueError('recording preflight duplicated')
        seen.add(rid)
        for name,obj in d.items():
            data=encoded(obj);unique.setdefault(digest(obj),len(data))
        predicted_events=2*len(source['records'])+2*len(d['tasks'])+2*len(source['sources'])+16
        predicted_source=predicted_events*2048
        if predicted_source>1024**3:raise ValueError('original source journal preflight exceeds one GiB')
        # Context body64KiB plus separately bounded result metadata =>128KiB/task.
        source_json=sum(len(encoded(d[k]))+1 for k in ('source_manifest','tasks','profiles'))
        source_json+=len(d['tasks'])*128*1024+2*1024**2
        if source_json>80*1024**2:raise ValueError('source JSON artifact bound exceeded')
        source_bytes+=3*(predicted_source+source_json)
        metadata_bytes+=3*(sum(len(encoded(v))+1 for v in d.values()))
    if seen!={r['recording_id'] for r in collection['recordings']}:raise ValueError('all original recording preflight bundles required')
    if sum(unique.values())>MAX_DECLARATIONS:raise ValueError('joint distinct declaration snapshot exceeds16MiB')
    native=preflight_native_capacity(collection)['predicted_native_bytes']
    cap=collection['capacity'];runs=len(collection['invocations'])
    financial=0
    if recording_declarations[0]['scope_protocol']['retain_financial_documents']:
        financial=128*1024**2+min(16384,cap['max_collection_claims'])*65536
    total=source_bytes+metadata_bytes+2*(native+financial)
    if total>cap['max_collection_bytes']:raise ValueError('joint source/native/metadata artifact budget exceeded')
    return dict(schema_version=1,collection_sha256=identity[0],predicted_artifact_bytes=total,
        predicted_source_bytes=source_bytes,predicted_native_bytes=native,
        parent_budget_precharge=source_bytes+metadata_bytes+native+financial,
        retained_native_copy_count=2,
        distinct_declaration_bytes=sum(unique.values()),invocation_count=runs,
        original_task_count=sum(len(r['task_ids']) for r in collection['recordings']),
        sample_gate_eligible=len(collection['recordings'])>=recording_declarations[0]['plan']['comparison']['min_pairs'],
        actual_acceptance='incomplete')


def preflight_native_capacity(collection):
    """Child-visible native maxima only; parent checks every original source."""
    cap=collection['capacity'];runs=len(collection['invocations'])
    if runs*cap['max_entries_per_run']>cap['max_collection_entries']:
        raise ValueError('per-run entry reservations exceed joint collection capacity')
    if runs*cap['max_chunks_per_run']>cap['max_collection_chunks']:
        raise ValueError('per-run chunk reservations exceed joint collection capacity')
    # Worst-case fragmentation is bounded by event, entry and byte rotations,
    # each charged separately rather than assuming ideal full packing.
    max_events=cap['max_events_per_run'];max_entries=cap['max_entries_per_run']
    rotations=1+(max_events+255)//256+(max_entries+127)//128+(max_events*2048+1024**2-1)//(1024**2-6144)
    if rotations>cap['max_chunks_per_run']:raise ValueError('fragmented native chunk preflight exceeds declared cap')
    native=runs*(max_events*2048+cap['max_chunks_per_run']*6144+
        ((cap['max_chunks_per_run']+127)//128+17)*65536)
    if 2*native>cap['max_collection_bytes']:raise ValueError('joint native copies exceed collection artifact bound')
    return dict(collection_sha256=digest(collection),predicted_native_bytes=native,
        invocation_count=runs,full_source_preflight=False,actual_acceptance='incomplete')
