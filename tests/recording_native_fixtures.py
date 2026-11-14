"""Explicit synthetic declarations for real owned SDK/native4 fixture execution."""
from copy import deepcopy
from benchmarks.recording_native_schema import CAPS,digest,validate_recording_scope_protocol,run_context
from benchmarks.replay_clock import ReplayClock
from test_three_arm_comparison import fixture as plan_fixture
from test_recorded_replay import ReplaySDKTests


def declarations(root,provider='test.client',revision='v1',role='text_llm',boundary='async_client_callback',config=None):
    plan,_=plan_fixture('synthetic')
    source,_,profiles=ReplaySDKTests().inputs(root)
    payload={'question':'fixture question'}
    task=deepcopy(plan['comparison']['tasks'][0]);task['input_sha256']=digest(payload)
    plan['comparison']['tasks']=[task]
    tasks=[dict(task_id=task['id'],input=payload,input_sha256=digest(payload),opens_ns=0,
        snapshot_ns=100000000,deadline_ns=300000000,window='1h')]
    binding=dict(binding_id='semantic',provider=provider,model_revision=revision,role=role,
        config_sha256=digest({} if config is None else config),source_ids=['mic'],entry_boundary=boundary,
        operation_kinds=['semantic'],semantic_lane=0)
    collection=dict(schema_version=2,kind='recording_collection',comparison_plan_sha256=digest(plan),
        runtime_policy_sha256=None,recordings=[dict(recording_id=source['recording_id'],
            source_manifest_sha256=digest(source),tasks_sha256=digest(tasks),profiles_sha256=digest(profiles),
            task_ids=[task['id']],duration_ns=source['duration_ns'])],
        arms=[dict(arm=a,config_sha256=digest(plan['systems'][a]),provider_bindings=[deepcopy(binding)]) for a in 'ABC'],
        invocations=[dict(ordinal=n,recording_index=0,arm=a,stream_id='stream-'+a) for n,a in enumerate('ABC')],
        capacity=dict(CAPS,max_entries_per_run=4096,max_events_per_run=40000,max_chunks_per_run=512))
    scopes=[]
    for n in range(3):
        scopes.extend([dict(invocation_ordinal=n,scope_index=0,scope_kind='stream',task_id=None,input_sha256=None),
            dict(invocation_ordinal=n,scope_index=1,scope_kind='task',task_id=task['id'],input_sha256=task['input_sha256'])])
    scope=dict(schema_version=1,kind='recording_scope_protocol',collection_sha256=digest(collection),
        comparison_plan_sha256=digest(plan),stream_protocol_sha256=None,allocation_protocol_sha256=None,
        recording_reconciliation_sha256=None,cost_card_sha256=None,retention_policy='explicit_external_evaluation_hashes',
        retain_financial_documents=False,scopes=scopes)
    instrumentation=dict(schema_version=2,kind='recording_instrumentation',scope_protocol_sha256=digest(scope),
        collection_sha256=digest(collection),hello_timeout_ms=1000,close_timeout_ms=1000,fd_environment_key='GAMST_RECORDING_FD_V2')
    return dict(plan=plan,collection=collection,scope_protocol=scope,instrumentation_protocol=instrumentation,
        source_manifest=source,tasks=tasks,profiles=profiles)


def rehash(d):
    d['collection']['comparison_plan_sha256']=digest(d['plan'])
    d['scope_protocol'].update(collection_sha256=digest(d['collection']),comparison_plan_sha256=digest(d['plan']))
    d['instrumentation_protocol'].update(collection_sha256=digest(d['collection']),scope_protocol_sha256=digest(d['scope_protocol']))
    return d


async def session(root,**kwargs):
    from benchmarks.recording_native_writer import RecordingNativeWriter
    from benchmarks.recording_evaluation import RecordingEvaluationSession
    d=declarations(root,**kwargs);validate_recording_scope_protocol(d)
    context=run_context(d,'a'*64,0)
    writer=await RecordingNativeWriter.open(root/'native',context)
    observer=RecordingEvaluationSession.from_frozen(d,writer,run_nonce='a'*64,execution_ordinal=0,
        retain_evaluation_hashes=True)
    return observer,writer,ReplayClock(d['source_manifest'])


def matched_recordings(root,count=24):
    """Short synthetic admission fixtures; not independent real recording proof."""
    from itertools import permutations
    base=declarations(root);plan=deepcopy(base['plan']);plan['comparison']['tasks']=[]
    records=[];bundles=[]
    for n in range(count):
        source=deepcopy(base['source_manifest']);source['recording_id']='recording-'+str(n)
        task=deepcopy(base['tasks'][0]);task['task_id']=str(n)
        original=deepcopy(base['plan']['comparison']['tasks'][0]);original['id']=str(n)
        plan['comparison']['tasks'].append(original)
        records.append(dict(recording_id=source['recording_id'],source_manifest_sha256=digest(source),
            tasks_sha256=digest([task]),profiles_sha256=digest(base['profiles']),task_ids=[str(n)],duration_ns=source['duration_ns']))
        bundles.append(dict(base,source_manifest=source,tasks=[task]))
    collection=deepcopy(base['collection']);collection.update(recordings=records,comparison_plan_sha256=digest(plan),invocations=[])
    collection['capacity'].update(max_entries_per_run=4,max_events_per_run=64,max_chunks_per_run=8)
    orders=list(permutations('ABC'))
    for n in range(count):
        for arm in orders[n%6]:
            collection['invocations'].append(dict(ordinal=len(collection['invocations']),recording_index=n,arm=arm,stream_id='stream-'+str(n)+'-'+arm))
    q=deepcopy(base['scope_protocol']);q['scopes']=[]
    for inv in collection['invocations']:
        n=inv['ordinal'];tid=str(inv['recording_index'])
        q['scopes'].extend([dict(invocation_ordinal=n,scope_index=0,scope_kind='stream',task_id=None,input_sha256=None),
            dict(invocation_ordinal=n,scope_index=1,scope_kind='task',task_id=tid,input_sha256=plan['comparison']['tasks'][int(tid)]['input_sha256'])])
    for bundle in bundles:
        bundle.update(plan=deepcopy(plan),collection=deepcopy(collection),scope_protocol=deepcopy(q),
            instrumentation_protocol=deepcopy(base['instrumentation_protocol']))
        rehash(bundle)
    return bundles
