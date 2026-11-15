"""Actual owned ABC input files, synthetic pixels/completions, no model claims.

This is a test-only builder. It hashes existing installations and creates only
files under the caller's empty, task-owned directory; it never installs assets.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import metadata as package_metadata
from itertools import permutations
from pathlib import Path
import sysconfig
import unittest

from benchmarks.recorded_profiles import FixtureBGR, PCMLevel
from benchmarks.recording_dependencies import distribution_descriptor, verify_dependencies
from benchmarks.recording_native_schema import CAPS, digest, encoded
from benchmarks import recording_transports
from test_recording_runtime_contract import fixture as policy_fixture


@dataclass(frozen=True)
class RuntimeFixture:
    root: Path
    execution_sha256s: tuple
    documents: dict
    source_bundles: dict
    dependency_inventory: object

    def declarations(self, recording_index=0):
        d = self.documents
        rid = d['runtime/collection.json']['recordings'][recording_index]['recording_id']
        bundle = self.source_bundles[rid]
        return deepcopy(dict(plan=d['runtime/plan.json'], collection=d['runtime/collection.json'],
            scope_protocol=d['runtime/scopes.json'], instrumentation_protocol=d['runtime/instrumentation.json'],
            runtime_policy=d['runtime/policy.json'], source_manifest=bundle['manifest'],
            tasks=bundle['tasks'], profiles=bundle['profiles']))


def installed_dependencies():
    """Verify existing visual dependencies through the production owned reader."""
    # Core CI intentionally has no vision extras. Missing packages are optional;
    # installed-but-invalid metadata must still fail the owned production reader.
    for name in ('numpy', 'opencv-python'):
        try:
            package_metadata.distribution(name)
        except package_metadata.PackageNotFoundError as exc:
            raise unittest.SkipTest('existing optional vision distributions required') from exc
    roots = (Path(sysconfig.get_paths()['data']),)
    metadata = (Path(sysconfig.get_paths()['purelib']),)
    entries = []
    for name in ('numpy', 'opencv-python'):
        descriptor = distribution_descriptor(name, allowed_roots=roots, metadata_roots=metadata)
        entries.append(dict(distribution=name, version=descriptor['version'],
                            installation_sha256=digest(descriptor)))
    inventory = verify_dependencies(entries, allowed_roots=roots, metadata_roots=metadata)
    return entries, inventory


def build_runtime_fixture(root, *, count=24, duration_ns=700_000_000,
                          tasks_per_recording=1, phase_ns=500_000_000,
                          fixture_delay_ms=0, dependency_evidence=None,
                          include_audio=False, vision_delay_ms=0, include_fill=False,
                          task_snapshot_ns=None):
    """Freeze balanced recording arms and actual files before any execution.

    Defaults yield one1Hz tick at500ms and a task snapshot at600ms. A phase0
    variant deliberately allows a truthful initial no-frame outcome. Multiple
    tasks overlap by construction; they never count as extra recordings.
    dependency_evidence may reuse a previously verified tuple within one test
    setup; production activation still re-verifies the frozen installation.
    """
    if type(count) is not int or not 20 <= count <= 32:
        raise ValueError('20..32 original recordings required; sample gate is unchanged')
    if type(tasks_per_recording) is not int or not 1 <= tasks_per_recording <= 8:
        raise ValueError('one through eight actual tasks per recording required')
    if type(duration_ns) is not int or not 200_000_000 <= duration_ns <= 3_000_000_000:
        raise ValueError('bounded fixture recording duration required')
    if type(phase_ns) is not int or not 0 <= phase_ns < min(duration_ns, 10**9):
        raise ValueError('actual phase within recording and one-second period required')
    if type(fixture_delay_ms) is not int or not 0 <= fixture_delay_ms <= 500:
        raise ValueError('bounded explicit synthetic callback delay required')
    if type(include_audio) is not bool or type(vision_delay_ms) is not int or not 0 <= vision_delay_ms <= 500:
        raise ValueError('explicit audio selection and bounded vision delay required')
    if type(include_fill) is not bool:raise ValueError('explicit fill selection required')
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
        raise ValueError('empty caller-owned existing directory required')
    root.chmod(0o700)
    entries, inventory = installed_dependencies() if dependency_evidence is None else dependency_evidence
    u, plan, latency, template_bundles = policy_fixture()
    base = next(iter(template_bundles.values()))
    original_task = deepcopy(plan['comparison']['tasks'][0])
    plan['comparison']['tasks'] = []
    plan['comparison']['held_out'] = False
    plan['baseline_sampling']['phase_ms'] = phase_ns // 10**6
    if phase_ns % 10**6:
        raise ValueError('baseline phase must have exact millisecond representation')
    plan['baseline_sampling']['audio_policy'] = ('Same PCM level features in all arms' if include_audio
                                                else 'No audio source in this declared fixture')
    u['recordings'] = []
    for arm in u['arms']:
        arm['recording_policies'] = []
    fill_config=None
    if include_fill:
        from gamst.providers.fill import _configuration
        from gamst.fill_conflict_rule import RULE_SHA,PREDICATE
        fill_config=_configuration(width=64,height=64,roi=[8,8,56,56],
            hsv_lower=[110,100,100],hsv_upper=[130,255,255],volume_curve_ml=[[0,0],[1,100]],
            calibration_id='cal1',container_id='cup1',coordinate_frame='pixels')
        c=u['arms'][2]
        controller=deepcopy(c['controllers'][0]);controller['controller_id']='fill'
        controller['policy']['cooldown_ms']=0;c['controllers'].append(controller)
        for lid,role in (('fill_small','fill_small'),('fill_strong','fill_strong')):
            c['lanes'].append(dict(lane_id=lid,role=role,adapter_id='fixture',controller_id='fill',
                maximum_charge=5,ttl_ms=1000,binding_id=lid))
    question = 'Describe the observed synthetic pixels; do not infer model accuracy.'
    prompt = dict(schema_version=1, kind='recording_visual_prompt', question=question)
    provider_config = dict(delay_ms=vision_delay_ms)
    budget = dict(unit='fixture', pricing_revision='1', ceiling=10000,
                  max_calls=256, max_inflight=4, receipt_limit=128)
    bundles = {}; crows = []; payloads = {}
    snapshot_ns = max(phase_ns + 50_000_000, duration_ns - 100_000_000)
    if task_snapshot_ns is not None:
        if type(task_snapshot_ns) is not int or not 0<=task_snapshot_ns<duration_ns:
            raise ValueError('explicit original snapshot must be within recording')
        snapshot_ns=task_snapshot_ns
    if snapshot_ns >= duration_ns:
        raise ValueError('fixture needs snapshot after tick and before recording end')
    for n in range(count):
        rid = f'original-{n}'
        manifest = deepcopy(base['manifest'])
        manifest.update(recording_id=rid, duration_ns=duration_ns)
        manifest['availability'][0]['intervals'][0]['end_ns'] = duration_ns
        if include_fill:manifest['sources'][0]['calibration_id']='cal1'
        # Three real captures include a brief event before the periodic tick.
        # Capture identities and immutable bytes are unique original inputs.
        offsets = sorted(set((0, duration_ns // 4, duration_ns // 2)))
        manifest['records'] = []
        for j, offset in enumerate(offsets):
            data = bytes(((n + j) % 256, (n + 2*j) % 256, (n + 3*j) % 256)) * (32*32)
            if include_fill:data=bytes(64*32*3)+bytes((255,0,0))*(64*32)
            path = f'frame-{j}.bin'; payloads[f'recordings/{n:02}/payload/{path}'] = data
            manifest['records'].append(dict(source_id='camera', sensor_id='video', sequence=j+1,
                capture_ns=manifest['origin_capture_ns']+offset, offset_ns=offset,
                format='bgr8_hwc', stream_epoch='old', payload_path=path,
                payload_bytes=len(data), payload_sha256=hashlib.sha256(data).hexdigest(),
                shape=[64,64,3] if include_fill else [32,32,3]))
        if include_audio:
            manifest['sources'].append(dict(source_id='mic',sensor_id='microphone',stream_epoch='old',
                format='pcm_s16le_mono',modality='audio',calibration_id=None))
            manifest['availability'].append(dict(source_id='mic',intervals=[dict(
                start_ns=0,end_ns=duration_ns,state='available')]))
            # Complete 50ms windows only; never stretch the last PCM packet.
            for j,offset in enumerate(range(0,duration_ns-50_000_000+1,50_000_000)):
                data=(100+n+j).to_bytes(2,'little',signed=True)*800
                path=f'audio-{j}.bin';payloads[f'recordings/{n:02}/payload/{path}']=data
                manifest['records'].append(dict(source_id='mic',sensor_id='microphone',sequence=j+1,
                    capture_ns=manifest['origin_capture_ns']+offset,offset_ns=offset,
                    format='pcm_s16le_mono',stream_epoch='old',payload_path=path,
                    payload_bytes=len(data),payload_sha256=hashlib.sha256(data).hexdigest(),
                    sample_rate=16000,sample_count=800))
            manifest['records'].sort(key=lambda r:(r['offset_ns'],r['source_id'],r['sequence']))
        tasks = []
        for k in range(tasks_per_recording):
            tid = f'{n}-{k}'; task_input = dict(question=question)
            tasks.append(dict(task_id=tid, input=task_input, input_sha256=digest(task_input),
                opens_ns=max(0, snapshot_ns-100_000_000), snapshot_ns=snapshot_ns,
                deadline_ns=duration_ns, window='1h'))
            row = deepcopy(original_task); row.update(id=tid, input_sha256=digest(task_input))
            plan['comparison']['tasks'].append(row)
        profile_arms = []
        for arm in u['arms']:
            name = arm['arm']
            policy = deepcopy(policy_fixture()[0]['arms']['ABC'.index(name)]['recording_policies'][0])
            policy['recording_id'] = rid
            delivery = policy['deliveries'][0]
            delivery['config_sha256'] = None if name == 'A' else digest(provider_config)
            if include_audio:
                policy['deliveries'].append(dict(source_id='mic',profile='pcm_level',
                    config_sha256=digest({}),binding_id='audio'))
            if policy['periodic'] is not None:
                policy['periodic'].update(phase_ns=phase_ns, prompt_policy_sha256=digest(prompt))
            if include_fill and name=='C':
                policy['fill_chain']=dict(implementation='fill_conflict_v1',source_id='camera',
                    selection_period_ns=duration_ns+1,phase_ns=phase_ns,max_lateness_ns=50_000_000,
                    fill_config_sha256=digest(fill_config),rule_sha256=RULE_SHA,
                    small_lane_id='fill_small',strong_lane_id='fill_strong',strong_opt_in=True,
                    charge_ceilings=dict(initial=5,reinspection=5,strong=5),chain_ceiling=15,
                    pricing_revision='1',unit='fixture',stage_timeout_ms=1000,retention_ms=5000)
            arm['recording_policies'].append(policy)
            profile_arms.append(dict(arm=name, deliveries=deepcopy(policy['deliveries'])))
            for controller in arm['controllers']:
                controller['budget_sha256'] = digest(budget)
        profiles = dict(schema_version=2, kind='recording_runtime_profiles', recording_id=rid,
                        source_manifest_sha256=digest(manifest), arms=profile_arms)
        bundles[rid] = dict(manifest=manifest, tasks=tasks, profiles=profiles)
        declaration = deepcopy(policy_fixture()[0]['recordings'][0])
        declaration.update(recording_id=rid, source_manifest_sha256=digest(manifest),
                           tasks_sha256=digest(tasks), profiles_sha256=digest(profiles))
        if include_fill:declaration['source_bindings'][0]['coordinate_frame']='pixels'
        if include_audio:
            declaration['source_bindings'].append(dict(source_id='mic',coordinate_frame=None,channels=None))
        u['recordings'].append(declaration)
        crows.append(dict(recording_id=rid, source_manifest_sha256=digest(manifest),
            tasks_sha256=digest(tasks), profiles_sha256=digest(profiles),
            task_ids=[t['task_id'] for t in tasks], duration_ns=duration_ns))
    invocations = []; orders = tuple(permutations('ABC'))
    for n in range(count):
        for arm in orders[n % 6]:
            invocations.append(dict(ordinal=len(invocations), recording_index=n,
                                    arm=arm, stream_id=f'stream-{n}-{arm}'))
    responses = dict(schema_version=1, kind='fixture_completions', rows=[])
    for inv in invocations:
        calls = [('answer', tasks_per_recording)]
        if inv['arm'] != 'B':
            calls.append(('periodic', len(range(phase_ns, duration_ns, 10**9))))
        if include_fill and inv['arm']=='C':calls.extend((('fill_small',2),('fill_strong',1)))
        for lane, maximum in calls:
            for ordinal in range(maximum):
                text='Synthetic pixel fixture observation.'
                if lane in ('fill_small','fill_strong'):
                    text=encoded(dict(schema_version=1,predicate=PREDICATE,status='estimated',
                                      interval=[.48,.52] if lane=='fill_strong' else [.1,.2])).decode()
                responses['rows'].append(dict(invocation_ordinal=inv['ordinal'], lane_id=lane,
                    call_ordinal=ordinal, text=text,
                    usage=None, synthetic_receipt_sha256=None))
    adapter = u['adapters'][0]
    adapter.update(implementation_sha256=hashlib.sha256(Path(recording_transports.__file__).read_bytes()).hexdigest(),
                   dependency_inventory_sha256=inventory.sha256)
    adapter['config'].update(responses_sha256=digest(responses), delay_ms=fixture_delay_ms)
    u['comparison_plan_sha256'] = digest(plan)
    latency['comparison_plan_sha256'] = digest(plan)
    u['latency_policy_sha256'] = digest(latency)
    collection = dict(schema_version=2, kind='recording_collection', comparison_plan_sha256=digest(plan),
        runtime_policy_sha256=digest(u), recordings=crows, arms=[], invocations=invocations,
        capacity=dict(CAPS, max_entries_per_run=128 if include_audio else 64,
                      max_events_per_run=2048 if include_audio else 512,
                      max_chunks_per_run=16 if include_audio else 8))
    spec = FixtureBGR.capability(vision_delay_ms)
    for arm in u['arms']:
        bindings = []
        for ordinal, lane in enumerate(arm['lanes']):
            bindings.append(dict(binding_id=lane['binding_id'], provider=adapter['provider'],
                model_revision=adapter['model_revision'], role='text_llm' if lane['role']=='text_answer' else 'strong_vlm' if lane['role']=='fill_strong' else 'small_vlm',
                config_sha256=digest(adapter['config']), source_ids=['camera','mic'] if include_audio else ['camera'],
                entry_boundary='async_client_callback', operation_kinds=['semantic'], semantic_lane=ordinal))
        if arm['arm'] != 'A':
            bindings.append(dict(binding_id='vision', provider=spec.provider,
                model_revision=spec.version, role='signal', config_sha256=digest(provider_config),
                source_ids=['camera'], entry_boundary='in_process_backend', operation_kinds=['process'], semantic_lane=None))
        if include_audio:
            bindings.append(dict(binding_id='audio',provider=PCMLevel.spec.provider,
                model_revision=PCMLevel.spec.version,role='signal',config_sha256=digest({}),
                source_ids=['mic'],entry_boundary='in_process_backend',operation_kinds=['process'],semantic_lane=None))
        if include_fill and arm['arm']=='C':
            from gamst.providers.fill import CalibratedColorFill
            fill_spec=CalibratedColorFill.capability(**fill_config)
            bindings.append(dict(binding_id='fill_measurement',provider=fill_spec.provider,
                model_revision=fill_spec.version,role='signal',config_sha256=digest(fill_config),
                source_ids=['camera'],entry_boundary='in_process_backend',operation_kinds=['process'],semantic_lane=None))
        collection['arms'].append(dict(arm=arm['arm'], config_sha256=digest(plan['systems'][arm['arm']]),
                                       provider_bindings=bindings))
    scopes = []
    for inv in invocations:
        scopes.append(dict(invocation_ordinal=inv['ordinal'], scope_index=0,
                           scope_kind='stream', task_id=None, input_sha256=None))
        for ordinal, task in enumerate(bundles[crows[inv['recording_index']]['recording_id']]['tasks'], 1):
            scopes.append(dict(invocation_ordinal=inv['ordinal'], scope_index=ordinal,
                               scope_kind='task', task_id=task['task_id'], input_sha256=task['input_sha256']))
    q = dict(schema_version=1, kind='recording_scope_protocol', collection_sha256=digest(collection),
        comparison_plan_sha256=digest(plan), stream_protocol_sha256=None, allocation_protocol_sha256=None,
        recording_reconciliation_sha256=None, cost_card_sha256=None,
        retention_policy='explicit_external_evaluation_hashes', retain_financial_documents=False, scopes=scopes)
    # Freeze startup allowance before all dependent hashes: the real child
    # validates original inputs before HELLO. Playback/task clocks are unchanged.
    i = dict(schema_version=2, kind='recording_instrumentation', scope_protocol_sha256=digest(q),
             collection_sha256=digest(collection), hello_timeout_ms=10_000, close_timeout_ms=1000,
             fd_environment_key='GAMST_RECORDING_FD_V2')
    documents = {'configurations/000.json':provider_config, 'configurations/001.json':prompt, 'responses/000.json':responses,
                 'budgets/000.json':budget}
    if include_audio:documents['configurations/002.json']={}
    if include_fill:documents['configurations/003.json']=fill_config
    def ref(path, value):
        data = encoded(value)+b'\n'
        return dict(path=path, file_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    assets = dict(schema_version=1, kind='recording_runtime_assets', assets=[],
                  dependencies=deepcopy(entries), resident_state=[])
    for aid, kind, path in (('vision-config','provider_config','configurations/000.json'),
                            ('visual-prompt','system_template','configurations/001.json'),
                            ('fixture-responses','fixture_responses','responses/000.json')):
        r = ref(path, documents[path])
        assets['assets'].append(dict(asset_id=aid, kind=kind, path=path,
                                    sha256=r['file_sha256'], bytes=r['bytes']))
    if include_audio:
        r=ref('configurations/002.json',{})
        assets['assets'].append(dict(asset_id='audio-config',kind='provider_config',path=r['path'],
                                    sha256=r['file_sha256'],bytes=r['bytes']))
    if include_fill:
        r=ref('configurations/003.json',fill_config)
        assets['assets'].append(dict(asset_id='fill-config',kind='provider_config',path=r['path'],
                                    sha256=r['file_sha256'],bytes=r['bytes']))
    documents.update({'runtime/'+name+'.json':value for name,value in dict(policy=u, plan=plan,
        latency=latency, collection=collection, scopes=q, instrumentation=i, assets=assets).items()})
    index = dict(schema_version=1, kind='recording_runtime_input_sources', collection_sha256=digest(collection),
                 runtime_policy_sha256=digest(u), recordings=[])
    for n,(rid,bundle) in enumerate(bundles.items()):
        row = dict(recording_id=rid, recording_index=n)
        for kind,value in bundle.items():
            path = f'recordings/{n:02}/{kind}.json'; documents[path] = value; row[kind] = ref(path,value)
        index['recordings'].append(row)
    documents['runtime/source-bundles.json'] = index
    executions = []
    for inv in invocations:
        execution = dict(schema_version=1, kind='recording_execution', runtime_policy_sha256=digest(u),
            collection_sha256=digest(collection), scope_protocol_sha256=digest(q),
            instrumentation_protocol_sha256=digest(i), **inv,
            execution_namespace='execution-'+str(inv['ordinal']), cache_namespace='cache-'+str(inv['ordinal']),
            asset_inventory_sha256=digest(assets))
        execution['invocation_ordinal'] = execution.pop('ordinal')
        documents[f"invocations/{inv['ordinal']:02}/execution.json"] = execution
        executions.append(digest(execution))
    for path,value in documents.items():
        target=root/path; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output: output.write(encoded(value)+b'\n')
    for path,data in payloads.items():
        target=root/path; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output: output.write(data)
    return RuntimeFixture(root, tuple(executions), documents, bundles, inventory)
