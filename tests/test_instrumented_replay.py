import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.instrumented_replay import run
from benchmarks.perception_comparison import plan_digest
from benchmarks.runtime_channel import FD_ENVIRONMENT_KEY
from benchmarks.runtime_channel import ParentChannel,frame,receive,RuntimeChannelError
from test_stream_attempts import fixture

PROJECT=Path(__file__).resolve().parents[1]
SCRIPT='''import asyncio,json,os,sys
from benchmarks.runtime_channel import ChildChannel
from benchmarks.runtime_instrumentation import EvaluationSession
from benchmarks.perception_comparison import plan_digest
from test_text_context_call import Calls
async def main():
    channel=await ChildChannel.from_environment()
    request=json.loads(sys.stdin.readline())
    assert plan_digest(request)==channel.bootstrap['request_sha256']
    assert not os.get_inheritable(int(os.environ['GAMST_EVALUATION_FD']))
    session=EvaluationSession.from_channel(channel)
    helper=Calls();await helper.asyncSetUp()
    mode=request['input']['mode']
    async def callback(payload):
        if mode=='timeout':await asyncio.sleep(60)
        if mode=='error':raise ValueError('PRIVATE BACKEND EXCEPTION')
        return dict(text='PRIVATE BACKEND ANSWER',usage=None)
    status='success'
    try:
        adapter=helper.adapter(callback,evaluation_observer=session)
        rid,record=await helper.queued()
        try:await adapter.execute(rid,record,'PRIVATE QUESTION',maximum_charge=None if mode=='denied' else 1)
        except ValueError:status='error'
    finally:await helper.asyncTearDown()
    await session.close()
    print(json.dumps(dict(schema_version=1,status=status,answer='source answer',measurements={'request':request})),flush=True)
asyncio.run(main())
'''
VISUAL_SCRIPT=SCRIPT.replace('from test_text_context_call import Calls','from test_visual_context_call import Calls').replace(
    'rid,record=await helper.queued()','rid,record,pixels=await helper.queued()').replace(
    "adapter.execute(rid,record,'PRIVATE QUESTION',", "adapter.execute(rid,record,pixels,'PRIVATE QUESTION',")
SMOL_SCRIPT='''import asyncio,json,os,sys
from benchmarks.runtime_channel import ChildChannel
from benchmarks.runtime_instrumentation import EvaluationSession
from benchmarks.perception_comparison import plan_digest
from test_smolvlm_call import Calls,SmolVLMCalls,CONFIG
from test_smolvlm_context import WORKER
async def main():
    channel=await ChildChannel.from_environment()
    request=json.loads(sys.stdin.readline());assert plan_digest(request)==channel.bootstrap['request_sha256']
    session=EvaluationSession.from_channel(channel);helper=Calls();await helper.asyncSetUp()
    try:
        check="import os\\ntry:os.fstat(int(os.environ['GAMST_EVALUATION_FD']))\\nexcept OSError:pass\\nelse:raise AssertionError('nested worker inherited evaluation descriptor')\\n"
        with helper.fixture(code=check+WORKER.replace('DELAY','0')):
            human,controller,spec=helper.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,evaluation_observer=session) as adapter:
                    rid,record,pixels=await helper.request(human,controller,spec)
                    await adapter.execute(rid,record,pixels,maximum_charge=2)
    finally:await helper.asyncTearDown()
    await session.close()
    print(json.dumps(dict(schema_version=1,status='success',answer='source answer',measurements={'request':request})),flush=True)
asyncio.run(main())
'''


def setup(root,mode='ok',script=SCRIPT,*,provider='test.client',revision='v1',extra_input=None,timeout_ms=3000):
    plan,spec,inputs,stream,reconciliation=fixture(root)
    if script==SMOL_SCRIPT:
        from humansense.provider_profiles import capability
        from humansense.providers.smolvlm_native import configuration
        from test_smolvlm_native import CONFIG
        contract=capability('smolvlm2.500m.local',dict(configuration(CONFIG),window_frames=1))
        provider,revision=contract.provider,contract.version
    plan['comparison']['cost']['rate_card']={'request':dict(provider=provider,revision=revision,unit='request',category='request',per_unit=1)}
    inputs[0]['input']['mode']=mode
    if extra_input is not None:inputs[0]['input'].update(extra_input)
    plan['comparison']['tasks'][0]['input_sha256']=plan_digest(inputs[0]['input'])
    path=root/'instrumented-child.py';path.write_text(script)
    python=str(Path(sys.executable).resolve())
    command=dict(argv=[python,str(path)],cwd=str(root),environment={'PYTHONPATH':str(PROJECT)+os.pathsep+str(PROJECT/'tests')},
        artifacts={python:hashlib.sha256(Path(python).read_bytes()).hexdigest(),str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    spec['runner'].update(commands={arm:copy.deepcopy(command) for arm in 'ABC'},timeout_ms=timeout_ms)
    spec['comparison_plan_sha256']=plan_digest(plan)
    stream.update(comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec))
    targets=[];rules=[]
    for lane in stream['streams']:
        sid=lane['id'];arm=lane['arm']
        targets.append(dict(id=arm+'-task',stream_id=sid,kind='task',task_id='0',windows=[]))
        targets.extend(dict(id=arm+'-'+kind,stream_id=sid,kind=kind,task_id=None,windows=[]) for kind in ('startup','idle','overhead'))
        for cid,req in lane['requirements'].items():
            rules.extend(dict(stream_id=sid,component_id=cid,**resource,method='direct',target_id=arm+'-overhead') for resource in req['lanes'])
    allocation=dict(schema_version=1,comparison_plan_sha256=plan_digest(plan),stream_protocol_sha256=plan_digest(stream),
        rounding='largest_remainder_v1',tie_break='unicode_codepoint_target_id',duration_model='uniform_line_quantity_over_interval',targets=targets,rules=rules)
    reconciliation.update(comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec),
        stream_protocol_sha256=plan_digest(stream),allocation_protocol_sha256=plan_digest(allocation))
    instrumentation=dict(schema_version=1,comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec),
        stream_protocol_sha256=plan_digest(stream),allocation_protocol_sha256=plan_digest(allocation),
        reconciliation_protocol_sha256=plan_digest(reconciliation),transport='socketpair_be32_json_v1',hello_timeout_ms=1000,
        fd_environment_key=FD_ENVIRONMENT_KEY,retention_policy='external_owner_retains_task_and_evidence_hashes',phase='unknown')
    return plan,spec,inputs,stream,allocation,reconciliation,instrumentation


async def collect(values,out):
    return await run(*values,out,**dict(zip(('expected_plan_sha256','expected_collection_sha256','expected_inputs_sha256',
        'expected_stream_protocol_sha256','expected_allocation_protocol_sha256','expected_reconciliation_protocol_sha256',
        'expected_instrumentation_protocol_sha256'),map(plan_digest,values))))


class Collector(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='.');self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name).resolve()
    def validate_runtime(self,runtime,*,events=None,reports=None,summary=None):
        channel=ParentChannel(runtime['bootstrap'])
        try:
            previous='0'*64
            for index,item in enumerate(copy.deepcopy(runtime['journal'] if events is None else events)):
                item.update(sequence=index,previous_sha256=previous);item.pop('sha256');item['sha256']=plan_digest(item)
                channel._event(item);previous=item['sha256']
            for item in copy.deepcopy(runtime['native_reports'] if reports is None else reports):
                item.pop('sha256');item['sha256']=plan_digest(item);channel._report(item)
            channel._summary(copy.deepcopy(runtime['terminal']['summary'] if summary is None else summary))
        finally:channel.close()
    async def test_actual_sdk_automatic_attempts_original_source_and_private_sidecar(self):
        values=setup(self.root);result=await collect(values,self.root/'run')
        rows=[json.loads(line) for line in (self.root/'run/source-traces.jsonl').read_text().splitlines()]
        sides=[json.loads(line) for line in (self.root/'run/runtime-sidecars.jsonl').read_text().splitlines()]
        self.assertEqual(result['records'],3)
        for row,side in zip(rows,sides):
            self.assertEqual(row['status'],'success',row)
            self.assertEqual(row['request_sha256'],plan_digest(row['response']['measurements']['request']))
            self.assertEqual(side['original_trace_sha256'],row['trace_sha256'])
            self.assertEqual(side['original_response_sha256'],row['response_sha256'])
            raw=json.dumps(dict(schema_version=1,status='success',answer='source answer',
                measurements={'request':row['response']['measurements']['request']})).encode()
            self.assertEqual(row['response_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertNotEqual(row['response_sha256'],plan_digest(row['response']))
            runtime=side['runtime'];self.assertTrue(runtime['instrumentation_complete'],runtime)
            self.assertEqual(runtime['terminal']['summary']['provider_counters'][0]['count'],1)
            attempt=runtime['native_reports'][0]['attempts'][0]
            self.assertIsNone(attempt['cost_quantities']);self.assertIsNone(attempt['line_claims'])
            self.assertNotIn('PRIVATE',json.dumps(side))
            self.assertNotIn('GAMST_EVALUATION_FD',json.dumps(row['response']['measurements']['request']))
        completion=json.loads((self.root/'run/runtime-complete.json').read_text())
        self.assertTrue(completion['instrumentation_complete']);self.assertEqual(completion['actual_acceptance'],'incomplete')
        self.assertNotIn('monetary_cost',result)
        with self.assertRaises(FileExistsError):await collect(values,self.root/'run')
    async def test_denial_is_not_an_attempt_and_error_is_a_real_attempt(self):
        for mode,count in (('denied',0),('error',1)):
            sub=self.root/mode;sub.mkdir();values=setup(sub,mode);await collect(values,sub/'run')
            sides=[json.loads(line) for line in (sub/'run/runtime-sidecars.jsonl').read_text().splitlines()]
            for side in sides:
                runtime=side['runtime'];self.assertTrue(runtime['instrumentation_complete'],runtime)
                self.assertEqual(len(runtime['native_reports'][0]['attempts']),count)
    async def test_denial_preserves_native_attempt_credit_then_executes_same_request(self):
        script=SCRIPT.replace("try:await adapter.execute(rid,record,'PRIVATE QUESTION',maximum_charge=None if mode=='denied' else 1)",
            "try:\n            await adapter.execute(rid,record,'PRIVATE QUESTION',maximum_charge=None)\n            await adapter.execute(rid,record,'PRIVATE QUESTION',maximum_charge=1)")
        values=setup(self.root,script=script);await collect(values,self.root/'run')
        for line in (self.root/'run/runtime-sidecars.jsonl').read_text().splitlines():
            runtime=json.loads(line)['runtime'];self.assertTrue(runtime['instrumentation_complete'],runtime)
            self.assertEqual(sum(e['event']=='call_capacity_reserved' for e in runtime['journal']),2)
            self.assertEqual(len(runtime['native_reports'][0]['attempts']),1)
    async def test_parent_separates_cumulative_reservation_bound_from_native_attempt_bound(self):
        values=setup(self.root,'denied');await collect(values,self.root/'run')
        runtime=json.loads((self.root/'run/runtime-sidecars.jsonl').read_text().splitlines()[0])['runtime']
        reserve=next(e for e in runtime['journal'] if e['event']=='call_capacity_reserved')
        release=next(e for e in runtime['journal'] if e['event']=='call_capacity_released')
        events=[]
        for index in range(1024):
            for original in (reserve,release):
                item=copy.deepcopy(original);item.update(operation_id=f'{index:032x}',monotonic_ns=reserve['monotonic_ns']);events.append(item)
        self.validate_runtime(runtime,events=events)
        summary=copy.deepcopy(runtime['terminal']['summary']);summary['incomplete_reasons']=['call_reservation_capacity_exhausted','native_attempt_capacity_exhausted']
        self.validate_runtime(runtime,events=events,summary=summary)
        events.extend([dict(copy.deepcopy(item),operation_id='f'*32) for item in events[-2:]])
        with self.assertRaises(ValueError):self.validate_runtime(runtime,events=events,summary=summary)
    async def test_actual_visual_and_owned_smol_sdk_paths_preserve_one_semantic_charge(self):
        try:import cv2,numpy
        except ImportError:self.skipTest('optional vision fixture dependencies unavailable; no install')
        for name,script in (('visual',VISUAL_SCRIPT),('smol',SMOL_SCRIPT)):
            sub=self.root/name;sub.mkdir();values=setup(sub,script=script)
            await collect(values,sub/'run')
            sides=[json.loads(line) for line in (sub/'run/runtime-sidecars.jsonl').read_text().splitlines()]
            for side in sides:
                runtime=side['runtime'];self.assertTrue(runtime['instrumentation_complete'],runtime)
                self.assertEqual(len(runtime['native_reports'][0]['attempts']),1)
                self.assertEqual(len(runtime['native_reports'][1]['attempts']),0)
                summary=runtime['terminal']['summary'];self.assertEqual(summary['provider_counters'][0]['count'],1)
                if name=='smol':
                    worker=summary['worker_measurements'][0];self.assertTrue(worker['reap_confirmed'])
                    self.assertEqual(worker['observed_exchange_union_ns']+worker['observed_no_exchange_gap_ns'],worker['residency_elapsed_ns'])
                    self.assertTrue(any(o['parent_operation_id'] is not None for o in worker['operations']))
    async def test_parent_rejects_rehashed_report_injection_summary_changes_and_trailing_terminal(self):
        values=setup(self.root);await collect(values,self.root/'run')
        side=json.loads((self.root/'run/runtime-sidecars.jsonl').read_text().splitlines()[0]);runtime=side['runtime']
        def parent():
            channel=ParentChannel(runtime['bootstrap'])
            for item in runtime['journal']:channel._event(item)
            return channel
        original=runtime['native_reports'][0]
        for key,value in (('input_sha256','f'*64),('configuration_sha256','f'*64),('recording_sha256','f'*64),
                          ('schema_version',True),('rejected',1),('scope','f'*32)):
            item=copy.deepcopy(original);item[key]=value;item.pop('sha256');item['sha256']=plan_digest(item)
            channel=parent()
            try:
                with self.subTest(key=key),self.assertRaises(ValueError):channel._report(item)
            finally:channel.close()
        item=copy.deepcopy(original);item['attempts'][0]['status']='running';item.pop('sha256');item['sha256']=plan_digest(item)
        channel=parent()
        try:
            with self.assertRaises(ValueError):channel._report(item)
        finally:channel.close()
        for change in ({'PRIVATE_BODY':'raw text'},{'monetary_cost':0},{'measurement_truth_verified':True},
                       {'provider_counters':[]},{'closed':1},{'required_boundary_lanes':[]}):
            channel=parent()
            try:
                for report in runtime['native_reports']:channel._report(report)
                with self.subTest(change=change),self.assertRaises(ValueError):channel._summary(dict(runtime['terminal']['summary'],**change))
            finally:channel.close()
        channel=ParentChannel(runtime['bootstrap']);channel.prepare({},2)
        peer=channel.child.dup();peer.setblocking(False);channel.spawned();reader=asyncio.create_task(channel.collect())
        bootstrap=await receive(peer)
        await asyncio.get_running_loop().sock_sendall(peer,frame(dict(schema_version=1,type='hello',bootstrap=bootstrap)))
        sequence=0;previous='0'*64
        payloads=[('event',e) for e in runtime['journal']]+[('report',r) for r in runtime['native_reports']]+[('terminal',runtime['terminal'])]
        for kind,payload in payloads+[('event',runtime['journal'][0])]:
            envelope=dict(schema_version=1,type=kind,nonce=bootstrap['nonce'],sequence=sequence,previous_sha256=previous,payload=payload)
            envelope['sha256']=plan_digest(envelope);previous=envelope['sha256'];sequence+=1
            await asyncio.get_running_loop().sock_sendall(peer,frame(envelope))
        peer.close()
        with self.assertRaises(RuntimeChannelError):await reader
        channel.close()
        self.assertIsNotNone(channel.terminal);self.assertFalse(channel.snapshot()['instrumentation_complete'])
    async def test_native_attempt_crossrefs_and_unique_ordered_call_milestones(self):
        values=setup(self.root);await collect(values,self.root/'run')
        runtime=json.loads((self.root/'run/runtime-sidecars.jsonl').read_text().splitlines()[0])['runtime']
        self.validate_runtime(runtime)
        attempt=runtime['native_reports'][0]['attempts'][0]
        entry=next(e for e in runtime['journal'] if e['event']=='backend_entry_admission')
        finish=next(e for e in runtime['journal'] if e['event']=='backend_invocation_finished')
        # Native timings use a relative perf-counter origin; journal is absolute.
        self.assertLess(attempt['started_ns'],entry['monotonic_ns'])
        for key,value in (('prompt_sha256','f'*64),('evidence_sha256','f'*64),('execution_sha256','f'*64),
                          ('role','strong_vlm'),('phase','cold'),
                          ('ended_ns',attempt['started_ns']+finish['monotonic_ns']-entry['monotonic_ns']+1)):
            reports=copy.deepcopy(runtime['native_reports']);reports[0]['attempts'][0][key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):self.validate_runtime(runtime,reports=reports)
        reports=copy.deepcopy(runtime['native_reports']);reports[0]['attempts'][0].update(
            cost_quantities={'request':1},line_claims=['a'*64],execution_instance_sha256='f'*64)
        with self.assertRaises(ValueError):self.validate_runtime(runtime,reports=reports)
        for name in ('call_capacity_reserved','controller_dispatched','backend_entry_admission',
                     'backend_invocation_finished','call_publication_finished'):
            events=copy.deepcopy(runtime['journal']);index=next(i for i,e in enumerate(events) if e['event']==name)
            events.insert(index,copy.deepcopy(events[index]))
            with self.subTest(duplicate=name),self.assertRaises(ValueError):self.validate_runtime(runtime,events=events)
        for name in ('call_capacity_reserved','controller_dispatched','backend_entry_admission','backend_invocation_finished','call_publication_finished'):
            events=[e for e in runtime['journal'] if e['event']!=name]
            with self.subTest(missing=name),self.assertRaises(ValueError):self.validate_runtime(runtime,events=events)
        for name,key,value in (('call_publication_finished','outcome','PRIVATE ERROR'),
                               ('backend_invocation_finished','status','PRIVATE ERROR'),
                               ('private_mappings_invalidated','reason','PRIVATE ERROR'),
                               ('controller_dispatched','relationship','PRIVATE ERROR')):
            events=copy.deepcopy(runtime['journal']);next(e for e in events if e['event']==name)['fields'][key]=value
            with self.subTest(vocabulary=name),self.assertRaises(ValueError):self.validate_runtime(runtime,events=events)
    async def test_worker_identity_ownership_and_unique_milestones(self):
        try:import cv2,numpy
        except ImportError:self.skipTest('optional vision fixture dependencies unavailable; no install')
        values=setup(self.root,script=SMOL_SCRIPT);await collect(values,self.root/'run')
        runtime=json.loads((self.root/'run/runtime-sidecars.jsonl').read_text().splitlines()[0])['runtime']
        self.validate_runtime(runtime)
        life=next(e for e in runtime['journal'] if e['event']=='work_capacity_reserved' and e['fields']['kind']=='worker_lifecycle')
        for mutation in ('duplicate_worker','unknown_worker','worker_parent','unknown_parent','duplicate_milestone','unknown_status','unknown_outcome'):
            events=copy.deepcopy(runtime['journal'])
            operation=next(e for e in events if e['event']=='work_capacity_reserved' and e['fields']['kind']=='worker_operation')
            if mutation=='duplicate_worker':
                extra=copy.deepcopy(life);extra['operation_id']='f'*32;events.insert(1,extra)
            elif mutation=='unknown_worker':operation['fields']['worker_id']='unknown'
            elif mutation in ('worker_parent','unknown_parent'):
                operation['fields']['parent_operation_id']=life['operation_id'] if mutation=='worker_parent' else 'f'*32
            elif mutation=='duplicate_milestone':
                index=next(i for i,e in enumerate(events) if e['event']=='handle_acquired');events.insert(index,copy.deepcopy(events[index]))
            elif mutation=='unknown_status':next(e for e in events if e['event']=='handle_acquired')['fields']['status']='PRIVATE ERROR'
            else:next(e for e in events if e['event']=='work_finished')['fields']['outcome']='PRIVATE ERROR'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.validate_runtime(runtime,events=events)
    async def test_early_declared_billing_binds_unique_original_admission_without_authenticating(self):
        values=setup(self.root);await collect(values,self.root/'run')
        runtime=json.loads((self.root/'run/runtime-sidecars.jsonl').read_text().splitlines()[0])['runtime']
        events=copy.deepcopy(runtime['journal']);index=next(i for i,e in enumerate(events) if e['event']=='backend_entry_admission')
        entry=events[index];billing=copy.deepcopy(entry);billing.update(operation_id=None,event='external_billing_attached',
            fields=dict(operation_id=entry['operation_id'],evidence_sha256='a'*64))
        events.insert(index+1,billing)
        reports=copy.deepcopy(runtime['native_reports']);reports[0]['attempts'][0].update(
            cost_quantities={'request':1},line_claims=['b'*64],execution_instance_sha256=entry['fields']['execution_instance_sha256'])
        summary=copy.deepcopy(runtime['terminal']['summary']);summary['unresolved_attempt_billing']=False
        summary['unresolved_billing']=any(lane['kind']=='monetary' for lane in summary['required_boundary_lanes'])
        self.validate_runtime(runtime,events=events,reports=reports,summary=summary)
        self.assertIsNone(summary['monetary_cost']);self.assertFalse(summary['receipt_authenticity_verified'])
        for mode in ('before_entry','duplicate_attachment','wrong_instance','missing_attachment','duplicate_scope'):
            altered=copy.deepcopy(events);changed=copy.deepcopy(reports)
            if mode=='before_entry':altered[index],altered[index+1]=altered[index+1],altered[index]
            elif mode=='duplicate_attachment':altered.insert(index+1,copy.deepcopy(billing))
            elif mode=='wrong_instance':changed[0]['attempts'][0]['execution_instance_sha256']='f'*64
            elif mode=='missing_attachment':altered.pop(index+1)
            else:changed[1]['scope']=changed[0]['scope']
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate_runtime(runtime,events=altered,reports=changed,summary=summary)
    async def test_partial_channel_and_missing_hello_keep_source_separate(self):
        code='''import json,os,socket,sys
s=socket.socket(fileno=int(os.environ['GAMST_EVALUATION_FD']));s.sendall(b'\\x00\\x00');s.close()
r=json.loads(sys.stdin.readline());print(json.dumps(dict(schema_version=1,status='success',answer='original',measurements={'request':r})))
'''
        values=setup(self.root,script=code);await collect(values,self.root/'run')
        rows=[json.loads(line) for line in (self.root/'run/source-traces.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['status']=='error' for row in rows))
        self.assertTrue(all(row['diagnostic']=='RuntimeChannelError' for row in rows))
        runtime=json.loads((self.root/'run/runtime-complete.json').read_text())
        self.assertFalse(runtime['instrumentation_complete'])
    async def test_deadline_and_repeated_cancellation_reap_and_keep_prefix(self):
        values=setup(self.root,'timeout');values[1]['runner']['timeout_ms']=500
        values[1]['comparison_plan_sha256']=plan_digest(values[0])
        # Updating timeout changes C and all dependent frozen declarations.
        values[3]['collection_sha256']=plan_digest(values[1]);values[4]['stream_protocol_sha256']=plan_digest(values[3])
        values[5].update(collection_sha256=plan_digest(values[1]),stream_protocol_sha256=plan_digest(values[3]),allocation_protocol_sha256=plan_digest(values[4]))
        values[6].update(collection_sha256=plan_digest(values[1]),stream_protocol_sha256=plan_digest(values[3]),allocation_protocol_sha256=plan_digest(values[4]),reconciliation_protocol_sha256=plan_digest(values[5]))
        children=[];real=asyncio.create_subprocess_exec
        async def spawn(*a,**kw):
            child=await real(*a,**kw);children.append(child);return child
        with patch('asyncio.create_subprocess_exec',spawn):await collect(values,self.root/'timeout-run')
        self.assertTrue(children and all(c.returncode is not None for c in children))
        rows=[json.loads(line) for line in (self.root/'timeout-run/source-traces.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['status']=='timeout' for row in rows))
        children.clear();sub=self.root/'cancel';sub.mkdir();values=setup(sub,'timeout')
        with patch('asyncio.create_subprocess_exec',spawn):
            task=asyncio.create_task(collect(values,sub/'run'))
            for _ in range(100):
                if children:break
                await asyncio.sleep(.01)
            task.cancel();task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(children and all(c.returncode is not None for c in children))
        side=json.loads((sub/'run/runtime-sidecars.jsonl').read_text().splitlines()[0])
        self.assertIsNone(side['original_trace_sha256']);self.assertFalse(side['runtime']['instrumentation_complete'])


if __name__=='__main__':unittest.main()
