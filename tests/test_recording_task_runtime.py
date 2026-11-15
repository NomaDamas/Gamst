"""Actual SDK/native4 task operations on synthetic recorded source metadata."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.recorded_source import digest
from benchmarks.recording_native_schema import run_context
from benchmarks.recording_native_writer import ArtifactBudget,RecordingNativeWriter
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_runtime_artifacts import RecordingRuntimeArtifacts,read_ref
from benchmarks.recording_task_runtime import RecordingTaskRuntime
from benchmarks.recording_transports import FixtureCompletion,TransportCapacity,TransportMemoryBudget
from benchmarks.replay_clock import ReplayClock
from gamst import Gamst,SensorMetadata,TextContextCalls,evaluation_snapshot_implementation_sha256
from gamst.call_controller import CallController
from gamst.call_policy import CallPolicy
from gamst.call_budget import CallBudget
from gamst.capabilities import CapabilitySpec,OutputField
from recording_native_fixtures import rehash
from test_recording_runtime_native import runtime_declarations


def originals(snapshots,*,delay_ms=0,queue=64,deadline_ns=1_900_000_000,parallel=False,answer_text=None):
    d,l,bundles=runtime_declarations();u=d['runtime_policy'];p=d['plan'];source=d['source_manifest']
    duration=max(2_000_000_000,max(deadline_ns) + 100_000_000 if isinstance(deadline_ns,list) else deadline_ns+100_000_000)
    source['duration_ns']=duration;source['availability'][0]['intervals'][0]['end_ns']=duration
    prototype=deepcopy(p['comparison']['tasks'][0]);p['comparison']['tasks']=[];tasks=[]
    for n,snapshot in enumerate(snapshots):
        question=dict(question='Question '+str(n));original=deepcopy(prototype);original.update(id='task'+str(n),input_sha256=digest(question))
        p['comparison']['tasks'].append(original)
        tasks.append(dict(task_id=original['id'],input=question,input_sha256=digest(question),opens_ns=0,
            snapshot_ns=snapshot,deadline_ns=deadline_ns[n] if isinstance(deadline_ns,list) else deadline_ns,window='1h'))
    d['tasks']=tasks;l['comparison_plan_sha256']=digest(p)
    profiles=d['profiles'];profiles['source_manifest_sha256']=digest(source)
    responses=dict(schema_version=1,kind='fixture_completions',rows=[dict(invocation_ordinal=0,lane_id='answer',
        call_ordinal=n,text='Answer '+str(n) if answer_text is None else answer_text,usage=None,synthetic_receipt_sha256=None) for n in range(len(tasks))])
    if parallel:
        responses['rows'].extend([dict(row,lane_id='answer2') for row in list(responses['rows'])])
        for arm in u['arms']:
            lane=dict(arm['lanes'][0],lane_id='answer2',binding_id='text2');arm['lanes'].append(lane)
            arm['recording_policies'][0]['text_lane_ids'].append('answer2')
        for arm,policy in zip(d['collection']['arms'],u['arms']):
            lane_index=len(policy['lanes'])-1
            binding=dict(arm['provider_bindings'][0],binding_id='text2',semantic_lane=lane_index)
            arm['provider_bindings'].append(binding)
    u['adapters'][0]['config'].update(responses_sha256=digest(responses),delay_ms=delay_ms)
    u.update(comparison_plan_sha256=digest(p),latency_policy_sha256=digest(l))
    u['recordings'][0].update(source_manifest_sha256=digest(source),tasks_sha256=digest(tasks),profiles_sha256=digest(profiles))
    for arm in u['arms']:
        arm['task_queue']['capacity']=queue
        arm['controllers'][0]['policy']=asdict(CallPolicy(mode='escalation',cooldown_ms=0,max_age_ms=60000))
    c=d['collection'];c['runtime_policy_sha256']=digest(u)
    c['recordings'][0].update(source_manifest_sha256=digest(source),tasks_sha256=digest(tasks),profiles_sha256=digest(profiles),
        task_ids=[t['task_id'] for t in tasks],duration_ns=source['duration_ns'])
    for arm in c['arms']:
        for binding in arm['provider_bindings']:
            if binding['operation_kinds']==['semantic']:binding['config_sha256']=digest(u['adapters'][0]['config'])
    d['scope_protocol']['scopes']=[]
    for inv in c['invocations']:
        ordinal=inv['ordinal'];d['scope_protocol']['scopes'].append(dict(invocation_ordinal=ordinal,scope_index=0,scope_kind='stream',task_id=None,input_sha256=None))
        d['scope_protocol']['scopes'].extend(dict(invocation_ordinal=ordinal,scope_index=n+1,scope_kind='task',task_id=t['task_id'],input_sha256=t['input_sha256']) for n,t in enumerate(tasks))
    rehash(d);bundles={'original':dict(manifest=source,tasks=tasks,profiles=profiles)}
    return d,l,bundles,responses


class TaskRuntime(unittest.IsolatedAsyncioTestCase):
    async def until(self,predicate):
        async def wait():
            while not predicate():await asyncio.sleep(.002)
        await asyncio.wait_for(wait(),2)

    async def setup_runtime(self,snapshots,**changes):
        d,l,bundles,responses=originals(snapshots,**changes)
        temporary=tempfile.TemporaryDirectory(dir='.');self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        human=Gamst(consent=True);self.addAsyncCleanup(human.close)
        human.register_capability(CapabilitySpec('source.fill','source.fill','video','test.source','1',(OutputField('fraction','number'),),'frame'))
        await human.push_observation('source.fill',{'fraction':.4},sensor=SensorMetadata('camera','video',1,stream_epoch='original'))
        publisher=human.create_evaluation_snapshot_publisher(implementation_sha256=evaluation_snapshot_implementation_sha256())
        policy=d['runtime_policy']['arms'][0];controller=CallController(policy=CallPolicy(**policy['controllers'][0]['policy']),
            budget=CallBudget(unit='USD.micro',pricing_revision='fixture1',ceiling=10000,max_inflight=32),max_requests=256)
        human.register_call_controller('main',controller)
        native=await RecordingNativeWriter.open(root/'native',run_context(d,'a'*64,0))
        owner=RecordingEvaluationSession.from_runtime_frozen(d,native,latency_policy=l,source_bundles=bundles,
            run_nonce='a'*64,execution_ordinal=0,retain_evaluation_hashes=True)
        self.addAsyncCleanup(owner.close)
        memory=TransportMemoryBudget(64*1024**2)
        client=await FixtureCompletion.open(d['runtime_policy']['adapters'][0],responses,invocation_ordinal=0,lane_id='answer',binding_index=0,
            context=dict(owner.context),capacity=TransportCapacity(262144),memory=memory)
        self.addAsyncCleanup(client.close)
        owner.bind_semantic_lane(controller,0);owner.bind_backend(0,client,'callback')
        adapter=TextContextCalls(human,'main',client,provider='test.fixture',model_revision='1',evaluation_observer=owner)
        self.addAsyncCleanup(adapter.close)
        context=owner.context;execution=dict(schema_version=1,kind='recording_execution',runtime_policy_sha256=digest(d['runtime_policy']),
            collection_sha256=context['collection_sha256'],scope_protocol_sha256=context['scope_protocol_sha256'],instrumentation_protocol_sha256=context['instrumentation_protocol_sha256'],
            invocation_ordinal=0,recording_index=0,arm='A',stream_id='stream-A',execution_namespace='exec',cache_namespace='cache',asset_inventory_sha256='a'*64)
        artifacts=await RecordingRuntimeArtifacts.open(root/'runtime',execution,budget=ArtifactBudget(),retain_evaluation_artifacts=True)
        self.addAsyncCleanup(artifacts.abort)
        lane_values={'answer':dict(adapter=adapter,completion=client,controller=controller,policy=policy['lanes'][0],binding_index=0)}
        if changes.get('parallel'):
            index=next(i for i,b in enumerate(d['collection']['arms'][0]['provider_bindings']) if b['binding_id']=='text2')
            second=await FixtureCompletion.open(d['runtime_policy']['adapters'][0],responses,invocation_ordinal=0,lane_id='answer2',binding_index=index,
                context=dict(owner.context),capacity=TransportCapacity(262144),memory=memory)
            self.addAsyncCleanup(second.close);owner.bind_semantic_lane(controller,index);owner.bind_backend(index,second,'callback')
            second_adapter=TextContextCalls(human,'main',second,provider='test.fixture',model_revision='1',evaluation_observer=owner)
            self.addAsyncCleanup(second_adapter.close)
            lane_values['answer2']=dict(adapter=second_adapter,completion=second,controller=controller,
                policy=next(row for row in policy['lanes'] if row['lane_id']=='answer2'),binding_index=index)
        clock=ReplayClock(d['source_manifest']);owner.bind_mapping(clock.mapping,clock.sha256)
        receipts=[]
        async def ready(task,ref,sha):
            value=read_ref(artifacts.output,ref);self.assertEqual(ref['file_sha256'],sha);receipts.append(task)
        watermarks=[]
        def marks():
            value=[dict(source_index=0,last_acquired_ordinal=None,last_submitted_sequence=None,processed_sequence=None,published_sequence=None,processed_known=False,published_known=False)]
            watermarks.append(value);return value
        runtime=RecordingTaskRuntime(human,owner,clock,d['source_manifest'],d['tasks'],'local-user',publisher,
            lane_values,policy['recording_policies'][0]['text_lane_ids'],policy['task_queue']['capacity'],artifacts,ready,marks)
        self.addAsyncCleanup(runtime.close)
        runtime._test_latency_policy=l
        return runtime,human,controller,client,adapter,owner,native,artifacts,receipts

    async def test_eight_sequential_answers_and_overlapping_windows_preserve_original_context(self):
        runtime,human,controller,client,adapter,owner,native,artifacts,receipts=await self.setup_runtime(
            [100_000_000+n*1_200_000_000 for n in range(8)]+[10_000_000_000,10_000_000_000],delay_ms=5,deadline_ns=12_000_000_000)
        await runtime.run();self.assertEqual(len(receipts),10)
        self.assertTrue(all(result['status']=='answer' for result in runtime.result_values),runtime.result_values)
        contexts=[]
        for n,result in enumerate(runtime.result_values):
            self.assertEqual(result['answer_sha256'],hashlib.sha256(result['answer_text'].encode()).hexdigest())
            evidence=read_ref(artifacts.output,result['evidence']);context=read_ref(artifacts.output,evidence['context']);contexts.append(context)
            record=read_ref(artifacts.output,evidence['snapshot_record']);self.assertEqual(digest(record),evidence['snapshot_record_sha256'])
            self.assertEqual(record['timestamp_ms'],context['timestamp_ms']);self.assertEqual(record['observation']['value']['scheduled_offset_ns'],runtime.tasks[n]['snapshot_ns'])
            self.assertGreaterEqual(record['observation']['value']['observed_begin_ns'],runtime.clock.monotonic_ns+runtime.tasks[n]['snapshot_ns'])
            self.assertEqual(context['summary']['record_count'],contexts[0]['summary']['record_count'])
            self.assertNotIn('evaluation.context_snapshot',json.dumps(context['observations']))
        default=await human.get_context();self.assertEqual(default['summary']['record_count'],contexts[0]['summary']['record_count']+10)
        self.assertEqual(controller.snapshot()['budget']['reserved_liability'],100)
        self.assertFalse(client._entries);self.assertIsNone(adapter._active)
        await artifacts.close(cleanup_finished=True)
        await adapter.close();await client.close();owner.backend_closed(0,client);await owner.close()
        from benchmarks.recording_runtime_native import validate_runtime_native_inventory
        index=validate_runtime_native_inventory(owner.declarations,native.output,source_inventory=runtime.manifest,
            latency_policy=runtime._test_latency_policy,
            source_bundles={'original':dict(manifest=runtime.manifest,tasks=runtime.tasks,profiles=owner.declarations['profiles'])})
        self.assertEqual(index.count,10)

    async def test_accepted_sdk_response_with_escaped_text_retains_exact_bounded_sidecar(self):
        text='\x01'*8191+'X'
        runtime,human,controller,client,adapter,owner,native,artifacts,receipts=await self.setup_runtime(
            [100_000_000],answer_text=text)
        await runtime.run();self.assertEqual(runtime.result_values[0]['status'],'answer')
        path=artifacts.output/'requests/request-000000.json'
        raw=path.read_bytes();self.assertGreater(len(raw),32768);self.assertLessEqual(len(raw),65536)
        response=json.loads(raw);self.assertEqual(response['sdk_result']['text'],text)
        self.assertEqual(response['native_raw_result_sha256'],digest(dict(text=text,usage=None)))

    async def test_bounded_waiting_queue_refuses_without_refresh_or_hidden_call(self):
        runtime,human,controller,client,adapter,owner,native,artifacts,receipts=await self.setup_runtime(
            [100_000_000]*4,queue=1)
        release=asyncio.Event();original=client._invoke
        async def held(*args):
            await release.wait();return await original(*args)
        client._invoke=held
        running=asyncio.create_task(runtime.run())
        await self.until(lambda:runtime.failure_counts.get('busy',0)>=2)
        release.set();await running;results=runtime.result_values
        self.assertEqual([r['reason'] for r in results],['completed','completed','busy','busy'])
        self.assertEqual(client._call_ordinal,2);self.assertEqual(len(runtime._queued),0)
        evidence=read_ref(artifacts.output,results[1]['evidence']);envelope=read_ref(artifacts.output,evidence['snapshot_envelope'])
        self.assertLess(envelope['observed_end_ns'],results[0]['child_finished_ns'])
        self.assertEqual(controller.snapshot()['budget']['reserved_liability'],20)

    async def test_waiting_tasks_use_original_deadline_then_frozen_index(self):
        runtime,human,controller,client,*rest=await self.setup_runtime([100_000_000]*3,queue=2,
            deadline_ns=[1_900_000_000,1_800_000_000,1_500_000_000])
        release=asyncio.Event();original=client._invoke
        async def held(*args):
            await release.wait();return await original(*args)
        client._invoke=held;running=asyncio.create_task(runtime.run())
        await self.until(lambda:len(runtime._queued)==2);release.set();await running
        self.assertEqual([r['answer_text'] for r in runtime.result_values],['Answer 0','Answer 2','Answer 1'])

    async def test_two_persistent_lanes_enter_overlapping_actual_calls(self):
        runtime,*rest=await self.setup_runtime([100_000_000,100_000_000],parallel=True)
        entered=[asyncio.Event(),asyncio.Event()];release=asyncio.Event()
        for index,lane in enumerate(runtime._lanes):
            original=lane.completion._invoke
            def wrapper(index,original):
                async def held(*args):
                    entered[index].set();await release.wait();return await original(*args)
                return held
            lane.completion._invoke=wrapper(index,original)
        running=asyncio.create_task(runtime.run())
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)),1)
        self.assertEqual(len(runtime._active),2);release.set();await running
        self.assertEqual([r['status'] for r in runtime.result_values],['answer','answer'])
        instances=[r['operation_instance'] for r in runtime.result_values];self.assertEqual(len(set(instances)),2)

    async def test_deadline_cancels_and_joins_actual_call_retains_unknown_liability(self):
        runtime,human,controller,client,adapter,owner,native,artifacts,receipts=await self.setup_runtime(
            [100_000_000],deadline_ns=1_500_000_000)
        entered=asyncio.Event()
        async def held(*args):
            entered.set();await asyncio.Event().wait()
        client._invoke=held
        await runtime.run();self.assertTrue(entered.is_set());result=runtime.result_values[0]
        self.assertEqual(result['reason'],'timeout');self.assertIsNone(result['answer_text']);self.assertTrue(result['cleanup_finished'])
        evidence=read_ref(artifacts.output,result['evidence']);self.assertIsNotNone(evidence['operation_instance'])
        self.assertEqual(controller.snapshot()['budget']['reserved_liability'],10)
        self.assertIsNone(adapter._active);self.assertIsNone(client._prepared);self.assertFalse(client._entries)
        self.assertFalse(runtime._active)

    async def test_result_channel_failure_is_fatal_after_sealed_result(self):
        runtime,*rest=await self.setup_runtime([100_000_000])
        async def failed(*args):raise OSError('test owned channel failed')
        runtime._answer_ready=failed
        with self.assertRaisesRegex(RuntimeError,'sealed task result delivery failed'):await runtime.run()
        self.assertEqual(len(runtime.results),1);self.assertFalse(runtime._active)
        self.assertEqual(runtime.failure_counts.get('transport_failed'),1)

    async def test_writer_failure_preserves_prefix_and_never_reports_success(self):
        runtime,human,controller,client,adapter,owner,native,artifacts,receipts=await self.setup_runtime([100_000_000])
        original=artifacts._store
        def failed(path,data):
            if path.endswith('-result.json'):raise OSError('test owned disk failed')
            return original(path,data)
        artifacts._store=failed
        with self.assertRaisesRegex(RuntimeError,'task artifact write failed'):await runtime.run()
        self.assertFalse(runtime.results);self.assertFalse(receipts);self.assertFalse(runtime._active)
        self.assertEqual(runtime.failure_counts.get('writer_failed'),1)
        self.assertTrue((artifacts.output/'tasks/task-0000-evidence.json').exists())
        self.assertEqual(controller.snapshot()['budget']['reserved_liability'],10)

    async def test_repeated_cancellation_finishes_all_original_results_and_keeps_callers_open(self):
        runtime,human,controller,client,adapter,owner,native,artifacts,receipts=await self.setup_runtime(
            [100_000_000,100_000_000,600_000_000],delay_ms=500,queue=1)
        run=asyncio.create_task(runtime.run())
        await self.until(lambda:client._call_ordinal>0)
        run.cancel();await asyncio.sleep(0);run.cancel()
        with self.assertRaises(asyncio.CancelledError):await run
        await runtime.close()
        self.assertEqual(len(runtime.results),3);self.assertTrue(all(r['reason']=='cancelled' for r in runtime.result_values))
        self.assertEqual(controller.snapshot()['budget']['reserved_liability'],10)
        self.assertFalse(adapter._closed);self.assertFalse(client._closed);self.assertIsNone(adapter._active);self.assertIsNone(client._prepared)
        self.assertFalse(runtime._active);self.assertEqual(len(receipts),3)
