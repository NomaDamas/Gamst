"""SDK scope routing plus synthetic real native4 session/writer integration.

ProbeOwner observes immutable routing without fabricating native evidence. The
Native* fixtures run actual callbacks/owned pipes with RecordingEvaluationSession
and its writer; they do not establish original-recording replay or model quality.
"""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

from humansense.recording_scope import RecordingScope,RecordingScopeOwner
from humansense.call_controller import CallController
from humansense.call_budget import CallBudget
from humansense.call_policy import CallPolicy
import test_text_context_call as text_tests
import test_visual_context_call as visual_tests
import test_smolvlm_call as smol_tests
from test_smolvlm_native import CONFIG


class ProbeOwner(RecordingScopeOwner):
    def __init__(self):
        self.scopes=[];self.reserved=[];self.entered=[];self.finished=[];self.released=[]
        self.invalidations=[];self.work=[];self.revoked=False;self.backends=[];self.validated=[];self.worker_configs=[]
    def issue(self):
        scope=RecordingScope(self,len(self.scopes),'a'*64,object());self.scopes.append(scope);return scope
    def validate_scope(self,scope):
        if scope.owner is not self or not any(scope is value for value in self.scopes):raise ValueError('unissued recording scope')
    def bind_backend(self,backend):self.backends.append(backend)
    def validate_worker_scope(self,scope):
        self.validate_scope(scope)
        if scope.scope_index!=0:raise ValueError('stream worker scope required')
    def validate_worker_configuration(self,scope,provider,revision,config):
        self.validate_worker_scope(scope)
        self.worker_configs.append((scope,provider,revision,config))
    def validate_backend(self,scope,backend):
        self.validate_scope(scope)
        if not any(backend is value for value in self.backends):raise ValueError('unbound recording backend')
        self.validated.append((scope,backend))
    def reserve_call(self,controller,request_id,*,provider,model_revision,prompt_sha256,evaluation_scope):
        self.validate_scope(evaluation_scope)
        if self.revoked:raise RuntimeError('recording owner revoked')
        self.reserved.append((evaluation_scope,request_id,provider,model_revision,prompt_sha256))
        return ProbeCall(self,evaluation_scope)
    def reserve_work(self,kind,metadata,*,evaluation_scope=None):
        if evaluation_scope is not None:self.validate_scope(evaluation_scope)
        self.work.append((kind,metadata));return ProbeWork()
    def invalidate(self,reason,subject_id=None):
        self.invalidations.append((reason,subject_id));self.revoked=True


class ProbeCall:
    def __init__(self,owner,scope):self.owner=owner;self.scope=scope;self.id=uuid.uuid4().hex
    def dispatched(self):pass
    async def invoke(self,callback):
        if self.owner.revoked:raise RuntimeError('recording owner revoked before callback')
        self.owner.entered.append(self.scope);return await callback()
    def finish(self,outcome):self.owner.finished.append((self.scope,outcome))
    def release(self,outcome):self.owner.released.append((self.scope,outcome))


class ProbeWork:
    def event(self,*args,**kwargs):pass
    def finish(self,*args,**kwargs):pass


class TextScopes(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    def adapter(self,callback,**kwargs):
        observer=kwargs.get('evaluation_observer')
        owner=observer.owner if type(observer) is RecordingScope else observer
        if isinstance(owner,ProbeOwner):owner.bind_backend(callback)
        return text_tests.Calls.adapter(self,callback,**kwargs)

    async def test_scope_is_captured_before_prepare_await_without_mutable_adapter_scope(self):
        owner=ProbeOwner();scope=owner.issue();foreign=ProbeOwner();foreign.issue()
        calls=[]
        async def callback(request):calls.append('original');return dict(text='answer',usage=None)
        async def other_backend(request):calls.append('wrong');return dict(text='wrong',usage=None)
        adapter=self.adapter(callback,evaluation_observer=owner);started=asyncio.Event();release=asyncio.Event()
        async def prepare(request):started.set();await release.wait();return request
        adapter._prepare_execution=prepare
        rid,record=await self.queued()
        task=asyncio.create_task(adapter.execute(rid,record,'question',maximum_charge=1,evaluation_scope=scope))
        await started.wait();adapter._evaluation_observer=foreign;adapter._completion=other_backend;release.set()
        self.assertTrue((await task)['accepted'])
        self.assertEqual(owner.entered,[scope]);self.assertFalse(foreign.reserved)
        self.assertEqual(calls,['original']);self.assertEqual(owner.validated,[(scope,callback)])
        self.assertEqual(owner.reserved[0][1],rid);self.assertEqual(self.c.snapshot()['budget']['dispatched'],1)

    async def test_missing_foreign_forged_fixed_override_and_legacy_scope_fail_before_work(self):
        owner=ProbeOwner();scope=owner.issue();other_scope=owner.issue();foreign=ProbeOwner().issue();calls=[]
        async def callback(request):calls.append(True);return dict(text='answer',usage=None)
        rid,record=await self.queued();root=self.adapter(callback,evaluation_observer=owner)
        forged=replace(scope)
        for kwargs in ({},{'evaluation_scope':foreign},{'evaluation_scope':forged},{'evaluation_scope':object()}):
            with self.assertRaises((TypeError,ValueError)):await root.execute(rid,record,'q',maximum_charge=1,**kwargs)
        fixed=self.adapter(callback,evaluation_observer=scope)
        with self.assertRaises(ValueError):await fixed.execute(rid,record,'q',maximum_charge=1,evaluation_scope=other_scope)
        legacy=self.adapter(callback)
        with self.assertRaises(ValueError):await legacy.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope)
        self.assertFalse(calls);self.assertFalse(owner.reserved);self.assertEqual(self.c.snapshot()['budget']['accepted'],0)
        self.assertTrue((await fixed.execute(rid,record,'q',maximum_charge=1))['accepted'])
        self.assertEqual(owner.entered,[scope])

    async def test_65_handles_one_privacy_owner_and_unchanged_owner_controller_caps(self):
        owner=ProbeOwner();handles=[owner.issue() for _ in range(65)]
        async def callback(request):return dict(text='answer',usage=None)
        for handle in handles:self.adapter(callback,evaluation_observer=handle)
        self.assertEqual(self.h._evaluation_observers,[owner]);self.assertEqual(len(self.h._call_controllers),1)
        await self.h.pause();self.assertEqual(owner.invalidations,[('pause',None)])
        for i in range(31):self.h._register_evaluation_observer(ProbeOwner())
        with self.assertRaises(ValueError):self.h._register_evaluation_observer(ProbeOwner())
        await self.h.resume()
        for i in range(7):
            controller=CallController(policy=CallPolicy(),budget=CallBudget(unit='fixture',pricing_revision='1',ceiling=1))
            self.h.register_call_controller('lane'+str(i),controller)
        with self.assertRaises(ValueError):
            self.h.register_call_controller('ninth',CallController(policy=CallPolicy(),budget=CallBudget(unit='fixture',pricing_revision='1',ceiling=1)))

    async def test_unbound_backend_fails_before_prepare_reservation_and_callback(self):
        owner=ProbeOwner();scope=owner.issue();called=[]
        async def callback(request):called.append(True);return dict(text='answer',usage=None)
        adapter=self.adapter(callback,evaluation_observer=owner);owner.backends.clear()
        async def forbidden_prepare(request):raise AssertionError('unbound backend prepared')
        adapter._prepare_execution=forbidden_prepare;rid,record=await self.queued()
        with self.assertRaisesRegex(ValueError,'unbound recording backend'):
            await adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope)
        self.assertFalse(called);self.assertFalse(owner.reserved)
        self.assertEqual(self.c.snapshot()['budget']['accepted'],0)

    async def test_many_overlapping_windows_keep_one_active_adapter_and_one_unknown_bill(self):
        owner=ProbeOwner();handles=[owner.issue() for _ in range(12)];started=asyncio.Event();release=asyncio.Event()
        async def callback(request):started.set();await release.wait();return dict(text='answer',usage=None)
        adapter=self.adapter(callback,evaluation_observer=owner);rid,record=await self.queued()
        active=asyncio.create_task(adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=handles[0]))
        await started.wait()
        for scope in handles[1:]:
            with self.assertRaisesRegex(RuntimeError,'busy'):
                await adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope)
        self.assertEqual(owner.entered,[handles[0]]);self.assertEqual(len(self.h._call_controllers),1)
        release.set();await active
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],1)

    async def test_pause_invalidates_one_owner_across_two_entered_scopes_and_preserves_debt(self):
        owner=ProbeOwner();handles=[owner.issue(),owner.issue()];entered=0
        both_entered=asyncio.Event();release=asyncio.Event()
        async def callback(request):
            nonlocal entered
            entered+=1
            if entered==2:both_entered.set()
            await release.wait();return dict(text='PRIVATE LATE BODY',usage=None)
        adapters=[self.adapter(callback,evaluation_observer=handle) for handle in handles]
        requests=[await self.queued(),await self.queued()]
        tasks=[asyncio.create_task(adapter.execute(rid,record,'q',maximum_charge=1))
            for adapter,(rid,record) in zip(adapters,requests)]
        await both_entered.wait();await self.h.pause();release.set()
        results=await asyncio.gather(*tasks)
        self.assertEqual(owner.entered,handles);self.assertEqual(owner.invalidations,[('pause',None)])
        self.assertTrue(all(value['status']=='discarded' and 'text' not in value for value in results))
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],2)

    async def test_legacy_observer_fake_owner_attribute_is_not_resolved(self):
        class Legacy:
            def __init__(self):self.invalidations=[]
            @property
            def evaluation_owner(self):raise AssertionError('unsupported owner attribute accessed')
            def invalidate(self,reason,subject_id=None):self.invalidations.append((reason,subject_id))
        legacy=Legacy();self.h._register_evaluation_observer(legacy)
        self.assertIs(self.h._evaluation_observers[0],legacy)
        await self.h.delete_data('local-user');self.assertEqual(legacy.invalidations,[('delete','local-user')])


@unittest.skipIf(visual_tests.np is None,'optional vision dependencies absent')
class VisualScopes(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=visual_tests.Calls.asyncSetUp
    asyncTearDown=visual_tests.Calls.asyncTearDown
    queued=visual_tests.Calls.queued
    def adapter(self,callback,**kwargs):
        observer=kwargs.get('evaluation_observer')
        owner=observer.owner if type(observer) is RecordingScope else observer
        if isinstance(owner,ProbeOwner):owner.bind_backend(callback)
        return visual_tests.Calls.adapter(self,callback,**kwargs)
    async def test_fixed_original_task_scope_survives_owned_png_preparation(self):
        owner=ProbeOwner();scope=owner.issue();other=ProbeOwner();other.issue();called=[]
        async def callback(request):called.append(request['images'][0]['source_frame_sha256']);return dict(text='answer',usage=None)
        adapter=self.adapter(callback,evaluation_observer=owner)
        original=adapter._prepare_execution;started=asyncio.Event();release=asyncio.Event()
        async def prepare(request):started.set();await release.wait();return await original(request)
        adapter._prepare_execution=prepare;rid,record,frame=await self.queued()
        task=asyncio.create_task(adapter.execute(rid,record,frame,'q',maximum_charge=1,evaluation_scope=scope))
        await started.wait();adapter._evaluation_observer=other;release.set()
        self.assertTrue((await task)['accepted']);self.assertEqual(len(called),1)
        self.assertEqual(owner.entered,[scope]);self.assertFalse(other.reserved)


@unittest.skipIf(smol_tests.np is None,'optional numpy absent')
class SmolScopes(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=smol_tests.Calls.asyncSetUp
    asyncTearDown=smol_tests.Calls.asyncTearDown
    fixture=smol_tests.Calls.fixture
    setup_human=smol_tests.Calls.setup_human
    request=smol_tests.Calls.request
    async def test_owned_process_scope_survives_reset_and_preserves_original_context(self):
        owner=ProbeOwner();scope=owner.issue();other=ProbeOwner();other.issue()
        with self.fixture():
            human,controller,spec=self.setup_human()
            async with human:
                async with await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,
                        evaluation_observer=owner,evaluation_worker_scope=scope) as adapter:
                    owner.bind_backend(adapter.worker)
                    original=adapter.worker.reset;started=asyncio.Event();release=asyncio.Event()
                    async def reset(**kwargs):started.set();await release.wait();return await original(**kwargs)
                    adapter.worker.reset=reset;rid,record,frame=await self.request(human,controller,spec)
                    task=asyncio.create_task(adapter.execute(rid,record,frame,maximum_charge=2,evaluation_scope=scope))
                    await started.wait();adapter._evaluation_observer=other;release.set()
                    result=await task;self.assertTrue(result['accepted'])
                    self.assertEqual(result['observation']['timestamp_ms'],frame.timestamp_ms)
                    self.assertEqual(owner.entered,[scope]);self.assertFalse(other.reserved)
                    self.assertEqual(human._evaluation_observers,[owner])
                    self.assertEqual(owner.worker_configs[0][1:3],(adapter.worker.spec.provider,adapter.worker.spec.version))
                    self.assertEqual(owner.worker_configs[0][3],adapter.worker.config)
            self.assertTrue(all(child.returncode is not None for child in self.children))
            self.assertEqual(controller.snapshot()['budget']['reserved_liability'],2)


class NativeFixture:
    """Real native4 writer/session with synthetic frozen recording declarations."""
    async def recording(self,**kwargs):
        from recording_native_fixtures import session
        directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(directory.cleanup)
        observer,writer,clock=await session(Path(directory.name).resolve(),**kwargs)
        self.addAsyncCleanup(observer.close)
        return observer,writer,clock

    @staticmethod
    def events(writer):
        return [json.loads(line) for path in sorted(writer.output.glob('native-*.jsonl'))
            for line in path.read_bytes().splitlines() if '"event":' in line.decode()]

    @staticmethod
    def terminal(writer):return json.loads((writer.output/'terminal.json').read_bytes())


class NativeTextScopes(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    adapter=text_tests.Calls.adapter

    async def test_actual_callback_has_task_scope_and_original_decision_hash_with_unknown_money(self):
        observer,writer,clock=await self.recording();entered=[]
        async def callback(request):
            entered.append(observer._entries)
            return dict(text='PRIVATE NATIVE ANSWER',usage=None)
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,callback,'callback')
        observer.bind_mapping(clock.mapping,clock.sha256)
        scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],binding_index=0)
        adapter=self.adapter(callback,evaluation_observer=observer);rid,record=await self.queued()
        decision=next(r['decision'] for r in self.c.snapshot()['requests'] if r['id']==rid)
        result=await adapter.execute(rid,record,'q',maximum_charge=2,evaluation_scope=scope)
        self.assertTrue(result['accepted']);self.assertEqual(entered,[1])
        await adapter.close();observer.backend_closed(0,callback);await observer.close()
        rows=self.events(writer);reserved=next(r['fields'] for r in rows if r['event']=='reserved')
        self.assertEqual(reserved['scope'],1);self.assertEqual(reserved['binding'],0)
        self.assertEqual(reserved['request_sha256'],decision['decision_sha256'])
        self.assertEqual(reserved['prompt_sha256'],result['prompt_sha256'])
        self.assertEqual([r['fields']['outcome'] for r in rows if r['event']=='published'],['accepted'])
        terminal=self.terminal(writer);self.assertEqual(terminal['state'],'closed')
        self.assertEqual(terminal['entry_count'],1);self.assertEqual(terminal['unsettled_entries'],1)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],2)
        self.assertNotIn('PRIVATE',json.dumps(rows));self.assertFalse(terminal['receipt_authenticity_verified'])

    async def test_real_owner_denied_then_retry_preserves_one_callback_and_releases_unentered_ticket(self):
        observer,writer,clock=await self.recording();called=[]
        async def callback(request):called.append(True);return dict(text='answer',usage=None)
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,callback,'callback')
        observer.bind_mapping(clock.mapping,clock.sha256)
        scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],binding_index=0)
        adapter=self.adapter(callback,evaluation_observer=scope);rid,record=await self.queued()
        with self.assertRaises(ValueError):
            await smol_tests.SmolVLMCalls.open(self.h,'main',sys.executable,CONFIG,
                evaluation_observer=observer,evaluation_worker_scope=observer.stream_scope(0))
        denied=await adapter.execute(rid,record,'q',maximum_charge=None)
        self.assertEqual(denied['status'],'blocked');self.assertFalse(called)
        self.assertTrue((await adapter.execute(rid,record,'q',maximum_charge=1))['accepted'])
        await adapter.close();observer.backend_closed(0,callback);await observer.close()
        terminal=self.terminal(writer)
        self.assertEqual(terminal['reservation_count'],2);self.assertEqual(terminal['released_count'],1)
        self.assertEqual(terminal['entry_count'],1);self.assertEqual(terminal['state'],'closed')
        self.assertEqual(called,[True]);self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],1)


@unittest.skipIf(visual_tests.np is None,'optional vision dependencies absent')
class NativeVisualScopes(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=visual_tests.Calls.asyncSetUp
    asyncTearDown=visual_tests.Calls.asyncTearDown
    queued=visual_tests.Calls.queued
    adapter=visual_tests.Calls.adapter

    async def test_actual_png_callback_is_entered_once_under_fixed_task_scope(self):
        observer,writer,clock=await self.recording(role='small_vlm');images=[]
        async def callback(request):images.extend(request['images']);return dict(text='answer',usage=None)
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,callback,'callback')
        observer.bind_mapping(clock.mapping,clock.sha256)
        scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],binding_index=0)
        adapter=self.adapter(callback,evaluation_observer=scope);rid,record,frame=await self.queued()
        result=await adapter.execute(rid,record,frame,'q',maximum_charge=1)
        self.assertTrue(result['accepted']);self.assertEqual(len(images),1)
        self.assertEqual(images[0]['source_frame_sha256'],result['source_frame_sha256'])
        await adapter.close();observer.backend_closed(0,callback);await observer.close()
        self.assertEqual(self.terminal(writer)['entry_count'],1)
        self.assertEqual(self.terminal(writer)['state'],'closed')
        self.assertNotIn('data_base64',json.dumps(self.events(writer)))


@unittest.skipIf(smol_tests.np is None,'optional numpy absent')
class NativeSmolScopes(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=smol_tests.Calls.asyncSetUp
    asyncTearDown=smol_tests.Calls.asyncTearDown
    fixture=smol_tests.Calls.fixture
    setup_human=smol_tests.Calls.setup_human
    request=smol_tests.Calls.request

    async def test_real_owned_worker_stream_lifetime_and_distinct_task_execution_scope(self):
        from humansense.provider_profiles import capability
        from humansense.providers.smolvlm_native import configuration
        config=dict(configuration(CONFIG),window_frames=1)
        spec=capability('smolvlm2.500m.local',config)
        observer,writer,clock=await self.recording(provider=spec.provider,revision=spec.version,
            role='small_vlm',boundary='process_adapter',config=config)
        with self.fixture():
            human,controller,detect_spec=self.setup_human()
            async with human:
                observer.bind_semantic_lane(controller,0)
                stream_scope=observer.stream_scope(0)
                task_scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],binding_index=0)
                for kwargs in ({},{'evaluation_worker_scope':task_scope},
                        {'evaluation_worker_scope':replace(stream_scope)},
                        {'evaluation_worker_scope':ProbeOwner().issue()}):
                    with self.assertRaises((TypeError,ValueError)):
                        await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,
                            evaluation_observer=observer,**kwargs)
                with self.assertRaisesRegex(ValueError,'pre-spawn provider/revision/config'):
                    await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,dict(CONFIG,max_new_tokens=3),
                        evaluation_observer=observer,evaluation_worker_scope=stream_scope)
                self.assertFalse(self.children)
                adapter=await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,
                    evaluation_observer=observer,evaluation_worker_scope=stream_scope)
                try:
                    observer.bind_backend(0,adapter.worker,'process')
                    observer.bind_mapping(clock.mapping,clock.sha256)
                    rid,record,frame=await self.request(human,controller,detect_spec)
                    result=await adapter.execute(rid,record,frame,maximum_charge=2,evaluation_scope=task_scope)
                    self.assertTrue(result['accepted'])
                    self.assertEqual(result['observation']['timestamp_ms'],frame.timestamp_ms)
                    self.assertEqual(human._evaluation_observers,[observer])
                finally:await adapter.close()
                await observer.close()
                terminal=self.terminal(writer);self.assertEqual(terminal['state'],'closed')
                self.assertEqual(terminal['entry_count'],1);self.assertEqual(terminal['unclosed_workers'],0)
                rows=self.events(writer)
                self.assertEqual([r['fields']['scope'] for r in rows if r['event']=='reserved'],[1])
                self.assertEqual([r['fields']['outcome'] for r in rows if r['event']=='published'],['accepted'])
                self.assertEqual(controller.snapshot()['budget']['reserved_liability'],2)
                from benchmarks.recording_native_validation import validate_recording_native_inventory,iter_canonical_inventory,canonical_lane_inventory
                index=validate_recording_native_inventory(observer.declarations,writer.output,
                    source_inventory=observer.declarations['source_manifest'])
                inventory=list(iter_canonical_inventory(index));work=[r for r in inventory if r['kind']=='worker_interval']
                self.assertTrue({'startup','residency','reset','transport','close'} <= {r['work_kind'] for r in work})
                self.assertTrue(all(r['worker_kind']=='process' and r['ended_ns']>=r['started_ns'] for r in work))
                delegated=[r for r in work if r['parent_entry_ordinal'] is not None]
                self.assertTrue(delegated);self.assertEqual({r['parent_entry_ordinal'] for r in delegated},{0})
                lane=canonical_lane_inventory(index,1,0)
                self.assertEqual(lane['counts']['entered'],1);self.assertFalse(lane['unobserved_entry_inventory_possible'])

            self.assertTrue(all(child.returncode is not None for child in self.children))


if __name__=='__main__':unittest.main()
