"""Real SDK ownership boundaries with a deterministic external observer."""
import asyncio
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider
import test_text_context_call as text_tests
import test_process_provider as worker_tests
import test_smolvlm_call as smol_tests
from test_smolvlm_native import CONFIG


class Ticket:
    def __init__(self, observer, kind, metadata):
        self.observer=observer;self.kind=kind;self.metadata=metadata
        self.id=str(len(observer.tickets));self.entered=False
        observer.tickets.append(self)
    def event(self,name,fields=None):
        self.observer.events.append((self.id,name))
        if self.observer.fail==name:raise RuntimeError('observer failed')
    def dispatched(self):
        self.event('dispatched')
        if self.observer.cancel_on_dispatch:
            raise asyncio.CancelledError()
    async def invoke(self,callback):
        self.entered=True;self.event('invoke')
        return await callback()
    def finish(self,outcome):self.event('finish:'+outcome)
    def release(self,reason):self.event('release:'+reason)


class Observer:
    def __init__(self,fail=None,cancel_on_dispatch=False):
        self.fail=fail;self.cancel_on_dispatch=cancel_on_dispatch
        self.tickets=[];self.events=[];self.invalidations=[]
    def reserve_call(self,controller,request_id,**fields):
        self.controller_state=controller.snapshot()['budget']['dispatched']
        return Ticket(self,'call',fields)
    def reserve_work(self,kind,metadata):return Ticket(self,kind,metadata)
    def invalidate(self,reason,subject_id=None):
        self.invalidations.append((reason,subject_id))
        if self.fail=='invalidate':raise RuntimeError('observer failed')


class Calls(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    adapter=text_tests.Calls.adapter

    async def test_success_and_preentry_dispatch_failure(self):
        for fail in (None,'dispatched'):
            observer=Observer(fail);called=[]
            async def backend(request):called.append(request);return dict(text='ok',usage=None)
            adapter=self.adapter(backend,evaluation_observer=observer)
            request,record=await self.queued()
            if fail:
                with self.assertRaisesRegex(RuntimeError,'observer failed'):
                    await adapter.execute(request,record,'question',maximum_charge=1)
                self.assertFalse(called);self.assertFalse(observer.tickets[0].entered)
            else:
                result=await adapter.execute(request,record,'question',maximum_charge=1)
                self.assertEqual(result['status'],'completed');self.assertEqual(len(called),1)
                self.assertIn(('0','finish:completed'),observer.events)
            self.assertIsNone(adapter._backend_task)

    async def test_cancellation_before_backend_entry_has_no_attempt(self):
        observer=Observer(cancel_on_dispatch=True);called=[]
        async def backend(request):called.append(True);return dict(text='ok',usage=None)
        adapter=self.adapter(backend,evaluation_observer=observer)
        request,record=await self.queued()
        with self.assertRaises(asyncio.CancelledError):
            await adapter.execute(request,record,'question',maximum_charge=1)
        self.assertFalse(called);self.assertFalse(observer.tickets[0].entered)
        self.assertEqual(self.c.snapshot()['budget']['unknown_reports'],1)
        self.assertIn(('0','release:cancelled'),observer.events)

    async def test_external_cancellation_drains_backend_before_terminal_hook_failure(self):
        started=asyncio.Event();drained=[]
        observer=Observer('finish:cancelled')
        async def backend(request):
            started.set()
            try:await asyncio.sleep(30)
            finally:drained.append(True)
        adapter=self.adapter(backend,evaluation_observer=observer)
        request,record=await self.queued()
        task=asyncio.create_task(adapter.execute(request,record,'question',maximum_charge=1))
        await started.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(drained,[True]);self.assertIsNone(adapter._backend_task)
        self.assertIn(('0','release:cancelled'),observer.events)

    async def test_privacy_invalidation_failure_never_blocks_cleanup(self):
        observer=Observer('invalidate')
        async def backend(request):return dict(text='ok',usage=None)
        self.adapter(backend,evaluation_observer=observer)
        await self.queued()
        await self.h.delete_data('local-user')
        self.assertEqual(await self.h.get_history('local-user'),[])
        await self.h.reset_processing();await self.h.pause()
        self.h.consent=False;await self.h.close()
        self.assertEqual(observer.invalidations,[('delete','local-user'),('reset',None),('pause',None),('consent_revoked',None),('close',None)])


class Workers(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=worker_tests.Processes.asyncSetUp
    asyncTearDown=worker_tests.Processes.asyncTearDown
    fixture=worker_tests.Processes.fixture

    async def test_actual_worker_operations_parent_and_verified_reap(self):
        observer=Observer()
        with self.fixture():
            provider=await ProcessProvider.open(sys.executable,'audio.prosody',{},evaluation_observer=observer)
            await provider.process(worker_tests.frame(),evaluation_operation_id='semantic-1')
            await provider.reset();await provider.finish();await provider.close()
        self.assertIsNotNone(self.children[0].returncode)
        lifecycle=observer.tickets[0]
        names=[name for ident,name in observer.events if ident==lifecycle.id]
        self.assertEqual(names,['spawn_begin','handle_acquired','handshake_ready','reap_confirmed','finish:closed'])
        process=next(t for t in observer.tickets if t.metadata.get('operation')=='process')
        self.assertEqual(process.metadata['parent_operation_id'],'semantic-1')
        self.assertEqual(process.metadata['worker_id'],lifecycle.metadata['worker_id'])

    async def test_failed_handshake_and_sink_error_reap(self):
        for code,fail in [('raise RuntimeError("fixture")',None),(worker_tests.FAKE_WORKER,'operation_started')]:
            observer=Observer(fail)
            with self.fixture(code):
                with self.assertRaises(Exception):
                    await ProcessProvider.open(sys.executable,'audio.prosody',{},evaluation_observer=observer)
            self.assertIsNotNone(self.children[-1].returncode)
            names=[name for ident,name in observer.events if ident=='0']
            self.assertLess(names.index('handshake_failed'),names.index('reap_confirmed'))


@unittest.skipIf(smol_tests.np is None,'optional numpy absent')
class Smol(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=smol_tests.Calls.asyncSetUp
    asyncTearDown=smol_tests.Calls.asyncTearDown
    fixture=smol_tests.Calls.fixture
    setup_human=smol_tests.Calls.setup_human
    request=smol_tests.Calls.request
    async def test_real_worker_has_one_semantic_call_and_child_process_operation(self):
        observer=Observer()
        with self.fixture():
            human,controller,spec=self.setup_human()
            async with human:
                async with await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,evaluation_observer=observer) as adapter:
                    request,record,frame=await self.request(human,controller,spec)
                    result=await adapter.execute(request,record,frame,maximum_charge=2)
                    self.assertEqual(result['status'],'published')
        calls=[t for t in observer.tickets if t.kind=='call']
        self.assertEqual(len(calls),1);self.assertTrue(calls[0].entered)
        process=next(t for t in observer.tickets if t.metadata.get('operation')=='process')
        self.assertEqual(process.metadata['parent_operation_id'],calls[0].id)


class Registry(unittest.TestCase):
    def test_bounded_deduplicated_callable_registry(self):
        human=HumanSense(consent=True)
        observers=[Observer() for _ in range(32)]
        for observer in observers:human._register_evaluation_observer(observer)
        for observer in observers:human._register_evaluation_observer(observer)
        self.assertEqual(len(human._evaluation_observers),32)
        with self.assertRaises(ValueError):human._register_evaluation_observer(Observer())
        with self.assertRaises(TypeError):human._register_evaluation_observer(object())
        asyncio.run(human.close())
