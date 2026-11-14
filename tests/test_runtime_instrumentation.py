"""Automatic native reports from actual SDK invocation boundaries."""
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from benchmarks.runtime_instrumentation import EvaluationSession
from benchmarks.perception_comparison import plan_digest
from test_stream_attempts import fixture
import test_text_context_call as text_tests
import test_smolvlm_call as smol_tests
from test_smolvlm_native import CONFIG
from humansense.provider_profiles import capability
from humansense.providers.smolvlm_native import configuration


def session(root,provider='test.client',revision='v1',**options):
    plan,spec,inputs,stream,protocol=fixture(root)
    plan['comparison']['cost']['rate_card']={'request':dict(provider=provider,revision=revision,
        unit='request',category='request',per_unit=1)}
    spec['comparison_plan_sha256']=plan_digest(plan)
    stream.update(comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec))
    protocol.update(comparison_plan_sha256=plan_digest(plan),collection_sha256=plan_digest(spec),
        stream_protocol_sha256=plan_digest(stream))
    args=dict(expected_plan_sha256=plan_digest(plan),expected_collection_sha256=plan_digest(spec),
        expected_stream_protocol_sha256=plan_digest(stream),expected_reconciliation_protocol_sha256=plan_digest(protocol),
        task_id='0',arm='A',stream_id='stream-A',retain_evaluation_hashes=True)
    args.update(options)
    return EvaluationSession(plan,spec,inputs,stream,protocol,**args)


class Sink:
    def __init__(self,event):self.event=event
    def emit(self,item):
        if item['event']==self.event:raise RuntimeError('private sink message')


class Text(unittest.IsolatedAsyncioTestCase):
    queued=text_tests.Calls.queued
    adapter=text_tests.Calls.adapter
    async def asyncSetUp(self):
        await text_tests.Calls.asyncSetUp(self)
        self.directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name).resolve()
    asyncTearDown=text_tests.Calls.asyncTearDown

    async def test_success_unknown_bills_and_immutable_terminal(self):
        observer=session(self.root)
        async def callback(request):return dict(text='PRIVATE RESPONSE',usage=dict(input_tokens=10,output_tokens=2,cached_input_tokens=0))
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        self.assertEqual((await adapter.execute(request,record,'PRIVATE QUESTION',maximum_charge=1))['status'],'completed')
        await adapter.close();report=await observer.close()
        self.assertEqual(report['provider_counters'],[dict(provider='test.client',revision='v1',role='text_llm',count=1)])
        self.assertTrue(report['unresolved_billing']);self.assertIsNone(report['monetary_cost'])
        attempt=report['reports'][0]['attempts'][0]
        self.assertIsNone(attempt['cost_quantities']);self.assertIsNone(attempt['line_claims'])
        self.assertEqual(attempt['status'],'success');self.assertNotIn('PRIVATE',str(report))
        self.assertEqual(report['actual_acceptance'],'incomplete')
        report['provider_counters'].clear()
        self.assertEqual(len(observer.snapshot()['provider_counters']),1)

    async def test_sink_failures_before_entry_retain_dispatched_liability_without_attempt(self):
        for event in ('call_capacity_reserved','controller_dispatched','backend_entry_admission'):
            with self.subTest(event=event):
                observer=session(self.root,journal=Sink(event));called=[]
                async def callback(request):called.append(True);return dict(text='ok',usage=None)
                adapter=self.adapter(callback,evaluation_observer=observer)
                request,record=await self.queued()
                with self.assertRaises(RuntimeError):await adapter.execute(request,record,'q',maximum_charge=1)
                report=await observer.close()
                self.assertFalse(called);self.assertEqual(report['reports'][0]['attempts'],[])
                self.assertIn('journal_sink_failed',report['incomplete_reasons'])
                self.assertEqual(report['dispatched_without_invocation'],int(event!='call_capacity_reserved'))

    async def test_cancellation_after_dispatch_before_callback_has_no_native_attempt(self):
        class CancelAtDispatch:
            def emit(self,event):
                if event['event']=='controller_dispatched':raise asyncio.CancelledError()
        observer=session(self.root,journal=CancelAtDispatch());called=[]
        async def callback(request):called.append(True);return dict(text='ok',usage=None)
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        with self.assertRaises(asyncio.CancelledError):await adapter.execute(request,record,'q',maximum_charge=1)
        report=await observer.close()
        self.assertFalse(called);self.assertEqual(report['reports'][0]['attempts'],[])
        self.assertEqual(report['provider_counters'],[])
        self.assertEqual(report['dispatched_without_invocation'],1)
        self.assertTrue(report['unresolved_billing']);self.assertEqual(self.c.snapshot()['budget']['unknown_reports'],1)

    async def test_error_and_cancellation_keep_counts_and_drain(self):
        for cancel in (False,True):
            observer=session(self.root);started=asyncio.Event();drained=[]
            async def callback(request):
                started.set()
                try:
                    if cancel:await asyncio.sleep(30)
                    raise ValueError('PRIVATE EXCEPTION')
                finally:drained.append(True)
            adapter=self.adapter(callback,evaluation_observer=observer)
            request,record=await self.queued()
            task=asyncio.create_task(adapter.execute(request,record,'q',maximum_charge=1))
            await started.wait()
            if cancel:task.cancel()
            with self.assertRaises(asyncio.CancelledError if cancel else ValueError):await task
            report=await observer.close()
            self.assertEqual(drained,[True]);self.assertEqual(report['provider_counters'][0]['count'],1)
            self.assertEqual(report['reports'][0]['attempts'][0]['status'],'cancelled' if cancel else 'error')
            self.assertNotIn('PRIVATE EXCEPTION',str(report))

    async def test_privacy_redaction_allows_opaque_late_lines_only_before_terminal(self):
        observer=session(self.root)
        async def callback(request):return dict(text='ok',usage=None)
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        await adapter.execute(request,record,'q',maximum_charge=1)
        operation=next(e['operation_id'] for e in observer.snapshot()['journal'] if e['event']=='controller_dispatched')
        await self.h.delete_data('local-user')
        self.assertEqual(observer.snapshot()['private_subject_mappings'],0)
        args=dict(quantities={'request':1},line_claims=['b'*64],evidence_sha256='c'*64)
        observer.attach_declared_lines(operation,**args)
        self.assertFalse(observer.snapshot()['unresolved_attempt_billing'])
        self.assertTrue(observer.snapshot()['unresolved_billing'])
        report=await observer.close()
        with self.assertRaises(RuntimeError):observer.attach_declared_lines(operation,**args)
        observer.invalidate('delete',subject_id='local-user')
        self.assertEqual(observer.snapshot(),report)
        self.assertFalse(report['receipt_authenticity_verified']);self.assertIsNone(report['monetary_cost'])

    async def test_actual_execution_binding_and_explicit_retained_evidence_retry(self):
        from humansense.schema import timestamp
        observer=session(self.root);executions=[];fail=[True]
        async def callback(request):
            current=next(r for r in self.c.snapshot()['requests'] if r['state']=='dispatched' and not r['finished'])
            executions.append(plan_digest(current['execution']))
            if fail[0]:raise RuntimeError('controlled first attempt')
            return dict(text='ok',usage=None)
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        with self.assertRaises(RuntimeError):await adapter.execute(request,record,'q',maximum_charge=1)
        operation=next(e['operation_id'] for e in observer.snapshot()['journal'] if e['event']=='controller_dispatched')
        source=next(iter(self.c.snapshot()['sources']))
        retry=self.c.submit(source,record,sequence=2,now_ms=timestamp(),query=True)['request_id']
        observer.link_call(retry,predecessor_operation_id=operation,relationship='retry')
        fail[0]=False
        await adapter.execute(retry,record,'q',maximum_charge=1)
        result=await observer.close();attempts=result['reports'][0]['attempts']
        self.assertEqual([a['execution_sha256'] for a in attempts],executions)
        self.assertEqual([a['status'] for a in attempts],['error','success'])
        self.assertEqual(attempts[1]['predecessor'],attempts[0]['attempt_id'])
        self.assertEqual(attempts[1]['relationship'],'retry')
        self.assertEqual(result['provider_counters'][0]['count'],2)
        self.assertTrue(result['unresolved_billing'])

    async def test_active_subject_deletion_scrubs_private_mapping_without_refund(self):
        observer=session(self.root);started=asyncio.Event();release=asyncio.Event()
        async def callback(request):
            started.set();await release.wait();return dict(text='PRIVATE LATE RESPONSE',usage=None)
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        task=asyncio.create_task(adapter.execute(request,record,'q',maximum_charge=1))
        await started.wait()
        self.assertEqual(observer.snapshot()['private_subject_mappings'],1)
        operation=next(e['operation_id'] for e in observer.snapshot()['journal'] if e['event']=='controller_dispatched')
        await self.h.delete_data('local-user')
        self.assertEqual(observer.snapshot()['private_subject_mappings'],0)
        release.set();result=await task
        self.assertEqual(result['status'],'discarded')
        observer.attach_declared_lines(operation,quantities={'request':1},line_claims=['b'*64],evidence_sha256='c'*64)
        report=await observer.close()
        self.assertEqual(report['provider_counters'][0]['count'],1)
        self.assertNotIn('PRIVATE',str(report));self.assertNotIn('local-user',str(report))
        self.assertFalse(report['sdk_deletion_erases_evaluation_artifacts'])
        self.assertFalse(report['receipt_authenticity_verified'])

    async def test_repeated_close_cancellation_drains_owned_writer_and_freezes_bundle(self):
        entered=asyncio.Event();release=asyncio.Event();drained=[]
        class Writer:
            def emit(self,event):pass
            async def close(self,**kwargs):
                entered.set();await release.wait();drained.append(True)
        observer=session(self.root,journal=Writer())
        closing=asyncio.create_task(observer.close());await entered.wait()
        closing.cancel();await asyncio.sleep(0);closing.cancel();await asyncio.sleep(0)
        self.assertFalse(closing.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):await closing
        self.assertEqual(drained,[True])
        frozen=observer.snapshot()
        self.assertEqual(await observer.close(),frozen)
        self.assertTrue(frozen['terminal'])

    async def test_native_admission_refusal_does_not_claim_callback_entry(self):
        from unittest.mock import patch
        observer=session(self.root);called=[]
        async def callback(request):called.append(True);return dict(text='ok',usage=None)
        async def refused(*args,**kwargs):raise ValueError('controlled native admission refusal')
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        with patch.object(observer._task,'invoke',refused):
            with self.assertRaises(ValueError):await adapter.execute(request,record,'q',maximum_charge=1)
        report=await observer.close()
        self.assertFalse(called);self.assertEqual(report['reports'][0]['attempts'],[])
        publication=next(e for e in report['journal'] if e['event']=='call_publication_finished')
        self.assertFalse(publication['fields']['invoked'])
        self.assertIn('native_admission_failed_before_callback',report['incomplete_reasons'])
        self.assertEqual(report['dispatched_without_invocation'],1)

    async def test_private_text_cannot_be_used_as_structural_terminal_reason(self):
        observer=session(self.root)
        worker=observer.reserve_work('worker_lifecycle',dict(worker_id='worker'))
        with self.assertRaises(ValueError):worker.finish('PRIVATE EXCEPTION')
        worker.finish('closed')
        request,record=await self.queued()
        call=observer.reserve_call(self.c,request,provider='test.client',model_revision='v1',prompt_sha256='a'*64)
        with self.assertRaises(ValueError):call.release('PRIVATE EXCEPTION')
        call.release('unadmitted')
        self.assertNotIn('PRIVATE',str(await observer.close()))

    async def test_explicit_retention_and_capacity_reject_before_dispatch(self):
        with self.assertRaises(PermissionError):session(self.root,retain_evaluation_hashes=False)
        observer=session(self.root,max_events=8)
        async def callback(request):self.fail('capacity failure reached backend')
        adapter=self.adapter(callback,evaluation_observer=observer)
        request,record=await self.queued()
        with self.assertRaises(RuntimeError):await adapter.execute(request,record,'q',maximum_charge=1)
        self.assertEqual(self.c.snapshot()['budget']['dispatched'],0)
        report=await observer.close()
        self.assertIn('journal_capacity_exhausted',report['incomplete_reasons'])
        self.assertEqual(report['provider_counters'],[])


@unittest.skipIf(smol_tests.np is None,'optional numpy absent')
class Smol(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=smol_tests.Calls.asyncSetUp
    asyncTearDown=smol_tests.Calls.asyncTearDown
    fixture=smol_tests.Calls.fixture
    setup_human=smol_tests.Calls.setup_human
    request=smol_tests.Calls.request
    async def test_actual_worker_lifecycle_and_one_semantic_count(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            contract=capability('smolvlm2.500m.local',dict(configuration(CONFIG),window_frames=1))
            observer=session(Path(directory).resolve(),contract.provider,contract.version)
            with self.fixture():
                human,controller,spec=self.setup_human()
                async with human:
                    async with await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,evaluation_observer=observer) as adapter:
                        request,record,frame=await self.request(human,controller,spec)
                        result=await adapter.execute(request,record,frame,maximum_charge=2)
                        self.assertEqual(result['status'],'published')
            report=await observer.close()
            self.assertEqual(report['provider_counters'][0]['count'],1)
            self.assertTrue(report['unresolved_billing'])
            worker=report['worker_measurements'][0]
            self.assertTrue(worker['reap_confirmed'])
            self.assertGreater(worker['startup_elapsed_ns'],0)
            marks={e['event']:e['monotonic_ns'] for e in report['journal'] if e['event'] in ('spawn_begin','handle_acquired','handshake_ready','reap_confirmed')}
            self.assertEqual(worker['residency_elapsed_ns'],marks['reap_confirmed']-marks['handle_acquired'])
            self.assertEqual(worker['observed_exchange_union_ns']+worker['observed_no_exchange_gap_ns'],worker['residency_elapsed_ns'])
            self.assertEqual(worker['startup_elapsed_ns'],marks['handshake_ready']-marks['spawn_begin'])
            self.assertEqual(report['incomplete_reasons'],[])

    async def test_cancelled_actual_worker_reaped_and_retained_as_one_attempt(self):
        class Started:
            def __init__(self):self.ready=asyncio.Event();self.process_id=None
            def emit(self,item):
                if item['event']=='work_capacity_reserved' and item['fields'].get('operation')=='process':
                    self.process_id=item['operation_id']
                if item['event']=='operation_started' and item['operation_id']==self.process_id:
                    self.ready.set()
        with tempfile.TemporaryDirectory(dir='.') as directory:
            contract=capability('smolvlm2.500m.local',dict(configuration(CONFIG),window_frames=1))
            sink=Started();observer=session(Path(directory).resolve(),contract.provider,contract.version,journal=sink)
            with self.fixture(delay=60):
                human,controller,spec=self.setup_human()
                async with human:
                    adapter=await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,evaluation_observer=observer)
                    request,record,frame=await self.request(human,controller,spec)
                    job=asyncio.create_task(adapter.execute(request,record,frame,maximum_charge=2))
                    await asyncio.wait_for(sink.ready.wait(),3)
                    job.cancel()
                    with self.assertRaises(asyncio.CancelledError):await job
                    await adapter.close()
            report=await observer.close()
            self.assertEqual(report['reports'][0]['attempts'][0]['status'],'cancelled')
            self.assertEqual(report['provider_counters'][0]['count'],1)
            self.assertTrue(report['worker_measurements'][0]['reap_confirmed'])
            self.assertTrue(report['unresolved_billing'])

    async def test_failed_actual_worker_startup_has_interval_without_semantic_attempt(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            observer=session(Path(directory).resolve())
            with self.fixture(code='raise RuntimeError("PRIVATE STARTUP ERROR")'):
                human,controller,spec=self.setup_human()
                async with human:
                    with self.assertRaises(Exception):
                        await smol_tests.SmolVLMCalls.open(human,'calls',sys.executable,CONFIG,evaluation_observer=observer)
            report=await observer.close()
            self.assertEqual(report['provider_counters'],[])
            self.assertEqual(report['reports'][0]['attempts'],[])
            worker=report['worker_measurements'][0]
            self.assertTrue(worker['reap_confirmed']);self.assertIsNotNone(worker['startup_elapsed_ns'])
            marks={e['event']:e['monotonic_ns'] for e in report['journal']}
            self.assertLessEqual(marks['handshake_failed'],marks['reap_confirmed'])
            self.assertEqual(worker['startup_elapsed_ns'],marks['handshake_failed']-marks['spawn_begin'])
            self.assertNotIn('PRIVATE STARTUP ERROR',str(report))
