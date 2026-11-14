import asyncio
import copy
import unittest

from benchmarks.controller_attempts import RecordedContextClient
from benchmarks.call_attempts import AttemptRecorder,cost_quantities
from benchmarks.perception_comparison import plan_digest
from humansense import TextContextCalls,VisualContextCalls
from humansense.call_controller import _digest
from humansense.schema import timestamp
from test_itemized_costs import itemized
import test_text_context_call as text_tests
import test_visual_context_call as visual_tests


class Clients(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued

    def recorder(self,**kwargs):
        plan,_=itemized()
        return plan,AttemptRecorder(plan,expected_plan_sha256=plan_digest(plan),task_id='0',system='candidate',**kwargs)

    def adapter(self,callback,provider='text',kind=TextContextCalls,**kwargs):
        a=kind(self.h,'main',callback,provider=provider,model_revision='frozen-v1',**kwargs)
        self.adapters.append(a);return a

    def reduce(self,plan,recorder):
        report=recorder.snapshot()
        return cost_quantities(plan,report,expected_plan_sha256=plan_digest(plan),expected_report_sha256=report['sha256'],task_id='0',system='candidate')

    async def test_actual_text_call_binds_controller_and_late_verified_receipt(self):
        plan,rec=self.recorder();rid,r=await self.queued();captured=[]
        async def client(request):captured.append(copy.deepcopy(request));return dict(text='PRIVATE ANSWER',usage=None)
        wrapped=RecordedContextClient(self.c,rid,rec,client,phase='cold')
        a=self.adapter(wrapped)
        result=await a.execute(rid,r,'PRIVATE QUESTION',maximum_charge=10)
        row=rec.snapshot()['attempts'][0];pending=self.c.snapshot()['requests'][0]
        self.assertEqual(row['attempt_id'],wrapped.attempt_id)
        self.assertEqual(row['execution_sha256'],_digest(pending['execution']))
        self.assertEqual(row['prompt_sha256'],_digest(captured[0]));self.assertEqual(row['prompt_sha256'],result['prompt_sha256'])
        self.assertEqual(row['evidence_sha256'],_digest(r));self.assertEqual(row['role'],'text_llm')
        self.assertNotIn('PRIVATE',str(rec.snapshot()));self.assertEqual(row['status'],'success')
        rec.close()
        with self.assertRaises(ValueError):self.reduce(plan,rec)
        rec.report(wrapped.attempt_id,quantities={'text.output':3},receipt_sha256='d'*64)
        self.assertEqual(self.reduce(plan,rec)['text.output'],3)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],10)
        self.c.settle(rid,components={'other':2},now_ms=timestamp())
        self.assertEqual(self.c.snapshot()['budget']['spent_known'],2)
        with self.assertRaises(RuntimeError):await wrapped(captured[0])

    async def test_blocked_and_binding_mismatch_never_invoke_provider(self):
        plan,rec=self.recorder();rid,r=await self.queued();calls=[]
        async def client(request):calls.append(request);return dict(text='ok',usage=None)
        wrapped=RecordedContextClient(self.c,rid,rec,client,phase='unknown');a=self.adapter(wrapped)
        self.assertEqual((await a.execute(rid,r,'q',maximum_charge=None))['status'],'blocked')
        self.assertIsNone(wrapped.attempt_id);self.assertEqual(rec.snapshot()['attempts'],[])
        # A middleware that mutates the dispatched prompt must fail before network invocation.
        async def tamper(request):request['messages'][1]['content']='changed';return await wrapped(request)
        b=self.adapter(tamper)
        with self.assertRaises(ValueError):await b.execute(rid,r,'q',maximum_charge=10)
        self.assertEqual(calls,[]);self.assertEqual(rec.snapshot()['attempts'],[])
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],10)
        with self.assertRaises(ValueError):RecordedContextClient(self.c,rid,rec,lambda x:None,phase='warm')

    async def test_failure_cancel_invalid_output_and_discard_preserve_attempts(self):
        plan,rec=self.recorder()
        for mode in ('failure','cancel','invalid','delete'):
            rid,r=await self.queued();started=asyncio.Event();release=asyncio.Event()
            async def client(request):
                started.set();await release.wait()
                if mode=='failure':raise RuntimeError('PRIVATE')
                return dict(text='PRIVATE',usage={} if mode=='invalid' else None)
            wrapped=RecordedContextClient(self.c,rid,rec,client,phase='warm');a=self.adapter(wrapped)
            task=asyncio.create_task(a.execute(rid,r,'q',maximum_charge=10));await started.wait()
            if mode=='delete':await self.h.delete_data()
            if mode=='cancel':task.cancel()
            else:release.set()
            if mode in ('failure','cancel','invalid'):
                with self.assertRaises({'failure':RuntimeError,'cancel':asyncio.CancelledError,'invalid':ValueError}[mode]):await task
            else:self.assertEqual((await task)['status'],'discarded')
            row=rec.snapshot()['attempts'][-1]
            self.assertEqual(row['status'],{'failure':'error','cancel':'cancelled','invalid':'success','delete':'success'}[mode])
            self.assertIsNotNone(wrapped.attempt_id)
            rec.report(wrapped.attempt_id,quantities={'text.output':1},receipt_sha256='d'*64)
            self.c.settle(rid,components={'other':1},now_ms=timestamp())
        rec.close();self.assertEqual(self.reduce(plan,rec)['text.output'],4)
        self.assertNotIn('PRIVATE',str(rec.snapshot()))

    async def test_closed_or_exhausted_recorder_never_calls_client(self):
        for closed in (False,True):
            plan,rec=self.recorder(max_attempts=1);calls=[]
            async def client(request):calls.append(1);return dict(text='ok',usage=None)
            if closed:rec.close()
            else:
                async def existing(token):return None
                await rec.invoke(existing,provider='text',revision='frozen-v1',role='text_llm',phase='cold',evidence_sha256='a'*64)
            rid,r=await self.queued();wrapped=RecordedContextClient(self.c,rid,rec,client,phase='warm');a=self.adapter(wrapped)
            with self.assertRaises(RuntimeError):await a.execute(rid,r,'q',maximum_charge=10)
            self.assertEqual(calls,[]);self.assertIsNone(wrapped.attempt_id)
            self.assertEqual(rec.snapshot()['rejected'],1)

    @unittest.skipIf(visual_tests.np is None,'optional vision dependencies absent')
    async def test_actual_visual_client_exact_image_prompt_execution(self):
        plan,rec=self.recorder()
        rid,r,f=await visual_tests.Calls.queued(self,'strong');captured=[]
        async def client(request):captured.append(copy.deepcopy(request));return dict(text='visual answer',usage=None)
        wrapped=RecordedContextClient(self.c,rid,rec,client,phase='unknown')
        a=self.adapter(wrapped,provider='strong',kind=VisualContextCalls)
        result=await a.execute(rid,r,f,'q',maximum_charge=10)
        self.assertTrue(result['accepted']);row=rec.snapshot()['attempts'][0]
        self.assertEqual(row['role'],'strong_vlm');self.assertEqual(row['prompt_sha256'],_digest(captured[0]))
        self.assertTrue(captured[0]['images'][0]['data_base64'])
        self.assertNotIn('data_base64',str(rec.snapshot()))
        rec.report(wrapped.attempt_id,quantities={'strong.input':1,'cache.read':0,'cache.write':0},receipt_sha256='d'*64)
        rec.close();self.assertEqual(self.reduce(plan,rec)['strong.input'],1)
