import json
import unittest
from unittest.mock import patch

from humansense import HumanSense, SensorMetadata
from humansense.capabilities import CapabilitySpec, OutputField
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.call_budget import CallBudget
from humansense.mcp import MCPServer
from humansense.schema import timestamp


class Integration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now=timestamp()
        self.clock=patch('humansense.engine.timestamp',side_effect=lambda value=None:self.now if value is None else value)
        self.clock.start()
        self.human=HumanSense(consent=True)
        self.spec=CapabilitySpec('test.context','test.context','video','test.provider','1',
            (OutputField('text','string'),),'frame',freshness_ms=1000)
        self.human.register_capability(self.spec)
        self.c=self.make();self.human.register_call_controller('main',self.c)
        self.server=MCPServer(self.human)

    def make(self):
        return CallController(policy=CallPolicy(mode='sentinel',cooldown_ms=0),
            budget=CallBudget(unit='USD.micro',pricing_revision='v1',ceiling=100),max_requests=4)

    async def asyncTearDown(self):
        await self.human.close();await self.server.close();self.clock.stop()

    async def queued(self,subject='alice',sequence=1):
        r=await self.human.push_observation(self.spec.id,{'text':'PRIVATE-BODY'},subject_id=subject,
            sensor=SensorMetadata('cam','video',sequence,coordinate_frame='pixels',stream_epoch='e'),timestamp_ms=self.now)
        h=self.c.open_source(r,now_ms=self.now)
        d=self.c.submit(h,r,sequence=sequence,now_ms=self.now,visual_reference=dict(source_id='cam',sensor_id='video',stream_epoch='e',capture_ms=self.now,sha256='a'*64))
        return h,d,r

    def reserve(self,d):
        return self.c.reserve(d['request_id'],provider='test.vlm',model_revision='v1',prompt_sha256='b'*64,
                              maximum_charge=20,now_ms=self.now)

    def dispatch(self,d):
        self.reserve(d);return self.c.dispatch(d['request_id'],now_ms=self.now)

    async def test_wire_status_is_bounded_and_never_contains_bodies(self):
        h,d,r=await self.queued();self.dispatch(d)
        response=await self.server.handle({'jsonrpc':'2.0','id':1,'method':'tools/call',
            'params':{'name':'get_call_status','arguments':{'controller_id':'main'}}})
        self.assertFalse(response['result']['isError'])
        text=response['result']['content'][0]['text'];self.assertNotIn('PRIVATE-BODY',text)
        status=json.loads(text);self.assertEqual(status['entries'][0]['budget']['reserved_liability'],20)
        small=await self.human.get_call_status(limit=1,max_bytes=1024)
        self.assertTrue(small['truncated']);self.assertEqual(small['omitted_count'],small['total_count']-small['returned_count'])
        self.assertLessEqual(sum(len(json.dumps(r,ensure_ascii=True).encode()) for r in small['entries']),1024)
        with self.assertRaises(ValueError):await self.human.get_call_status(max_bytes=10)

    async def test_subject_delete_detaches_bill_and_rejects_late_result(self):
        h,d,r=await self.queued();self.dispatch(d);t=d['request_id']
        await self.human.delete_data('alice')
        state=self.c.snapshot();self.assertEqual(state['detached_financial_requests'],[t])
        self.assertNotIn('alice',json.dumps(state));self.assertNotIn('PRIVATE-BODY',repr(self.c._requests))
        self.assertNotIn(d['decision']['evidence']['record_sha256'],json.dumps(state))
        with self.assertRaises(ValueError):self.c.finish(t,success=True,evidence_sha256=d['decision']['evidence']['record_sha256'],result_sha256='c'*64,now_ms=self.now)
        self.c.settle(t,components=None,now_ms=self.now)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],20)
        self.c.settle(t,components={'input':9},now_ms=self.now)
        self.assertEqual(self.c.snapshot()['budget']['spent_known'],9)
        self.assertFalse(self.c.snapshot()['detached_financial_requests'])

    async def test_pause_and_resume_do_not_refund_or_accept_old_sources(self):
        h,d,r=await self.queued();self.dispatch(d)
        await self.human.pause()
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],20)
        with self.assertRaises(RuntimeError):self.c.open_source(r,now_ms=self.now)
        result=self.c.finish(d['request_id'],success=True,evidence_sha256=d['decision']['evidence']['record_sha256'],result_sha256='c'*64,now_ms=self.now)
        self.assertFalse(result['accepted'])
        await self.human.resume()
        with self.assertRaises(ValueError):self.c.open_source(r,now_ms=self.now)
        self.c.settle(d['request_id'],components={'input':2},now_ms=self.now)
        self.assertEqual(self.c.snapshot()['budget']['spent_known'],2)

    async def test_pending_reserved_delete_and_close_outstanding(self):
        h,d,r=await self.queued();self.reserve(d)
        await self.human.delete_data('alice')
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],0)
        self.assertFalse(self.c.snapshot()['requests'])
        h,d,r=await self.queued('bob',2);self.dispatch(d)
        await self.human.close()
        self.assertNotIn('bob',json.dumps(self.c.snapshot()))
        self.c.settle(d['request_id'],components={'other':3},now_ms=self.now)
        status=await self.human.get_call_status()
        self.assertTrue(status['closed']);self.assertEqual(status['entries'][0]['budget']['spent_known'],3)

    async def test_revocation_and_epoch_race_reject_results(self):
        h,d,r=await self.queued();self.dispatch(d)
        self.human._generation+=1  # Interval before asynchronous reset cleanup runs.
        result=self.c.finish(d['request_id'],success=True,evidence_sha256=d['decision']['evidence']['record_sha256'],result_sha256='c'*64,now_ms=self.now)
        self.assertFalse(result['accepted'])
        self.human.consent=False
        with self.assertRaises(PermissionError):self.c.open_source(r,now_ms=self.now)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],20)
        self.human.consent=True
        with self.assertRaises(ValueError):self.c.open_source(r,now_ms=self.now)

    async def test_binding_limits_provenance_and_runtime_clock(self):
        with self.assertRaises(ValueError):self.human.register_call_controller('main',self.make())
        with self.assertRaises(ValueError):self.human.register_call_controller('other',self.c)
        other=HumanSense(consent=False)
        try:
            with self.assertRaises(PermissionError):other.register_call_controller('x',self.make())
        finally:await other.close()
        for i in range(7):self.human.register_call_controller('extra'+str(i),self.make())
        with self.assertRaises(ValueError):self.human.register_call_controller('overflow',self.make())
        h,d,r=await self.queued();self.now+=2000
        admission=self.c.reserve(d['request_id'],provider='test.vlm',model_revision='v1',prompt_sha256='b'*64,maximum_charge=1,now_ms=r['timestamp_ms'])
        self.assertIn('stale_capture',admission['blockers'])
        r['models'][0]['version']='fake'
        with self.assertRaises(ValueError):self.c.open_source(r,now_ms=self.now)

    async def test_executable_wiring_success_and_failure_liability(self):
        from examples.call_controller_demo import main, execute
        result=await main()
        self.assertEqual(result['calls'],1);self.assertTrue(result['synthetic_example'])
        r=await self.human.push_observation(self.spec.id,{'text':'PRIVATE-BODY'},
            sensor=SensorMetadata('cam','video',0,coordinate_frame='pixels',stream_epoch='e'),timestamp_ms=self.now)
        async def failing(*args):
            raise RuntimeError('transport failed')
        with self.assertRaisesRegex(RuntimeError,'transport failed'):
            await execute(self.human,self.c,r,b'fixture',failing)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],1)
        self.assertEqual(self.c.snapshot()['budget']['unknown_reports'],1)

    async def test_delete_completed_receipt_and_other_subject_liability(self):
        h,a,r=await self.queued();self.dispatch(a)
        self.c.finish(a['request_id'],success=True,evidence_sha256=a['decision']['evidence']['record_sha256'],result_sha256='c'*64,now_ms=self.now)
        self.c.settle(a['request_id'],components={'other':4},now_ms=self.now)
        h,b,r=await self.queued('bob',2);self.dispatch(b)
        await self.human.delete_data('alice')
        state=self.c.snapshot()
        self.assertNotIn('alice',json.dumps(state))
        self.assertEqual(state['budget']['spent_known'],4)
        self.assertEqual(state['budget']['reserved_liability'],20)
        self.c.settle(b['request_id'],components={'other':3},now_ms=self.now)
        self.assertEqual(self.c.snapshot()['budget']['spent_known'],7)
