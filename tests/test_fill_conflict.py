"""Actual OpenCV + owned controller/visual calls; model responses are fixtures."""
import asyncio
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:cv2=np=None

from humansense import HumanSense,SensorMetadata
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.call_budget import CallBudget
from humansense.fill_conflict import FillConflictRuntime
from humansense.fill_conflict_rule import PREDICATE
from humansense.plugins import InputFrame
from humansense.schema import timestamp
from humansense.mcp import MCPServer


def configuration():
    return dict(width=64,height=64,roi=[8,8,56,56],hsv_lower=[110,100,100],hsv_upper=[130,255,255],
        volume_curve_ml=[[0,0],[1,100]],calibration_id='cal1',container_id='cup1',coordinate_frame='pixels')


def frame(seq=1,ts=None):
    pixels=np.zeros((64,64,3),np.uint8);pixels[32:,:,0]=255
    return InputFrame(pixels,'video',timestamp() if ts is None else ts,sensor=SensorMetadata('camera','image',seq,'pixels',('B','G','R'),'cal1','capture1',seq))


def answer(interval):
    return dict(text=json.dumps(dict(schema_version=1,predicate=PREDICATE,status='unknown' if interval is None else 'estimated',interval=interval)),usage=None)


@unittest.skipIf(np is None,'optional vision required')
class Runtime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h=HumanSense(consent=True);self.drivers=[]
    async def asyncTearDown(self):
        for driver in self.drivers:await driver.close()
        await self.h.close()

    def create(self,responses,*,strong=True,settle=True,cooldown=0,max_inflight=4,max_requests=128,ceilings=None,callback=None):
        controller=CallController(policy=CallPolicy(mode='escalation',cooldown_ms=cooldown,max_age_ms=1000),
            budget=CallBudget(unit='USD.micro',pricing_revision='v1',ceiling=100,max_inflight=max_inflight),max_requests=max_requests)
        self.h.register_call_controller('fill',controller);requests=[];holder={}
        async def completion(request):
            requests.append(deepcopy(request))
            if settle:
                rid=holder['driver'].snapshot()['active']['attempts'][-1]['request_id']
                controller.settle(rid,components={'other':1},now_ms=timestamp())
            if callback is not None:return await callback(request)
            return answer(responses[len(requests)-1])
        client=dict(completion=completion,provider='fixture.client',model_revision='fixture.v1')
        driver=FillConflictRuntime(self.h,'fill','fill',fill_config=configuration(),small_client=client,
            strong_client=client if strong else None,strong_opt_in=strong,
            charge_ceilings=ceilings or dict(initial=5,reinspection=5,strong=5),chain_ceiling=15,unit='USD.micro',pricing_revision='v1')
        holder['driver']=driver;self.drivers.append(driver)
        return driver,controller,requests

    async def test_actual_fill_reinspection_strong_mcp_and_no_target_leak(self):
        driver,c,requests=self.create([[.1,.2],[.1,.2],[.48,.52]])
        result=await driver.inspect(frame());value=result['observation']['value']['comparison']
        self.assertEqual(value['outcome'],'resolved_under_rule');self.assertEqual(len(requests),3)
        self.assertEqual([x['stage'] for x in value['attempts']],['initial','reinspection','strong'])
        self.assertEqual(c.snapshot()['budget']['spent_known'],3);self.assertEqual(c.snapshot()['budget']['reserved_liability'],0)
        for request in requests:
            user=json.loads(request['messages'][1]['content']);record=user['observation_record']
            self.assertEqual(record['observation']['capability']['id'],'vision.fill_inspection_input')
            encoded=json.dumps(request)
            for forbidden in ('height_resolution_interval','volume_ml','prior_claim','measurement_record_id','0.5'):
                self.assertNotIn(forbidden,encoded)
            self.assertEqual(request['images'][0]['source_frame_sha256'],value['source_sha256'])
        observations=await self.h.get_observations(capability_id='vision.calibrated_fill')
        numeric=observations['items'][0]['observation']['value']['fill']
        self.assertEqual(numeric['height_fraction'],.5);self.assertFalse(numeric['calibration_verified'])
        reply=await MCPServer(self.h).call_tool('get_observations',dict(capability_id='vision.fill_comparison'))
        self.assertEqual(reply['items'][0]['observation']['value']['comparison']['id'],value['id'])

    async def test_consistent_unknown_and_unknown_fill_do_not_escalate(self):
        driver,c,requests=self.create([[.48,.52]])
        first=await driver.inspect(frame(ts=timestamp()-10));self.assertEqual(first['observation']['value']['comparison']['outcome'],'consistent_under_rule')
        self.assertEqual(len(requests),1)
        original=frame(2,timestamp());original.data[:]=0
        result=await driver.inspect(original);self.assertEqual(result['observation']['value']['comparison']['outcome'],'indeterminate_fill')
        self.assertEqual(len(requests),1)

    async def test_same_frame_owned_despite_caller_mutation_and_no_confidence_route(self):
        entered=asyncio.Event();release=asyncio.Event()
        async def callback(request):entered.set();await release.wait();return dict(text='{"confidence":1,"status":"conflict"}',usage=None)
        driver,c,requests=self.create([],callback=callback)
        capture=frame();task=asyncio.create_task(driver.inspect(capture));await asyncio.wait_for(entered.wait(),5)
        capture.data[:]=0;release.set();result=await task
        value=result['observation']['value']['comparison'];self.assertEqual(value['outcome'],'indeterminate_invalid_claim')
        self.assertEqual(len(requests),1);self.assertEqual(requests[0]['images'][0]['source_frame_sha256'],value['source_sha256'])

    async def test_default_cooldown_and_unknown_liability_remain_blockers(self):
        driver,c,requests=self.create([[.1,.2]],cooldown=1000)
        result=await driver.inspect(frame());value=result['observation']['value']['comparison']
        self.assertEqual(value['outcome'],'blocked_stage');self.assertIn('cooldown',value['attempts'][-1]['blockers']);self.assertEqual(len(requests),1)

    async def test_unknown_bill_blocks_reinspection_without_ledger_reset(self):
        driver,c,requests=self.create([[.1,.2]],settle=False,max_inflight=1)
        result=await driver.inspect(frame());value=result['observation']['value']['comparison']
        self.assertEqual(value['outcome'],'blocked_stage');self.assertEqual(len(requests),1)
        budget=c.snapshot()['budget'];self.assertEqual(budget['reserved_liability'],5);self.assertEqual(budget['spent_known'],0)
        self.assertEqual(budget['unresolved_dispatched'],1)
        rid=value['attempts'][0]['request_id'];await driver.close()
        c.settle(rid,components={'other':2},now_ms=timestamp());self.assertEqual(c.snapshot()['budget']['spent_known'],2)

    async def test_slow_response_cannot_publish_fresh_current_claim(self):
        advanced=[];now=timestamp()
        async def callback(request):advanced.append(True);return answer([.48,.52])
        driver,c,requests=self.create([],callback=callback)
        with patch('humansense.fill_conflict.timestamp',side_effect=lambda:now+1001 if advanced else now):
            result=await driver.inspect(frame(ts=now))
        value=result['observation']['value']['comparison']
        self.assertEqual(value['outcome'],'blocked_stale_evidence');self.assertFalse(value['current_evidence'])
        self.assertEqual(result['timestamp_ms'],now);self.assertEqual(result['quality']['status'],'insufficient')
        self.assertEqual(len(requests),1)

    async def test_pause_scrubs_and_joins_during_client_without_late_publication(self):
        entered=asyncio.Event()
        async def callback(request):entered.set();await asyncio.Event().wait()
        driver,c,requests=self.create([],callback=callback,settle=False)
        task=asyncio.create_task(driver.inspect(frame()));await asyncio.wait_for(entered.wait(),5)
        await self.h.pause()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(driver.closed);self.assertIsNone(driver.snapshot()['active']);self.assertEqual(driver.snapshot()['history'],[])
        observed=await self.h.get_observations(capability_id='vision.fill_comparison');self.assertEqual(observed['items'],[])
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],5)

    async def test_each_stage_host_lifecycle_and_independent_context_progress(self):
        await self.h.close()
        for stage,action in ((1,'consent'),(2,'reset_processing'),(3,'delete_data'),(2,'close')):
            with self.subTest(stage=stage,action=action):
                self.h=HumanSense(consent=True);entered=asyncio.Event();count=[]
                async def callback(request):
                    count.append(True)
                    if len(count)==stage:entered.set();await asyncio.Event().wait()
                    return answer([.1,.2])
                driver,c,requests=self.create([],callback=callback,settle=False)
                task=asyncio.create_task(driver.inspect(frame()))
                await asyncio.wait_for(entered.wait(),5)
                # Independent audio/context APIs progress while visual callback waits.
                await self.h.push_signal('audio.rms',.1,'audio')
                self.assertTrue((await self.h.get_context())['observations'])
                if action=='consent':
                    self.h.consent=False;self.assertTrue(driver.closed);await self.h._join_call_drivers()
                else:await getattr(self.h,action)()
                with self.assertRaises(asyncio.CancelledError):await task
                self.assertTrue(driver._close_task.done());self.assertIsNone(driver.snapshot()['active'])
                self.assertEqual(driver.snapshot()['history'],[])
                self.assertEqual(c.snapshot()['budget']['reserved_liability'],5*stage)
                await self.h.close()

    async def test_newer_capture_supersedes_without_queue_and_reuses_same_source(self):
        entered=asyncio.Event();count=[]
        async def callback(request):
            count.append(True)
            if len(count)==1:entered.set();await asyncio.Event().wait()
            return answer([.48,.52])
        driver,c,requests=self.create([],callback=callback)
        first=frame(ts=timestamp()-10);task=asyncio.create_task(driver.inspect(first))
        await asyncio.wait_for(entered.wait(),5);source=driver._source
        newer=frame(2)
        with self.assertRaisesRegex(RuntimeError,'not queued'):await driver.inspect(newer)
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(driver._source,source);self.assertIsNone(driver.snapshot()['active'])
        result=await driver.inspect(newer)
        self.assertEqual(result['timestamp_ms'],newer.timestamp_ms)
        self.assertEqual(driver._source,source);self.assertEqual(len(c.snapshot()['sources']),1)

    async def test_unknown_claim_does_not_create_conflict(self):
        driver,c,requests=self.create([None])
        result=await driver.inspect(frame());self.assertEqual(result['observation']['value']['comparison']['outcome'],'indeterminate_claim')
        self.assertEqual(len(requests),1)

    async def test_actual_backend_observer_is_consumed_not_inferred_from_dispatch(self):
        from test_evaluation_hooks import Observer
        observer=Observer();driver,c,requests=self.create([[.48,.52]])
        driver.small._evaluation_observer=observer
        self.h._register_evaluation_observer(observer)
        result=await driver.inspect(frame());self.assertEqual(len(observer.tickets),1)
        self.assertTrue(observer.tickets[0].entered)
        self.assertIn(('0','finish:completed'),observer.events)
        self.assertFalse(result['observation']['value']['comparison']['whole_chain_cost_complete'])
        self.assertIsNone(result['observation']['value']['comparison']['local_overhead_cost'])

    async def test_strong_optout_retains_conflict(self):
        driver,c,requests=self.create([[.1,.2],[.1,.2]],strong=False)
        result=await driver.inspect(frame())
        self.assertEqual(result['observation']['value']['comparison']['outcome'],'strong_opt_out')
        self.assertEqual(len(requests),2)

    async def test_unknown_quote_never_enters_backend(self):
        driver,c,requests=self.create([],ceilings=dict(initial=None,reinspection=5,strong=5))
        result=await driver.inspect(frame());value=result['observation']['value']['comparison']
        self.assertEqual(requests,[]);self.assertEqual(c.snapshot()['budget']['dispatched'],0)
        self.assertIn('unknown_charge_ceiling',value['attempts'][0]['blockers'])

    async def test_capture_configuration_and_order_fail_closed(self):
        driver,c,requests=self.create([[.48,.52]])
        original=frame()
        for bad in (replace(original,sensor=replace(original.sensor,calibration_id='wrong')),
            replace(original,sensor=replace(original.sensor,stream_epoch=None)),
            replace(original,data=original.data.reshape(32,128,3))):
            with self.assertRaises(ValueError):await driver.inspect(bad)
        self.assertEqual(requests,[])
        await driver.inspect(original)
        changed=replace(original,data=original.data.copy());changed.data[:]=0
        with self.assertRaises(ValueError):await driver.inspect(changed)
        self.assertEqual(len(requests),1)

    async def test_atomic_next_submission_preserves_existing_order(self):
        driver,c,requests=self.create([[.1,.2],[.48,.52]])
        c._submission_seq=40
        result=await driver.inspect(frame())
        self.assertEqual(c._submission_seq,42)
        self.assertEqual(len(result['observation']['value']['comparison']['attempts']),2)

    async def test_case_deadline_joins_actual_visual_backend_and_preserves_liability(self):
        entered=asyncio.Event();cleaned=asyncio.Event()
        async def callback(request):
            entered.set()
            try:await asyncio.Event().wait()
            finally:cleaned.set()
        driver,c,requests=self.create([],settle=False,callback=callback)
        original=driver._execute_stage
        async def bounded(adapter,args,kwargs,deadline):
            return await original(adapter,args,kwargs,asyncio.get_running_loop().time()+.1)
        with patch.object(driver,'_execute_stage',bounded):
            result=await driver.inspect(frame())
        self.assertTrue(entered.is_set());self.assertTrue(cleaned.is_set())
        self.assertEqual(result['observation']['value']['comparison']['outcome'],'stage_failed')
        self.assertIsNone(driver.small._active);self.assertIsNone(driver.small._backend_task)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],5)
        self.assertEqual(len(requests),1)

    async def test_invalid_busy_capture_cannot_cancel_or_poison_owner(self):
        entered=asyncio.Event();release=asyncio.Event()
        async def callback(request):entered.set();await release.wait();return answer([.48,.52])
        driver,c,requests=self.create([],callback=callback)
        first=frame(ts=timestamp()-10);task=asyncio.create_task(driver.inspect(first))
        await asyncio.wait_for(entered.wait(),5)
        valid=frame(2)
        invalid=(replace(valid,modality='audio'),replace(valid,timestamp_ms=2**60),
            replace(valid,timestamp_ms=2**53-1),replace(valid,timestamp_ms=timestamp()+500),
            replace(valid,sensor=replace(valid.sensor,source_id='unrelated')),
            replace(valid,data=np.zeros((32,32,3),np.uint8)))
        for candidate in invalid:
            with self.assertRaises(ValueError):await driver.inspect(candidate)
            self.assertFalse(task.done());self.assertEqual(driver._last,(first.timestamp_ms,1))
            self.assertIsNone(driver._superseded_order);self.assertFalse(driver.closed)
        release.set();await task
        result=await driver.inspect(frame(2))
        self.assertEqual(result['observation']['value']['comparison']['outcome'],'consistent_under_rule')

    async def test_invalid_idle_future_does_not_advance_capture_order(self):
        driver,c,requests=self.create([[.48,.52]])
        for future in (2**53-1,timestamp()+500):
            with self.assertRaises(ValueError):await driver.inspect(frame(ts=future))
            self.assertIsNone(driver._last);self.assertIsNone(driver._owner)
        await driver.inspect(frame())
        self.assertEqual(len(requests),1)

    async def test_blocked_followups_release_capacity_across_fresh_cases(self):
        now=[timestamp()]
        # Advance the SDK/controller/client capture clocks together, not sleep or
        # silently relax production cooldown/freshness/queue capacity.
        with ExitStack() as stack:
            for name in ('humansense.fill_conflict.timestamp','humansense.engine.timestamp',
                         'humansense.text_context_call.timestamp','test_fill_conflict.timestamp'):
                stack.enter_context(patch(name,side_effect=lambda value=None:now[0] if value is None else value))
            driver,c,requests=self.create([[.1,.2]]*4,cooldown=1000,max_requests=2)
            for sequence in range(1,5):
                result=await driver.inspect(frame(sequence,now[0]))
                value=result['observation']['value']['comparison']
                self.assertEqual(value['outcome'],'blocked_stage')
                self.assertIn('cooldown',value['attempts'][-1]['blockers'])
                self.assertEqual(c.snapshot()['requests'],[])
                self.assertEqual(c.snapshot()['budget']['reserved_liability'],0)
                now[0]+=1001
        self.assertEqual(len(requests),4)
        self.assertEqual(c.snapshot()['overflow_rejections'],0)
