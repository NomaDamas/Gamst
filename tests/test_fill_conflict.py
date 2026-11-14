"""Actual OpenCV + owned controller/visual calls; model responses are fixtures."""
import asyncio
from copy import deepcopy
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

    def create(self,responses,*,strong=True,settle=True,cooldown=0,max_inflight=4,ceilings=None,callback=None):
        controller=CallController(policy=CallPolicy(mode='escalation',cooldown_ms=cooldown,max_age_ms=1000),
            budget=CallBudget(unit='USD.micro',pricing_revision='v1',ceiling=100,max_inflight=max_inflight))
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
        first=await driver.inspect(frame());self.assertEqual(first['observation']['value']['comparison']['outcome'],'consistent_under_rule')
        self.assertEqual(len(requests),1)
        original=frame(2,timestamp()+1);original.data[:]=0
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
