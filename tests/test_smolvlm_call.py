import asyncio
from dataclasses import replace
import hashlib
import sys
import unittest

from humansense import HumanSense,SensorMetadata
from humansense.capabilities import CapabilitySpec,OutputField
from humansense.call_budget import CallBudget
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.mcp import MCPServer
from humansense.schema import timestamp
from humansense.smolvlm_call import SmolVLMCalls,smolvlm_frame_sha256
import test_smolvlm_context as smol_tests
from test_smolvlm_context import frame,np
from test_smolvlm_native import CONFIG


@unittest.skipIf(np is None,'optional numpy absent')
class Calls(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=smol_tests.Managed.asyncSetUp
    asyncTearDown=smol_tests.Managed.asyncTearDown
    fixture=smol_tests.Managed.fixture

    def setup_human(self):
        human=HumanSense(consent=True)
        spec=CapabilitySpec('thin.detect','thin.objects','video','thin.fixture','1',(OutputField('count','integer'),),'frame',freshness_ms=1000)
        human.register_capability(spec)
        controller=CallController(policy=CallPolicy(mode='escalation',cooldown_ms=0),
            budget=CallBudget(unit='fixture',pricing_revision='1',ceiling=20,max_inflight=8))
        human.register_call_controller('calls',controller)
        return human,controller,spec

    async def request(self,human,controller,spec,*,index=0,**rules):
        f=frame(index,timestamp());f=replace(f,sensor=replace(f.sensor,stream_epoch="camera-1",source_id="camera-"+str(index)))
        f=replace(f,data=np.arange(f.data.size,dtype=np.uint8).reshape(f.data.shape))
        r=await human.push_observation(spec.id,{'count':1},sensor=f.sensor,subject_id=f.subject_id,timestamp_ms=f.timestamp_ms)
        source=controller.open_source(r,now_ms=timestamp())
        ref=dict(source_id=f.sensor.source_id,sensor_id=f.sensor.sensor_id,stream_epoch=f.sensor.stream_epoch,
            capture_ms=f.timestamp_ms,sha256=smolvlm_frame_sha256(f.data))
        d=controller.submit(source,r,sequence=index,now_ms=timestamp(),visual_reference=ref,**(rules or {'unknown_candidate':True}))
        return d['request_id'],r,f

    async def test_actual_worker_admission_context_mcp_and_late_cost(self):
        with self.fixture():
            human,c,spec=self.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                    request,r,f=await self.request(human,c,spec)
                    result=await adapter.execute(request,r,f,maximum_charge=2)
                    self.assertEqual(result['status'],'published');self.assertIsNone(result['actual_cost'])
                    output=result['observation'];self.assertEqual(output['timestamp_ms'],f.timestamp_ms)
                    self.assertEqual(output['observation']['value']['semantic']['text'],'worker description')
                    self.assertEqual(c.snapshot()['budget']['reserved_liability'],2)
                    self.assertEqual(c.snapshot()['budget']['unknown_reports'],1)
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':adapter.worker.spec.id})
                    self.assertIn('worker description',str(reply))
                    c.settle(request,components={'local_compute':1},now_ms=timestamp())
                    self.assertEqual(c.snapshot()['budget']['spent_known'],1)

    async def test_denial_route_record_and_pixels_never_dispatch(self):
        with self.fixture():
            human,c,spec=self.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                    request,r,f=await self.request(human,c,spec)
                    reshaped=replace(f,data=f.data.reshape(48,32,3))
                    self.assertEqual(f.data.tobytes(),reshaped.data.tobytes())
                    self.assertNotEqual(smolvlm_frame_sha256(f.data),smolvlm_frame_sha256(reshaped.data))
                    with self.assertRaises(ValueError):await adapter.execute(request,r,reshaped,maximum_charge=2)
                    self.assertEqual(c.snapshot()['budget']['dispatched'],0)
                    bad=replace(f,data=f.data.copy());bad.data[:]=77
                    with self.assertRaises(ValueError):await adapter.execute(request,r,bad,maximum_charge=2)
                    with self.assertRaises(ValueError):await adapter.execute(request,dict(r,id='changed'),f,maximum_charge=2)
                    with self.assertRaises(ValueError):await adapter.execute(request,r,replace(f,sensor=replace(f.sensor,sequence=99)),maximum_charge=2)
                    result=await adapter.execute(request,r,f,maximum_charge=None)
                    self.assertEqual(result['blockers'],['unknown_charge_ceiling'])
                    strong,r2,f2=await self.request(human,c,spec,index=1,conflict_rule='verified-rule')
                    with self.assertRaises(ValueError):await adapter.execute(strong,r2,f2,maximum_charge=2)
                    self.assertEqual(c.snapshot()['budget']['dispatched'],0)

    async def test_lifecycle_discards_late_results_without_erasing_bill(self):
        for action in ('pause','reset','delete','consent','close'):
            with self.subTest(action=action),self.fixture(delay=.15):
                human,c,spec=self.setup_human()
                async with human:
                    async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                        request,r,f=await self.request(human,c,spec)
                        task=asyncio.create_task(adapter.execute(request,r,f,maximum_charge=2))
                        while c.snapshot()['budget']['dispatched']<1:await asyncio.sleep(.001)
                        if action=='pause':await human.pause()
                        elif action=='reset':await human.reset_processing()
                        elif action=='delete':await human.delete_data(f.subject_id)
                        elif action=='consent':human.consent=False
                        else:await human.close()
                        result=await task
                        self.assertEqual(result['status'],'discarded')
                        self.assertEqual(c.snapshot()['budget']['reserved_liability'],2)
                        c.settle(request,components={'local_compute':1},now_ms=timestamp())
                        self.assertEqual(c.snapshot()['budget']['spent_known'],1)
                        if not human.closed:
                            self.assertFalse(human.store.query(f.subject_id,kind='observation.signal',name=adapter.worker.spec.output_name))

    async def test_busy_mutation_copy_and_cancel_owned_worker(self):
        with self.fixture(delay=60):
            human,c,spec=self.setup_human()
            async with human:
                adapter=await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG)
                try:
                    request,r,f=await self.request(human,c,spec)
                    task=asyncio.create_task(adapter.execute(request,r,f,maximum_charge=2))
                    while c.snapshot()['budget']['dispatched']<1:await asyncio.sleep(.001)
                    with self.assertRaises(RuntimeError):await adapter.execute(request,r,f,maximum_charge=2)
                    f.data[:]=99
                    await adapter.close()
                    with self.assertRaises(asyncio.CancelledError):await task
                    self.assertTrue(adapter.worker.closed)
                    self.assertEqual(c.snapshot()['budget']['reserved_liability'],2)
                    self.assertEqual(c.snapshot()['budget']['unknown_reports'],1)
                finally:await adapter.close()

    async def test_retained_copy_and_empty_or_mismatched_results(self):
        with self.fixture(delay=.1):
            human,c,spec=self.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                    request,r,f=await self.request(human,c,spec)
                    digest=hashlib.sha256(f.data.tobytes()).hexdigest()
                    task=asyncio.create_task(adapter.execute(request,r,f,maximum_charge=2))
                    while c.snapshot()['budget']['dispatched']<1:await asyncio.sleep(.001)
                    f.data[:]=99
                    result=await task
                    self.assertEqual(result['observation']['observation']['value']['window']['captures'][0]['pixels_sha256'],digest)
        for mode in ('empty','wrong_prompt'):
            code=smol_tests.WORKER.replace('DELAY','0')
            if mode=='empty':code=code.replace("text='worker description'","text=''")
            else:
                extra="original=SmolVLMContext._observation\ndef bad(self,*args):\n    value=original(self,*args);value['model']['prompt_sha256']='0'*64;return value\nSmolVLMContext._observation=bad\n"
                code=code.replace('w._construct=',extra+'w._construct=')
            with self.subTest(mode=mode),self.fixture(code=code):
                human,c,spec=self.setup_human()
                async with human:
                    async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                        request,r,f=await self.request(human,c,spec)
                        if mode=='empty':
                            self.assertEqual((await adapter.execute(request,r,f,maximum_charge=2))['status'],'discarded')
                        else:
                            with self.assertRaises(ValueError):await adapter.execute(request,r,f,maximum_charge=2)
                        self.assertEqual(c.snapshot()['budget']['reserved_liability'],2)
                        self.assertTrue(all(s['last_semantic'] is None for s in c.snapshot()['sources'].values()))
