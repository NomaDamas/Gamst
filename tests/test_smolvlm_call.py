import asyncio
from dataclasses import replace
import hashlib
import sys
import unittest

from gamst import Gamst,SensorMetadata
from gamst.capabilities import CapabilitySpec,OutputField
from gamst.call_controller import CallController
from gamst.call_policy import CallPolicy
from gamst.mcp import MCPServer
from gamst.schema import timestamp
from gamst.smolvlm_call import SmolVLMCalls,smolvlm_frame_sha256
import test_smolvlm_context as smol_tests
from test_smolvlm_context import frame,np
from test_smolvlm_native import CONFIG


@unittest.skipIf(np is None,'optional numpy absent')
class Calls(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=smol_tests.Managed.asyncSetUp
    asyncTearDown=smol_tests.Managed.asyncTearDown
    fixture=smol_tests.Managed.fixture

    def setup_human(self):
        human=Gamst(consent=True)
        spec=CapabilitySpec('thin.detect','thin.objects','video','thin.fixture','1',(OutputField('count','integer'),),'frame',freshness_ms=1000)
        human.register_capability(spec)
        controller=CallController(policy=CallPolicy(mode='escalation',cooldown_ms=0))
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

    async def test_actual_worker_admission_context_and_mcp(self):
        with self.fixture():
            human,c,spec=self.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                    request,r,f=await self.request(human,c,spec)
                    result=await adapter.execute(request,r,f)
                    self.assertEqual(result['status'],'published')
                    output=result['observation'];self.assertEqual(output['timestamp_ms'],f.timestamp_ms)
                    self.assertEqual(output['observation']['value']['semantic']['text'],'worker description')
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':adapter.worker.spec.id})
                    self.assertIn('worker description',str(reply))
                    self.assertFalse(c.snapshot()['requests'])

    async def test_denial_route_record_and_pixels_never_dispatch(self):
        with self.fixture():
            human,c,spec=self.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                    request,r,f=await self.request(human,c,spec)
                    reshaped=replace(f,data=f.data.reshape(48,32,3))
                    self.assertEqual(f.data.tobytes(),reshaped.data.tobytes())
                    self.assertNotEqual(smolvlm_frame_sha256(f.data),smolvlm_frame_sha256(reshaped.data))
                    with self.assertRaises(ValueError):await adapter.execute(request,r,reshaped)
                    self.assertTrue(all(r['state']=='pending' for r in c.snapshot()['requests']))
                    bad=replace(f,data=f.data.copy());bad.data[:]=77
                    with self.assertRaises(ValueError):await adapter.execute(request,r,bad)
                    with self.assertRaises(ValueError):await adapter.execute(request,dict(r,id='changed'),f)
                    with self.assertRaises(ValueError):await adapter.execute(request,r,replace(f,sensor=replace(f.sensor,sequence=99)))
                    strong,r2,f2=await self.request(human,c,spec,index=1,conflict_rule='verified-rule')
                    with self.assertRaises(ValueError):await adapter.execute(strong,r2,f2)
                    self.assertTrue(all(r['state']=='pending' for r in c.snapshot()['requests']))

    async def test_lifecycle_discards_late_results(self):
        for action in ('pause','reset','delete','consent','close'):
            with self.subTest(action=action),self.fixture(delay=.15):
                human,c,spec=self.setup_human()
                async with human:
                    async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                        request,r,f=await self.request(human,c,spec)
                        task=asyncio.create_task(adapter.execute(request,r,f))
                        while not any(r['state']=='dispatched' for r in c.snapshot()['requests']):await asyncio.sleep(.001)
                        if action=='pause':await human.pause()
                        elif action=='reset':await human.reset_processing()
                        elif action=='delete':await human.delete_data(f.subject_id)
                        elif action=='consent':human.consent=False
                        else:await human.close()
                        result=await task
                        self.assertEqual(result['status'],'discarded')
                        self.assertFalse(c.snapshot()['requests'])
                        if not human.closed:
                            self.assertFalse(human.store.query(f.subject_id,kind='observation.signal',name=adapter.worker.spec.output_name))

    async def test_busy_mutation_copy_and_cancel_owned_worker(self):
        with self.fixture(delay=60):
            human,c,spec=self.setup_human()
            async with human:
                adapter=await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG)
                try:
                    request,r,f=await self.request(human,c,spec)
                    task=asyncio.create_task(adapter.execute(request,r,f))
                    while not any(r['state']=='dispatched' for r in c.snapshot()['requests']):await asyncio.sleep(.001)
                    with self.assertRaises(RuntimeError):await adapter.execute(request,r,f)
                    f.data[:]=99
                    await adapter.close()
                    with self.assertRaises(asyncio.CancelledError):await task
                    self.assertTrue(adapter.worker.closed)
                finally:await adapter.close()

    async def test_retained_copy_and_empty_or_mismatched_results(self):
        with self.fixture(delay=.1):
            human,c,spec=self.setup_human()
            async with human:
                async with await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG) as adapter:
                    request,r,f=await self.request(human,c,spec)
                    digest=hashlib.sha256(f.data.tobytes()).hexdigest()
                    task=asyncio.create_task(adapter.execute(request,r,f))
                    while not any(r['state']=='dispatched' for r in c.snapshot()['requests']):await asyncio.sleep(.001)
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
                            self.assertEqual((await adapter.execute(request,r,f))['status'],'discarded')
                        else:
                            with self.assertRaises(ValueError):await adapter.execute(request,r,f)
                            self.assertTrue(all(s['last_semantic'] is None for s in c.snapshot()['sources'].values()))

    async def test_external_close_does_not_join_context_owner(self):
        with self.fixture(delay=60):
            human,c,spec=self.setup_human()
            async with human:
                adapter=await SmolVLMCalls.open(human,'calls',sys.executable,CONFIG)
                request,r,f=await self.request(human,c,spec)
                async def owner():
                    async with adapter:
                        await adapter.execute(request,r,f)
                task=asyncio.create_task(owner())
                try:
                    async def dispatched():
                        while not any(r['state']=='dispatched' for r in c.snapshot()['requests']):await asyncio.sleep(.001)
                    await asyncio.wait_for(dispatched(),5)
                    await asyncio.wait_for(adapter.close(),5)
                    with self.assertRaises(asyncio.CancelledError):await task
                    self.assertTrue(adapter.worker.closed)
                    self.assertTrue(all(child.returncode is not None for child in self.children))
                finally:
                    if not task.done():task.cancel()
                    await asyncio.gather(task,return_exceptions=True)
                    await adapter.close()

