import asyncio
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense,ProcessProvider,SensorMetadata,SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.smolvlm_context import SmolVLMContext
from humansense.schema import timestamp
from humansense.mcp import MCPServer
from test_smolvlm_native import CONFIG
try:
    import numpy as np
except ImportError:np=None


def frame(i,ts=None,**changes):
    sensor=SensorMetadata('cam','video',i,coordinate_frame='pixels')
    value=InputFrame(np.full((32,48,3),i%255,np.uint8),'video',1000+i*500 if ts is None else ts,sensor=sensor)
    return replace(value,**changes)


class Backend:
    def __init__(self):self.calls=[];self.closed=False;self.text='visible scene'
    def predict(self,images,times):
        self.calls.append((images.copy(),times.copy()))
        return dict(text=self.text,input_tokens=100,output_tokens=2,output_limit_reached=True,
            cache_reused_across_windows=False,interpretation='untrusted_semantic_hypothesis',quality_validated=False)
    def close(self):self.closed=True


class Discovery(unittest.TestCase):
    def test_optional_free_profile(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("smolvlm2.500m.local",'+repr(CONFIG)+');assert not {"torch","numpy","PIL","transformers"}&sys.modules.keys()'
        p=subprocess.run([sys.executable,'-c',code],capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
        for bad in ({'window_frames':0},{'window_frames':True},{'max_gap_ms':10001},{'sequence_step':0}):
            with self.assertRaises(ValueError):SmolVLMContext.capability(**dict(CONFIG,**bad))


@unittest.skipIf(np is None,'optional numpy absent')
class Window(unittest.TestCase):
    def test_window_hash_capture_copy_and_completed_retention(self):
        b=Backend();p=SmolVLMContext(b,**CONFIG)
        f=frame(0);expected=hashlib.sha256(f.data.tobytes()).hexdigest()
        first=p.process(f);f.data[:]=99
        self.assertIsNone(first['semantic']['text']);self.assertEqual(first['semantic']['status'],'collecting')
        self.assertIsNone(p.process(frame(1)));self.assertIsNone(p.process(frame(2)))
        value=p.process(frame(3));self.assertEqual(value['semantic']['text'],'visible scene')
        self.assertEqual(value['window']['captures'][0]['pixels_sha256'],expected)
        self.assertEqual(b.calls[0][0][0][0,0].tolist(),[0,0,0]);self.assertEqual(b.calls[0][1],[1000,1500,2000,2500])
        self.assertEqual(value['window']['capture_end_ms'],2500);self.assertTrue(value['window']['complete'])
        self.assertEqual(p.images,[]);self.assertIsNone(p.process(frame(4)))
        p.close();p.close();self.assertTrue(b.closed)
        with self.assertRaises(RuntimeError):p.reset()
        with self.assertRaises(RuntimeError):p.process(frame(5))

    def test_gap_owner_geometry_clock_calibration_and_malformed_reset(self):
        for mutation in (
            lambda f:replace(f,sensor=replace(f.sensor,source_id='other')),
            lambda f:replace(f,sensor=replace(f.sensor,stream_epoch='new')),
            lambda f:replace(f,sensor=replace(f.sensor,calibration_id='changed')),
            lambda f:replace(f,data=np.zeros((48,48,3),np.uint8)),
            lambda f:replace(f,timestamp_ms=9999),
            lambda f:replace(f,sensor=replace(f.sensor,sequence=9))):
            p=SmolVLMContext(Backend(),**CONFIG);a=p.process(frame(0));v=p.process(mutation(frame(1)))
            self.assertEqual(v['window']['observed_frames'],1);self.assertNotEqual(a['window']['segment_id'],v['window']['segment_id'])
        p=SmolVLMContext(Backend(),**CONFIG);p.process(frame(0))
        with self.assertRaises(ValueError):p.process(frame(0))
        self.assertEqual(p.images,[])
        self.assertEqual(p.process(frame(1))['semantic']['status'],'collecting')
        for bad in (replace(frame(2),timestamp_ms=True),replace(frame(2),sample_rate=16000),replace(frame(2),sensor=None),replace(frame(2),data=np.ma.array(frame(2).data))):
            with self.assertRaises(ValueError):p.process(bad)
            self.assertEqual(p.images,[])
        p=SmolVLMContext(Backend(),**dict(CONFIG,max_gap_ms=10000))
        p.process(frame(0,0));p.process(frame(1,6000));v=p.process(frame(2,11000))
        self.assertEqual(v['window']['observed_frames'],1)

    def test_declared_sampling_stride_partial_finish_empty_and_invalid_result(self):
        b=Backend();p=SmolVLMContext(b,**dict(CONFIG,sequence_step=30))
        for i in range(2):p.process(replace(frame(i),sensor=replace(frame(i).sensor,capture_sequence=i*30)))
        v=p.finish();self.assertTrue(v['window']['final_partial']);self.assertFalse(v['window']['complete'])
        self.assertEqual([x['capture_sequence'] for x in v['window']['captures']],[0,30]);self.assertIsNone(p.finish())
        b.text='';p=SmolVLMContext(b,**dict(CONFIG,window_frames=1));self.assertEqual(p.process(frame(0))['semantic']['status'],'empty_generation')
        b.text='x'*8193
        with self.assertRaises(ValueError):p.process(frame(1))
        self.assertEqual(p.images,[])


WORKER='''
from humansense import provider_worker as w
from humansense.providers.smolvlm_context import SmolVLMContext
import time
class Backend:
    def predict(self,images,times):
        time.sleep(DELAY)
        return dict(text='worker description',input_tokens=100,output_tokens=2,output_limit_reached=True,cache_reused_across_windows=False,interpretation='untrusted_semantic_hypothesis',quality_validated=False)
    def close(self):pass
w._construct=lambda profile,config:(SmolVLMContext(Backend(),**config),w.capability(profile,config))
w.main()
'''


@unittest.skipIf(np is None,'optional numpy absent')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):self.children=[];self.spawn=asyncio.create_subprocess_exec
    async def asyncTearDown(self):
        leaked=[p for p in self.children if p.returncode is None]
        for p in leaked:p.kill();await p.wait()
        self.assertEqual(leaked,[])
    def fixture(self,delay=0,code=None):
        async def spawn(*args,**kwargs):
            p=await self.spawn(sys.executable,'-u','-c',code or WORKER.replace('DELAY',str(delay)),**kwargs)
            self.children.append(p);return p
        return patch('asyncio.create_subprocess_exec',spawn)

    async def test_owned_worker_mcp_history_partial_and_original_timestamp(self):
        with self.fixture():
            worker=await ProcessProvider.open(sys.executable,'smolvlm2.500m.local',CONFIG)
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                await streams.open('cam','video',[worker.spec.id]);start=timestamp()-3000
                try:
                    for i in range(4):await streams.submit(frame(i,start+i*500));await streams.drain()
                    records=human.store.latest_signals('local-user',0,timestamp());self.assertEqual(records[0]['timestamp_ms'],start+1500)
                    self.assertEqual(records[0]['observation']['value']['semantic']['text'],'worker description')
                    await streams.submit(frame(4,start+2000));await streams.drain()
                    self.assertEqual(human.store.latest_signals('local-user',0,timestamp())[0]['timestamp_ms'],start+1500)
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    self.assertIn('worker description',str(reply))
                    await streams.finish('cam','video')
                    last=human.store.latest_signals('local-user',0,timestamp())[0]
                    self.assertEqual(last['timestamp_ms'],start+2000);self.assertTrue(last['observation']['value']['window']['final_partial'])
                finally:await streams.clear();await worker.close()
            self.assertIsNotNone(self.children[-1].returncode)

    async def test_inference_timeout_and_cancellation_reap(self):
        for cancel in (False,True):
            with self.fixture(delay=60):
                worker=await ProcessProvider.open(sys.executable,'smolvlm2.500m.local',dict(CONFIG,window_frames=1),inference_timeout=.15)
                try:
                    task=asyncio.create_task(worker.process(frame(0)))
                    if cancel:
                        await asyncio.sleep(.02);task.cancel()
                        with self.assertRaises(asyncio.CancelledError):await task
                    else:
                        with self.assertRaises(asyncio.TimeoutError):await task
                finally:await worker.close()
                self.assertTrue(worker.closed);self.assertIsNotNone(self.children[-1].returncode)

    async def test_slow_inference_keeps_polling_and_clear_suppresses_late_result(self):
        with self.fixture(delay=.4):
            worker=await ProcessProvider.open(sys.executable,'smolvlm2.500m.local',dict(CONFIG,window_frames=1))
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);streams.register(worker.spec,lambda:worker);await streams.open('cam','video',[worker.spec.id])
                try:
                    await streams.submit(frame(0,timestamp()))
                    polls=0
                    for _ in range(5):
                        await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id});polls+=1;await asyncio.sleep(.01)
                    self.assertEqual(polls,5)
                    await streams.clear()
                    self.assertEqual(human.store.latest_signals('local-user',0,timestamp()),[])
                finally:await streams.clear();await worker.close()
            self.assertIsNotNone(self.children[-1].returncode)

    async def test_real_loader_missing_assets_reaps(self):
        # Unmodified worker construction, not a backend double.
        async def spawn(*args,**kwargs):
            p=await self.spawn(*args,**kwargs);self.children.append(p);return p
        with patch('asyncio.create_subprocess_exec',spawn):
            with self.assertRaises(RuntimeError):await ProcessProvider.open(sys.executable,'smolvlm2.500m.local',CONFIG)
        self.assertIsNotNone(self.children[-1].returncode)
