import unittest
from dataclasses import replace
import test_sam_visual_window as windows
import test_sam_audio_context as audio
from humansense.sam_visual_packet import visual_policy,unpack_visual_packet
from humansense.providers.sam_audio_context import SAMAudioContext


class Backend(audio.FixtureBackend):
    def visual_parameters(self):return dict(image_size=224,batch_size=16)
    def predict(self,window,*,visual=None):
        self.visual=visual
        return super().predict(window)


@unittest.skipIf(windows.fixtures.np is None,'optional numpy missing')
class VisualContext(unittest.TestCase):
    def setUp(self):
        fixture=windows.Window();fixture.setUp()
        self.packet=fixture.fixture.packet()
        policy=visual_policy(unpack_visual_packet(self.packet.data)[0])
        self.config=dict(audio.CONFIG,prompt='',visual_policy=policy)
        self.backend=Backend()
        self.context=SAMAudioContext(self.backend,self.config)

    def tearDown(self):self.context.close()

    def test_actual_context_consumes_aligned_visual_and_empty_description(self):
        pair=self.context.process(self.packet)
        self.assertEqual(self.backend.last.prompt,'')
        self.assertEqual(len(self.backend.visual.indices),25)
        self.assertEqual(pair.observation['prompt']['utf8_bytes'],0)
        self.assertEqual(pair.observation['prompt']['visual']['mask_polarity'],'zero-kept')
        self.assertFalse(pair.observation['prompt']['visual']['clock_verified'])
        self.assertEqual(len(pair.target),48000*4)

    def test_bad_capture_never_reaches_backend(self):
        with self.assertRaises(ValueError):
            self.context.process(replace(self.packet,timestamp_ms=2001))
        self.assertFalse(hasattr(self.backend,'visual'))

    def test_reset_clears_both_histories(self):
        self.context.process(self.packet)
        self.context.reset()
        pair=self.context.process(self.packet)
        self.assertEqual(pair.observation['window']['discontinuity_reason'],'reset')


class OwnedVisual(unittest.IsolatedAsyncioTestCase):
    async def test_owned_worker_stream_context_and_mcp_visual_evidence(self):
        if windows.fixtures.np is None:self.skipTest('optional numpy missing')
        import asyncio,json,os,sys
        from unittest.mock import patch
        from humansense import HumanSense,ProcessProvider,SensorStreams
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        from humansense.separation_media import PROFILE
        fixture=windows.Window();fixture.setUp(); f=fixture.fixture
        now=timestamp(); shift=(now-f.audio.timestamp_ms)*1000
        f.audio=replace(f.audio,timestamp_ms=now)
        f.kw['intervals_us']=[[a+shift,b+shift] for a,b in f.kw['intervals_us']]
        packet=f.packet(); policy=visual_policy(unpack_visual_packet(packet.data)[0])
        config=dict(audio.CONFIG,prompt='',visual_policy=policy)
        child_code='''
from humansense import provider_worker as w
from humansense.providers.sam_audio_native import SAMAudioNative
from test_sam_visual_context import Backend
SAMAudioNative.from_local=classmethod(lambda cls,config:Backend())
w.main()
'''
        real_spawn=asyncio.create_subprocess_exec;children=[]
        async def spawn(*args,**kwargs):
            env=dict(kwargs.get('env',{}));env['PYTHONPATH']=os.path.abspath('tests')+os.pathsep+os.path.abspath('.')
            kwargs['env']=env
            child=await real_spawn(sys.executable,'-u','-c',child_code,**kwargs)
            children.append(child);return child
        async with HumanSense(consent=True) as human:
            owner=human.enable_separation_media(subject_id='subject',source_id='microphone',sensor_id='mic0',
                visual_source={'source_id':'camera','sensor_id':'cam0'})
            with patch('asyncio.create_subprocess_exec',spawn):
                provider=await ProcessProvider.open(sys.executable,PROFILE,config,separation_owner=owner)
            streams=SensorStreams(human);streams.register(provider.spec,lambda:provider)
            await streams.open('microphone','mic0',[provider.spec.id],subject_id='subject')
            try:
                await streams.submit(replace(packet,sensor=replace(packet.sensor,stream_epoch=None)))
                await streams.drain()
                context=await human.get_context(subject_id='subject')
                value=next(iter(context['observations'].values()))['value']
                evidence=value['window']['source_evidence']
                self.assertEqual(evidence['visual_source_id'],'camera')
                self.assertEqual(value['prompt']['visual']['feature_frames'],25)
                self.assertTrue(value['model']['fixture_engine'])
                first_ref=value['media']['reference']
                epoch=evidence['sensor']['stream_epoch']
                self.assertEqual(len(owner.retrieve(first_ref,'target',stream_epoch=epoch)),192000)
                response=await MCPServer(human).handle(dict(jsonrpc='2.0',id=1,method='tools/call',
                    params=dict(name='get_context',arguments=dict(subject_id='subject'))))
                text=json.dumps(response)
                self.assertIn('mapping_sha256',text)
                self.assertNotIn('masked_videos',text)
                self.assertNotIn('"rgb":',text)
                # Overlapping audio consumes the SAME visual capture IDs/hashes.
                f.audio=replace(f.audio,timestamp_ms=now+40,sensor=replace(f.audio.sensor,sequence=2))
                again=f.packet()
                await streams.submit(replace(again,sensor=replace(again.sensor,stream_epoch=None)))
                await streams.drain()
                context=await human.get_context(subject_id='subject')
                value=next(iter(context['observations'].values()))['value']
                self.assertEqual(value['window']['capture_end_ms'],now+40)
                self.assertEqual(value['prompt']['visual']['feature_frames'],25)
            finally:await streams.close()
            self.assertEqual(owner._store.bytes,0)
            self.assertTrue(all(c.returncode is not None for c in children))

    async def test_visual_grant_and_reservation_reject_before_worker(self):
        if windows.fixtures.np is None:self.skipTest('optional numpy missing')
        import sys
        from unittest.mock import patch,AsyncMock
        from humansense import HumanSense,ProcessProvider
        from humansense.separation_media import PROFILE
        fixture=windows.Window();fixture.setUp()
        policy=visual_policy(unpack_visual_packet(fixture.packet)[0])
        config=dict(audio.CONFIG,visual_policy=policy)
        async with HumanSense(consent=True) as human:
            owner=human.enable_separation_media(subject_id='subject',source_id='microphone',sensor_id='mic0')
            with patch('asyncio.create_subprocess_exec',new_callable=AsyncMock) as spawn:
                with self.assertRaises(PermissionError):
                    await ProcessProvider.open(sys.executable,PROFILE,config,separation_owner=owner)
                spawn.assert_not_called()
            visual=human.enable_separation_media(subject_id='subject',source_id='microphone',sensor_id='mic0',
                visual_source={'source_id':'camera','sensor_id':'cam0'})
            with self.assertRaises(ValueError):visual._reserve(192000,request_bytes=16*1024*1024)
            self.assertNotIn(visual,visual._store.reservations)


class NativeVisualCall(unittest.TestCase):
    def test_actual_native_predict_consumes_pinned_processor_masked_frames(self):
        import os,types
        import test_sam_audio_native as native_tests
        path=os.environ.get('GAMST_SAM_PROCESSOR_SOURCE')
        if not path or not native_tests.HAS_TORCH:
            self.skipTest('explicit pinned processor and optional Torch required')
        import torch
        from humansense.sam_visual_window import prepare_visual
        from humansense.sam_audio_windows import prepare_window
        f=windows.Window();f.setUp();source=f.fixture
        for i in range(10):source.rgb[i]=i+1
        source.masks[:,:,:,8:]=1
        packet=source.packet().data
        policy=visual_policy(unpack_visual_packet(packet)[0])
        visual=prepare_visual(packet,policy=policy,sample_count=48000,hop_length=1920,image_size=224,batch_size=16)
        window=prepare_window(source.audio.data,sample_rate=48000,prompt='',hop_length=1920,allow_empty_prompt=True)
        backend=native_tests.ActualCallBoundary().backend()
        backend.model.vision_encoder=types.SimpleNamespace(image_size=224,batch_size=16)
        original=backend.model.separate;observed=[]
        def separate(batch,**kwargs):
            observed.append(batch.masked_video[0].clone())
            return original(batch,**kwargs)
        backend.model.separate=separate
        with native_tests.pinned_processor(path) as Processor:
            backend.processor=Processor(1920,48000)
            target,residual,tokens,_=backend.predict(window,visual=visual)
        expected=(source.rgb*(source.masks==0))[list(visual.indices)]
        self.assertTrue(torch.equal(observed[0],torch.from_numpy(expected)))
        self.assertEqual(len(target),192000);self.assertEqual(len(residual),192000)
        self.assertEqual(tokens['token_count_after'],4)
        backend.model.vision_encoder.image_size=336
        with self.assertRaises(ValueError):backend.predict(window,visual=visual)
