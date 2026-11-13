from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.region_appearance import RegionAppearance,region_evidence
from humansense.provider_profiles import capability
try:
    import numpy as np
except ImportError:np=None


def frame(seq=0,ts=1000,epoch='one'):
    image=np.zeros((80,120,3),dtype=np.uint8);image[:,:60,2]=255;image[:,60:,0]=255
    return InputFrame(image,'video',ts,sensor=SensorMetadata('cam','sensor',seq,coordinate_frame='camera.normalized',stream_epoch=epoch))


def objects():
    return [dict(track_id='cup1',label='cup',location=dict(unit='normalized_image',bbox=dict(x=0,y=0,width=.5,height=1))),
            dict(track_id='cup2',label='cup',location=dict(unit='normalized_image',bbox=dict(x=.5,y=0,width=.5,height=1)))]


def evidence(f,rows=None):return region_evidence(f,objects() if rows is None else rows,detector_id='detector',detector_version='v1',complete_snapshot=True)


def provider(**kw):
    def predict(rgb):
        out=np.zeros(384);out[0]=rgb[:,:,0].mean();out[1]=rgb[:,:,2].mean();return out
    encoder=SimpleNamespace(predictor=predict,close=lambda:None)
    return RegionAppearance(encoder,sha256='0'*64,**kw)


class Metadata(unittest.TestCase):
    def test_profile(self):
        self.assertTrue(capability('dinov3.regions',dict(path='/missing',sha256='0'*64)).stateful)
        for options in (dict(max_regions=9),dict(top_k=3),dict(top_k=True),dict(capacity=0)):
            with self.assertRaises(ValueError):RegionAppearance.capability('0'*64,**options)


@unittest.skipIf(np is None,'optional numpy absent')
class Regions(unittest.TestCase):
    def test_same_capture_crop_and_no_same_frame_candidates(self):
        p=provider();a=frame();first=p.process_regions(a,evidence(a))
        self.assertTrue(all(not o['attributes']['appearance_candidates'] for o in first['objects']))
        b=frame(1,1001);second=p.process_regions(b,evidence(b))
        for index,item in enumerate(second['objects']):
            candidates=item['attributes']['appearance_candidates']
            self.assertEqual(candidates[0]['reference_id'],first['objects'][index]['attributes']['reference_id'])
            self.assertEqual([c['cosine_similarity'] for c in candidates],[1,0])
            self.assertIsNone(candidates[0]['current_location']);self.assertFalse(candidates[0]['identity_confirmed'])
        self.assertNotIn('embedding',str(second));self.assertEqual(len(p._entries),4)

    def test_capture_mismatch_fails_before_encoder(self):
        p=provider();f=frame();bound=evidence(f);calls=[];p.encoder.predictor=lambda rgb:calls.append(rgb)
        for altered in (replace(f,timestamp_ms=1001),replace(f,sensor=replace(f.sensor,sequence=1)),
                        replace(f,subject_id='other'),replace(f,data=f.data[:,::-1].copy())):
            with self.assertRaises(ValueError):p.process_regions(altered,bound)
        self.assertEqual(calls,[]);self.assertEqual(len(p._entries),0)

    def test_bounds_abstention_and_atomic_failure(self):
        p=provider(max_regions=1);f=frame();result=p.process_regions(f,evidence(f))
        self.assertFalse(result['summary']['complete_snapshot']);self.assertEqual(result['summary']['regions_omitted'],1)
        f=frame(1,1001);rows=objects()[:1];rows[0]['location']['bbox']['width']=.01
        self.assertEqual(p.process_regions(f,evidence(f,rows))['objects'][0]['attributes']['appearance_status'],'unavailable_crop')
        f=frame(2,1002);p.encoder.predictor=lambda _:np.zeros(384)
        before=list(p._entries)
        with self.assertRaises(ValueError):p.process_regions(f,evidence(f))
        self.assertEqual(p._last,(1001,1));self.assertEqual(len(p._entries),len(before))
        rows=objects();rows[1]['track_id']='cup1'
        with self.assertRaises(ValueError):p.process_regions(f,evidence(f,rows))
        rows=objects();rows[0]['location']['bbox']['x']=.7
        with self.assertRaises(ValueError):p.process_regions(f,evidence(f,rows))

    def test_lifecycle_ownership_expiry_and_disappearance(self):
        p=provider(capacity=2,retention_ms=10)
        with patch('humansense.providers.region_appearance.time.monotonic',return_value=1):
            f=frame();p.process_regions(f,evidence(f))
            f=frame(10,1005);empty=p.process_regions(f,evidence(f,[]));self.assertEqual(empty['objects'],[])
            f=frame(11,1006);self.assertEqual(len(p.process_regions(f,evidence(f))['objects'][0]['attributes']['appearance_candidates']),2)
            self.assertEqual(len(p._entries),2)
        with patch('humansense.providers.region_appearance.time.monotonic',return_value=2):
            f=frame(12,1007);self.assertFalse(p.process_regions(f,evidence(f))['objects'][0]['attributes']['appearance_candidates'])
        f=frame(epoch='two');self.assertFalse(p.process_regions(f,evidence(f))['objects'][0]['attributes']['appearance_candidates'])
        p.reset();self.assertFalse(p._entries);p.close();p.close()
        with self.assertRaises(RuntimeError):p.process_regions(f,evidence(f))


@unittest.skipIf(np is None,'optional numpy absent')
class Pipeline(unittest.IsolatedAsyncioTestCase):
    def build(self):
        from humansense.object_appearance import ObjectAppearancePipeline
        from humansense.providers.detection import RFDETRObjects
        class Detector:
            spec=RFDETRObjects.capability('test')
            closed=False;resets=0;rows=objects()
            async def process(self,f):return dict(objects=self.rows,summary=dict(complete_snapshot=True))
            async def reset(self):self.resets+=1
            async def close(self):self.closed=True
        class Regions:
            def __init__(self):self.native=provider();self.spec=self.native.spec;self.closed=False
            async def process_regions(self,f,e):return self.native.process_regions(f,e)
            async def reset(self):self.native.reset()
            async def close(self):self.native.close();self.closed=True
        d,r=Detector(),Regions();return ObjectAppearancePipeline(d,r),d,r

    async def test_sampling_disappearance_memory_and_mcp(self):
        from humansense import HumanSense,SensorStreams
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        p,d,r=self.build()
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);streams.register(p.spec,lambda:p)
            await streams.open('cam','sensor',[p.spec.id]);now=timestamp()-100
            for seq in (0,10,11):
                if seq==10:d.rows=[]
                if seq==11:d.rows=objects()
                f=frame(seq,now+seq,epoch=None);await streams.submit(f);await streams.drain()
                record=human.store.latest_signals('local-user',0,timestamp())[0]
                await human.remember_objects(record['id'],complete_snapshot=record['observation']['value']['summary']['complete_snapshot'])
                if seq==10:
                    memory=await MCPServer(human).call_tool('get_object_memory',{})
                    self.assertEqual(len(memory['items']),2)
                    self.assertTrue(all(item['visibility']=='previously_seen' and item['current_location'] is None for item in memory['items']))
            self.assertEqual(d.resets,1)
            value=record['observation']['value'];self.assertEqual(value['objects'][0]['attributes']['appearance_candidates'][0]['timestamp_ms'],now)
            self.assertEqual(value['objects'][0]['attributes']['appearance_candidates'][0]['cosine_similarity'],1)
            await streams.clear();self.assertTrue(d.closed);self.assertTrue(r.closed)

    async def test_native_input_snapshot_before_await(self):
        import asyncio
        p,d,r=self.build();f=frame();entered=asyncio.Event();release=asyncio.Event()
        async def detect(copy):
            entered.set();await release.wait();copy.data[:]=0
            return dict(objects=objects(),summary=dict(complete_snapshot=True))
        d.process=detect
        task=asyncio.create_task(p.process(f));await entered.wait();f.data[:]=0;release.set()
        result=await task
        self.assertTrue(all(o['attributes']['appearance_status']=='estimated' for o in result['objects']))
        await p.close()

    async def test_cancellation_closes_both_owned_workers(self):
        import asyncio
        p,d,r=self.build();entered=asyncio.Event()
        async def detect(f):entered.set();await asyncio.Future()
        d.process=detect
        task=asyncio.create_task(p.process(frame()));await entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(p.closed);self.assertTrue(d.closed);self.assertTrue(r.closed)

    async def test_close_waits_active_and_failure_closes_other(self):
        import asyncio
        p,d,r=self.build();entered=asyncio.Event();release=asyncio.Event()
        async def detect(f):entered.set();await release.wait();return dict(objects=[],summary=dict(complete_snapshot=True))
        d.process=detect
        task=asyncio.create_task(p.process(frame()));await entered.wait()
        closing=asyncio.create_task(p.close());await asyncio.sleep(0);self.assertFalse(r.closed)
        release.set();await task;await closing;self.assertTrue(r.closed)
        p,d,r=self.build()
        async def fail():d.closed=True;raise RuntimeError('failed close')
        d.close=fail
        with self.assertRaises(RuntimeError):await p.close()
        self.assertTrue(r.closed)

    async def test_partial_open_failure_releases_first_runtime(self):
        from humansense.object_appearance import ObjectAppearancePipeline
        from unittest.mock import AsyncMock
        p,d,r=self.build()
        config=dict(path='/explicit',sha256='0'*64)
        opening=AsyncMock(side_effect=[d,RuntimeError('second environment unavailable')])
        with patch('humansense.object_appearance.ProcessProvider.open',opening):
            with self.assertRaises(RuntimeError):await ObjectAppearancePipeline.open('/det/python',config,'/embed/python',config)
        self.assertTrue(d.closed);self.assertEqual(opening.call_args_list[0].args[0],'/det/python')
        self.assertEqual(opening.call_args_list[1].args[0],'/embed/python')
        await r.close()

    async def test_real_worker_region_operation_and_capture_rejection(self):
        import asyncio,sys
        from humansense import ProcessProvider
        children=[];real_spawn=asyncio.create_subprocess_exec
        code='''
from humansense import provider_worker as w
from humansense.providers.region_appearance import RegionAppearance
from types import SimpleNamespace
import numpy as np
w._construct=lambda profile,config:(RegionAppearance(SimpleNamespace(predictor=lambda rgb:np.ones(384),close=lambda:None),sha256=config['sha256']),w.capability(profile,config))
w.main()
'''
        async def spawn(*args,**kw):
            child=await real_spawn(sys.executable,'-u','-c',code,**kw);children.append(child);return child
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                worker=await ProcessProvider.open(sys.executable,'dinov3.regions',dict(path='/fixture',sha256='0'*64))
                try:
                    f=frame();result=await worker.process_regions(f,evidence(f));self.assertEqual(len(result['objects']),2)
                    bad=frame(1,1001)
                    with self.assertRaises(RuntimeError):await worker.process_regions(bad,evidence(f))
                    self.assertTrue(worker.closed)
                finally:await worker.close()
            self.assertTrue(all(c.returncode is not None for c in children))
        finally:
            for child in children:
                if child.returncode is None:child.kill();await child.wait()
