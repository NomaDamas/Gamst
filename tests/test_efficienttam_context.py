import asyncio
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None

from humansense import HumanSense, SensorStreams, ProcessProvider, MaskTrackingPipeline
from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.efficienttam_context import EfficientTAMContext
from humansense.providers.efficienttam_native import configuration, _read, _weights
from humansense.providers.region_appearance import region_evidence

CONFIG = dict(source_dir='/source',checkpoint_path='/model.pt',checkpoint_sha256='a'*64,runtime_root='/runtime')


class Backend:
    def __init__(self): self.calls = []; self.resets = 0; self.closed = False; self.empty = False; self.bad = False
    def reset(self): self.resets += 1
    def close(self): self.closed = True
    def step(self, image, *, boxes=None):
        self.calls.append(boxes)
        if boxes is not None: self.count = len(boxes)
        h,w = image.shape[:2]
        masks = np.zeros((self.count,h,w),dtype=bool)
        if not self.empty: masks[:,2:10,3:12] = True
        return masks.astype(np.float32) if self.bad else masks


def frame(i=0, *, epoch='e1', timestamp_ms=None):
    return InputFrame(np.zeros((32,45,3),dtype=np.uint8),'video',1000+40*i if timestamp_ms is None else timestamp_ms,
        sensor=SensorMetadata('cam','video',i,coordinate_frame='image1',stream_epoch=epoch))


def rows(n=2):
    return [dict(track_id='seed'+str(i),label='cup',location=dict(unit='normalized_image',
        bbox=dict(x=.1,y=.1,width=.5,height=.5))) for i in range(n)]


def evidence(f,n=2):
    return region_evidence(f,rows(n),detector_id='fixture',detector_version='1',complete_snapshot=False)


def context(**kw):
    b=Backend(); return EfficientTAMContext(b,checkpoint_sha256='a'*64,**kw),b


class Discovery(unittest.TestCase):
    def test_dependency_light_profile_and_config(self):
        code = 'import sys;from humansense import ProcessProvider;ProcessProvider.capability("efficienttam.tiny512",'+repr(CONFIG)+');assert not {"numpy","torch","hydra","efficient_track_anything"}&sys.modules.keys()'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for bad in ({'device':'mps'},{'max_frames':True},{'max_gap_ms':0},{'source_dir':'relative'}, {'checkpoint_sha256':'X'*64}):
            with self.assertRaises(ValueError):ProcessProvider.capability('efficienttam.tiny512',dict(CONFIG,**bad))
        with self.assertRaises(ValueError):configuration(dict(CONFIG,extra=True))

    def test_verified_local_snapshot(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            p=Path(root).resolve()/'source.py';p.write_bytes(b'fixture')
            digest=hashlib.sha256(b'fixture').hexdigest()
            self.assertEqual(_read(p,digest,7),b'fixture')
            with self.assertRaises(ValueError):_read(p,digest,6)
            with self.assertRaises(ValueError):_read(p,'a'*64,7)

    @unittest.skipIf(torch is None,'optional torch unavailable')
    def test_strict_checkpoint_tensors(self):
        model=torch.nn.Linear(2,2).float()
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        def payload(value):
            buffer=io.BytesIO();torch.save(value,buffer);return buffer.getvalue()
        _weights(model,payload({'model':state}),torch)
        bad=[{}, {'model':{}}, {'model':state,'extra':1}, {'model':dict(state,weight=state['weight'].double())},
             {'model':dict(state,weight=torch.full_like(state['weight'],float('nan')))}]
        for value in bad:
            with self.assertRaises(ValueError):_weights(model,payload(value),torch)


@unittest.skipIf(np is None,'optional numpy unavailable')
class Context(unittest.TestCase):
    def test_seed_exact_capture_masks_and_continuity(self):
        p,b=context();f=frame()
        self.assertEqual(p.process(f)['summary']['status'],'needs_prompt')
        initial=p.process_regions(f,evidence(f));track=initial['objects'][0]['track_id']
        self.assertEqual(initial['objects'][0]['attributes']['mask']['foreground_pixels'],72)
        self.assertEqual(p.process(frame(1))['objects'][0]['track_id'],track)
        self.assertIsNone(b.calls[-1]);self.assertFalse(initial['summary']['complete_snapshot'])
        self.assertFalse(initial['summary']['model_quality_verified'])
        self.assertEqual(p.process(frame(3))['summary']['status'],'needs_prompt')
        self.assertEqual(len(b.calls),2)
        reseed=p.process_regions(frame(3),evidence(frame(3)))
        self.assertNotEqual(reseed['objects'][0]['track_id'],track)
        b.empty=True;empty=p.process(frame(4))
        self.assertEqual(empty['objects'],[]);self.assertEqual(empty['summary']['empty_tracks'][0]['mask_status'],'empty')
        p.close();p.close();self.assertTrue(b.closed)
        with self.assertRaises(RuntimeError):p.reset()

    def test_capture_hash_and_invalid_input_invalidate_state(self):
        p,b=context();f=frame();p.process_regions(f,evidence(f))
        changed=frame(1);bad=evidence(changed);changed.data[0,0,0]=1
        with self.assertRaises(ValueError):p.process_regions(changed,bad)
        self.assertIsNone(p.segment)
        self.assertEqual(p.process(frame(2))['summary']['status'],'needs_prompt')
        p.process_regions(frame(2),evidence(frame(2)))
        with self.assertRaises(ValueError):p.process(frame(2))
        self.assertIsNone(p.segment)
        p.process_regions(frame(3),evidence(frame(3)))
        b.bad=True
        with self.assertRaises(ValueError):p.process(frame(4))
        self.assertIsNone(p.segment)

    def test_owner_shape_time_and_capture_sequence_reset(self):
        for change in (lambda:frame(1,epoch='e2'),lambda:replace(frame(1),data=np.zeros((33,45,3),dtype=np.uint8)),
                       lambda:frame(1,timestamp_ms=3000),
                       lambda:replace(frame(1),sensor=replace(frame(1).sensor,capture_sequence=100))):
            p,b=context();p.process_regions(frame(),evidence(frame()))
            self.assertEqual(p.process(change())['summary']['status'],'needs_prompt')
            self.assertEqual(len(b.calls),1)
        p,b=context(max_frames=1);p.process_regions(frame(),evidence(frame()))
        self.assertEqual(p.process(frame(1))['summary']['status'],'needs_prompt')

    def test_seed_limits_and_payload_budget(self):
        p,b=context()
        for n in (0,5):
            with self.assertRaises(ValueError):p.process_regions(frame(),evidence(frame(),n))
        seed=evidence(frame(),4)
        for row in seed['objects']:row['label']='컵'*128
        result=p.process_regions(frame(),seed)
        from humansense.capabilities import bounded_json
        for row in result['objects']:bounded_json(row,max_bytes=4096,forbid_raw=True)
        bad=evidence(frame(1));bad['objects'][0]['location']['bbox']['width']=float('nan')
        with self.assertRaises(ValueError):p.process_regions(frame(1),bad)


@unittest.skipIf(np is None,'optional numpy unavailable')
class Managed(unittest.IsolatedAsyncioTestCase):
    def build(self):
        from humansense.providers.detection import RFDETRObjects
        class Detector:
            spec=RFDETRObjects.capability('test');calls=0;closed=False
            async def process(self,f):self.calls+=1;return dict(objects=rows(),summary=dict(complete_snapshot=True))
            async def reset(self):pass
            async def close(self):self.closed=True
        class Tracker:
            def __init__(self):self.native,self.backend=context();self.spec=self.native.spec;self.closed=False
            async def process(self,f):return self.native.process(f)
            async def process_regions(self,f,e):return self.native.process_regions(f,e)
            async def reset(self):self.native.reset()
            async def close(self):self.native.close();self.closed=True
        d,t=Detector(),Tracker();return MaskTrackingPipeline(d,t),d,t

    async def test_sensorstreams_mcp_memory_and_gap_reseed(self):
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        p,d,t=self.build()
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);streams.register(p.spec,lambda:p)
            await streams.open('cam','video',[p.spec.id]);now=timestamp()-1000
            tracks=[]
            for i in (0,1,3):
                await streams.submit(frame(i,epoch=None,timestamp_ms=now+i*40));await streams.drain()
                signals=human.store.latest_signals('local-user',0,timestamp())
                self.assertEqual(len(signals),1)
                record=signals[0];tracks.append(record['observation']['value']['objects'][0]['track_id'])
                await human.remember_objects(record['id'],complete_snapshot=False)
            self.assertEqual(tracks[0],tracks[1]);self.assertNotEqual(tracks[1],tracks[2]);self.assertEqual(d.calls,2)
            memory=await MCPServer(human).call_tool('get_object_memory',{})
            self.assertEqual(len(memory['items']),4)
            last_seen={item['track_id']:item['last_seen_ms'] for item in memory['items']}
            t.backend.empty=True
            await streams.submit(frame(4,epoch=None,timestamp_ms=now+160));await streams.drain()
            record=human.store.latest_signals('local-user',0,timestamp())[0]
            self.assertEqual(record['observation']['value']['objects'],[])
            await human.remember_objects(record['id'],complete_snapshot=False)
            memory=await MCPServer(human).call_tool('get_object_memory',{})
            self.assertEqual({item['track_id']:item['last_seen_ms'] for item in memory['items']},last_seen)
            expired=human.object_memory.find(now_ms=timestamp()+human.object_memory.visibility_ms+1)
            self.assertTrue(all(not item['currently_visible'] for item in expired))
            await streams.clear();self.assertTrue(d.closed);self.assertTrue(t.closed)

    async def test_cancel_and_partial_open_close_owned_workers(self):
        p,d,t=self.build();entered=asyncio.Event()
        async def stuck(f):entered.set();await asyncio.Future()
        d.process=stuck;task=asyncio.create_task(p.process(frame()));await entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(d.closed);self.assertTrue(t.closed)
        p,d,t=self.build()
        with patch('humansense.mask_tracking.ProcessProvider.open',AsyncMock(side_effect=[d,RuntimeError('missing runtime')])):
            with self.assertRaises(RuntimeError):await MaskTrackingPipeline.open('/d',dict(path='/fixture',sha256='b'*64),'/t',CONFIG)
        self.assertTrue(d.closed);await t.close()

    async def test_subpixel_detector_boxes_do_not_close_pipeline(self):
        p,d,t=self.build()
        candidates=rows();candidates[0]['location']['bbox']['width']=.01
        async def detect(f):return dict(objects=candidates,summary=dict(complete_snapshot=True))
        d.process=detect
        result=await p.process(frame())
        self.assertEqual(len(result['objects']),1)
        self.assertEqual(result['summary']['seed_candidates_unseedable'],1)
        await p.reset()
        candidates[1]['location']['bbox']['height']=.01
        result=await p.process(frame(1))
        self.assertEqual(result['summary']['status'],'needs_prompt');self.assertFalse(p.closed)
        candidates[1]['location']['bbox']['height']=.5
        result=await p.process(frame(2))
        self.assertEqual(len(result['objects']),1);await p.close()

    async def test_worker_regions_and_missing_assets_reap(self):
        children=[];real_spawn=asyncio.create_subprocess_exec
        code='''
import numpy as np
from humansense import provider_worker as w
from humansense.providers.efficienttam_context import EfficientTAMContext
class Backend:
 def reset(self):pass
 def close(self):pass
 def step(self,image,*,boxes=None):
  if boxes is not None:self.count=len(boxes)
  return np.ones((self.count,*image.shape[:2]),dtype=bool)
w._construct=lambda profile,config:(EfficientTAMContext(Backend(),checkpoint_sha256=config['checkpoint_sha256']),w.capability(profile,config))
w.main()
'''
        async def spawn(*args,**kw):
            child=await real_spawn(sys.executable,'-u','-c',code,**kw);children.append(child);return child
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                worker=await ProcessProvider.open(sys.executable,'efficienttam.tiny512',CONFIG)
                try:
                    self.assertEqual((await worker.process(frame()))['summary']['status'],'needs_prompt')
                    result=await worker.process_regions(frame(),evidence(frame()))
                    self.assertEqual(len(result['objects']),2)
                    self.assertEqual((await worker.process(frame(1)))['summary']['processed_frames'],2)
                finally:await worker.close()
            self.assertTrue(all(c.returncode is not None for c in children))
            # Unpatched native worker fails before importing/constructing a model.
            with self.assertRaises(RuntimeError):
                await ProcessProvider.open(sys.executable,'efficienttam.tiny512',CONFIG)
        finally:
            for child in children:
                if child.returncode is None:child.kill();await child.wait()

    async def test_replay_runner_marks_injected_gap_without_reusing_box(self):
        try:
            from PIL import Image
        except ImportError:self.skipTest('optional Pillow unavailable')
        import importlib.util,json
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('efficienttam_replay',Path(__file__).resolve().parents[1]/'examples/validate_efficienttam.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        p,d,t=self.build()
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();source=root/'input.png';output=root/'report.json'
            Image.new('RGB',(45,32)).save(source)
            args=SimpleNamespace(image=[str(source)]*3,period_ms=40,gap_at=2,source_dir='/source',
                checkpoint='/model.pt',sha256='a'*64,runtime_root=str(root),device='cpu',python=sys.executable,
                box=[.1,.1,.5,.5],label='cup',output=str(output))
            with patch.object(module.ProcessProvider,'open',AsyncMock(return_value=t)):
                await module.run(args)
            report=json.loads(output.read_text())
            self.assertFalse(report['quality_acceptance'])
            self.assertEqual([r['context']['summary']['status'] for r in report['observations']],['tracking','tracking','needs_prompt'])
            self.assertEqual(len(t.backend.calls),2);self.assertTrue(t.closed)
        await d.close()


if __name__ == '__main__':unittest.main()
