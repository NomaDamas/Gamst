import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_profiles import capability
from humansense.providers.landmarker_video import MediaPipeVideoContext
from humansense.schema import timestamp
try:
    import numpy as np
except ImportError:
    np = None


def output(kind):
    return SimpleNamespace(**{'body': {'pose_landmarks': []},
        'hand': {'hand_landmarks': [], 'handedness': []},
        'face': {'face_landmarks': [], 'face_blendshapes': []}}[kind])


class Detector:
    def __init__(self, kind):
        self.kind, self.calls, self.closed, self.fail = kind, [], 0, False
    def detect_for_video(self, image, stamp):
        self.calls.append((image.copy(), stamp))
        if self.fail: raise RuntimeError('native failure')
        return output(self.kind)
    def close(self): self.closed += 1


def provider(kind='face', **kwargs):
    made = []
    def factory():
        detector = Detector(kind); made.append(detector); return detector
    api = SimpleNamespace(Image=lambda **kw: kw['data'], ImageFormat=SimpleNamespace(SRGB=1))
    return MediaPipeVideoContext(kind, factory, api, sha256='0'*64, **kwargs), made


def frame(sequence=0, stamp=1000, **metadata):
    return InputFrame(np.zeros((100,200,3),dtype=np.uint8), 'video', stamp,
        sensor=SensorMetadata(metadata.pop('source_id','cam'), metadata.pop('sensor_id','sensor'),
                              sequence, **metadata))


class DiscoveryLoader(unittest.TestCase):
    def test_strict_profiles_and_version_identity(self):
        for kind in ('body','hand','face'):
            name='mediapipe.'+kind+'.video'
            config=dict(path='/not-installed',sha256='0'*64)
            spec=capability(name,config)
            self.assertEqual(spec.streaming_mode,'native')
            self.assertTrue(spec.stateful and spec.reset_on_gap)
            self.assertNotEqual(spec.id,capability('mediapipe.'+kind,config).id)
            for update in ({'max_entities':1},{'max_gap_ms':1001}):
                self.assertNotEqual(spec.version,capability(name,dict(config,**update)).version)
            for update in ({'max_entities':True},{'max_entities':5},{'max_gap_ms':0},
                           {'max_gap_ms':60001},{'max_gap_ms':1.5},{'unknown':1}):
                with self.assertRaises(ValueError): capability(name,dict(config,**update))

    def test_local_integrity_precedes_import(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; path.write_bytes(b'fixture')
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): MediaPipeVideoContext.from_local('hand',str(path),'0'*64)
                version.assert_not_called()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError):
                    MediaPipeVideoContext.from_local('hand',str(path),hashlib.sha256(b'fixture').hexdigest())

    @unittest.skipIf(np is None,'numpy not installed')
    def test_lazy_video_factories_reuse_verified_bytes_after_reset(self):
        for kind,name,count_key in (('body','Pose','num_poses'),('hand','Hand','num_hands'),('face','Face','num_faces')):
            captured=[]; made=[]
            def options(**kw): captured.append(kw); return kw
            def create(options):
                detector=Detector(kind); made.append(detector); return detector
            class Base:
                Delegate=SimpleNamespace(CPU=0)
                def __init__(self,**kw): self.__dict__.update(kw)
            vision=SimpleNamespace(**{name+'LandmarkerOptions':options,
                name+'Landmarker':SimpleNamespace(create_from_options=create)},
                RunningMode=SimpleNamespace(VIDEO='VIDEO'))
            python=SimpleNamespace(BaseOptions=Base,vision=vision)
            mp=SimpleNamespace(tasks=SimpleNamespace(python=python),
                Image=lambda **kw:kw['data'],ImageFormat=SimpleNamespace(SRGB=1))
            modules={'mediapipe':mp,'mediapipe.tasks':mp.tasks,'mediapipe.tasks.python':python}
            with tempfile.TemporaryDirectory(dir='.') as directory:
                path=Path(directory).resolve()/'model.task'; payload=b'not-a-model'; path.write_bytes(payload)
                with patch('importlib.metadata.version',return_value='0.10.35'),patch.dict('sys.modules',modules):
                    p=MediaPipeVideoContext.from_local(kind,str(path),hashlib.sha256(payload).hexdigest(),max_entities=2)
                self.assertEqual(made,[])  # first valid frame owns native initialization
                p.process(frame()); p.reset(); path.write_bytes(b'changed-after-verification')
                p.process(frame()); p.close()
                self.assertEqual(len(captured),2)
                for call in captured:
                    self.assertEqual(call['base_options'].model_asset_buffer,payload)
                    self.assertEqual(call['base_options'].delegate,0)
                    self.assertEqual(call['running_mode'],'VIDEO'); self.assertEqual(call[count_key],2)
                    if kind=='face':
                        self.assertTrue(call['output_face_blendshapes'])
                        self.assertFalse(call['output_facial_transformation_matrixes'])
                    if kind=='body': self.assertFalse(call['output_segmentation_masks'])
                self.assertTrue(all(d.closed==1 for d in made))


@unittest.skipIf(np is None,'numpy not installed')
class Continuity(unittest.TestCase):
    def test_all_three_modes_retain_native_state_and_relative_clock(self):
        for kind in ('body','hand','face'):
            p,made=provider(kind)
            first=p.process(frame(0,1000)); second=p.process(frame(1,1037))
            self.assertEqual(len(made),1)
            self.assertEqual([call[1] for call in made[0].calls],[0,37])
            self.assertEqual(first['summary']['state_segment'],second['summary']['state_segment'])
            self.assertTrue(second['summary']['temporal_tracking'])
            self.assertEqual(second['summary']['identity'],'unavailable')
            self.assertEqual(second['summary']['native_timestamp_ms'],37)
            p.close(); p.close(); self.assertEqual(made[0].closed,1)
            with self.assertRaises(RuntimeError): p.process(frame())

    def test_each_continuity_boundary_recreates_native_state(self):
        variants=[frame(2,1030), frame(1,2001), frame(1,1030,source_id='other'),
                  frame(1,1030,sensor_id='other'),frame(1,1030,stream_epoch='new'),
                  frame(1,1030,coordinate_frame='new'), frame(1,1030,calibration_id='new'),
                  frame(1,1030,channels=('new',)),
                  InputFrame(np.zeros((120,200,3),dtype=np.uint8),'video',1030,sensor=SensorMetadata('cam','sensor',1)),
                  InputFrame(np.zeros((100,200,3),dtype=np.uint8),'video',1030,'other',sensor=SensorMetadata('cam','sensor',1))]
        for f in variants:
            p,made=provider(); a=p.process(frame()); b=p.process(f)
            self.assertEqual(len(made),2); self.assertEqual(made[0].closed,1)
            self.assertNotEqual(a['summary']['state_segment'],b['summary']['state_segment'])
            self.assertEqual(made[1].calls[0][1],0); p.close()

    def test_bad_metadata_or_image_clears_state_before_recovery(self):
        for f in (frame(0,1030),frame(1,1000),frame(1,999),frame(1,1000.5),frame(1,2**53),
                  InputFrame(np.zeros((100,200,3),dtype=np.uint8),'video',1030),
                  InputFrame(np.zeros((100,200,3)),'video',1030,sensor=SensorMetadata('cam','sensor',1))):
            p,made=provider(); p.process(frame())
            with self.assertRaises(ValueError): p.process(f)
            self.assertEqual(made[0].closed,1)
            result=p.process(frame(2,1060)); self.assertEqual(result['summary']['native_timestamp_ms'],0)
            p.close()
        p,made=provider()
        with self.assertRaises(ValueError): p.process(InputFrame(np.zeros((4,4)),'video',1,sensor=SensorMetadata('cam','sensor',0)))
        self.assertEqual(made,[])

    def test_close_waits_for_owned_native_call(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        p,made=provider(); p.process(frame())
        entered,release,closing=threading.Event(),threading.Event(),threading.Event()
        def blocked(image,stamp):
            entered.set()
            if not release.wait(5): raise RuntimeError('test release timeout')
            return output('face')
        def close():
            closing.set(); p.close()
        made[0].detect_for_video=blocked
        with ThreadPoolExecutor(max_workers=2) as pool:
            running=pool.submit(p.process,frame(1,1030))
            try:
                self.assertTrue(entered.wait(2))
                closed=pool.submit(close)
                self.assertTrue(closing.wait(2))
                self.assertFalse(closed.done())
                self.assertEqual(made[0].closed,0)
            finally:
                release.set()
            self.assertEqual(running.result(timeout=2)['summary']['native_timestamp_ms'],30)
            closed.result(timeout=2)
        self.assertEqual(made[0].closed,1)

    def test_native_error_bad_results_reset_and_close_failure_stops(self):
        p,made=provider(); p.process(frame()); made[0].fail=True
        with self.assertRaises(RuntimeError): p.process(frame(1,1030))
        self.assertEqual(made[0].closed,1)
        p.process(frame(2,1060)); made[1].detect_for_video=lambda *args:SimpleNamespace(face_landmarks=[[]],face_blendshapes=[[]])
        with self.assertRaises(ValueError): p.process(frame(3,1090))
        self.assertEqual(made[1].closed,1)
        p.process(frame(4,1120))
        def fail_close(): raise RuntimeError('close failure')
        made[2].close=fail_close
        with self.assertRaises(RuntimeError): p.reset()
        self.assertTrue(p.closed)
        with self.assertRaises(RuntimeError): p.process(frame(5,1150))


@unittest.skipIf(np is None,'numpy not installed')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_video_context_managed_reset_and_mcp_provenance(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p,made=provider(); streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('cam','sensor',[p.spec.id])
            now=timestamp()-100
            await streams.submit(frame(0,now)); await streams.drain()
            await streams.submit(frame(2,now+40)); await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            self.assertEqual(result['returned_count'],2)
            self.assertEqual(len(made),2)
            items=sorted(result['items'],key=lambda item:item['observation']['sensor']['capture_sequence'])
            self.assertEqual([item['observation']['sensor']['capture_sequence'] for item in items],[0,2])
            self.assertNotEqual(items[0]['observation']['value']['summary']['state_segment'],
                                items[1]['observation']['value']['summary']['state_segment'])
            await streams.clear(); self.assertTrue(p.closed)
            self.assertTrue(all(d.closed==1 for d in made))
