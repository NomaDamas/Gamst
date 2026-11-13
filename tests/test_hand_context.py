import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_profiles import capability
from humansense.providers.hand_context import MediaPipeHandContext
from humansense.schema import timestamp
def points():
    return [SimpleNamespace(x=.2+(i%5)*.1,y=.2+(i//5)*.1,z=0.) for i in range(21)]
try:
    import numpy as np
except ImportError:
    np=None


class Detector:
    def __init__(self,bodies): self.bodies=bodies; self.images=[]; self.closed=False
    def detect(self,image): self.images.append(image.copy()); return SimpleNamespace(hand_landmarks=self.bodies,handedness=[[SimpleNamespace(category_name='Left',score=.9)] for _ in self.bodies])
    def close(self): self.closed=True


def provider(bodies):
    detector=Detector(bodies)
    api=SimpleNamespace(Image=lambda **kw:kw['data'],ImageFormat=SimpleNamespace(SRGB=1))
    return MediaPipeHandContext(detector,api,sha256='0'*64)


def frame():
    return InputFrame(np.zeros((100,200,3),dtype=np.uint8),'video',timestamp()-100,
                      sensor=SensorMetadata('camera','sensor',0))


class Loader(unittest.TestCase):
    def test_metadata_validation_and_preimport_integrity(self):
        spec=capability('mediapipe.hand',dict(path='/not-installed',sha256='0'*64))
        self.assertEqual(spec.streaming_mode,'frame'); self.assertFalse(spec.stateful)
        self.assertNotEqual(spec.version,capability('mediapipe.hand',dict(path='x',sha256='0'*64,max_hands=1)).version)
        for args in ({},dict(path='x',sha256='bad'),dict(path='x',sha256='0'*64,max_hands=True),
                     dict(path='x',sha256='0'*64,max_hands=5),dict(path='x',sha256='0'*64,extra=1)):
            with self.assertRaises(ValueError): capability('mediapipe.hand',args)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; path.write_bytes(b'fixture')
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): MediaPipeHandContext.from_local(str(path),'0'*64)
                version.assert_not_called()
            digest=hashlib.sha256(b'fixture').hexdigest()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): MediaPipeHandContext.from_local(str(path),digest)

    def test_loader_passes_exact_verified_buffer_and_image_mode(self):
        detector=Detector([]); captured={}
        def options(**kwargs): captured.update(kwargs); return kwargs
        class Base:
            Delegate=SimpleNamespace(CPU=0)
            def __init__(self,**kwargs): self.__dict__.update(kwargs)
        vision=SimpleNamespace(HandLandmarker=SimpleNamespace(create_from_options=lambda o:detector),
                               HandLandmarkerOptions=options,RunningMode=SimpleNamespace(IMAGE='IMAGE'))
        python=SimpleNamespace(BaseOptions=Base,vision=vision)
        mp=SimpleNamespace(tasks=SimpleNamespace(python=python))
        modules={'mediapipe':mp,'mediapipe.tasks':mp.tasks,'mediapipe.tasks.python':python}
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; payload=b'fixture-not-a-model'; path.write_bytes(payload)
            with patch('importlib.metadata.version',return_value='0.10.35'),patch.dict('sys.modules',modules):
                p=MediaPipeHandContext.from_local(str(path),hashlib.sha256(payload).hexdigest(),max_hands=2)
            self.assertEqual(captured['base_options'].model_asset_buffer,payload)
            self.assertEqual(captured['running_mode'],'IMAGE')
            self.assertEqual(captured['num_hands'],2)
            p.close(); self.assertTrue(detector.closed)


@unittest.skipIf(np is None,'numpy not installed')
class Geometry(unittest.TestCase):
    def test_projected_geometry_and_rgb_copy(self):
        hand=points()
        for i,xy in {5:(.25,.5),6:(.5,.5),7:(.75,1.),17:(.75,.5),4:(.5,.25),8:(.5,.75)}.items():
            hand[i].x,hand[i].y=xy
        p=provider([hand])
        f=frame(); f.data[:,:,0]=17; f.data[:,:,2]=42
        result=p.process(f); geometry=result['hands'][0]['geometry']
        self.assertAlmostEqual(geometry['index_pip_projected_angle_deg'],135.,places=3)
        self.assertEqual(geometry['thumb_index_distance_over_palm_width'],.5)
        larger=InputFrame(np.zeros((200,400,3),dtype=np.uint8),'video',1)
        self.assertEqual(p.process(larger)['hands'][0]['geometry'],geometry)
        self.assertEqual(p.detector.images[0][0,0].tolist(),[42,0,17])
        self.assertEqual(f.data[0,0].tolist(),[17,0,42])
        self.assertLess(len(json.dumps(result)),8192)
        self.assertNotIn('landmarks',json.dumps(result))
        self.assertEqual(result['hands'][0]['joint_confidence'],'unavailable')
        self.assertFalse(result['summary']['metric_3d'])
        self.assertFalse(result['summary']['temporal_tracking'])

    def test_offimage_and_degenerate_abstain(self):
        for mode in ('offimage','degenerate'):
            hand=points()
            if mode=='offimage': hand[6].x=1.1
            else: hand[5].x,hand[5].y=hand[6].x,hand[6].y
            self.assertIsNone(provider([hand]).process(frame())['hands'][0]['geometry']['index_pip_projected_angle_deg'])
        for mode in ('offimage','subpixel'):
            hand=points()
            if mode=='offimage': hand[8].x=1.1
            else: hand[17].x,hand[17].y=hand[5].x+.001,hand[5].y
            result=provider([hand]).process(frame())['hands'][0]
            self.assertIsNone(result['geometry']['thumb_index_distance_over_palm_width'])
            if mode=='offimage': self.assertIsNone(result['index_tip_xy'])
        self.assertEqual(provider([]).process(frame())['hands'],[])
        hand=points()
        for point in hand: point.x=2.
        self.assertIsNone(provider([hand]).process(frame())['hands'][0]['bbox'])

    def test_handedness_association_and_scores(self):
        p=provider([points()])
        for categories in ([SimpleNamespace(category_name='Other',score=.5)],
                           [SimpleNamespace(category_name='Left',score=float('nan'))],
                           [SimpleNamespace(category_name='Left',score=.5)]*2):
            p.detector.detect=lambda image:SimpleNamespace(hand_landmarks=[points()],handedness=[categories])
            with self.assertRaises(ValueError): p.process(frame())
        p.detector.detect=lambda image:SimpleNamespace(hand_landmarks=[points()],handedness=[])
        with self.assertRaises(ValueError): p.process(frame())
        p.detector.detect=lambda image:SimpleNamespace(hand_landmarks=[points()],handedness=[[]])
        self.assertEqual(p.process(frame())['hands'][0]['handedness'],[])

    def test_input_native_output_limits_and_close(self):
        p=provider([])
        for data in (np.zeros((10,10)),np.zeros((10,10,3)),np.zeros((1081,1920,3),dtype=np.uint8)):
            with self.assertRaises(ValueError): p.process(InputFrame(data,'video',1))
        for hands in ([points()]*5,[points()[:2]]):
            with self.assertRaises(ValueError): provider(hands).process(frame())
        for bad in (float('nan'),float('inf'),101):
            hand=points(); hand[0].x=bad
            with self.assertRaises(ValueError): provider([hand]).process(frame())
        p.close(); p.close()
        with self.assertRaises(RuntimeError): p.process(frame())


@unittest.skipIf(np is None,'numpy not installed')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_hand_context_through_managed_lane_and_mcp(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p=provider([points()]); streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('camera','sensor',[p.spec.id])
            await streams.submit(frame()); await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            self.assertEqual(result['returned_count'],1)
            self.assertEqual(result['items'][0]['observation']['value']['summary']['count'],1)
            self.assertEqual(result['items'][0]['observation']['sensor']['capture_sequence'],0)
            await streams.clear(); self.assertTrue(p.detector.closed)
