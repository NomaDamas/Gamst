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
from humansense.providers.face_context import MediaPipeFaceContext
from humansense.schema import timestamp
def points():
    return [SimpleNamespace(x=.2+(i%20)*.02,y=.2+(i//20)*.02,z=0.) for i in range(478)]
try:
    import numpy as np
except ImportError:
    np=None


class Detector:
    def __init__(self,bodies): self.bodies=bodies; self.images=[]; self.closed=False
    def detect(self,image): self.images.append(image.copy()); return SimpleNamespace(face_landmarks=self.bodies,face_blendshapes=[[SimpleNamespace(category_name='jawOpen',score=.9)] for _ in self.bodies])
    def close(self): self.closed=True


def provider(bodies):
    detector=Detector(bodies)
    api=SimpleNamespace(Image=lambda **kw:kw['data'],ImageFormat=SimpleNamespace(SRGB=1))
    return MediaPipeFaceContext(detector,api,sha256='0'*64)


def frame():
    return InputFrame(np.zeros((100,200,3),dtype=np.uint8),'video',timestamp()-100,
                      sensor=SensorMetadata('camera','sensor',0))


class Loader(unittest.TestCase):
    def test_metadata_validation_and_preimport_integrity(self):
        spec=capability('mediapipe.face',dict(path='/not-installed',sha256='0'*64))
        self.assertEqual(spec.streaming_mode,'frame'); self.assertFalse(spec.stateful)
        self.assertNotEqual(spec.version,capability('mediapipe.face',dict(path='x',sha256='0'*64,max_faces=1)).version)
        for args in ({},dict(path='x',sha256='bad'),dict(path='x',sha256='0'*64,max_faces=True),
                     dict(path='x',sha256='0'*64,max_faces=5),dict(path='x',sha256='0'*64,extra=1)):
            with self.assertRaises(ValueError): capability('mediapipe.face',args)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; path.write_bytes(b'fixture')
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): MediaPipeFaceContext.from_local(str(path),'0'*64)
                version.assert_not_called()
            digest=hashlib.sha256(b'fixture').hexdigest()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): MediaPipeFaceContext.from_local(str(path),digest)

    def test_loader_passes_exact_verified_buffer_and_image_mode(self):
        detector=Detector([]); captured={}
        def options(**kwargs): captured.update(kwargs); return kwargs
        class Base:
            Delegate=SimpleNamespace(CPU=0)
            def __init__(self,**kwargs): self.__dict__.update(kwargs)
        vision=SimpleNamespace(FaceLandmarker=SimpleNamespace(create_from_options=lambda o:detector),
                               FaceLandmarkerOptions=options,RunningMode=SimpleNamespace(IMAGE='IMAGE'))
        python=SimpleNamespace(BaseOptions=Base,vision=vision)
        mp=SimpleNamespace(tasks=SimpleNamespace(python=python))
        modules={'mediapipe':mp,'mediapipe.tasks':mp.tasks,'mediapipe.tasks.python':python}
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; payload=b'fixture-not-a-model'; path.write_bytes(payload)
            with patch('importlib.metadata.version',return_value='0.10.35'),patch.dict('sys.modules',modules):
                p=MediaPipeFaceContext.from_local(str(path),hashlib.sha256(payload).hexdigest(),max_faces=2)
            self.assertEqual(captured['base_options'].model_asset_buffer,payload)
            self.assertEqual(captured['running_mode'],'IMAGE')
            self.assertEqual(captured['num_faces'],2)
            self.assertTrue(captured['output_face_blendshapes'])
            self.assertFalse(captured['output_facial_transformation_matrixes'])
            p.close(); self.assertTrue(detector.closed)


@unittest.skipIf(np is None,'numpy not installed')
class Geometry(unittest.TestCase):
    def test_projected_ratios_rgb_copy_and_missing_confidence(self):
        face=points()
        for i,xy in {13:(.5,.25),14:(.5,.75),61:(.25,.5),291:(.75,.5),
                     386:(.5,.4),374:(.5,.6),263:(.4,.5),362:(.6,.5),
                     159:(.5,.45),145:(.5,.55),33:(.4,.5),133:(.6,.5)}.items():
            face[i].x,face[i].y=xy
        p=provider([face])
        f=frame(); f.data[:,:,0]=17; f.data[:,:,2]=42
        result=p.process(f); geometry=result['faces'][0]['geometry']
        self.assertEqual(geometry['mouth_gap_over_width'],.5)
        self.assertEqual(geometry['left_eye_gap_over_width'],.5)
        self.assertEqual(geometry['right_eye_gap_over_width'],.25)
        larger=InputFrame(np.zeros((200,400,3),dtype=np.uint8),'video',1)
        self.assertEqual(p.process(larger)['faces'][0]['geometry'],geometry)
        self.assertEqual(p.detector.images[0][0,0].tolist(),[42,0,17])
        self.assertEqual(f.data[0,0].tolist(),[17,0,42])
        self.assertLess(len(json.dumps(result)),8192)
        self.assertNotIn('landmarks',json.dumps(result))
        self.assertEqual(result['faces'][0]['joint_confidence'],'unavailable')
        self.assertFalse(result['summary']['metric_3d'])
        self.assertFalse(result['summary']['temporal_tracking'])
        self.assertEqual(result['summary']['emotion'],'unavailable')
        self.assertEqual(result['summary']['gaze'],'unavailable')

    def test_offimage_and_subpixel_abstain(self):
        for mode in ('offimage','subpixel'):
            face=points()
            if mode=='offimage': face[13].x=1.1
            else: face[291].x,face[291].y=face[61].x+.001,face[61].y
            result=provider([face]).process(frame())['faces'][0]
            self.assertIsNone(result['geometry']['mouth_gap_over_width'])
        self.assertEqual(provider([]).process(frame())['faces'],[])
        face=points()
        for point in face: point.x=2.
        self.assertIsNone(provider([face]).process(frame())['faces'][0]['bbox'])

    def test_blendshape_association_scores_and_semantics(self):
        from humansense.providers.face_context import BLENDSHAPES
        self.assertEqual(len(BLENDSHAPES),52)
        p=provider([points()])
        for categories in ([SimpleNamespace(category_name='happy',score=.5)],
                           [SimpleNamespace(category_name='jawOpen',score=float('nan'))],
                           [SimpleNamespace(category_name='jawOpen',score=.5)]*2,
                           [SimpleNamespace(category_name='jawOpen',score=-.1)],
                           [SimpleNamespace(category_name='jawOpen',score=True)],
                           [SimpleNamespace(category_name='jawOpen',score=.5)]*53):
            p.detector.detect=lambda image:SimpleNamespace(face_landmarks=[points()],face_blendshapes=[categories])
            with self.assertRaises(ValueError): p.process(frame())
        p.detector.detect=lambda image:SimpleNamespace(face_landmarks=[points()],face_blendshapes=[])
        with self.assertRaises(ValueError): p.process(frame())
        p.detector.detect=lambda image:SimpleNamespace(face_landmarks=[points()],face_blendshapes=[[]])
        self.assertFalse(p.process(frame())['faces'][0]['blendshapes_available'])
        coefficients=[SimpleNamespace(category_name=name,score=.5) for name in BLENDSHAPES]
        p.detector.detect=lambda image:SimpleNamespace(face_landmarks=[points()]*4,face_blendshapes=[coefficients]*4)
        result=p.process(frame())
        self.assertEqual(len(result['faces']),4)
        self.assertEqual(len(result['faces'][0]['blendshapes']),52)
        self.assertLess(len(json.dumps(result)),16384)

    def test_input_native_output_limits_and_close(self):
        p=provider([])
        for data in (np.zeros((10,10)),np.zeros((10,10,3)),np.zeros((1081,1920,3),dtype=np.uint8)):
            with self.assertRaises(ValueError): p.process(InputFrame(data,'video',1))
        for faces in ([points()]*5,[points()[:2]]):
            with self.assertRaises(ValueError): provider(faces).process(frame())
        for bad in (float('nan'),float('inf'),101):
            face=points(); face[0].x=bad
            with self.assertRaises(ValueError): provider([face]).process(frame())
        p.close(); p.close()
        with self.assertRaises(RuntimeError): p.process(frame())


@unittest.skipIf(np is None,'numpy not installed')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_face_context_through_managed_lane_and_mcp(self):
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
