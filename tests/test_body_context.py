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
from humansense.providers.body_context import MediaPipeBodyContext
from humansense.schema import timestamp
from test_pose import points
try:
    import numpy as np
except ImportError:
    np=None


class Detector:
    def __init__(self,bodies): self.bodies=bodies; self.images=[]; self.closed=False
    def detect(self,image): self.images.append(image.copy()); return SimpleNamespace(pose_landmarks=self.bodies)
    def close(self): self.closed=True


def provider(bodies):
    detector=Detector(bodies)
    api=SimpleNamespace(Image=lambda **kw:kw['data'],ImageFormat=SimpleNamespace(SRGB=1))
    return MediaPipeBodyContext(detector,api,sha256='0'*64)


def frame():
    return InputFrame(np.zeros((100,200,3),dtype=np.uint8),'video',timestamp()-100,
                      sensor=SensorMetadata('camera','sensor',0))


class Loader(unittest.TestCase):
    def test_metadata_validation_and_preimport_integrity(self):
        spec=capability('mediapipe.body',dict(path='/not-installed',sha256='0'*64))
        self.assertEqual(spec.streaming_mode,'frame'); self.assertFalse(spec.stateful)
        self.assertNotEqual(spec.version,capability('mediapipe.body',dict(path='x',sha256='0'*64,max_poses=1)).version)
        for args in ({},dict(path='x',sha256='bad'),dict(path='x',sha256='0'*64,max_poses=True),
                     dict(path='x',sha256='0'*64,max_poses=5),dict(path='x',sha256='0'*64,extra=1)):
            with self.assertRaises(ValueError): capability('mediapipe.body',args)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; path.write_bytes(b'fixture')
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): MediaPipeBodyContext.from_local(str(path),'0'*64)
                version.assert_not_called()
            digest=hashlib.sha256(b'fixture').hexdigest()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): MediaPipeBodyContext.from_local(str(path),digest)

    def test_loader_passes_exact_verified_buffer_and_image_mode(self):
        detector=Detector([]); captured={}
        def options(**kwargs): captured.update(kwargs); return kwargs
        class Base:
            Delegate=SimpleNamespace(CPU=0)
            def __init__(self,**kwargs): self.__dict__.update(kwargs)
        vision=SimpleNamespace(PoseLandmarker=SimpleNamespace(create_from_options=lambda o:detector),
                               PoseLandmarkerOptions=options,RunningMode=SimpleNamespace(IMAGE='IMAGE'))
        python=SimpleNamespace(BaseOptions=Base,vision=vision)
        mp=SimpleNamespace(tasks=SimpleNamespace(python=python))
        modules={'mediapipe':mp,'mediapipe.tasks':mp.tasks,'mediapipe.tasks.python':python}
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.task'; payload=b'fixture-not-a-model'; path.write_bytes(payload)
            with patch('importlib.metadata.version',return_value='0.10.35'),patch.dict('sys.modules',modules):
                p=MediaPipeBodyContext.from_local(str(path),hashlib.sha256(payload).hexdigest(),max_poses=2)
            self.assertEqual(captured['base_options'].model_asset_buffer,payload)
            self.assertEqual(captured['running_mode'],'IMAGE')
            self.assertEqual(captured['num_poses'],2); self.assertFalse(captured['output_segmentation_masks'])
            p.close(); self.assertTrue(detector.closed)


@unittest.skipIf(np is None,'numpy not installed')
class Geometry(unittest.TestCase):
    def test_projected_angle_uses_pixels_and_context_is_bounded(self):
        body=points()
        body[11].x,body[11].y=.25,.5
        body[13].x,body[13].y=.5,.5
        body[15].x,body[15].y=.75,1.
        p=provider([body,points(.1)])
        f=frame(); f.data[:,:,0]=17; f.data[:,:,2]=42
        result=p.process(f)
        self.assertAlmostEqual(result['poses'][0]['geometry']['left_elbow_projected_angle_deg'],135.,places=3)
        self.assertEqual(p.detector.images[0][0,0].tolist(),[42,0,17])
        self.assertEqual(f.data[0,0].tolist(),[17,0,42])
        self.assertLess(len(json.dumps(result)),8192)
        self.assertNotIn('landmarks',json.dumps(result))
        self.assertFalse(result['summary']['metric_3d'])
        self.assertFalse(result['summary']['temporal_tracking'])

    def test_low_visibility_offimage_degenerate_and_empty_abstain(self):
        for mode in ('low','offimage','degenerate'):
            body=points()
            if mode=='low': body[13].visibility=.1
            if mode=='offimage': body[13].x=1.1
            if mode=='degenerate': body[11].x,body[11].y=body[13].x,body[13].y
            result=provider([body]).process(frame())
            self.assertIsNone(result['poses'][0]['geometry']['left_elbow_projected_angle_deg'])
        self.assertEqual(provider([]).process(frame())['poses'],[])

    def test_input_native_output_limits_and_close(self):
        p=provider([])
        for data in (np.zeros((10,10)),np.zeros((10,10,3)),np.zeros((1081,1920,3),dtype=np.uint8)):
            with self.assertRaises(ValueError): p.process(InputFrame(data,'video',1))
        for bodies in ([points()]*5,[points()[:2]]):
            with self.assertRaises(ValueError): provider(bodies).process(frame())
        for bad in (float('nan'),float('inf'),101):
            body=points(); body[0].x=bad
            with self.assertRaises(ValueError): provider([body]).process(frame())
        body=points(); body[0].presence=2
        with self.assertRaises(ValueError): provider([body]).process(frame())
        p.close(); p.close()
        with self.assertRaises(RuntimeError): p.process(frame())


@unittest.skipIf(np is None,'numpy not installed')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_body_context_through_managed_lane_and_mcp(self):
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
