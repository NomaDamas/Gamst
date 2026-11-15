"""Synthetic camera projection and native face boundary; no human pose claims."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from gamst import Gamst, ProcessProvider, SensorMetadata, SensorStreams
from gamst.plugins import InputFrame
from gamst.providers.head_orientation import HeadOrientationContext
from gamst.providers.landmarker_video import MediaPipeVideoContext
from gamst.schema import timestamp
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


def config():
    # Explicit synthetic cuboid correspondences, not a human anatomical model.
    points=[[-.03,-.04,-.01],[.03,-.04,-.01],[-.03,.04,-.01],[.03,.04,-.01],
            [-.02,-.03,.02],[.02,-.03,.02],[-.02,.03,.02],[.02,.03,.02]]
    return dict(path='/explicit/face.task',sha256='0'*64,
        camera=dict(width=640,height=480,fx=700,fy=710,cx=320,cy=240,distortion=[.01,-.02,0,0,0],mirrored=False),
        sensor=dict(source_id='cam',sensor_id='sensor',calibration_id='camera1',coordinate_frame='camera.optical'),
        model_frame='synthetic.model',model_points=[dict(landmark_index=i,xyz_m=p) for i,p in enumerate(points)],
        max_reprojection_error_px=1,max_faces=1,max_gap_ms=100)


def frame(sequence=0,stamp=1000):
    return InputFrame(np.zeros((480,640,3),np.uint8),'video',stamp,
        sensor=SensorMetadata('cam','sensor',sequence,calibration_id='camera1',coordinate_frame='camera.optical'))


def native_result(c=None, rvec=None):
    c=c or config(); rvec=np.array([.12,-.18,.07] if rvec is None else rvec,dtype=np.float64)
    camera=c['camera']; k=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]],np.float64)
    xyz=np.array([row['xyz_m'] for row in c['model_points']],np.float64)
    pixels,_=cv2.projectPoints(xyz,rvec,np.array([.01,.02,.6]),k,np.array(camera['distortion']))
    landmarks=[NS(x=.5,y=.5,z=0.) for _ in range(478)]
    for row,pixel in zip(c['model_points'],pixels.reshape(-1,2)):
        landmarks[row['landmark_index']]=NS(x=float(pixel[0]/camera['width']),y=float(pixel[1]/camera['height']),z=0.)
    return NS(face_landmarks=[landmarks],face_blendshapes=[[]])


class Detector:
    def __init__(self): self.calls=[]; self.closed=0; self.result=native_result()
    def detect_for_video(self,image,stamp): self.calls.append(stamp); return self.result
    def close(self): self.closed+=1


def provider():
    made=[]
    def factory():
        detector=Detector(); made.append(detector); return detector
    api=NS(Image=lambda **kw:kw['data'],ImageFormat=NS(SRGB=1)); c=config()
    video=MediaPipeVideoContext('face',factory,api,sha256=c['sha256'],max_entities=1,max_gap_ms=100)
    return HeadOrientationContext(video,**c),made


def native_modules():
    class Base:
        Delegate=NS(CPU=0)
        def __init__(self,**kw): self.__dict__.update(kw)
    vision=NS(FaceLandmarker=NS(create_from_options=lambda options:Detector()),
        FaceLandmarkerOptions=lambda **kw:NS(**kw),RunningMode=NS(VIDEO='VIDEO'))
    python=NS(BaseOptions=Base,vision=vision)
    mp=NS(tasks=NS(python=python),Image=lambda **kw:kw['data'],ImageFormat=NS(SRGB=1))
    return {'mediapipe':mp,'mediapipe.tasks':mp.tasks,'mediapipe.tasks.python':python}


class Configuration(unittest.TestCase):
    def test_pure_discovery_hash_binding_and_invalid_geometry(self):
        code='from gamst import ProcessProvider; import sys; ProcessProvider.capability("mediapipe.head_orientation.video",'+repr(config())+'); assert not {"cv2","numpy","mediapipe"}&sys.modules.keys()'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        spec=ProcessProvider.capability('mediapipe.head_orientation.video',config())
        changed=config(); changed['camera']['fx']+=1
        self.assertNotEqual(spec.version,ProcessProvider.capability('mediapipe.head_orientation.video',changed).version)
        invalid=[]
        c=config(); c['camera']['fx']=float('nan'); invalid.append(c)
        c=config(); c['camera']['distortion']=[]; invalid.append(c)
        c=config(); c['camera']['mirrored']=True; invalid.append(c)
        c=config(); del c['camera']['mirrored']; invalid.append(c)
        c=config(); c['model_points'][0]['landmark_index']=True; invalid.append(c)
        c=config(); c['model_points'][1]['landmark_index']=0; invalid.append(c)
        c=config(); c['model_points'][1]['xyz_m']=c['model_points'][0]['xyz_m']; invalid.append(c)
        c=config()
        for p in c['model_points']: p['xyz_m'][2]=0
        invalid.append(c)
        c=config(); c['sensor']['calibration_id']=''; invalid.append(c)
        c=config(); c['model_frame']=c['sensor']['coordinate_frame']; invalid.append(c)
        c=config(); c['extra']=1; invalid.append(c)
        for c in invalid:
            with self.assertRaises((ValueError,TypeError)): ProcessProvider.capability('mediapipe.head_orientation.video',c)


@unittest.skipIf(cv2 is None,'optional OpenCV/numpy unavailable')
class Orientation(unittest.TestCase):
    def test_actual_pnp_recovers_declared_model_to_camera_rotation(self):
        p,made=provider(); self.addCleanup(p.close)
        value=p.process(frame()); pose=value['faces'][0]['orientation']
        expected,_=cv2.Rodrigues(np.array([.12,-.18,.07]))
        self.assertEqual(pose['status'],'estimated')
        np.testing.assert_allclose(pose['rotation_model_to_camera'],expected,atol=1e-6)
        np.testing.assert_allclose(pose['rodrigues_model_to_camera_rad'],[.12,-.18,.07],atol=1e-6)
        self.assertLess(pose['reprojection_rmse_px'],1e-5)
        self.assertFalse(value['summary']['calibration_verified'])
        self.assertFalse(value['summary']['pose_accuracy_verified'])
        self.assertFalse(value['summary']['translation_published'])
        self.assertEqual(value['summary']['mirror_status'],'caller_declared_unmirrored')
        self.assertNotIn('translation_m',pose)
        self.assertIsNone(p.result)
        second=p.process(frame(1,1030)); self.assertEqual(made[0].calls,[0,30])
        third=p.process(frame(3,1060)); self.assertEqual(made[0].closed,1)
        self.assertNotEqual(second['summary']['state_segment'],third['summary']['state_segment'])

    def test_reprojection_degenerate_outside_and_negative_depth_abstain(self):
        p,made=provider(); self.addCleanup(p.close); p.process(frame())
        for label in ('outside','degenerate','inconsistent'):
            raw=native_result()
            if label=='outside': raw.face_landmarks[0][0].x=-.1
            elif label=='degenerate':
                for point in raw.face_landmarks[0]: point.x=point.y=.5
            else: raw.face_landmarks[0][0].x+=.1
            made[-1].result=raw
            value=p.process(frame(len(made[-1].calls),1000+10*len(made[-1].calls)))
            self.assertEqual(value['faces'][0]['orientation']['status'],'unavailable')
        made[-1].result=native_result()
        with patch.object(cv2,'solvePnP',return_value=(True,np.zeros((3,1)),np.array([[0.],[0.],[-1.]]))):
            self.assertEqual(p.process(frame(10,1200))['faces'][0]['orientation']['reason'],'invalid_pose_or_cheirality')
        made[-1].result=NS(face_landmarks=[],face_blendshapes=[])
        self.assertEqual(p.process(frame(11,1230))['faces'],[])

    def test_calibration_geometry_bad_clock_and_close(self):
        p,made=provider(); p.process(frame())
        for bad in (replace(frame(1,1030),sensor=replace(frame().sensor,calibration_id='wrong')),
                    replace(frame(1,1030),data=np.zeros((240,320,3),np.uint8)),frame(0,1000)):
            with self.assertRaises(ValueError): p.process(bad)
            self.assertIsNone(p.video.detector)
            p.process(frame())
        p.close(); p.close(); self.assertTrue(all(d.closed==1 for d in made))
        with self.assertRaises(RuntimeError): p.process(frame())


@unittest.skipIf(cv2 is None,'optional OpenCV/numpy unavailable')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_real_worker_native_face_boundary_to_mcp(self):
        code='''
import sys,importlib.metadata
from test_head_orientation import native_modules
sys.modules.update(native_modules())
importlib.metadata.version=lambda name:'0.10.35'
from gamst.provider_worker import main
main()
'''
        code = 'import sys; sys.path.insert(0, '+repr(str(Path(__file__).resolve().parent))+')\n'+code
        real=asyncio.create_subprocess_exec; children=[]
        async def spawn(*args,**kwargs):
            child=await real(args[0],'-u','-c',code,**kwargs); children.append(child); return child
        with tempfile.TemporaryDirectory(dir='.') as temporary:
            c=config(); path=Path(temporary).resolve()/'face.task'; payload=b'not-a-model'; path.write_bytes(payload)
            c.update(path=str(path),sha256=hashlib.sha256(payload).hexdigest())
            with patch('asyncio.create_subprocess_exec',spawn):
                async with Gamst(consent=True) as human:
                    spec=ProcessProvider.capability('mediapipe.head_orientation.video',c)
                    async def factory(): return await ProcessProvider.open(sys.executable,'mediapipe.head_orientation.video',c)
                    streams=SensorStreams(human); streams.register(spec,factory)
                    await streams.open('cam','sensor',[spec.id]); await streams.submit(frame(0,timestamp()-10)); await streams.drain()
                    from gamst.mcp import MCPServer
                    answer=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                    self.assertEqual(answer['returned_count'],1)
                    self.assertEqual(answer['items'][0]['observation']['value']['faces'][0]['orientation']['status'],'estimated')
                    await streams.clear()
            self.assertTrue(children and all(child.returncode is not None for child in children))


if __name__=='__main__': unittest.main()
