import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from humansense import HumanSense, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.provider_profiles import capability
from humansense.providers.gaze import L2CSGaze
from humansense.schema import timestamp

try:
    import numpy as np
except ImportError:
    np=None


class Face:
    def __init__(self, boxes): self.boxes=boxes; self.closed=False
    def process(self, frame): return {'faces':[dict(bbox=b) for b in self.boxes]}
    def close(self): self.closed=True


BOX=dict(x=.2,y=.2,width=.6,height=.6)

def logits(count, index):
    result=np.full((count,90),-1000.)
    result[:,index]=0
    return result


def provider(boxes=None, predictor=None):
    return L2CSGaze(Face([BOX] if boxes is None else boxes),
        predictor or (lambda crops:(logits(len(crops),45),logits(len(crops),45))),
        sha256='0'*64,face_sha256='1'*64)


def frame():
    data=np.zeros((100,200,3),dtype=np.uint8)
    data[:,:,0]=10; data[:,:,2]=90
    return InputFrame(data,'video',timestamp()-100,sensor=SensorMetadata('camera','sensor',0))


class Metadata(unittest.TestCase):
    def test_profiles_and_preimport_artifact_checks(self):
        config=dict(path='/not-installed',sha256='0'*64,face_path='/not-installed',face_sha256='1'*64)
        spec=capability('l2cs.gaze',config)
        self.assertEqual(spec.streaming_mode,'frame'); self.assertFalse(spec.stateful)
        for update in (dict(arch='ResNet18'),dict(max_faces=1),dict(face_sha256='2'*64),dict(sha256='3'*64)):
            self.assertNotEqual(spec.version,capability('l2cs.gaze',dict(config,**update)).version)
        for update in (dict(arch='ResNet101'),dict(max_faces=True),dict(extra=1),dict(sha256='x'),dict(face_path=1)):
            with self.assertRaises(ValueError): capability('l2cs.gaze',dict(config,**update))
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'fixture.bin'; path.write_bytes(b'not-a-model')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): L2CSGaze.from_local(str(path),digest,str(path),'1'*64)
                version.assert_not_called()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): L2CSGaze.from_local(str(path),digest,str(path),digest)


@unittest.skipIf(np is None,'optional numpy not installed')
class Gaze(unittest.TestCase):
    def test_rgb_copy_head_order_and_direction(self):
        crops=[]
        def predict(batch):
            crops.extend(batch)
            return logits(len(batch),60),logits(len(batch),45)
        p=provider(predictor=predict); f=frame(); result=p.process(f)
        face=result['faces'][0]
        self.assertEqual(face['yaw_deg'],60); self.assertEqual(face['pitch_deg'],0)
        self.assertAlmostEqual(face['direction']['x'],-3**.5/2)
        self.assertAlmostEqual(face['direction']['z'],-.5)
        self.assertEqual(crops[0].shape,(60,120,3))
        self.assertEqual(crops[0][0,0].tolist(),[90,0,10])
        crops[0][:]=0; self.assertEqual(f.data[20,40].tolist(),[10,0,90])
        self.assertEqual(face['confidence'],'unavailable')
        self.assertEqual(result['summary']['gaze_target'],'unavailable')
        self.assertFalse(result['summary']['temporal_tracking'])
        p=provider(predictor=lambda c:(logits(len(c),45),logits(len(c),60)))
        self.assertAlmostEqual(p.process(frame())['faces'][0]['direction']['y'],-3**.5/2)

    def test_softmax_shift_invariance_and_extreme_bins(self):
        self.assertEqual(L2CSGaze._angles(np.zeros((1,90)),1),[-2.])
        for index,angle in ((0,-180),(89,176)):
            self.assertEqual(L2CSGaze._angles(logits(1,index),1),[float(angle)])
        a=np.arange(90,dtype=float).reshape(1,90)
        self.assertEqual(L2CSGaze._angles(a,1),L2CSGaze._angles(a+10000,1))

    def test_empty_small_missing_faces_never_invoke_model_and_association(self):
        def forbidden(crops): raise AssertionError('no usable crop')
        for boxes in ([],[None],[dict(x=0,y=0,width=.01,height=.01)]):
            result=provider(boxes,forbidden).process(frame())
            self.assertEqual(result['summary']['estimated_count'],0)
        result=provider([None,BOX]).process(frame())
        self.assertEqual([f['status'] for f in result['faces']],['unavailable_crop','estimated'])
        self.assertEqual(result['faces'][1]['frame_face_index'],1)
        self.assertIsNone(result['faces'][0]['yaw_deg'])

    def test_native_output_input_and_crop_failures(self):
        for output in (None,(np.zeros((1,90)),), (np.zeros((2,90)),np.zeros((2,90))),
                       (np.full((1,90),np.nan),np.zeros((1,90))),
                       (np.zeros((1,90),dtype=bool),np.zeros((1,90)))):
            with self.assertRaises(ValueError): provider(predictor=lambda c:output).process(frame())
        for box in ({},dict(BOX,x=.9),dict(BOX,x=float('nan')),dict(BOX,width=True)):
            with self.assertRaises(ValueError): provider([box]).process(frame())
        with self.assertRaises(ValueError): provider([BOX]*5).process(frame())
        for data in (np.zeros((10,10)),np.zeros((10,10,3)),np.zeros((1081,1920,3),dtype=np.uint8)):
            with self.assertRaises(ValueError): provider().process(InputFrame(data,'video',1))
        p=provider(); p.close(); p.close(); self.assertTrue(p.face_provider.closed)
        with self.assertRaises(RuntimeError): p.process(frame())

    def test_local_loader_real_torchvision_with_synthetic_state(self):
        # Executes native preprocessing/backbone/heads, but no trained model.
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ('torch','torchvision','cv2')):
            self.skipTest('optional native gaze runtime not installed')
        import torch, torchvision
        reference=torchvision.models.resnet18(weights=None)
        reference.fc=torch.nn.Identity()
        for name in ('fc_yaw_gaze','fc_pitch_gaze'):
            layer=torch.nn.Linear(512,90)
            torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)
            setattr(reference,name,layer)
        reference.fc_finetune=torch.nn.Linear(515,3)
        state=reference.state_dict()
        captured={}
        def load(source,**kwargs):
            captured.update(kwargs); captured['payload']=source.read()
            return state
        versions={'torch':'2.8.0','torchvision':'0.23.0','mediapipe':'0.10.35'}
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'fixture.bin'; path.write_bytes(b'fixture-not-weights')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('importlib.metadata.version',side_effect=versions.__getitem__), \
                 patch('torch.load',load), \
                 patch('humansense.providers.gaze.MediaPipeFaceContext.from_local',return_value=Face([BOX])):
                p=L2CSGaze.from_local(str(path),digest,str(path),digest,arch='ResNet18')
            try:
                result=p.process(frame())
                self.assertEqual(result['faces'][0]['yaw_deg'],-2.)
                self.assertEqual(result['faces'][0]['pitch_deg'],-2.)
            finally: p.close()
        self.assertEqual(captured,dict(map_location='cpu',weights_only=True,payload=b'fixture-not-weights'))

    def test_close_waits_for_native_work_and_releases_predictor(self):
        import threading
        started,release,closed=threading.Event(),threading.Event(),threading.Event()
        def predict(crops):
            started.set()
            if not release.wait(5): raise RuntimeError('test gate timed out')
            return logits(len(crops),45),logits(len(crops),45)
        p=provider(predictor=predict)
        worker=threading.Thread(target=p.process,args=(frame(),))
        closer=threading.Thread(target=lambda:(p.close(),closed.set()))
        worker.start()
        try:
            self.assertTrue(started.wait(5)); closer.start()
            self.assertFalse(closed.wait(.05))
        finally:
            release.set(); worker.join(5)
            if closer.ident is not None: closer.join(5)
        self.assertTrue(closed.is_set()); self.assertIsNone(p.predictor)


@unittest.skipIf(np is None,'optional numpy not installed')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_managed_observation_and_mcp(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p=provider(); streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('camera','sensor',[p.spec.id])
            await streams.submit(frame()); await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            item=result['items'][0]['observation']
            self.assertEqual(item['value']['faces'][0]['yaw_deg'],0)
            self.assertEqual(item['sensor']['capture_sequence'],0)
            await streams.clear(); self.assertTrue(p.closed)
