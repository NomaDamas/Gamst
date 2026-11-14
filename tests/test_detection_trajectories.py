import asyncio
from dataclasses import replace
import hashlib
import json
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorStreams
from humansense.mcp import MCPServer
from humansense.providers.detection_trajectories import RFDETRTrajectories
from humansense.schema import timestamp
import test_detection_provider as detection

np = detection.np
CONFIG = dict(path='/explicit/local/checkpoint.pth',sha256='a'*64,camera_motion='stationary')


def frame(seq=0,ts=None):
    f=detection.frame(1000+seq*100 if ts is None else ts,seq)
    return replace(f,sensor=replace(f.sensor,stream_epoch='e1'))


class Discovery(unittest.TestCase):
    def test_discovery_no_optional_import_and_strict_config(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("rfdetr.trajectories",'+repr(CONFIG)+');assert not {"numpy","torch","rfdetr"}&sys.modules.keys()'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for change in ({'max_objects':9},{'max_samples':True},{'device':'bad'},{'camera_motion':'auto'},{'sha256':'bad'}):
            with self.assertRaises(ValueError):ProcessProvider.capability('rfdetr.trajectories',dict(CONFIG,**change))
        self.assertNotEqual(ProcessProvider.capability('rfdetr.trajectories',CONFIG).version,
                            ProcessProvider.capability('rfdetr.trajectories',dict(CONFIG,camera_motion='unknown')).version)


@unittest.skipIf(np is None,'optional numpy absent')
class Pipeline(unittest.TestCase):
    def test_pixels_boxes_velocity_and_digest(self):
        b=detection.Backend();p=RFDETRTrajectories(b,version='fixture',camera_motion='stationary')
        f=frame();digest=hashlib.sha256(b'gamst.rfdetr.trajectory.uint8.HWC.BGR.v1\0'+struct.pack('>III',*f.data.shape)+f.data.tobytes()).hexdigest()
        first=p.process(f);b.rows=((20,10,50,40,.9,1,'cup'),)
        second=p.process(frame(1))
        obj=second['objects'][0]
        self.assertAlmostEqual(obj['velocity_image_s'][0],1)
        self.assertEqual(obj['observations'][0]['evidence_sha256'],digest)
        self.assertIsNone(obj['observations'][1]['association_confidence'])
        self.assertEqual(second['detections'][0]['attributes']['score'],.9)
        self.assertEqual(second['detections'][0]['attributes']['association'],'matched')
        self.assertEqual(obj['segment_id'],first['objects'][0]['segment_id'])
        self.assertEqual(b.received[0,0].tolist(),[30,20,10])

    def test_truncation_disappearance_and_ambiguity_abstain(self):
        b=detection.Backend();p=RFDETRTrajectories(b,version='fixture',camera_motion='stationary')
        old=p.process(frame())['objects'][0]['segment_id']
        b.rows=tuple((10,10,40,40,.9,i,'item') for i in range(9))
        result=p.process(frame(1));self.assertEqual(result['objects'],[])
        self.assertEqual(len(result['detections']),8)
        self.assertFalse(result['geometry']['complete_snapshot'])
        self.assertEqual(result['geometry']['trajectory_status'],'incomplete_detection_snapshot')
        b.rows=detection.Backend.rows
        self.assertIsNone(p.process(frame(2))['objects'][0]['velocity_image_s'])
        b.rows=();self.assertEqual(p.process(frame(3))['objects'],[])
        b.rows=detection.Backend.rows
        self.assertNotEqual(old,p.process(frame(4))['objects'][0]['segment_id'])
        p.reset();b.rows=((10,10,50,50,.9,1,'cup'),(30,10,70,50,.9,1,'cup'))
        p.process(frame());b.rows=((20,10,60,50,.9,1,'cup'),)
        result=p.process(frame(1));obj=result['objects'][0]
        self.assertEqual(result['detections'][0]['attributes']['association'],'uncertain_new')
        self.assertIsNone(obj['velocity_image_s'])

    def test_maximum_escaped_labels_and_full_history_stay_bounded(self):
        b=detection.Backend();p=RFDETRTrajectories(b,version='v'*128,camera_motion='stationary')
        b.rows=tuple((10.123+i,10.456,100.789+i,100.123,.987654321,i,chr(0x10000+i)*256) for i in range(8))
        for i in range(8):
            f=frame(i,2**53-8+i)
            f=replace(f,data=np.zeros((511,1023,3),np.uint8),sensor=replace(f.sensor,
                source_id='s'*128,sensor_id='i'*128,coordinate_frame='f'*128,
                calibration_id='c'*128,stream_epoch='e'*128,channels=tuple('channel'+str(n) for n in range(64))))
            result=p.process(f)
            self.assertLessEqual(len(json.dumps(result,ensure_ascii=True).encode()),65536)
        self.assertEqual([len(x['observations']) for x in result['objects']],[8]*8)
        self.assertEqual(len(result['detections'][0]['label']),256)
        self.assertNotIn('label',result['objects'][0])

    def test_unknown_motion_gap_owner_failure_and_copied_input(self):
        b=detection.Backend();p=RFDETRTrajectories(b,version='fixture')
        p.process(frame());self.assertIsNone(p.process(frame(1))['objects'][0]['velocity_image_s'])
        p=RFDETRTrajectories(b,version='fixture',camera_motion='stationary')
        for sensor_change in ({'stream_epoch':'e2'},{'calibration_id':'cal2'},{'channels':('x',)},{'source_id':'other'}):
            p.reset();p.process(frame());f=frame(1)
            self.assertIsNone(p.process(replace(f,sensor=replace(f.sensor,**sensor_change)))['objects'][0]['velocity_image_s'])
        p.reset();p.process(frame());self.assertIsNone(p.process(frame(2))['objects'][0]['velocity_image_s'])
        with self.assertRaises(ValueError):p.process(frame(2))
        self.assertIsNone(p.process(frame(3))['objects'][0]['velocity_image_s'])
        original=frame(4);expected=original.data.copy();predict=b.predict
        def mutate(data,**kwargs):
            original.data[:]=0
            return predict(data,**kwargs)
        b.predict=mutate;result=p.process(original)
        self.assertEqual(b.received[0,0].tolist(),[30,20,10])
        self.assertNotEqual(result['geometry']['input_sha256'],hashlib.sha256(original.data.tobytes()).hexdigest())
        self.assertTrue((expected[0,0]==[10,20,30]).all())
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame(5))


WORKER = '''
import numpy as np
from types import SimpleNamespace
from humansense.providers.detection import RFDETRObjects
import humansense.provider_worker as w
class Backend:
    def __init__(self): self.index=0
    def predict(self,data,**kwargs):
        x=10+self.index; self.index+=1
        return SimpleNamespace(xyxy=np.array([[x,10,x+30,40]]),confidence=np.array([.9]),class_id=np.array([1]),data={'class_name':['cup']})
def local(cls,path,sha256,device,**kwargs):
    return cls(Backend(),version='1.12.1.eager.sha256.'+sha256,**kwargs)
RFDETRObjects.from_local=classmethod(local)
w.main()
'''


@unittest.skipIf(np is None,'optional numpy absent')
class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_owned_worker_stream_context_mcp_and_close(self):
        original_spawn=asyncio.create_subprocess_exec;children=[]
        async def spawn(*args,**kwargs):
            child=await original_spawn(sys.executable,'-u','-c',WORKER,**kwargs)
            children.append(child);return child
        with patch('asyncio.create_subprocess_exec',spawn):
            worker=await ProcessProvider.open(sys.executable,'rfdetr.trajectories',CONFIG)
            try:
                async with HumanSense(consent=True) as human:
                    streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                    await streams.open('cam','sensor',[worker.spec.id]);now=timestamp()-200
                    for i in range(2):
                        f=frame(i,now+i*100);f=replace(f,sensor=replace(f.sensor,stream_epoch=None))
                        await streams.submit(f);await streams.drain()
                    value=next(iter((await human.get_context())['observations'].values()))['value']
                    self.assertEqual(len(value['objects'][0]['observations']),2)
                    self.assertAlmostEqual(value['objects'][0]['velocity_image_s'][0],.1)
                    self.assertEqual(value['geometry']['timestamp_ms'],now+100)
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    self.assertIn('evidence_sha256',json.dumps(reply))
                    await human.pause()
                    self.assertTrue(worker.closed)
                self.assertIsNotNone(children[0].returncode)
            finally:await worker.close()
