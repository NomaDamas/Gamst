import asyncio
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorStreams
from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.mcp import MCPServer
from humansense.schema import timestamp
from humansense.stereo_packet import stereo_frame, unpack_stereo
from humansense.providers.stereo_context import StereoMetricContext
from humansense.providers.detection_trajectories import RFDETRTrajectories
from humansense.stereo_replay import recording, replay
from test_stereo_metric import calibration, np, cv2
from test_detection_provider import Backend


CONFIG = dict(path='/explicit/fixture/weights',sha256='a'*64,calibration=calibration(),
    left_sensor=dict(source_id='rig',sensor_id='left',coordinate_frame='left.optical'),
    right_sensor=dict(source_id='rig',sensor_id='right',coordinate_frame='right.optical'),
    clock_id='capture.clock',alignment_id='alignment1',rectified_frame='left.rectified',
    max_skew_ms=2,camera_motion='stationary',disparities=32)


def packet(seq=0, ts=1000, *, disparity=8, right_ts=None):
    gray = np.random.default_rng(49).integers(0,256,(160,320),dtype=np.uint8)
    left = np.repeat(gray[:,:,None],3,axis=2)
    right = np.zeros_like(left); right[:,:-disparity] = left[:,disparity:]
    frames = []
    for name,data,time in (('left',left,ts),('right',right,ts if right_ts is None else right_ts)):
        frames.append(InputFrame(data,'video',time,sensor=SensorMetadata('rig',name,seq,
            coordinate_frame=name+'.optical',calibration_id='fixture',stream_epoch='capture1',capture_sequence=seq)))
    return stereo_frame(*frames,left_interval_ms=[ts,ts],right_interval_ms=[frames[1].timestamp_ms]*2,
                        clock_id='capture.clock',alignment_id='alignment1')


class Discovery(unittest.TestCase):
    def test_no_optional_imports(self):
        code = ('from humansense import ProcessProvider; import sys; '
                'ProcessProvider.capability("opencv.rfdetr.stereo",'+repr(CONFIG)+'); '
                'assert not {"cv2","numpy","torch","rfdetr"}&sys.modules.keys()')
        got = subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(got.returncode,0,got.stderr)
        for changes in ({'max_skew_ms':True},{'camera_motion':'moving'},{'disparities':15},
                        {'rectified_frame':'left.optical'},{'rectified_frame':'right.optical'},
                        {'right_sensor':dict(CONFIG['right_sensor'],sensor_id='left')}):
            with self.assertRaises(ValueError):
                ProcessProvider.capability('opencv.rfdetr.stereo',dict(CONFIG,**changes))


@unittest.skipIf(cv2 is None,'optional OpenCV unavailable')
class Pipeline(unittest.TestCase):
    def make(self):
        self.backend = Backend(); self.backend.rows = ((80,40,240,120,.9,1,'fixture'),)
        detector = RFDETRTrajectories(self.backend,version='fixture',camera_motion='stationary',max_samples=2)
        model = StereoMetricContext(detector,**CONFIG)
        self.addCleanup(model.close)
        return model

    def test_actual_stereo_to_motion_and_absence(self):
        model = self.make()
        first = model.process(packet())
        self.assertAlmostEqual(first['objects'][0]['position_m'][2],2.5,places=2)
        self.assertIsNone(first['objects'][0]['velocity_m_s'])
        self.assertEqual(first['objects'][0]['detection']['label'],'fixture')
        self.assertFalse(first['objects'][0]['detection']['identity_verified'])
        second = model.process(packet(1,1100,disparity=10))
        self.assertAlmostEqual(second['objects'][0]['velocity_m_s'][2],-5,places=1)
        self.assertFalse(second['geometry']['physical_accuracy_verified'])
        self.assertEqual(second['stereo']['support'][0]['position_semantics'],'visible_surface_median')
        self.backend.rows = ()
        self.assertEqual(model.process(packet(2,1200))['objects'],[])
        self.backend.rows = ((80,40,240,120,.9,1,'fixture'),)
        self.assertIsNone(model.process(packet(3,1300))['objects'][0]['velocity_m_s'])

    def test_mismatches_reject_and_reset(self):
        model = self.make(); model.process(packet())
        with self.assertRaises(ValueError): model.process(packet(1,1100,right_ts=1110))
        self.assertIsNone(model.process(packet(2,1200))['objects'][0]['velocity_m_s'])
        f = packet(3,1300)
        for changed in (replace(f,timestamp_ms=1301),replace(f,subject_id='wrong'),
                        replace(f,sensor=replace(f.sensor,calibration_id='wrong'))):
            with self.assertRaises(ValueError): model.process(changed)
        self.assertIsNone(model.process(f)['objects'][0]['velocity_m_s'])
        next_f = packet(4,1400)
        self.assertIsNone(model.process(replace(next_f,sensor=replace(next_f.sensor,stream_epoch='runtime2')))['objects'][0]['velocity_m_s'])

    def test_missing_depth_is_incomplete_and_breaks_history(self):
        model = self.make(); model.process(packet())
        with patch.object(model._geometry,'compute',return_value=(
                np.full((160,320,3),np.nan,np.float32),np.zeros((160,320),bool))):
            result = model.process(packet(1,1100))
        self.assertEqual(result['objects'],[])
        self.assertFalse(result['geometry']['complete_snapshot'])
        self.assertTrue(result['geometry']['detection_snapshot_complete'])
        self.assertEqual(result['stereo']['support'][0]['status'],'insufficient_support')
        self.assertIsNone(model.process(packet(2,1200))['objects'][0]['velocity_m_s'])

    def test_nonidentity_rectification_reports_actual_coordinate_transform(self):
        c = calibration()
        angle = .04
        rotation = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        c['right_from_left_rotation'] = rotation.tolist()
        backend = Backend(); backend.rows = ((80,40,240,120,.9,1,'fixture'),)
        detector = RFDETRTrajectories(backend,version='fixture',camera_motion='stationary',max_samples=2)
        model = StereoMetricContext(detector,**dict(CONFIG,calibration=c))
        self.addCleanup(model.close)
        f = packet()
        _,raw,_ = unpack_stereo(f.data)
        left = np.frombuffer(raw,np.uint8).reshape(160,320,3).copy()
        k = np.array(c['left_intrinsics'],float)
        # Exact homography for a frontoparallel plane at original-left Z=2.5m.
        homography = k @ (rotation+np.outer([-.1,0,0],[0,0,1])/2.5) @ np.linalg.inv(k)
        right = cv2.warpPerspective(left,homography,(320,160))
        lf = replace(f,data=left)
        rf = replace(lf,data=right,sensor=replace(lf.sensor,sensor_id='right',coordinate_frame='right.optical'))
        paired = stereo_frame(lf,rf,left_interval_ms=[1000,1000],right_interval_ms=[1000,1000],
                              clock_id='capture.clock',alignment_id='alignment1')
        result = model.process(paired)
        provenance = result['stereo']['rectification']
        r1 = np.array(provenance['rectified_from_original_left_rotation'])
        self.assertGreater(np.max(np.abs(r1-np.eye(3))),.01)
        np.testing.assert_allclose(r1 @ r1.T,np.eye(3),atol=1e-8)
        self.assertEqual(provenance['original_left_frame'],'left.optical')
        self.assertEqual(provenance['rectified_left_frame'],'left.rectified')
        self.assertEqual(result['geometry']['input_coordinate_frame'],'left.rectified')
        original_point = r1.T @ np.array(result['objects'][0]['position_m'])
        self.assertAlmostEqual(original_point[2],2.5,delta=.05)
        self.assertAlmostEqual(provenance['rectified_baseline_m'],.1)
        provenance['rectified_from_original_left_rotation'][0][0] = 99
        self.assertNotEqual(model._geometry.provenance()['rectified_from_original_left_rotation'][0][0],99)

    def test_packet_integrity_and_metadata_bounds(self):
        f = packet()
        h,left,right = unpack_stereo(f.data)
        self.assertEqual(len(left),160*320*3)
        self.assertEqual(h['left']['interval_ms'],[1000,1000])
        for raw in (f.data[:-1],f.data+b'x',f.data[:-1]+bytes([f.data[-1]^1]),bytearray(f.data)):
            with self.assertRaises(ValueError): unpack_stereo(raw)
        model = self.make()
        result = model.process(f)
        self.assertLess(len(json.dumps(result)),10000)
        self.assertEqual(set(result['stereo']['left']),{'interval_ms','sensor','sha256','subject_id'})


WORKER = '''
import numpy as np
from types import SimpleNamespace
from humansense.providers.detection import RFDETRObjects
import humansense.provider_worker as w
class Fixture:
    def predict(self,data,**kwargs):
        return SimpleNamespace(xyxy=np.array([[80,40,240,120]]),confidence=np.array([.9]),class_id=np.array([1]),data={'class_name':['fixture']})
def local(cls,path,sha256,device,**kwargs):
    return cls(Fixture(),version='1.12.1.eager.sha256.'+sha256,**kwargs)
RFDETRObjects.from_local=classmethod(local)
w.main()
'''


@unittest.skipIf(cv2 is None,'optional OpenCV unavailable')
class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_hash_bound_recording_actual_worker_and_failure(self):
        spawn = asyncio.create_subprocess_exec
        async def fixture_spawn(*args,**kwargs):
            return await spawn(sys.executable,'-u','-c',WORKER,**kwargs)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).resolve()
            rows = []
            for i in range(2):
                data = packet(i,1000+i*100,disparity=8+2*i).data
                name = str(i)+'.gstereo'; (root/name).write_bytes(data)
                rows.append(dict(file=name,sha256=hashlib.sha256(data).hexdigest()))
            path = root/'manifest.json'
            data = json.dumps(dict(schema_version=1,config=CONFIG,packets=rows)).encode()
            path.write_bytes(data); digest = hashlib.sha256(data).hexdigest()
            with patch('asyncio.create_subprocess_exec',fixture_spawn):
                results = [row async for row in replay(str(path),digest)]
            self.assertEqual([r['capture_timestamp_ms'] for r in results],[1000,1100])
            self.assertAlmostEqual(results[1]['value']['objects'][0]['velocity_m_s'][2],-5,places=1)
            self.assertFalse(results[0]['physical_accuracy_verified'])
            (root/'1.gstereo').write_bytes(b'changed')
            with self.assertRaises(ValueError): recording(str(path),digest)

    async def test_owned_worker_sensorstreams_context_mcp(self):
        spawn = asyncio.create_subprocess_exec; children = []
        async def fixture_spawn(*args,**kwargs):
            child = await spawn(sys.executable,'-u','-c',WORKER,**kwargs)
            children.append(child)
            return child
        with patch('asyncio.create_subprocess_exec',fixture_spawn):
            worker = await ProcessProvider.open(sys.executable,'opencv.rfdetr.stereo',CONFIG,inference_timeout=10)
            try:
                async with HumanSense(consent=True) as human:
                    streams = SensorStreams(human)
                    streams.register(worker.spec,lambda:worker)
                    await streams.open('rig','left',[worker.spec.id])
                    now = timestamp()-200
                    for i in range(2):
                        f = packet(i,now+i*100,disparity=8+i*2)
                        await streams.submit(replace(f,sensor=replace(f.sensor,stream_epoch=None)))
                        await streams.drain()
                    observations = (await human.get_context())['observations']
                    value = next(iter(observations.values()))['value']
                    self.assertAlmostEqual(value['objects'][0]['velocity_m_s'][2],-5,places=1)
                    reply = await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    self.assertIn('visible_surface_median',json.dumps(reply))
                    await human.pause()
                    self.assertTrue(worker.closed)
            finally:
                await worker.close()
        self.assertIsNotNone(children[0].returncode)
