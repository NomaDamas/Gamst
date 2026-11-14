"""Real owned protocol/ctypes/C++ bridge, with a test-double SLAM System only."""
import asyncio
from dataclasses import replace
import hashlib
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
import test_orbslam_native as fixture
from humansense import HumanSense,ProcessProvider,SensorStreams,SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.orbslam_context import ORBSLAMContext
from humansense.mcp import MCPServer
from humansense.schema import timestamp
try:
    import numpy as np
    import cv2
except ImportError:np=None

CONFIG=dict(library_path='/local/bridge.so',vocabulary_path='/local/vocab',settings_path='/local/camera.yaml',
    library_sha256='a'*64,vocabulary_sha256='b'*64,settings_sha256='c'*64,calibration_id='cal1')


def frame(i,ts=None,color=2):
    return InputFrame(np.full((96,96,3),color,np.uint8),'video',1000+i*100 if ts is None else ts,
        sensor=SensorMetadata('cam','video',i,coordinate_frame='camera',calibration_id='cal1',stream_epoch='epoch1'))


def assets(test,slow=False):
    f=fixture.Native();test.addCleanup(f.doCleanups)
    header=fixture.HEADER
    if slow:
        header=header.replace('#include <stdexcept>','#include <stdexcept>\n#include <thread>\n#include <chrono>')
        header=header.replace('if(im.data[0]==254)', 'if(im.data[0]==253)std::this_thread::sleep_for(std::chrono::seconds(60));if(im.data[0]==254)')
    with patch.object(fixture,'HEADER',header):f.setUp()
    return dict(f.args(),calibration_id='cal1')


class Discovery(unittest.TestCase):
    def test_no_optional_import_or_asset_io(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("orbslam3.monocular",'+repr(CONFIG)+');assert not {"numpy","cv2"}&sys.modules.keys()'
        p=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        for changes in ({'max_frames':True},{'max_gap_ms':0},{'sequence_step':1001},{'library_path':'relative'}):
            with self.assertRaises(ValueError):ORBSLAMContext.capability(**dict(CONFIG,**changes))


@unittest.skipIf(np is None or shutil.which('c++') is None,'requires numpy/OpenCV/compiler')
class Context(unittest.TestCase):
    def provider(self,**changes):
        p=ORBSLAMContext.from_local(**dict(assets(self),**changes));self.addCleanup(p.close);return p

    def test_pose_capture_gaps_owner_and_budget(self):
        p=self.provider(max_frames=4)
        a=p.process(frame(0,color=0));self.assertIsNone(a['pose']['world_to_camera'])
        b=p.process(frame(1));self.assertAlmostEqual(b['pose']['world_to_camera'][0][3],.1)
        self.assertFalse(b['pose']['coordinate_continuity_verified'])
        c=p.process(frame(3));self.assertEqual(c['provenance']['reset_count'],1)
        self.assertNotEqual(a['provenance']['reset_segment'],c['provenance']['reset_segment'])
        other=frame(0,100);other=replace(other,sensor=replace(other.sensor,source_id='other'))
        d=p.process(other);self.assertEqual(d['provenance']['reset_count'],2)
        self.assertEqual(d['provenance']['capture_ms'],100)
        with self.assertRaisesRegex(RuntimeError,'budget'):p.process(replace(other,timestamp_ms=200,sensor=replace(other.sensor,sequence=1)))
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.reset()

    def test_validation_error_resets_before_next_valid_capture(self):
        p=self.provider(sequence_step=2,max_gap_ms=500)
        p.process(frame(0));a=p.process(frame(2));self.assertEqual(a['provenance']['reset_count'],0)
        for bad in (frame(2),replace(frame(4),sensor=replace(frame(4).sensor,calibration_id='wrong')),
                    replace(frame(4),data=np.zeros((32,32,3),np.uint8)),replace(frame(4),timestamp_ms=True)):
            with self.assertRaises(ValueError):p.process(bad)
        b=p.process(frame(4));self.assertEqual(b['provenance']['reset_count'],1)
        c=p.process(frame(6,3000));self.assertEqual(c['provenance']['reset_count'],2)
        p.reset();self.assertEqual(p.process(frame(0,0))['provenance']['reset_count'],3)


@unittest.skipIf(np is None or shutil.which('c++') is None,'requires numpy/OpenCV/compiler')
class Owned(unittest.IsolatedAsyncioTestCase):
    async def test_concrete_worker_sensorstreams_mcp_and_reap(self):
        config=dict(assets(self),sequence_step=2)
        worker=await ProcessProvider.open(sys.executable,'orbslam3.monocular',config)
        child=worker._process
        try:
            with self.assertRaises(ValueError):
                await worker.process(replace(frame(0),data=np.ma.array(frame(0).data,mask=True)))
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                await streams.open('cam','video',[worker.spec.id]);start=timestamp()-1000
                try:
                    f=frame(0,start);f=replace(f,sensor=replace(f.sensor,stream_epoch=None))
                    await streams.submit(f);await streams.drain()
                    f2=frame(2,start+200);f2=replace(f2,sensor=replace(f2.sensor,stream_epoch=None))
                    await streams.submit(f2);await streams.drain()
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    item=next(row['observation'] for row in reply['items'] if row['observation']['value']['provenance']['capture_ms']==start+200);value=item['value']
                    self.assertEqual(value['provenance']['capture_ms'],start+200)
                    self.assertEqual(value['provenance']['reset_count'],0)
                    self.assertAlmostEqual(value['pose']['world_to_camera'][0][3],.2)
                    self.assertEqual(value['provenance']['calibration_id'],'cal1')
                    self.assertTrue(value['pose']['tracked']);self.assertFalse(value['pose']['metric'])
                    self.assertEqual(item['sensor']['capture_sequence'],2)
                    f6=frame(6,start+600);f6=replace(f6,sensor=replace(f6.sensor,stream_epoch=None))
                    await streams.submit(f6);await streams.drain()
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                    latest=next(row['observation']['value'] for row in reply['items'] if row['observation']['value']['provenance']['capture_ms']==start+600)
                    self.assertEqual(latest['provenance']['reset_count'],1)
                finally:await streams.clear()
        finally:await worker.close()
        self.assertIsNotNone(child.returncode)

    async def test_native_block_timeout_reaps_owned_worker(self):
        worker=await ProcessProvider.open(sys.executable,'orbslam3.monocular',assets(self,slow=True),inference_timeout=3)
        child=worker._process
        try:
            # Warm imports before shortening deadline; pixel253 blocks only native tracking.
            await worker.process(frame(0))
            worker.timeout=.15
            with self.assertRaises(asyncio.TimeoutError):await worker.process(frame(1,color=253))
        finally:await worker.close()
        self.assertIsNotNone(child.returncode)

    async def test_native_error_and_cancellation_reap_without_publication(self):
        config=assets(self,slow=True)
        for mode in ('error','cancel'):
            worker=await ProcessProvider.open(sys.executable,'orbslam3.monocular',config)
            child=worker._process
            try:
                await worker.process(frame(0))
                if mode=='error':
                    with self.assertRaises(RuntimeError):await worker.process(frame(1,color=254))
                else:
                    task=asyncio.create_task(worker.process(frame(1,color=253)))
                    await asyncio.sleep(.1);task.cancel()
                    with self.assertRaises(asyncio.CancelledError):await task
            finally:await worker.close()
            self.assertIsNotNone(child.returncode)
