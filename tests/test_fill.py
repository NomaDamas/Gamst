import dataclasses
import json
import subprocess
import sys
import time
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2=np=None

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.fill import CalibratedColorFill
from humansense.provider_profiles import capability

CONFIG=dict(width=128,height=128,roi=[16,16,112,112],hsv_lower=[100,100,50],hsv_upper=[130,255,255],
            volume_curve_ml=[[0,0],[.5,100],[1,400]],calibration_id='beaker-v1',container_id='beaker',coordinate_frame='image')


def frame(fraction=.5,sequence=0):
    image=np.full((128,128,3),255,np.uint8)
    boundary=112-round(96*fraction)
    image[boundary:112,16:112]=(255,0,0)
    return InputFrame(image,'video',int(time.time()*1000),sensor=SensorMetadata('test','camera',sequence,'image',calibration_id='beaker-v1'))


class MetadataTests(unittest.TestCase):
    def test_core_discovery_and_configuration(self):
        code="from humansense.provider_profiles import capability; import sys; capability('opencv.fill',"+repr(CONFIG)+"); assert 'cv2' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable,'-S','-c',code],check=True)
        bad=({'width':True},{'width':4096},{'roi':[0,0,5,15]},{'hsv_lower':[180,100,50]},
             {'hsv_lower':[100,0,0]},{'hsv_upper':[90,255,255]},
             {'volume_curve_ml':[[0,0],[.5,100],[.4,400]]},
             {'volume_curve_ml':[[0,1],[1,400]]},{'volume_curve_ml':[[0,0],[1,float('nan')]]})
        for changes in bad:
            with self.assertRaises(ValueError):capability('opencv.fill',dict(CONFIG,**changes))
        self.assertNotEqual(capability('opencv.fill',CONFIG).version,capability('opencv.fill',dict(CONFIG,container_id='different')).version)


@unittest.skipIf(cv2 is None,'optional OpenCV/NumPy absent')
class FillTests(unittest.TestCase):
    def test_actual_hsv_and_nonlinear_volume_curve(self):
        p=CalibratedColorFill(**CONFIG)
        for fraction,volume in ((.25,50),(.5,100),(.75,250)):
            result=p.process(frame(fraction))['fill']
            self.assertEqual(result['status'],'estimated')
            self.assertEqual(result['height_fraction'],fraction)
            self.assertAlmostEqual(result['volume_ml'],volume)
            self.assertEqual(result['meniscus_image_y_px'],112-96*fraction)
            lo,hi=result['volume_resolution_interval_ml']
            self.assertLess(lo,volume);self.assertGreater(hi,volume)
            self.assertFalse(result['calibration_verified'])
            self.assertEqual(result['opencv_version'],cv2.__version__)

    def test_unknown_not_empty_full_or_transparent_estimate(self):
        p=CalibratedColorFill(**CONFIG)
        for fraction,reason in ((0,'no_color_evidence'),(1,'headspace_not_visible')):
            result=p.process(frame(fraction))['fill']
            self.assertEqual(result['status'],'unknown');self.assertEqual(result['reason'],reason)
            self.assertIsNone(result['volume_ml'])
        f=frame(.5);f.data[64:112,16:112]=(128,128,128)
        self.assertEqual(p.process(f)['fill']['reason'],'no_color_evidence')
        f=frame(.5);f.data[110:112,16:112]=(255,255,255)
        self.assertEqual(p.process(f)['fill']['reason'],'bottom_color_support_missing')

    def test_occlusion_stripes_and_slanted_boundary_rejected(self):
        p=CalibratedColorFill(**CONFIG)
        f=frame(.5);f.data[80:90,16:112]=(255,255,255)
        self.assertEqual(p.process(f)['fill']['reason'],'fragmented_or_sloped_profile')
        f=frame(.5);f.data[32:40,16:112]=(255,0,0)
        self.assertEqual(p.process(f)['fill']['status'],'unknown')
        f=frame(0)
        for x in range(16,112):f.data[50+(x-16)//4:112,x]=(255,0,0)
        self.assertEqual(p.process(f)['fill']['status'],'unknown')

    def test_transition_resolution_is_not_confidence(self):
        p=CalibratedColorFill(**CONFIG);f=frame(.5)
        f.data[62:64,16:64]=(255,0,0)
        r=p.process(f)['fill']
        self.assertEqual(r['transition_rows'],2)
        self.assertEqual(r['meniscus_image_y_px'],63)
        for observed,expected in zip(r['height_resolution_interval'],[(49-1.5)/96,(49+1.5)/96]):
            self.assertAlmostEqual(observed,expected)

    def test_invalid_frames_and_close(self):
        p=CalibratedColorFill(**CONFIG);f=frame()
        for bad in (dataclasses.replace(f,data=np.ma.array(f.data,mask=True)),dataclasses.replace(f,timestamp_ms=True),
                    dataclasses.replace(f,sensor=None),dataclasses.replace(f,data=np.zeros((32,32,3),np.uint8)),
                    dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,calibration_id='wrong'))):
            with self.assertRaises(ValueError):p.process(bad)
        p.reset();r=p.process(f);r['fill']['volume_curve_ml'][1][1]=999
        self.assertEqual(p.process(f)['fill']['volume_ml'],100)
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(f)


@unittest.skipIf(cv2 is None,'optional OpenCV/NumPy absent')
class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_worker_stream_mcp_and_unknown_replacement(self):
        async with HumanSense(consent=True) as human:
            spec=capability('opencv.fill',CONFIG)
            streams=SensorStreams(human)
            async def create():return await ProcessProvider.open(sys.executable,'opencv.fill',CONFIG)
            streams.register(spec,create);await streams.open('test','camera',[spec.id])
            await streams.submit(frame(.5,0));await streams.drain()
            value=next(iter((await human.get_context())['observations'].values()))['value']['fill']
            self.assertEqual(value['volume_ml'],100)
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
            self.assertIn('volume_ml',json.dumps(result))
            await streams.submit(frame(0,1));await streams.drain()
            value=next(iter((await human.get_context())['observations'].values()))['value']['fill']
            self.assertEqual(value['status'],'unknown');self.assertIsNone(value['volume_ml'])
            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)
