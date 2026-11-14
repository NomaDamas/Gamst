import asyncio
from dataclasses import replace
import json
import struct
import subprocess
import sys
import unittest

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.efficientphys_context import EfficientPhysContext, face_frame, decode, spectral_candidate

try:
    import numpy as np
    import cv2
except ImportError:
    np = None

CONFIG=dict(source_path='/source.py',checkpoint_path='/model.pth',checkpoint_sha256='a'*64)


class Backend:
    def __init__(self): self.calls=0; self.closed=False; self.bad=False
    def predict(self,data):
        self.calls+=1; self.data=data.copy()
        if self.bad:return np.zeros((180,1),np.float32)
        # A synthetic derivative at1.5Hz is a spectral oracle, not real physiology.
        return np.diff(np.sin(np.arange(181)*2*np.pi*1.5/30)).astype(np.float32)
    def close(self):self.closed=True


def frame(i=0,track='face1',ts=None,epoch=None,variation=True):
    image=np.zeros((80,90,3),np.uint8)
    image[...,0]=20+(i%30 if variation else 0)
    image[...,1]=100
    image[...,2]=190
    f=InputFrame(image,'video',1000+round(i*1000/30) if ts is None else ts,
        sensor=SensorMetadata('cam','video',i,coordinate_frame='image1',stream_epoch=epoch))
    return face_frame(f,track_id=track,box=[2,3,76,78] if track else None,
        detector_id='caller-face',detector_version='1',clock_id='capture-clock')


def _capture_mode(f,sequence,capture_sequence):
    size=struct.unpack('!I',f.data[8:12])[0]
    meta=json.loads(f.data[12:12+size])
    meta['provenance']['sequence_kind']='capture'
    meta['provenance']['capture_sequence']=capture_sequence
    head=json.dumps(meta,separators=(',',':')).encode()
    return replace(f,sensor=replace(f.sensor,sequence=sequence,capture_sequence=capture_sequence),
        data=f.data[:8]+struct.pack('!I',len(head))+head+f.data[12+size:])


class Discovery(unittest.TestCase):
    def test_profile_import_and_errors(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("efficientphys.local",'+repr(CONFIG)+');assert not {"torch","numpy","cv2"}&sys.modules.keys()'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for bad in ({'source_path':'relative'},{'device':'auto'},{'checkpoint_sha256':'x'},{'window_frames':10}):
            with self.assertRaises(ValueError): ProcessProvider.capability('efficientphys.local',dict(CONFIG,**bad))


@unittest.skipIf(np is None,'optional NumPy/OpenCV absent')
class Context(unittest.TestCase):
    def test_packet_capture_binding_color_and_loss(self):
        f=frame();meta,rgb=decode(f)
        np.testing.assert_array_equal(rgb[0,0],[190,100,20])
        self.assertEqual(meta['box'],[2,3,76,78])
        self.assertIsNone(decode(frame(1,track=None))[1])
        invalid=InputFrame(np.ma.array(np.ones((80,90,3),np.uint8),mask=True),'video',1000,
            sensor=SensorMetadata('cam','video',0,coordinate_frame='image1'))
        with self.assertRaises(ValueError):
            face_frame(invalid,track_id='f',box=[0,0,40,40],detector_id='d',detector_version='1',clock_id='c')
        for bad in (replace(f,timestamp_ms=f.timestamp_ms+1),replace(f,data=f.data[:-1]),
                    replace(f,data=f.data+b'x'),replace(f,data=f.data[:-1]+bytes([f.data[-1]^1])),
                    replace(f,sensor=replace(f.sensor,source_id='other'))):
            with self.assertRaises(ValueError):decode(bad)
        # Stream lifecycle epoch is managed by SensorStreams, not copied into header.
        self.assertEqual(decode(replace(f,sensor=replace(f.sensor,stream_epoch='manager')))[0],meta)
        size=struct.unpack('!I',f.data[8:12])[0]
        header=f.data[12:12+size]
        duplicate=b'{"track_id":"bad",'+header[1:]
        with self.assertRaises(ValueError):decode(replace(f,data=f.data[:8]+struct.pack('!I',len(duplicate))+duplicate+f.data[12+size:]))

    def test_complete_window_normalization_and_spectral_truth(self):
        backend=Backend();p=EfficientPhysContext(backend,checkpoint_sha256='a'*64)
        for i in range(179):
            value=p.process(frame(i))
            if i == 0:
                self.assertEqual(value['summary']['status'],'collecting')
                self.assertIsNone(value['signal']['spectral_candidate'])
            else:self.assertIsNone(value)
        self.assertEqual(backend.calls,0)
        value=p.process(frame(179));self.assertEqual(backend.calls,1)
        self.assertEqual(backend.data.shape,(180,72,72,3))
        self.assertAlmostEqual(float(backend.data.mean()),0,places=5)
        self.assertAlmostEqual(float(backend.data.std()),1,places=5)
        summary=value['summary'];self.assertTrue(summary['window_complete'])
        self.assertEqual(summary['capture_start_ms'],1000)
        self.assertEqual(summary['capture_end_ms'],6967)
        self.assertFalse(summary['model_quality_validated'])
        self.assertFalse(summary['offline_preprocessing_parity'])
        self.assertIsNone(value['signal']['pulse_bpm'])
        self.assertAlmostEqual(value['signal']['spectral_candidate']['equivalent_bpm'],90,delta=.02)
        self.assertLess(len(json.dumps(value)),4096)
        self.assertEqual(len(p.images),0)
        self.assertIsNone(p.process(frame(180)))
        self.assertEqual(len(p.images),1)
        p.close();p.close();self.assertTrue(backend.closed)
        with self.assertRaises(RuntimeError):p.process(frame(181))
        with self.assertRaises(RuntimeError):p.reset()

    def test_gap_identity_clock_duplicate_and_reset(self):
        p=EfficientPhysContext(Backend(),checkpoint_sha256='a'*64)
        a=p.process(frame(0));p.process(frame(1))
        b=p.process(frame(3));self.assertEqual(b['summary']['accumulated_frames'],1)
        self.assertNotEqual(a['summary']['state_segment'],b['summary']['state_segment'])
        b=p.process(frame(4,track='face2'));self.assertEqual(b['summary']['accumulated_frames'],1)
        b=p.process(frame(5,track=None));self.assertEqual(b['summary']['status'],'face_unavailable')
        self.assertEqual(len(p.images),0)
        p.process(frame(6))
        with self.assertRaises(ValueError):p.process(frame(6))
        self.assertEqual(len(p.images),0)
        p.process(frame(7));p.reset();self.assertEqual(len(p.images),0)
        # Explicit original capture_sequence can differ from delivery sequence.
        f=frame(0); f=_capture_mode(f,99,0)
        self.assertEqual(p.process(f)['summary']['accumulated_frames'],1)
        f=frame(1); f=_capture_mode(f,100,1)
        self.assertIsNone(p.process(f));self.assertEqual(len(p.images),2)
        self.assertEqual(p.process(frame(2))['summary']['accumulated_frames'],1)
        p.reset()
        p.process(frame(8,ts=2000));b=p.process(frame(9,ts=2038))
        self.assertEqual(b['summary']['accumulated_frames'],1)
        # Cumulative drift even if each individual interval is within tolerance.
        p.reset()
        for i in range(4): b=p.process(frame(i,ts=3000+i*34))
        self.assertIsNone(b);self.assertEqual(len(p.images),4)
        b=p.process(frame(4,ts=3136));self.assertEqual(b['summary']['accumulated_frames'],1)

    def test_static_textured_face_and_bad_backend_rejected(self):
        backend=Backend();p=EfficientPhysContext(backend,checkpoint_sha256='a'*64)
        for i in range(180):v=p.process(frame(i,variation=False))
        self.assertEqual(v['summary']['status'],'insufficient_variation');self.assertEqual(backend.calls,0)
        backend.bad=True
        for i in range(179):p.process(frame(180+i))
        with self.assertRaises(ValueError):p.process(frame(359))
        self.assertEqual(len(p.images),0)
        self.assertIsNone(spectral_candidate(np.zeros(180),30))
        with self.assertRaises(ValueError):spectral_candidate(np.zeros(180)+np.nan,30)


@unittest.skipIf(np is None,'optional NumPy/OpenCV absent')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_sensor_stream_and_mcp_observation(self):
        from humansense.schema import timestamp
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);backend=Backend()
            p=EfficientPhysContext(backend,checkpoint_sha256='a'*64)
            streams.register(p.spec,lambda:p);await streams.open('cam','video',[p.spec.id])
            start=timestamp()-6000
            for i in range(180):
                await streams.submit(frame(i,ts=start+round(i*1000/30)))
                await streams.drain()
            records=human.store.latest_signals('local-user',0,timestamp())
            value=records[0]['observation']['value']
            self.assertEqual(value['summary']['status'],'estimated')
            completed_timestamp=records[0]['timestamp_ms']
            # Polling at1Hz after a completion must not lose that completed window.
            for i in range(180,240):
                await streams.submit(frame(i,ts=start+round(i*1000/30)))
                await streams.drain()
                if i in (209,239):
                    latest=human.store.latest_signals('local-user',0,start+9000)[0]
                    self.assertEqual(latest['timestamp_ms'],completed_timestamp)
                    self.assertEqual(latest['observation']['value']['summary']['status'],'estimated')
            answer=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            self.assertIn('spectral_candidate',str(answer))
            await streams.submit(frame(240,track=None,ts=start+8000))
            await streams.drain()
            latest=human.store.latest_signals('local-user',0,start+9000)[0]
            self.assertEqual(latest['observation']['value']['summary']['status'],'face_unavailable')
            self.assertIsNone(latest['observation']['value']['signal']['spectral_candidate'])
            await streams.clear();self.assertTrue(backend.closed)
