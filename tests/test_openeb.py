import asyncio
from dataclasses import replace
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.openeb import OpenEBContext,event_packet,decode_event_packet,HEADER,MAX_EVENTS
from humansense.providers.openeb_capture import OpenEBCapture

try:import numpy as np
except ImportError:np=None

CONFIG=dict(width=16,height=8,sensor_origin_us='0',capture_origin_ms=1000,calibration_id='clock1')


def events(rows=()):
    return np.array(list(rows),dtype=[('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')])


def frame(i=0,rows=None,start=None,end=None):
    start=i*10000 if start is None else start
    end=start+10000 if end is None else end
    if rows is None:rows=[(0,0,0,start),(15,7,1,end),(15,7,1,end)]
    packet=event_packet(events(rows),width=16,height=8,start_us=start,end_us=end)
    return InputFrame(packet,'events',1000+end//1000,
        sensor=SensorMetadata('eventcam','events',i,coordinate_frame='pixels',calibration_id='clock1'))


class Discovery(unittest.TestCase):
    def test_import_isolation_and_config(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("openeb.eventcd",'+repr(CONFIG)+');assert not {"numpy","metavision_core","metavision_hal"}&sys.modules.keys()'
        run=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(run.returncode,0,run.stderr)
        for bad in ({'width':True},{'height':5000},{'sensor_origin_us':'00'},{'sensor_origin_us':str(2**63)},
                    {'capture_origin_ms':True},{'calibration_id':''}):
            with self.assertRaises(ValueError):ProcessProvider.capability('openeb.eventcd',dict(CONFIG,**bad))
        with self.assertRaises(ValueError):OpenEBCapture.camera('')
        with self.assertRaises(ValueError):OpenEBCapture.recording('relative.raw')


@unittest.skipIf(np is None,'optional NumPy absent')
class Context(unittest.TestCase):
    def test_exact_field_packet_and_activity_units(self):
        f=frame();width,height,start,end,values=decode_event_packet(f.data)
        self.assertEqual((width,height,start,end),(16,8,0,10000))
        self.assertEqual(f.data[HEADER.size:HEADER.size+13],struct.pack('<HHBQ',0,0,0,0))
        v=OpenEBContext(**CONFIG).process(f)
        a=v['activity'];self.assertEqual(a['event_rate_hz'],300)
        self.assertEqual((a['positive_count'],a['negative_count'],a['active_pixel_count']),(2,1,2))
        self.assertEqual(a['polarity_grid']['negative'][0],1)
        self.assertEqual(a['polarity_grid']['positive'][-1],2)
        self.assertEqual(a['activity_centroid']['x'],10)
        self.assertFalse(v['summary']['object_detection_available'])
        self.assertFalse(v['summary']['clock_calibration_verified'])

    def test_empty_equal_time_and_large_native_clock(self):
        p=OpenEBContext(**CONFIG)
        self.assertEqual(p.process(frame(rows=[]))['activity']['event_rate_hz'],0)
        v=p.process(frame(1,start=10000,end=10000,rows=[(1,2,1,10000)]))
        self.assertIsNone(v['activity']['event_rate_hz']);self.assertTrue(v['summary']['continuous'])
        self.assertTrue(p.process(frame(2,start=10000,end=10000,rows=[(2,2,0,10000)]))['summary']['continuous'])
        origin=2**63-10001
        config=dict(CONFIG,sensor_origin_us=str(origin))
        f=frame(start=origin,end=origin+10000,rows=[])
        f=replace(f,timestamp_ms=1010)
        v=OpenEBContext(**config).process(f)
        self.assertEqual(v['summary']['native_start_us'],str(origin))

    def test_sequence_gap_owner_overlap_clock_and_close(self):
        p=OpenEBContext(**CONFIG);a=p.process(frame());b=p.process(frame(1))
        self.assertEqual(a['summary']['state_segment'],b['summary']['state_segment'])
        c=p.process(frame(3));self.assertFalse(c['summary']['continuous'])
        self.assertNotEqual(b['summary']['state_segment'],c['summary']['state_segment'])
        with self.assertRaises(ValueError):p.process(frame(3))
        self.assertIsNone(p.last)
        p.process(frame())
        with self.assertRaises(ValueError):p.process(frame(1,start=9999))
        with self.assertRaises(ValueError):p.process(replace(frame(),timestamp_ms=1011))
        f=frame();f=replace(f,sensor=replace(f.sensor,calibration_id='other'))
        with self.assertRaises(ValueError):p.process(f)
        p.reset();p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame())
        with self.assertRaises(RuntimeError):p.reset()

    def test_malformed_crc_geometry_masked_and_bounds(self):
        f=frame();bad=bytearray(f.data);bad[-1]^=1
        for data in (b'',f.data[:-1],f.data+b'x',bytes(bad)):
            with self.assertRaises(ValueError):decode_event_packet(data)
        for data in (events([(16,0,0,0)]),events([(0,0,2,0)]),events([(0,0,0,10001)]),
                     events([(0,0,0,2),(0,0,0,1)]),np.ma.array(events(),mask=False),
                     np.zeros(MAX_EVENTS+1,dtype=events().dtype)):
            with self.assertRaises(ValueError):event_packet(data,width=16,height=8,start_us=0,end_us=10000)
        # Valid CRC does not override semantic field validation.
        data=bytearray(f.data);data[HEADER.size+4]=2
        data[-4:]=struct.pack('<I',zlib.crc32(data[:-4]))
        with self.assertRaises(ValueError):decode_event_packet(bytes(data))

    def test_max_packet_no_native_padding_and_owned_decode(self):
        source=np.zeros(MAX_EVENTS,dtype=events().dtype);source['p']=1
        data=event_packet(source,width=16,height=8,start_us=0,end_us=0)
        self.assertEqual(len(data),HEADER.size+MAX_EVENTS*13+4)
        source['p']=0
        decoded=decode_event_packet(data)[4];self.assertEqual(int(decoded['p'].sum()),MAX_EVENTS)
        decoded['p']=0;self.assertEqual(int(decode_event_packet(data)[4]['p'].sum()),MAX_EVENTS)


@unittest.skipIf(np is None,'optional NumPy absent')
class Capture(unittest.TestCase):
    def test_native_api_arguments_boundaries_and_early_close(self):
        instances=[]
        class Reader:
            def __init__(self,*args,**kwargs):
                self.args=args;self.kwargs=kwargs;self.reader=self;self.stops=0;self.time=np.int64(0);instances.append(self)
            def get_size(self):return (8,16)
            def get_current_time(self):return self.time
            def __iter__(self):
                self.time=np.int64(10000);yield events([(1,2,1,10000)])
                self.time=np.int64(10000);yield events([(2,3,0,10000)])
                self.time=np.int64(20000);yield events()
            def __exit__(self,*args):self.stops+=1
        options=dict(source_id='eventcam',sensor_id='events',sensor_origin_us='0',capture_origin_ms=1000,
            calibration_id='clock1',coordinate_frame='pixels')
        with patch('humansense.providers.openeb_capture._installed_iterator',return_value=Reader):
            with OpenEBCapture.camera('SERIAL1',**options) as capture:
                reader=instances[-1]
                self.assertEqual(reader.kwargs,dict(start_ts=0,mode='mixed',delta_t=10000,n_events=8192,relative_timestamps=False))
                p=OpenEBContext(**capture.config)
                a=p.process(capture.next_frame());b=p.process(capture.next_frame());c=p.process(capture.next_frame())
                self.assertEqual(a['activity']['event_count'],1)
                self.assertIsNone(b['activity']['event_rate_hz'])
                self.assertEqual(c['activity']['event_count'],0)
                with self.assertRaises(StopIteration):capture.next_frame()
                self.assertTrue(capture.closed)
            self.assertGreater(reader.stops,0)
            capture=OpenEBCapture.camera('SERIAL1',**options);reader=instances[-1]
            capture.close();capture.close();self.assertEqual(reader.stops,1)
            with self.assertRaises(RuntimeError):capture.next_frame()
            with self.assertRaises(ValueError):OpenEBCapture.camera('SERIAL1',**dict(options,sensor_origin_us='1'))


@unittest.skipIf(np is None,'optional NumPy absent')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_real_context_worker_sensor_stream_mcp(self):
        from humansense.schema import timestamp
        from humansense.mcp import MCPServer
        config=dict(CONFIG,capture_origin_ms=timestamp()-10)
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);spec=ProcessProvider.capability('openeb.eventcd',config)
            async def factory():return await ProcessProvider.open(sys.executable,'openeb.eventcd',config)
            streams.register(spec,factory);await streams.open('eventcam','events',[spec.id])
            await streams.submit(replace(frame(),timestamp_ms=config['capture_origin_ms']+10))
            await streams.drain()
            result=human.store.latest_signals('local-user',0,timestamp())[0]
            self.assertEqual(result['observation']['value']['activity']['event_count'],3)
            value=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
            self.assertIn('polarity_grid',str(value))
            await streams.clear()
