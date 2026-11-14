import asyncio
from dataclasses import replace
import json
import struct
import subprocess
import sys
import unittest
import zlib

from humansense import HumanSense,ProcessProvider,SensorStreams
from humansense.providers.visual_microphone_context import VisualMicrophoneContext,window_frame,decode,configuration,PREFIX,MAGIC

try:
    import numpy as np
    import scipy
except ImportError:np=None

CONFIG=dict(fps_numerator=2000,fps_denominator=1,jitter_ns=100,native_origin_ns='0',capture_origin_ms=1000,calibration_id='clock1')


def frame(window=0,config=None,flat=False,offset=0):
    config=config or CONFIG;n=256;x,y=np.meshgrid(np.arange(32),np.arange(32))
    indices=np.arange(window*n+offset,(window+1)*n+offset,dtype=np.int64)
    times=np.array([int(i)*1000000000*config['fps_denominator']//config['fps_numerator'] for i in indices],np.int64)
    shift=.15*np.sin(2*np.pi*125*indices/2000)
    pixels=np.array([128+45*np.cos(2*np.pi*(x-d)/4)+45*np.cos(2*np.pi*(y-d)/4) for d in shift]).round().astype(np.uint8)
    if flat:pixels[:]=128
    return window_frame(pixels,times,indices,config=config,source_id='cam',sensor_id='video',coordinate_frame='pixels',
        window_sequence=window,roi_id='foil1',roi_box=[10,20,42,52],source_shape=[80,90],source_sha256='a'*64,shutter='global',exposure_ns=1000)


def remeta(f,change):
    size=PREFIX.unpack(f.data[:PREFIX.size])[1];meta=json.loads(f.data[PREFIX.size:PREFIX.size+size]);change(meta)
    head=json.dumps(meta,separators=(',',':')).encode();data=PREFIX.pack(MAGIC,len(head))+head+f.data[PREFIX.size+size:-4]
    return replace(f,data=data+struct.pack('<I',zlib.crc32(data)))


class Discovery(unittest.TestCase):
    def test_dependency_light_and_configuration(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("visual_microphone.phase",'+repr(CONFIG)+');assert not {"numpy","scipy","torch"}&sys.modules.keys()'
        p=subprocess.run([sys.executable,'-c',code],capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
        for bad in ({'fps_numerator':True},{'fps_numerator':99},{'jitter_ns':5001},{'native_origin_ns':'01'},{'capture_origin_ms':-1},{'other':0}):
            with self.assertRaises(ValueError):configuration(dict(CONFIG,**bad))


@unittest.skipIf(np is None,'optional NumPy/SciPy absent')
class Windows(unittest.TestCase):
    def test_actual_numeric_output_and_outer_binding(self):
        f=frame();p=VisualMicrophoneContext(**CONFIG);value=p.process(f)
        self.assertEqual(value['vibration']['peak_hz'],125)
        self.assertEqual(value['summary']['original_last_frame_index'],255)
        self.assertEqual(value['summary']['native_end_ns'],'127500000')
        self.assertFalse(value['summary']['clock_calibration_verified'])
        self.assertNotIn('waveform',value);self.assertEqual(value['vibration']['quality'],'not_assessed')
        for bad in (replace(f,timestamp_ms=f.timestamp_ms+1),replace(f,sensor=replace(f.sensor,source_id='other')),
                    replace(f,sensor=replace(f.sensor,capture_sequence=1)),replace(f,sample_rate=2000),
                    replace(f,sensor=replace(f.sensor,channels=('gray',)))):
            with self.assertRaises(ValueError):p.process(bad)
        # Delivery index may differ, original window index remains binding.
        self.assertEqual(p.process(replace(f,sensor=replace(f.sensor,sequence=999)))['summary']['window_sequence'],0)

    def test_lifecycle_gaps_overlap_and_abstention(self):
        p=VisualMicrophoneContext(**CONFIG);v=p.process(frame());segment=v['summary']['state_segment']
        v=p.process(frame(1));self.assertTrue(v['summary']['continuous']);self.assertEqual(v['summary']['state_segment'],segment)
        with self.assertRaises(ValueError):p.process(frame(1))
        v=p.process(frame(2));self.assertFalse(v['summary']['continuous'])
        v=p.process(frame(4));self.assertFalse(v['summary']['continuous'])
        v=p.process(frame(5,flat=True));self.assertIsNone(v['vibration']['peak_hz']);self.assertEqual(v['vibration']['status'],'insufficient_texture')
        p.reset();self.assertFalse(p.process(frame())['summary']['continuous'])
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame())

    def test_metadata_crc_length_and_type_fail_closed(self):
        f=frame()
        for bad in (replace(f,data=f.data[:-1]),replace(f,data=f.data[:-1]+bytes([f.data[-1]^1])),
                    remeta(f,lambda m:m.update(shape=[True,32,32])),remeta(f,lambda m:m.update(extra=1)),
                    remeta(f,lambda m:m.update(roi_box=[0,0,31,32])),remeta(f,lambda m:m.update(shutter='rolling')),
                    remeta(f,lambda m:m.update(source_sha256='x'))):
            with self.assertRaises(ValueError):decode(bad.data)
        size=PREFIX.unpack(f.data[:PREFIX.size])[1];head=f.data[PREFIX.size:PREFIX.size+size]
        head=b'{"shape":[256,32,32],'+head[1:];data=PREFIX.pack(MAGIC,len(head))+head+f.data[PREFIX.size+size:-4]
        with self.assertRaises(ValueError):decode(data+struct.pack('<I',zlib.crc32(data)))

    def test_timing_indices_and_owned_snapshot(self):
        meta,pixels,times,indices,_=decode(frame().data)
        def make(t=times,i=indices,p=pixels,conf=CONFIG):
            return window_frame(p,t,i,config=conf,**{k:meta[k] for k in ('subject_id','source_id','sensor_id','coordinate_frame','window_sequence','roi_id','roi_box','source_shape','source_sha256','shutter','exposure_ns')})
        for t,i in ((times+np.arange(256,dtype=np.int64)*10,indices),(times,indices+np.arange(256,dtype=np.int64)),
                    (times[::-1],indices),(times.astype(np.uint64)+2**63,indices)):
            with self.assertRaises(ValueError):make(t,i)
        captured=make();pixels[:]=0;times[:]=0;indices[:]=0
        self.assertGreater(decode(captured.data)[1].max(),0)
        with self.assertRaises(ValueError):make(p=np.ma.array(pixels))
        with self.assertRaises(ValueError):make(p=np.broadcast_to(np.uint8(0),(4096,128,128)))
        rational=dict(CONFIG,fps_numerator=2200,jitter_ns=1)
        self.assertEqual(decode(frame(config=rational).data)[-1],1115)


@unittest.skipIf(np is None,'optional NumPy/SciPy absent')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_owned_worker_sensorstreams_mcp_and_freshness(self):
        from humansense.schema import timestamp
        from humansense.mcp import MCPServer
        config=dict(CONFIG,capture_origin_ms=timestamp()-400)
        worker=await ProcessProvider.open(sys.executable,'visual_microphone.phase',config)
        child=worker._process
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);spec=worker.spec
            streams.register(spec,lambda:worker);await streams.open('cam','video',[spec.id])
            try:
                f=frame(config=config);await streams.submit(f);await streams.drain()
                records=human.store.latest_signals('local-user',0,timestamp());record=records[0]
                self.assertEqual(record['observation']['value']['vibration']['peak_hz'],125)
                self.assertEqual(record['timestamp_ms'],f.timestamp_ms)
                server=MCPServer(human)
                for _ in range(3):
                    value=await server.call_tool('get_observations',{'capability_id':spec.id})
                    self.assertIn('125.0',str(value));await asyncio.sleep(.02)
                self.assertEqual(human.store.latest_signals('local-user',0,timestamp())[0]['timestamp_ms'],f.timestamp_ms)
                await streams.submit(frame(1,config,flat=True));await streams.drain()
                value=human.store.latest_signals('local-user',0,timestamp())[0]['observation']['value']
                self.assertIsNone(value['vibration']['peak_hz'])
                self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)
            finally:await streams.clear()
        self.assertIsNotNone(child.returncode);self.assertTrue(worker.closed)

    async def test_example_independent_polling_and_join(self):
        import contextlib
        import io
        from unittest.mock import patch
        from examples.visual_microphone_context import deliver
        from humansense.mcp import MCPServer
        calls=[];original_call=MCPServer.call_tool;original_process=ProcessProvider.process
        async def call(server,*a,**kw):
            calls.append(asyncio.get_running_loop().time())
            return await original_call(server,*a,**kw)
        async def slow(provider,value):
            await asyncio.sleep(.2)
            return await original_process(provider,value)
        before=set(asyncio.all_tasks())
        with patch.object(MCPServer,'call_tool',call),patch.object(ProcessProvider,'process',slow),contextlib.redirect_stdout(io.StringIO()):
            from humansense.schema import timestamp
            result=await deliver(frame(config=dict(CONFIG,capture_origin_ms=timestamp()-200)).data,sys.executable,20)
        self.assertIn('125.0',str(result));self.assertGreaterEqual(len(calls),5)
        count=len(calls);await asyncio.sleep(.05);self.assertEqual(count,len(calls))
        self.assertFalse([t for t in asyncio.all_tasks()-before if not t.done()])

    async def test_owned_profile_timeout_and_cancel_reap_only_child(self):
        from unittest.mock import patch
        real_spawn=asyncio.create_subprocess_exec;children=[]
        code="import time;from humansense.providers.visual_microphone_context import VisualMicrophoneContext;VisualMicrophoneContext.process=lambda *a:time.sleep(60);from humansense.provider_worker import main;main()"
        async def spawn(*args,**kwargs):
            if 'humansense.provider_worker' in args:
                value=await real_spawn(sys.executable,'-u','-c',code,**kwargs)
            else:value=await real_spawn(*args,**kwargs)
            children.append(value);return value
        with patch('asyncio.create_subprocess_exec',spawn):
            worker=await ProcessProvider.open(sys.executable,'visual_microphone.phase',CONFIG,inference_timeout=.1)
            with self.assertRaises(asyncio.TimeoutError):await worker.process(frame())
            self.assertTrue(worker.closed);self.assertIsNotNone(children[-1].returncode)
            await worker.close()
            worker=await ProcessProvider.open(sys.executable,'visual_microphone.phase',CONFIG)
            task=asyncio.create_task(worker.process(frame()));await asyncio.sleep(.05);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            await worker.close();self.assertIsNotNone(children[-1].returncode)
        self.assertEqual(len(children),2)
