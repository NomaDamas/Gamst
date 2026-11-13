import dataclasses
import importlib.metadata
import json
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from humansense import HumanSense, SensorMetadata, SensorStreams
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.piv import OpenPIVFlow
from humansense.provider_profiles import capability

CONFIG = dict(meters_per_pixel=.001, calibration_id='fixture', coordinate_frame='image')


def frame(seq=0,ts=0,side=128):
    rng=np.random.default_rng(seq)
    return InputFrame(rng.integers(0,256,(side,side,3),dtype=np.uint8),'video',ts,
        sensor=SensorMetadata('test','camera',seq,'image',calibration_id='fixture',stream_epoch='e1'))


def backend(a,b,**options):
    shape=((a.shape[0]-64)//32+1,(a.shape[1]-64)//32+1)
    return np.full(shape,3/options['dt']),np.full(shape,2/options['dt']),np.full(shape,2.)


class MetadataTests(unittest.TestCase):
    def test_discovery_stays_stdlib_and_config_validates(self):
        code="from humansense.provider_profiles import capability; import sys; capability('openpiv.flow',"+repr(CONFIG)+"); assert 'numpy' not in sys.modules; assert 'openpiv' not in sys.modules"
        subprocess.run([sys.executable,'-S','-c',code],check=True)
        for change in ({'meters_per_pixel':True},{'meters_per_pixel':0},{'snr_threshold':float('nan')},{'max_gap_ms':0},{'max_speed_m_s':-1}):
            with self.assertRaises(ValueError):capability('openpiv.flow',dict(CONFIG,**change))
        self.assertNotEqual(capability('openpiv.flow',CONFIG).version,capability('openpiv.flow',dict(CONFIG,max_gap_ms=999)).version)


@unittest.skipIf(np is None,'optional numpy absent')
class PIVTests(unittest.TestCase):
    def test_pinned_loader_missing_and_wrong_version(self):
        with patch('importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError,'absent'):OpenPIVFlow.from_installed(**CONFIG)
        with patch('importlib.metadata.version',return_value='0.25'):
            with self.assertRaisesRegex(RuntimeError,'0.26.1'):OpenPIVFlow.from_installed(**CONFIG)
        package=types.ModuleType('openpiv');module=types.ModuleType('openpiv.pyprocess');module.extended_search_area_piv=backend
        with patch('importlib.metadata.version',return_value='0.26.1'),patch.dict(sys.modules,{'openpiv':package,'openpiv.pyprocess':module}):
            p=OpenPIVFlow.from_installed(**CONFIG)
            self.assertIs(p.backend,backend);p.close()

    def test_units_directions_pair_and_owned_inputs(self):
        calls=[]
        def run(a,b,**kw):
            calls.append((a.copy(),b.copy(),kw))
            result=backend(a,b,**kw);a[:]=0;b[:]=0
            return result
        p=OpenPIVFlow(run,**CONFIG);first=frame()
        self.assertEqual(p.process(first)['flow']['status'],'warmup')
        first.data[:]=0
        value=p.process(frame(1,100))['flow']
        self.assertTrue(np.any(calls[0][0]))
        self.assertEqual(calls[0][2]['backend'],'scipy')
        self.assertEqual(calls[0][2]['correlation_method'],'linear')
        self.assertEqual(calls[0][2]['dt'],.1)
        self.assertEqual(value['vectors'][0]['velocity_m_s'],[.03,.02])
        self.assertEqual(value['vectors'][0]['center_m'],[.032,.032])
        self.assertEqual(value['valid_vectors'],9)
        self.assertTrue(np.any(p._previous))
        value['vectors'][0]['velocity_m_s'][0]=999
        self.assertEqual(p.process(frame(2,300))['flow']['vectors'][0]['velocity_m_s'],[.015,.01])

    def test_invalid_vectors_are_null_not_filled(self):
        def run(a,b,**kw):
            u,v,snr=backend(a,b,**kw)
            u[0,0]=float('nan');v[0,1]=float('inf');snr[0,2]=.5
            u[1,0]=1e9;snr[1,1]=float('inf')
            return u,v,snr
        p=OpenPIVFlow(run,**CONFIG);p.process(frame())
        value=p.process(frame(1,100))['flow']
        self.assertEqual(value['rejected_vectors'],5)
        self.assertEqual(value['valid_vectors'],4)
        self.assertIsNone(value['vectors'][0]['velocity_m_s'])
        self.assertIsNone(value['vectors'][4]['snr'])
        json.dumps(value,allow_nan=False)

    def test_blank_pair_does_not_call_backend(self):
        p=OpenPIVFlow(lambda *a,**k:self.fail('blank backend call'),**CONFIG)
        f=frame();f.data[:]=0;p.process(f)
        result=p.process(frame(1,100))['flow']
        self.assertEqual(result['status'],'insufficient_texture')
        self.assertEqual(result['vectors'],[])
        self.assertEqual(result['interval_ms'],[0,100])

    def test_discontinuity_and_invalid_input_clear_history(self):
        p=OpenPIVFlow(backend,**CONFIG)
        for nextframe in (frame(2,100),frame(1,1001),frame(1,100,256),
                          dataclasses.replace(frame(1,100),subject_id='other')):
            p.reset();p.process(frame());self.assertEqual(p.process(nextframe)['flow']['status'],'warmup')
        for bad in (frame(0,100),frame(1,0),frame(1,100,32),
                    dataclasses.replace(frame(1,100),sensor=None),
                    dataclasses.replace(frame(1,100),timestamp_ms=True)):
            p.reset();p.process(frame())
            with self.assertRaises(ValueError):p.process(bad)
            self.assertEqual(p.process(frame(2,200))['flow']['status'],'warmup')
        f=frame(1,100)
        for changes in ({'stream_epoch':'e2'},{'source_id':'new'},{'sensor_id':'new'},{'capture_sequence':9}):
            p.reset();p.process(frame())
            self.assertEqual(p.process(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,**changes)))['flow']['status'],'warmup')
        p.close();self.assertIsNone(p._previous);p.close()
        with self.assertRaises(RuntimeError):p.process(frame())

    def test_bad_native_output_clears_and_grid_bound(self):
        p=OpenPIVFlow(lambda *a,**kw: (np.zeros((1,1)),)*3,**CONFIG)
        p.process(frame())
        with self.assertRaises(ValueError):p.process(frame(1,100))
        self.assertIsNone(p._previous)
        p=OpenPIVFlow(backend,**CONFIG);p.process(frame(0,0,256))
        result=p.process(frame(1,100,256))
        self.assertEqual(len(result['flow']['vectors']),49)
        self.assertLess(len(json.dumps(result)),65536)


@unittest.skipIf(np is None,'optional numpy absent')
class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_stream_and_mcp(self):
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human)
            spec=capability('openpiv.flow',CONFIG)
            streams.register(spec,lambda:OpenPIVFlow(backend,**CONFIG))
            await streams.open('test','camera',[spec.id])
            now=int(time.time()*1000)
            for seq,ts in ((0,now-100),(1,now)):
                f=frame(seq,ts)
                await streams.submit(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,stream_epoch=None)))
                await streams.drain()
            context=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
            self.assertIn('velocity_m_s',json.dumps(context))
            value=next(iter((await human.get_context())['observations'].values()))['value']
            self.assertEqual(value['flow']['vectors'][0]['velocity_m_s'],[.03,.02])
            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)

    async def test_worker_profile_dispatch_and_real_pipe(self):
        import asyncio
        from humansense import ProcessProvider
        children=[];spawn_real=asyncio.create_subprocess_exec
        script='''
import numpy as np
from humansense.providers.piv import OpenPIVFlow
from humansense.provider_worker import main

def backend(a,b,**kw):
    shape=((a.shape[0]-64)//32+1,(a.shape[1]-64)//32+1)
    return np.full(shape,3/kw['dt']),np.full(shape,2/kw['dt']),np.full(shape,2.)
OpenPIVFlow.from_installed=classmethod(lambda cls,**config:cls(backend,**config))
main()
'''
        async def spawn(*args,**kwargs):
            child=await spawn_real(sys.executable,'-u','-c',script,**kwargs)
            children.append(child);return child
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                worker=await ProcessProvider.open(sys.executable,'openpiv.flow',CONFIG)
                try:
                    self.assertEqual((await worker.process(frame()))['flow']['status'],'warmup')
                    result=await worker.process(frame(1,100))
                    self.assertEqual(result['flow']['vectors'][0]['velocity_m_s'],[.03,.02])
                    await worker.reset()
                    self.assertEqual((await worker.process(frame(2,200)))['flow']['status'],'warmup')
                finally:await worker.close()
            self.assertTrue(all(child.returncode is not None for child in children))
        finally:
            for child in children:
                if child.returncode is None:
                    child.kill();await child.wait()
