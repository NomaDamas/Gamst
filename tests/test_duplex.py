from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_profiles import capability
from humansense.providers.duplex import WebRTCDuplex
try:
    import numpy as np
except ImportError:np=None


def frame(seq=0,end=1000,rate=16000,reference=True):
    data=np.zeros((rate//100,2 if reference else 1),dtype=np.float32)
    data[:,0]=.5
    if reference:data[:,1]=.25
    return InputFrame(data,'audio',end,sample_rate=rate,sensor=SensorMetadata('duplex','paired',seq,
        channels=('microphone','render') if reference else ('microphone',),calibration_id='declared-pair'))


class Metadata(unittest.TestCase):
    def test_discovery_and_runtime_pins(self):
        spec=capability('webrtc.aec3',{})
        self.assertTrue(spec.stateful);self.assertEqual(spec.window_ms,10)
        self.assertNotEqual(spec.version,capability('webrtc.aec3',dict(stream_delay_ms=40)).version)
        for config in (dict(sample_rate=8000),dict(stream_delay_ms=-1),dict(stream_delay_ms=True),
                       dict(speech_sensor_id='ambient'),dict(other=1)):
            with self.assertRaises(ValueError):capability('webrtc.aec3',config)
        with patch('importlib.metadata.version',return_value='wrong'):
            with self.assertRaises(RuntimeError):WebRTCDuplex.from_installed()


@unittest.skipIf(np is None,'optional numpy absent')
class Duplex(unittest.TestCase):
    def provider(self,**options):
        created=[]
        class Backend:
            closed=False
            def process(self,near,far):
                self.near=near.copy();self.far=far.copy()
                result=near-far;near[:]=0;far[:]=0
                return result
            def close(self):self.closed=True
        def factory():
            b=Backend();created.append(b);return b
        return WebRTCDuplex(factory,**options),created

    def test_owned_ambient_and_cleaned_lane_all_native_rates(self):
        for rate in (16000,32000,48000):
            p,created=self.provider(sample_rate=rate);f=frame(rate=rate);before=f.data.copy()
            out=p.transform(f)
            np.testing.assert_array_equal(out.ambient.data,np.full(rate//100,.5,dtype=np.float32))
            np.testing.assert_array_equal(out.speech.data,np.full(rate//100,.25,dtype=np.float32))
            np.testing.assert_array_equal(f.data,before)
            self.assertEqual(out.ambient.sensor.sensor_id,'ambient');self.assertEqual(out.speech.sensor.sensor_id,'speech')
            self.assertFalse(out.context['processing']['echo_reduction_verified'])
            self.assertTrue(out.context['processing']['high_pass_filter_applied'])
            self.assertEqual(created[0].near.shape,(rate//100,));self.assertTrue(created[0].near.flags.c_contiguous)
            p.close();p.close();self.assertTrue(created[0].closed)
            with self.assertRaises(RuntimeError):p.transform(f)

    def test_missing_reference_distinct_from_explicit_silence(self):
        p,created=self.provider();out=p.transform(frame(reference=False))
        self.assertIsNone(out.speech);self.assertEqual(created,[])
        self.assertEqual(out.context['processing']['status'],'unavailable_reference')
        f=frame(1,1010);f.data[:,1]=0
        out=p.transform(f);self.assertTrue(out.context['processing']['reference_silent']);self.assertIsNotNone(out.speech)
        out=p.transform(frame(2,1020,reference=False));self.assertTrue(created[0].closed);self.assertIsNone(out.speech)
        p.transform(frame(3,1030));self.assertEqual(len(created),2)

    def test_discontinuities_and_bad_inputs_reset_native_state(self):
        for kind in ('sequence','clock','subject','epoch','calibration'):
            p,created=self.provider();p.transform(frame());f=frame(1,1010)
            if kind=='sequence':f=replace(f,sensor=replace(f.sensor,sequence=4))
            if kind=='clock':f=replace(f,timestamp_ms=1012)
            if kind=='subject':f=replace(f,subject_id='other')
            if kind=='epoch':f=replace(f,sensor=replace(f.sensor,stream_epoch='other'))
            if kind=='calibration':f=replace(f,sensor=replace(f.sensor,calibration_id='other'))
            p.transform(f);self.assertTrue(created[0].closed);self.assertEqual(len(created),2)
        p,created=self.provider();p.transform(frame())
        with self.assertRaises(ValueError):p.transform(frame())
        self.assertTrue(created[0].closed);self.assertIsNone(p.backend)
        for f in (replace(frame(),data=np.zeros((159,2),dtype=np.float32)),
                  replace(frame(),data=np.full((160,2),np.nan,dtype=np.float32)),
                  replace(frame(),data=np.ones((160,2),dtype=np.float64)),
                  replace(frame(),sensor=replace(frame().sensor,calibration_id=None)),
                  replace(frame(),sensor=replace(frame().sensor,channels=('render','microphone')))):
            with self.assertRaises(ValueError):p.transform(f)

    def test_int16_normalization_output_clip_and_invalid_native(self):
        p,created=self.provider();f=frame();data=np.zeros((160,2),dtype=np.int16);data[:,0]=-32768
        out=p.transform(replace(f,data=data));self.assertTrue(np.all(out.ambient.data==-1))
        self.assertTrue(out.context['processing']['input_clipping_present'])
        data[:,0]=32767
        out=p.transform(replace(frame(1,1010),data=data))
        self.assertTrue(out.context['processing']['input_clipping_present'])
        self.assertTrue(np.all(out.ambient.data==32767/32768))
        class Backend:
            def process(self,near,far):return np.full(160,1.25,dtype=np.float32)
        p=WebRTCDuplex(Backend);out=p.transform(frame());self.assertTrue(np.all(out.speech.data==1))
        self.assertEqual(out.context['processing']['output_samples_clipped'],160)
        self.assertEqual(out.context['processing']['speech_peak_before_clip'],1.25)
        for output in (np.zeros(159,dtype=np.float32),np.full(160,np.inf,dtype=np.float32),np.ones(160),np.full(160,17,dtype=np.float32)):
            backend=SimpleNamespace(process=lambda a,b:output)
            p=WebRTCDuplex(lambda:backend)
            with self.assertRaises(ValueError):p.transform(frame())
            self.assertIsNone(p.backend)

    def test_loader_is_lazy_and_uses_echo_canceller_only(self):
        calls=[]
        def constructor(**kwargs):calls.append(kwargs);return SimpleNamespace(process=lambda near,far:near)
        with patch('importlib.metadata.version',return_value='0.2.0'),patch.dict('sys.modules',{'pywebrtc_audio':SimpleNamespace(EchoCanceller=constructor)}):
            p=WebRTCDuplex.from_installed(sample_rate=32000,stream_delay_ms=40)
            self.assertEqual(calls,[]);p.transform(frame(rate=32000))
        self.assertEqual(calls,[dict(sample_rate=32000,num_channels=1,stream_delay_ms=40)])
        p.reset();self.assertIsNone(p.backend);p.close()

    def test_close_serializes_native_work(self):
        import threading
        entered,release,closed=threading.Event(),threading.Event(),threading.Event();errors=[]
        class Backend:
            def process(self,near,far):
                entered.set()
                if not release.wait(5):raise RuntimeError('gate timeout')
                return near
        p=WebRTCDuplex(Backend)
        def work():
            try:p.transform(frame())
            except BaseException as exc:errors.append(exc)
        worker=threading.Thread(target=work);closer=threading.Thread(target=lambda:(p.close(),closed.set()))
        worker.start()
        try:
            self.assertTrue(entered.wait(5));closer.start();self.assertFalse(closed.wait(.05))
        finally:
            release.set();worker.join(5)
            if closer.ident is not None:closer.join(5)
        self.assertEqual(errors,[]);self.assertTrue(closed.is_set())


@unittest.skipIf(np is None,'optional numpy absent')
class Context(unittest.IsolatedAsyncioTestCase):
    async def test_ambient_and_speech_feed_separate_prosody_and_mcp(self):
        from humansense import HumanSense,SensorStreams
        from humansense.mcp import MCPServer
        from humansense.providers.prosody import AudioProsody
        from humansense.schema import timestamp
        p=WebRTCDuplex(lambda:SimpleNamespace(process=lambda near,far:near-far))
        ambient,speech=AudioProsody(window_ms=80),AudioProsody(window_ms=80)
        start=timestamp()-100;ambient_result=speech_result=None
        async with HumanSense(consent=True) as human:
            human.register_capability(p.spec)
            for seq in range(8):
                f=frame(seq,start+10*seq);out=p.transform(f)
                ambient_result=ambient.process(out.ambient) or ambient_result
                speech_result=speech.process(out.speech) or speech_result
                await human.push_observation(p.spec.id,out.context,sensor=f.sensor,timestamp_ms=f.timestamp_ms)
            self.assertIsNotNone(ambient_result);self.assertIsNotNone(speech_result)
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            processing=result['items'][0]['observation']['value']['processing']
            self.assertEqual(processing['input_rms'],.5);self.assertEqual(processing['speech_rms'],.25)
            self.assertNotIn('data',processing);self.assertFalse(processing['recognition_improvement_verified'])
            p.close();ambient.close();speech.close()

    async def test_managed_context_reference_gap_and_clear(self):
        from humansense import HumanSense,SensorStreams
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        p=WebRTCDuplex(lambda:SimpleNamespace(process=lambda near,far:near-far))
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);streams.register(p.spec,lambda:p)
            await streams.open('duplex','paired',[p.spec.id]);now=timestamp()-100
            await streams.submit(frame(0,now));await streams.drain()
            await streams.submit(frame(10,now+10,reference=False));await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            self.assertEqual(result['items'][0]['observation']['value']['processing']['status'],'unavailable_reference')
            await streams.clear();self.assertTrue(p.closed)


@unittest.skipIf(np is None,'optional numpy absent')
class Smoke(unittest.TestCase):
    def test_paired_wave_runner_preservation_and_partial_block_rejection(self):
        import importlib.util,json,tempfile,wave
        import importlib.metadata
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('duplex_smoke',Path(__file__).resolve().parents[1]/'examples/validate_duplex.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original_version=importlib.metadata.version
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve()
            def write(name,value,count=480):
                path=root/name
                with wave.open(str(path),'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
                    f.writeframes(np.full(count,value,dtype='<i2').tobytes())
                return str(path)
            args=SimpleNamespace(microphone=write('mic.wav',16384),render=write('render.wav',8192),
                output=str(root/'report.json'),delay_ms=40)
            p=WebRTCDuplex(lambda:SimpleNamespace(process=lambda near,far:near-far),stream_delay_ms=40)
            with patch.object(module.WebRTCDuplex,'from_installed',return_value=p),patch('importlib.metadata.version',
                    side_effect=lambda name:'0.2.0' if name=='pywebrtc-audio' else original_version(name)):
                module.run(args)
            report=json.loads(Path(args.output).read_text())
            self.assertEqual(report['blocks'],3);self.assertTrue(report['ambient_preserved']);self.assertTrue(p.closed)
            self.assertFalse(report['echo_reduction_verified']);self.assertEqual(report['output_samples_clipped'],0)
            self.assertEqual(report['last_context']['processing']['speech_rms'],.25)
            partial=write('partial.wav',0,161)
            with self.assertRaises(ValueError):module.read_pcm(partial)
