import asyncio
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorStreams, TalkNetWindows, talknet_stream_frame
from humansense.mcp import MCPServer
from humansense.providers import talknet
from humansense.providers.talknet import TalkNetContext, mfcc
from test_audiovisual import audio, video

try:
    import numpy as np
    import scipy
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


def predictor(features, faces):
    assert features.shape == (len(faces)*4, 13)
    assert features.dtype == np.float32
    return np.tile(np.array([1., 2.], np.float32), (len(faces), 1))


def packet(join=None, offset=0):
    join = join or TalkNetWindows(clock_id='clock')
    for i in range(25):
        join.push_face(video(i, offset+i*40), track_id='f', clock_id='clock')
    join.push_audio(audio(0, offset, 1000), clock_id='clock')
    return join.push_audio(audio(1, offset+1000), clock_id='clock')[0]


class Discovery(unittest.TestCase):
    def test_profile_dependency_light_and_configuration(self):
        code="from humansense import ProcessProvider; import sys; ProcessProvider.capability('talknet.local', {'source_dir':'/x','checkpoint_path':'/x','checkpoint_sha256':'a'*64}); assert not {'torch','numpy','scipy'} & sys.modules.keys()"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for options in ({'window_frames':True},{'device':'auto'},{'checkpoint_sha256':'bad'}):
            config={'checkpoint_sha256':'a'*64};config.update(options)
            with self.assertRaises(ValueError):TalkNetContext.capability(**config)


@unittest.skipIf(np is None, 'optional NumPy/SciPy absent')
class Numeric(unittest.TestCase):
    def test_mfcc_pinned_reference_golden_and_bounds(self):
        fixture=json.loads(Path('tests/fixtures/talknet-mfcc.json').read_text())
        for kind, expected in fixture['cases'].items():
            size=16240
            if kind=='silence': pcm=np.zeros(size,np.int16)
            elif kind=='constant': pcm=np.full(size,1234,np.int16)
            elif kind=='ramp': pcm=((np.arange(size,dtype=np.int64)*113)%65536-32768).astype(np.int16)
            elif kind=='rails': pcm=np.resize(np.array([-32768,32767],np.int16),size)
            else:
                pcm=np.zeros(size,np.int16);pcm[0]=32767;pcm[399]=-32768;pcm[-1]=12345
            np.testing.assert_allclose(mfcc(pcm)[fixture['rows']],expected,rtol=2e-6,atol=2e-5)
        for bad in (np.zeros(16000,np.int16),np.zeros(16240,np.float32),np.zeros((16240,1),np.int16)):
            with self.assertRaises(ValueError):mfcc(bad)

    def test_face_crop_geometry_border_and_owned_pixels(self):
        try: import cv2
        except ImportError: self.skipTest('optional OpenCV absent')
        bgr=np.arange(128*160*3,dtype=np.uint8).reshape(128,160,3)
        # Source geometry for xyxy[0,0,40,50]: size25, center20/25,
        # context rectangle x[-15,55), y[0,70), padding value110.
        expected=np.full((70,70,3),110,np.uint8)
        expected[:,15:]=bgr[:70,:55]
        expected=cv2.cvtColor(cv2.resize(expected,(224,224)),cv2.COLOR_BGR2GRAY)[56:168,56:168]
        actual=talknet.prepare_talknet_face(bgr,[0,0,40,50])
        np.testing.assert_array_equal(actual,expected)
        bgr[:]=0
        np.testing.assert_array_equal(actual,expected)
        for box in ([0,0,4,5],[-1,0,40,50],[0,0,161,50],[0,0,float('nan'),50]):
            with self.assertRaises(ValueError):talknet.prepare_talknet_face(bgr,box)

    def test_scores_provenance_capture_order_and_closure(self):
        provider=TalkNetContext(predictor,checkpoint_sha256='a'*64)
        frame=packet()
        value=provider.process(frame)['association']
        self.assertEqual(len(value['scores']),25)
        self.assertEqual(value['scores'][0]['speaking_logit'],2.)
        self.assertAlmostEqual(value['scores'][0]['speaking_probability'],1/(1+np.exp(-1)))
        self.assertEqual(value['scores'][-1]['end_ms'],1000)
        self.assertFalse(value['artifact_hashes_verified'])
        self.assertFalse(value['synchronization_verified'])
        self.assertIsNone(value['diarization_slot'])
        with self.assertRaises(ValueError):provider.process(frame)
        for bad in (dataclasses.replace(frame,timestamp_ms=1016),dataclasses.replace(frame,subject_id='other'),
                    dataclasses.replace(frame,sample_rate=16000.),dataclasses.replace(frame,data=frame.data[:-1])):
            with self.assertRaises(ValueError):provider.process(bad)
        provider.predictor=lambda a,v:np.full((25,2),np.nan,np.float32)
        with self.assertRaises(ValueError):provider.process(frame)
        provider.close();provider.reset()
        with self.assertRaises(RuntimeError):provider.process(frame)

    def test_stream_binding_retains_capture_session_and_lifetime_sequence(self):
        join=TalkNetWindows(clock_id='clock')
        one=packet(join)
        bound=talknet_stream_frame(one)
        self.assertIsNone(bound.sensor.stream_epoch)
        self.assertEqual(bound.data,one.data)
        join.reset()
        two=packet(join,2000)
        self.assertGreater(two.sensor.sequence,one.sensor.sequence)
        self.assertNotEqual(two.sensor.stream_epoch,one.sensor.stream_epoch)
        with self.assertRaises(ValueError):talknet_stream_frame(dataclasses.replace(one,timestamp_ms=999))


@unittest.skipIf(torch is None or np is None, 'optional Torch/NumPy/SciPy absent')
class Loader(unittest.TestCase):
    def test_verified_source_namespace_and_complete_checkpoint(self):
        source=b'''import torch
class talkNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2))
    def forward_audio_frontend(self,x):return x
    def forward_visual_frontend(self,x):return x
    def forward_cross_attention(self,a,v):return a,v
    def forward_audio_visual_backend(self,a,v):return torch.zeros(v.shape[1],256)
'''
        state={'model.weight':torch.ones(2),'lossAV.FC.weight':torch.zeros(2,256),'lossAV.FC.bias':torch.tensor([.25,-.75])}
        for key in ('lossA','lossV'):
            state[key+'.FC.weight']=torch.zeros(2,128);state[key+'.FC.bias']=torch.zeros(2)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory);(root/'talkNetModel.py').write_bytes(source)
            def save():
                blob=io.BytesIO();torch.save(state,blob);payload=blob.getvalue()
                (root/'weights.pt').write_bytes(payload)
                return hashlib.sha256(payload).hexdigest()
            with patch.object(talknet,'SOURCES',{'talkNetModel.py':(len(source),hashlib.sha256(source).hexdigest())}),patch.object(talknet,'version',return_value='2.8.0'):
                selected=TalkNetContext.from_local(root,root/'weights.pt',checkpoint_sha256=save())
                self.assertTrue(selected.artifact_hashes_verified)
                value=selected.process(packet())['association']['scores'][0]
                self.assertEqual(value['speaking_logit'],-.75)
                self.assertFalse(any(k.startswith('_gamst_talknet_') for k in sys.modules))
                selected.close()
                del state['model.weight']
                with self.assertRaisesRegex(ValueError,'every exact'):
                    TalkNetContext.from_local(root,root/'weights.pt',checkpoint_sha256=save())
                state['model.weight']=torch.ones(2,dtype=torch.float64)
                with self.assertRaisesRegex(ValueError,'dtype'):
                    TalkNetContext.from_local(root,root/'weights.pt',checkpoint_sha256=save())
                (root/'talkNetModel.py').write_bytes(b'not verified')
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    TalkNetContext.from_local(root,root/'weights.pt',checkpoint_sha256=save())


@unittest.skipIf(np is None, 'optional NumPy/SciPy absent')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_actual_worker_stream_and_mcp_preserve_window_after_reset(self):
        children=[];real_spawn=asyncio.create_subprocess_exec
        script='''import sys
sys.path.insert(0,'tests')
from test_talknet import predictor
from humansense.providers.talknet import TalkNetContext
from humansense.provider_worker import main
TalkNetContext.from_local=classmethod(lambda cls,source_dir,checkpoint_path,**kw:cls(predictor,**kw))
main()
'''
        async def spawn(*args,**kw):
            child=await real_spawn(sys.executable,'-u','-c',script,**kw);children.append(child);return child
        config={'source_dir':'/fixture','checkpoint_path':'/fixture','checkpoint_sha256':'a'*64}
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                async with HumanSense(consent=True) as human:
                    streams=SensorStreams(human)
                    spec=ProcessProvider.capability('talknet.local',config)
                    async def create():return await ProcessProvider.open(sys.executable,'talknet.local',config)
                    streams.register(spec,create)
                    await streams.open('audiovisual','face-audio',[spec.id])
                    join=TalkNetWindows(clock_id='clock')
                    start=int(time.time()*1000)-3000
                    for offset in (start,start+1500):
                        await streams.submit(talknet_stream_frame(packet(join,offset)))
                        await streams.drain()
                        result=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                        self.assertTrue(result['items'])
                        values=[v['observation']['value']['association'] for v in result['items']]
                        self.assertTrue(any(v['window']['start_ms']==offset for v in values))
                        join.reset()
                    await streams.finish('audiovisual','face-audio')
            self.assertTrue(children and all(c.returncode is not None for c in children))
        finally:
            for child in children:
                if child.returncode is None:child.kill();await child.wait()


if __name__=='__main__':unittest.main()
