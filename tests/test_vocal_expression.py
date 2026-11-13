import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np=None

from humansense import HumanSense, SensorMetadata, SensorStreams
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.vocal_expression import Emotion2VecContext, LABELS, SOURCE_SHA256
from humansense.provider_profiles import capability

CONFIG=dict(model_sha256='0'*64,head_sha256='1'*64,window_ms=1000,hop_ms=500)


def frame(seq=0,ts=1000,n=16000,value=.1):
    return InputFrame(np.full(n,value,np.float32),'audio',ts,sample_rate=16000,
        sensor=SensorMetadata('test','mic',seq,channels=('mono',),stream_epoch='e1'))


def predictor(samples):return np.ones(9)/9


class MetadataTests(unittest.TestCase):
    def test_discovery_and_config_bounds(self):
        code="from humansense.providers.vocal_expression import Emotion2VecContext; import sys; Emotion2VecContext.capability(**"+repr(CONFIG)+"); assert 'numpy' not in sys.modules; assert 'onnxruntime' not in sys.modules"
        subprocess.run([sys.executable,'-S','-c',code],check=True)
        for change in ({'window_ms':999},{'window_ms':4001},{'hop_ms':99},{'model_sha256':'a'},{'window_ms':True}):
            with self.assertRaises(ValueError):Emotion2VecContext.capability(**dict(CONFIG,**change))
        capability('emotion2vec.seed.onnx',dict(CONFIG,model_path='/local/model',head_path='/local/head'))


@unittest.skipIf(np is None,'optional numpy absent')
class WindowTests(unittest.TestCase):
    def test_window_hop_ownership_and_silence(self):
        seen=[]
        def run(x):seen.append(x.copy());x[:]=0;return predictor(x)
        p=Emotion2VecContext(run,**CONFIG)
        self.assertIsNone(p.process(frame(0,500,8000)))
        result=p.process(frame(1,1000,8000))['classification']
        self.assertEqual((result['start_ms'],result['end_ms']),(0,1000))
        self.assertEqual([v['label'] for v in result['scores']],list(LABELS))
        self.assertTrue(np.all(p._buffer>.09))
        self.assertIsNone(p.process(frame(2,1250,4000)))
        self.assertIsNotNone(p.process(frame(3,1500,4000)))
        self.assertEqual(len(seen),2)
        result=p.process(frame(4,2500,16000,0))['classification']
        self.assertEqual(result['status'],'insufficient_energy');self.assertEqual(result['scores'],[])
        self.assertEqual(len(seen),2)

    def test_gap_reset_clock_and_channel_ownership(self):
        p=Emotion2VecContext(predictor,**CONFIG)
        for bad_next in (frame(2,1000,8000),frame(1,1001,8000),
                         dataclasses.replace(frame(1,1000,8000),subject_id='new')):
            p.reset();p.process(frame(0,500,8000));self.assertIsNone(p.process(bad_next))
        p.reset();p.process(frame())
        p.reset();self.assertIsNone(p.process(frame(0,500,8000)))
        self.assertIsNotNone(p.process(frame(1,1000,8000)))
        f=frame(2,1500,8000)
        self.assertIsNone(p.process(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,channels=('other',)))))

    def test_invalid_input_output_clear_and_close(self):
        p=Emotion2VecContext(predictor,**CONFIG)
        for bad in (dataclasses.replace(frame(),sample_rate=8000),frame(n=15),frame(value=float('nan')),
                    frame(value=2),dataclasses.replace(frame(),data=np.ma.array(frame().data,mask=True))):
            p.process(frame())
            with self.assertRaises(ValueError):p.process(bad)
            self.assertEqual(len(p._buffer),0)
        p=Emotion2VecContext(lambda x:np.full(9,float('nan')),**CONFIG)
        with self.assertRaises(ValueError):p.process(frame())
        self.assertEqual(len(p._buffer),0)
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame())

    def test_pcm16_rails_and_max_window_bound(self):
        p=Emotion2VecContext(predictor,**dict(CONFIG,window_ms=4000))
        for seq in range(20):
            values=np.full(16000,32767 if seq%2 else -32768,np.int16)
            result=p.process(dataclasses.replace(frame(seq,(seq+1)*1000),data=values.tobytes()))
            self.assertLessEqual(p._buffer.nbytes,256000)
        self.assertTrue(result['classification']['last_block_clipped'])
        self.assertEqual(result['classification']['start_ms'],16000)

    def test_local_hash_contract_and_linear_head(self):
        module=types.ModuleType('onnxruntime')
        class Session:
            def __init__(self,blob,**kw):
                self.asserted=kw['providers']==['CPUExecutionProvider'] and blob==b'fixture'
            def get_inputs(self):return [types.SimpleNamespace(name='input',type='tensor(float)',shape=['batch_size','sequence_length'])]
            def get_outputs(self):return [types.SimpleNamespace(name='output',type='tensor(float)',shape=['batch_size','sequence_length',768])]
            def get_modelmeta(self):return types.SimpleNamespace(custom_metadata_map={'gamst.source_checkpoint_sha256':SOURCE_SHA256,'gamst.export_contract':'emotion2vec_seed_features_v1'})
            def run(self,names,feed):
                self_shape=feed['input'].shape
                assert self.asserted and self_shape==(1,16000)
                return [np.stack([np.full((768,),2,np.float32),np.full((768,),4,np.float32)])[None,:,:]]
        module.SessionOptions=types.SimpleNamespace;module.InferenceSession=Session
        head=dict(source_checkpoint_sha256=SOURCE_SHA256,labels=list(LABELS),weight=np.zeros((9,768)).tolist(),bias=list(range(9)))
        head['weight'][0][0]=2
        with tempfile.TemporaryDirectory(dir='.') as directory:
            m=Path(directory)/'model.onnx';h=Path(directory)/'head.json';m.write_bytes(b'fixture');h.write_text(json.dumps(head))
            config=dict(CONFIG,model_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),head_sha256=hashlib.sha256(h.read_bytes()).hexdigest())
            with patch.dict(sys.modules,{'onnxruntime':module}),patch('importlib.metadata.version',return_value='1.23.2'):
                p=Emotion2VecContext.from_local(m,h,**config)
                scores=p.process(frame())['classification']['scores']
                logits=np.arange(9);logits[0]=6
                expected=np.exp(logits-8);expected/=expected.sum()
                np.testing.assert_allclose([x['score'] for x in scores],expected)
                p.close()
                with self.assertRaises(ValueError):Emotion2VecContext.from_local(m,h,**dict(config,head_sha256='f'*64))
                with patch.object(Session,'get_modelmeta',return_value=types.SimpleNamespace(custom_metadata_map={})):
                    with self.assertRaises(ValueError):Emotion2VecContext.from_local(m,h,**config)


@unittest.skipIf(np is None,'optional numpy absent')
class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_window_context(self):
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);spec=Emotion2VecContext.capability(**CONFIG)
            streams.register(spec,lambda:Emotion2VecContext(predictor,**CONFIG))
            await streams.open('test','mic',[spec.id]);now=int(time.time()*1000)
            f=frame(0,now)
            await streams.submit(dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,stream_epoch=None)))
            await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
            self.assertIn('inner_emotion_verified',json.dumps(result))
            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)

    async def test_worker_profile_partial_window_reset_and_close(self):
        import asyncio
        from humansense import ProcessProvider
        children=[];spawn_real=asyncio.create_subprocess_exec
        script='''
import numpy as np
from humansense.providers.vocal_expression import Emotion2VecContext
from humansense.provider_worker import main
Emotion2VecContext.from_local=classmethod(lambda cls,model_path,head_path,**config:cls(lambda x:np.ones(9)/9,**config))
main()
'''
        async def spawn(*args,**kwargs):
            child=await spawn_real(sys.executable,'-u','-c',script,**kwargs)
            children.append(child);return child
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                worker=await ProcessProvider.open(sys.executable,'emotion2vec.seed.onnx',dict(CONFIG,model_path='/fixture',head_path='/fixture'))
                try:
                    self.assertIsNone(await worker.process(frame(0,500,8000)))
                    result=await worker.process(frame(1,1000,8000))
                    self.assertEqual(len(result['classification']['scores']),9)
                    self.assertFalse(result['classification']['export_parity_verified'])
                    await worker.reset()
                    self.assertIsNone(await worker.process(frame(2,1500,8000)))
                finally:await worker.close()
            self.assertTrue(all(child.returncode is not None for child in children))
        finally:
            for child in children:
                if child.returncode is None:child.kill();await child.wait()
