import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_profiles import capability
from humansense.providers.visual_retrieval import DINOv3Retrieval
try:
    import numpy as np
except ImportError:
    np=None


def frame(sequence=0,ts=1000,subject='person',epoch='one',data=None):
    if data is None:
        data=np.zeros((40,60,3),dtype=np.uint8); data[:,:,2]=100
    return InputFrame(data,'video',ts,subject_id=subject,sensor=SensorMetadata('cam','sensor',sequence,stream_epoch=epoch))


class Metadata(unittest.TestCase):
    def test_profile_and_preimport_integrity(self):
        config=dict(path='/not-installed',sha256='0'*64)
        spec=capability('dinov3.retrieval',config)
        self.assertTrue(spec.stateful); self.assertFalse(spec.reset_on_gap)
        self.assertNotEqual(spec.version,capability('dinov3.retrieval',dict(config,capacity=2)).version)
        for extra in (dict(sha256='bad'),dict(path=2),dict(capacity=True),dict(retention_ms=0),dict(top_k=21),dict(other=1)):
            with self.assertRaises(ValueError): capability('dinov3.retrieval',dict(config,**extra))
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'weights'; path.write_bytes(b'fixture')
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): DINOv3Retrieval.from_local(str(path),'0'*64)
                version.assert_not_called()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): DINOv3Retrieval.from_local(str(path),hashlib.sha256(b'fixture').hexdigest())


@unittest.skipIf(np is None,'optional numpy absent')
class Retrieval(unittest.TestCase):
    def vector(self,index=0):
        value=np.zeros(384); value[index]=1
        return value

    def test_similarity_order_negative_scores_and_owned_rgb(self):
        vectors=iter([self.vector(),self.vector(1),-self.vector(),self.vector()]); seen=[]
        def predictor(rgb): seen.append(rgb); return next(vectors)
        p=DINOv3Retrieval(predictor,sha256='0'*64)
        original=frame(); first=p.process(original)
        self.assertEqual(first['candidates'],[])
        self.assertEqual(seen[0][0,0].tolist(),[100,0,0]); seen[0][:]=0
        self.assertEqual(original.data[0,0,2],100)
        second=p.process(frame(1,1001)); third=p.process(frame(2,1002)); final=p.process(frame(3,1003))
        self.assertEqual([c['cosine_similarity'] for c in final['candidates']],[1,0,-1])
        self.assertEqual(final['candidates'][0]['reference_id'],first['summary']['reference_id'])
        self.assertEqual(final['candidates'][1]['reference_id'],second['summary']['reference_id'])
        self.assertEqual(final['candidates'][2]['reference_id'],third['summary']['reference_id'])
        self.assertTrue(all(c['current_location']=='unknown' and not c['identity_confirmed'] for c in final['candidates']))
        self.assertNotIn('embedding',str(final))

    def test_capacity_expiry_monotonic_and_capture_time(self):
        p=DINOv3Retrieval(lambda _:self.vector(),sha256='0'*64,capacity=2,retention_ms=10)
        with patch('humansense.providers.visual_retrieval.time.monotonic',return_value=10):
            a=p.process(frame()); b=p.process(frame(1,1001)); c=p.process(frame(2,1002))
            self.assertTrue(c['summary']['capacity_evicted']); self.assertEqual(len(p._entries),2)
            result=p.process(frame(3,1013)); self.assertEqual(result['candidates'],[])
        with patch('humansense.providers.visual_retrieval.time.monotonic',return_value=11):
            self.assertEqual(p.process(frame(4,1014))['candidates'],[])

    def test_owner_reset_explicit_clear_and_sequence_gaps(self):
        p=DINOv3Retrieval(lambda _:self.vector(),sha256='0'*64)
        p.process(frame()); self.assertEqual(len(p.process(frame(10,1010))['candidates']),1)
        for f in (frame(10,1011),frame(11,1010)):
            with self.assertRaises(ValueError): p.process(f)
        changed=p.process(frame(0,1000,subject='other'))
        self.assertTrue(changed['summary']['owner_changed']); self.assertEqual(changed['candidates'],[])
        self.assertEqual(p.process(frame(0,1000,epoch='two'))['candidates'],[])
        p.reset(); self.assertEqual(p.process(frame())['candidates'],[])
        p.close(); p.close(); self.assertEqual(len(p._entries),0); self.assertIsNone(p.predictor)
        with self.assertRaises(RuntimeError): p.process(frame())
        with self.assertRaises(RuntimeError): p.reset()

    def test_invalid_embedding_does_not_mutate_history(self):
        p=DINOv3Retrieval(lambda _:self.vector(),sha256='0'*64); p.process(frame())
        for bad in (np.zeros(384),np.ones(383),np.full(384,np.nan),np.full(384,1e7),np.ones(384,dtype=bool)):
            p.predictor=lambda _,bad=bad:bad
            with self.assertRaises(ValueError): p.process(frame(1,1001))
            self.assertEqual(len(p._entries),1); self.assertEqual(p._last,(1000,0))
        for bad in (np.zeros((31,40,3),dtype=np.uint8),np.zeros((32,129,3),dtype=np.uint8),np.zeros((40,60,3))):
            with self.assertRaises(ValueError): p.process(frame(1,1001,data=bad))

    def test_close_serializes_active_inference(self):
        import threading
        entered,release,closed=threading.Event(),threading.Event(),threading.Event(); errors=[]
        def predict(rgb):
            entered.set()
            if not release.wait(5): raise RuntimeError('gate timeout')
            return self.vector()
        p=DINOv3Retrieval(predict,sha256='0'*64)
        def work():
            try:p.process(frame())
            except BaseException as exc:errors.append(exc)
        worker=threading.Thread(target=work); closer=threading.Thread(target=lambda:(p.close(),closed.set()))
        worker.start()
        try:
            self.assertTrue(entered.wait(5)); closer.start(); self.assertFalse(closed.wait(.05))
        finally:
            release.set(); worker.join(5)
            if closer.ident is not None:closer.join(5)
        self.assertEqual(errors,[]);self.assertTrue(closed.is_set());self.assertEqual(len(p._entries),0)

    def test_loader_uses_verified_bytes_fixed_model_and_bounded_tensor(self):
        import contextlib
        from types import SimpleNamespace
        calls={}
        class Tensor:
            def __init__(self,array): self.array=array; self.shape=array.shape
            def __getitem__(self,key): return Tensor(self.array[key])
            def cpu(self):return self
            def numpy(self):return self.array
        class Model:
            def __init__(self,config):calls['config']=config
            def to(self,**kwargs):calls['device']=kwargs
            def load_state_dict(self,state,**kwargs):calls['state']=state;calls['strict']=kwargs
            def eval(self):calls['eval']=True
            def __call__(self,**kwargs):return SimpleNamespace(pooler_output=Tensor(np.ones(calls.get('output_shape',(1,384)))))
        class Processor:
            def __init__(self,**kwargs):calls['processor']=kwargs
            def __call__(self,**kwargs):
                calls['input']=kwargs
                return dict(pixel_values=Tensor(np.zeros(calls.get('input_shape',(1,3,224,224)))))
        def load(payload):calls['payload']=payload;return 'synthetic-state'
        st=SimpleNamespace(load=load)
        torch=SimpleNamespace(device=lambda _:contextlib.nullcontext(),float32='float32',
            inference_mode=contextlib.nullcontext,isfinite=lambda t:np.isfinite(t.array))
        tf=SimpleNamespace(DINOv3ViTConfig=lambda **kw:kw,DINOv3ViTModel=Model,DINOv3ViTImageProcessorFast=Processor)
        modules={'torch':torch,'transformers':tf,'safetensors':SimpleNamespace(torch=st),'safetensors.torch':st}
        versions={'torch':'2.8.0','torchvision':'0.23.0','transformers':'4.57.6'}
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'weights';payload=b'fixture';path.write_bytes(payload)
            with patch.dict('sys.modules',modules),patch('importlib.metadata.version',side_effect=versions.__getitem__):
                p=DINOv3Retrieval.from_local(str(path),hashlib.sha256(payload).hexdigest())
                path.write_bytes(b'changed'); p.process(frame())
                calls['input_shape']=(2,3,224,224)
                with self.assertRaises(ValueError):p.process(frame(1,1001))
                calls['input_shape']=(1,3,224,224);calls['output_shape']=(1,385)
                with self.assertRaises(ValueError):p.process(frame(1,1001))
            p.close()
        self.assertEqual(calls['payload'],payload);self.assertEqual(calls['strict'],{'strict':True})
        self.assertEqual(calls['config']['num_register_tokens'],4)
        self.assertEqual(calls['config']['hidden_size'],384)
        self.assertEqual(calls['processor']['size'],dict(height=224,width=224))
        self.assertEqual(calls['processor']['resample'],2);self.assertFalse(calls['processor']['do_center_crop'])
        self.assertEqual(calls['device'],dict(device='cpu',dtype='float32'))


@unittest.skipIf(np is None,'optional numpy absent')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_managed_gallery_and_mcp_clear(self):
        from humansense import HumanSense,SensorStreams
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        p=DINOv3Retrieval(lambda _:np.ones(384),sha256='0'*64)
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);streams.register(p.spec,lambda:p)
            await streams.open('cam','sensor',[p.spec.id],subject_id='person')
            now=timestamp()-100
            await streams.submit(frame(ts=now,epoch=None));await streams.drain()
            await streams.submit(frame(10,now+10,epoch=None));await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id,'subject_id':'person'})
            items=[item['observation'] for item in result['items']]
            matched=[item for item in items if item['value']['candidates']]
            self.assertEqual(len(matched),1)
            self.assertEqual(matched[0]['sensor']['capture_sequence'],10)
            self.assertFalse(matched[0]['value']['candidates'][0]['identity_confirmed'])
            # Ordinary sampled gaps preserve history; errors still force recovery reset.
            previous_epoch=matched[0]['sensor']['stream_epoch']
            p.predictor=lambda _:np.zeros(384)
            await streams.submit(frame(11,now+11,epoch=None));await streams.drain()
            p.predictor=lambda _:np.ones(384)
            await streams.submit(frame(12,now+12,epoch=None));await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id,'subject_id':'person'})
            recovered=[i['observation'] for i in result['items'] if i['observation']['sensor']['capture_sequence']==12][0]
            self.assertEqual(recovered['value']['candidates'],[])
            self.assertNotEqual(recovered['sensor']['stream_epoch'],previous_epoch)
            await streams.clear();self.assertTrue(p.closed);self.assertEqual(len(p._entries),0)
