import contextlib
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humansense import HumanSense, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.provider_profiles import capability
from humansense.providers.depth import DepthAnythingContext
from humansense.schema import timestamp
try:
    import numpy as np
except ImportError:
    np=None


def frame(data=None):
    if data is None:
        data=np.zeros((60,100,3),dtype=np.uint8); data[:,:,0]=10; data[:,:,2]=90
    return InputFrame(data,'video',timestamp()-100,sensor=SensorMetadata('camera','sensor',0))


def provider(depth,grid_size=3):
    return DepthAnythingContext(lambda image:depth,sha256='0'*64,grid_size=grid_size)


class Metadata(unittest.TestCase):
    def test_discovery_and_artifact_validation_before_import(self):
        config=dict(path='/not-installed',sha256='0'*64)
        spec=capability('depth_anything_v2.small',config)
        self.assertFalse(spec.stateful); self.assertEqual(spec.streaming_mode,'frame')
        self.assertNotEqual(spec.version,capability('depth_anything_v2.small',dict(config,grid_size=4)).version)
        for extra in (dict(sha256='bad'),dict(path=1),dict(grid_size=True),dict(grid_size=9),dict(extra=1)):
            with self.assertRaises(ValueError): capability('depth_anything_v2.small',dict(config,**extra))
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.safetensors'; path.write_bytes(b'fixture-not-weights')
            with patch('importlib.metadata.version') as version:
                with self.assertRaises(ValueError): DepthAnythingContext.from_local(str(path),'0'*64)
                version.assert_not_called()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): DepthAnythingContext.from_local(str(path),digest)


@unittest.skipIf(np is None,'optional numpy not installed')
class Depth(unittest.TestCase):
    def test_rgb_copy_spatial_order_and_affine_invariance(self):
        raw=np.tile(np.arange(12,dtype=float),(12,1)); seen=[]
        def predict(image): seen.append(image); return raw
        p=DepthAnythingContext(predict,sha256='0'*64)
        f=frame(); result=p.process(f); values=result['grid']['relative_nearness']
        self.assertEqual(len(values),9)
        np.testing.assert_allclose(result['grid']['x_edges'],[0,1/3,2/3,1])
        self.assertLess(values[0],values[1]); self.assertLess(values[1],values[2])
        np.testing.assert_allclose(values[:3],values[3:6])
        np.testing.assert_allclose(values,provider(7*raw+20).process(frame())['grid']['relative_nearness'],atol=1e-12)
        self.assertEqual(seen[0][0,0].tolist(),[90,0,10])
        seen[0][:]=0; self.assertEqual(f.data[0,0].tolist(),[10,0,90])
        self.assertFalse(result['summary']['metric_depth'])
        self.assertFalse(result['summary']['temporal_comparability'])
        self.assertEqual(result['summary']['confidence'],'unavailable')

    def test_flat_and_low_spread_abstain_and_robust_clipping(self):
        for depth in (np.zeros((12,12)),np.ones((12,12))*100,np.ones((12,12))+np.eye(12)*1e-8):
            result=provider(depth).process(frame())
            self.assertEqual(result['summary']['status'],'unavailable_low_spread')
            self.assertEqual(result['grid']['relative_nearness'],[None]*9)
        raw=np.arange(400).reshape(20,20).astype(float); raw[-1,-1]=1e6
        values=provider(raw,8).process(frame())['grid']['relative_nearness']
        self.assertEqual(len(values),64)
        self.assertTrue(all(0<=value<=1 for value in values))

    def test_input_output_limits_no_retained_map_and_close(self):
        for output in (np.zeros((2,2)),np.zeros((4,4,1)),np.zeros((1037,3)),
                       np.zeros((518,1036),dtype=bool),np.full((8,8),np.nan),
                       np.full((8,8),-1),np.full((8,8),1e7),np.full((8,8),'x')):
            with self.assertRaises(ValueError): provider(output).process(frame())
        for data in (np.zeros((30,30,3),dtype=np.uint8),np.zeros((32,100,3),dtype=np.uint8),
                     np.zeros((100,100,3)),np.zeros((1081,1920,3),dtype=np.uint8)):
            with self.assertRaises(ValueError): provider(np.zeros((8,8))).process(frame(data))
        p=provider(np.arange(64).reshape(8,8)); result=p.process(frame())
        self.assertNotIn('depth_map',result)
        p.close(); p.close(); self.assertIsNone(p.predictor)
        with self.assertRaises(RuntimeError): p.process(frame())

    def test_close_waits_for_active_native_call(self):
        import threading
        entered,release,closed=threading.Event(),threading.Event(),threading.Event()
        errors=[]
        def predict(rgb):
            entered.set()
            if not release.wait(5): raise RuntimeError('test gate timed out')
            return np.arange(144).reshape(12,12)
        p=DepthAnythingContext(predict,sha256='0'*64)
        def work():
            try: p.process(frame())
            except BaseException as exc: errors.append(exc)
        worker=threading.Thread(target=work)
        closer=threading.Thread(target=lambda:(p.close(),closed.set()))
        worker.start()
        try:
            self.assertTrue(entered.wait(5)); closer.start()
            self.assertFalse(closed.wait(.05))
        finally:
            release.set(); worker.join(5)
            if closer.ident is not None: closer.join(5)
        self.assertEqual(errors,[]); self.assertTrue(closed.is_set())
        self.assertIsNone(p.predictor)

    def test_local_loader_exact_bytes_fixed_config_and_native_output_bounds(self):
        calls={}; raw=np.arange(518*518).reshape(518,518).astype(float)
        class Tensor:
            def __init__(self,array): self.array=array; self.shape=array.shape; self.ndim=array.ndim
            def __getitem__(self,key): return Tensor(self.array[key])
            def cpu(self): return self
            def numpy(self): return self.array
        class Model:
            def __init__(self,config): calls['config']=config
            def to(self,**kwargs): calls['device']=kwargs
            def load_state_dict(self,state,**kwargs): calls['state']=state; calls['strict']=kwargs
            def eval(self): calls['eval']=True
            def __call__(self,**kwargs): calls['model_input']=kwargs; return SimpleNamespace(predicted_depth=Tensor(np.zeros(calls['output_shape']) if 'output_shape' in calls else raw[None]))
        class Processor:
            def __init__(self,**kwargs): calls['processor']=kwargs
            def __call__(self,**kwargs):
                calls['image']=kwargs
                return dict(pixel_values=Tensor(np.zeros(calls.get('tensor_shape',(1,3,518,518)))))
        def config(**kwargs): return kwargs
        def load(payload): calls['payload']=payload; return 'synthetic-state'
        tf=SimpleNamespace(DepthAnythingConfig=config,DepthAnythingForDepthEstimation=Model,
            Dinov2Config=config,DPTImageProcessor=Processor)
        torch=SimpleNamespace(device=lambda value:contextlib.nullcontext(),float32='float32',
                              inference_mode=contextlib.nullcontext)
        st=SimpleNamespace(load=load)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'model.safetensors'; payload=b'fixture'; path.write_bytes(payload)
            modules={'torch':torch,'transformers':tf,'safetensors':SimpleNamespace(torch=st),'safetensors.torch':st}
            versions={'torch':'2.8.0','transformers':'4.57.6'}
            with patch.dict('sys.modules',modules),patch('importlib.metadata.version',side_effect=versions.__getitem__):
                p=DepthAnythingContext.from_local(str(path),hashlib.sha256(payload).hexdigest())
                path.write_bytes(b'changed-after-verification')
                p.process(frame())
                for shape in ((1,3,518,1050),(2,3,518,518),(1,3,519,518),(1,3,1036,1036),(1,3,0,518)):
                    calls['tensor_shape']=shape
                    with self.assertRaises(ValueError): p.process(frame())
                calls['tensor_shape']=(1,3,518,518)
                calls['output_shape']=(1,14,14)
                with self.assertRaises(ValueError): p.process(frame())
            p.close()
        self.assertEqual(calls['payload'],payload)
        self.assertEqual(calls['strict'],{'strict':True})
        self.assertEqual(calls['device'],dict(device='cpu',dtype='float32'))
        self.assertEqual(calls['config']['depth_estimation_type'],'relative')
        self.assertFalse(calls['config']['use_pretrained_backbone'])
        self.assertEqual(calls['config']['backbone_config']['out_indices'],[3,6,9,12])
        self.assertEqual(calls['processor']['size'],dict(height=518,width=518))
        self.assertTrue(calls['processor']['keep_aspect_ratio']); self.assertEqual(calls['processor']['ensure_multiple_of'],14)
        self.assertEqual(calls['image']['images'][0,0].tolist(),[90,0,10])


@unittest.skipIf(np is None,'optional numpy not installed')
class Publication(unittest.IsolatedAsyncioTestCase):
    async def test_managed_depth_and_mcp_capture_provenance(self):
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            p=provider(np.arange(144).reshape(12,12)); streams=SensorStreams(human)
            streams.register(p.spec,lambda:p)
            await streams.open('camera','sensor',[p.spec.id])
            await streams.submit(frame()); await streams.drain()
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
            item=result['items'][0]['observation']
            self.assertEqual(item['value']['summary']['status'],'estimated')
            self.assertEqual(item['sensor']['capture_sequence'],0)
            await streams.clear(); self.assertTrue(p.closed)
