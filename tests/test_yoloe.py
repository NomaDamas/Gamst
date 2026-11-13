import dataclasses
import asyncio
import contextlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.yoloe import YOLOEContext, mask_context, prompt_labels, visual_examples
from humansense.providers.yoloe_native import configuration, _read, NativeYOLOE

CONFIG = dict(labels=['cup','glass'],prompt_sha256='a'*64,checkpoint_sha256='b'*64,mode='text')
NATIVE = dict(labels=['cup','glass'],checkpoint_path='/fixture.pt',checkpoint_sha256='b'*64,
    mode='text',runtime_root='/fixture',encoder_path='/fixture.ts',encoder_sha256='a'*64,
    encoder_family='mobileclip:blt')


class Backend:
    def __init__(self): self.closed = False; self.output = None
    def predict(self,image):
        if self.output is not None: return self.output
        h,w = image.shape[:2]
        mask = np.zeros((1,h,w),dtype=np.uint8); mask[:,:,:w//2] = 1
        image[:] = 255  # Provider must own its native input, not mutate caller.
        return np.asarray([[0,0,w/2,h,.75,0]],dtype=np.float32),mask
    def close(self): self.closed = True


def frame(i=0,h=33,w=47):
    return InputFrame(np.zeros((h,w,3),dtype=np.uint8),'video',1000+i*50,
        sensor=SensorMetadata('camera1','video',i,coordinate_frame='image1',stream_epoch='e1'))


class Labels(unittest.TestCase):
    def test_order_normalization_and_rejection(self):
        self.assertEqual(prompt_labels(['cafe\u0301','컵']),('café','컵'))
        for value in ([],['x']*17,['x','x'],['café','cafe\u0301'],[' x'],['x\n'],['x\x00'],['x\u200b'],[True]):
            with self.assertRaises(ValueError): prompt_labels(value)

    def test_configuration_is_dependency_light_and_hash_binds_prompt_assets(self):
        a,_,_ = configuration(NATIVE)
        b,_,_ = configuration(dict(NATIVE,encoder_sha256='c'*64))
        self.assertNotEqual(a['prompt_sha256'],b['prompt_sha256'])
        for change in ({'imgsz':True},{'device':'mps'},{'threshold':float('nan')},{'encoder_family':'clip:other'}):
            with self.assertRaises(ValueError): configuration(dict(NATIVE,**change))
        code = 'import sys;from humansense import ProcessProvider;ProcessProvider.capability("yoloe.local",'+repr(NATIVE)+');assert not {"numpy","torch","ultralytics","clip"}&sys.modules.keys()'
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_asset_snapshot_hash_bounds_and_child_network_guard(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            path = Path(root)/'asset'; path.write_bytes(b'fixture')
            digest = hashlib.sha256(b'fixture').hexdigest()
            self.assertEqual(_read(path,digest,7),b'fixture')
            with self.assertRaises(ValueError): _read(path,digest,6)
            with self.assertRaises(ValueError): _read(path,'a'*64,7)
            code = '''import os,socket,sys
from humansense.providers.yoloe_native import _bootstrap
os.environ['GAMST_PROVIDER_WORKER']='1'
_bootstrap(sys.argv[1])
assert os.environ['YOLO_AUTOINSTALL']=='False'
assert os.environ['ULTRALYTICS_SAFE_LOAD']=='True'
assert os.environ['YOLO_OFFLINE']=='True'
for call in (lambda:socket.getaddrinfo('example.com',443),lambda:socket.create_connection(('example.com',443))):
    try:call()
    except RuntimeError:pass
    else:raise AssertionError('network guard missing')
assert not {'torch','ultralytics'}&sys.modules.keys()
'''
            result = subprocess.run([sys.executable,'-c',code,root],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipIf(np is None,'NumPy optional runtime is unavailable')
class Context(unittest.TestCase):
    def test_native_call_bridge_preserves_reference_once_and_mask_shape(self):
        # This verifies API arguments/result conversion, not Ultralytics inference.
        class Tensor:
            def __init__(self,value): self.value=value; self.shape=value.shape
            def detach(self): return self
            def cpu(self): return self
            def float(self): return self
            def numpy(self): return self.value
        class Model:
            def __init__(self): self.calls=[]
            def predict(self,**kw):
                self.calls.append(kw)
                h,w=kw['source'].shape[:2]
                return [SimpleNamespace(orig_shape=(h,w),boxes=SimpleNamespace(data=Tensor(
                    np.array([[0,0,w,h,.5,0]],np.float32))),masks=SimpleNamespace(data=Tensor(np.ones((1,h,w),np.uint8))))]
        model=Model(); ref=np.zeros((32,32,3),np.uint8)
        native=NativeYOLOE(model,device='cpu',imgsz=320,threshold=.25,max_objects=16,
            reference=ref,prompts={'bboxes':np.array([[0,0,16,16]],np.float32),'cls':np.array([0])})
        with patch.dict(sys.modules,{'torch':SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            for _ in range(2): boxes,masks=native.predict(ref)
        self.assertEqual(boxes.shape,(1,6)); self.assertEqual(masks.shape,(1,32,32))
        self.assertIn('refer_image',model.calls[0]); self.assertNotIn('refer_image',model.calls[1])
        self.assertTrue(model.calls[0]['retina_masks']); self.assertFalse(model.calls[0]['save'])
        self.assertEqual(model.calls[0]['max_det'],16)
        # Shape must be bounded before a GPU/host transfer or copy is attempted.
        model.predict = lambda **kw:[SimpleNamespace(orig_shape=(32,32),
            boxes=SimpleNamespace(data=SimpleNamespace(shape=(17,6))),masks=None)]
        with patch.dict(sys.modules,{'torch':SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            with self.assertRaises(ValueError): native.predict(ref)
        native.close(); self.assertIsNone(native.model)

    def test_reference_class_mapping_owned_bytes_and_hash(self):
        image = np.zeros((50,60,3),np.uint8)
        boxes = [[0,0,20,30],[20,5,59,49],[30,20,40,30]]
        a = visual_examples(image,boxes,[0,1,0],['cup','glass'])
        b = visual_examples(image,boxes,[0,1,0],['cup','glass'])
        self.assertEqual(a[-1],b[-1]); self.assertEqual(a[1].dtype,np.float32)
        image[:] = 9; self.assertFalse(a[0].any())
        c = visual_examples(image,boxes,[0,1,0],['cup','glass'])
        self.assertNotEqual(a[-1],c[-1])
        for badboxes,badids in ((boxes,[0,2,0]),(boxes,[0,0,0]),([[-1,0,2,2]],[0]),
                               ([[0,0,1,1]],[True]),([[0,0,float('nan'),2]],[0])):
            with self.assertRaises(ValueError): visual_examples(image,badboxes,badids,['cup','glass'])

    def test_mask_grid_exact_integer_partitions_and_area(self):
        rng = np.random.default_rng(7)
        for h,w in ((32,32),(33,47),(101,67)):
            mask = rng.integers(0,2,(h,w),dtype=np.uint8)
            result = mask_context(mask)
            grid = np.frombuffer(bytes.fromhex(result['coverage_hex']),np.uint8).reshape(16,16)
            for y in range(16):
                for x in range(16):
                    cell = mask[y*h//16:(y+1)*h//16,x*w//16:(x+1)*w//16]
                    self.assertEqual(grid[y,x],round(255*float(cell.mean())))
            self.assertEqual(result['foreground_pixels'],int(mask.sum()))
            self.assertEqual(result['area_fraction'],float(mask.sum())/(h*w))
        self.assertEqual(set(bytes.fromhex(mask_context(np.ones((32,32),np.bool_))['coverage_hex'])),{255})
        for mask in (np.full((32,32),.5,np.float32),np.ones((31,32),np.uint8),np.full((32,32),np.nan,np.float32)):
            with self.assertRaises(ValueError): mask_context(mask)

    def test_context_geometry_provenance_and_no_mutation(self):
        backend = Backend(); p = YOLOEContext(backend,**CONFIG)
        f = frame(); result = p.process(f)
        obj = result['objects'][0]
        self.assertEqual(obj['label'],'cup'); self.assertEqual(obj['bbox']['width'],.5)
        self.assertEqual(obj['mask']['foreground_pixels'],33*23)
        self.assertFalse(f.data.any())
        self.assertFalse(result['summary']['model_quality_verified'])
        self.assertFalse(result['summary']['tracked'])
        self.assertEqual(result['summary']['prompt_sha256'],'a'*64)
        with self.assertRaises(ValueError): p.process(f)
        p.process(f)  # Error resets capture guard; no tracker state to leak.
        p.reset(); p.process(frame())
        p.close(); p.reset(); self.assertTrue(backend.closed)
        with self.assertRaises(RuntimeError): p.process(frame(1))

    def test_empty_bad_native_output_and_metadata(self):
        backend = Backend(); p = YOLOEContext(backend,**CONFIG)
        backend.output = (np.empty((0,6),np.float32),np.empty((0,33,47),np.uint8))
        self.assertEqual(p.process(frame())['objects'],[])
        for boxes,masks in ((np.zeros((1,6),np.float64),np.zeros((1,33,47),np.uint8)),
            (np.asarray([[0,0,50,20,.5,0]],np.float32),np.zeros((1,33,47),np.uint8)),
            (np.asarray([[0,0,20,20,.5,.1]],np.float32),np.zeros((1,33,47),np.uint8)),
            (np.asarray([[0,0,20,20,.5,0]],np.float32),np.zeros((1,32,47),np.uint8))):
            backend.output = boxes,masks
            with self.assertRaises(ValueError): p.process(frame(1))
        with self.assertRaises(ValueError): p.process(dataclasses.replace(frame(),timestamp_ms=True))
        with self.assertRaises(ValueError): p.process(dataclasses.replace(frame(),sensor=None))


@unittest.skipIf(np is None,'NumPy optional runtime is unavailable')
class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_owned_worker_context_to_mcp_with_explicit_test_backend(self):
        from humansense import HumanSense,ProcessProvider,SensorStreams
        from humansense.mcp import MCPServer
        children=[]; real_spawn=asyncio.create_subprocess_exec
        script='''import sys
sys.path.insert(0,'tests')
from test_yoloe import Backend
from humansense.providers.yoloe import YOLOEContext
from humansense.providers import yoloe_native
from humansense.provider_worker import main
yoloe_native.load_worker=lambda config:YOLOEContext(Backend(),**yoloe_native.configuration(config)[0])
main()
'''
        async def spawn(*args,**kw):
            child=await real_spawn(sys.executable,'-u','-c',script,**kw); children.append(child); return child
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                async with HumanSense(consent=True) as human:
                    streams=SensorStreams(human);spec=ProcessProvider.capability('yoloe.local',NATIVE)
                    async def create():return await ProcessProvider.open(sys.executable,'yoloe.local',NATIVE)
                    streams.register(spec,create);await streams.open('camera1','video',[spec.id])
                    f=frame();f=dataclasses.replace(f,timestamp_ms=int(time.time()*1000),
                        sensor=dataclasses.replace(f.sensor,stream_epoch=None))
                    await streams.submit(f);await streams.drain()
                    output=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                    self.assertTrue(output['items'])
                    value=output['items'][0]['observation']['value']
                    self.assertEqual(value['objects'][0]['label'],'cup')
                    self.assertFalse(value['summary']['artifact_integrity_verified'])
                    await streams.finish('camera1','video')
            self.assertTrue(children and all(c.returncode is not None for c in children))
        finally:
            for child in children:
                if child.returncode is None:child.kill();await child.wait()


if __name__ == '__main__': unittest.main()
