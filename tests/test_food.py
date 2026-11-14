from dataclasses import replace
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from humansense import ProcessProvider,HumanSense,SensorStreams,SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.food import FoodContext,SegFormerFoodNative,configuration,preprocess,model_config

CONFIG=dict(checkpoint_path='/explicit/food.safetensors',checkpoint_sha256='a'*64,labels=['background','rice','tofu','unknown'],
    background_id=0,unknown_id=3,training_dataset='fixture',target_cuisine='fixture-only',model_source='generated-test',
    checkpoint_license='fixture-only',input_size=256)
try:
    import numpy as np
    from PIL import Image
except ImportError:np=None


def runtime_available():
    try:return all(version(p).split('+')[0]==v for p,v in (('transformers','4.57.6'),('torch','2.8.0'),('huggingface-hub','0.36.0'),('safetensors','0.7.0')))
    except PackageNotFoundError:return False


class Discovery(unittest.TestCase):
    def test_discovery_no_ml_import_and_bad_config(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("food.segformer_b0",'+repr(CONFIG)+');assert not {"torch","transformers","numpy","PIL"}&sys.modules.keys()'
        p=subprocess.run([sys.executable,'-c',code],capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
        for bad in ({'checkpoint_path':'relative'},{'labels':['same','same']},{'background_id':True},{'unknown_id':0},
                    {'input_size':1024},{'threshold':float('nan')},{'device':'auto'},{'extra':1}):
            with self.assertRaises(ValueError):configuration(dict(CONFIG,**bad))
    def test_unavailable_local_asset_and_hash_reject(self):
        with self.assertRaises(ValueError):SegFormerFoodNative.from_local(CONFIG)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            p=Path(directory)/'bad.safetensors';p.write_bytes(b'not weights')
            with self.assertRaises(ValueError):SegFormerFoodNative.from_local(dict(CONFIG,checkpoint_path=str(p)))


class Backend:
    def __init__(self,classes=4,size=64):
        self.value=np.full((classes,size,size),-10,np.float32);self.value[0]=10;self.closed=False
    def predict(self,data):self.input=data.copy();return self.value
    def close(self):self.closed=True


@unittest.skipIf(np is None,'optional NumPy/Pillow absent')
class Context(unittest.TestCase):
    def test_exact_partition_mask_grid_boxes_and_abstention(self):
        b=Backend();b.value[:,0:32,0:32]=-10;b.value[1,0:32,0:32]=10
        b.value[:,32:64,0:32]=-10;b.value[2,32:64,0:32]=10
        b.value[:,0:16,32:64]=-10;b.value[3,0:16,32:64]=10
        b.value[:,16:32,32:64]=0
        p=FoodContext(b,**CONFIG);f=InputFrame(np.zeros((80,120,3),np.uint8),'video',1000)
        v=p.process(f);self.assertEqual([r['label'] for r in v['regions']],['rice','tofu'])
        self.assertEqual(v['regions'][0]['image_area_fraction'],.25)
        self.assertEqual(sum(v['regions'][0]['mask']['coverage'])/256,.25)
        self.assertEqual(v['regions'][0]['box_xyxy_normalized'],[0,0,.5,.5])
        s=v['summary'];self.assertEqual(s['declared_unknown_class_fraction'],.125);self.assertEqual(s['below_threshold_fraction'],.125)
        self.assertEqual(sum(r['image_area_fraction'] for r in v['regions'])+sum(s[k] for k in ('background_fraction','declared_unknown_class_fraction','below_threshold_fraction','omitted_food_fraction')),1)
        self.assertIsNone(s['calories_kcal']);self.assertFalse(s['unknown_detection_validated'])
        b.value[:]=0;self.assertEqual(p.process(f)['regions'],[])
        p.close();p.close();self.assertTrue(b.closed)
        with self.assertRaises(RuntimeError):p.process(f)

    def test_preprocess_bgr_rgb_and_owned_input(self):
        bgr=np.zeros((32,64,3),np.uint8);bgr[:]=[0,128,255]
        x=preprocess(bgr,256);expected=(np.array([1,128/255,0],np.float32)-[.485,.456,.406])/[.229,.224,.225]
        np.testing.assert_allclose(x[0,:,0,0],expected,rtol=1e-6)
        self.assertEqual(x.shape,(1,3,256,256));self.assertEqual(x.dtype,np.float32)
        self.assertEqual(bgr[0,0].tolist(),[0,128,255])
        for value in (np.ma.array(bgr),bgr.astype(np.float32),bgr[:20],np.broadcast_to(np.uint8(0),(2160,3840,3))):
            with self.assertRaises(ValueError):preprocess(value,256)

    def test_truncation_extreme_logits_and_malformed_output(self):
        config=dict(CONFIG,max_labels=1);b=Backend();b.value[1,:,:32]=np.finfo(np.float32).max;b.value[2,:,32:]=np.finfo(np.float32).max
        p=FoodContext(b,**config);f=InputFrame(np.zeros((32,32,3),np.uint8),'video',1000)
        v=p.process(f);self.assertTrue(v['summary']['truncated']);self.assertEqual(v['summary']['omitted_food_fraction'],.5)
        for value in (np.zeros((4,63,64),np.float32),np.zeros((4,64,64),np.float64),np.full((4,64,64),np.nan,np.float32)):
            b.value=value
            with self.assertRaises(ValueError):p.process(f)
        with self.assertRaises(ValueError):p.process(replace(f,data=np.ma.array(f.data)))

    def test_maximum_label_mask_output_fits_context_budget(self):
        config=dict(CONFIG,labels=['食'*64+str(i) for i in range(16)])
        # Label length is bounded64 including identifier suffix.
        config['labels']=['食'*62+str(i) for i in range(16)]
        config['unknown_id']=None
        b=Backend(classes=16)
        for i in range(16):b.value[i,i*4:(i+1)*4,:]=20
        value=FoodContext(b,**config).process(InputFrame(np.zeros((32,32,3),np.uint8),'video',1000))
        self.assertEqual(len(value['regions']),15)
        self.assertLess(len(json.dumps(value,ensure_ascii=True).encode()),65536)


@unittest.skipUnless(runtime_available() and np is not None,'dedicated pinned food runtime absent; no automatic install')
class Native(unittest.TestCase):
    def test_generated_untrained_graph_strict_loader_and_reference(self):
        import torch
        from transformers import SegformerConfig,SegformerForSemanticSegmentation
        from safetensors.torch import save
        previous_threads=torch.get_num_threads()
        self.addCleanup(torch.set_num_threads,previous_threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17);model=SegformerForSemanticSegmentation(model_config(SegformerConfig,CONFIG['labels'])).float().eval()
        payload=save(model.state_dict());digest=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path=Path(directory)/'untrained.safetensors';path.write_bytes(payload)
            config=dict(CONFIG,checkpoint_path=str(path),checkpoint_sha256=digest)
            native=SegFormerFoodNative.from_local(config)
            try:
                image=np.random.default_rng(17).integers(0,256,(48,80,3),np.uint8)
                actual=native.predict(image)
                with torch.inference_mode():expected=model(pixel_values=torch.from_numpy(preprocess(image,256))).logits[0].numpy()
                np.testing.assert_allclose(actual,expected,atol=0,rtol=0)
                self.assertEqual(actual.shape,(4,64,64))
            finally:native.close()
            # Exercise the real generated graph in the owned worker and MCP path.
            import asyncio
            from humansense.schema import timestamp
            from humansense.mcp import MCPServer
            async def managed():
                worker=await ProcessProvider.open(sys.executable,'food.segformer_b0',config,inference_timeout=60)
                child=worker._process
                try:
                    async with HumanSense(consent=True) as human:
                        streams=SensorStreams(human)
                        streams.register(worker.spec,lambda:worker)
                        await streams.open('food-camera','video',[worker.spec.id])
                        try:
                            captured=timestamp()-100
                            frame=InputFrame(image,'video',captured,sensor=SensorMetadata(source_id='food-camera',sensor_id='fixture-camera',sequence=0))
                            await streams.submit(frame);await streams.drain()
                            records=human.store.latest_signals('local-user',0,timestamp())
                            self.assertEqual(len(records),1)
                            self.assertEqual(records[0]['timestamp_ms'],captured)
                            value=records[0]['observation']['value']
                            self.assertEqual(value['summary']['checkpoint_sha256'],digest)
                            self.assertFalse(value['summary']['model_quality_validated'])
                            reply=await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})
                            self.assertIn('semantic_not_instances',str(reply))
                            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)
                        finally:await streams.clear()
                finally:await worker.close()
                self.assertIsNotNone(child.returncode)
            asyncio.run(managed())
            # Missing one learned tensor must never silently keep random initialization.
            state=model.state_dict();state.pop(next(iter(state)));bad=save(state);path.write_bytes(bad)
            with self.assertRaises(ValueError):SegFormerFoodNative.from_local(dict(config,checkpoint_sha256=hashlib.sha256(bad).hexdigest()))
