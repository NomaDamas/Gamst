import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from humansense.plugins import InputFrame
from humansense.yolo import UltralyticsYOLOPlugin


class Tensor:
    def __init__(self,values): self.values=values
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class YOLOTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'local.pt'
        self.path.write_bytes(b'explicit trusted mock weights')
        self.image=SimpleNamespace(ndim=3,shape=(100,200,3),dtype='uint8')
        async def inline(fn,*args,**kwargs): return fn(*args,**kwargs)
        self.worker=patch('humansense.yolo.asyncio.to_thread',inline); self.worker.start()
    async def asyncTearDown(self):
        self.worker.stop(); self.directory.cleanup()
    def plugin(self,coordinates=None,scores=None,classes=None):
        result=SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(coordinates if coordinates is not None else [[-10,0,100,100]]),
            conf=Tensor(scores if scores is not None else [.8]),cls=Tensor(classes if classes is not None else [0.])))
        class Model:
            def __init__(self,path,task): self.path,self.task,self.calls=path,task,[]
            def predict(self,**kwargs): self.calls.append(kwargs); return [result]
        with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLO=Model)}):
            plugin=UltralyticsYOLOPlugin(str(self.path))
        return plugin,result
    async def test_local_model_predict_and_tracking_flags(self):
        plugin,_=self.plugin()
        first=await plugin.process(InputFrame(self.image,'video',1000,'cam'))
        calls=plugin.model.calls
        self.assertEqual(plugin.model.path,str(self.path)); self.assertEqual(plugin.model.task,'detect')
        self.assertFalse(calls[0]['save']); self.assertFalse(calls[0]['show']); self.assertFalse(calls[0]['verbose'])
        self.assertEqual(calls[0]['classes'],[0]); self.assertEqual(calls[0]['device'],'cpu')
        values={p['name']:p['value'] for p in first}
        self.assertEqual(values['scene.person_count'],1)
        self.assertEqual(values['scene.person_boxes'],[dict(x=0.,y=0.,width=.5,height=1.)])
        self.assertFalse(values['scene.person_tracks'][0]['identity_verified'])
        self.assertIsNone(values['scene.person_detection_scores']['probability'])
        second=await plugin.process(InputFrame(self.image,'video',1100,'cam'))
        nexttracks=next(p['value'] for p in second if p['name']=='scene.person_tracks')
        self.assertEqual(nexttracks[0]['track_id'],values['scene.person_tracks'][0]['track_id'])
        self.assertIn('sha256:',first[0]['model_version'])
        await plugin.close()
        with self.assertRaises(RuntimeError): await plugin.process(InputFrame(self.image,'video',1200,'cam'))
    async def test_invalid_model_outputs_fail_instead_of_inventing_boxes(self):
        for coords,scores,classes in [([[0,0,10,10]],[],[0]),([[0,0,10,float('nan')]],[.8],[0]),
            ([[10,0,0,10]],[.8],[0]),([[0,0,10,10]],[.8],[.5])]:
            with self.subTest(coords=coords):
                plugin,_=self.plugin(coords,scores,classes)
                with self.assertRaises(ValueError): await plugin.process(InputFrame(self.image,'video',1000))
    async def test_nonperson_and_low_score_filtered(self):
        plugin,_=self.plugin([[0,0,10,10],[0,0,10,10]],[.9,.1],[1.,0.])
        result=await plugin.process(InputFrame(self.image,'video',1000))
        self.assertEqual(result[0]['value'],0)
    async def test_reset_discards_native_results_and_tracking_history(self):
        plugin,_=self.plugin()
        entered,release=asyncio.Event(),asyncio.Event()
        async def delay(fn,*args,**kwargs):
            entered.set(); await release.wait(); return fn(*args,**kwargs)
        with patch('humansense.yolo.asyncio.to_thread',delay):
            task=asyncio.create_task(plugin.process(InputFrame(self.image,'video',1000)))
            await entered.wait(); plugin.reset(); release.set()
            self.assertEqual(await task,[])
        self.assertEqual(plugin.model.calls,[])
    async def test_nonlocal_paths_rejected_before_model_import(self):
        with self.assertRaises(ValueError): UltralyticsYOLOPlugin('yolo-latest.pt')
        with self.assertRaises(ValueError): UltralyticsYOLOPlugin('https://example.org/model.pt')


if __name__=='__main__': unittest.main()
