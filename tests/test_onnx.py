import asyncio
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from humansense.onnx import ONNXPersonPlugin, ONNXFeaturePlugin
from humansense.plugins import InputFrame


class Array:
    def __init__(self, data, shape=None):
        self.data = data
        self.shape = shape or self._shape(data)
        self.ndim = len(self.shape)
    @classmethod
    def _shape(cls, data):
        return (len(data),) + (cls._shape(data[0]) if data and isinstance(data[0], list) else ())
    def __getitem__(self, index): return Array(self.data[index])
    def tolist(self): return self.data


class NP:
    @staticmethod
    def asarray(data): return data if isinstance(data, Array) else Array(data)
    @staticmethod
    def isfinite(array):
        def finite(data): return all(finite(x) for x in data) if isinstance(data, list) else math.isfinite(data)
        return SimpleNamespace(all=lambda: finite(array.data))


class Session:
    def __init__(self, output, contract='person_xyxy_score_class_v1'):
        self.output, self.contract = output, contract
        self.calls = 0
    def get_modelmeta(self): return SimpleNamespace(custom_metadata_map={'humansense.contract': self.contract})
    def get_inputs(self): return [SimpleNamespace(name='image', shape=[1,3,640,640], type='tensor(float)')]
    def get_outputs(self): return [SimpleNamespace(name='output')]
    def run(self, names, feed):
        self.calls += 1
        return [self.output]


class ONNXTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'model.onnx'
        self.path.write_bytes(b'explicit mock artifact')
        async def inline(fn, *args, **kwargs): return fn(*args, **kwargs)
        self.worker = patch('humansense.onnx.asyncio.to_thread', inline)
        self.worker.start()
    async def asyncTearDown(self):
        self.worker.stop()
        self.directory.cleanup()
    def plugin(self, session, feature=False, **kwargs):
        ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'],
                              InferenceSession=lambda *a, **kw: session)
        with patch.dict('sys.modules', {'onnxruntime': ort, 'numpy': NP, 'cv2': SimpleNamespace()}):
            if feature:
                return ONNXFeaturePlugin(str(self.path), modality='audio', output_labels={'output':['voice.expression.a','voice.expression.b']},
                                         preprocessor=lambda frame: {'image':frame.data}, **kwargs)
            return ONNXPersonPlugin(str(self.path), **kwargs)

    async def test_person_clips_filters_and_tracks_without_identity_claim(self):
        session = Session(Array([[[-10,0,320,640,.9,0],[10,10,20,20,.2,0],[0,0,100,100,.9,1]]]))
        plugin = self.plugin(session)
        plugin._detect = lambda image: plugin._parse(session.output)
        first = await plugin.process(InputFrame('pixels','video',1000,'stream'))
        boxes = next(p['value'] for p in first if p['name']=='scene.person_boxes')
        self.assertEqual(boxes,[{'x':0.,'y':0.,'width':.5,'height':1.}])
        tracks = next(p['value'] for p in first if p['name']=='scene.person_tracks')
        self.assertFalse(tracks[0]['identity_verified'])
        second = await plugin.process(InputFrame('pixels','video',1100,'stream'))
        self.assertEqual(tracks[0]['track_id'], next(p['value'] for p in second if p['name']=='scene.person_tracks')[0]['track_id'])
        self.assertIn('sha256:',first[0]['model_version'])
        await plugin.close()
        with self.assertRaises(RuntimeError): await plugin.process(InputFrame('pixels','video',1200,'stream'))

    async def test_raw_head_nonfinite_score_and_class_are_rejected(self):
        plugin = self.plugin(Session(Array([],(0,6))))
        self.assertEqual(plugin._parse(Array([],(0,6))), ([],[]))
        for raw in [Array([[1]*8400]*84), Array([[0,0,10,10,float('nan'),0]]),
                    Array([[0,0,10,10,1.1,0]]), Array([[0,0,10,10,.9,.5]]), Array([[10,0,0,10,.9,0]])]:
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError): plugin._parse(raw)

    async def test_metadata_provider_and_shape_contracts_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError,'metadata'):
            self.plugin(Session(Array([],(0,6)),contract='raw_yolo'))
        with self.assertRaisesRegex(ValueError,'providers'):
            self.plugin(Session(Array([],(0,6))),providers=['CUDAExecutionProvider'])
        session = Session(Array([],(0,6)))
        session.get_inputs = lambda: [SimpleNamespace(name='image',shape=[1,640,640,3],type='tensor(float)')]
        with self.assertRaisesRegex(ValueError,'shape'): self.plugin(session)

    async def test_feature_labels_emit_scores_without_probability(self):
        plugin = self.plugin(Session(Array([[.7,.2]]),'feature_model_scores_v1'),feature=True)
        result = await plugin.process(InputFrame([0.],'audio',1000,'owner',16000))
        self.assertEqual([p['name'] for p in result],['voice.expression.a','voice.expression.b'])
        self.assertEqual(result[0]['value'],{'model_score':.7,'probability':None,'calibrated':False})
        self.assertEqual(result[0]['subject_id'],'owner')
        for raw in [Array([.5]),Array([1.1,.2]),Array([float('nan'),.2])]:
            plugin.session.output=raw
            with self.assertRaises(ValueError): await plugin.process(InputFrame([0.],'audio',1001))

    async def test_reset_discards_inflight_feature_results(self):
        entered, release = asyncio.Event(),asyncio.Event()
        async def delay(fn,*args,**kwargs):
            entered.set(); await release.wait(); return fn(*args,**kwargs)
        plugin = self.plugin(Session(Array([.1,.2]),'feature_model_scores_v1'),feature=True)
        with patch('humansense.onnx.asyncio.to_thread',delay):
            task=asyncio.create_task(plugin.process(InputFrame([0.],'audio',1000)))
            await entered.wait(); plugin.reset(); release.set()
            self.assertEqual(await task,[])
        self.assertEqual(plugin.session.calls,0)

    async def test_missing_models_rejected_without_download(self):
        with self.assertRaises(ValueError): ONNXPersonPlugin('/does-not-exist.onnx')
        with self.assertRaises(ValueError): ONNXFeaturePlugin('/does-not-exist.onnx',modality='audio',output_labels={},preprocessor=lambda x:x)


if __name__=='__main__': unittest.main()
