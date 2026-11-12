import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from humansense import HumanSense
from humansense.extensions import classification, discover_plugins, load_plugin
from humansense.plugins import InputFrame
from humansense.providers.huggingface import HuggingFaceClassifier


class ExtensionTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_preserves_labels_provenance_and_context(self):
        def pipeline(data,top_k):
            self.assertEqual(data['sampling_rate'],16000)
            return [{'label':'speech','score':.7},{'label':'music','score':.3}]
        plugin=HuggingFaceClassifier(pipeline,modality='audio',name='classification.sound',model_id='fixture',revision='pinned-sha',exclusive=True)
        numpy=SimpleNamespace(asarray=lambda data,dtype:data,float32='float32')
        async with HumanSense(consent=True) as h:
            h.register(plugin)
            with patch.dict('sys.modules',{'numpy':numpy}):
                await h.push_audio([0]*100)
            context=await h.get_context()
            obs=context['observations']['classification.sound']
            self.assertEqual(obs['value']['labels']['speech'],.7)
            self.assertEqual(obs['models'][0]['id'],'fixture')
            self.assertEqual(obs['value']['score_kind'],'score')

    async def test_invalid_classification_cannot_leak_raw_context(self):
        async with HumanSense(consent=True) as h:
            await h.push_signal('classification.bad',{'labels':{'raw':'secret'},'jpeg':'secret'},'video')
            self.assertNotIn('classification.bad',(await h.get_context())['observations'])
            value=classification({f'c{i}':.01 for i in range(32)})
            await h.push_signal('classification.full',value,'video')
            self.assertEqual(len((await h.get_context())['observations']['classification.full']['value']['labels']),32)

    def test_invalid_score_semantics(self):
        for labels in ({'a':float('nan')},{'a':-1},{'a':True}):
            with self.assertRaises(ValueError): classification(labels)
        with self.assertRaises(ValueError):classification({'a':.2},score_kind='probability',exclusive=True)

    def test_discovery_never_loads_until_explicitly_selected(self):
        called=[]
        class Entry:
            name,value,dist='fixture','package:factory',None
            def load(self):
                called.append(1)
                return lambda:SimpleNamespace(modalities=('audio',),process=lambda f:[])
        with patch('humansense.extensions.entry_points',return_value=[Entry()]):
            self.assertEqual(discover_plugins()[0]['name'],'fixture')
            self.assertEqual(called,[])
            load_plugin('fixture')
            self.assertEqual(called,[1])
