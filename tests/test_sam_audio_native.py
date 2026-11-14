"""Strict loader/call fixtures, not official checkpoint acceptance."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from humansense.sam_audio_sources import VerifiedSources
from humansense.sam_audio_source_hashes import SOURCE_HASHES
from humansense.providers import sam_audio_native as native

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_SAFETENSORS = importlib.util.find_spec('safetensors') is not None


class Sources(unittest.TestCase):
    def test_fixed_inventory_and_detached_code_not_caller_authentication(self):
        self.assertEqual({k:len(v) for k,v in SOURCE_HASHES.items()},dict(sam_audio=20,perception_models=43,dacvae=11))
        self.assertEqual(SOURCE_HASHES['sam_audio']['sam_audio/model/text_encoder.py'],
                         'eead9f300a1d84810010543bd5a915e9998f0003b0e7ced9346698d20f8959c5')
        # A deliberately tiny trusted inventory isolates the importer mechanics;
        # production uses the fixed official inventory, never caller digests.
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root = Path(directory).absolute(); (root/'sam_audio').mkdir()
            data = b'VALUE = 17\n'; path = root/'sam_audio/__init__.py'; path.write_bytes(data)
            trusted = {'sam_audio':{'sam_audio/__init__.py':hashlib.sha256(data).hexdigest()}}
            declaration = dict(sources={'sam_audio':dict(root=str(root),files=trusted['sam_audio'])})
            with patch('humansense.sam_audio_sources.SOURCE_HASHES',trusted):
                finder = VerifiedSources(declaration)
                path.write_text('raise AssertionError("changed disk or stale pyc executed")')
                finder.install()
                try:
                    import sam_audio
                    self.assertEqual(sam_audio.VALUE,17)
                    with self.assertRaises(ModuleNotFoundError): importlib.util.find_spec('sam_audio.extra')
                    with self.assertRaises(RuntimeError): VerifiedSources(declaration)
                finally:
                    finder.remove(); sys.modules.pop('sam_audio',None)
                forged = dict(sources={'sam_audio':dict(root=str(root),files={'sam_audio/__init__.py':'0'*64})})
                with self.assertRaises(ValueError): VerifiedSources(forged)

    def test_pinned_zero_byte_package_and_worker_dependency_closure(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).absolute();(root/'core').mkdir();(root/'core/__init__.py').write_bytes(b'')
            trusted={'perception_models':{'core/__init__.py':SOURCE_HASHES['perception_models']['core/__init__.py']}}
            declaration=dict(sources={'perception_models':dict(root=str(root),files=trusted['perception_models'])})
            with patch('humansense.sam_audio_sources.SOURCE_HASHES',trusted):
                finder=VerifiedSources(declaration);finder.install()
                try:
                    import core
                    self.assertEqual(core.__file__,str(root/'core/__init__.py'))
                    self.assertIs(core.__loader__,finder)
                finally: finder.remove();sys.modules.pop('core',None)
        proxy=types.SimpleNamespace(modules={'fixture_dependency':types.SimpleNamespace(__file__='/fixture/dependency.py')},
                                   stdlib_module_names=frozenset())
        with patch.object(native,'sys',proxy),patch.object(native.importlib.metadata,'packages_distributions',return_value={'fixture_dependency':['fixture-dist']}),\
             patch.object(native.importlib.metadata,'version',return_value='1.0'):
            with self.assertRaises(RuntimeError): native._runtime_closure({'runtime':{'packages':{}}})
            native._runtime_closure({'runtime':{'packages':{'fixture-dist':'1.0'}}})
            with self.assertRaises(RuntimeError): native._runtime_closure({'runtime':{'packages':{'fixture-dist':'2.0'}}})

    def test_aggregate_assets_and_private_staging_fail_before_loader(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).absolute();(root/'asset.py').write_bytes(b'abc')
            declaration=dict(sources={'fixture':dict(root=str(root),files={'asset.py':'0'*64})},
                sam=dict(config_path=str(root/'asset.py'),checkpoint_path=str(root/'asset.py')),
                t5=dict(root=str(root),files={'asset.py':'0'*64}))
            with patch.object(native,'MAX_SERIALIZED_ASSETS',11):
                with self.assertRaises(ValueError): native._asset_budget(declaration)
            with patch.object(native,'MAX_SERIALIZED_ASSETS',12): native._asset_budget(declaration)
            with patch.dict(os.environ,GAMST_SAM_TOKENIZER_DIR=''):
                with self.assertRaises(RuntimeError): native._local_tokenizer(object(),declaration)
            stage=root/'stage';stage.mkdir(mode=0o700);stage.chmod(0o755)
            with patch.dict(os.environ,GAMST_SAM_TOKENIZER_DIR=str(stage)):
                with self.assertRaises(ValueError): native._local_tokenizer(object(),declaration)
                stage.chmod(0o700);(stage/'unexpected').write_bytes(b'x')
                with self.assertRaises(ValueError): native._local_tokenizer(object(),declaration)

    def test_tokenizer_config_cannot_redirect_assets_or_activate_slow_conversion(self):
        base={'tokenizer_class':'T5TokenizerFast','model_max_length':512,'extra_ids':0,
              'add_prefix_space':None,'pad_token':{'content':'<pad>','special':True}}
        native._tokenizer_configuration(json.dumps(base).encode())
        native._tokenizer_configuration(json.dumps(dict(base,model_max_length=1000000000000000019884624838656,extra_special_tokens={})).encode())
        for change in ({'tokenizer_file':'/outside.json'},{'vocab_file':'/outside.model'},
                       {'fast_tokenizer_files':['/outside.json']},{'tokenizer_config_file':'/outside.json'},
                       {'name_or_path':'remote-id'},{'from_slow':True},{'add_prefix_space':True},
                       {'tokenizer_class':'CustomTokenizer'},{'extra_ids':1000000},{'unknown':{}},
                       {'added_tokens_decoder':{'1':{'content':{'content':'nested'}}}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                native._tokenizer_configuration(json.dumps(dict(base,**change)).encode())

    def test_immutable_staged_tokenizer_rejects_unhashed_asset_reference_before_factory(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).absolute();assets=root/'assets';assets.mkdir();stage=root/'stage';stage.mkdir(mode=0o700)
            (assets/'tokenizer.json').write_text('{}')
            outside=root/'outside.json';outside.write_text('{"not":"declared"}')
            for field,value in (('tokenizer_file',str(outside)),('vocab_file',str(outside)),('fast_tokenizer_files',[str(outside)])):
                data=json.dumps({'tokenizer_class':'T5TokenizerFast',field:value}).encode()
                (assets/'tokenizer_config.json').write_bytes(data)
                declaration={'t5':{'root':str(assets),'files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in assets.iterdir()}}}
                factory=Mock(side_effect=AssertionError('redirected asset must never reach builtin constructor'))
                with patch.dict(os.environ,GAMST_SAM_TOKENIZER_DIR=str(stage)):
                    with self.assertRaises(ValueError): native._local_tokenizer(types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=factory)),declaration)
                factory.assert_not_called();self.assertEqual(list(stage.iterdir()),[])


@unittest.skipUnless(HAS_TORCH,'optional Torch unavailable')
class TensorPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_exact_tensor_coverage_shape_dtype_finite_and_meta_budget(self):
        t = self.torch
        expected = {'x':((2,),t.float32)}
        native._state({'x':t.ones(2)},expected,t)
        for state in ({},{'x':t.ones(2),'extra':t.ones(1)},{'x':t.ones(3)},
                      {'x':t.ones(2,dtype=t.float64)},{'x':t.tensor([1,float('nan')])}):
            with self.assertRaises(ValueError): native._state(state,expected,t)
        with self.assertRaises(ValueError): native._plan(t.nn.Linear(100000,100000,device='meta'),t)
        for rates in ([12,10,8,2],[1,1920]):
            native._codec_bounds(types.SimpleNamespace(encoder_rates=[1920],decoder_rates=rates,sample_rate=48000),1920)
        for rates in ([10]*16,[12,10,8,3],[],[True,1920],[-1,-1920]):
            with self.assertRaises(ValueError): native._codec_bounds(types.SimpleNamespace(encoder_rates=[1920],decoder_rates=rates,sample_rate=48000),1920)
        for config in ({'num_layers':10000000},{'n_heads':1000000},{'dimension':10000001}):
            with self.assertRaises(ValueError): native._config_bounds(config)

    def test_torch_archive_hash_and_storage_before_loading(self):
        t = self.torch
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path = Path(directory)/'checkpoint.pt'
            t.save({'weight':t.ones(2)},path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(t.equal(native._torch_checkpoint(str(path.absolute()),digest,{'weight':((2,),t.float32)},t)['weight'],t.ones(2)))
            with self.assertRaises(ValueError): native._torch_checkpoint(str(path.absolute()),'0'*64,{'weight':((2,),t.float32)},t)
            with patch.object(t,'load',wraps=t.load) as load:
                with self.assertRaises(ValueError): native._torch_checkpoint(str(path.absolute()),digest,{'weight':((3,),t.float32)},t)
                self.assertEqual([call.kwargs['map_location'] for call in load.call_args_list],['meta'])
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('archive/data.pkl',b'x'*100)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError),patch.object(t,'load',side_effect=AssertionError('must reject before load')):
                native._torch_checkpoint(str(path.absolute()),digest,{'weight':((2,),t.float32)},t)

    def test_declared_storage_extent_and_zip_records_precede_cpu_materialization(self):
        t=self.torch
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory)/'checkpoint.pt';t.save({'weight':t.ones(2)},path)
            with zipfile.ZipFile(path) as archive: records=[(info.filename,archive.read(info)) for info in archive.infolist()]
            for source,replacement in ((b'K\x02t',b'M\xe8\x03t'), # storage count2 ->1000
                                       (b'QK\x00K\x02',b'QM\xe8\x03K\x02'), # offset0 ->1000
                                       (b'K\x01\x85',b'M\xe8\x03\x85')): # stride1 ->1000
                with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
                    for name,data in records:
                        if name.endswith('/data.pkl'):
                            self.assertIn(source,data);data=data.replace(source,replacement,1)
                        archive.writestr(name,data)
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                for bound in (2048,8192):
                    with patch.object(native,'MAX_STORAGE',bound),patch.object(t,'load',wraps=t.load) as load:
                        with self.assertRaises(ValueError): native._torch_checkpoint(str(path.absolute()),digest,{'weight':((2,),t.float32)},t)
                        self.assertEqual([call.kwargs['map_location'] for call in load.call_args_list],['meta'])
            shared=t.ones(4);state={'weight':shared[:2],'other':shared[2:]}
            t.save(state,path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            actual=native._torch_checkpoint(str(path.absolute()),digest,{key:((2,),t.float32) for key in state},t)
            self.assertTrue(t.equal(actual['weight'],t.ones(2)))
            state=__import__('collections').OrderedDict(weight=t.ones(2));state._metadata={'':{'version':1}}
            t.save(state,path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            native._torch_checkpoint(str(path.absolute()),digest,{'weight':((2,),t.float32)},t)
            state._metadata={'':{'hidden':t.ones(1000)}}
            t.save(state,path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(t,'load',wraps=t.load) as load:
                with self.assertRaises(ValueError): native._torch_checkpoint(str(path.absolute()),digest,{'weight':((2,),t.float32)},t)
                self.assertEqual([call.kwargs['map_location'] for call in load.call_args_list],['meta'])

    def test_fake_planning_does_not_materialize_real_storage(self):
        t=self.torch
        real_open=t.serialization._open_zipfile_reader;real_load=t.load
        phase={'value':None};reads=[]
        class Proxy:
            def __init__(self,reader): self.reader=reader
            def __getattr__(self,name): return getattr(self.reader,name)
            def get_storage_from_record(self,*args):
                reads.append(phase['value'])
                if phase['value']!='cpu': raise AssertionError('planning read CPU tensor storage')
                return self.reader.get_storage_from_record(*args)
        class Reader:
            def __init__(self,path): self.reader=real_open(path)
            def __enter__(self): return Proxy(self.reader.__enter__())
            def __exit__(self,*args): return self.reader.__exit__(*args)
        def load(*args,**kwargs):
            phase['value']=kwargs['map_location']
            try: return real_load(*args,**kwargs)
            finally: phase['value']=None
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory)/'checkpoint.pt';t.save({'weight':t.ones(2)},path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(t.serialization,'_open_zipfile_reader',Reader),patch.object(t,'load',side_effect=load):
                native._torch_checkpoint(str(path.absolute()),digest,{'weight':((2,),t.float32)},t)
            self.assertTrue(reads);self.assertEqual(set(reads),{'cpu'})

    @unittest.skipUnless(HAS_SAFETENSORS,'optional safetensors unavailable')
    def test_t5_explicit_tied_alias_and_no_full_decoder_ignore(self):
        from safetensors.torch import save_file
        t = self.torch
        expected = {'shared.weight':((2,2),t.float32),'encoder.embed_tokens.weight':((2,2),t.float32)}
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path = Path(directory)/'model.safetensors'
            def read(values):
                save_file(values,str(path));digest=hashlib.sha256(path.read_bytes()).hexdigest()
                return native._t5_state(str(path.absolute()),digest,expected,t)
            state = read({'shared.weight':t.ones(2,2)})
            self.assertTrue(t.equal(state['shared.weight'],state['encoder.embed_tokens.weight']))
            for values in ({'shared.weight':t.ones(2,2),'decoder.weight':t.ones(1)},
                           {'shared.weight':t.ones(2,2),'encoder.embed_tokens.weight':t.zeros(2,2)},
                           {'shared.weight':t.ones(2,2,dtype=t.float64)},
                           {'shared.weight':t.full((2,2),float('inf'))}):
                with self.assertRaises(ValueError): read(values)

    def test_offline_guard_blocks_sockets_hub_and_auxiliary_without_parent_env_leak(self):
        import socket
        try: import huggingface_hub
        except ImportError: self.skipTest('optional Hub unavailable')
        before = dict(os.environ)
        with native.offline():
            for action in (lambda:socket.create_connection(('127.0.0.1',1)),
                           lambda:huggingface_hub.snapshot_download('must-not-fetch')):
                with self.assertRaises(RuntimeError): action()
            with self.assertRaises(ModuleNotFoundError): importlib.util.find_spec('laion_clap')
        self.assertEqual(dict(os.environ),before)


@contextmanager
def module_overrides(mapping):
    absent=object()
    previous={key:sys.modules.get(key,absent) for key in mapping}
    sys.modules.update(mapping)
    try: yield
    finally:
        for key,value in previous.items():
            if value is absent: sys.modules.pop(key,None)
            else: sys.modules[key]=value


@contextmanager
def pinned_processor(path):
    """Execute authenticated official processor with inert optional imports.

    No model graph, pretrained asset, network or package installation occurs.
    """
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=SOURCE_HASHES['sam_audio']['sam_audio/processor.py']:
        raise ValueError('processor fixture is not the pinned official file')
    config = types.ModuleType('sam_audio.model.config')
    config.SAMAudioConfig = config.SAMAudioJudgeConfig = object
    decoders = types.ModuleType('torchcodec.decoders')
    decoders.AudioDecoder = decoders.VideoDecoder = object
    package = types.ModuleType('sam_audio');package.__path__=[]
    model = types.ModuleType('sam_audio.model');model.__path__=[]
    transformers = types.ModuleType('transformers')
    transformers.AutoTokenizer=object;transformers.BatchFeature=dict
    with module_overrides({'sam_audio':package,'sam_audio.model':model,
        'sam_audio.model.config':config,'torchcodec.decoders':decoders,'transformers':transformers}):
        module = types.ModuleType('_pinned_sam_processor_fixture')
        exec(compile(data,str(path),'exec'),module.__dict__)
        yield module.SAMAudioProcessor


@unittest.skipUnless(HAS_TORCH,'optional Torch unavailable')
class ActualProcessorParity(unittest.TestCase):
    def test_pinned_processor_mono_ceil_anchors_masks_and_padding(self):
        path = os.environ.get('GAMST_SAM_PROCESSOR_SOURCE')
        if not path: self.skipTest('explicit authenticated local upstream processor source not supplied')
        import numpy as np
        import torch
        from humansense.sam_audio_windows import prepare_window
        with pinned_processor(path) as Processor:
            processor = Processor(1920,48000)
            for count,anchors in ((48000,None),(48001,[['+',0,.2],['-',.2,.5]]),(48100,[['+',.001,.2]])):
                window=prepare_window(np.full(count,.25,np.float32),sample_rate=48000,prompt='fixture',hop_length=1920,anchors=anchors)
                batch=processor(descriptions=[window.prompt],audios=[torch.from_numpy(window.array()).reshape(1,-1)],anchors=[list(window.anchors)] if anchors else None)
                self.assertEqual(batch.audios.numpy().tobytes(),window.pcm)
                self.assertEqual(batch.sizes.tolist(),[window.padded_samples//1920])
                self.assertTrue(batch.audio_pad_mask.all().item())
                expected=torch.zeros_like(batch.anchor_alignment)
                ids=[0,3]
                for index,(token,start,end) in enumerate(window.feature_spans):
                    ids.append(1 if token=='+' else 2);expected[0,start:end]=index+2
                self.assertEqual(batch.anchor_ids.tolist(),[ids])
                self.assertTrue(torch.equal(batch.anchor_alignment,expected))


class FixtureTokenizer:
    def __call__(self,texts,**kwargs):
        import torch
        if not kwargs.get('truncation'): return {'input_ids':[[1,2,3,4,5,6]]}
        return {'input_ids':torch.tensor([[1,2,3,4]]),'attention_mask':torch.ones(1,4,dtype=torch.long)}


class FixtureProcessor:
    def __call__(self,descriptions,audios,anchors):
        import torch
        count=audios[0].shape[-1];features=(count+1919)//1920
        ids=[0,3];alignment=torch.zeros(1,features,dtype=torch.long)
        import math
        for index,(token,start,end) in enumerate(anchors[0] if anchors else []):
            ids.append(1 if token=='+' else 2)
            alignment[0,math.ceil(start*48000/1920):math.ceil(end*48000/1920)]=index+2
        batch=types.SimpleNamespace(audios=audios[0].reshape(1,1,-1),sizes=torch.tensor([features]),
            wav_sizes=torch.tensor([count]),audio_pad_mask=torch.ones(1,features,dtype=torch.bool),
            anchor_ids=torch.tensor([ids]),anchor_alignment=alignment,descriptions=descriptions)
        batch.to=lambda device:batch
        return batch


@unittest.skipUnless(HAS_TORCH,'optional Torch unavailable')
class ActualCallBoundary(unittest.TestCase):
    def backend(self):
        import threading
        import torch
        backend=native.SAMAudioNative()
        backend.lock=threading.RLock();backend.hop_length=1920
        backend.config=dict(device='cpu',ode_steps=4,random_seed=7)
        backend.processor=FixtureProcessor()
        class Model:
            text_encoder=types.SimpleNamespace(tokenizer=native._Tokenizer(FixtureTokenizer()))
            def separate(self,batch,**kwargs):
                self.called=kwargs
                self.text_encoder.tokenizer(batch.descriptions,truncation=True,max_length=4,padding='longest',return_tensors='pt')
                length=int(batch.sizes[0])*1920
                return types.SimpleNamespace(target=[torch.full((length,),.125)],residual=[torch.full((length,),-.5)])
        backend.model=Model()
        return backend

    def test_actual_separate_signature_tokenizer_evidence_and_trimmed_pair(self):
        import numpy as np
        from humansense.sam_audio_windows import prepare_window
        backend=self.backend()
        window=prepare_window(np.full(48001,.25,np.float32),sample_rate=48000,prompt='private fixture',hop_length=1920,anchors=[['+',0,.2]])
        target,residual,tokens,elapsed=backend.predict(window)
        self.assertEqual(len(target),48001*4);self.assertEqual(len(residual),48001*4)
        self.assertEqual(tokens,dict(token_count_before=6,token_count_after=4,max_tokens=4,truncated=True))
        self.assertEqual(backend.model.called,dict(predict_spans=False,reranking_candidates=1,
            ode_opt={'method':'midpoint','options':{'step_size':.25}}))
        self.assertGreaterEqual(elapsed,0)

    def test_malformed_native_output_or_processor_cannot_be_published(self):
        import numpy as np
        import torch
        from humansense.sam_audio_windows import prepare_window
        window=prepare_window(np.full(48000,.25,np.float32),sample_rate=48000,prompt='fixture',hop_length=1920)
        for values in (torch.ones(48000,dtype=torch.float64),torch.full((48000,),float('nan')),torch.ones(47999)):
            backend=self.backend()
            backend.model.separate=lambda *a,**kw:types.SimpleNamespace(target=[values],residual=[torch.ones(48000)])
            with self.assertRaises(ValueError): backend.predict(window)
        backend=self.backend();backend.processor=lambda **kwargs:types.SimpleNamespace(audios=torch.zeros(1,2,48000))
        with self.assertRaises((ValueError,AttributeError)): backend.predict(window)


@unittest.skipUnless(HAS_TORCH and HAS_SAFETENSORS,'optional tensor readers unavailable')
class ConstructorFixture(unittest.TestCase):
    def test_local_constructor_exact_state_local_tokenizer_and_override_bypass(self):
        """Real tiny tensors/serialization and production constructor, no SAM assets."""
        from safetensors.torch import save_file
        import torch
        from copy import deepcopy
        class Encoder(torch.nn.Module):
            def __init__(self,config):
                super().__init__();self.shared=torch.nn.Embedding(2,2)
                self.encoder=torch.nn.Module();self.encoder.embed_tokens=self.shared
        class EncoderBase(torch.nn.Module): pass
        upstream=types.ModuleType('sam_audio.model.model');upstream.T5TextEncoder=EncoderBase
        class Graph(torch.nn.Module):
            def __init__(self,config):
                super().__init__();self.text_encoder=upstream.T5TextEncoder(types.SimpleNamespace(pad_mode='longest',max_length=4))
                self.core=torch.nn.Parameter(torch.zeros(2));self.audio_codec=torch.nn.Module()
                self.audio_codec.weight=torch.nn.Parameter(torch.zeros(1));self.audio_codec.hop_length=1920;self.sample_rate=48000
            def load_state_dict(self,*args,**kwargs):
                raise AssertionError('must bypass permissive upstream override')
        upstream.SAMAudio=Graph
        transformer=types.ModuleType('transformers');transformer.T5Config=lambda **config:config;transformer.T5EncoderModel=Encoder;transformer.T5TokenizerFast=FixtureTokenizer
        calls=[]
        def tokenizer(root,**kwargs):
            calls.append((root,kwargs));self.assertEqual((Path(root)/'config.json').read_bytes(),b'{}');return FixtureTokenizer()
        transformer.AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer)
        configmod=types.ModuleType('sam_audio.model.config');configmod.SAMAudioConfig=lambda **config:types.SimpleNamespace(audio_codec=types.SimpleNamespace(encoder_rates=[2,8,10,12],decoder_rates=[12,10,8,2],sample_rate=48000))
        textmod=types.ModuleType('sam_audio.model.text_encoder');textmod.T5TextEncoder=EncoderBase
        processormod=types.ModuleType('sam_audio.processor');processormod.SAMAudioProcessor=lambda *args:FixtureProcessor()
        package=types.ModuleType('sam_audio');package.__path__=[]
        modelpackage=types.ModuleType('sam_audio.model');modelpackage.__path__=[];modelpackage.model=upstream
        class Finder:
            installed=False
            instances=[]
            def __init__(self,*args): self.instances.append(self)
            def install(self): self.installed=True
            def remove(self): self.installed=False
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).absolute();(root/'config.json').write_text('{}')
            (root/'tokenizer_config.json').write_text('{"tokenizer_class":"T5TokenizerFast"}')
            (root/'tokenizer.json').write_text('{}')
            t5path=root/'model.safetensors';save_file({'shared.weight':torch.full((2,2),3.)},str(t5path))
            checkpoint=root/'sam.pt';torch.save({'core':torch.full((2,),7.),'audio_codec.weight':torch.full((1,),9.)},checkpoint)
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            declaration=dict(sources={},runtime=dict(packages={key:'fixture' for key in native.REQUIRED_PACKAGES}),
                t5=dict(root=str(root),files={name:digest(root/name) for name in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors')}),
                sam=dict(config_path=str(root/'config.json'),checkpoint_path=str(checkpoint),checkpoint_sha256=digest(checkpoint)))
            report=dict(effective_config={},hop_length=1920,**{key:'0'*64 for key in (
                'runtime_manifest_sha256','manifest_sha256','t5_manifest_sha256','original_config_sha256','effective_config_sha256')})
            config=dict(manifest_path=str(root/'manifest.json'),manifest_sha256='0'*64,prompt='fixture',device='cpu',ode_steps=4,random_seed=7)
            modules={'sam_audio':package,'sam_audio.model':modelpackage,'sam_audio.model.model':upstream,
                'sam_audio.model.config':configmod,'sam_audio.model.text_encoder':textmod,
                'sam_audio.processor':processormod,'transformers':transformer}
            proxy=types.SimpleNamespace(version_info=(3,11),modules=sys.modules,meta_path=sys.meta_path)
            stage=root/'stage';stage.mkdir(mode=0o700)
            with patch.dict(os.environ,GAMST_SAM_TOKENIZER_DIR=str(stage)),module_overrides(modules),patch.object(native,'sys',proxy),patch.object(native,'VerifiedSources',Finder),\
                 patch.object(native,'read_manifest',return_value=deepcopy(declaration)) as read,\
                 patch.object(native,'verify_manifest',return_value=report) as verify,\
                 patch.object(native,'verify_runtime') as runtime,patch.object(native,'_runtime_closure') as closure:
                backend=native.SAMAudioNative.from_local(config)
                self.assertTrue(torch.equal(backend.model.core,torch.full((2,),7.)))
                self.assertTrue(torch.equal(backend.model.audio_codec.weight,torch.full((1,),9.)))
                self.assertTrue(torch.equal(backend.model.text_encoder.model.shared.weight,torch.full((2,2),3.)))
                self.assertEqual(len(calls),1)
                self.assertNotEqual(calls[0][0],str(root))
                self.assertEqual(calls[0][0],str(stage))
                self.assertEqual(calls[0][1],dict(local_files_only=True,trust_remote_code=False,use_fast=True,tokenizer_type='t5',
                    tokenizer_file=str(stage/'tokenizer.json'),vocab_file=None,from_slow=False))
                self.assertEqual(verify.call_count,2);runtime.assert_called_once();closure.assert_called_once();read.assert_called_once()
                for name in ('config.json','tokenizer_config.json','tokenizer.json'): (stage/name).unlink()
                stage.rmdir()
                backend.model.text_encoder.tokenizer(['fixture'],truncation=True,max_length=4,padding='longest',return_tensors='pt')
                self.assertEqual(backend.model.text_encoder.tokenizer.stats['token_count_after'],4)
                self.assertTrue(backend.finder.installed);backend.close();self.assertFalse(backend.finder.installed)
                self.assertIsNone(backend.model)
                # Exact on-disk tensor geometry fails and removes the importer;
                # the upstream constructor adaptation is also restored.
                torch.save({'core':torch.ones(3),'audio_codec.weight':torch.ones(1)},checkpoint)
                declaration['sam']['checkpoint_sha256']=digest(checkpoint)
                read.return_value=deepcopy(declaration)
                with self.assertRaises(ValueError): native.SAMAudioNative.from_local(config)
                self.assertFalse(Finder.instances[-1].installed)
                self.assertIs(upstream.T5TextEncoder,EncoderBase)


@unittest.skipUnless(HAS_TORCH,'optional Torch unavailable')
class BuiltinTokenizerSnapshot(unittest.TestCase):
    def test_builtin_tokenizer_repeated_calls_after_parent_snapshot_cleanup(self):
        try:
            import transformers
            from transformers import T5TokenizerFast
            from tokenizers import Tokenizer,models,pre_tokenizers,processors
        except ImportError as error:
            self.skipTest('installed optional Transformers runtime incompatible: '+str(error))
        raw=Tokenizer(models.WordLevel({'<pad>':0,'</s>':1,'<unk>':2,'hello':3},unk_token='<unk>'))
        raw.pre_tokenizer=pre_tokenizers.Whitespace()
        raw.post_processor=processors.TemplateProcessing(single='$A </s>',special_tokens=[('</s>',1)])
        original=T5TokenizerFast(tokenizer_object=raw,eos_token='</s>',unk_token='<unk>',pad_token='<pad>',extra_ids=0)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).absolute();assets=root/'assets';assets.mkdir()
            original.save_pretrained(str(assets));(assets/'config.json').write_text('{"model_type":"t5"}')
            declaration={'t5':{'root':str(assets),'files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in assets.iterdir()}}}
            with tempfile.TemporaryDirectory(dir='.') as stage:
                with patch.dict(os.environ,GAMST_SAM_TOKENIZER_DIR=str(Path(stage).absolute())):
                    tokenizer=native._local_tokenizer(transformers,declaration)
            self.assertFalse(Path(stage).exists())
            wrapper=native._Tokenizer(tokenizer)
            for _ in range(2):
                result=wrapper(['hello hello hello'],truncation=True,max_length=2,padding='longest',return_tensors='pt')
                self.assertEqual(result['input_ids'].shape,(1,2))
                self.assertEqual(wrapper.stats,dict(token_count_before=4,token_count_after=2,max_tokens=2,truncated=True))
