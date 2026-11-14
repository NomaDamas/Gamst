import hashlib
import json
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from humansense.providers.smolvlm_native import (RUNTIME,SPECIAL,SmolVLMNative,
    configuration,verified_file,model_config,build_processor,prepare,load_state,materialize)
CONFIG=dict(checkpoint_path='/explicit/model.safetensors',checkpoint_sha256='a'*64,
    tokenizer_path='/explicit/tokenizer.json',tokenizer_sha256='b'*64,max_new_tokens=2)


def available():
    try:return all(version(k).split('+')[0]==v for k,v in RUNTIME.items())
    except PackageNotFoundError:return False


class Contract(unittest.TestCase):
    def test_discovery_and_configuration(self):
        p=subprocess.run([sys.executable,'-c','from humansense.providers.smolvlm_native import configuration;import sys;configuration('+repr(CONFIG)+');assert not {"torch","numpy","transformers","PIL"}&sys.modules.keys()'],capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        for bad in ({'device':'auto'},{'precision':'float16'},{'precision':'bfloat16'},
            {'checkpoint_path':'model-id'},{'tokenizer_sha256':'X'*64},{'max_new_tokens':True},{'max_new_tokens':129},{'extra':1}):
            with self.assertRaises(ValueError):configuration(dict(CONFIG,**bad))

    def test_bounded_hash_regular_file_without_optional_runtime(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path=Path(directory)/'asset';path.write_bytes(b'owned-fixture')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(verified_file(path,digest,100)[0],path)
            for expected,limit in (('a'*64,100),(digest,2)):
                with self.assertRaises(ValueError):verified_file(path,expected,limit)
            link=Path(directory)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):verified_file(link,digest,100)
            with self.assertRaises(FileNotFoundError):SmolVLMNative.from_local(**CONFIG)


@unittest.skipUnless(available(),'dedicated pinned SmolVLM runtime absent; no install')
class Native(unittest.TestCase):
    def test_actual_tiny_graph_processor_generation_and_strict_state(self):
        import numpy as np
        import torch
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import (SmolVLMConfig,SmolVLMForConditionalGeneration,PreTrainedTokenizerFast,
            SmolVLMProcessor,SmolVLMImageProcessor,SmolVLMVideoProcessor)
        from safetensors import safe_open
        from safetensors.torch import save_file
        previous=torch.get_num_threads();self.addCleanup(torch.set_num_threads,previous);torch.set_num_threads(1)
        # Full production architecture is materialized on meta only; no trained weights.
        with torch.device('meta'):full=SmolVLMForConditionalGeneration(model_config(SmolVLMConfig))
        header=json.loads((Path(__file__).parent/'fixtures/smolvlm_500m_header.json').read_text())
        self.assertEqual({k:list(v.shape) for k,v in full.state_dict().items()},
                         {k:v['shape'] for k,v in header['tensors'].items()})
        self.assertTrue(450_000_000<sum(p.numel() for p in full.parameters())<550_000_000)
        del full
        vocab={'fixture_'+str(i):i for i in range(49280)}
        for token,i in dict(SPECIAL,**{'<|endoftext|>':0}).items():del vocab['fixture_'+str(i)];vocab[token]=i
        raw=Tokenizer(models.WordLevel(vocab,unk_token='<|endoftext|>'));raw.pre_tokenizer=pre_tokenizers.Whitespace()
        tok=PreTrainedTokenizerFast(tokenizer_object=raw,bos_token='<|im_start|>',eos_token='<end_of_utterance>',pad_token='<|im_end|>',unk_token='<|endoftext|>',additional_special_tokens=['<fake_token_around_image>','<image>','<global-img>'],model_input_names=['input_ids','attention_mask'])
        processor=build_processor(tok,SmolVLMProcessor,SmolVLMImageProcessor,SmolVLMVideoProcessor)
        config=SmolVLMConfig(image_token_id=49190,scale_factor=4,pad_token_id=2,
            text_config=dict(model_type='llama',hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,vocab_size=49280,pad_token_id=2,bos_token_id=1,eos_token_id=49279),
            vision_config=dict(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,image_size=512,patch_size=16))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13);model=SmolVLMForConditionalGeneration(config).float().eval()
        images=[np.full((48,80,3),[0,128,255],np.uint8),np.full((80,48,3),32,np.uint8)]
        inputs=prepare(processor,images,[1000,1200])
        self.assertEqual(tuple(inputs['pixel_values'].shape),(1,2,3,512,512))
        np.testing.assert_allclose(inputs['pixel_values'][0,0,:,0,0].numpy(),[1,128/255*2-1,-1],atol=1e-6)
        self.assertEqual(int((inputs['input_ids']==49190).sum()),128)
        for frames,times in ((images,[1,1]),(images,[1,10002]),(images*3,list(range(6))),([np.ma.array(images[0])],[1])):
            with self.assertRaises(ValueError):prepare(processor,frames,times)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path=Path(directory)/'untrained.safetensors';save_file(model.state_dict(),str(path))
            with torch.device('meta'):loaded=SmolVLMForConditionalGeneration(config)
            state=load_state(path,loaded,torch,safe_open,'float32');loaded.load_state_dict(state,strict=True,assign=True);materialize(loaded,'cpu')
            self.assertFalse(any(v.is_meta for v in loaded.buffers()))
            self.assertIs(loaded.model.text_model.rotary_emb.original_inv_freq,loaded.model.text_model.rotary_emb.inv_freq)
            native=SmolVLMNative(loaded,processor,CONFIG)
            try:
                actual=native.predict(images,[1000,1200])
                with torch.inference_mode():reference=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=2,eos_token_id=49279,pad_token_id=2,bos_token_id=1,use_cache=True,return_dict_in_generate=False)
                self.assertEqual(actual['text'],tok.decode(reference[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
                self.assertEqual(actual['output_tokens'],2);self.assertFalse(actual['cache_reused_across_windows'])
                self.assertEqual(native.predict(images,[2000,2200]),actual)
                four=prepare(processor,images*2,[0,1,2,3]);self.assertEqual(tuple(four['pixel_values'].shape),(1,4,3,512,512))
            finally:native.close()
            with self.assertRaises(RuntimeError):native.predict(images,[0,1])
            # Real owned worker, production loader and Context/MCP, substituting
            # ONLY the architecture with this explicitly generated tiny fixture.
            import asyncio
            from unittest.mock import patch
            from humansense import HumanSense,ProcessProvider,SensorStreams,SensorMetadata
            from humansense.plugins import InputFrame
            from humansense.schema import timestamp
            from humansense.mcp import MCPServer
            tokenizer_path=Path(directory)/'tokenizer.json';tok.backend_tokenizer.save(str(tokenizer_path))
            worker_config=dict(CONFIG,checkpoint_path=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                tokenizer_path=str(tokenizer_path),tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),window_frames=2)
            code=('import torch;torch.set_num_threads(1)\n'
                'from humansense.providers import smolvlm_native as n\n'
                'n.model_config=lambda cls:cls.from_dict('+repr(config.to_dict())+')\n'
                'from humansense.provider_worker import main\nmain()')
            async def managed():
                real_spawn=asyncio.create_subprocess_exec
                children=[]
                async def spawn(*args,**kwargs):
                    child=await real_spawn(sys.executable,'-u','-c',code,**kwargs);children.append(child);return child
                with patch('asyncio.create_subprocess_exec',spawn):
                    worker=await ProcessProvider.open(sys.executable,'smolvlm2.500m.local',worker_config,inference_timeout=60)
                try:
                    async with HumanSense(consent=True) as human:
                        streams=SensorStreams(human);streams.register(worker.spec,lambda:worker)
                        await streams.open('camera','video',[worker.spec.id]);start=timestamp()-500
                        try:
                            for i in range(2):
                                f=InputFrame(images[0].copy(),'video',start+i*200,sensor=SensorMetadata('camera','video',i,coordinate_frame='pixels'))
                                await streams.submit(f);await streams.drain()
                            records=human.store.latest_signals('local-user',0,timestamp())
                            self.assertEqual(len(records),1);self.assertEqual(records[0]['timestamp_ms'],start+200)
                            value=records[0]['observation']['value'];self.assertTrue(value['window']['complete'])
                            self.assertEqual(value['model']['checkpoint_sha256'],worker_config['checkpoint_sha256'])
                            self.assertEqual(value['window']['observed_frames'],2)
                            self.assertIn('untrusted_semantic_hypothesis',str(await MCPServer(human).call_tool('get_observations',{'capability_id':worker.spec.id})))
                            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)
                        finally:await streams.clear()
                finally:await worker.close()
                self.assertIsNotNone(children[0].returncode)
            asyncio.run(managed())
            async def on_demand():
                from humansense import SmolVLMCalls,smolvlm_frame_sha256
                from humansense.capabilities import CapabilitySpec,OutputField
                from humansense.call_policy import CallPolicy
                from humansense.call_controller import CallController
                from humansense.call_budget import CallBudget
                real_spawn=asyncio.create_subprocess_exec;children=[]
                async def spawn(*args,**kwargs):
                    child=await real_spawn(sys.executable,'-u','-c',code,**kwargs);children.append(child);return child
                async with HumanSense(consent=True) as human:
                    thin=CapabilitySpec('fixture.detect','fixture.objects','video','fixture','1',(OutputField('count','integer'),),'frame',freshness_ms=1000)
                    human.register_capability(thin)
                    controller=CallController(policy=CallPolicy(mode='sentinel'),budget=CallBudget(unit='fixture',pricing_revision='1',ceiling=2))
                    human.register_call_controller('native',controller)
                    with patch('asyncio.create_subprocess_exec',spawn):
                        adapter=await SmolVLMCalls.open(human,'native',sys.executable,{k:v for k,v in worker_config.items() if k!='window_frames'},inference_timeout=60)
                    async with adapter:
                        ts=timestamp();sensor=SensorMetadata('camera','video',0,coordinate_frame='pixels',stream_epoch='native-1')
                        record=await human.push_observation(thin.id,{'count':1},sensor=sensor,timestamp_ms=ts)
                        source=controller.open_source(record,now_ms=timestamp())
                        frame=InputFrame(images[0].copy(),'video',ts,sensor=sensor)
                        ref=dict(source_id='camera',sensor_id='video',stream_epoch='native-1',capture_ms=ts,sha256=smolvlm_frame_sha256(frame.data))
                        queued=controller.submit(source,record,sequence=0,now_ms=timestamp(),visual_reference=ref,unknown_candidate=True)
                        result=await adapter.execute(queued['request_id'],record,frame,maximum_charge=1)
                        self.assertIn(result['status'],('published','discarded'))
                        self.assertEqual(controller.snapshot()['budget']['dispatched'],1)
                        self.assertEqual(controller.snapshot()['budget']['reserved_liability'],1)
                        if result['status']=='published':
                            self.assertEqual(result['observation']['observation']['value']['model']['checkpoint_sha256'],worker_config['checkpoint_sha256'])
                            self.assertFalse(result['observation']['observation']['value']['semantic']['quality_validated'])
                self.assertIsNotNone(children[0].returncode)
            asyncio.run(on_demand())
            # The same strict loader used for production rejects missing/nonfinite tensors.
            bad=model.state_dict();bad.pop(next(iter(bad)));save_file(bad,str(path))
            with self.assertRaises(ValueError):load_state(path,model,torch,safe_open,'float32')
            bad=model.state_dict();key=next(iter(bad));bad[key]=torch.full_like(bad[key],float('nan'));save_file(bad,str(path))
            with self.assertRaises(ValueError):load_state(path,model,torch,safe_open,'float32')
            bad=model.state_dict();key=next(iter(bad));bad[key]=torch.full_like(bad[key],torch.finfo(torch.float32).max);save_file(bad,str(path))
            with self.assertRaises(ValueError):load_state(path,model,torch,safe_open,'bfloat16')
