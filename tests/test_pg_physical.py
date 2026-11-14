"""PG protocol/source machinery fixtures; never trained PG quality evidence."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_pg_canonical import basic
from humansense.pg_artifact import manifest_digest
from humansense.pg_generation import (PROMPT_SHA,TRANSPORT_ALLOWANCE,policy,pack_capture,unpack_capture,
    validate_result,original_generate,preprocess,infer,HEADER_LIMIT)
from humansense.pg_physical_process import PGPhysicalProcess,capability
from humansense.process_provider import ProcessProvider
from humansense.plugins import InputFrame
from humansense.capabilities import SensorMetadata


def config(root):
    bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
    graph,sha,lock,lock_sha,spec,_=basic(bundle)
    return dict(graph=graph,graph_sha=sha,lock=lock,lock_sha=lock_sha,canonical=spec,canonical_sha=manifest_digest(spec),
        stage_root=str(stage),generation=dict(schema_version=1,prompt_sha256=PROMPT_SHA,execution=graph['execution'],
            transport_allowance_bytes=TRANSPORT_ALLOWANCE,generation_workspace_bytes=1024**3),max_gap_ms=1000)


def frame(seq=1,ts=100,concept='transparency',pixels=None):
    import numpy as np
    if pixels is None:pixels=np.zeros((8,9,3),dtype=np.uint8)
    return InputFrame(pack_capture(pixels,[0,0,pixels.shape[1],pixels.shape[0]],concept,track_provenance='caller.track'),
        'video',ts,subject_id='object1',sensor=SensorMetadata('camera','sensor',seq,'pixels',('R','G','B'),stream_epoch='captureepoch',capture_sequence=seq))


# Real child transport exercises the parent boundary without importing a model.
CHILD=r'''
import hashlib,json,pathlib,sys,time
r=json.loads(pathlib.Path(sys.argv[1]).read_text());mode=sys.argv[2]
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
print(json.dumps(dict(status='loaded',canonical_sha256=r['canonical_sha'],graph_schema_sha256=r['canonical']['graph_schema_sha256'],planned_storage_bytes=4,trained_identity_verified=False,reference_parity_verified=False,runtime_compatibility_verified=False,resource_admission='unavailable',actual_acceptance='incomplete',generation_sha256=digest(r['generation']))),flush=True)
while True:
 line=sys.stdin.buffer.readline()
 if not line:break
 h=json.loads(line);pixels=sys.stdin.buffer.read(h['payload_bytes'])
 assert len(pixels)==h['payload_bytes'] and hashlib.sha256(pixels).hexdigest()==h['pixels_sha256']
 sys.stderr.write('R');sys.stderr.flush()
 if mode=='hang':time.sleep(60)
 if mode=='gated':sys.stdin.buffer.read(1)
 if mode=='oversize':print('x'*70000,flush=True);continue
 if mode=='bad':print('{}',flush=True);continue
 text='unknown' if h['binding']['timestamp_ms']>100 else 'opaque'
 result=dict(alternatives=[dict(text=text,beam_score=-.1)]*3,input_tokens=20,output_tokens=None,inference_ms=1.)
 response=dict(status='generated',request_id=h['request_id'],request_sha256=digest(h),canonical_sha256=r['canonical_sha'],generation_sha256=digest(r['generation']),result=result)
 if mode=='wrong':response['request_sha256']='0'*64
 print(json.dumps(response),flush=True)
'''


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional numpy')
class Contract(unittest.TestCase):
    def test_packet_ownership_geometry_and_policy_abstention(self):
        import numpy as np
        p=np.zeros((5,7,3),dtype=np.uint8);packet=pack_capture(p,[1,1,6,4],'mass',track_provenance='caller')
        p[:]=255;head,raw=unpack_capture(packet);self.assertEqual(raw,bytes(105))
        for bad in (packet+b'x',packet[:-1],b'PGC1'+b'\xff'*4,bytearray(packet)):
            with self.assertRaises(ValueError):unpack_capture(bad)
        for crop in ([True,0,1,1],[0,0,8,1],[0,0,0,1]):
            with self.assertRaises(ValueError):pack_capture(p,crop,'mass',track_provenance='caller')
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());execution=cfg['graph']['execution']
            policy(cfg['generation'],execution)
            row=dict(alternatives=[dict(text='opaque',beam_score=-1.)]*3,input_tokens=2,output_tokens=None,inference_ms=1)
            self.assertEqual(validate_result(row,'transparency',execution)[1],'opaque')
            row['alternatives'][-1]=dict(text='transparent',beam_score=-100)
            self.assertIsNone(validate_result(row,'transparency',execution)[1])
            row['alternatives'][0]['beam_score']=float('nan')
            with self.assertRaises(ValueError):validate_result(row,'transparency',execution)
            self.assertEqual(capability(cfg).streaming_mode,'frame')

    def test_mutating_caller_crop_after_snapshot_cannot_change_owned_packet(self):
        import threading
        import numpy as np
        from humansense import pg_generation
        crop=[0,0,8,8];arr=np.zeros((8,8,3),np.uint8)
        entered=threading.Event();mutated=threading.Event();original=pg_generation.layout
        def mutate():
            if entered.wait(5):crop[:]=[0,0,999,8]+list(range(10000));mutated.set()
        thread=threading.Thread(target=mutate);thread.start()
        def checked(shape,owned):
            self.assertIsNot(owned,crop);entered.set();self.assertTrue(mutated.wait(5))
            return original(shape,owned)
        try:
            with patch('humansense.pg_generation.layout',side_effect=checked):
                packet=pack_capture(arr,crop,'mass',track_provenance='caller')
        finally:entered.set();thread.join(5)
        self.assertFalse(thread.is_alive());head,payload=unpack_capture(packet)
        self.assertEqual(head['crop'],[0,0,8,8]);self.assertEqual(len(payload),192)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional numpy')
class Owned(unittest.IsolatedAsyncioTestCase):
    async def opened(self,cfg,mode='normal',timeout=10):
        real=asyncio.create_subprocess_exec;children=[]
        async def spawn(*args,**kwargs):
            request=args[args.index('--request')+1];kwargs['stderr']=asyncio.subprocess.PIPE
            child=await real(sys.executable,'-u','-c',CHILD,request,mode,**kwargs);children.append(child);return child
        with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn):
            owner=await ProcessProvider.open(sys.executable,'pg.physical.local',cfg,inference_timeout=timeout)
        self.assertIsInstance(owner,PGPhysicalProcess)
        return owner,children[0]

    async def test_actual_child_context_duplicates_concepts_unknown_and_reset(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg)
            try:
                result=await owner.process(frame());self.assertEqual(result['physical']['value'],'opaque')
                self.assertEqual(result['evidence']['timestamp_ms'],100);self.assertIsNone(result['model']['output_tokens'])
                with self.assertRaisesRegex(ValueError,'duplicate'):await owner.process(frame())
                result=await owner.process(frame(concept='material'));self.assertEqual(result['physical']['status'],'unknown')
                result=await owner.process(frame(2,101));self.assertIsNone(result['physical']['value'])
                self.assertEqual(owner._states['object1']['transparency']['status'],'unknown')
                await owner.reset();self.assertEqual(owner._states,{})
                await owner.process(frame())
            finally:await asyncio.gather(owner.close(),owner.close())
            self.assertIsNotNone(child.returncode);self.assertTrue(child._transport.is_closing())
            self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_cancel_one_inflight_and_reap(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg,'hang')
            task=asyncio.create_task(owner.process(frame()))
            await asyncio.wait_for(child.stderr.readexactly(1),10)
            with self.assertRaisesRegex(RuntimeError,'one in-flight'):await owner.process(frame(2,101))
            task.cancel();task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertTrue(owner.closed);self.assertIsNotNone(child.returncode);self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_malformed_mismatch_oversize_and_actual_timeout_poison(self):
        for mode in ('bad','wrong','oversize','hang'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir='.') as temp:
                cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg,mode,timeout=.5 if mode=='hang' else 10)
                with self.assertRaises((ValueError,TimeoutError)):await owner.process(frame())
                self.assertTrue(owner.closed);self.assertIsNotNone(child.returncode);self.assertTrue(child._transport.is_closing())
                self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_maximum_packet_actual_pipe_and_reject_before_state(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg)
            try:
                pixels=np.zeros((2048,4096,3),dtype=np.uint8)
                value=frame(pixels=pixels)
                result=await owner.process(value);self.assertEqual(result['evidence']['shape'],[2048,4096,3])
                prior=deepcopy(owner._states)
                with self.assertRaises(ValueError):await owner.process(InputFrame(value.data+b'x','video',101,sensor=value.sensor))
                self.assertEqual(prior,owner._states);self.assertFalse(owner._busy)
                with self.assertRaises(ValueError):pack_capture(np.zeros((2049,4096,3),dtype=np.uint8),[0,0,4096,2049],'mass',track_provenance='caller')
            finally:await owner.close()
            self.assertIsNotNone(child.returncode)

    async def test_evaluation_observer_is_not_silently_dropped_by_special_dispatch(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve())
            with patch('humansense.pg_physical_process.PGPhysicalProcess.open') as opened:
                with self.assertRaisesRegex(ValueError,'instrumentation is unavailable'):
                    await ProcessProvider.open(sys.executable,'pg.physical.local',cfg,evaluation_observer=object())
                opened.assert_not_called()
            self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_native_missing_runtime_unavailable_not_fixture_result(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve())
            with self.assertRaises(RuntimeError):await ProcessProvider.open(sys.executable,'pg.physical.local',cfg)
            self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_reset_invalidates_late_response_without_releasing_inflight_guard(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg,'gated')
            try:
                task=asyncio.create_task(owner.process(frame()))
                await asyncio.wait_for(child.stderr.readexactly(1),10)
                await owner.reset()
                with self.assertRaisesRegex(RuntimeError,'one in-flight'):await owner.process(frame(2,101))
                child.stdin.write(b'X');await child.stdin.drain()
                self.assertIsNone(await task);self.assertEqual(owner._states,{})
                self.assertFalse(owner.closed);self.assertEqual(owner._sequence,0)
            finally:await owner.close()
            self.assertIsNotNone(child.returncode)

    async def test_external_close_joins_owned_operation_and_repeated_cancellation(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg,'hang')
            caller=asyncio.create_task(owner.process(frame()))
            await asyncio.wait_for(child.stderr.readexactly(1),10)
            operation=owner._operation
            closing=asyncio.create_task(owner.close())
            # Wait on the actual owned close start, not an arbitrary delay.
            while owner._close_task is None:await asyncio.sleep(0)
            closing.cancel();closing.cancel()
            with self.assertRaises(asyncio.CancelledError):await closing
            self.assertTrue(operation.done());self.assertFalse(owner._busy)
            self.assertTrue(owner._close_task.done());self.assertIsNotNone(child.returncode)
            self.assertTrue(child._transport.is_closing())
            with self.assertRaises(asyncio.CancelledError):await caller
            await owner.close();self.assertIsNone(owner._operation)
            self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_cancel_before_owned_operation_first_step_still_reaps(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg,'hang')
            real=asyncio.shield;hooked=[]
            def shield(task):
                if task is owner._operation and not hooked:
                    hooked.append(True);task.cancel();asyncio.current_task().cancel()
                return real(task)
            with patch('humansense.pg_physical_process.asyncio.shield',side_effect=shield):
                caller=asyncio.create_task(owner.process(frame()))
                with self.assertRaises(asyncio.CancelledError):await caller
            self.assertEqual(hooked,[True]);self.assertTrue(owner.closed);self.assertFalse(owner._busy)
            self.assertIsNone(owner._operation);self.assertIsNotNone(child.returncode)
            self.assertEqual(list(Path(cfg['stage_root']).iterdir()),[])

    async def test_state_subject_cap_gap_and_epoch_reset(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg)
            try:
                # State admission mechanics: no model/callback substitution.
                original=frame()
                for index in range(129):
                    header,_=owner._capture(replace(original,subject_id='object'+str(index)))
                    owner._admit(header)
                self.assertEqual(len(owner._states),128);self.assertNotIn('object0',owner._states)
                header,_=owner._capture(replace(original,subject_id='object128',timestamp_ms=1200,
                    sensor=replace(original.sensor,sequence=2,capture_sequence=2)))
                owner._admit(header);self.assertEqual(len(owner._states['object128']),1)
                header,_=owner._capture(replace(original,sensor=replace(original.sensor,stream_epoch='new-epoch')))
                owner._admit(header);self.assertEqual(list(owner._states),['object1'])
            finally:await owner.close()
            self.assertIsNotNone(child.returncode)

    async def test_sdk_stream_mcp_capture_time_and_explicit_packet_budget(self):
        from dataclasses import replace
        from humansense import HumanSense,SensorStreams
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        with tempfile.TemporaryDirectory(dir='.') as temp:
            cfg=config(Path(temp).resolve());owner,child=await self.opened(cfg)
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human,max_packet_bytes=32*1024**2,max_buffer_bytes=64*1024**2,capacity=1)
                streams.register(owner.spec,lambda:owner)
                await streams.open('camera','sensor',[owner.spec.id],subject_id='object1')
                now=timestamp()-100;capture=frame(ts=now)
                capture=replace(capture,sensor=replace(capture.sensor,stream_epoch=None))
                await streams.submit(capture);await streams.drain()
                self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'],0)
                context=await human.get_context(subject_id='object1');value=next(iter(context['observations'].values()))['value']
                self.assertEqual(value['evidence']['timestamp_ms'],now)
                self.assertNotEqual(value['evidence']['worker_epoch'],value['evidence']['sensor']['stream_epoch'])
                server=MCPServer(human)
                response=await server.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'get_context','arguments':{'subject_id':'object1'}}})
                observed=json.loads(response['result']['content'][0]['text'])
                self.assertEqual(next(iter(observed['observations'].values()))['value'],value)
                await streams.close()
            self.assertIsNotNone(child.returncode)


@unittest.skipUnless(os.environ.get('GAMST_PG_SOURCE_ROOT') and importlib.util.find_spec('torchvision'),'explicit pinned sources and existing Torch/vision required')
class OriginalSources(unittest.TestCase):
    def test_original_generation_and_eval_recipe_with_explicit_small_machinery_graph(self):
        import ast
        from contextlib import nullcontext
        from types import SimpleNamespace
        import numpy as np
        import torch
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image
        from humansense.pg_sources import SOURCE_HASHES
        from test_pg_artifact import declaration
        root=Path(os.environ['GAMST_PG_SOURCE_ROOT']);source=root/'pg-generate.py'
        manifest=declaration(root);manifest['sources']['pg']['files']['generate.py'].update(path='pg-generate.py',size=source.stat().st_size)
        for name in ('lavis/processors/blip_processors.py','lavis/configs/models/blip2/blip2_instruct_flant5xxl.yaml'):
            manifest['sources']['lavis_candidate']['files'][name].update(path=name,size=(root/name).stat().st_size)
        generate=original_generate(manifest)
        processor_path=root/'lavis/processors/blip_processors.py';raw=processor_path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),SOURCE_HASHES['lavis_candidate']['lavis/processors/blip_processors.py'])
        tree=ast.parse(raw);classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('BlipImageBaseProcessor','BlipImageEvalProcessor')]
        class BaseProcessor:pass
        for node in classes:node.decorator_list=[]
        ns=dict(BaseProcessor=BaseProcessor,transforms=transforms,InterpolationMode=InterpolationMode)
        exec(compile(ast.Module(body=classes,type_ignores=[]),'original-blip-eval','exec'),ns)
        pixels=np.arange(23*37*3,dtype=np.uint8).reshape(23,37,3);crop=[2,3,34,20]
        expected=ns['BlipImageEvalProcessor'](image_size=224)(Image.fromarray(pixels).crop(tuple(crop))).unsqueeze(0)
        self.assertTrue(torch.equal(preprocess(pixels.tobytes(),list(pixels.shape),crop),expected))
        class Batch:
            input_ids=torch.ones((1,4),dtype=torch.long)
            attention_mask=torch.ones((1,4),dtype=torch.long)
            def to(self,device):return self
        class Tokenizer:
            def __call__(self,*args,**kw):return Batch()
            def batch_decode(self,sequences,**kwargs):return ['opaque']*3
        calls=[]
        def native_generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(sequences=torch.tensor([[1,2]]*3),sequences_scores=torch.tensor([-.1,-.2,-.3]))
        model=SimpleNamespace(qformer_text_input=False,query_tokens=torch.ones((1,2,4)),device=torch.device('cpu'),
            maybe_autocast=lambda **kw:nullcontext(),ln_vision=lambda x:x,visual_encoder=lambda x:torch.ones((1,2,4)),
            Qformer=SimpleNamespace(bert=lambda **kw:SimpleNamespace(last_hidden_state=torch.ones((1,2,4)))),
            t5_proj=lambda x:x,t5_tokenizer=Tokenizer(),t5_model=SimpleNamespace(generate=native_generate,
                encoder=SimpleNamespace(embed_tokens=lambda x:torch.ones((1,4,4)))))
        generation=dict(execution=manifest['execution'])
        header=dict(concept='transparency',shape=list(pixels.shape),crop=crop)
        result=infer(model,generate,manifest,generation,header,pixels.tobytes())
        self.assertEqual(result['input_tokens'],4);self.assertIsNone(result['output_tokens'])
        self.assertEqual(calls[0]['max_new_tokens'],32);self.assertEqual(calls[0]['num_beams'],5)
        self.assertEqual(calls[0]['length_penalty'],0);self.assertEqual(calls[0]['repetition_penalty'],1)
        self.assertTrue(calls[0]['output_scores']);self.assertFalse(calls[0]['do_sample'])
        self.assertEqual(tuple(calls[0]['inputs_embeds'].shape),(1,6,4))
        generation['execution']['prompt_tokens']=3
        with self.assertRaisesRegex(ValueError,'tokenizer prompt limit'):infer(model,generate,manifest,generation,header,pixels.tobytes())
        self.assertEqual(len(calls),1)
