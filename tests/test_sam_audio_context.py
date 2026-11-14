"""Native integration fixtures only; no pretrained model or quality evidence."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.sam_audio_context import SAMAudioContext, configuration
from humansense.schema import timestamp
from humansense.separation_media import PROFILE

try:
    import numpy as np
except ImportError:
    np = None

CONFIG = dict(manifest_path='/fixture/not-a-model/manifest.json',manifest_sha256='0'*64,
              prompt='fixture ringing',anchors=[['+',0,.2]],ode_steps=4,random_seed=7)


class Discovery(unittest.TestCase):
    def test_strict_config_and_dependency_light_discovery(self):
        result = subprocess.run([sys.executable,'-c',
            'from humansense import ProcessProvider;import sys;ProcessProvider.capability('+repr(PROFILE)+','+repr(CONFIG)+');assert not {"numpy","torch","transformers","sam_audio"}&sys.modules.keys()'],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        before = deepcopy(CONFIG)
        config = configuration(CONFIG)
        config['anchors'][0][1] = .1
        self.assertEqual(CONFIG,before)
        for change in ({'fixture':True},{'device':'mps'},{'ode_steps':True},{'ode_steps':65},
                       {'random_seed':-1},{'manifest_path':'model-id'},{'manifest_path':'/fixture/../manifest.json'},
                       {'manifest_sha256':'bad'},
                       {'anchors':[['+',0,float('nan')]]},{'prompt':'x'*4097}):
            with self.subTest(change=change),self.assertRaises(ValueError): configuration(dict(CONFIG,**change))
        self.assertEqual(SAMAudioContext.capability(CONFIG),SAMAudioContext.capability(configuration(CONFIG)))
        self.assertNotEqual(SAMAudioContext.capability(CONFIG).version,
                            SAMAudioContext.capability(dict(CONFIG,prompt='another fixture')).version)


class FixtureBackend:
    fixture_engine = True
    hop_length = 1920
    provenance = dict(source_revision='bb4c6999d2677c7402360e426afc01ddfad6dce0',
        runtime_manifest_sha256='0'*64,sam_manifest_sha256='1'*64,t5_manifest_sha256='2'*64,
        checkpoint_sha256='3'*64,original_config_sha256='4'*64,effective_config_sha256='5'*64,
        device='cpu',precision='float32',predict_spans=False,reranking_candidates=1,
        ode_steps=4,ode_method='midpoint',random_seed=7)
    def predict(self,window):
        self.last = window
        pcm = window.array()
        return (pcm*.5).astype('<f4').tobytes(),(pcm*-.25).astype('<f4').tobytes(),dict(
            token_count_before=8,token_count_after=4,max_tokens=4,truncated=True),2.5
    def close(self): self.closed = True


@unittest.skipIf(np is None,'optional NumPy unavailable')
class Context(unittest.TestCase):
    def setUp(self):
        self.backend = FixtureBackend()
        self.context = SAMAudioContext(self.backend,CONFIG)

    def tearDown(self): self.context.close()

    def frame(self,sequence=0,end=10000,epoch='epoch',count=48000):
        return InputFrame(np.full(count,.25,np.float32),'audio',end,'alice',48000,
            SensorMetadata('mic','device',sequence,channels=('mono',),stream_epoch=epoch))

    def test_capture_prompt_padding_and_ambient_preservation(self):
        frame = self.frame(count=48001)
        before = frame.data.tobytes()
        pair = self.context.process(frame)
        self.assertEqual(frame.data.tobytes(),before)
        self.assertEqual(pair.padded_samples,49920)
        self.assertEqual(len(pair.target),len(before))
        obs = pair.observation
        self.assertEqual(obs['window']['capture_start_ms'],10000-48001/48)
        self.assertEqual(obs['window']['trimmed_padding_samples'],1919)
        self.assertEqual(obs['prompt']['feature_spans'],[['+',0,5]])
        self.assertTrue(obs['prompt']['truncated'])
        self.assertNotIn(CONFIG['prompt'],json.dumps(obs))
        self.assertEqual(obs['media'],{})
        self.assertEqual(obs['separation']['status'],'fixture-estimate')
        self.assertFalse(obs['separation']['quality_validated'])

    def test_order_gap_reset_reconnect_and_no_partial_tail(self):
        self.context.process(self.frame())
        pair = self.context.process(self.frame(1,11000))
        self.assertFalse(pair.observation['window']['discontinuity'])
        pair = self.context.process(self.frame(3,14000))
        self.assertEqual(pair.observation['window']['discontinuity_reason'],'gap')
        for frame in (self.frame(3,15000),self.frame(4,14000),self.frame(4,15000,'new')):
            with self.assertRaises(ValueError): self.context.process(frame)
        self.context.reset()
        pair = self.context.process(self.frame(0,15000,'new'))
        self.assertEqual(pair.observation['window']['discontinuity_reason'],'reset')
        self.assertIsNone(self.context.finish())

    def test_bad_capture_and_native_bytes_do_not_advance_state(self):
        for frame in (self.frame(count=47999),self.frame(epoch=None)):
            with self.assertRaises(ValueError): self.context.process(frame)
        self.backend.predict = lambda window:(b'bad',b'bad',{},0)
        with self.assertRaises(ValueError): self.context.process(self.frame())
        self.assertIsNone(self.context._previous)


FIXTURE_CHILD = '''
from humansense import provider_worker as w
from humansense.providers.sam_audio_native import SAMAudioNative
from test_sam_audio_context import FixtureBackend
SAMAudioNative.from_local=classmethod(lambda cls,config:FixtureBackend())
w.main()
'''


@unittest.skipIf(np is None,'optional NumPy unavailable')
class OwnedContext(unittest.IsolatedAsyncioTestCase):
    async def test_real_profile_context_owned_worker_and_metadata_only_mcp(self):
        from humansense.mcp import MCPServer
        children = [];stage_paths=[]
        real_spawn = asyncio.create_subprocess_exec
        async def spawn(*args,**kwargs):
            # Only the native loader is replaced by a clearly labeled fixture.
            # Discovery, _construct, Context, framing and parent ownership run.
            import os
            env = dict(kwargs.get('env',{}))
            env['PYTHONPATH'] = os.path.abspath('tests')+os.pathsep+os.path.abspath('.')
            kwargs['env'] = env
            stage_paths.append(Path(env['GAMST_SAM_TOKENIZER_DIR']))
            child = await real_spawn(sys.executable,'-u','-c',FIXTURE_CHILD,**kwargs)
            children.append(child)
            return child
        async with HumanSense(consent=True) as human:
            owner = human.enable_separation_media(subject_id='alice',source_id='mic',sensor_id='device')
            with patch('asyncio.create_subprocess_exec',spawn):
                provider = await ProcessProvider.open(sys.executable,PROFILE,CONFIG,separation_owner=owner)
            self.assertTrue(stage_paths and all(not path.exists() for path in stage_paths))
            streams = SensorStreams(human)
            streams.register(provider.spec,lambda:provider)
            await streams.open('mic','device',[provider.spec.id],subject_id='alice')
            try:
                pcm = np.full(48000,.25,np.float32); before = pcm.tobytes()
                await streams.submit(InputFrame(pcm,'audio',timestamp(),'alice',48000,
                    SensorMetadata('mic','device',0,channels=('mono',))))
                await streams.drain()
                context = await human.get_context(subject_id='alice')
                value = next(iter(context['observations'].values()))['value']
                epoch = streams.status()['streams'][0]['providers'][0]['stream_epoch']
                target = owner.retrieve(value['media']['reference'],'target',stream_epoch=epoch)
                self.assertEqual(target,(pcm*.5).astype('<f4').tobytes())
                self.assertEqual(pcm.tobytes(),before)
                self.assertTrue(value['model']['fixture_engine'])
                self.assertEqual(value['window']['source_evidence']['sensor']['stream_epoch'],epoch)
                response = await MCPServer(human).handle(dict(jsonrpc='2.0',id=1,method='tools/call',
                    params=dict(name='get_context',arguments=dict(subject_id='alice'))))
                self.assertNotIn(CONFIG['prompt'],json.dumps(response))
                self.assertNotIn('\"target\": [',json.dumps(response))
                # A capture gap runs the real parent reset, changes provider
                # epoch and revokes the first pair before admitting a new one.
                reference=value['media']['reference']
                await streams.submit(InputFrame(pcm,'audio',timestamp(),'alice',48000,
                    SensorMetadata('mic','device',2,channels=('mono',))))
                await streams.drain()
                new_epoch=streams.status()['streams'][0]['providers'][0]['stream_epoch']
                values=(await human.get_context(subject_id='alice'))['observations'].values()
                value=next(item['value'] for item in values if item['value']['window']['source_evidence']['sensor']['stream_epoch']==new_epoch)
                self.assertNotEqual(new_epoch,epoch)
                self.assertEqual(value['window']['discontinuity_reason'],'reset')
                with self.assertRaises(PermissionError): owner.retrieve(reference,'target',stream_epoch=epoch)
                self.assertEqual(owner.retrieve(value['media']['reference'],'target',stream_epoch=new_epoch),target)
            finally:
                await streams.close()
            self.assertEqual(owner._store.bytes,0)
            self.assertTrue(all(child.returncode is not None for child in children))

    async def test_parent_removes_private_startup_snapshots_after_cancel_or_timeout(self):
        from pathlib import Path
        import os
        code='''
import os,time
from pathlib import Path
from humansense import provider_worker as w
from humansense.providers.sam_audio_native import SAMAudioNative
def fixture_load(cls,config):
    (Path(os.environ['GAMST_SAM_TOKENIZER_DIR'])/'fixture-consumed').write_bytes(b'fixture tokenizer snapshot')
    time.sleep(30)
SAMAudioNative.from_local=classmethod(fixture_load)
w.main()
'''
        real_spawn=asyncio.create_subprocess_exec
        for cancel in (True,False):
            children=[];paths=[]
            async def spawn(*args,**kwargs):
                path=Path(kwargs['env']['GAMST_SAM_TOKENIZER_DIR']);paths.append(path)
                self.assertEqual(path.stat().st_mode & 0o777,0o700)
                child=await real_spawn(sys.executable,'-u','-c',code,**kwargs)
                children.append(child);return child
            async with HumanSense(consent=True) as human:
                owner=human.enable_separation_media(subject_id='alice',source_id='mic',sensor_id='device')
                with patch('asyncio.create_subprocess_exec',spawn):
                    task=asyncio.create_task(ProcessProvider.open(sys.executable,PROFILE,CONFIG,separation_owner=owner,
                        startup_timeout=5 if cancel else .5))
                    if cancel:
                        for _ in range(200):
                            if paths and (paths[0]/'fixture-consumed').exists(): break
                            await asyncio.sleep(.01)
                        self.assertTrue(paths and (paths[0]/'fixture-consumed').exists())
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else asyncio.TimeoutError): await task
                self.assertTrue(paths and all(not path.exists() for path in paths))
                self.assertTrue(children and all(child.returncode is not None for child in children))
                self.assertEqual(owner._store.bytes,0)
