"""Pinned admission and actual owned SDK pipes with synthetic model/metadata.

Tiny local metadata/weight bytes and worker predict are explicit fixtures. They
exercise machinery, never assert installation of a trained Smol runtime.
"""
import asyncio
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from benchmarks.recording_dependencies import distribution_descriptor,verify_dependencies
from benchmarks.recorded_source import digest,encoded
import benchmarks.recording_smol_runtime as smol
from gamst.provider_profiles import capability
from gamst.providers.smolvlm_native import configuration,RUNTIME,MODEL_ID
import test_smolvlm_call as calls
import test_smolvlm_context as context_fixture
from test_recording_sdk_scopes import NativeFixture


def fixture(root):
    site=root/'site';site.mkdir();entries=[]
    for name,version in dict(RUNTIME,numpy='2.2.6',pillow='12.2.0').items():
        directory=site/(name.replace('-','_')+'-'+version+'.dist-info');directory.mkdir()
        (directory/'METADATA').write_text('Name: '+name+'\nVersion: '+version+'\n\nExplicit test fixture\n')
        (directory/'RECORD').write_text(directory.name+'/METADATA,,\n'+directory.name+'/RECORD,,\n')
        descriptor=distribution_descriptor(name,allowed_roots=(root,),metadata_roots=(site,))
        entries.append(dict(distribution=name,version=version,installation_sha256=digest(descriptor)))
    inventory=verify_dependencies(entries,allowed_roots=(root,),metadata_roots=(site,))
    weights=root/'weights.bin';weights.write_bytes(b'explicit non-model fixture')
    tokens=root/'tokens.json';tokens.write_bytes(b'explicit non-tokenizer fixture')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    config=configuration(dict(checkpoint_path=str(weights),checkpoint_sha256=sha(weights),
                              tokenizer_path=str(tokens),tokenizer_sha256=sha(tokens)))
    raw=encoded(config)+b'\n';path='configurations/000.json';docs={path:config}
    assets=dict(schema_version=1,kind='recording_runtime_assets',assets=[],dependencies=entries,resident_state=[])
    for aid,kind,p in (('python','python',Path(sys.executable)),('weights','checkpoint',weights),('tokens','checkpoint',tokens)):
        assets['assets'].append(dict(asset_id=aid,kind=kind,path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    assets['assets'].append(dict(asset_id='worker',kind='worker_config',path=path,
                               sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    spec=capability('smolvlm2.500m.local',dict(config,window_frames=1))
    adapter=dict(adapter_id='small',implementation='smolvlm_process_v1',evidence_class='real',
        provider=spec.provider,model_id=MODEL_ID,model_revision=spec.version,
        implementation_sha256=smol.implementation_sha256(),dependency_inventory_sha256=inventory.sha256,
        config=dict(python_asset_sha256=assets['assets'][0]['sha256'],
                    worker_config_sha256=assets['assets'][-1]['sha256'],startup_timeout_ms=2000,inference_timeout_ms=2000))
    return adapter,assets,docs,inventory,config,spec


class Admission(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir='.');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.case=fixture(self.root)
    def admit(self,case=None):
        a,assets,docs,inventory,_,_=case or self.case
        return smol._admit(a,assets,docs,inventory,'smolvlm_sentinel_v1')

    def test_owned_metadata_loader_resolves_original_fixed_smol_prompt(self):
        from benchmarks.recording_runtime_inputs import OwnedRuntimeInputs
        from gamst.providers.smolvlm_native import PROMPT
        a,assets,docs,_,_,_=deepcopy(self.case)
        prompt=dict(schema_version=1,kind='recording_visual_prompt',question=PROMPT)
        path='configurations/001.json';docs[path]=prompt
        raw=encoded(prompt)+b'\n'
        assets['assets'].append(dict(asset_id='prompt',kind='system_template',path=path,
            sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        docs['runtime/assets.json']=assets
        policy=dict(adapters=[a],arms=[dict(controllers=[],recording_policies=[dict(periodic=dict(
            scheduler='smolvlm_sentinel_v1',prompt_policy_sha256=digest(prompt)),deliveries=[],fill_chain=None)])])
        owner=OwnedRuntimeInputs.__new__(OwnedRuntimeInputs);owner._root_path=str(self.root)
        self.assertEqual(owner._metadata_assets(docs,policy),{'configurations/000.json',path})
    def test_actual_hashes_retained_fds_and_mutation_detection(self):
        _,_,python,pins=self.admit()
        try:
            self.assertEqual(python,str(Path(sys.executable).resolve()))
            self.assertEqual(len(pins.files),3)
            self.assertTrue(all(not os.get_inheritable(fd) for fd,_,_ in pins.files))
            (self.root/'weights.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):pins.check()
        finally:pins.close()
        self.assertEqual(pins.files,[])
    def test_wrong_scheduler_identity_python_and_assets_refuse_before_spawn(self):
        a,assets,docs,inv,_,_=self.case
        with self.assertRaises(ValueError):smol._admit(a,assets,docs,inv,'recorded_visual_v1')
        mutations=(lambda a,s,d:a.update(implementation_sha256='f'*64),
            lambda a,s,d:a.update(model_id='other'),lambda a,s,d:a.update(model_revision='other'),
            lambda a,s,d:s['assets'][0].update(path=str(self.root/'other-python')),
            lambda a,s,d:s['assets'][1].update(sha256='e'*64),
            lambda a,s,d:d['configurations/000.json'].update(max_new_tokens=3))
        for mutation in mutations:
            aa,ss,dd=deepcopy((a,assets,docs));mutation(aa,ss,dd)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):smol._admit(aa,ss,dd,inv,'smolvlm_sentinel_v1')
    def test_fifo_and_symlink_assets_refuse_without_blocking(self):
        weights=self.root/'weights.bin';saved=self.root/'original.bin';weights.rename(saved)
        os.mkfifo(weights)
        with self.assertRaises(ValueError):self.admit()
        weights.unlink();weights.symlink_to(saved)
        with self.assertRaises(OSError):self.admit()

    def test_symlink_dependent_virtualenv_is_not_silently_reselected(self):
        alias=self.root/'current-python';alias.symlink_to(Path(sys.executable).resolve())
        with patch.object(sys,'executable',str(alias)),patch.object(sys,'prefix',str(self.root/'venv')):
            a,assets,docs,inventory,_,_=self.case
            assets=deepcopy(assets);assets['assets'][0]['path']=str(alias)
            with self.assertRaisesRegex(ValueError,'virtual environment'):
                smol._admit(a,assets,docs,inventory,'smolvlm_sentinel_v1')


@unittest.skipIf(calls.np is None,'existing optional numpy required')
class OwnedWorker(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=calls.Calls.asyncSetUp
    asyncTearDown=calls.Calls.asyncTearDown
    worker_fixture=calls.Calls.fixture
    setup_human=calls.Calls.setup_human
    request=calls.Calls.request
    def inputs(self):
        temporary=tempfile.TemporaryDirectory(dir='.');self.addCleanup(temporary.cleanup)
        return fixture(Path(temporary.name).resolve())
    async def test_actual_owned_process_entry_and_cleanup(self):
        a,assets,docs,inventory,config,spec=self.inputs()
        owner,writer,clock=await self.recording(provider=spec.provider,revision=spec.version,
            role='small_vlm',boundary='process_adapter',config=dict(config,window_frames=1))
        human,controller,detect=self.setup_human()
        async with human:
            owner.bind_semantic_lane(controller,0)
            with self.worker_fixture():
                lane=await smol.open_smol_lane(human,'calls',a,assets,docs,owner,0,inventory,scheduler='smolvlm_sentinel_v1')
                owner.bind_mapping(clock.mapping,clock.sha256)
                try:
                    rid,record,frame=await self.request(human,controller,detect)
                    decision=next(r['decision']['decision_sha256'] for r in controller.snapshot()['requests'] if r['id']==rid)
                    self.assertIsNone(lane.entry_for_decision(decision))
                    result=await lane.adapter.execute(rid,record,frame,maximum_charge=2,
                        evaluation_scope=owner.task_scope(owner.declarations['tasks'][0]['task_id'],binding_index=0))
                    self.assertTrue(result['accepted'])
                    entry=lane.entry_for_decision(decision).snapshot()
                    self.assertEqual(entry['prompt_sha256'],lane.prompt_sha256(frame.timestamp_ms))
                    self.assertEqual(entry['entry']['request_decision_sha256'],decision)
                    lane.release_entry(decision)
                finally:await lane.close()
                self.assertTrue(lane.worker.closed);self.assertEqual(lane._pins.files,[])
                self.assertIsNotNone(self.children[-1].returncode)
        await owner.close()
        self.assertEqual(self.terminal(writer)['entry_count'],1)
        self.assertEqual(self.terminal(writer)['unclosed_workers'],0)
    async def test_cancel_during_pin_read_joins_and_closes_before_return(self):
        a,assets,docs,inventory,_,_=self.inputs()
        entered=threading.Event();release=threading.Event();original=smol._admit;captured=[]
        def slow(*args):
            value=original(*args);captured.append(value[3]);entered.set();release.wait(5);return value
        with patch.object(smol,'_admit',side_effect=slow):
            task=asyncio.create_task(smol.open_smol_lane(None,'calls',a,assets,docs,None,0,inventory,scheduler='smolvlm_sentinel_v1'))
            self.assertTrue(await asyncio.to_thread(entered.wait,2));task.cancel();await asyncio.sleep(0)
            self.assertFalse(task.done());release.set()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(captured[0].files,[]);self.assertEqual(self.children,[])

    async def test_cancel_after_actual_spawn_reaps_worker_and_releases_pins(self):
        a,assets,docs,inventory,config,spec=self.inputs()
        owner,_,_=await self.recording(provider=spec.provider,revision=spec.version,
            role='small_vlm',boundary='process_adapter',config=dict(config,window_frames=1))
        human,controller,_=self.setup_human();captured=[];original=smol._admit
        def capture(*args):
            result=original(*args);captured.append(result[3]);return result
        async with human:
            owner.bind_semantic_lane(controller,0)
            code='import time; time.sleep(60)\n'+context_fixture.WORKER.replace('DELAY','0')
            with patch.object(smol,'_admit',side_effect=capture),self.worker_fixture(code=code):
                task=asyncio.create_task(smol.open_smol_lane(human,'calls',a,assets,docs,owner,0,inventory,
                    scheduler='smolvlm_sentinel_v1'))
                async def spawned():
                    while not self.children:
                        if task.done():await task
                        await asyncio.sleep(.001)
                await asyncio.wait_for(spawned(),2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
                self.assertIsNotNone(self.children[-1].returncode)
                self.assertEqual(captured[0].files,[])

    async def test_alias_retarget_never_executes_unverified_marker(self):
        # No spawn/SDK mock: the real worker rejects our synthetic non-model
        # assets. A substituted executable must never run, even on failure.
        temporary=tempfile.TemporaryDirectory(dir='.');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name).resolve();real=Path(sys.executable).resolve()
        alias=root/'current-python';alias.symlink_to(real)
        marker=root/'UNVERIFIED_EXECUTED';replacement=root/'replacement-python'
        replacement.write_text('#!'+str(real)+'\nfrom pathlib import Path\nPath('+repr(str(marker))+').write_text("executed")\n')
        replacement.chmod(0o700)
        with patch.object(sys,'executable',str(alias)):
            a,assets,docs,inventory,config,spec=fixture(root)
            owner,writer,_=await self.recording(provider=spec.provider,revision=spec.version,
                role='small_vlm',boundary='process_adapter',config=dict(config,window_frames=1))
            human,controller,_=self.setup_human();original=smol._admit;captured=[]
            def retarget(*args):
                result=original(*args);captured.append(result[3])
                alias.unlink();alias.symlink_to(replacement)
                return result
            async with human:
                owner.bind_semantic_lane(controller,0)
                with patch.object(smol,'_admit',side_effect=retarget):
                    with self.assertRaises((RuntimeError,ValueError,asyncio.TimeoutError)):
                        await smol.open_smol_lane(human,'calls',a,assets,docs,owner,0,inventory,
                            scheduler='smolvlm_sentinel_v1')
            self.assertFalse(marker.exists())
            self.assertEqual(captured[0].files,[])
            await owner.close()
            self.assertEqual(self.terminal(writer)['unclosed_workers'],0)
