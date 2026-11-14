"""Strict state machinery fixtures; not trained/reference PG acceptance."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from humansense.pg_artifact import manifest_digest
from humansense.pg_sources import PG_REVISION,PG_BYTES,PG_SHA256
from humansense.pg_graph import _inventory
from humansense.pg_canonical import (FORMAT,MIN_WORKSPACE,validate_canonical,schema_digest,
    _snapshot_into,_headers,_coverage,_materialize,_planned)
from humansense.pg_canonical_process import PGCanonicalProcess
from test_pg_artifact import declaration


def basic(root):
    graph=declaration(root);graph_sha=manifest_digest(graph);lock={};lock_sha=manifest_digest(lock)
    raw=struct.pack('<f',1.)
    header=json.dumps({'value':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode()
    payload=struct.pack('<Q',len(header))+header+raw
    spec=dict(schema_version=1,format=FORMAT,graph_manifest_sha256=graph_sha,runtime_lock_sha256=lock_sha,
        graph_schema_sha256='0'*64,graph_buffer_sha256={},
        source_identity=dict(pg_revision=PG_REVISION,pg_sha256=PG_SHA256,pg_bytes=PG_BYTES),
        export_provenance=dict(tool_revision='0'*40,source_evidence_sha256='0'*64,reference_replay_sha256='0'*64),
        tensors={'value':dict(logical_names=['value'],shard=0,tensor_key='value')},
        shards=[dict(path='state.safetensors',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())],
        budgets=dict(disk_bytes=len(payload)+8*1024**2,ram_destination_bytes=1024,vram_destination_bytes=0,ram_workspace_bytes=MIN_WORKSPACE))
    return graph,graph_sha,lock,lock_sha,spec,payload


def snapshots(root,spec,stage):
    opened=[]
    try:_snapshot_into(root,spec['shards'],stage,opened)
    except BaseException:
        for item in opened:item.close()
        raise
    return opened


class CanonicalContract(unittest.TestCase):
    def test_manifest_bijection_bounds_and_frozen_identity(self):
        *_,base,payload=basic(Path('/bundle'))
        validate_canonical(base,expected_sha256=manifest_digest(base))
        bads=[]
        bad=deepcopy(base);bad['tensors']['other']=dict(logical_names=['other'],shard=0,tensor_key='value');bads.append(bad)
        bad=deepcopy(base);bad['tensors']['value']['logical_names']=['value','value'];bads.append(bad)
        bad=deepcopy(base);bad['budgets']['ram_workspace_bytes']=MIN_WORKSPACE-1;bads.append(bad)
        bad=deepcopy(base);bad['budgets']['disk_bytes']=True;bads.append(bad)
        bad=deepcopy(base);bad['shards'][0]['path']='../outside';bads.append(bad)
        bad=deepcopy(base);bad['source_identity']['pg_bytes']=1;bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):validate_canonical(bad,expected_sha256=manifest_digest(bad))
        with self.assertRaises(ValueError):validate_canonical(base,expected_sha256='0'*64)

    def test_real_snapshot_fd_rejects_path_replacement_and_mutation(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            *_,spec,payload=basic(bundle);(bundle/'state.safetensors').write_bytes(payload)
            owned=snapshots(bundle,spec,stage)
            try:
                header=_headers(owned,spec['shards']);self.assertEqual(header[0]['value']['shape'],[1])
                # The loader reads its verified FD, never a later path occupant.
                replacement=stage/'replacement';replacement.write_bytes(b'not a tensor')
                os.replace(replacement,owned[0].path)
                with self.assertRaisesRegex(ValueError,'identity'):_headers(owned,spec['shards'])
            finally:
                for item in owned:item.close()
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            *_,spec,payload=basic(bundle);(bundle/'state.safetensors').write_bytes(payload)
            owned=snapshots(bundle,spec,stage)
            try:
                os.chmod(owned[0].path,0o600);owned[0].path.write_bytes(payload+b'x')
                with self.assertRaisesRegex(ValueError,'identity'):_headers(owned,spec['shards'])
            finally:
                for item in owned:item.close()

    def test_source_mutations_during_stream_rejected(self):
        for mutation in ('append','truncate','replace'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir='.') as temp:
                root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
                *_,spec,_=basic(bundle);raw=b'x'*(1024*1024+17);source=bundle/'state.safetensors';source.write_bytes(raw)
                spec['shards'][0].update(size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                original=os.fdopen;fired=False
                class Reader:
                    def __init__(self,stream):self.stream=stream
                    def __enter__(self):return self
                    def __exit__(self,*args):return self.stream.__exit__(*args)
                    def fileno(self):return self.stream.fileno()
                    def read(self,size):
                        nonlocal fired
                        value=self.stream.read(size)
                        if not fired:
                            fired=True
                            if mutation=='append':
                                with source.open('ab') as target:target.write(b'y')
                            elif mutation=='truncate':
                                with source.open('wb') as target:target.write(b'x')
                            else:
                                replacement=bundle/'replacement';replacement.write_bytes(raw);os.replace(replacement,source)
                        return value
                def opened(fd,mode,*args,**kwargs):
                    stream=original(fd,mode,*args,**kwargs)
                    return Reader(stream) if mode=='rb' else stream
                with patch('humansense.pg_canonical.os.fdopen',side_effect=opened):
                    with self.assertRaises(ValueError):snapshots(bundle,spec,stage)
                self.assertTrue(fired)

    def test_source_hash_symlink_and_header_gap_fail(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            *_,spec,payload=basic(bundle);(bundle/'state.safetensors').write_bytes(payload)
            bad=deepcopy(spec);bad['shards'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'digest'):snapshots(bundle,bad,stage)
            stage2=root/'stage2';stage2.mkdir();(bundle/'link').symlink_to(bundle/'state.safetensors')
            bad=deepcopy(spec);bad['shards'][0]['path']='link'
            with self.assertRaisesRegex(ValueError,'symlink'):snapshots(bundle,bad,stage2)
            raw=json.dumps({'value':dict(dtype='F32',shape=[1],data_offsets=[1,5])}).encode()
            content=struct.pack('<Q',len(raw))+raw+b'x'+struct.pack('<f',1.)
            (bundle/'gap').write_bytes(content);bad['shards'][0]=dict(path='gap',size=len(content),sha256=hashlib.sha256(content).hexdigest())
            owned=snapshots(bundle,bad,stage2)
            try:
                with self.assertRaisesRegex(ValueError,'gap'):_headers(owned,bad['shards'])
            finally:
                for item in owned:item.close()


@unittest.skipUnless(importlib.util.find_spec('torch'),'existing Torch needed; no installs')
class Materialize(unittest.TestCase):
    def fixture(self,root,*,nonfinite=False,bad_position=False):
        import torch
        model=torch.nn.Module();model.left=torch.nn.Parameter(torch.empty(3,dtype=torch.bfloat16,device='meta'))
        model.right=model.left;model.independent=torch.nn.Parameter(torch.empty(3,dtype=torch.bfloat16,device='meta'))
        model.narrow=torch.nn.Parameter(torch.empty(2,dtype=torch.float16,device='meta'))
        model.register_buffer('persistent_half',torch.tensor([1.001],dtype=torch.float16),persistent=True)
        model.register_buffer('position',torch.arange(3),persistent=True)
        model.register_buffer('constant',torch.tensor(1.001,dtype=torch.float16),persistent=False)
        model.register_buffer('constant_alias',model.constant,persistent=False)
        model.qformer_text_input=True
        schema=_inventory(model,torch)
        graph,graph_sha,lock,lock_sha,spec,_=basic(root)
        values=torch.tensor([1.001,1.003,1.007],dtype=torch.bfloat16)
        if nonfinite:values[0]=float('nan')
        tensors={'left':values,'independent':values.clone(),'narrow':torch.tensor([1.001,1.002],dtype=torch.float16),
                 'persistent_half':model.persistent_half.clone(),'position':torch.arange(3)+(1 if bad_position else 0)}
        headers={};data=bytearray();mapping={}
        for rep,tensor in tensors.items():
            raw=tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            start=len(data);data.extend(raw)
            headers[rep]=dict(dtype={torch.int64:'I64',torch.bfloat16:'BF16',torch.float16:'F16'}[tensor.dtype],shape=list(tensor.shape),data_offsets=[start,len(data)])
            names=sorted(row['name'] for row in schema['rows'] if row['alias']==rep)
            mapping[rep]=dict(logical_names=names,shard=0,tensor_key=rep)
        encoded=json.dumps(headers).encode();payload=struct.pack('<Q',len(encoded))+encoded+data
        spec.update(tensors=mapping,graph_schema_sha256=schema_digest(schema,graph_sha,lock_sha),graph_buffer_sha256=schema['buffer_sha256'])
        spec['shards'][0].update(size=len(payload),sha256=hashlib.sha256(payload).hexdigest());spec['budgets']['disk_bytes']=len(payload)
        (root/'state.safetensors').write_bytes(payload)
        return model,schema,spec,values

    def test_real_alias_constructor_rounding_buffer_and_complete_materialization(self):
        import torch
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();stage=root/'stage';stage.mkdir()
            model,schema,spec,values=self.fixture(root);validate_canonical(spec,expected_sha256=manifest_digest(spec))
            owned=snapshots(root,spec,stage)
            try:_materialize(model,schema,spec,_headers(owned,spec['shards']),owned,'cpu')
            finally:
                for item in owned:item.close()
            self.assertIs(model.left,model.right);self.assertIsNot(model.left,model.independent)
            self.assertTrue(torch.equal(model.left,values.float()))
            self.assertFalse(torch.equal(model.left,torch.tensor([1.001,1.003,1.007])))
            self.assertTrue(torch.equal(model.narrow,torch.tensor([1.001,1.002],dtype=torch.float16).float()))
            self.assertEqual(model.persistent_half.dtype,torch.float32)
            self.assertEqual(model.persistent_half.item(),torch.tensor(1.001,dtype=torch.float16).float().item())
            self.assertIs(model.constant,model.constant_alias);self.assertEqual(model.constant.dtype,torch.float32)
            self.assertFalse(model.qformer_text_input);self.assertFalse(model.training)
            self.assertTrue(all(not p.is_meta and not p.requires_grad for p in model.parameters()))

    def test_missing_alias_dtype_and_budget_deny_before_allocation(self):
        import torch
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();stage=root/'stage';stage.mkdir()
            model,schema,spec,_=self.fixture(root);owned=snapshots(root,spec,stage)
            try:
                headers=_headers(owned,spec['shards'])
                cases=[]
                bad=deepcopy(spec);del bad['tensors']['independent'];cases.append((bad,headers))
                bad=deepcopy(spec);bad['tensors']['left']['logical_names']=['left'];cases.append((bad,headers))
                bad=deepcopy(spec);bad['budgets']['ram_destination_bytes']=0;cases.append((bad,headers))
                wrong=deepcopy(headers);wrong[0]['left']['dtype']='F32';cases.append((spec,wrong))
                for bad,head in cases:
                    with patch('torch.empty',side_effect=AssertionError('must reject before allocation')):
                        with self.assertRaises(ValueError):_materialize(model,schema,bad,head,owned,'cpu')
            finally:
                for item in owned:item.close()

    def test_nonfinite_and_deterministic_buffer_fail_no_inference(self):
        for options in ({'nonfinite':True},{'bad_position':True}):
            with self.subTest(options=options),tempfile.TemporaryDirectory(dir='.') as temp:
                root=Path(temp).resolve();stage=root/'stage';stage.mkdir()
                model,schema,spec,_=self.fixture(root,**options);owned=snapshots(root,spec,stage)
                try:
                    with self.assertRaises(ValueError):_materialize(model,schema,spec,_headers(owned,spec['shards']),owned,'cpu')
                    self.assertTrue(model.qformer_text_input)
                finally:
                    for item in owned:item.close()


class OwnedWorker(unittest.IsolatedAsyncioTestCase):
    async def test_actual_unavailable_worker_reaped_and_stage_removed(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            graph,sha,lock,lock_sha,spec,_=basic(bundle);children=[]
            real=asyncio.create_subprocess_exec
            async def spawn(*args,**kwargs):
                child=await real(*args,**kwargs);children.append(child);return child
            with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError,'runtime_mismatch'):
                    await PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage))
            self.assertEqual(len(children),1);self.assertIsNotNone(children[0].returncode)
            self.assertTrue(children[0]._transport.is_closing());self.assertEqual(list(stage.iterdir()),[])

    async def test_cancellation_reaps_actual_child_before_cleanup(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            graph,sha,lock,lock_sha,spec,_=basic(bundle);children=[];entered=asyncio.Event()
            real=asyncio.create_subprocess_exec
            async def spawn(*args,**kwargs):
                request=args[args.index('--request')+1]
                code="import pathlib,sys,time; p=pathlib.Path(sys.argv[1]).parent; (p/'shard-00.safetensors').write_bytes(b'owned'); sys.stderr.write('R');sys.stderr.flush();time.sleep(60)"
                kwargs['stderr']=asyncio.subprocess.PIPE
                child=await real(sys.executable,'-u','-c',code,request,**kwargs)
                children.append(child);entered.set();return child
            with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn):
                task=asyncio.create_task(PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage)))
                await asyncio.wait_for(entered.wait(),5)
                await asyncio.wait_for(children[0].stderr.readexactly(1),5)
                task.cancel();task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
            self.assertIsNotNone(children[0].returncode);self.assertTrue(children[0]._transport.is_closing())
            self.assertEqual(list(stage.iterdir()),[])

    async def test_timeout_reaps_child_and_preserves_unexpected_stage_entry(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            graph,sha,lock,lock_sha,spec,_=basic(bundle);children=[];paths=[]
            real=asyncio.create_subprocess_exec
            async def spawn(*args,**kwargs):
                request=args[args.index('--request')+1];paths.append(Path(request).parent)
                code="import pathlib,sys,time; p=pathlib.Path(sys.argv[1]).parent; (p/'unexpected.txt').write_text('preserve');sys.stderr.write('R');sys.stderr.flush();time.sleep(60)"
                kwargs['stderr']=asyncio.subprocess.PIPE
                child=await real(sys.executable,'-u','-c',code,request,**kwargs);children.append(child)
                await asyncio.wait_for(child.stderr.readexactly(1),5)
                return child
            with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn):
                with self.assertRaises(TimeoutError) as caught:
                    await PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage),startup_timeout=.01)
            self.assertIsNotNone(children[0].returncode);self.assertTrue(children[0]._transport.is_closing())
            self.assertEqual((paths[0]/'unexpected.txt').read_text(),'preserve')
            self.assertIsInstance(caught.exception.__cause__,RuntimeError)

    async def test_normal_concurrent_close_and_late_spawn_cancel_are_owned(self):
        for late in (False,True):
            with self.subTest(late=late),tempfile.TemporaryDirectory(dir='.') as temp:
                root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
                graph,sha,lock,lock_sha,spec,_=basic(bundle);children=[];spawned=asyncio.Event();release=asyncio.Event()
                real=asyncio.create_subprocess_exec
                async def spawn(*args,**kwargs):
                    request=args[args.index('--request')+1]
                    code="import pathlib,sys,json; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(json.dumps(dict(status='loaded',canonical_sha256=r['canonical_sha'],graph_schema_sha256=r['canonical']['graph_schema_sha256'],planned_storage_bytes=4,trained_identity_verified=False,reference_parity_verified=False,runtime_compatibility_verified=False,resource_admission='unavailable',actual_acceptance='incomplete')),flush=True);sys.stdin.buffer.read()"
                    child=await real(sys.executable,'-u','-c',code,request,**kwargs);children.append(child);spawned.set()
                    if late:await release.wait()
                    return child
                with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn):
                    task=asyncio.create_task(PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage)))
                    await asyncio.wait_for(spawned.wait(),5)
                    if late:
                        task.cancel();release.set()
                        with self.assertRaises(asyncio.CancelledError):await task
                    else:
                        owner=await task
                        await asyncio.gather(owner.close(),owner.close())
                        await owner.close();self.assertTrue(owner.closed)
                self.assertIsNotNone(children[0].returncode);self.assertTrue(children[0]._transport.is_closing())
                self.assertEqual(list(stage.iterdir()),[])

    async def test_actual_exec_failure_cleans_owned_request_directory(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            graph,sha,lock,lock_sha,spec,_=basic(bundle)
            invalid=root/'not-python';invalid.write_text('not an executable image');invalid.chmod(0o700)
            with self.assertRaises(OSError):
                await PGCanonicalProcess.open(str(invalid),graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage))
            self.assertEqual(list(stage.iterdir()),[])
            self.assertEqual(invalid.read_text(),'not an executable image')

    async def test_spawn_deadline_keeps_late_handle_only_for_reap(self):
        from humansense import pg_canonical_process as process_module
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            graph,sha,lock,lock_sha,spec,_=basic(bundle);children=[]
            spawned=asyncio.Event();release=asyncio.Event();expired=asyncio.Event()
            real=asyncio.create_subprocess_exec;real_deadline=process_module._deadline
            async def spawn(*args,**kwargs):
                request=args[args.index('--request')+1]
                code="import pathlib,sys,json; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(json.dumps(dict(status='loaded',canonical_sha256=r['canonical_sha'],graph_schema_sha256=r['canonical']['graph_schema_sha256'],planned_storage_bytes=4,trained_identity_verified=False,reference_parity_verified=False,runtime_compatibility_verified=False,resource_admission='unavailable',actual_acceptance='incomplete')),flush=True);sys.stdin.buffer.read()"
                child=await real(sys.executable,'-u','-c',code,request,**kwargs)
                children.append(child);spawned.set();await release.wait();return child
            async def deadline(task,seconds):
                try:return await real_deadline(task,seconds)
                except TimeoutError:expired.set();raise
            with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn), \
                 patch('humansense.pg_canonical_process._deadline',side_effect=deadline):
                task=asyncio.create_task(PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage),startup_timeout=.01))
                try:
                    await asyncio.wait_for(spawned.wait(),5)
                    await asyncio.wait_for(expired.wait(),5)
                    self.assertFalse(task.done())  # Parent is still acquiring ownership for cleanup.
                finally:release.set()
                with self.assertRaises(TimeoutError):await task
            self.assertIsNotNone(children[0].returncode);self.assertTrue(children[0]._transport.is_closing())
            self.assertEqual(list(stage.iterdir()),[])

    async def test_response_uses_remaining_startup_budget(self):
        from humansense import pg_canonical_process as process_module
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();bundle=root/'bundle';bundle.mkdir();stage=root/'stage';stage.mkdir()
            graph,sha,lock,lock_sha,spec,_=basic(bundle);children=[];budgets=[]
            real=asyncio.create_subprocess_exec;real_deadline=process_module._deadline
            async def spawn(*args,**kwargs):
                request=args[args.index('--request')+1]
                code="import pathlib,sys,json; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(json.dumps(dict(status='loaded',canonical_sha256=r['canonical_sha'],graph_schema_sha256=r['canonical']['graph_schema_sha256'],planned_storage_bytes=4,trained_identity_verified=False,reference_parity_verified=False,runtime_compatibility_verified=False,resource_admission='unavailable',actual_acceptance='incomplete')),flush=True);sys.stdin.buffer.read()"
                child=await real(sys.executable,'-u','-c',code,request,**kwargs);children.append(child);return child
            async def deadline(task,seconds):
                budgets.append(seconds);return await real_deadline(task,seconds)
            # Controlled application clock; actual asyncio timers/subprocesses
            # are unchanged. No host scheduling-speed assumption is involved.
            with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=spawn), \
                 patch('humansense.pg_canonical_process._deadline',side_effect=deadline), \
                 patch('humansense.pg_canonical_process._now',side_effect=[100.,100.,104.,104.,105.]):
                owner=await PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(stage),startup_timeout=10)
                await owner.close()
            self.assertEqual(budgets[:2],[10.,6.])
            self.assertIsNotNone(children[0].returncode);self.assertEqual(list(stage.iterdir()),[])

    async def test_source_stage_overlap_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root=Path(temp).resolve();graph,sha,lock,lock_sha,spec,_=basic(root)
            with patch('humansense.pg_canonical_process.asyncio.create_subprocess_exec',side_effect=AssertionError('must not spawn')):
                with self.assertRaisesRegex(ValueError,'overlap'):
                    await PGCanonicalProcess.open(sys.executable,graph,sha,lock,lock_sha,spec,manifest_digest(spec),stage_root=str(root))


if __name__=='__main__':unittest.main()
