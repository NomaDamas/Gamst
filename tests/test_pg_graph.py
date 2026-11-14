"""Graph machinery tests; never evidence of a trained PG runtime."""
import ast
import hashlib
import importlib.util
import json
import os
import platform
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from humansense.pg_artifact import manifest_digest
from humansense.pg_graph import (FILES,project_sources,inspect_graph,_tokenizer_dir,_inventory,
                                _execute_projection,_unavailable,_verify_runtime_lock,_check_defaults)
from test_pg_artifact import declaration


class Contract(unittest.TestCase):
    def test_bad_or_missing_source_never_reaches_ast(self):
        raw={name:b'raise RuntimeError("must not execute")' for name in FILES}
        with patch('humansense.pg_graph.ast.parse',side_effect=AssertionError('parsed unauthenticated source')):
            with self.assertRaisesRegex(ValueError,'source hash'):project_sources(raw)
            with self.assertRaisesRegex(ValueError,'exact original'):project_sources({})

    def test_runtime_mismatch_is_import_light_and_no_file_access(self):
        value=declaration(Path('/missing/pg'))
        with patch('humansense.pg_graph._runtime',return_value=['transformers','timm']), \
             patch('humansense.pg_graph._read_entry',side_effect=AssertionError('must not open files')):
            result=inspect_graph(value,expected_sha256=manifest_digest(value))
        self.assertEqual(result['reason'],'declared_runtime_mismatch')
        self.assertFalse(result['native_ready']);self.assertFalse(result['trained_state_loaded'])
        self.assertFalse(result['runtime_compatibility_verified'])
        self.assertEqual(result['resource_admission'],'unavailable')
        self.assertEqual(result['actual_acceptance'],'incomplete')
        # Actual isolated CLI must also fail unavailable without importing ML.
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'manifest.json';path.write_text(json.dumps(value))
            code='''import sys,runpy,json
sys.argv=['pg_graph','--manifest',sys.argv[1],'--sha256',sys.argv[2]]
runpy.run_module('humansense.pg_graph',run_name='__main__')
assert not any(x in sys.modules for x in ('torch','transformers','accelerate','timm','lavis'))
'''
            run=subprocess.run([sys.executable,'-c',code,str(path),manifest_digest(value)],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(run.stdout)['reason'],'declared_runtime_mismatch')

    def test_private_import_does_not_fall_back_to_lavis_or_network(self):
        # This probes the import guard itself, not a synthetic supported graph.
        trees={name:ast.parse('') for name in FILES}
        trees['eva']=ast.parse('import lavis')
        with self.assertRaisesRegex(ImportError,'outside PG source import closure'):
            _execute_projection(trees,{})
        trees['eva']=ast.parse('import socket')
        with self.assertRaises(ImportError):_execute_projection(trees,{})

    def test_complete_worker_metadata_lock_and_missing_lock_gate(self):
        lock=dict(schema_version=1,python=platform.python_version(),platform=sys.platform,
                  machine=platform.machine(),packages={'package-a':'1.0','package-b':'2.0'})
        items=[SimpleNamespace(metadata={'Name':'package_a'},version='1.0'),
               SimpleNamespace(metadata={'Name':'Package.B'},version='2.0')]
        with patch('humansense.pg_graph.importlib.metadata.distributions',return_value=items):
            self.assertTrue(_verify_runtime_lock(lock,manifest_digest(lock)))
            missing=dict(lock,packages={'package-a':'1.0'})
            self.assertFalse(_verify_runtime_lock(missing,manifest_digest(missing)))
            bad=dict(lock,packages={'Package_A':'1.0'})
            with self.assertRaises(ValueError):_verify_runtime_lock(bad,manifest_digest(bad))
            with self.assertRaises(ValueError):_verify_runtime_lock(lock,'0'*64)
        with patch('humansense.pg_graph.importlib.metadata.distributions',return_value=items+items[:1]):
            with self.assertRaisesRegex(ValueError,'duplicate'):_verify_runtime_lock(lock,manifest_digest(lock))
        value=declaration(Path('/missing/pg'))
        with patch('humansense.pg_graph._runtime',return_value=[]), \
             patch('humansense.pg_graph._read_entry',side_effect=AssertionError('no complete lock')):
            result=inspect_graph(value,expected_sha256=manifest_digest(value))
        self.assertEqual(result['reason'],'complete_worker_runtime_lock_required')
        self.assertFalse(result['native_ready'])

    def test_tokenizer_scope_hash_and_undeclared_files(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();folder=root/'tokenizer';folder.mkdir()
            raw=b'ordinary tokenizer fixture';(folder/'vocab.txt').write_bytes(raw)
            value=dict(files=[dict(path='tokenizer/vocab.txt',size=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
            self.assertEqual(_tokenizer_dir(root,value,kind='bert'),(folder,{'vocab_file':str(folder/'vocab.txt'),'tokenizer_file':None}))
            (folder/'tokenizer_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'undeclared'):_tokenizer_dir(root,value,kind='bert')
            # No deletion needed: independent supplied role with wrong digest.
            value['files'].append(dict(path='tokenizer/tokenizer_config.json',size=2,sha256='0'*64))
            with self.assertRaises(ValueError):_tokenizer_dir(root,value,kind='bert')

    def test_tokenizer_config_cannot_redirect_to_unhashed_files(self):
        for kind,vocab in [('bert','vocab.txt'),('t5','spiece.model')]:
            for field in ('vocab_file','tokenizer_file','merges_file'):
                for target in ('/outside/unhashed-vocab.txt','../unhashed-vocab.txt'):
                    with self.subTest(kind=kind,field=field,target=target), tempfile.TemporaryDirectory(dir='.') as directory:
                        root=Path(directory).resolve();folder=root/'tokenizer';folder.mkdir()
                        (folder/vocab).write_bytes(b'vocab fixture')
                        config=json.dumps({field:target}).encode()
                        (folder/'tokenizer_config.json').write_bytes(config)
                        files=[]
                        for path in folder.iterdir():
                            raw=path.read_bytes();files.append(dict(path='tokenizer/'+path.name,size=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                        with self.assertRaisesRegex(ValueError,'redirect'):
                            _tokenizer_dir(root,dict(files=files),kind=kind)
        # Ordinary special token contents are not interpreted as paths.
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();folder=root/'tokenizer';folder.mkdir()
            data={'tokenizer.json':b'{}','tokenizer_config.json':b'{"eos_token":"</s>"}'}
            files=[]
            for name,raw in data.items():
                (folder/name).write_bytes(raw);files.append(dict(path='tokenizer/'+name,size=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            _,paths=_tokenizer_dir(root,dict(files=files),kind='t5')
            self.assertEqual(paths,{'vocab_file':None,'tokenizer_file':str(folder/'tokenizer.json')})

    def test_fixed_default_dtype_and_device_guard(self):
        fake=SimpleNamespace(float32='float32',get_default_dtype=lambda:'float32',
                             get_default_device=lambda:'cpu',device=lambda value:value)
        _check_defaults(fake)
        fake.get_default_dtype=lambda:'float64'
        with self.assertRaises(ValueError):_check_defaults(fake)
        fake.get_default_dtype=lambda:'float32';fake.get_default_device=lambda:'meta'
        with self.assertRaises(ValueError):_check_defaults(fake)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'existing torch required for machinery-only inventory')
    def test_duplicate_alias_buffer_and_precision_inventory(self):
        import torch
        model=torch.nn.Module()
        model.left=torch.nn.Parameter(torch.empty(2,3,device='meta',dtype=torch.bfloat16))
        model.right=model.left
        model.register_buffer('position',torch.arange(3),persistent=False)
        model.register_buffer('scale',torch.ones(2,dtype=torch.float32),persistent=True)
        result=_inventory(model,torch);rows={row['name']:row for row in result['rows']}
        self.assertEqual(set(rows),{'left','right','position','scale'})
        self.assertEqual(rows['right']['alias'],'left')
        self.assertEqual(rows['left']['constructor_dtype'],'torch.bfloat16')
        self.assertEqual(rows['left']['cpu_target_dtype'],'torch.float32')
        self.assertFalse(rows['position']['persistent']);self.assertTrue(rows['scale']['persistent'])
        self.assertEqual(result['unique_constructor_bytes'],12+24+8)
        self.assertEqual(result['unique_cpu_target_bytes'],24+24+8)
        self.assertEqual(len(result['buffer_sha256']),2)
        self.assertIsNone(result['peak_memory_bytes'])
        model.register_buffer('scalar',torch.tensor(1.),persistent=False)
        self.assertIn('scalar',_inventory(model,torch)['buffer_sha256'])
        model.scalar.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):_inventory(model,torch)
        model.scalar.fill_(1.)
        model.actual=torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError,'materialized parameter'):_inventory(model,torch)


@unittest.skipUnless(os.environ.get('GAMST_PG_SOURCE_ROOT'),'supplied pinned original sources required; no downloads')
class OriginalSource(unittest.TestCase):
    def test_actual_original_projection_retains_graph_and_removes_asset_loading(self):
        root=Path(os.environ['GAMST_PG_SOURCE_ROOT'])
        payload={key:(root/path).read_bytes() for key,path in FILES.items()}
        trees=project_sources(payload)
        for key,tree in trees.items():compile(tree,'verified/'+FILES[key],'exec')
        eva=next(n for n in trees['eva'].body if isinstance(n,ast.FunctionDef) and n.name=='create_eva_vit_g')
        self.assertEqual(len(eva.body),3)
        self.assertIn('embed_dim=1408',ast.unparse(eva))
        self.assertNotIn('download_cached_file',ast.unparse(eva))
        self.assertNotIn('torch.load',ast.unparse(eva))
        t5=ast.unparse(trees['t5'])
        self.assertNotIn('apex',t5);self.assertNotIn('ALL_LAYERNORM_LAYERS.append',t5)
        model=next(n for n in trees['model'].body if isinstance(n,ast.ClassDef))
        original=next(n for n in ast.parse(payload['model']).body if isinstance(n,ast.ClassDef))
        original_init=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        projected_init=next(n for n in model.body if isinstance(n,ast.FunctionDef))
        self.assertEqual(ast.dump(original_init),ast.dump(projected_init))
        self.assertEqual(model.decorator_list,[])
        self.assertIn('qformer_text_input=True',ast.unparse(projected_init))
        for tree in trees.values():
            self.assertFalse(any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('lavis') for n in ast.walk(tree)))
        tampered=dict(payload);tampered['t5']+=b'\n'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):project_sources(tampered)


if __name__=='__main__':unittest.main()
