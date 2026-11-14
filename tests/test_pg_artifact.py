from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from humansense.pg_artifact import (ORIGINS,PACKAGES,CANDIDATE_PACKAGES,MAX_WEIGHT,MAX_AGGREGATE,
    manifest_digest,validate_manifest,inspect_manifest,read_manifest,_file)
from humansense.pg_sources import SOURCE_HASHES,PG_REVISION,LAVIS_CANDIDATE,PG_BYTES,PG_SHA256


def declaration(root):
    sources={}
    for group,pins in SOURCE_HASHES.items():
        sources[group]=dict(revision=PG_REVISION if group=='pg' else LAVIS_CANDIDATE,
            files={name:dict(path=group+'/'+name,size=1,sha256=digest) for name,digest in pins.items()})
    artifacts={}
    for role,origin in ORIGINS.items():
        artifacts[role]=dict(origin=origin,revision='unresolved',license='unresolved',
            files=[dict(path='assets/'+role+'.fixture',size=1,sha256='0'*64)])
    artifacts['pg_weights'].update(revision=PG_REVISION,
        files=[dict(path='assets/pgvlm_weights.bin',size=PG_BYTES,sha256=PG_SHA256)])
    return dict(schema_version=1,bundle_root=str(root),sources=sources,artifacts=artifacts,
        runtime=dict(python='3.10.11',packages={name:CANDIDATE_PACKAGES.get(name,'1.0.0') for name in PACKAGES}),
        execution=dict(construction_qformer_text_input=True,inference_qformer_text_input=False,
            device='cpu',precision='float32',batch_size=1,image_size=224,prompt_tokens=128,max_new_tokens=32,
            num_beams=5,num_captions=3,use_nucleus_sampling=False,length_penalty=0,repetition_penalty=1))


def validate(value):return validate_manifest(value,expected_sha256=manifest_digest(value))


class Manifest(unittest.TestCase):
    def test_contract_retained_graph_and_failclosed_readiness(self):
        value=declaration(Path('/missing/bundle'))
        result=inspect_manifest(value,expected_sha256=manifest_digest(value))
        self.assertEqual(result['inspection'],'manifest_only')
        self.assertFalse(result['native_ready']);self.assertFalse(result['graph_coverage_verified'])
        self.assertFalse(result['legacy_checkpoint_schema_verified'])
        self.assertFalse(result['runtime_compatibility_verified'])
        self.assertEqual(result['resource_admission'],'unavailable')
        self.assertEqual(result['actual_acceptance'],'incomplete')
        self.assertTrue(all(row['status']=='not_inspected' for row in result['files']))
        self.assertIn('t5_weights',result['unresolved_revisions'])
        self.assertEqual(next(row for row in result['files'] if row['role']=='pg_weights')['size'],PG_BYTES)
        self.assertEqual(next(row for row in result['files'] if row['role']=='t5_weights')['identity_authority'],'caller_declared')
        for key,bad in [('construction_qformer_text_input',False),('inference_qformer_text_input',True),
                        ('batch_size',True),('num_beams',1),('num_captions',4),('prompt_tokens',129),
                        ('use_nucleus_sampling',True),('max_new_tokens',33),('length_penalty',.1)]:
            changed=deepcopy(value);changed['execution'][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):validate(changed)
        for change in ({'size':1},{'sha256':'0'*64},{'path':'assets/fake.bin'}):
            changed=deepcopy(value);changed['artifacts']['pg_weights']['files'][0].update(change)
            with self.assertRaises(ValueError):validate(changed)

    def test_matching_runtime_metadata_still_never_claims_compatibility(self):
        value=declaration(Path('/bundle'))
        with patch('humansense.pg_artifact.metadata.version',side_effect=lambda name:value['runtime']['packages'][name]), \
             patch('humansense.pg_artifact.platform.python_version',return_value=value['runtime']['python']):
            result=inspect_manifest(value,expected_sha256=manifest_digest(value))
        self.assertEqual(result['runtime_mismatches'],[])
        self.assertFalse(result['runtime_compatibility_verified'])
        self.assertFalse(result['native_ready'])

    def test_missing_roles_source_runtime_and_path_bounds(self):
        base=declaration(Path('/bundle'))
        changes=[]
        bad=deepcopy(base);del bad['artifacts']['eva_weights'];changes.append(bad)
        bad=deepcopy(base);bad['artifacts']['unknown']=bad['artifacts']['eva_weights'];changes.append(bad)
        bad=deepcopy(base);bad['runtime']['packages']['transformers']='4.57.6';changes.append(bad)
        bad=deepcopy(base);bad['sources']['pg']['revision']='0'*40;changes.append(bad)
        bad=deepcopy(base);bad['sources']['pg']['files']['test.py']['sha256']='0'*64;changes.append(bad)
        for name in ('../escape','/absolute','a//b','a/./b','a\\b'):
            bad=deepcopy(base);bad['artifacts']['bert_config']['files'][0]['path']=name;changes.append(bad)
        bad=deepcopy(base);bad['artifacts']['bert_config']['files'][0]['path']='assets/bert_weights.fixture';changes.append(bad)
        bad=deepcopy(base);bad['artifacts']['bert_config']['files'][0]['size']=32*1024**2+1;changes.append(bad)
        bad=deepcopy(base);bad['artifacts']['t5_weights']['files'][0]['size']=MAX_WEIGHT+1;changes.append(bad)
        bad=deepcopy(base)
        for role in ('t5_weights','bert_weights'):bad['artifacts'][role]['files'][0]['size']=MAX_WEIGHT
        changes.append(bad)
        bad=deepcopy(base)
        for role in ('t5_weights','bert_weights'):
            bad['artifacts'][role]['files']=[dict(path=f'{role}/{i}',size=1,sha256='0'*64) for i in range(64)]
        changes.append(bad)
        for bad in changes:
            with self.assertRaises(ValueError):validate(bad)
        bad=deepcopy(base);bad['execution']['length_penalty']=float('nan')
        with self.assertRaises(ValueError):manifest_digest(bad)

    def test_manifest_read_json_duplicates_size_and_frozen_digest(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();value=declaration(root);file=root/'manifest.json'
            file.write_text(json.dumps(value))
            self.assertEqual(read_manifest(str(file),expected_sha256=manifest_digest(value)),value)
            with self.assertRaises(ValueError):read_manifest(str(file),expected_sha256='0'*64)
            file.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(ValueError):read_manifest(str(file),expected_sha256='0'*64)
            file.write_bytes(b'x'*(1024**2+1))
            with self.assertRaises(ValueError):read_manifest(str(file),expected_sha256='0'*64)

    def test_explicit_file_mode_reports_missing_not_ready(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();value=declaration(root)
            result=inspect_manifest(value,expected_sha256=manifest_digest(value),inspect_files=True)
            self.assertEqual(len(result['missing_paths']),result['declared_file_count'])
            self.assertFalse(result['native_ready']);self.assertEqual(result['actual_acceptance'],'incomplete')
            self.assertFalse(result['known_source_inventory_complete_for_execution'])

    def test_dependency_light_actual_cli(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();value=declaration(root);path=root/'manifest.json';path.write_text(json.dumps(value))
            code='''
import sys,socket
socket.create_connection=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('network'))
from humansense.pg_artifact import main
main()
assert not {'torch','transformers','timm','fairscale','lavis','safetensors','numpy'} & sys.modules.keys()
'''
            result=subprocess.run([sys.executable,'-c',code,'--manifest',str(path),'--sha256',manifest_digest(value)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout)
            self.assertEqual(report['inspection'],'manifest_only');self.assertFalse(report['native_ready'])


class StreamedFiles(unittest.TestCase):
    def test_real_hash_limits_symlinks_and_special_files(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();data=b'x'*(1024**2+5);path=root/'file';path.write_bytes(data)
            digest=hashlib.sha256(data).hexdigest()
            self.assertEqual(_file(path,digest,len(data)),len(data))
            with self.assertRaises(ValueError):_file(path,digest,len(data)-1)
            with self.assertRaises(ValueError):_file(path,'0'*64,len(data))
            link=root/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):_file(link,digest,len(data))
            directory=root/'dir';directory.mkdir();(directory/'f').write_bytes(data)
            linked=root/'linked';linked.symlink_to(directory,target_is_directory=True)
            with self.assertRaises(ValueError):_file(linked/'f',digest,len(data))
            fifo=root/'fifo';os.mkfifo(fifo)
            with self.assertRaises(ValueError):_file(fifo,digest,len(data))

    def test_append_truncate_and_replace_during_stream_rejected(self):
        real_fdopen=os.fdopen
        for mutation in ('append','truncate','replace'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir='.') as tmp:
                root=Path(tmp).resolve();path=root/'file';data=b'x'*(1024**2+8);path.write_bytes(data)
                done=[]
                class Reader:
                    def __init__(self,fd,*args,**kwargs):self.stream=real_fdopen(fd,*args,**kwargs)
                    def __enter__(self):return self
                    def __exit__(self,*args):return self.stream.__exit__(*args)
                    def fileno(self):return self.stream.fileno()
                    def read(self,n):
                        result=self.stream.read(n)
                        if not done:
                            done.append(True)
                            if mutation=='append':
                                with path.open('ab') as out:out.write(b'y')
                            elif mutation=='truncate':path.write_bytes(b'')
                            else:
                                replacement=root/'replacement';replacement.write_bytes(data);replacement.replace(path)
                        return result
                with patch('humansense.sam_audio_artifact.os.fdopen',side_effect=Reader):
                    with self.assertRaises(ValueError):_file(path,hashlib.sha256(data).hexdigest(),len(data))
                self.assertEqual(done,[True])


if __name__=='__main__':unittest.main()
