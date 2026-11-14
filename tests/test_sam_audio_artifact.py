import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from humansense.sam_audio_artifact import (
    SOURCE_PINS, manifest_digest, validate_manifest, verify_manifest, verify_runtime, read_manifest,
)


class ArtifactPreflight(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        sources = {}
        for name, pin in SOURCE_PINS.items():
            path = self.root/name; path.mkdir()
            digest = self.put(path/'model.py', b'# test fixture, not official source\n')
            sources[name] = dict(revision=pin,root=str(path),files={'model.py':digest})
        config = dict(audio_codec={'sample_rate':48000,'encoder_rates':[2,8,10,12]},
                      text_encoder={'name':'remote-name-not-used','max_length':512},
                      transformer={'fixture':True}, vision_encoder={'fixture':True})
        self.config = self.root/'config.json'
        config_hash = self.put(self.config,json.dumps(config).encode())
        checkpoint = self.root/'checkpoint.pt'
        checkpoint_hash = self.put(checkpoint,b'not a trained checkpoint')
        t5 = self.root/'t5'; t5.mkdir()
        files = {'config.json':self.put(t5/'config.json',b'{"model_type":"t5"}'),
                 'tokenizer_config.json':self.put(t5/'tokenizer_config.json',b'{"tokenizer_class":"T5TokenizerFast"}'),
                 'tokenizer.json':self.put(t5/'tokenizer.json',b'{}'),
                 'model.safetensors':self.put(t5/'model.safetensors',b'not tensor evidence')}
        self.manifest = dict(schema_version=1,sources=sources,
            runtime={'python':'3.11.12','packages':{'torch':'2.8.0'}},
            sam=dict(config_path=str(self.config),config_sha256=config_hash,
                     checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_hash),
            t5=dict(root=str(t5),files=files))

    @staticmethod
    def put(path, data):
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def verify(self):
        return verify_manifest(self.manifest,expected_sha256=manifest_digest(self.manifest))

    def change_config(self, data):
        self.manifest['sam']['config_sha256'] = self.put(self.config,data)

    def test_byte_identity_is_not_native_or_tensor_acceptance(self):
        result = self.verify()
        self.assertTrue(result['declared_file_bytes_verified'])
        for key in ('imported_source_verified','tensor_coverage_verified','runtime_compatibility_verified'):
            self.assertFalse(result[key])
        self.assertEqual(result['hop_length'],1920)
        self.assertEqual(result['effective_config']['text_encoder']['name'],self.manifest['t5']['root'])
        for key in ('span_predictor','text_ranker','visual_ranker'):
            self.assertIsNone(result['effective_config'][key])
            self.assertNotIn(key,result['original_config'])
        self.assertEqual(result['original_config_bytes'],self.config.read_bytes())
        self.assertEqual(result['original_config']['text_encoder']['name'],'remote-name-not-used')

    def test_detached_declarations_and_effective_config(self):
        result = self.verify()
        self.manifest['runtime']['packages']['torch'] = '0'
        result['effective_config']['text_encoder']['name'] = 'changed'
        self.assertEqual(result['manifest']['runtime']['packages']['torch'],'2.8.0')
        self.assertEqual(result['original_config']['text_encoder']['name'],'remote-name-not-used')
        self.assertEqual(result['original_config_sha256'],hashlib.sha256(result['original_config_bytes']).hexdigest())

    def test_frozen_digest_and_unknown_fields(self):
        digest = manifest_digest(self.manifest)
        self.manifest['unknown'] = True
        with self.assertRaises(ValueError): validate_manifest(self.manifest,expected_sha256=digest)
        with self.assertRaises(ValueError): self.verify()

    def test_manifest_file_strict_json_and_canonical_identity(self):
        path = self.root/'manifest.json'
        path.write_text(json.dumps(self.manifest,indent=2))
        digest = manifest_digest(self.manifest)
        self.assertEqual(read_manifest(str(path),expected_sha256=digest),self.manifest)
        data = json.dumps(self.manifest).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        path.write_text(data)
        with self.assertRaises(ValueError): read_manifest(str(path),expected_sha256=digest)

    def test_source_revision_inventory_and_content(self):
        source = self.manifest['sources']['sam_audio']
        source['revision'] = '0'*40
        with self.assertRaises(ValueError): self.verify()
        source['revision'] = SOURCE_PINS['sam_audio']
        path = Path(source['root'])/'model.py'
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.verify()
        source['files']['../model.py'] = source['files'].pop('model.py')
        with self.assertRaises(ValueError): self.verify()

    def test_missing_extra_and_modified_t5_files(self):
        root = Path(self.manifest['t5']['root'])
        (root/'extra.json').write_text('{}')
        with self.assertRaises(ValueError): self.verify()
        (root/'extra.json').unlink()
        (root/'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.verify()
        (root/'model.safetensors').unlink()
        with self.assertRaises(ValueError): self.verify()

    def test_symlink_file_and_parent_rejected(self):
        root = Path(self.manifest['t5']['root'])
        original = root/'model.safetensors'
        original.rename(self.root/'weights.bin')
        original.symlink_to(self.root/'weights.bin')
        with self.assertRaises(ValueError): self.verify()
        original.unlink(); original.write_bytes(b'not tensor evidence')
        alias = self.root/'alias'; alias.symlink_to(root,target_is_directory=True)
        self.manifest['t5']['root'] = str(alias)
        with self.assertRaises(ValueError): self.verify()

    def test_special_file_rejected_before_read(self):
        path = Path(self.manifest['sam']['checkpoint_path'])
        path.unlink(); os.mkfifo(path)
        with self.assertRaises(ValueError): self.verify()

    def test_file_swapped_for_fifo_cannot_block_open(self):
        # A separate owned fixture process bounds this regression even against
        # the old blocking implementation. No unrelated process is signalled.
        script = '''
import os, sys
from pathlib import Path
from unittest.mock import patch
from humansense.sam_audio_artifact import _file
target = Path(sys.argv[1]); expected = sys.argv[2]
original_open = os.open
def replace(path, flags):
    target.unlink(); os.mkfifo(target)
    return original_open(path, flags)
with patch('os.open', side_effect=replace):
    try: _file(target, expected, 1024)
    except ValueError: print('rejected replacement before read')
    else: raise AssertionError('replaced file accepted')
'''
        result = subprocess.run([sys.executable,'-c',script,self.manifest['sam']['checkpoint_path'],
                                 self.manifest['sam']['checkpoint_sha256']],
                                text=True,capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('rejected replacement before read',result.stdout)

    def test_sparse_oversize_checkpoint_rejected_before_hash(self):
        path = Path(self.manifest['sam']['checkpoint_path'])
        with path.open('wb') as stream: stream.truncate(16*1024**3+1)
        with self.assertRaises(ValueError): self.verify()

    def test_strict_config_json_and_geometry(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e400}', b'[]'):
            self.change_config(raw)
            with self.assertRaises(ValueError): self.verify()
        for codec, text in (({'sample_rate':16000,'encoder_rates':[2]}, {'max_length':512}),
                            ({'sample_rate':48000,'encoder_rates':[True]}, {'max_length':512}),
                            ({'sample_rate':48000,'encoder_rates':[48000,2]}, {'max_length':512}),
                            ({'sample_rate':48000,'encoder_rates':[2]}, {'max_length':None})):
            self.change_config(json.dumps(dict(audio_codec=codec,text_encoder=text)).encode())
            with self.assertRaises(ValueError): self.verify()

    def test_builtin_t5_only(self):
        root = Path(self.manifest['t5']['root'])
        for filename, content in (
            ('config.json',b'{"model_type":"custom"}'),
            ('config.json',b'{"model_type":"t5","auto_map":{}}'),
            ('tokenizer_config.json',b'{"tokenizer_class":"CustomTokenizer"}')):
            with self.subTest(filename=filename,content=content):
                previous = (root/filename).read_bytes()
                self.manifest['t5']['files'][filename] = self.put(root/filename,content)
                with self.assertRaises(ValueError): self.verify()
                self.manifest['t5']['files'][filename] = self.put(root/filename,previous)

    def test_paths_and_runtime_are_exact_declarations(self):
        runtime = self.manifest['runtime']
        for version in ('3.10.11','>=3.11','3.11',True):
            runtime['python'] = version
            with self.assertRaises(ValueError): self.verify()
        runtime['python'] = '3.11.12'
        self.manifest['sam']['config_path'] = 'https://example.invalid/config.json'
        with self.assertRaises(ValueError): self.verify()

    def test_runtime_versions_checked_without_importing_model_packages(self):
        digest = manifest_digest(self.manifest)
        with patch('platform.python_version',return_value='3.11.12'), \
                patch('importlib.metadata.version',return_value='2.8.0') as version:
            result = verify_runtime(self.manifest,expected_sha256=digest)
            version.assert_called_once_with('torch')
            self.assertTrue(result['declared_runtime_versions_verified'])
            self.assertFalse(result['runtime_compatibility_verified'])
        with patch('platform.python_version',return_value='3.10.11'):
            with self.assertRaises(RuntimeError): verify_runtime(self.manifest,expected_sha256=digest)
        with patch('platform.python_version',return_value='3.11.12'), \
                patch('importlib.metadata.version',return_value='2.8.0+cpu'):
            with self.assertRaises(RuntimeError): verify_runtime(self.manifest,expected_sha256=digest)


if __name__ == '__main__':
    unittest.main()
