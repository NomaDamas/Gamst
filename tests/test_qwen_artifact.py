import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from humansense.qwen_artifact import REQUIRED, artifact_version, verify_bundle
from humansense.provider_profiles import capability
from humansense.providers.qwen_asr import QwenStreamingASR


class Artifacts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='.'); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        for name in REQUIRED: (self.root/name).write_bytes(b'fixture-not-a-model')
        (self.root/'config.json').write_text(json.dumps(dict(model_type='qwen3_asr',architectures=['Qwen3ASRForConditionalGeneration'])))
        self.files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()}
    def test_identity_discovery_and_verified_bytes(self):
        root,version=verify_bundle(str(self.root),self.files)
        self.assertEqual(root,str(self.root))
        self.assertEqual(version,artifact_version(dict(reversed(list(self.files.items())))))
        self.assertEqual(capability('qwen3.asr',dict(path=str(self.root),files=self.files)).version,version)
    def test_missing_changed_extra_and_linked_files(self):
        for path in ('Qwen/Qwen3-ASR-0.6B','https://example.com/model'):
            with self.assertRaises(ValueError): verify_bundle(path,self.files)
        target=self.root/'merges.txt'; original=target.read_bytes()
        target.write_bytes(b'changed')
        with self.assertRaises(ValueError): verify_bundle(str(self.root),self.files)
        target.write_bytes(original)
        extra=self.root/'extra.py'; extra.write_text('fixture')
        with self.assertRaises(ValueError): verify_bundle(str(self.root),self.files)
        extra.unlink(); target.unlink(); target.symlink_to(self.root/'vocab.json')
        with self.assertRaises(ValueError): verify_bundle(str(self.root),self.files)
    def test_loader_validation_precedes_optional_import_and_offline_required(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError): QwenStreamingASR.from_local(str(self.root),self.files)
        with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_ENABLE_V1_MULTIPROCESSING':'0'}):
            with patch('importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(RuntimeError): QwenStreamingASR.from_local(str(self.root),self.files)
    def test_exact_loader_arguments_with_explicit_double_not_inference(self):
        captured={}
        engine=SimpleNamespace(backend='vllm',sampling_params=SimpleNamespace(max_tokens=128),
            init_streaming_state=lambda **kw:None,streaming_transcribe=lambda *args:None,
            finish_streaming_transcribe=lambda *args:None)
        def load(**kwargs): captured.update(kwargs); return engine
        versions={'qwen-asr':'0.0.6','vllm':'0.14.0','transformers':'4.57.6'}
        with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_ENABLE_V1_MULTIPROCESSING':'0'}), patch('importlib.metadata.version',side_effect=versions.get), patch.dict('sys.modules',{'qwen_asr':SimpleNamespace(Qwen3ASRModel=SimpleNamespace(LLM=load))}):
            p=QwenStreamingASR.from_local(str(self.root),self.files)
        self.assertEqual(captured['model'],str(self.root)); self.assertEqual(captured['tokenizer'],str(self.root))
        self.assertFalse(captured['trust_remote_code']); self.assertEqual(captured['load_format'],'safetensors')
        self.assertEqual(p.max_segment_ms,30000); p.close()
