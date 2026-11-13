import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from humansense.provider_profiles import capability
from humansense.provider_worker import _construct
from humansense.providers.endpoint_asr import SileroEndpointASR
from humansense.providers.qwen_asr import QwenStreamingASR
from humansense.providers.vad import SileroActivity
from humansense.qwen_artifact import REQUIRED, artifact_version
from test_qwen_asr import EngineDouble, np
from test_qwen_endpoint import ActivityModel


def config(path='/fixture-not-installed'):
    return dict(asr=dict(path=path,files={name:'0'*64 for name in REQUIRED}),
        vad=dict(path=path,sha256='1'*64),minimum_speech_ms=160,silence_ms=640)


class EndpointDiscovery(unittest.TestCase):
    def test_discovery_is_import_light_and_does_not_touch_files(self):
        code="""
import sys
from humansense.provider_profiles import capability
from humansense.qwen_artifact import REQUIRED
config=dict(asr=dict(path='/not-installed',files={name:'0'*64 for name in REQUIRED}),
    vad=dict(path='/not-installed',sha256='1'*64),minimum_speech_ms=160,silence_ms=640)
spec=capability('qwen3.asr.endpoint',config)
assert spec.id=='audio.qwen3_asr.silero_endpoint'
assert not {'numpy','torch','qwen_asr','vllm','transformers'} & set(sys.modules)
"""
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        baseline=capability('qwen3.asr.endpoint',config())
        changed=config(); changed['silence_ms']=672
        self.assertNotEqual(baseline.version,capability('qwen3.asr.endpoint',changed).version)
        for value in ({},dict(config(),extra=True),dict(config(),minimum_speech_ms=True),
                      dict(config(),silence_ms=33),dict(config(),vad={'path':'x','sha256':'bad'}),
                      dict(config(),asr={'path':'x','files':{}})):
            with self.assertRaises(ValueError): capability('qwen3.asr.endpoint',value)

    def test_small_artifact_is_checked_before_asr_loader(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory).resolve()/'vad.jit'; path.write_bytes(b'not-a-model')
            selected=config(str(path))
            with patch.object(QwenStreamingASR,'from_local') as load:
                with self.assertRaises(ValueError): SileroEndpointASR.from_local(**selected)
                load.assert_not_called()


@unittest.skipIf(np is None,'numpy not installed')
class EndpointLoader(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='.'); self.addCleanup(self.temp.cleanup)
        path=Path(self.temp.name).resolve()/'vad.jit'; path.write_bytes(b'fixture-not-a-model')
        self.config=config(str(path))
        self.config['vad']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()

    def providers(self):
        speech=QwenStreamingASR(EngineDouble(),model_version=artifact_version(self.config['asr']['files']))
        model=ActivityModel([])
        activity=SileroActivity(model,model.score,version=SileroActivity.artifact_version(self.config['vad']['sha256']))
        return speech,activity

    def test_worker_construct_matches_discovery_and_closes_both(self):
        speech,activity=self.providers()
        with patch.object(QwenStreamingASR,'from_local',return_value=speech) as a, patch.object(SileroActivity,'from_local',return_value=activity) as v:
            provider,spec=_construct('qwen3.asr.endpoint',self.config)
            a.assert_called_once_with(**self.config['asr']); v.assert_called_once_with(**self.config['vad'])
            self.assertEqual(provider.spec,spec)
            self.assertEqual(spec,capability('qwen3.asr.endpoint',self.config))
            provider.close()
        self.assertTrue(speech.closed and activity.closed)

    def test_failed_load_or_contract_match_releases_partial_ownership(self):
        speech,activity=self.providers()
        with patch.object(QwenStreamingASR,'from_local',return_value=speech), patch.object(SileroActivity,'from_local',side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError): SileroEndpointASR.from_local(**self.config)
        self.assertTrue(speech.closed)
        speech,activity=self.providers()
        activity.start_threshold=.7
        with patch.object(QwenStreamingASR,'from_local',return_value=speech), patch.object(SileroActivity,'from_local',return_value=activity):
            with self.assertRaisesRegex(ValueError,'differs'): SileroEndpointASR.from_local(**self.config)
        self.assertTrue(speech.closed and activity.closed)
