import ast
from pathlib import Path
import re
import stat
import sys
import tempfile
import textwrap
import unittest

import numpy as np

from gamst.crew.modules import RoleUnavailable, role_module
from gamst.crew.registry import ROLES
from gamst.provider_profiles import capability
from gamst.providers import model_workers

PROVIDERS = Path(model_workers.__file__).resolve().parent
SHA = 'a' * 64


class WorkerProtocolTest(unittest.IsolatedAsyncioTestCase):
    """The real worker process and wire with a stand-in model (no weights): what crosses, and what cannot."""

    def python(self, directory):
        """An interpreter wrapper whose worker builds stand-in models instead of loading weights."""
        script = Path(directory) / 'worker.py'
        script.write_text(textwrap.dedent('''
            from gamst.providers import model_workers as m
            class Extractor:
                def extract(self, mixture, enrollment):
                    return mixture * float(enrollment[0])
            class Separator:
                def separate(self, window):
                    return [window, -window, window * 0] if len(window) else []
            class Aligner:
                def align(self, samples, rate, text, language):
                    return [dict(text=text, start_s=0.0, end_s=len(samples) / rate)]
            class Pose:
                version = 'stand-in'
                def infer(self, image):
                    return [dict(bbox=dict(x=0., y=0., width=1., height=1.),
                                 landmarks=[dict(x=.5, y=.25, visibility=.9)] * 308)] * 4
            def embed(frame):
                if abs(frame.data).max() == 0:
                    raise ValueError('silent')
                return [float(frame.data.mean())] * 1024
            def load(profile, config):
                return {'speaker.embedding': lambda: m._Embedding(embed), 'sapiens2.pose': lambda: m._Pose(Pose()),
                        'qwen3.forced_aligner': lambda: m._Aligner(Aligner()),
                        'target_speech.local': lambda: m._Extraction(Extractor()),
                        'pixit.sources': lambda: m._Separation(Separator())}[profile]()
            m.load = load
            from gamst.provider_worker import main
            main()
        '''))
        wrapper = Path(directory) / 'python'
        root = PROVIDERS.parents[1]
        wrapper.write_text(f'#!/bin/sh\nPYTHONPATH="{root}" exec "{sys.executable}" -u "{script}"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        return str(wrapper)

    async def test_models_answer_their_module_from_another_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            python = self.python(directory)
            embedder = await model_workers.RemoteEmbedder.open(python, 'speaker.embedding', dict(
                backend='sherpa-onnx', path='/m.onnx', sha256=SHA))
            try:
                self.assertEqual(embedder.model_id, f'sherpa-onnx:{SHA[:16]}')
                self.assertEqual(len(await embedder.embed(np.full(800, .5, np.float32), 16000)), 1024)
                with self.assertRaises(ValueError):   # unusable audio is an answer; the worker stays up
                    await embedder.embed(np.zeros(800, np.float32), 16000)
                self.assertAlmostEqual((await embedder.embed(np.full(8, .25, np.float32), 16000))[0], .25)
            finally:
                await embedder.close()
            extractor = await model_workers.RemoteExtractor.open(python, 'target_speech.local', dict(
                backend='wesep', directory='/m', sha256=SHA))
            try:
                clip, mixture = np.full(160, .5, np.float32), np.arange(32000, dtype=np.float32) / 1e5
                np.testing.assert_allclose(await extractor.extract(mixture, clip), mixture / 2)
                np.testing.assert_allclose(await extractor.extract(mixture[:500], clip), mixture[:500] / 2)
            finally:
                await extractor.close()
            separator = await model_workers.RemoteSeparator.open(python, 'pixit.sources', dict(
                config='/c', separation='/s', embedding='/e', sha256=SHA))
            try:
                self.assertEqual(separator.model_id, f'pyannote-pixit:{SHA[:16]}')
                mixture = np.arange(16000, dtype=np.float32) / 1e5
                sources = await separator.separate(mixture)   # speakers x samples cross as a private payload
                self.assertEqual(sources.shape, (3, 16000))
                np.testing.assert_allclose(sources[1], -mixture)
            finally:
                await separator.close()
            aligner = await model_workers.RemoteAligner.open(python, 'qwen3.forced_aligner', dict(path='/m', sha256=SHA))
            try:
                self.assertEqual(await aligner.align(np.zeros(16000, np.float32), 16000, '대박', 'ko'),
                                 [dict(text='대박', start_s=0., end_s=1.)])
            finally:
                await aligner.close()
            pose = await model_workers.RemotePose.open(python, 'sapiens2.pose', dict(
                pose_dir='/p', detector_dir='/d', sha256=SHA, accept_license=True))
            try:
                poses = await pose.infer(np.zeros((64, 48, 3), np.uint8))   # 4 people x 308 keypoints fit the wire
                self.assertEqual((len(poses), len(poses[0]['landmarks'])), (4, 308))
            finally:
                await pose.close()

    async def test_a_model_worker_never_becomes_a_context_module(self):
        for profile in model_workers.PROFILES:
            with self.assertRaises(RoleUnavailable):
                await role_module(None, 'audio.speaker_verify', dict(
                    model='x', provider_profile=profile, python=sys.executable, config={}))

    def test_opt_in_licences_and_digests_are_required_before_a_process_starts(self):
        with self.assertRaisesRegex(PermissionError, 'NC'):
            capability('target_speech.local', dict(backend='usef', source='s', checkpoint='c', sha256=SHA))
        with self.assertRaisesRegex(PermissionError, 'Sapiens2'):
            capability('sapiens2.pose', dict(pose_dir='/p', detector_dir='/d', sha256=SHA))
        with self.assertRaisesRegex(ValueError, 'sha256'):
            capability('face.embedding', dict(path='/m.onnx'))


# ---------------------------------------------------------------- version pins

# registry provider_profile (or candidate id when it has no worker profile) -> the file that loads it
LOADERS = {
    'rfdetr.nano': 'detection', 'rfdetr.large': 'detection', 'yoloe.local': 'yoloe_native',
    'efficienttam.tiny512': 'efficienttam_native', 'mediapipe.body.video': 'landmarker_video',
    'mediapipe.hand.video': 'landmarker_video', 'mediapipe.face.video': 'landmarker_video',
    'depth_anything_v2.small': 'depth', 'video_depth_anything.large': 'video_depth',
    'smolvlm.500m': 'smolvlm_native', 'qwen3.asr': 'qwen_asr', 'moonshine_voice.native': 'moonshine_asr',
    'yamnet.events': 'environment_audio', 'ced.onnx': 'ced_events', 'nemotron3.diarization': 'diarization',
    'emotion2vec.seed.onnx': 'vocal_expression', 'factorizephys.local': 'factorizephys_context',
    'silero.activity': 'vad', 'qwen3.forced_aligner': 'asr_words', 'speaker.embedding': 'speaker_verify',
    'face.embedding': 'person_match', 'body.embedding': 'person_match', 'target_speech.local': 'target_speech',
    'pixit.sources': 'pixit', 'sepformer.sources': 'sepformer',
    'ultralytics.objects': 'ultralytics_sports', 'ultralytics.pose': 'ultralytics_sports', 'tracknet.ball': 'tracknet_ball',
    'image.embedding': 'siglip_embedding',
    'sapiens2.pose': 'sapiens2_pose', 'rtmw.pose': 'rtmw_pose', 'active_speaker.lightasd': 'active_speaker', 'audio.prosody': 'prosody', 'opencv.flow': 'motion',
    'streamingvlm.local': 'streaming_semantic', 'vision.rotation': 'rotation', 'faster_whisper.small': None, 'faster_whisper.large_v3_turbo': None,
    'qwen3.asr.words': 'qwen_words', 'hosted.asr.words': None,   # a hosted API loads no package
    'sortformer.diarization': 'sortformer', 'pyannoteai.live.diarization': None, 'pyannoteai.offline.diarization': None,
    'offline.diarization': None,   # slot_sources.py checks DiariZen's and community-1's pins in their own environments
    'streaming.asr.words': None, 'faster_whisper.align': None,
    'pg.physical.local': None, 'opencv': None, 'sounddevice': None, 'pillow': None,   # no version check in a provider file
}
VERSION = re.compile(r'\d+(\.\d+)+')


def checked_versions(path):
    """{package: version} the provider compares with the installed one (read from its source)."""
    tree = ast.parse(path.read_text())
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) for target in node.targets if isinstance(target, ast.Name)}
    for node in tree.body:   # e.g. yoloe_native imports ULTRALYTICS_VERSION from yoloe
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            other = path.with_name(node.module + '.py')
            if other.is_file() and any('VERSION' in item.name for item in node.names):
                for sub in ast.parse(other.read_text()).body:
                    if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant):
                        constants.update({t.id: sub.value.value for t in sub.targets if isinstance(t, ast.Name)})

    def literal(node):
        value = node.value if isinstance(node, ast.Constant) else constants.get(getattr(node, 'id', None))
        return value if isinstance(value, str) else None

    def version_call(node):
        while isinstance(node, (ast.Subscript, ast.Attribute)) or (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'split'):
            node = node.func.value if isinstance(node, ast.Call) else node.value
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', None)) == 'version' \
                and node.args:
            return literal(node.args[0])
        return None

    found, names = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and version_call(node.value):
            names.update({t.id: version_call(node.value) for t in node.targets if isinstance(t, ast.Name)})
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            package, version = map(literal, node.elts)
            if package and version and VERSION.fullmatch(version):
                found[package] = version
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                package, version = literal(key) if key else None, literal(value)
                if package and version and VERSION.fullmatch(version):
                    found[package] = version
        elif isinstance(node, ast.Compare) and len(node.comparators) == 1:
            package = version_call(node.left) or names.get(getattr(node.left, 'id', None))
            version = literal(node.comparators[0])
            if package and version and VERSION.fullmatch(version):
                found[package] = version
    return found


class RegistryPinTest(unittest.TestCase):
    def test_the_reader_finds_each_style_of_check(self):
        self.assertEqual(checked_versions(PROVIDERS / 'qwen_asr.py'),
                         {'qwen-asr': '0.0.6', 'vllm': '0.14.0', 'transformers': '4.57.6'})
        self.assertEqual(checked_versions(PROVIDERS / 'detection.py'), {'rfdetr': '1.12.1'})
        self.assertEqual(checked_versions(PROVIDERS / 'yoloe_native.py')['ultralytics'], '8.4.212')
        self.assertEqual(checked_versions(PROVIDERS / 'environment_audio.py'), {'mediapipe': '0.10.35'})
        self.assertEqual(checked_versions(PROVIDERS / 'moonshine_asr.py'), {'moonshine-voice': '0.0.45'})

    def test_registry_pins_agree_with_the_versions_providers_check(self):
        problems = []
        for role in ROLES.values():
            for name, candidate in role.candidates.items():
                key = candidate.provider_profile or name
                if candidate.in_process and key not in LOADERS:
                    continue   # GAMST's own code or a pinned native build; no Python package is loaded
                self.assertTrue(name == 'custom' or key in LOADERS, f'{role.id} {name}: add {key} to LOADERS')
                loader = LOADERS.get(key)
                if loader is None:
                    continue
                pins = dict(re.sub(r'\[.*\]', '', line).split('==', 1) for line in candidate.packages
                            if '==' in line and '@' not in line)
                for package, version in checked_versions(PROVIDERS / f'{loader}.py').items():
                    if pins.get(package) != version:
                        problems.append(f'{role.id} {name}: {loader}.py checks {package}=={version}, '
                                        f'registry has {pins.get(package)}')
        self.assertEqual(problems, [])

    def test_manual_and_licence_bound_candidates_are_never_applied(self):
        for role, name in (('audio.target_speech', 'usef.tfgridnet'), ('vision.pose', 'sapiens2.5b'),
                           ('audio.speaker_verify', 'wespeaker.w2vbert2_mfa_voxblink2'),
                           ('vision.body_identity', 'solider.swin_b_msmt17'),
                           ('vision.semantic_scene', 'qwen3.8-flash.hosted')):
            self.assertFalse(ROLES[role].candidates[name].automatic)
        for role in ROLES.values():
            for name, candidate in role.candidates.items():
                if candidate.provider_profile in model_workers.PROFILES and candidate.automatic:
                    self.assertTrue(all(weight.sha256 and weight.url for weight in candidate.weights), name)


if __name__ == '__main__':
    unittest.main()
