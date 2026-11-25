from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock
import wave

import numpy as np

from gamst import Gamst, recipes
from gamst.pipeline import Derived, Pipeline
from gamst.private_payload import PrivatePoses
from gamst.providers.asr_words import words_capability
from gamst.providers.streaming_semantic import OpenAIStreamingBackend
from gamst.recipes.parts import PARTS
from gamst.recipes.perception_parts import MATH
from tests.test_recipes import HOST
from tests.test_speaker_verify import ToneEmbedder, now, tone

try:
    import cv2  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False


class Embedder(ToneEmbedder):
    """Stand-in for the speaker model worker (async, closable), used to test the recipe, not a model."""
    closed = 0

    async def embed(self, samples, sample_rate=16000):
        return super().embed(samples, sample_rate)

    async def close(self):
        type(self).closed += 1


class Words:
    """Stand-in for the ASR worker: one final utterance per second of non-silent audio."""
    spec = words_capability('test')

    def __init__(self):
        self.heard, self.start = [], None

    def process(self, frame):
        self.start = frame.timestamp_ms - 100 if self.start is None else self.start
        self.heard.append(frame.data)
        return self._final(frame.timestamp_ms) if len(self.heard) == 10 else None

    def _final(self, end):
        loud, start = np.abs(np.concatenate(self.heard)).mean() > .05, self.start
        self.heard, self.start = [], None
        return self.spec.validate(dict(utterance_id=f'u{start}', status='final', text='대박이다', start_ms=start, end_ms=end,
                                       words=[dict(text='대박이다', start_ms=start + 100, end_ms=end - 100)])) if loud else None


async def fake_open(client, entry):
    return dict(RemoteEmbedder=Embedder, RemoteWhisper=Words)[client]()


def write_wav(path, samples):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1), out.setsampwidth(2), out.setframerate(16000)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    return str(path)


class SpeechRecipeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.profile = self.tmp / 'profile.json'
        self.profile.write_text(json.dumps(dict(version=1, roles={
            'audio.speaker_verify': dict(model='sherpa.campplus_zh_en', python=sys.executable,
                                         config=dict(backend='sherpa-onnx', path='/m.onnx', sha256='a' * 64)),
            'audio.asr': dict(model='faster_whisper.small', python=sys.executable, config=dict(model_path='/m')),
            'audio.prosody': dict(model='dsp', python=sys.executable, config={})})))
        patcher = mock.patch('gamst.recipes.perception_parts._open', fake_open)
        patcher.start()
        self.addCleanup(patcher.stop)

    def enroll_voice(self):
        clip = write_wav(self.tmp / 'me.wav', tone(220, 4.))
        recipes.main(['enroll', 'person-term-count', '--kind', 'voice', '--audio', clip, '--seconds', '4', '--consent',
                      'owner, word counting', '--out', str(self.tmp / 'me.json'), '--profile', str(self.profile)])
        return self.tmp / 'me.json'

    def test_enrollment_keeps_embeddings_and_consent_only(self):
        with self.assertRaisesRegex(SystemExit, 'consent'):
            recipes.main(['enroll', 'person-term-count', '--kind', 'voice', '--audio', 'x.wav', '--out',
                          str(self.tmp / 'no.json'), '--profile', str(self.profile)])
        path = self.enroll_voice()
        saved = json.loads(path.read_text())
        self.assertEqual(set(saved), {'kind', 'model', 'dim', 'embeddings', 'recipe', 'seconds', 'consent'})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertGreaterEqual(Embedder.closed, 1)
        del saved['consent']
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(SystemExit, 'no consent record'):
            recipes.main(['run', 'person-term-count', '--audio', 'x.wav', '--voice', str(path), '--terms', '대박',
                          '--headless', '--profile', str(self.profile)])

    def test_only_the_enrolled_voice_is_counted_and_scored_against_a_human_count(self):
        voice = self.enroll_voice()
        # enrolled 3 s, somebody else 4 s, enrolled 3 s, silence; the stand-in ASR hears one term per second
        audio = np.concatenate([tone(220, 3.), tone(1500, 4.), tone(220, 3.), np.zeros(32000, np.float32)])
        truth = self.tmp / 'truth.json'
        truth.write_text(json.dumps(dict(capability='speech.term_events', event='term.spoken', count=6)))
        report = recipes.main(['run', 'person-term-count', '--audio', write_wav(self.tmp / 'room.wav', audio),
                               '--voice', str(voice), '--terms', '대박', '--speaker-threshold', '.95', '--headless',
                               '--truth', str(truth), '--report', str(self.tmp / 'r.json'),
                               '--profile', str(self.profile)])
        self.assertEqual(report['score']['truth_count'], 6)
        self.assertEqual(report['score']['counted'], report['counts']['speech.term_events']['term.spoken'])
        self.assertLessEqual(abs(report['score']['count_error']), 2)    # gate edges, 1.5 s segments
        times = [event['start_ms'] - report['start_ms'] for event in report['events']['speech.term_events']]
        self.assertFalse([t for t in times if 3500 < t < 6000], times)  # nothing inside the other voice
        self.assertTrue(json.loads((self.tmp / 'r.json').read_text())['pipeline']['modules']['target_voice']['enabled'])

    def test_a_way_of_speaking_is_enrolled_and_counted(self):
        def call(seconds=1.):
            t = np.arange(int(16000 * seconds)) / 16000
            return (.3 * np.sin(2 * np.pi * (150 * t + 120 * t * t))).astype(np.float32)   # rising pitch
        flat = tone(200, 2.)
        demo = write_wav(self.tmp / 'demo.wav', np.concatenate([flat, call(), flat]))
        recipes.main(['enroll', 'person-voice-style-count', '--audio', demo, '--start', '2', '--seconds', '1',
                      '--out', str(self.tmp / 'style.json'), '--profile', str(self.profile)])
        saved = json.loads((self.tmp / 'style.json').read_text())
        self.assertEqual(saved['mode'], 'sequence')
        self.assertGreater(len(saved['positives'][0]['value']), 4)

    @unittest.skipUnless(HAS_CV, 'needs opencv')
    def test_a_person_is_enrolled_only_from_frames_that_show_one_person(self):
        import cv2
        from gamst.plugins import PerceptionPlugin, _signal
        from tests.test_person_match import Color, person

        class Pose(PerceptionPlugin):
            count = 0

            async def process(self, frame):
                self.count += 1
                people = [person(.25, 'a')] + ([person(.75, 'b')] if self.count % 2 else [])
                return [_signal(frame, 'scene.pose_count', len(people), 'pose', '1'),
                        Derived('pose_joints', PrivatePoses(people), 'video')]

        async def build(options, context):
            return dict(module=Pose())

        async def fake(client, entry):
            return Color('face')

        path = self.tmp / 'me.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (200, 200))
        for _ in range(20):
            writer.write(np.full((200, 200, 3), 120, np.uint8))
        writer.release()
        self.profile.write_text(json.dumps(dict(version=1, roles={'vision.face_identity': dict(
            model='lvface.b', python=sys.executable, config=dict(path='/m.onnx', sha256='a' * 64))})))
        with mock.patch('gamst.recipes.perception_parts._open', fake), \
                mock.patch.object(PARTS['mediapipe-pose'], 'build', build):
            recipes.main(['enroll', 'person-action-count', '--kind', 'person', '--video', str(path), '--seconds',
                          '2', '--consent', 'owner, pose counting', '--out', str(self.tmp / 'me.json'),
                          '--profile', str(self.profile)])
        saved = json.loads((self.tmp / 'me.json').read_text())
        self.assertEqual(set(saved), {'kind', 'recipe', 'face_model', 'body_model', 'face', 'body', 'consent'})
        self.assertEqual((len(saved['face']), saved['body']), (10, []))   # every second frame had one person

    def test_optional_modules_choose_their_links_and_references(self):
        recipe = recipes.load('person-voice-style-count')
        self.assertEqual([item['name'] for item in recipe['references']], ['voice', 'style'])
        more = recipes.load('person-voice-style-count', ['asr', 'speaker_check', 'terms'])
        self.assertEqual([item['name'] for item in more['references']], ['voice', 'style', 'terms'])
        with self.assertRaisesRegex(SystemExit, 'no optional module'):
            recipes.load('person-term-count', ['semantic'])
        action = recipes.load('person-action-count', exclude=['target_person'])
        channels = next(item for item in action['modules'] if item['name'] == 'pose_channels')
        self.assertEqual(channels['consumes'], ['pose_joints'])
        self.assertEqual([item['name'] for item in action['references']], ['exemplar'])

    def test_every_recipe_role_exists_in_the_one_registry(self):
        for name in recipes.names():
            raw = json.loads((recipes.RECIPES / f'{name}.json').read_text())
            optional = [item['name'] for item in raw['modules'] if item.get('optional')]
            steps = recipes.plan(recipes.load(name, optional), hardware=HOST)
            self.assertTrue(steps['roles'], name)


class MathPartTest(unittest.IsolatedAsyncioTestCase):
    async def test_physics_modules_are_recipe_parts(self):
        frames = dict(calibration_id='c', input_frame='a', output_frame='b')
        configs = {'kinematics': frames, 'physical-state': dict(frames, clock_id='clock')}
        for name in MATH:
            if name == 'optical-flow' and not HAS_CV:
                continue
            built = await PARTS[name].build(dict(config=configs.get(name, {})), {})
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'math')
                pipeline.add_input('input', modality=built['spec'].modality)
                pipeline.add_module(name, built['module'], consumes='input', spec=built['spec'])

    @unittest.skipUnless(HAS_CV, 'needs opencv')
    async def test_optical_flow_part_measures_a_moving_pattern(self):
        built = await PARTS['optical-flow'].build({}, {})
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'flow')
            pipeline.add_input('camera', modality='video')
            pipeline.add_module('flow', built['module'], consumes='camera', spec=built['spec'])
            texture = np.random.default_rng(0).integers(0, 255, (120, 160, 3), np.uint8)
            records = []
            for i in range(3):
                records += await pipeline.push('camera', np.roll(texture, 2 * i, axis=1), timestamp_ms=now() - 500 + 40 * i)
            self.assertTrue(records)


class SemanticLaneTest(unittest.IsolatedAsyncioTestCase):
    def test_a_hosted_endpoint_is_an_explicit_opt_in(self):
        url = 'https://example.invalid/v1'
        with self.assertRaisesRegex(ValueError, 'local'):
            OpenAIStreamingBackend(url, 'm', 'r')
        with self.assertRaisesRegex(ValueError, 'https'):
            OpenAIStreamingBackend(url.replace('https', 'http'), 'm', 'r', allow_remote=True)
        with self.assertRaisesRegex(ValueError, 'api_key_env'):
            OpenAIStreamingBackend(url, 'm', 'r', 'GAMST_TEST_UNSET_KEY', allow_remote=True)

    @unittest.skipUnless(HAS_CV, 'needs opencv')
    async def test_one_description_per_second_in_its_own_process_and_it_can_be_switched_off(self):
        seen = []

        class Server(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(body)
                text = json.dumps(dict(state={}, events=[], summary='a desk', evidence_ids=['v']))
                reply = json.dumps(dict(choices=[dict(message=dict(content=text))])).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        built = await PARTS['semantic-scene'].build(dict(every_ms=1000, config=dict(
            endpoint=f'http://127.0.0.1:{server.server_port}/v1', model='m', revision='r')), {})
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'glasses')
            pipeline.add_input('camera', modality='video')
            pipeline.add_module('semantic', built['module'], consumes='camera', spec=built['spec'],
                                every_ms=built['every_ms'], queue=built['queue'])
            start, image = now() - 9000, np.zeros((48, 64, 3), np.uint8)
            for i in range(50):              # 5 s at 10 fps
                await pipeline.push('camera', image, timestamp_ms=start + 100 * i)
                if i % 10 == 0:
                    await pipeline.drain()
            self.assertEqual(len(seen), 5)
            self.assertEqual(sum(item['type'] == 'image_url' for item in seen[0]['messages'][0]['content']), 1)
            pipeline.disable('semantic')
            await pipeline.push('camera', image, timestamp_ms=start + 7000)
            await pipeline.drain()
            self.assertEqual(len(seen), 5)
            history = await human.get_history()
            self.assertEqual(len([r for r in history if r['observation'].get('name') == 'vision.semantic_state']), 5)
            await pipeline.close()


if __name__ == '__main__':
    unittest.main()
