"""Overlapping voices: separated sources scored per word, hosted identification as evidence, the PixIT model."""
import copy
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.pyannoteai import HostedIdentify, PyannoteAI, overlap_ms, turns
from gamst.providers.speaker_verify import SpeakerVerify, centroid
from gamst.providers.target_speech import SourceBlock, SpeakerSources
from tests.test_speaker_verify import ToneEmbedder, now, tone

KEY = 'test-key-' + 'x' * 24   # a stand-in; no real key is used or stored by these tests


class Split:
    """Test double separator: the window's low and high bands as two 'speakers'."""
    model_id = 'split'

    def separate(self, window):
        spectrum = np.fft.rfft(window)
        low, high = spectrum.copy(), spectrum.copy()
        low[len(spectrum) // 8:], high[:len(spectrum) // 8] = 0, 0
        return np.stack([np.fft.irfft(low, len(window)), np.fft.irfft(high, len(window))]).astype(np.float32)


def utterance(start, words):
    return dict(utterance_id='u', revision=0, status='final', text=' '.join(w for w, _, _ in words),
                start_ms=start + words[0][1], end_ms=start + words[-1][2], language='ko',
                words=[dict(word=w, start_ms=start + a, end_ms=start + b) for w, a, b in words])


class SourcesTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocks_carry_left_context_and_are_never_copied(self):
        module, out = SpeakerSources(Split(), emits='sources', block_ms=1000, context_ms=500), []
        for i in range(25):
            out += await module.process(InputFrame(tone(220, .1), 'audio', 1000 + 100 * (i + 1), sample_rate=16000))
        out += await module.finish()
        blocks = [item.data for item in out]
        self.assertEqual([(b.start_ms, b.keep_ms, b.end_ms) for b in blocks],
                         [(1000, 1000, 2000), (1500, 2000, 3000), (2500, 3000, 3500)])
        self.assertEqual(blocks[1].sources.shape, (2, 24000))
        for action in (pickle.dumps, copy.copy, copy.deepcopy):
            with self.assertRaises(TypeError):
                action(blocks[0])

    async def test_a_quiet_voice_under_a_loud_one_is_scored_on_its_own_source(self):
        """The enrolled voice (220 Hz) overlaps a louder other voice (1500 Hz): the mixture scores low,
        the separated source of the person decides."""
        embedder = ToneEmbedder()
        target = centroid([embedder.embed(tone(220, 1.))])
        verify = SpeakerVerify(embedder, threshold=.9, segments=False, context_ms=1500, words=True,
                               sources='sources')
        verify.set_reference(dict(model='test-tone', embeddings=[target]))
        start = 10000
        mixture = np.concatenate([tone(220, 2., amplitude=.05) + tone(1500, 2., amplitude=.5), tone(1500, 2.)])
        for i in range(0, len(mixture), 1600):
            await verify.process(InputFrame(mixture[i:i + 1600], 'audio', start + (i + 1600) // 16, sample_rate=16000))
        asked = utterance(start, [('뽀또', 500, 900), ('뽀또', 2500, 2900)])
        held = await verify.process(InputFrame(asked, 'audio', start + 3000))
        self.assertEqual(held, [])                          # waits for the separated blocks
        self.assertEqual(verify.pending_since_ms, start + 500)
        block = SourceBlock(start, start, start + 4000, Split().separate(mixture))
        out = await verify.process(InputFrame(block, 'audio', start + 4000))
        first, second = out[0]['words']
        self.assertLess(first['mixture_score'], .9)
        self.assertGreaterEqual(first['speaker_score'], .9)
        self.assertEqual((first['match'], first['scored_on'], first['sources_active']), (True, 'sources', 2))
        self.assertFalse(second['match'])
        self.assertIsNone(verify.pending_since_ms)

    async def test_a_word_no_block_covers_keeps_the_mixture_score_at_the_end(self):
        embedder = ToneEmbedder()
        verify = SpeakerVerify(embedder, threshold=.9, segments=False, context_ms=1500, words=True, sources='sources')
        verify.set_reference(dict(model='test-tone', embeddings=[embedder.embed(tone(220, 1.))]))
        for i in range(20):
            await verify.process(InputFrame(tone(220, .1), 'audio', 100 * (i + 1), sample_rate=16000))
        self.assertEqual(await verify.process(InputFrame(utterance(0, [('뽀또', 800, 1200)]), 'audio', 2000)), [])
        word = (await verify.finish())[0]['words'][0]
        self.assertEqual((word['scored_on'], word['match'], word['speaker_score']), ('mixture', True, word['mixture_score']))

    async def test_turns_are_evidence_beside_the_voice_score(self):
        embedder = ToneEmbedder()
        verify = SpeakerVerify(embedder, threshold=.9, segments=False, context_ms=1500, words=True, turns=True)
        verify.set_reference(dict(model='test-tone', embeddings=[embedder.embed(tone(220, 1.))]))
        for i in range(30):
            await verify.process(InputFrame(tone(1500, .1), 'audio', 100 * (i + 1), sample_rate=16000))
        self.assertEqual(await verify.process(InputFrame(utterance(0, [('a', 500, 900), ('b', 2000, 2400)]),
                                                         'audio', 3000)), [])
        record = dict(kind='turns', reference='voice', start_ms=0, end_ms=3000, speakers=2, overlap_ms=0, model={},
                      turns=[dict(speaker='S0', start_ms=400, end_ms=1000, match=True, confidence=.8),
                             dict(speaker='S1', start_ms=1900, end_ms=2600, match=False)])
        first, second = (await verify.process(InputFrame(record, 'audio', 3000)))[0]['words']
        self.assertEqual((first['turn_match'], first['turn_confidence'], first['match']), (True, .8, False))
        self.assertEqual((second['turn_match'], second['turn_speakers']), (False, 1))


# ---------------------------------------------------------------- pyannoteAI against a stand-in server

class StandIn(BaseHTTPRequestHandler):
    """The documented pyannoteAI API shape, answered locally; records what it was sent."""
    log, jobs = [], {}

    def log_message(self, *_):
        pass

    def _send(self, code, value):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        size = int(self.headers['Content-Length'])
        StandIn.log.append(('PUT', self.path, self.headers.get('Authorization'), self.rfile.read(size)[:4]))
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        StandIn.log.append(('POST', self.path, self.headers.get('Authorization'), body))
        if self.headers.get('Authorization') != f'Bearer {KEY}':
            return self._send(401, dict(message='bad key', requestId='r'))
        if self.path == '/v1/media/input':
            return self._send(201, dict(url=f'http://127.0.0.1:{self.server.server_port}/upload/{body["url"][8:]}'))
        job = f'job{len(StandIn.jobs)}'
        if self.path == '/v1/voiceprint':
            StandIn.jobs[job] = dict(voiceprint='VP-stand-in')
        else:
            StandIn.jobs[job] = dict(identification=[
                dict(speaker='voice', diarizationSpeaker='SPEAKER_00', match='voice', start=.5, end=1.5,
                     confidence={'voice': 77}),
                dict(speaker='SPEAKER_01', diarizationSpeaker='SPEAKER_01', match=None, start=1., end=2.5)])
        return self._send(200, dict(jobId=job, status='created'))

    def do_GET(self):
        StandIn.log.append(('GET', self.path, self.headers.get('Authorization'), None))
        self._send(200, dict(jobId=self.path.rsplit('/', 1)[1], status='succeeded',
                             output=StandIn.jobs[self.path.rsplit('/', 1)[1]]))


class HostedTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        StandIn.log, StandIn.jobs = [], {}
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), StandIn)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}'
        os.environ['GAMST_TEST_PYANNOTEAI_KEY'] = KEY

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        os.environ.pop('GAMST_TEST_PYANNOTEAI_KEY', None)

    def client(self, out):
        return PyannoteAI(endpoint=self.endpoint, api_key_env='GAMST_TEST_PYANNOTEAI_KEY', eur_per_hour=.112,
                          eur_per_voiceprint=.015, allow_remote=True, poll_s=0, out=out)

    def test_audio_leaves_the_device_only_by_opt_in_and_only_over_https(self):
        with self.assertRaisesRegex(PermissionError, 'leaves this device'):
            PyannoteAI(eur_per_hour=1, eur_per_voiceprint=1)
        with self.assertRaisesRegex(ValueError, 'https'):
            PyannoteAI(endpoint='http://api.example.com', eur_per_hour=1, eur_per_voiceprint=1, allow_remote=True)
        with self.assertRaisesRegex(PermissionError, 'GAMST_TEST_UNSET'):
            PyannoteAI(endpoint=self.endpoint, api_key_env='GAMST_TEST_UNSET', eur_per_hour=1, eur_per_voiceprint=1,
                       allow_remote=True)._key()

    async def test_voiceprint_once_then_turns_with_a_cost_line_before_each_call(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            stored = Path(directory) / 'voiceprint'
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'hosted')
                pipeline.add_input('microphone', modality='audio')
                pipeline.add_reference('voice', 'voice', dict(samples=[tone(220, 40.)], sample_rate=16000),
                                       consent=dict(by='owner'))
                module = HostedIdentify(self.client(out), block_ms=20000, voiceprint_file=str(stored))
                pipeline.add_module('identify', module, consumes='microphone', spec=module.spec)
                start, audio = now() - 60000, tone(300, 25.)
                for i in range(0, len(audio), 16000):
                    await pipeline.push('microphone', audio[i:i + 16000], timestamp_ms=start + (i + 16000) // 16,
                                        sample_rate=16000)
                records = await pipeline.finish()
                history = await human.get_history()
            self.assertEqual(oct(stored.stat().st_mode & 0o777), '0o600')
            self.assertEqual(stored.read_text(), 'VP-stand-in')
        lines = out.getvalue().splitlines()
        self.assertEqual([line.split(':')[0] for line in lines],
                         ['pyannoteAI voiceprint', 'pyannoteAI identify', 'pyannoteAI identify'])
        self.assertIn('30.0 s', lines[0])                    # the clip is cut to the 30 s a voiceprint allows
        self.assertIn('billed as 20.0 s', lines[2])          # the 5 s rest is billed as the 20 s minimum
        self.assertIn('0.0006 EUR', lines[1])                # 20 s at 0.112 EUR/h
        posts = [item for item in StandIn.log if item[0] == 'POST']
        self.assertEqual([item[1] for item in posts].count('/v1/voiceprint'), 1)
        identify = next(item[3] for item in posts if item[1] == '/v1/identify')
        self.assertEqual(identify['voiceprints'], [dict(label='voice', voiceprint='VP-stand-in')])
        self.assertTrue(all(item[2] is None for item in StandIn.log if item[0] == 'PUT'))   # storage gets no key
        self.assertEqual({item[3] for item in StandIn.log if item[0] == 'PUT'}, {b'RIFF'})
        value = records[0]['observation']['value'] if records else history[0]['observation']['value']
        self.assertEqual(value['turns'][0]['match'], True)
        self.assertEqual(value['speakers'], 2)
        self.assertNotIn(KEY, json.dumps(history))
        self.assertNotIn('VP-stand-in', json.dumps(history))
        self.assertNotIn(KEY, out.getvalue())

    def test_turn_conversion_and_overlap(self):
        found = turns(dict(identification=[dict(diarizationSpeaker='A', match='voice', start=1, end=2,
                                                confidence={'voice': 90}),
                                           dict(diarizationSpeaker='B', match=None, start=1.5, end=3)]), 'voice', 1000)
        self.assertEqual(found[0], dict(speaker='A', start_ms=2000, end_ms=3000, match=True, confidence=90.))
        self.assertEqual(overlap_ms(found), 500)


# ---------------------------------------------------------------- the real PixIT model, in its own environment

PIXIT = Path(os.environ.get('GAMST_TEST_PIXIT', Path.home() / '.cache' / 'gamst' / 'models' / 'pyannote.pixit_ami'))
PIXIT_PYTHON = os.environ.get('GAMST_TEST_PIXIT_PYTHON', '')
PIXIT_SHA256 = '3c94804b873ecc6c972dd095f1c33a4f9a5c9802dabf0275067ecbdcf76bf024'


@unittest.skipUnless((PIXIT / 'pytorch_model.bin').is_file() and Path(PIXIT_PYTHON).is_file(),
                     'PixIT weights (GAMST_TEST_PIXIT) and its pyannote.audio 3.3.2 interpreter '
                     '(GAMST_TEST_PIXIT_PYTHON) not available')
class PixITTest(unittest.IsolatedAsyncioTestCase):
    """Two macOS TTS voices, one after the other with 1 s together: PixIT gives each its own source."""

    async def test_two_voices_get_two_sources_and_a_solo_stretch_one(self):
        if sys.platform != 'darwin' or not shutil.which('say'):
            self.skipTest('needs macOS say for test voices')
        from gamst.providers.model_workers import RemoteSeparator
        from tests.test_speaker_verify import _tts
        with tempfile.TemporaryDirectory() as directory:
            yuna = _tts(directory, 'Yuna', '오늘 방송에서는 새로운 게임을 해 보겠습니다 정말 기대가 되네요', 'a')
            eddy = _tts(directory, 'Eddy (Korean (South Korea))', '잠깐만요 채팅 좀 읽어볼게요 여러분 고맙습니다', 'b')
        offset = len(yuna) - 16000
        mixture = np.zeros(offset + len(eddy), np.float32)
        mixture[:len(yuna)] += yuna
        mixture[offset:] += eddy
        config = dict(config=str(PIXIT / 'config.yaml'), separation=str(PIXIT / 'pytorch_model.bin'),
                      embedding=str(PIXIT.parent / 'speechbrain.spkrec_ecapa_voxceleb'),
                      wavlm=str(PIXIT.parent / 'wavlm-large-config'), sha256=PIXIT_SHA256)
        separator = await RemoteSeparator.open(PIXIT_PYTHON, 'pixit.sources', config, inference_timeout=600)
        try:
            sources = await separator.separate(mixture)
        finally:
            await separator.close()
        self.assertGreaterEqual(sources.shape[0], 2)
        self.assertEqual(sources.shape[1], len(mixture))
        solo = sources[:, 16000:offset - 8000]                  # only the first voice
        loud = np.sqrt((solo ** 2).mean(axis=1))
        self.assertEqual(int((loud > loud.max() * .1).sum()), 1)


if __name__ == '__main__':
    unittest.main()
