"""Qwen3-ASR phrases with aligned words, and the hosted ASR adapters against a stand-in server (no paid call).

The real model runs when its snapshots are given (in an environment with qwen-asr):
GAMST_TEST_QWEN_ASR=<Qwen3-ASR-1.7B dir>:<Qwen3-ForcedAligner-0.6B dir>  (GAMST_TEST_DEVICE=cpu|mps)
"""
import asyncio
import base64
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest

from gamst.plugins import InputFrame

try:
    import numpy as np
except ImportError:      # pragma: no cover
    np = None


class TextSpanTest(unittest.TestCase):
    def test_aligned_words_find_their_place_in_the_text(self):
        from gamst.providers.qwen_words import split_phrases, text_spans
        text = '뽀또 하나, 주세요! 네 뽀또요.'
        words = ['뽀또', '하나', '주세요', '네', '뽀또요']
        self.assertEqual([text[a:b] for a, b in text_spans(text, words)], words)
        self.assertIsNone(text_spans(text, ['뽀또', '둘']))
        items = [('뽀또', .1, .4), ('하나', .45, .7), ('주세요', .75, 1.1), ('네', 2.0, 2.2), ('뽀또요', 2.25, 2.7)]
        phrases = split_phrases(text, items, .5)
        self.assertEqual([phrase.text for phrase in phrases], ['뽀또 하나, 주세요', '네 뽀또요'])
        self.assertEqual([phrase.end_s for phrase in phrases], [1.1, 2.7])
        # text and words disagree: the words are the text
        self.assertEqual(split_phrases('다른 말', items[:2], .5)[0].text, '뽀또 하나')


class Decoder:
    """Says each (start, end, text) of the absolute timeline that lies inside the decoded samples."""
    name, version, language = 'stand-in', '1', 'ko'

    def __init__(self, said):
        self.said, self.calls = said, 0

    def decode(self, samples):
        from gamst.providers.whisper_phrases import Phrase
        self.calls += 1
        zero, length = self.zero(), len(samples) / 16000
        return [Phrase(text, [(text, start - zero, end - zero)], end - zero)
                for start, end, text in self.said if start >= zero - .01 and end <= zero + length]


@unittest.skipUnless(np is not None, 'needs numpy')
class GenericPhraseTest(unittest.TestCase):
    def test_any_decoder_gives_final_phrases_once_with_capture_times(self):
        from gamst.providers.whisper_phrases import Phrases
        decoder = Decoder([(.5, 1.2, '뽀또'), (2.9, 3.4, '하나'), (7.6, 8.4, '뽀또'), (9.5, 9.9, '네')])
        module = Phrases(decoder, hop_seconds=2., margin_seconds=1., max_seconds=8.)
        decoder.zero = lambda: (module._end_ms - 50_000) / 1000 - len(module._buffer) / 16000
        out = []
        for index in range(100):
            out += module.process(InputFrame(np.full(1600, .1, np.float32), 'audio', 50_000 + 100 * (index + 1),
                                             sample_rate=16000))
        out += module.finish()
        values = [item['value'] for item in out]
        self.assertEqual([value['text'] for value in values], ['뽀또', '하나', '뽀또', '네'])
        self.assertEqual([value['words'][0]['end_ms'] - 50_000 for value in values], [1200, 3400, 8400, 9900])
        self.assertTrue(all(item['name'] == 'speech.transcript_words' for item in out))


class Server:
    """A local stand-in for both hosted APIs; it records what it was sent."""

    def __init__(self):
        self.requests = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                owner.requests.append((self.path, dict(self.headers), body))
                if self.path.startswith('/dashscope'):
                    answer = dict(output=dict(sentence=dict(text='뽀또 주세요', begin_time=100, end_time=900, words=[
                        dict(text='뽀또', begin_time=100, end_time=400, punctuation=''),
                        dict(text='주세요', begin_time=450, end_time=900, punctuation='')])), usage=dict(duration=3))
                else:
                    answer = dict(language_code='kor', text='뽀또 주세요', words=[
                        dict(text='뽀또', type='word', start=.1, end=.4), dict(text=' ', type='spacing', start=.4, end=.45),
                        dict(text='주세요', type='word', start=.45, end=.9), dict(text='(웃음)', type='audio_event', start=1., end=1.5)])
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@unittest.skipUnless(np is not None, 'needs numpy')
class HostedTest(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        self.addCleanup(self.server.close)

    def config(self, api, **extra):
        return dict(api=api, endpoint=f'{self.server.url}/{api}', model='m', **extra)

    def test_dashscope_words_in_milliseconds_become_phrases(self):
        from gamst.providers.hosted_asr import HostedDecoder
        decoder = HostedDecoder(self.config('dashscope', terms=['뽀또']))
        [phrase] = decoder.decode(np.zeros(16000 * 2, np.float32))
        self.assertEqual((phrase.text, phrase.words[0], phrase.end_s), ('뽀또 주세요', ('뽀또', .1, .4), .9))
        _, headers, body = self.server.requests[0]
        sent = json.loads(body)
        self.assertEqual(sent['parameters']['language_hints'], ['ko'])
        self.assertEqual(sent['parameters']['vocabulary'], {'뽀또': 4})
        audio = sent['input']['messages'][0]['content'][0]['input_audio']['data']
        self.assertTrue(base64.b64decode(audio.split(',', 1)[1]).startswith(b'RIFF'))
        self.assertNotIn('Authorization', headers)     # a local endpoint gets no key

    def test_elevenlabs_words_in_seconds_and_key_terms(self):
        from gamst.providers.hosted_asr import HostedDecoder
        decoder = HostedDecoder(self.config('elevenlabs', terms=['뽀또'], pause_seconds=.5))
        [phrase] = decoder.decode(np.zeros(16000 * 2, np.float32))
        self.assertEqual([word for word, _, _ in phrase.words], ['뽀또', '주세요'])     # spacing and events dropped
        _, headers, body = self.server.requests[0]
        self.assertIn(b'name="keyterms"\r\n\r\n\xeb\xbd\x80\xeb\x98\x90', body)
        self.assertIn(b'name="timestamps_granularity"\r\n\r\nword', body)
        self.assertEqual((decoder.requests, decoder.seconds_sent), (1, 2.))

    def test_a_hosted_endpoint_needs_opt_in_https_and_a_key_from_the_environment(self):
        from gamst.providers.hosted_asr import HostedDecoder, check_config
        hosted = dict(api='elevenlabs', endpoint='https://api.example.com/v1/speech-to-text', model='scribe_v2',
                      api_key_env='GAMST_TEST_UNSET_KEY')
        with self.assertRaisesRegex(PermissionError, 'leave this device'):
            check_config(hosted)
        with self.assertRaisesRegex(ValueError, 'https'):
            check_config(dict(hosted, allow_remote=True, endpoint='http://api.example.com/x'))
        with self.assertRaisesRegex(ValueError, 'environment'):
            check_config(dict(hosted, allow_remote=True, api_key='secret'))
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            check_config(dict(hosted, allow_remote=True, endpoint='https://{WorkspaceId}.example.com/x'))
        os.environ.pop('GAMST_TEST_UNSET_KEY', None)
        with self.assertRaisesRegex(ValueError, 'not set'):
            HostedDecoder(dict(hosted, allow_remote=True))
        os.environ['GAMST_TEST_UNSET_KEY'] = 'k'
        self.addCleanup(os.environ.pop, 'GAMST_TEST_UNSET_KEY', None)
        self.assertEqual(HostedDecoder(dict(hosted, allow_remote=True)).key, 'k')

    def test_the_cost_line_says_where_the_sound_goes_and_how_often(self):
        from gamst.providers.hosted_asr import cost_line
        line = cost_line(dict(endpoint='https://api.elevenlabs.io/v1/speech-to-text', model='scribe_v2',
                              price='$0.20 per hour'), dict(hop_seconds=8., max_seconds=24.))
        self.assertIn('api.elevenlabs.io', line)
        self.assertIn('$0.20 per hour', line)
        self.assertIn('3 times', line)

    def test_the_worker_gives_transcript_words_records(self):
        import sys
        from gamst.providers import model_workers

        async def run():
            module = await model_workers.RemoteWhisper.open(
                sys.executable, 'hosted.asr.words', dict(self.config('elevenlabs'), phrases=dict(hop_seconds=2., margin_seconds=1.)),
                startup_timeout=60)
            try:
                out = []
                for index in range(30):
                    out += await module.process(InputFrame(np.full(1600, .1, np.float32), 'audio', 10_000 + 100 * index,
                                                           sample_rate=16000)) or []
                return out + (await module.finish() or [])
            finally:
                await module.close()
        values = asyncio.run(run())
        self.assertTrue(values)
        self.assertEqual(values[0]['words'][0]['text'], '뽀또')
        self.assertEqual(values[0]['status'], 'final')


class RegistryTest(unittest.TestCase):
    def test_qwen_runs_on_a_mac_from_pinned_snapshots(self):
        from gamst.crew.registry import ROLES
        asr = ROLES['audio.asr'].candidates['qwen3.asr_1_7b.transformers']
        aligner = ROLES['audio.word_alignment'].candidates['qwen3.forced_aligner_0_6b']
        for candidate in (asr, aligner):
            self.assertIn('mps', candidate.devices)
            self.assertTrue(candidate.bundle and all(weight.sha256 and weight.url for weight in candidate.weights))
            self.assertTrue(all('/resolve/' in weight.url and weight.url.split('/resolve/')[1][:40].isalnum()
                                for weight in candidate.weights))
        self.assertEqual(asr.environment, aligner.environment)       # one worker holds both
        for name in ('alibaba.qwen_audio_3_1_asr_flash.hosted', 'elevenlabs.scribe_v2.hosted'):
            hosted = ROLES['audio.asr'].candidates[name]
            self.assertFalse(hosted.automatic or hosted.config['allow_remote'])
            self.assertNotIn('api_key', hosted.config)

    def test_the_recipe_part_sets_up_the_aligner_too(self):
        from gamst.recipes import roles
        recipe = dict(name='r', inputs=[dict(id='microphone', use='microphone')], view='count',
                      modules=[dict(name='asr', use='qwen-words', consumes=['microphone'])])
        found = roles(recipe)
        self.assertEqual((found['audio.asr'], found['audio.word_alignment']),
                         ('qwen3.asr_1_7b.transformers', 'qwen3.forced_aligner_0_6b'))

    def test_setup_writes_the_worker_config(self):
        from gamst.crew.setup import _role_config
        weights = [dict(name=name, path=f'/m/{name}', sha256=str(index) * 64) for index, name in
                   enumerate(('config.json', 'model-00001-of-00002.safetensors'))]
        config = _role_config(dict(model='qwen3.asr_1_7b.transformers', config=dict(device='cpu')), weights, None)
        self.assertEqual((config['model_path'], config['sha256'], len(config['files'])), ('/m', '1' * 64, 2))
        config = _role_config(dict(model='qwen3.forced_aligner_0_6b'), [dict(name='model.safetensors', path='/a/model.safetensors',
                                                                             sha256='2' * 64)], None)
        self.assertEqual(config, dict(path='/a', sha256='2' * 64))


QWEN = os.environ.get('GAMST_TEST_QWEN_ASR', '')


@unittest.skipUnless(QWEN and shutil.which('say'), 'set GAMST_TEST_QWEN_ASR=<asr dir>:<aligner dir> (macOS say for speech)')
class RealQwenTest(unittest.TestCase):
    def test_a_spoken_word_is_written_with_its_time(self):
        import wave
        from gamst.crew.registry import QWEN_ALIGNER_FILES, QWEN_ASR_FILES
        from gamst.providers.qwen_words import QwenDecoder
        asr, aligner = QWEN.split(':')
        files = {name: sha for name, (_, sha) in QWEN_ASR_FILES.items()}
        aligner_sha = QWEN_ALIGNER_FILES['model.safetensors'][1]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'say.wav')
            subprocess.run(['say', '-v', 'Yuna', '-o', path, '--data-format=LEI16@16000', '안녕하세요. 과자 하나 주세요.'], check=True)
            with wave.open(path) as handle:
                samples = np.frombuffer(handle.readframes(handle.getnframes()), '<i2').astype(np.float32) / 32768
        decoder = QwenDecoder.from_local(asr, files, aligner, aligner_sha, device=os.environ.get('GAMST_TEST_DEVICE', 'cpu'))
        phrases = decoder.decode(samples)
        words = [word for phrase in phrases for word in phrase.words]
        # A synthetic voice, so only that words come back in order with times; '뽀또' itself was heard as '포도'
        # (2026-11-26): which words are heard is measured on real recordings, not here.
        self.assertTrue(any('주세요' in word for word, _, _ in words), words)
        self.assertEqual([start for _, start, _ in words], sorted(start for _, start, _ in words))
        self.assertTrue(all(0 <= start <= end <= len(samples) / 16000 + .1 for _, start, end in words))


if __name__ == '__main__':
    unittest.main()
