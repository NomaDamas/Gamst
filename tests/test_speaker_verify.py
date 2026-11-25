import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.audio_ring import AudioRing, HopClock, resample
from gamst.providers.speaker_verify import Hysteresis, SpeakerVerify, centroid, cosine, enroll
from gamst.providers.term_events import TermEvents

MODELS = Path(os.environ.get('GAMST_MODELS', Path.home() / '.cache' / 'gamst' / 'models'))
CAMPP = Path(os.environ.get('GAMST_TEST_CAMPP', MODELS / 'speaker' /
                            '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'))
CAMPP_SHA256 = '53cab20ba8f2eb2f6e2d89ecc18e88ef3222169b61b4419cbc13666744a4574b'


def now():
    return time.time_ns() // 1_000_000


class ToneEmbedder:
    """Test double: the embedding is the normalized magnitude spectrum in 8 bands (pitch-sensitive)."""
    model_id = 'test-tone'

    def embed(self, samples, sample_rate=16000):
        spectrum = np.abs(np.fft.rfft(np.asarray(samples, np.float32)))
        bands = np.array_split(spectrum[:len(spectrum) // 4], 8)
        return [float(band.mean()) + 1e-6 for band in bands]


def tone(hz, seconds, rate=16000, amplitude=.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * math.pi * hz * t)).astype(np.float32)


class MathTest(unittest.TestCase):
    def test_centroid_weighs_each_clip_the_same_and_cosine_is_scale_free(self):
        self.assertAlmostEqual(cosine([1, 0], [10, 0]), 1.)
        self.assertAlmostEqual(cosine([1, 0], [0, 3]), 0.)
        middle = centroid([[1, 0], [0, 100]])
        self.assertAlmostEqual(middle[0], middle[1])
        with self.assertRaises(ValueError):
            centroid([[1, 0], [1, 0, 0]])

    def test_hysteresis_holds_through_a_dip_above_exit(self):
        state = Hysteresis(.5, .4)
        self.assertEqual([state.update(s) for s in (.3, .55, .45, .41, .39, .45, .5)],
                         [False, True, True, True, False, False, True])

    def test_ring_cuts_by_capture_time_and_hop_clock_keeps_a_gap_free_sample_clock(self):
        ring = AudioRing(16000, 5)
        for i in range(10):                       # 100 ms chunks whose values are their index
            ring.append(np.full(1600, i / 10, np.float32), 1000 + 100 * (i + 1))
        cut = ring.span(1250, 1450)
        self.assertEqual(len(cut), 3200)
        self.assertAlmostEqual(float(cut[0]), .2, places=5)
        self.assertAlmostEqual(float(cut[-1]), .4, places=5)
        clock = HopClock(16000, 20)
        hops, gap = clock.push(np.zeros(1000, np.float32), 1000)   # 3 hops + 40 samples pending
        self.assertEqual([end for end, _ in hops], [958, 978, 998])
        hops, gap = clock.push(np.zeros(600, np.float32), 1037)
        self.assertFalse(gap)
        self.assertEqual([end for end, _ in hops], [1018, 1038])
        jittered = AudioRing(16000, 5)             # a live callback arrives early or late by a few ms
        for i, jitter in enumerate((0, 7, -9, 4)):
            self.assertFalse(jittered.append(np.zeros(1600, np.float32), 1000 + 100 * i + jitter))
        self.assertEqual(jittered.end_ms - jittered.start_ms, 400)
        self.assertTrue(jittered.append(np.zeros(1600, np.float32), 5000))
        hops, gap = clock.push(np.zeros(320, np.float32), 2000)    # capture jumped: discontinuity
        self.assertTrue(gap)

    def test_resample_keeps_a_tone_and_its_level(self):
        source = tone(440, 1.)
        up = resample(source, 16000, 48000)
        self.assertEqual(len(up), 48000)
        spectrum = np.abs(np.fft.rfft(up))
        self.assertAlmostEqual(np.argmax(spectrum) * 48000 / len(up), 440, delta=2)
        self.assertAlmostEqual(float(np.sqrt(np.mean(up[1000:-1000] ** 2))), .3 / math.sqrt(2), delta=.01)
        down = resample(up, 48000, 8000)
        self.assertEqual(len(down), 8000)


class ModuleTest(unittest.IsolatedAsyncioTestCase):
    async def test_segments_follow_the_enrolled_voice_and_terms_count_only_its_utterances(self):
        embedder = ToneEmbedder()
        reference = await enroll(embedder, [tone(220, 1.), tone(220, 1.5)])
        self.assertEqual(reference['model'], 'test-tone')
        self.assertNotIn('samples', reference)
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'voice')
            pipeline.add_input('microphone', modality='audio')
            pipeline.add_reference('voice', 'voice', reference, consent=dict(by='owner', purpose='test'))
            pipeline.add_reference('terms', 'terms', ['대박'])
            pipeline.add_input('asr', modality='audio')
            verify = SpeakerVerify(embedder, threshold=.9, hysteresis=.05, segment_ms=1000, hop_ms=500)
            pipeline.add_module('speaker', verify, consumes=['microphone', 'asr'], spec=verify.spec)
            terms = TermEvents()
            pipeline.add_module('terms', terms, consumes='speaker', spec=terms.spec)
            start = now() - 10000
            audio = np.concatenate([tone(220, 2.), tone(1500, 2.), np.zeros(16000, np.float32)])
            records = []
            for i in range(0, len(audio), 1600):
                records += await pipeline.push('microphone', audio[i:i + 1600], timestamp_ms=start + (i + 1600) // 16,
                                               sample_rate=16000)
            segments = [r['observation']['value'] for r in records
                        if r['observation']['value'].get('kind') == 'segment']
            states = [(s['end_ms'] - start, s['state']) for s in segments]
            self.assertEqual([s for t, s in states if t <= 2000], ['match'] * 3)
            self.assertTrue(all(s == 'other' for t, s in states if 3000 <= t <= 4000))
            self.assertTrue(all(s == 'silent' for t, s in states if t >= 4600))
            # Two final utterances: the enrolled voice says the term once, another voice once.
            for utterance, (begin, end) in (('a', (0, 2000)), ('b', (2000, 4000))):
                await pipeline.push('asr', dict(utterance_id=utterance, revision=0, status='final', text='대박이다',
                                                start_ms=start + begin, end_ms=start + end, language='ko'),
                                    timestamp_ms=start + end + 5000)
            counted = await human.count_events('speech.term_events', window='24h')
            self.assertEqual(counted['count'], 1)

    async def test_text_patterns_are_terms(self):
        terms = TermEvents(terms=['re:(거든요|잖아요)$'])
        value = terms.process(InputFrame(dict(utterance_id='a', status='final', text='제가 했거든요!', start_ms=1, end_ms=2),
                                         'audio', 2))
        self.assertEqual([event['spoken'] for event in value['events']], ['거든요'])
        self.assertEqual(TermEvents(terms=[r're:\S']).terms, {r're:\S': [r're:\S']})
        with self.assertRaises(ValueError):
            TermEvents(terms=['re:('])

    def test_a_reference_enrolled_with_another_model_is_refused(self):
        verify = SpeakerVerify(ToneEmbedder())
        with self.assertRaises(ValueError):
            verify.set_reference(dict(model='other-model', embeddings=[[1., 0.]]))


def _tts(directory, voice, text, name):
    aiff, wav = Path(directory) / f'{name}.aiff', Path(directory) / f'{name}.wav'
    subprocess.run(['say', '-v', voice, '-o', str(aiff), text], check=True)
    subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', str(aiff), str(wav)], check=True)
    from gamst.providers.target_speech import read_wav
    return read_wav(wav)[0]


@unittest.skipUnless(CAMPP.is_file(), f'CAM++ ONNX not installed at {CAMPP}')
class CamPlusPlusTest(unittest.IsolatedAsyncioTestCase):
    """Runs the real CAM++ model (sherpa-onnx). Voices are macOS Korean TTS voices, not people."""

    async def test_same_voice_scores_above_other_voice(self):
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            self.skipTest('sherpa-onnx not installed')
        if sys.platform != 'darwin' or not shutil.which('say'):
            self.skipTest('needs macOS say for test voices')
        from gamst.providers.speaker_verify import SherpaOnnxEmbedder
        embedder = SherpaOnnxEmbedder.from_local(CAMPP, sha256=CAMPP_SHA256)
        lines = ('오늘 방송에서는 새로운 게임을 해 보겠습니다', '대박 이거 진짜 말도 안 되는데요',
                 '잠깐만요 채팅 좀 읽어볼게요 여러분')
        with tempfile.TemporaryDirectory() as directory:
            yuna = [_tts(directory, 'Yuna', line, f'yuna{i}') for i, line in enumerate(lines)]
            eddy = [_tts(directory, 'Eddy (Korean (South Korea))', line, f'eddy{i}') for i, line in enumerate(lines)]
        self.assertEqual(embedder.dim, len(embedder.embed(yuna[0])))
        target = centroid((await enroll(embedder, yuna[:2]))['embeddings'])
        same = cosine(embedder.embed(yuna[2]), target)
        other = cosine(embedder.embed(eddy[2]), target)
        print(f'\nCAM++ zh_en advanced: same-voice {same:.3f}, other-voice {other:.3f}')
        self.assertGreater(same, other)


if __name__ == '__main__':
    unittest.main()
