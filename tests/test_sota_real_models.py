"""Invokes each real model when its local files are given; skipped otherwise (nothing is downloaded).

GAMST_TEST_WESPEAKER=<model dir>            GAMST_TEST_WESEP=<model dir>
GAMST_TEST_USEF=<source>:<checkpoint>       GAMST_TEST_QWEN_ALIGNER=<snapshot dir>
GAMST_TEST_WHISPER=<faster-whisper dir>     GAMST_TEST_LVFACE=<onnx>
GAMST_TEST_SOLIDER=<source>:<checkpoint>    GAMST_TEST_SAPIENS2=<pose dir>:<yolox onnx>
GAMST_TEST_PERSON_IMAGE=<image of a person> GAMST_TEST_DEVICE=cpu|mps|cuda (default cpu)

These check that the official package loads the weights and answers in the declared shape. They measure
no accuracy; that needs a human-counted take (``gamst recipe run --truth``). CAM++ has its own test in
``tests/test_speaker_verify.py``.
"""
import os
import unittest

import numpy as np

from tests.test_speaker_verify import tone

DEVICE = os.environ.get('GAMST_TEST_DEVICE', 'cpu')


def given(name):
    value = os.environ.get('GAMST_TEST_' + name)
    return unittest.skipUnless(value, f'set GAMST_TEST_{name} to local files'), (value or '').split(':')


WESPEAKER, wespeaker = given('WESPEAKER')
WESEP, wesep = given('WESEP')
USEF, usef = given('USEF')
ALIGNER, aligner = given('QWEN_ALIGNER')
WHISPER, whisper = given('WHISPER')
LVFACE, lvface = given('LVFACE')
SOLIDER, solider = given('SOLIDER')
SAPIENS2, sapiens2 = given('SAPIENS2')
IMAGE = os.environ.get('GAMST_TEST_PERSON_IMAGE')


def speech():
    """A voiced, modulated signal; not a person."""
    t = np.arange(48000) / 16000
    return (tone(150, 3.) * (.6 + .4 * np.sin(2 * np.pi * 3 * t)) + .3 * tone(450, 3.)).astype(np.float32)


class RealModelTest(unittest.IsolatedAsyncioTestCase):
    @WESPEAKER
    def test_wespeaker(self):
        from gamst.providers.speaker_verify import WeSpeakerEmbedder, cosine
        model = WeSpeakerEmbedder.from_local(wespeaker[0], device=DEVICE)
        first = model.embed(speech())
        self.assertEqual(len(first), 256)
        self.assertAlmostEqual(cosine(first, model.embed(speech())), 1., places=3)

    def extraction(self, model):
        estimate = model.extract(speech(), tone(150, 3.))
        self.assertEqual((estimate.shape, estimate.dtype), ((48000,), np.float32))
        self.assertTrue(np.isfinite(estimate).all())

    @WESEP
    def test_wesep(self):
        from gamst.providers.target_speech import WeSepExtractor
        self.extraction(WeSepExtractor.from_local(wesep[0], device=DEVICE))

    @USEF
    def test_usef(self):
        from gamst.providers.target_speech import USEFExtractor
        self.extraction(USEFExtractor.from_local(usef[0], usef[1], device=DEVICE, allow_noncommercial=True))

    @ALIGNER
    def test_qwen_forced_aligner(self):
        from gamst.providers.asr_words import QwenForcedAligner
        model = QwenForcedAligner.from_local(aligner[0], device=DEVICE)
        items = model.align(speech(), 16000, '대박 진짜', 'ko')
        self.assertTrue(items and all(0 <= item['start_s'] <= item['end_s'] <= 3.1 for item in items))

    @WHISPER
    async def test_faster_whisper_words(self):
        from gamst.plugins import FasterWhisperPlugin, InputFrame
        plugin = FasterWhisperPlugin(whisper[0], language='ko', word_timestamps=True)
        for signal in await plugin.process(InputFrame(speech(), 'audio', 3000, sample_rate=16000)):
            if signal['name'] == 'speech.transcript_words':
                self.assertTrue(all(0 <= word['start_ms'] <= word['end_ms'] for word in signal['value']['words']))

    @LVFACE
    def test_lvface(self):
        from gamst.providers.person_match import LVFaceEmbedder
        model = LVFaceEmbedder.from_local(lvface[0])
        face = np.random.default_rng(0).integers(0, 255, (112, 112, 3), np.uint8)
        self.assertEqual(len(model.embed(face)), 512)
        np.testing.assert_allclose(model.embed(face), model.embed(face), rtol=1e-4)

    @SOLIDER
    def test_solider(self):
        from gamst.providers.person_match import SOLIDERBodyEmbedder
        model = SOLIDERBodyEmbedder.from_local(solider[0], solider[1], device=DEVICE)
        self.assertEqual(len(model.embed(np.zeros((384, 128, 3), np.uint8))), 1024)

    @SAPIENS2
    @unittest.skipUnless(IMAGE, 'set GAMST_TEST_PERSON_IMAGE')
    def test_sapiens2(self):
        import cv2
        from pathlib import Path
        from gamst.providers.rtmw_pose import _sha256
        from gamst.providers.sapiens2_pose import Sapiens2Models
        models = Sapiens2Models(sapiens2[0], _sha256(Path(sapiens2[0]) / 'model.safetensors'), sapiens2[1],
                                _sha256(sapiens2[1]), device=DEVICE, det_device='cpu', accept_license=True)
        boxes, keypoints, scores = models.infer(cv2.imread(IMAGE))
        self.assertGreaterEqual(len(boxes), 1)
        self.assertEqual((keypoints.shape[1:], scores.shape[1:]), ((308, 2), (308,)))


if __name__ == '__main__':
    unittest.main()
