import unittest

import numpy as np

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.target_speech import (Blocks, SAMSpanSpeech, SpeakerGate, TargetSpeech, USEFExtractor,
                                           enrollment_audio, span_anchors)
from tests.test_speaker_verify import now, tone


class Recorder:
    """Test double extractor: returns the mixture, remembers what it was given."""

    def __init__(self):
        self.calls = []

    def extract(self, mixture, enrollment):
        self.calls.append((len(mixture), len(enrollment)))
        return mixture


class Sink:
    name, version = 'sink', '1'

    def __init__(self):
        self.frames = []

    def process(self, frame):
        self.frames.append(frame)
        return []


class BlockTest(unittest.TestCase):
    def test_every_sample_is_kept_once_with_left_context(self):
        blocks = Blocks(block_ms=1000, context_ms=500)
        samples, out = np.arange(16000 * 3 + 800, dtype=np.float32) / 1e6, []
        for i in range(0, len(samples), 700):
            out += blocks.push(samples[i:i + 700], 10000 + round((i + 700) / 16))
        out += blocks.flush()
        kept = np.concatenate([window[-keep:] for _, _, window, keep in out])
        np.testing.assert_array_equal(kept, samples)
        self.assertEqual([len(window) for _, _, window, _ in out][:3], [16000, 24000, 24000])
        self.assertEqual(out[1][1], out[2][0])

    def test_anchors_prefer_the_person_and_never_overlap(self):
        anchors = span_anchors([(0, 1500, False), (500, 1500, True), (1400, 2200, True), (1900, 2000, False)], 0, 2000)
        self.assertEqual(anchors, [('-', 0., .5), ('+', .5, 2.)])
        self.assertEqual(span_anchors([(0, 1000, False)], 0, 2000, include_other=False), [])

    def test_embeddings_cannot_condition_an_extractor_and_noncommercial_needs_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'embeddings'):
            enrollment_audio(dict(embeddings=[[1.]]))
        with self.assertRaisesRegex(PermissionError, 'NC'):
            USEFExtractor.from_local('src', 'checkpoint')


class ModuleTest(unittest.IsolatedAsyncioTestCase):
    async def test_separated_voice_is_a_derived_input_with_its_own_capture_times(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'voice')
            pipeline.add_input('microphone', modality='audio')
            pipeline.add_reference('me', 'voice', dict(samples=[tone(200, .5)], sample_rate=16000),
                                   consent=dict(by='owner'))
            extractor, sink = Recorder(), Sink()
            module = TargetSpeech(extractor, reference='me', block_ms=1000, context_ms=1000)
            pipeline.add_module('target', module, consumes='microphone', emits=('voice:me',))
            pipeline.add_module('sink', sink, consumes='voice:me')   # an input id may contain ':'
            start, audio = now() - 9000, tone(300, 2.5)
            for i in range(0, len(audio), 1600):
                await pipeline.push('microphone', audio[i:i + 1600], timestamp_ms=start + (i + 1600) // 16,
                                    sample_rate=16000)
            await pipeline.finish()
            self.assertEqual([frame.timestamp_ms - start for frame in sink.frames], list(range(100, 2600, 100)))
            np.testing.assert_allclose(np.concatenate([f.data for f in sink.frames]), audio)
            self.assertEqual(extractor.calls[0][1], 8000)
            self.assertEqual(await human.get_history(), [])   # media moved between modules; nothing was stored

    async def test_gate_passes_only_matched_chunks(self):
        gate = SpeakerGate(reference='voice', emits='voice', segment_ms=1000)
        out = []
        for i in range(30):
            end = 100 * (i + 1)
            out += gate.process(InputFrame(np.full(1600, .1, np.float32), 'audio', end, sample_rate=16000))
            if end >= 1000 and end % 500 == 0:
                gate.process(InputFrame(dict(kind='segment', reference='voice', start_ms=end - 1000, end_ms=end,
                                             match=end <= 2000), 'audio', end))
        out += gate.finish()
        self.assertEqual([item.timestamp_ms for item in out], list(range(100, 3100, 100)))
        heard = [bool(item.data.any()) for item in out]
        self.assertEqual(heard, [True] * heard.index(False) + [False] * (30 - heard.index(False)))
        self.assertAlmostEqual(100 * heard.index(False), 1750, delta=250)

    async def test_spans_drive_the_sam_prompt_and_silence_needs_no_model(self):
        class Backend:
            calls = []

            def separate(self, window, anchors):
                self.calls.append(anchors)
                return window

            def close(self):
                pass

        module = SAMSpanSpeech(Backend(), emits='voice', window_ms=1000)
        await module.process(InputFrame(dict(kind='segment', reference='voice', start_ms=200, end_ms=800, match=True,
                                             state='match'), 'audio', 800))
        out = []
        for i in range(20):
            out += await module.process(InputFrame(np.full(1600, .1, np.float32), 'audio', 100 * (i + 1),
                                                   sample_rate=16000))
        self.assertEqual(Backend.calls, [[('+', .2, .8)]])
        self.assertTrue(out[0].data.any())
        self.assertFalse(out[-1].data.any())


if __name__ == '__main__':
    unittest.main()
