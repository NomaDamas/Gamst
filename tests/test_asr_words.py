import unittest

import numpy as np

from gamst.plugins import InputFrame
from gamst.providers.asr_words import AlignedASR, QwenForcedAligner, WhisperWords
from tests.test_speaker_verify import tone


class Engine:
    """Test double with QwenStreamingASR's contract: strict 20 ms hops, one final every second."""
    hop_ms = 20

    def __init__(self):
        self.reset()

    def reset(self):
        self.last, self.count, self.start = None, 0, None

    def process(self, frame):
        assert frame.data.shape == (320,) and len(frame.sensor.channels) == 1
        assert self.last is None or (frame.timestamp_ms == self.last[0] + 20 and
                                     frame.sensor.sequence == self.last[1] + 1)
        self.last = (frame.timestamp_ms, frame.sensor.sequence)
        self.start = frame.timestamp_ms - 20 if self.start is None else self.start
        self.count += 1
        return self._final() if self.count == 50 else None

    def _final(self):
        value = dict(utterance_id=f'u{self.start}', revision=0, text='대박 진짜', status='final', start_ms=self.start,
                     end_ms=self.last[0], language='ko', finalization_reason='vad_endpoint')
        self.count, self.start = 0, None
        return value

    def finish(self):
        return self._final() if self.count else None


class Aligner:
    model_id = 'test-aligner'

    def __init__(self):
        self.calls = []

    def align(self, samples, sample_rate, text, language):
        self.calls.append((len(samples), language))
        return [dict(text='대박', start_s=.1, end_s=.4), dict(text='진짜', start_s=.5, end_s=.9)]


class PartialEngine(Engine):
    def process(self, frame):
        final = super().process(frame)
        return final if final or self.count < 10 else dict(utterance_id='u', status='partial', text='대', start_ms=self.start)


class AlignedTest(unittest.IsolatedAsyncioTestCase):
    async def test_an_open_utterance_is_declared_held_until_it_is_final(self):
        module, held = AlignedASR(PartialEngine()), []
        audio = tone(200, 1.2)
        for end in range(100, 1201, 100):
            at = (end - 100) * 16
            values = await module.process(InputFrame(audio[at:at + 1600], 'audio', 5000 + end, sample_rate=16000))
            held.append((module.pending_since_ms, bool(values)))      # partial text itself is not emitted
        self.assertEqual(held[0], (None, False))
        self.assertEqual(held[2:9], [(5000, False)] * 7)
        self.assertEqual(held[9], (None, True))
        await module.finish()
        self.assertIsNone(module.pending_since_ms)

    async def test_jittery_chunks_become_exact_hops_and_words_get_capture_times(self):
        aligner = Aligner()
        module = AlignedASR(Engine(), aligner)
        audio, values = tone(200, 2.5), []
        for i, at in enumerate(range(0, len(audio), 1700)):
            end = 5000 + round(min(at + 1700, len(audio)) / 16) + (i % 3) * 8   # callback jitter
            values += await module.process(InputFrame(audio[at:at + 1700], 'audio', end, sample_rate=16000)) or []
        values += await module.finish() or []
        self.assertEqual([value['status'] for value in values], ['final'] * 3)
        self.assertEqual(aligner.calls[0], (16000, 'ko'))
        first = values[0]
        self.assertEqual(first['words'][0], dict(text='대박', start_ms=first['start_ms'] + 100,
                                                 end_ms=first['start_ms'] + 400))
        self.assertAlmostEqual(first['start_ms'], 5000, delta=10)

    async def test_a_capture_gap_finishes_the_utterance_and_restarts_the_engine(self):
        module = AlignedASR(Engine(), None)
        first = await module.process(InputFrame(tone(200, .5), 'audio', 1500, sample_rate=16000))
        second = await module.process(InputFrame(tone(200, .5), 'audio', 9500, sample_rate=16000))
        self.assertIsNone(first)
        self.assertEqual(second[0]['finalization_reason'], 'vad_endpoint')
        self.assertFalse(second[0]['aligned'])

    async def test_whisper_tail_is_padded_once(self):
        class Plugin:
            name, version, window_seconds = 'whisper', '1', 1.
            buffers, seen = {}, []

            async def process(self, frame):
                self.seen.append((len(frame.data), frame.timestamp_ms))
                self.buffers.setdefault(frame.subject_id, []).extend(frame.data)
                return []

        plugin = Plugin()
        module = WhisperWords(plugin)
        await module.process(InputFrame(np.zeros(4000, np.float32), 'audio', 1000, sample_rate=16000))
        await module.finish()
        self.assertEqual(plugin.seen[-1], (12000, 1750))

    def test_official_result_shape_is_read(self):
        class Item:
            def __init__(self, text, start, end):
                self.text, self.start_time, self.end_time = text, start, end

        class Result:
            items = [Item('대박', .08, .4)]

        class Model:
            def align(self, audio, text, language):
                assert audio[1] == 16000 and language == 'Korean'
                return [Result()]

        self.assertEqual(QwenForcedAligner(Model(), 'x').align(np.zeros(10), 16000, '대박', 'ko'),
                         [dict(text='대박', start_s=.08, end_s=.4)])


if __name__ == '__main__':
    unittest.main()
