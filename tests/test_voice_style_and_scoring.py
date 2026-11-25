import importlib.util
import math
import time
import unittest

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.providers.pose_exemplar import ExemplarEvents
from gamst.providers.term_events import TermEvents
from gamst.providers.voice_style import VoiceChannels
from gamst.scoring import score

HAS_NUMPY = importlib.util.find_spec('numpy') is not None


def now():
    return time.time_ns() // 1_000_000


def tone(frequency, seconds, level, rate=16000):
    import numpy as np
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * math.pi * frequency * t)).astype(np.float32) if frequency else \
        np.zeros(int(seconds * rate), np.float32)


@unittest.skipUnless(HAS_NUMPY, 'needs numpy')
class VoiceStyleTest(unittest.IsolatedAsyncioTestCase):
    async def test_counts_a_demonstrated_rising_shout_through_prosody(self):
        import numpy as np
        from gamst.providers.prosody import AudioProsody
        rate = 16000
        quiet = lambda s: tone(150, s, .05)

        def shout():  # a loud rising call: 150 -> 300 Hz in 0.6 s
            t = np.arange(int(.6 * rate)) / rate
            phase = 2 * math.pi * (150 * t + 125 * t * t / .6)
            return (.6 * np.sin(phase)).astype(np.float32)

        async def channels_of(signal):
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'enroll')
                pipeline.add_input('mic', modality='audio')
                prosody = AudioProsody()
                pipeline.add_module('prosody', prosody, consumes='mic', spec=AudioProsody.capability())
                voice = VoiceChannels()
                pipeline.add_module('voice', voice, consumes='prosody', spec=voice.spec)
                start = now() - 60000
                for index in range(0, len(signal), 320):
                    records = await pipeline.push('mic', signal[index:index + 320], sample_rate=rate,
                                                  timestamp_ms=start + (index + 320) * 1000 // rate)
                rows = await human.get_observations(capability_id='audio.voice_channels', window='2h', limit=100)
                return [dict(channels=row['observation']['value']['channels'], mask=row['observation']['value']['mask'])
                        for row in reversed(rows['items'])]

        enrolled = await channels_of(np.concatenate([quiet(2), shout()]))
        exemplar = enrolled[-5:]  # the shout part
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'style')
            pipeline.add_reference('call', 'exemplar', exemplar)
            pipeline.add_input('mic', modality='audio')
            pipeline.add_module('prosody', AudioProsody(), consumes='mic', spec=AudioProsody.capability())
            voice = VoiceChannels()
            pipeline.add_module('voice', voice, consumes='prosody', spec=voice.spec)
            matcher = ExemplarEvents(reference='call', mode='sequence', enter=.15, exit=.3, modality='audio',
                                     capability_id='audio.exemplar_events')
            pipeline.add_module('call', matcher, consumes='voice', spec=matcher.spec)
            signal = np.concatenate([quiet(2), shout(), quiet(1.5), shout(), quiet(1.5), tone(150, .6, .05), quiet(1)])
            start = now() - 60000
            for index in range(0, len(signal), 320):
                await pipeline.push('mic', signal[index:index + 320], sample_rate=rate,
                                    timestamp_ms=start + (index + 320) * 1000 // rate)
            counted = await human.count_events('audio.exemplar_events', event='exemplar.matched', window='2h')
            self.assertEqual(counted['count'], 2)


class TranscriptWindowTest(unittest.IsolatedAsyncioTestCase):
    async def test_plain_text_windows_count_once_each(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'asr')
            pipeline.add_reference('words', 'terms', ['대박'])
            pipeline.add_input('asr', modality='audio')
            terms = TermEvents(reference='words')
            pipeline.add_module('terms', terms, consumes='asr', spec=terms.spec)
            start = now() - 9000
            for index, text in enumerate(['와 대박', '그냥 말', '대박 대박']):
                await pipeline.push('asr', text, timestamp_ms=start + 3000 * index)
            self.assertEqual((await human.count_events('speech.term_events', event='term.spoken'))['count'], 3)


class ScoringTest(unittest.TestCase):
    def test_count_times_and_values_against_truth(self):
        start = 1_000_000
        events = [dict(event='throw', timestamp_ms=start + 1000, flips=5.0),
                  dict(event='throw', timestamp_ms=start + 4100, flips=7.5),
                  dict(event='throw', timestamp_ms=start + 9000, flips=2.0),
                  dict(event='other', timestamp_ms=start + 9000)]
        truth = dict(capability='x', event='throw', count=3, value_key='flips', tolerance_s=.5,
                     events=[dict(t_s=1.1, value=5), dict(t_s=4.0, value=9), dict(t_s=6.0, value=3)])
        result = score(truth, events, start_ms=start)
        self.assertEqual((result['counted'], result['count_error']), (3, 0))
        self.assertEqual(result['matched'], 2)
        self.assertEqual(result['recall'], round(2 / 3, 4))
        self.assertEqual([pair.get('value_error') for pair in result['events']], [0.0, -1.5, None])
        self.assertEqual(result['values_within_one'], 1)


if __name__ == '__main__':
    unittest.main()
