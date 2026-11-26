"""What the two test-bed clips needed: rates in rules, sound onsets, coincidence, phrases, each word's own voice."""
import asyncio
import types
import unittest

import importlib.util

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.coincidence import Coincidence
from gamst.providers.movement_rule import RuleScorer, validate_rule
from gamst.providers.term_events import TermEvents

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HEIGHT = {'height': ['right_wrist', 'right_shoulder']}


class RateTest(unittest.TestCase):
    def rule(self):
        return RuleScorer({'steps': [{'when': {'measure': {'rate': HEIGHT}, 'op': 'lt', 'value': -5}}]})

    def points(self, height):
        return {'left_shoulder': (0., 0.), 'right_shoulder': (1., 0.), 'right_wrist': (1., -height)}

    def violation(self, rule, ms, height):
        size = len(rule._index)
        points = self.points(height) if height is not None else {}
        rule.observe(ms, [0.] * size, [False] * size, points)
        return rule.violation(0, [0.] * size, [False] * size, points)

    def test_fast_fall_satisfies_and_slow_does_not(self):
        rule = self.rule()
        for index in range(4):
            fast = self.violation(rule, 1000 + 25 * index, .5 - .4 * index)
        self.assertEqual(fast, 0)
        rule.reset()
        for index in range(4):
            slow = self.violation(rule, 1000 + 25 * index, .5 - .05 * index)
        self.assertGreater(slow, 0)

    def test_unseen_now_is_unknown_and_a_long_hole_is_no_speed(self):
        rule = self.rule()
        self.violation(rule, 1000, .5)
        self.assertIsNone(self.violation(rule, 1050, None))
        self.assertIsNone(self.violation(rule, 2000, -2.))

    def test_rate_of_rate_is_refused(self):
        with self.assertRaises(ValueError):
            validate_rule({'steps': [{'when': {'measure': {'rate': {'rate': HEIGHT}}, 'op': 'lt', 'value': 0}}]})


class CoincidenceTest(unittest.TestCase):
    def frame(self, name, ms, events):
        return InputFrame(dict(events=events), 'record', ms, sensor=SensorMetadata(source_id='t', sensor_id=name, sequence=0))

    def setUp(self):
        self.module = Coincidence('move', 'sound', of_event='m', by_event='s')

    def move(self, ms):
        return self.frame('move', ms, [dict(event='m', end_ms=ms)])

    def sound(self, ms, *times):
        return self.frame('sound', ms, [dict(event='s', timestamp_ms=at) for at in times])

    def test_either_may_arrive_first(self):
        self.assertIsNone(self.module.process(self.move(1000)))
        value = self.module.process(self.sound(1300, 1050))
        self.assertEqual((value['count'], value['events'][0]['delay_ms']), (1, 50))
        self.module.process(self.sound(2300, 2050))
        self.assertEqual(self.module.process(self.move(2000))['count'], 2)

    def test_waits_for_the_answer_and_one_sound_confirms_one_movement(self):
        self.module.process(self.move(1000))
        self.module.process(self.move(1060))
        self.assertIsNone(self.module.process(self.sound(1050)))
        value = self.module.process(self.sound(1400, 1070))
        self.assertEqual((value['count'], value['unconfirmed']), (1, 1))

    def test_end_of_input_decides_what_is_left(self):
        self.module.process(self.move(1000))
        self.assertEqual(self.module.finish()['unconfirmed'], 1)


@unittest.skipUnless(HAS_NUMPY, 'needs numpy')
class OnsetTest(unittest.TestCase):
    def test_knocks_are_found_once_each_at_their_time(self):
        import numpy as np
        from gamst.providers.audio_onsets import AudioOnsets
        rng = np.random.default_rng(1)
        x = rng.normal(0, .002, 4 * 16000)
        for at in (1., 1.6, 3.):
            n = np.arange(1600)
            x[int(at * 16000):int(at * 16000) + 1600] += .5 * np.sin(2 * np.pi * 120 * n / 16000) * np.exp(-n / 400)
        module, found = AudioOnsets(), []
        answers = 0
        for i in range(0, len(x), 1600):
            value = module.process(InputFrame(x[i:i + 1600].astype(np.float32), 'audio', 100_000 + (i + 1600) // 16, sample_rate=16000))
            answers += value is not None
            found += [event['timestamp_ms'] - 100_000 for event in (value or {}).get('events', ())]
        self.assertEqual(len(found), 3, found)
        for seen, true in zip(found, (1000, 1600, 3000)):
            self.assertLess(abs(seen - true), 40)
        self.assertGreater(answers, 3)      # silence is answered too


class Plugin:
    name, version, language = 'stand-in', '1', 'ko'

    def __init__(self, phrases):
        self.phrases, self.model = phrases, self

    def transcribe(self, samples, **options):
        zero, length, out = self.zero(), len(samples) / 16000, []
        for start, end, text in self.phrases:
            if start >= zero - .01 and end <= zero + length:
                word = types.SimpleNamespace(word=text, start=start - zero, end=end - zero)
                out.append(types.SimpleNamespace(text=text, words=[word], end=end - zero, avg_logprob=0., no_speech_prob=0.,
                                                 compression_ratio=1.))
        return out, None


@unittest.skipUnless(HAS_NUMPY, 'needs numpy')
class PhraseTest(unittest.TestCase):
    def test_every_phrase_is_final_once_with_capture_times(self):
        import numpy as np
        from gamst.providers.whisper_phrases import WhisperPhrases
        plugin = Plugin([(.5, 1.2, 'a'), (2.9, 3.4, 'b'), (7.6, 8.4, 'c'), (9.5, 9.9, 'd')])
        module = WhisperPhrases(plugin, hop_seconds=2., margin_seconds=1., max_seconds=8.)
        plugin.zero = lambda: (module._end_ms - 50_000) / 1000 - len(module._buffer) / 16000
        out = []
        for index in range(100):
            out += module.process(InputFrame(np.full(1600, .1, np.float32), 'audio', 50_000 + 100 * (index + 1), sample_rate=16000))
        out += module.finish()
        values = [item['value'] for item in out]
        self.assertEqual([value['text'] for value in values], ['a', 'b', 'c', 'd'])
        self.assertEqual([value['start_ms'] - 50_000 for value in values], [500, 2900, 7600, 9500])


class WordVoiceTest(unittest.TestCase):
    def value(self):
        return dict(utterance_id='u', status='final', text='뽀또 뽀또', start_ms=0, end_ms=900, speaker=dict(match=False, score=.2),
                    words=[dict(text='뽀또', start_ms=0, end_ms=400, match=True, speaker_score=.5),
                           dict(text='뽀또', start_ms=500, end_ms=900, match=False, speaker_score=.1)])

    def test_a_shared_phrase_counts_only_the_enrolled_persons_word(self):
        for report, names in ((True, ['term.spoken', 'term.other_voice']), (False, ['term.spoken'])):
            module = TermEvents(terms=['뽀또'], report_others=report)
            value = module.process(InputFrame(self.value(), 'record', 1000))
            self.assertEqual((value['count'], [event['event'] for event in value['events']]), (1, names))
            self.assertEqual(value['events'][0]['speaker_score'], .5)


if __name__ == '__main__':
    unittest.main()


class ScoreAtEndTest(unittest.TestCase):
    def test_truth_may_be_given_by_when_an_event_ended(self):
        from gamst.scoring import score
        events = [dict(event='e', start_ms=1000, end_ms=1500)]
        truth = dict(event='e', count=1, tolerance_s=.1, events=[dict(t_s=0.)])
        self.assertEqual(score(truth, events, start_ms=1000)['matched'], 1)                  # by start
        self.assertEqual(score(dict(truth, at='end'), events, start_ms=1000)['matched'], 0)
        self.assertEqual(score(dict(truth, at='end', events=[dict(t_s=.5)]), events, start_ms=1000)['matched'], 1)
