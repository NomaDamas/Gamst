"""Whole-word terms (``word:18``, ``word:달*``): only whole recogniser words, each repetition counted, numerals flagged."""
import asyncio
import time
import unittest

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.term_events import TermEvents, matches, parse_terms, spelling_parts, term_occurrences
from gamst.recipes.perception_parts import bias_terms

try:
    import numpy as np
except ImportError:
    np = None


def utterance(text, words=None, key='u'):
    words = words or [(item, 1000 * index, 1000 * index + 900) for index, item in enumerate(text.split())]
    return dict(utterance_id=key, status='final', text=text, start_ms=words[0][1], end_ms=words[-1][2],
                words=[dict(text=item, start_ms=start, end_ms=end) for item, start, end in words])


def counted(terms, value):
    found = TermEvents(terms=terms).process(InputFrame(value, 'audio', value['end_ms']))
    return found['count'] if found else 0


class WholeWordTest(unittest.TestCase):
    def test_a_repeated_sound_word_counts_each_repetition_and_never_inside_another_word(self):
        for text, count in (('달', 1), ('달달', 2), ('달달달', 3), ('달달달!', 3), ('달라', 0), ('달걀', 0), ('달려', 0),
                            ('달이', 0), ('아 달 달달 그래', 3), ('달다', 0)):
            self.assertEqual(counted(['word:달*'], utterance(text)), count, text)
        self.assertEqual(counted(['달'], utterance('달라 달걀')), 2)          # plain text: found inside words too

    def test_a_whole_word_without_star_is_exactly_that_word(self):
        for text, count in (('18', 1), ('18!', 1), ('180', 0), ('2018', 0), ('18살', 0), ('1818', 0), ('아 18 진짜', 1)):
            self.assertEqual(counted(['word:18'], utterance(text)), count, text)
        self.assertEqual(counted(['word:달'], utterance('달달')), 0)

    def test_words_are_the_recognisers_own_not_the_aligners_pieces(self):
        # An aligner may cut one written word in two ("달달" + "달"): the written word decides, the pieces give times.
        value = utterance('달달달 좋아', [('달달', 0, 400), ('달', 400, 600), ('좋아', 700, 900)])
        found = term_occurrences(value, parse_terms(['word:달*'])[0])
        self.assertEqual([(item['start_ms'], item['end_ms'], item['part']) for item in found],
                         [(0, 200, [0, 3]), (200, 400, [1, 3]), (400, 600, [2, 3])])
        self.assertEqual(counted(['word:달*'], utterance('달라', [('달', 0, 100), ('라', 100, 200)])), 0)

    def test_numerals_are_flagged_and_a_flag_can_be_written(self):
        self.assertEqual(spelling_parts('word:18')['flags'], ('may_be_number',))
        self.assertEqual(spelling_parts('십팔#may_be_number'), dict(kind='text', body='십팔', repeat=False,
                                                                   flags=('may_be_number',)))
        self.assertEqual(spelling_parts('C#')['flags'], ())
        terms = {'시발': ['시발', '씨발', '씨팔', '십팔#may_be_number', 'word:18']}
        found = TermEvents(terms=terms).process(InputFrame(utterance('아 씨발 18 180 십팔'), 'audio', 1))
        self.assertEqual([(event['spoken'], event.get('flags')) for event in sorted(found['events'],
                                                                                   key=lambda item: item['start_ms'])],
                         [('씨발', None), ('18', ['may_be_number']), ('십팔', ['may_be_number'])])
        self.assertEqual(found['counts'], {'시발': 3})

    def test_a_term_given_as_a_list_is_named_without_the_syntax_and_recognisers_get_plain_words(self):
        self.assertEqual(parse_terms(['word:달*', 'word:18', '뽀또'])[0],
                         {'달': ['word:달*'], '18': ['word:18'], '뽀또': ['뽀또']})
        context = dict(references=dict(terms=dict(시발=['시발', '십팔#may_be_number', 'word:18'], 달=['word:달*'],
                                                    끝=['re:(다|요)$'])))
        self.assertEqual(bias_terms(dict(bias='terms'), context), ['시발', '십팔', '18', '달', '끝'])   # no pattern
        self.assertEqual(matches('달 달달', 'word:달*'), [(0, 1), (1, 1), (2, 1)])
        with self.assertRaises(ValueError):
            parse_terms(['word:'])


class FakeEmbedder:
    model_id = 'fake'

    def embed(self, samples, sample_rate=16000):
        return [1., 0.] if float(np.median(samples)) > .15 else [0., 1.]


class FakeDiarizer:
    name = 'fake'

    def diarize(self, samples):
        step = 1600
        return [(first / 16000, (first + step) / 16000, 'P') for first in range(0, len(samples), step)
                if float(np.median(samples[first:first + step])) > 0]


class TwoOfThree:
    """A second hearing that writes the sound word twice where the first wrote it three times in one word."""
    name = 'second'
    pending_since_ms = None

    def __init__(self):
        self.base = None

    async def process(self, frame):
        index = int(round(float(frame.data[0]) * 10000))
        if index != 8:
            return []
        base = frame.timestamp_ms - 900
        return [dict(utterance_id='s', status='final', text='bo bo', start_ms=base + 300, end_ms=base + 500,
                     words=[dict(text='bo', start_ms=base + 300, end_ms=base + 400),
                            dict(text='bo', start_ms=base + 400, end_ms=base + 500)])]

    async def finish(self):
        return None


@unittest.skipIf(np is None, 'numpy is needed')
class RepetitionVerificationTest(unittest.TestCase):
    def test_each_repetition_inside_one_word_is_decided_on_its_own(self):
        from gamst.counting import count_events
        from gamst.providers.word_verification import WordVerification

        async def main():
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'verify')
                pipeline.add_input('microphone', None, modality='audio')
                pipeline.add_input('first', None, modality='text')
                pipeline.add_reference('voice', 'voice', dict(model='fake', embeddings=[[1., 0.]]), consent='test')
                pipeline.add_reference('terms', 'terms', ['word:bo*'])
                verify = WordVerification(TwoOfThree(), FakeDiarizer(), FakeEmbedder(), audio='microphone',
                                          rule=dict(keep_clear_voice=False, near_ms=60),
                                          windows=dict(step_ms=1000, before_ms=500, after_ms=200))
                pipeline.add_module('verification', verify, consumes=['microphone', 'first'], spec=verify.spec, queue=500)
                terms = TermEvents(report_others=True)
                pipeline.add_module('terms', terms, consumes=['first', 'verification'], spec=terms.spec)
                base = time.time_ns() // 1_000_000 - 60000
                for index in range(30):
                    samples = np.full(1600, .2 if index < 10 else 0., np.float32)
                    samples[0] = index / 10000
                    await pipeline.push('microphone', samples, timestamp_ms=base + (index + 1) * 100, sample_rate=16000)
                word = dict(text='bobobo', start_ms=base + 300, end_ms=base + 600, word_id='u#0', state='provisional',
                            match=True, speaker_score=.9)
                await pipeline.push('first', dict(kind='utterance', reference='voice', utterance_id='u', status='final',
                                                  text='bobobo', start_ms=word['start_ms'], end_ms=word['end_ms'],
                                                  words=[word], speaker=dict(match=True)), timestamp_ms=base + 600)
                await pipeline.drain()
                await pipeline.finish()
                rows = list(human.store.query('local-user', kind='observation.signal', limit=500, registered_only=True,
                                              capability_id='speech.term_events'))
                verdicts = {item['word_id']: item for record in human.store.query(
                    'local-user', kind='observation.signal', limit=500, registered_only=True,
                    capability_id='speech.word_verdicts') for item in record['observation']['value'].get('verdicts') or ()}
                await pipeline.close()
                return count_events(rows, event='term.spoken'), verdicts

        result, verdicts = asyncio.run(main())
        self.assertEqual(sorted(verdicts), ['u#0~0', 'u#0~1', 'u#0~2'])
        self.assertEqual([verdicts[f'u#0~{index}']['match'] for index in range(3)], [True, True, False])
        self.assertEqual(verdicts['u#0~2']['reason'], 'not_heard')
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['states'], dict(provisional=0, final=2, revoked=1))


if __name__ == '__main__':
    unittest.main()
