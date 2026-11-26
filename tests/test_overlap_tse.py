"""The TSE lane (overlapped speech, live): where voices meet, the enrolled voice is extracted, heard and scored, and the
counted words there are decided from it; it catches up with forks when it falls behind. Stand-ins only (no model)."""
import asyncio
import time
import unittest

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.overlap_tse import MultiVoice, RULE, TSEDecision, TargetSpeechWords, decide

try:
    import numpy as np
except ImportError:
    np = None

RATE = 16000
FRAME_MS = 100


def activity(session, start, end, slots):
    """One diarization chunk: ``slots`` = {slot: [[start, end], ...]}."""
    return {'diarization': dict(session_id=session, chunks=[dict(start_ms=start, end_ms=end, activity=[
        dict(slot=slot, intervals_ms=intervals) for slot, intervals in slots.items()])])}


class MultiVoiceTest(unittest.TestCase):
    def test_overlap_rule_finds_two_slots_at_once_and_waits_until_a_stretch_is_closed(self):
        detector = MultiVoice(rule='overlap', min_slot_ms=100)
        detector.add(activity('s', 0, 3000, {0: [[0, 2000]], 1: [[1500, 2600]], 2: [[2900, 2950]]}))
        self.assertEqual(detector.stretches(), [(1500, 2000)])
        detector = MultiVoice(rule='overlap', min_slot_ms=100)
        detector.add(activity('s', 0, 2050, {0: [[0, 2050]], 1: [[1500, 2050]]}))
        closed, opened, clean = detector.scan()
        self.assertEqual((closed, opened, clean), ([], 1500, None))          # still open at the end of what is known

    def test_near_rule_flags_voices_within_the_context_and_says_where_nothing_more_can_be_found(self):
        detector = MultiVoice(rule='near', context_ms=1000, min_slot_ms=100)
        detector.add(activity('s', 0, 6000, {0: [[1000, 2000]], 1: [[2300, 2800]]}))   # 300 ms apart: they meet
        (low, high), = detector.stretches()
        self.assertTrue(1700 <= low <= 2000 and 2300 <= high <= 2600, (low, high))
        detector = MultiVoice(rule='near', context_ms=1000, min_slot_ms=100)
        detector.add(activity('s', 0, 6000, {0: [[1000, 2000]], 1: [[4000, 4500]]}))   # 2 s apart: they do not
        self.assertEqual(detector.scan(), ([], None, 5000))

    def test_a_new_session_does_not_join_the_old_one(self):
        detector = MultiVoice(rule='overlap')
        detector.add(activity('a', 0, 1000, {0: [[0, 1000]]}))
        detector.add(activity('b', 1000, 3000, {0: [[1000, 3000]]}))
        self.assertEqual(detector.stretches(), [])


class DecisionRuleTest(unittest.TestCase):
    def test_person_not_the_person_and_uncertain(self):
        record = dict(score=.8, terms=[dict(term='bbo', end_ms=1000, score=.7)])
        self.assertEqual(decide(1200, 'bbo', record, RULE)[:3], ('person', True, None))
        self.assertEqual(decide(1200, 'bbo', dict(record, terms=[dict(term='bbo', end_ms=1000, score=.2)]), RULE)[:3],
                         ('uncertain', False, 'uncertain_overlap'))
        self.assertEqual(decide(5000, 'bbo', record, RULE)[:3], ('other', False, 'tse_not_heard'))
        self.assertEqual(decide(5000, 'bbo', dict(record, score=.1), RULE)[:3], ('uncertain', False, 'uncertain_overlap'))
        self.assertEqual(decide(5000, 'bbo', dict(record, score=None), RULE)[:3], ('uncertain', False, 'uncertain_overlap'))


class FakeEmbedder:
    """The person's sound is .2, another voice .1: the 'embedding' says which."""
    model_id = 'fake'

    def embed(self, samples, sample_rate=RATE):
        level = float(np.median(np.abs(samples)))
        return [1., 0.] if level > .15 else [0., 1.]


class FakeExtractor:
    """The person is .2 of the level: .3 (both) and .2 give .2, .1 (the other voice alone) gives silence."""
    model_id = 'fake-tse'

    def __init__(self, delay_s=0.):
        self.delay_s, self.calls = delay_s, 0

    async def extract(self, mixture, enrollment):
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        out = np.where(mixture > .15, .2, 0.).astype(np.float32)
        out[np.abs(mixture) < .05] = mixture[np.abs(mixture) < .05]       # the frame marks pass as they are
        return out


class FakeHearing:
    """A recogniser session: writes 'bbo' for 100 ms of sound with the person's voice whose first sample marks a word."""
    name = 'fake-asr'

    def __init__(self):
        self.out = []

    async def process(self, frame):
        index = int(round(float(frame.data[0]) * 10000))
        if float(np.median(frame.data)) > .15 and index % 80 in HEARD:
            self.out.append(dict(utterance_id=f'h{index}', status='final', text='bbo', start_ms=frame.timestamp_ms - 100,
                                 end_ms=frame.timestamp_ms, words=[dict(text='bbo', start_ms=frame.timestamp_ms - 100,
                                                                        end_ms=frame.timestamp_ms)]))
        return None

    async def _commit(self):
        pass

    pending_since_ms = None

    def _take(self):
        out, self.out = self.out, []
        return out

    async def close(self):
        pass


class FakeLive:
    """The live recogniser module: only its fork() is used here."""
    def __init__(self):
        self.forks = 0

    def process(self, frame):
        return None

    def fork(self):
        self.forks += 1
        return FakeHearing()


HEARD = {15, 19}        # frames where the person says the word (16 and 18: the live recogniser's own words)


def level(index):
    """100 ms frames of an 8 s round: the person speaks at 0-2 s and 4-6 s (.2), another voice at 1.5-3 s (.1)."""
    index %= 80
    return .2 * (index < 20 or 40 <= index < 60) + .1 * (15 <= index < 30)


def frame_samples(index):
    samples = np.full(RATE * FRAME_MS // 1000, level(index), np.float32)
    samples[0] = (index % 80) / 10000
    return samples


def slots(base, index):
    """Slot activity of frame ``index``: slot 0 the person, slot 1 the other voice (1.5-3 s)."""
    start, end = base + index * FRAME_MS, base + (index + 1) * FRAME_MS
    index %= 80
    active = {0: [[start, end]] if index < 20 or 40 <= index < 60 else [], 1: [[start, end]] if 15 <= index < 30 else []}
    return activity('s', start, end, active)


def utterance(word_id, end, match, score):
    word = dict(text='bbo', start_ms=end - 100, end_ms=end, word_id=word_id, state='provisional', match=match,
                speaker_score=score)
    return dict(kind='utterance', reference='voice', start_ms=end - 100, end_ms=end, match=match,
                state='match' if match else 'other', utterance_id=word_id[:-2], status='final', text='bbo',
                words=[word], speaker=dict(reference='voice', match=match), model=dict(id='fake'))


@unittest.skipIf(np is None, 'numpy is needed')
class LaneTest(unittest.TestCase):
    def run_lane(self):
        async def main():
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'tse')
                pipeline.add_input('microphone', None, modality='audio')
                pipeline.add_input('diarization', None, modality='text')
                pipeline.add_input('first', None, modality='text')
                pipeline.add_reference('voice', 'voice', dict(model='fake', embeddings=[[1., 0.]],
                                                              samples=[np.full(RATE, .2, np.float32)]), consent='test')
                pipeline.add_reference('terms', 'terms', ['bbo'])
                pipeline.add_module('asr', FakeLive(), consumes='microphone')
                lane = TargetSpeechWords(FakeExtractor(), FakeEmbedder(), detect=dict(rule='overlap', min_slot_ms=100),
                                         window=dict(margin_ms=500, word_context_ms=400))
                pipeline.add_module('tse', lane, consumes=['microphone', 'diarization'], spec=lane.spec, queue=500,
                                    catch_up=False)     # pushed frames are a minute old: no catch-up here
                decision = TSEDecision(tse='tse', rule=dict(near_ms=150))
                pipeline.add_module('decision', decision, consumes=['microphone', 'first', 'tse'], spec=decision.spec)
                base = time.time_ns() // 1_000_000 - 60000
                # The live recogniser's words: 1.6 s (the person, over the other voice: heard in the extracted sound),
                # 1.8 s (said by the other voice over the person: not in the extracted sound), 4.6 s (no overlap).
                first = [('u16#0', base + 1600, False, .3), ('u18#0', base + 1800, True, .6), ('u46#0', base + 4600, True, .9)]
                for index in range(80):
                    end = base + (index + 1) * FRAME_MS
                    await pipeline.push('microphone', frame_samples(index), timestamp_ms=end, sample_rate=RATE)
                    await pipeline.push('diarization', slots(base, index), timestamp_ms=end)
                    for word_id, at, match, score in first:
                        if at == end - 300:
                            await pipeline.push('first', utterance(word_id, at, match, score), timestamp_ms=end)
                await pipeline.drain()
                await pipeline.finish()
                rows = human.store.query('local-user', kind='observation.signal', limit=500, registered_only=True,
                                         capability_id=decision.spec.id)
                words = {word.get('word_id', word['text'] + str(word['end_ms'] - base)): word
                         for row in rows for word in row['observation']['value'].get('words') or ()}
                windows = [row['observation']['value'] for row in human.store.query(
                    'local-user', kind='observation.signal', limit=500, registered_only=True, capability_id=lane.spec.id)]
                await pipeline.close()
                return words, windows, decision.stats, base
        return asyncio.run(main())

    def test_overlapped_words_are_decided_from_the_extracted_voice_and_the_others_pass_unchanged(self):
        words, windows, stats, base = self.run_lane()
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0]['start_ms'] - base, windows[0]['end_ms'] - base), (1500, 2000))
        self.assertEqual(words['u16#0']['tse']['decision'], 'person')     # heard in the extracted sound, the person's
        self.assertTrue(words['u16#0']['match'])
        self.assertEqual(words['u16#0']['mixture_score'], .3)
        self.assertEqual(words['u18#0']['reason'], 'tse_not_heard')       # not in the extracted (the person's) sound
        self.assertFalse(words['u18#0']['match'])
        self.assertNotIn('tse', words['u46#0'])                           # no overlap there: as it was
        self.assertTrue(words['u46#0']['match'])
        heard_only = [word for key, word in words.items() if word.get('basis') == 'tse_heard_only']
        self.assertEqual([word['end_ms'] - base for word in heard_only], [2000])    # only the extracted sound has it
        self.assertEqual(heard_only[0]['state'], 'provisional')
        self.assertEqual(stats['decided'], 2)


class SlowLane(TargetSpeechWords):
    pass


class PacedMicrophone:
    modality, live, missed = 'audio', True, 0

    def __init__(self, frames):
        self.count = frames

    async def frames(self):
        base = time.time_ns() // 1_000_000
        for index in range(self.count):
            due = base + (index + 1) * FRAME_MS
            wait = due / 1000 - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            yield InputFrame(frame_samples(index), 'audio', due, sample_rate=RATE)


class Diarizer:
    """Slot activity of each frame, at once (a stand-in for the live diarization in its own lane)."""
    def process(self, frame):
        index = int(round((frame.timestamp_ms - self.base) / FRAME_MS)) - 1 if hasattr(self, 'base') else 0
        if not hasattr(self, 'base'):
            self.base = frame.timestamp_ms - FRAME_MS
            index = 0
        return slots(self.base, index)


@unittest.skipIf(np is None, 'numpy is needed')
class CatchUpTest(unittest.TestCase):
    def test_a_slow_lane_skips_to_now_and_forks_extract_what_it_skipped_once(self):
        async def main(delay):
            async with Gamst(consent=True) as human:
                pipeline = Pipeline(human, 'tse')
                pipeline.add_input('microphone', PacedMicrophone(160), modality='audio')
                pipeline.add_reference('voice', 'voice', dict(model='fake', embeddings=[[1., 0.]],
                                                              samples=[np.full(RATE, .2, np.float32)]), consent='test')
                pipeline.add_reference('terms', 'terms', ['bbo'])
                pipeline.add_module('asr', FakeLive(), consumes='microphone')
                from gamst import CapabilitySpec, OutputField
                spec = CapabilitySpec('audio.speaker_activity', 'audio.speaker_activity', 'audio', 'test', '1',
                                      (OutputField('diarization', 'object'),), 'native', stateful=True)
                pipeline.add_module('diarization', Diarizer(), consumes='microphone', spec=spec)
                extractors = []

                async def open_extractor():
                    extractors.append(FakeExtractor(delay))
                    return extractors[-1]

                lane = TargetSpeechWords(await open_extractor(), FakeEmbedder(), workers=2, open_extractor=open_extractor,
                                         detect=dict(rule='overlap', min_slot_ms=100),
                                         window=dict(margin_ms=500, word_context_ms=400))
                pipeline.add_module('tse', lane, consumes=['microphone', 'diarization'], spec=lane.spec, queue=1000,
                                    catch_up=dict(lag_ms=600, max_lookback_ms=3000, tail_ms=300, workers=2))
                await pipeline.run()
                status = pipeline.status()
                windows = [row['observation']['value'] for row in human.store.query(
                    'local-user', kind='observation.signal', limit=500, registered_only=True, capability_id=lane.spec.id)]
                await pipeline.close()
                return windows, status, extractors
        steady, _, _ = asyncio.run(main(0.))
        slowed, status, extractors = asyncio.run(main(1.2))
        catch = status['modules']['tse']['catch_up']
        span = lambda found: sorted((item['start_ms'] % 8000, item['end_ms'] - item['start_ms']) for item in found)
        self.assertEqual(len(steady), 2)                       # 1.5-2 s of each 8 s round
        self.assertGreaterEqual(catch['skips'], 1)             # it fell behind
        self.assertEqual(catch['done'], catch['skips'])
        self.assertEqual(status['modules']['tse']['gaps'], 0)  # nothing given up
        self.assertEqual(len(slowed), len(steady))             # every stretch once
        self.assertGreater(len(extractors), 1)                 # a fork used an extraction worker of its own


class VerificationUsesTSETest(unittest.TestCase):
    def test_an_overlapped_word_takes_the_tse_decision_and_other_words_keep_the_old_rule(self):
        from gamst.providers.word_verification import WordVerification

        class Embedder:
            model_id = 'fake'

        verify = WordVerification(None, None, Embedder())
        window = dict(turns=[(0, 1000, 'P'), (0, 1000, 'O')], speakers=[dict(speaker='P', score=.9),
                                                                       dict(speaker='O', score=.1)],
                      person={'P'}, unknown=set())
        tse = lambda decision: [dict(word_id='w', tse=dict(decision=decision, score=.7))]
        found = asyncio.run(verify._voice_of(window, 100, 400, tse('person')))
        self.assertEqual(found[:3], (True, 'overlap_tse_person', None))
        self.assertEqual(asyncio.run(verify._voice_of(window, 100, 400, tse('other')))[:3],
                         (False, 'overlap_tse_other', 'other_speaker'))
        self.assertEqual(asyncio.run(verify._voice_of(window, 100, 400, tse('uncertain')))[:3],
                         (False, 'overlap_tse_uncertain', 'uncertain_overlap'))
        alone = dict(window, turns=[(0, 1000, 'P')])
        self.assertEqual(asyncio.run(verify._voice_of(alone, 100, 400, tse('other')))[:3],
                         (True, 'person_speaker', None))          # no overlap there: the person's speaker decides


class RecipeTest(unittest.TestCase):
    def test_with_tse_puts_the_decision_between_speaker_check_and_terms_and_verification(self):
        from gamst.recipes import choose, read_recipe
        recipe = read_recipe('person-term-count')[1]
        reads = lambda chosen: {item['name']: item['consumes'] for item in chosen['modules']}
        both = reads(choose(recipe, include=('tse', 'verification')))
        self.assertEqual(both['terms'], ['tse_decision', 'verification'])
        self.assertEqual(both['verification'], ['microphone', 'tse_decision'])
        self.assertIn('diarization', both)
        self.assertEqual(reads(choose(recipe, include=('tse',)))['terms'], ['tse_decision'])
        self.assertEqual(reads(choose(recipe, include=('verification',)))['terms'], ['speaker_check', 'verification'])
        self.assertNotIn('tse', reads(choose(recipe)))


if __name__ == '__main__':
    unittest.main()
