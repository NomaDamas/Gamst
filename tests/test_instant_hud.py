"""The HUD shows L1's count at once and every later correction at the moment it was decided (owner decision 26)."""
import unittest

from gamst.count_video import change_line, hud_timeline, item_state
from gamst.recipes import event_states


def record(events, ms=0, rid=None):
    return dict(id=rid or f'r{ms}', timestamp_ms=ms, observation=dict(capability=dict(id='x'), value=dict(events=events)))


class EventHistoryTest(unittest.TestCase):
    def test_every_revision_is_kept_with_its_capture_time_and_the_moment_it_was_decided(self):
        rows = [record([dict(event='term.spoken', event_id='a', state='provisional', end_ms=5)], 10, 'r1'),
                record([dict(event='event.revision', revises='a', state='uncertain', reason='uncertain_overlap',
                             by='verification')], 20, 'r2'),
                record([dict(event='event.revision', revises='a', state='final', by='tse')], 30, 'r3')]
        paced = event_states(rows, {'r1': 100, 'r2': 300, 'r3': 400}, paced=True)[0]
        self.assertEqual([(step['state'], step['capture_ms'], step['decided_ms'], step.get('by'))
                          for step in paced['history']],
                         [('provisional', 10, 100, None), ('uncertain', 20, 300, 'verification'), ('final', 30, 400, 'tse')])
        self.assertEqual(paced['history'][1]['reason'], 'uncertain_overlap')
        recorded = event_states(rows, {'r1': 100, 'r2': 300, 'r3': 400})[0]     # not paced: the capture clock
        self.assertEqual([step['decided_ms'] for step in recorded['history']], [10, 20, 30])
        self.assertEqual(recorded['state'], 'final')

    def test_a_revision_says_which_lane_settled_it(self):
        from gamst.plugins import InputFrame
        from gamst.providers.term_events import TermEvents
        module = TermEvents(terms=['뽀또'])
        word = dict(text='뽀또', start_ms=0, end_ms=300, match=True, state='provisional', word_id='u#0')
        module.process(InputFrame(dict(utterance_id='u', status='final', text='뽀또', words=[word]), 'audio', 300))
        out = module.process(InputFrame(dict(kind='verdicts', verdicts=[
            dict(word_id='u#0', match=False, reason='other_speaker', settled_by='verification')]), 'audio', 900))
        revision = out['events'][0]
        self.assertEqual((revision['state'], revision['by'], revision['reason']), ('revoked', 'verification', 'other_speaker'))


def stated(event_id, kind, end_s, *steps, **extra):
    return dict(event=kind, event_id=event_id, end_ms=1000 + round(1000 * end_s), state=steps[-1][1],
                history=[dict(state=state, decided_ms=1000 + round(1000 * at), capture_ms=0, **more)
                         for at, state, *rest in steps for more in [rest[0] if rest else {}]], **extra)


class HudTimelineTest(unittest.TestCase):
    def report(self, events):
        return dict(start_ms=1000, paced=True, event_states={'speech.term_events': events})

    def test_live_words_count_at_once_and_corrections_come_when_they_were_decided(self):
        report = self.report([
            stated('a', 'term.spoken', 2., (2.5, 'provisional'), (20., 'final')),
            stated('b', 'term.spoken', 3., (3.4, 'provisional'), (20., 'revoked', dict(reason='other_speaker'))),
            stated('c', 'term.other_voice', 4., (4.3, 'provisional'), (21., 'revoked')),
            stated('c2', 'term.spoken', 4., (21., 'final'), replaces='c'),
            stated('d', 'term.spoken', 6., (22., 'final'), decided_by='verification')])
        hud = hud_timeline(report, 'speech.term_events:term.spoken', 'speech.term_events:term.other_voice')
        count = lambda t: sum(item_state(item, t) in ('provisional', 'final') for item in hud['items'])
        self.assertEqual([count(t) for t in (2.4, 2.6, 3.5, 19.9, 20.1, 21.1, 22.1)], [0, 1, 2, 2, 1, 2, 3])
        kinds = [(round(at, 1), delta, kind, item['number']) for at, delta, kind, item in hud['changes']]
        self.assertEqual(kinds, [(2.5, 1, 'live', 1), (3.4, 1, 'live', 2), (20., 0, 'final', 1), (20., -1, 'revoked', 2),
                                 (21., 1, 'added', 3), (22., 1, 'added', 4)])
        other = next(item for item in hud['items'] if item['event_id'] == 'c')
        self.assertEqual((item_state(other, 5.), item_state(other, 21.)), ('other', 'final'))
        lines = {kind: change_line(kind, item) for _, _, kind, item in hud['changes']}
        self.assertEqual(lines['revoked'], ('−1 취소 #2', '다른 사람 목소리'))
        self.assertEqual(change_line('added', other), ('+1 추가 #3', '감스트 목소리로 정정'))

    def test_an_uncertain_word_leaves_the_count_and_comes_back_when_settled(self):
        report = self.report([stated('a', 'term.spoken', 2., (2.5, 'provisional'),
                                     (10., 'uncertain', dict(reason='uncertain_overlap')), (15., 'final'))])
        hud = hud_timeline(report, 'speech.term_events:term.spoken')
        self.assertEqual([(kind, delta) for _, delta, kind, _ in hud['changes']],
                         [('live', 1), ('uncertain', -1), ('added', 1)])
        self.assertEqual(change_line('uncertain', hud['items'][0]), ('−1 보류 #1', '불확실: 목소리 겹침'))

    def test_a_report_without_history_falls_back_to_committed_and_settled_times(self):
        report = dict(start_ms=1000, paced=True, event_states={'speech.term_events': [
            dict(event='term.spoken', event_id='a', state='revoked', end_ms=2000, committed_ms=2500, settled_ms=9000,
                 committed_state='provisional', revised_by=dict(reason='not_heard'))]})
        hud = hud_timeline(report, 'speech.term_events:term.spoken')
        self.assertEqual([(round(at, 1), kind) for at, _, kind, _ in hud['changes']], [(1.5, 'live'), (8., 'revoked')])


if __name__ == '__main__':
    unittest.main()


class SameClipVoiceTest(unittest.TestCase):
    def test_the_main_speaker_is_who_speaks_alone_longest(self):
        from gamst.recipes.enrollment import choose_main_speaker
        turns = [(0., 4., 'A'), (3., 10., 'B'), (11., 12., 'A')]
        speaker, pieces, seconds = choose_main_speaker(turns, 0., 15.)
        self.assertEqual((speaker, pieces, seconds), ('B', [(4., 10.)], 6.))
        self.assertIsNone(choose_main_speaker([], 0., 15.))

    def test_owner_scores_count_repetitions_and_leave_out_the_voice_reference(self):
        from gamst.recipes.bench import error_types, match, view
        truth = dict(tolerance_s=.6, exclude=[dict(from_s=0., to_s=5.)],
                     words=[dict(end_s=3., by='person'), dict(end_s=8., by='person', count=3), dict(end_s=12., by='other')])
        reps = [dict(event='term.spoken', event_id=f'r{i}', state='final', start_ms=1000 + 6500 + 500 * i,
                     end_ms=1000 + 7000 + 500 * i, repetition=[i, 3], word_end_ms=1000 + 8000) for i in range(3)]
        events = [dict(event='term.spoken', event_id='x', state='final', end_ms=1000 + 3000), *reps,
                  dict(event='term.spoken', event_id='o', state='final', end_ms=1000 + 12000)]
        self.assertEqual([label for _, label, _ in match(truth, events, 1000)],
                         ['excluded', 'person', 'person', 'person', 'other'])
        result = view(truth, events, 1000)
        self.assertEqual((result['counted'], result['right'], result['other'], result['recall']), (4, 3, 1, 1.))
        errors = error_types(truth, dict(start_ms=1000, event_states={'speech.term_events': events}))
        self.assertEqual((errors['person'], errors['counted_right'], errors['other_to_person']), (3, 3, 1))


class VoiceBankTest(unittest.TestCase):
    def test_a_word_counts_only_when_closer_to_the_person_than_to_every_known_other_voice(self):
        from gamst.providers.voice_bank import VoiceBank, shared
        bank = VoiceBank([[1., 0., 0.]], grow=True, combine='max')
        near_other = [.8, .6, 0.]       # 0.8 to the person
        self.assertTrue(bank.decide(near_other, floor=.5, margin=.1)[0])
        bank.add_negative([.6, .8, 0.], 'w1:B')      # another speaker of a window where the person was clear
        matched, pos, neg = bank.decide(near_other, floor=.5, margin=.1)
        self.assertEqual((matched, pos, neg), (False, .8, .96))
        self.assertTrue(bank.add_person([.7, .7, 0.], 'w2:A'))
        self.assertFalse(bank.add_person([.7, .7, 0.], 'w2:A'))     # once per window speaker
        self.assertEqual(bank.status()['references'], 2)
        pipeline = type('P', (), {})()
        self.assertIs(shared(pipeline, 'voice', [[1., 0., 0.]]), shared(pipeline, 'voice', [[0., 1., 0.]]))

    def test_the_live_lane_decides_a_word_against_the_bank(self):
        from gamst.providers.speaker_verify import SpeakerVerify

        class Embedder:
            model_id = 'stand-in'

            def embed(self, samples, rate):
                return [.8, .6, 0.]
        module = SpeakerVerify(Embedder(), words=True, context_ms=1500, segments=False, relative=dict(floor=.5, margin=.1))
        pipeline = type('P', (), dict(references={'voice': dict(kind='voice', value=dict(
            model='stand-in', embeddings=[[1., 0., 0.]]))}))()
        module.bind(pipeline)
        module.bank.add_negative([.6, .8, 0.], 'B')
        import asyncio
        import numpy as np
        module._covers = lambda first, last: True
        module._span = lambda first, last: np.full(16000, .1, dtype=np.float32)
        word = asyncio.run(module._word(dict(text='a', start_ms=500, end_ms=800)))
        self.assertEqual((word['match'], word['speaker_score'], word['negative_score'], word['decision']),
                         (False, .8, .96, 'relative'))

    def test_verification_names_the_closest_speaker_with_a_margin_and_keeps_the_others_as_negatives(self):
        import asyncio
        import numpy as np
        from gamst.providers.voice_bank import VoiceBank
        from gamst.providers.word_verification import WordVerification
        voices = {0: [.9, .436, 0.], 1: [.8, .6, 0.]}      # speaker A (0-2 s) and B (2-4 s): both near the person

        class Memory:
            start_ms, end_ms = 0, 4000

            def span(self, low, high, rate):
                return np.full(int((high - low) * rate / 1000), .1 + .1 * (low >= 2000), dtype=np.float32)

        class Embedder:
            model_id = 'stand-in'

            def embed(self, samples, rate):
                return voices[int(samples[0] > .15)]

        class Diarizer:
            def diarize(self, samples):
                return [(0., 2., 'A'), (2., 4., 'B')]
        module = WordVerification(object(), Diarizer(), Embedder(), voice=dict(decision='relative', floor=.5, margin=.05))
        module.memory, module.target = Memory(), [1., 0., 0.]
        module.bank = VoiceBank([[1., 0., 0.]])
        window = asyncio.run(module._diarized(0, 4000))
        self.assertEqual(window['person'], {'A'})
        self.assertEqual(module.bank.status()['negatives'], 1)
        module.voice['margin'] = .2      # not clearly closer: nobody is named, nothing is learnt
        module.bank = VoiceBank([[1., 0., 0.]])
        window = asyncio.run(module._diarized(0, 4000))
        self.assertEqual((window['person'], module.bank.status()['negatives']), (set(), 0))
