"""Counted words in the live recogniser's partial hypotheses count at once (heard), and the final corrects them."""
import asyncio
import unittest

import numpy as np

from gamst.counting import count_events, revisions, with_state
from gamst.count_video import hud_timeline, item_state
from gamst.plugins import InputFrame
from gamst.providers.partial_terms import PartialSpotter, reconcile
from gamst.providers.speaker_verify import SpeakerVerify
from gamst.providers.term_events import TermEvents, parse_terms
from gamst.recipes import event_states
from gamst.recipes.bench import heard_events, live_events


def partial(text, heard, key='s:0', start=0):
    return dict(utterance_id=key, status='partial', text=text, start_ms=start, end_ms=heard, heard_to_ms=heard, words=[])


def heard_word(index=0, start=1000, end=1300, key='s:0', match=True):
    return dict(text='뽀또', state='heard', term='뽀또', index=index, word_id=f'heard:{key}:뽀또:{index}',
                start_ms=start, end_ms=end, heard_to_ms=end + 500, word_timed=False, match=match, speaker_score=.4)


def final(text, words, key='s:0'):
    return dict(utterance_id=key, status='final', text=text, start_ms=words[0]['start_ms'], end_ms=words[-1]['end_ms'],
                words=words)


def said(text, start, end, index=0, key='s:0', state='provisional', match=True):
    return dict(text=text, start_ms=start, end_ms=end, match=match, state=state, word_id=f'{key}#{index}')


class SpotterTest(unittest.TestCase):
    def test_a_term_counts_once_it_survived_the_revisions_and_a_flicker_never_does(self):
        spotter = PartialSpotter(parse_terms(['뽀또'])[0], min_revisions=2)
        self.assertEqual(spotter.update(partial('뽀또', 1000)), [])          # first seen
        self.assertEqual(spotter.update(partial('포도', 1300)), [])          # gone: a flicker
        self.assertEqual(spotter.update(partial('뽀또 주세요', 1600)), [])
        found = spotter.update(partial('뽀또 주세요', 1900))
        self.assertEqual([(item['term'], item['index'], item['revisions']) for item in found], [('뽀또', 0, 2)])
        self.assertEqual(spotter.update(partial('뽀또 주세요', 2200)), [])   # once per utterance
        stats = spotter.stats()
        self.assertEqual((stats['appeared'], stats['flickered'], stats['emitted']), (2, 1, 1))
        self.assertEqual(spotter.trace()[1], ['s:0', 1300, {}])           # counts only, no text

    def test_the_time_is_estimated_from_when_it_was_heard_and_what_follows_it(self):
        spotter = PartialSpotter(parse_terms(['뽀또'])[0], min_revisions=1, lag_ms=400, ms_per_char=150)
        item = spotter.update(partial('뽀또 주세요', 5000, start=2000))[0]
        self.assertEqual((item['end_ms'], item['start_ms'], item['timed']), (5000 - 400 - 3 * 150, 5000 - 850 - 300, False))

    def test_min_ms_makes_it_stable_by_time_too_and_a_final_closes_the_utterance(self):
        spotter = PartialSpotter(parse_terms(['뽀또'])[0], min_revisions=5, min_ms=300)
        spotter.update(partial('뽀또', 1000))
        self.assertEqual(len(spotter.update(partial('뽀또', 1350))), 1)
        spotter.final('s:0')
        self.assertEqual(spotter.update(partial('뽀또 뽀또', 1700)), [])

    def test_reconcile_pairs_in_order_nearest_first(self):
        pairs = reconcile([(1000, 'a'), (3000, 'b')], [(1100, 'x'), (2900, 'y'), (9000, 'z')], max_ms=500)
        self.assertEqual(pairs, [('a', 'x'), ('b', 'y')])
        self.assertEqual(reconcile([(1000, 'a'), (1200, 'b')], [(1250, 'x')], max_ms=500), [('b', 'x')])


class HeardEventsTest(unittest.TestCase):
    def run_frames(self, module, values):
        out = []
        for at, value in values:
            found = module.process(InputFrame(value, 'audio', at))
            out += found if isinstance(found, list) else [found] if found else []
        return out

    def test_heard_counts_at_once_and_the_final_confirms_the_same_event(self):
        module = TermEvents(terms=['뽀또'])
        out = self.run_frames(module, [
            (1800, dict(partial('', 1800), text='뽀또', words=[heard_word()])),
            (4000, final('뽀또 주세요', [said('뽀또', 1100, 1400), said('주세요', 1500, 1900, 1)]))])
        self.assertEqual([record['count'] for record in out], [1, 1])
        heard, revision = out[0]['events'][0], out[1]['events'][0]
        self.assertEqual(heard['state'], 'heard')
        self.assertEqual((revision['event'], revision['revises'], revision['state'], revision['by'], revision['end_ms']),
                         ('event.revision', heard['event_id'], 'provisional', 'l1', 1400))
        settled = module.process(InputFrame(dict(kind='verdicts', verdicts=[dict(word_id='s:0#0', match=True)]),
                                            'audio', 9000))
        self.assertEqual((settled['events'][0]['revises'], settled['events'][0]['state'], settled['count']),
                         (heard['event_id'], 'final', 1))

    def test_a_heard_word_the_final_does_not_hold_is_withdrawn(self):
        module = TermEvents(terms=['뽀또'])
        out = self.run_frames(module, [
            (1800, dict(partial('', 1800), text='뽀또', words=[heard_word()])),
            (4000, final('포도 주세요', [said('포도', 1100, 1400), said('주세요', 1500, 1900, 1)]))])
        self.assertEqual(out[-1]['count'], 0)
        self.assertEqual((out[-1]['events'][0]['state'], out[-1]['events'][0]['reason']), ('revoked', 'partial_withdrawn'))

    def test_a_clearly_other_voice_is_not_heard_and_a_late_partial_is_ignored(self):
        module = TermEvents(terms=['뽀또'])
        self.assertIsNone(module.process(InputFrame(dict(partial('', 1800), text='뽀또', words=[heard_word(match=False)]),
                                                    'audio', 1800)))
        module.process(InputFrame(final('뽀또', [said('뽀또', 1100, 1400)]), 'audio', 4000))
        self.assertIsNone(module.process(InputFrame(dict(partial('', 4100), text='뽀또', words=[heard_word(1)]),
                                                    'audio', 4100)))

    def test_a_turn_whose_final_never_came_is_withdrawn_by_a_later_final_or_at_the_end(self):
        module = TermEvents(terms=['뽀또'])
        self.run_frames(module, [(1800, dict(partial('', 1800), text='뽀또', words=[heard_word()]))])
        out = self.run_frames(module, [(9000, final('네', [said('네', 6000, 6300, key='s:1')], key='s:1'))])
        self.assertEqual((out[0]['events'][0]['reason'], out[0]['count']), ('partial_withdrawn', 0))
        self.run_frames(module, [(9500, dict(partial('', 9500, key='s:2'), text='뽀또', words=[heard_word(key='s:2')]))])
        self.assertEqual(module.finish()['events'][0]['reason'], 'partial_withdrawn')

    def test_raw_partials_are_spotted_here_without_a_voice_module(self):
        module = TermEvents(terms=['뽀또'], partials=dict(min_revisions=2))
        out = self.run_frames(module, [(1000, partial('뽀또', 1000)), (1300, partial('뽀또', 1300)),
                                       (4000, final('뽀또', [dict(text='뽀또', start_ms=500, end_ms=800)]))])
        self.assertEqual([record['count'] for record in out], [1, 1])
        self.assertEqual(out[1]['events'][0]['state'], 'final')       # no later lane: the final hearing is final


class StatesTest(unittest.TestCase):
    def rows(self):
        module = TermEvents(terms=['뽀또'])
        values = [(1800, dict(partial('', 1800), text='뽀또', words=[heard_word()])),
                  (4000, final('뽀또', [said('뽀또', 1100, 1400)]))]
        rows = []
        for index, (at, value) in enumerate(values):
            rows.append(dict(id=f'r{index}', timestamp_ms=at, observation=dict(
                capability=dict(id='speech.term_events'), value=module.process(InputFrame(value, 'audio', at)))))
        return rows

    def test_a_heard_event_is_counted_once_and_takes_the_final_times(self):
        rows = self.rows()
        counted = count_events(rows)
        self.assertEqual((counted['count'], counted['states']['provisional']), (1, 1))
        event = with_state(rows[0]['observation']['value']['events'][0], revisions(rows))
        self.assertEqual((event['state'], event['start_ms'], event['end_ms'], event['heard_end_ms']),
                         ('provisional', 1100, 1400, 1300))

    def test_the_hud_counts_a_heard_word_once_and_l1_keeps_it(self):
        rows = self.rows()
        report = dict(start_ms=0, paced=True, event_states={'speech.term_events': event_states(
            rows, {'r0': 1850, 'r1': 4050}, paced=True)})
        hud = hud_timeline(report, 'speech.term_events:term.spoken')
        self.assertEqual([(at, delta, kind) for at, delta, kind, _ in hud['changes']], [(1.85, 1, 'heard')])
        self.assertEqual([item_state(hud['items'][0], t) for t in (1.8, 2., 4.1)], [None, 'heard', 'provisional'])
        self.assertEqual(len(live_events(report)), 1)
        self.assertEqual(len(heard_events(report)), 1)
        replay = hud_timeline(report, 'speech.term_events:term.spoken', at_spoken=True)     # presentation mode
        self.assertEqual([(at, delta, kind) for at, delta, kind, _ in replay['changes']], [(1.4, 1, 'live')])
        self.assertEqual(item_state(replay['items'][0], 1.5), 'provisional')


class Embedder:
    model_id = 'stand-in'

    def __init__(self, vector):
        self.vector = vector

    def embed(self, samples, rate):
        return self.vector


class SpeakerCheckTest(unittest.TestCase):
    def module(self, vector):
        found = SpeakerVerify(Embedder(vector), words=True, context_ms=1500, segments=False, threshold=.5,
                              provisional=True, l1=dict(policy='instant', clear_other=.1),
                              partials=dict(terms='terms', min_revisions=2))
        pipeline = type('P', (), dict(references={
            'voice': dict(kind='voice', value=dict(model='stand-in', embeddings=[[1., 0., 0.]])),
            'terms': dict(kind='terms', value=['뽀또'])}))()
        found.bind(pipeline)
        found.ring.append(np.full(16000 * 3, .1, dtype=np.float32), 3000, 16000)
        return found

    def test_a_stable_term_in_a_partial_leaves_with_its_voice_and_nothing_else_of_the_text(self):
        found = self.module([.45, .89, 0.])
        first = asyncio.run(found.process(InputFrame(partial('뽀또 주세요', 2500), 'audio', 2500)))
        self.assertIsNone(first)
        out = asyncio.run(found.process(InputFrame(partial('뽀또 주세요', 2800), 'audio', 2800)))
        word = out['words'][0]
        self.assertEqual((out['status'], out['text'], word['state'], word['match'], word['l1']),
                         ('partial', '뽀또', 'heard', True, 'instant'))
        self.assertAlmostEqual(word['speaker_score'], .45, places=2)
        self.assertEqual(found.status()['partials']['emitted'], 1)

    def test_a_clearly_other_voice_is_marked(self):
        found = self.module([.05, .99, 0.])
        asyncio.run(found.process(InputFrame(partial('뽀또', 2500), 'audio', 2500)))
        out = asyncio.run(found.process(InputFrame(partial('뽀또', 2800), 'audio', 2800)))
        self.assertFalse(out['words'][0]['match'])


if __name__ == '__main__':
    unittest.main()
