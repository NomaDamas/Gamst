import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from gamst import Gamst
from gamst.counting import count_events, event_rows
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.providers.speaker_tracks import SpeakerTracks, TrackVerification, WholeBlocks
from gamst.providers.speaker_verify import centroid, enroll
from gamst.providers.term_events import TermEvents
from gamst.recipes import choose, event_states, load
from gamst.recipes.bench import match, score_run, table, view, words_of


def now():
    return time.time_ns() // 1_000_000


class ToneEmbedder:
    """Test double: the embedding is the normalized magnitude spectrum in 8 bands (pitch-sensitive)."""
    model_id = 'test-tone'

    def __init__(self):
        self.calls = []

    def embed(self, samples, sample_rate=16000):
        self.calls.append(len(samples))
        spectrum = np.abs(np.fft.rfft(np.asarray(samples, np.float32)))
        bands = np.array_split(spectrum[:len(spectrum) // 4], 8)
        return [float(band.mean()) + 1e-6 for band in bands]


def tone(hz, seconds, rate=16000, amplitude=.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def record(events, ms=0, rid=None):
    return dict(id=rid or f'r{ms}', timestamp_ms=ms, observation=dict(capability=dict(id='x'), value=dict(events=events)))


class RevisionProtocolTest(unittest.TestCase):
    def test_counts_are_provisional_plus_final_and_revoked_ones_are_listed_not_counted(self):
        rows = [record([dict(event='hit', event_id='a', state='provisional'), dict(event='hit', event_id='b',
                                                                                   state='provisional'),
                        dict(event='hit', event_id='c', state='provisional'), dict(event='hit')], 1),
                record([dict(event='event.revision', revises='a', state='final'),
                        dict(event='event.revision', revises='b', state='revoked', reason='another_voice')], 2)]
        counted = count_events(rows, event='hit')
        self.assertEqual(counted['count'], 3)
        self.assertEqual(counted['states'], dict(provisional=1, final=2, revoked=1))
        states = {item['event'].get('event_id'): item['event'].get('state') for item in counted['evidence']}
        self.assertEqual(states, {'a': 'final', 'c': 'provisional', None: None})
        self.assertEqual(count_events(rows, event='hit', where={'state': 'revoked'})['count'], 1)
        self.assertEqual(count_events(rows, event='hit', where={'state': 'final'})['count'], 2)   # 'd' has no state: final
        self.assertEqual(count_events(rows, event='event.revision')['count'], 2)
        grouped = count_events(rows, group_by='@event')
        self.assertEqual(grouped['groups'], {'hit': 3})
        self.assertEqual(grouped['group_states']['hit']['revoked'], 1)
        revoked = next(row for row in event_rows(rows) if row.get('event_id') == 'b')
        self.assertEqual((revoked['state'], revoked['revised_by']['reason']), ('revoked', 'another_voice'))

    def test_without_states_nothing_changes(self):
        counted = count_events([record([dict(event='hit'), dict(event='hit')])], event='hit')
        self.assertEqual(counted['count'], 2)
        self.assertNotIn('states', counted)

    def test_the_newest_revision_decides(self):
        rows = [record([dict(event='hit', event_id='a', state='provisional')], 1),
                record([dict(event='event.revision', revises='a', state='revoked')], 2),
                record([dict(event='event.revision', revises='a', state='final')], 3)]
        self.assertEqual(count_events(rows, event='hit')['states'], dict(provisional=0, final=1, revoked=0))


def words(utterance, start, *spans, provisional=True, matches=None):
    items = []
    for index, (first, last) in enumerate(spans):
        item = dict(text='대박', start_ms=start + first, end_ms=start + last, word_id=f'{utterance}#{index}')
        if provisional:
            item.update(state='provisional', match=True if matches is None else matches[index], slot='s1:0')
        items.append(item)
    return dict(utterance_id=utterance, status='final', text=' '.join('대박' for _ in spans), start_ms=start + spans[0][0],
                end_ms=start + spans[-1][1], words=items)


class TermEventsRevisionTest(unittest.TestCase):
    def frame(self, value, ms=0):
        return InputFrame(value, 'audio', ms)

    def test_provisional_events_are_settled_by_verdicts(self):
        terms = TermEvents(terms=['대박'], report_others=True)
        first = terms.process(self.frame(words('u1', 0, (0, 300), (400, 700), matches=[True, False])))
        events = first['events']
        self.assertEqual([(event['event'], event['state']) for event in events],
                         [('term.spoken', 'provisional'), ('term.other_voice', 'provisional')])
        self.assertEqual(first['count'], 1)
        self.assertEqual(len({event['event_id'] for event in events}), 2)
        # The counted word was another voice; the other one was the enrolled voice after all.
        settled = terms.process(self.frame(dict(kind='verdicts', verdicts=[
            dict(word_id='u1#0', match=False, reason='slot_identity_other'), dict(word_id='u1#1', match=True, slot_score=.8)])))
        revisions = [event for event in settled['events'] if event['event'] == 'event.revision']
        self.assertEqual([(item['revises'], item['state']) for item in revisions],
                         [(events[0]['event_id'], 'revoked'), (events[1]['event_id'], 'revoked')])
        self.assertEqual(revisions[0]['reason'], 'slot_identity_other')
        promoted = next(event for event in settled['events'] if event['event'] == 'term.spoken')
        self.assertEqual((promoted['state'], promoted['replaces']), ('final', events[1]['event_id']))
        self.assertEqual(settled['count'], 1)
        rows = [record(first['events'], 1, 'a'), record(settled['events'], 2, 'b')]
        self.assertEqual(count_events(rows, event='term.spoken')['states'], dict(provisional=0, final=1, revoked=1))
        self.assertIsNone(terms.process(self.frame(dict(kind='verdicts', verdicts=[dict(word_id='u1#0', match=True)]))))

    def test_a_confirmed_word_becomes_final_and_the_same_utterance_has_the_same_ids(self):
        ids = []
        for _ in range(2):
            terms = TermEvents(terms=['대박'])
            ids.append(terms.process(self.frame(words('u1', 0, (0, 300))))['events'][0]['event_id'])
        self.assertEqual(ids[0], ids[1])
        settled = terms.process(self.frame(dict(kind='verdicts', verdicts=[dict(word_id='u1#0', match=True)])))
        self.assertEqual([(item['event'], item['state']) for item in settled['events']], [('event.revision', 'final')])

    def test_words_without_states_are_final_as_committed(self):
        value = TermEvents(terms=['대박']).process(self.frame(words('u1', 0, (0, 300), provisional=False)))
        self.assertNotIn('state', value['events'][0])
        self.assertIn('event_id', value['events'][0])


def activity(session, start, end, slots):
    return dict(diarization=dict(session_id=session, chunks=[dict(start_ms=start, end_ms=end, frame_ms=10, activity=[
        dict(slot=slot, intervals_ms=intervals) for slot, intervals in slots.items()])]))


class SpeakerTracksTest(unittest.IsolatedAsyncioTestCase):
    async def build(self, **options):
        embedder = ToneEmbedder()
        reference = await enroll(embedder, [tone(220, 1.5)])
        tracks = SpeakerTracks(embedder, threshold=.9, piece_ms=1000, min_ms=500, stable_ms=3000, **options)
        tracks.target = centroid(reference['embeddings'])
        embedder.calls.clear()
        return tracks, embedder

    async def test_each_word_goes_to_its_slot_and_the_slot_is_decided_by_its_own_sound(self):
        tracks, embedder = await self.build()
        audio = np.concatenate([tone(220, 2.), tone(1500, 2.)])     # person 0-2 s, another voice 2-4 s
        for i in range(0, len(audio), 1600):
            await tracks.process(InputFrame(audio[i:i + 1600], 'audio', 1000 + (i + 1600) // 16, sample_rate=16000))
        utterance = dict(utterance_id='u', status='final', text='대박 대박 대박', start_ms=2300, end_ms=4700, words=[
            dict(text='대박', start_ms=2300, end_ms=2700), dict(text='대박', start_ms=3800, end_ms=4200),
            dict(text='대박', start_ms=4400, end_ms=4700)])
        self.assertIsNone(await tracks.process(InputFrame(utterance, 'audio', 5000)))    # activity does not cover it yet
        self.assertEqual(tracks.pending_since_ms, 2300)
        out = await tracks.process(InputFrame(activity('x', 1000, 5000, {0: [[1000, 3000]], 1: [[3000, 5000]]}), 'audio', 5000))
        slots = next(item for item in out if item['kind'] == 'slots')['slots']
        self.assertEqual({slot['slot']: slot['enrolled'] for slot in slots}, {'s1:0': True, 's1:1': False})
        self.assertTrue(all(slot['speech_ms'] == 2000 for slot in slots))
        attributed = next(item for item in out if item['kind'] == 'utterance')['words']
        self.assertEqual([(word['slot'], word['match'], word['state']) for word in attributed],
                         [('s1:0', True, 'provisional'), ('s1:1', False, 'provisional'), ('s1:1', False, 'provisional')])
        self.assertEqual(attributed[0]['word_id'], 'u#0')
        self.assertIsNone(tracks.pending_since_ms)
        self.assertEqual(embedder.calls, [32000, 32000])   # each slot's own sound embedded once (pieces of >= 1 s)

    async def test_overlap_is_left_out_and_a_word_without_a_slot_is_not_counted(self):
        tracks, embedder = await self.build()
        audio = tone(220, 2.)
        for i in range(0, len(audio), 1600):
            await tracks.process(InputFrame(audio[i:i + 1600], 'audio', (i + 1600) // 16, sample_rate=16000))
        calls = len(embedder.calls)
        out = await tracks.process(InputFrame(activity('x', 0, 2000, {0: [[0, 2000]], 1: [[0, 2000]]}), 'audio', 2000))
        self.assertIsNone(out)                       # both slots at once: nobody's own sound
        self.assertEqual(len(embedder.calls), calls)
        found = await tracks.process(InputFrame(dict(utterance_id='v', status='final', text='대박', start_ms=100, end_ms=400,
                                                     words=[dict(text='대박', start_ms=100, end_ms=400)]), 'audio', 2000))
        word = found[0]['words'][0]
        self.assertEqual((word['slots_active'], word['decided_by'], word['match']), (2, 'slot_unknown', False))

    async def test_a_new_session_has_new_slots(self):
        tracks, _ = await self.build()
        audio = tone(220, 2.)
        for i in range(0, len(audio), 1600):
            await tracks.process(InputFrame(audio[i:i + 1600], 'audio', (i + 1600) // 16, sample_rate=16000))
        first = await tracks.process(InputFrame(activity('x', 0, 1000, {0: [[0, 1000]]}), 'audio', 1000))
        second = await tracks.process(InputFrame(activity('y', 1000, 2000, {0: [[1000, 2000]]}), 'audio', 2000))
        self.assertEqual([item['slots'][0]['slot'] for item in (first[0], second[0])], ['s1:0', 's2:0'])


class VerificationTest(unittest.TestCase):
    def frame(self, value):
        return InputFrame(value, 'audio', 0)

    def slots(self, end, **slots):
        return dict(kind='slots', reference='voice', start_ms=end, end_ms=end,
                    slots=[dict(slot=key.replace('_', ':'), **value) for key, value in slots.items()])

    def test_words_settle_when_their_slot_is_stable_or_after_the_delay(self):
        verify = TrackVerification(settle_ms=2000, window_ms=5000)
        utterance = dict(kind='utterance', reference='voice', utterance_id='u', words=[
            dict(word_id='u#0', start_ms=100, end_ms=400, slot='s1:0', match=True, state='provisional', slot_score=.6),
            dict(word_id='u#1', start_ms=500, end_ms=800, slot='s1:1', match=True, state='provisional', slot_score=.6),
            dict(word_id='u#2', start_ms=900, end_ms=1000, slot=None, match=False, state='provisional')])
        self.assertIsNone(verify.process(self.frame(utterance)))
        self.assertEqual(verify.pending_since_ms, 100)
        out = verify.process(self.frame(self.slots(1500, s1_0=dict(score=.7, enrolled=True, stable=True),
                                                   s1_1=dict(score=.3, enrolled=False, stable=False))))
        self.assertEqual([(item['word_id'], item['match'], item['changed'], item['settled_by']) for item in out['verdicts']],
                         [('u#0', True, False, 'stable')])
        out = verify.process(self.frame(self.slots(3000, s1_1=dict(score=.3, enrolled=False, stable=False))))
        self.assertEqual([(item['word_id'], item['match'], item['changed'], item.get('reason')) for item in out['verdicts']],
                         [('u#1', False, True, 'slot_identity_other'), ('u#2', False, False, None)])
        self.assertIsNone(verify.pending_since_ms)

    def test_what_waits_at_the_end_is_decided(self):
        verify = TrackVerification(settle_ms=2000)
        verify.process(self.frame(dict(kind='utterance', reference='voice', utterance_id='u', words=[
            dict(word_id='u#0', start_ms=100, end_ms=400, slot='s1:0', match=True, state='provisional')])))
        self.assertEqual(verify.finish()['verdicts'][0]['settled_by'], 'final')


class WholeBlocksTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocks_are_whole_10ms_on_a_gap_free_clock(self):
        class Seen:
            spec = None

            def __init__(self):
                self.frames = []

            def process(self, frame):
                self.frames.append((len(frame.data), frame.timestamp_ms))

        seen = Seen()
        blocks = WholeBlocks(seen)
        for size, ms in ((1600, 100), (1000, 162), (1080, 230), (1600, 3000)):
            await blocks.process(InputFrame(np.zeros(size, np.float32), 'audio', ms, sample_rate=16000))
        self.assertEqual(seen.frames, [(1600, 100), (960, 160), (1120, 230), (1600, 3000)])


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_provisional_counts_at_once_then_settled_in_its_own_lane(self):
        embedder = ToneEmbedder()
        reference = await enroll(embedder, [tone(220, 1.5)])
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'tracks')
            for name in ('microphone', 'slots', 'asr'):
                pipeline.add_input(name, modality='audio')
            pipeline.add_reference('voice', 'voice', reference, consent=dict(by='owner', purpose='test'))
            pipeline.add_reference('terms', 'terms', ['대박'])
            tracks = SpeakerTracks(embedder, threshold=.9, piece_ms=1000, min_ms=500, stable_ms=3000)
            pipeline.add_module('tracks', tracks, consumes=['microphone', 'slots', 'asr'], spec=tracks.spec)
            verify = TrackVerification(settle_ms=2000)
            pipeline.add_module('verify', verify, consumes='tracks', spec=verify.spec, queue=16)
            terms = TermEvents(report_others=True)
            pipeline.add_module('terms', terms, consumes=['tracks', 'verify'], spec=terms.spec)
            start = now() - 20000
            audio = np.concatenate([tone(220, 1.), tone(1500, 1.), tone(220, 3.)])    # 0-1 person, 1-2 other, 2-5 person
            for i in range(0, len(audio), 1600):
                await pipeline.push('microphone', audio[i:i + 1600], timestamp_ms=start + (i + 1600) // 16, sample_rate=16000)
            await pipeline.push('asr', dict(utterance_id='u', status='final', text='대박 대박', start_ms=start + 300,
                                            end_ms=start + 1700, words=[
                dict(text='대박', start_ms=start + 300, end_ms=start + 700),
                dict(text='대박', start_ms=start + 1300, end_ms=start + 1700)]), timestamp_ms=start + 1800)
            await pipeline.push('slots', activity('x', start, start + 2000, {0: [[start, start + 1000]],
                                                                             1: [[start + 1000, start + 2000]]}),
                                timestamp_ms=start + 2000)
            await pipeline.drain()
            counted = await human.count_events('speech.term_events', event='term.spoken', window='24h')
            self.assertEqual((counted['count'], counted['states']['provisional']), (1, 1))
            await pipeline.push('slots', activity('x', start + 2000, start + 5000, {0: [[start + 2000, start + 5000]]}),
                                timestamp_ms=start + 5000)
            await pipeline.drain()
            counted = await human.count_events('speech.term_events', window='24h', group_by='@event')
            self.assertEqual(counted['group_states'], {'term.spoken': dict(provisional=0, final=1, revoked=0),
                                                        'term.other_voice': dict(provisional=0, final=1, revoked=0)})
            timeline = await human.get_timeline(start_ms=start, end_ms=start + 6000, step_ms=1000,
                                                capabilities=['speech.term_events'])
            values = [entry['value'] for entry in timeline['evidence'].values()] if 'evidence' in timeline else []
            shown = [value for value in values if isinstance(value, dict) and value.get('event', '').startswith('term.')]
            self.assertTrue(shown and all(value['state'] == 'final' for value in shown), timeline.keys())
            self.assertFalse(any(value.get('event') == 'event.revision' for value in values if isinstance(value, dict)))


class RecipeAndBenchTest(unittest.TestCase):
    def test_tracks_replace_the_word_snippet_check_and_feed_the_terms(self):
        recipe = load('person-term-count', ['tracks'])
        modules = {item['name']: item for item in recipe['modules']}
        self.assertNotIn('speaker_check', modules)
        self.assertEqual(modules['terms']['consumes'], ['tracks', 'verify'])
        self.assertEqual(modules['verify']['queue'], 256)
        self.assertIn('speaker_check', {item['name'] for item in load('person-term-count')['modules']})
        both = choose(json.loads((Path('gamst/recipes/person-term-count.json')).read_text()), ['tracks', 'speaker_check'])
        self.assertIn('speaker_check', {item['name'] for item in both['modules']})

    def test_bench_scores_by_end_time_with_labels_and_exclusions(self):
        truth = dict(tolerance_s=.5, words=[dict(end_s=1., by='person'), dict(end_s=2., by='person', exclude='reference'),
                                            dict(end_s=3., by='other'), dict(end_s=4., by='none'), dict(end_s=6., by='person'),
                                            dict(from_s=8., to_s=9., by='other')])
        events = [dict(end_ms=1000 + 1100), dict(end_ms=1000 + 2000), dict(end_ms=1000 + 3200), dict(end_ms=1000 + 4000),
                  dict(end_ms=1000 + 8500), dict(end_ms=1000 + 7000)]
        self.assertEqual([label for _, label, _ in match(truth, events, 1000)],
                         ['person', 'excluded', 'other', 'none', 'other', 'unlabelled'])
        result = view(truth, events, 1000)
        self.assertEqual((result['counted'], result['right'], result['invented'], result['other'], result['unlabelled'],
                          result['missed']), (5, 1, 1, 2, 1, [6.]))
        self.assertEqual((result['precision'], result['recall']), (.2, .5))

    def test_run_rows_have_final_and_at_once_views_and_latency(self):
        truth = dict(tolerance_s=.5, words=[dict(end_s=1., by='person'), dict(end_s=3., by='other')])
        stated = [dict(event='term.spoken', event_id='a', state='final', end_ms=2000, committed_ms=2500, settled_ms=4500,
                       slot='s1:0', slot_score=.7),
                  dict(event='term.spoken', event_id='b', state='revoked', end_ms=4000, committed_ms=4300, settled_ms=7300,
                       slot='s1:1', slot_score=.4)]
        report = dict(start_ms=1000, events={}, event_states={'speech.term_events': stated}, complete=True)
        row = score_run(truth, report, finished_ms=1000 + 11000, duration_s=10.)
        self.assertEqual((row['final']['counted'], row['final']['right'], row['at_once']['counted'], row['at_once']['other']),
                         (1, 1, 2, 1))
        self.assertEqual((row['settle_ms']['p50'], row['provisional_ms']['p50'], row['real_time_factor'], row['revoked']),
                         (3000, 500, 1.1, 1))
        rows = words_of(truth, report)
        self.assertEqual([(item['truth'], item['provisional'], item['final']) for item in rows],
                         [('person', 'counted', 'counted'), ('other', 'counted', 'not counted')])
        self.assertIn('settle p50/p95', table([dict(clip='a', **row)]))

    def test_event_states_say_when_each_event_was_known_and_settled(self):
        rows = [record([dict(event='hit', event_id='a', state='provisional', end_ms=5)], 10, 'r1'),
                record([dict(event='event.revision', revises='a', state='revoked', reason='x')], 20, 'r2')]
        found = event_states(rows, {'r1': 100, 'r2': 300})
        self.assertEqual([(item['state'], item['committed_ms'], item['settled_ms'], item['settled_capture_ms'])
                          for item in found], [('revoked', 100, 300, 20)])


if __name__ == '__main__':
    unittest.main()


class TranscriptReplayTest(unittest.TestCase):
    def test_words_come_when_the_asr_had_them_and_only_the_terms_keep_their_text(self):
        from gamst.providers.transcript_replay import TranscriptReplay, export
        records = [dict(id='a', observation=dict(name='speech.transcript_words', value=dict(
            status='final', language='ko', words=[dict(text='오늘', start_ms=1100, end_ms=1300),
                                                  dict(text='뽀또가', start_ms=1300, end_ms=1600)])))]
        rows = export(records, {'a': 2500}, ['뽀또'], 1000)
        self.assertEqual([word['text'] for word in rows[0]['words']], ['_', '뽀또가'])
        self.assertEqual((rows[0]['start_s'], rows[0]['end_s'], rows[0]['known_s']), (.1, .6, 1.5))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'words.jsonl'
            path.write_text(json.dumps(rows[0], ensure_ascii=False) + '\n')
            replay = TranscriptReplay(path)
        audio = np.zeros(1600, np.float32)
        self.assertIsNone(replay.process(InputFrame(audio, 'audio', 50100, sample_rate=16000)))   # starts at 50000
        given = [(ms, replay.process(InputFrame(audio, 'audio', ms, sample_rate=16000))) for ms in range(50200, 51700, 100)]
        self.assertEqual([ms for ms, value in given if value], [51500])
        out = next(value for _, value in given if value)
        self.assertEqual((out[0]['start_ms'], out[0]['words'][1]['end_ms'], out[0]['status']), (50100, 50600, 'final'))
        self.assertIsNone(replay.finish())
