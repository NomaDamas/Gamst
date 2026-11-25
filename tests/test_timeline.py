"""Rows of capture time: statuses, completeness, revisions, knowledge time, cursors, clocks and bounds."""
import asyncio
import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from gamst import Gamst, SensorMetadata
from gamst.capabilities import CapabilitySpec, OutputField
from gamst.mcp import MCPServer, tool_definitions
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.timeline import MAX_AFFECTED, History, progress


def now():
    return time.time_ns() // 1_000_000


def spec(name, *, freshness=1000, mode='frame', window=0, lookahead=0):
    return CapabilitySpec(name, name, 'video', 'fake.' + name, '1',
                          (OutputField('value', 'number', required=False), OutputField('label', 'string', required=False),
                           OutputField('count', 'integer', required=False), OutputField('events', 'array', required=False),
                           OutputField('words', 'array', required=False), OutputField('window', 'object', required=False),
                           OutputField('utterance_id', 'string', required=False), OutputField('text', 'string', required=False),
                           OutputField('revision', 'integer', required=False), OutputField('status', 'string', required=False),
                           OutputField('start_ms', 'integer', required=False), OutputField('end_ms', 'integer', required=False),
                           OutputField('evidence_ids', 'array', required=False)),
                          mode, window_ms=window, lookahead_ms=lookahead, freshness_ms=freshness)


class Module:
    def __init__(self, name, delay=0., every=1, value=None, **options):
        self.spec, self.delay, self.every, self.seen = spec(name, **options), delay, every, 0
        self.value = value or (lambda frame: dict(value=float(frame.data)))

    async def process(self, frame):
        await asyncio.sleep(self.delay)
        self.seen += 1
        return self.value(frame) if (self.seen - 1) % self.every == 0 else None


class ReportingPipeline(Pipeline):
    """The completeness report of feat/realtime-pipeline, on this branch's Pipeline (which has none)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.gaps, self.extra, self.start = {}, {}, {}

    async def _run_module(self, name, entry, frame):
        self.start.setdefault(name, frame.timestamp_ms)
        return await super()._run_module(name, entry, frame)

    def status(self):
        status = super().status()
        for name, state in status['modules'].items():
            gaps = self.gaps.get(name, [])
            state.update(complete_until_ms=self.modules[name]['last_run_ms'], complete_from_ms=self.start.get(name),
                         gap_intervals=gaps, gaps=len(gaps), **self.extra.get(name, {}))
        return status


class Source:
    modality = 'video'

    def __init__(self, start, count, step):
        self.start, self.count, self.step = start, count, step

    async def frames(self):
        for index in range(self.count):
            yield InputFrame(index, 'video', self.start + index * self.step)


def grid(step=100, ago=4000):
    return (now() - ago) // step * step


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = Gamst(consent=True)
        self.sequence = 0

    async def asyncTearDown(self):
        await self.human.close()

    async def put(self, name, value, at, *, level=None, supersedes=None, source='cam', clock=None, epoch=None, **options):
        self.human.register_capability(spec(name, **options))
        self.sequence += 1
        return await self.human.push_observation(
            name, value, timestamp_ms=at, supersedes=supersedes,
            quality=None if level is None else dict(status='sufficient', level=level),
            sensor=SensorMetadata(source, 'sensor', self.sequence, clock_id=clock, clock_epoch=epoch))

    async def rows(self, start, end, step=100, **options):
        answer = await self.human.get_timeline(start_ms=start, end_ms=end, step_ms=step, **options)
        return answer, {row['t_ms'] - start: row for row in answer['rows']}

    def report(self, **modules):
        """A Pipeline that only reports: module name -> status fields."""
        class Fake:
            def __init__(self):
                self.modules = {}

            def status(self):
                return dict(modules={name: dict(dict(enabled=True, capability=name, emits=[]), **state)
                                     for name, state in modules.items()})
        self.human.pipeline = Fake()
        return modules

    def statuses(self, rows, name):
        return [rows[key]['cells'][name]['status'] for key in sorted(rows)]


class PipelineTimelineTest(Base):
    async def test_three_module_speeds_are_aligned_by_capture_time_and_pending_is_not_none(self):
        start = grid()
        pipeline = ReportingPipeline(self.human, 'p')
        pipeline.add_input('camera')
        for name, delay, every in (('fast', 0, 1), ('slow', .03, 1), ('rare', 0, 5)):
            module = Module(name, delay, every, freshness=0 if name == 'rare' else 150)
            pipeline.add_module(name, module, consumes='camera', spec=module.spec, queue=32)
        for index in range(10):
            await pipeline.push('camera', index, timestamp_ms=start + 100 * index, modality='video')
        await pipeline.push('camera', 10, timestamp_ms=start + 999, modality='video')
        await asyncio.sleep(.1)
        early, rows = await self.rows(start, start + 1000)
        self.assertEqual(self.statuses(rows, 'fast'), ['exact'] * 10)
        slow = self.statuses(rows, 'slow')
        self.assertEqual(slow[0], 'exact')
        self.assertEqual(slow[-1], 'pending')
        self.assertFalse(rows[900]['complete'])
        column = {item['key']: item for item in early['columns']}
        self.assertLess(column['slow']['complete_until_ms'], column['fast']['complete_until_ms'])
        self.assertEqual(early['complete_until_ms'], column['slow']['complete_until_ms'])
        await pipeline.drain()
        done, rows = await self.rows(start, start + 1000)
        self.assertEqual(self.statuses(rows, 'slow'), ['exact'] * 10)
        self.assertEqual(self.statuses(rows, 'rare'), (['exact'] + ['none'] * 4 + ['exact'] + ['none'] * 3 + ['exact']))
        self.assertTrue(all(row['complete'] for row in rows.values()))
        for offset, row in rows.items():      # the same moment in every column, whatever the delay
            values = {key: done['evidence'][cell['ref']]['value'].get('value') for key, cell in row['cells'].items()
                      if cell['status'] == 'exact' and cell.get('records', 1) == 1}
            self.assertLessEqual(len(set(values.values())), 1)
        # What was known at the first read is reproduced exactly, completeness included.
        replay, _ = await self.rows(start, start + 1000, as_of_ms=early['known_ms'])
        self.assertEqual(replay['rows'], early['rows'])
        self.assertEqual(replay['evidence'], early['evidence'])
        self.assertEqual(replay['complete_until_ms'], early['complete_until_ms'])
        settled, rows = await self.rows(start, start + 1000, settled_only=True, as_of_ms=early['known_ms'])
        self.assertTrue(rows and all(row['complete'] for row in rows.values()))
        self.assertLess(len(rows), 10)
        await pipeline.close()

    async def test_without_a_report_nothing_is_guessed(self):
        start = grid()
        pipeline = Pipeline(self.human, 'plain')
        pipeline.add_input('camera', Source(start, 3, 100))
        module = Module('fast', every=2, freshness=0)
        pipeline.add_module('fast', module, consumes='camera', spec=module.spec)
        pipeline.add_module('quiet', Module('quiet', value=lambda frame: None), consumes='camera', spec=spec('quiet'))
        await pipeline.run()
        answer, rows = await self.rows(start, start + 500)
        self.assertEqual(self.statuses(rows, 'fast'), ['exact', 'unknown', 'exact', 'unknown', 'unknown'])
        self.assertEqual(self.statuses(rows, 'quiet'), ['unknown'] * 5)
        self.assertIsNone(answer['complete_until_ms'])
        self.assertFalse(any(row['complete'] for row in rows.values()))
        await pipeline.close()

    async def test_off_paused_gap_and_sampled(self):
        start = grid()
        for index in (0, 1, 2, 8):
            await self.put('pose', dict(value=1.), start + 100 * index, freshness=350)
        for index in (0, 5):
            await self.put('vlm', dict(label='x'), start + 100 * index, freshness=1000)
        modules = self.report(
            pose=dict(complete_until_ms=start + 899, complete_from_ms=start, gaps=2, gap_intervals=[
                dict(start_ms=start + 400, end_ms=start + 450, frames=2, reason='overflow', source='camera'),
                dict(start_ms=start + 600, end_ms=start + 650, frames=2, reason='disabled', source='camera')]),
            vlm=dict(complete_until_ms=start + 899, complete_from_ms=start, every_ms=500, sampled=dict(every_ms=500)),
            stopped=dict(complete_until_ms=None))
        modules['stopped']['enabled'] = False
        answer, rows = await self.rows(start, start + 1000)
        self.assertEqual(self.statuses(rows, 'pose'), ['exact', 'exact', 'exact', 'held', 'gap', 'held', 'off', 'none',
                                                       'exact', 'pending'])
        self.assertEqual(rows[300]['cells']['pose']['age_ms'], 100)
        self.assertTrue(rows[500]['cells']['pose']['across_gap'])
        self.assertNotIn('ref', rows[400]['cells']['pose'])
        self.assertEqual(rows[900]['cells']['pose']['age_ms'], 100)
        self.assertEqual(self.statuses(rows, 'vlm'), ['exact'] + ['sampled'] * 4 + ['exact'] + ['sampled'] * 3 + ['pending'])
        self.assertEqual(self.statuses(rows, 'stopped'), ['off'] * 10)
        column = {item['key']: item for item in answer['columns']}
        self.assertEqual([gap['reason'] for gap in column['pose']['gaps']], ['overflow', 'disabled'])
        self.assertTrue(column['vlm']['sampled'] and column['stopped']['off'])
        self.assertEqual(answer['complete_until_ms'], start + 899)      # a module that is off does not hold it back

    async def test_rows_before_the_module_started_and_gaps_no_longer_listed_are_unknown(self):
        start = grid()
        await self.put('pose', dict(value=1.), start + 500, freshness=0)
        modules = self.report(pose=dict(complete_until_ms=start + 600))
        await self.rows(start, start + 1000)            # first seen here: from here on its account is known to hold
        modules['pose']['complete_until_ms'] = start + 999
        _, rows = await self.rows(start, start + 1000)
        self.assertEqual(self.statuses(rows, 'pose'), ['unknown'] * 5 + ['exact'] + ['none'] * 4)
        modules['pose'].update(complete_from_ms=start, gaps=70, gap_intervals=[
            dict(start_ms=start + 300, end_ms=start + 310, reason='overflow')])
        _, rows = await self.rows(start, start + 1000)
        self.assertEqual(self.statuses(rows, 'pose'), ['unknown'] * 3 + ['gap', 'none', 'exact'] + ['none'] * 4)

    async def test_gap_records_of_an_earlier_run_are_declared_gaps(self):
        start = grid()
        await self.put('pose', dict(value=1.), start, freshness=0)
        self.human.register_capability(spec('pipeline.gaps'))
        await self.put('pipeline.gaps', dict(events=[dict(
            event='pipeline.gap', start_ms=start + 200, end_ms=start + 250, reasons=['error', 'disabled'],
            modules=['pose'], capabilities=['pose'])]), start + 250)
        answer, rows = await self.rows(start, start + 400)
        self.assertEqual(self.statuses(rows, 'pose'), ['exact', 'unknown', 'gap', 'unknown'])
        self.assertEqual([item['capability'] for item in answer['columns']], ['pose'])
        self.assertEqual(rows[200]['cells']['pose']['reason'], 'error')

    async def test_declared_window_and_open_utterance_hold_back_completeness(self):
        start = grid()
        await self.put('tone', dict(window=dict(start_ms=start, end_ms=start + 300), label='calm'), start + 300,
                       mode='window', window=300)
        await self.put('asr', dict(utterance_id='u', revision=0, status='partial', text='he', start_ms=start + 200,
                                   end_ms=start + 400), start + 400)
        self.report(tone=dict(complete_until_ms=start + 999, complete_from_ms=start),
                    asr=dict(complete_until_ms=start + 999, complete_from_ms=start))
        answer, rows = await self.rows(start, start + 1000)
        column = {item['key']: item for item in answer['columns']}
        self.assertEqual(column['tone']['complete_until_ms'], start + 699)
        self.assertEqual(column['tone']['processed_until_ms'], start + 999)
        self.assertEqual(column['asr']['complete_until_ms'], start + 199)
        self.assertTrue(column['asr']['late_intervals'])
        self.assertEqual(answer['complete_until_ms'], start + 199)
        self.assertEqual([row['complete'] for row in rows.values()], [True, True] + [False] * 8)

    async def test_progress_adapter_accepts_what_is_there_and_nothing_else(self):
        self.assertEqual(progress(None), {})
        self.report(a=dict(), b=dict(complete_until_ms=50, holding_from_ms=20, gap_intervals=[[1, 2]], gaps=1))
        parts = progress(self.human.pipeline)
        self.assertFalse(parts['a'].reported)
        self.human.pipeline.status = lambda: 1 / 0      # a module's own status() failed: said, not guessed
        answer, _ = await self.rows(grid(), grid() + 100)
        self.assertEqual((answer['completeness_error'], answer['complete_until_ms']), ('ZeroDivisionError', None))
        self.assertEqual((parts['b'].until, parts['b'].gaps, parts['b'].gaps_since), (19, ((1, 2, 'lost'),), None))


class EvidenceTest(Base):
    async def test_intervals_appear_where_they_happened_not_when_they_arrived(self):
        start = grid()
        await self.put('asr', dict(utterance_id='u1', revision=0, status='partial', text='red', start_ms=start + 100,
                                   end_ms=start + 250), start + 300)
        await self.put('asr', dict(utterance_id='u1', revision=1, status='final', text='red car', start_ms=start + 100,
                                   end_ms=start + 500, words=[
                                       dict(text='red', start_ms=start + 100, end_ms=start + 200),
                                       dict(text='car', start_ms=start + 300, end_ms=start + 500)]), start + 1900)
        await self.put('match', dict(count=1, events=[dict(event='exemplar.matched', index=1, reference='m',
                                                          start_ms=start + 200, end_ms=start + 300)]), start + 300)
        await self.put('match', dict(count=1, events=[dict(event='exemplar.ended', index=1, reference='m',
                                                          start_ms=start + 200, end_ms=start + 600)]), start + 1600)
        answer, rows = await self.rows(start, start + 800)

        def texts(offset, name):
            return [answer['evidence'][ref]['value'].get('text', answer['evidence'][ref]['value'].get('event'))
                    for ref in rows[offset]['cells'][name].get('intervals', [])]

        self.assertEqual(texts(0, 'asr'), [])
        self.assertEqual(sorted(texts(100, 'asr')), ['red', 'red car'])
        self.assertEqual(texts(200, 'asr'), ['red car'])
        self.assertEqual(sorted(texts(300, 'asr')), ['car', 'red car'])
        self.assertEqual(rows[500]['cells']['asr']['status'], 'unknown')
        self.assertEqual([texts(offset, 'match') for offset in (100, 200, 500, 600)],
                         [[], ['exemplar.ended'], ['exemplar.ended'], []])
        utterance = next(entry for entry in answer['evidence'].values() if entry['value'].get('text') == 'red car')
        self.assertEqual((utterance['start_ms'], utterance['end_ms']), (start + 100, start + 500))
        self.assertGreater(utterance['known_ms'], utterance['end_ms'])
        self.assertNotIn('words', utterance['value'])

    async def test_out_of_order_and_several_records_in_one_step(self):
        start = grid()
        for offset, value in ((250, 3.), (0, 1.), (260, 4.), (100, 2.)):
            await self.put('pose', dict(value=value, events=[dict(event='tick')] if value == 3 else []), start + offset)
        answer, rows = await self.rows(start, start + 400)
        cells = [rows[key]['cells']['pose'] for key in sorted(rows)]
        self.assertEqual([cell['status'] for cell in cells], ['exact', 'exact', 'exact', 'held'])
        self.assertEqual([answer['evidence'][cell['ref']]['value']['value'] for cell in cells], [1., 2., 4., 4.])
        self.assertEqual(cells[2]['records'], 2)
        self.assertEqual(cells[3]['age_ms'], 40)
        # The event of the record that is not shown is not lost: it is an interval of its own.
        self.assertEqual(answer['evidence'][cells[2]['intervals'][0]]['value'], dict(event='tick'))

    async def test_bounds_truncation_and_private_fields(self):
        start = grid()
        big = dict(value=1., label='x' * 3000, events=[dict(event='e', start_ms=start, end_ms=start + 100, n=n)
                                                       for n in range(20)])
        await self.put('big', big, start)
        await self.human.push_signal('scene.poses', [dict(track_id=1, bbox=[0, 0, 1, 1], landmarks=[[.1, .2]] * 33)],
                                     'video', timestamp_ms=start)
        await self.human.push_signal('custom.points', [[.1, .2]] * 40, 'video', timestamp_ms=start)
        await self.put('nested', dict(window=dict(start_ms=start, end_ms=start + 50, inner=dict(a=[1] * 50))), start)
        answer, rows = await self.rows(start, start + 200)
        text = json.dumps(answer)
        for word in ('landmarks"', 'custom.points', 'keypoints', 'embedding'):
            self.assertNotIn(word, text)
        cell = rows[0]['cells']['big']
        self.assertEqual((len(cell['intervals']), cell['intervals_omitted']), (8, 12))
        entry = answer['evidence'][cell['ref']]
        self.assertEqual(entry['flags'], ['truncated'])
        self.assertEqual(entry['items_as_intervals'], 20)
        self.assertLessEqual(len(json.dumps(entry['value'])), 2048)
        for index in range(60):
            await self.put('pose', dict(value=float(index), label='y' * 300), start + 20 * index, freshness=0)
        seen, cursor, pages = [], None, 0
        while True:
            answer = await self.human.get_timeline(start_ms=start, end_ms=start + 1200, step_ms=20, max_bytes=8192,
                                                   capabilities=['pose'], cursor=cursor)
            self.assertLessEqual(len(json.dumps(answer)), 8192)
            seen += [row['t_ms'] for row in answer['rows']]
            pages, cursor = pages + 1, answer['next_cursor']
            if not answer['has_more']:
                break
        self.assertEqual(seen, [start + 20 * index for index in range(60)])
        self.assertGreater(pages, 2)
        with self.assertRaisesRegex(ValueError, 'max_bytes'):
            for index in range(30):
                await self.put(f'cap{index:02}', dict(value=1.), start)
            await self.rows(start, start + 100, max_bytes=4096)
        with self.assertRaisesRegex(ValueError, 'at most 601'):
            await self.rows(start - 100_000, start)

    async def test_retention_is_not_read_as_nothing_happened(self):
        await self.human.close()
        self.human = Gamst(consent=True, max_records=6, sensor_inference_interval_ms=5000)
        start = grid()
        for index in range(10):
            await self.put('pose', dict(value=float(index)), start + 100 * index, freshness=0)
        self.report(pose=dict(complete_until_ms=start + 999, complete_from_ms=start))
        answer, rows = await self.rows(start, start + 1000)
        lost = [key for key in sorted(rows) if rows[key]['cells']['pose']['status'] == 'unknown']
        self.assertTrue(lost)
        self.assertEqual(answer['removed_until_ms'], start + lost[-1])
        self.assertFalse(any(rows[key]['complete'] for key in lost))

    async def test_uncalibrated_clock_is_kept_out_and_calibration_places_it(self):
        start, shift = grid(), -7_200_000           # the remote clock is two hours behind
        await self.put('near', dict(value=1.), start)
        await self.put('far', dict(value=2., events=[dict(event='e', start_ms=start + shift, end_ms=start + shift + 100)]),
                       start + shift + 200, source='glasses', clock='glasses', epoch='boot1')
        answer, rows = await self.rows(start, start + 400)
        self.assertEqual(set(self.statuses(rows, 'far')), {'unknown'})
        self.assertEqual(rows[0]['cells']['far']['reason'], 'uncalibrated_clock')
        self.assertFalse(rows[0]['complete'])
        self.assertEqual(answer['uncalibrated_clocks'][0]['clock'], 'glasses@boot1')
        before = answer['known_ms']
        for index in range(8):
            self.human.add_clock_sample('glasses', start + shift + 100 * index, start + 100 * index, clock_epoch='boot1')
        answer, rows = await self.rows(start, start + 400)
        self.assertEqual(answer['uncalibrated_clocks'], [])
        self.assertEqual(self.statuses(rows, 'far'), ['exact', 'unknown', 'exact', 'held'])
        entry = answer['evidence'][rows[200]['cells']['far']['ref']]
        self.assertEqual((entry['at_ms'], entry['source_time_ms']), (start + 200, start + shift + 200))
        _, rows = await self.rows(start, start + 400, as_of_ms=before)      # it was not calibrated then
        self.assertEqual(rows[200]['cells']['far']['reason'], 'uncalibrated_clock')
        # A restarted clock does not inherit the calibration; a worker's operation clock is not a source clock.
        await self.put('far', dict(value=3.), start + shift, source='glasses', clock='glasses', epoch='boot2')
        await self.put('local', dict(value=3.), start, clock='provider:pose')
        answer, rows = await self.rows(start, start + 400)
        self.assertEqual([item['clock'] for item in answer['uncalibrated_clocks']], ['glasses@boot2'])
        self.assertEqual(rows[0]['cells']['local']['status'], 'exact')

    async def test_sensor_stream_worker_keeps_the_clock_of_its_source(self):
        from types import SimpleNamespace
        from gamst.streams import SensorStreams
        lane = dict(spec=spec('s'), provider=SimpleNamespace(clock_id='provider:s', clock_epoch='e'), output_sequence=0)
        self.human.register_capability(lane['spec'])
        streams, clocks = SensorStreams(self.human), []
        for sensor in (SensorMetadata('a', 'b', 1), SensorMetadata('a', 'b', 2, clock_id='phone')):
            await streams._publish(lane, dict(value=1.), InputFrame(0, 'video', now(), sensor=sensor), 1)
        for record in self.human.store.query(name='s'):
            clocks.append((record['observation']['sensor']['clock_id'], record['observation']['sensor']['clock_epoch']))
        self.assertEqual(clocks, [('provider:s', 'e'), ('phone', None)])


class RevisionTest(Base):
    async def count(self, **options):
        return await self.human.count_events('match', **options)

    async def match(self, at, start, *, value=True, **options):
        events = [dict(event='matched', start_ms=start, end_ms=start + 100)] if value else []
        return await self.put('match', dict(count=len(events), events=events), at, freshness=0, **options)

    async def mark(self):
        await asyncio.sleep(.003)
        answer = await self.human.get_timeline(window='1s', capabilities=['match'])
        return answer['known_ms']

    async def test_provisional_replaced_by_final_agree_disagree_retracted(self):
        start = grid()
        one = await self.match(start + 100, start + 100, level='provisional')
        two = await self.match(start + 400, start + 400, level='provisional')
        three = await self.match(start + 700, start + 700, level='provisional')
        first, rows = await self.rows(start, start + 1000)
        result = await self.count(max_evidence=0)
        self.assertEqual((result['count'], result['final'], result['provisional']), (3, 0, 3))
        self.assertEqual({entry.get('level') for entry in first['evidence'].values()}, {'provisional'})
        agree = await self.match(start + 100, start + 100)
        disagree = await self.match(start + 400, 0, value=False)
        retraction = await self.human.retract_observation(three['id'], 'full processing found nothing')
        for record, root in ((agree, one), (disagree, two), (retraction, three)):
            self.assertEqual((record['supersedes'], record['revision']), (root['id'], 1))
        self.assertEqual(self.human.store.get_many([one['id']]), [one])     # nothing stored was changed
        result = await self.count()
        self.assertEqual((result['count'], result['final'], result['provisional']), (1, 1, 0))
        self.assertEqual((result['evidence'][0]['record_id'], result['evidence'][0]['level']), (agree['id'], 'final'))
        before = await self.count(as_of_ms=first['known_ms'])
        self.assertEqual((before['count'], before['provisional'], before['final']), (3, 3, 0))
        delta = await self.human.get_timeline(start_ms=start, end_ms=start + 1000, step_ms=100, cursor=first['next_cursor'])
        changed = {row['t_ms'] - start: row for row in delta['rows']}
        self.assertEqual({key: row['change'] for key, row in changed.items()}, {100: 'revised', 400: 'revised', 700: 'retracted'})
        agreed = changed[100]
        old, new = (delta['evidence'][agreed[side]['match']['intervals'][0]] for side in ('was', 'cells'))
        self.assertEqual((old['level'], old['current'], new.get('level'), old['value']), ('provisional', False, None, new['value']))
        self.assertEqual((new['supersedes'], new['revision']), (one['id'], 1))
        self.assertEqual(changed[400]['was']['match']['status'], 'exact')
        self.assertNotIn('intervals', changed[400]['cells']['match'])
        self.assertEqual(changed[700]['cells']['match']['status'], 'unknown')
        self.assertEqual({(item['change'], item['start_ms'] - start) for item in delta['affected']['intervals']},
                         {('revised', 100), ('revised', 400), ('retracted', 700)})
        replay, _ = await self.rows(start, start + 1000, as_of_ms=first['known_ms'])
        self.assertEqual((replay['rows'], replay['evidence']), (first['rows'], first['evidence']))
        again = await self.human.get_timeline(start_ms=start, end_ms=start + 1000, step_ms=100, cursor=delta['next_cursor'])
        self.assertEqual((again['rows'], again['affected']['intervals']), ([], []))

    async def test_final_first_late_provisional_and_moved_events_are_not_counted_twice(self):
        start = grid()
        await self.match(start + 100, start + 100)
        await self.match(start + 100, start + 100, level='provisional')     # learned later, still worth less
        await self.match(start + 300, start + 300, level='provisional')
        await self.match(start + 340, start + 300)                          # same event, recognized at another time
        result = await self.count()
        self.assertEqual((result['count'], result['final'], result['provisional']), (2, 2, 0))
        answer, rows = await self.rows(start, start + 600)
        self.assertEqual([len(rows[key]['cells']['match'].get('intervals', [])) for key in (100, 300)], [1, 1])

    async def test_late_match_between_two_earlier_ones_as_of_and_affected(self):
        start = grid()
        first, third = await self.match(start + 100, start + 100), await self.match(start + 700, start + 700)
        before, _ = await self.rows(start, start + 1000)
        middle = await self.match(start + 450, start + 400)
        self.assertIn('out_of_order', middle['quality']['flags'])
        self.assertEqual([(await self.count())['count'], (await self.count(as_of_ms=before['known_ms']))['count']], [3, 2])
        order = [item['record_id'] for item in (await self.count())['evidence']]
        self.assertEqual(order, [first['id'], middle['id'], third['id']])
        delta, rows = await self.rows(start, start + 1000, cursor=before['next_cursor'])
        self.assertEqual(sorted(rows), [400])
        self.assertEqual((rows[400]['change'], rows[400]['was']['match']['status']), ('revised', 'unknown'))
        self.assertGreater(rows[400]['known_ms'], before['known_ms'])
        self.assertEqual([(item['change'], item['start_ms'] - start, item['end_ms'] - start)
                          for item in delta['affected']['intervals']], [('added', 400, 500)])

    async def test_corrections_chain_and_dependencies_show_that_they_changed(self):
        start = grid()
        pose = await self.put('pose', dict(value=1.), start)
        await self.put('derived', dict(value=2., evidence_ids=[pose['id']]), start)
        await self.put('match', dict(events=[dict(event='m', evidence_ids=[pose['id']])]), start)
        before, _ = await self.rows(start, start + 100)
        second = await self.put('pose', dict(value=1.5), start, supersedes=pose['id'])
        third = await self.put('pose', dict(value=1.6), start, supersedes=second['id'])
        self.assertEqual((third['supersedes'], third['revision']), (pose['id'], 2))
        delta, rows = await self.rows(start, start + 100, cursor=before['next_cursor'])
        cell = rows[0]['cells']['pose']
        self.assertEqual((delta['evidence'][cell['ref']]['value']['value'], cell.get('records')), (1.6, None))
        entry = delta['evidence'][rows[0]['cells']['derived']['ref']]
        self.assertEqual((entry['depends_on'], entry['depends_on_changed']), ([pose['id']], [pose['id']]))
        self.assertEqual((await self.count())['evidence'][0]['depends_on'], [pose['id']])
        # One event with three revisions for the temporal packet, and one value in get_context.
        packet = await self.human.build_temporal_context()
        poses = [event for event in packet['events'] if event['event_type'] == 'pose']
        self.assertEqual([(event['event_id'], event['revision'], event['value']['value']) for event in poses],
                         [(pose['id'], 2, 1.6)])
        context = await self.human.get_context()
        self.assertEqual([item['value'] for item in context['observations'].values() if item.get('name') == 'pose'],
                         [dict(value=1.6)])
        for bad in (dict(supersedes='nope'), dict(supersedes=pose['id'], name='derived')):
            with self.assertRaises(ValueError):
                await self.put(bad.get('name', 'pose'), dict(value=1.), start, supersedes=bad['supersedes'])
        with self.assertRaises(ValueError):
            await self.put('pose', dict(value=1.), start, level='guess')
        with self.assertRaises(ValueError):
            await self.human.retract_observation('nope')
        await self.human.retract_observation(pose['id'])
        context = await self.human.get_context()
        self.assertFalse([item for item in context['observations'].values() if item.get('name') == 'pose'])
        self.assertFalse([event for event in (await self.human.build_temporal_context())['events']
                          if event['event_type'] == 'pose'])

    async def test_cursor_moves_with_the_window_and_old_rows_keep_changing(self):
        start = grid(ago=3000)
        await self.match(start, start)
        answer = await self.human.get_timeline(window='2s', step_ms=100, capabilities=['match'])
        cursor, first = answer['next_cursor'], answer['start_ms']
        await asyncio.sleep(.25)
        late = start + 1300 if start + 1300 >= first else first
        await self.match(late, late)
        delta = await self.human.get_timeline(window='200ms', step_ms=100, capabilities=['match'], cursor=cursor)
        self.assertEqual(delta['start_ms'], first)          # rows the client holds stay covered
        changes = {row['t_ms']: row['change'] for row in delta['rows']}
        self.assertEqual(changes[late], 'revised')
        self.assertTrue(all(change == 'added' for at, change in changes.items() if at >= answer['end_ms']))
        self.assertIn(answer['end_ms'], changes)
        self.assertNotIn('not_covered', delta)
        with self.assertRaisesRegex(ValueError, 'different timeline query'):
            await self.human.get_timeline(window='2s', step_ms=200, capabilities=['match'], cursor=cursor)
        with self.assertRaisesRegex(ValueError, 'invalid or expired'):
            await self.human.get_timeline(window='2s', step_ms=100, capabilities=['match'], cursor=cursor[:-4] + 'AAAA')

    async def test_rows_past_the_row_limit_are_named_and_still_reported_as_affected(self):
        start = grid(ago=14_000)
        await self.match(start, start)
        answer = await self.human.get_timeline(start_ms=start, end_ms=start + 1000, step_ms=20, capabilities=['match'])
        while answer['has_more']:
            answer = await self.human.get_timeline(start_ms=start, end_ms=start + 1000, step_ms=20,
                                                   capabilities=['match'], cursor=answer['next_cursor'])
        await self.match(start + 100, start + 100)
        delta = await self.human.get_timeline(window='1s', step_ms=20, capabilities=['match'], cursor=answer['next_cursor'])
        self.assertEqual(delta['not_covered'], [dict(start_ms=start, end_ms=start + 1000)])
        self.assertEqual([(item['change'], item['start_ms']) for item in delta['affected']['intervals']],
                         [('added', start + 100)])

    async def test_expired_cursor_says_restart(self):
        await self.human.close()
        self.human = Gamst(consent=True, max_records=12, sensor_inference_interval_ms=5000)
        start = grid()
        await self.match(start, start)
        answer, _ = await self.rows(start, start + 1000)
        for index in range(30):
            await self.match(start + 20 * index, start + 20 * index)
        delta, _ = await self.rows(start, start + 1000, cursor=answer['next_cursor'])
        self.assertEqual((delta['restart'], delta['reason'], delta['rows'], delta['next_cursor']),
                         (True, 'retention', [], None))
        answer, _ = await self.rows(start, start + 1000)
        await self.human.delete_data('local-user')
        with self.assertRaisesRegex(ValueError, 'restart without cursor'):
            await self.rows(start, start + 1000, cursor=answer['next_cursor'])
        answer, _ = await self.rows(start, start + 1000)
        self.human._timeline_history = History()
        delta, _ = await self.rows(start, start + 1000, cursor=answer['next_cursor'])
        self.assertEqual((delta['restart'], delta['reason']), (True, 'completeness_history'))

    async def test_as_of_before_the_kept_history_says_so_and_future_is_refused(self):
        start = grid()
        await self.match(start, start)
        answer, rows = await self.rows(start, start + 100, as_of_ms=start + 50)
        self.assertEqual((answer['completeness_history'], answer['evidence']), ('unavailable', {}))
        for value in (now() + 60_000, -1, 1.5):
            with self.assertRaises(ValueError):
                await self.human.get_timeline(as_of_ms=value)
            with self.assertRaises(ValueError):
                await self.count(as_of_ms=value)

    async def test_affected_list_is_bounded(self):
        start = grid(ago=9000)
        answer, _ = await self.rows(start, start + 9000)
        for index in range(MAX_AFFECTED + 6):
            await self.put(f'cap{index % 20:02}', dict(value=1.), start + 120 * index, freshness=0)
        delta, _ = await self.rows(start, start + 9000, cursor=answer['next_cursor'], max_bytes=262144)
        self.assertEqual((len(delta['affected']['intervals']), delta['affected']['omitted']), (MAX_AFFECTED, 6))

    async def test_files_written_before_knowledge_time_was_kept_open_and_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'old.db')
            async with Gamst(consent=True, store=path) as human:
                human.register_capability(spec('pose'))
                start = grid()
                await human.push_observation('pose', dict(value=1.), timestamp_ms=start, sensor=SensorMetadata('a', 'b', 1))
            connection = sqlite3.connect(path)
            connection.execute('DROP INDEX known_time')
            connection.execute('ALTER TABLE record_offsets DROP COLUMN known_ms')
            connection.commit()
            connection.close()
            async with Gamst(consent=True, store=path) as human:
                answer = await human.get_timeline(start_ms=start, end_ms=start + 100, step_ms=100)
                self.assertEqual(answer['rows'][0]['cells']['pose']['status'], 'exact')
                self.assertIsNone(next(iter(answer['evidence'].values()))['known_ms'])


class MCPTimelineTest(Base):
    async def test_schema_validation_and_round_trip(self):
        server = MCPServer(self.human)
        start = grid()
        await self.put('pose', dict(value=1.), start)
        tool = next(tool for tool in tool_definitions() if tool['name'] == 'get_timeline')
        self.assertFalse(tool['inputSchema']['additionalProperties'])
        reply = await server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {
            'name': 'get_timeline', 'arguments': {'start_ms': start, 'end_ms': start + 500, 'step_ms': 100,
                                                  'capabilities': ['pose']}}})
        answer = json.loads(reply['result']['content'][0]['text'])
        self.assertEqual(answer['rows'][0]['cells']['pose']['status'], 'exact')
        again = await server.call_tool('get_timeline', {'start_ms': start, 'end_ms': start + 500, 'step_ms': 100,
                                                        'capabilities': ['pose'], 'cursor': answer['next_cursor']})
        self.assertEqual(again['rows'], [])
        self.assertEqual((await server.call_tool('get_timeline', {}))['step_ms'], 200)
        self.assertEqual((await server.call_tool('count_events', {'capability_id': 'pose', 'as_of_ms': answer['known_ms']}))['count'], 0)
        for arguments in ({'step_ms': 10}, {'step_ms': 1.5}, {'capabilities': []}, {'capabilities': 'pose'},
                          {'capabilities': ['a b']}, {'nope': 1}, {'start_ms': 5}, {'start_ms': 9, 'end_ms': 5},
                          {'window': '10s', 'start_ms': 1, 'end_ms': 2}, {'max_bytes': 10}, {'settled_only': 1},
                          {'cursor': 'x'}, {'cursor': 'x' * 3000}, {'window': '0s'}, {'window': '10m'},
                          {'as_of_ms': now(), 'cursor': answer['next_cursor']}, {'subject_id': ''}):
            with self.assertRaises(ValueError, msg=str(arguments)):
                await server.call_tool('get_timeline', arguments)

    async def test_reading_changes_nothing(self):
        start = grid()
        await self.put('pose', dict(value=1.), start)
        records = self.human.store.query()
        context = await self.human.get_context()
        with mock.patch.object(self.human, '_emit', side_effect=AssertionError('a read committed a record')):
            await self.rows(start, start + 500)
        self.assertEqual(self.human.store.query(), records)
        after = await self.human.get_context()
        self.assertEqual(set(after), set(context))
        self.assertEqual([set(item) for item in after['observations'].values()],
                         [set(item) for item in context['observations'].values()])


if __name__ == '__main__':
    unittest.main()
