"""Per-record aggregation equals the one computed from the records and stays cheap; a long read does not hold up capture."""
import asyncio
import json
import os
import random
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from gamst import Gamst, SensorMetadata
from gamst.capabilities import CapabilitySpec, OutputField
from gamst.context_memory import ContextMemory
from gamst.schema import envelope
from gamst.mcp import MCPServer
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.storage import Store


def now():
    return time.time_ns() // 1_000_000


def signal(rng, ts, subject, value=None, **extra):
    capability = rng.choice([None, 'a', 'b'])
    observation = dict(name=rng.choice(['x', 'y', 'é', 'statement.claim', 'z.MODEL_SCORE']), modality=rng.choice(['audio', 'video']),
                       value=rng.choice([rng.random() * 10 ** rng.randrange(-6, 9), rng.randrange(9), True, 'text', None,
                                         dict(value=1)]) if value is None else value)
    if capability is not None:
        observation.update(capability=dict(id=capability), sensor=dict(
            source_id=rng.choice(['s1', 's2']), sensor_id=rng.choice(['c1', None]), sequence=1,
            clock_id=rng.choice([None, None, 'provider:a', 'phone']), clock_epoch=rng.choice([None, 'e'])))
    record = envelope(rng.choice(['observation.signal'] * 5 + ['observation.event', 'human.inference']), subject, ts,
                      observation=observation, models=[dict(id=rng.choice('mn'), version=rng.choice(['1', '2']))],
                      quality=dict(status=rng.choice(['sufficient'] * 4 + ['insufficient']), flags=[]))
    record.update(extra)
    if record['type'] == 'observation.event':
        record['event'] = rng.choice(['e1', 'e2'])
    return record


class EqualityTest(unittest.TestCase):
    def check(self, store, rng, base):
        for subject in ('u', 'v'):
            start, end = sorted(base + rng.randrange(-50, 700) for _ in range(2))
            self.assertEqual(store.latest_signals(subject, start, end), store._latest_signals_scan(subject, start, end))
            self.assertEqual(repr(store.numeric_summary(subject, start, end)),     # bit for bit, and int or float
                             repr(store._numeric_summary_scan(subject, start, end)))
            known = rng.randrange(store._opened_ms - 5, store.known_now() + 5)
            self.assertEqual(store.latest_signals(subject, start, end, known_until=known),
                             store._latest_signals_scan(subject, start, end, known_until=known))
            clocks = {}
            for record in store.query(subject, None, None, 'observation.signal', limit=100_000):
                sensor = record['observation'].get('sensor') or {}
                clock = sensor.get('clock_id')
                if clock and not clock.startswith('provider:'):
                    clocks[clock, sensor['clock_epoch']] = clocks.get((clock, sensor['clock_epoch']), 0) + 1
            found = {}
            for item in store.source_clocks(subject):
                key = item['clock_id'], item['clock_epoch']
                found[key] = found.get(key, 0) + item['records']
            self.assertEqual(found, clocks)

    def test_random_sequences_with_eviction_corrections_and_times_out_of_order(self):
        for seed in range(25):
            rng = random.Random(seed)
            base = now() - 5000
            store = Store(max_records=rng.choice([7, 40, 5000]), retention_hours=1)
            ids = []
            for step in range(250):
                ts = base + rng.choice([step * 2, rng.randrange(600), step // 25 * 50])     # many share a time
                record = signal(rng, ts, rng.choice('uuv'))
                roll = rng.random()
                if ids and roll < .2:            # a correction or a retraction, of a record that may be gone already
                    target, subject, revision = rng.choice(ids)
                    record.update(subject_id=subject, supersedes=target, revision=revision + rng.randrange(1, 4))
                    if roll < .05:
                        record['type'] = 'observation.retraction'
                store.add(record)
                ids.append((record.get('supersedes') or record['id'], record['subject_id'], record.get('revision', 0)))
                if rng.random() < .01:
                    store.delete(rng.choice([None, 'v']))
                if rng.random() < .02:
                    store.prune(base + rng.randrange(300) + store.retention_ms)
                if rng.random() < .02:           # a failed write rolls the index back with the records
                    with self.assertRaises(Exception):
                        store.add(dict(record, id=record['id']))
                        store.add(dict(record, id=record['id']))
                if step % 5 == 0:
                    self.check(store, rng, base)
            self.check(store, rng, base)
            live = store.connection.execute('SELECT (SELECT COUNT(*) FROM live_signals),'
                                            "(SELECT COUNT(*) FROM records WHERE kind='observation.signal'),"
                                            '(SELECT COUNT(*) FROM live_series WHERE id NOT IN (SELECT series FROM live_signals)),'
                                            '(SELECT COUNT(*) FROM live_metrics WHERE id NOT IN (SELECT metric FROM live_numbers))').fetchone()
            self.assertEqual((live[0], live[2], live[3]), (live[1], 0, 0))
            store.close()

    def test_another_connection_writing_the_file_is_seen(self):
        rng = random.Random(1)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 's.db')
            first, second = Store(path), Store(path)
            base = now() - 5000
            for step in range(60):
                rng.choice([first, second]).add(signal(rng, base + step, 'u'))
                if step % 4 == 0:
                    self.check(first, rng, base)
                    self.check(second, rng, base)
            second.delete('u')
            self.assertEqual(first.latest_signals('u', 0, now()), [])
            self.check(first, rng, base)
            names = {row[0] for row in first.connection.execute('SELECT name FROM main.sqlite_master')}
            self.assertFalse({name for name in names if name.startswith('live_')})      # nothing of it is in the file
            first.close()
            second.close()

    def test_context_memory_counts_what_it_holds(self):
        for seed in range(15):
            rng = random.Random(seed)
            memory = ContextMemory(max_recent=rng.choice([1, 8]), max_episodic=rng.choice([1, 16]),
                                   max_current=rng.choice([1, 9]), max_bytes=rng.choice([600, 2000, 100_000]))
            for step in range(120):
                memory.append(dict(event_id=f'e{rng.randrange(60)}', stored_at_ms=1, value='x' * rng.randrange(120)),
                              entity_id=rng.choice([None, f'é{rng.randrange(6)}']), episodic=rng.random() < .4)
                if rng.random() < .05:
                    other = ContextMemory()
                    other.append(dict(event_id=f'd{rng.randrange(9)}', stored_at_ms=1), entity_id='q')
                    memory.replay(other.delta(None))
                body = json.dumps(memory.snapshot(), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(memory._snapshot_bytes(), len(body))
                retained = {item['evidence_id'] for item in memory._retained() if item.get('evidence_id')}
                memory._order_evidence()
                self.assertEqual(set(memory._evidence), retained)


class PublishedTest(unittest.IsolatedAsyncioTestCase):
    async def run_sequence(self, seed, scan, fixed):
        rng = random.Random(seed)
        base, published, ids = fixed - 20_000, [], {}

        def plain(value):
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, list):
                return [plain(item) for item in value]
            return ids.setdefault(value, f'id{len(ids)}') if isinstance(value, str) and len(value) == 36 else value

        patches = [mock.patch.object(Store, 'latest_signals', Store._latest_signals_scan),
                   mock.patch.object(Store, 'numeric_summary', Store._numeric_summary_scan)] if scan else []
        for patch in patches:
            patch.start()
        try:
            human = Gamst(consent=True, max_records=rng.choice([60, 5000]), window_ms=rng.choice([400, 30_000]))
            for name in 'ab':
                human.register_capability(CapabilitySpec(name, name, 'video', 'f', '1', (OutputField('value', 'number'),), 'frame'))
            stored = []
            for step in range(150):
                at = base + step * 20 - rng.choice([0, 0, 0, rng.randrange(400)])
                kind = rng.random()
                if kind < .5:
                    name, target = rng.choice('ab'), rng.choice([item[0] for item in stored[-20:]] or [None])
                    target = target if rng.random() < .2 and human.store.get_many([target]) else None
                    name = dict(stored).get(target, name)       # a correction is of the same capability
                    record = await human.push_observation(name, dict(value=rng.random()), timestamp_ms=at, supersedes=target,
                                                          sensor=SensorMetadata('s', 'c', step))
                    stored.append((record['id'], name))
                    if rng.random() < .1:
                        await human.retract_observation(record['id'])
                elif kind < .85:
                    record = await human.push_signal(rng.choice(['rms', 'pitch']), rng.choice([rng.random(), rng.randrange(5)]),
                                                     'audio', timestamp_ms=at, model_id='m', model_version='1')
                elif kind < .95:
                    record = await human.push_signal('speech.activity', rng.random() < .5, 'audio', timestamp_ms=at)
                else:
                    record = await human.push_text(rng.choice(['hello', 'explain again']), timestamp_ms=at)
                if step % 10 == 0:
                    published.append(await human.get_trends(window='1s', end_ms=at))
                    context = await human.get_context()
                    published.append({key: context[key] for key in ('summary', 'trends')})
                    published.append(sorted(json.dumps(plain({k: v for k, v in item.items() if k != 'age_ms'}), sort_keys=True)
                                            for item in context['observations'].values()))
            published += [plain(record) for record in human.store.query(limit=100_000)]
            published.append(plain([{k: v for k, v in item.items() if k != 'stored_at_ms'}
                                    for item in human.context_memory.snapshot()['recent_timeline']]))
            await human.close()
            return json.loads(json.dumps(published).replace(human._session_id, 'session'))
        finally:
            for patch in patches:
                patch.stop()

    async def test_every_published_record_is_the_same_as_before(self):
        for seed in range(6):
            fixed = now()
            with mock.patch('gamst.engine.timestamp', lambda value=None: fixed if value is None else int(value)):
                expected, found = await self.run_sequence(seed, True, fixed), await self.run_sequence(seed, False, fixed)
            self.assertGreater(sum(item.get('type') == 'human.inference' for item in expected if isinstance(item, dict)), 20)
            self.assertEqual(found, expected)

    async def test_a_record_does_not_cost_more_as_records_accumulate(self):
        human = Gamst(consent=True, max_records=20_000)
        human.register_capability(CapabilitySpec('a', 'a', 'video', 'f', '1', (OutputField('boxes', 'array'),), 'frame'))
        steps, base = [], now() - 30_000

        async def push(count):
            for _ in range(count):
                await human.push_observation('a', dict(boxes=[dict(label='x', bbox=[.1, .2, .3, .4])] * 4),
                                             timestamp_ms=base + len(steps) * 15, sensor=SensorMetadata('s', 'c', len(steps)))
                steps.append(0)

        async def work():       # SQLite's own count of steps: the same on a busy machine
            count = [0]

            def step():
                count[0] += 1
            human.store.connection.set_progress_handler(step, 100)
            await push(20)
            human.store.connection.set_progress_handler(None, 0)
            return count[0]

        await push(50)
        early = await work()
        await push(1200)
        late = await work()
        self.assertLess(late, early * 1.5 + 20)
        await human.close()


def fill(store, count, start, span_ms):
    """``count`` registered records of 5 capabilities over ``span_ms``."""
    for index in range(count):
        ts = start + index * span_ms // count
        store.add(envelope('observation.signal', 'local-user', ts, observation=dict(
            name='abcde'[index % 5], modality='video', capability=dict(
                id='abcde'[index % 5], streaming_mode='frame', window_ms=0, lookahead_ms=0, freshness_ms=1000,
                interpretation='estimate'), sensor=dict(source_id='cam', sensor_id='0', sequence=index),
            value=dict(value=index * .5, boxes=[dict(label='x', bbox=[.1, .2, .3, .4])] * 3,
                       events=[dict(event='e', start_ms=ts - 200, end_ms=ts)] if index % 50 == 0 else []))))


class ReadTest(unittest.IsolatedAsyncioTestCase):
    def test_scans_return_what_a_query_returns(self):
        for seed in range(6):
            rng = random.Random(seed)
            base, store = now() - 5000, Store()
            for step in range(300):
                store.add(signal(rng, base + rng.randrange(120), rng.choice('uuv')))
            known = store.read_mark()
            for _ in range(12):
                start, end = sorted(base + rng.randrange(-5, 130) for _ in range(2))
                about = rng.choice([{}, dict(any_of=['a', 'x', 'm']), dict(clock_id='phone'), dict(spans_only=True)])
                found = [json.loads(payload) for rows in store.scan('u', start, end, known_until=known, step=rng.choice([1, 7, 200]),
                                                                    **about) for payload, _ in rows]
                self.assertEqual(found[::-1], store.query('u', start, end, 'observation.signal', limit=9999, known_until=known, **about))
            learned = [json.loads(payload) for rows in store.scan_learned('u', 0, known, ('observation.signal',), step=9) for payload in rows]
            self.assertEqual({record['id'] for record in learned},
                             {record['id'] for record in store.query('u', kind='observation.signal', limit=9999)})
            self.assertEqual(list(store.scan_learned('u', known, known + 5, ('observation.signal',))), [])

    async def test_pipeline_keeps_its_cadence_during_a_read_of_10k_records(self):
        class Live:
            spec = CapabilitySpec('live', 'live', 'video', 'f', '1', (OutputField('value', 'number'),), 'frame')

            async def process(self, frame):
                await asyncio.sleep(.001)
                return dict(value=float(frame.data))

        class Camera:
            modality, live, stop = 'video', True, False

            def __init__(self):
                self.times = []

            async def frames(self):
                due = time.perf_counter()
                while not self.stop:
                    due += 1 / 30
                    await asyncio.sleep(max(0, due - time.perf_counter()))
                    self.times.append(time.perf_counter())
                    yield InputFrame(len(self.times), 'video', now())

        for store in (':memory:', 'file'):
            with tempfile.TemporaryDirectory() as folder:
                human = Gamst(consent=True, max_records=20_000, max_payload_bytes=256 << 20, sensor_inference_interval_ms=1000)
                start = now() - 60_000
                fill(human.store, 10_000, start, 59_000)
                if store == 'file':         # the same records in a file
                    path = os.path.join(folder, 's.db')
                    target = sqlite3.connect(path)
                    human.store.connection.backup(target)
                    target.close()
                    await human.close()
                    human = Gamst(consent=True, store=path, max_records=20_000, max_payload_bytes=256 << 20,
                                  sensor_inference_interval_ms=1000)
                camera, pipeline = Camera(), Pipeline(human, 'p')
                pipeline.add_input('camera', camera)
                pipeline.add_module('live', Live(), consumes='camera', spec=Live.spec, queue=8)
                run, beats = asyncio.create_task(pipeline.run()), []

                async def beat():
                    last = time.perf_counter()
                    while not camera.stop:
                        await asyncio.sleep(.001)
                        beats.append(time.perf_counter() - last)
                        last = time.perf_counter()

                beating = asyncio.create_task(beat())
                await asyncio.sleep(.3)
                best = None
                for _ in range(4):      # the best of a few: other processes may take the CPU
                    beats.clear()
                    camera.times.clear()
                    began = time.perf_counter()
                    answer = await human.get_timeline(start_ms=start, end_ms=start + 60_000, step_ms=1000, max_bytes=262144)
                    took = time.perf_counter() - began
                    stall = max(beats)
                    best = min(best or (9, 0, 0), (stall, took, len(camera.times)))
                self.assertGreater(len(answer['rows']), 20)
                self.assertIn(answer['rows_total'], (60, 61))
                stall, took, frames = best
                self.assertLess(stall, .06, f'{store}: read {took:.3f} s')
                self.assertGreaterEqual(frames, int(took * 30) - 1)
                camera.stop = True
                await run
                await beating
                self.assertEqual(pipeline.status()['modules']['live']['gaps'], 0)
                await pipeline.close()
                await human.close()

    async def test_a_read_sees_one_state_whatever_happens_meanwhile(self):
        human = Gamst(consent=True, max_records=3000)
        human.register_capability(CapabilitySpec('a', 'a', 'video', 'f', '1', (OutputField('value', 'number'),), 'frame',
                                                 freshness_ms=1000))
        start = now() - 20_000
        for index in range(900):
            await human.push_observation('a', dict(value=float(index)), timestamp_ms=start + index * 20,
                                         sensor=SensorMetadata('s', 'c', index))
        query = dict(start_ms=start, end_ms=start + 18_000, step_ms=200, capabilities=['a'], max_bytes=262144)
        expected = await human.get_timeline(**query)
        scan, pushed = Store.scan, []

        def busy(store, *args, **about):
            for rows in scan(store, *args, **about):
                yield rows
                if len(pushed) < 5:     # committed between two steps of the same read
                    record = envelope('observation.signal', 'local-user', start + 100 * len(pushed), observation=store.query(
                        limit=1, name='a')[0]['observation'])
                    pushed.append(store.add(record))

        with mock.patch.object(Store, 'scan', busy):
            answer = await human.get_timeline(**query)
        self.assertEqual(len(pushed), 5)
        self.assertEqual((answer['rows'], answer['evidence']), (expected['rows'], expected['evidence']))
        # Records removed meanwhile: what they covered is not called complete.
        answer = await human.get_timeline(**query)
        with mock.patch.object(Store, 'scan', lambda store, *args, **about: self.evicting(human, scan(store, *args, **about))):
            answer = await human.get_timeline(**query)
        self.assertGreater(answer['removed_until_ms'], start)
        with mock.patch.object(Store, 'scan', lambda store, *args, **about: self.deleting(human, scan(store, *args, **about))):
            with self.assertRaisesRegex(RuntimeError, 'discarded'):
                await human.get_timeline(**query)
        task = asyncio.create_task(human.get_timeline(**query))
        await asyncio.sleep(0)
        await human.close()
        with self.assertRaises(RuntimeError):
            await task

    @staticmethod
    def evicting(human, steps):
        for rows in steps:
            yield rows
            human.store.max_records = 200
            human.store.add(envelope('human.inference', 'local-user', now()))

    @staticmethod
    def deleting(human, steps):
        for rows in steps:
            yield rows
            human.store.delete('local-user')

    async def test_other_tools_do_not_wait_for_a_read(self):
        human = Gamst(consent=True, max_records=20_000, max_payload_bytes=256 << 20)
        start = now() - 60_000
        fill(human.store, 4000, start, 59_000)
        server = MCPServer(human)
        read = asyncio.create_task(server.call_tool('get_timeline', dict(start_ms=start, end_ms=start + 60_000, step_ms=1000)))
        await asyncio.sleep(.01)
        await server.call_tool('get_state', {})
        self.assertFalse(read.done())
        self.assertIn((await read)['rows_total'], (60, 61))
        await human.close()


if __name__ == '__main__':
    unittest.main()
