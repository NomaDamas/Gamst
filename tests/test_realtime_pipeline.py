"""Lossless live Pipeline: lanes, gaps, completeness, replicas, pausing, inputs that can lose data, real workers."""
import argparse
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gamst import CapabilitySpec, Gamst, OutputField
from gamst.crew.modules import ReplicatedWorker, open_worker
from gamst.inputs import CameraInput, MicrophoneInput
from gamst.pipeline import Derived, Pipeline
from gamst.plugins import InputFrame
from gamst.schema import timestamp
try:
    import numpy as np
except ImportError:
    np = None

MODELS = Path(os.path.expanduser('~/.cache/gamst/demo/models'))
POSE = Path(os.path.expanduser('~/.cache/gamst/models/mediapipe/pose_landmarker_full.task'))


def spec(name, *, modality='video', stateful=False):
    return CapabilitySpec(name, name, modality, 'test', '1', (OutputField('events', 'array'),), 'native',
                          stateful=stateful, reset_on_gap=stateful)


class Slow:
    """A module whose frames wait until released."""

    def __init__(self, seconds=None):
        self.release, self.seen, self.resets, self.seconds = asyncio.Event(), [], 0, seconds

    async def process(self, frame):
        if self.seconds is None:
            await self.release.wait()
        else:
            await asyncio.sleep(self.seconds)
        self.seen.append(frame.timestamp_ms)
        return []

    def reset(self):
        self.resets += 1


class Live:
    """A live source of given (timestamp offset, data); ``before`` runs before a frame."""
    modality, live, missed, pace = 'video', True, 0, 0

    def __init__(self, items, before=None):
        self.items, self.before = items, before or {}

    async def frames(self):
        base = timestamp() - 60000
        for index, (offset, data, rate) in enumerate(self.items):
            if index in self.before:
                self.before[index](self)
            yield InputFrame(data, self.modality, base + offset, sample_rate=rate)
            await asyncio.sleep(self.pace)


async def until(condition, seconds=5):
    for _ in range(int(seconds * 200)):
        if condition():
            return
        await asyncio.sleep(.005)
    raise AssertionError('not reached')


class Lossless(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_module_builds_a_backlog_and_still_gets_every_frame(self):
        async with Gamst(consent=True) as human:
            pipeline, slow, seen = Pipeline(human, 'live'), Slow(), []
            pipeline.add_input('camera')
            pipeline.add_module('slow', slow, consumes='camera', queue=100)
            pipeline.add_module('math', SimpleNamespace(process=lambda frame: seen.append(frame.timestamp_ms) or []),
                                consumes='camera')
            now = timestamp()
            for index in range(30):
                await pipeline.push('camera', index, timestamp_ms=now - 3000 + index * 33, modality='video')
            await asyncio.sleep(.01)
            self.assertEqual(len(seen), 30)       # the slow module never delayed it
            status = pipeline.status()['modules']['slow']
            self.assertEqual((status['backlog'], status['processed'], status['gaps']), (30, 0, 0))
            self.assertGreaterEqual(status['backlog_ms'], 2900)
            self.assertEqual(status['complete_until_ms'], now - 3000 - 1)
            self.assertEqual(pipeline.status()['modules']['math']['complete_until_ms'], now - 3000 + 29 * 33)
            slow.release.set()
            await pipeline.drain()
            self.assertEqual(slow.seen, [now - 3000 + index * 33 for index in range(30)])
            status = pipeline.status()['modules']['slow']
            self.assertEqual((status['complete_until_ms'], status['backlog'], status['complete']),
                             (now - 3000 + 29 * 33, 0, True))
            self.assertEqual(status['latency_ms']['samples'], 30)
            self.assertTrue(pipeline.status()['complete'])
            await pipeline.close()

    async def test_an_exceeded_bound_is_a_declared_gap_on_the_module_its_events_and_the_count(self):
        class Counter:
            def process(self, frame):
                return dict(events=[dict(event='seen', start_ms=frame.timestamp_ms - 100, end_ms=frame.timestamp_ms)])

        async with Gamst(consent=True) as human:
            pipeline, slow = Pipeline(human, 'live'), Slow()
            pipeline.add_input('camera')
            pipeline.add_module('slow', slow, consumes='camera', queue=4)
            pipeline.add_module('count', Counter(), consumes='camera', spec=spec('test.count'))
            pipeline.add_module('other', Counter(), consumes='camera', spec=spec('test.other'), queue=100)
            now = timestamp() - 5000
            await pipeline.push('camera', 0, timestamp_ms=now, modality='video')
            await asyncio.sleep(0)
            for index in range(1, 10):
                await pipeline.push('camera', index, timestamp_ms=now + index * 33, modality='video')
            slow.release.set()
            await pipeline.drain()
            status = pipeline.status()['modules']
            # 10 frames for a lane of 4: five go, but not five in a row, and the first and newest are seen
            self.assertEqual(len(slow.seen), 5)
            self.assertEqual((slow.seen[0], slow.seen[-1]), (now, now + 9 * 33))
            self.assertLessEqual(max(b - a for a, b in zip(slow.seen, slow.seen[1:])), 4 * 33)
            self.assertEqual((status['slow']['dropped'], status['slow']['gap_frames']), (5, 5))
            self.assertEqual({gap['reason'] for gap in status['slow']['gap_intervals']}, {'thinned'})
            # 'count' reads the same lane's input and has the same holes; 'other' has its own lane
            self.assertTrue(status['other']['complete'])
            self.assertEqual(status['slow']['complete_until_ms'], now + 9 * 33)   # processed or declared
            counted = await human.count_events('test.other')
            self.assertTrue(counted['complete'])
            await pipeline.close()

    async def test_an_overloaded_video_lane_never_gives_up_a_whole_interval(self):
        async with Gamst(consent=True) as human:
            for offered, longest in ((1000, 2), (1700, 2), (3000, 4)):
                pipeline, slow = Pipeline(human, 'live'), Slow()
                pipeline.add_input('camera')
                pipeline.add_module('slow', slow, consumes='camera', queue=64)
                now = timestamp() - 200000
                capacity = pipeline.modules['slow']['lane']['capacity']
                share = offered / 900    # the same overload as a 900-frame lane offered ``offered`` frames
                total = int(capacity * share)
                for index in range(total):
                    await pipeline.push('camera', index, timestamp_ms=now + index * 33, modality='video')
                    await asyncio.sleep(0)
                slow.release.set()
                await pipeline.drain()
                status = pipeline.status()['modules']['slow']
                self.assertEqual(status['processed'] + status['dropped'], total)
                self.assertEqual(status['gap_frames'], status['dropped'])
                # every stretch of the recording is still seen: never more than a few frames missing in a row
                self.assertLessEqual(max(b - a for a, b in zip(slow.seen, slow.seen[1:])), longest * 33)
                self.assertFalse(status['complete'])
                self.assertEqual(slow.seen, sorted(slow.seen))
                await pipeline.close()

    async def test_audio_is_not_thinned(self):
        async with Gamst(consent=True) as human:
            pipeline, slow = Pipeline(human, 'live'), Slow()
            pipeline.add_input('microphone', modality='audio')
            pipeline.add_module('slow', slow, consumes='microphone', queue=20)
            now = timestamp() - 60000
            for index in range(24):
                await pipeline.push('microphone', [0.], timestamp_ms=now + index * 100, sample_rate=16000)
                await asyncio.sleep(0)
            slow.release.set()
            await pipeline.drain()
            reasons = {gap['reason'] for gap in pipeline.status()['modules']['slow']['gap_intervals']}
            self.assertEqual(reasons, {'backlog'})   # speech with every second 100 ms missing is not speech
            await pipeline.close()

    async def test_gaps_reach_events_counts_and_the_export(self):
        class Counter:
            def process(self, frame):
                return dict(events=[dict(event='seen', start_ms=frame.timestamp_ms - 40, end_ms=frame.timestamp_ms)])

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'run.db')
            async with Gamst(consent=True, store=path) as human:
                pipeline, release = Pipeline(human, 'live'), asyncio.Event()

                class Waiting(Counter):
                    async def process(self, frame):
                        await release.wait()
                        return super().process(frame)

                pipeline.add_input('camera')
                pipeline.add_module('count', Waiting(), consumes='camera', spec=spec('test.count'), queue=2)
                now = timestamp() - 5000
                for index in range(6):
                    await pipeline.push('camera', index, timestamp_ms=now + index * 33, modality='video')
                    await asyncio.sleep(0)
                release.set()
                await pipeline.drain()
                counted = await human.count_events('test.count')
                self.assertEqual((counted['complete'], counted['lost']['frames']), (False, 3))
                marked = [item['event'].get('gap_frames') for item in counted['evidence']]
                self.assertEqual(marked[0], None)
                self.assertIn(1, marked)          # an event that spans a thinned frame says so
                self.assertEqual(pipeline.lost('test.count')['frames'], 3)
                await pipeline.close()
            from gamst.recipes.movement import export
            out, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errors):
                export(argparse.Namespace(store=path, subject='local-user', capability='test.count', event=None,
                                          out=None, format='jsonl'))
            rows = [json.loads(line) for line in out.getvalue().splitlines()]
            gaps = [row for row in rows if row['event'] == 'pipeline.gap']
            self.assertEqual(sum(row['lost_frames'] for row in gaps), 3)
            self.assertEqual([row.get('gap_frames') for row in rows if row['event'] == 'seen'], marked)
            summary = json.loads(errors.getvalue())
            self.assertEqual((summary['events'], summary['complete']), (3, False))

    async def test_waiting_frames_are_bounded_in_bytes_and_shared(self):
        async with Gamst(consent=True) as human:
            pipeline, first, second = Pipeline(human, 'live', max_backlog_bytes=10_000), Slow(), Slow()
            pipeline.add_input('camera')
            pipeline.add_module('first', first, consumes='camera', queue=100)
            pipeline.add_module('second', second, consumes='camera', queue=100)
            now = timestamp() - 5000
            frames = [bytes(1000 + index) for index in range(9)]
            for index, data in enumerate(frames):
                await pipeline.push('camera', data, timestamp_ms=now + index, modality='video')
            self.assertEqual(pipeline.status()['backlog_bytes'], sum(map(len, frames)))   # once, not per module
            self.assertTrue(pipeline.status()['complete'])
            for index in range(9, 20):
                await pipeline.push('camera', bytes(2000), timestamp_ms=now + index, modality='video')
                self.assertLessEqual(pipeline.status()['backlog_bytes'], 10_000 + 4000)   # + frames being processed
            self.assertFalse(pipeline.status()['complete'])
            first.release.set(), second.release.set()
            await pipeline.drain()
            self.assertEqual(pipeline.status()['backlog_bytes'], 0)
            for slow, name in ((first, 'first'), (second, 'second')):
                self.assertEqual(len(slow.seen) + pipeline.status()['modules'][name]['gap_frames'], 20)
            await pipeline.close()

    async def test_declared_sampling_is_not_loss(self):
        async with Gamst(consent=True) as human:
            pipeline, slow = Pipeline(human, 'live'), Slow()
            slow.owns_process = True
            pipeline.add_input('camera')
            pipeline.add_module('scene', slow, consumes='camera', queue='auto', every_ms=100)
            now = timestamp() - 5000
            for index in range(30):
                await pipeline.push('camera', index, timestamp_ms=now + index * 33, modality='video')
                await asyncio.sleep(0)
            slow.release.set()
            await pipeline.drain()
            status = pipeline.status()['modules']['scene']
            self.assertEqual(status['processed'] + status['sampled']['skipped'], 30)
            self.assertEqual((status['queue'], status['gaps'], status['dropped'], status['complete']), (1, 0, 0, True))
            await pipeline.close()

    async def test_off_time_is_a_declared_gap_and_holds_the_watermark(self):
        async with Gamst(consent=True) as human:
            pipeline, seen = Pipeline(human, 'live'), []
            pipeline.add_input('camera')
            pipeline.add_module('math', SimpleNamespace(process=lambda frame: seen.append(frame.data) or []),
                                consumes='camera')
            now = timestamp() - 5000

            async def push(index):
                await pipeline.push('camera', index, timestamp_ms=now + index, modality='video')
            await push(1)
            pipeline.disable('math')
            await push(2)
            await push(3)
            self.assertEqual(pipeline.status()['modules']['math']['complete_until_ms'], now + 1)
            pipeline.enable('math')
            await push(4)
            status = pipeline.status()['modules']['math']
            self.assertEqual((status['complete_until_ms'], seen), (now + 4, [1, 4]))
            gap = status['gap_intervals'][0]
            self.assertEqual((gap['start_ms'], gap['end_ms'], gap['reason']), (now + 2, now + 3, 'disabled'))
            await pipeline.close()

    async def test_one_chain_runs_in_one_lane_and_pairs_the_same_capture(self):
        class Pose:
            owns_process = True
            name, version = 'pose', '1'

            async def process(self, frame):
                await asyncio.sleep(.001)
                return [Derived('joints', frame.timestamp_ms, 'video')]

        class Person:
            """Pairs a camera frame with the joints of the same capture, joints first (like TargetPerson)."""
            def __init__(self):
                self.joints, self.pairs = None, []

            def process(self, frame):
                if isinstance(frame.data, int):
                    self.joints = frame.data
                else:
                    self.pairs.append((self.joints, frame.timestamp_ms))
                return []

        async with Gamst(consent=True) as human:
            pipeline, person, scene = Pipeline(human, 'live'), Person(), Slow()
            pipeline.add_input('camera')
            pipeline.add_module('pose', Pose(), consumes='camera', queue='auto', emits=('joints',))
            pipeline.add_module('person', person, consumes=['camera', 'joints'], queue='auto')
            pipeline.add_module('scene', scene, consumes='camera', queue=200, optional=True)
            now = timestamp() - 5000
            for index in range(50):
                await pipeline.push('camera', 'image', timestamp_ms=now + index * 33, modality='video')
            await until(lambda: len(person.pairs) == 50)      # the stuck optional module did not delay the chain
            self.assertEqual(person.pairs, [(now + i * 33, now + i * 33) for i in range(50)])
            status = pipeline.status()['modules']
            self.assertEqual((status['person']['lane'], status['scene']['processed']), ('pose', 0))
            scene.release.set()
            await pipeline.close()

    async def test_optional_modules_pause_before_a_needed_one_loses_frames(self):
        async with Gamst(consent=True) as human:
            pipeline, needed, optional = Pipeline(human, 'live'), Slow(), Slow(.0005)
            pipeline.add_input('camera')
            pipeline.add_module('needed', needed, consumes='camera', queue=10)
            pipeline.add_module('optional', optional, consumes='camera', queue=100, optional=True)
            pipeline.add_module('in_chain', Slow(0), consumes='camera', queue=100, optional=True, emits=('chain',))
            pipeline.add_module('reads_chain', Slow(0), consumes='chain')
            now = timestamp() - 5000
            for index in range(10):
                await pipeline.push('camera', index, timestamp_ms=now + index, modality='video')
                await asyncio.sleep(.002)
            status = pipeline.status()['modules']
            self.assertTrue(status['optional']['paused'])
            self.assertFalse(status['in_chain']['paused'])       # a module that is needed reads it
            self.assertEqual((status['needed']['gaps'], status['optional']['pauses']), (0, 1))
            needed.release.set()
            await until(lambda: not pipeline.status()['modules']['optional']['paused'])
            await pipeline.push('camera', 10, timestamp_ms=now + 10, modality='video')
            await pipeline.drain()
            status = pipeline.status()['modules']
            self.assertEqual(status['optional']['gap_intervals'][0]['reason'], 'paused')
            self.assertEqual(len(optional.seen) + status['optional']['gap_frames'], 11)
            self.assertEqual((len(needed.seen), status['needed']['complete']), (11, True))
            await pipeline.close()

    async def test_lost_audio_is_never_joined_also_further_down(self):
        class Gate:
            name, version = 'gate', '1'

            def __init__(self):
                self.resets = 0

            def reset(self):
                self.resets += 1

            def process(self, frame):
                return [Derived('voice', frame.data, 'audio', sample_rate=16000)]

        class Windows(Gate):
            """Joins three chunks, like a windowed ASR; a reset starts a new window."""
            def __init__(self):
                super().__init__()
                self.buffer, self.windows = [], []

            def reset(self):
                super().reset()
                self.buffer = []

            def process(self, frame):
                self.buffer.append(frame.data)
                if len(self.buffer) == 3:
                    self.windows.append(self.buffer)
                    self.buffer = []
                return []

        def lose(source):
            source.missed += 2
        source = Live([(100 * i, i, 16000) for i in (1, 2, 5, 6, 7)], {2: lose})
        source.modality = 'audio'
        async with Gamst(consent=True) as human:
            pipeline, gate, windows = Pipeline(human, 'live'), Gate(), Windows()
            pipeline.add_input('microphone', source)
            pipeline.add_module('gate', gate, consumes='microphone', queue=10, emits=('voice',))
            pipeline.add_module('asr', windows, consumes='voice', queue=10)
            await pipeline.run()
            self.assertEqual((gate.resets, windows.resets, windows.windows), (1, 1, [[5, 6, 7]]))
            status = pipeline.status()
            self.assertEqual((status['inputs']['microphone']['missed'], status['modules']['asr']['gap_frames']), (2, 2))
            self.assertEqual(status['modules']['asr']['gap_intervals'][0]['reason'], 'input')
            await pipeline.close()

    async def test_recorded_input_waits_and_failures_of_needed_modules_stop_the_run(self):
        class Failing:
            def process(self, frame):
                raise ValueError('broken')

        async def run(module, **options):
            source = Live([(10 * i, i, None) for i in range(20)])
            source.live = False
            async with Gamst(consent=True) as human:
                pipeline, slow = Pipeline(human, 'file'), Slow(.001)
                pipeline.add_input('video', source)
                pipeline.add_module('slow', slow, consumes='video', queue=1)
                pipeline.add_module('second', module, consumes='video', queue=1, **options)
                try:
                    await pipeline.run()
                finally:
                    status = pipeline.status()['modules']
                    await pipeline.close()
                return len(slow.seen), status
        seen, status = await run(Slow(0))
        self.assertEqual((seen, status['slow']['dropped'], status['slow']['backlog_ms']), (20, 0, None))
        with self.assertRaisesRegex(ValueError, 'broken'):
            await run(Failing())
        seen, status = await run(Failing(), optional=True)
        self.assertEqual((seen, status['second']['errors'], status['second']['complete']), (20, 20, False))

    async def test_lane_records_reach_on_frame(self):
        class Counter:
            async def process(self, frame):
                await asyncio.sleep(.001)
                return dict(events=[dict(event='seen', start_ms=frame.timestamp_ms, end_ms=frame.timestamp_ms)])
        found = []
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'live')
            pipeline.add_input('camera', Live([(10 * i, i, None) for i in range(20)]))
            pipeline.add_module('count', Counter(), consumes='camera', spec=spec('test.count'), queue=50)
            found += await pipeline.run(on_frame=lambda input_id, frame, records: found.extend(records))
            self.assertEqual([r['observation']['value']['events'][0]['start_ms'] for r in found],
                             sorted(r['timestamp_ms'] for r in found))
            self.assertEqual(len({r['id'] for r in found}), 20)
            await pipeline.close()


class Replicas(unittest.IsolatedAsyncioTestCase):
    class Worker:
        private = closed = False

        def __init__(self, index, spec):
            self.index, self.spec, self.busy, self.seen = index, spec, False, []

        async def process(self, frame):
            assert not self.busy
            self.busy = True
            await asyncio.sleep((.009, .003, .006)[frame.data % 3])   # later frames finish earlier
            self.busy = False
            self.seen.append(frame.data)
            return dict(events=[dict(event='seen', start_ms=frame.timestamp_ms, end_ms=frame.timestamp_ms)])

        async def close(self):
            self.closed = True

    async def test_frames_are_shared_once_each_and_committed_in_capture_order(self):
        contract = spec('test.detector')
        workers = [self.Worker(index, contract) for index in range(3)]
        async with Gamst(consent=True) as human:
            pipeline, linked = Pipeline(human, 'live'), []
            pipeline.add_input('camera')
            pipeline.add_module('detect', ReplicatedWorker(workers), consumes='camera', spec=contract, queue='auto')
            pipeline.add_module('tracks', SimpleNamespace(process=lambda frame: linked.append(frame.timestamp_ms) or []),
                                consumes='detect')
            now = timestamp() - 5000
            for index in range(30):
                await pipeline.push('camera', index, timestamp_ms=now + index, modality='video')
            await pipeline.drain()
            self.assertEqual(sorted(sum((worker.seen for worker in workers), [])), list(range(30)))
            self.assertTrue(all(worker.seen for worker in workers))
            self.assertEqual(linked, [now + index for index in range(30)])
            status = pipeline.status()['modules']['detect']
            self.assertEqual((status['replicas'], status['processed'], status['complete_until_ms']), (3, 30, now + 29))
            await pipeline.close()
        self.assertTrue(all(worker.closed for worker in workers))

    async def test_stateful_modules_are_refused(self):
        contract = spec('test.tracker', stateful=True)
        with self.assertRaisesRegex(ValueError, 'stateful'):
            ReplicatedWorker([self.Worker(0, contract), self.Worker(1, contract)])
        with self.assertRaisesRegex(ValueError, 'stateful'):
            await open_worker(sys.executable, 'mediapipe.body.video',
                              dict(path='/x.task', sha256='0' * 64), replicas=2)
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'live')
            pipeline.add_input('camera')
            module = SimpleNamespace(process=lambda frame: None, replicas=2)
            for contract in (None, contract):
                with self.assertRaisesRegex(ValueError, 'stateless'):
                    pipeline.add_module('x', module, consumes='camera', spec=contract)

    def test_yoloe_is_replicable_only_when_declared_stateless(self):
        from gamst.providers.yoloe import YOLOEContext
        options = dict(labels=['cup'], prompt_sha256='0' * 64, checkpoint_sha256='1' * 64, mode='text')
        default, stateless = YOLOEContext.capability(**options), YOLOEContext.capability(**options, stateless=True)
        self.assertTrue(default.stateful and default.reset_on_gap)
        self.assertFalse(stateless.stateful or stateless.reset_on_gap)
        self.assertEqual(stateless.version, default.version + '.stateless')


@unittest.skipIf(np is None, 'numpy required')
class Inputs(unittest.IsolatedAsyncioTestCase):
    class Microphone:
        """A device stream: ``script`` of (samples, overflowed, waiting samples)."""
        latency = 0.

        def __init__(self, script):
            self.script, self.read_available, self.dropped = iter(script), 0, 0

        def start(self):
            pass

        stop = close = start

        def read(self, count):
            if count != 1600:
                self.dropped += count
                return np.zeros((count, 1), np.float32), False
            samples, overflowed, self.read_available = next(self.script)
            return np.full((samples, 1), .1, np.float32), overflowed

    async def collect(self, script, arrivals, **options):
        stream, clock = self.Microphone(script), iter(arrivals)
        microphone = MicrophoneInput(stream_factory=lambda **kwargs: stream, max_chunks=len(arrivals) - options.pop('lost', 0))
        with patch('gamst.inputs._now_ms', lambda: next(clock)):
            frames = []
            async for frame in microphone.frames():
                frames.append((frame.timestamp_ms, microphone.missed))
        return microphone, stream, frames

    async def test_chunks_follow_the_sample_clock_not_arrival(self):
        # arrival is late by 0..90 ms; the second read left 800 samples (50 ms) in the device
        _, _, frames = await self.collect([(1600, False, 0), (1600, False, 800), (1600, False, 0), (1600, False, 0)],
                                          [10100, 10290, 10300, 10490])
        self.assertEqual([stamp for stamp, _ in frames], [10100, 10200, 10300, 10400])

    async def test_overflow_is_counted_dropped_around_and_restarts_the_clock(self):
        microphone, stream, frames = await self.collect(
            [(1600, False, 0), (1600, True, 800), (1600, False, 0), (1600, False, 0)], [10100, 10900, 11000, 11100],
            lost=1)
        self.assertEqual(frames, [(10100, 0), (11000, 8), (11100, 8)])
        self.assertEqual((microphone.overflows, microphone.lost_ms, stream.dropped), (1, 800, 800))

    async def test_unreported_loss_restarts_the_clock(self):
        microphone, _, frames = await self.collect([(1600, False, 0)] * 3, [10100, 10200, 11300])
        self.assertEqual(frames, [(10100, 0), (10200, 0), (11300, 10)])
        self.assertEqual((microphone.resyncs, microphone.lost_ms), (1, 1000))

    async def camera(self, count, limit, pause):
        class Capture:
            def __init__(self, *_):
                self.left, self.released = count, False

            def isOpened(self):
                return True

            def read(self):
                self.left -= 1
                return self.left >= 0, np.full((10, 10, 3), count - self.left - 1, np.uint8)

            def release(self):
                self.released = True
        fake = SimpleNamespace(VideoCapture=Capture, CAP_AVFOUNDATION=1, CAP_ANY=0)
        source, seen = CameraInput(0), []
        source.MAX_WAITING_BYTES = limit
        with patch.dict(sys.modules, cv2=fake):
            async for frame in source.frames():
                if not seen:
                    await asyncio.sleep(pause)
                seen.append((int(frame.data[0, 0, 0]), source.missed))
        return source, seen

    async def test_camera_frames_wait_for_a_busy_consumer(self):
        source, seen = await self.camera(40, 1_000_000, .1)
        self.assertEqual(([index for index, _ in seen], source.missed), (list(range(40)), 0))

    async def test_camera_frames_beyond_the_bound_are_counted_where_they_are_missing(self):
        source, seen = await self.camera(40, 300 * 5, .1)
        indexes = [index for index, _ in seen]
        self.assertEqual(source.missed, 40 - len(seen))
        self.assertGreater(source.missed, 0)
        for (before, missed_before), (index, missed) in zip(seen, seen[1:]):
            self.assertEqual(index - before - 1, missed - missed_before)
        self.assertEqual(indexes[-1], 39)


@unittest.skipIf(np is None, 'numpy required')
class RecipeLanes(unittest.IsolatedAsyncioTestCase):
    async def test_the_counting_chain_is_one_lane_and_the_scene_is_sampled(self):
        from gamst.recipes import load
        recipe = load('person-action-count', ['semantic'])

        class Worker(Slow):
            owns_process = True
            name, version = 'fake', '1'
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'recipe')
            pipeline.add_input('camera')
            for item in recipe['modules']:
                worker = item['use'] in ('mediapipe-pose', 'semantic-scene')
                pipeline.add_module(item['name'], Worker() if worker else Slow(), consumes=item['consumes'],
                                    emits=tuple(item.get('emits', ())), every_ms=item.get('every_ms'), queue='auto',
                                    optional=bool(item.get('optional')))
            status = pipeline.status()['modules']
            self.assertEqual({name: value.get('lane') for name, value in status.items()},
                             {name: 'semantic' if name == 'semantic' else 'pose' for name in status})
            self.assertEqual([name for name in status if pipeline._is_pausable(name)], ['semantic'])
            await pipeline.close()

    def test_modules_that_own_a_process_say_so(self):
        from gamst.crew.modules import WorkerModule
        from gamst.mediapipe_graph import MediaPipeGraphModule
        from gamst.providers.asr_words import AlignedASR
        from gamst.providers.model_workers import RemoteWhisper
        from gamst.providers.person_match import TargetPerson
        from gamst.providers.pose import PoseWorkerPlugin
        from gamst.providers.sapiens2_pose import Sapiens2PosePlugin
        from gamst.providers.speaker_verify import SpeakerVerify
        from gamst.providers.target_speech import SAMSpanSpeech, TargetSpeech
        for cls in (WorkerModule, MediaPipeGraphModule, AlignedASR, RemoteWhisper, PoseWorkerPlugin,
                    Sapiens2PosePlugin, SpeakerVerify, TargetSpeech, SAMSpanSpeech):
            self.assertTrue(cls.owns_process, cls)
        self.assertFalse(getattr(TargetPerson, 'owns_process', False))   # it runs in the pose lane


class RealWorkers(unittest.IsolatedAsyncioTestCase):
    """The worker contract itself (sensor metadata, coordinates, wire format), not a stand-in of it."""

    def setUp(self):
        if np is None:
            self.skipTest('numpy required')

    def frames(self, count):
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (256, 256, 3), np.uint8) for _ in range(count)]

    async def run_worker(self, module, count=12, pace=0, **options):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'real')
            source = Live([(33 * i, image, None) for i, image in enumerate(self.frames(count))])
            source.pace = pace
            pipeline.add_input('camera', source)
            pipeline.add_module('model', module, consumes='camera', spec=getattr(module, 'spec', None), queue='auto',
                                **options)
            try:
                await pipeline.run()
                status = pipeline.status()['modules']['model']
                records = human.store.query('local-user', kind='observation.signal', limit=100)
            finally:
                await pipeline.close()
        skipped = status.get('sampled', {}).get('skipped', 0)
        self.assertEqual((status['processed'] + skipped, status['errors'], status['complete']), (count, 0, True),
                         status['last_error'])
        return status, records

    async def test_yoloe_replicas_accept_every_pipeline_frame(self):
        from gamst.demos.rotation import MODEL_ASSETS
        if any(not (MODELS / name).is_file() for name in MODEL_ASSETS):
            self.skipTest('YOLOE weights are not on this disk')
        try:
            import torch, ultralytics  # noqa: F401,E401
        except ImportError:
            self.skipTest('YOLOE runtime is not installed')
        with tempfile.TemporaryDirectory() as runtime:
            config = dict(checkpoint_path=str(MODELS / 'yoloe-26s-seg.pt'),
                          checkpoint_sha256=MODEL_ASSETS['yoloe-26s-seg.pt'][1], runtime_root=runtime, mode='text',
                          labels=['person'], encoder_path=str(MODELS / 'mobileclip2_b.ts'),
                          encoder_sha256=MODEL_ASSETS['mobileclip2_b.ts'][1], encoder_family='mobileclip2:b',
                          device='cpu', imgsz=320, stateless=True)
            module = await open_worker(sys.executable, 'yoloe.local', config, replicas=2)
            status, records = await self.run_worker(module)
        self.assertEqual([record['timestamp_ms'] for record in records], sorted(r['timestamp_ms'] for r in records))
        self.assertEqual((len(records), status['replicas']), (12, 2))
        self.assertEqual(records[0]['observation']['value']['summary']['coordinate_frame'], 'image')

    async def test_mediapipe_pose_keeps_one_tracking_state_over_skipped_frames(self):
        if not POSE.is_file():
            self.skipTest('pose weights are not on this disk')
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            self.skipTest('mediapipe is not installed')
        import hashlib
        digest = hashlib.sha256(POSE.read_bytes()).hexdigest()
        module = await open_worker(sys.executable, 'mediapipe.body.video',
                                   dict(path=str(POSE), sha256=digest, max_entities=1, allow_skips=True))
        status, records = await self.run_worker(module, every_ms=60, pace=.1)     # every second frame: sequence jumps by 2
        segments = {record['observation']['value']['summary']['state_segment'] for record in records}
        self.assertEqual(len(segments), 1)
        self.assertGreater(status['sampled']['skipped'], 0)
        self.assertEqual((len(records), status['processed'] > 1), (status['processed'], True))


if __name__ == '__main__':
    unittest.main()
