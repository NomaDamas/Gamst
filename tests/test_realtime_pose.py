"""Realtime pose lanes: a lane that samples the newest frame, a lane that keeps every frame in the video memory, a
lane that batches waiting frames, RTMW batches, keypoint-set-aware joint reading, shared-memory frames for workers,
and the recipe options that select them."""
import argparse
import asyncio
from types import SimpleNamespace
import unittest

from gamst import CapabilitySpec, Gamst, OutputField
import time

from gamst.pipeline import Derived, Pipeline
from gamst.plugins import InputFrame
from gamst.schema import timestamp
try:
    import numpy as np
except ImportError:
    np = None


class Slow:
    """Each frame waits until released."""

    def __init__(self):
        self.release, self.seen = asyncio.Event(), []

    async def process(self, frame):
        await self.release.wait()
        self.seen.append((frame.timestamp_ms, None if frame.data is None else int(np.asarray(frame.data).sum())))
        return []


@unittest.skipIf(np is None, 'numpy')
class VideoMemoryTest(unittest.TestCase):
    def test_frames_beyond_memory_spill_and_come_back_the_same(self):
        from gamst.providers.video_memory import VideoMemory
        memory = VideoMemory(memory_bytes=2 * 300, disk_bytes=10 * 300, disk_floor_bytes=0)
        frames = [np.full((10, 10, 3), index, np.uint8) for index in range(5)]
        kept = [memory.keep(frame) for frame in frames]
        status = memory.status()
        self.assertEqual((status['spilled_frames'], status['in_memory_bytes'], status['spilled_bytes']), (3, 600, 900))
        for frame, item in zip(frames, kept):
            self.assertTrue(np.array_equal(memory.take(item), frame))
            memory.release(item)
        self.assertEqual((memory.in_memory, memory.spilled, memory._end), (0, 0, 0))   # the file starts over
        memory.close()

    def test_the_spill_bound_keeps_a_free_disk_floor(self):
        from gamst.providers import video_memory
        free = [5000]
        original = video_memory.VideoMemory._free
        video_memory.VideoMemory._free = lambda self: free[0]
        try:
            memory = video_memory.VideoMemory(memory_bytes=0, disk_bytes=10_000, disk_floor_bytes=4000)
            self.assertEqual((memory.disk_bytes, memory.status()['disk_limited']), (1000, True))   # min(asked, free - floor)
            self.assertIsNotNone(memory.keep(np.zeros((10, 10, 3), np.uint8)))      # 300 bytes: still above the floor
            free[0] = 4100                                                          # the disk filled meanwhile
            self.assertIsNone(memory.keep(np.zeros((10, 10, 3), np.uint8)))         # declared, never below the floor
            memory.close()
        finally:
            video_memory.VideoMemory._free = original

    def test_a_frame_beyond_the_disk_bound_is_dropped_not_kept(self):
        from gamst.providers.video_memory import VideoMemory
        memory = VideoMemory(memory_bytes=0, disk_bytes=300, disk_floor_bytes=0)
        self.assertIsNotNone(memory.keep(np.zeros((10, 10, 3), np.uint8)))
        self.assertIsNone(memory.keep(np.zeros((10, 10, 3), np.uint8)))
        self.assertEqual(memory.status()['dropped'], 1)
        memory.close()


@unittest.skipIf(np is None, 'numpy')
class Lanes(unittest.IsolatedAsyncioTestCase):
    async def test_a_newest_frame_lane_skips_by_choice_and_loses_nothing(self):
        async with Gamst(consent=True) as human:
            pipeline, slow = Pipeline(human, 'live'), Slow()
            pipeline.add_input('camera')
            pipeline.add_module('pose', slow, consumes='camera', sample='newest')
            now = timestamp() - 5000
            for index in range(10):
                await pipeline.push('camera', np.full((2, 2, 3), index, np.uint8), timestamp_ms=now + index * 17,
                                    modality='video')
                await asyncio.sleep(0)
            slow.release.set()
            await pipeline.drain()
            status = pipeline.status()['modules']['pose']
            self.assertEqual([ms for ms, _ in slow.seen], [now, now + 9 * 17])   # the first, then the newest
            self.assertEqual(status['sampled'], dict(every_ms=None, skipped=8, choice='newest'))
            self.assertEqual((status['gaps'], status['gap_frames'], status['sample']), (0, 0, 'newest'))
            self.assertEqual(pipeline.lost(), dict(gaps=0, frames=0))
            await pipeline.close()

    async def test_a_spilling_lane_processes_every_frame_with_its_data(self):
        async with Gamst(consent=True) as human:
            pipeline, slow = Pipeline(human, 'live'), Slow()
            pipeline.add_input('camera')
            pipeline.add_module('verify', slow, consumes='camera', backlog='spill')
            memory = pipeline.video_memory(memory_bytes=2 * 12, disk_bytes=1 << 20, disk_floor_bytes=0)
            now = timestamp() - 5000
            for index in range(20):
                await pipeline.push('camera', np.full((2, 2, 3), index, np.uint8), timestamp_ms=now + index * 17,
                                    modality='video')
            self.assertEqual(memory.status()['spilled_frames'], 18)
            self.assertEqual(pipeline.status()['backlog_bytes'], 0)     # not held against the lanes' byte bound
            slow.release.set()
            await pipeline.drain()
            self.assertEqual(slow.seen, [(now + index * 17, index * 12) for index in range(20)])
            status = pipeline.status()
            self.assertEqual((status['modules']['verify']['gaps'], status['modules']['verify']['keeps']), (0, 'spill'))
            self.assertEqual((status['video_memory']['in_memory_bytes'], status['video_memory']['spilled_bytes']), (0, 0))
            await pipeline.close()

    def test_sample_and_backlog_are_checked(self):
        pipeline = Pipeline(SimpleNamespace(register_capability=lambda spec: None, pipeline=None), 'x')
        pipeline.add_input('camera')
        with self.assertRaises(ValueError):
            pipeline.add_module('a', Slow(), consumes='camera', sample='oldest')
        with self.assertRaises(ValueError):
            pipeline.add_module('b', Slow(), consumes='camera', sample='newest', backlog='spill')


class Batching:
    """process_batch answers one output per frame; each call waits until released."""

    def __init__(self, seconds=0.):
        self.release, self.calls, self.seconds = asyncio.Event(), [], seconds

    async def process(self, frame):
        return (await self.process_batch([frame]))[0]

    async def process_batch(self, frames):
        await self.release.wait()
        await asyncio.sleep(self.seconds)
        self.calls.append([frame.timestamp_ms for frame in frames])
        return [[] for _ in frames]


@unittest.skipIf(np is None, 'numpy')
class BatchLanes(unittest.IsolatedAsyncioTestCase):
    async def test_a_batching_lane_takes_what_waits_and_processes_every_frame_in_order(self):
        async with Gamst(consent=True) as human:
            pipeline, module = Pipeline(human, 'live'), Batching()
            pipeline.add_input('camera')
            pipeline.add_module('pose', module, consumes='camera', backlog='spill', batch=4)
            now = timestamp() - 5000
            for index in range(10):
                await pipeline.push('camera', np.full((2, 2, 3), index, np.uint8), timestamp_ms=now + index * 17,
                                    modality='video')
            await asyncio.sleep(0)
            module.release.set()
            await pipeline.drain()
            self.assertEqual(sum(module.calls, []), [now + index * 17 for index in range(10)])
            self.assertEqual([len(call) for call in module.calls], [4, 4, 2])   # what waits, up to 4 a call
            status = pipeline.status()['modules']['pose']
            self.assertEqual((status['processed'], status['gaps']), (10, 0))
            self.assertEqual(status['batch'], dict(most=4, calls=3, batched_frames=10, largest=4))
            await pipeline.close()

    async def test_a_recording_fills_its_batches(self):
        async with Gamst(consent=True) as human:
            pipeline, module = Pipeline(human, 'file'), Batching(.01)
            module.release.set()
            base = timestamp() - 60000

            class Recording:
                modality, live, missed = 'video', False, 0

                async def frames(self):
                    for index in range(9):
                        yield InputFrame(np.full((2, 2, 3), index, np.uint8), 'video', base + index * 17)
                        await asyncio.sleep(0)
            pipeline.add_input('camera', Recording())
            pipeline.add_module('pose', module, consumes='camera', batch=4)
            await pipeline.run()
            self.assertEqual(sum(module.calls, []), [base + index * 17 for index in range(9)])
            self.assertGreater(max(len(call) for call in module.calls), 1)
            await pipeline.close()

    def test_batch_is_checked_and_ignored_without_process_batch(self):
        pipeline = Pipeline(SimpleNamespace(register_capability=lambda spec: None, pipeline=None), 'x')
        pipeline.add_input('camera')
        with self.assertRaises(ValueError):
            pipeline.add_module('a', Batching(), consumes='camera', batch=0)
        pipeline.add_module('b', Slow(), consumes='camera', batch=4)
        self.assertIsNone(pipeline.modules['b']['lane'])     # no process_batch: inline, one by one


HITS = CapabilitySpec('test.hits', 'test.hits', 'video', 'test', '1',
                     (OutputField('count', 'integer'), OutputField('events', 'array')), 'native', stateful=True)
FRAME = 17


class Camera:
    """A live camera stand-in: frames every 17 ms, stamped when captured; the first pixel says which frame it is."""
    modality, live, missed = 'video', True, 0

    def __init__(self, count):
        self.count, self.base = count, None

    async def frames(self):
        self.base = time.time_ns() // 1_000_000
        for index in range(self.count):
            due = self.base + index * FRAME
            wait = due / 1000 - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            yield InputFrame(np.full((2, 2, 3), index % 256, np.uint8), 'video', due)


class Poser:
    """A pose model stand-in: slow on ``slow`` frames (live only); its joints are the frame's number."""
    owns_process, modalities = True, ('video',)

    def __init__(self, slow=(), fork_delay=0.):
        self.slow, self.forks, self.seen, self.delay, self.fork_delay = set(slow), [], [], 0., fork_delay

    async def process_batch(self, frames):
        outputs = []
        for frame in frames:
            index = int(frame.data[0, 0, 0])
            self.seen.append(index)
            if index in self.slow:
                await asyncio.sleep(.05)
            elif self.delay:
                await asyncio.sleep(self.delay)
            outputs.append([Derived('joints', index, 'video')])
        return outputs

    async def process(self, frame):
        return (await self.process_batch([frame]))[0]

    async def fork(self):
        self.forks.append(Poser())
        self.forks[-1].delay = self.fork_delay
        return self.forks[-1]


class Hits:
    """A movement stand-in with state: a hit is frames 4, 5, 6 of every 12 up, then one down; it ends at the down."""

    def __init__(self):
        self.spec, self.run, self.start, self.count = HITS, 0, None, 0

    def process(self, frame):
        index = frame.data
        if index % 12 in (4, 5, 6):
            self.start = frame.timestamp_ms if self.run == 0 else self.start
            self.run += 1
            return None
        found, self.run = self.run == 3, 0
        if not found:
            return None
        self.count += 1
        return dict(count=self.count, events=[dict(event='hit', start_ms=self.start, end_ms=frame.timestamp_ms)])

    def fork(self):
        return Hits()


@unittest.skipIf(np is None, 'numpy')
class VideoCatchUp(unittest.IsolatedAsyncioTestCase):
    async def run_cut(self, slow, tail_ms=150, fork_delay=0.):
        async with Gamst(consent=True) as human:
            pipeline, camera = Pipeline(human, 'live'), Camera(150)
            poser = Poser(slow=slow, fork_delay=fork_delay)                     # 50 ms of work per 17 ms frame there: it falls behind
            pipeline.add_input('camera', camera)
            pipeline.add_module('pose', poser, consumes='camera', emits=('joints',), backlog='spill', batch=4,
                                catch_up=dict(lag_ms=150, max_lookback_ms=150, tail_ms=tail_ms, workers=1))
            pipeline.add_module('hits', Hits(), consumes='joints', spec=HITS)
            await pipeline.run()
            status = pipeline.status()['modules']['pose']
            rows = human.store.query('local-user', kind='observation.signal', limit=500, registered_only=True,
                                     capability_id='test.hits')
            events = [event for row in rows for event in row['observation']['value']['events']]
            await pipeline.close()
        frame = lambda ms: (ms - camera.base) / FRAME     # noqa: E731  (in frames; owned ends may fall between)
        cuts = [(frame(cut['from_ms']), frame(cut['start_ms']), frame(cut['end_ms'])) for cut in status['catch_up']['todos']]
        found = [(frame(event['start_ms']), frame(event['end_ms']), bool(event.get('correction'))) for event in events]
        seen = set(poser.seen + sum((fork.seen for fork in poser.forks), []))
        return status, cuts, found, seen

    async def test_frames_the_live_lane_did_not_reach_are_counted_once_by_an_l2_chain(self):
        straddled = []
        for slow in (range(20, 45), range(22, 45), range(25, 49)):
            status, cuts, found, seen = await self.run_cut(slow)
            self.assertEqual(sorted(round(end) for _, end, _ in found), [i for i in range(150) if i % 12 == 7],
                             (cuts, found))
            self.assertEqual((status['catch_up']['done'], status['gaps']), (len(cuts), 0))
            self.assertTrue(cuts and seen == set(range(150)))
            for begin, end, late in found:     # an event is the lane's whose owned range holds its end
                self.assertEqual(late, any(start <= end <= until for _, start, until in cuts), (cuts, found))
                if any(begin < start <= end or begin <= until < end for _, start, until in cuts):
                    straddled.append((begin, end, late))
        self.assertTrue(straddled)     # some hits crossed a cut, and still counted once

    async def test_a_second_interval_waits_for_the_one_worker_and_is_processed(self):
        # the one L2 worker is still busy with the first interval when the second is cut
        status, cuts, found, seen = await self.run_cut(list(range(15, 35)) + list(range(70, 100)), fork_delay=.06)
        self.assertGreaterEqual(len(cuts), 2)
        self.assertEqual((status['catch_up']['done'], status['catch_up']['pending']), (len(cuts), 0))
        self.assertEqual(seen, set(range(150)))
        self.assertEqual(sorted(round(end) for _, end, _ in found), [i for i in range(150) if i % 12 == 7])


class CoreMLTemp(unittest.TestCase):
    def test_only_onnx_runtime_temp_copies_of_dead_processes_are_removed(self):
        import os
        import tempfile
        from pathlib import Path
        from gamst.providers.rtmw_pose import clear_dead_coreml_temp
        with tempfile.TemporaryDirectory() as folder:
            dead, alive = 999999, os.getpid()
            names = [f'onnxruntime-510A2ABA-8A23-47E0-8FD0-5F8ADF818781-{dead}-00000212F880E38E',
                     f'onnxruntime-1786B2EE-CDFB-4A6D-A38A-AD559C1ACB99-{dead}-00000212F5A705E8.mlmodelc',
                     f'onnxruntime-E0D35E30-F43E-43F3-8312-E9760EF3863B-{alive}-00000212F84DE435',
                     f'onnxruntime-notes-{dead}-x']
            for name in names:
                (Path(folder) / name).mkdir()
            self.assertEqual(clear_dead_coreml_temp(folder), 2)
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()), sorted(names[2:]))


class SafetyNet(unittest.TestCase):
    def test_the_same_event_from_the_other_lane_within_100_ms_is_stored_once(self):
        pipeline = Pipeline(SimpleNamespace(register_capability=lambda spec: None, pipeline=None), 'x')
        entry = {}
        hit = lambda end: dict(count=1, events=[dict(event='hit', label='slam', start_ms=end - 200, end_ms=end)])  # noqa
        self.assertEqual(pipeline._once(entry, hit(1000), False)['events'][0]['end_ms'], 1000)
        self.assertIsNone(pipeline._once(entry, hit(1060), True))                 # L2 found the live one again
        self.assertIsNotNone(pipeline._once(entry, hit(1060), False))             # the same lane twice: two hits
        self.assertIsNotNone(pipeline._once(entry, hit(1300), True))              # another hit


@unittest.skipIf(np is None, 'numpy')
class PostChunks(unittest.IsolatedAsyncioTestCase):
    async def test_a_recording_in_chunks_counts_each_hit_once_and_nothing_is_late(self):
        async with Gamst(consent=True) as human:
            pipeline, poser = Pipeline(human, 'file'), Poser(slow=range(0, 150))    # slow everywhere: chunks help
            base = timestamp() - 60000

            class Recording:
                modality, live, missed = 'video', False, 0

                async def frames(self):
                    for index in range(150):
                        yield InputFrame(np.full((2, 2, 3), index, np.uint8), 'video', base + index * FRAME)
            pipeline.add_input('camera', Recording())
            pipeline.add_module('pose', poser, consumes='camera', emits=('joints',), backlog='spill', batch=4,
                                catch_up=dict(lag_ms=150, max_lookback_ms=150, tail_ms=150, workers=1,
                                              post_chunk_ms=500))
            pipeline.add_module('hits', Hits(), consumes='joints', spec=HITS)
            await pipeline.run()
            status = pipeline.status()['modules']['pose']
            rows = human.store.query('local-user', kind='observation.signal', limit=500, registered_only=True,
                                     capability_id='test.hits')
            events = [event for row in rows for event in row['observation']['value']['events']]
            await pipeline.close()
        ends = sorted(round((event['end_ms'] - base) / FRAME) for event in events)
        self.assertEqual(ends, [index for index in range(150) if index % 12 == 7])
        self.assertGreaterEqual(status['catch_up']['done'], 1)                      # some ran as chunks
        self.assertFalse(any(event.get('correction') for event in events))           # nothing is late in post
        self.assertEqual(status['gaps'], 0)


@unittest.skipIf(np is None, 'numpy')
class RTMWBatch(unittest.TestCase):
    def models(self, det_every_ms=500):
        from gamst.providers.rtmw_pose import POINTS, RTMWModels
        models = object.__new__(RTMWModels)
        models.det_every_ms, models.max_poses, models.max_crops, models._fixed = det_every_ms, 4, 16, {}
        models.detections = []

        def detector(image):
            models.detections.append(int(image[0, 0, 0]))
            return np.array([[100., 50., 300., 400.]])

        def postprocess(outputs, center, scale):
            points = np.zeros((1, POINTS, 2), 'f4')
            points[0, :17] = [[150. + 10 * k, 100. + 15 * k] for k in range(17)]
            return points, np.ones((1, POINTS), 'f4')
        models.detector = detector
        models.calls = []
        models.pose = SimpleNamespace(
            session=SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='input')],
                                    run=lambda _, feed: models.calls.append(len(feed['input'])) or
                                    [np.zeros((len(feed['input']), POINTS, 576), 'f4'),
                                     np.zeros((len(feed['input']), POINTS, 768), 'f4')]),
            postprocess=postprocess)
        models._fixed = {4: models.pose.session, 2: models.pose.session}
        models._crop = lambda image, box: (np.zeros((4, 3, 3), np.uint8), (0, 0), (1, 1))
        models._mean, models._std = np.zeros(3, 'f4'), np.ones(3, 'f4')
        models.reset()
        return models

    def test_one_pose_call_for_a_batch_and_the_detector_only_when_due(self):
        models = self.models()
        images = [np.full((480, 640, 3), index, np.uint8) for index in range(6)]
        found = models.infer_batch(images, [0, 17, 33, 50, 517, 534])
        self.assertEqual(models.calls, [4, 2])              # the crops of the batch in fixed-size calls
        self.assertEqual(models.detections, [0, 4])          # at the start and 500 ms later
        self.assertEqual([len(boxes) for boxes, _, _ in found], [1] * 6)
        self.assertEqual(found[1][0], [[100., 50., 300., 400.]])   # within the batch: the detection before it
        after = models.infer_batch(images[:1], [551])
        self.assertNotEqual(after[0][0], found[0][0])        # next batch: the box around the last keypoints
        self.assertEqual(models.detections, [0, 4])

    def test_the_worker_splits_stacked_frames_and_names_each_pose_s_frame(self):
        from gamst.providers.model_workers import _RTMW
        models = self.models()
        models.version = 'rtmw:test'
        stacked = np.concatenate([np.full((480, 640, 3), index, np.uint8) for index in range(3)])
        public, joints = _RTMW(models).query(SimpleNamespace(data=stacked), dict(frames=3, timestamps_ms=[0, 17, 33]))
        self.assertEqual([pose['in_batch'] for pose in public['poses']], [0, 1, 2])
        self.assertEqual((public['image_height'], public['frames'], joints.shape), (480, 3, (3, 133, 3)))
        from gamst.providers.model_workers import capability
        spec = capability('rtmw.pose', dict(path='p', sha256='0' * 64, det_path='d', det_sha256='1' * 64))
        self.assertEqual(spec.validate(public)['frames'], 3)     # what the worker answers passes its contract
        with self.assertRaises(ValueError):
            _RTMW(models).query(SimpleNamespace(data=stacked), dict(frames=7, timestamps_ms=[0] * 7))   # 1440 rows


def rtmw_pose(points):
    """An RTMW-shaped pose (coco_wholebody133) with ``points`` {index: (x, y)} visible, the rest invisible."""
    landmarks = [dict(x=0., y=0., visibility=0., presence=1.) for _ in range(133)]
    for index, (x, y) in points.items():
        landmarks[index] = dict(x=x, y=y, visibility=.9, presence=1.)
    return dict(keypoint_set='coco_wholebody133', landmarks=landmarks, image_width=1280, image_height=720)


class KeypointSets(unittest.TestCase):
    def test_face_points_of_an_rtmw_pose_are_its_own_indices(self):
        from gamst.providers.person_match import face_points, face_reading
        face = {2: (.40, .30), 1: (.46, .30), 0: (.43, .34), 71: (.41, .38), 77: (.45, .38)}
        pose = rtmw_pose(face)
        indices, visibility = face_reading(pose)
        points = face_points(pose['landmarks'], 1000, 1000, indices, min_visibility=visibility)
        self.assertEqual(points, [(face[index][0] * 1000, face[index][1] * 1000) for index in (2, 1, 0, 71, 77)])
        # MediaPipe's indices on the same pose would read body and face points that are not the eyes and mouth
        self.assertIsNone(face_points(pose['landmarks'], 1000, 1000, (5, 2, 0, 10, 9)))
        self.assertEqual(face_reading(dict(keypoint_set='goliath308'))[0], None)    # unknown face points: no face
        self.assertEqual(face_reading(dict(landmarks=[]))[0], (5, 2, 0, 10, 9))      # MediaPipe names no set

    def test_pose_channels_read_an_rtmw_pose_by_its_keypoint_set(self):
        from gamst.providers.pose_exemplar import MEDIAPIPE_BODY, largest_pose
        from gamst.providers.rtmw_pose import COCO_WHOLEBODY_BODY
        body = {index: (.3 + .01 * index, .2 + .02 * index) for index in COCO_WHOLEBODY_BODY.values()}
        values, mask, _, _ = largest_pose([rtmw_pose(body)], MEDIAPIPE_BODY)     # the wrong mapping is not used
        self.assertTrue(all(mask))
        with self.assertRaises(ValueError):
            largest_pose([dict(rtmw_pose(body), keypoint_set='unknown17')], MEDIAPIPE_BODY)


@unittest.skipIf(np is None, 'numpy')
class FrameMemoryTest(unittest.TestCase):
    def test_a_large_frame_goes_through_shared_memory_and_back(self):
        from gamst.capabilities import SensorMetadata
        from gamst.plugins import InputFrame
        from gamst.provider_wire import FrameMemory, decode_frame
        image = np.arange(720 * 1280 * 3, dtype=np.uint32).astype(np.uint8).reshape(720, 1280, 3)
        sent = InputFrame(image, 'video', 5, 'local-user', None, SensorMetadata(source_id='s', sensor_id='c', sequence=1))
        memory = FrameMemory()
        header, payload = memory.put(sent)
        self.assertEqual((header['layout']['kind'], payload), ('shared', b''))
        got = decode_frame(header, payload)
        self.assertTrue(np.array_equal(got.data, image))
        self.assertEqual((got.timestamp_ms, got.sensor.sequence), (5, 1))
        small = InputFrame(np.zeros((4, 4, 3), np.uint8), 'video', 6, 'local-user', None, sent.sensor)
        self.assertIsNone(memory.put(small))     # small frames keep the pipe
        batch = np.concatenate([image] * 8)      # 22 MB: beyond the pipe's limit, within shared memory's
        header, _ = memory.put(InputFrame(batch, 'video', 7, 'local-user', None, sent.sensor))
        self.assertTrue(np.array_equal(decode_frame(header, b'').data, batch))
        memory.close()


class RecipeOptions(unittest.TestCase):
    def test_pose_candidate_replaces_the_part_and_how_joints_are_read(self):
        from gamst.recipes import load, with_pose, with_settings
        recipe = with_pose(load('person-action-count', exclude=['target_person']), 'rtmw.l_384x288')
        items = {item['name']: item for item in recipe['modules']}
        self.assertEqual((items['pose']['use'], items['pose']['candidate']), ('rtmw-pose', 'rtmw.l_384x288'))
        self.assertEqual((items['pose_channels'].get('source'), items['exemplar'].get('source')), ('rtmw', 'rtmw'))
        recipe = with_settings(recipe, ['pose.det_every_ms=500', 'pose.device=cpu'])
        items = {item['name']: item for item in recipe['modules']}
        self.assertEqual((items['pose']['det_every_ms'], items['pose']['device']), (500, 'cpu'))
        with self.assertRaises(SystemExit):
            with_settings(recipe, ['nobody.x=1'])
        with self.assertRaises(SystemExit):
            with_pose(recipe, 'no.such_model')

    def test_one_pose_model_counts_every_frame_by_default(self):
        from gamst.recipes import load, with_pose
        recipe = load('person-action-count', exclude=['target_person'])
        items = {item['name']: item for item in recipe['modules']}
        pose = items['pose']
        self.assertEqual((pose['use'], pose['candidate'], pose['backlog'], pose['batch']),
                         ('rtmw-pose', 'rtmw.l_384x288', 'spill', 8))
        self.assertNotIn('sample', pose)                                      # no frame is skipped
        self.assertEqual([item for item in recipe['modules'] if item['use'] in ('rtmw-pose', 'mediapipe-pose')], [pose])
        self.assertFalse({'verification', 'verify_pose', 'verify_exemplar'} & set(items))
        mediapipe = {item['name']: item for item in with_pose(recipe, 'mediapipe.pose_full')['modules']}
        self.assertEqual(mediapipe['pose']['use'], 'mediapipe-pose')
        self.assertIsNone(mediapipe['exemplar'].get('source'))


if __name__ == '__main__':
    unittest.main()
