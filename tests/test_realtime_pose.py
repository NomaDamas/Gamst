"""Realtime pose lanes: a lane that samples the newest frame, a lane that keeps every frame in the video memory,
the count verification lane, shared-memory frames for workers, and the recipe options that select them."""
import argparse
import asyncio
from types import SimpleNamespace
import unittest

from gamst import CapabilitySpec, Gamst, OutputField
from gamst.pipeline import Pipeline
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
        memory = VideoMemory(memory_bytes=2 * 300, disk_bytes=10 * 300)
        frames = [np.full((10, 10, 3), index, np.uint8) for index in range(5)]
        kept = [memory.keep(frame) for frame in frames]
        status = memory.status()
        self.assertEqual((status['spilled_frames'], status['in_memory_bytes'], status['spilled_bytes']), (3, 600, 900))
        for frame, item in zip(frames, kept):
            self.assertTrue(np.array_equal(memory.take(item), frame))
            memory.release(item)
        self.assertEqual((memory.in_memory, memory.spilled, memory._end), (0, 0, 0))   # the file starts over
        memory.close()

    def test_a_frame_beyond_the_disk_bound_is_dropped_not_kept(self):
        from gamst.providers.video_memory import VideoMemory
        memory = VideoMemory(memory_bytes=0, disk_bytes=300)
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
            memory = pipeline.video_memory(memory_bytes=2 * 12, disk_bytes=1 << 20)
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


SPEC = CapabilitySpec('test.events', 'test.events', 'video', 'test', '1',
                      (OutputField('count', 'integer'), OutputField('status', 'string'), OutputField('events', 'array'),
                       OutputField('reference_id', 'string', required=False)), 'native')


def frame(source, ms, events=(), **value):
    return SimpleNamespace(data=dict(events=list(events), **value), timestamp_ms=ms,
                           sensor=SimpleNamespace(sensor_id=source))


def matched(start, end, event_id=None):
    item = dict(event='exemplar.matched', start_ms=start, end_ms=end, label='slam')
    if event_id:
        item.update(state='provisional', event_id=event_id)
    return item


class CountVerificationTest(unittest.TestCase):
    def test_final_revoked_uncertain_and_added(self):
        from gamst.providers.count_verification import CountVerification
        verify = CountVerification(SPEC, live='live', verifier='check', near_ms=100)
        out = []
        for result in (
                verify.process(frame('live', 1000, [matched(800, 1000, 'a')], reference_id='r1')),
                verify.process(frame('live', 2000, [matched(1800, 2000, 'b')])),
                verify.process(frame('live', 4000, [matched(3800, 4000, 'c')])),
                verify.process(frame('check', 1040, [matched(840, 1040)])),       # pairs with a
                verify.process(frame('check', 1900)),                             # a pose there, no count: b revoked
                verify.process(frame('check', 3000, [matched(2800, 3000)])),      # nobody live: added
                verify.process(frame('check', 4500)),
                verify.finish()):                                                 # no pose during c: uncertain
            out += (result or {}).get('events', [])
        decided = {item.get('revises') or item.get('event_id'): item['state'] for item in out}
        self.assertEqual(decided['a'], 'final')
        self.assertEqual(decided['b'], 'revoked')
        self.assertEqual(decided['c'], 'uncertain')
        added = [item for item in out if item.get('decided_by') == 'verification']
        self.assertEqual([(item['end_ms'], item['state'], item['reference_id']) for item in added], [(3000, 'final', 'r1')])
        self.assertEqual(verify.status()['decided'], dict(final=1, revoked=1, uncertain=1, added=1))

    def test_nothing_is_decided_before_the_verifier_has_seen_past_it(self):
        from gamst.providers.count_verification import CountVerification
        verify = CountVerification(SPEC, live='live', verifier='check', near_ms=100)
        verify.process(frame('live', 1000, [matched(800, 1000, 'a')]))
        self.assertIsNone(verify.process(frame('check', 1050)))
        self.assertEqual(verify.finish()['events'][0]['state'], 'revoked')   # the end decides what is left


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

    def test_verification_parts_settle_the_live_exemplar(self):
        from gamst.recipes import load
        recipe = load('person-action-count', include=['verification'], exclude=['target_person'])
        items = {item['name']: item for item in recipe['modules']}
        self.assertEqual(items['verify_pose']['backlog'], 'spill')
        self.assertEqual((items['verification']['verifies'], items['verification']['verifier']), ('exemplar', 'verify_exemplar'))
        self.assertIn('verify_exemplar', [reference['name'] for reference in recipe['references']])
        plain = load('person-action-count', exclude=['target_person'])
        self.assertNotIn('verification', {item['name'] for item in plain['modules']})


if __name__ == '__main__':
    unittest.main()
