"""Body model contract tests use a native-detector fixture, not a real camera."""
import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

from humansense.plugins import InputFrame
from humansense.providers.pose import MediaPipePosePlugin
from humansense.tracking import PersonTracker


class ImageFixture:
    def __getitem__(self, key): return self
    def copy(self): return self


def points(offset=0):
    result = [SimpleNamespace(x=.2+offset+(i % 3)*.1, y=.2+(i % 5)*.1,
                              z=-.1, visibility=.9, presence=.9) for i in range(33)]
    result[11].y = result[12].y = .5
    result[15].y, result[16].y = .3, .7
    return result


def fixture_plugin(detect):
    plugin = MediaPipePosePlugin.__new__(MediaPipePosePlugin)
    plugin.max_poses, plugin.model_hash = 4, 'fixture-sha'
    plugin._native_lock = threading.Lock()
    plugin._process_lock = asyncio.Lock()
    plugin._generation, plugin._closed = 0, False
    plugin.tracker = PersonTracker()
    plugin.mp = SimpleNamespace(Image=lambda **kwargs: kwargs['data'], ImageFormat=SimpleNamespace(SRGB=1))
    plugin.detector = SimpleNamespace(detect=detect, close=lambda: None)
    return plugin


class PoseTests(unittest.IsolatedAsyncioTestCase):
    async def test_keypoints_geometry_and_continuity(self):
        plugin = fixture_plugin(lambda _: SimpleNamespace(pose_landmarks=[points()]))
        first = await plugin.process(InputFrame(ImageFixture(), 'video', 100, 'camera'))
        data = {item['name']: item['value'] for item in first}
        pose = data['scene.poses'][0]
        self.assertEqual(data['scene.pose_count'], 1)
        self.assertEqual(len(pose['landmarks']), 33)
        self.assertTrue(pose['geometry']['left_wrist_above_shoulder'])
        self.assertFalse(pose['geometry']['right_wrist_above_shoulder'])
        self.assertFalse(pose['identity_verified'])
        self.assertEqual(pose['speaker_association'], 'unavailable')
        self.assertEqual(first[0]['subject_id'], 'camera')
        self.assertIn('sha256:fixture-sha', first[0]['model_version'])
        self.assertTrue(pose['track_id'].startswith('camera:body:person-'))
        second = await plugin.process(InputFrame(ImageFixture(), 'video', 200, 'camera'))
        self.assertEqual(second[0]['value'][0]['track_id'], pose['track_id'])
        self.assertEqual(second[0]['value'][0]['association_quality'], 'matched')
        plugin.reset()
        third = await plugin.process(InputFrame(ImageFixture(), 'video', 300, 'camera'))
        self.assertNotEqual(third[0]['value'][0]['track_id'], pose['track_id'])

    async def test_missing_visibility_abstains_geometry_and_tracking(self):
        low = points()
        for p in low: p.visibility = .1
        plugin = fixture_plugin(lambda _: SimpleNamespace(pose_landmarks=[low]))
        result = await plugin.process(InputFrame(ImageFixture(), 'video', 100))
        pose = result[0]['value'][0]
        self.assertIsNone(pose['bbox'])
        self.assertIsNone(pose['track_id'])
        self.assertIsNone(pose['geometry']['left_wrist_above_shoulder'])
        self.assertIsNone(pose['geometry']['right_wrist_above_shoulder'])
        self.assertEqual(result[2]['value'], [])

    async def test_duplicate_people_do_not_share_assigned_identity(self):
        plugin = fixture_plugin(lambda _: SimpleNamespace(pose_landmarks=[points(), points()]))
        result = await plugin.process(InputFrame(ImageFixture(), 'video', 100))
        self.assertEqual(result[1]['value'], 2)
        self.assertTrue(all(p['track_id'] is None and p['association_quality'] == 'uncertain'
                            for p in result[0]['value']))

    async def test_result_bounds_and_invalid_native_output(self):
        malformed = points(); malformed[0].x = float('nan')
        plugin = fixture_plugin(lambda _: SimpleNamespace(pose_landmarks=[malformed, points()[:2], points()]))
        result = await plugin.process(InputFrame(ImageFixture(), 'video', 100))
        self.assertEqual(result[1]['value'], 1)
        self.assertEqual(result[0]['value'][0]['frame_pose_index'], 2)
        clipped = points()
        clipped[0].x, clipped[1].y = -1, 2
        box = plugin._bbox(plugin._landmarks(clipped))
        self.assertEqual(box['x'], 0)
        self.assertLessEqual(box['y']+box['height'], 1)

    async def test_reset_invalidates_native_result(self):
        entered, released = threading.Event(), threading.Event()
        def detect(_):
            entered.set()
            released.wait(2)
            return SimpleNamespace(pose_landmarks=[points()])
        plugin = fixture_plugin(detect)
        task = asyncio.create_task(plugin.process(InputFrame(ImageFixture(), 'video', 100)))
        await asyncio.to_thread(entered.wait, 1)
        self.assertTrue(entered.is_set())
        plugin.reset(); released.set()
        self.assertEqual(await task, [])
        self.assertEqual(plugin.tracker._streams, {})
        await plugin.close()
        with self.assertRaises(RuntimeError):
            await plugin.process(InputFrame(ImageFixture(), 'video', 200))

    def test_model_path_is_explicit(self):
        with self.assertRaises(ValueError):
            MediaPipePosePlugin(str(Path(__file__).with_name('missing-pose-model.task')))
