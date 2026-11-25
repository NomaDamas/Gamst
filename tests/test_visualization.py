"""What a screen draws is not what a record holds: real plugins with stand-in detectors."""
import asyncio
import json
import pickle
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from gamst import Gamst
from gamst.dashboard import Dashboard
from gamst.mcp import MCPServer
from gamst.plugins import MediaPipeFacePlugin
from gamst.realtime import QueuedPlugin
from gamst.schema import timestamp
from tests.test_pose import fixture_plugin, points

IMAGE = np.zeros((48, 64, 3), np.uint8)


def face_plugin():
    plugin = MediaPipeFacePlugin.__new__(MediaPipeFacePlugin)
    plugin.mp = SimpleNamespace(Image=lambda **kwargs: kwargs['data'], ImageFormat=SimpleNamespace(SRGB=1))
    plugin.model_hash, plugin._native_lock, plugin._closed = 'fixture-sha', threading.Lock(), False
    found = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=.123, y=.456, z=.01)] * 478],
                            face_blendshapes=[[SimpleNamespace(category_name='jawOpen', score=.5)]])
    plugin.detector = SimpleNamespace(detect=lambda image: found, close=lambda: None)
    return plugin


class VisualizationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = await Gamst(consent=True).__aenter__()
        self.face = face_plugin()
        self.pose = QueuedPlugin(self.human, fixture_plugin(lambda _: SimpleNamespace(pose_landmarks=[points()])), capacity=1)
        self.human.register(self.face).register(self.pose)
        self.dashboard = Dashboard(self.human, asyncio.get_running_loop())

    async def asyncTearDown(self):
        await self.human.close()

    async def seen(self):
        await self.human.push_video(IMAGE)
        await self.pose.task
        return await self.dashboard.dispatch({'op': 'state'})

    async def test_the_dashboard_draws_points_that_no_record_holds(self):
        data = await self.seen()
        for name, count in (('scene.faces', 478), ('scene.poses', 33)):
            self.assertEqual(len(data['guides'][name]['value'][0]['landmarks']), count)
            self.assertTrue(data['guides'][name]['fresh'])
            self.assertNotIn('landmarks', data['state']['signals'][name]['observation']['value'][0])
        self.assertEqual(data['guides']['scene.faces']['value'][0]['landmarks'][0], dict(x=.123, y=.456, z=.01))
        self.assertEqual(data['context']['observations']['scene.poses']['value'][0]['landmark_count'], 33)
        self.assertTrue(data['guides']['scene.poses']['value'][0]['geometry']['left_wrist_above_shoulder'])
        server = MCPServer(self.human)
        answers = [await server.call_tool('get_state'), await server.call_tool('get_context'),
                   await server.call_tool('inspect_interval', dict(start_ms=0, end_ms=timestamp() + 1000)),
                   await server.call_tool('get_timeline'), self.human.store.query('local-user', limit=1000)]
        for answer in answers:
            self.assertNotIn('"landmarks"', json.dumps(answer))
            self.assertNotIn('.123', json.dumps(answer))

    async def test_it_is_gone_after_a_pause_or_a_deletion_and_for_another_frame_or_subject(self):
        await self.seen()
        drawing = self.face.drawing
        stamp = (await self.human.get_state())['signals']['scene.faces']['timestamp_ms']
        self.assertIsNotNone(drawing.points('scene.faces', 'local-user', stamp))
        self.assertIsNone(drawing.points('scene.faces', 'someone', stamp))
        self.assertIsNone(drawing.points('scene.faces', 'local-user', stamp + 1))
        await self.human.delete_data()
        self.assertIsNone(drawing.points('scene.faces', 'local-user', stamp))
        self.assertEqual((await self.dashboard.dispatch({'op': 'state'}))['guides'], {})
        await self.seen()
        await self.human.pause()
        self.assertEqual(drawing._kept, {})
        self.assertEqual(self.pose.plugin.drawing._kept, {})

    async def test_it_is_bounded_and_not_serializable(self):
        for _ in range(5):
            await self.seen()
        self.assertEqual(len(self.face.drawing._kept['scene.faces']), 2)
        self.assertNotIn('.123', repr(self.face.drawing))
        with self.assertRaises(TypeError):
            pickle.dumps(self.face.drawing)

    async def test_a_signal_that_carries_points_is_refused(self):
        for value in ([dict(landmarks=[dict(x=.1, y=.2)])], dict(world_landmarks=[[.1, .2]])):
            with self.assertRaises(PermissionError):
                await self.human.push_signal('scene.poses', value, 'video')
