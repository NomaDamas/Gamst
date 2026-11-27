"""Worker-private payloads: what crosses to the owner, and that nothing of it can be committed."""
import asyncio
import copy
import json
from pathlib import Path
import pickle
import stat
import sys
import tempfile
import textwrap
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gamst import Gamst, ProcessProvider, SensorMetadata, SensorStreams
from gamst.capabilities import CapabilitySpec, OutputField, bounded_json, is_private_key, is_raw_key
from gamst.context import compact_observation
from gamst.crew.modules import RoleUnavailable, WorkerModule, role_module
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.private_payload import CONTRACTS, PRIVATE_ONLY, PROTOCOL, Private, PrivatePoses, expected_size, pack, unpack
from gamst.schema import timestamp

NAMES = ('keypoints', 'keypoint', 'key_points', 'keyPoints', 'KEY-POINTS', 'keypoints_2d', 'joints', 'joint_positions',
         'skeleton', 'pose_landmarks', 'left_hand_landmarks', 'face_landmarks', 'embedding', 'embeddings', 'voiceprint',
         'waveform', 'pcm', 'landmarks', 'world_landmarks')
SHA = '0' * 64
WORKER = '''
import sys
from types import SimpleNamespace
import numpy as np
from gamst.providers.landmarker_video import MediaPipeVideoContext
mode = "MODE"
class Detector:
    def detect_for_video(self, image, stamp):
        return SimpleNamespace(pose_landmarks=[[SimpleNamespace(x=i / 64, y=.25, z=0., visibility=.5, presence=.75)
                                                for i in range(33)]])
    def close(self): pass
def from_local(kind, path, sha256, **options):
    api = SimpleNamespace(Image=lambda **kw: kw["data"], ImageFormat=SimpleNamespace(SRGB=1))
    provider = MediaPipeVideoContext(kind, Detector, api, sha256=sha256, **options)
    original = provider.query
    def query(frame, request=None):
        value, joints = original(frame, request)
        return value, dict(honest=joints, shape=joints[:, :20], more=np.repeat(joints, 2, axis=0),
                           huge=joints * 1e6, nan=joints * np.nan)[mode]
    provider.query = query
    return provider
MediaPipeVideoContext.from_local = staticmethod(from_local)
from gamst.provider_worker import main
main()
'''


class GuardTest(unittest.TestCase):
    def test_joint_identity_and_waveform_names_are_refused(self):
        for name in NAMES:
            self.assertTrue(is_raw_key(name), name)
            with self.assertRaisesRegex(ValueError, 'raw media'):
                bounded_json(dict(a=[{name: [1]}]), forbid_raw=True)
            with self.assertRaises(ValueError):
                OutputField(name.replace(' ', ''), 'array')
            self.assertNotIn(name, compact_observation('x', {name: [.5], 'kept': 1}))
        for name in ('joint_count', 'visible_joint_count', 'keypoint_set', 'landmark_count', 'embedding_model'):
            self.assertFalse(is_private_key(name), name)

    def test_neither_a_registered_nor_an_unregistered_record_can_carry_them(self):
        spec = CapabilitySpec('test.private', 'test.private', 'video', 'test', '1', (OutputField('value', 'object'),), 'frame')

        async def run():
            async with Gamst(consent=True) as human:
                human.register_capability(spec)
                sensor = SensorMetadata('s', 'c', 0)
                for value in [{name: [1.]} for name in NAMES] + [PrivatePoses([]), Private('joints', None)]:
                    with self.assertRaises((ValueError, TypeError)):
                        await human.push_observation(spec.id, dict(value=value), sensor=sensor, timestamp_ms=timestamp())
                    with self.assertRaises(PermissionError):
                        await human.push_signal('scene.poses', [dict(deep=[value])], 'video')
                return human.store.query('local-user')
        self.assertEqual(asyncio.run(run()), [])

    def test_a_private_payload_is_not_json_and_cannot_be_copied(self):
        for value in (Private('joints', [.123]), PrivatePoses([dict(landmarks=[.123])])):
            for action in (json.dumps, pickle.dumps, copy.copy, copy.deepcopy, bounded_json):
                with self.assertRaises((TypeError, ValueError)):
                    action(value)
            self.assertNotIn('.123', repr(value))
        holder = Private('embedding', [1.])
        with self.assertRaises(PermissionError):
            holder.take('joints')


@unittest.skipIf(np is None, 'needs numpy')
class ContractTest(unittest.TestCase):
    def test_shapes_bounds_and_the_public_part_are_checked_on_both_sides(self):
        block = dict(layout=dict(shape=[160]))
        good = [('mediapipe.body.video', dict(poses=[{}]), np.zeros((1, 33, 4))), ('sapiens2.pose', dict(poses=[{}, {}]), np.zeros((2, 308, 3))),
                ('face.embedding', dict(usable=True), np.ones(512)), ('speaker.embedding', dict(usable=False), np.zeros(0)),
                ('target_speech.local', dict(samples_returned=160), np.zeros(160)), ('target_speech.local', dict(samples_returned=0), np.zeros(0)),
                ('rtmw.pose', dict(poses=[{}]), np.zeros((1, 133, 3))), ('rtmw.pose', dict(poses=[{}]), np.zeros((1, 133, 4)))]
        for profile, public, value in good:
            descriptor, payload = pack(profile, public, value, block)
            array = unpack(profile, public, descriptor, payload, block).take(CONTRACTS[profile].kind)
            self.assertEqual(array.shape, value.shape)
        bad = [('mediapipe.body.video', dict(poses=[{}]), np.zeros((1, 32, 4))), ('mediapipe.body.video', dict(poses=[]), np.zeros((1, 33, 4))),
               ('mediapipe.body.video', dict(poses=[{}] * 5), np.zeros((5, 33, 4))), ('sapiens2.pose', dict(poses=[{}]), np.zeros((1, 133, 3))),
               ('face.embedding', dict(usable=True), np.ones(5000)), ('face.embedding', dict(usable=True), np.ones(3)),
               ('face.embedding', dict(usable=False), np.ones(512)), ('face.embedding', dict(usable=True), np.full(512, np.inf)),
               ('target_speech.local', dict(samples_returned=161), np.zeros(161)), ('target_speech.local', dict(samples_returned=160), np.full(160, 1e5)),
               ('rtmw.pose', dict(poses=[{}]), np.zeros((1, 133, 5))), ('sapiens2.pose', dict(poses=[{}]), np.zeros((1, 308, 4)))]
        for profile, public, value in bad:
            with self.assertRaises(ValueError):
                pack(profile, public, value, block)
        for descriptor in (None, {}, dict(protocol='x', kind='joints', shape=[1, 33, 4]),
                           dict(protocol=PROTOCOL, kind='embedding', shape=[512]),
                           dict(protocol=PROTOCOL, kind='joints', shape=[1, 33, 4], extra=1),
                           dict(protocol=PROTOCOL, kind='joints', shape=[True, 33, 4]),
                           dict(protocol=PROTOCOL, kind='joints', shape=[-1, 33, 4])):
            with self.assertRaises(ValueError):
                expected_size('mediapipe.body.video', descriptor, block)


@unittest.skipIf(np is None, 'needs numpy')
class WorkerTest(unittest.IsolatedAsyncioTestCase):
    """The real worker process and wire; MediaPipe itself is a stand-in (no model is loaded)."""

    def python(self, mode='honest'):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script, wrapper = Path(directory.name) / 'worker.py', Path(directory.name) / 'python'
        script.write_text(textwrap.dedent(WORKER).replace('MODE', mode))
        root = Path(__file__).resolve().parents[1]
        wrapper.write_text(f'#!/bin/sh\nPYTHONPATH="{root}" exec "{sys.executable}" -u "{script}"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        return str(wrapper)

    async def open(self, mode='honest', **options):
        provider = await ProcessProvider.open(self.python(mode), 'mediapipe.body.video', dict(path='/m', sha256=SHA), **options)
        self.addAsyncCleanup(provider.close)
        return provider

    def frame(self, sequence=0):
        return InputFrame(np.zeros((64, 96, 3), np.uint8), 'video', timestamp() - 500 + sequence,
                          sensor=SensorMetadata('cam', 'video', sequence))

    async def test_the_owner_gets_the_joints_beside_a_public_value_without_them(self):
        worker = await self.open(private=True)
        for sequence in range(2):
            public, private = await worker.query(self.frame(sequence))
            joints = private.take('joints')
            self.assertEqual(joints.shape, (1, 33, 4))
            np.testing.assert_allclose(joints[0, 8], [8 / 64, .25, .5, .75])
            self.assertEqual(public['summary']['image_width'], 96)
            self.assertNotIn('0.125', json.dumps(public['poses'][0]['geometry']))
            for key in json.dumps(public).split('"'):
                self.assertFalse(is_raw_key(key) and key != 'raw', key)
        for call in (worker.process(self.frame(2)), worker.process_regions(self.frame(2), {}), worker.finish(), worker.endpoint()):
            with self.assertRaises(PermissionError):
                await call

    async def test_a_public_worker_has_no_query_and_the_same_contract_as_before(self):
        worker = await self.open()
        self.assertFalse(worker.private)
        with self.assertRaises(PermissionError):
            await worker.query(self.frame())
        self.assertNotIn('joints', await worker.process(self.frame()))

    async def test_a_dishonest_worker_is_refused(self):
        for mode in ('shape', 'more', 'huge', 'nan'):
            worker = await self.open(mode, private=True)
            with self.assertRaises((RuntimeError, ValueError)):
                await worker.query(self.frame())
            self.assertTrue(worker.closed)

    async def test_private_workers_are_refused_wherever_output_is_committed(self):
        worker = await self.open(private=True)
        async with Gamst(consent=True) as human:
            with self.assertRaises(PermissionError):
                WorkerModule(worker)
            with self.assertRaises(PermissionError):
                human.register(worker)
            pipeline = Pipeline(human, 'p')
            pipeline.add_input('camera')
            with self.assertRaises(PermissionError):
                pipeline.add_module('pose', worker, consumes=['camera'], spec=worker.spec)
            streams = SensorStreams(human)
            streams.register(worker.spec, lambda: worker)
            with self.assertRaises(PermissionError):
                await streams.open('cam', 'video', [worker.spec.id])
        with self.assertRaises(PermissionError):
            await ProcessProvider.open(sys.executable, 'mediapipe.hand.video', dict(path='/m', sha256=SHA), private=True)
        for profile in PRIVATE_ONLY:
            with self.assertRaises(PermissionError):
                await ProcessProvider.open(sys.executable, profile, {})
            with self.assertRaises(RoleUnavailable):
                await role_module(None, 'vision.pose', dict(model='x', provider_profile=profile, python=sys.executable, config={}))


if __name__ == '__main__':
    unittest.main()
