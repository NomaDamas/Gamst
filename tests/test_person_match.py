import json
import math
import unittest

import numpy as np

from gamst import Gamst
from gamst.pipeline import Pipeline
from gamst.plugins import InputFrame
from gamst.private_payload import PrivatePoses
from gamst.providers.person_match import ARCFACE_112, TargetPerson, body_box, face_points, umeyama
from gamst.providers.pose_exemplar import PoseChannels
from gamst.providers.sapiens2_pose import GOLIATH_BODY, Sapiens2Models, to_poses
from tests.test_speaker_verify import now

try:
    import cv2  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False


class Color:
    """Test double embedder: mean color of the crop (a person is a color in these frames)."""

    def __init__(self, model_id):
        self.model_id = model_id

    def embed(self, image):
        return [float(v) + 1e-6 for v in image.reshape(-1, 3).mean(0)]


def person(x, track, face=True):
    landmarks = [dict(x=x, y=.5, visibility=0., presence=1.) for _ in range(33)]
    if face:
        for index, dx, dy in ((5, -.04, 0), (2, .04, 0), (0, 0, .04), (10, -.03, .08), (9, .03, .08)):
            landmarks[index] = dict(x=x + dx, y=.2 + dy, visibility=1., presence=1.)
    return dict(landmarks=landmarks, bbox=dict(x=x - .1, y=.1, width=.2, height=.8), track_id=track)


def frame_of(left, right):
    image = np.zeros((400, 400, 3), np.uint8)
    image[:, :200], image[:, 200:] = left, right
    return image


RED, BLUE = (0, 0, 255), (255, 0, 0)


class GeometryTest(unittest.TestCase):
    def test_umeyama_recovers_a_similarity_and_not_a_mirror(self):
        angle, scale = .4, 1.7
        rotation = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        source = np.array(ARCFACE_112)
        matrix = umeyama(source, source @ rotation.T + [5, -3])
        np.testing.assert_allclose(matrix[:, :2], rotation, atol=1e-9)
        np.testing.assert_allclose(matrix[:, 2], [5, -3], atol=1e-9)
        mirrored = umeyama(source * [-1, 1], source)
        self.assertGreater(np.linalg.det(mirrored[:, :2]), 0)

    def test_hidden_face_points_and_tiny_boxes_abstain(self):
        self.assertIsNone(face_points(person(.3, 'a', face=False)['landmarks'], 400, 400))
        self.assertIsNone(body_box(dict(x=.5, y=.5, width=.01, height=.01), 100, 100))
        self.assertEqual(body_box(dict(x=0, y=0, width=1, height=1), 100, 80), (0, 0, 100, 80))

    def test_sapiens_output_maps_to_the_same_pose_channels(self):
        keypoints = np.zeros((1, 308, 2)) + 5
        for i, (name, index) in enumerate(GOLIATH_BODY.items()):
            keypoints[0, index] = (20 + 7 * i, 30 + (i * i) % 40)
        poses = to_poses(keypoints, np.ones((1, 308)), [[0, 0, 100, 100]], 100, 100)
        module = PoseChannels(mapping=GOLIATH_BODY, source='sapiens2', min_visibility=.3)
        from_landmarks = module.process(InputFrame(poses, 'video', 1))
        self.assertEqual(from_landmarks['visible'], 13)
        private = InputFrame(PrivatePoses(poses), 'video', 1)
        self.assertEqual(module.process(private)['channels'], from_landmarks['channels'])
        self.assertEqual(from_landmarks['image_aspect'], 1.)
        wide = to_poses(keypoints * [2, 1], np.ones((1, 308)), [[0, 0, 200, 100]], 200, 100)
        self.assertEqual((wide[0]['image_width'], wide[0]['image_height']), (200, 100))
        self.assertEqual(module.process(InputFrame(wide, 'video', 1))['image_aspect'], 2.)
        with self.assertRaisesRegex(PermissionError, 'biometric'):
            Sapiens2Models('pose', 'a' * 64, 'detector', 'b' * 64)


@unittest.skipUnless(HAS_CV, 'needs opencv')
class TargetTest(unittest.IsolatedAsyncioTestCase):
    def module(self, **options):
        module = TargetPerson(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, body_threshold=.9,
                              face_every_ms=0, body_every_ms=0, **options)
        module.set_reference(dict(face_model='face', face=[[0., 0., 1.]], body_model='body', body=[[0., 0., 1.]]))
        return module

    async def step(self, module, ts, poses, image):
        await module.process(InputFrame(poses, 'video', ts))
        out = await module.process(InputFrame(image, 'video', ts))
        return out[0]['value']

    async def test_face_locks_track_holds_then_expires_and_a_contradicting_face_releases(self):
        module = self.module(hold_ms=3000)
        module.body = None
        two = frame_of(RED, BLUE)
        value = await self.step(module, 1000, [person(.25, 'a'), person(.75, 'b')], two)
        self.assertEqual((value['track_id'], value['state']), ('a', 'face'))
        value = await self.step(module, 3500, [person(.25, 'a', face=False), person(.75, 'b')], two)
        self.assertEqual((value['track_id'], value['state']), ('a', 'held'))
        value = await self.step(module, 4500, [person(.25, 'a', face=False)], two)
        self.assertEqual(value['state'], 'absent')       # not confirmed within hold_ms
        await self.step(module, 5000, [person(.25, 'a')], two)
        value = await self.step(module, 5100, [person(.25, 'a')], frame_of(BLUE, BLUE))   # track now another person
        self.assertEqual(value['state'], 'absent')

    async def test_an_l2_fork_takes_over_the_lock_by_box_overlap_or_waits_for_a_face(self):
        live = self.module(hold_ms=3000)
        live.body = None
        two = frame_of(RED, BLUE)
        await self.step(live, 1000, [person(.25, 'a'), person(.75, 'b')], two)          # the face locks a
        cut = live.snapshot()
        self.assertEqual((cut['bbox']['x'], cut['confirmed_ms']), (.25 - .1, 1000))
        fork = live.fork()
        fork.body = None
        fork.restore(cut)
        value = await self.step(fork, 1200, [person(.26, 'x', face=False), person(.75, 'y')], two)   # new track ids
        self.assertEqual((value['track_id'], value['state']), ('x', 'held'))         # the overlapping box: held
        other = live.fork()
        other.body = None
        other.restore(dict(cut, bbox=dict(x=.5, y=.5, width=.05, height=.05)))        # nobody where the lock was
        value = await self.step(other, 1200, [person(.26, 'x', face=False)], two)
        self.assertEqual(value['state'], 'absent')                                    # unattributed until a face
        value = await self.step(other, 1300, [person(.26, 'x')], two)
        self.assertEqual((value['track_id'], value['state']), ('x', 'face'))
        await other.close()     # a fork never closes the models it shares
        self.assertIsNotNone(live.face)

    async def test_an_rtmw_shaped_pose_is_matched_by_its_own_face_points(self):
        module = self.module()
        module.body = None
        landmarks = [dict(x=.25, y=.5, visibility=0., presence=1.) for _ in range(133)]
        for index, dx, dy in ((2, -.04, 0), (1, .04, 0), (0, 0, .04), (71, -.03, .08), (77, .03, .08)):
            landmarks[index] = dict(x=.25 + dx, y=.2 + dy, visibility=.9, presence=1.)
        pose = dict(landmarks=landmarks, keypoint_set='coco_wholebody133', bbox=dict(x=.15, y=.1, width=.2, height=.8),
                    track_id='a')
        value = await self.step(module, 1000, [pose], frame_of(RED, BLUE))
        self.assertEqual((value['track_id'], value['state']), ('a', 'face'))
        wrong = dict(pose, keypoint_set=None)            # read as MediaPipe: those indices are not its face
        value = await self.step(self.module(), 1000, [dict(wrong, track_id='b')], frame_of(RED, BLUE))
        self.assertNotEqual(value['state'], 'face')

    async def test_body_bridges_without_a_face_but_not_over_another_face(self):
        module = self.module()
        value = await self.step(module, 1000, [person(.25, 'a', face=False)], frame_of(RED, BLUE))
        self.assertEqual(value['state'], 'body')
        value = await self.step(module, 2000, [person(.75, 'b', face=False)], frame_of(RED, BLUE))
        self.assertEqual(value['state'], 'absent')

    async def test_only_the_target_pose_and_crop_reach_downstream(self):
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'person')
            pipeline.add_input('camera', modality='video')
            pipeline.add_input('poses', modality='video')
            value = dict(face_model='face', face=[[0., 0., 1.]], body_model='body', body=[[0., 0., 1.]])
            with self.assertRaisesRegex(ValueError, 'consent'):
                pipeline.add_reference('person', 'person', value)
            pipeline.add_reference('person', 'person', value, consent=dict(by='owner'))
            module = TargetPerson(Color('face'), Color('body'), face_threshold=.9, face_reject=.5, emits='person:me', joints='joints:me')
            pipeline.add_module('target', module, consumes=['camera', 'poses'], emits=('person:me', 'joints:me'))
            crops, poses = [], []

            class Sink:
                def __init__(self, to):
                    self.to = to

                def process(self, frame):
                    self.to.append(frame.data)
                    return []

            pipeline.add_module('crop', Sink(crops), consumes='person:me')
            pipeline.add_module('pose', Sink(poses), consumes='joints:me')
            ts = now() - 1000
            await pipeline.push('poses', PrivatePoses([person(.25, 'a'), person(.75, 'b')]), timestamp_ms=ts)
            records = await pipeline.push('camera', frame_of(BLUE, RED), timestamp_ms=ts)
            self.assertEqual([pose['track_id'] for pose in poses[0].poses], ['b'])
            self.assertEqual(len(poses[0].poses[0]['landmarks']), 33)
            self.assertNotIn('landmarks', json.dumps(records))   # the joints never become a record
            self.assertGreater(crops[0][..., 2].mean(), 200)

    def test_embeddings_of_another_model_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'other'):
            TargetPerson(Color('face')).set_reference(dict(face_model='other', face=[[1., 0., 0.]]))


if __name__ == '__main__':
    unittest.main()
