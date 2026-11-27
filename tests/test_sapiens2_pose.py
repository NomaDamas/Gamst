"""Sapiens2 pose over YOLOX: the opt-in licence, its settings, joints read by name like RTMW's, and the worker's answer.

No weights are needed. The decoding test runs the real transformers ``Sapiens2ImageProcessor`` on a made-up heatmap and
is skipped when torch or that transformers port is absent.
"""
import unittest

from gamst.providers.model_workers import capability, model_id
from gamst.providers.sapiens2_pose import GOLIATH_BODY, POINTS, Sapiens2Models, check_config, to_poses
from gamst.providers.rtmw_pose import COCO_WHOLEBODY_BODY
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    from transformers import Sapiens2ImageProcessor
except ImportError:
    torch = None

SHA = 'a' * 64
CONFIG = dict(path='/weights/sapiens2.pose_1b', sha256=SHA, det_path='/weights/yolox.onnx', det_sha256='b' * 64,
              accept_license=True)


class ConfigTest(unittest.TestCase):
    def test_the_licence_is_refused_before_anything_else(self):
        for config in ({}, dict(CONFIG, accept_license=False), {k: v for k, v in CONFIG.items() if k != 'accept_license'},
                       dict(CONFIG, accept_license='yes')):
            with self.assertRaisesRegex(PermissionError, 'biometric'):
                check_config(config)
        with self.assertRaisesRegex(PermissionError, 'Sapiens2'):
            capability('sapiens2.pose', {k: v for k, v in CONFIG.items() if k != 'accept_license'})

    def test_settings_are_checked(self):
        check_config(CONFIG)
        check_config(dict(CONFIG, device='cpu', dtype='bfloat16', det_device='cpu', det_every_ms=500, max_poses=8,
                          flip_test=False, compute_units='ALL', det_score=.4))
        for bad in (dict(device='coreml'), dict(dtype='int8'), dict(det_device='mps'), dict(det_every_ms=-1),
                    dict(det_every_ms=.5), dict(max_poses=9), dict(max_poses=0), dict(flip_test=1),
                    dict(compute_units='GPU'), dict(pose_dir='/p')):
            with self.assertRaises(ValueError, msg=bad):
                check_config(dict(CONFIG, **bad))
        with self.assertRaises(ValueError):
            check_config({k: v for k, v in CONFIG.items() if k != 'det_path'})
        self.assertEqual(model_id('sapiens2.pose', CONFIG), f'sapiens2:{SHA[:16]}')
        self.assertEqual(capability('sapiens2.pose', CONFIG).id, 'model.sapiens2.pose')

    def test_the_weights_are_never_downloaded_or_applied_by_gamst(self):
        from gamst.crew.registry import ROLES
        candidate = ROLES['vision.pose'].candidates['sapiens2.1b']
        self.assertFalse(candidate.automatic)
        self.assertEqual(candidate.provider_profile, 'sapiens2.pose')
        self.assertIn('biometric', candidate.license)
        self.assertEqual(candidate.weights[0].sha256,
                         '2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f')
        with self.assertRaisesRegex(PermissionError, 'biometric'):
            Sapiens2Models('/nowhere', SHA, '/nowhere.onnx', SHA)


@unittest.skipIf(np is None, 'numpy')
class JointsTest(unittest.TestCase):
    def test_the_rule_reads_the_same_joints_as_from_rtmw(self):
        """hit_10_3.json names right_wrist/right_shoulder: both models give the same channels for the same body."""
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        from gamst.providers.pose_exemplar import PoseChannels
        from gamst.providers.rtmw_pose import to_poses as rtmw_poses
        self.assertEqual(set(GOLIATH_BODY), set(COCO_WHOLEBODY_BODY))
        sapiens, rtmw = np.full((1, POINTS, 2), 7.), np.full((1, 133, 2), 7.)
        for i, name in enumerate(sorted(GOLIATH_BODY)):
            sapiens[0, GOLIATH_BODY[name]] = rtmw[0, COCO_WHOLEBODY_BODY[name]] = (40 + 9 * i, 20 + (7 * i * i) % 90)
        box = [[0, 0, 160, 120]]
        ours = to_poses(sapiens, np.full((1, POINTS), 1.4), box, 160, 120)   # heatmap peaks above 1 are clipped
        self.assertEqual((ours[0]['keypoint_set'], ours[0]['landmarks'][0]['visibility']), ('goliath308', 1.))
        theirs = rtmw_poses(box, rtmw, np.ones((1, 133)), 160, 120)
        a = PoseChannels(mapping=GOLIATH_BODY, source='sapiens2', min_visibility=.3).process(
            InputFrame(PrivatePoses(ours), 'video', 1))
        b = PoseChannels(mapping=COCO_WHOLEBODY_BODY, source='rtmw', min_visibility=.3).process(
            InputFrame(PrivatePoses(theirs), 'video', 1))
        self.assertEqual((a['channels'], a['mask'], a['image_aspect']), (b['channels'], b['mask'], b['image_aspect']))

    def test_the_pose_option_reads_sapiens2_joints_its_way(self):
        from gamst.recipes import load, with_pose
        from gamst.recipes.parts import _pose_source
        recipe = with_pose(load('person-action-count', exclude=['target_person']), 'sapiens2.1b')
        items = {item['name']: item for item in recipe['modules']}
        self.assertEqual((items['pose']['use'], items['pose']['candidate']), ('sapiens2-pose', 'sapiens2.1b'))
        self.assertEqual((items['pose_channels'].get('source'), items['exemplar'].get('source')), ('sapiens2', 'sapiens2'))
        self.assertEqual(_pose_source(items['exemplar']), dict(mapping=GOLIATH_BODY, min_visibility=.3))


@unittest.skipIf(np is None, 'numpy')
class WorkerTest(unittest.TestCase):
    def models(self, every_ms):
        """The worker class with stand-ins for YOLOX and the pose model."""
        models = object.__new__(Sapiens2Models)
        models.det_every_ms, models.max_poses, models.detected, models.version = every_ms, 2, 0, 'stand-in'

        def detector(image):
            models.detected += 1
            return [[10., 10., 60., 110.], [70., 10., 120., 110.], [0., 0., 5., 5.]]

        def keypoints(image, boxes):
            points = np.zeros((len(boxes), POINTS, 2), 'f4')
            for index, (x0, y0, x1, y1) in enumerate(boxes):
                points[index, :, 0] = np.linspace(x0 + 5, x1 - 5, POINTS)
                points[index, :, 1] = np.linspace(y0 + 5, y1 - 5, POINTS)
            return points, np.full((len(boxes), POINTS), .9, 'f4')
        models.detector, models.keypoints = detector, keypoints
        models.reset()
        return models

    def test_every_frame_and_interval_detection(self):
        image = np.zeros((120, 160, 3), np.uint8)
        models = self.models(0)
        for t in (0, 17, 33):
            boxes, points, scores = models.infer(image, t)
        self.assertEqual((models.detected, len(boxes), points.shape, scores.shape), (3, 2, (2, POINTS, 2), (2, POINTS)))
        models = self.models(500)
        for t in (0, 17, 33, 499):
            boxes, _, _ = models.infer(image, t)
        self.assertEqual(models.detected, 1)    # between detections: boxes around the last keypoints
        self.assertEqual(boxes[0][0::2], [10.0, 60.0])
        models.infer(image, 500)
        self.assertEqual(models.detected, 2)

    def test_nobody_is_an_empty_answer_of_308_points(self):
        models = self.models(0)
        models.detector = lambda image: []
        boxes, points, scores = models.infer(np.zeros((8, 8, 3), np.uint8), 0)
        self.assertEqual((boxes, points.shape, scores.shape), ([], (0, POINTS, 2), (0, POINTS)))
        from gamst.providers.model_workers import _RTMW
        from gamst.private_payload import CONTRACTS
        frame = type('F', (), dict(data=np.zeros((8, 8, 3), np.uint8)))()
        public, joints = _RTMW(models).query(frame, {'timestamp_ms': 0})
        self.assertEqual((public['poses'], joints.shape), ([], (0, POINTS, 3)))
        CONTRACTS['sapiens2.pose'].check_shape(list(joints.shape), None)


@unittest.skipIf(torch is None, 'torch and transformers with the Sapiens2 port')
class DecodingTest(unittest.TestCase):
    def test_a_heatmap_peak_lands_where_it_was_in_the_image(self):
        """The real processor crops a padded box to 1024x768 and maps the heatmap peak back to image pixels."""
        processor = Sapiens2ImageProcessor(size=dict(height=1024, width=768), do_rescale=True,
                                           rescale_factor=1 / 255, do_normalize=True,
                                           image_mean=[.485, .456, .406], image_std=[.229, .224, .225])
        image = np.zeros((720, 1280, 3), np.uint8)
        box = [400., 100., 700., 700.]           # xyxy, 300 x 600
        target = (550., 250.)                    # a point inside the box
        models = object.__new__(Sapiens2Models)
        models.torch, models.device, models.dtype, models.processor = torch, torch.device('cpu'), torch.float32, processor
        models.flip_pairs = None

        class Pose:
            def __call__(self, pixel_values, flip_pairs=None):
                self.shape = tuple(pixel_values.shape)
                heatmaps = torch.zeros(1, POINTS, 256, 192)
                # where target falls in the 1.25x padded, 3:4 crop (scale 375 x 750, centred on the box)
                hx = (target[0] - (550 - 375 / 2)) / 375 * 191
                hy = (target[1] - (400 - 750 / 2)) / 750 * 255
                ys, xs = torch.meshgrid(torch.arange(256.), torch.arange(192.), indexing='ij')
                heatmaps[0, :] = torch.exp(-((xs - hx) ** 2 + (ys - hy) ** 2) / 8)
                return type('Out', (), dict(heatmaps=heatmaps))()
        models.pose = Pose()
        points, scores = models.keypoints(image, [box])
        self.assertEqual(models.pose.shape, (1, 3, 1024, 768))
        self.assertEqual((points.shape, scores.shape), ((1, POINTS, 2), (1, POINTS)))
        self.assertLess(abs(points[0, 41, 0] - target[0]) + abs(points[0, 41, 1] - target[1]), 3.)


if __name__ == '__main__':
    unittest.main()
