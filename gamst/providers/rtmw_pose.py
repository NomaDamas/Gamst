"""RTMW whole-body pose (133 COCO-WholeBody keypoints, hands included) through rtmlib's ONNX models.

The usable state of the art for 2D whole-body pose in docs/research/speaker-pose-sota-2026-11/report.md: RTMW-x
384x288 is COCO-WholeBody 70.2 AP, RTMW-l 384x288 70.1 AP (mmpose RTMPose project table). OpenMMLab publishes an ONNX
export of RTMW-l 384x288 only (``rtmw-dw-x-l_simcc-cocktail14_270e-384x288``, rtmlib's ``performance`` whole-body
model); RTMW-x 384x288 is a ``.pth`` checkpoint without an ONNX file. Code: rtmlib and mmpose, Apache-2.0.

Top-down: a YOLOX-m person detector (HumanArt, rtmlib's ``performance`` detector) gives boxes, RTMW the keypoints
in each box. ``det_every_ms`` 0 (default) detects on every frame, as rtmlib's ``Wholebody`` does; a positive value
detects at most that often and in between uses the box around the previous frame's body keypoints (rtmlib
``PoseTracker``), which is cheaper but follows a cut or a new person only at the next detection.

``device``: ``cpu`` (ONNX Runtime CPU) or ``coreml`` (ONNX Runtime CoreML execution provider, ML Program format, on
the GPU and CPU: ``compute_units``; macOS). Joint positions leave the worker only as the private payload of
``rtmw.pose`` and leave the module only as the derived input ``joints`` (:class:`PoseChannels` reads them with
``mapping=COCO_WHOLEBODY_BODY``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..plugins import PerceptionPlugin
from ..tracking import create_tracker
from .pose import associate, committed

POINTS = 133
# COCO-WholeBody: the first 17 points are COCO-17 (then 6 feet, 68 face, 21 left hand, 21 right hand).
COCO_WHOLEBODY_BODY = dict(left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8, left_wrist=9,
                           right_wrist=10, left_hip=11, right_hip=12, left_knee=13, right_knee=14,
                           left_ankle=15, right_ankle=16)
MIN_VISIBILITY = .3     # rtmlib's keypoint threshold (``format_result``, ``draw_skeleton``)
DEVICES = ('cpu', 'coreml')
COMPUTE_UNITS = ('ALL', 'CPUAndGPU', 'CPUOnly', 'CPUAndNeuralEngine')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_config(config):
    """The worker's settings, refused before anything starts."""
    known = {'path', 'sha256', 'det_path', 'det_sha256', 'device', 'compute_units', 'det_every_ms', 'max_poses',
             'det_score', 'input_size'}
    if set(config) - known or not {'path', 'sha256', 'det_path', 'det_sha256'} <= set(config):
        raise ValueError(f'rtmw.pose needs path, sha256, det_path, det_sha256; optional {sorted(known)}')
    if config.get('device', 'cpu') not in DEVICES or config.get('compute_units', 'CPUAndGPU') not in COMPUTE_UNITS:
        raise ValueError(f'rtmw.pose device is one of {DEVICES}, compute_units one of {COMPUTE_UNITS}')
    every = config.get('det_every_ms', 0)
    if type(every) is not int or not 0 <= every <= 10000:
        raise ValueError('det_every_ms is integer milliseconds 0..10000 (0: every frame)')


def track_box(points, scores, width, height, threshold=MIN_VISIBILITY, scale=1.25):
    """The box around a person's visible keypoints, enlarged as rtmlib ``PoseTracker`` does; None when too few."""
    visible = [(float(x), float(y)) for (x, y), score in zip(points, scores) if score >= threshold]
    if len(visible) < 4:
        return None
    xs, ys = [p[0] for p in visible], [p[1] for p in visible]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w, half_h = (max(xs) - min(xs)) * scale / 2, (max(ys) - min(ys)) * scale / 2
    box = [max(0., cx - half_w), max(0., cy - half_h), min(float(width), cx + half_w), min(float(height), cy + half_h)]
    return box if box[2] - box[0] > 4 and box[3] - box[1] > 4 else None


class RTMWModels:
    """The worker side: rtmlib's YOLOX and RTMPose sessions (in the ``rtmlib`` environment)."""

    def __init__(self, path, sha256, det_path, det_sha256, *, device='cpu', compute_units='CPUAndGPU', det_every_ms=0,
                 max_poses=4, det_score=None, input_size=(288, 384)):
        check_config(dict(path=path, sha256=sha256, det_path=det_path, det_sha256=det_sha256, device=device,
                          compute_units=compute_units, det_every_ms=det_every_ms))
        for weights, expected in ((path, sha256), (det_path, det_sha256)):
            if not Path(weights).is_file() or _sha256(weights) != expected:
                raise ValueError(f'{weights} is missing or does not match its pinned sha256')
        import onnxruntime as ort
        from rtmlib import RTMPose, YOLOX
        self.detector = YOLOX(str(det_path), model_input_size=(640, 640), backend='onnxruntime', device='cpu')
        if det_score is not None:
            self.detector.score_thr = float(det_score)
        self.pose = RTMPose(str(path), model_input_size=tuple(input_size), backend='onnxruntime', device='cpu')
        if device == 'coreml':
            providers = [('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': compute_units}),
                         'CPUExecutionProvider']
            self.detector.session = ort.InferenceSession(str(det_path), providers=providers)
            self.pose.session = ort.InferenceSession(str(path), providers=providers)
        self.det_every_ms, self.max_poses = int(det_every_ms), int(max_poses)
        self.version = f'rtmw:{sha256[:16]}:yolox:{det_sha256[:16]}:{device}' + (f':det{det_every_ms}' if det_every_ms else '')
        self.reset()

    def reset(self):
        self._boxes, self._detected_ms = None, None

    def infer(self, image, timestamp_ms=None):
        """BGR image -> (boxes xyxy, keypoints (n, 133, 2) pixels, scores (n, 133))."""
        import numpy as np
        height, width = image.shape[:2]
        due = (self._boxes is None or not self.det_every_ms or timestamp_ms is None or self._detected_ms is None
               or timestamp_ms - self._detected_ms >= self.det_every_ms or timestamp_ms < self._detected_ms)
        if due:
            boxes = [list(map(float, box)) for box in self.detector(image)][:self.max_poses]
            self._detected_ms = timestamp_ms
        else:
            boxes = self._boxes
        if not boxes:
            self._boxes = None if due else self._boxes
            return [], np.zeros((0, POINTS, 2), 'f4'), np.zeros((0, POINTS), 'f4')
        keypoints, scores = self.pose(image, bboxes=boxes)
        if self.det_every_ms:
            tracked = [track_box(points[:17], values[:17], width, height) for points, values in zip(keypoints, scores)]
            self._boxes = [box for box in tracked if box is not None] or None
        return boxes, np.asarray(keypoints, 'f4'), np.asarray(scores, 'f4')


def to_poses(boxes, keypoints, scores, width, height):
    """Pixel results -> scene.poses dicts with normalized landmarks (``keypoint_set`` coco_wholebody133)."""
    poses = []
    for index, (box, points, values) in enumerate(zip(boxes, keypoints, scores)):
        landmarks = [dict(x=float(x) / width, y=float(y) / height, visibility=float(min(1., max(0., score))),
                          presence=1.) for (x, y), score in zip(points, values)]
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = max(0., x0 / width), min(1., x1 / width)
        y0, y1 = max(0., y0 / height), min(1., y1 / height)
        poses.append(dict(frame_pose_index=index, landmarks=landmarks, keypoint_set='coco_wholebody133',
                          bbox=dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0) if x1 > x0 and y1 > y0 else None,
                          image_width=int(width), image_height=int(height), track_id=None,
                          association_quality='unavailable', identity_verified=False,
                          speaker_association='unavailable'))
    return poses


class RTMWPosePlugin(PerceptionPlugin):
    """The module in this process: the models answer from their own environment (``RemoteRTMW``)."""
    name = 'rtmw_pose'
    modalities = ('video',)
    owns_process = True

    def __init__(self, remote, *, joints=None, tracker='ocsort'):
        self.remote, self.joints, self.version = remote, joints, remote.model_id
        self.tracker, self.tracker_name = create_tracker(tracker), tracker

    async def process(self, frame):
        poses = await self.remote.infer(frame.data, frame.timestamp_ms)
        tracks = associate(poses, self.tracker, frame)
        return committed(frame, poses, tracks, self.name, self.version, self.tracker_name, self.joints)

    async def reset(self):
        self.tracker.reset()
        await self.remote.reset()

    async def close(self):
        await self.remote.close()
