"""Sapiens2 whole-body pose (308 Goliath keypoints), top-down over a YOLOX-m person detector. Opt-in only.

Highest-accuracy 2D whole-body option in docs/research/speaker-pose-sota-2026-11/report.md (Sapiens2 paper, Table 3,
308-kp mAP: 0.4B 76.9, 1B 80.4, 5B 82.3; Meta's own test set, no COCO-WholeBody number). The 1B backbone is 1.462 B
parameters and 4.715 TFLOPs per 1024x768 crop (model card); Meta publishes no latency. Measured in GAMST on the owner's
shotgun clip: docs/measurements/testbed-2026-11-26.md ("Sapiens2 1B").

**Licence, read before use.** Code and weights are under the Sapiens2 License
(https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md), not Apache-2.0. It forbids biometric processing
and surveillance, and gives Meta audit and deletion rights. GAMST does not bundle or download these weights: the user
downloads them from Meta's release (Hugging Face ``facebook/sapiens2-pose-<size>``, linked from the official README),
passes their local directory and sha256, and sets ``accept_license`` to say they read and accepted the licence. Use it
for movement (gesture/repetition counting) only; identity comes from :mod:`gamst.providers.person_match`, never from
Sapiens2 keypoints, and its keypoints are never used to find, align or recognise a face.

Like :mod:`gamst.providers.rtmw_pose`: rtmlib's YOLOX-m (HumanArt) finds people, Sapiens2 the keypoints in each box
(1.25x padded crop warped to 1024x768, UDP-DARK heatmap decoding, both from the transformers port
``Sapiens2ForPoseEstimation`` / ``Sapiens2ImageProcessor``, transformers 5.24.1). ``flip_test`` (default on, as the
official 1B config's ``val_cfg``) averages with the mirrored crop, at twice the cost. ``det_every_ms`` as in RTMW.
``device`` is the torch device of the pose model (``mps``, ``cuda``, ``cpu``), ``dtype`` its weights' precision;
``det_device`` runs YOLOX on ONNX Runtime ``cpu`` or ``coreml``.

``path`` is a local directory holding Meta's ``config.json``, ``preprocessor_config.json`` and ``model.safetensors``
(``sha256`` is that file's). Joint positions leave the worker only as the private payload of ``sapiens2.pose`` and
leave the module only as the derived input ``joints``, which :class:`PoseChannels` reads with ``mapping=GOLIATH_BODY``.
"""
from __future__ import annotations

from pathlib import Path

from .rtmw_pose import RTMWPosePlugin, _sha256, track_box

POINTS = 308
# Goliath-308 indices (sapiens2 ``sapiens/pose/configs/_base_/keypoints308.py`` at 7e5bae8, 36 teeth points removed).
GOLIATH_BODY = dict(left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8, left_wrist=62,
                    right_wrist=41, left_hip=9, right_hip=10, left_knee=11, right_knee=12,
                    left_ankle=13, right_ankle=14)
UPSTREAM_REVISION = '7e5bae88456ac418ff0e58e74106c9fe192055d4'
MIN_VISIBILITY = .3     # the official demo's --kpt-thr (sapiens/pose/scripts/demo/keypoints308.sh)
TRANSFORMERS_VERSION = '5.24.1'   # the Sapiens2 port read for this adapter
LICENSE_URL = 'https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md'
SIZES = ('0.4b', '0.8b', '1b', '5b')
DEVICES = ('mps', 'cuda', 'cpu')
DTYPES = ('float32', 'float16', 'bfloat16')
DET_DEVICES = ('cpu', 'coreml')
COMPUTE_UNITS = ('ALL', 'CPUAndGPU', 'CPUOnly', 'CPUAndNeuralEngine')
KNOWN = frozenset({'path', 'sha256', 'det_path', 'det_sha256', 'accept_license', 'device', 'dtype', 'det_device',
                   'compute_units', 'det_every_ms', 'max_poses', 'det_score', 'flip_test'})


def check_config(config):
    """The worker's settings, refused before anything starts (the licence first)."""
    if config.get('accept_license') is not True:
        raise PermissionError(f'the Sapiens2 License forbids biometric processing and surveillance and gives Meta audit '
                              f'rights ({LICENSE_URL}): read it and set accept_license to opt in')
    if set(config) - KNOWN or not {'path', 'sha256', 'det_path', 'det_sha256'} <= set(config):
        raise ValueError(f'sapiens2.pose needs path, sha256, det_path, det_sha256, accept_license; '
                         f'optional {sorted(KNOWN)}')
    if (config.get('device', 'mps') not in DEVICES or config.get('dtype', 'float32') not in DTYPES
            or config.get('det_device', 'coreml') not in DET_DEVICES
            or config.get('compute_units', 'CPUAndGPU') not in COMPUTE_UNITS):
        raise ValueError(f'sapiens2.pose device is one of {DEVICES}, dtype one of {DTYPES}, det_device one of '
                         f'{DET_DEVICES}, compute_units one of {COMPUTE_UNITS}')
    every = config.get('det_every_ms', 0)
    if type(every) is not int or not 0 <= every <= 10000:
        raise ValueError('det_every_ms is integer milliseconds 0..10000 (0: every frame)')
    poses = config.get('max_poses', 4)
    if type(poses) is not int or not 1 <= poses <= 8:      # the private payload carries at most 8 people
        raise ValueError('max_poses is an integer 1..8')
    if type(config.get('flip_test', True)) is not bool:
        raise ValueError('flip_test is true or false')


def to_poses(keypoints, scores, boxes, width, height):
    """Pixel keypoints (n, 308, 2) + scores (n, 308) + xyxy boxes -> scene.poses dicts (normalized)."""
    poses = []
    for index, (points, values, box) in enumerate(zip(keypoints, scores, boxes)):
        landmarks = [dict(x=float(x) / width, y=float(y) / height, visibility=float(min(1., max(0., score))),
                          presence=1.) for (x, y), score in zip(points, values)]
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = max(0., x0 / width), min(1., x1 / width)
        y0, y1 = max(0., y0 / height), min(1., y1 / height)
        bbox = dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0) if x1 > x0 and y1 > y0 else None
        poses.append(dict(frame_pose_index=index, landmarks=landmarks, bbox=bbox, image_width=int(width),
                          image_height=int(height), keypoint_set='goliath308', track_id=None,
                          association_quality='unavailable', identity_verified=False,
                          speaker_association='unavailable'))
    return poses


class Sapiens2Models:
    """The worker side (``sapiens2`` environment): rtmlib's YOLOX session and the Sapiens2 pose model."""
    points = POINTS

    def __init__(self, path, sha256, det_path, det_sha256, *, accept_license=False, device='mps', dtype='float32',
                 det_device='coreml', compute_units='CPUAndGPU', det_every_ms=0, max_poses=4, det_score=None,
                 flip_test=True):
        check_config(dict(path=path, sha256=sha256, det_path=det_path, det_sha256=det_sha256,
                          accept_license=accept_license, device=device, dtype=dtype, det_device=det_device,
                          compute_units=compute_units, det_every_ms=det_every_ms, max_poses=max_poses,
                          flip_test=flip_test))
        weights = Path(path) / 'model.safetensors'
        for file, expected in ((weights, sha256), (Path(det_path), det_sha256)):
            if not file.is_file() or _sha256(file) != expected:
                raise ValueError(f'{file} is missing or does not match its pinned sha256')
        from importlib.metadata import version
        if version('transformers') != TRANSFORMERS_VERSION:
            raise RuntimeError(f'the Sapiens2 adapter was written against transformers=={TRANSFORMERS_VERSION}')
        import onnxruntime as ort
        import torch
        from rtmlib import YOLOX
        from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor
        self.detector = YOLOX(str(det_path), model_input_size=(640, 640), backend='onnxruntime', device='cpu')
        if det_score is not None:
            self.detector.score_thr = float(det_score)
        if det_device == 'coreml':
            providers = [('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': compute_units}),
                         'CPUExecutionProvider']
            self.detector.session = ort.InferenceSession(str(det_path), providers=providers)
        self.torch, self.device = torch, torch.device(device)
        self.processor = Sapiens2ImageProcessor.from_pretrained(str(path), local_files_only=True)
        self.pose = Sapiens2ForPoseEstimation.from_pretrained(str(path), local_files_only=True,
                                                              dtype=getattr(torch, dtype)).to(self.device).eval()
        self.dtype = getattr(torch, dtype)
        self.flip_pairs = torch.tensor(self.pose.config.flip_pairs, device=self.device) if flip_test else None
        self.det_every_ms, self.max_poses = int(det_every_ms), int(max_poses)
        self.version = (f'sapiens2:{sha256[:16]}:yolox:{det_sha256[:16]}:{device}:{dtype}' + ('' if flip_test else ':noflip')
                        + (f':det{det_every_ms}' if det_every_ms else ''))
        self.reset()

    def reset(self):
        self._boxes, self._detected_ms = None, None

    def keypoints(self, image, boxes):
        """BGR image + xyxy boxes -> keypoints (n, 308, 2) pixels, scores (n, 308)."""
        import numpy as np
        torch = self.torch
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        coco = [[[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in boxes]]
        inputs = self.processor(rgb, boxes=coco, return_tensors='pt')
        pixels = inputs['pixel_values'].to(self.device, self.dtype)
        with torch.inference_mode():
            outputs = self.pose(pixel_values=pixels)
            flipped = None if self.flip_pairs is None else self.pose(pixel_values=pixels.flip(-1),
                                                                     flip_pairs=self.flip_pairs)
            found = self.processor.post_process_pose_estimation(outputs, boxes=coco, outputs_flipped=flipped)[0]
        keypoints = np.stack([item['keypoints'].float().cpu().numpy() for item in found]).astype('f4')
        scores = np.stack([item['scores'].float().cpu().numpy() for item in found]).astype('f4')
        return keypoints, scores

    def infer(self, image, timestamp_ms=None):
        """BGR image -> (boxes xyxy, keypoints (n, 308, 2) pixels, scores (n, 308)); the RTMW worker's answer."""
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
        keypoints, scores = self.keypoints(image, boxes)
        if self.det_every_ms:
            tracked = [track_box(points, values, width, height, MIN_VISIBILITY) for points, values in zip(keypoints, scores)]
            self._boxes = [box for box in tracked if box is not None] or None
        return boxes, keypoints, scores


class Sapiens2PosePlugin(RTMWPosePlugin):
    """The module in this process: the models answer from their own environment (``RemotePose``)."""
    name = 'sapiens2_pose'
