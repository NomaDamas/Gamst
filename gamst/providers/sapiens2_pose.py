"""Sapiens2 whole-body pose (308 Goliath keypoints) with a DETR-R101-DC5 person detector.

Highest-accuracy 2D whole-body option in the research notes (Sapiens2 paper, Table 3, 308-kp mAP:
0.4B 76.9, 1B 80.4, 5B 82.3; Meta's own test set, no COCO-WholeBody number). GPU-server model: the
1B backbone is 4.7 TFLOPs per 1024x768 crop; no latency is published.

Loading uses the native transformers port (``Sapiens2ForPoseEstimation`` / ``Sapiens2ImageProcessor``,
transformers >= 5.10.3, Python >= 3.11) from local snapshots of ``facebook/sapiens2-pose-<size>``,
and ``DetrForObjectDetection`` from a local ``facebook/detr-resnet-101-dc5`` snapshot (the detector the
official Sapiens2 demo uses). Detection keeps COCO label 1 (person) above ``bbox_threshold`` with NMS,
as in the official ``vis_pose.py``; post-processing is the processor's UDP-DARK decoding, optionally
with flip test.

**Licence:** the Sapiens2 License forbids biometric processing and identifying or re-identifying
individuals. Use this module for pose (gesture/repetition counting) only; identity comes from
:mod:`gamst.providers.person_match`, never from Sapiens2 keypoints.

Output matches :class:`gamst.providers.pose.MediaPipePosePlugin` (``scene.poses`` with normalized
``landmarks``, ``bbox`` and a stream-local ``track_id``), so :class:`PoseChannels` uses it with
``mapping=GOLIATH_BODY``. Its keypoints are never used to find or align a face.
``accept_license=True`` records that the owner read and accepted the licence.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import threading

from ..plugins import PerceptionPlugin, _signal
from ..tracking import create_tracker

# Goliath-308 indices (sapiens2 ``configs/_base_/keypoints308.py`` at 7e5bae8, teeth removed).
GOLIATH_BODY = dict(left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8, left_wrist=62,
                    right_wrist=41, left_hip=9, right_hip=10, left_knee=11, right_knee=12,
                    left_ankle=13, right_ankle=14)
SIZES = ('0.4b', '0.8b', '1b', '5b')
UPSTREAM_REVISION = '7e5bae88456ac418ff0e58e74106c9fe192055d4'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def nms(boxes, scores, threshold):
    """Greedy IoU non-maximum suppression over xyxy boxes; indices kept, best first."""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        area = max(0., x1 - x0) * max(0., y1 - y0)
        ok = True
        for j in keep:
            a0, b0, a1, b1 = boxes[j]
            inter = max(0., min(x1, a1) - max(x0, a0)) * max(0., min(y1, b1) - max(y0, b0))
            union = area + max(0., a1 - a0) * max(0., b1 - b0) - inter
            if union > 0 and inter / union > threshold:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def to_poses(keypoints, scores, boxes, width, height):
    """Pixel keypoints (n, 308, 2) + scores (n, 308) + xyxy boxes -> scene.poses dicts (normalized)."""
    poses = []
    for index, (points, values, box) in enumerate(zip(keypoints, scores, boxes)):
        landmarks = [dict(x=float(x) / width, y=float(y) / height, visibility=float(score), presence=1.)
                     for (x, y), score in zip(points, values)]
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = max(0., x0 / width), min(1., x1 / width)
        y0, y1 = max(0., y0 / height), min(1., y1 / height)
        bbox = dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0) if x1 > x0 and y1 > y0 else None
        poses.append(dict(frame_pose_index=index, landmarks=landmarks, bbox=bbox, keypoint_set='goliath308',
                          track_id=None, association_quality='unavailable', identity_verified=False,
                          speaker_association='unavailable'))
    return poses


class Sapiens2PosePlugin(PerceptionPlugin):
    name = 'sapiens2_pose'
    modalities = ('video',)

    def __init__(self, pose, processor, detector, detector_processor, *, device, version, max_poses=4,
                 bbox_threshold=.3, nms_threshold=.3, flip_test=True, tracker='ocsort',
                 remote=None):
        self.pose, self.processor = pose, processor
        self.detector, self.detector_processor = detector, detector_processor
        self.device, self.version, self.max_poses = device, version, int(max_poses)
        self.bbox_threshold, self.nms_threshold, self.flip_test = bbox_threshold, nms_threshold, flip_test
        self.tracker, self.tracker_name = create_tracker(tracker), tracker
        self.remote = remote   # RemotePose: the models live in their own environment
        self._native_lock = threading.Lock()

    @classmethod
    def from_local(cls, pose_dir, detector_dir, *, pose_sha256=None, detector_sha256=None, device='cuda',
                   dtype='float32', accept_license=False, **options):
        if accept_license is not True:
            raise PermissionError('the Sapiens2 License forbids biometric processing and re-identification: read it '
                                  'and pass accept_license=True to opt in')
        import torch
        from transformers import (AutoImageProcessor, AutoModelForPoseEstimation, DetrForObjectDetection,
                                  DetrImageProcessor)
        pose_dir, detector_dir = Path(pose_dir), Path(detector_dir)
        digests = []
        for directory, expected in ((pose_dir, pose_sha256), (detector_dir, detector_sha256)):
            weights = directory / 'model.safetensors'
            if not weights.is_file():
                raise ValueError(f'{directory} has no model.safetensors (a local HF snapshot is required)')
            digest = _sha256(weights)
            if expected is not None and digest != expected:
                raise ValueError(f'{weights} does not match its pinned sha256 {expected}')
            digests.append(digest)
        torch_dtype = getattr(torch, dtype)
        processor = AutoImageProcessor.from_pretrained(str(pose_dir), local_files_only=True)
        pose = AutoModelForPoseEstimation.from_pretrained(str(pose_dir), local_files_only=True,
                                                          dtype=torch_dtype).to(device).eval()
        detector_processor = DetrImageProcessor.from_pretrained(str(detector_dir), local_files_only=True)
        detector = DetrForObjectDetection.from_pretrained(str(detector_dir), local_files_only=True).to(device).eval()
        version = f'sapiens2:{digests[0][:16]}:detr:{digests[1][:16]}'
        return cls(pose, processor, detector, detector_processor, device=device, version=version, **options)

    def _detect(self, rgb):
        import torch
        height, width = rgb.shape[:2]
        inputs = self.detector_processor(images=rgb, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        result = self.detector_processor.post_process_object_detection(
            outputs, threshold=self.bbox_threshold, target_sizes=torch.tensor([[height, width]]))[0]
        boxes, scores = [], []
        for label, score, box in zip(result['labels'].tolist(), result['scores'].tolist(), result['boxes'].tolist()):
            if label == 1:
                boxes.append(box)
                scores.append(score)
        keep = nms(boxes, scores, self.nms_threshold)[:self.max_poses]
        return [boxes[i] for i in keep]

    def infer(self, image_bgr):
        import torch
        rgb = image_bgr[:, :, ::-1].copy()
        height, width = rgb.shape[:2]
        with self._native_lock:
            boxes = self._detect(rgb)
            if not boxes:
                return []
            coco = [[[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in boxes]]
            inputs = self.processor(rgb, boxes=coco, return_tensors='pt').to(self.device)
            inputs['pixel_values'] = inputs['pixel_values'].to(next(self.pose.parameters()).dtype)
            with torch.inference_mode():
                outputs = self.pose(**inputs)
                flipped = None
                if self.flip_test:
                    flipped = self.pose(inputs['pixel_values'].flip(-1),
                                        flip_pairs=torch.tensor(self.pose.config.flip_pairs, device=self.device))
            results = self.processor.post_process_pose_estimation(outputs, boxes=coco, outputs_flipped=flipped)[0]
        keypoints = [item['keypoints'].detach().cpu().numpy() if hasattr(item['keypoints'], 'detach')
                     else item['keypoints'] for item in results]
        scores = [item['scores'].detach().cpu().numpy() if hasattr(item['scores'], 'detach')
                  else item['scores'] for item in results]
        return to_poses(keypoints, scores, boxes, width, height)

    async def process(self, frame):
        poses = await (self.remote.infer(frame.data) if self.remote is not None
                       else asyncio.to_thread(self.infer, frame.data))
        boxes = [pose['bbox'] for pose in poses if pose['bbox'] is not None]
        tracks = self.tracker.update(boxes, frame.timestamp_ms, stream_id=frame.subject_id + ':body')
        for pose in poses:
            matches = [track for track in tracks if track['visible'] and track['bbox'] == pose['bbox']]
            if len(matches) == 1 and matches[0]['association_quality'] != 'uncertain':
                pose['track_id'] = matches[0]['track_id']
                pose['association_quality'] = matches[0]['association_quality']
        return [_signal(frame, 'scene.poses', poses, self.name, self.version),
                _signal(frame, 'scene.pose_count', len(poses), self.name, self.version),
                _signal(frame, 'scene.body_tracks', tracks, f'{self.name}:{self.tracker_name}-tracker', '0.1')]

    def reset(self):
        self.tracker.reset()

    async def close(self):
        if self.remote is not None:
            await self.remote.close()
