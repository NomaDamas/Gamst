"""Enrolled-person identity in video: which tracked body is the person the owner enrolled?

Pipeline module :class:`TargetPerson` consumes the camera and the pose record of the same capture
(``"pose:scene.poses"`` from :class:`gamst.providers.pose.MediaPipePosePlugin`, tracker ``ocsort``).
For every visible person it scores

- the **face** (LVFace, MIT code and weights): a 112x112 crop aligned to the ArcFace 5-point template
  with the eye, nose and mouth-corner keypoints the pose model already gives, then cosine against the
  enrolled face centroid;
- the **body** (SOLIDER Swin-B re-ID, Apache-2.0): a 384x128 crop of the pose box, cosine against the
  enrolled body centroid and the body embeddings collected this session while the face confirmed the
  person (clothing changes between sessions, so body re-ID is short-horizon glue between face
  confirmations, never the primary identity).

Math here (no model): 5-point similarity alignment (Umeyama), crop geometry, centroids, cosine, and a
track lock with hysteresis: a face score >= ``face_threshold`` locks the track; the lock holds while
the track is visible, its face does not contradict (score < ``face_reject``) and the face confirmed
it within ``hold_ms`` (or the body still matches); without a lock, a body score >= ``body_threshold`` selects a track tentatively (``state="body"``).

Output (plugin signals, so downstream modules link by name):

- ``vision.target_person``: ``{reference, state, track_id, bbox, face_score, body_score, candidates}``;
- ``scene.target_poses``: the target's pose only (``[]`` when absent), the input of
  :class:`gamst.providers.pose_exemplar.PoseChannels` so a count follows one person;
- optional derived input (``emits=('person:<ref>',)``): the target's crop as a video frame, for any
  module that should see only that person. Derived media never enters the Store.

A ``person`` reference needs a consent record (``Pipeline.add_reference``). Its value is what
``gamst recipe enroll`` saves: ``{"face_model", "face": [[...]], "body_model", "body": [[...]]}``
(embeddings only, never images). Scores are uncalibrated cosine similarities; set thresholds on the
owner's own video.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
import threading

from .speaker_verify import centroid, cosine, l2_normalize

# ArcFace / InsightFace 112x112 destination points: image-left eye, image-right eye, nose,
# image-left mouth corner, image-right mouth corner (insightface ``face_align.arcface_dst``).
ARCFACE_112 = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655),
               (70.7299, 92.2041))
# MediaPipe Pose landmarks in that order for a frontal face: the subject's right eye and right mouth
# corner appear on the image's left. 0 nose, 2 left_eye, 5 right_eye, 9 mouth_left, 10 mouth_right.
MEDIAPIPE_FACE_POINTS = (5, 2, 0, 10, 9)


def _np():
    import numpy as np
    return np


def umeyama(source, target):
    """2x3 similarity transform (rotation, uniform scale, translation) mapping ``source`` onto ``target``."""
    np = _np()
    src, dst = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 2:
        raise ValueError('need matching (n, 2) point sets, n >= 2')
    mean_src, mean_dst = src.mean(0), dst.mean(0)
    a, b = src - mean_src, dst - mean_dst
    covariance = b.T @ a / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    variance = (a ** 2).sum() / len(src)
    if variance <= 0:
        raise ValueError('degenerate source points')
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    matrix = np.zeros((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = mean_dst - scale * rotation @ mean_src
    return matrix


def face_points(landmarks, width, height, indices=MEDIAPIPE_FACE_POINTS, *, min_visibility=.5):
    """Five face points in pixels from normalized keypoints, or None when any is not visible."""
    points = []
    for index in indices:
        if index >= len(landmarks):
            return None
        point = landmarks[index]
        if min(point.get('visibility', 1.), point.get('presence', 1.)) < min_visibility:
            return None
        points.append((float(point['x']) * width, float(point['y']) * height))
    return points


def align_face(image, points, *, min_eye_px=12.):
    """112x112 ArcFace-aligned BGR crop, or None when the eyes are closer than ``min_eye_px``."""
    import cv2
    if points is None:
        return None
    if math.dist(points[0], points[1]) < min_eye_px:
        return None
    matrix = umeyama(points, ARCFACE_112)
    return cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)


def body_box(bbox, width, height, *, pad=.1):
    """Pixel box (x0, y0, x1, y1) of a normalized pose box grown by ``pad`` of its size, clipped."""
    x, y, w, h = (float(bbox[key]) for key in ('x', 'y', 'width', 'height'))
    x0, y0 = (x - pad * w) * width, (y - pad * h) * height
    x1, y1 = (x + w + pad * w) * width, (y + h + pad * h) * height
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(width, int(math.ceil(x1))), min(height, int(math.ceil(y1)))
    return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 16 else None


def body_crop(image, bbox, *, pad=.1, size=(128, 384)):
    """BGR crop of the person resized to ``size`` (width, height), or None for a tiny box."""
    import cv2
    height, width = image.shape[:2]
    box = body_box(bbox, width, height, pad=pad)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return cv2.resize(image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)


def _verify(path, expected):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise ValueError(f'{path} does not match its pinned sha256 {expected}')
    return value


# ---------------------------------------------------------------- identity gallery and embedding backends

class PersonGallery:
    """Enrolled face/body centroids plus body embeddings seen while the face confirmed the person."""

    def __init__(self, *, face=None, body=None, session_size=32):
        self.face = centroid(face) if face else None
        self.body = centroid(body) if body else None
        self.session = deque(maxlen=session_size)

    def face_score(self, embedding):
        return None if self.face is None or embedding is None else round(cosine(embedding, self.face), 5)

    def body_score(self, embedding):
        if embedding is None:
            return None
        scores = []
        if self.body is not None:
            scores.append(cosine(embedding, self.body))
        if self.session:
            scores.append(cosine(embedding, centroid(list(self.session))))
        return round(max(scores), 5) if scores else None

    def remember_body(self, embedding):
        if embedding is not None:
            self.session.append(l2_normalize(embedding))


class LVFaceEmbedder:
    """LVFace ONNX (bytedance-research/LVFace, MIT): aligned 112x112 BGR face -> 512-d embedding.

    Preprocessing follows the official ``inference_onnx.py``: BGR -> RGB, CHW, ``(x / 255 - 0.5) / 0.5``.
    The model does not normalise its output; :func:`cosine` does.
    """

    def __init__(self, session, model_id):
        self.session, self.model_id = session, model_id
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @classmethod
    def from_local(cls, path, *, sha256=None, providers=None):
        import onnxruntime
        digest = _verify(path, sha256)
        session = onnxruntime.InferenceSession(str(path), providers=providers or ['CPUExecutionProvider'])
        return cls(session, f'lvface:{digest[:16]}')

    def embed(self, face):
        np = _np()
        if face is None:
            return None
        if face.shape[:2] != (112, 112):
            import cv2
            face = cv2.resize(face, (112, 112))
        tensor = ((face[:, :, ::-1].astype(np.float32) / 255. - .5) / .5).transpose(2, 0, 1)[None]
        output = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(tensor)})[0]
        return [float(v) for v in output.reshape(-1)]


class SOLIDERBodyEmbedder:
    """SOLIDER-REID Swin (tinyvision/SOLIDER-REID, MIT; SOLIDER backbone Apache-2.0): person crop -> 1024-d.

    Built with the official ``make_model`` from a pinned source checkout and a fine-tuned re-ID
    checkpoint (``load_param``). Three shims keep the 2023 code importable on torch 2.x without editing
    it: ``torch._six.container_abcs`` (removed in torch 2) is provided from ``collections.abc``;
    ``mmcv.runner.load_checkpoint`` (used only to load ImageNet/SOLIDER *pretraining*, which inference
    from a fine-tuned checkpoint skips via ``MODEL.PRETRAIN_PATH ''``) raises if called; and the backbone
    is called with an explicit ``semantic_weight`` tensor on the model's device (its default path calls
    ``.cuda()``). The feature is the backbone's ``global_feat`` (``TEST.NECK_FEAT 'before'``), as in
    the official test; it is L2-normalised by :func:`cosine`.
    """

    SOURCE_REVISION = '8c08e1c3255e8e1e51e006bf189e52cc57b009ed'

    def __init__(self, model, model_id, device, semantic_weight):
        self.model, self.model_id, self.device, self.semantic_weight = model, model_id, device, semantic_weight

    @staticmethod
    def _shims():
        import collections.abc
        import sys
        import types
        import torch
        if not hasattr(torch, '_six') and 'torch._six' not in sys.modules:
            six = types.ModuleType('torch._six')
            six.container_abcs = collections.abc
            six.string_classes = (str,)
            six.int_classes = (int,)
            sys.modules['torch._six'] = six
        try:
            from mmcv.runner import load_checkpoint  # noqa: F401  (mmcv < 2 installed: use it)
        except ImportError:
            def load_checkpoint(*args, **kwargs):
                raise RuntimeError('SOLIDER pretraining weights are not loaded for inference')
            mmcv = sys.modules.setdefault('mmcv', types.ModuleType('mmcv'))
            runner = types.ModuleType('mmcv.runner')
            runner.load_checkpoint = load_checkpoint
            mmcv.runner = runner
            sys.modules['mmcv.runner'] = runner

    @classmethod
    def from_local(cls, source, checkpoint, *, sha256=None, config='configs/msmt17/swin_base.yml',
                   semantic_weight=.2, device='cuda'):
        import os
        import subprocess
        import sys
        source = os.path.abspath(source)
        head = subprocess.run(['git', '-C', source, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        if head != cls.SOURCE_REVISION:
            raise ValueError(f'{source} must be tinyvision/SOLIDER-REID at {cls.SOURCE_REVISION}, found {head or "no git"}')
        digest = _verify(checkpoint, sha256)
        cls._shims()
        if source not in sys.path:
            sys.path.insert(0, source)
        from config import cfg as base_cfg
        from model import make_model
        cfg = base_cfg.clone()
        cfg.merge_from_file(os.path.join(source, config))
        cfg.merge_from_list(['MODEL.SEMANTIC_WEIGHT', float(semantic_weight), 'MODEL.PRETRAIN_PATH', '',
                             'TEST.WEIGHT', str(checkpoint), 'TEST.NECK_FEAT', 'before'])
        model = make_model(cfg, num_class=1, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        model.load_param(str(checkpoint))
        model.eval().to(device)
        return cls(model, f'solider-reid:{digest[:16]}', device, float(semantic_weight))

    def embed(self, crop):
        import torch
        np = _np()
        if crop is None:
            return None
        tensor = ((crop[:, :, ::-1].astype(np.float32) / 255. - .5) / .5).transpose(2, 0, 1)[None]
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(tensor)).to(self.device)
            weight = torch.full((1, 1), self.semantic_weight, device=self.device)
            feature, _ = self.model.base(x, semantic_weight=weight)
        return [float(v) for v in feature.detach().cpu().reshape(-1).tolist()]


def enroll_person(face_embedder, body_embedder, images, poses, face_indices=MEDIAPIPE_FACE_POINTS):
    """Face/body embeddings of the owner's enrollment frames (one visible person each). Images are not kept."""
    faces, bodies = [], []
    for image, pose in zip(images, poses):
        height, width = image.shape[:2]
        landmarks = pose.get('landmarks') or []
        if face_embedder is not None:
            face = align_face(image, face_points(landmarks, width, height, face_indices))
            if face is not None:
                faces.append(face_embedder.embed(face))
        if body_embedder is not None and pose.get('bbox'):
            crop = body_crop(image, pose['bbox'])
            if crop is not None:
                bodies.append(body_embedder.embed(crop))
    if not faces and not bodies:
        raise ValueError('no usable face or body in the enrollment frames')
    return dict(kind='person', face_model=getattr(face_embedder, 'model_id', None),
                body_model=getattr(body_embedder, 'model_id', None), face=faces, body=bodies)


class TargetPerson:
    """Pipeline plugin module: camera + ``scene.poses`` of the same capture in, the enrolled person out."""

    name = 'target_person'

    def __init__(self, face=None, body=None, *, reference='person', face_threshold=.35, face_reject=.15,
                 body_threshold=.6, face_every_ms=200, body_every_ms=500, hold_ms=10000, emits=None, crop_pad=.15,
                 face_indices=MEDIAPIPE_FACE_POINTS):
        if face is None and body is None:
            raise ValueError('target_person needs a face or a body embedder')
        if not face_reject <= face_threshold:
            raise ValueError('require face_reject <= face_threshold')
        self.face, self.body, self.reference_name = face, body, reference
        self.face_threshold, self.face_reject, self.body_threshold = face_threshold, face_reject, body_threshold
        self.face_every_ms, self.body_every_ms, self.hold_ms = int(face_every_ms), int(body_every_ms), int(hold_ms)
        self.emits, self.crop_pad, self.face_indices = emits, crop_pad, tuple(face_indices)
        self.version = f"1.{getattr(face, 'model_id', 'noface')}.{getattr(body, 'model_id', 'nobody')}"
        self.gallery = None
        self._lock = threading.RLock()
        self.reset()

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'person':
            raise ValueError(f'target_person needs a person reference named {self.reference_name!r}')
        self.set_reference(reference['value'])

    def set_reference(self, value):
        if not isinstance(value, dict):
            raise ValueError('a person reference is {"face_model", "face": [...], "body_model", "body": [...]}')
        face = value.get('face') or []
        body = value.get('body') or []
        if face and (self.face is None or value.get('face_model') != self.face.model_id):
            raise ValueError(f"person reference face embeddings come from {value.get('face_model')!r}; "
                             f"this module runs {getattr(self.face, 'model_id', None)!r}")
        if body and (self.body is None or value.get('body_model') != self.body.model_id):
            body = []   # another body model: keep only face identity (body glue is rebuilt this session)
        if not face and not body:
            raise ValueError('the person reference has no embeddings this module can use')
        self.gallery = PersonGallery(face=face, body=body)

    def reset(self):
        with self._lock:
            self._poses = {}
            self._tracks = {}
            self._locked = self._confirmed_ms = None
            if self.gallery is not None:
                self.gallery.session.clear()

    def _score(self, track, pose, image, ts):
        height, width = image.shape[:2]
        state = self._tracks.setdefault(track, dict(face=None, face_ms=None, body=None, body_ms=None))
        state['seen_ms'] = ts
        if self.face is not None and self.gallery.face is not None and \
                (state['face_ms'] is None or ts - state['face_ms'] >= self.face_every_ms):
            crop = align_face(image, face_points(pose.get('landmarks') or [], width, height, self.face_indices))
            if crop is not None:
                state['face'], state['face_ms'] = self.gallery.face_score(self.face.embed(crop)), ts
        if self.body is not None and pose.get('bbox') and \
                (state['body_ms'] is None or ts - state['body_ms'] >= self.body_every_ms):
            crop = body_crop(image, pose['bbox'])
            if crop is not None:
                embedding = self.body.embed(crop)
                state['body_embedding'] = embedding
                state['body'], state['body_ms'] = self.gallery.body_score(embedding), ts
        for key in ('face', 'body'):   # a score is evidence about now, not about a minute ago
            if state[key + '_ms'] is not None and ts - state[key + '_ms'] > 1000:
                state[key] = None
        return state

    def process(self, frame):
        from ..plugins import InputFrame, _signal
        from ..pipeline import Derived
        with self._lock:
            if self.gallery is None:
                raise RuntimeError('no person reference bound')
            if isinstance(frame.data, list):           # the pose record arrives before its camera frame
                self._poses[frame.timestamp_ms] = frame.data
                while len(self._poses) > 8:
                    self._poses.pop(next(iter(self._poses)))
                return []
            if frame.modality != 'video' or not hasattr(frame.data, 'shape'):
                return []
            ts = frame.timestamp_ms
            poses = self._poses.pop(ts, None)
            if poses is None:
                return []
            candidates = {}
            for index, pose in enumerate(poses):
                if not isinstance(pose, dict) or not pose.get('bbox'):
                    continue
                track = pose.get('track_id') or f'frame:{ts}:{index}'
                candidates[track] = (pose, self._score(track, pose, frame.data, ts))
            for track in [key for key, state in self._tracks.items() if ts - state['seen_ms'] > 5000]:
                del self._tracks[track]
            target, state_name = self._select(candidates, ts)
            pose, scores = candidates[target] if target is not None else (None, {})
            if state_name == 'face' and scores.get('body_embedding') is not None:
                self.gallery.remember_body(scores['body_embedding'])
            value = dict(reference=self.reference_name, state=state_name, track_id=target,
                         bbox=pose.get('bbox') if pose else None, face_score=scores.get('face'),
                         body_score=scores.get('body'), candidates=len(candidates))
            frame_like = InputFrame(None, 'video', ts, frame.subject_id)
            output = [_signal(frame_like, 'vision.target_person', value, self.name, self.version),
                      _signal(frame_like, 'scene.target_poses', [pose] if pose else [], self.name, self.version)]
            if self.emits and pose is not None:
                box = body_box(pose['bbox'], frame.data.shape[1], frame.data.shape[0], pad=self.crop_pad)
                if box is not None:
                    x0, y0, x1, y1 = box
                    output.append(Derived(self.emits, frame.data[y0:y1, x0:x1].copy(), 'video'))
            return output

    def _select(self, candidates, ts):
        """(track_id, state): best matching face; else the locked track while it holds; else best body."""
        faces = [(state['face'], track) for track, (_, state) in candidates.items()
                 if state.get('face') is not None and state['face'] >= self.face_threshold]
        if faces:
            locked = next((track for _, track in faces if track == self._locked), None)
            self._locked, self._confirmed_ms = locked or max(faces)[1], ts
            return self._locked, 'face'
        if self._locked in candidates:
            scores = candidates[self._locked][1]
            face, body = scores.get('face'), scores.get('body')
            fresh = ts - self._confirmed_ms <= self.hold_ms or (body is not None and body >= self.body_threshold)
            if fresh and (face is None or face >= self.face_reject):
                return self._locked, 'held'
        self._locked = None   # gone, contradicted by its face, or unconfirmed for too long
        bodies = [(state['body'], track) for track, (_, state) in candidates.items()
                  if state.get('body') is not None and state['body'] >= self.body_threshold
                  and not (state.get('face') is not None and state['face'] < self.face_reject)]
        if bodies:
            return max(bodies)[1], 'body'
        return None, 'absent'
