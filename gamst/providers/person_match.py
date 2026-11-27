"""Enrolled-person identity in video: which tracked body is the person the owner enrolled?

Pipeline module :class:`TargetPerson` consumes the camera and the pose record of the same capture
(``"pose:scene.poses"`` from :class:`gamst.providers.pose.MediaPipePosePlugin`, tracker ``ocsort``).
For every visible person it scores

- the **face** (LVFace, MIT code and weights): a 112x112 crop aligned to the ArcFace 5-point template
  with the eye, nose and mouth-corner keypoints the pose model already gives, then cosine against the
  enrolled face centroid;
- the **body** (whole-body appearance, person re-ID: PersonViT ViT-B, or OSNet-AIN, :class:`ReIDEmbedder`): a crop of the pose box,
  cosine against the enrolled body centroid and the body embeddings collected this session while the face confirmed the
  person (clothing changes between sessions, so body re-ID is short-horizon glue between face
  confirmations, never the primary identity).

Math here (no model): 5-point similarity alignment (Umeyama), crop geometry, centroids, cosine, and a
track lock with hysteresis: a face score >= ``face_threshold`` locks the track; the lock holds while
the track is visible, its face does not contradict (score < ``face_reject``) and the face confirmed
it within ``hold_ms`` (or the body still matches); without a lock, a body score >= ``body_threshold`` selects a track tentatively (``state="body"``).
Search mode (``mode="search"``, owner decision 33, :data:`gamst.counting.SEARCH`): never lose the person: a face that does
not contradict (score >= ``face_reject``) locks the track, and a body score >= the lower of ``body_threshold`` and the
search ``body_threshold`` selects one; :meth:`set_mode` switches it while the run goes on.

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
from pathlib import Path

from .speaker_verify import centroid, cosine, embed, l2_normalize

# ArcFace / InsightFace 112x112 destination points: image-left eye, image-right eye, nose,
# image-left mouth corner, image-right mouth corner (insightface ``face_align.arcface_dst``).
ARCFACE_112 = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655),
               (70.7299, 92.2041))
# MediaPipe Pose landmarks in that order for a frontal face: the subject's right eye and right mouth
# corner appear on the image's left. 0 nose, 2 left_eye, 5 right_eye, 9 mouth_left, 10 mouth_right.
MEDIAPIPE_FACE_POINTS = (5, 2, 0, 10, 9)
# COCO-WholeBody (RTMW) in the same order: 2 right_eye, 1 left_eye, 0 nose, then the 68 face points from index 23
# (iBUG 68: 48 is the mouth corner on the image's left of a frontal face, 54 the other): 23 + 48, 23 + 54.
COCO_WHOLEBODY_FACE_POINTS = (2, 1, 0, 71, 77)
FACE_POINTS = {'mediapipe33': (MEDIAPIPE_FACE_POINTS, .5), 'coco_wholebody133': (COCO_WHOLEBODY_FACE_POINTS, .3)}


HANDOVER_IOU = .3    # an L2 chain holds the cut's locked person only on a box that overlaps it at least this much


def box_iou(a, b):
    """Overlap of two normalized boxes {x, y, width, height} (0 when either is missing)."""
    if not a or not b:
        return 0.
    ax1, ay1, bx1, by1 = a['x'] + a['width'], a['y'] + a['height'], b['x'] + b['width'], b['y'] + b['height']
    width, height = min(ax1, bx1) - max(a['x'], b['x']), min(ay1, by1) - max(a['y'], b['y'])
    inter = max(0., width) * max(0., height)
    union = a['width'] * a['height'] + b['width'] * b['height'] - inter
    return inter / union if union > 0 else 0.


DUPLICATE_IOU = .5        # two poses of one frame are one person seen twice when their boxes overlap this much ...
DUPLICATE_JOINTS = .05    # ... and their joints agree: median distance of the joints both see, over the box height
DUPLICATE_MIN_JOINTS = 5  # joints both must see for the joint test (fewer: never a duplicate)


def same_body(a, b, *, iou=DUPLICATE_IOU, joints=DUPLICATE_JOINTS, min_joints=DUPLICATE_MIN_JOINTS, min_visibility=.5):
    """Whether two poses of one frame are the same person detected twice (a tracker that keeps a second box on one
    body): boxes overlap at least ``iou`` and the joints both poses see lie within ``joints`` x the box height of each
    other (median). Two people standing close overlap in box but not in joints (their heads and shoulders differ)."""
    if not isinstance(a, dict) or not isinstance(b, dict) or box_iou(a.get('bbox'), b.get('bbox')) < iou:
        return False
    first, second = a.get('landmarks') or [], b.get('landmarks') or []
    if len(first) != len(second) or a.get('keypoint_set') != b.get('keypoint_set'):
        return False
    seen = lambda point: min(point.get('visibility', 1.), point.get('presence', 1.)) >= min_visibility
    gaps = sorted(math.dist((p['x'], p['y']), (q['x'], q['y'])) for p, q in zip(first, second) if seen(p) and seen(q))
    if len(gaps) < min_joints:
        return False
    height = max(a['bbox']['height'], b['bbox']['height'], 1e-6)
    return gaps[len(gaps) // 2] / height <= joints


def duplicates(poses, first_seen):
    """``{duplicate track: kept track}`` among ``{track: pose}`` of one frame (:func:`same_body`). The kept track is the
    one seen first (``first_seen``: track -> ms; then the larger box, then the track id), so the choice is stable over
    frames and the same in every module that applies it (the counter counts the track the binding binds)."""
    order = sorted(poses, key=lambda track: (first_seen.get(track, float('inf')),
                                             -(poses[track].get('bbox') or {}).get('width', 0) *
                                             (poses[track].get('bbox') or {}).get('height', 0), str(track)))
    found = {}
    for index, track in enumerate(order):
        if track in found:
            continue
        for other in order[index + 1:]:
            if other not in found and same_body(poses[track], poses[other]):
                found[other] = track
    return found


def face_reading(pose, indices=MEDIAPIPE_FACE_POINTS, min_visibility=.5):
    """(indices, min_visibility) of the five face points of this pose: by its ``keypoint_set`` when it names one
    (a set without known face points, e.g. Sapiens2 ``goliath308`` here, gives None: no face is aligned); a pose
    without one (MediaPipe) uses ``indices``."""
    kind = pose.get('keypoint_set') if isinstance(pose, dict) else None
    if kind is None:
        return indices, min_visibility
    return FACE_POINTS.get(kind, (None, min_visibility))


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
    if indices is None:
        return None
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


# Head points (nose, eyes, ears) and shoulders of each keypoint set, for the region a face detector looks in.
HEAD_POINTS = {'coco_wholebody133': ((0, 1, 2, 3, 4), (5, 6)), 'mediapipe33': ((0, 2, 5, 7, 8), (11, 12))}
HEAD_REGION = 1.3       # the region's side, in shoulder widths (a frontal face is about 0.4 of it)


def head_region(image, pose, *, min_visibility=.1, scale=HEAD_REGION, min_px=24, max_px=448, points=None):
    """A square crop centred on a pose's head (the mean of its visible nose, eye and ear points), ``scale`` shoulder
    widths a side (without both shoulders: four times the spread of the head points, at least a third of the box
    width), for a face detector (:class:`gamst.providers.face_detect.DetectedFaceEmbedder`); larger than ``max_px``:
    shrunk to it. None without a visible head point, or when the square is smaller than ``min_px`` or mostly outside."""
    import cv2
    np = _np()
    heads, shoulders = (points or HEAD_POINTS).get(pose.get('keypoint_set') or 'mediapipe33', (None, None))
    marks, box = pose.get('landmarks') or [], pose.get('bbox')
    if heads is None or not box:
        return None
    height, width = image.shape[:2]

    def point(index):
        if index >= len(marks):
            return None
        mark = marks[index]
        if min(mark.get('visibility', 1.), mark.get('presence', 1.)) < min_visibility:
            return None
        return float(mark['x']) * width, float(mark['y']) * height
    head = [p for p in (point(i) for i in heads) if p is not None]
    if not head:
        return None
    cx, cy = sum(p[0] for p in head) / len(head), sum(p[1] for p in head) / len(head)
    sides = [point(i) for i in shoulders]
    if sides and all(sides):
        side = scale * math.dist(*sides)
    else:
        spread = max(max(p[0] for p in head) - min(p[0] for p in head), max(p[1] for p in head) - min(p[1] for p in head))
        side = 4 * spread
    side = max(side, float(box['width']) * width / 3)
    half = side / 2
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    if x1 - x0 < min_px:
        return None
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if (cx1 - cx0) * (cy1 - cy0) < .4 * (x1 - x0) * (y1 - y0):
        return None
    region = np.zeros((y1 - y0, x1 - x0, 3), image.dtype)      # outside the frame: black, so the head stays centred
    region[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = image[cy0:cy1, cx0:cx1]
    if region.shape[0] > max_px:
        region = cv2.resize(region, (max_px, max_px), interpolation=cv2.INTER_AREA)
    return region


def points_region(image, points, *, scale=3.):
    """A square crop centred on five face points, ``scale`` times their extent a side (for a region face embedder given
    a face another model found, e.g. the active speaker's)."""
    if points is None:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    side = scale * max(max(xs) - min(xs), max(ys) - min(ys), 1.)
    landmarks = [dict(x=(sum(xs) / len(xs)) / image.shape[1], y=(sum(ys) / len(ys)) / image.shape[0], visibility=1.)]
    box = dict(width=side / image.shape[1])
    return head_region(image, dict(keypoint_set='single', landmarks=landmarks, bbox=box), scale=1., points={'single': ((0,), ())})


def face_crop(embedder, image, pose, indices=MEDIAPIPE_FACE_POINTS):
    """What a face embedder is given for one pose: the ArcFace-aligned crop from the pose model's own face points
    (``face_input`` 'aligned', LVFace alone), or the region around the head (``face_input`` 'region': the embedder finds
    the face and its five points itself, :class:`gamst.providers.face_detect.DetectedFaceEmbedder`). None: nothing to
    look at."""
    if getattr(embedder, 'face_input', 'aligned') == 'region':
        return head_region(image, pose)
    height, width = image.shape[:2]
    chosen, visibility = face_reading(pose, indices)
    return align_face(image, face_points(pose.get('landmarks') or [], width, height, chosen, min_visibility=visibility))


def body_box(bbox, width, height, *, pad=.1):
    """Pixel box (x0, y0, x1, y1) of a normalized pose box grown by ``pad`` of its size, clipped."""
    x, y, w, h = (float(bbox[key]) for key in ('x', 'y', 'width', 'height'))
    x0, y0 = (x - pad * w) * width, (y - pad * h) * height
    x1, y1 = (x + w + pad * w) * width, (y + h + pad * h) * height
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(width, int(math.ceil(x1))), min(height, int(math.ceil(y1)))
    return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 16 else None


def body_crop(image, bbox, *, pad=.1, size=(128, 256)):
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


def onnx_session(path, *, device='cpu', compute_units='ALL', cache_dir=None, providers=None, free_size=1):
    """ONNX Runtime session on the CPU, or on Apple's CoreML execution provider (``device='coreml'``: ML Program format,
    ``compute_units`` ALL / CPUAndGPU / CPUAndNeuralEngine). ``cache_dir`` keeps CoreML's compiled model there
    (``ModelCacheDirectory``) instead of a new temporary folder per session. Free input dimensions are fixed to
    ``free_size`` (one crop per call; a detector's free height and width: its input size)."""
    import onnxruntime
    if providers is None and device == 'coreml':
        options = {'ModelFormat': 'MLProgram', 'MLComputeUnits': compute_units}
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            options['ModelCacheDirectory'] = str(cache_dir)
        providers = [('CoreMLExecutionProvider', options), 'CPUExecutionProvider']
    elif device not in ('cpu', 'coreml'):
        raise ValueError("device is 'cpu' or 'coreml'")
    options = onnxruntime.SessionOptions()
    probe = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for item in probe.get_inputs():     # one crop per call: a fixed batch, which CoreML compiles whole
        for dimension in item.shape:
            if isinstance(dimension, str):
                options.add_free_dimension_override_by_name(dimension, free_size)
    del probe
    return onnxruntime.InferenceSession(str(path), options, providers=providers or ['CPUExecutionProvider'])


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

    face_input = 'aligned'      # it is given an aligned 112 x 112 face (person_match.face_crop)

    def __init__(self, session, model_id):
        self.session, self.model_id = session, model_id
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @classmethod
    def from_local(cls, path, *, sha256=None, providers=None, device='cpu', compute_units='ALL', cache_dir=None):
        digest = _verify(path, sha256)
        model = cls(onnx_session(path, device=device, compute_units=compute_units, cache_dir=cache_dir, providers=providers),
                    f'lvface:{digest[:16]}')
        model.embed(_np().zeros((112, 112, 3), 'uint8'))     # CoreML specializes on the first call: here, not on a frame
        return model

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


class ReIDEmbedder:
    """Whole-body appearance (person re-ID) ONNX: a person crop -> an embedding, compared by cosine.

    Default (registry ``vision.body_identity``, owner 2026-11-27 "무조건 SOTA"): PersonViT ViT-B/16 fine-tuned on MSMT17
    (hustvl/PersonViT; official log: MSMT17 mAP 80.8, Rank-1 92.0 without re-ranking), converted once by
    :mod:`gamst.providers.personvit_export`: a 256 x 128 crop, mean and std 0.5, the 768-d class token. Option: OSNet-AIN
    x1.0 MSMT17 (:mod:`gamst.providers.osnet_export`, ImageNet mean and std, 512-d). Both: BGR -> RGB and a plain resize,
    as in their official test code. The feature is not normalised by the model; :func:`cosine` does. Appearance is
    evidence about one video only (clothes change): the person binding never trusts it across sessions
    (:mod:`gamst.person_reference`).
    """

    IMAGENET = ((.485, .456, .406), (.229, .224, .225))

    def __init__(self, session, model_id, *, height=256, width=128, mean=IMAGENET[0], std=IMAGENET[1]):
        self.session, self.model_id, self.height, self.width = session, model_id, int(height), int(width)
        self.mean, self.std = tuple(float(v) for v in mean), tuple(float(v) for v in std)
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @classmethod
    def from_local(cls, path, *, sha256=None, providers=None, height=256, width=128, name='osnet-ain', device='cpu',
                   compute_units='ALL', cache_dir=None, mean=IMAGENET[0], std=IMAGENET[1]):
        digest = _verify(path, sha256)
        model = cls(onnx_session(path, device=device, compute_units=compute_units, cache_dir=cache_dir, providers=providers),
                    f'{name}:{digest[:16]}', height=height, width=width, mean=mean, std=std)
        model.embed(_np().zeros((height, width, 3), 'uint8'))
        return model

    def embed(self, crop):
        np = _np()
        if crop is None:
            return None
        if crop.shape[:2] != (self.height, self.width):
            import cv2
            crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.
        tensor = ((rgb - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)).transpose(2, 0, 1)[None]
        output = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(tensor, np.float32)})[0]
        return [float(v) for v in output.reshape(-1)]


async def enroll_person(face_embedder, body_embedder, images, poses, face_indices=MEDIAPIPE_FACE_POINTS):
    """Face/body embeddings of the owner's enrollment frames (one visible person each). Images are not kept."""
    faces, bodies = [], []
    for image, pose in zip(images, poses):
        if face_embedder is not None:
            face = face_crop(face_embedder, image, pose, face_indices)
            if face is not None:
                try:
                    faces.append(await embed(face_embedder, face))
                except ValueError:      # no face the model can use
                    pass
        if body_embedder is not None and pose.get('bbox'):
            crop = body_crop(image, pose['bbox'])
            if crop is not None:
                bodies.append(await embed(body_embedder, crop))
    if not faces and not bodies:
        raise ValueError('no usable face or body in the enrollment frames')
    return dict(kind='person', face_model=getattr(face_embedder, 'model_id', None),
                body_model=getattr(body_embedder, 'model_id', None), face=faces, body=bodies)


class TargetPerson:
    """Pipeline plugin module: camera + the pose module's joints of the same capture in, the enrolled person out."""

    name = 'target_person'
    # Its face model has a worker, but it declares no process of its own: it pairs a camera frame with the joints
    # of the same capture, so it runs in the pose module's lane (gamst.pipeline), never beside it.

    def __init__(self, face=None, body=None, *, reference='person', face_threshold=.35, face_reject=.15,
                 body_threshold=.6, face_every_ms=200, body_every_ms=500, hold_ms=10000, emits=None, joints=None, crop_pad=.15,
                 face_indices=MEDIAPIPE_FACE_POINTS, mode='count', search=None):
        if face is None and body is None:
            raise ValueError('target_person needs a face or a body embedder')
        if not face_reject <= face_threshold:
            raise ValueError('require face_reject <= face_threshold')
        self.face, self.body, self.reference_name = face, body, reference
        self.face_threshold, self.face_reject, self.body_threshold = face_threshold, face_reject, body_threshold
        self._count_thresholds = (face_threshold, body_threshold)
        self.set_mode(mode, search)
        self.face_every_ms, self.body_every_ms, self.hold_ms = int(face_every_ms), int(body_every_ms), int(hold_ms)
        self.joints = joints   # derived input that receives the matched person's joints (never a record)
        self.emits, self.crop_pad, self.face_indices = emits, crop_pad, tuple(face_indices)
        self.version = f"1.{getattr(face, 'model_id', 'noface')}.{getattr(body, 'model_id', 'nobody')}"
        self.gallery = None
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
        self._reference_value = value

    # ------------------------------------------------------------------ an L2 chain (gamst.pipeline): handover at a cut

    def snapshot(self):
        """The lock at a cut: the locked person's last box, when the face last confirmed it, the session's body glue."""
        return dict(bbox=self._locked_bbox if self._locked is not None else None, confirmed_ms=self._confirmed_ms,
                    session=list(self.gallery.session) if self.gallery is not None else [])

    def fork(self):
        """The same person matcher from a fresh state; it shares this module's face and body models (never closes
        them)."""
        twin = type(self)(self.face, self.body, reference=self.reference_name, face_threshold=self._count_thresholds[0],
                          face_reject=self.face_reject, body_threshold=self._count_thresholds[1],
                          face_every_ms=self.face_every_ms, body_every_ms=self.body_every_ms, hold_ms=self.hold_ms,
                          emits=self.emits, joints=self.joints, crop_pad=self.crop_pad, face_indices=self.face_indices,
                          mode=self.mode)
        twin._shares_models = True
        if getattr(self, '_reference_value', None) is not None:
            twin.set_reference(self._reference_value)
        return twin

    def restore(self, state):
        """Take over a cut's lock: during the pre-roll the person whose box overlaps the locked box most (IoU at least
        ``HANDOVER_IOU``) is held as before; if none does, only a face confirmation inside the interval locks, and
        what comes before it stays unattributed."""
        self._handover = dict(state) if state and state.get('bbox') else None
        if self.gallery is not None:
            for embedding in (state or {}).get('session') or ():
                self.gallery.session.append(embedding)

    def set_mode(self, mode, search=None):
        """``count`` or ``search`` (module doc) from the next frame on; ``search``: :data:`gamst.counting.SEARCH`."""
        from ..counting import MODES, search_settings
        if mode not in MODES:
            raise ValueError(f"mode is one of {', '.join(MODES)}")
        face, body = self._count_thresholds
        if mode == 'search':
            face, body = self.face_reject, min(body, float(search_settings(search)['body_threshold']))
        self.mode, self.face_threshold, self.body_threshold = mode, face, body

    def reset(self):
        self._poses = {}
        self._tracks = {}
        self._locked = self._confirmed_ms = self._locked_bbox = None
        self._handover = None
        if self.gallery is not None:
            self.gallery.session.clear()

    async def _score(self, track, pose, image, ts):
        state = self._tracks.setdefault(track, dict(face=None, face_ms=None, body=None, body_ms=None))
        state['seen_ms'] = ts
        if self.face is not None and self.gallery.face is not None and \
                (state['face_ms'] is None or ts - state['face_ms'] >= self.face_every_ms):
            crop = face_crop(self.face, image, pose, self.face_indices)
            if crop is not None:
                try:
                    state['face'], state['face_ms'] = self.gallery.face_score(await embed(self.face, crop)), ts
                except ValueError:      # the region had no usable face
                    pass
        if self.body is not None and pose.get('bbox') and \
                (state['body_ms'] is None or ts - state['body_ms'] >= self.body_every_ms):
            crop = body_crop(image, pose['bbox'])
            if crop is not None:
                embedding = await embed(self.body, crop)
                state['body_embedding'] = embedding
                state['body'], state['body_ms'] = self.gallery.body_score(embedding), ts
        for key in ('face', 'body'):   # a score is evidence about now, not about a minute ago
            if state[key + '_ms'] is not None and ts - state[key + '_ms'] > 1000:
                state[key] = None
        return state

    async def process(self, frame):
        from ..plugins import InputFrame, _signal
        from ..pipeline import Derived
        from ..private_payload import PrivatePoses, poses_of, public_pose
        if self.gallery is None:
            raise RuntimeError('no person reference bound')
        found = poses_of(frame.data)
        if found is not None:                      # the joints arrive before their camera frame
            self._poses[frame.timestamp_ms] = found
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
            candidates[track] = (pose, await self._score(track, pose, frame.data, ts))
        for track in [key for key, state in self._tracks.items() if ts - state['seen_ms'] > 5000]:
            del self._tracks[track]
        target, state_name = self._select(candidates, ts)
        pose, scores = candidates[target] if target is not None else (None, {})
        self._locked_bbox = pose.get('bbox') if pose and target == self._locked else self._locked_bbox
        if state_name == 'face' and scores.get('body_embedding') is not None:
            self.gallery.remember_body(scores['body_embedding'])
        value = dict(reference=self.reference_name, state=state_name, track_id=target,
                     bbox=pose.get('bbox') if pose else None, face_score=scores.get('face'),
                     body_score=scores.get('body'), candidates=len(candidates))
        frame_like = InputFrame(None, 'video', ts, frame.subject_id)
        output = [_signal(frame_like, 'vision.target_person', value, self.name, self.version),
                  _signal(frame_like, 'scene.target_poses', [public_pose(pose)] if pose else [], self.name, self.version)]
        if self.joints and pose is not None:
            output.append(Derived(self.joints, PrivatePoses([pose]), 'video'))
        if self.emits and pose is not None:
            box = body_box(pose['bbox'], frame.data.shape[1], frame.data.shape[0], pad=self.crop_pad)
            if box is not None:
                x0, y0, x1, y1 = box
                output.append(Derived(self.emits, frame.data[y0:y1, x0:x1].copy(), 'video'))
        return output

    async def close(self):
        if getattr(self, '_shares_models', False):
            return
        for embedder in (self.face, self.body):
            close = getattr(embedder, 'close', None)
            if callable(close):
                value = close()
                if hasattr(value, '__await__'):
                    await value

    def _select(self, candidates, ts):
        """(track_id, state): best matching face; else the locked track while it holds; else best body."""
        if self._handover is not None and self._locked is None:     # an L2 chain takes over the lock at its cut
            overlaps = [(box_iou(pose.get('bbox'), self._handover['bbox']), track)
                        for track, (pose, _) in candidates.items() if pose.get('bbox')]
            best = max(overlaps, default=(0., None))
            if best[0] >= HANDOVER_IOU:
                self._locked, self._confirmed_ms = best[1], self._handover['confirmed_ms'] or ts
            self._handover = None
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
