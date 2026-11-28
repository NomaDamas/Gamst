"""Face detection with five landmarks (SCRFD), for aligning a face that the pose model's own face points miss.

The person binding aligns a face for LVFace from five points: the eyes, the nose and the mouth corners. The pose model
(RTMW) gives them, but only when all five are visible (raw score >= 0.3) and the eyes are at least 12 px apart; a
turned, small or partly hidden face then gives no crop at all (measured on the owner's clips,
docs/measurements/identity-recall-2026-11-28.md). A face detector trained for it finds the face and its five points in
those cases too.

**Model:** SCRFD-10GF with keypoints (InsightFace, ``det_10g.onnx`` of the ``buffalo_l`` package, 16,923,827 bytes,
sha256 ``5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91``). WIDER FACE val AP 95.40 / 94.01 / 82.80
(easy / medium / hard) at 640 px, the best published InsightFace detector that also gives five landmarks (the 34GF
model has no landmarks). The InsightFace code is MIT; its pretrained models are for non-commercial research use only,
accepted because GAMST is not used commercially (owner, 2026-11-27).

Decoding follows the official ``insightface/python-package/insightface/model_zoo/scrfd.py`` (strides 8, 16, 32, two
anchors per place, distances times the stride, NMS 0.4, input ``(x - 127.5) / 128`` RGB, padded to 640 x 640).

:class:`DetectedFaceEmbedder` is what the ``vision.face_identity`` worker runs when its config names a detector
(``detector_path`` + ``detector_sha256``): the binding sends the region around a track's head
(:func:`gamst.providers.person_match.head_region`, a square centred on the head), the worker finds the face that
covers the centre, aligns it to the ArcFace template with the detector's five points and embeds it with the same LVFace
model, so its ``model_id`` (and every enrolled reference) is unchanged.
"""
from __future__ import annotations

import math

SCRFD_SHA256 = '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'
SIZE = 640
STRIDES = (8, 16, 32)
ANCHORS = 2


def _np():
    import numpy as np
    return np


def nms(boxes, scores, threshold=.4):
    """Indices kept by greedy non-maximum suppression (``boxes`` as x0, y0, x1, y1)."""
    np = _np()
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0, yy0 = np.maximum(boxes[i, 0], boxes[order[1:], 0]), np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1, yy1 = np.minimum(boxes[i, 2], boxes[order[1:], 2]), np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0., xx1 - xx0 + 1) * np.maximum(0., yy1 - yy0 + 1)
        overlap = inter / (area[i] + area[order[1:]] - inter)
        order = order[1:][overlap <= threshold]
    return keep


def decode(outputs, scale, *, threshold=.5, nms_threshold=.4, size=SIZE):
    """``[(score, (x0, y0, x1, y1), [(x, y) * 5])]`` in the input image's pixels, best first, from the nine SCRFD outputs
    (scores, boxes and keypoints of strides 8, 16, 32) of one ``size`` x ``size`` input resized by ``scale``."""
    np = _np()
    scores, boxes, points = [], [], []
    for index, stride in enumerate(STRIDES):
        rows = cols = size // stride
        score = np.asarray(outputs[index]).reshape(-1)
        distance = np.asarray(outputs[index + 3]).reshape(-1, 4) * stride
        marks = np.asarray(outputs[index + 6]).reshape(-1, 10) * stride
        centers = np.stack(np.mgrid[:rows, :cols][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
        centers = np.repeat(centers, ANCHORS, axis=0)
        keep = np.where(score >= threshold)[0]
        if not keep.size:
            continue
        c, d, m = centers[keep], distance[keep], marks[keep]
        boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=-1))
        points.append(np.stack([c[:, 0:1] + m[:, 0::2], c[:, 1:2] + m[:, 1::2]], axis=-1))
        scores.append(score[keep])
    if not scores:
        return []
    scores, boxes, points = np.concatenate(scores), np.concatenate(boxes) / scale, np.concatenate(points) / scale
    return [(float(scores[i]), tuple(float(v) for v in boxes[i]), [tuple(float(v) for v in p) for p in points[i]])
            for i in nms(boxes, scores, nms_threshold)]


def blob(image, size=SIZE):
    """(1 x 3 x size x size float32 input, scale): ``image`` (BGR) resized to fit, padded right and bottom."""
    import cv2
    np = _np()
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    tensor = ((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), scale


class SCRFD:
    """SCRFD ONNX face detector: a BGR image -> faces with five landmarks (image-left eye, image-right eye, nose,
    image-left mouth corner, image-right mouth corner: the ArcFace template's order)."""

    def __init__(self, session, *, threshold=.5, nms_threshold=.4):
        self.session, self.threshold, self.nms_threshold = session, float(threshold), float(nms_threshold)
        self.input_name = session.get_inputs()[0].name
        self.outputs = [item.name for item in session.get_outputs()]
        if len(self.outputs) != 9:
            raise ValueError('an SCRFD model with keypoints has 9 outputs (scores, boxes, keypoints x 3 strides)')

    @classmethod
    def from_local(cls, path, *, sha256=SCRFD_SHA256, device='cpu', compute_units='ALL', cache_dir=None, **options):
        from .person_match import _verify, onnx_session
        _verify(path, sha256)
        model = cls(onnx_session(path, device=device, compute_units=compute_units, cache_dir=cache_dir, free_size=SIZE),
                    **options)
        model.detect(_np().zeros((SIZE, SIZE, 3), 'uint8'))      # CoreML specializes on the first call
        return model

    def detect(self, image):
        tensor, scale = blob(image)
        outputs = self.session.run(self.outputs, {self.input_name: tensor})
        return decode(outputs, scale, threshold=self.threshold, nms_threshold=self.nms_threshold)


def eye_distance(points):
    return math.dist(points[0], points[1])


def pick(faces, width, height, *, cover=.25):
    """The face of a head region (a square centred on the track's head): the best-scoring face whose box, grown by
    ``cover`` of its size, holds the region's centre; None when no face does (a face of someone else beside the head is
    not this track's)."""
    cx, cy = width / 2, height / 2
    for score, (x0, y0, x1, y1), points in faces:
        gx, gy = cover * (x1 - x0), cover * (y1 - y0)
        if x0 - gx <= cx <= x1 + gx and y0 - gy <= cy <= y1 + gy:
            return score, (x0, y0, x1, y1), points
    return None


INTERPOLATION = {'linear': 1, 'cubic': 2, 'lanczos': 4}       # cv2.INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4


def align(region, points, *, interpolation='linear'):
    """112 x 112 ArcFace-aligned crop of ``region`` from five points (``interpolation`` of the warp; a face smaller than
    112 px is enlarged by it)."""
    import cv2
    from .person_match import ARCFACE_112, umeyama
    matrix = umeyama(points, ARCFACE_112)
    return cv2.warpAffine(region, matrix, (112, 112), flags=INTERPOLATION[interpolation], borderValue=0.0)


class DetectedFaceEmbedder:
    """Head region -> SCRFD face that covers its centre -> ArcFace alignment from the detector's five points -> LVFace.

    Raises ValueError when there is no such face, or its eyes are closer than ``min_eye_px`` (the worker then answers
    "not usable", no evidence). ``model_id`` is the LVFace model's: the embedding is the same model's."""

    face_input = 'region'

    def __init__(self, detector, embedder, *, min_eye_px=6., interpolation='linear', min_score=.5):
        self.detector, self.embedder, self.model_id = detector, embedder, embedder.model_id
        self.min_eye_px, self.interpolation, self.min_score = float(min_eye_px), interpolation, float(min_score)
        if interpolation not in INTERPOLATION:
            raise ValueError(f'interpolation is one of {sorted(INTERPOLATION)}')

    @classmethod
    def from_local(cls, path, *, sha256=None, detector_path, detector_sha256=SCRFD_SHA256, device='cpu',
                   compute_units='ALL', cache_dir=None, min_eye_px=6., interpolation='linear', min_score=.5):
        from .person_match import LVFaceEmbedder
        embedder = LVFaceEmbedder.from_local(path, sha256=sha256, device=device, compute_units=compute_units,
                                             cache_dir=cache_dir)
        detector = SCRFD.from_local(detector_path, sha256=detector_sha256, device=device, compute_units=compute_units,
                                    cache_dir=cache_dir, threshold=min_score)
        return cls(detector, embedder, min_eye_px=min_eye_px, interpolation=interpolation, min_score=min_score)

    def face(self, region):
        """(score, box, points, aligned crop) of the region's face, or None."""
        found = pick(self.detector.detect(region), region.shape[1], region.shape[0])
        if found is None or found[0] < self.min_score or eye_distance(found[2]) < self.min_eye_px:
            return None
        return (*found, align(region, found[2], interpolation=self.interpolation))

    def embed(self, region):
        if region is None:
            return None
        found = self.face(region)
        if found is None:
            raise ValueError('no face at the centre of the head region')
        return self.embedder.embed(found[3])
