"""YOLO26 through the ``ultralytics`` package: the Ultralytics stack's detector with its own tracker, and its pose.

The Ultralytics sports page (https://www.ultralytics.com/solutions/computer-vision-in-sport, read 2026-11-27)
shows YOLO models with Ultralytics' trackers (BoT-SORT, ByteTrack) and its pose models; it publishes no sports
weights. This adapter runs exactly that combination on the COCO weights of the ``v8.4.0`` release (``sports ball``,
``person``, ``tennis racket`` ... are COCO classes), each in its own environment:

- :class:`UltralyticsObjects` (worker profile ``ultralytics.objects``): ``model.track(persist=True,
  tracker="botsort.yaml" | "bytetrack.yaml")`` per frame, or ``model.predict`` with ``tracker: null``. It commits the
  ``objects`` contract of every GAMST detector (``docs/vision-detection.md``), track IDs from Ultralytics' tracker.
- :class:`UltralyticsPoseModels` (model worker ``ultralytics.pose``): YOLO26-pose, 17 COCO keypoints per person;
  joints leave the worker only as a private payload, like RTMW's (``gamst/providers/rtmw_pose.py``), and the same
  ``RTMWPosePlugin`` follows the people in this process.

Ultralytics and its weights are AGPL-3.0 (or an Ultralytics Enterprise licence). GAMST stays Apache-2.0: this file
imports ``ultralytics`` only inside the worker's own environment, which the owner sets up on purpose
(``gamst recipe setup RECIPE --stack ultralytics``).
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from ..capabilities import CapabilitySpec, OutputField

TRACKERS = ('botsort', 'bytetrack')
DEVICES = ('cpu', 'mps', 'cuda')
POINTS = 17
VERSION = '8.4.212'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_config(config, *, pose=False):
    known = {'path', 'sha256', 'device', 'imgsz', 'conf', 'iou', 'max_objects', 'half'} | (
        set() if pose else {'tracker', 'labels'}) | ({'max_poses'} if pose else set())
    if set(config) - known or not {'path', 'sha256'} <= set(config):
        raise ValueError(f'the ultralytics worker needs path and sha256; optional {sorted(known - {"path", "sha256"})}')
    if not isinstance(config['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', config['sha256']):
        raise ValueError('a lowercase checkpoint SHA256 is required')
    if config.get('device', 'cpu') not in DEVICES:
        raise ValueError(f'device is one of {DEVICES}')
    if not pose and config.get('tracker', 'botsort') not in (*TRACKERS, None):
        raise ValueError(f'tracker is one of {TRACKERS} or null (detection only)')
    imgsz = config.get('imgsz', 640)
    if type(imgsz) is not int or not 160 <= imgsz <= 2560 or imgsz % 32:
        raise ValueError('imgsz is a multiple of 32 in 160..2560')
    for key, default in (('conf', .25), ('iou', .7)):
        if type(config.get(key, default)) not in (int, float) or not 0 < config.get(key, default) <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    labels = config.get('labels')
    if labels is not None and (not isinstance(labels, list) or not labels or not all(isinstance(v, str) for v in labels)):
        raise ValueError('labels is a list of class names to keep')
    return config


def _load(path, sha256, device):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or _sha256(path) != sha256:
        raise ValueError('existing checkpoint with the pinned SHA256 required')
    from importlib.metadata import version
    if version('ultralytics') != VERSION:
        raise RuntimeError(f'the ultralytics adapter is validated with ultralytics=={VERSION}')
    from ultralytics import YOLO
    model = YOLO(str(path))
    model.to(device)
    return model


class UltralyticsObjects:
    """One stream per instance (the tracker keeps state across calls: ``persist=True``)."""

    @staticmethod
    def capability(config):
        config = check_config(dict(config))
        tracker = config.get('tracker', 'botsort')
        version = '1.' + hashlib.sha256(repr(sorted(config.items())).encode()).hexdigest()[:16]
        return CapabilitySpec('vision.ultralytics.objects', 'vision.objects', 'video', 'ultralytics.yolo26', version,
                              (OutputField('objects', 'array'), OutputField('summary', 'object')), 'frame',
                              stateful=tracker is not None, reset_on_gap=tracker is not None, freshness_ms=1000,
                              devices=DEVICES, sensor_requirements=('uint8-bgr-camera',),
                              license='adapter:Apache-2.0;ultralytics:AGPL-3.0', interpretation='estimate')

    @classmethod
    def from_local(cls, **config):
        config = check_config(dict(config))
        return cls(_load(config['path'], config['sha256'], config.get('device', 'cpu')), config)

    def __init__(self, model, config):
        self.model, self.config = model, dict(config)
        self.spec = self.capability(config)
        self.tracker = config.get('tracker', 'botsort')
        names = getattr(model, 'names', {}) or {}
        labels = config.get('labels')
        self.classes = None if labels is None else [index for index, name in names.items() if name in labels]
        if labels is not None and not self.classes:
            raise ValueError(f'none of {labels} is a class of this model')

    def reset(self):
        predictor = getattr(self.model, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', None) or ():
            tracker.reset()

    def close(self):
        self.model = None

    def process(self, frame):
        import numpy as np
        image = frame.data
        if frame.modality != 'video' or not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError('the ultralytics detector needs a uint8 BGR image')
        height, width = image.shape[:2]
        options = dict(imgsz=self.config.get('imgsz', 640), conf=self.config.get('conf', .25),
                       iou=self.config.get('iou', .7), classes=self.classes, verbose=False,
                       device=self.config.get('device', 'cpu'), max_det=self.config.get('max_objects', 64))
        if self.tracker is None:
            result = self.model.predict(image, **options)[0]
        else:
            result = self.model.track(image, persist=True, tracker=f'{self.tracker}.yaml', **options)[0]
        found = result.boxes
        objects = []
        if found is not None and len(found):
            xyxy = found.xyxy.cpu().numpy()
            scores = found.conf.cpu().numpy()
            classes = found.cls.cpu().numpy().astype(int)
            ids = None if found.id is None else found.id.cpu().numpy().astype(int)
            for row, (box, score, cls) in enumerate(zip(xyxy, scores, classes)):
                x0, y0, x1, y1 = (float(v) for v in box)
                x0, x1 = max(0., x0 / width), min(1., x1 / width)
                y0, y1 = max(0., y0 / height), min(1., y1 / height)
                if x1 <= x0 or y1 <= y0 or not math.isfinite(float(score)):
                    continue
                track = None if ids is None else f'track-{int(ids[row])}'
                objects.append(dict(track_id=track or f'object-{row}', label=str(result.names[int(cls)]),
                                    attributes=dict(class_id=int(cls), score=round(float(score), 4),
                                                    association='ultralytics.' + (self.tracker or 'none'),
                                                    identity_verified=False),
                                    location=dict(bbox=dict(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                                                  unit='normalized_image')))
        return self.spec.validate(dict(objects=objects, summary=dict(
            detected_count=len(objects), returned_count=len(objects), tracking=f'ultralytics.{self.tracker}',
            width=int(width), height=int(height))))


class UltralyticsPoseModels:
    """YOLO26-pose in the worker: the same calls as :class:`gamst.providers.rtmw_pose.RTMWModels`."""

    def __init__(self, path, sha256, *, device='cpu', imgsz=640, conf=.25, max_poses=4, **_):
        check_config(dict(path=path, sha256=sha256, device=device, imgsz=imgsz, conf=conf, max_poses=max_poses), pose=True)
        self.model = _load(path, sha256, device)
        self.device, self.imgsz, self.conf, self.max_poses = device, imgsz, conf, int(max_poses)
        self.version = f'yolo26-pose:{sha256[:16]}'

    def infer_batch(self, images, timestamps_ms=None):
        """BGR images -> per image (boxes xyxy, keypoints (n, 17, 2) pixels, scores (n, 17))."""
        import numpy as np
        results = self.model.predict(list(images), imgsz=self.imgsz, conf=self.conf, device=self.device,
                                     max_det=self.max_poses, verbose=False)
        found = []
        for result in results:
            if result.keypoints is None or result.boxes is None or not len(result.boxes):
                found.append((np.zeros((0, 4)), np.zeros((0, POINTS, 2)), np.zeros((0, POINTS))))
                continue
            points = result.keypoints.xy.cpu().numpy()
            scores = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else \
                np.ones(points.shape[:2])
            found.append((result.boxes.xyxy.cpu().numpy(), points, scores))
        return found

    def benchmark(self, images, detect=False):
        import time
        started = time.perf_counter()
        self.infer_batch(images)
        return dict(ms_per_frame=round((time.perf_counter() - started) * 1000 / len(images), 2), detect_ms=0.,
                    det_every_ms=0, frames=len(images), version=self.version)

    def reset(self):
        pass
