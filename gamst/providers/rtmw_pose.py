"""RTMW whole-body pose (133 COCO-WholeBody keypoints, hands included) through rtmlib's ONNX models.

The usable state of the art for 2D whole-body pose in docs/research/speaker-pose-sota-2026-11/report.md: RTMW-x
384x288 is COCO-WholeBody 70.2 AP, RTMW-l 384x288 70.1 AP (mmpose RTMPose project table). OpenMMLab publishes an ONNX
export of RTMW-l 384x288 only (``rtmw-dw-x-l_simcc-cocktail14_270e-384x288``, rtmlib's ``performance`` whole-body
model); RTMW-x 384x288 is a ``.pth`` checkpoint without an ONNX file. Code: rtmlib and mmpose, Apache-2.0.

Top-down: a YOLOX-m person detector (HumanArt, rtmlib's ``performance`` detector) gives boxes, RTMW the keypoints
in each box. ``det_every_ms`` 0 (default) detects on every frame, as rtmlib's ``Wholebody`` does; a positive value
detects at most that often and in between uses the box around the previous frame's body keypoints (rtmlib
``PoseTracker``), which is cheaper but follows a cut or a new person only at the next detection.

Frames may come in a batch of consecutive ones (:meth:`RTMWModels.infer_batch`, ``PerceptionPlugin.process_batch``):
the pose model then runs once on all their crops. Measured on an M2 Pro (CoreML, GPU): 23 ms for one crop, 14-15 ms
per crop for 4-8 (docs/measurements/testbed-2026-11-26.md). In a batch, a frame without a detection uses the box of
the latest frame before the batch.

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
# One call of the worker: at most MAX_FRAMES consecutive frames and MAX_POSES people in each (the private payload
# carries their joints together); the pose model runs on at most MAX_CROPS crops per session call.
MAX_FRAMES, MAX_POSES, MAX_CROPS = 16, 4, 16
FIXED_BATCHES = (8, 4, 2)   # CoreML: a session of each fixed batch size (about 300 MB each); 1 crop uses the plain one


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_config(config):
    """The worker's settings, refused before anything starts."""
    known = {'path', 'sha256', 'det_path', 'det_sha256', 'device', 'compute_units', 'det_every_ms', 'max_poses',
             'det_score', 'input_size', 'max_crops', 'fixed_batches', 'cache_dir'}
    if set(config) - known or not {'path', 'sha256', 'det_path', 'det_sha256'} <= set(config):
        raise ValueError(f'rtmw.pose needs path, sha256, det_path, det_sha256; optional {sorted(known)}')
    if config.get('device', 'cpu') not in DEVICES or config.get('compute_units', 'CPUAndGPU') not in COMPUTE_UNITS:
        raise ValueError(f'rtmw.pose device is one of {DEVICES}, compute_units one of {COMPUTE_UNITS}')
    if type(config.get('max_poses', 4)) is not int or not 1 <= config.get('max_poses', 4) <= MAX_POSES:
        raise ValueError(f'rtmw.pose max_poses is 1..{MAX_POSES}')
    if type(config.get('max_crops', MAX_CROPS)) is not int or not 1 <= config.get('max_crops', MAX_CROPS) <= 64:
        raise ValueError('rtmw.pose max_crops is 1..64')
    fixed = config.get('fixed_batches', list(FIXED_BATCHES))
    if not isinstance(fixed, (list, tuple)) or any(type(size) is not int or not 2 <= size <= 64 for size in fixed):
        raise ValueError('rtmw.pose fixed_batches is a list of batch sizes 2..64')
    every = config.get('det_every_ms', 0)
    if type(every) is not int or not 0 <= every <= 10000:
        raise ValueError('det_every_ms is integer milliseconds 0..10000 (0: every frame)')


def coreml_cache(weights, cache_dir=None):
    """Where compiled CoreML models are kept: ``cache_dir``, else ``$GAMST_COREML_CACHE``, else ``cache/coreml`` beside
    the weights' root (``<root>/models/<name>/file`` -> ``<root>/cache/coreml``); None if it cannot be made."""
    import os
    folder = cache_dir or os.environ.get('GAMST_COREML_CACHE') or Path(weights).resolve().parent.parent.parent / 'cache' / 'coreml'
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return Path(folder)


TEMP_NAME = r'onnxruntime-[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}-(\d+)-[0-9A-F]+(\.mlmodelc|\.mlpackage)?'


def clear_dead_coreml_temp(folder=None):
    """ONNX Runtime's CoreML temp copies (``$TMPDIR/onnxruntime-<UUID>-<pid>-<hex>[.mlmodelc]``) of processes that no
    longer run: a killed worker leaves 200-400 MB per session there. Only names of exactly that form are touched."""
    import os
    import re
    import shutil
    import tempfile
    removed = 0
    for entry in Path(folder or tempfile.gettempdir()).glob('onnxruntime-*'):
        found = re.fullmatch(TEMP_NAME, entry.name)
        if not found:
            continue
        try:
            os.kill(int(found.group(1)), 0)
            continue                     # its process still runs
        except ProcessLookupError:
            pass
        except (PermissionError, OverflowError, ValueError):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        removed += 1
    return removed


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
                 max_poses=4, det_score=None, input_size=(288, 384), max_crops=MAX_CROPS, fixed_batches=FIXED_BATCHES,
                 cache_dir=None):
        check_config(dict(path=path, sha256=sha256, det_path=det_path, det_sha256=det_sha256, device=device,
                          compute_units=compute_units, det_every_ms=det_every_ms, max_poses=max_poses,
                          max_crops=max_crops, fixed_batches=list(fixed_batches)))
        for weights, expected in ((path, sha256), (det_path, det_sha256)):
            if not Path(weights).is_file() or _sha256(weights) != expected:
                raise ValueError(f'{weights} is missing or does not match its pinned sha256')
        import numpy as np
        import onnxruntime as ort
        from rtmlib import RTMPose, YOLOX
        self._fixed = {}
        self.detector = YOLOX(str(det_path), model_input_size=(640, 640), backend='onnxruntime', device='cpu')
        if det_score is not None:
            self.detector.score_thr = float(det_score)
        self.pose = RTMPose(str(path), model_input_size=tuple(input_size), backend='onnxruntime', device='cpu')
        if device == 'coreml':
            cache = coreml_cache(path, cache_dir)
            clear_dead_coreml_temp()

            def providers(sha, name):
                # one compiled copy per model, unit and batch size, reused by every run (without it CoreML writes a
                # 200-400 MB copy per session into $TMPDIR, left behind whenever a worker is killed)
                options = {'ModelFormat': 'MLProgram', 'MLComputeUnits': compute_units}
                if cache is not None:
                    folder = cache / f'{sha[:16]}-{compute_units}-{name}'
                    folder.mkdir(parents=True, exist_ok=True)
                    options['ModelCacheDirectory'] = str(folder)
                return [('CoreMLExecutionProvider', options), 'CPUExecutionProvider']
            self.detector.session = ort.InferenceSession(str(det_path), providers=providers(det_sha256, 'det'))
            self.pose.session = ort.InferenceSession(str(path), providers=providers(sha256, 'dynamic'))
            # A session whose batch size changes from call to call specializes again for every new size (about
            # 300 ms on CoreML); one session per fixed size never does. The session above takes what is left.
            for size in sorted(set(fixed_batches), reverse=True):
                options = ort.SessionOptions()
                options.add_free_dimension_override_by_name('batch', int(size))
                self._fixed[int(size)] = ort.InferenceSession(str(path), options, providers=providers(sha256, f'b{size}'))
            # The first call of a CoreML session specializes it (hundreds of ms): done here, not on the first frames.
            height, width = self.pose.model_input_size[1], self.pose.model_input_size[0]
            name = self.pose.session.get_inputs()[0].name
            for size, session in [(1, self.pose.session), *self._fixed.items()]:
                session.run(None, {name: np.zeros((size, 3, height, width), 'f4')})
        self.det_every_ms, self.max_poses, self.max_crops = int(det_every_ms), int(max_poses), int(max_crops)
        self._mean = np.asarray(self.pose.mean, 'f4')
        self._std = np.asarray(self.pose.std, 'f4')
        self.version = f'rtmw:{sha256[:16]}:yolox:{det_sha256[:16]}:{device}' + (f':det{det_every_ms}' if det_every_ms else '')
        self.reset()

    def reset(self):
        self._boxes, self._detected_ms = None, None

    def benchmark(self, images, detect=False):
        """The pose path of :meth:`infer_batch` (crop, normalize, model, keypoints) on ``images`` with one fixed box
        each (no detector), and, with ``detect``, the time of one detection of the first image (None if the detector
        fails on it). Synthetic input; what it measures is this machine's speed, not accuracy."""
        import time
        detect_ms = None
        if detect:
            try:
                started = time.perf_counter()
                self.detector(images[0])
                detect_ms = round((time.perf_counter() - started) * 1000, 1)
            except Exception:
                detect_ms = None
        height, width = images[0].shape[:2]
        box = [width / 4, height / 8, 3 * width / 4, 7 * height / 8]
        name = self.pose.session.get_inputs()[0].name
        started = time.perf_counter()
        crops = [self._crop(image, box) for image in images]
        start = 0
        while start < len(crops):
            size = next((size for size in self._fixed if size <= len(crops) - start), 1)
            session = self._fixed.get(size, self.pose.session)
            outputs = session.run(None, {name: self._normalized([crop[0] for crop in crops[start:start + size]])})
            for row, (_, center, scale) in enumerate(crops[start:start + size]):
                self.pose.postprocess([outputs[0][row:row + 1], outputs[1][row:row + 1]], center, scale)
            start += size
        return dict(ms_per_frame=round((time.perf_counter() - started) * 1000 / len(images), 2), detect_ms=detect_ms,
                    det_every_ms=self.det_every_ms, frames=len(images), version=self.version)

    def infer(self, image, timestamp_ms=None):
        """BGR image -> (boxes xyxy, keypoints (n, 133, 2) pixels, scores (n, 133))."""
        return self.infer_batch([image], [timestamp_ms])[0]

    def _due(self, timestamp_ms):
        return (self._boxes is None or not self.det_every_ms or timestamp_ms is None or self._detected_ms is None
                or timestamp_ms - self._detected_ms >= self.det_every_ms or timestamp_ms < self._detected_ms)

    def _crop(self, image, box):
        """rtmlib ``RTMPose.preprocess`` without its normalization (same box padding and affine warp): (HWC uint8,
        center, scale). :meth:`_normalized` normalizes a whole batch at once."""
        import numpy as np
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
        center, scale = bbox_xyxy2cs(np.array(box), padding=1.25)
        warped, scale = top_down_affine(self.pose.model_input_size, scale, center, image)
        return warped, center, scale

    def _normalized(self, warped):
        """Crops (HWC uint8) -> the model input (N, 3, H, W) float32, normalized as rtmlib does ((x - mean) / std; rtmlib
        computes it in float64 and casts, float32 differs by at most one rounding step)."""
        import numpy as np
        batch = np.empty((len(warped), 3, *warped[0].shape[:2]), 'f4')
        for index, crop in enumerate(warped):
            batch[index] = crop.transpose(2, 0, 1)
        batch -= self._mean[None, :, None, None]
        batch /= self._std[None, :, None, None]
        return batch

    def infer_batch(self, images, timestamps_ms):
        """Consecutive frames of one stream -> one (boxes, keypoints, scores) per frame; the pose model runs once for
        all their crops. The detector runs where it is due; a frame where it is not due uses the box around the body
        keypoints of the latest frame before the batch (``track_box``), or the detection of an earlier frame of it."""
        import numpy as np
        boxes_of = []
        for image, timestamp_ms in zip(images, timestamps_ms):
            if self._due(timestamp_ms):
                boxes = [list(map(float, box)) for box in self.detector(image)][:self.max_poses]
                self._detected_ms, self._boxes = timestamp_ms, boxes or None
            boxes_of.append(list(self._boxes or ()))
        crops = [(index, box, *self._crop(images[index], box)) for index, boxes in enumerate(boxes_of) for box in boxes]
        found = [([], np.zeros((0, POINTS, 2), 'f4'), np.zeros((0, POINTS), 'f4')) for _ in images]
        if not crops:
            self._boxes = None if self.det_every_ms else self._boxes
            return found
        name = self.pose.session.get_inputs()[0].name
        outputs = [], []
        start = 0
        while start < len(crops):
            left = min(len(crops) - start, self.max_crops)
            size = next((size for size in self._fixed if size <= left), 1)     # largest fixed size that fits
            part = self._normalized([crop[2] for crop in crops[start:start + size]])
            session = self._fixed.get(size, self.pose.session)
            for kept, output in zip(outputs, session.run(None, {name: part})):
                kept.append(output)
            start += size
        simcc_x, simcc_y = (np.concatenate(kept) for kept in outputs)
        per_frame = {}
        for row, (index, box, _, center, scale) in enumerate(crops):
            points, values = self.pose.postprocess([simcc_x[row:row + 1], simcc_y[row:row + 1]], center, scale)
            per_frame.setdefault(index, []).append((box, points[0], values[0]))
        for index, items in per_frame.items():
            found[index] = ([box for box, _, _ in items], np.asarray([p for _, p, _ in items], 'f4'),
                            np.asarray([v for _, _, v in items], 'f4'))
        if self.det_every_ms:     # the boxes of the next frames: around the last frame's body keypoints
            height, width = images[-1].shape[:2]
            last_boxes, last_points, last_scores = found[-1]
            tracked = [track_box(points[:17], values[:17], width, height)
                       for points, values in zip(last_points, last_scores)]
            self._boxes = [box for box in tracked if box is not None] or None
        return found


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
    """The module in this process: the models answer from their own environment (``RemoteRTMW``).

    ``batch``: a lane may hand it up to this many waiting frames at once (:meth:`process_batch`); 1 is one by one.
    ``opener(settings)``: opens another worker of the same model (``settings`` override its config, e.g. another
    compute unit); with it the module can :meth:`fork` (an L2 chain of the Pipeline)."""
    name = 'rtmw_pose'
    modalities = ('video',)
    owns_process = True

    def __init__(self, remote, *, joints=None, tracker='ocsort', batch=1, opener=None, fork_settings=None):
        if type(batch) is not int or not 1 <= batch <= MAX_FRAMES:
            raise ValueError(f'rtmw pose batch is 1..{MAX_FRAMES} frames')
        self.remote, self.joints, self.version = remote, joints, remote.model_id
        self.tracker, self.tracker_name, self.batch = create_tracker(tracker), tracker, batch
        self._opener, self._fork_settings, self.spare = opener, dict(fork_settings or {}), None
        if opener is None:
            self.fork = None

    async def fork(self):
        """Another worker of the same model (the same weights; ``fork_settings`` may place it on another compute
        unit) with a fresh tracker: the one :meth:`prestart` opened, else a new one."""
        if self.spare is not None:
            spare, self.spare = self.spare, None
            return spare
        remote = await self._opener(self._fork_settings)
        return type(self)(remote, joints=self.joints, tracker=self.tracker_name, batch=self.batch, opener=self._opener,
                          fork_settings=self._fork_settings)

    async def prestart(self):
        """Open the next fork now (an L2 worker that starts later does not wait for its model to load)."""
        if self.spare is None and self._opener is not None:
            self.spare = await self.fork()
        return self.spare

    async def give_back(self, fork):
        """A fork that finished its interval: kept (reset) as the next one, or closed if one is kept already."""
        if self.spare is None and not getattr(self, '_closed', False):
            await fork.reset()
            self.spare = fork
        else:
            await fork.close()

    async def benchmark(self, **options):
        """This module's worker speed now (``RemoteRTMW.benchmark``)."""
        return await self.remote.benchmark(**options)

    async def process(self, frame):
        return (await self.process_batch([frame]))[0]

    async def process_batch(self, frames):
        """Consecutive frames -> one output each, in order (the tracker follows them one by one)."""
        found = await self.remote.infer_batch([frame.data for frame in frames], [frame.timestamp_ms for frame in frames])
        outputs = []
        for frame, poses in zip(frames, found):
            tracks = associate(poses, self.tracker, frame)
            outputs.append(committed(frame, poses, tracks, self.name, self.version, self.tracker_name, self.joints))
        return outputs

    async def reset(self):
        self.tracker.reset()
        await self.remote.reset()

    async def close(self):
        self._closed = True
        spare, self.spare = self.spare, None
        if spare is not None:
            await spare.close()
        await self.remote.close()
