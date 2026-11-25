"""Training-free pose features and exemplar matching: count an owner-demonstrated pose or movement.

Two math modules, no trained weights:

``PoseChannels`` turns one person's body keypoints (any keypoint model mapped to the COCO-style
body joints below: MediaPipe Pose, Sapiens2, RTMW) into a fixed vector that does not depend on where
the person stands or how large they appear: joint angles, plus wrist positions relative to the
shoulders in shoulder widths. The same vector feeds :class:`gamst.providers.repetition.RepetitionCounter`
(periodic movements) and :class:`ExemplarEvents` (a demonstrated pose or movement).

``ExemplarEvents`` compares that stream with an exemplar the owner demonstrated once (a user-defined
``exemplar`` reference input):

- ``hold``: the exemplar is one pose. A match starts when the distance stays below ``enter`` for
  ``min_hold_ms`` and ends when it rises above ``exit`` (hysteresis); each match is one event.
- ``sequence``: the exemplar is a movement (a list of vectors). Subsequence dynamic time warping
  over a sliding window finds where the recent motion ends a close copy of the exemplar; matches
  closer together than half the exemplar's length are one event.

Distances are mean absolute differences per channel over channels visible in both, so a value
reads as "average joint-angle difference" (angles are in units of pi radians).

Keypoints are normalized per axis (x by the image width, y by its height), so on a non-square image
angles and lengths only mean something after x is scaled by the aspect ratio. ``PoseChannels`` does
that from the image size the pose source reports (``CHANNELS_VERSION`` 2; version 1 did not, so
exemplars enrolled with it do not match on non-square images and have to be enrolled again).
"""
from __future__ import annotations

from collections import deque
import math
import threading

from ..capabilities import CapabilitySpec, OutputField

# COCO-17 indices into MediaPipe's 33 landmarks; other keypoint models pass their own mapping.
MEDIAPIPE_BODY = dict(left_shoulder=11, right_shoulder=12, left_elbow=13, right_elbow=14, left_wrist=15,
                      right_wrist=16, left_hip=23, right_hip=24, left_knee=25, right_knee=26,
                      left_ankle=27, right_ankle=28)
ANGLES = (('left_elbow', 'left_shoulder', 'left_wrist'), ('right_elbow', 'right_shoulder', 'right_wrist'),
          ('left_shoulder', 'left_elbow', 'left_hip'), ('right_shoulder', 'right_elbow', 'right_hip'),
          ('left_hip', 'left_shoulder', 'left_knee'), ('right_hip', 'right_shoulder', 'right_knee'),
          ('left_knee', 'left_hip', 'left_ankle'), ('right_knee', 'right_hip', 'right_ankle'))
CHANNELS = tuple(f'angle_{vertex}' for vertex, _, _ in ANGLES) + (
    'left_wrist_dx', 'left_wrist_dy', 'right_wrist_dx', 'right_wrist_dy', 'wrist_gap')
CHANNELS_VERSION = 2


def _angle(a, vertex, b):
    ax, ay, bx, by = a[0] - vertex[0], a[1] - vertex[1], b[0] - vertex[0], b[1] - vertex[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return None
    cosine = max(-1., min(1., (ax * bx + ay * by) / (na * nb)))
    return math.acos(cosine) / math.pi


def pose_channels(landmarks, mapping=MEDIAPIPE_BODY, *, min_visibility=.5, aspect=1.):
    """``landmarks``: list of {x, y, visibility?}. Returns (channels, mask); masked channels are 0.

    ``aspect`` is image width / height when x and y are normalized per axis.
    """
    points = {}
    for name, index in mapping.items():
        if index < len(landmarks):
            point = landmarks[index]
            if point.get('visibility', 1.) >= min_visibility:
                points[name] = (float(point['x']) * aspect, float(point['y']))
    values, mask = [], []
    for vertex, a, b in ANGLES:
        angle = _angle(points[a], points[vertex], points[b]) if {vertex, a, b} <= points.keys() else None
        values.append(angle if angle is not None else 0.)
        mask.append(angle is not None)
    shoulders = ('left_shoulder', 'right_shoulder')
    if set(shoulders) <= points.keys():
        (lx, ly), (rx, ry) = points['left_shoulder'], points['right_shoulder']
        cx, cy, width = (lx + rx) / 2, (ly + ry) / 2, math.hypot(lx - rx, ly - ry)
    else:
        width = 0
    for wrist in ('left_wrist', 'right_wrist'):
        ok = width > 0 and wrist in points
        values += [(points[wrist][0] - cx) / width if ok else 0., (points[wrist][1] - cy) / width if ok else 0.]
        mask += [ok, ok]
    ok = width > 0 and {'left_wrist', 'right_wrist'} <= points.keys()
    values.append(math.hypot(points['left_wrist'][0] - points['right_wrist'][0],
                             points['left_wrist'][1] - points['right_wrist'][1]) / width if ok else 0.)
    mask.append(ok)
    return values, mask


def distance(a, mask_a, b, mask_b):
    pairs = [abs(x - y) for x, y, m, n in zip(a, b, mask_a, mask_b) if m and n]
    return sum(pairs) / len(pairs) if len(pairs) >= 4 else None


def _landmarks(pose):
    """``landmarks`` [{x, y, visibility}] or ``keypoints`` [[x, y, score], ...] (normalized image units)."""
    if not isinstance(pose, dict):
        return None
    if pose.get('landmarks'):
        return pose['landmarks']
    keypoints = pose.get('keypoints')
    if isinstance(keypoints, list) and keypoints:
        return [dict(x=point[0], y=point[1], visibility=point[2] if len(point) > 2 else 1.) for point in keypoints]
    return None


def _aspect(*sources):
    """Image width / height from the first source that reports both, else None."""
    for source in sources:
        if isinstance(source, dict) and source.get('image_width') and source.get('image_height'):
            return source['image_width'] / source['image_height']
    return None


class PoseChannels:
    """Pose record -> the largest visible person's feature vector.

    Reads a pose list (``scene.poses``) or a ``{"poses": [...], "summary": {...}}`` value. The image
    size comes from each pose or the summary; a source that reports neither is taken as square and its
    record has no ``image_aspect``.
    """

    def __init__(self, *, mapping=MEDIAPIPE_BODY, source='mediapipe_pose', min_visibility=.5):
        self.mapping, self.min_visibility = mapping, min_visibility
        self.spec = CapabilitySpec(
            'vision.pose_channels', 'vision.pose_channels', 'video', 'gamst.pose_channels', f'{CHANNELS_VERSION}.{source}',
            (OutputField('channels', 'array'), OutputField('mask', 'array'), OutputField('visible', 'integer'),
             OutputField('pose_index', 'integer', required=False),
             OutputField('image_aspect', 'number', unit='width/height', required=False)),
            'frame', freshness_ms=1000, interpretation='measurement', devices=('cpu',), license='Apache-2.0')

    def process(self, frame):
        # A pose list (``scene.poses``, ``scene.target_poses``) or a record with ``poses`` (a worker's value).
        value = frame.data
        poses = value if isinstance(value, list) else (value.get('poses') or []) if isinstance(value, dict) else []
        summary = value.get('summary') if isinstance(value, dict) else None
        best = None
        for index, pose in enumerate(poses):
            landmarks = _landmarks(pose)
            if not landmarks:
                continue
            aspect = _aspect(pose, summary)
            values, mask = pose_channels(landmarks, self.mapping, min_visibility=self.min_visibility,
                                         aspect=1. if aspect is None else aspect)
            if best is None or sum(mask) > sum(best[1]):
                best = (values, mask, index, aspect)
        if best is None:
            return None
        values, mask, index, aspect = best
        result = dict(channels=[round(v, 5) for v in values], mask=mask, visible=sum(mask), pose_index=index)
        if aspect is not None:
            result['image_aspect'] = round(aspect, 5)
        return self.spec.validate(result)


def subsequence_dtw(template, series):
    """Distance of the best warp of ``template`` onto a suffix of ``series`` ending at its last item.

    Items are (values, mask). Open beginning (any start in ``series``), fixed end (the latest item):
    the cost of ``template`` ending now. Returns the mean step cost, or None.
    """
    n, m = len(template), len(series)
    if n == 0 or m == 0:
        return None
    infinity = float('inf')
    previous = [0.] * (m + 1)  # open start: the template may begin anywhere
    steps_previous = [0] * (m + 1)
    for i in range(1, n + 1):
        current, steps = [infinity] * (m + 1), [0] * (m + 1)
        for j in range(1, m + 1):
            cost = distance(template[i - 1][0], template[i - 1][1], series[j - 1][0], series[j - 1][1])
            cost = 1. if cost is None else cost
            choices = ((previous[j - 1], steps_previous[j - 1]), (previous[j], steps_previous[j]),
                       (current[j - 1], steps[j - 1]))
            best, length = min(choices, key=lambda item: item[0])
            current[j], steps[j] = best + cost, length + 1
        previous, steps_previous = current, steps
    if previous[m] == infinity:
        return None
    return previous[m] / max(1, steps_previous[m])


class ExemplarEvents:
    """Match a feature-channel stream against a demonstrated exemplar; one event per match.

    Any fixed-length channel stream works: pose channels (video), voice-style channels (audio), IMU.
    ``modality`` and ``capability_id`` name the record so a video and an audio matcher can run together.
    """

    def __init__(self, *, reference='exemplar', mode='hold', enter=.12, exit=.18, min_hold_ms=200,
                 refractory_ms=None, window_s=None, max_gap_ms=1000, modality='video',
                 capability_id='temporal.exemplar_events'):
        if mode not in ('hold', 'sequence'):
            raise ValueError("mode must be 'hold' or 'sequence'")
        if not 0 < enter <= exit:
            raise ValueError('require 0 < enter <= exit')
        self.reference_name, self.mode = reference, mode
        self.enter, self.exit, self.min_hold_ms = enter, exit, min_hold_ms
        self.refractory_ms, self.window_s, self.max_gap_ms = refractory_ms, window_s, max_gap_ms
        self.template = None
        self.spec = CapabilitySpec(
            capability_id, 'temporal.exemplar_events', modality, 'gamst.exemplar_events',
            f'1.{mode}', (OutputField('count', 'integer'), OutputField('status', 'string'),
                          OutputField('distance', 'number', required=False), OutputField('events', 'array')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms, interpretation='estimate',
            devices=('cpu',), license='Apache-2.0')
        self._lock = threading.RLock()
        self._clear()

    def bind(self, pipeline):
        """Read the owner's exemplar: a channel vector (hold) or a list of them (sequence)."""
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'exemplar':
            raise ValueError(f'exemplar_events needs an exemplar reference named {self.reference_name!r}')
        self.set_exemplar(reference['value'])
        self.label = reference['label']

    def set_exemplar(self, value):
        items = value if self.mode == 'sequence' else [value]
        template = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('channels'), list):
                raise ValueError('exemplar items are {"channels": [...], "mask": [...]} pose-channel values')
            channels = [float(v) for v in item['channels']]
            mask = [bool(v) for v in item.get('mask', [True] * len(channels))]
            template.append((channels, mask))
        if not template:
            raise ValueError('empty exemplar')
        self.template = template
        if self.mode == 'sequence':
            self.window = max(4, 2 * len(template))

    def _clear(self):
        self._count, self._inside, self._candidate_ms = 0, False, None
        self._started_ms = self._last_ms = self._last_match_ms = None
        self._history = deque()
        self._best = None

    def reset(self):
        with self._lock:
            self._clear()

    def process(self, frame):
        with self._lock:
            if self.template is None:
                raise RuntimeError('no exemplar: add an exemplar reference before the module')
            value = frame.data if isinstance(frame.data, dict) else {}
            channels, mask = value.get('channels'), value.get('mask')
            if not isinstance(channels, list) or not isinstance(mask, list):
                return None
            ts = frame.timestamp_ms
            if self._last_ms is not None and ts - self._last_ms > self.max_gap_ms:
                count = self._count
                self._clear()
                self._count = count  # a gap ends a match; it does not undo counted ones
            self._last_ms = ts
            item = ([float(v) for v in channels], [bool(v) for v in mask])
            events = []
            if self.mode == 'hold':
                current = distance(self.template[0][0], self.template[0][1], *item)
                events = self._hold(ts, current)
            else:
                self._history.append((ts, item))
                while len(self._history) > self.window:
                    self._history.popleft()
                current = subsequence_dtw(self.template, [entry for _, entry in self._history]) \
                    if len(self._history) >= max(2, len(self.template) // 2) else None
                events = self._sequence(ts, current)
            status = 'matching' if self._inside else 'watching'
            result = dict(count=self._count, status=status, events=events)
            if current is not None:
                result['distance'] = round(current, 5)
            return self.spec.validate(result)

    def _hold(self, ts, current):
        if current is None:
            return []
        if not self._inside:
            if current <= self.enter:
                self._candidate_ms = self._candidate_ms or ts
                if ts - self._candidate_ms >= self.min_hold_ms:
                    self._inside, self._started_ms, self._best = True, self._candidate_ms, current
                    self._count += 1
                    return [dict(event='exemplar.matched', index=self._count, reference=self.reference_name,
                                 start_ms=self._started_ms, distance=round(current, 5))]
            else:
                self._candidate_ms = None
            return []
        self._best = min(self._best, current)
        if current > self.exit:
            self._inside, self._candidate_ms = False, None
            return [dict(event='exemplar.ended', index=self._count, reference=self.reference_name,
                         start_ms=self._started_ms, end_ms=ts, best_distance=round(self._best, 5))]
        return []

    def _sequence(self, ts, current):
        if current is None:
            return []
        refractory = self.refractory_ms
        if refractory is None and len(self._history) >= 2:
            span = self._history[-1][0] - self._history[0][0]
            refractory = int(.5 * span * len(self.template) / max(1, len(self._history) - 1))
        if current <= self.enter and not self._inside:
            if self._last_match_ms is not None and ts - self._last_match_ms < (refractory or 0):
                return []
            self._inside, self._last_match_ms = True, ts
            self._count += 1
            return [dict(event='exemplar.matched', index=self._count, reference=self.reference_name,
                         end_ms=ts, distance=round(current, 5))]
        if current > self.exit:
            self._inside = False
        return []
