"""Training-free pose features and exemplar matching: count an owner-demonstrated pose or movement.

Two math modules, no trained weights:

``PoseChannels`` turns one person's body keypoints (any keypoint model mapped to the COCO-style
body joints below: MediaPipe Pose, Sapiens2, RTMW) into a fixed vector that does not depend on where
the person stands or how large they appear: joint angles, plus wrist positions relative to the
shoulders in shoulder widths. The same vector feeds :class:`gamst.providers.repetition.RepetitionCounter`
(periodic movements) and :class:`ExemplarEvents` (a demonstrated pose or movement).

``ExemplarEvents`` is the one real-time counter. It compares that stream with a movement the owner
specified (a user-defined ``exemplar`` reference input), by examples or by a rule:

- ``hold``: the examples are poses. A match starts when the distance stays below ``enter`` for
  ``min_hold_ms`` and ends when it rises above ``exit`` (hysteresis); each match is one event.
- ``sequence``: the examples are movements (lists of vectors). Subsequence dynamic time warping
  finds where the recent motion ends a close copy of an example; matches closer together than half the
  example's length are one event.
- ``rule``: a declarative rule over channels and joint relations (:mod:`gamst.providers.movement_rule`).

A reference holds several positive and optional negative examples; the matching and the derivation of
thresholds from them are in :mod:`gamst.providers.movement_match`.

Distances are mean absolute differences per channel over channels visible in both, so a value
reads as "average joint-angle difference" (angles are in units of pi radians).

Keypoints are normalized per axis (x by the image width, y by its height), so on a non-square image
angles and lengths only mean something after x is scaled by the aspect ratio. ``PoseChannels`` does
that from the image size the pose source reports (``CHANNELS_VERSION`` 2; version 1 did not, so
exemplars enrolled with it do not match on non-square images and have to be enrolled again).
"""
from __future__ import annotations

import math
import threading

from ..capabilities import CapabilitySpec, OutputField
from ..private_payload import poses_of
from .movement_match import ExampleScorer, SequenceTrack, distance, example  # noqa: F401  (distance: shared metric)
from .movement_rule import RuleScorer

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


def joint_angle(a, vertex, b):
    """Angle at ``vertex`` in units of pi radians, None for a zero-length side."""
    ax, ay, bx, by = a[0] - vertex[0], a[1] - vertex[1], b[0] - vertex[0], b[1] - vertex[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return None
    cosine = max(-1., min(1., (ax * bx + ay * by) / (na * nb)))
    return math.acos(cosine) / math.pi


def body_points(landmarks, mapping=MEDIAPIPE_BODY, *, min_visibility=.5, aspect=1.):
    """{joint: (x, y)} of the visible mapped joints, x scaled so both axes have one unit."""
    points = {}
    for name, index in mapping.items():
        if index < len(landmarks):
            point = landmarks[index]
            if point.get('visibility', 1.) >= min_visibility:
                points[name] = (float(point['x']) * aspect, float(point['y']))
    return points


def pose_channels(landmarks, mapping=MEDIAPIPE_BODY, *, min_visibility=.5, aspect=1.):
    """``landmarks``: list of {x, y, visibility?}. Returns (channels, mask); masked channels are 0.

    ``aspect`` is image width / height when x and y are normalized per axis.
    """
    points = body_points(landmarks, mapping, min_visibility=min_visibility, aspect=aspect)
    values, mask = [], []
    for vertex, a, b in ANGLES:
        angle = joint_angle(points[a], points[vertex], points[b]) if {vertex, a, b} <= points.keys() else None
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


def _aspect(pose):
    """Width / height of the image this pose is normalized to, else None."""
    if pose.get('image_width') and pose.get('image_height'):
        return pose['image_width'] / pose['image_height']
    return None


def largest_pose(poses, mapping=MEDIAPIPE_BODY, min_visibility=.5):
    """(channels, mask, index, aspect) of the pose with the most visible channels, or None."""
    best = None
    for index, pose in enumerate(poses):
        landmarks = pose.get('landmarks') if isinstance(pose, dict) else None
        if not landmarks:
            continue
        aspect = _aspect(pose)
        values, mask = pose_channels(landmarks, mapping, min_visibility=min_visibility,
                                     aspect=1. if aspect is None else aspect)
        if best is None or sum(mask) > sum(best[1]):
            best = (values, mask, index, aspect)
    return best


class PoseChannels:
    """Pose record -> the largest visible person's feature vector.

    Reads the joints input of a pose or target-person module (:class:`gamst.private_payload.PrivatePoses`),
    or a legacy ``scene.poses`` value with landmarks. Each pose names the size of its own image (a crop's
    own size for poses found in a crop); a pose without one is taken as square and its record has no
    ``image_aspect``.
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
        poses = poses_of(frame.data)
        if poses is None:
            return None
        poses = [pose for pose in poses if isinstance(pose, dict)]
        if poses and not any('landmarks' in pose for pose in poses):
            raise ValueError('these poses carry no joint positions; consume the pose module\'s joints input')
        best = largest_pose(poses, self.mapping, self.min_visibility)
        if best is None:
            return None
        values, mask, index, aspect = best
        result = dict(channels=[round(v, 5) for v in values], mask=mask, visible=sum(mask), pose_index=index)
        if aspect is not None:
            result['image_aspect'] = round(aspect, 5)
        return self.spec.validate(result)


def subsequence_dtw(template, series):
    """Distance of the best warp of ``template`` onto a suffix of ``series`` ending at its last item.

    Items are (values, mask). Open beginning (any start in ``series``), fixed end (the latest item): the
    cost of ``template`` ending now. Returns the mean step cost, or None.
    """
    if not template or not series:
        return None
    track = SequenceTrack(template, window=len(series))
    for index, item in enumerate(series):
        track.update(index, item)
    return track.cost()


class ExemplarEvents:
    """The movement counter: match a channel stream against a specified movement; one event per match.

    Any fixed-length channel stream works: pose channels (video), voice-style channels (audio), IMU.
    ``modality`` and ``capability_id`` name the record so a video and an audio matcher can run together.

    It may also consume (both optional):

    - an identity record (``vision.target_person``, ``audio.speaker_match``): its newest value is
      attached to each event as ``person``;
    - a pose module's joints input: needed only by a rule that uses joint relations. Joints stay in memory.

    Events: ``exemplar.matched`` (``start_ms``, ``end_ms`` = when it was recognized) and, when a held pose or
    a sequence is over, ``exemplar.ended`` with the same ``index`` and ``start_ms``, its final ``end_ms`` and
    ``best_distance``. Both carry ``reference``, ``reference_id``, ``label``, ``method`` (``examples``,
    ``rule`` or ``text_finder``) and ``confirmed`` (false when an unconfirmed proposal is part of the
    movement), and the revision of the movement (``reference_revision``): the Pipeline may give another movement
    while a run goes on (:meth:`prepare`, :meth:`schedule`). It takes over at the first frame captured at or after
    the stated time; a match in progress ends there (``cut: instruction_changed``) and the count starts at 0.
    """

    def __init__(self, *, reference='exemplar', mode=None, enter=.12, exit=.18, min_hold_ms=200,
                 refractory_ms=None, max_gap_ms=1000, modality='video', capability_id='temporal.exemplar_events',
                 fixed_thresholds=False, mapping=MEDIAPIPE_BODY, min_visibility=.5, contract=None, speaker_bound=True):
        """``contract``: what a sensor's channels are; a movement learned from others is refused.
        ``speaker_bound=False``: the sound is not tied to an enrolled voice; every event says so."""
        if mode not in (None, 'hold', 'sequence'):
            raise ValueError("mode must be 'hold' or 'sequence'")
        if not 0 < enter <= exit:
            raise ValueError('require 0 < enter <= exit')
        self.reference_name, self._wanted_mode, self.mode = reference, mode, mode
        self.enter, self.exit, self.min_hold_ms = enter, exit, min_hold_ms
        self.fixed_thresholds = fixed_thresholds     # the caller's enter / exit win over the reference's
        self.refractory_ms, self.max_gap_ms = refractory_ms, max_gap_ms
        self.contract, self.speaker_bound = contract, bool(speaker_bound)
        self.mapping, self.min_visibility = mapping, min_visibility
        self.label, self.scorer, self.rule, self.info = reference, None, None, {}
        self.identity, self.from_ms, self._methods, self._next = {}, 0, [], []
        self._defaults = dict(enter=enter, exit=exit)
        self.spec = CapabilitySpec(
            capability_id, 'temporal.exemplar_events', modality, 'gamst.exemplar_events',
            '3', (OutputField('count', 'integer'), OutputField('status', 'string'),
                  OutputField('distance', 'number', required=False), OutputField('events', 'array'),
                  OutputField('reference_id', 'string', required=False),
                  OutputField('reference_revision', 'integer', required=False)),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms, interpretation='estimate',
            devices=('cpu',), license='Apache-2.0')
        self._lock = threading.RLock()
        self._clear()

    # ------------------------------------------------------------------ the movement

    def bind(self, pipeline):
        """Read the owner's movement: a saved reference (examples or a rule), or the bare value of one example."""
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'exemplar':
            raise ValueError(f'exemplar_events needs an exemplar reference named {self.reference_name!r}')
        self._adopt(self.prepare(reference), reference.get('from_ms', 0))

    # ------------------------------------------------------------------ another movement while a run goes on

    def prepare(self, reference):
        """A checked movement to run from a stated time on; nothing of the running one changes."""
        ready = type(self)(reference=self.reference_name, mode=self._wanted_mode, min_hold_ms=self.min_hold_ms,
                           refractory_ms=self.refractory_ms, max_gap_ms=self.max_gap_ms, modality=self.spec.modality,
                           capability_id=self.spec.id, fixed_thresholds=self.fixed_thresholds, mapping=self.mapping,
                           min_visibility=self.min_visibility, contract=self.contract, speaker_bound=self.speaker_bound,
                           **self._defaults)
        ready.set_exemplar(reference['value'])
        ready.label = reference['label']
        ready.identity = {f'reference_{key}': reference[key] for key in ('id', 'revision') if reference.get(key) is not None}
        return ready

    def offline(self, reference):
        """A separate counter for records that were kept (nothing of it is committed)."""
        ready = self.prepare(reference)
        if ready.rule is not None and ready.rule.needs_joints:
            raise ValueError('this rule uses joint relations; joint positions are never kept')
        return ready

    def schedule(self, ready, from_ms):
        """``ready`` (None: nothing is looked for) takes over at the first frame captured at or after ``from_ms``."""
        with self._lock:
            self._next = [item for item in self._next if item[0] < from_ms] + [(from_ms, ready)]

    def _adopt(self, ready, from_ms):
        for name in ('label', 'scorer', 'rule', 'mode', 'enter', 'exit', '_methods', 'identity'):
            setattr(self, name, getattr(ready, name) if ready is not None else None if name in ('scorer', 'rule', 'mode') else
                    getattr(self, name))
        self.info, self.from_ms = dict(ready.info, removed=False) if ready is not None else dict(self.info, removed=True), from_ms
        person, joints = self._person, self._saw_joints
        self._clear()
        self._person, self._saw_joints = person, joints

    def _value(self, **value):
        return self.spec.validate(dict(value, count=self._count, **self.identity))

    def _switch(self, ts):
        """Take over what is due at ``ts``; what the old movement still says about its last match."""
        said = []
        while self._next and self._next[0][0] <= ts:
            from_ms, ready = self._next.pop(0)
            if self._inside:
                events = [self._ended(self._last_ms, cut='instruction_changed')]
                said.append(self._value(status='watching', events=events))
            self._adopt(ready, from_ms)
        return said

    def set_exemplar(self, value):
        """``{"positives": [...], ...}`` or ``{"rule": {...}}`` as saved by ``gamst recipe enroll``; or one bare
        example, ``{"channels", "mask"}`` (hold) or a list of them (sequence), as files before format 2 hold."""
        with self._lock:
            self.scorer = self.rule = None
            learned = value.get('channels') if isinstance(value, dict) and 'mask' not in value else None
            if isinstance(learned, str) and learned != self.contract and (self.contract or learned.startswith('sensor.')):
                raise ValueError(f'the movement was specified on {learned}; this module reads {self.contract or "other channels"}')
            if isinstance(value, dict) and 'rule' in value:
                self.mode, self.rule = 'rule', RuleScorer(value['rule'])
                self.enter, self.exit = 0., self.rule.margin
                self._methods = ['rule']
            else:
                if isinstance(value, dict) and 'positives' in value:
                    entries, mode = value, value.get('mode')
                    if not self.fixed_thresholds and 'enter' in value:
                        enter, exit_ = float(value['enter']), float(value['exit'])
                        if not 0 < enter <= exit_ < float('inf'):
                            raise ValueError('require 0 < enter <= exit')
                        self.enter, self.exit = enter, exit_
                else:
                    if value == [] or value is None:
                        raise ValueError('empty exemplar')
                    entries = dict(positives=[dict(value=value)])
                    mode = 'sequence' if isinstance(value, list) else 'hold'
                if mode not in ('hold', 'sequence') or self._wanted_mode not in (None, mode):
                    raise ValueError(f'this is a {mode} exemplar; the module runs {self._wanted_mode or "hold or sequence"}')
                positives, negatives = ([example(item['value'], span_ms=item.get('span_ms'))
                                         for item in entries.get(key) or ()] for key in ('positives', 'negatives'))
                self.mode, self.scorer = mode, ExampleScorer(mode, positives, negatives)
                self._methods = [item.get('method', 'examples') for item in entries['positives']]
            saved = value if isinstance(value, dict) else {}
            self.identity = {'reference_id': saved['id']} if saved.get('id') else {}
            self.info = dict(confirmed=bool(saved.get('confirmed', True)),
                             threshold_source='caller' if self.fixed_thresholds and self.rule is None else
                             saved.get('threshold_source', 'rule' if self.rule is not None else 'default'),
                             positives=len(self._methods) if self.rule is None else 0,
                             negatives=len(self.scorer.negatives) if self.rule is None else 0)
            self._clear()

    def status(self):
        return dict(self.info, **self.identity, from_ms=self.from_ms, scheduled=[item[0] for item in self._next],
                    label=self.label, mode=self.mode, enter=self.enter, exit=self.exit, count=self._count,
                    methods=sorted(set(self._methods)) if self.scorer or self.rule else [])

    # ------------------------------------------------------------------ the counter

    def _clear(self):
        self._count, self._inside, self._candidate_ms = 0, False, None
        self._started_ms = self._last_ms = self._last_match_ms = self._end_ms = None
        self._best = self._method = None
        self._step = 0
        self._person, self._saw_joints, self._channel_frames = None, False, 0
        if self.scorer is not None:
            self.scorer.reset()
        if getattr(self, 'rule', None) is not None:
            self.rule.reset()

    def reset(self):
        with self._lock:
            self._clear()

    def _event(self, name, **values):
        event = dict(event=name, index=self._count, reference=self.reference_name, label=self.label,
                     method=self._method, confirmed=self.info['confirmed'], **values)
        if not self.speaker_bound:
            event['speaker'] = 'not_bound'
        event.update(self.identity)
        if self._person is not None and self._last_ms - self._person['at_ms'] <= self.max_gap_ms:
            event['person'] = dict(self._person)
        return event

    def _ended(self, end_ms, **extra):
        self._inside, self._candidate_ms = False, None
        return self._event('exemplar.ended', start_ms=self._started_ms, end_ms=end_ms,
                           best_distance=round(self._best, 5), **extra)

    def _identity(self, frame, value):
        """The newest identity verdict about the person in view (a record another module committed)."""
        self._person = dict(reference=value['reference'], state=value['state'], at_ms=frame.timestamp_ms)
        for key in ('face_score', 'body_score', 'score'):
            if type(value.get(key)) in (int, float):
                self._person[key] = value[key]

    def _input(self, frame):
        """(channels, mask, points or None) of a frame to match, else None."""
        poses = poses_of(frame.data)
        if poses is not None:
            if self.rule is None or not self.rule.needs_joints:
                return None
            self._saw_joints = True
            best = largest_pose(poses, self.mapping, self.min_visibility)
            if best is None:
                return None
            channels, mask, index, aspect = best
            return channels, mask, body_points(poses[index]['landmarks'], self.mapping,
                                               min_visibility=self.min_visibility,
                                               aspect=1. if aspect is None else aspect)
        value = frame.data if isinstance(frame.data, dict) else {}
        channels, mask = value.get('channels'), value.get('mask')
        if not isinstance(channels, list) or not isinstance(mask, list):
            if isinstance(value.get('reference'), str) and isinstance(value.get('state'), str):
                self._identity(frame, value)
            return None
        if self.rule is not None and self.rule.needs_joints:
            self._channel_frames += 1
            if self._channel_frames > 2 and not self._saw_joints:
                raise ValueError('this rule uses joint relations: link the module to the pose module\'s joints input')
            return None
        return [float(v) for v in channels], [bool(v) for v in mask], None

    def process(self, frame):
        with self._lock:
            said = self._switch(frame.timestamp_ms)
            if self.info.get('removed') or frame.timestamp_ms < self.from_ms:
                return said or None
            if self.scorer is None and self.rule is None:
                raise RuntimeError('no exemplar: add an exemplar reference before the module')
            found = self._input(frame)
            if found is None:
                return said or None
            channels, mask, points = found
            ts, events = frame.timestamp_ms, []
            if self._last_ms is not None and ts - self._last_ms > self.max_gap_ms:
                if self._inside:   # a gap ends a match; it does not undo counted ones
                    events.append(self._ended(self._last_ms, cut='gap'))
                count, person, joints = self._count, self._person, self._saw_joints
                self._clear()
                self._count, self._person, self._saw_joints = count, person, joints
            self._last_ms = ts
            if self.rule is not None:
                current = self._rule(ts, channels, mask, points, events)
            else:
                current, index, start, rival = self.scorer.update(ts, (channels, mask), limit=self.exit)
                (self._hold if self.mode == 'hold' else self._sequence)(ts, current, index, start, rival, events)
            result = self._value(status='matching' if self._inside else 'watching', events=events,
                                 **({} if current is None else {'distance': round(current, 5)}))
            return [*said, result] if said else result

    def finish(self):
        """End of input: the match in progress ends at the last frame."""
        with self._lock:
            if not self._inside:
                return None
            events = [self._ended(self._last_ms, cut='end_of_input')]
            return self._value(status='watching', events=events)

    def _match(self, ts, start, current, index=None):
        self._count += 1
        self._inside, self._started_ms, self._last_match_ms = True, start, ts
        self._best, self._end_ms, self._method = current, ts, self._methods[index or 0]
        extra = {} if index is None else dict(example=index)
        return self._event('exemplar.matched', start_ms=start, end_ms=ts, distance=round(current, 5), **extra)

    def _hold(self, ts, current, index, start, rival, events):
        if current is None:
            return
        if not self._inside:
            if current <= self.enter and (rival is None or current < rival):
                self._candidate_ms = ts if self._candidate_ms is None else self._candidate_ms
                if ts - self._candidate_ms >= self.min_hold_ms:
                    events.append(self._match(ts, self._candidate_ms, current, index))
            else:
                self._candidate_ms = None
            return
        self._best = min(self._best, current)
        # A negative ends the match by the same hysteresis gap as the threshold does.
        if current > self.exit or (rival is not None and current - rival > self.exit - self.enter):
            events.append(self._ended(ts))

    def _sequence(self, ts, current, index, start, rival, events):
        value = float('inf') if current is None else current
        copy = value <= self.enter and (rival is None or value < rival)
        if copy and (not self._inside or start >= self._end_ms):     # nothing yet, or a copy after the last one
            refractory = self.scorer.half_duration_ms(index) if self.refractory_ms is None else self.refractory_ms
            if self._last_match_ms is None or ts - self._last_match_ms >= refractory:
                if self._inside:
                    events.append(self._ended(self._end_ms))
                events.append(self._match(ts, start, current, index))
                return
        if self._inside:
            if value < self._best:
                self._best, self._end_ms = value, ts
            if value > self.exit:
                events.append(self._ended(self._end_ms))

    def _rule(self, ts, channels, mask, points, events):
        steps = self.rule.steps
        self.rule.observe(ts, channels, mask, points)
        if len(steps) == 1:
            current = self.rule.violation(0, channels, mask, points)
            hold_ms = steps[0].get('hold_ms', self.min_hold_ms)
            if current is None:
                if not self._inside:
                    self._candidate_ms = None
            elif not self._inside:
                if current <= 0:
                    self._candidate_ms = ts if self._candidate_ms is None else self._candidate_ms
                    if ts - self._candidate_ms >= hold_ms:
                        events.append(self._match(ts, self._candidate_ms, current))
                else:
                    self._candidate_ms = None
            else:
                self._best = min(self._best, current)
                if current > self.exit:
                    events.append(self._ended(ts))
            return current
        if self._step and ts - self._end_ms > steps[self._step]['within_ms']:
            self._step, self._candidate_ms = 0, None     # too late: start over with this frame
        current = self.rule.violation(self._step, channels, mask, points)
        if current is None or current > 0:
            self._candidate_ms = None
            return current
        self._candidate_ms = ts if self._candidate_ms is None else self._candidate_ms
        if ts - self._candidate_ms < steps[self._step].get('hold_ms', 0):
            return current
        if self._step == 0:
            self._started_ms = self._candidate_ms
        self._step, self._end_ms, self._candidate_ms = self._step + 1, ts, None
        if self._step == len(steps):
            events.append(self._match(ts, self._started_ms, current))
            self._inside, self._step = False, 0          # ordered movements are events, not states
        return current
