"""Body turns about the vertical axis and time in the air, from 2D joints (figure skating, diving, gymnastics, dance).

A person turning about the vertical axis shows the camera a shoulder line whose signed width (right shoulder x minus
left shoulder x, hips averaged in) changes sign every half turn: facing, side, back, side, facing. Counting the sign
changes with hysteresis (``min_width`` of the torso height on each side) counts half turns; two make a turn. The
direction of a turn cannot be told from 2D joints alone, so it is not reported.

Time in the air: the ankles leave the ground line (the lowest recent ankle position) by more than ``lift`` torso
heights; a jump has a start, an end, its airtime and the turns counted during it.

This is **experimental**: it was checked on synthetic joints only, never on a skater (docs/sports.md). It reads the
joints input of a pose module (the derived input ``pose_joints`` or ``target_joints``) by the pose's own keypoint set (RTMW, YOLO26-pose,
Sapiens2, MediaPipe), or by ``mapping`` (``"coco"``, ``"mediapipe"``) when given. Sampling: a turn shows two
sign changes, so ``gamst recipe check`` requires ``fps >= 4 x max_rev_s`` (rotation sampling with symmetry 2).
"""
from __future__ import annotations

import threading

from ..capabilities import CapabilitySpec, OutputField

COCO = dict(left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12, left_ankle=15, right_ankle=16)
MEDIAPIPE = dict(left_shoulder=11, right_shoulder=12, left_hip=23, right_hip=24, left_ankle=27, right_ankle=28)
MAPPINGS = dict(coco=COCO, mediapipe=MEDIAPIPE)


def _points(pose, mapping, min_visibility):
    if mapping is None:     # the pose's own keypoint set decides (RTMW, YOLO26-pose, Sapiens2, MediaPipe)
        from .pose_exemplar import keypoint_reading
        mapping, min_visibility = keypoint_reading(pose)
    landmarks = pose.get('landmarks') or ()
    found = {}
    for name, index in mapping.items():
        if index < len(landmarks):
            point = landmarks[index]
            if point.get('visibility', 1.) >= min_visibility:
                found[name] = (float(point['x']), float(point['y']))
    return found


class BodySpin:
    def __init__(self, *, mapping=None, min_visibility=.3, min_width=.15, lift=.25, ground_ms=1500, min_air_ms=120,
                 aspect=1., capability_id='sports.spin'):
        self.mapping = None if mapping is None else MAPPINGS[mapping] if isinstance(mapping, str) else dict(mapping)
        self.min_visibility, self.min_width, self.lift = float(min_visibility), float(min_width), float(lift)
        self.ground_ms, self.min_air_ms = int(ground_ms), int(min_air_ms)
        self.aspect = float(aspect)     # image width / height when the poses do not say
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'video', 'gamst.sports_spin', f'1.{min_width}-{lift}',
            (OutputField('turns', 'number'), OutputField('airborne', 'boolean'), OutputField('jumps', 'integer'),
             OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=1000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._side = 0              # sign of the last confident shoulder width
            self._half_turns = 0
            self._ankles = []           # (ms, lowest ankle y) for the ground line
            self._air = None            # (start_ms, half turns at start)
            self._jumps = 0

    def process(self, frame):
        from ..private_payload import poses_of
        with self._lock:
            poses = poses_of(frame.data)
            if not poses:
                return None
            pose = max((pose for pose in poses if isinstance(pose, dict)), default=None,
                       key=lambda pose: ((pose.get('bbox') or {}).get('width', 0) * (pose.get('bbox') or {}).get('height', 0)))
            points = {} if pose is None else _points(pose, self.mapping, self.min_visibility)
            need = ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')
            if not all(name in points for name in need):
                return None
            at, events = frame.timestamp_ms, []
            shoulders_y = (points['left_shoulder'][1] + points['right_shoulder'][1]) / 2
            hips_y = (points['left_hip'][1] + points['right_hip'][1]) / 2
            # in pixels: x is normalised by the image width and y by its height (a landscape frame would otherwise
            # shrink the width against the torso by height / width)
            aspect = (float(pose.get('image_width') or 1) / float(pose.get('image_height') or 1)) \
                if pose.get('image_width') and pose.get('image_height') else self.aspect
            torso = max(1e-6, abs(hips_y - shoulders_y))
            width = aspect * ((points['right_shoulder'][0] - points['left_shoulder'][0]) +
                              (points['right_hip'][0] - points['left_hip'][0])) / (2 * torso)
            side = 1 if width > self.min_width else -1 if width < -self.min_width else 0
            if side and self._side and side != self._side:
                self._half_turns += 1
                events.append(dict(event='spin.half_turn', index=self._half_turns, timestamp_ms=at))
            if side:
                self._side = side
            ankles = [points[name][1] for name in ('left_ankle', 'right_ankle') if name in points]
            if ankles:
                lowest = max(ankles)        # image y grows downward: the lower ankle has the larger y
                self._ankles = [(ms, y) for ms, y in self._ankles if at - ms <= self.ground_ms] + [(at, lowest)]
                ground = max(y for _, y in self._ankles)
                up = ground - lowest > self.lift * torso
                if up and self._air is None:
                    self._air = (at, self._half_turns)
                elif not up and self._air is not None:
                    start, turns_at = self._air
                    self._air = None
                    if at - start >= self.min_air_ms:
                        self._jumps += 1
                        events.append(dict(event='spin.jump', index=self._jumps, start_ms=start, end_ms=at,
                                           timestamp_ms=at, airtime_ms=at - start,
                                           turns=(self._half_turns - turns_at) / 2))
            return self.spec.validate(dict(turns=self._half_turns / 2, airborne=self._air is not None,
                                           jumps=self._jumps, events=events))
