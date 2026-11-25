"""A small declarative rule language for a movement over named channels and joint relations.

A rule is JSON data supplied by the owner or proposed by an agent; nothing is evaluated as code and GAMST
runs no model to write it. It is validated once (bounded size, depth and names) and then drives the same
counter as examples do (:class:`gamst.providers.pose_exemplar.ExemplarEvents`).

::

    {"channels": "vision.pose_channels",          # default; or "audio.voice_channels"
     "steps": [                                   # 1..8, in order
       {"when": CONDITION, "hold_ms": 200},       # true for this long (default: the module's min_hold_ms
       {"when": CONDITION, "within_ms": 1500}],   #   for a single step, 0 in an ordered rule)
     "margin": 0.05}                              # how far a held rule may be violated before it ends (default 0)

    CONDITION   {"all": [CONDITION, ...]} | {"any": [CONDITION, ...]} |
                {"measure": MEASURE, "op": "gt" | "gte" | "lt" | "lte", "value": number}
    MEASURE     {"channel": NAME}                 a recorded channel, in its own unit
                {"angle": [A, VERTEX, B]}         degrees at VERTEX, 0..180
                {"height": [A, B]}                how far A is above B, in shoulder widths
                {"offset": [A, B]}                how far A is to the image right of B, in shoulder widths
                {"distance": [A, B]}              in shoulder widths

One step is a *hold*: it matches after ``hold_ms`` and ends when its violation exceeds ``margin``. Several
steps are an *ordered movement*: each later step must be reached ``within_ms`` of the step before it, else
the rule starts over. The violation of a comparison is how far the measure is on the wrong side (angles in
units of 180 degrees); ``all`` takes the largest, ``any`` the smallest; a measure whose joints are not
visible is unknown, which is never satisfied.

Measures over joints are computed in memory from the pose module's joints input; only the rule's single
violation number and its events become records.
"""
from __future__ import annotations

import json
import math

MAX_STEPS, MAX_NODES, MAX_DEPTH, MAX_BYTES = 8, 64, 6, 16384
OPS = ('gt', 'gte', 'lt', 'lte')
RELATIONS = dict(angle=3, height=2, offset=2, distance=2)
STRICT = 1e-9      # a strict comparison is violated at equality
POSE = 'vision.pose_channels'
VOICE = 'audio.voice_channels'


def channel_names(channels):
    from .pose_exemplar import CHANNELS as pose
    from .voice_style import CHANNELS as voice
    names = {POSE: pose, VOICE: voice}.get(channels)
    if names is None:
        raise ValueError(f'rule channels must be {POSE} or {VOICE}')
    return names


def joint_names():
    from .pose_exemplar import MEDIAPIPE_BODY
    return tuple(MEDIAPIPE_BODY)


def _keys(value, allowed, required, where):
    if type(value) is not dict:
        raise ValueError(f'{where}: must be an object')
    unknown = set(value) - set(allowed)
    if unknown or not set(required) <= set(value):
        raise ValueError(f'{where}: needs {sorted(required)}, allows {sorted(allowed)}')


def _finite(value, where, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{where}: a finite number in [{low:g}, {high:g}]')
    return value


def _milliseconds(value, where):
    if type(value) is not int or not 0 <= value <= 60_000:
        raise ValueError(f'{where}: integer milliseconds 0..60000')
    return value


def _measure(value, channels, where):
    if type(value) is not dict or len(value) != 1:
        raise ValueError(f'{where}: exactly one of channel, angle, height, offset or distance')
    [(kind, argument)] = value.items()
    if kind == 'channel':
        if argument not in channel_names(channels):
            raise ValueError(f'{where}: unknown channel; one of {", ".join(channel_names(channels))}')
        return dict(channel=argument)
    if kind not in RELATIONS:
        raise ValueError(f'{where}: unknown measure')
    if channels != POSE:
        raise ValueError(f'{where}: joint relations need {POSE}')
    names = joint_names()
    if type(argument) is not list or len(argument) != RELATIONS[kind] or any(
            type(name) is not str or name not in names for name in argument):
        raise ValueError(f'{where}: {kind} takes {RELATIONS[kind]} of {", ".join(names)}')
    if len(set(argument)) != len(argument):
        raise ValueError(f'{where}: {kind} takes different joints')
    return {kind: list(argument)}


def validate_rule(rule):
    """The checked copy of a rule; ValueError with the place of the first problem."""
    try:
        if len(json.dumps(rule, allow_nan=False)) > MAX_BYTES:
            raise ValueError('rule: too large')
    except (TypeError, RecursionError) as error:
        raise ValueError('rule: must be JSON') from error
    _keys(rule, ('channels', 'steps', 'margin'), ('steps',), 'rule')
    channels = rule.get('channels', POSE)
    channel_names(channels)
    nodes = [0]

    def condition(value, where, depth):
        nodes[0] += 1
        if nodes[0] > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError(f'rule: at most {MAX_NODES} conditions, {MAX_DEPTH} deep')
        if type(value) is not dict:
            raise ValueError(f'{where}: must be an object')
        for group in ('all', 'any'):
            if group in value:
                _keys(value, (group,), (group,), where)
                if type(value[group]) is not list or not 1 <= len(value[group]) <= 16:
                    raise ValueError(f'{where}.{group}: 1..16 conditions')
                return {group: [condition(item, f'{where}.{group}[{i}]', depth + 1)
                                for i, item in enumerate(value[group])]}
        _keys(value, ('measure', 'op', 'value'), ('measure', 'op', 'value'), where)
        if value['op'] not in OPS:
            raise ValueError(f'{where}.op: one of {OPS}')
        measure = _measure(value['measure'], channels, f'{where}.measure')
        low, high = (0, 180) if 'angle' in measure else (-100, 100)
        return dict(measure=measure, op=value['op'], value=_finite(value['value'], f'{where}.value', low, high))

    steps = rule['steps']
    if type(steps) is not list or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError(f'rule.steps: 1..{MAX_STEPS} steps')
    checked = []
    for index, step in enumerate(steps):
        where = f'rule.steps[{index}]'
        _keys(step, ('when', 'hold_ms', 'within_ms'), ('when',) if index == 0 else ('when', 'within_ms'), where)
        if index == 0 and 'within_ms' in step:
            raise ValueError(f'{where}: the first step has no within_ms')
        item = dict(when=condition(step['when'], f'{where}.when', 1))
        for key in ('hold_ms', 'within_ms'):
            if key in step:
                item[key] = _milliseconds(step[key], f'{where}.{key}')
        if item.get('within_ms') == 0:
            raise ValueError(f'{where}.within_ms: must be positive')
        checked.append(item)
    return dict(channels=channels, steps=checked, margin=_finite(rule.get('margin', 0), 'rule.margin', 0, 100))


class RuleScorer:
    """Violation of each step of a validated rule for one frame."""

    def __init__(self, rule):
        self.rule = validate_rule(rule)
        self.steps, self.margin = self.rule['steps'], float(self.rule['margin'])
        self._index = {name: i for i, name in enumerate(channel_names(self.rule['channels']))}
        self.needs_joints = any(self._relations(step['when']) for step in self.steps)

    def _relations(self, condition):
        for group in ('all', 'any'):
            if group in condition:
                return any(self._relations(item) for item in condition[group])
        return 'channel' not in condition['measure']

    def violation(self, step, channels, mask, points=None):
        """0 when satisfied, None when unknown. ``points``: {joint: (x, y)} with square units, y down."""
        if len(channels) != len(self._index):
            raise ValueError(f"this rule reads {self.rule['channels']} ({len(self._index)} channels); "
                             f'the stream has {len(channels)}')
        return self._violation(self.steps[step]['when'], channels, mask, points or {})

    def _violation(self, condition, channels, mask, points):
        for group, pick in (('all', max), ('any', min)):
            if group in condition:
                values = [self._violation(item, channels, mask, points) for item in condition[group]]
                if group == 'all' and None in values:
                    return None
                return pick((value for value in values if value is not None), default=None)
        measure, op, target = condition['measure'], condition['op'], condition['value']
        observed = self._observe(measure, channels, mask, points)
        if observed is None:
            return None
        wrong = target - observed if op in ('gt', 'gte') else observed - target
        if wrong < 0 or (wrong == 0 and op in ('gte', 'lte')):
            return 0.
        return max(wrong, STRICT) / (180. if 'angle' in measure else 1.)

    def _observe(self, measure, channels, mask, points):
        [(kind, argument)] = measure.items()
        if kind == 'channel':
            index = self._index[argument]
            return float(channels[index]) if mask[index] else None
        if any(name not in points for name in argument):
            return None
        if kind == 'angle':
            from .pose_exemplar import joint_angle
            a, vertex, b = (points[name] for name in argument)
            value = joint_angle(a, vertex, b)
            return None if value is None else 180. * value
        shoulders = ('left_shoulder', 'right_shoulder')
        if any(name not in points for name in shoulders):
            return None
        width = math.dist(points[shoulders[0]], points[shoulders[1]])
        if width <= 0:
            return None
        a, b = (points[name] for name in argument)
        if kind == 'height':
            return (b[1] - a[1]) / width
        if kind == 'offset':
            return (a[0] - b[0]) / width
        return math.dist(a, b) / width
