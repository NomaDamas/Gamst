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
                {"hand_open": "left" | "right"}   how open that hand is, in palm lengths (a fist is low)
                {"rate": MEASURE}                 how fast a measure changes, its unit per second (over 50 ms)

Joints are the 12 body joints (``left_shoulder``, ``right_wrist``, ``left_knee``, ...) and the 21 points of each hand:
``left_hand_wrist``, ``left_thumb_cmc|mcp|ip|tip``, ``left_index|middle|ring|pinky_mcp|pip|dip|tip``, the same for
``right_`` (:data:`gamst.providers.pose_exemplar.HAND_JOINTS`). Each pose model is read by its own keypoint set and
score gate (:func:`gamst.providers.pose_exemplar.keypoint_reading`): RTMW and Sapiens2 have hands, MediaPipe Pose has
none, so a hand joint or ``hand_open`` is unknown there.

``hand_open`` is the mean distance from the hand's wrist to its fingertips (index, middle, ring, pinky; the thumb
folds across the palm in a fist and stays out) divided by the palm length (hand wrist to middle-finger MCP); it needs
the wrist, the middle MCP and at least two of the four tips visible, and averages the visible tips. It does not depend
on how large the hand appears, so it needs no shoulders. Measured with RTMW-l on the owner's 맨까송 clip (hands judged by
eye, docs/measurements/testbed-2026-11-26.md "Hand joints"): fists 0.65-0.92 in 7 of 9, open hands 1.0-2.1 (median
1.62); with the 0.3 gate below 1.0 held 7 of 9 fists and 0 of 28 open hands, but also 12 of 43 crops where no hand was
in view (RTMW scores are clipped to 1 on the wire, so that gate passed such hands). RTMW hands are now gated on the raw
score (3.0, :data:`gamst.providers.rtmw_pose.HAND_MIN_RAW_SCORE`): 9/9 fists and 27/28 open hands stay known, 25 of the
43 crops without a hand become unknown, and below 1.0 holds 3 of them. Still require the arm's own movement as well.

One step is a *hold*: it matches after ``hold_ms`` and ends when its violation exceeds ``margin``. Several
steps are an *ordered movement*: each later step must be reached ``within_ms`` of the step before it, else
the rule starts over. The violation of a comparison is how far the measure is on the wrong side (angles in
units of 180 degrees); ``all`` takes the largest, ``any`` the smallest; a measure whose joints are not
visible is unknown, which is never satisfied.

Measures over joints are computed in memory from the pose module's joints input; only the rule's single
violation number and its events become records.
"""
from __future__ import annotations

from collections import deque
import json
import math

MAX_STEPS, MAX_NODES, MAX_DEPTH, MAX_BYTES = 8, 64, 6, 16384
OPS = ('gt', 'gte', 'lt', 'lte')
RELATIONS = dict(angle=3, height=2, offset=2, distance=2)
HAND_MEASURES = ('hand_open',)
TIPS = ('index_tip', 'middle_tip', 'ring_tip', 'pinky_tip')
MIN_TIPS = 2       # hand_open: fingertips that must be visible (a fist hides some)
RATE_MS = 50       # a rate is the change over about this long: two frames at 30 fps, three at 60
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
    from .pose_exemplar import HAND_JOINTS, MEDIAPIPE_BODY
    return tuple(MEDIAPIPE_BODY) + HAND_JOINTS


def hand_open(points, side):
    """Mean wrist-to-fingertip distance over the palm length of one hand (``points``: {joint: (x, y)}), or None."""
    wrist, middle = points.get(f'{side}_hand_wrist'), points.get(f'{side}_middle_mcp')
    tips = [points[name] for name in (f'{side}_{tip}' for tip in TIPS) if name in points]
    if wrist is None or middle is None or len(tips) < MIN_TIPS:
        return None
    palm = math.dist(wrist, middle)
    if palm <= 0:
        return None
    return sum(math.dist(wrist, tip) for tip in tips) / len(tips) / palm


def _hands(measure):
    """Whether a checked measure reads hand points."""
    [(kind, argument)] = measure.items()
    if kind == 'rate':
        return _hands(argument)
    if kind in HAND_MEASURES:
        return True
    from .pose_exemplar import HAND_JOINTS
    return kind in RELATIONS and any(name in HAND_JOINTS for name in argument)


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
        raise ValueError(f'{where}: exactly one of channel, angle, height, offset, distance, hand_open or rate')
    [(kind, argument)] = value.items()
    if kind == 'rate':
        if type(argument) is dict and 'rate' in argument:
            raise ValueError(f'{where}: a rate of a rate')
        return dict(rate=_measure(argument, channels, f'{where}.rate'))
    if kind == 'channel':
        if argument not in channel_names(channels):
            raise ValueError(f'{where}: unknown channel; one of {", ".join(channel_names(channels))}')
        return dict(channel=argument)
    if kind in HAND_MEASURES:
        if channels != POSE:
            raise ValueError(f'{where}: {kind} needs {POSE}')
        if argument not in ('left', 'right'):
            raise ValueError(f'{where}: {kind} is "left" or "right"')
        return {kind: argument}
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
        low, high = (-10000, 10000) if 'rate' in measure else (0, 180) if 'angle' in measure else (-100, 100)
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
        self.needs_hands = any(self._relations(step['when'], _hands) for step in self.steps)
        self._now = None
        self._history = {}      # key of a measure -> deque of (ms, value) for its rate
        for step in self.steps:
            self._rates(step['when'])

    def _rates(self, condition):
        for group in ('all', 'any'):
            for item in condition.get(group, ()):
                self._rates(item)
        if 'rate' in condition.get('measure', ()):
            inner = condition['measure']['rate']
            self._history[json.dumps(inner, sort_keys=True)] = (inner, deque(maxlen=64))

    def observe(self, timestamp_ms, channels, mask, points=None):
        """One frame, before its violations are asked: what the rates need to remember."""
        for inner, history in self._history.values():
            value = self._observe(inner, channels, mask, points or {})
            if value is not None:   # unseen frames leave a hole; _rate looks back 3 * RATE_MS at most
                history.append((timestamp_ms, value))
            self._now = timestamp_ms

    def reset(self):
        for _, history in self._history.values():
            history.clear()

    def _rate(self, inner):
        """Change per second over the last RATE_MS (one frame longer at most), or None."""
        history = self._history[json.dumps(inner, sort_keys=True)][1]
        if len(history) < 2 or self._now != history[-1][0]:
            return None     # not seen now: no speed now
        now, value = history[-1]
        earlier = next((item for item in reversed(list(history)[:-1]) if now - item[0] >= RATE_MS - 5), None)
        if earlier is None or now - earlier[0] > 3 * RATE_MS:
            return None
        return 1000 * (value - earlier[1]) / (now - earlier[0])

    def _relations(self, condition, test=None):
        """Whether a condition reads joints (``test``: whether one of its measures passes it)."""
        for group in ('all', 'any'):
            if group in condition:
                return any(self._relations(item, test) for item in condition[group])
        measure = condition['measure']
        return test(measure) if test is not None else 'channel' not in measure.get('rate', measure)

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
        return max(wrong, STRICT) / (180. if 'angle' in measure.get('rate', measure) else 1.)

    def _observe(self, measure, channels, mask, points):
        [(kind, argument)] = measure.items()
        if kind == 'rate':
            return self._rate(argument)
        if kind == 'channel':
            index = self._index[argument]
            return float(channels[index]) if mask[index] else None
        if kind == 'hand_open':
            return hand_open(points, argument)
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


# ---------------------------------------------------------------- calibration from example clips

# How far beyond the examples' least satisfying value a calibrated number is placed, per measure, in its own unit.
SLACK = dict(angle=10., height=.15, offset=.15, distance=.15, hand_open=.1, channel=.05)
RATE_SLACK = .2        # a rate: this fraction of the examples' least value
MIN_WITHIN_MS, WITHIN_FACTOR = 300, 2.
CONTAINS = .5          # a clip shows a step when, as written, the step is violated by at most this much at its place


def _leaves(condition, path=()):
    """[(path, leaf, any-group paths above it)] of a checked condition."""
    for group in ('all', 'any'):
        if group in condition:
            return [(found, leaf, ([path] if group == 'any' else []) + above)
                    for i, item in enumerate(condition[group])
                    for found, leaf, above in _leaves(item, (*path, group, i))]
    return [(path, condition, [])]


def _at(condition, path):
    for key in path:
        condition = condition[key]
    return condition


def _rate_kind(measure):
    [(kind, argument)] = measure.items()
    return (True, next(iter(argument))) if kind == 'rate' else (False, kind)


def _walk(scorer, condition, path, channels, mask, points, out):
    """Violation of ``condition`` (None: unknown); every node's (violation, observed) into ``out[path]``."""
    for group, pick in (('all', max), ('any', min)):
        if group in condition:
            values = [_walk(scorer, item, (*path, group, i), channels, mask, points, out)
                      for i, item in enumerate(condition[group])]
            value = None if group == 'all' and None in values else pick(
                (v for v in values if v is not None), default=None)
            out[path] = (value, None)
            return value
    value = scorer._violation(condition, channels, mask, points)
    out[path] = (value, scorer._observe(condition['measure'], channels, mask, points))
    return value


def _series(scorer, clip):
    """Per frame: (ts, [violation of each step], [{path: (violation, observed)} of each step]) of one clip."""
    scorer.reset()
    rows = []
    for ts, channels, mask, points in clip:
        points = points or {}
        scorer.observe(ts, channels, mask, points)
        violations, nodes = [], []
        for step in scorer.steps:
            out = {}
            violations.append(_walk(scorer, step['when'], (), channels, mask, points, out))
            nodes.append(out)
        rows.append((ts, violations, nodes))
    return rows


def _aligned(rows, steps):
    """(frame index of each step, in order, with the least summed violation (unknown: 1), that sum); None if empty."""
    if not rows:
        return None
    cost = [[1. if v is None else min(1., v) for v in violations] for _, violations, _ in rows]
    best, choice = [cost[f][0] for f in range(len(rows))], [[f] for f in range(len(rows))]
    for k in range(1, steps):
        run, at, total, picks = float('inf'), None, [], []
        for f in range(len(rows)):
            if best[f] < run:
                run, at = best[f], f
            total.append(run + cost[f][k])
            picks.append(choice[at] + [f])
        best, choice = total, picks
    end = min(range(len(rows)), key=lambda f: best[f])
    return choice[end], best[end]


def _held(nodes, path, above):
    """Whether the comparison at ``path`` is the one that holds in each ``any`` group above it (or as close as any)."""
    for group in above:
        mine = nodes.get(path[:len(group) + 2], (None, None))[0]
        siblings = [value for key, (value, _) in nodes.items() if len(key) == len(group) + 2 and key[:len(group)] == group
                    and key[len(group)] == 'any' and value is not None]
        if mine is None or mine > min(siblings, default=float('inf')):
            return False
    return True


def _placed(scorer, clips, steps):
    placed = []
    for clip in clips:
        rows = _series(scorer, clip)
        found = _aligned(rows, steps)
        if found is not None:
            placed.append(dict(rows=rows, frames=found[0], cost=round(found[1], 4)))
    return placed


def calibrate(rule, positives, negatives=()):
    """The rule with its numbers set from example clips: ``(rule, evidence)``.

    ``positives`` / ``negatives``: clips as ``[(timestamp_ms, channels, mask, points), ...]`` (points: {joint: (x, y)} as
    the counter reads them). The rule's skeleton (the text's: which joints, which comparisons, in which order) is kept.
    In each clip every step is placed on one frame, in order, where it is closest to satisfied; a clip shows a step when
    the written step is violated there by at most :data:`CONTAINS` (a clip trimmed to start at the second step does not
    show the first), and only such clips teach that step. As with example thresholds
    (:func:`gamst.providers.movement_match.derive`), positives only widen: a comparison a positive does not pass moves to
    the least satisfying value the positives show, plus :data:`SLACK` (a rate: :data:`RATE_SLACK` of it); one they all
    pass keeps the text's number. A comparison inside ``any`` learns only from the clips where it was the one that held.
    A step's ``within_ms`` grows to twice the longest gap the positives show between it and the step before (at least
    :data:`MIN_WITHIN_MS`). Only a negative narrows: a negative whose placed values would satisfy every step moves the
    comparison that separates it most from the positives to halfway between. The evidence lists, per comparison, the
    text's number, the positives' values and the number chosen.
    """
    checked = validate_rule(rule)
    scorer = RuleScorer(checked)
    steps = len(checked['steps'])
    placed = _placed(scorer, positives, steps)
    calibrated = json.loads(json.dumps(checked))
    evidence = dict(rule='gamst.providers.movement_rule.calibrate', positives=len(positives), negatives=len(negatives),
                    placed_cost=[item['cost'] for item in placed], leaves=[], within_ms=[], negatives_moved=[])
    learned = {}
    shows = [[(item['rows'][item['frames'][k]][1][k] or 0) <= CONTAINS and item['rows'][item['frames'][k]][1][k] is not None
              for k in range(steps)] for item in placed]
    evidence['shows_step'] = shows
    for k, step in enumerate(checked['steps']):
        for path, leaf, above in _leaves(step['when']):
            values = [item['rows'][item['frames'][k]][2][k][path][1] for item, seen in zip(placed, shows)
                      if seen[k] and _held(item['rows'][item['frames'][k]][2][k], path, above)]
            values = [value for value in values if value is not None]
            target = _at(calibrated['steps'][k]['when'], path)
            if values:
                rate, kind = _rate_kind(leaf['measure'])
                worst = min(values) if leaf['op'] in ('gt', 'gte') else max(values)
                slack = abs(worst) * RATE_SLACK if rate else SLACK.get(kind, .1)
                value = min(leaf['value'], worst - slack) if leaf['op'] in ('gt', 'gte') else \
                    max(leaf['value'], worst + slack)
                low, high = (-10000, 10000) if rate else (0, 180) if kind == 'angle' else (-100, 100)
                target['value'] = round(min(high, max(low, value)), 4)
                learned[k, path] = worst
            evidence['leaves'].append(dict(step=k, path='.'.join(map(str, path)) or 'when', measure=leaf['measure'],
                                           op=leaf['op'], text=leaf['value'], value=target['value'],
                                           positives=[round(value, 4) for value in values]))
        both = [item for item, seen in zip(placed, shows) if seen[k] and seen[k - 1]]
        if k and both:
            gaps = [item['rows'][item['frames'][k]][0] - item['rows'][item['frames'][k - 1]][0] for item in both]
            within = int(min(60_000, max(step['within_ms'], MIN_WITHIN_MS, WITHIN_FACTOR * max(gaps))))
            evidence['within_ms'].append(dict(step=k, text=step.get('within_ms'), gaps_ms=gaps, value=within))
            calibrated['steps'][k]['within_ms'] = within
    calibrated = validate_rule(calibrated)
    for index, item in enumerate(_placed(RuleScorer(calibrated), negatives, steps)):
        frames, rows = item['frames'], item['rows']
        if any(rows[frames[k]][1][k] is None or rows[frames[k]][1][k] > 0 for k in range(steps)):
            continue        # this negative is not counted by the calibrated rule as placed
        best = None         # the comparison with the widest gap between the negative and the positives' worst
        for (k, path), worst in learned.items():
            value = rows[frames[k]][2][k].get(path, (None, None))[1]
            if value is None:
                continue
            leaf = _at(calibrated['steps'][k]['when'], path)
            gap = (worst - value) if leaf['op'] in ('gt', 'gte') else (value - worst)
            if gap > 0 and (best is None or gap > best[0]):
                best = (gap, k, path, value, worst)
        if best is None:
            evidence['negatives_moved'].append(dict(negative=index, moved=None, note='not separable by one comparison'))
            continue
        _, k, path, value, worst = best
        leaf = _at(calibrated['steps'][k]['when'], path)
        leaf['value'] = round((value + worst) / 2, 4)
        evidence['negatives_moved'].append(dict(negative=index, step=k, path='.'.join(map(str, path)), value=leaf['value']))
    return validate_rule(calibrated), evidence
