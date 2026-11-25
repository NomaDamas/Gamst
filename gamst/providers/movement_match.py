"""Few-shot matching math for any channel stream (pose, voice style, IMU): no weights, no training.

A movement is given by *positive* examples and, optionally, *negative* ones (look-alikes that must not
count). All examples of one movement are either a ``hold`` (one channel vector) or a ``sequence`` (a list
of them with the demonstration's duration).

Matching
--------
The distance of "now" to the movement is the smallest distance over the positives. A negative at least as
close as every positive rejects the match. A sequence is compared with :class:`SequenceTrack`, a streaming
subsequence dynamic time warp (open beginning, end = now). It updates one column per frame, so the cost per
frame is (number of examples x example length), whatever the length of the recording, and it also yields
where the copy began. The matched stretch must last between half and twice the example's own duration and
cover at least half of the example's range of motion (mean over channels of largest - smallest value);
without that, standing still in a pose of the movement would warp onto it at a small distance.

Thresholds (:func:`derive`)
---------------------------
The *critical distance* of a clip is the smallest ``enter`` at which the counter would count it. Let
``d_i`` be positive i's critical distance when only the other positives are templates (leave one out), and
``n`` the smallest critical distance of any negative against all positives. With the recipe's default
``enter0`` / ``exit0``::

    p     = max of the d_i that are below n          (0 with a single positive)
    base  = max(enter0, p), at most exit0 when no negative limits it
    enter = min(base, (p + n) / 2)
    exit  = min(enter * exit0 / enter0, (enter + n) / 2)

So the examples of the same movement can only widen the default, and only a negative can place an upper
limit; no target count is ever used. ``threshold_source`` says what the numbers rest on:

- ``default``: one positive, nothing to derive from;
- ``default_limited_by_negatives``: one positive and negatives;
- ``leave_one_out``: at least two positives, no negative (how different other movements are is unknown);
- ``leave_one_out_and_negatives``.

``uncovered`` lists positives whose leave-one-out distance is above ``enter``: without their own template
they would not have been counted. With one positive no such check exists, and the saved file says so.
"""
from __future__ import annotations

from collections import deque
import math

MAX_EXAMPLES = 16        # positives, and the same number of negatives
MAX_ITEMS = 60           # vectors of one sequence example
MAX_CHANNELS = 64
RESOLUTION = 1e-5        # channel values are rounded to this
MODES = ('hold', 'sequence')
INFINITY = float('inf')


def distance(a, mask_a, b, mask_b):
    """Mean absolute channel difference over channels visible in both; None when fewer than 4 are."""
    pairs = [abs(x - y) for x, y, m, n in zip(a, b, mask_a, mask_b) if m and n]
    return sum(pairs) / len(pairs) if len(pairs) >= 4 else None


def example(value, *, span_ms=None):
    """``{"items": [(channels, mask), ...], "span_ms"}`` from one saved vector or a list of them."""
    items = []
    values = value if isinstance(value, list) else [value]
    if not 1 <= len(values) <= MAX_ITEMS:
        raise ValueError(f'an example holds 1..{MAX_ITEMS} channel vectors')
    for item in values:
        channels = item.get('channels') if isinstance(item, dict) else None
        if not isinstance(channels, list) or not 1 <= len(channels) <= MAX_CHANNELS:
            raise ValueError('exemplar items are {"channels": [...], "mask": [...]} channel values')
        mask = item.get('mask', [True] * len(channels))
        if not isinstance(mask, list) or len(mask) != len(channels):
            raise ValueError('an exemplar mask has one entry per channel')
        numbers = [float(v) for v in channels]
        if any(not math.isfinite(v) or abs(v) > 1e6 for v in numbers):
            raise ValueError('exemplar channels must be finite and bounded')
        items.append((numbers, [bool(v) for v in mask]))
    if len({len(channels) for channels, _ in items}) != 1:
        raise ValueError('every vector of an example has the same channels')
    if span_ms is not None and (type(span_ms) not in (int, float) or not 0 < span_ms <= 600_000):
        raise ValueError('span_ms must be a positive duration of at most 10 minutes')
    return dict(items=items, span_ms=span_ms)


def motion(items):
    """Mean over channels of largest - smallest visible value: {channel: range} and its mean (0 if unseen)."""
    ranges = {}
    for channel in range(len(items[0][0])):
        values = [channels[channel] for channels, mask in items if mask[channel]]
        if len(values) >= 2:
            ranges[channel] = max(values) - min(values)
    return ranges, sum(ranges.values()) / len(ranges) if ranges else 0.


class SequenceTrack:
    """Streaming subsequence DTW of one template: the cost of the template ending at the newest item.

    With ``span_ms`` the copy must last span/2 .. 2 x span and move (:meth:`moved`). Without it (exemplar files from before the
    duration was saved) the search window is the last ``2 x len(template)`` items.
    """

    def __init__(self, template, span_ms=None, window=None):
        if not template:
            raise ValueError('empty exemplar')
        self.template, self.span_ms = template, span_ms
        self.window = window or max(4, 2 * len(template))
        self.ranges, self.motion = motion(template)
        self.reset()

    def reset(self):
        n = len(self.template)
        self._cost, self._steps, self._start = [INFINITY] * n, [0] * n, [0] * n
        self._times = deque(maxlen=self.window)

    def update(self, ts, item):
        """(mean step distance or None, start time of that copy) with ``item`` as the newest vector."""
        self._times.append(ts)
        earliest = self._times[0] if self.span_ms is None else ts - 2 * self.span_ms
        channels, mask = item
        old_cost, old_steps, old_start = self._cost, self._steps, self._start
        cost, steps, start = [], [], []
        for i, (reference, reference_mask) in enumerate(self.template):
            local = distance(reference, reference_mask, channels, mask)
            local = 1. if local is None else local
            if i == 0:
                best, length, begin = 0., 0, ts           # open beginning: a copy may start at any item
            else:
                best, length, begin = INFINITY, 0, ts
                if old_cost[i - 1] < best and old_start[i - 1] >= earliest:
                    best, length, begin = old_cost[i - 1], old_steps[i - 1], old_start[i - 1]
                if cost[i - 1] < best:
                    best, length, begin = cost[i - 1], steps[i - 1], start[i - 1]
                if old_cost[i] < best and old_start[i] >= earliest:
                    best, length, begin = old_cost[i], old_steps[i], old_start[i]
            cost.append(best + local)
            steps.append(length + 1)
            start.append(begin)
        self._cost, self._steps, self._start = cost, steps, start
        if self.span_ms is None:
            if len(self._times) < max(2, len(self.template) // 2):
                return None, None
        elif 2 * (ts - start[-1]) < self.span_ms:
            return None, None                            # faster than twice the demonstrated speed
        return self.cost(), start[-1]

    def cost(self):
        """Mean step distance of the best copy ending at the newest item, whatever its duration."""
        return self._cost[-1] / max(1, self._steps[-1])

    def moved(self, items):
        """Does this stretch cover at least half of the example's range of motion?"""
        if self.span_ms is None:
            return True
        ranges, _ = motion(items)
        shared = [channel for channel in self.ranges if channel in ranges]
        return bool(shared) and 2 * sum(ranges[c] for c in shared) >= sum(self.ranges[c] for c in shared)

    def half_duration_ms(self):
        """Half the demonstration's length: two matches closer together are one event."""
        if self.span_ms is not None:
            return int(self.span_ms / 2)
        if len(self._times) < 2:
            return 0
        return int(.5 * (self._times[-1] - self._times[0]) * len(self.template) / (len(self._times) - 1))


class HoldTrack:
    def __init__(self, template, span_ms=None):
        if len(template) != 1:
            raise ValueError('a hold example is one channel vector')
        self.template = template[0]

    def reset(self):
        pass

    def update(self, ts, item):
        return distance(*self.template, *item), ts

    def moved(self, items):
        return True

    def half_duration_ms(self):
        return 0


class ExampleScorer:
    """Distance of the newest item to the movement: best over the positives, and the closest negative."""

    def __init__(self, mode, positives, negatives=()):
        if mode not in MODES:
            raise ValueError("mode must be 'hold' or 'sequence'")
        if not 1 <= len(positives) <= MAX_EXAMPLES or len(negatives) > MAX_EXAMPLES:
            raise ValueError(f'a movement has 1..{MAX_EXAMPLES} positive and at most {MAX_EXAMPLES} negative examples')
        track = HoldTrack if mode == 'hold' else SequenceTrack
        self.size = len(positives[0]['items'][0][0])
        if any(len(entry['items'][0][0]) != self.size for entry in (*positives, *negatives)):
            raise ValueError('every example of a movement has the same channels')
        self.positives = [track(entry['items'], entry.get('span_ms')) for entry in positives]
        self.negatives = [track(entry['items'], entry.get('span_ms')) for entry in negatives]
        spans = [track.span_ms for track in (*self.positives, *self.negatives) if getattr(track, 'span_ms', None)]
        self._keep_ms, self._recent = 2 * max(spans, default=0), deque()

    def reset(self):
        self._recent.clear()
        for track in (*self.positives, *self.negatives):
            track.reset()

    def update(self, ts, item, limit=INFINITY):
        """(distance or None, index of the best positive, start time, closest negative's distance or None).

        A distance above ``limit`` is returned unchecked: the caller treats it as no match either way.
        """
        if len(item[0]) != self.size or len(item[1]) != self.size:
            raise ValueError(f'this movement was enrolled from {self.size} channels; the stream has {len(item[0])}')
        if self._keep_ms:
            self._recent.append((ts, item))
            while ts - self._recent[0][0] > self._keep_ms:
                self._recent.popleft()
        best, rival = (None, None, None), None
        for index, track in enumerate(self.positives):
            found, start = self._valid(track, ts, item, limit)
            if found is not None and (best[0] is None or found < best[0]):
                best = (found, index, start)
        for track in self.negatives:
            found, _ = self._valid(track, ts, item, limit)
            if found is not None and (rival is None or found < rival):
                rival = found
        return (*best, rival)

    def _valid(self, track, ts, item, limit):
        found, start = track.update(ts, item)
        if found is not None and found <= limit and not track.moved(
                [entry for at, entry in self._recent if at >= start]):
            return None, None
        return found, start

    def half_duration_ms(self, index):
        return self.positives[index].half_duration_ms()


def critical_distance(mode, templates, clip, *, min_hold_ms=0):
    """The smallest ``enter`` at which a counter with these positives would count ``clip``.

    ``clip``: [(timestamp_ms, channels, mask), ...]. Infinity when no threshold counts it.
    """
    scorer = ExampleScorer(mode, templates)
    values = [(ts, scorer.update(ts, (channels, mask))[0]) for ts, channels, mask in clip]
    values = [(ts, value) for ts, value in values if value is not None]
    if mode == 'sequence' or min_hold_ms <= 0:
        return min((value for _, value in values), default=INFINITY)
    best = INFINITY
    for i, (begin, _) in enumerate(values):              # the best run that lasts min_hold_ms
        worst = 0.
        for ts, value in values[i:]:
            worst = max(worst, value)
            if worst >= best:
                break
            if ts - begin >= min_hold_ms:
                best = worst
                break
    return best


def same(a, b):
    """Two example templates are the same data (to the channel resolution)."""
    if len(a['items']) != len(b['items']):
        return False
    return all(mask == other_mask and all(abs(x - y) <= RESOLUTION for x, y in zip(channels, other))
               for (channels, mask), (other, other_mask) in zip(a['items'], b['items']))


def _number(value):
    return None if value == INFINITY else round(value, 5)


def derive(mode, positives, negatives, *, enter0, exit0, min_hold_ms=0):
    """Thresholds of one movement from its examples (module docstring).

    ``positives`` / ``negatives``: ``{"template": example(...), "clip": [(ts, channels, mask), ...]}``.
    """
    if not 0 < enter0 <= exit0:
        raise ValueError('require 0 < enter <= exit')
    templates = [item['template'] for item in positives]
    if mode == 'sequence':
        for i, template in enumerate(templates):
            if motion(template['items'])[1] <= RESOLUTION:
                raise ValueError(f'positive example {i} does not move; a held pose is enrolled with mode hold')
    for k, negative in enumerate(negatives):
        for i, template in enumerate(templates):
            if same(template, negative['template']):
                raise ValueError(f'negative example {k} is identical to positive example {i}')
    loo = [] if len(positives) < 2 else [
        critical_distance(mode, templates[:i] + templates[i + 1:], item['clip'], min_hold_ms=min_hold_ms)
        for i, item in enumerate(positives)]
    against = [critical_distance(mode, templates, item['clip'], min_hold_ms=min_hold_ms) for item in negatives]
    n = min(against, default=INFINITY)
    if n <= RESOLUTION:
        raise ValueError(f'negative example {against.index(n)} cannot be told from the positive examples')
    p = max((value for value in loo if value < n), default=0.)
    base = max(enter0, p)
    if n == INFINITY:
        base = min(base, exit0)
    enter = min(base, (p + n) / 2)
    exit_ = min(enter * exit0 / enter0, (enter + n) / 2)
    source = ('leave_one_out' if loo else 'default') + (
        '' if not negatives else '_and_negatives' if loo else '_limited_by_negatives')
    uncovered = [i for i, value in enumerate(loo) if value > enter]
    notes = []
    if not loo:
        notes.append('one positive example: the thresholds are not checked against another demonstration')
    if not negatives:
        notes.append('no negative example: how close other movements come is unknown')
    if uncovered:
        notes.append(f'positive examples {uncovered} are farther from every other example than enter; '
                     'each is counted through its own template only')
    # Not rounded up: enter and exit must stay below the closest negative.
    return dict(enter=math.floor(enter * 1e6) / 1e6, exit=math.floor(exit_ * 1e6) / 1e6, threshold_source=source,
                threshold_evidence=dict(
                    rule='gamst.providers.movement_match.derive', default_enter=enter0, default_exit=exit0,
                    min_hold_ms=min_hold_ms, positive_leave_one_out=[_number(value) for value in loo],
                    negative=[_number(value) for value in against], uncovered=uncovered, notes=notes))
