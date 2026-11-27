"""Form against a reference: for every counted movement, how its joint angles compare with the demonstrated one.

The movement counter (:class:`gamst.providers.pose_exemplar.ExemplarEvents`, ``exemplar``) already matches each
occurrence to the owner's example clips and reports a DTW ``distance``. This module adds what a coach reads: per joint
angle (``gamst.providers.pose_exemplar.CHANNELS``) the range the person reached during the occurrence (``min`` /
``max``) next to the range in the example clips, and the difference. It reads the pose channels (``channels``, a
``pose-channels`` module) and the counter's events, and binds the same exemplar reference (``reference``), whose
example vectors give the reference ranges. With a rule instead of examples there is no reference range: only the
reached ranges are reported.

``sports.form`` events: ``form.compared`` with ``index`` (the counter's), ``start_ms``, ``end_ms``, ``distance``,
``score`` (1 at distance 0, 0 at the counter's exit threshold or above) and ``angles`` {channel: {min, max, ref_min,
ref_max, range_diff}}; ``largest`` names the channel whose range differs most. Angles are in degrees (the
channels hold them in units of pi radians, ``joint_angle``).
"""
from __future__ import annotations

import threading
from collections import deque

from ..capabilities import CapabilitySpec, OutputField, identifier
from .pose_exemplar import CHANNELS

ANGLE_CHANNELS = tuple(name for name in CHANNELS if name.startswith('angle_'))


def _vectors(value):
    """Every saved channel vector of an exemplar value (examples of any nesting), as lists of numbers."""
    found = []

    def walk(item):
        if isinstance(item, dict):
            if isinstance(item.get('channels'), list) and all(type(v) in (int, float) for v in item['channels']):
                mask = item.get('mask') or [True] * len(item['channels'])
                found.append([v if seen else None for v, seen in zip(item['channels'], mask)])
                return
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return found


class FormCompare:
    def __init__(self, *, channels, exemplar, reference='exemplar', exit=None, history_ms=20_000,
                 capability_id='sports.form'):
        self.channels, self.exemplar = identifier(channels, 'module name'), identifier(exemplar, 'module name')
        self.reference_name, self.exit, self.history_ms = reference, exit, int(history_ms)
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'events', 'gamst.sports_form', '1',
            (OutputField('compared', 'integer'), OutputField('mean_score', 'number', required=False),
             OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=5000, interpretation='measurement', devices=('cpu',),
            license='Apache-2.0')
        self.reference_ranges = {}
        self.reset()

    def reset(self):
        with self._lock:
            self._history = deque()     # (ms, [values]) of the pose channels
            self._scores = []
            self._compared = 0

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        self.reference_ranges = {} if reference is None else self.ranges(_vectors(reference.get('value')))

    @staticmethod
    def ranges(vectors):
        ranges = {}
        for index, name in enumerate(CHANNELS):
            if name not in ANGLE_CHANNELS:
                continue
            values = [vector[index] for vector in vectors if index < len(vector) and vector[index] is not None]
            if values:      # channels hold angles in units of pi radians
                ranges[name] = (180. * min(values), 180. * max(values))
        return ranges

    def _compare(self, event):
        start, end = event.get('start_ms'), event.get('end_ms', event.get('timestamp_ms'))
        if type(start) not in (int, float) or type(end) not in (int, float):
            return None
        vectors = [values for ms, values in self._history if start <= ms <= end]
        reached = self.ranges(vectors)
        angles, largest = {}, None
        for name, (low, high) in reached.items():
            item = dict(min=round(low, 1), max=round(high, 1))
            if name in self.reference_ranges:
                ref_low, ref_high = self.reference_ranges[name]
                diff = (high - low) - (ref_high - ref_low)
                item.update(ref_min=round(ref_low, 1), ref_max=round(ref_high, 1), range_diff=round(diff, 1))
                if largest is None or abs(diff) > abs(angles[largest]['range_diff']):
                    largest = name
            angles[name] = item
        result = dict(event='form.compared', index=event.get('index'), start_ms=int(start), end_ms=int(end),
                      timestamp_ms=int(end), angles=angles, frames=len(vectors),
                      reference='examples' if self.reference_ranges else 'none')
        distance = event.get('distance')
        if type(distance) in (int, float):
            result['distance'] = distance
            if self.exit:
                result['score'] = round(max(0., 1. - distance / float(self.exit)), 3)
                self._scores.append(result['score'])
        if largest is not None:
            result['largest'] = largest
        return result

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            value = frame.data if isinstance(frame.data, dict) else {}
            if source == self.channels:
                channels = value.get('channels')
                mask = value.get('mask') or [True] * len(channels or ())
                if isinstance(channels, list):
                    self._history.append((frame.timestamp_ms, [v if seen else None for v, seen in zip(channels, mask)]))
                    while self._history and frame.timestamp_ms - self._history[0][0] > self.history_ms:
                        self._history.popleft()
                return None
            if source != self.exemplar:
                return None
            events = [found for found in (self._compare(event) for event in value.get('events') or ()
                                          if isinstance(event, dict) and event.get('event') == 'exemplar.matched')
                      if found is not None]
            if not events:
                return None
            self._compared += len(events)
            result = dict(compared=self._compared, events=events)
            if self._scores:
                result['mean_score'] = round(sum(self._scores) / len(self._scores), 3)
            return self.spec.validate(result)
