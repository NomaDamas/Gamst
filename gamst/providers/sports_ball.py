"""One ball, followed frame by frame: the candidate of a ball model or a detector that continues the flight.

A detector sees several round things and misses the ball in some frames; a heatmap ball model (TrackNetV3) gives at
most one candidate per frame but can jump to a shoe or a light. This module keeps one flight: a constant-velocity
prediction from the last samples gates the candidates (``max_speed``: the fastest image speed, in image widths per
second), the nearest candidate inside the gate continues it, and a flight with no candidate for ``lost_ms`` ends.
It reads one module (``ball``): a ``balls`` record of a ball model or an ``objects`` record of a detector, whose ball
label it takes (``labels``).

Its record is the ball state at a capture time (``at_ms``, the frame it describes; a ball model with temporal
ensembling answers for a frame a few frames back): ``tracking`` with a position and an image velocity, ``missing``
(no candidate yet, the flight goes on), ``lost`` (the flight ended), or ``idle``. ``ball.found`` and ``ball.lost``
events mark flights. Positions are normalised image coordinates; nothing here is in metres, since a ball in the air
is off the court plane (docs/sports.md).
"""
from __future__ import annotations

import math
import threading
from collections import deque

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.objects import ball_points


class BallTrack:
    def __init__(self, *, ball, labels=('sports ball',), min_score=.3, max_speed=3., gate=.03, lost_ms=400,
                 history=6, capability_id='sports.ball'):
        for name, value, low, high in (('min_score', min_score, 0, 1), ('max_speed', max_speed, .01, 100),
                                        ('gate', gate, 0, 1)):
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f'{name} must be {low}..{high}')
        if type(lost_ms) is not int or not 1 <= lost_ms <= 10_000:
            raise ValueError('lost_ms must be 1..10000 milliseconds')
        self.ball, self.labels = identifier(ball, 'module name'), tuple(labels)
        self.min_score, self.max_speed, self.gate, self.lost_ms = float(min_score), float(max_speed), float(gate), lost_ms
        self.history = int(history)
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'video', 'gamst.sports_ball', f'1.{max_speed}-{gate}-{lost_ms}',
            (OutputField('state', 'string'), OutputField('at_ms', 'integer'),
             OutputField('x', 'number', 'normalized_image', required=False),
             OutputField('y', 'number', 'normalized_image', required=False),
             OutputField('vx', 'number', 'image_widths/s', required=False),
             OutputField('vy', 'number', 'image_heights/s', required=False),
             OutputField('speed', 'number', 'normalized_image/s', required=False),
             OutputField('flight', 'integer'), OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=lost_ms, interpretation='estimate', devices=('cpu',),
            license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._samples = deque(maxlen=self.history)      # (at_ms, x, y) of the current flight
            self._missing_since = None
            self._flight = 0
            self._last_at = None

    def _velocity(self):
        if len(self._samples) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = self._samples[max(0, len(self._samples) - 3)], self._samples[-1]
        dt = (t1 - t0) / 1000
        return None if dt <= 0 else ((x1 - x0) / dt, (y1 - y0) / dt)

    def _choose(self, candidates, at_ms):
        candidates = [item for item in candidates if item[2] >= self.min_score]
        if not candidates:
            return None
        if not self._samples:
            return max(candidates, key=lambda item: item[2])
        t, x, y = self._samples[-1]
        dt = max(0., (at_ms - t) / 1000)
        velocity = self._velocity() or (0., 0.)
        px, py = x + velocity[0] * dt, y + velocity[1] * dt
        radius = self.gate + self.max_speed * dt
        near = [(math.hypot(cx - px, cy - py), item) for item in candidates for cx, cy in [item[:2]]]
        near = [pair for pair in near if pair[0] <= radius]
        return min(near, key=lambda pair: pair[0])[1] if near else None

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            if source != self.ball or not isinstance(frame.data, dict):
                return None
            if isinstance(frame.data.get('frames'), list):     # a ball model: one answer per frame it decided
                values = [self._step(ball_points(item, self.labels), int(item.get('frame_ms', frame.timestamp_ms)))
                          for item in frame.data['frames'] if isinstance(item, dict)]
                values = [value for value in values if value is not None]
                return values or None
            candidates = ball_points(frame.data, self.labels)
            at_ms = next((int(item[3]) for item in candidates if item[3] is not None), None)
            return self._step(candidates, int(frame.data.get('frame_ms', frame.timestamp_ms)) if at_ms is None else at_ms)

    def _step(self, candidates, at_ms):
        if self._last_at is not None and at_ms <= self._last_at:
            return None         # a ball model answers each frame once; an older answer changes nothing
        self._last_at = at_ms
        events = []
        chosen = self._choose(candidates, at_ms)
        if chosen is not None:
            if not self._samples:
                self._flight += 1
                events.append(dict(event='ball.found', flight=self._flight, timestamp_ms=at_ms))
            self._samples.append((at_ms, float(chosen[0]), float(chosen[1])))
            self._missing_since = None
            velocity = self._velocity()
            value = dict(state='tracking', at_ms=at_ms, x=round(chosen[0], 5), y=round(chosen[1], 5),
                         flight=self._flight, events=events)
            if velocity is not None:
                value.update(vx=round(velocity[0], 4), vy=round(velocity[1], 4),
                             speed=round(math.hypot(*velocity), 4))
            return self.spec.validate(value)
        if not self._samples:
            return self.spec.validate(dict(state='idle', at_ms=at_ms, flight=self._flight, events=events))
        self._missing_since = self._samples[-1][0] if self._missing_since is None else self._missing_since
        if at_ms - self._missing_since >= self.lost_ms:
            events.append(dict(event='ball.lost', flight=self._flight, timestamp_ms=self._samples[-1][0]))
            self._samples.clear()
            self._missing_since = None
            return self.spec.validate(dict(state='lost', at_ms=at_ms, flight=self._flight, events=events))
        return self.spec.validate(dict(state='missing', at_ms=at_ms, flight=self._flight, events=events))
