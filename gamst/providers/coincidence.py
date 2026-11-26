"""Events of one module that another module saw too: a movement that also made a sound, a word at a gesture.

Two records are read. An event of the first (``of``) is confirmed when an event of the second (``by``) lies
between ``before_ms`` before and ``after_ms`` after it, by capture time, whichever of the two arrives first;
the nearest free one confirms it, and it confirms one event only. An event is decided only once the second
module has answered for the whole interval, so a slow module delays the answer and never changes it.
"""
from __future__ import annotations

from collections import deque
import threading

from ..capabilities import CapabilitySpec, OutputField, identifier


def _time(event):
    for key in ('timestamp_ms', 'end_ms', 'start_ms'):
        if type(event.get(key)) in (int, float):
            return event[key]
    return None


class Coincidence:
    def __init__(self, of, by, *, of_event, by_event, before_ms=30, after_ms=150, max_pending=256,
                 capability_id='temporal.coincidence'):
        for value in (before_ms, after_ms):
            if type(value) is not int or not 0 <= value <= 60_000:
                raise ValueError('before_ms and after_ms must be integer milliseconds 0..60000')
        if of == by:
            raise ValueError('a module cannot confirm itself')
        self.of, self.by = identifier(of, 'module name'), identifier(by, 'module name')
        self.of_event, self.by_event = of_event, by_event
        self.before_ms, self.after_ms = before_ms, after_ms
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'events', 'gamst.coincidence', f'1.{before_ms}-{after_ms}',
            (OutputField('count', 'integer'), OutputField('unconfirmed', 'integer'), OutputField('events', 'array')),
            'window', window_ms=before_ms + after_ms + 1, freshness_ms=5000, interpretation='estimate',
            devices=('cpu',), license='Apache-2.0')
        self._pending, self._seen = deque(maxlen=max_pending), deque(maxlen=max_pending)
        self._count = self._unconfirmed = 0
        self._answered_ms = None      # up to where the second module has answered

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            if source not in (self.of, self.by):
                raise ValueError(f'{source!r} is neither {self.of!r} nor {self.by!r}')
            value = frame.data if isinstance(frame.data, dict) else {}
            events = [event for event in value.get('events') or () if isinstance(event, dict)]
            events = [event for event in events if event.get('event') ==
                      (self.of_event if source == self.of else self.by_event) and _time(event) is not None]
            if source == self.of:
                self._pending.extend(events)
            else:
                self._seen.extend(events)
                self._answered_ms = max(frame.timestamp_ms, self._answered_ms or frame.timestamp_ms)
            return self._decide()

    def _decide(self, *, last=False):
        events, waiting = [], []
        for event in sorted(self._pending, key=_time):
            at = _time(event)
            if not last and (waiting or self._answered_ms is None or self._answered_ms < at + self.after_ms):
                waiting.append(event)
                continue
            near = [item for item in self._seen if at - self.before_ms <= _time(item) <= at + self.after_ms]
            if not near:
                self._unconfirmed += 1
                events.append(dict(event='coincidence.unconfirmed', timestamp_ms=round(at)))
                continue
            found = min(near, key=lambda item: abs(_time(item) - at))
            self._seen.remove(found)
            self._count += 1
            events.append(dict(event='coincidence.confirmed', index=self._count, timestamp_ms=round(at),
                               by_ms=round(_time(found)), delay_ms=round(_time(found) - at),
                               label=event.get('label', self.of)))
        self._pending.clear()
        self._pending.extend(waiting)
        if not events:
            return None
        return self.spec.validate(dict(count=self._count, unconfirmed=self._unconfirmed, events=events))

    def finish(self):
        with self._lock:
            return self._decide(last=True)
