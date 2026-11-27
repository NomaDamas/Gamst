"""Rallies: the contacts of one exchange, from the first hit until play stops.

Reads :class:`gamst.providers.sports_contacts.BallContacts` (``contacts``) and the ball state (``ball``, for time
and for a lost ball). A rally starts with a hit after a pause and ends when

- no contact follows for ``max_gap_ms`` (the ball left the view or play stopped), or
- the ball is lost for ``lost_ms`` with no contact, or
- with a court reference, two bounces on the same half follow each other with no hit between (the point is over in
  tennis, padel and table tennis, where a ball may bounce once per side): ``ended_by`` says which.

``rally.ended`` carries the hits, bounces, players who hit and the duration; ``rally.started`` its first hit. The
longest rally so far is in every record. A rally of fewer than ``min_hits`` hits is reported with ``short: true``
(a serve fault, a ball thrown back).
"""
from __future__ import annotations

import threading

from ..capabilities import CapabilitySpec, OutputField, identifier


class Rally:
    def __init__(self, *, contacts, ball, max_gap_ms=2500, lost_ms=1500, min_hits=2, double_bounce=True,
                 capability_id='sports.rally'):
        for name, value in (('max_gap_ms', max_gap_ms), ('lost_ms', lost_ms)):
            if type(value) is not int or not 100 <= value <= 60_000:
                raise ValueError(f'{name} must be 100..60000 milliseconds')
        self.contacts, self.ball = identifier(contacts, 'module name'), identifier(ball, 'module name')
        self.max_gap_ms, self.lost_ms, self.min_hits, self.double_bounce = max_gap_ms, lost_ms, int(min_hits), double_bounce
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'events', 'gamst.sports_rally', f'1.{max_gap_ms}-{lost_ms}-{min_hits}',
            (OutputField('rallies', 'integer'), OutputField('longest_hits', 'integer'),
             OutputField('in_rally', 'boolean'), OutputField('hits', 'integer'), OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=max_gap_ms, interpretation='estimate', devices=('cpu',),
            license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._open = None
            self._count = self._longest = 0
            self._lost_since = None

    def _start(self, event):
        self._open = dict(start_ms=event['timestamp_ms'], last_ms=event['timestamp_ms'], hits=0, bounces=0,
                          deflections=0, who=[], last_bounce_half=None)

    def _end(self, at_ms, why):
        rally, self._open = self._open, None
        self._count += 1
        self._longest = max(self._longest, rally['hits'])
        return dict(event='rally.ended', index=self._count, start_ms=rally['start_ms'], end_ms=int(rally['last_ms']),
                    timestamp_ms=int(rally['last_ms']), hits=rally['hits'], bounces=rally['bounces'],
                    deflections=rally['deflections'], duration_ms=int(rally['last_ms'] - rally['start_ms']),
                    who=rally['who'], ended_by=why, short=rally['hits'] < self.min_hits, decided_ms=int(at_ms))

    def _contact(self, event, events):
        kind = event.get('kind')
        if self._open is None:
            if kind != 'hit':
                return
            self._start(event)
            events.append(dict(event='rally.started', index=self._count + 1, timestamp_ms=event['timestamp_ms'],
                               who=event.get('who')))
        rally = self._open
        rally['last_ms'] = event['timestamp_ms']
        if kind == 'hit':
            rally['hits'] += 1
            rally['last_bounce_half'] = None
            if event.get('who') is not None:
                rally['who'].append(event['who'])
        elif kind == 'bounce':
            rally['bounces'] += 1
            half = event.get('half')
            if self.double_bounce and half is not None and half == rally['last_bounce_half']:
                events.append(self._end(event['timestamp_ms'], 'double_bounce'))
                return
            rally['last_bounce_half'] = half
        else:
            rally['deflections'] += 1

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            value = frame.data if isinstance(frame.data, dict) else {}
            events = []
            if source == self.contacts:
                for event in sorted((item for item in value.get('events') or () if isinstance(item, dict)),
                                    key=lambda item: item.get('timestamp_ms', 0)):
                    self._contact(event, events)
            elif source == self.ball:
                now = int(value.get('at_ms', frame.timestamp_ms))
                if value.get('state') == 'tracking':
                    self._lost_since = None
                elif value.get('state') in ('lost', 'idle', 'missing') and self._lost_since is None:
                    self._lost_since = now
                if self._open is not None:
                    if now - self._open['last_ms'] >= self.max_gap_ms:
                        events.append(self._end(now, 'no_contact'))
                    elif self._lost_since is not None and now - max(self._lost_since, self._open['last_ms']) >= self.lost_ms:
                        events.append(self._end(now, 'ball_lost'))
            else:
                return None
            if not events:
                return None
            return self.spec.validate(dict(rallies=self._count, longest_hits=self._longest, in_rally=self._open is not None,
                                           hits=0 if self._open is None else self._open['hits'], events=events))

    def finish(self):
        with self._lock:
            if self._open is None:
                return None
            event = self._end(self._open['last_ms'], 'input_ended')
            return self.spec.validate(dict(rallies=self._count, longest_hits=self._longest, in_rally=False, hits=0,
                                           events=[event]))
