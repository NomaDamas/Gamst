"""Possession: who has the ball, and when it goes from one player to another (an estimated pass or turnover).

Reads the ball state (``ball``, :class:`gamst.providers.sports_ball.BallTrack`) and players (``players``: an
``objects`` module; a team module's objects carry ``team``). At a ball sample, the nearest player whose box is within
``reach`` of the ball is the candidate; a candidate who stays for ``hold_ms`` becomes the holder. When the holder
changes from A to B:

- ``possession.pass``: same team (estimated pass; the ball may also have been deflected on the way);
- ``possession.turnover``: other team;
- ``possession.changed``: teams unknown.

No holder for ``loose_ms`` makes the ball loose (``possession.loose``). A long pass flies loose: the next holder within
``max_pass_ms`` of the last one is still classified against that last holder (``via_loose: true``). The ball is often hidden by the player who
holds it; ``hidden_ms`` keeps the holder while the ball is missing near them. These are estimates from one camera,
never official statistics: every event says so (``estimated: true``).
"""
from __future__ import annotations

import threading
from collections import deque

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.objects import boxes


class Possession:
    def __init__(self, *, ball, players, player_labels=('person',), reach=.03, hold_ms=200, loose_ms=800,
                 hidden_ms=1000, max_pass_ms=6000, capability_id='sports.possession'):
        for name, value in (('hold_ms', hold_ms), ('loose_ms', loose_ms), ('hidden_ms', hidden_ms)):
            if type(value) is not int or not 0 <= value <= 30_000:
                raise ValueError(f'{name} must be 0..30000 milliseconds')
        self.ball, self.players = identifier(ball, 'module name'), identifier(players, 'module name')
        self.player_labels, self.reach = tuple(player_labels), float(reach)
        self.hold_ms, self.loose_ms, self.hidden_ms, self.max_pass_ms = hold_ms, loose_ms, hidden_ms, int(max_pass_ms)
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'events', 'gamst.sports_possession', f'1.{reach}-{hold_ms}-{loose_ms}',
            (OutputField('holder', 'string', required=False), OutputField('team', 'string', required=False),
             OutputField('passes', 'integer'), OutputField('turnovers', 'integer'),
             OutputField('team_time_ms', 'object'), OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=loose_ms, interpretation='estimate', devices=('cpu',),
            license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._players = deque(maxlen=32)
            self._holder = None         # (track_id, team, since_ms, last_ms)
            self._previous = None       # the holder before the ball went loose
            self._candidate = None      # (track_id, team, since_ms)
            self._passes = self._turnovers = 0
            self._team_time = {}
            self._last_ms = None
            self._loose_reported = False

    def _snapshot(self, at_ms):
        found = min(self._players, key=lambda item: abs(item[0] - at_ms), default=None)
        return [] if found is None or abs(found[0] - at_ms) > 250 else found[1]

    def _near(self, at_ms, x, y):
        pairs = [(box.distance(x, y), box) for box in self._snapshot(at_ms) if box.track_id is not None]
        pairs = [pair for pair in pairs if pair[0] <= self.reach]
        return min(pairs, key=lambda pair: pair[0])[1] if pairs else None

    def _value(self, events):
        holder = self._holder
        value = dict(passes=self._passes, turnovers=self._turnovers,
                     team_time_ms={key: int(ms) for key, ms in self._team_time.items()}, events=events)
        if holder is not None:
            value['holder'] = holder[0]
            if holder[1]:
                value['team'] = holder[1]
        return self.spec.validate(value)

    def _change(self, track, team, at_ms):
        events = []
        old, via_loose = self._holder, False
        if old is None and self._previous is not None and at_ms - self._previous[3] <= self.max_pass_ms:
            old, via_loose = self._previous, True       # a pass that flew loose for a while
        self._previous = None
        if old is not None and old[0] != track:
            kind = 'changed' if not (old[1] and team) else 'pass' if old[1] == team else 'turnover'
            if kind == 'pass':
                self._passes += 1
            elif kind == 'turnover':
                self._turnovers += 1
            events.append(dict(event=f'possession.{kind}', timestamp_ms=int(at_ms), start_ms=int(old[3]),
                               end_ms=int(at_ms), **{'from': old[0], 'to': track}, from_team=old[1], to_team=team,
                               index=self._passes if kind == 'pass' else self._turnovers if kind == 'turnover' else 0,
                               via_loose=via_loose, estimated=True))
        elif old is None or via_loose:
            events.append(dict(event='possession.started', timestamp_ms=int(at_ms), who=track, team=team, estimated=True))
        self._holder = (track, team, at_ms, at_ms)
        self._loose_reported = False
        return events

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            value = frame.data if isinstance(frame.data, dict) else {}
            if source == self.players:
                self._players.append((frame.timestamp_ms, boxes(value, self.player_labels)))
                return None
            if source != self.ball:
                return None
            at_ms = int(value.get('at_ms', frame.timestamp_ms))
            if self._last_ms is not None and self._holder is not None and self._holder[1]:
                team = self._holder[1]
                self._team_time[team] = self._team_time.get(team, 0) + max(0, at_ms - self._last_ms)
            self._last_ms = at_ms
            events = []
            if value.get('state') == 'tracking':
                near = self._near(at_ms, float(value['x']), float(value['y']))
                if near is None:
                    self._candidate = None
                elif self._holder is not None and near.track_id == self._holder[0]:
                    self._holder = (*self._holder[:3], at_ms)
                    self._candidate = None
                elif self._candidate is None or self._candidate[0] != near.track_id:
                    self._candidate = (near.track_id, near.team, at_ms)
                elif at_ms - self._candidate[2] >= self.hold_ms:
                    events += self._change(near.track_id, near.team, self._candidate[2])
                    self._candidate = None
            elif self._holder is not None and value.get('state') == 'missing':
                if at_ms - self._holder[3] <= self.hidden_ms:
                    return None     # hidden by the holder: possession stays
            if self._holder is not None and not self._loose_reported and at_ms - self._holder[3] >= self.loose_ms:
                events.append(dict(event='possession.loose', timestamp_ms=int(self._holder[3]), last=self._holder[0],
                                   team=self._holder[1], estimated=True))
                self._previous, self._holder, self._loose_reported = self._holder, None, True
            return self._value(events) if events else None
