"""Where each player was: positions, distance covered, speed and an occupancy heat map, per track and per team.

Reads players (``players``: an ``objects`` module). A player's position is where the box touches the ground (its
bottom centre); with a ``court`` reference it is mapped to metres on the court plane (:class:`gamst.sports.court.
CourtMap`), otherwise it stays in normalised image units (``unit`` says which). Distance adds steps longer than
``min_step`` (box jitter is not running); speed is the distance of the last ``speed_window_ms``. The heat map counts
samples in a ``cols`` x ``rows`` grid over the court (or the image), all players and per team.

A record is committed every ``every_ms`` of capture time: a summary for the LLM, not one value per frame.
"""
from __future__ import annotations

import math
import threading
from collections import deque

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.court import court_map
from ..sports.objects import boxes


class Occupancy:
    def __init__(self, *, players, player_labels=('person',), court=None, cols=12, rows=6, every_ms=1000,
                 min_step=None, speed_window_ms=1000, max_tracks=40, capability_id='sports.occupancy'):
        if type(cols) is not int or type(rows) is not int or not 1 <= cols <= 64 or not 1 <= rows <= 64:
            raise ValueError('cols and rows must be 1..64')
        if type(every_ms) is not int or not 100 <= every_ms <= 600_000:
            raise ValueError('every_ms must be 100..600000')
        self.players, self.player_labels = identifier(players, 'module name'), tuple(player_labels)
        self.court = court_map(court)
        self.unit = 'm' if self.court is not None else 'normalized_image'
        self.min_step = float(min_step if min_step is not None else (.15 if self.court is not None else .002))
        self.cols, self.rows, self.every_ms, self.speed_window_ms = cols, rows, every_ms, int(speed_window_ms)
        self.max_tracks = int(max_tracks)
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'video', 'gamst.sports_occupancy', f'1.{cols}x{rows}-{every_ms}-{self.unit}',
            (OutputField('unit', 'string'), OutputField('tracks', 'array'), OutputField('teams', 'object'),
             OutputField('grid', 'object')),
            'frame', stateful=True, freshness_ms=2 * every_ms, interpretation='measurement', devices=('cpu',),
            license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._tracks = {}       # track_id -> dict(team, last, distance, recent deque, first_ms, last_ms)
            self._grid = [0] * (self.cols * self.rows)
            self._team_grid = {}
            self._next_ms = None

    def _position(self, box):
        x, y = box.foot
        if self.court is None:
            return x, y, 0. <= x <= 1. and 0. <= y <= 1.
        point = self.court.to_court(x, y)
        if point is None:
            return None
        return point[0], point[1], self.court.inside(*point)

    def _cell(self, x, y):
        width, height = (self.court.width, self.court.length) if self.court is not None else (1., 1.)
        col = min(self.cols - 1, max(0, int(x / width * self.cols)))
        row = min(self.rows - 1, max(0, int(y / height * self.rows)))
        return row * self.cols + col

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            if source != self.players or not isinstance(frame.data, dict):
                return None
            at = frame.timestamp_ms
            for box in boxes(frame.data, self.player_labels):
                if box.track_id is None:
                    continue
                found = self._position(box)
                if found is None:
                    continue
                x, y, inside = found
                track = self._tracks.get(box.track_id)
                if track is None:
                    if len(self._tracks) >= self.max_tracks:
                        oldest = min(self._tracks, key=lambda key: self._tracks[key]['last_ms'])
                        del self._tracks[oldest]
                    track = self._tracks[box.track_id] = dict(team=box.team, last=(x, y), distance=0., first_ms=at,
                                                              last_ms=at, recent=deque())
                step = math.hypot(x - track['last'][0], y - track['last'][1])
                if step >= self.min_step:
                    track['distance'] += step
                    track['recent'].append((at, step))
                    track['last'] = (x, y)
                track['last_ms'], track['team'] = at, box.team or track['team']
                track['position'] = (x, y)
                while track['recent'] and at - track['recent'][0][0] > self.speed_window_ms:
                    track['recent'].popleft()
                if inside:
                    cell = self._cell(x, y)
                    self._grid[cell] += 1
                    if box.team:
                        self._team_grid.setdefault(box.team, [0] * len(self._grid))[cell] += 1
            if self._next_ms is None:
                self._next_ms = at + self.every_ms
            if at < self._next_ms:
                return None
            self._next_ms = at + self.every_ms
            return self._summary(at)

    def _summary(self, at):
        tracks, teams = [], {}
        for key, track in sorted(self._tracks.items()):
            if 'position' not in track:
                continue
            speed = sum(step for _, step in track['recent']) * 1000 / self.speed_window_ms
            tracks.append(dict(track_id=key, team=track['team'], x=round(track['position'][0], 3),
                               y=round(track['position'][1], 3), distance=round(track['distance'], 2),
                               speed=round(speed, 3), seen_ms=int(track['last_ms'] - track['first_ms']),
                               current=at - track['last_ms'] <= self.every_ms))
            if track['team']:
                total = teams.setdefault(track['team'], dict(distance=0., players=0))
                total['distance'] = round(total['distance'] + track['distance'], 2)
                total['players'] += 1
        grid = dict(cols=self.cols, rows=self.rows, over='court' if self.court is not None else 'image',
                    cells=list(self._grid), teams={team: list(cells) for team, cells in self._team_grid.items()})
        return self.spec.validate(dict(unit=self.unit, tracks=tracks, teams=teams, grid=grid))

    def finish(self):
        with self._lock:
            return self._summary(max((track['last_ms'] for track in self._tracks.values()), default=0)) \
                if self._tracks else None
