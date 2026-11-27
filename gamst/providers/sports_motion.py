"""Tracked objects that start, stop and touch: shots and ball-to-ball contacts (billiards, bowling, curling, boules).

Reads an ``objects`` module (``objects``, with track IDs; ``labels`` the ones to follow). Per track it keeps the
centre and its speed over the last ``window_ms`` (image widths per second, or metres per second on the table with a
``court`` reference, where balls roll on the plane). A **shot** starts when any object starts moving after everything was still for
``still_ms``, and ends after everything was still for ``still_ms`` again; it reports which object moved first (the cue ball), how many
contacts happened and how many objects moved. A **contact** is two objects whose centres come within ``touch`` times
their mean size while at least one of them changes velocity by ``min_dv`` (units per second; the collision), once per pair per
``refractory_ms``.
"""
from __future__ import annotations

import math
import threading
from collections import deque

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.court import court_map
from ..sports.objects import boxes


class ObjectMotion:
    def __init__(self, *, objects, labels=('sports ball',), court=None, moving=.05, still_ms=600, touch=1.2,
                 min_dv=.05, refractory_ms=300, window_ms=150, capability_id='sports.motion'):
        self.objects, self.labels = identifier(objects, 'module name'), tuple(labels)
        self.court = court_map(court)
        self.moving, self.still_ms, self.touch = float(moving), int(still_ms), float(touch)
        self.min_dv, self.refractory_ms, self.window_ms = float(min_dv), int(refractory_ms), int(window_ms)
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'video', 'gamst.sports_motion', f'1.{moving}-{still_ms}-{touch}',
            (OutputField('shots', 'integer'), OutputField('contacts', 'integer'), OutputField('moving', 'integer'),
             OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=1000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._tracks = {}           # id -> (ms, x, y, vx, vy, size), velocity over the last window_ms
            self._history = {}          # id -> deque of (ms, x, y)
            self._still_since = None
            self._shot = None           # dict(start_ms, first, contacts, moved)
            self._shots = self._contacts = 0
            self._touched = {}

    def _point(self, box):
        if self.court is None:
            return box.cx, box.cy, (box.width + box.height) / 2
        centre = self.court.to_court(box.cx, box.cy)
        corner = self.court.to_court(box.x + box.width, box.cy)
        if centre is None or corner is None:
            return None
        return centre[0], centre[1], 2 * math.hypot(corner[0] - centre[0], corner[1] - centre[1])

    def _velocity(self, key, at, x, y):
        """Displacement over the last ``window_ms`` (box jitter of one frame averages out)."""
        history = self._history.setdefault(key, deque())
        history.append((at, x, y))
        while len(history) > 2 and at - history[1][0] >= self.window_ms:
            history.popleft()
        t0, x0, y0 = history[0]
        dt = (at - t0) / 1000
        return (0., 0.) if dt <= 0 else ((x - x0) / dt, (y - y0) / dt)

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            if source != self.objects or not isinstance(frame.data, dict):
                return None
            at, events, now, changed = frame.timestamp_ms, [], {}, set()
            for box in boxes(frame.data, self.labels):
                point = None if box.track_id is None else self._point(box)
                if point is None:
                    continue
                x, y, size = point
                old = self._tracks.get(box.track_id)
                vx, vy = self._velocity(box.track_id, at, x, y)
                if old is not None and math.hypot(vx - old[3], vy - old[4]) >= self.min_dv:
                    changed.add(box.track_id)
                now[box.track_id] = (at, x, y, vx, vy, size)
            self._tracks.update(now)
            movers = [key for key, item in now.items() if math.hypot(item[3], item[4]) >= self.moving]
            if movers:
                if self._shot is None and self._still_since is not None and at - self._still_since >= self.still_ms:
                    self._shots += 1
                    first = max(movers, key=lambda key: math.hypot(now[key][3], now[key][4]))
                    self._shot = dict(start_ms=at, first=first, contacts=0, moved={first})
                    events.append(dict(event='shot.started', index=self._shots, timestamp_ms=at, first=first))
                if self._shot is not None:
                    self._shot['moved'].update(movers)
                self._still_since = None
            else:
                if self._still_since is None:
                    self._still_since = at
                if self._shot is not None and at - self._still_since >= self.still_ms:
                    # a shot ends after still_ms of stillness, not at one frame where the moving ball was missed
                    shot, self._shot = self._shot, None
                    events.append(dict(event='shot.ended', index=self._shots, start_ms=shot['start_ms'],
                                       end_ms=self._still_since, timestamp_ms=self._still_since, first=shot['first'],
                                       contacts=shot['contacts'], moved=len(shot['moved'])))
            keys = sorted(now)
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    pa, pb = now[a], now[b]
                    if math.hypot(pa[1] - pb[1], pa[2] - pb[2]) > self.touch * (pa[5] + pb[5]) / 2:
                        continue
                    if not ({a, b} & changed):
                        continue
                    pair = (a, b)
                    if at - self._touched.get(pair, -10 ** 9) < self.refractory_ms:
                        continue
                    self._touched[pair] = at
                    self._contacts += 1
                    if self._shot is not None:
                        self._shot['contacts'] += 1
                    events.append(dict(event='contact.objects', index=self._contacts, timestamp_ms=at, a=a, b=b))
            return self.spec.validate(dict(shots=self._shots, contacts=self._contacts, moving=len(movers),
                                           events=events)) if events else None
