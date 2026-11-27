"""Ball contacts: where a flight breaks, and who or what broke it (hit, bounce, deflection).

A ball in flight changes direction only when something touches it. This module reads the ball state of
:class:`gamst.providers.sports_ball.BallTrack` (``ball``) and finds the breaks of each flight: at sample k, the
image velocity over the ``span`` samples before and the ``span`` after differ by at least ``min_turn_deg`` (or the
speed changes by ``speed_ratio``). Breaks closer than ``min_contact_ms`` are one contact, the sharpest. A break is
decided ``span`` samples after it, so a contact is reported that many frames late (a few tens of milliseconds).

What broke it:

- **hit**: the ball is within ``reach`` of a player box at that time (``players``: an ``objects`` module, its
  ``player_labels``); the event names the player's ``track_id`` (``who``) and team when a team module labelled it;
- **bounce**: no player is near and the image vertical velocity turns from down to up (the ball met the ground);
  with a ``court`` reference its court position in metres, whether it is inside the lines, and which half;
- **deflection**: anything else (net, wall, frame of a racket nobody detected).

``sound`` (an ``audio-onsets`` module, optional) adds whether an impact sound lies within ``before_ms`` /
``after_ms`` of the contact and its delay; a contact then waits until that module has answered past it, so a slow
module delays the answer and never changes it. The sound never makes a contact by itself: a racket sound with no
seen ball break is not counted (docs/sports.md, "what is not counted").
"""
from __future__ import annotations

import math
import threading
from collections import deque

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.court import court_map
from ..sports.objects import boxes

KINDS = ('hit', 'bounce', 'deflection')


def _angle(a, b):
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return 0.
    cos = max(-1., min(1., (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cos))


class BallContacts:
    def __init__(self, *, ball, players=None, sound=None, player_labels=('person',), reach=.04, span=2,
                 min_turn_deg=40., speed_ratio=2.5, min_speed=.05, min_contact_ms=120, before_ms=40, after_ms=120,
                 max_gap_ms=500, court=None, sound_event='audio.onset', capability_id='sports.contacts'):
        if type(span) is not int or not 1 <= span <= 10:
            raise ValueError('span must be 1..10 samples')
        if type(min_contact_ms) is not int or not 1 <= min_contact_ms <= 5000:
            raise ValueError('min_contact_ms must be 1..5000')
        self.ball = identifier(ball, 'module name')
        self.players = None if players is None else identifier(players, 'module name')
        self.sound = None if sound is None else identifier(sound, 'module name')
        self.player_labels, self.reach, self.span = tuple(player_labels), float(reach), span
        self.min_turn, self.speed_ratio, self.min_speed = float(min_turn_deg), float(speed_ratio), float(min_speed)
        self.min_contact_ms, self.before_ms, self.after_ms = min_contact_ms, int(before_ms), int(after_ms)
        self.max_gap_ms, self.sound_event = int(max_gap_ms), sound_event
        self.court = court_map(court)
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'events', 'gamst.sports_contacts',
            f'1.{span}-{min_turn_deg}-{speed_ratio}-{reach}-{min_contact_ms}',
            (OutputField('hits', 'integer'), OutputField('bounces', 'integer'), OutputField('deflections', 'integer'),
             OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=5000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._samples = deque(maxlen=4 * self.span + 4)     # (at_ms, x, y, flight) of tracking states
            self._players = deque(maxlen=64)                     # (at_ms, [Box]) recent player snapshots
            self._sounds = deque(maxlen=256)                     # onset times
            self._sound_until = None
            self._candidate = None                               # the sharpest break not yet decided
            self._pending = []                                   # breaks waiting for the sound module
            self._counts = dict.fromkeys(KINDS, 0)
            self._last_contact = None

    # ------------------------------------------------------------------ reading the three inputs

    def process(self, frame):
        with self._lock:
            source = None if frame.sensor is None else frame.sensor.sensor_id
            value = frame.data if isinstance(frame.data, dict) else {}
            if source == self.players:
                self._players.append((frame.timestamp_ms, boxes(value, self.player_labels)))
                return self._decide()
            if source == self.sound:
                for event in value.get('events') or ():
                    if isinstance(event, dict) and event.get('event') == self.sound_event:
                        at = event.get('timestamp_ms', event.get('start_ms'))
                        if type(at) in (int, float):
                            self._sounds.append(float(at))
                self._sound_until = max(frame.timestamp_ms, self._sound_until or frame.timestamp_ms)
                return self._decide()
            if source != self.ball:
                return None
            if value.get('state') == 'tracking':
                sample = (int(value['at_ms']), float(value['x']), float(value['y']), int(value.get('flight', 0)))
                if self._samples and (sample[3] != self._samples[-1][3] or
                                      sample[0] - self._samples[-1][0] > self.max_gap_ms):
                    self._flush_candidate()
                    self._samples.clear()
                self._samples.append(sample)
                self._look()
                if self._candidate is not None and sample[0] - self._candidate['at_ms'] >= self.min_contact_ms:
                    self._flush_candidate()     # nothing sharper can follow inside min_contact_ms any more
            elif value.get('state') == 'lost':
                self._flush_candidate()
                self._samples.clear()
            return self._decide()

    def _look(self):
        """The break at the sample ``span`` back from the newest, if any; the sharpest within min_contact_ms wins."""
        if len(self._samples) < 2 * self.span + 1:
            return
        samples = list(self._samples)
        k = len(samples) - 1 - self.span
        before, here, after = samples[k - self.span], samples[k], samples[k + self.span]
        dt_before, dt_after = (here[0] - before[0]) / 1000, (after[0] - here[0]) / 1000
        if dt_before <= 0 or dt_after <= 0:
            return
        v0 = ((here[1] - before[1]) / dt_before, (here[2] - before[2]) / dt_before)
        v1 = ((after[1] - here[1]) / dt_after, (after[2] - here[2]) / dt_after)
        s0, s1 = math.hypot(*v0), math.hypot(*v1)
        if max(s0, s1) < self.min_speed:
            return
        turn = _angle(v0, v1)
        ratio = max(s0, s1) / max(1e-9, min(s0, s1))
        if turn < self.min_turn and ratio < self.speed_ratio:
            return
        sharpness = turn + 10 * math.log(ratio)
        candidate = dict(at_ms=here[0], x=here[1], y=here[2], v0=v0, v1=v1, turn=turn, ratio=ratio, sharpness=sharpness)
        if self._candidate is not None and here[0] - self._candidate['at_ms'] < self.min_contact_ms:
            if sharpness > self._candidate['sharpness']:
                self._candidate = candidate
            return
        self._flush_candidate()
        self._candidate = candidate

    def _flush_candidate(self):
        if self._candidate is not None:
            if self._last_contact is None or self._candidate['at_ms'] - self._last_contact >= self.min_contact_ms:
                self._pending.append(self._candidate)
                self._last_contact = self._candidate['at_ms']
            self._candidate = None

    # ------------------------------------------------------------------ deciding

    def _nearest_player(self, at_ms, x, y):
        snapshot = min(self._players, key=lambda item: abs(item[0] - at_ms), default=None)
        if snapshot is None or abs(snapshot[0] - at_ms) > 250:
            return None, None
        found = [(box.distance(x, y), box) for box in snapshot[1]]
        found = [pair for pair in found if pair[0] <= self.reach]
        if not found:
            return None, None
        distance, box = min(found, key=lambda pair: pair[0])
        return box, distance

    def _event(self, contact):
        at, x, y = contact['at_ms'], contact['x'], contact['y']
        player, distance = self._nearest_player(at, x, y) if self.players else (None, None)
        if player is not None:
            kind = 'hit'
        elif contact['v0'][1] > 0 and contact['v1'][1] < 0:
            kind = 'bounce'
        else:
            kind = 'deflection'
        self._counts[kind] += 1
        event = dict(event=f'contact.{kind}', kind=kind, index=self._counts[kind], timestamp_ms=int(at),
                     x=round(x, 5), y=round(y, 5), turn_deg=round(contact['turn'], 1),
                     speed_before=round(math.hypot(*contact['v0']), 4), speed_after=round(math.hypot(*contact['v1']), 4))
        if player is not None:
            event.update(who=player.track_id, reach=round(distance, 4), **({'team': player.team} if player.team else {}))
        if self.court is not None and kind == 'bounce':
            point = self.court.to_court(x, y)
            if point is not None:
                event.update(court_x_m=round(point[0], 2), court_y_m=round(point[1], 2),
                             in_court=self.court.inside(*point, margin_m=0.), half=self.court.half(point[1]))
        if self.sound is not None:
            near = [onset for onset in self._sounds if at - self.before_ms <= onset <= at + self.after_ms]
            event['sound'] = bool(near)
            if near:
                event['sound_delay_ms'] = int(round(min(near, key=lambda onset: abs(onset - at)) - at))
        return event

    def _decide(self, *, last=False):
        events, waiting = [], []
        for contact in self._pending:
            if not last and self.sound is not None and (
                    self._sound_until is None or self._sound_until < contact['at_ms'] + self.after_ms):
                waiting.append(contact)
                continue
            events.append(self._event(contact))
        self._pending = waiting
        if not events:
            return None
        return self.spec.validate(dict(hits=self._counts['hit'], bounces=self._counts['bounce'],
                                       deflections=self._counts['deflection'], events=events))

    def finish(self):
        with self._lock:
            self._flush_candidate()
            return self._decide(last=True)
