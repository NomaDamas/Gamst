"""The verification lane of a live counter: a second counter over every frame settles each live count.

The video counterpart of the word verification lane (docs/catch-up-verification.md). The live counter (L1, e.g. the
recipe's ``exemplar`` on MediaPipe in a ``sample="newest"`` lane) commits each count at once as ``provisional``, with
an ``event_id``. A verification counter (the same movement on a more accurate pose model, in a ``backlog="spill"``
lane that processes every frame, behind) counts too. This module reads both, in capture order, and once the
verifier has seen past a count by ``near_ms`` it commits, under the live counter's capability, one ``event.revision``
per count (:mod:`gamst.counting`):

- ``final``: the verifier counted it too (its end within ``near_ms``; paired one to one, nearest first);
- ``revoked`` (``not_seen_by_verifier``): the verifier saw the person there and did not count it;
- ``uncertain`` (``verifier_saw_no_pose``): the verifier had no pose during the count, so it cannot say; it is shown,
  not counted, and never "not done";
- a verifier count no live count pairs with is **added**: a new ``final`` event (``decided_by: "verification"``).

Every number is a parameter; nothing here knows a movement.
"""
from __future__ import annotations

from collections import deque

REVISION = 'event.revision'


class CountVerification:
    name = 'count_verification'
    modalities = ('video',)

    def __init__(self, spec, *, live, verifier, event='exemplar.matched', near_ms=150, by='verification'):
        if type(near_ms) is not int or not 0 < near_ms <= 10000:
            raise ValueError('near_ms is integer milliseconds 1..10000')
        self.spec, self.live, self.verifier, self.event = spec, live, verifier, event
        self.near_ms, self.by = near_ms, by
        self._open = []           # live counts waiting: their events
        self._seen = []           # verifier counts not paired or added yet
        self._posed = deque(maxlen=4096)   # capture times at which the verifier had a pose
        self._live_ms = self._verified_ms = None
        self._identity = {}       # the live counter's reference, given to added counts
        self._counted, self._added = 0, 0
        self.decided = dict(final=0, revoked=0, uncertain=0, added=0)

    def status(self):
        return dict(waiting=len(self._open), verifier_unpaired=len(self._seen), decided=dict(self.decided),
                    live_ms=self._live_ms, verified_ms=self._verified_ms, near_ms=self.near_ms)

    def process(self, frame):
        value = frame.data if isinstance(frame.data, dict) else {}
        source = getattr(frame.sensor, 'sensor_id', None)
        events = [item for item in value.get('events') or () if isinstance(item, dict) and item.get('event') == self.event]
        if source == self.live:
            self._live_ms = max(frame.timestamp_ms, self._live_ms or frame.timestamp_ms)
            self._identity.update({key: value[key] for key in ('reference_id', 'reference_revision') if key in value})
            for item in events:
                if item.get('state') == 'provisional' and isinstance(item.get('event_id'), str):
                    self._open.append(item)
                    self._counted += 1
        elif source == self.verifier:
            self._verified_ms = max(frame.timestamp_ms, self._verified_ms or frame.timestamp_ms)
            self._posed.append(frame.timestamp_ms)
            self._seen += events
        else:
            return None
        return self._decide()

    def finish(self):
        """End of input: what is still open is decided on what the verifier saw."""
        return self._decide(end=True)

    def _decide(self, end=False):
        if not end and (self._live_ms is None or self._verified_ms is None):
            return None
        horizon = None if end else min(self._live_ms, self._verified_ms) - self.near_ms
        out = []
        for item in sorted(self._open, key=lambda e: e['end_ms']):
            if horizon is not None and item['end_ms'] > horizon:
                break
            near = [seen for seen in self._seen if abs(seen['end_ms'] - item['end_ms']) <= self.near_ms]
            self._open.remove(item)
            if near:
                best = min(near, key=lambda seen: abs(seen['end_ms'] - item['end_ms']))
                self._seen.remove(best)
                out.append(self._revision(item, 'final', basis='seen_by_verifier',
                                          verifier_end_ms=best['end_ms']))
            elif any(item['start_ms'] - self.near_ms <= ms <= item['end_ms'] + self.near_ms for ms in self._posed):
                out.append(self._revision(item, 'revoked', reason='not_seen_by_verifier'))
                self._counted -= 1
            else:
                out.append(self._revision(item, 'uncertain', reason='verifier_saw_no_pose'))
                self._counted -= 1
        for seen in sorted(self._seen, key=lambda e: e['end_ms']):
            if horizon is not None and seen['end_ms'] > horizon:
                break
            if any(abs(item['end_ms'] - seen['end_ms']) <= self.near_ms for item in self._open):
                continue    # a live count that may still pair with it is not decided yet
            self._seen.remove(seen)
            self._added += 1
            self._counted += 1
            self.decided['added'] += 1
            added = {key: seen[key] for key in ('reference', 'label', 'method', 'confirmed', 'start_ms', 'end_ms',
                                               'distance', 'example') if key in seen}
            out.append(dict(added, event=self.event, event_id=f'{self.by}:{seen["end_ms"]}:{self._added}',
                            state='final', decided_by=self.by, **self._identity))
        if not out:
            return None
        return self.spec.validate(dict(count=self._counted, status='verified', events=out, **self._identity))

    def _revision(self, item, state, **extra):
        self.decided[state] += 1
        return dict(event=REVISION, revises=item['event_id'], state=state, by=self.by, start_ms=item['start_ms'],
                    end_ms=item['end_ms'], **extra)
