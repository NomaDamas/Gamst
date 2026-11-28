"""The movement counter for every tracked person: one :class:`ExemplarEvents` per track, each event with its track and who.

Design section 2 (owner decision 43): matchers run for every track, identity is bound afterwards. :class:`TrackExemplar`
reads the pose module's joints (every visible person, with the tracker's ``track_id``) and, optionally, the
``vision.person_binding`` records of :class:`gamst.providers.person_binding.PersonBinding`. Each track has its own copy
of the movement counter (examples, a rule; the same thresholds and states), so two people never mix in one match.

Every event carries ``track_id`` and ``who``: the person bound to that track when it was counted, else ``unknown``. A
binding decided later reaches back over its track (retroactive): the value's ``by_who`` is recomputed from the bindings,
and the stored events are resolved the same way when they are read (:func:`gamst.providers.person_binding.resolve`, MCP
``count_events`` / ``get_timeline`` with ``who``). "A given person's action" is the events whose who is that person.

An instruction may name the person it is about (``who`` in the movement value, MCP ``set_instruction``): then ``count`` is
that person's count and ``count_all`` everyone's. It is the same capability (``temporal.exemplar_events``) as the
single-person counter, so the HUD, Watches and count_events read it as before.
"""
from __future__ import annotations

from ..capabilities import CapabilitySpec, OutputField
from ..plugins import InputFrame
from ..private_payload import PrivatePoses, poses_of
from .person_binding import CAPABILITY as BINDING, UNKNOWN, PersonOnce, TrackBindings, intervals_of, resolve
from .pose_exemplar import ExemplarEvents, largest_pose

MAX_TRACKS = 32          # counters kept at once (a crowd beyond it: the longest unseen ones go first)
TRACK_LOST_MS = 10000    # a track not seen this long is finished
MAX_EVENTS_KEPT = 4096   # event times kept for the retroactive by_who
CONTINUE_IOU = .5        # an untracked pose continues the track it overlaps this much in the frame before
UNTRACKED = 'untracked'
FIRST_SEEN_MS = 5000     # a track unseen this long starts again (the person binding's lost_ms): who is kept of two
                         # duplicate tracks of one body is the one seen first, the same choice as the binding's


class TrackExemplar(ExemplarEvents):
    """:class:`ExemplarEvents` per track (it inherits how a movement is set, prepared and scheduled)."""

    def __init__(self, **options):
        super().__init__(**options)
        self.spec = CapabilitySpec(
            self.spec.id, 'temporal.exemplar_events', self.spec.modality, 'gamst.exemplar_events', '3.tracks',
            (*self.spec.fields, OutputField('count_all', 'integer', required=False),
             OutputField('who', 'string', required=False), OutputField('by_who', 'object', required=False),
             OutputField('tracks', 'integer', required=False), OutputField('by_person', 'object', required=False)),
            'native', stateful=True, reset_on_gap=True, freshness_ms=self.max_gap_ms, interpretation='estimate',
            devices=('cpu',), license='Apache-2.0')
        self._options = dict(options)
        self._binding_events, self._forked, self._namespace = [], False, None
        self.bindings = TrackBindings()
        self._first_seen, self._last_seen, self.duplicates = {}, {}, 0     # duplicate boxes of one body: not counted

    def _clear(self):
        super()._clear()
        self._tracks, self._counted, self._indices = {}, [], {}   # track -> counter; [(track, end_ms)] counted

    def fork(self):
        twin = super().fork()
        twin._forked = True
        return twin

    def _counter(self):
        options = {key: value for key, value in self._options.items() if key != 'capability_id'}
        counter = ExemplarEvents(**options)
        counter._host = self._host       # count or search mode: how a rule and examples combine
        counter.set_exemplar(self._reference['value'])
        counter.label, counter.identity, counter.info = self.label, dict(self.identity), dict(self.info)
        counter._methods = list(self._methods)
        if not self.fixed_thresholds:
            counter.enter, counter.exit = self.enter, self.exit
        return counter

    @property
    def wanted(self):
        """The person the current instruction is about, or None."""
        value = (getattr(self, '_reference', None) or {}).get('value')
        who = value.get('who') if isinstance(value, dict) else None
        return who if isinstance(who, str) and who else None

    # ------------------------------------------------------------------ who

    def _track_name(self, pose, ts, index):
        track = pose.get('track_id') or f'frame:{ts}:{index}'
        if self._forked:
            self._namespace = ts if self._namespace is None else self._namespace
            track = f'{track}@{self._namespace}'
        return track

    def _learn(self, value):
        """A ``vision.person_binding`` value: who each track is from now on (and before, within the track)."""
        for item in value.get('events') or ():
            if not isinstance(item, dict):
                continue
            self._binding_events.append(item)
            if item.get('event') == 'person.bound':
                self.bindings.tracks.setdefault(item['track_id'], dict(first_ms=item['start_ms'], last_ms=item['end_ms'],
                                                                       who=None, from_ms=None, by=None, bound_before=True))
                self.bindings.tracks[item['track_id']].update(who=item['who'], from_ms=item['start_ms'], by=item.get('by'),
                                                              person_id=item.get('person_id'))
            elif item.get('event') == 'person.unbound' and item.get('track_id') in self.bindings.tracks:
                self.bindings.tracks[item['track_id']].update(who=None, from_ms=None, by=None, person_id=None)
        del self._binding_events[:-MAX_EVENTS_KEPT]

    def by_who(self, key='who'):
        """Counts per person, resolved from the bindings (retroactive); one action of a person counted on two tracks
        within ``SAME_EVENT_MS`` counts once (:class:`PersonOnce`). ``key='person_id'``: per person id (named or
        not; ``unknown`` for tracks nobody was bound to)."""
        intervals = intervals_of(self._binding_events, key=key)
        counts, once = {}, PersonOnce()
        for track, end_ms in sorted(self._counted, key=lambda item: item[1]):
            who = resolve(dict(track_id=track, end_ms=end_ms), intervals, key) or UNKNOWN
            if once.fresh(who, track, end_ms):
                counts[who] = counts.get(who, 0) + 1
        return counts

    # ------------------------------------------------------------------ frames

    def _tracked(self, poses):
        """``{track: pose}``. A pose the tracker gave no track continues the track whose box it overlaps most in the
        frame before (IoU >= :data:`CONTINUE_IOU`), else the one ``untracked`` stream (a pose model without a tracker:
        one person); further untracked poses of that frame are not counted."""
        from .person_match import box_iou, duplicates
        found, loose = {}, []
        for index, pose in enumerate(poses):
            if not isinstance(pose, dict) or not pose.get('landmarks'):
                continue
            if pose.get('track_id'):
                found[self._track_name(pose, self._frame_ms, index)] = pose
            else:
                loose.append(pose)
        boxes = getattr(self, '_boxes', {})
        for pose in loose:
            best = max(((box_iou(pose.get('bbox'), box), track) for track, box in boxes.items() if track not in found),
                       default=(0., None))
            track = best[1] if best[0] >= CONTINUE_IOU else UNTRACKED
            if track not in found:
                found[track] = pose
        self._boxes = {track: pose.get('bbox') for track, pose in found.items() if pose.get('bbox')}
        first = self._first_seen
        for track in [track for track, ms in first.items()
                      if self._frame_ms - self._last_seen.get(track, ms) > FIRST_SEEN_MS]:
            first.pop(track, None)
            self._last_seen.pop(track, None)
        for track in found:
            first.setdefault(track, self._frame_ms)
            self._last_seen[track] = self._frame_ms
        doubles = duplicates({track: pose for track, pose in found.items() if track != UNTRACKED}, first)
        self.duplicates += len(doubles)
        return {track: pose for track, pose in found.items() if track not in doubles}

    def _feed(self, counter, pose, ts):
        if self.rule is not None and self.rule.needs_joints:
            return counter.process(InputFrame(PrivatePoses([pose]), 'video', ts))
        best = largest_pose([pose], self.mapping, self.min_visibility)
        if best is None:
            return None
        channels, mask, _, _ = best
        return counter.process(InputFrame(dict(channels=[round(v, 5) for v in channels], mask=mask), 'video', ts))

    def _tagged(self, track, events):
        tagged = []
        for item in events:
            who = self.bindings.who(track) or UNKNOWN
            person = (self.bindings.tracks.get(track) or {}).get('person_id')
            item = dict(item, track_id=track, who=who, track_index=item.get('index'), **({'person_id': person}
                                                                                          if person else {}))
            if item['event'] == 'exemplar.matched':
                self._count += 1
                self._counted.append((track, item['end_ms']))
                del self._counted[:-MAX_EVENTS_KEPT]
                self._indices[track, item['track_index']] = self._count
            item['index'] = self._indices.get((track, item['track_index']), self._count)
            tagged.append(item)
        return tagged

    def _result(self, events, distance=None):
        counts = self.by_who()
        everyone = sum(counts.values())
        wanted = self.wanted
        value = dict(status='matching' if any(counter._inside for counter in self._tracks.values()) else 'watching',
                     events=events, tracks=len(self._tracks), by_who=dict(list(counts.items())[:16]),
                     by_person=dict(list(self.by_who('person_id').items())[:32]), **self.identity)
        if distance is not None:
            value['distance'] = round(distance, 5)
        if wanted is not None:
            value.update(who=wanted, count=counts.get(wanted, 0), count_all=everyone)
        else:
            value['count'] = everyone
        return self.spec.validate(value)

    def _switch(self, ts):
        said = []
        while self._next and self._next[0][0] <= ts:
            from_ms, ready = self._next.pop(0)
            ended = []
            for track, counter in self._tracks.items():
                if counter._inside:
                    ended += self._tagged(track, [counter._ended(counter._last_ms, cut='instruction_changed')])
            if ended:
                said.append(self._result(ended))
            self._adopt(ready, from_ms)
        return said

    def process(self, frame):
        with self._lock:
            said = self._switch(frame.timestamp_ms)
            value = frame.data if isinstance(frame.data, dict) else None
            if value is not None and isinstance(value.get('tracks'), list) and isinstance(value.get('bound'), dict):
                self._learn(value)          # a vision.person_binding record
                return said or None
            if self.info.get('removed') or frame.timestamp_ms < self.from_ms:
                return said or None
            if self.scorer is None and self.rule is None:
                raise RuntimeError('no exemplar: add an exemplar reference before the module')
            poses = poses_of(frame.data)
            if poses is None:
                return said or None
            ts, events, distances = frame.timestamp_ms, [], []
            self._frame_ms = ts
            for track, pose in self._tracked(poses).items():
                counter = self._tracks.get(track)
                if counter is None:
                    counter = self._tracks[track] = self._counter()
                counter._seen_ms = ts
                out = self._feed(counter, pose, ts)
                for item in out if isinstance(out, list) else [out] if out else []:
                    events += self._tagged(track, item.get('events') or ())
                    if item.get('distance') is not None:
                        distances.append(item['distance'])
            for track, counter in list(self._tracks.items()):      # finished tracks
                if ts - getattr(counter, '_seen_ms', ts) > TRACK_LOST_MS or len(self._tracks) > MAX_TRACKS and \
                        counter is min(self._tracks.values(), key=lambda item: getattr(item, '_seen_ms', 0)):
                    if counter._parts and counter._pending:
                        events += self._tagged(track, counter._decide(counter._last_ms, counter.combining(), final=True))
                    if counter._inside:
                        events += self._tagged(track, [counter._ended(counter._last_ms, cut='track_lost')])
                    del self._tracks[track]
            result = self._result(events, min(distances) if distances else None)
            return [*said, result] if said else result

    def finish(self):
        with self._lock:
            events = []
            for track, counter in self._tracks.items():
                if counter._parts and counter._pending:     # a rule's match still waiting for its example
                    events += self._tagged(track, counter._decide(counter._last_ms, counter.combining(), final=True))
                if counter._inside:
                    events += self._tagged(track, [counter._ended(counter._last_ms, cut='end_of_input')])
            return self._result(events) if events else None

    def reset(self):
        with self._lock:
            super().reset()
            self.bindings, self._binding_events, self._namespace = TrackBindings(), [], None
            self._first_seen, self._last_seen = {}, {}

    def status(self):
        counts = self.by_who()
        return dict(super().status(), count=counts.get(self.wanted, 0) if self.wanted else sum(counts.values()),
                    tracks=len(self._tracks), who=self.wanted, per_track=True, duplicates=self.duplicates)


def reads_bindings(options):
    """Whether an exemplar item reads ``vision.person_binding`` records (then it counts every track)."""
    return any(str(source).endswith(':' + BINDING) for source in options.get('consumes') or ())
