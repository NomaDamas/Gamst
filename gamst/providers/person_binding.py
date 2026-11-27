"""Who is each tracked person: track_id <-> enrolled person P over time, retroactive within a track (design section 2).

Owner decision 43: actions are counted for every track, and identity is bound afterwards, like words heard first and
attributed by voice. This module does the binding; the matchers (:class:`gamst.providers.track_exemplar.TrackExemplar`)
count every track and carry its ``track_id``; ``who`` is resolved from the bindings, also for events counted before the
person was confirmed (:func:`resolve`).

Pipeline module :class:`PersonBinding` consumes the camera and the pose module's joints of the same capture (every
visible person, with the tracker's ``track_id``). For each track it scores, against every enrolled person reference
(:mod:`gamst.person_reference`):

- the **face** (LVFace, the ``vision.face_identity`` role): ArcFace-aligned from the pose model's own face points
  (:func:`gamst.providers.person_match.face_reading`);
- the **whole-body appearance** (person re-ID, the ``vision.body_identity`` role): the pose box crop. Appearance is
  trusted within one video or session only (clothes change): the enrolled body samples are used only when they come from
  this run's own recording (``session``), and bodies of tracks the face confirmed in this run are added as it goes.
  A voice in the reference binds speakers, not tracks (the speech modules use it); it is not scored here.

Decisions (math only): a track is bound to P when its face scores at least ``face_threshold`` and beats every other
person by ``margin``; an unbound track is bound by body when its appearance scores at least ``body_threshold`` against
P's appearance of this session and no face contradicts it (default 0.8: on the shop clip, OSNet-AIN, the person's
face-confirmed tracks scored median 0.87, the best other track median 0.64 and 90th percentile 0.79; one clip, not
labelled, docs/measurements/person-identity-2026-11-27.md). A pose the tracker gave no track is never bound. A bound track is released when its face contradicts P (score
below ``face_reject``: the tracker may have switched people) or when it is lost (not seen for ``lost_ms``). One person is
on one track at a time: a face binding moves P off any other track. A person moves to another track only when that track's face
beats the holding track's face of the same frame by ``margin``. Scores are uncalibrated cosines; the face thresholds are
the same defaults as ``target_person``; none is measured on labelled people yet.

Retroactive: the first binding of a track starts at the track's first frame (``retroactive: true``): what that track did
before its face was seen was the same person. After a release, a new binding starts when it is decided.

Output, capability ``vision.person_binding`` (records, so count_events, get_timeline and :func:`resolve` read them):
``{tracks, bound, events}``; ``events`` are ``person.bound`` (``track_id, who, start_ms`` = from when, ``end_ms`` = when it
was decided, ``by: face | body | handover``, scores, ``retroactive``) and ``person.unbound`` (``track_id, who, start_ms,
end_ms, reason``). Embeddings and crops never leave this process.

L2 (gamst.pipeline): a fork names its tracks ``<track>@<first frame ms>`` (a forked pose tracker numbers its tracks
again) and takes over the bindings of the cut by box overlap (:meth:`snapshot`, :meth:`restore`); a track it cannot hand
over stays unknown until its own face confirms it, never a guess.
"""
from __future__ import annotations

from ..capabilities import CapabilitySpec, OutputField
from .person_match import (HANDOVER_IOU, MEDIAPIPE_FACE_POINTS, align_face, body_crop, box_iou, face_points,
                           face_reading)
from .speaker_verify import centroid, cosine, embed, l2_normalize

CAPABILITY = 'vision.person_binding'
UNKNOWN = 'unknown'


# ---------------------------------------------------------------- bindings as data (no model)

class TrackBindings:
    """track_id <-> who intervals. The first binding of a track reaches back to the track's first frame."""

    def __init__(self):
        self.tracks = {}       # track -> {first_ms, last_ms, who, from_ms, by, bound_before}
        self.closed = []       # finished intervals {track_id, who, from_ms, to_ms, by}

    def seen(self, track, ts):
        state = self.tracks.setdefault(track, dict(first_ms=ts, last_ms=ts, who=None, from_ms=None, by=None,
                                                   bound_before=False))
        state['last_ms'] = max(state['last_ms'], ts)
        return state

    def who(self, track):
        state = self.tracks.get(track)
        return state['who'] if state else None

    def holder(self, who):
        return next((track for track, state in self.tracks.items() if state['who'] == who), None)

    def bind(self, track, who, ts, by, **scores):
        """A ``person.bound`` event, or None when the track is already ``who``'s."""
        state = self.seen(track, ts)
        if state['who'] == who:
            return None
        events = []
        if state['who'] is not None:
            events.append(self.unbind(track, ts, 'another_person'))
        retroactive = not state['bound_before']
        start = state['first_ms'] if retroactive else ts
        state.update(who=who, from_ms=start, by=by, bound_before=True)
        event = dict(event='person.bound', track_id=track, who=who, start_ms=start, end_ms=ts, by=by,
                     retroactive=retroactive, **{key: round(value, 5) for key, value in scores.items() if value is not None})
        return [*events, event] if events else event

    def unbind(self, track, ts, reason):
        state = self.tracks.get(track)
        if state is None or state['who'] is None:
            return None
        self.closed.append(dict(track_id=track, who=state['who'], from_ms=state['from_ms'], to_ms=ts, by=state['by']))
        event = dict(event='person.unbound', track_id=track, who=state['who'], start_ms=state['from_ms'], end_ms=ts,
                     reason=reason)
        state.update(who=None, from_ms=None, by=None)
        return event

    def forget(self, track):
        """The track is lost: its binding ends at its last frame."""
        state = self.tracks.get(track)
        event = self.unbind(track, state['last_ms'], 'track_lost') if state else None
        self.tracks.pop(track, None)
        return event


def intervals_of(events):
    """``{track_id: [(from_ms, to_ms or None, who)]}`` from ``person.bound`` / ``person.unbound`` events (in order)."""
    found = {}
    for item in events:
        if not isinstance(item, dict) or not item.get('track_id'):
            continue
        spans = found.setdefault(item['track_id'], [])
        if item.get('event') == 'person.bound':
            if spans and spans[-1][1] is None:
                spans[-1] = (spans[-1][0], item['end_ms'], spans[-1][2])
            spans.append((item['start_ms'], None, item['who']))
        elif item.get('event') == 'person.unbound' and spans and spans[-1][1] is None and spans[-1][2] == item.get('who'):
            spans[-1] = (spans[-1][0], item['end_ms'], spans[-1][2])
    return found


def resolve(event, intervals):
    """Who did ``event`` (it carries ``track_id`` and a time): the binding of its track that covers its end (or
    start) time, else what it said itself, else :data:`UNKNOWN`. A binding decided later reaches back (retroactive)."""
    track = event.get('track_id') if isinstance(event, dict) else None
    at = event.get('end_ms', event.get('start_ms')) if isinstance(event, dict) else None
    if track is not None and type(at) in (int, float):
        for low, high, who in reversed(intervals.get(track, ())):
            if low <= at and (high is None or at <= high):
                return who
    own = event.get('who') if isinstance(event, dict) else None
    return own if isinstance(own, str) and own else UNKNOWN


def binding_events(rows):
    """The ``person.bound`` / ``person.unbound`` events of stored ``vision.person_binding`` records (oldest first)."""
    found = []
    for record in rows:
        value = (record.get('observation') or {}).get('value')
        for item in (value.get('events') if isinstance(value, dict) else None) or ():
            if isinstance(item, dict) and item.get('event') in ('person.bound', 'person.unbound'):
                found.append(item)
    return found


def seen_intervals(intervals, who, start_ms=None, end_ms=None):
    """``[{track_id, start_ms, end_ms}]``: when ``who`` was on screen (bound to a track), clipped to the window."""
    shown = []
    for track, spans in intervals.items():
        for low, high, person in spans:
            if person != who:
                continue
            high = end_ms if high is None else high if end_ms is None else min(high, end_ms)
            low = low if start_ms is None else max(low, start_ms)
            if high is None or high >= low:
                shown.append(dict(track_id=track, start_ms=low, end_ms=high))
    return sorted(shown, key=lambda item: item['start_ms'])


# ---------------------------------------------------------------- the module

class _Person:
    def __init__(self, name, face, body, session_size):
        from collections import deque
        self.name = name
        self.face = centroid(face) if face else None
        self.body = centroid(body) if body else None       # this session's enrolled appearance only
        self.session = deque(maxlen=session_size)          # bodies seen while the face confirmed the person

    def face_score(self, embedding):
        return None if self.face is None or embedding is None else cosine(embedding, self.face)

    def body_score(self, embedding):
        if embedding is None:
            return None
        scores = [cosine(embedding, self.body)] if self.body is not None else []
        if self.session:
            scores.append(cosine(embedding, centroid(list(self.session))))
        return max(scores) if scores else None


class PersonBinding:
    """Pipeline module: camera + every person's joints in, ``vision.person_binding`` records out."""

    name = 'person_binding'

    def __init__(self, face=None, body=None, *, references=('person',), session=None, face_threshold=.35,
                 face_reject=.15, body_threshold=.8, margin=.05, face_every_ms=200, confirm_every_ms=1000,
                 body_every_ms=500, lost_ms=5000, continue_ms=1000, continue_iou=.3, body_reject=.6, session_size=32,
                 max_images=32, face_indices=MEDIAPIPE_FACE_POINTS):
        """Without a face and a body model it binds nobody (no person can be enrolled into it either)."""
        if not face_reject <= face_threshold:
            raise ValueError('require face_reject <= face_threshold')
        import secrets
        # a live run is its own session: appearance seen in it (camera seconds) is trusted in it only
        session = session or f'live:{secrets.token_hex(6)}'
        self.face, self.body, self.reference_names, self.session = face, body, tuple(references), session
        self.face_threshold, self.face_reject, self.body_threshold, self.margin = face_threshold, face_reject, body_threshold, margin
        self.face_every_ms, self.confirm_every_ms = int(face_every_ms), int(confirm_every_ms)
        self.body_every_ms, self.lost_ms = int(body_every_ms), int(lost_ms)
        self.continue_ms, self.continue_iou, self.body_reject = int(continue_ms), float(continue_iou), float(body_reject)
        self.session_size, self.max_images, self.face_indices = session_size, int(max_images), tuple(face_indices)
        self._args = dict(references=self.reference_names, session=self.session, face_threshold=face_threshold,
                          face_reject=face_reject, body_threshold=body_threshold, margin=margin,
                          face_every_ms=face_every_ms, confirm_every_ms=confirm_every_ms, body_every_ms=body_every_ms,
                          lost_ms=lost_ms, continue_ms=continue_ms, continue_iou=continue_iou, body_reject=body_reject,
                          session_size=session_size, max_images=max_images, face_indices=self.face_indices)
        self.version = f"2.{getattr(face, 'model_id', 'noface')}.{getattr(body, 'model_id', 'nobody')}"
        self.spec = CapabilitySpec(
            CAPABILITY, CAPABILITY, 'video', 'gamst.person_binding', self.version,
            (OutputField('tracks', 'array'), OutputField('bound', 'object'), OutputField('events', 'array')),
            'frame', stateful=True, freshness_ms=1000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.persons, self._values = [], {}
        self.counts = dict(face_embeddings=0, body_embeddings=0, unpaired_images=0, frames=0, scored_frames=0)
        self.reset()

    # ------------------------------------------------------------------ references

    def bind(self, pipeline):
        """The run's person references that are given (none is allowed: a person can be enrolled while it runs)."""
        from ..person_reference import check, name_of, refusal
        found = {}
        for key in self.reference_names:
            reference = pipeline.references.get(key)
            if reference is None:
                continue
            if reference['kind'] != 'person':
                raise ValueError(f'person_binding reads person references; {key!r} is a {reference["kind"]} reference')
            verdict = check(reference['value'], session=self.session)
            if not verdict['ok']:
                raise ValueError(refusal(name_of(reference['value'], key), verdict['problems']))
            found[key] = reference['value']
        self.set_references(found)

    def set_references(self, values):
        """``{reference name: person reference value}``; embeddings of another face model are refused."""
        from ..person_reference import name_of
        persons = []
        for key, value in values.items():
            face, body = value.get('face') or [], value.get('body') or []
            if face and (self.face is None or value.get('face_model') != self.face.model_id):
                raise ValueError(f"person reference {key!r}: face embeddings come from {value.get('face_model')!r}; "
                                 f"this module runs {getattr(self.face, 'model_id', None)!r}")
            if body and (self.body is None or value.get('body_model') != self.body.model_id
                         or value.get('session') is None or value.get('session') != self.session):
                body = []      # another model, or clothes seen in another video: not trusted here
            persons.append(_Person(name_of(value, key), face, body, self.session_size))
        if persons and self.face is None and self.body is None:
            raise ValueError('person_binding has no face or body model set up: run gamst recipe setup')
        names = [person.name for person in persons]
        if len(set(names)) != len(names):
            raise ValueError(f"two person references have the same name: {', '.join(names)}")
        kept = {person.name: person.session for person in self.persons}
        for person in persons:          # a person added while the run goes on keeps the others' appearance glue
            person.session.extend(kept.get(person.name, ()))
        self.persons, self._values = persons, dict(values)

    # ------------------------------------------------------------------ L2 (gamst.pipeline)

    def fork(self):
        twin = type(self)(self.face, self.body, **self._args)
        twin._shares_models, twin._forked = True, True
        if self._values:
            twin.set_references(self._values)
        return twin

    def snapshot(self):
        """At a cut: the bound tracks' last boxes and who they are, and this session's appearance glue."""
        return dict(bound=[dict(bbox=self._boxes.get(track), who=state['who']) for track, state in self.bindings.tracks.items()
                           if state['who'] is not None and self._boxes.get(track)],
                    session={person.name: list(person.session) for person in self.persons})

    def restore(self, state):
        self._handover = [item for item in (state or {}).get('bound') or () if item.get('bbox')]
        for person in self.persons:
            for embedding in ((state or {}).get('session') or {}).get(person.name) or ():
                person.session.append(embedding)

    async def collect(self, seconds, *, every_ms=200):
        """Face and body embeddings of the one person in view during the next ``seconds`` of this run's frames
        (MCP ``enroll_person`` with camera seconds): ``{face, body, samples, skipped}``; a frame with nobody or with
        several people is skipped. Nothing else of the frames is kept."""
        import asyncio
        if self._collect is not None:
            raise ValueError('a person is being enrolled from the camera already')
        future = asyncio.get_running_loop().create_future()
        self._collect = dict(seconds=float(seconds), until=None, last=None, every=every_ms, face=[], body=[], samples=[],
                             skipped=0, future=future)
        try:
            return await asyncio.wait_for(asyncio.shield(future), seconds + 30)
        except asyncio.TimeoutError:
            raise ValueError(f'no camera frames reached the person binding within {seconds + 30:g} s') from None
        finally:
            self._collect = None

    async def _collected(self, candidates, image, ts):
        job = self._collect
        job['until'] = ts + 1000 * job['seconds'] if job['until'] is None else job['until']
        if ts > job['until']:
            if not job['future'].done():
                job['future'].set_result({key: job[key] for key in ('face', 'body', 'samples', 'skipped')})
            return
        if job['last'] is not None and ts - job['last'] < job['every']:
            return
        if len(candidates) != 1:
            job['skipped'] += 1
            return
        job['last'] = ts
        pose = next(iter(candidates.values()))[0]
        height, width = image.shape[:2]
        indices, visibility = face_reading(pose, self.face_indices)
        face = await self._embed(self.face, align_face(image, face_points(pose.get('landmarks') or [], width, height,
                                                                        indices, min_visibility=visibility)))
        body = await self._embed(self.body, body_crop(image, pose['bbox']))
        for key, value in (('face', face), ('body', body)):
            if value is not None:
                job[key].append(value)
        job['samples'].append(dict(source='camera', at=ts, persons=1, face=face is not None, body=body is not None))

    def reset(self):
        self._collect = getattr(self, '_collect', None)
        self.bindings, self._scores, self._poses, self._boxes = TrackBindings(), {}, {}, {}
        self._images, self._camera_ms, self._kept_ms = {}, None, None
        self._handover, self._namespace = [], None
        for person in self.persons:
            person.session.clear()

    async def close(self):
        if getattr(self, '_shares_models', False):
            return
        for embedder in (self.face, self.body):
            close = getattr(embedder, 'close', None)
            value = close() if callable(close) else None
            if hasattr(value, '__await__'):
                await value

    # ------------------------------------------------------------------ frames

    def _track(self, pose, ts, index):
        track = pose.get('track_id') or f'frame:{ts}:{index}'
        if getattr(self, '_forked', False):
            self._namespace = ts if self._namespace is None else self._namespace
            track = f'{track}@{self._namespace}'
        return track

    async def _embed(self, embedder, crop):
        if embedder is None or crop is None:
            return None
        try:
            return await embed(embedder, crop)
        except ValueError:      # too small or unclear for the model
            return None

    def _due(self, track, ts, kind):
        """Whether a track's face (or body) is looked at in this frame: often while it is unconfirmed, again only after
        ``confirm_every_ms`` once it is bound. Every track is looked at; nothing is looked at on every frame."""
        state = self._scores.get(track) or {}
        last = state.get(kind + '_ms')
        every = self.confirm_every_ms if self.bindings.who(track) is not None else \
            self.face_every_ms if kind == 'face' else self.body_every_ms
        return last is None or ts - last >= every

    async def _score(self, track, pose, image, ts):
        state = self._scores.setdefault(track, dict(face=None, face_ms=None, body=None, body_ms=None, body_embedding=None))
        if image is not None and self.persons:
            height, width = image.shape[:2]
            if self.face is not None and any(person.face is not None for person in self.persons) and self._due(track, ts, 'face'):
                indices, visibility = face_reading(pose, self.face_indices)
                crop = align_face(image, face_points(pose.get('landmarks') or [], width, height, indices,
                                                     min_visibility=visibility))
                state['face_ms'] = ts      # tried: a turned head is tried again at the same pace, not on the next frame
                embedding = await self._embed(self.face, crop)
                if embedding is not None:
                    self.counts['face_embeddings'] += 1
                    state['face'] = {person.name: person.face_score(embedding) for person in self.persons}
                    state['face_seen_ms'] = ts
            if self.body is not None and pose.get('bbox') and self._due(track, ts, 'body'):
                state['body_ms'] = ts
                embedding = await self._embed(self.body, body_crop(image, pose['bbox']))
                if embedding is not None:
                    self.counts['body_embeddings'] += 1
                    state['body_embedding'] = embedding
                    state['body'] = {person.name: person.body_score(embedding) for person in self.persons}
                    state['body_seen_ms'] = ts
        for key in ('face', 'body'):     # a score is evidence about now, not about a minute ago
            if state.get(key + '_seen_ms') is not None and ts - state[key + '_seen_ms'] > 1000:
                state[key] = None
                if key == 'body':
                    state['body_embedding'] = None
        return state

    def _best(self, scores, threshold):
        """(name, score) of the person above ``threshold`` who beats every other by ``margin``, else None."""
        ranked = sorted(((value, name) for name, value in (scores or {}).items() if value is not None), reverse=True)
        if not ranked or ranked[0][0] < threshold:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.margin:
            return None
        return ranked[0][1], ranked[0][0]

    def _fresh(self, scores, kind, ts):
        """The scores of this frame only (a decision is made on what was just seen)."""
        return (scores.get(kind) or {}) if scores.get(kind + '_seen_ms') == ts else {}

    def _decide(self, track, scores, ts, events, candidates):
        bindings = self.bindings
        if track.startswith('frame:'):      # the tracker gave this pose no track: nothing to bind it over time
            return
        current = bindings.who(track)
        face, body = self._fresh(scores, 'face', ts), self._fresh(scores, 'body', ts)
        if current is not None and face.get(current) is not None and face[current] < self.face_reject:
            events.append(bindings.unbind(track, ts, 'face_contradicts'))
            current = None
        chosen = self._best(face, self.face_threshold)
        if chosen is not None:
            name, score = chosen
            other = bindings.holder(name)
            if other is not None and other != track:
                held = ((candidates.get(other) or (None, {}))[1].get('face') or {}).get(name)
                if other in candidates and held is not None and held >= score - self.margin:
                    return      # the track that holds the person still shows their face as well: no flip
                events.append(bindings.unbind(other, ts, 'seen_on_another_track'))
            events.append(bindings.bind(track, name, ts, 'face', face_score=score, body_score=body.get(name)))
            self.bindings.tracks[track]['confirmed_ms'] = ts
            person = next(item for item in self.persons if item.name == name)
            if body and scores.get('body_embedding') is not None:
                person.session.append(l2_normalize(scores['body_embedding']))
            return
        if current is not None:
            if face.get(current) is not None and face[current] >= self.face_threshold:
                self.bindings.tracks[track]['confirmed_ms'] = ts
            return
        chosen = self._best(body, self.body_threshold)
        if chosen is not None:
            name, score = chosen
            contradicted = face.get(name) is not None and face[name] < self.face_reject
            if not contradicted and bindings.holder(name) is None:
                events.append(bindings.bind(track, name, ts, 'body', body_score=score, face_score=face.get(name)))

    def _continue(self, candidates, ts, events):
        """A tracker that starts a new track for the same person (a swap): when the track that holds a person has just
        left the frame (within ``continue_ms``) and a track that was never bound appears where it was (box overlap
        ``continue_iou``), the person goes over to it (``by: continuation``), unless its face or appearance of this
        frame says otherwise (below ``face_reject`` or ``body_reject``). The face confirms it later as for any track."""
        bindings = self.bindings
        for person in self.persons:
            holder = bindings.holder(person.name)
            if holder is None or holder in candidates:
                continue
            last = bindings.tracks[holder]
            if ts - last['last_ms'] > self.continue_ms or not self._boxes.get(holder):
                continue
            options = []
            for track, (pose, scores) in candidates.items():
                state = bindings.tracks.get(track) or {}
                if track.startswith('frame:') or state.get('who') is not None or state.get('bound_before'):
                    continue
                if state.get('first_ms', ts) < last['last_ms'] - self.continue_ms:
                    continue        # a track that was there all along is someone else
                face, body = self._fresh(scores, 'face', ts), self._fresh(scores, 'body', ts)
                if face.get(person.name) is not None and face[person.name] < self.face_reject or \
                        body.get(person.name) is not None and body[person.name] < self.body_reject:
                    continue
                options.append((box_iou(pose.get('bbox'), self._boxes[holder]), track))
            iou, track = max(options, default=(0., None))
            if track is not None and iou >= self.continue_iou:
                events.append(bindings.unbind(holder, ts, 'track_replaced'))
                events.append(bindings.bind(track, person.name, ts, 'continuation', iou=iou))

    def _hand_over(self, candidates, ts, events):
        """An L2 fork's first frame: each bound person of the cut goes to the track whose box overlaps theirs most."""
        for item in self._handover:
            best = max(((box_iou(pose.get('bbox'), item['bbox']), track) for track, (pose, _) in candidates.items()),
                       default=(0., None))
            if best[0] >= HANDOVER_IOU and self.bindings.who(best[1]) is None and self.bindings.holder(item['who']) is None:
                events.append(self.bindings.bind(best[1], item['who'], ts, 'handover', iou=best[0]))
        self._handover = []

    def _wants_image(self, ts):
        """Keep this camera frame until its joints come (this module in a lane of its own sees the frame first)? Only
        at the pace faces are looked at, and only while someone is enrolled or being enrolled."""
        if not (self.persons or self._collect is not None) or (self.face is None and self.body is None):
            return False
        every = min(self.face_every_ms, self.body_every_ms, (self._collect or {}).get('every', 10 ** 9))
        return self._kept_ms is None or ts - self._kept_ms >= every

    async def process(self, frame):
        """Joints and the camera frame of one capture, in either order (in the pose module's lane the joints come
        first; in a lane of its own the camera frame does): a frame is decided when both, or the joints of a camera
        frame that was not kept, have arrived."""
        from ..private_payload import poses_of
        found = poses_of(frame.data)
        ts = frame.timestamp_ms
        if found is not None:
            if ts in self._images:
                return await self._frame(ts, found, self._images.pop(ts))
            if self._camera_ms is not None and ts <= self._camera_ms:
                return await self._frame(ts, found, None)       # its camera frame went by unkept: decide on joints
            self._poses[ts] = found
            while len(self._poses) > 8:
                self._poses.pop(next(iter(self._poses)))
            return None
        if frame.modality != 'video' or not hasattr(frame.data, 'shape'):
            return None
        self._camera_ms = ts if self._camera_ms is None else max(self._camera_ms, ts)
        poses = self._poses.pop(ts, None)
        if poses is not None:
            return await self._frame(ts, poses, frame.data if self._wants_image(ts) or self._kept_ms is None else None,
                                     kept=True)
        if self._wants_image(ts):
            import numpy as np
            self._images[ts], self._kept_ms = np.array(frame.data, copy=True), ts   # the lane may reuse its buffer
            while len(self._images) > self.max_images:
                self._images.pop(next(iter(self._images)))
                self.counts['unpaired_images'] += 1
        return None

    async def _frame(self, ts, poses, image, kept=False):
        if image is not None:
            self._kept_ms = ts if kept else self._kept_ms
            for old in [key for key in self._images if key < ts]:       # their joints never came
                del self._images[old]
                self.counts['unpaired_images'] += 1
        self.counts['frames'] += 1
        self.counts['scored_frames'] += image is not None
        candidates, events = {}, []
        for index, pose in enumerate(poses):
            if not isinstance(pose, dict) or not pose.get('bbox'):
                continue
            track = self._track(pose, ts, index)
            self.bindings.seen(track, ts)
            candidates[track] = (pose, await self._score(track, pose, image, ts))
        for track in [key for key, state in self.bindings.tracks.items() if ts - state['last_ms'] > self.lost_ms]:
            events.append(self.bindings.forget(track))
            self._scores.pop(track, None)
            self._boxes.pop(track, None)
        if self._collect is not None and image is not None:
            await self._collected(candidates, image, ts)
        if self._handover:
            self._hand_over(candidates, ts, events)
        for track, (_, scores) in candidates.items():
            self._decide(track, scores, ts, events, candidates)
        self._continue(candidates, ts, events)
        for track, (pose, _) in candidates.items():
            self._boxes[track] = pose['bbox']
        flat = []
        for item in events:
            flat.extend(item if isinstance(item, list) else [item] if item else [])
        tracks = []
        for track, (_, scores) in candidates.items():
            who = self.bindings.who(track)
            entry = dict(track_id=track, who=who or UNKNOWN)
            for key in ('face', 'body'):
                values = self._fresh(scores, key, ts)
                value = values.get(who) if who else max((v for v in values.values() if v is not None), default=None)
                if value is not None:
                    entry[key + '_score'] = round(value, 5)
            tracks.append(entry)
        bound = {person.name: self.bindings.holder(person.name) for person in self.persons}
        return self.spec.validate(dict(tracks=tracks[:16], bound=bound, events=flat))

    def status(self):
        return dict(self.counts, persons=[person.name for person in self.persons], waiting_images=len(self._images),
                    tracks=len(self.bindings.tracks))
