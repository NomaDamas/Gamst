"""Replay the person binding on a run's recorded poses with face and body evidence computed once (measurement only).

The binding's decisions (thresholds, which face crop, scene rules) have to be measured many times on the owner's
labelled clips, and running the pose model every time is slow. A pose dump (:mod:`gamst.eval.pose_dump`) has every
track's box and joints; an evidence table has, for some frames, the embeddings the face and body models gave for each
track (computed by the owner's tool on the owner's machine, never stored in the repository). :func:`replay` feeds the
dump to the real :class:`gamst.providers.person_binding.PersonBinding` with that evidence in place of its models:

- a track's crops are "taken" only where the binding asks for them (its own ``face_every_ms`` / ``confirm_every_ms``
  pace) and the table has that frame (the nearest table frame at or before it, at most ``slack_ms`` earlier; the
  evidence is then of that frame's time);
- a result is decided on at the first frame at least ``delay_ms`` after its frame (the run's scorer answers beside the
  lane, 100 ms median on the shop clip);
- scene cuts are the run's own ``scene.cut`` events, applied at the frame that decided them.

So the binding code under test is the code that runs, and only the models' timing is simulated. The replay of the
run's own settings is compared with the run in the measurement doc.
"""
from __future__ import annotations

import asyncio


class CachedEmbedder:
    """Stands in for a model: its "crop" is already the embedding (None: the model found nothing usable)."""

    def __init__(self, model_id):
        self.model_id = model_id

    def embed(self, crop):
        if crop is None:
            raise ValueError('no evidence')
        return crop


def _binding_class():
    from ..providers.person_binding import PersonBinding

    class ReplayBinding(PersonBinding):
        """:class:`PersonBinding` whose crops come from an evidence table and whose scorer is a delay queue."""

        def setup_replay(self, evidence, *, delay_ms=100, slack_ms=250):
            self.evidence, self.delay_ms, self.slack_ms = evidence, int(delay_ms), int(slack_ms)
            self._queue, self._now = [], None
            self._times = {}
            for ts, track in evidence:
                self._times.setdefault(track, []).append(ts)
            for times in self._times.values():
                times.sort()

        def _lookup(self, track, ts):
            import bisect
            times = self._times.get(track)
            if not times:
                return None
            index = bisect.bisect_right(times, ts) - 1
            if index < 0 or ts - times[index] > self.slack_ms:
                return None
            return times[index]

        def _crops(self, pose, image, face, body):
            at = self._lookup(pose.get('track_id'), self._now)
            if at is None:
                return [None, None]
            found = self.evidence[(at, pose.get('track_id'))]
            self._taken = at
            return [found[0] if face else None, found[1] if body else None]

        def _submit(self, key, ts, face, body):
            self.counts['jobs'] = self.counts.get('jobs', 0) + 1
            at = getattr(self, '_taken', None)
            at = ts if at is None else at      # the frame the evidence is of
            self._taken = None
            self._queue.append((at + self.delay_ms, key, at, face, body))

        def deliver_due(self, now):
            ready = [item for item in self._queue if item[0] <= now]
            self._queue = [item for item in self._queue if item[0] > now]
            for _, key, ts, face, body in ready:
                self._deliver(key, ts, face, body)

    return ReplayBinding


def with_bindings(dump, frames):
    """A copy of ``dump`` whose person binding records are ``frames`` ([(ts, events)], e.g. a replay's
    ``replay_frames``), each placed after the poses of its frame, so :func:`gamst.eval.pose_dump.replay` counts with them."""
    at = {}
    for ts, events in frames:
        at.setdefault(ts, []).extend(events)
    out = []
    for kind, ts, payload in dump['frames']:
        if kind == 'binding':
            continue
        out.append((kind, ts, payload))
        if kind == 'poses' and ts in at:
            out.append(('binding', ts, at.pop(ts)))
    out += [('binding', ts, events) for ts, events in sorted(at.items())]
    return dict(dump, frames=out)


def replay(dump, evidence, references, *, face_model, body_model=None, delay_ms=100, slack_ms=250, binding_class=None,
           **options):
    """Every event of a :class:`PersonBinding` (``options``: its keyword arguments) run over the dump's frames.

    ``evidence``: ``{(ts, track_id): (face embedding or None, body embedding or None)}``; ``references``:
    ``{name: person reference value}`` (their ``face_model`` must be ``face_model``); ``binding_class``: a subclass of
    the replay class to measure a change."""
    from .pose_dump import full_pose
    from ..providers.person_binding import PersonBinding
    cls = binding_class or _binding_class()
    face = CachedEmbedder(face_model)
    body = CachedEmbedder(body_model) if body_model else None
    module = cls(face, body, **options)
    module.setup_replay(evidence, delay_ms=delay_ms, slack_ms=slack_ms)
    module.replay_frames = []       # [(ts, the binding events of that frame)], for a pose dump with these bindings
    module.set_references(references)
    indices = dump['header']['indices']
    cuts = sorted(((item['end_ms'], item['start_ms']) for kind, _, payload in dump['frames'] if kind == 'binding'
                   for item in payload if item.get('event') == 'scene.cut'), reverse=True)
    events = []

    async def run():
        for kind, ts, payload in dump['frames']:
            if kind != 'poses':
                continue
            while cuts and cuts[-1][0] <= ts:
                module._pending_cuts.append(cuts.pop()[1])
            module._now = ts
            module.deliver_due(ts)
            poses = [full_pose(row, indices) for row in payload]
            out = await module._frame(ts, poses, True if any(row[0] for row in payload) else None)
            events.extend((out or {}).get('events') or ())
            module.replay_frames.append((ts, (out or {}).get('events') or []))
        out = module.finish()
        events.extend((out or {}).get('events') or ())
        if out:
            module.replay_frames.append((module.replay_frames[-1][0] if module.replay_frames else 0, out.get('events') or []))
    asyncio.run(run())
    assert isinstance(module, PersonBinding)
    return events, module
