"""Record what the per-track movement counter saw in one run, and replay it with another movement (measurement only).

Joint positions are never records (:mod:`gamst.private_payload`). For measuring rules and examples on the owner's
labelled clips the same frames have to be counted many times, so this tool, run by the owner on the owner's machine,
writes them to a local file the owner names (gzip JSON lines). Nothing in a recipe reads or writes such a file; MCP
cannot reach it.

  python -m gamst.eval.pose_dump record OUT.jsonl.gz -- [gamst recipe run person-action-count arguments]
      runs the recipe unchanged and keeps every frame its exemplar module was given: the poses of every tracked person
      (track, box, the 12 body joints and 42 hand joints with visibility and raw score) and the person binding's
      records, in the order they arrived; the run's own events are kept as well (``kind: "events"`` lines).

:func:`replay` feeds the frames to a fresh :class:`gamst.providers.track_exemplar.TrackExemplar` with any movement
value (examples, a rule, or both) and returns its events, so a rule or a threshold is measured on exactly the poses the
run had, by the code that counts in a run. A replay of the run's own movement gives the run's own events
(``tests/test_eval_multi_person.py`` checks this on a synthetic run; the measurement doc checks it on the clips).
"""
from __future__ import annotations

import gzip
import json
import sys

VERSION = 1
COLUMNS = ('x', 'y', 'visibility', 'raw_score')


def kept_indices():
    """{joint name: landmark index} of RTMW (COCO-WholeBody 133) that a rule or the channels can read."""
    from ..providers.pose_exemplar import COCO_WHOLEBODY_HANDS
    from ..providers.rtmw_pose import COCO_WHOLEBODY_BODY
    return {**COCO_WHOLEBODY_BODY, **COCO_WHOLEBODY_HANDS}


def _r(value, digits=4):
    return None if value is None else round(float(value), digits)


def compact_pose(pose, indices):
    """One pose as a JSON row: [track_id, bbox [x, y, w, h], image_w, image_h, keypoint_set, [[x, y, vis, raw]...]]."""
    marks = pose.get('landmarks') or []
    rows = []
    for index in indices:
        point = marks[index] if index < len(marks) else None
        rows.append(None if point is None else [_r(point.get('x')), _r(point.get('y')), _r(point.get('visibility', 1.), 3),
                                                _r(point.get('raw_score'), 3)])
    box = pose.get('bbox') or {}
    return [pose.get('track_id'), [_r(box.get(key)) for key in ('x', 'y', 'width', 'height')] if box else None,
            pose.get('image_width'), pose.get('image_height'), pose.get('keypoint_set'), rows]


def full_pose(row, indices):
    """A compact row as the pose dict the counter reads (landmarks at their own indices, others not visible)."""
    track, box, width, height, kind, rows = row
    size = max(indices) + 1
    marks = [dict(x=0., y=0., visibility=0.) for _ in range(size)]
    for index, point in zip(indices, rows):
        if point is not None:
            mark = dict(x=point[0], y=point[1], visibility=point[2])
            if point[3] is not None:
                mark['raw_score'] = point[3]
            marks[index] = mark
    pose = dict(landmarks=marks, keypoint_set=kind)
    if track:
        pose['track_id'] = track
    if box:
        pose['bbox'] = dict(x=box[0], y=box[1], width=box[2], height=box[3])
    if width and height:
        pose.update(image_width=width, image_height=height)
    return pose


class Recorder:
    """Patches :meth:`TrackExemplar.process` for one run; every frame it is given goes to ``out``."""

    def __init__(self, out):
        self.out, self.frames, self.bindings, self.events = out, 0, 0, []
        self.indices = sorted(set(kept_indices().values()))
        self.names = {index: name for name, index in kept_indices().items()}

    def write(self, item):
        self.out.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')

    def __enter__(self):
        from ..private_payload import poses_of
        from ..providers import track_exemplar
        self._original = original = track_exemplar.TrackExemplar.process
        recorder = self
        self.write(dict(kind='header', version=VERSION, columns=COLUMNS, indices=self.indices,
                        joints=[self.names[index] for index in self.indices]))

        def process(module, frame):
            value = frame.data if isinstance(frame.data, dict) else None
            if value is not None and isinstance(value.get('tracks'), list) and isinstance(value.get('bound'), dict):
                recorder.bindings += 1
                recorder.write(dict(kind='binding', ts=frame.timestamp_ms, events=value.get('events') or []))
            else:
                poses = poses_of(frame.data)
                if poses is not None:
                    recorder.frames += 1
                    recorder.write(dict(kind='poses', ts=frame.timestamp_ms, poses=[
                        compact_pose(pose, recorder.indices) for pose in poses if isinstance(pose, dict)]))
            result = original(module, frame)
            for item in result if isinstance(result, list) else [result] if result else []:
                for event in item.get('events') or ():
                    recorder.events.append(event)
            return result

        track_exemplar.TrackExemplar.process = process
        return self

    def __exit__(self, *exc):
        from ..providers import track_exemplar
        track_exemplar.TrackExemplar.process = self._original
        self.write(dict(kind='events', events=self.events))
        return False


def record(out_path, argv):
    """Run ``gamst recipe run`` with ``argv`` and write its exemplar frames to ``out_path``."""
    from .. import recipes
    with gzip.open(out_path, 'wt', encoding='utf-8') as out, Recorder(out) as recorder:
        recipes.main(['run', *argv])
    print(json.dumps(dict(out=str(out_path), frames=recorder.frames, binding_records=recorder.bindings,
                          events=len(recorder.events))), file=sys.stderr, flush=True)


def load(path):
    """``{header, frames: [(kind, ts, payload)], events}`` of a recorded file."""
    header, frames, events = None, [], []
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        for line in source:
            item = json.loads(line)
            if item['kind'] == 'header':
                header = item
            elif item['kind'] == 'poses':
                frames.append(('poses', item['ts'], item['poses']))
            elif item['kind'] == 'binding':
                frames.append(('binding', item['ts'], item['events']))
            elif item['kind'] == 'events':
                events = item['events']
    if header is None or header.get('version') != VERSION:
        raise ValueError(f'{path}: not a pose dump of version {VERSION}')
    return dict(header=header, frames=frames, events=events)


def binding_events(dump):
    """Every person binding event of the run, in order."""
    return [event for kind, _, payload in dump['frames'] if kind == 'binding' for event in payload]


def counter_for(value, *, label='movement', min_hold_ms=200):
    """A per-track counter of ``value`` (a movement file's content), as the recipe builds it for RTMW poses."""
    from ..pipeline import reference_id
    from ..providers.rtmw_pose import COCO_WHOLEBODY_BODY, MIN_VISIBILITY
    from ..providers.track_exemplar import TrackExemplar
    counter = TrackExemplar(reference='exemplar', mapping=COCO_WHOLEBODY_BODY, min_visibility=MIN_VISIBILITY,
                            min_hold_ms=min_hold_ms)
    reference = dict(kind='exemplar', value=value, label=value.get('label') or label, id=reference_id(value), revision=1,
                     from_ms=0)
    counter._adopt(counter.prepare(reference), 0)
    return counter


def replay(dump, value, *, label='movement', min_hold_ms=200, until_ms=None):
    """Every event (``exemplar.matched`` and ``exemplar.ended``) the counter gives for ``value`` on the recorded frames."""
    from ..plugins import InputFrame
    from ..private_payload import PrivatePoses
    counter = counter_for(value, label=label, min_hold_ms=min_hold_ms)
    indices, events = dump['header']['indices'], []
    for kind, ts, payload in dump['frames']:
        if until_ms is not None and ts > until_ms:
            break
        if kind == 'binding':
            data = dict(tracks=[], bound={}, events=payload)
        else:
            data = PrivatePoses([full_pose(row, indices) for row in payload])
        out = counter.process(InputFrame(data, 'video', ts))
        for item in out if isinstance(out, list) else [out] if out else []:
            events += item.get('events') or ()
    out = counter.finish()
    if out:
        events += out.get('events') or ()
    return events


def track_series(dump, track_ids=None):
    """{track: [(ts, pose dict)]} of the recorded frames (for templates and measures of one person's frames)."""
    indices, found = dump['header']['indices'], {}
    for kind, ts, payload in dump['frames']:
        if kind != 'poses':
            continue
        for row in payload:
            if row[0] and (track_ids is None or row[0] in track_ids):
                found.setdefault(row[0], []).append((ts, full_pose(row, indices)))
    return found


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != 'record' or '--' not in argv:
        raise SystemExit('usage: python -m gamst.eval.pose_dump record OUT.jsonl.gz -- RECIPE-RUN-ARGUMENTS')
    split = argv.index('--')
    record(argv[1], argv[split + 1:])


if __name__ == '__main__':
    main()


def clip_samples(dump, track, start_ms, end_ms):
    """[(ts, channels, mask, points)] of one track between two capture times: the channels and joints the per-track counter
    reads (RTMW poses: :func:`gamst.providers.pose_exemplar.largest_pose` of that one pose, joints with the hands)."""
    from ..providers.pose_exemplar import body_points, keypoint_reading, largest_pose, raw_score_gates
    from ..providers.rtmw_pose import COCO_WHOLEBODY_BODY, MIN_VISIBILITY
    found = []
    for ts, pose in track_series(dump, {track}).get(track, ()):
        if not start_ms <= ts <= end_ms:
            continue
        best = largest_pose([pose], COCO_WHOLEBODY_BODY, MIN_VISIBILITY)
        if best is None:
            continue
        channels, mask, _, aspect = best
        own, visibility = keypoint_reading(pose, COCO_WHOLEBODY_BODY, MIN_VISIBILITY, hands=True)
        points = body_points(pose['landmarks'], own, min_visibility=visibility, aspect=1. if aspect is None else aspect,
                             raw_gates=raw_score_gates(pose, own))
        found.append((ts, [round(value, 5) for value in channels], mask, points))
    return found


def movement_value(positives, negatives=(), *, rule=None, mode='sequence', calibrated=True, combine=None, label='movement',
                   enter0=.12, exit0=.18, min_hold_ms=200):
    """The movement file ``gamst recipe enroll`` writes, from clips given as :func:`clip_samples` lists: examples only, or
    a rule with examples (``rule``; its numbers calibrated from the clips' joints unless ``calibrated`` is False).
    The same functions as enrollment (template, derive, calibrate), so a measured value is what enroll would save for
    the same frames."""
    from ..providers.movement_match import derive, example
    from ..providers.movement_rule import calibrate, validate_rule
    from ..recipes.movement import template
    entries = dict(positives=[], negatives=[])
    for key, clips in (('positives', positives), ('negatives', negatives)):
        for samples in clips:
            plain = [item[:3] for item in samples]
            saved = template(mode, plain)
            entries[key].append(dict(saved=dict(saved, frames=len(plain), method='examples', confirmed=True),
                                     template=example(saved['value'], span_ms=saved.get('span_ms')), clip=plain))
    thresholds = derive(mode, entries['positives'], entries['negatives'], enter0=enter0, exit0=exit0,
                        min_hold_ms=min_hold_ms)
    from ..providers.pose_exemplar import CHANNELS_VERSION
    value = dict(kind='exemplar', format=2, label=label, mode=mode, channels=f'vision.pose_channels@{CHANNELS_VERSION}.rtmw',
                 method='examples', confirmed=True, **thresholds,
                 positives=[entry['saved'] for entry in entries['positives']],
                 negatives=[entry['saved'] for entry in entries['negatives']])
    if rule is None:
        return value
    written = validate_rule(rule)
    used, evidence = written, None
    if calibrated:
        used, evidence = calibrate(written, [[item for item in clip if item[3]] for clip in positives],
                                   [[item for item in clip if item[3]] for clip in negatives])
    return dict(value, mode=f'rule+{mode}', examples_mode=mode, method='rule+examples', rule=used, rule_text=written,
                rule_source='text+clips' if evidence else 'text', **({'calibration': evidence} if evidence else {}),
                **({'combine': combine} if combine else {}))
