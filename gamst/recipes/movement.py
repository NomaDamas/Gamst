"""Specify a movement, find it in recordings, and turn its events into data (``gamst recipe`` actions).

- ``enroll``: example clips (``--video FILE[@start-end]``, ``--negative ...``), confirmed proposals of a text,
  a declarative rule, or a rule with example clips (owner decision 58: the text's rule is the skeleton, the clips set
  its numbers and are compared too) become one movement file. It holds channel vectors (and/or the rule), thresholds
  and where they come from; never frames, joints or audio.
- ``propose`` / ``proposals``: a VLM looks for a text in a recording; the owner confirms or rejects each candidate.
- ``find``: the recipe's own modules over recordings; matched intervals as rows.
- ``export``: the events of a store as JSONL or CSV.

Every movement file runs in the same counter, :class:`gamst.providers.pose_exemplar.ExemplarEvents`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

MAX_CLIP_S = 30.
MAX_SEQUENCE_ITEMS = 30
SOURCES = ('camera', 'video', 'take', 'audio', 'microphone', 'iphone', 'sensor')
PROPOSALS = 'gamst-movement-proposals.json'
FEATURES = ('vision.pose_channels', 'audio.voice_channels')
FEATURES_OF_POSES = ('pose-channels',)
FEATURE_PARTS = ('mediapipe-pose', 'mediapipe-graph', 'sapiens2-pose', 'rtmw-pose', 'pose-channels', 'prosody', 'voice-channels')


@dataclass
class Clip:
    """One demonstration: a file (or a live source) and the part of it that is the movement."""
    kind: str
    path: object
    start: float | None = None
    end: float | None = None
    method: str = 'examples'
    confirmed: bool = True
    proposal: str | None = None
    at: tuple | None = None         # the person to learn from: the one at this point (x, y in 0..1 of the frame)

    def name(self):
        place = Path(self.path).name if isinstance(self.path, str) else self.kind
        place = place if self.start is None else f'{place}@{self.start:g}-{self.end:g}'
        return place if self.at is None else f'{place}@{self.at[0]:g},{self.at[1]:g}'

    def footage(self):
        path = str(Path(self.path).expanduser().resolve()) if isinstance(self.path, str) else self.path
        return (self.kind, path), self.start or 0., float('inf') if self.end is None else self.end


def parse_at(text):
    """``X,Y`` in 0..1 of the frame -> (x, y)."""
    found = re.fullmatch(r'(\d*\.?\d+),(\d*\.?\d+)', str(text).strip())
    if found is None or not all(0 <= float(value) <= 1 for value in found.groups()):
        raise SystemExit(f'{text}: a point is X,Y in 0..1 of the frame (the person to learn from)')
    return float(found[1]), float(found[2])


def parse_clip(kind, text):
    """``FILE``, ``FILE@start-end`` in seconds, or ``FILE@start-end@x,y`` (learn from the person at that point, x and y
    in 0..1 of the frame; the track nearest the click is followed). A file whose name looks like that wins."""
    found = re.fullmatch(r'(.+)@(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?:@(\d*\.?\d+,\d*\.?\d+))?', text)
    if found is None or Path(text).is_file():
        return Clip(kind, text)
    start, end = float(found[2]), float(found[3])
    if not 0 < end - start <= MAX_CLIP_S:
        raise SystemExit(f'{text}: an interval is start < end, at most {MAX_CLIP_S:g} s')
    return Clip(kind, found[1], start, end, at=parse_at(found[4]) if found[4] else None)


def _exemplar_item(recipe):
    item = next((item for item in recipe['modules'] if item['use'] == 'exemplar'), None)
    if item is None:
        raise SystemExit(f"{recipe['name']} has no exemplar module to enroll for")
    return item


def feature_modules(recipe):
    """The modules that turn the recipe's input into the channels its exemplar module reads.

    The demonstration itself is the only person or voice in it, so identity modules are not used here.
    """
    from . import choose, inputs_of, upstream
    wanted = _exemplar_item(recipe)['name']
    chain = upstream(recipe['modules'], wanted)
    features = {item['name'] for item in chain if item['use'] in FEATURE_PARTS}
    plain = choose(dict(recipe, modules=chain), exclude=[
        item['name'] for item in chain if item.get('optional') and item['name'] not in features | {wanted}])
    sound = next((item['id'] for item in inputs_of(recipe, chain) if item['use'] == 'microphone'), None)
    found = [dict(item, consumes=[sound]) if item['use'] == 'prosody' else item
             for item in plain['modules'] if item['name'] in features]
    if not any(item['use'] in FEATURES_OF_POSES for item in found):     # a per-track counter reads the joints itself:
        joints = {name for item in found for name in (item.get('joints'), *item.get('emits', ())) if name}
        found += [item for item in recipe['modules'] if item['use'] in FEATURES_OF_POSES  # its channels, as it computes them
                  and item.get('consumes') and set(item['consumes']) <= joints][:1]
    return found


def exemplar_input(recipe):
    """The input the recipe's exemplar module matches on, through the modules between them."""
    from . import inputs_of, upstream
    item = _exemplar_item(recipe)
    direct = [entry for entry in inputs_of(recipe, [item]) if entry['use'] == 'sensor']
    return (direct or inputs_of(recipe, feature_modules(recipe)) or inputs_of(recipe, upstream(recipe['modules'], item['name'])))[0]


class _JointsTap:
    """A module of a demonstration only: per frame, the joints of the largest person as a rule reads them (for
    :func:`gamst.providers.movement_rule.calibrate`) and, with ``everyone``, every tracked person's channels, joints and
    box (to learn from a chosen person, :func:`chosen_person`). In memory only; it commits nothing and nothing of it is
    saved."""

    def __init__(self, mapping, min_visibility, everyone=False):
        self.mapping, self.min_visibility, self.everyone = mapping, min_visibility, everyone
        self.points, self.people = {}, {}

    def _read(self, pose):
        from ..providers.pose_exemplar import body_points, keypoint_reading, largest_pose, raw_score_gates
        best = largest_pose([pose], self.mapping, self.min_visibility)
        if best is None:
            return None
        channels, mask, _, aspect = best
        own, visibility = keypoint_reading(pose, self.mapping, self.min_visibility, hands=True)
        return [round(value, 5) for value in channels], mask, body_points(
            pose['landmarks'], own, min_visibility=visibility, aspect=1. if aspect is None else aspect,
            raw_gates=raw_score_gates(pose, own))

    def process(self, frame):
        from ..private_payload import poses_of
        from ..providers.pose_exemplar import largest_pose
        poses = [pose for pose in poses_of(frame.data) or () if isinstance(pose, dict) and pose.get('landmarks')]
        best = largest_pose(poses, self.mapping, self.min_visibility) if poses else None
        if best is not None:
            read = self._read(poses[best[2]])
            if read is not None:
                self.points[frame.timestamp_ms] = read[2]
        if self.everyone:
            row = []
            for pose in poses:
                read = self._read(pose)
                if read is not None:
                    row.append((pose.get('track_id'), pose.get('bbox'), *read))
            self.people[frame.timestamp_ms] = row
        return None


def chosen_person(people, *, at=None, who=None, bindings=()):
    """{timestamp_ms: (channels, mask, points)} of one person in a demonstration's frames (``people``: what
    :class:`_JointsTap` kept with ``everyone``).

    ``at`` (x, y): the person whose box holds the point (else the nearest box centre) in the first frame with anyone,
    then that track (the label tool's ``enroll_clicked.py``). ``who``: in each frame, the track the person binding bound
    to that name (bindings resolved retroactively over the clip)."""
    from ..providers.person_binding import intervals_of, resolve
    chosen, track = {}, None
    intervals = intervals_of(list(bindings)) if who else None
    for ts in sorted(people):
        row = people[ts]
        if who:
            for item in row:
                if item[0] and resolve(dict(track_id=item[0], end_ms=ts), intervals) == who:
                    chosen[ts] = item[2:]
                    break
            continue
        if track is None:
            x, y = at
            best = None
            for item in row:
                box = item[1]
                if not box:
                    continue
                inside = box['x'] <= x <= box['x'] + box['width'] and box['y'] <= y <= box['y'] + box['height']
                gap = ((box['x'] + box['width'] / 2 - x) ** 2 + (box['y'] + box['height'] / 2 - y) ** 2) ** .5 - inside
                if best is None or gap < best[0]:
                    best = (gap, item)
            if best is None:
                continue
            track = best[1][0] or ''
            chosen[ts] = best[1][2:]
            continue
        found = next((item for item in row if item[0] == track), None) if track else None
        if found is not None:
            chosen[ts] = found[2:]
    return chosen


def _joints_input(modules):
    """(name of the pose module's joints input, its mapping options) among the demonstration's modules, or None."""
    from .parts import _pose_source
    for item in modules:
        if item['use'] in ('mediapipe-pose', 'mediapipe-graph', 'sapiens2-pose', 'rtmw-pose') and item.get('joints'):
            source = {'rtmw-pose': 'rtmw', 'sapiens2-pose': 'sapiens2'}.get(item['use'])
            return item['joints'], _pose_source(dict(source=source))
    return None


async def demonstration(recipe, args, clip, *, joints=False):
    """([(timestamp_ms, channels, mask), ...], channel contract) of one clip; frames and joints are not kept.

    ``joints``: each sample also carries the person's joints of that frame (a fourth item, ``{joint: (x, y)}`` or
    None), held in memory for calibrating a rule and then dropped.

    The person learnt from is the largest visible one, unless the clip names a point (``clip.at``, ``FILE@a-b@x,y``):
    then the one there; or ``--from-who NAME`` (with ``--person``): then the track the person binding names so
    (:func:`chosen_person`)."""
    from .. import Gamst
    from ..inputs import open_source
    from . import build_pipeline
    samples, seen, place, modules = [], {}, exemplar_input(recipe), feature_modules(recipe)
    options = {}
    if clip.kind == 'video':
        if getattr(args, 'capture_fps', None):
            options['capture_fps'] = args.capture_fps
        if clip.start is not None:
            options.update(start_s=max(0., clip.start - 1), end_s=clip.end)  # a second for the tracker to settle
    if clip.kind == 'sensor':
        options = {key: place[key] for key in ('columns', 'time', 'modality') if key in place}
    sound = clip.kind == 'video' and place['use'] == 'microphone'
    source = open_source('sound' if sound else clip.kind, clip.path if clip.path is not True else None, **options)
    run = argparse.Namespace(**dict(vars(args), offset_ms=None, input=None, who=None,
                                    **{name: None for name in SOURCES}))
    setattr(run, clip.kind, clip.path)
    start, end = clip.start or 0., clip.end
    from_who = getattr(args, 'from_who', None)
    choose = clip.at is not None or bool(from_who)
    references = []
    if from_who:
        if not getattr(args, 'person', None):
            raise SystemExit(f'--from-who {from_who} learns from the person the binding names so: give --person P.json '
                             '(an enrolled person, photos)')
        binding = next((item for item in recipe['modules'] if item['use'] == 'person-binding'), None)
        if binding is None:
            raise SystemExit(f"--from-who: {recipe['name']} has no person-binding module")
        modules = [*modules, dict(binding, optional=False)]
        references = [item for item in recipe.get('references', ()) if item['kind'] == 'person']
    bindings = []
    async with Gamst(consent=True) as human:
        if place['use'] == 'sensor':    # its samples are the channels: no module between
            from ..pipeline import Pipeline
            pipeline = Pipeline(human, 'demonstration').add_input(place['id'], source)
        else:
            pipeline = await build_pipeline(dict(recipe, references=references), run, human,
                                            sources={place['id']: source}, modules=modules, lanes=False)
        tap = None
        if (joints or choose) and place['use'] != 'sensor' and _joints_input(modules):
            name, options = _joints_input(modules)
            from ..providers.pose_exemplar import MEDIAPIPE_BODY
            tap = _JointsTap(options.get('mapping') or MEDIAPIPE_BODY, options.get('min_visibility', .5), everyone=choose)
            pipeline.add_module('demonstration_joints', tap, consumes=[name])
        elif choose:
            raise SystemExit(f'{clip.name()}: choosing a person needs a pose model with tracks')

        async def on_frame(input_id, frame, records):
            origin = getattr(source, 'origin_ms', None)
            seen.setdefault('ms', frame.timestamp_ms if origin is None else origin)
            # A block of sound begins before its time, which is that of its last sample.
            block = 1000 * len(frame.data) / frame.sample_rate if frame.sample_rate and origin is not None else 0
            seconds = (frame.timestamp_ms - block - seen['ms']) / 1000
            if end is not None and seconds > end:
                return False
            if end is None and seconds > MAX_CLIP_S:
                seen['too_long'] = True
                return False
            if seconds >= start and place['use'] == 'sensor':
                samples.append((frame.timestamp_ms, frame.data['channels'], frame.data['mask']))
                seen['channels'] = source.contract
            elif seconds >= start or from_who:
                for record in records:
                    capability = (record['observation'].get('capability') or {}).get('id')
                    value = record['observation'].get('value')
                    if capability == 'vision.person_binding' and isinstance(value, dict):
                        bindings.extend(item for item in value.get('events') or () if isinstance(item, dict))
                    if seconds < start:
                        continue
                    if capability in FEATURES:
                        value = record['observation']['value']
                        samples.append((record['timestamp_ms'], value['channels'], value['mask']))
                        seen['channels'] = f"{capability}@{record['models'][0]['version']}"
            return True

        try:
            await pipeline.run(on_frame=on_frame, stop_on_failure=True)
        finally:
            await pipeline.close()
    if 'too_long' in seen:
        raise SystemExit(f'{clip.name()} is longer than {MAX_CLIP_S:g} s: one movement per clip; trim it or name the '
                         'part, FILE@start-end')
    if not samples:
        raise SystemExit(f'nothing to learn in {clip.name()} (no visible person, no audio, or no samples)')
    if choose:
        people = chosen_person({ts: row for ts, row in tap.people.items() if ts >= samples[0][0]}, at=clip.at,
                               who=from_who, bindings=bindings)
        if len(people) < 2:
            missing = 'nobody at that point' if clip.at else f'{from_who!r} is not bound to anyone'
            raise SystemExit(f'{clip.name()}: {missing} in this clip; give another point or clip')
        samples = [(ts, channels, mask, points) for ts, (channels, mask, points) in sorted(people.items())]
        return (samples if joints else [item[:3] for item in samples]), seen['channels']
    if joints:
        found = tap.points if tap is not None else {}
        samples = [(*item, found.get(item[0])) for item in samples]
    return samples, seen['channels']


def template(mode, samples):
    """The saved value of one example: the median pose of a hold, or the movement as at most 30 vectors."""
    if mode == 'hold':
        size = len(samples[0][1])
        mask = [sum(item[2][i] for item in samples) * 2 > len(samples) for i in range(size)]
        channels = []
        for i in range(size):
            values = sorted(item[1][i] for item in samples if item[2][i])  # item: (ts, channels, mask[, joints])
            channels.append(values[len(values) // 2] if values and mask[i] else 0.)
        return dict(value=dict(channels=channels, mask=mask))
    span = samples[-1][0] - samples[0][0]
    if len(samples) < 2 or span <= 0:
        raise ValueError('a movement needs at least two samples; use --mode hold for a pose')
    step = -(-len(samples) // MAX_SEQUENCE_ITEMS)
    return dict(value=[dict(channels=item[1], mask=item[2]) for item in samples[::step]], span_ms=span)


def _refuse_repeats(positives, negatives):
    """The same footage twice is not two examples, and cannot be both a positive and a negative."""
    clips = [(clip, False) for clip in positives] + [(clip, True) for clip in negatives]
    for index, (clip, negative) in enumerate(clips):
        place, start, end = clip.footage()
        for other, other_negative in clips[:index]:
            other_place, other_start, other_end = other.footage()
            if place != other_place or place[1] in (None, True) or type(place[1]) is int:
                continue
            if negative != other_negative and min(end, other_end) > max(start, other_start):
                raise SystemExit(f'{clip.name()} and {other.name()} overlap: one is an example, the other a negative')
            if negative == other_negative and (start, end) == (other_start, other_end):
                raise SystemExit(f'{clip.name()} is given twice')


def _positives(recipe, args):
    """Clips of the command line: files with optional intervals, or the one live source."""
    from . import _given, _source
    if not any(name != 'iphone' for name in _given(args)):
        return []
    _, kind = _source(recipe, args, [exemplar_input(recipe)])
    values = getattr(args, kind, None)
    values = values if isinstance(values, list) else [values]
    clips = [parse_clip(kind, value) if kind in ('video', 'audio', 'sensor') else Clip(kind, value) for value in values]
    window = args.start is not None or args.seconds is not None
    for clip in clips:
        if clip.start is None and (window or kind in ('camera', 'microphone', 'take')):
            clip.start = args.start or 0.
            clip.end = clip.start + (1.5 if args.seconds is None else args.seconds)
            if clip.end - clip.start > MAX_CLIP_S:
                raise SystemExit(f'one movement per clip, at most {MAX_CLIP_S:g} s')
    return clips


def _proposed(recipe, args):
    """(positive clips, negative clips, rule) of the owner's decisions in a proposals file."""
    from ..movement_proposals import Proposals, video_identity
    store = Proposals(args.proposals)
    try:
        chosen, videos = store.selected(text=args.text, ids=args.proposal)
    except ValueError as error:
        raise SystemExit(f'{args.proposals}: {error}') from error
    chosen = [item for item in chosen if item['recipe'] == recipe['name'] and item['kind'] != 'term']
    wanted = ('open', 'confirmed') if args.no_confirm else ('confirmed',)
    waiting = [item for item in chosen if item['status'] == 'open' and 'open' not in wanted]
    if waiting:
        print(f'{len(waiting)} open proposals are not used; confirm or reject them: gamst recipe proposals list '
              f'--proposals {args.proposals}', file=sys.stderr)
    positives, negatives, rules, checked = [], [], [], set()
    for item in chosen:
        if item['kind'] == 'rule':
            if item['status'] in wanted:
                rules.append(item)
            continue
        if item['status'] not in (*wanted, 'rejected'):
            continue
        video = videos[item['video']]
        if item['video'] not in checked:
            if not Path(video['path']).is_file() or video_identity(video['path'])['sha256'] != item['video']:
                raise SystemExit(f"{video['name']} is not the file the proposals were made from")
            checked.add(item['video'])
        clip = Clip('video', video['path'], item['start_s'], item['end_s'], 'text_finder',
                    item['status'] != 'open', item['id'])
        (negatives if item['status'] == 'rejected' else positives).append(clip)
    if len(rules) > 1 or (rules and positives):
        raise SystemExit('more than one way to specify the movement: choose with --proposal ID')
    if not rules and not positives:
        raise SystemExit('no confirmed proposal to enroll from (gamst recipe proposals confirm ID)')
    return positives, negatives, rules[0] if rules else None


def _write(args, saved):
    from .enrollment import _save
    content = {key: saved[key] for key in ('mode', 'channels', 'rule', 'positives', 'negatives', 'enter', 'exit', 'combine')
               if key in saved}
    saved['id'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    if args.out is None:        # an instruction of a running recipe: kept by the run, not in a file
        return saved
    _save(args.out, saved)      # readable by the owner only
    summary = {key: saved[key] for key in ('id', 'label', 'mode', 'method', 'confirmed', 'enter', 'exit',
                                           'threshold_source', 'rule_source', 'combine') if key in saved}
    evidence = saved.get('threshold_evidence') or {}
    print(json.dumps(dict(out=args.out, **summary, positives=len(saved.get('positives', ())),
                          negatives=len(saved.get('negatives', ())), **({'evidence': evidence} if evidence else {})),
                     ensure_ascii=False), flush=True)
    for note in evidence.get('notes', ()):
        print(f'note: {note}', file=sys.stderr)
    return saved


def combined(recipe, args, rule, mode, channels, thresholds, entries, joints, *, confirmed):
    """The movement file of a rule with example clips (owner decision 58).

    The text's rule is the skeleton (joints, comparisons, order); unless ``--keep-rule-numbers``, its numbers are
    calibrated from the clips' joints (:func:`gamst.providers.movement_rule.calibrate`; a clip whose joints were not
    seen keeps the text's numbers). The clips are also kept as examples, with thresholds from
    :func:`gamst.providers.movement_match.derive`. Counting uses both (``combine``: ``all`` in count mode, ``any`` in search
    mode, unless ``--combine`` fixes it)."""
    from ..providers.movement_rule import calibrate, validate_rule
    try:
        written = validate_rule(rule)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    used, evidence = written, None
    if not getattr(args, 'keep_rule_numbers', False):
        clips = {key: [[item for item in samples if item[3]] for samples in joints[key]] for key in joints}
        if any(clips['positives']):
            try:
                used, evidence = calibrate(written, [clip for clip in clips['positives'] if clip],
                                           [clip for clip in clips['negatives'] if clip])
            except ValueError as error:
                raise SystemExit(f'the clips do not calibrate the rule: {error}') from error
    combine = getattr(args, 'combine', None)
    return dict(kind='exemplar', format=2, recipe=recipe['name'], label=args.name or args.text, text=args.text,
                mode=f'rule+{mode}', examples_mode=mode, channels=channels, method='rule+examples', confirmed=confirmed,
                rule=used, rule_text=written, rule_source='text+clips' if evidence else 'text',
                **({'calibration': evidence} if evidence else {}), **({'combine': combine} if combine else {}),
                **thresholds, positives=[entry['saved'] for entry in entries['positives']],
                negatives=[entry['saved'] for entry in entries['negatives']])


def _rule(recipe, args, rule, *, confirmed, proposal=None):
    from ..providers.movement_rule import POSE, validate_rule
    from ..providers.pose_exemplar import CHANNELS_VERSION
    try:
        rule = validate_rule(rule)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    channels = f'{POSE}@{CHANNELS_VERSION}.rule' if rule['channels'] == POSE else f"{rule['channels']}@1"
    return _write(args, dict(kind='exemplar', format=2, recipe=recipe['name'], label=args.name or args.text,
                             text=args.text, mode='rule', channels=channels, method='rule', confirmed=confirmed,
                             rule=rule, **({'proposal': proposal} if proposal else {})))


async def enroll(recipe, args):
    """One movement file from example clips, confirmed proposals or a rule."""
    from ..movement_finder import movement_text
    from ..providers.movement_match import MAX_EXAMPLES, derive, example
    item = _exemplar_item(recipe)
    if args.text:
        try:
            args.text = movement_text(args.text)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    given, negatives = getattr(args, 'clips', None) or (_positives(recipe, args), [])
    rule = None
    if args.rule:
        if args.proposals:
            raise SystemExit('--rule takes example clips (--video, --negative), not proposals')
        if not args.text:
            raise SystemExit('--rule needs the --text it stands for')
        rule = getattr(args, 'rule_value', None)
        if rule is None:
            try:
                rule = json.loads(Path(args.rule).read_text())
            except (OSError, ValueError) as error:
                raise SystemExit(f'{args.rule}: {error}') from error
        if not given and not args.negative:
            return _rule(recipe, args, rule, confirmed=not args.no_confirm)
        if not given:
            raise SystemExit('--negative needs example clips of the movement as well (--video FILE@START-END)')
    positives = list(given)
    if args.proposals:
        if not args.text and not args.proposal:
            raise SystemExit('--proposals needs --text (every proposal of it) or --proposal ID')
        found, rejected, rule = _proposed(recipe, args)
        negatives = negatives + rejected
        if rule is not None:
            if given or args.negative:
                raise SystemExit('a proposed rule specifies the movement alone; it takes no clips')
            args.text = args.text or rule['text']
            return _rule(recipe, args, rule['rule'], confirmed=rule['status'] == 'confirmed', proposal=rule['id'])
        positives += found
    elif args.text and not given:
        raise SystemExit('a text alone is not counted: search a recording (gamst recipe propose), then enroll with '
                         '--proposals; or give a --rule')
    if not positives:
        raise SystemExit('give example clips: --video FILE[@start-end] (repeat it), --camera, --audio or --microphone')
    kind = {'microphone': 'audio', 'sensor': 'sensor'}.get(exemplar_input(recipe)['use'], 'video')
    negatives = negatives + [parse_clip(kind, value) for value in args.negative]
    if len(positives) > MAX_EXAMPLES or len(negatives) > MAX_EXAMPLES:
        raise SystemExit(f'at most {MAX_EXAMPLES} positive and {MAX_EXAMPLES} negative examples')
    _refuse_repeats(positives, negatives)
    mode = args.mode or ('sequence' if rule is not None else item.get('mode', 'hold'))
    entries, channels, joints = dict(positives=[], negatives=[]), None, dict(positives=[], negatives=[])
    for key, clips in (('positives', positives), ('negatives', negatives)):
        for clip in clips:
            samples, contract = await demonstration(recipe, args, clip, joints=rule is not None)
            if rule is not None:
                joints[key].append(samples)
                samples = [item[:3] for item in samples]
            if channels not in (None, contract):
                raise SystemExit(f'{clip.name()} gives {contract}, the other clips {channels}')
            channels = contract
            try:
                saved = template(mode, samples)
                entries[key].append(dict(saved=dict(
                    saved, frames=len(samples), source=clip.name(), method=clip.method, confirmed=clip.confirmed,
                    **({'person': dict(at=list(clip.at))} if clip.at else {'person': dict(who=args.from_who)}
                       if getattr(args, 'from_who', None) else {}),
                    **({'proposal': clip.proposal} if clip.proposal else {})),
                    template=example(saved['value'], span_ms=saved.get('span_ms')), clip=samples))
            except ValueError as error:
                raise SystemExit(f'{clip.name()}: {error}') from error
    try:
        thresholds = derive(mode, entries['positives'], entries['negatives'], enter0=float(item.get('enter', .12)),
                            exit0=float(item.get('exit', .18)), min_hold_ms=int(item.get('min_hold_ms', 200)))
    except ValueError as error:
        raise SystemExit(f'{error}; the examples do not say which is the movement') from error
    used = [entry['saved'] for entry in (*entries['positives'], *entries['negatives'])]
    methods = sorted({entry['method'] for entry in used[:len(positives)]})
    if rule is not None:
        return _write(args, combined(recipe, args, rule, mode, channels, thresholds, entries, joints,
                                     confirmed=not args.no_confirm and all(entry['confirmed'] for entry in used)))
    return _write(args, dict(
        kind='exemplar', format=2, recipe=recipe['name'],
        label=args.name or args.text or next((reference.get('label') for reference in recipe['references']
                                              if reference['name'] == item.get('reference', 'exemplar')), None),
        **({'text': args.text} if args.text else {}), mode=mode, channels=channels,
        method=methods[0] if len(methods) == 1 else 'examples+text_finder',
        confirmed=all(entry['confirmed'] for entry in used), **thresholds,
        positives=[entry['saved'] for entry in entries['positives']],
        negatives=[entry['saved'] for entry in entries['negatives']]))


# ---------------------------------------------------------------- text: proposals

def _semantic_backend(name, args):
    """The semantic lane's own client, from the profile setup wrote (or the recipe item's ``config``)."""
    from ..providers.streaming_semantic import OpenAIStreamingBackend
    from . import RECIPES, installed
    path = Path(name) if name.endswith('.json') and Path(name).is_file() else RECIPES / f'{name}.json'
    recipe = json.loads(path.read_text())
    item = next((item for item in recipe['modules'] if item['use'] == 'semantic-scene'), None)
    if item is None:
        raise SystemExit(f'{name} has no semantic-scene module: finding a text needs a VLM endpoint')
    try:
        config = item.get('config') or installed(recipe, item, args.profile)['config']
    except SystemExit as error:
        raise SystemExit(f"{error} --with {item['name']}") from error
    if not {'endpoint', 'model', 'revision'} <= set(config):
        raise SystemExit('the set up vision.semantic_scene candidate is not an endpoint; finding a text needs one')
    return OpenAIStreamingBackend.from_config(config)


async def propose(recipe, args, *, backend=None):
    """Search recordings for a text and write the candidates as proposals; counts nothing."""
    from ..inputs import open_source
    from ..movement_finder import MAX_WINDOWS, check_windows, find, movement_text
    from ..movement_proposals import Proposals, checked_interval, video_identity
    _exemplar_item(recipe)
    if exemplar_input(recipe)['use'] in ('microphone', 'sensor'):
        raise SystemExit('a VLM looks at video frames; give example clips for a way of speaking')
    if not args.video:
        raise SystemExit('propose needs --video FILE')
    try:
        text = movement_text(args.text)
        size, overlap = check_windows(fps=args.fps, window_s=args.window_s, overlap_s=args.overlap_s)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    backend = backend or _semantic_backend(args.recipe, args)
    store, result = Proposals(args.proposals), []
    for value in args.video:
        options = dict(capture_fps=args.capture_fps) if args.capture_fps else {}
        source = open_source('video', value, **options)
        duration = source.duration_s()
        windows = None if duration is None else max(1, -(-int(duration * args.fps) // (size - overlap)))
        limit = args.max_windows or MAX_WINDOWS
        if windows is not None and windows > limit:
            raise SystemExit(f'{value}: about {windows} VLM calls, more than {limit}; search a shorter recording')
        print(json.dumps(dict(video=Path(value).name, vlm_calls_about=windows, frames_per_call=size,
                              remote=bool(getattr(backend, 'remote', False)),
                              model=backend.model_id)), flush=True)
        try:
            found = await find(source, backend, text, fps=args.fps, window_s=args.window_s, overlap_s=args.overlap_s,
                               max_windows=limit)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        finally:
            source.close()
        candidates, dropped = [], 0
        for item in found.pop('candidates'):
            try:
                checked_interval(item['start_s'], item['end_s'], duration)
                candidates.append(item)
            except ValueError:      # e.g. joined to something longer than one movement
                dropped += 1
        added = store.propose_intervals(recipe=recipe['name'], text=text, video=video_identity(value),
                                        candidates=candidates, duration_s=duration, model=backend.model_id)
        result.append(dict(video=Path(value).name, **found, dropped_candidates=dropped, proposed=len(added)))
        print(json.dumps(dict(result[-1], proposals=added), ensure_ascii=False), flush=True)
    print(f'nothing is counted yet: gamst recipe proposals list --proposals {args.proposals}', file=sys.stderr)
    return result


def proposals(args):
    """``gamst recipe proposals list|confirm|reject``: the owner's decisions, made in the owner's own terminal."""
    from ..movement_proposals import OWNER, Proposals
    store = Proposals(args.proposals)
    try:
        if args.decision == 'list':
            result = store.list(status=args.status, text=args.text, limit=512)
            for item in result['proposals']:
                place = f"{item['video']['name']} {item['start_s']:.2f}-{item['end_s']:.2f} s" \
                    if item['kind'] == 'interval' else json.dumps(item['rule']) if item['kind'] == 'rule' \
                    else 'word: ' + ', '.join(item['spellings'])
                score = '' if item.get('score') is None else f" score {item['score']:.2f}"
                print(f"{item['id'][:12]}  {item['status']:9}  {item['proposer']:6}  {place}{score}  "
                      f"{json.dumps(item['text'], ensure_ascii=False)}")
            print(f"{result['total']} proposals", file=sys.stderr)
            return result
        everything, resolved = store.list(limit=512)['proposals'], []
        for wanted in args.ids:
            found = re.fullmatch(r'([0-9a-f]+)(?:@(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)|=(.+))?', wanted)
            matches = [item for item in everything if found and item['id'].startswith(found[1])]
            if len(matches) != 1 or len(found[1]) < 6:
                raise ValueError(f'{wanted}: not the start (6 or more characters) of one proposal ID')
            interval = {} if found[2] is None else dict(start_s=float(found[2]), end_s=float(found[3]))
            if found[4] is not None:
                interval = dict(spellings=found[4].split(','))
            resolved.append(store.resolve(proposal_id=matches[0]['id'], actor=OWNER, decision=args.decision,
                                          expected_version=matches[0]['version'], **interval))
        for item in resolved:
            print(json.dumps(item, ensure_ascii=False))
        return resolved
    except ValueError as error:
        raise SystemExit(str(error)) from error


# ---------------------------------------------------------------- find and export

def counting_modules(recipe):
    """The exemplar module and everything it reads, directly or through other modules or derived inputs."""
    from . import upstream
    return upstream(recipe['modules'], _exemplar_item(recipe)['name'])


async def find_in(recipe, args, kind, path):
    """The movement's events in one recording: the recipe's own counter, its own records."""
    from .. import Gamst
    from ..counting import event_rows
    from . import build_pipeline, store_options
    item = _exemplar_item(recipe)
    modules = counting_modules(recipe)
    place = exemplar_input(recipe)['id']
    capability = item.get('capability_id', 'temporal.exemplar_events')
    run = argparse.Namespace(**dict(vars(args), **{name: None for name in SOURCES}))
    setattr(run, kind, path)
    records, seen = [], dict(frames=0)

    def keep(found):
        records.extend(record for record in found if ((record.get('observation') or {}).get('capability') or {})
                       .get('id') == capability and record['observation']['value'].get('events'))

    async with Gamst(consent=True, **store_options(args)) as human:
        pipeline = await build_pipeline(recipe, run, human, modules=modules)

        async def on_frame(input_id, frame, found):
            seen['frames'] += input_id == place
            seen.setdefault('start_ms', frame.timestamp_ms - (
                round(1000 * len(frame.data) / frame.sample_rate) if frame.sample_rate else 0))
            keep(found)
            return True

        try:
            keep(await pipeline.run(on_frame=on_frame, stop_on_failure=True) or [])
            movement = pipeline.modules[item['name']]['module'].status()
            lost = pipeline.lost(capability)
        finally:
            await pipeline.close()
    rows = []
    for row in event_rows(records, capability=capability, event='exemplar.matched'):
        row['source'] = Path(path).name
        for key in ('start', 'end'):
            row[f'{key}_s'] = round((row[f'{key}_ms'] - seen['start_ms']) / 1000, 3)
        rows.append(row)
    return rows, dict(source=Path(path).name, frames=seen['frames'], count=len(rows), movement=movement,
                      complete=not lost['gaps'], **({'lost': lost} if lost['gaps'] else {}))


async def find(recipe, args):
    from ..counting import write_rows
    from . import check, source_fps
    place = exemplar_input(recipe)['use']
    kind = {'microphone': 'audio', 'sensor': 'sensor'}.get(place, 'video')
    paths = getattr(args, kind) or []
    if not paths:
        raise SystemExit(f'find needs --{kind} FILE (repeat it for more recordings)')
    everything, summaries = [], []
    for path in paths:
        verdict = check(recipe, fps=source_fps(kind, path, args.capture_fps), max_reps_s=args.max_reps_s)
        if verdict.get('ok') is False:
            raise SystemExit(f"{path}: {'; '.join(verdict['problems'])}")
        rows, summary = await find_in(recipe, args, kind, path)
        everything += rows
        summaries.append(summary)
        for row in rows:
            print(f"{row['source']}  {row['start_s']:.2f}-{row['end_s']:.2f} s  distance {row['distance']:.3f}  "
                  f"{row.get('label')}", flush=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, 'w', newline='') as out:
            write_rows(everything, out, args.format)
    return everything, summaries


def export(args):
    """Events of a store file as JSONL or CSV. The store is opened read-only.

    What the Pipeline lost is part of the data: ``pipeline.gap`` rows that affect the exported capability come
    with its events (unless ``--event`` asks for one kind), and the summary line counts them.
    """
    from ..counting import GAP_CAPABILITY, event_rows, gaps_of, write_rows
    from ..storage import Store
    try:
        store = Store.open_read_only(args.store)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    try:
        def stored(capability):
            records, position = [], 0
            while True:
                page, _ = store.observation_page_rows(subject=args.subject, after=position, limit=1000,
                                                      capability_id=capability, events_only=True)
                if not page:
                    return records
                records += [record for _, record in page]
                position = page[-1][0]

        records, gaps = stored(args.capability), gaps_of(stored(GAP_CAPABILITY), args.capability)
        if args.capability is not None:
            others = gaps if args.capability != GAP_CAPABILITY else []
            records = sorted([*records, *others], key=lambda record: record['timestamp_ms'])
        removed = store.removed_records()
    finally:
        store.close()
    rows = event_rows(records, event=args.event)
    if args.out:
        with open(args.out, 'w', newline='') as out:
            write_rows(rows, out, args.format)
    else:
        write_rows(rows, sys.stdout, args.format)
    print(json.dumps(dict(events=sum(row['event'] != 'pipeline.gap' for row in rows), records_no_longer_in_the_store=removed, complete=not gaps and not removed,
                          gap_records=len(gaps))), file=sys.stderr)
    return rows
