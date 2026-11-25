"""Specify a movement, find it in recordings, and turn its events into data (``gamst recipe`` actions).

- ``enroll``: example clips (``--video FILE[@start-end]``, ``--negative ...``), confirmed proposals of a text,
  or a declarative rule become one movement file. It holds channel vectors (or the rule), thresholds and where
  they come from; never frames, joints or audio.
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
PROPOSALS = 'gamst-movement-proposals.json'
FEATURES = ('vision.pose_channels', 'audio.voice_channels')
FEATURE_PARTS = ('mediapipe-pose', 'mediapipe-graph', 'sapiens2-pose', 'pose-channels', 'prosody', 'voice-channels')


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

    def name(self):
        place = Path(self.path).name if isinstance(self.path, str) else self.kind
        return place if self.start is None else f'{place}@{self.start:g}-{self.end:g}'

    def footage(self):
        path = str(Path(self.path).expanduser().resolve()) if isinstance(self.path, str) else self.path
        return (self.kind, path), self.start or 0., float('inf') if self.end is None else self.end


def parse_clip(kind, text):
    """``FILE`` or ``FILE@start-end`` in seconds (a file whose name looks like that wins)."""
    found = re.fullmatch(r'(.+)@(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', text)
    if found is None or Path(text).is_file():
        return Clip(kind, text)
    start, end = float(found[2]), float(found[3])
    if not 0 < end - start <= MAX_CLIP_S:
        raise SystemExit(f'{text}: an interval is start < end, at most {MAX_CLIP_S:g} s')
    return Clip(kind, found[1], start, end)


def _exemplar_item(recipe):
    item = next((item for item in recipe['modules'] if item['use'] == 'exemplar'), None)
    if item is None:
        raise SystemExit(f"{recipe['name']} has no exemplar module to enroll for")
    return item


def feature_modules(recipe):
    """The modules that turn the recipe's input into the channels its exemplar module reads.

    The demonstration itself is the only person or voice in it, so identity modules are not used here.
    """
    from . import choose
    plain = choose(recipe, exclude=[item['name'] for item in recipe['modules'] if item.get('optional')])
    return [dict(item, consumes=[recipe['inputs'][0]['id']]) if item['use'] == 'prosody' else item
            for item in plain['modules'] if item['use'] in FEATURE_PARTS]


async def demonstration(recipe, args, clip):
    """([(timestamp_ms, channels, mask), ...], channel contract) of one clip; frames and joints are not kept."""
    from .. import Gamst
    from ..inputs import open_source
    from . import build_pipeline
    samples, seen = [], {}
    options = {}
    if clip.kind == 'video':
        if getattr(args, 'capture_fps', None):
            options['capture_fps'] = args.capture_fps
        if clip.start is not None:
            options.update(start_s=max(0., clip.start - 1), end_s=clip.end)  # a second for the tracker to settle
    source = open_source(clip.kind, clip.path if clip.path is not True else None, **options)
    run = argparse.Namespace(**dict(vars(args), camera=None, video=None, take=None, audio=None, microphone=None,
                                    iphone=None))
    setattr(run, clip.kind, clip.path)
    start, end = clip.start or 0., clip.end
    async with Gamst(consent=True) as human:
        pipeline = await build_pipeline(dict(recipe, references=[]), run, human, source=source,
                                        modules=feature_modules(recipe))

        async def on_frame(input_id, frame, records):
            origin = getattr(source, 'origin_ms', None)
            seen.setdefault('ms', frame.timestamp_ms if origin is None else origin)
            seconds = (frame.timestamp_ms - seen['ms']) / 1000
            if end is not None and seconds > end:
                return False
            if end is None and seconds > MAX_CLIP_S:
                seen['too_long'] = True
                return False
            if seconds >= start:
                for record in records:
                    capability = (record['observation'].get('capability') or {}).get('id')
                    if capability in FEATURES:
                        value = record['observation']['value']
                        samples.append((record['timestamp_ms'], value['channels'], value['mask']))
                        seen['channels'] = f"{capability}@{record['models'][0]['version']}"
            return True

        try:
            await pipeline.run(on_frame=on_frame)
        finally:
            await pipeline.close()
    if 'too_long' in seen:
        raise SystemExit(f'{clip.name()} is longer than {MAX_CLIP_S:g} s: one movement per clip; trim it or name the '
                         'part, FILE@start-end')
    if not samples:
        raise SystemExit(f'nothing to learn in {clip.name()} (no visible person, or no audio)')
    return samples, seen['channels']


def template(mode, samples):
    """The saved value of one example: the median pose of a hold, or the movement as at most 30 vectors."""
    if mode == 'hold':
        size = len(samples[0][1])
        mask = [sum(item[2][i] for item in samples) * 2 > len(samples) for i in range(size)]
        channels = []
        for i in range(size):
            values = sorted(item[1][i] for item in samples if item[2][i])
            channels.append(values[len(values) // 2] if values and mask[i] else 0.)
        return dict(value=dict(channels=channels, mask=mask))
    span = samples[-1][0] - samples[0][0]
    if len(samples) < 2 or span <= 0:
        raise ValueError('a movement needs at least two samples; use --mode hold for a pose')
    step = -(-len(samples) // MAX_SEQUENCE_ITEMS)
    return dict(value=[dict(channels=channels, mask=mask) for _, channels, mask in samples[::step]], span_ms=span)


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
    from . import _source
    if not any(getattr(args, name, None) is not None and getattr(args, name) is not False
               for name in ('camera', 'video', 'take', 'audio', 'microphone')):
        return []
    _, kind = _source(recipe, args)
    values = getattr(args, kind, None)
    values = values if isinstance(values, list) else [values]
    clips = [parse_clip(kind, value) if kind in ('video', 'audio') else Clip(kind, value) for value in values]
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
    chosen = [item for item in chosen if item['recipe'] == recipe['name']]
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
    content = {key: saved[key] for key in ('mode', 'channels', 'rule', 'positives', 'negatives', 'enter', 'exit')
               if key in saved}
    saved['id'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    _save(args.out, saved)      # readable by the owner only
    summary = {key: saved[key] for key in ('id', 'label', 'mode', 'method', 'confirmed', 'enter', 'exit',
                                           'threshold_source') if key in saved}
    evidence = saved.get('threshold_evidence') or {}
    print(json.dumps(dict(out=args.out, **summary, positives=len(saved.get('positives', ())),
                          negatives=len(saved.get('negatives', ())), **({'evidence': evidence} if evidence else {})),
                     ensure_ascii=False), flush=True)
    for note in evidence.get('notes', ()):
        print(f'note: {note}', file=sys.stderr)
    return saved


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
    given, negatives, rule = _positives(recipe, args), [], None
    if args.rule:
        if given or args.negative or args.proposals:
            raise SystemExit('--rule specifies the movement alone; it takes no clips or proposals')
        if not args.text:
            raise SystemExit('--rule needs the --text it stands for')
        try:
            written = json.loads(Path(args.rule).read_text())
        except (OSError, ValueError) as error:
            raise SystemExit(f'{args.rule}: {error}') from error
        return _rule(recipe, args, written, confirmed=not args.no_confirm)
    positives = list(given)
    if args.proposals:
        if not args.text and not args.proposal:
            raise SystemExit('--proposals needs --text (every proposal of it) or --proposal ID')
        found, negatives, rule = _proposed(recipe, args)
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
    kind = 'audio' if recipe['inputs'][0]['use'] == 'microphone' else 'video'
    negatives = negatives + [parse_clip(kind, value) for value in args.negative]
    if len(positives) > MAX_EXAMPLES or len(negatives) > MAX_EXAMPLES:
        raise SystemExit(f'at most {MAX_EXAMPLES} positive and {MAX_EXAMPLES} negative examples')
    _refuse_repeats(positives, negatives)
    mode = args.mode or item.get('mode', 'hold')
    entries, channels = dict(positives=[], negatives=[]), None
    for key, clips in (('positives', positives), ('negatives', negatives)):
        for clip in clips:
            samples, contract = await demonstration(recipe, args, clip)
            if channels not in (None, contract):
                raise SystemExit(f'{clip.name()} gives {contract}, the other clips {channels}')
            channels = contract
            try:
                saved = template(mode, samples)
                entries[key].append(dict(saved=dict(
                    saved, frames=len(samples), source=clip.name(), method=clip.method, confirmed=clip.confirmed,
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
    if recipe['inputs'][0]['use'] == 'microphone':
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
                    if item['kind'] == 'interval' else json.dumps(item['rule'])
                score = '' if item.get('score') is None else f" score {item['score']:.2f}"
                print(f"{item['id'][:12]}  {item['status']:9}  {item['proposer']:6}  {place}{score}  "
                      f"{json.dumps(item['text'], ensure_ascii=False)}")
            print(f"{result['total']} proposals", file=sys.stderr)
            return result
        everything, resolved = store.list(limit=512)['proposals'], []
        for wanted in args.ids:
            found = re.fullmatch(r'([0-9a-f]+)(?:@(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?))?', wanted)
            matches = [item for item in everything if found and item['id'].startswith(found[1])]
            if len(matches) != 1 or len(found[1]) < 6:
                raise ValueError(f'{wanted}: not the start (6 or more characters) of one proposal ID')
            interval = {} if found[2] is None else dict(start_s=float(found[2]), end_s=float(found[3]))
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
    wanted, needed = [_exemplar_item(recipe)['name']], set()
    provider = {name: item['name'] for item in recipe['modules'] for name in (item['name'], *item.get('emits', ()))}
    while wanted:
        name = wanted.pop()
        if name in needed:
            continue
        needed.add(name)
        item = next(item for item in recipe['modules'] if item['name'] == name)
        for source in item['consumes']:
            owner = provider.get(source) or provider.get(source.split(':', 1)[0])
            if owner is not None:
                wanted.append(owner)
    return [item for item in recipe['modules'] if item['name'] in needed]


async def find_in(recipe, args, kind, path):
    """The movement's events in one recording: the recipe's own counter, its own records."""
    from .. import Gamst
    from ..counting import event_rows
    from . import build_pipeline, store_options
    item = _exemplar_item(recipe)
    capability = item.get('capability_id', 'temporal.exemplar_events')
    run = argparse.Namespace(**dict(vars(args), video=None, audio=None, camera=None, take=None, microphone=None))
    setattr(run, kind, path)
    records, seen = [], dict(frames=0)

    def keep(found):
        records.extend(record for record in found if ((record.get('observation') or {}).get('capability') or {})
                       .get('id') == capability and record['observation']['value'].get('events'))

    async with Gamst(consent=True, **store_options(args)) as human:
        pipeline = await build_pipeline(recipe, run, human, modules=counting_modules(recipe))

        async def on_frame(input_id, frame, found):
            seen['frames'] += 1
            seen.setdefault('start_ms', frame.timestamp_ms - (
                round(1000 * len(frame.data) / frame.sample_rate) if frame.sample_rate else 0))
            keep(found)
            return True

        try:
            keep(await pipeline.run(on_frame=on_frame) or [])
            movement = pipeline.modules[item['name']]['module'].status()
        finally:
            await pipeline.close()
    rows = []
    for row in event_rows(records, capability=capability, event='exemplar.matched'):
        row['source'] = Path(path).name
        for key in ('start', 'end'):
            row[f'{key}_s'] = round((row[f'{key}_ms'] - seen['start_ms']) / 1000, 3)
        rows.append(row)
    return rows, dict(source=Path(path).name, frames=seen['frames'], count=len(rows), movement=movement)


async def find(recipe, args):
    from ..counting import write_rows
    from . import check, source_fps
    kind = 'audio' if recipe['inputs'][0]['use'] == 'microphone' else 'video'
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
    """Events of a store file as JSONL or CSV. The store is opened read-only."""
    from ..counting import event_rows, write_rows
    from ..storage import Store
    try:
        store = Store.open_read_only(args.store)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    try:
        records, position = [], 0
        while True:
            page, _ = store.observation_page_rows(subject=args.subject, after=position, limit=1000,
                                                  capability_id=args.capability, events_only=True)
            if not page:
                break
            records += [record for _, record in page]
            position = page[-1][0]
        removed = store.removed_records()
    finally:
        store.close()
    rows = event_rows(records, event=args.event)
    if args.out:
        with open(args.out, 'w', newline='') as out:
            write_rows(rows, out, args.format)
    else:
        write_rows(rows, sys.stdout, args.format)
    print(json.dumps(dict(events=len(rows), records_no_longer_in_the_store=removed)), file=sys.stderr)
    return rows
