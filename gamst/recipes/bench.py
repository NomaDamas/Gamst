"""``gamst recipe bench person-term-count --clips DIR``: one recipe on several owner clips, paced as live, one table.

A clips folder has one subfolder per clip::

    DIR/<clip>/
        clip.json      optional: {"media": "a.mp4", "voice": ..., "consent": "...", "terms": [...] | "terms.json",
                                  "truth": "truth.json", "language": "ko"}
        a.mp4 | a.wav  the recording (default: the only media file of the folder)
        voice.json     the enrolled voice (gamst recipe enroll --kind voice), or in clip.json
                       "voice": "file.json" | {"file": "voice.wav"} | {"from_s": 11.0, "to_s": 12.5} (a range of this clip;
                       it needs "consent": then words inside that range are not fairly scored: exclude them)
        terms.txt      the words to count, one per line or comma-separated (or "terms" in clip.json)
        truth.json     who said what, by the time the word ENDS (seconds from the start of the recording)::

            {"tolerance_s": 0.6,
             "words": [{"end_s": 12.04, "by": "person"},           # the enrolled person said it: it counts
                       {"end_s": 12.64, "by": "other"},            # another person: it must not count
                       {"end_s": 24.70, "by": "none"},             # nobody said it (counted: the ASR invented it)
                       {"from_s": 49.0, "to_s": 54.0, "by": "other"},   # every word in a stretch
                       {"end_s": 12.04, "by": "person", "exclude": "inside the voice reference"}]}   # not scored

Each clip runs the recipe with ``--as-live`` (the recording at its own pace, as a microphone would give it). The
table has, per clip and in total: what was counted and how much of it was right (precision, recall), counted words
nobody said (``invented``), counted words of another person (``other``), counted words the truth does not mention
(``unlabelled``), missed words, provisional -> final latency (p50/p95 of the time between a word's commit and its
revision), word end -> provisional (p50/p95), and the real-time factor (from the first frame to the end of the run,
over the recording's length). ``final`` is the count after revisions; ``at once`` is what was counted provisionally.
Arguments it does not know go to ``gamst recipe run`` (``--with tracks``, ``--profile``, ``--speaker-threshold``).
No transcript is kept: the result holds the counted term and its times only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

MEDIA = ('.mp4', '.mov', '.mkv', '.webm', '.m4a', '.wav')
EVENT = 'term.spoken'


def _media(folder, manifest):
    if manifest.get('media'):
        return folder / manifest['media']
    found = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in MEDIA and not path.stem.startswith('voice')]
    if len(found) != 1:
        raise SystemExit(f'{folder}: put one recording in the folder, or name it in clip.json ("media")')
    return found[0]


def _terms(folder, manifest):
    value = manifest.get('terms')
    if isinstance(value, list):
        return ','.join(value)
    path = folder / (value or 'terms.txt')
    if not path.is_file():
        raise SystemExit(f'{folder}: no terms (terms.txt, or "terms" in clip.json)')
    if path.suffix == '.json':
        return str(path)
    return ','.join(item.strip() for line in path.read_text().splitlines() for item in line.split(',') if item.strip())


def load_clip(folder):
    """What one clip folder holds (nothing is run)."""
    folder = Path(folder)
    manifest = json.loads((folder / 'clip.json').read_text()) if (folder / 'clip.json').is_file() else {}
    truth = json.loads((folder / manifest.get('truth', 'truth.json')).read_text())
    voice = manifest.get('voice', 'voice.json')
    media = _media(folder, manifest)
    return dict(name=folder.name, folder=folder, media=media, voice=voice, consent=manifest.get('consent'),
                terms=_terms(folder, manifest), truth=truth, language=manifest.get('language'))


def _source(media):
    return ['--audio' if media.suffix.lower() == '.wav' else '--video', str(media)]


def _voice(clip, extra, work, run):
    """The enrolled voice file of a clip, enrolled first when it is a clip or a range."""
    voice = clip['voice']
    if isinstance(voice, str):
        return str(clip['folder'] / voice)
    if not clip.get('consent'):
        raise SystemExit(f"{clip['name']}: a voice from a recording needs \"consent\" in clip.json")
    out = str(Path(work) / f"{clip['name']}.voice.json")
    options = [name for name in ('--profile', '--worker-python') if name in extra]
    passed = [value for name in options for value in (name, extra[extra.index(name) + 1])]
    if 'file' in voice:
        source, timing = _source(clip['folder'] / voice['file']), []
    else:
        source = _source(clip['media'])
        timing = ['--start', str(voice['from_s']), '--seconds', str(voice['to_s'] - voice['from_s'])]
    run(['enroll', 'person-term-count', '--kind', 'voice', *source, *timing, '--consent', clip['consent'], '--out', out,
         *passed])
    return out


def _labels(truth):
    """(points, ranges): points ``(end_ms, by, excluded)``, ranges ``(from_ms, to_ms, by)``.

    A word with ``count`` (a repeated sound word, "달달달달달": 5) is that many points at its end. A word inside a range
    of ``truth["exclude"]`` (``[{from_s, to_s, why}]``, e.g. the voice reference) is excluded."""
    points, ranges = [], []
    out = [(round(1000 * item['from_s']), round(1000 * item['to_s'])) for item in truth.get('exclude') or ()]
    for word in truth.get('words') or ():
        if 'end_s' in word:
            end = round(1000 * word['end_s'])
            inside = any(low <= end <= high for low, high in out)
            points += [(end, word['by'], bool(word.get('exclude')) or inside)] * max(1, int(word.get('count') or 1))
        else:
            ranges.append((round(1000 * word['from_s']), round(1000 * word['to_s']), word['by']))
    return points, ranges


def _end(event):
    """Where an event is matched: the end of the whole recogniser word for a repetition inside it, else its end."""
    if 'word_end_ms' in event:
        return event['word_end_ms']
    part = event.get('repetition')     # an older report: repetitions share the word's time evenly
    if part and type(event.get('start_ms')) in (int, float):
        return event['end_ms'] + (part[1] - 1 - part[0]) * (event['end_ms'] - event['start_ms'])
    return event['end_ms']


def match(truth, events, start_ms):
    """Each event (``end_ms``) paired with a truth word by end time, nearest first: ``[(event, label, truth index)]``.

    ``label``: the truth word's ``by``, ``excluded``, the ``by`` of a stretch it ends in, or ``unlabelled``. An event
    that ends inside an excluded range of ``truth["exclude"]`` is ``excluded`` too.
    """
    points, ranges = _labels(truth)
    out = [(round(1000 * item['from_s']), round(1000 * item['to_s'])) for item in truth.get('exclude') or ()]
    tolerance = round(1000 * float(truth.get('tolerance_s', .5)))
    ends = [_end(event) - start_ms for event in events]
    pairs = sorted((abs(end - point[0]), index, slot) for index, end in enumerate(ends)
                   for slot, point in enumerate(points) if abs(end - point[0]) <= tolerance)
    taken, used = {}, set()
    for _, index, slot in pairs:
        if index not in taken and slot not in used:
            taken[index] = slot
            used.add(slot)
    found = []
    for index, event in enumerate(events):
        if index in taken:
            point = points[taken[index]]
            found.append((event, 'excluded' if point[2] else point[1], taken[index]))
            continue
        if any(low <= ends[index] <= high for low, high in out):
            found.append((event, 'excluded', None))
            continue
        stretch = next((by for low, high, by in ranges if low <= ends[index] <= high), 'unlabelled')
        found.append((event, stretch, None))
    return found


def error_types(truth, report, capability='speech.term_events'):
    """Where the person's words were lost or others' were counted, by kind: ``person`` (the person's words scored),
    ``heard`` (any recogniser wrote it: a term event of any state was reported there), ``counted_right``,
    ``person_to_other`` (heard, given to another voice or revoked), ``uncertain`` (heard, left uncertain),
    ``not_heard``, ``other_to_person`` (counted where another voice said it), ``nothing_to_person`` (counted where
    nobody said it), ``unlabelled`` (counted where the truth says nothing)."""
    start = report['start_ms']
    stated = (report.get('event_states') or {}).get(capability) or []
    replaced = {event['replaces']: event for event in stated if event.get('replaces')}
    words = [event for event in stated if not event.get('replaces')]
    now = lambda event: replaced.get(event.get('event_id'), event)
    points, _ = _labels(truth)
    person = {index for index, point in enumerate(points) if point[1] == 'person' and not point[2]}
    result = dict(person=len(person), heard=0, counted_right=0, person_to_other=0, uncertain=0, not_heard=0,
                  other_to_person=0, nothing_to_person=0, unlabelled=0)
    seen = {}
    for event, label, slot in match(truth, words, start):
        state = now(event)
        counted = state.get('event') == EVENT and state.get('state') in ('provisional', 'final')
        if slot in person:
            seen[slot] = 'counted' if counted else 'uncertain' if state.get('state') == 'uncertain' else 'other'
        elif counted and label in ('other', 'none', 'unlabelled'):
            result[dict(other='other_to_person', none='nothing_to_person', unlabelled='unlabelled')[label]] += 1
    for slot in person:
        kind = seen.get(slot)
        result['heard'] += kind is not None
        result[dict(counted='counted_right', uncertain='uncertain', other='person_to_other').get(kind, 'not_heard')] += 1
    return result


def live_events(report, capability='speech.term_events'):
    """What L1 counted the moment it committed each word (the HUD's instant count before any correction)."""
    stated = (report.get('event_states') or {}).get(capability) or []
    return [event for event in stated if event.get('event') == EVENT and 'replaces' not in event
            and event.get('decided_by') is None and event.get('committed_state', 'provisional') in ('provisional', 'final')]


def _percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(p50=ordered[len(ordered) // 2], p95=ordered[min(len(ordered) - 1, int(.95 * len(ordered)))], n=len(ordered))


def view(truth, events, start_ms):
    """Counted, right, precision, recall, invented, other, unlabelled, missed of counted ``events``."""
    points, _ = _labels(truth)
    pairs = [item for item in match(truth, events, start_ms) if item[1] != 'excluded']
    person = [index for index, point in enumerate(points) if point[1] == 'person' and not point[2]]
    right = {slot for _, label, slot in pairs if label == 'person'}
    counted = len(pairs)
    return dict(counted=counted, right=len(right), precision=round(len(right) / counted, 3) if counted else None,
                recall=round(len(right) / len(person), 3) if person else None,
                invented=sum(label == 'none' for _, label, _ in pairs), other=sum(label == 'other' for _, label, _ in pairs),
                unlabelled=sum(label == 'unlabelled' for _, label, _ in pairs),
                missed=[round(points[index][0] / 1000, 2) for index in person if index not in right])


def score_run(truth, report, *, finished_ms, duration_s, capability='speech.term_events'):
    """The table row of one run's report."""
    start = report['start_ms']
    stated = report.get('event_states', {}).get(capability)
    if not stated:      # nothing was provisional: as committed is final
        stated = [dict(event, state='final') for event in report['events'].get(capability, ()) if event.get('event') == EVENT]
    spoken = [event for event in stated if event.get('event') == EVENT]
    final = [event for event in spoken if event['state'] in ('provisional', 'final')]
    first = [event for event in spoken if 'replaces' not in event and event.get('decided_by') != 'verification']
    excluded = {id(event) for event, label, _ in match(truth, spoken, start) if label == 'excluded'}
    timed = [event for event in stated if id(event) not in excluded and event.get('committed_ms') is not None]
    settle = [event['settled_ms'] - event['committed_ms'] for event in timed if event.get('settled_ms') is not None
              and 'replaces' not in event]
    at_once = [event['committed_ms'] - event['end_ms'] for event in timed if 'replaces' not in event
               and event.get('decided_by') != 'verification']
    return dict(final=view(truth, final, start), at_once=view(truth, first, start),
                revoked=sum(event['state'] == 'revoked' for event in spoken),
                uncertain=sum(event['state'] == 'uncertain' for event in spoken),
                settle_ms=_percentiles(settle), provisional_ms=_percentiles(at_once),
                real_time_factor=round((finished_ms - start) / 1000 / duration_s, 2) if duration_s else None,
                complete=report.get('complete'))


def words_of(truth, report, capability='speech.term_events'):
    """Per word (every reported term event, and every truth word nothing was reported for): what was decided."""
    start, rows = report['start_ms'], []
    stated = report.get('event_states', {}).get(capability) or []
    points, _ = _labels(truth)
    seen = set()
    events = [event for event in stated if 'replaces' not in event]
    promoted = {event['replaces']: event for event in stated if 'replaces' in event}
    for event, label, slot in match(truth, events, start):
        seen.add(slot)
        later = promoted.get(event.get('event_id'))
        now = later or event
        final = 'counted' if now['event'] == EVENT and now['state'] in ('provisional', 'final') else 'uncertain' \
            if now['state'] == 'uncertain' else 'not counted'
        settled = (later or event).get('settled_ms') if later is None else later.get('committed_ms')
        added = event.get('decided_by') == 'verification'
        revised = event.get('revised_by') or {}
        rows.append(dict(end_s=round((event['end_ms'] - start) / 1000, 2), truth=label, slot=event.get('slot'),
                         slot_score=event.get('slot_score'), final_slot_score=(later or event).get('settled_slot_score', (
                             later or {}).get('slot_score')),
                         speaker_score=event.get('speaker_score'),
                         provisional='not heard at once' if added else 'counted' if event['event'] == EVENT
                         else 'not counted', final=final,
                         reason=revised.get('reason') or ((later or {}).get('revised_by') or {}).get('reason'),
                         basis=revised.get('basis') or event.get('basis'),
                         latency_ms=None if settled is None or event.get('committed_ms') is None or added
                         else settled - event['committed_ms']))
    for index, point in enumerate(points):
        if index not in seen:
            rows.append(dict(end_s=round(point[0] / 1000, 2), truth='excluded' if point[2] else point[1], slot=None,
                             slot_score=None, provisional='not transcribed', final='not transcribed', latency_ms=None))
    return sorted(rows, key=lambda row: row['end_s'])


def _total(rows):
    keys = ('counted', 'right', 'invented', 'other', 'unlabelled')
    total = {}
    for name in ('final', 'at_once'):
        part = {key: sum(row[name][key] for row in rows) for key in keys}
        persons = sum(row[name]['right'] + len(row[name]['missed']) for row in rows)
        part.update(precision=round(part['right'] / part['counted'], 3) if part['counted'] else None,
                    recall=round(part['right'] / persons, 3) if persons else None,
                    missed=sum(len(row[name]['missed']) for row in rows))
        total[name] = part
    return total


def table(results):
    """The printed table: one line per clip and view, then the totals."""
    head = f"{'clip':14} {'view':8} {'counted':>7} {'right':>5} {'prec':>5} {'recall':>6} {'invent':>6} {'other':>5} " \
           f"{'unlab':>5} {'missed':>6} {'settle p50/p95 ms':>18} {'at once p50/p95 ms':>19} {'rtf':>5}"
    lines = [head, '-' * len(head)]

    def cell(value, width):
        return f"{'-' if value is None else value:>{width}}"

    def spread(value):
        return '-' if not value else f"{value['p50']}/{value['p95']}"

    def line(name, kind, part, row):
        missed = part['missed'] if isinstance(part['missed'], int) else len(part['missed'])
        settle, once = (row.get('settle_ms'), row.get('provisional_ms')) if kind == 'final' else (None, None)
        lines.append(f"{name:14} {kind:8} {part['counted']:>7} {part['right']:>5} {cell(part['precision'], 5)} "
                     f"{cell(part['recall'], 6)} {part['invented']:>6} {part['other']:>5} {part['unlabelled']:>5} {missed:>6} "
                     f"{spread(settle):>18} {spread(once):>19} "
                     f"{cell(row.get('real_time_factor') if kind == 'final' else None, 5)}")

    for result in results:
        for kind in ('final', 'at_once'):
            line(result['clip'], kind, result[kind], result)
    if len(results) > 1:
        total = _total(results)
        lines.append('-' * len(head))
        for kind in ('final', 'at_once'):
            line('total', kind, total[kind], {})
    return '\n'.join(lines)


def main(argv):
    parser = argparse.ArgumentParser(prog='gamst recipe bench', description=__doc__.split('\n')[0])
    parser.add_argument('recipe')
    parser.add_argument('--clips', required=True, help='a folder with one subfolder per clip')
    parser.add_argument('--out', help='write the table, per-word rows and settings as JSON')
    parser.add_argument('--words', action='store_true', help='also print what was decided for each word')
    parser.add_argument('--recorded', action='store_true', help='run the recordings as fast as they go (not paced)')
    args, extra = parser.parse_known_args(argv)
    from . import read_recipe
    if read_recipe(args.recipe)[1].get('name') != 'person-term-count':
        raise SystemExit('bench scores person-term-count runs (words by one person), or a file of that recipe')
    from . import main as recipe_main
    from ..inputs import media_streams
    folders = sorted(path for path in Path(args.clips).iterdir() if path.is_dir())
    if not folders:
        raise SystemExit(f'{args.clips}: no clip folders')
    results = []
    with tempfile.TemporaryDirectory(prefix='gamst-bench-') as work:
        for folder in folders:
            clip = load_clip(folder)
            voice = _voice(clip, extra, work, recipe_main)
            report_path = str(Path(work) / f"{clip['name']}.report.json")
            argv = ['run', args.recipe, *_source(clip['media']), '--voice', voice, '--terms', clip['terms'], '--headless',
                    '--report', report_path, *(['--language', clip['language']] if clip['language'] else []),
                    *([] if args.recorded else ['--as-live']), *extra]
            print(f"bench: {clip['name']}", file=sys.stderr, flush=True)
            report = recipe_main(argv)
            finished = time.time_ns() // 1_000_000
            duration = media_streams(str(clip['media']))['duration_s']
            row = dict(clip=clip['name'], **score_run(clip['truth'], report, finished_ms=finished, duration_s=duration))
            if args.recorded:
                row['real_time_factor'] = None     # capture times are not the clock of a recorded run
            row['words'] = words_of(clip['truth'], report)
            results.append(row)
    print(table(results))
    if args.words:
        for row in results:
            for word in row['words']:
                print(json.dumps(dict(clip=row['clip'], **word), ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(dict(recipe=args.recipe, arguments=extra, paced=not args.recorded,
                                                  clips=results, total=_total(results) if len(results) > 1 else None),
                                             ensure_ascii=False, indent=2))
    return results
