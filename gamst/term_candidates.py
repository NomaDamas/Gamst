"""Every place a recogniser wrote a counted word, for the owner to label: a markdown table from a run's report.

    python -m gamst.term_candidates --report run.json --out candidates.md [--title "감스트 뽀또: 뽀또"]

A ``person-term-count`` run ``--with verification`` (and ``--with vote``) reports what each recogniser wrote of the
counted words, without the rest of the transcript: the first hearing's term events (``event_states``), the second
recogniser's and the voters' occurrences (``heard_terms``, ``voted_terms`` of the verification lane's state). Here
they are paired by end time (within ``--near-ms``, nearest first, one to one, as the verification lane pairs them) into
one row per place: when (seconds from the start of the recording), which recognisers wrote it and how, what was
decided at once (provisional) and at the end (final, revoked with its reason, uncertain), and the voice evidence
(the basis of the verdict and the per-word voice score). The last column is left empty for the owner: 감스트 / 다른 사람
/ 말 안 함 (the person said it, another person, nobody said it). No model runs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CAPABILITY = 'speech.term_events'
NAMES = {'first': 'Scribe (L1)', 'second': 'gpt-live (L4)', 'third': 'Speechmatics (vote)'}


def _pair(left, right, near_ms):
    """One-to-one pairs (index in left, index in right) by end time, nearest first."""
    options = sorted((abs(a['end_ms'] - b['end_ms']), i, j) for i, a in enumerate(left) for j, b in enumerate(right)
                     if abs(a['end_ms'] - b['end_ms']) <= near_ms)
    taken, used = {}, set()
    for _, i, j in options:
        if i not in taken and j not in used:
            taken[i] = j
            used.add(j)
    return taken


def _decision(event, replacement):
    now = replacement or event
    state = now.get('state') or 'final'
    if now.get('event') == 'term.spoken':
        if state in ('final', 'provisional'):
            return f'counted ({state})'
        if state == 'uncertain':
            return 'uncertain (shown, not counted)'
        return 'revoked: ' + str((now.get('revised_by') or {}).get('reason') or now.get('reason') or '?')
    return 'not counted: another voice' + ('' if state == 'final' else f' ({state})')


def _first_state(event):
    if event.get('decided_by') == 'verification':
        return '- (added by verification)'
    if event.get('event') == 'term.other_voice':
        return 'not counted (another voice)'
    return {'provisional': 'counted (provisional)', 'uncertain': 'uncertain'}.get(event.get('state'), 'counted')


def rows(report, near_ms=1000):
    """One dict per place a counted word was written, in time order."""
    start = report['start_ms']
    stated = (report.get('event_states') or {}).get(CAPABILITY) or [
        dict(event, state='final') for event in (report.get('events') or {}).get(CAPABILITY, ())]
    replaced = {event['replaces']: event for event in stated if event.get('replaces')}
    first = [event for event in stated if not event.get('replaces') and event.get('decided_by') != 'verification']
    added = [event for event in stated if not event.get('replaces') and event.get('decided_by') == 'verification']
    lane = ((report.get('pipeline') or {}).get('modules') or {}).get('verification') or {}
    state = lane.get('state') or {}
    second = list(state.get('heard_terms') or ())
    third = list(state.get('voted_terms') or ())
    found = []
    with_second, with_third = _pair(first, second, near_ms), _pair(first, third, near_ms)
    for index, event in enumerate(first):
        heard = dict(first=event.get('spoken'))
        if index in with_second:
            heard['second'] = second[with_second[index]].get('spoken', '?')
        if index in with_third:
            heard['third'] = third[with_third[index]].get('spoken', '?')
        found.append(dict(event=event, heard=heard, end_ms=event['end_ms'], start_ms=event.get('start_ms')))
    rest_second = [item for j, item in enumerate(second) if j not in set(with_second.values())]
    rest_third = [item for j, item in enumerate(third) if j not in set(with_third.values())]
    pairs = _pair(rest_second, rest_third, near_ms)
    others = []
    for j, item in enumerate(rest_second):
        heard = dict(second=item.get('spoken', '?'))
        if j in pairs:
            heard['third'] = rest_third[pairs[j]].get('spoken', '?')
        others.append(dict(event=None, heard=heard, end_ms=item['end_ms'], start_ms=item.get('start_ms')))
    used = set(pairs.values())
    others += [dict(event=None, heard=dict(third=item.get('spoken', '?')), end_ms=item['end_ms'],
                    start_ms=item.get('start_ms')) for j, item in enumerate(rest_third) if j not in used]
    to_added = _pair(others, added, near_ms)
    for index, row in enumerate(others):
        if index in to_added:
            row['event'] = added[to_added[index]]
    found += others
    out = []
    for row in sorted(found, key=lambda item: item['end_ms']):
        event = row['event']
        replacement = replaced.get(event.get('event_id')) if event else None
        now = replacement or event or {}
        revised = now.get('revised_by') or (event or {}).get('revised_by') or {}
        out.append(dict(
            start_s=None if row['start_ms'] is None else round((row['start_ms'] - start) / 1000, 2),
            end_s=round((row['end_ms'] - start) / 1000, 2),
            heard={name: spoken for name, spoken in row['heard'].items()},
            provisional=_first_state(event) if event else '- (not written at once)',
            final=_decision(event, replacement) if event else 'not counted: one recogniser only',
            basis=revised.get('basis') or now.get('basis') or (event or {}).get('basis'),
            reason=revised.get('reason') or now.get('reason'),
            speaker_score=(event or {}).get('speaker_score'),
            flags=sorted(set((event or {}).get('flags') or ())),
            repetition=(event or {}).get('repetition')))
    return out


def summary(report):
    """final / provisional / uncertain / revoked of ``term.spoken`` at the end of the run."""
    states = ((report.get('states') or {}).get(CAPABILITY) or {}).get('term.spoken') or {}
    return dict(final=states.get('final', 0), provisional=states.get('provisional', 0),
                uncertain=states.get('uncertain', 0), revoked=states.get('revoked', 0))


def markdown(report, title, near_ms=1000, note=None):
    found = rows(report, near_ms)
    counts = summary(report)
    lines = [f'# {title}', '',
             f"Counted at the end: final {counts['final']}, provisional {counts['provisional']}, uncertain "
             f"{counts['uncertain']} (shown, not counted), revoked {counts['revoked']}. One row per place any recogniser "
             f'wrote the word ({len(found)} rows); times are word ends in seconds from the start of the recording.', '']
    if note:
        lines += [note, '']
    lines += ['| # | start s | end s | written by | provisional | final | voice / speaker | flags | owner: 감스트 / 다른 사람 / 말 안 함 |',
              '|---|---|---|---|---|---|---|---|---|']
    for number, row in enumerate(found, 1):
        heard = ', '.join(f'{NAMES[name]} "{spoken}"' for name, spoken in row['heard'].items())
        voice = '; '.join(item for item in (
            row['basis'], None if row['reason'] in (None, '') or row['final'].startswith('counted')
            or (row['basis'] or '').endswith(str(row['reason']))
            else f"reason {row['reason']}", None if row['speaker_score'] is None
            else f"word voice score {row['speaker_score']:.2f}") if item)
        flags = ', '.join(row['flags'] + ([f"repetition {row['repetition'][0] + 1}/{row['repetition'][1]}"]
                                          if row['repetition'] else []))
        lines.append(f"| {number} | {'' if row['start_s'] is None else row['start_s']} | {row['end_s']} | {heard} | "
                     f"{row['provisional']} | {row['final']} | {voice or '-'} | {flags or '-'} |  |")
    lines += ['', 'Recognisers: Scribe v3 Live (live, L1, the words as key terms); gpt-live-transcribe with the words as '
              'keywords and Qwen3-ForcedAligner times (verification, L4); Speechmatics Enhanced with the words as additional '
              'vocabulary (third voter). A word counts when two of the three wrote it and the voice says it is the person '
              '(pyannoteAI Precision-3 speakers named by CAM++ against the voice sample; the per-word CAM++ score where '
              'speakers overlap). "basis" says which recognisers wrote it (heard_by_N(...)) and how the voice decided.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m gamst.term_candidates', description=__doc__.split('\n')[0])
    parser.add_argument('--report', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--title', default='Candidates')
    parser.add_argument('--note', help='a line under the title (e.g. how the run was made)')
    parser.add_argument('--near-ms', type=int, default=1000)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    Path(args.out).write_text(markdown(report, args.title, args.near_ms, args.note))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
