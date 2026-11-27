"""Score a run's movement events and person binding against the owner's truth of a multi-person clip.

The truth files are those of ``docs/ground-truth/2026-11-28/<clip>/`` (``truth_action.json``, ``truth_identity.json``,
``examples.json``). The rules are those of the label tool's ``score_multi.py`` (the owner's scorer), so numbers agree:

- **Action**, per person (감스트 / everyone else / everyone): GAMST events are matched one to one to the truth occurrences
  of that person by time (greedy in start order; an overlap, or a gap of at most :data:`TOL_S`, the closest start + end
  first). Precision = matched / GAMST events, recall = matched / truth occurrences, count error = GAMST - truth, start and
  end error = mean absolute difference over matched pairs. An event whose who is ``unknown`` is not 감스트.
- **Excluded ranges** (lesson 3): truth occurrences and events that overlap a few-shot example's range are left out of
  both sides, so an example never scores itself.
- **Halves**: :func:`scene_halves` splits a clip at the scene cuts the run's own person binding found (``scene.cut``
  events) and gives the odd and the even scenes; a threshold tuned on one half is tested on the other.
- **Errors** (:func:`breakdown`): each wrong event and each missed occurrence of a person gets one reason: ``identity``
  (the motion was right but the person was wrong), ``split`` (a second event on one occurrence), ``merged`` (one event over
  two occurrences), ``extra`` (no occurrence of anyone there) or ``missed`` (no event there at all).
- **Identity**, per frame (:func:`score_identity`): each labelled box is matched to the run's box of the same moment (IoU
  >= 0.5, within one frame); the run says 감스트 when that track is bound to 감스트 then (bindings resolved retroactively).
"""
from __future__ import annotations

import json
from pathlib import Path

TOL_S = .5
IOU = .5
TARGET = '감스트'
OTHERS = '다른 사람'
EVERYONE = '모두'


# ---------------------------------------------------------------- truth and events

def load_truth(folder):
    """{action, identity, examples} of one clip's truth folder (missing files: None)."""
    folder = Path(folder)

    def read(name):
        path = folder / name
        return json.loads(path.read_text()) if path.is_file() else None
    action = read('truth_action.json')
    if action is not None:
        action = dict(action, occurrences=[item for item in action.get('occurrences') or ()
                                           if (item.get('source') or 'owner') == 'owner'])
    return dict(action=action, identity=read('truth_identity.json'), examples=read('examples.json'))


def example_ranges(examples):
    """[(start, end)] of the owner's few-shot examples, in seconds."""
    return [(float(item['start']), float(item['end'])) for item in (examples or {}).get('examples') or ()]


def occurrences(events, start_ms, bindings=None):
    """The counted occurrences of a run's events, in seconds from ``start_ms``: one per ``exemplar.matched``, ended where
    its ``exemplar.ended`` says (same track and index), with ``who`` resolved from ``bindings`` when given (retroactive)."""
    from ..providers.person_binding import intervals_of, resolve
    intervals = intervals_of(bindings) if bindings is not None else None
    ended = {(item.get('track_id'), item.get('track_index', item.get('index'))): item for item in events
             if item.get('event') == 'exemplar.ended'}
    found = []
    for item in events:
        if item.get('event') != 'exemplar.matched':
            continue
        end = ended.get((item.get('track_id'), item.get('track_index', item.get('index'))), item)
        who = item.get('who') or 'unknown'
        if intervals is not None:
            who = resolve(dict(track_id=item.get('track_id'), end_ms=item['end_ms']), intervals) or 'unknown'
        found.append(dict(start=(item['start_ms'] - start_ms) / 1000, end=(max(end['end_ms'], item['end_ms']) - start_ms) / 1000,
                          who=who, track_id=item.get('track_id'), evidence=item.get('evidence'),
                          distance=item.get('distance'), method=item.get('method')))
    return sorted(found, key=lambda item: item['start'])


def _overlaps(item, ranges):
    return any(min(item['end'], end) > max(item['start'], start) for start, end in ranges)


def _inside(item, ranges):
    middle = (item['start'] + item['end']) / 2
    return any(start <= middle < end for start, end in ranges)


def restrict(items, *, exclude=(), keep=None):
    """Items that overlap no ``exclude`` range and whose middle is inside a ``keep`` range (None: everywhere)."""
    return [item for item in items if not _overlaps(item, exclude) and (keep is None or _inside(item, keep))]


# ---------------------------------------------------------------- action

def match(gamst, truth, tol=TOL_S):
    """[(event, occurrence)]: greedy one to one by time overlap (then the closest start + end), within ``tol``."""
    pairs, used = [], set()
    for event in sorted(gamst, key=lambda item: item['start']):
        best = None
        for index, item in enumerate(truth):
            if index in used:
                continue
            gap = max(item['start'] - event['end'], event['start'] - item['end'], 0.)
            if gap <= tol:
                distance = abs(event['start'] - item['start']) + abs(event['end'] - item['end'])
                if best is None or distance < best[0]:
                    best = (distance, index)
        if best:
            used.add(best[1])
            pairs.append((event, truth[best[1]]))
    return pairs


def _mine(person):
    if person == TARGET:
        return lambda who: who == TARGET
    if person == OTHERS:
        return lambda who: who != TARGET
    return lambda who: True


def score_action(gamst, truth, *, exclude=(), keep=None, tol=TOL_S):
    """{person: {truth, gamst, error, matched, precision, recall, start_err_s, end_err_s}} for 감스트, others, everyone."""
    gamst, truth = restrict(gamst, exclude=exclude, keep=keep), restrict(truth, exclude=exclude, keep=keep)
    out = {}
    for person in (TARGET, OTHERS, EVERYONE):
        mine = _mine(person)
        t = [item for item in truth if mine(item['who'])]
        g = [item for item in gamst if mine(item['who'])]
        pairs = match(g, t, tol)
        n = len(pairs)
        out[person] = dict(
            truth=len(t), gamst=len(g), error=len(g) - len(t), matched=n,
            precision=round(n / len(g), 4) if g else None, recall=round(n / len(t), 4) if t else None,
            start_err_s=round(sum(abs(a['start'] - b['start']) for a, b in pairs) / n, 3) if n else None,
            end_err_s=round(sum(abs(a['end'] - b['end']) for a, b in pairs) / n, 3) if n else None)
    out['unknown_events'] = sum(item['who'] == 'unknown' for item in gamst)
    return out


def breakdown(gamst, truth, *, person=TARGET, exclude=(), keep=None, tol=TOL_S):
    """{wrong: {identity, split, extra}, missed: {identity, merged, missed}} of ``person``'s events and occurrences.

    For everyone (``person`` = :data:`EVERYONE`) there is no identity reason. A wrong event of the person is ``identity``
    when it matches an unmatched occurrence of someone else, ``split`` when it overlaps an occurrence of the person that
    another event already matched, else ``extra``. A missed occurrence is ``identity`` when an event of someone else
    (or unknown) is there, ``merged`` when an event matched to another occurrence covers it too, else ``missed``."""
    gamst, truth = restrict(gamst, exclude=exclude, keep=keep), restrict(truth, exclude=exclude, keep=keep)
    mine = _mine(person)
    g = [item for item in gamst if mine(item['who'])]
    t = [item for item in truth if mine(item['who'])]
    pairs = match(g, t, tol)
    matched_events = {id(a) for a, _ in pairs}
    matched_truth = {id(b) for _, b in pairs}
    others_t = [item for item in truth if not mine(item['who'])]
    others_g = [item for item in gamst if not mine(item['who'])]
    # who else's occurrences are matched by who else's events (by time alone)
    other_pairs = {id(b) for _, b in match(others_g, others_t, tol)}

    def near(a, b):
        return max(b['start'] - a['end'], a['start'] - b['end'], 0.) <= tol

    def overlap(a, b):
        return min(a['end'], b['end']) > max(a['start'], b['start'])
    wrong = dict(identity=[], split=[], extra=[])
    free_others = [item for item in others_t if id(item) not in other_pairs]
    for event in g:
        if id(event) in matched_events:
            continue
        if person != EVERYONE and any(near(event, item) for item in free_others):
            wrong['identity'].append(event)
        elif any(overlap(event, item) or near(event, item) for item in t if id(item) in matched_truth):
            wrong['split'].append(event)
        else:
            wrong['extra'].append(event)
    missed = dict(identity=[], merged=[], missed=[])
    for item in t:
        if id(item) in matched_truth:
            continue
        if person != EVERYONE and any(near(event, item) for event in others_g):
            missed['identity'].append(item)
        elif any(overlap(event, item) for event in g if id(event) in matched_events):
            missed['merged'].append(item)
        else:
            missed['missed'].append(item)
    return dict(wrong={key: len(value) for key, value in wrong.items()},
                missed={key: len(value) for key, value in missed.items()},
                wrong_at=[dict(reason=key, start=round(item['start'], 2), end=round(item['end'], 2), who=item['who'])
                          for key, value in wrong.items() for item in value],
                missed_at=[dict(reason=key, start=round(item['start'], 2), end=round(item['end'], 2))
                           for key, value in missed.items() for item in value])


# ---------------------------------------------------------------- held-out halves

def scenes(bindings, start_ms, duration_s):
    """[(start, end)] seconds of the scenes between the run's own ``scene.cut`` events (their start times)."""
    cuts = sorted({(item['start_ms'] - start_ms) / 1000 for item in bindings or ()
                   if item.get('event') == 'scene.cut' and type(item.get('start_ms')) in (int, float)})
    edges = [0.] + [cut for cut in cuts if 0 < cut < duration_s] + [float(duration_s)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def scene_halves(scene_list):
    """(odd, even): the 1st, 3rd, ... scenes and the 2nd, 4th, ... scenes."""
    return scene_list[0::2], scene_list[1::2]


def time_halves(duration_s, block_s):
    """(odd, even) blocks of ``block_s`` seconds, alternating (for a clip with too few scene cuts)."""
    blocks, at = [], 0.
    while at < duration_s:
        blocks.append((at, min(duration_s, at + block_s)))
        at += block_s
    return blocks[0::2], blocks[1::2]


# ---------------------------------------------------------------- identity

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0, x1, y1 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.


def score_identity(boxes, bindings, start_ms, truth, *, fps=30.):
    """Per-frame and per-segment precision and recall of 감스트 (the rules of ``score_multi.py``).

    ``boxes``: [(seconds, track_id, (x, y, w, h))] of the run; ``bindings``: its person binding events."""
    from ..providers.person_binding import intervals_of, resolve
    intervals = intervals_of(bindings)
    frames = {}
    for t, track, box in boxes:
        frames.setdefault(round(t * fps), []).append((box, track))

    def who(track, t):
        return resolve(dict(track_id=track, end_ms=start_ms + t * 1000), intervals)
    tp = fp = fn = tn = 0
    rows = []
    for item in (truth or {}).get('segments') or ():
        if item.get('label') not in ('G', 'O'):
            continue
        positive, said = item['label'] == 'G', []
        for row in item.get('boxes') or ():
            k, best = round(row[0] * fps), None
            for kk in (k, k - 1, k + 1):
                for box, track in frames.get(kk, ()):
                    value = iou(row[1:], box)
                    if value >= IOU and (best is None or value > best[0]):
                        best = (value, track)
                if best:
                    break
            predicted = best is not None and who(best[1], row[0]) == TARGET
            said.append(predicted)
            tp += positive and predicted
            fn += positive and not predicted
            fp += (not positive) and predicted
            tn += (not positive) and not predicted
        rows.append((positive, sum(said) * 2 > len(said) if said else False))
    for item in (truth or {}).get('added') or ():
        if item.get('label') != 'G':
            continue
        said = [any(who(track, k / fps) == TARGET for _, track in frames.get(k, ()))
                for k in range(round(item['start'] * fps), round(item['end'] * fps) + 1)]
        tp += sum(said)
        fn += len(said) - sum(said)
        rows.append((True, sum(said) * 2 > len(said) if said else False))

    def ratio(a, b):
        return round(a / b, 4) if b else None
    stp = sum(1 for p, m in rows if p and m)
    sfp = sum(1 for p, m in rows if not p and m)
    sfn = sum(1 for p, m in rows if p and not m)
    return dict(frames=dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn)),
                segments=dict(n=len(rows), tp=stp, fp=sfp, fn=sfn, precision=ratio(stp, stp + sfp),
                              recall=ratio(stp, stp + sfn)))


# ---------------------------------------------------------------- tables

def pct(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def table_row(name, scores, person):
    a = scores[person]
    sec = (lambda x: '—' if x is None else f'{x:.2f}')
    return (f"| {name} | {a['truth']} | {a['gamst']} | {a['error']:+d} | {a['matched']} | {pct(a['precision'])} | "
            f"{pct(a['recall'])} | {sec(a['start_err_s'])} | {sec(a['end_err_s'])} |")


TABLE_HEAD = ('| 방법 | 정답 | GAMST | 차이 | 짝 | 정밀도 | 재현율 | 시작오차(초) | 끝오차(초) |\n'
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|')
