"""Per-word table of a person-term-count run with --with tse on the shop clip (records of ``--store``).

  python scripts/qa_tse.py RUN.db RUN.json

Prints each of the owner's labelled words (try-it/testbed/labels_bbo.md, ends +-0.6 s): the live word, its voice score
on the room audio, whether a TSE stretch covered it, what the extracted voice's recogniser heard there and its score,
the TSE decision, the provisional count and the final decision. Only the counted term's text is printed.
"""
import json
import sqlite3
import sys

PERSON = [12.04, 14.0, 19.74, 20.14, 21.46, 22.30, 28.10, 38.20, 46.64, 47.74]
OTHER = [12.64, 20.34, 20.74, 27.44]
OVERLAPPED = {12.04, 12.64, 19.74, 20.14, 20.34, 20.74, 27.44, 28.10}
TOL = .6


def main(path, report):
    db = sqlite3.connect(path)
    rep = json.load(open(report))
    start = rep['start_ms']
    caps = {}
    for (payload,) in db.execute("select payload from records where kind='observation.signal'"):
        record = json.loads(payload)
        cap = ((record.get('observation') or {}).get('capability') or {}).get('id')
        caps.setdefault(cap, []).append(record)
    value = lambda record: record['observation'].get('value') or {}
    windows = [value(r) for r in caps.get('speech.target_words', [])]
    utterances = [value(r) for r in caps.get('speech.speaker_check_tse', [])] or [value(r) for r in caps.get('speech.speaker_check', [])]
    verdicts = {v['word_id']: v for r in caps.get('speech.word_verdicts', []) for v in value(r).get('verdicts') or ()}
    words = [dict(w, utterance=u.get('utterance_id')) for u in utterances for w in u.get('words') or ()
             if '뽀또' in str(w.get('text', '')) or w.get('basis') == 'tse_heard_only']
    rel = lambda ms: (ms - start) / 1000
    sys.path.insert(0, __file__.rsplit('/scripts/', 1)[0])
    from gamst.providers.overlap_tse import MultiVoice
    spans = {}
    for rule in ('near', 'overlap'):
        detector = MultiVoice(rule=rule)
        for record in sorted(caps.get('audio.nemotron3.diarization', []) + caps.get('audio.speaker_activity', []),
                             key=lambda r: r['timestamp_ms']):
            detector.add(value(record))
        found = detector.scan()
        spans[rule] = found[0] + ([(found[1], detector.known_until_ms)] if found[1] is not None else [])
        flagged = sum(high - low for low, high in spans[rule]) / 1000
        covered = lambda t: any(rel(low) - TOL / 3 <= t <= rel(high) + TOL / 3 for low, high in spans[rule])
        print(f"detector {rule}: {len(spans[rule])} stretches, {flagged:.1f} s flagged; overlapped labelled words covered: "
              f"{sum(covered(t) for t in OVERLAPPED)}/{len(OVERLAPPED)} "
              f"({', '.join(f'{t}' for t in sorted(OVERLAPPED) if covered(t))}); other person words flagged: "
              f"{', '.join(str(t) for t in PERSON if t not in OVERLAPPED and covered(t)) or 'none'}")
    print(f"TSE stretches: {len(windows)}; " + ' '.join(f"{rel(w['start_ms']):.2f}-{rel(w['end_ms']):.2f}"
                                                     f"(took {sum(w['took_ms'].values())} ms, heard {len(w['terms'])})"
                                                     for w in windows))
    print('| truth end | who | overlapped | stretch? | live word end | room score | heard in extracted (end, score) | window score | TSE decision | provisional | final |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    used = set()
    labels = sorted([(t, 'person') for t in PERSON] + [(t, 'other') for t in OTHER])
    pairs = sorted((abs(rel(w['end_ms']) - truth), truth, index) for truth, _ in labels for index, w in enumerate(words)
                   if abs(rel(w['end_ms']) - truth) <= TOL)
    paired, taken = {}, set()
    for _, truth, index in pairs:        # one to one, nearest first
        if truth not in paired and index not in taken:
            paired[truth] = index
            taken.add(index)
    for truth, who in labels:
        near = [words[paired[truth]]] if truth in paired else []
        stretch = [w for w in windows if rel(w['start_ms']) - .2 <= truth <= rel(w['end_ms']) + .2]
        heard = [f"{rel(t['end_ms']):.2f}/{t.get('score')}" for w in stretch for t in w['terms']
                 if abs(rel(t['end_ms']) - truth) <= TOL]
        wscore = ','.join(str(w.get('score')) for w in stretch) or '-'
        if not near:
            print(f"| {truth} | {who} | {'yes' if truth in OVERLAPPED else 'no'} | {'yes' if stretch else 'no'} | - | - | "
                  f"{' '.join(heard) or '-'} | {wscore} | - | not transcribed | - |")
            continue
        w = near[0]
        used.add(id(w))
        tse = w.get('tse') or {}
        verdict = verdicts.get(w.get('word_id'))
        final = '-' if verdict is None else ('counted' if verdict['match'] else f"not ({verdict.get('reason')})")
        print(f"| {truth} | {who} | {'yes' if truth in OVERLAPPED else 'no'} | {'yes' if stretch else 'no'} | "
              f"{rel(w['end_ms']):.2f}{' (tse only)' if w.get('basis') == 'tse_heard_only' else ''} | "
              f"{w.get('mixture_score', w.get('speaker_score'))} | {' '.join(heard) or 'no'} | {wscore} | "
              f"{tse.get('decision', '-')}{' (' + w['reason'] + ')' if w.get('reason') else ''} | "
              f"{'counted' if w.get('match') else 'not'} | {final} |")
    extra = [w for w in words if id(w) not in used]
    for w in extra:
        tse = w.get('tse') or {}
        print(f"| (no label) | - | - | - | {rel(w['end_ms']):.2f} | {w.get('mixture_score', w.get('speaker_score'))} | - | - | "
              f"{tse.get('decision', '-')} | {'counted' if w.get('match') else 'not'} | "
              f"{(lambda v: '-' if v is None else ('counted' if v['match'] else 'not (' + str(v.get('reason')) + ')'))(verdicts.get(w.get('word_id')))} |")
    score = rep.get('score') or {}
    print('score (after revisions):', {k: score.get(k) for k in ('counted', 'matched', 'precision', 'recall')})
    print('score_provisional:', {k: (rep.get('score_provisional') or {}).get(k) for k in ('counted', 'matched', 'precision', 'recall')})


if __name__ == '__main__':
    main(*sys.argv[1:3])
