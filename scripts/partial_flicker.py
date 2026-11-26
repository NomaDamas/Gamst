"""How a debounce of partial hypotheses would have done on runs already made (counts only; no transcript is read).

    python scripts/partial_flicker.py REPORT [REPORT ...] [--revisions 1 2 3 4] [--ms 0 300 600]

Reads the partial trace of the speaker check (``pipeline.modules.speaker_check.state.partial_trace``: per partial its
utterance, ``heard_to_ms`` and how many occurrences of each term it held) and the final occurrences per utterance
(the report's ``event_states``), and replays :class:`gamst.providers.partial_terms.PartialSpotter`'s rule for each
setting: an occurrence (term, k) is heard once it was in N partials in a row, or for T ms. Prints per setting: heard,
withdrawn (the final of its turn has fewer occurrences of the term than k + 1), flickers (appeared, then left before it
was heard) and the delay from the final's word end to the partial that made it heard (p50 / p95 / max, seconds).
"""
import argparse
import json


def finals(report, capability='speech.term_events'):
    """{utterance_id: {term: [end_ms, ...]}} of what the final hearing (L1) held."""
    found = {}
    for event in (report.get('event_states') or {}).get(capability) or ():
        if event.get('replaces') or event.get('decided_by') == 'verification' or not event.get('utterance_id'):
            continue
        if event.get('committed_state') == 'heard':
            step = next((step for step in event.get('history') or () if step.get('by') == 'l1'), None)
            if step is None or step['state'] == 'revoked':
                continue          # withdrawn, or another voice's (that word is its own term.other_voice event)
        found.setdefault(event['utterance_id'], {}).setdefault(event['term'], []).append(event['end_ms'])
    return {key: {term: sorted(ends) for term, ends in terms.items()} for key, terms in found.items()}


def replay(trace, final, revisions, ms):
    """The spotter's rule (gamst/providers/partial_terms.py) over a trace: once per (utterance, term, k)."""
    heard, withdrawn, flickers, delays = 0, 0, 0, []
    held, emitted, present = {}, set(), {}      # (utterance, term, k) -> [first heard_to, streak]
    for key, at, counts in trace:
        now = {(key, term, k) for term, number in counts.items() for k in range(number)}
        for place in [place for place in present.get(key, ()) if place not in now]:
            flickers += place not in emitted
            held.pop(place, None)
        present[key] = now
        for place in now:
            item = held.setdefault(place, [at, 0])
            item[1] += 1
            if place in emitted or not (item[1] >= revisions or ms is not None and at - item[0] >= ms):
                continue
            emitted.add(place)
            heard += 1
            ends = final.get(key, {}).get(place[1], [])
            if len(ends) <= place[2]:
                withdrawn += 1
            else:
                delays.append((at - ends[place[2]]) / 1000)
    delays.sort()
    pick = lambda share: round(delays[min(len(delays) - 1, int(share * len(delays)))], 2) if delays else None
    return dict(heard=heard, withdrawn=withdrawn, flickers=flickers, delay_p50=pick(.5), delay_p95=pick(.95),
                delay_max=round(delays[-1], 2) if delays else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--revisions', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--ms', type=float, nargs='+', default=[None])
    parser.add_argument('--module', default='speaker_check')
    args = parser.parse_args(argv)
    for path in args.reports:
        report = json.load(open(path))
        state = report['pipeline']['modules'][args.module].get('state') or {}
        trace, final = state.get('partial_trace') or [], finals(report)
        for revisions in args.revisions:
            for ms in args.ms:
                print(json.dumps(dict(report=path.rsplit('/', 1)[-1], partials=len(trace), revisions=revisions, ms=ms,
                                      **replay(trace, final, revisions, ms))))


if __name__ == '__main__':
    main()
