"""Score person-term-count runs against the owner's labels (``truth_owner.json`` of the labelling tool).

    python scripts/score_owner_clips.py REPORT TRUTH [--voice VOICE.json] [--exclude segments|window]

Prints one JSON object: what entered the count first, at once (``heard``: words heard in the live recogniser's partials,
and those L1 committed without being heard), what L1 counted the moment it committed each word (``l1``), the count
after every correction (``final``), the error types (gamst.recipes.bench.error_types), how many heard words the final
withdrew (``withdrawn``) and, per labelled word counted at once, the time from its end (the owner's label) to the moment
the HUD first counted it (``latency_s``: ``person`` words and ``all`` labelled words; p50 / p95 / max). Words inside the voice
reference are not scored (the reference holds their sound): the owner-marked range, or for the automatic main speaker
the segments that were enrolled (``segments``, the default, owner decision 2026-11-26) or its whole window (``window``:
stricter, a live run has that voice only once the window has passed). A report of a run that enrolled its own sample
(``voice_sample``) names the excluded seconds itself. No transcript is read or kept.
"""
import argparse
import json

from gamst.recipes.bench import _labels, error_types, first_count_ms, heard_events, live_events, match, view, withdrawn, EVENT


def exclusions(voice, mode='segments'):
    origin = (voice or {}).get('enrolled_from') or {}
    if origin.get('kind') == 'range':
        return [dict(from_s=origin['from_s'], to_s=origin['to_s'], why='owner-marked voice range')]
    if origin.get('kind') == 'main_speaker':
        if mode == 'segments':
            return [dict(from_s=low, to_s=high, why='enrolled main-speaker segment') for low, high in origin['segments_s']]
        low, high = origin['window_s']
        return [dict(from_s=low, to_s=high, why='main-speaker window')]
    return []


def score(report, truth, voice=None, mode='segments'):
    own = (report.get('voice_sample') or {}).get('excluded_s')
    out = [dict(from_s=low, to_s=high, why='the run\'s voice sample') for low, high in own] if own and mode == 'segments' \
        else exclusions(voice or dict(enrolled_from=report.get('voice_sample')), mode)
    truth = dict(truth, exclude=[*truth.get('exclude', ()), *out])
    stated = (report.get('event_states') or {}).get('speech.term_events') or []
    replaced = {event['replaces']: event for event in stated if event.get('replaces')}
    final = [event for event in stated if event.get('event') == EVENT and event.get('state') in ('provisional', 'final')]
    start = report['start_ms']
    first = heard_events(report)
    delays = {'person': [], 'all': []}
    points = [point[0] for point in _labels(truth)[0]]
    for event, label, slot in match(truth, first, start):
        if slot is None or label == 'excluded':
            continue
        at = first_count_ms(event)
        if at is None:
            continue
        if event.get('committed_state') != 'heard':     # the HUD never shows a word before its end
            at = max(at, event['end_ms'])
        delay = (at - start - points[slot]) / 1000
        delays['all'].append(delay)
        if label == 'person':
            delays['person'].append(delay)
    heard = [event for event in stated if event.get('committed_state') == 'heard']
    return dict(excluded=truth['exclude'], heard=view(truth, first, start), l1=view(truth, live_events(report), start),
                final=view(truth, final, start), errors=error_types(truth, report), replaced=len(replaced),
                heard_words=len(heard), withdrawn=sum(withdrawn(event) for event in heard),
                latency_s={name: spread(values) for name, values in delays.items()})


def spread(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda share: ordered[min(len(ordered) - 1, int(share * len(ordered)))]
    return dict(p50=round(pick(.5), 2), p95=round(pick(.95), 2), max=round(ordered[-1], 2), n=len(ordered))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('report')
    parser.add_argument('truth')
    parser.add_argument('--voice')
    parser.add_argument('--exclude', choices=['window', 'segments'], default='segments')
    args = parser.parse_args(argv)
    report, truth = json.load(open(args.report)), json.load(open(args.truth))
    voice = json.load(open(args.voice)) if args.voice else None
    print(json.dumps(score(report, truth, voice, args.exclude), ensure_ascii=False))


if __name__ == '__main__':
    main()
