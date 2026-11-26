"""Score person-term-count runs against the owner's labels (``truth_owner.json`` of the labelling tool).

    python scripts/score_owner_clips.py REPORT TRUTH [--voice VOICE.json] [--exclude segments|window]

Prints one JSON object: what L1 counted the moment it committed each word (the HUD's instant count, ``l1``), the count
after every correction (``final``), and the error types (gamst.recipes.bench.error_types). Words inside the voice
reference are not scored (the reference holds their sound): the owner-marked range, or for the automatic main speaker
the segments that were enrolled (``segments``, the default, owner decision 2026-11-26) or its whole window (``window``:
stricter, a live run has that voice only once the window has passed). A report of a run that enrolled its own sample
(``voice_sample``) names the excluded seconds itself. No transcript is read or kept.
"""
import argparse
import json

from gamst.recipes.bench import error_types, live_events, view, EVENT


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
    return dict(excluded=truth['exclude'], l1=view(truth, live_events(report), start), final=view(truth, final, start),
                errors=error_types(truth, report), replaced=len(replaced))


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
