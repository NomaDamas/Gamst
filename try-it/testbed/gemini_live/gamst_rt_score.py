"""Score a GAMST realtime run (``gamst_rt.sh``) with the same scorer and columns as the Gemini and Claude arms.

    python try-it/testbed/gemini_live/gamst_rt_score.py NAME gwan|man

Reads ``<try-it>/testbed/gemini_live/gamst/NAME.{jsonl.gz,report.json,wall_s}``. Who of an event = the run's person
binding resolved retroactively (as every GAMST measurement on these clips). Latency = when the Store knew the count
(``committed_ms`` on the capture clock, ``--realtime`` clock shift removed) − the truth occurrence's end, for matched
감스트 events; a second figure uses the event's own end (the counter's decision delay). Writes NAME.score.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eval as le  # noqa: E402
from gamst.eval import multi_person as mp  # noqa: E402
from gamst.eval import pose_dump as D  # noqa: E402

OUT = le.TRYIT / 'testbed' / 'gemini_live' / 'gamst'
CAP = 'temporal.exemplar_events'


def _key(e):
    return (e.get('track_id'), e.get('track_index', e.get('index')), e.get('start_ms'))


def _pct(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    p50 = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return dict(p50_s=round(p50, 3), p95_s=round(values[min(len(values) - 1, int(.95 * len(values)))], 3),
                max_s=round(values[-1], 3), n=len(values))


def score_run(name, clip):
    report = json.loads((OUT / f'{name}.report.json').read_text())
    dump = D.load(OUT / f'{name}.jsonl.gz')
    start = report['start_ms']
    shift = report.get('clock_shift_ms', 0)
    occ = mp.occurrences(dump['events'], start, D.binding_events(dump))
    stated = {}
    for e in (report.get('event_states') or {}).get(CAP) or ():
        if e.get('event') == 'exemplar.matched' and e.get('committed_ms') is not None:
            stated.setdefault(_key(e), e)
    by_key = {_key(e): e for e in dump['events'] if e.get('event') == 'exemplar.matched'}
    events = []
    for o, raw in zip(occ, sorted((e for e in dump['events'] if e.get('event') == 'exemplar.matched'),
                                  key=lambda e: e['start_ms'])):
        pass
    for o in occ:
        e = dict(start=o['start'], end=o['end'], who=le.TARGET if o['who'] == le.TARGET else le.OTHER,
                 track_id=o['track_id'])
        events.append(e)
    # knowledge time of each counted event (matched on track and start)
    known = {}
    for key, raw in by_key.items():
        s = stated.get(key)
        if s is not None:
            known[(raw.get('track_id'), round((raw['start_ms'] - start) / 1000, 3))] = (s['committed_ms'] - shift - start) / 1000
    for e in events:
        e['said_at'] = known.get((e['track_id'], round(e['start'], 3)))
    result = le.score(clip, events)
    truth = le.load_truth(clip)
    g = mp.restrict([e for e in events if e['who'] == le.TARGET], exclude=truth['examples'])
    t = mp.restrict([o for o in truth['occurrences'] if o['who'] == le.TARGET], exclude=truth['examples'])
    pairs = mp.match(g, t)
    decision = _pct([e['said_at'] - e['end'] for e, _ in pairs if e.get('said_at') is not None])
    wall = OUT / f'{name}.wall_s'
    info = le.CLIPS[clip]
    out = dict(name=name, clip=clip, timing=report.get('timing'), realtime=report.get('realtime'),
               frames=report.get('frames'), clip_fps=info['fps'], lost=report.get('lost'),
               complete=report.get('complete'), run_s=report.get('run_s'),
               wall_s=float(wall.read_text()) if wall.exists() else None,
               events=len(events), with_known_time=sum(e['said_at'] is not None for e in events),
               score=result, counter_decision_delay=decision)
    if out['wall_s']:
        out['real_time_factor'] = round(out['wall_s'] / info['duration_s'], 3)
    (OUT / f'{name}.score.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == '__main__':
    print(json.dumps(score_run(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=1))
