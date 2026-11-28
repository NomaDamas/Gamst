"""Collect the three arms' measured results into one table (markdown), the same columns for each.

    python try-it/testbed/gemini_live/compare.py [--json OUT.json]

Reads (all outside the repository): GAMST ``gamst/*.score.json`` (gamst_rt_score.py), Gemini ``runs/*/result.json``
(harness.py), Claude ``claude/*/result.r*.json`` (claude_arm.py). Scores are 감스트's (the target person) with the
owner's scorer; 관제탑 excludes the 7 example ranges (39 of 46), as every GAMST number on Main.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eval as le  # noqa: E402

BASE = le.TRYIT / 'testbed' / 'gemini_live'
G = le.TARGET


def _row(arm, clip, cond, run, score, *, wall_s, fps_seen, tok_in, tok_out, usd, latency=None, note=''):
    s = score[G]
    dur = le.CLIPS[clip]['duration_s']
    return dict(arm=arm, clip=clip, cond=cond, run=run, truth=s['truth'], said=s['said'], matched=s['matched'],
                precision=s['precision'], recall=s['recall'], f1=s['f1'] if s['f1'] is not None else 0.0,
                start_err=s.get('start_err_s', s.get('time_err_s')), end_err=s.get('end_err_s', s.get('time_err_s')),
                latency_p50=(latency or {}).get('p50_s'), latency_max=(latency or {}).get('max_s'),
                wall_s=wall_s, fps_seen=fps_seen, tok_in=tok_in, tok_out=tok_out, usd=usd,
                usd_per_min=None if usd is None else usd / (dur / 60), usd_per_hour=None if usd is None else usd / dur * 3600,
                usd_per_correct=None if usd is None or not s['matched'] else usd / s['matched'], note=note)


def gamst_rows():
    rows = []
    for path in sorted((BASE / 'gamst').glob('*.score.json')):
        d = json.loads(path.read_text())
        frames = d.get('frames') or 0
        lost = sum((v or {}).get('frames', 0) for v in (d.get('lost') or {}).values()) if isinstance(d.get('lost'), dict) else 0
        rows.append(_row('GAMST', d['clip'], d['name'], 1, d['score'], wall_s=d.get('wall_s'),
                         fps_seen=round(frames / le.CLIPS[d['clip']]['duration_s'], 1) if frames else None,
                         tok_in=0, tok_out=0, usd=0.0, latency=(d['score'][G].get('latency') or None),
                         note=f"realtime, RTF {d.get('real_time_factor')}, lost {lost} frames"))
    return rows


def gemini_rows():
    rows = []
    for path in sorted((BASE / 'runs').glob('*/result.json')):
        d = json.loads(path.read_text())
        if path.parent.name.startswith('smoke') or d.get('errors'):
            continue
        sp = d['spent']
        common = dict(wall_s=d['wall_s'], fps_seen=d['frames_per_clip_s'], tok_in=sp['tokens_in'], tok_out=sp['tokens_out'],
                      usd=sp['usd'])
        cond = f"{d['mode']}.{d['lang']}"
        if d['mode'] == 'rt' and d['realtime']['score']:
            rows.append(_row('Gemini Live', d['clip'], cond + ' 실시간', int(d['condition'][-1]), d['realtime']['score'],
                             latency=d['realtime']['score'][G].get('latency'), **common))
        rows.append(_row('Gemini Live', d['clip'], cond + ' 사후', int(d['condition'][-1]), d['after']['score'],
                         **dict(common, usd=None if d['mode'] == 'rt' else sp['usd']),
                         note='same run as the realtime row' if d['mode'] == 'rt' else ''))
    return rows


def claude_rows():
    rows = []
    for path in sorted((BASE / 'claude').glob('*/result.r*.json')):
        d = json.loads(path.read_text())
        u = d['usage']
        rows.append(_row('Claude Code', d['clip'], f"{d['fps']:g} fps", d['run'], d['score'], wall_s=u.get('wall_s'),
                         fps_seen=d['fps'], tok_in=(u.get('input_tokens') or 0) + (u.get('cache_read_tokens') or 0) +
                         (u.get('cache_write_tokens') or 0), tok_out=u.get('output_tokens'), usd=d['usd'],
                         note=f"cache read {u.get('cache_read_tokens'):,} / write {u.get('cache_write_tokens'):,}"))
    return rows


def fmt(v, digits=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{digits}f}'
    return f'{v:,}' if isinstance(v, int) else str(v)


def table(rows):
    head = ('| 팔 | 영상 | 조건 | 회 | 맞음/말함/정답 | 정밀 | 재현 | F1 | 시작/끝 오차(초) | 지연 p50 / 최대(초) | 걸린 시간(초) | '
            '초당 장수 | 토큰 입력 / 출력 | USD | KRW | USD/분 | USD/시간 | USD/맞힌 1개 |')
    out = [head, '|' + '---|' * 18]
    for r in rows:
        krw = None if r['usd'] is None else r['usd'] * le.USD_KRW
        out.append('| ' + ' | '.join([
            r['arm'], le.CLIPS[r['clip']]['name'], r['cond'], str(r['run']), f"{r['matched']}/{r['said']}/{r['truth']}",
            fmt(r['precision'], 2), fmt(r['recall'], 2), fmt(r['f1'], 3),
            f"{fmt(r['start_err'], 2)} / {fmt(r['end_err'], 2)}", f"{fmt(r['latency_p50'], 2)} / {fmt(r['latency_max'], 2)}",
            fmt(r['wall_s'], 1), fmt(r['fps_seen'], 1), f"{fmt(r['tok_in'])} / {fmt(r['tok_out'])}", fmt(r['usd'], 4),
            fmt(None if krw is None else round(krw)), fmt(r['usd_per_min'], 4), fmt(r['usd_per_hour'], 2),
            fmt(r['usd_per_correct'], 4)]) + ' |')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json')
    args = ap.parse_args(argv)
    rows = gamst_rows() + gemini_rows() + claude_rows()
    order = {'gwan': 0, 'man': 1, 'shot': 2}
    rows.sort(key=lambda r: (order[r['clip']], r['arm'], r['cond'], r['run']))
    print(table(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
