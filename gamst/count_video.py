"""A video with a recipe run's count drawn on it, in the style of the coin takes; with the original sound.

    python -m gamst.count_video --report run.json --video in.mp4 --out counted.mp4 --title "뽀또" \
        --count speech.term_events:term.spoken --other speech.term_events:term.other_voice

Only events of the report are drawn, each at the capture time it happened; no model runs here.
``--slow 4`` writes every frame that many times as long (no sound). ``--points`` is an optional JSON list, one
``[[x, y], ...]`` (pixels, or null) per frame, drawn as a trail.

Revision states (a report with ``event_states``, e.g. ``--with tracks``): a counted word appears when it was committed
(a run ``--as-live``: when it was known; otherwise the capture time of its record), orange while it is provisional,
green once final, red and struck out once revoked (it leaves the count then). The panel says how many are final,
provisional and revoked at that moment.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import shutil
import subprocess

from .halo import ACCENT, BG, DIM, GOOD, LINE, MUTED, PANEL, REC, WARN, WHITE, Text

KOREAN = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
WIDTH, HEIGHT, VIEW_W, VIEW_H, SIDE = 1920, 1080, 1376, 774, 1440
HOLD_S = .55


class KoreanText(Text):
    def _font(self, family, size, weight):
        key = (family, size, weight)
        if key not in self._fonts:
            try:
                self._fonts[key] = self._font_cls.truetype(KOREAN, size, index=0 if weight == 'regular' else 6)
            except OSError:
                return super()._font(family, size, weight)
        return self._fonts[key]


def event_time(event):
    """When it was complete: the moment it is counted."""
    for key in ('timestamp_ms', 'end_ms', 'start_ms'):
        if isinstance(event.get(key), (int, float)):
            return event[key]
    return None


def load_events(report, wanted):
    if not wanted:
        return []
    capability, _, name = wanted.partition(':')
    found = [dict(event, t=(event_time(event) - report['start_ms']) / 1000)
             for event in report['events'].get(capability, ()) if event.get('event') == name and event_time(event)]
    return sorted(found, key=lambda event: event['t'])


def load_states(report, wanted):
    """Counted events with revision states: ``t`` (word), ``seen`` (committed) and ``settled`` (revised), in seconds."""
    capability, _, name = wanted.partition(':')
    stated = (report.get('event_states') or {}).get(capability)
    if not stated:
        return None
    known = bool(report.get('paced'))
    at = lambda event, key: None if event.get(key) is None else (event[key] - report['start_ms']) / 1000
    found = []
    for event in stated:
        if event.get('event') != name or not event_time(event):
            continue
        seen = at(event, 'committed_ms' if known else 'capture_ms')
        found.append(dict(event, t=(event_time(event) - report['start_ms']) / 1000,
                          seen=max(seen if seen is not None else 0, (event_time(event) - report['start_ms']) / 1000),
                          settled=at(event, 'settled_ms' if known else 'settled_capture_ms'),
                          first='final' if 'replaces' in event else 'provisional'))
    return sorted(found, key=lambda event: event['seen'])


def state_at(event, t):
    """provisional, final or revoked at video time ``t`` (None before it was committed)."""
    if t < event['seen']:
        return None
    return event['state'] if event['settled'] is not None and t >= event['settled'] else event['first']


STATE_COLOR = {'provisional': WARN, 'final': GOOD, 'revoked': REC, 'uncertain': WARN}
STATE_WORD = {'provisional': '잠정', 'final': '확정', 'revoked': '취소', 'uncertain': '불확실'}


def render(report, video, out, *, title, count, other=None, support=None, count_label='counted', other_label='not counted',
           note='', slow=1, points=None, start=0., end=None):
    import cv2
    import numpy as np
    counted, others = load_events(report, count), load_events(report, other)
    stated = load_states(report, count)
    if stated is not None:
        counted = stated
        others = [event for event in others if not any(item.get('replaces') == event.get('event_id') for item in stated)]
    if support:     # another module's word on each counted event (e.g. whether it made a sound)
        notes = [event for event in report['events'].get(support, ()) if event_time(event)]
        for event in counted:
            near = min(notes, key=lambda item: abs(event_time(item) - event_time(event)), default=None)
            if near is not None and abs(event_time(near) - event_time(event)) <= 50:
                event['support'] = near
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f'cannot open {video}')
    fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    end = duration if end is None else min(end, duration)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise SystemExit('count_video needs ffmpeg')
    sound = ['-ss', str(start), '-t', str(end - start), '-i', str(video), '-map', '0:v', '-map', '1:a?', '-c:a', 'aac',
             '-b:a', '160k'] if slow == 1 else []
    sink = subprocess.Popen([ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s',
                             f'{WIDTH}x{HEIGHT}', '-r', f'{fps / slow:.6f}', '-i', '-', *sound, '-c:v', 'libx264',
                             '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                             str(out)], stdin=subprocess.PIPE)
    text, trail = KoreanText(), deque(maxlen=max(2, round(.25 * fps)))
    marks = [(event, True) for event in counted] + [(event, False) for event in others]
    index = -1
    while True:
        ok, frame = capture.read()
        index += 1
        t = index / fps
        if not ok or t >= end:
            break
        if t < start:
            continue
        h, w = frame.shape[:2]
        trail.append(points[index] if points is not None and index < len(points) else None)
        dots = [dot for dot in trail if dot]
        for a, b in zip(dots, dots[1:]):
            for p, q in zip(a, b):
                cv2.line(frame, tuple(map(round, p)), tuple(map(round, q)), ACCENT, max(2, w // 320), cv2.LINE_AA)
        scale = min(VIEW_W / w, VIEW_H / h)
        view = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
        x0, y0 = 32 + (VIEW_W - view.shape[1]) // 2, 104 + (VIEW_H - view.shape[0]) // 2
        canvas[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = view
        if stated is None:
            now = [event for event in counted if event['t'] <= t]
            since = t - now[-1]['t'] if now else 9
        else:
            states = [(event, state_at(event, t)) for event in counted]
            now = [event for event, state in states if state in ('provisional', 'final')]
            since = t - max((event['seen'] for event in now), default=-9)
            tally = {name: sum(state == name for _, state in states) for name in STATE_WORD}
        rejected = [event for event in others if 0 <= t - event['t'] < HOLD_S]
        flash = since < HOLD_S
        if flash:
            cv2.rectangle(canvas, (x0, y0), (x0 + view.shape[1] - 1, y0 + view.shape[0] - 1), GOOD,
                          max(2, round(14 * (1 - since / HOLD_S))))
            grow = round(60 * max(0, 1 - since / .2))
            text.put(canvas, f'{len(now)}', x0 + view.shape[1] - 40, y0 + 24, 150 + grow, GOOD, 'bold', anchor='right')
        elif rejected:
            cv2.rectangle(canvas, (x0, y0), (x0 + view.shape[1] - 1, y0 + view.shape[0] - 1), DIM, 6)
        text.put(canvas, f'GAMST  ·  {title}', 32, 40, 34, WHITE, 'bold')
        text.put(canvas, (f'{slow}배 느리게' if slow != 1 else '원본 속도'), VIEW_W + 32, 46, 24, MUTED, anchor='right')
        cv2.rectangle(canvas, (SIDE, 104), (WIDTH - 32, 104 + VIEW_H), PANEL, -1)
        text.put(canvas, '지금까지', SIDE + 32, 136, 28, MUTED)
        text.put(canvas, str(len(now)), SIDE + 32, 180, 230, GOOD if flash else WHITE, 'bold')
        text.put(canvas, count_label, SIDE + 32, 420, 28, GOOD, 'bold')
        if stated is not None:
            text.put(canvas, f"확정 {tally['final']}  ·  잠정 {tally['provisional']}  ·  취소 {tally['revoked']}"
                     + (f"  ·  불확실 {tally['uncertain']}" if tally['uncertain'] else ''),
                     SIDE + 32, 452, 22, MUTED)
        text.put(canvas, f'{t:5.2f}초', WIDTH - 64, 136, 28, MUTED, anchor='right')
        appeared = (lambda event: event['seen']) if stated is not None else (lambda event: event['t'])
        feed = sorted(((event, kind) for event, kind in marks if (appeared(event) if kind else event['t']) <= t),
                      key=lambda item: item[0]['t'])[-7:]
        for row, (event, kind) in enumerate(reversed(feed)):
            y = 486 + 52 * row
            state = state_at(event, t) if kind and stated is not None else None
            if state is not None:
                color = STATE_COLOR[state]
                text.put(canvas, STATE_WORD[state], SIDE + 32, y, 28, color, 'bold')
                text.put(canvas, f"{event['t']:.2f}초", SIDE + 130, y, 28, WHITE if state != 'revoked' else DIM)
                if state == 'revoked':
                    cv2.line(canvas, (SIDE + 128, y + 16), (SIDE + 236, y + 16), REC, 3, cv2.LINE_AA)
                text.put(canvas, count_detail(event, state), SIDE + 260, y + 3, 22, MUTED)
                continue
            number = sum(1 for item in counted if item['t'] <= event['t'])
            text.put(canvas, f'{number}번' if kind else '제외', SIDE + 32, y, 28, GOOD if kind else DIM, 'bold')
            text.put(canvas, f"{event['t']:.2f}초", SIDE + 130, y, 28, WHITE if kind else DIM)
            detail = count_detail(event) if kind else other_label
            text.put(canvas, detail, SIDE + 260, y + 3, 22, MUTED if kind else DIM)
        bar_y, bar_w = 940, WIDTH - 64
        cv2.rectangle(canvas, (32, bar_y - 34), (32 + bar_w, bar_y + 34), PANEL, -1)
        place = lambda at: 32 + round(bar_w * min(max((at - start) / (end - start), 0), 1))
        for event, kind in marks:
            if start <= event['t'] < end:
                state = state_at(event, t) if kind and stated is not None else None
                shown = event['t'] <= t if state is None and not (kind and stated is not None) else state is not None
                color = STATE_COLOR[state] if state is not None else (GOOD if kind else DIM) if shown else LINE
                cv2.line(canvas, (place(event['t']), bar_y - (30 if kind else 14)),
                         (place(event['t']), bar_y + (30 if kind else 14)), color, 5 if kind else 3, cv2.LINE_AA)
        cv2.line(canvas, (place(t), bar_y - 34), (place(t), bar_y + 34), WARN, 2)
        text.put(canvas, note, 32, 1000, 24, MUTED)
        sink.stdin.write(canvas.tobytes())
    sink.stdin.close()
    if sink.wait():
        raise SystemExit('ffmpeg failed')
    return dict(out=str(out), counted=len(counted), not_counted=len(others))


def count_detail(event, state=None):
    if state is not None:
        reason = (event.get('revised_by') or {}).get('reason') if state == 'revoked' else None
        score = f"슬롯 {event.get('slot')} 점수 {event['slot_score']:.2f}" if event.get('slot_score') is not None else ''
        return '  '.join(item for item in (score, reason) if item)
    near = event.get('support')
    if near is not None:
        return f"쿵 소리 +{near['delay_ms']}ms" if 'delay_ms' in near else '소리 없음, 동작으로 판정'
    if 'delay_ms' in event:
        return f"+{event['delay_ms']}ms 뒤 충돌음"
    if 'speaker_score' in event:
        return f"목소리 점수 {event['speaker_score']:.2f}"
    return ''


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m gamst.count_video', description=__doc__.split('\n')[0])
    parser.add_argument('--report', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--count', required=True, metavar='CAPABILITY:EVENT')
    parser.add_argument('--other', metavar='CAPABILITY:EVENT')
    parser.add_argument('--support', metavar='CAPABILITY', help='events of another module shown beside each counted one')
    parser.add_argument('--count-label', default='counted')
    parser.add_argument('--other-label', default='not counted')
    parser.add_argument('--note', default='')
    parser.add_argument('--points')
    parser.add_argument('--slow', type=int, default=1)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float)
    args = vars(parser.parse_args(argv))
    report = json.loads(Path(args.pop('report')).read_text())
    points = json.loads(Path(args['points']).read_text()) if args['points'] else None
    print(json.dumps(render(report, args.pop('video'), args.pop('out'), **dict(args, points=points)), ensure_ascii=False))


if __name__ == '__main__':
    main()
