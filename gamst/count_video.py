"""A video with a recipe run's count drawn on it, in the style of the coin takes; with the original sound.

    python -m gamst.count_video --report run.json --video in.mp4 --out counted.mp4 --title "뽀또" \
        --count speech.term_events:term.spoken --other speech.term_events:term.other_voice

Only events of the report are drawn, each at the capture time it happened; no model runs here.
``--slow 4`` writes every frame that many times as long (no sound). ``--hold-settled`` keeps the last picture (silent)
after the video until every word has settled: the verification lane decides the last stretch only after the input ends. ``--points`` is an optional JSON list, one
``[[x, y], ...]`` (pixels, or null) per frame, drawn as a trail.

Revision states (a report with ``event_states``, e.g. ``--with verification``; owner decision 26). At video time t the
HUD shows exactly what was known at t: each event's ``history`` says when every step was decided on the capture clock
(a run ``--as-live``: when it was known live; otherwise the capture time of its record). The big number is the count
as it stood then: a word the live recogniser's partial hypothesis held steadily enters it the moment that was known
(``heard``: pale, dashed "+1 들림"; gamst/providers/partial_terms.py); a word the live recogniser (L1) counted enters
it the moment L1 committed it (orange, provisional), and a heard word turns provisional then (same word, no second +1)
or leaves the count ("-1 취소 #n 부분 인식 철회");
a later lane's decision (catch-up, TSE, verification) changes it only at the moment it was decided, with a visible
correction in Korean: "+1 추가 #12" (a word given to the person later: another voice's at first, or heard again),
"-1 취소 #9" with the reason (red, struck out), "-1 보류" (violet, uncertain: shown, not counted), "확정" (green).
The number that drops is shown struck out for a moment, then the new one. Words are numbered in the order they
entered the count. The panel says how many are final, provisional, revoked and uncertain at that moment, each in its
colour; another voice's words are listed as "제외". ``--hold-settled`` keeps the last picture until the last
decision has been shown.
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
NANUM = ('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf')  # Linux
WIDTH, HEIGHT, VIEW_W, VIEW_H, SIDE = 1920, 1080, 1376, 774, 1440
HOLD_S = .55
HOLD_MAX_S = 30.


EN_PHRASES = [
    ('영상 끝 · 검증이 마지막 말을 정하는 중', 'Video ended · verifying the last words'),
    ('들림 즉시 · 잠정 L1 · 정정은 결정된 순간에', 'Heard at once · L1 provisional · corrections when decided'),
    ('말한 순간 기준 · 최종 결과', 'Shown when said · final result'),
    ('원본 속도', 'Original speed'), ('지금까지', 'Count so far'), ('배 느리게', 'x slower'),
    ('부분 인식 철회', 'partial withdrawn'), ('다른 사람 목소리', "another person's voice"), ('다시 들어서 찾음', 'found on re-listening'),
    ('감스트 목소리로 정정', "corrected to Gamst's voice"), ('불확실에서 확정', 'uncertain → final'), ('검증 결정', 'verified'),
    ('검증 통과', 'verified'), ('부분 인식', 'partial'), ('실시간', 'live'), ('목소리 점수', 'voice score'),
    ('+1 들림', '+1 heard'), ('+1 추가', '+1 added'), ('-1 취소', '-1 revoked'), ('-1 보류', '-1 held'),
    ('찾기 모드', 'search mode'), ('인식기끼리 다름', 'recognisers disagree'), ('목소리 점수 낮음', 'low voice score'),
    ('인식 신뢰도 낮음', 'low recogniser confidence'), ('세지 않았을 말', 'would not have counted'), ('후보', 'candidate'),
    ('목소리 겹침', 'voices overlap'), ('말소리 없음', 'no speech'), ('다시 들어도 없음', 'not heard again'),
    ('화자 없음', 'no speaker'), ('화자 불명', 'unknown speaker'), ('인식기 하나만 들음', 'heard by one recogniser'),
    ('확정', 'Final'), ('잠정', 'Provisional'), ('들림', 'Heard'), ('취소', 'Revoked'), ('불확실', 'Uncertain'), ('제외', 'Excluded'),
]


def to_english(text):
    """The HUD's Korean words in English (for an English version of a video); numbers and names are kept."""
    import re
    text = re.sub(r'(\d+(?:\.\d+)?)초에 한 말', r'said at \1 s', str(text))
    text = re.sub(r'외 (\d+)건', r'+\1 more', text)
    for korean, english in EN_PHRASES:
        text = text.replace(korean, english)
    text = re.sub(r'(\d+(?:\.\d+)?)초', r'\1 s', text)
    return re.sub(r'(\d+)번', r'#\1', text)


class KoreanText(Text):
    def _font(self, family, size, weight):
        key = (family, size, weight)
        if key not in self._fonts:
            try:
                import os
                if os.path.exists(KOREAN):
                    self._fonts[key] = self._font_cls.truetype(KOREAN, size, index=0 if weight == 'regular' else 6)
                else:       # Linux: Nanum Gothic (apt fonts-nanum)
                    self._fonts[key] = self._font_cls.truetype(NANUM[0 if weight == 'regular' else 1], size)
            except OSError:
                return super()._font(family, size, weight)
        return self._fonts[key]


class EnglishText(KoreanText):
    """The same HUD with its Korean words in English; given strings (title, labels, note) are used as they are."""

    def put(self, image, text, *args, **kwargs):
        return super().put(image, to_english(text), *args, **kwargs)


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
                          first=event.get('committed_state') or ('final' if 'replaces' in event else 'provisional')))
    return sorted(found, key=lambda event: event['seen'])


def state_at(event, t):
    """provisional, final or revoked at video time ``t`` (None before it was committed)."""
    if t < event['seen']:
        return None
    return event['state'] if event['settled'] is not None and t >= event['settled'] else event['first']


UNCERTAIN = (230, 130, 190)     # violet (BGR): neither counted nor revoked
HEARD = (200, 222, 236)         # pale (BGR): heard in a partial hypothesis, counted, not yet committed by L1
STATE_COLOR = {'heard': HEARD, 'provisional': WARN, 'final': GOOD, 'revoked': REC, 'uncertain': UNCERTAIN}
STATE_WORD = {'heard': '들림', 'provisional': '잠정', 'final': '확정', 'revoked': '취소', 'uncertain': '불확실'}
REASON_WORD = {'other_speaker': '다른 사람 목소리', 'another_voice': '다른 사람 목소리', 'heard_by_one': '인식기 하나만 들음',
               'no_speech': '말소리 없음', 'not_heard': '다시 들어도 없음', 'uncertain_overlap': '목소리 겹침',
               'no_speaker': '화자 없음', 'speaker_unidentified': '화자 불명', 'late': '늦게 도착',
               'partial_withdrawn': '부분 인식 철회', 'other_voice': '다른 사람 목소리', 'contested': '인식기끼리 다름',
               'below_threshold': '목소리 점수 낮음', 'vetoed': '인식 신뢰도 낮음', 'not_counted': '세지 않았을 말'}
FLAG_WORD = {'may_be_number': '숫자일 수 있음'}



CHANGE_S = 2.4      # how long a correction stays on the screen after it was decided
L1_BY = (None, 'live')


def _steps(event, start_ms, known):
    """``[(at_s, state, step)]`` of one stated event: its ``history`` (``decided_ms``), else the older fields."""
    at = lambda value: None if value is None else (value - start_ms) / 1000
    history = event.get('history')
    if history:
        return [(at(step.get('decided_ms', step.get('capture_ms'))), step['state'], step) for step in history
                if at(step.get('decided_ms', step.get('capture_ms'))) is not None]
    seen = at(event.get('committed_ms' if known else 'capture_ms'))
    first = event.get('committed_state') or ('final' if 'replaces' in event else 'provisional')
    steps = [(seen if seen is not None else 0, first, {})]
    settled = at(event.get('settled_ms' if known else 'settled_capture_ms'))
    if settled is not None:
        steps.append((settled, event['state'], dict(event.get('revised_by') or {})))
    return steps


def hud_timeline(report, count, other=None, at_spoken=False):
    """What the HUD knows at each moment, from a report's ``event_states`` (None without them).

    One item per word: ``t`` (its end, seconds), ``steps`` ``[(at_s, state, step)]`` with ``state`` one of provisional,
    final, revoked, uncertain or ``other`` (another voice's word: shown, never counted), each at the moment it was
    decided on the capture clock (when it appears live), and ``number`` (the order in which words entered the count).
    A word another voice was first given that a later lane gives to the person (an event that ``replaces`` it) is one
    item: shown as another voice's, then added. ``changes``: every moment a word's count or state changed after it was
    first shown, ``(at_s, delta, kind, item)`` with ``kind`` one of ``live`` (L1 counted it), ``added``, ``final``,
    ``revoked``, ``uncertain``; nothing is applied before it was decided.

    ``at_spoken``: a replay for presentation, not what was known live: each word shows only its final state, at the moment
    it was said (the video must say so; ``render`` writes a line).
    """
    capability, _, name = count.partition(':')
    stated = (report.get('event_states') or {}).get(capability)
    if not stated:
        return None
    other_name = other.partition(':')[2] if other and other.partition(':')[0] == capability else None
    known, start = bool(report.get('paced')), report['start_ms']
    replaced = {event['replaces']: event for event in stated if event.get('replaces')}
    items = []
    for event in stated:
        if event.get('replaces') or not event_time(event) or event.get('event') not in (name, other_name):
            continue
        t = (event_time(event) - start) / 1000
        steps = []
        for at, state, step in _steps(event, start, known):
            if event['event'] == other_name:
                if event['event_id'] in replaced and len(steps):
                    continue      # what replaced it carries on
                state = 'other'
            steps.append((at if state == 'heard' else max(at, t), state, step))
        later = replaced.get(event['event_id'])
        if later is not None:
            steps += [(max(at, t), state, step) for at, state, step in _steps(later, start, known)]
        if not steps:
            continue
        if at_spoken:      # presentation replay: the settled state, drawn when the word was said
            steps = [(t, sorted(steps, key=lambda item: item[0])[-1][1], sorted(steps, key=lambda item: item[0])[-1][2])]
        items.append(dict(event, t=t, steps=sorted(steps, key=lambda item: item[0]), now=later or event))
    items.sort(key=lambda item: (item['steps'][0][0], item['t']))
    changes, number = [], 0
    for item in sorted(items, key=lambda item: item['t']):
        item['number'] = None
    moments = sorted(((at, item['t'], index, position) for position, item in enumerate(items)
                      for index, (at, _, _) in enumerate(item['steps'])))
    for at, _, index, position in moments:
        item = items[position]
        state, step = item['steps'][index][1:]
        before = item['steps'][index - 1][1] if index else None
        was, now = before in COUNTED_STATES, state in COUNTED_STATES
        if now and item['number'] is None:
            number += 1
            item['number'] = number
        if index == 0:
            if now:
                changes.append((at, 1, 'live' if at_spoken else 'heard' if state == 'heard' else 'live'
                                if state == 'provisional' and step.get('by') in L1_BY else 'added', item))
            continue
        if state == before or before == 'heard' and state == 'provisional':
            continue      # a heard word L1 committed: the same word, already counted
        kind = 'added' if now and not was else 'final' if state == 'final' else state if was else None
        if kind is not None:
            changes.append((at, int(now) - int(was), kind, item))
    return dict(items=items, changes=changes, presentation=at_spoken)


COUNTED_STATES = ('heard', 'provisional', 'final')


def item_state(item, t):
    """The item's state at video time ``t`` (None before anything was known of it)."""
    found = None
    for at, state, _ in item['steps']:
        if at > t:
            break
        found = state
    return found


def change_line(kind, item):
    """(headline, why) of one correction in Korean (an ASCII minus: the Korean font has no U+2212), e.g. ``("-1 취소 #9", "다른 사람 목소리")``.

    Search mode (gamst.counting): a word counted only because it was searched for says so, "확정 #4 후보" with the reason
    it would not have counted."""
    reason, candidate = None, None
    for _, state, step in item['steps']:
        if state in ('revoked', 'uncertain') and step.get('reason'):
            reason = step['reason']
        if step.get('candidate'):
            candidate = step.get('reason') or 'not_counted'
    reason = REASON_WORD.get(reason, reason)
    number = f" #{item['number']}" if item.get('number') else ''
    if candidate is not None and kind in ('final', 'added', 'live'):
        return (f"확정{number} 후보" if kind == 'final' else f"+1 후보{number}"), \
            '찾기 모드: ' + REASON_WORD.get(candidate, candidate)
    if kind == 'revoked':
        return f"-1 취소{number}", reason or ''
    if kind == 'uncertain':
        return f"-1 보류{number}", '불확실' + (f': {reason}' if reason else '')
    if kind == 'added':
        first = item['steps'][0][1]
        return f"+1 추가{number}", '다시 들어서 찾음' if first in COUNTED_STATES else '감스트 목소리로 정정' \
            if first == 'other' else '불확실에서 확정' if first == 'uncertain' else '검증 결정'
    if kind == 'final':
        return f"확정{number}", '검증 통과'
    if kind == 'heard':
        return f"+1 들림{number}", '부분 인식'
    return f"+1{number}", '실시간'


CHANGE_COLOR = {'revoked': REC, 'uncertain': UNCERTAIN, 'added': GOOD, 'final': GOOD, 'live': GOOD, 'heard': HEARD}


def render(report, video, out, *, title, count, other=None, support=None, count_label='counted', other_label='not counted',
           note='', slow=1, points=None, start=0., end=None, hold_settled=False, at_spoken=False, lang='ko'):
    import cv2
    import numpy as np
    hud = hud_timeline(report, count, other, at_spoken=at_spoken)
    counted, others = load_events(report, count), load_events(report, other)
    if hud is not None:
        counted, others = [], []
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
    hold_until = end
    if hold_settled and hud is not None:     # the last stretch is decided after the input ends: show it settle
        last_change = max((at for item in hud['items'] for at, _, _ in item['steps']), default=0)
        hold_until = min(max(end, last_change + CHANGE_S), end + HOLD_MAX_S)
    from .voice_quality import hud_line
    warning = None if at_spoken else hud_line(report.get('voice_quality'))     # the voice reference is not one person's voice for sure
    text, trail = (EnglishText() if lang == 'en' else KoreanText()), deque(maxlen=max(2, round(.25 * fps)))
    last = None
    marks = [(event, True) for event in counted] + [(event, False) for event in others]
    index = -1
    while True:
        ok, frame = capture.read()
        index += 1
        t = index / fps
        holding = not ok or t >= end
        if holding:
            if last is None or t >= hold_until:
                break
            frame = last.copy()     # the last picture, no sound, while the last words settle
        elif t >= start:
            last = frame
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
        text.put(canvas, f'GAMST  ·  {title}', 32, 40, 34, WHITE, 'bold')
        text.put(canvas, '영상 끝 · 검증이 마지막 말을 정하는 중' if holding else (f'{slow}배 느리게' if slow != 1 else '원본 속도'),
                 VIEW_W + 32, 46, 24, WARN if holding else MUTED, anchor='right')
        cv2.rectangle(canvas, (SIDE, 104), (WIDTH - 32, 104 + VIEW_H), PANEL, -1)
        text.put(canvas, f'{t:5.2f}초', WIDTH - 64, 136, 28, MUTED, anchor='right')
        bar_y, bar_w = 940, WIDTH - 64
        cv2.rectangle(canvas, (32, bar_y - 34), (32 + bar_w, bar_y + 34), PANEL, -1)
        place = lambda at: 32 + round(bar_w * min(max((at - start) / (end - start), 0), 1))
        if hud is not None:
            _draw_hud(canvas, text, hud, t, x0, y0, view, count_label, other_label, place, bar_y, start, end)
        else:
            _draw_plain(canvas, text, counted, others, marks, t, x0, y0, view, count_label, other_label, place, bar_y,
                        start, end)
        cv2.line(canvas, (place(t), bar_y - 34), (place(t), bar_y + 34), WARN, 2)
        text.put(canvas, note, 32, 1000, 24, MUTED)
        if warning:
            text.put(canvas, f'주의: {warning}', 32, 1036, 24, WARN, 'bold')
        sink.stdin.write(canvas.tobytes())
    sink.stdin.close()
    if sink.wait():
        raise SystemExit('ffmpeg failed')
    if hud is not None:
        final = sum(item_state(item, 1e9) in COUNTED_STATES for item in hud['items'])
        return dict(out=str(out), counted=final, not_counted=len(hud['items']) - final, changes=len(hud['changes']))
    return dict(out=str(out), counted=len(counted), not_counted=len(others))


def _flash_border(canvas, x0, y0, view, color, since, span=HOLD_S):
    import cv2
    cv2.rectangle(canvas, (x0, y0), (x0 + view.shape[1] - 1, y0 + view.shape[0] - 1), color,
                  max(2, round(14 * (1 - since / span))))


def _dashed_border(canvas, x0, y0, view, color, since, span=HOLD_S):
    """A dashed frame: a word heard in a partial hypothesis (counted, not yet committed)."""
    import cv2
    thick, right, bottom = max(2, round(10 * (1 - since / span))), x0 + view.shape[1] - 1, y0 + view.shape[0] - 1
    for x in range(x0, right, 36):
        cv2.line(canvas, (x, y0), (min(x + 20, right), y0), color, thick)
        cv2.line(canvas, (x, bottom), (min(x + 20, right), bottom), color, thick)
    for y in range(y0, bottom, 36):
        cv2.line(canvas, (x0, y), (x0, min(y + 20, bottom)), color, thick)
        cv2.line(canvas, (right, y), (right, min(y + 20, bottom)), color, thick)


def _draw_hud(canvas, text, hud, t, x0, y0, view, count_label, other_label, place, bar_y, start, end):
    """The count as it was known at ``t``: L1's words at once, every later correction at the moment it was decided."""
    import cv2
    states = [(item, item_state(item, t)) for item in hud['items']]
    now = sum(state in COUNTED_STATES for _, state in states)
    recent = [change for change in hud['changes'] if 0 <= t - change[0] < CHANGE_S]
    live = [change for change in recent if change[2] in ('live', 'heard') and t - change[0] < HOLD_S]
    corrections = [change for change in recent if change[2] not in ('live', 'heard')]
    if live:
        newest = max(live, key=lambda change: change[0])
        since, color = t - newest[0], GOOD if newest[2] == 'live' else HEARD
        if newest[2] == 'heard':
            _dashed_border(canvas, x0, y0, view, color, since)
        else:
            _flash_border(canvas, x0, y0, view, color, since)
        grow = round(60 * max(0, 1 - since / .2))
        text.put(canvas, f'{now}', x0 + view.shape[1] - 40, y0 + 24, 150 + grow, color, 'bold', anchor='right')
        if newest[2] == 'heard':
            text.put(canvas, '+1 들림', x0 + view.shape[1] - 40, y0 + 210, 40, color, 'bold', anchor='right')
    elif corrections:
        newest = max(corrections, key=lambda change: change[0])
        since = t - newest[0]
        if since < HOLD_S * 1.5:
            _flash_border(canvas, x0, y0, view, CHANGE_COLOR[newest[2]], since, HOLD_S * 1.5)
    text.put(canvas, '지금까지', SIDE + 32, 136, 28, MUTED)
    text.put(canvas, '말한 순간 기준 · 최종 결과' if hud.get('presentation') else
             '들림 즉시 · 잠정 L1 · 정정은 결정된 순간에', SIDE + 32, 172, 20, DIM)
    big = (GOOD if newest[2] == 'live' else HEARD) if live else WHITE
    dropping = [change for change in corrections if change[1] < 0 and t - change[0] < .8]
    if dropping:      # the number that was there, struck out, for a moment: then the corrected one
        before = now - sum(change[1] for change in dropping)
        width = text.put(canvas, str(before), SIDE + 32, 196, 200, REC, 'bold')
        cv2.line(canvas, (SIDE + 24, 300), (SIDE + 40 + width, 300), REC, 10, cv2.LINE_AA)
        text.put(canvas, f'→ {now}', SIDE + 64 + width, 250, 110, WHITE, 'bold')
    else:
        rising = [change for change in corrections if change[1] > 0 and t - change[0] < .8]
        text.put(canvas, str(now), SIDE + 32, 196, 200, GOOD if rising else big, 'bold')
    text.put(canvas, count_label, SIDE + 32, 404, 26, GOOD, 'bold')
    tally = {name: sum(state == name for _, state in states) for name in STATE_WORD}
    if hud.get('presentation'):      # a replay for presentation: only what was counted
        tally = {name: n for name, n in tally.items() if name in COUNTED_STATES}
    x = SIDE + 32
    for name in ('final', 'provisional', 'heard', 'revoked', 'uncertain'):
        if name not in tally or name in ('uncertain', 'heard') and not tally[name]:
            continue
        x += text.put(canvas, f"{STATE_WORD[name]} {tally[name]}", x, 438, 22, STATE_COLOR[name], 'bold') + 22
    y = 474
    shown = sorted(corrections, key=lambda change: -change[0])
    for at, _, kind, item in shown[:3]:       # the corrections decided just now, newest first
        fade = 1 - max(0., t - at - (CHANGE_S - .6)) / .6
        color = tuple(round(c * fade + b * (1 - fade)) for c, b in zip(CHANGE_COLOR[kind], PANEL))
        cv2.rectangle(canvas, (SIDE + 24, y - 6), (WIDTH - 56, y + 54), tuple(round(c * .3 + b * .7) for c, b in zip(
            color, PANEL)), -1)
        headline, why = change_line(kind, item)
        text.put(canvas, headline, SIDE + 36, y, 24, color, 'bold')
        text.put(canvas, f"{item['t']:.2f}초에 한 말", WIDTH - 68, y + 2, 20, MUTED, anchor='right')
        text.put(canvas, why, SIDE + 36, y + 30, 18, WHITE)
        y += 66
    if len(shown) > 3:
        text.put(canvas, f'외 {len(shown) - 3}건', SIDE + 36, y, 20, MUTED)
        y += 30
    y = max(y + 8, 700) if shown else 520
    feed = sorted((item for item, state in states if state is not None and (not hud.get('presentation') or state in COUNTED_STATES)),
                  key=lambda item: item['t'])[-max(1, (870 - y) // 44):]
    for row, item in enumerate(reversed(feed)):
        top = y + 44 * row
        state = item_state(item, t)
        if state == 'other':
            text.put(canvas, '제외', SIDE + 32, top, 24, DIM, 'bold')
            text.put(canvas, f"{item['t']:.2f}초", SIDE + 112, top, 24, DIM)
            text.put(canvas, other_label, SIDE + 232, top + 3, 20, DIM)
            continue
        color = STATE_COLOR[state]
        width = text.put(canvas, STATE_WORD[state], SIDE + 32, top, 24, color, 'bold')
        if state == 'heard':        # dotted underline: not committed yet
            for dot in range(SIDE + 32, SIDE + 32 + width, 8):
                cv2.line(canvas, (dot, top + 32), (dot + 3, top + 32), color, 2)
        text.put(canvas, f"{item['t']:.2f}초", SIDE + 112, top, 24, WHITE if state != 'revoked' else DIM)
        if state == 'revoked':
            cv2.line(canvas, (SIDE + 110, top + 14), (SIDE + 206, top + 14), REC, 3, cv2.LINE_AA)
        label = f"#{item['number']}" if item.get('number') else ''
        detail = count_detail(item['now'], state, step=next((step for at, name, step in reversed(item['steps'])
                                                              if at <= t and name == state), None))
        text.put(canvas, '  '.join(part for part in (label, detail) if part), SIDE + 232, top + 3, 20, MUTED)
    for item, state in states:
        if state is None or not start <= item['t'] < end:
            continue
        kind = state != 'other'
        color = DIM if state == 'other' else STATE_COLOR[state]
        if state == 'heard':        # dashed tick: heard, not committed yet
            for y in range(bar_y - 30, bar_y + 30, 12):
                cv2.line(canvas, (place(item['t']), y), (place(item['t']), y + 6), color, 5, cv2.LINE_AA)
            continue
        cv2.line(canvas, (place(item['t']), bar_y - (30 if kind else 14)), (place(item['t']), bar_y + (30 if kind else 14)),
                 color, 5 if kind else 3, cv2.LINE_AA)


def _draw_plain(canvas, text, counted, others, marks, t, x0, y0, view, count_label, other_label, place, bar_y, start, end):
    """A report without revision states: each event counts at its capture time."""
    import cv2
    now = [event for event in counted if event['t'] <= t]
    since = t - now[-1]['t'] if now else 9
    rejected = [event for event in others if 0 <= t - event['t'] < HOLD_S]
    flash = since < HOLD_S
    if flash:
        _flash_border(canvas, x0, y0, view, GOOD, since)
        grow = round(60 * max(0, 1 - since / .2))
        text.put(canvas, f'{len(now)}', x0 + view.shape[1] - 40, y0 + 24, 150 + grow, GOOD, 'bold', anchor='right')
    elif rejected:
        cv2.rectangle(canvas, (x0, y0), (x0 + view.shape[1] - 1, y0 + view.shape[0] - 1), DIM, 6)
    text.put(canvas, '지금까지', SIDE + 32, 136, 28, MUTED)
    text.put(canvas, str(len(now)), SIDE + 32, 180, 230, GOOD if flash else WHITE, 'bold')
    text.put(canvas, count_label, SIDE + 32, 420, 28, GOOD, 'bold')
    feed = sorted(((event, kind) for event, kind in marks if event['t'] <= t), key=lambda item: item[0]['t'])[-7:]
    for row, (event, kind) in enumerate(reversed(feed)):
        y = 486 + 52 * row
        number = sum(1 for item in counted if item['t'] <= event['t'])
        text.put(canvas, f'{number}번' if kind else '제외', SIDE + 32, y, 28, GOOD if kind else DIM, 'bold')
        text.put(canvas, f"{event['t']:.2f}초", SIDE + 130, y, 28, WHITE if kind else DIM)
        text.put(canvas, count_detail(event) if kind else other_label, SIDE + 260, y + 3, 22, MUTED if kind else DIM)
    for event, kind in marks:
        if start <= event['t'] < end:
            color = (GOOD if kind else DIM) if event['t'] <= t else LINE
            cv2.line(canvas, (place(event['t']), bar_y - (30 if kind else 14)),
                     (place(event['t']), bar_y + (30 if kind else 14)), color, 5 if kind else 3, cv2.LINE_AA)


def count_detail(event, state=None, step=None):
    if state is not None:
        candidate = state in COUNTED_STATES and ((step or {}).get('candidate') or event.get('candidate'))
        reason = (step or {}).get('reason') or (event.get('revised_by') or {}).get('reason') or event.get('reason') \
            if state in ('revoked', 'uncertain') or candidate else None
        if candidate:      # search mode: counted, with the reason it would not have counted
            reason = '후보: ' + REASON_WORD.get(reason or 'not_counted', reason or 'not_counted')
        score = f"슬롯 {event.get('slot')} 점수 {event['slot_score']:.2f}" if event.get('slot_score') is not None else ''
        spoken = f"“{event['spoken']}”" if event.get('spoken') and event.get('spoken') != event.get('term') else ''
        flags = [FLAG_WORD.get(flag, flag) for flag in event.get('flags') or ()]
        return '  '.join(item for item in (spoken, score, REASON_WORD.get(reason, reason), *flags) if item)
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
    parser.add_argument('--lang', choices=('ko', 'en'), default='ko', help='language of the HUD words')
    parser.add_argument('--at-spoken', action='store_true',
                        help='presentation replay: each word\'s final result drawn when it was said (not live timing; the video says so)')
    parser.add_argument('--hold-settled', action='store_true',
                        help='after the video, keep its last picture (silent) until every word has settled (at most 30 s)')
    args = vars(parser.parse_args(argv))
    report = json.loads(Path(args.pop('report')).read_text())
    points = json.loads(Path(args['points']).read_text()) if args['points'] else None
    print(json.dumps(render(report, args.pop('video'), args.pop('out'), **dict(args, points=points)), ensure_ascii=False))


if __name__ == '__main__':
    main()
