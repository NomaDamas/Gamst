"""Clips, prompts, answer parsing, scoring and cost for the GAMST / Gemini Live / Claude comparison.

Pure functions only (no network, no model): the harness (``harness.py``) and the Claude arm (``claude_arm.py``) share
them, and the unit tests run them offline on a recorded fake response.

- Truth: ``docs/ground-truth/2026-11-28/v2/`` (관제탑, 맨까송) and ``<try-it>/testbed/truth_hit.json`` (slam clip).
- Scorer: ``gamst.eval.multi_person`` (the owner's ``score_multi.py`` rules: one to one by time, 0.5 s) for the two
  multi-person clips; the testbed rule (one to one by end time, nearest first, 0.2 s) for the slam clip. These are the
  rules GAMST was measured with.
- The truth is never put into a prompt: :func:`prompt_texts` reads only the action text and the clip times.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gamst.eval import multi_person as mp  # noqa: E402

TESTBED = Path(os.environ.get('GAMST_TESTBED', '/Users/jinminseong/Desktop/홍명보/gamst/GAMST TestBed'))
TRYIT = Path(os.environ.get('GAMST_TRYIT', REPO.parent / 'try-it' if (REPO.parent / 'try-it').is_dir()
                            else '/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it'))
TRUTH_V2 = REPO / 'docs' / 'ground-truth' / '2026-11-28' / 'v2'
# The photos 감스트's identity reference was checked against (person-identity-2026-11-27.md, `id-enroll-photos`): five
# frames of the shop clip a.mp4, 5.0-9.5 s. The #379/#381 runs used `testbed/identity/clips/감스트.person.json`,
# enrolled from 7 s of the same shop clip (embeddings only, no image kept); these photos are the image form of it.
PHOTOS = [TRYIT / 'testbed' / 'identity' / 'photos' / f'a_{t}.jpg' for t in ('5.0', '6.0', '7.5', '8.5', '9.5')]

TARGET = mp.TARGET            # '감스트'
OTHER = 'other'

# Action text as the owner defined it. ko: the owner's definition column of docs/ground-truth/2026-11-28/README.md.
# en: the owner decisions 62/63 of docs/decision-log-2026-11-26.md (recorded in English). The slam clip has no Korean
# owner text on Main: the English is the owner's definition in docs/measurements/testbed-2026-11-26.md ("a collision,
# heard or not"); the Korean is a translation made for this experiment (marked so in the report).
CLIPS = {
    'gwan': dict(
        name='관제탑', video='감스트-관제탑 모음집.mp4', duration_s=522.436, width=1280, height=720, fps=30.0,
        truth='gwan', kind='interval', who_matters=True, poll_s=5.0,
        action={'ko': '관제탑: 두 팔을 오므렸다가 → 양옆으로 폈다가 → 다시 접는 것 = 1회. 연속으로 하면 1회씩 따로 셈',
                'en': '관제탑 (control tower): both arms folded in front → spread out to BOTH SIDES → folded again = 1. '
                      'Each repetition counts separately.'}),
    'man': dict(
        name='맨까송', video='감스트 맨까송 - 맨가대(가습기X김바쏘).mp4', duration_s=67.255, width=640, height=360,
        fps=29.97, truth='man', kind='interval', who_matters=True, poll_s=5.0,
        action={'ko': '펀치: 감스트가 펀치를 한 사실 = 1회',
                'en': 'Punch: 감스트 punched = 1 (one strike).'}),
    'shot': dict(
        name='샷건 (슬램)', video=None, path='testbed/b.mp4', duration_s=11.9, width=1280, height=720, fps=60.0,
        truth='testbed/truth_hit.json', kind='point', who_matters=False, poll_s=2.0,
        action={'ko': '슬램: 주먹으로 무언가를 내리쳐 부딪히는 것 = 1회. 소리가 나든 안 나든 센다.',
                'en': 'Slam: a strike that collides with something = 1, whether or not it makes a sound.'}),
}


def clip_path(clip):
    info = CLIPS[clip]
    return TRYIT / info['path'] if info.get('path') else TESTBED / info['video']


# ---------------------------------------------------------------------------------------------------------- prompts

def prompt_texts(clip, lang, mode, *, window=None, fps_note='1 fps'):
    """{system, cue, final}: the instruction, the short cue sent at the end of each realtime window, and the question
    asked after a session's stream ends. ``mode``: 'rt' (realtime announcements, then the full list) or 'post' (full
    list only). ``window`` = (start_s, end_s) of the session. Nothing here reads the truth."""
    info = CLIPS[clip]
    action = info['action'][lang]
    point = info['kind'] == 'point'
    a, b = window or (0.0, info['duration_s'])
    if lang == 'ko':
        fmt_one = ('{"who":"감스트","t":초}' if point else '{"who":"감스트","start":초,"end":초}')
        fmt_any = ('{"who":"감스트" 또는 "다른 사람","t":초}' if point else
                   '{"who":"감스트" 또는 "다른 사람","start":초,"end":초}')
        system = (
            f"너는 영상을 실시간으로 지켜보는 관찰자다. 영상 프레임({fps_note})과 소리가 실시간으로 들어온다. "
            "각 프레임 바로 앞에 '[t=12.0s]'처럼 영상 시각(초)이 붙어 있다. 시각은 반드시 이 표시를 기준으로 말한다.\n"
            "맨 처음에 '감스트'라는 사람의 참고 사진 5장이 온다. 영상에는 다른 사람도 나온다.\n"
            f"찾을 동작 — {action}\n"
            + ("동작의 시각은 부딪히는 순간(t)이다.\n" if point else "동작의 시작(start)과 끝(end)을 초로 말한다.\n")
            + "답은 한 줄에 동작 하나씩 JSON으로만 쓴다. 다른 말, 설명, 코드 블록은 쓰지 않는다. 해당하는 것이 없으면 정확히 '없음'이라고만 쓴다.")
        cue = (f"[보고] 앞 보고 이후에 새로 끝난 감스트의 동작을 한 줄에 하나씩 {fmt_one} 형식으로 말해라. "
               "이미 말한 것은 다시 말하지 않는다. 없으면 '없음'.")
        final = (f"[전체 목록] 이 세션에서 본 영상 구간(t={a:.0f}–{b:.0f}s) 전체에서 이 동작을 한 것을 감스트와 다른 사람 모두 "
                 f"빠짐없이 시간 순서대로 한 줄에 하나씩 {fmt_any} 형식으로 나열해라. 없으면 '없음'.")
    else:
        fmt_one = ('{"who":"감스트","t":seconds}' if point else '{"who":"감스트","start":seconds,"end":seconds}')
        fmt_any = ('{"who":"감스트" or "other","t":seconds}' if point else
                   '{"who":"감스트" or "other","start":seconds,"end":seconds}')
        system = (
            f"You watch a video in real time. Video frames ({fps_note}) and sound arrive live. "
            "Each frame is preceded by its video time in seconds, like '[t=12.0s]'. Always state times on that clock.\n"
            "First come 5 reference photos of a person called '감스트'. Other people also appear in the video.\n"
            f"Action to find — {action}\n"
            + ("The time of an action is the moment of impact (t).\n" if point else
               "State the start and the end of each action in seconds.\n")
            + "Answer with one JSON object per line, one action per line, and nothing else: no prose, no code fences. "
              "If there is none, answer exactly 'none'.")
        cue = (f"[report] List each action by 감스트 that ended since your last report, one per line as {fmt_one}. "
               "Do not repeat ones already reported. If none, 'none'.")
        final = (f"[full list] For the whole part of the video this session saw (t={a:.0f}–{b:.0f}s), list every "
                 f"occurrence of the action, by 감스트 and by anyone else, in time order, one per line as {fmt_any}. "
                 "If none, 'none'.")
    if mode == 'post':
        system = system.replace('실시간으로 지켜보는 관찰자', '지켜보는 관찰자').replace('You watch a video in real time',
                                                                                'You watch a video')
    return dict(system=system, cue=cue, final=final, action=action)


# ---------------------------------------------------------------------------------------------------------- parsing

_OBJ = re.compile(r'\{[^{}]*\}')
_NONE = re.compile(r'^\s*(없음|none|nothing|n/?a)\s*\.?\s*$', re.I)


def _num(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        m = re.search(r'-?\d+(?:\.\d+)?', value)
        if m:
            return float(m.group())
    return None


def normalize_who(who):
    text = str(who or '').strip().strip('"\'')
    if '감스트' in text or text.lower() in ('gamst', 'gamsteu', 'gam-st'):
        return TARGET
    return OTHER if text else TARGET


def parse_answer(text, *, said_at=None, source=None):
    """Occurrences in one model answer: [{start, end, who, said_at, source, raw}] (seconds on the clip clock).

    Each JSON object on its own (code fences, prose and Korean quotes around it are tolerated); a ``t`` gives a point
    (start = end = t); ``start`` without ``end`` is a point too. 'none'/'없음' or no object = no occurrence. Malformed
    objects are counted in ``rejected`` rather than guessed."""
    found, rejected = [], []
    if not text or _NONE.match(text.strip()):
        return found, rejected
    for match in _OBJ.finditer(text.replace('“', '"').replace('”', '"')):
        raw = match.group()
        try:
            obj = json.loads(raw)
        except ValueError:
            try:
                obj = json.loads(re.sub(r"'", '"', raw))
            except ValueError:
                rejected.append(raw)
                continue
        t = _num(obj.get('t', obj.get('time')))
        start = _num(obj.get('start', t))
        end = _num(obj.get('end', start))
        if start is None or end is None:
            rejected.append(raw)
            continue
        if end < start:
            start, end = end, start
        found.append(dict(start=start, end=end, who=normalize_who(obj.get('who')), said_at=said_at,
                          source=source, raw=raw))
    return found, rejected


def dedupe_exact(items):
    """Drop exact repeats (same who, start, end) keeping the first said; near repeats stay (they are the model's)."""
    seen, out = set(), []
    for item in items:
        key = (item['who'], round(item['start'], 2), round(item['end'], 2))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------------------------------------- scoring

def load_truth(clip):
    info = CLIPS[clip]
    if info['kind'] == 'point':
        data = json.loads((TRYIT / info['truth']).read_text())
        return dict(kind='point', points=[float(e['t_s']) for e in data['events']], tol=float(data['tolerance_s']))
    truth = mp.load_truth(TRUTH_V2 / info['truth'])
    occ = [dict(start=float(o['start']), end=float(o['end']), who=TARGET if o['who'] == TARGET else OTHER)
           for o in truth['action']['occurrences']]
    return dict(kind='interval', occurrences=occ, examples=mp.example_ranges(truth['examples']),
                together=mp.target_moments(truth['identity']))


def f1(precision, recall):
    if not precision or not recall:
        return 0.0 if precision is not None and recall is not None else None
    return round(2 * precision * recall / (precision + recall), 4)


def match_points(events, points, tol):
    """The testbed slam rule: one to one by end time, nearest first, within ``tol`` seconds. [(event, point)]."""
    pairs = sorted((abs(e['end'] - p), i, j) for i, e in enumerate(events) for j, p in enumerate(points)
                   if abs(e['end'] - p) <= tol + 1e-9)
    used_e, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_e and j not in used_p:
            used_e.add(i)
            used_p.add(j)
            out.append((events[i], points[j]))
    return out


def _latency(pairs, end_of):
    lags = sorted(e['said_at'] - end_of(t) for e, t in pairs if e.get('said_at') is not None)
    if not lags:
        return None
    mid = len(lags) // 2
    p50 = lags[mid] if len(lags) % 2 else (lags[mid - 1] + lags[mid]) / 2
    return dict(p50_s=round(p50, 2), max_s=round(lags[-1], 2), min_s=round(lags[0], 2), n=len(lags))


def score(clip, events, *, tol=None, exclude_examples=True):
    """Score occurrences ``events`` of one clip against the truth with GAMST's rules. Returns a dict with, for 감스트
    (and, on the multi-person clips, everyone): truth, said, matched, precision, recall, F1, start/end error, and the
    latency from the truth occurrence's end to the moment the answer arrived (realtime answers only)."""
    truth = load_truth(clip)
    if truth['kind'] == 'point':
        tol = truth['tol'] if tol is None else tol
        pairs = match_points(events, truth['points'], tol)
        n, g, t = len(pairs), len(events), len(truth['points'])
        precision = round(n / g, 4) if g else None
        recall = round(n / t, 4) if t else None
        return {TARGET: dict(truth=t, said=g, error=g - t, matched=n, precision=precision, recall=recall,
                             f1=f1(precision, recall), tol_s=tol,
                             time_err_s=round(sum(abs(e['end'] - p) for e, p in pairs) / n, 3) if n else None,
                             latency=_latency(pairs, lambda p: p))}
    tol = mp.TOL_S if tol is None else tol
    exclude = truth['examples'] if exclude_examples else ()
    out = mp.score_action(events, truth['occurrences'], exclude=exclude, tol=tol, together=truth['together'])
    result = {}
    for person in (TARGET, mp.EVERYONE):
        row = dict(out[person])
        row['said'] = row.pop('gamst')
        row['f1'] = f1(row['precision'], row['recall'])
        mine = (lambda who: who == TARGET) if person == TARGET else (lambda who: True)
        g = mp.restrict([e for e in events if mine(e['who'])], exclude=exclude)
        t = mp.restrict([o for o in truth['occurrences'] if mine(o['who'])], exclude=exclude)
        row['latency'] = _latency(mp.match(g, t, tol), lambda o: o['end'])
        result[person] = row
    result['tol_s'] = tol
    result['examples_excluded'] = bool(exclude)
    return result


# ---------------------------------------------------------------------------------------------------------- cost

# Vertex AI (Gemini Enterprise Agent Platform) pricing, "Gemini 3.8 Live API", Standard PayGo, non-global, read
# 2026-11-28 from https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing . USD per 1M tokens.
GEMINI_LIVE_MODEL = 'gemini-3.8-live'
GEMINI_PRICE = dict(text_in=0.75, image_in=1.00, video_in=1.00, audio_in=3.00, text_out=4.50, audio_out=12.00)
# Same page, "Live API Billing Details": every turn bills all tokens in the session context window (past turns are
# re-processed each turn); audio 25 tokens/s; image input 258 tokens per image.
TOK_FRAME, TOK_AUDIO_S = 258, 25
TOK_TIME_MARK = 8               # the '[t=12.0s]' text before each frame (estimate)
TOK_SYSTEM = 450                # instruction + action text (estimate; Korean ~1 token per 1-2 characters)
TOK_CUE, TOK_FINAL_Q = 60, 90   # the text cue and the final question (estimate)
TOK_OUT_POLL, TOK_OUT_ITEM = 12, 25
N_PHOTOS = len(PHOTOS)
PRICING_URL = 'https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing'
LIMITS_URLS = ['https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api',
               'https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-live.md.txt',
               'https://aistudio.google.com/docs/live-api/capabilities',
               'https://aistudio.google.com/docs/live-api/session-management']
USD_KRW = float(os.environ.get('USD_KRW', '1400'))   # stated conversion, not a quote


def _usd_per_media_second(audio=True):
    """$ per second of clip held in the context once (video frame at 1 fps + time mark + audio)."""
    p = GEMINI_PRICE
    return (TOK_FRAME * p['video_in'] + TOK_TIME_MARK * p['text_in'] + (TOK_AUDIO_S * p['audio_in'] if audio else 0)) / 1e6


def gemini_estimate(clip, mode, *, session_s=110.0, poll_s=None, audio=True, expected_items=None):
    """Estimated tokens and USD of one run of one condition (one clip) on gemini-3.8-live.

    Sessions of ``session_s`` seconds of clip (the audio+video session limit is 2 min without compression). Inside a
    session, turn k (k = 1..K) happens after w_k seconds of clip; it bills the whole context then:
        prompt_k = system + photos + w_k * (258 + 8 + 25 [audio]) + text of earlier cues and answers
    'rt' has one turn per ``poll_s`` window plus a final full-list turn; 'post' has only the final turn.
    USD = sum_k (prompt_k by modality x price) + output tokens x 4.50/1M."""
    info = CLIPS[clip]
    poll_s = poll_s or info['poll_s']
    dur = info['duration_s']
    items = expected_items if expected_items is not None else _expected_items(clip)
    p = GEMINI_PRICE
    fixed = TOK_SYSTEM * p['text_in'] + N_PHOTOS * TOK_FRAME * p['image_in']
    media_rate = _usd_per_media_second(audio) * 1e6        # micro-USD per clip second in context
    tok_rate = TOK_FRAME + TOK_TIME_MARK + (TOK_AUDIO_S if audio else 0)
    usd = tokens_in = tokens_out = 0.0          # usd accumulates micro-USD (tokens x $/1M)
    turns = sessions = 0
    start = 0.0
    while start < dur - 1e-6:
        length = min(session_s, dur - start)
        sessions += 1
        share = items * length / dur
        history = 0.0                                       # text tokens of earlier cues + answers in this session
        windows = []
        if mode == 'rt':
            k = 1
            while (k - 1) * poll_s < length - 1e-6:
                windows.append(min(k * poll_s, length))
                k += 1
        for w in windows:
            prompt_usd = fixed + w * media_rate + (history + TOK_CUE) * p['text_in']
            out = TOK_OUT_POLL + TOK_OUT_ITEM * share / max(len(windows), 1)
            usd += prompt_usd + out * p['text_out']
            tokens_in += TOK_SYSTEM + N_PHOTOS * TOK_FRAME + w * tok_rate + history + TOK_CUE
            tokens_out += out
            history += TOK_CUE + out
            turns += 1
        out = TOK_OUT_ITEM * share * (2 if info['who_matters'] else 1) + 5     # final list: everyone's occurrences
        prompt_usd = fixed + length * media_rate + (history + TOK_FINAL_Q) * p['text_in']
        usd += prompt_usd + out * p['text_out']
        tokens_in += TOK_SYSTEM + N_PHOTOS * TOK_FRAME + length * tok_rate + history + TOK_FINAL_Q
        tokens_out += out
        turns += 1
        start += length
    return dict(clip=clip, mode=mode, sessions=sessions, turns=turns, poll_s=poll_s if mode == 'rt' else None,
                session_s=session_s, audio=audio, tokens_in=int(tokens_in), tokens_out=int(tokens_out),
                usd=round(usd / 1e6, 4))


def _expected_items(clip):
    """A planning number of answer lines (not the truth itself: a round figure of the clip's size, used for output
    tokens only, which are < 2 % of the estimate)."""
    return dict(gwan=70, man=20, shot=16)[clip]


def usage_cost(usage_rows):
    """USD from the API's own usage metadata, one row per completed turn: {prompt: {MODALITY: n}, response: n,
    thoughts: n, response_modality: 'TEXT'|'AUDIO'}. Unknown prompt modalities are billed at the text rate and listed."""
    p = GEMINI_PRICE
    rate = dict(TEXT=p['text_in'], IMAGE=p['image_in'], VIDEO=p['video_in'], AUDIO=p['audio_in'])
    usd, unknown, tin, tout = 0.0, set(), 0, 0
    for row in usage_rows:
        for modality, n in (row.get('prompt') or {}).items():
            if modality not in rate:
                unknown.add(modality)
            usd += n * rate.get(modality, p['text_in']) / 1e6
            tin += n
        out = (row.get('response') or 0) + (row.get('thoughts') or 0)
        usd += out * (p['audio_out'] if row.get('response_modality') == 'AUDIO' else p['text_out']) / 1e6
        tout += out
    return dict(usd=round(usd, 6), tokens_in=tin, tokens_out=tout, turns=len(usage_rows), unknown=sorted(unknown))


# Claude arm: Anthropic list price, claude-opus-5-5 (USD per 1M tokens; cache write 1.25x input, cache read $0.20 as
# listed in the claude-api reference of 2026-11-03). Image tokens ~ width x height / 750.
CLAUDE_MODEL = 'claude-opus-5-5'
CLAUDE_PRICE = dict(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20)


def claude_sheet_tokens(width=1536, height=936):
    return math.ceil(width * height / 750)


def claude_estimate(clip, fps, *, per_sheet=16, overhead_tokens=25000, reads_per_turn=8, output_tokens=20000):
    """Estimated tokens and USD of one Claude Code subagent run on contact sheets: each sheet holds ``per_sheet``
    frames (4 x 4, 384 px tiles, time-stamped). The agent reads ``reads_per_turn`` sheets per turn; every turn re-reads
    the conversation from the cache. USD = new tokens x cache write + sum over turns of context x cache read + output."""
    info = CLIPS[clip]
    frames = math.ceil(info['duration_s'] * fps)
    sheets = math.ceil(frames / per_sheet)
    sheet_tok = claude_sheet_tokens()
    photo_tok = N_PHOTOS * claude_sheet_tokens(768, 432)
    new = overhead_tokens + photo_tok + sheets * sheet_tok
    turns = math.ceil(sheets / reads_per_turn) + 2
    context, reread = overhead_tokens + photo_tok, 0
    for k in range(turns):
        reread += context
        context += min(reads_per_turn, max(sheets - k * reads_per_turn, 0)) * sheet_tok
    p = CLAUDE_PRICE
    usd = (new * p['cache_write'] + reread * p['cache_read'] + output_tokens * p['output']) / 1e6
    return dict(clip=clip, fps=fps, frames=frames, sheets=sheets, turns=turns, tokens_new=new, tokens_cache_read=reread,
                tokens_out=output_tokens, usd=round(usd, 4))
