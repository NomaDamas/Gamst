"""Gemini Live on the owner's truth: one condition per call.

    python try-it/testbed/gemini_live/harness.py --clip gwan --mode rt --lang ko --run 1          # plan + cost only
    python try-it/testbed/gemini_live/harness.py --plan                                           # every condition
    python try-it/testbed/gemini_live/harness.py --check --project P                              # auth + billing (read-only)
    python try-it/testbed/gemini_live/harness.py --clip gwan --mode rt --lang ko --run 1 --go --project P

Without ``--go`` nothing is sent anywhere. With ``--go`` the clip is streamed to the Vertex AI Live API
(``gemini-3.8-live``, ADC credentials, the project's own billing) and the answers, the API's usage metadata and the
scores are written to ``<try-it>/testbed/gemini_live/runs/<condition>/`` (outside the repository).

Modes (see README.md):
- ``rt``: the clip at its own pace, 1 frame/s (the Live API's maximum) + audio, in sessions of at most ``--session-s``
  seconds of clip (audio+video sessions are limited to 2 minutes without compression). Every ``--poll-s`` seconds the
  window is closed with a cue and the model answers with the actions by 감스트 that ended since its last answer
  (realtime announcements, each with the clip time it arrived). When a session's stream ends it is asked for the full
  list (after-the-fact; 감스트 and others).
- ``post``: the same stream, no cues, only the full-list question at the end of each session.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eval as le  # noqa: E402

GCLOUD = shutil.which('gcloud') or str(Path.home() / 'google-cloud-sdk' / 'bin' / 'gcloud')
RUNS = le.TRYIT / 'testbed' / 'gemini_live' / 'runs'
CACHE = le.TRYIT / 'testbed' / 'gemini_live' / 'media'
AUDIO_RATE, AUDIO_CHUNK_S = 16000, 0.1
FORBIDDEN_MODELS = ('gemini-3.1-pro',)          # standing owner rule


def condition_name(args):
    return f"{args.clip}.{args.mode}.{args.lang}{'' if args.audio else '.noaudio'}.r{args.run}"


# ---------------------------------------------------------------------------------------------------------- media

def prepare_media(clip, *, audio=True, cache=CACHE, long_side=768):
    """1 fps JPEG frames (long side <= 768 px, the Live API's standard input size) and 16 kHz mono PCM of a clip,
    cached under ``cache/<clip>/``. Returns (frames [(t_s, bytes)], pcm bytes or b'')."""
    src = le.clip_path(clip)
    out = Path(cache) / clip
    frames_dir = out / 'frames'
    if not (frames_dir / 'done').exists():
        frames_dir.mkdir(parents=True, exist_ok=True)
        scale = f"scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'"
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-vf', f'fps=1,{scale}', '-q:v', '3',
                        str(frames_dir / '%05d.jpg')], check=True)
        (frames_dir / 'done').write_text('ok')
    files = sorted(frames_dir.glob('*.jpg'))
    frames = [(float(i), f.read_bytes()) for i, f in enumerate(files)]
    pcm = b''
    if audio:
        wav = out / 'audio.s16le'
        if not wav.exists():
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-ac', '1', '-ar', str(AUDIO_RATE),
                            '-f', 's16le', str(wav)], check=True)
        pcm = wav.read_bytes()
    return frames, pcm


def media_items(frames, pcm, start, end):
    """Time-ordered items to send for clip time [start, end): ('frame', t, jpeg) and ('audio', t, pcm chunk)."""
    items = [('frame', t, data) for t, data in frames if start <= t < end]
    step = int(AUDIO_RATE * AUDIO_CHUNK_S) * 2
    k = int(round(start / AUDIO_CHUNK_S))
    while pcm and k * AUDIO_CHUNK_S < end - 1e-9:
        chunk = pcm[k * step:(k + 1) * step]
        if not chunk:
            break
        items.append(('audio', round(k * AUDIO_CHUNK_S, 3), chunk))
        k += 1
    return sorted(items, key=lambda item: (item[1], item[0] != 'frame'))


# ---------------------------------------------------------------------------------------------------------- SDK glue

def _types():
    try:
        from google.genai import types
        return types
    except ImportError:
        return None


def _blob(data, mime):
    t = _types()
    return t.Blob(data=data, mime_type=mime) if t else {'data': data, 'mime_type': mime}


def _activity(kind):
    t = _types()
    if t:
        return t.ActivityStart() if kind == 'start' else t.ActivityEnd()
    return {'activity': kind}


def live_config(system, *, modality='TEXT', labels=None):
    config = {
        'response_modalities': [modality],
        'system_instruction': system,
        'realtime_input_config': {
            'automatic_activity_detection': {'disabled': True},
            'activity_handling': 'NO_INTERRUPTION',
            'turn_coverage': 'TURN_INCLUDES_ALL_INPUT',
        },
    }
    if modality == 'AUDIO':                     # native-audio Live models answer in speech: answers come as
        config['output_audio_transcription'] = {}   # `report` tool calls; the transcription is kept in the log
        config['tools'] = [REPORT_TOOL]
    if labels:
        config['labels'] = labels
    return config


def _usage_row(usage, modality):
    if usage is None:
        return None
    details = {}
    for item in getattr(usage, 'prompt_tokens_details', None) or ():
        name = getattr(getattr(item, 'modality', None), 'name', None) or str(getattr(item, 'modality', 'TEXT'))
        name = name.split('.')[-1]
        details[name] = details.get(name, 0) + (getattr(item, 'token_count', 0) or 0)
    if not details and getattr(usage, 'prompt_token_count', None):
        details['TEXT'] = usage.prompt_token_count
    response_details = {}
    for item in getattr(usage, 'response_tokens_details', None) or ():
        name = (getattr(getattr(item, 'modality', None), 'name', None) or str(getattr(item, 'modality', ''))).split('.')[-1]
        response_details[name] = response_details.get(name, 0) + (getattr(item, 'token_count', 0) or 0)
    return dict(prompt=details, prompt_total=getattr(usage, 'prompt_token_count', None),
                response_details=response_details,
                response=getattr(usage, 'response_token_count', None) or 0,
                thoughts=getattr(usage, 'thoughts_token_count', None) or 0,
                total=getattr(usage, 'total_token_count', None), response_modality=modality)


# ---------------------------------------------------------------------------------------------------------- runner

class Clock:
    """Clip time on the wall clock: clip time t is due at ``t0 + t / pace``."""

    def __init__(self, pace=1.0):
        self.pace, self.t0 = pace, None

    def start(self, lead_s):
        self.t0 = time.monotonic() + lead_s

    def clip_now(self):
        return (time.monotonic() - self.t0) * self.pace

    async def until(self, clip_t):
        delay = self.t0 + clip_t / self.pace - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


REPORT_TOOL = {'function_declarations': [{
    'name': 'report', 'behavior': 'NON_BLOCKING',     # answered SILENT: no second model turn (no context re-bill)
    'description': 'Report occurrences of the action (one call per answer; an empty list when there are none).',
    'parameters': {'type': 'OBJECT', 'required': ['occurrences'], 'properties': {'occurrences': {
        'type': 'ARRAY', 'items': {'type': 'OBJECT', 'required': ['who'], 'properties': {
            'who': {'type': 'STRING', 'description': "'감스트', or '다른 사람' / 'other' for anyone else"},
            'start': {'type': 'NUMBER', 'description': 'start, seconds on the [t=...] clock'},
            'end': {'type': 'NUMBER', 'description': 'end, seconds on the [t=...] clock'},
            't': {'type': 'NUMBER', 'description': 'moment of impact (slam), seconds'}}}}}}}]}


class SessionRun:
    """One Live session over clip time [start, end): photos, paced media, cues every ``poll_s`` (mode rt), the final
    full-list question. Records answers as they arrive (``report`` tool calls, or JSON text), each with the clip time
    it arrived and its phase: ``rt`` (an answer to a cue) or ``final`` (the full list), and the usage of every turn."""

    def __init__(self, index, start, end, *, prompts, frames, pcm, photos, mode, poll_s, clock, modality, log):
        self.index, self.start, self.end = index, start, end
        self.prompts, self.frames, self.pcm, self.photos = prompts, frames, pcm, photos
        self.mode, self.poll_s, self.clock, self.modality, self.log = mode, poll_s, clock, modality, log
        self.polls = 0               # cues sent (each closes a turn the model answers)
        self.final_sent = False
        self.turns = []              # completed turns: phase, done_at, usage, text
        self.items, self.rejected, self.usage_rows = [], [], []
        self.frames_sent = 0
        self.errors, self.warnings = [], []
        self.tool_calls = 0
        self._last_msg = time.monotonic()

    def phase(self):
        """By time: what arrives after the final question was sent answers it. (Measured on gemini-3.8-live: the
        model does not answer every cue with a turn of its own; it answers several cues at once, so counting turns
        against cues would misfile the final answer.)"""
        return 'final' if self.final_sent else 'rt'

    async def run(self, connect, model, config, lead_s):
        await self.clock.until(self.start - lead_s)
        async with connect(model=model, config=config) as session:
            receiver = asyncio.create_task(self._receive(session))
            try:
                await self._stream(session)
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    finals = [t for t in self.turns if t['phase'] == 'final']
                    if finals and time.monotonic() - self._last_msg > (4.0 if self.clock.pace <= 1 else 0.2):
                        break
                    if receiver.done():
                        break
                    await asyncio.sleep(0.25)
                else:
                    self.errors.append('final answer timed out after 90 s')
            finally:
                receiver.cancel()
                try:
                    await receiver
                except (asyncio.CancelledError, Exception) as exc:
                    if not isinstance(exc, asyncio.CancelledError):
                        self.errors.append(f'receiver: {exc!r}')
        return self

    async def _stream(self, session):
        send = session.send_realtime_input
        await self.clock.until(self.start)
        await send(activity_start=_activity('start'))
        label = '감스트 참고 사진' if self.prompts['lang'] == 'ko' else 'reference photo of 감스트'
        for k, photo in enumerate(self.photos, 1):
            await send(text=f'[{label} {k}/{len(self.photos)}]')
            await send(video=_blob(photo, 'image/jpeg'))
        bounds = []
        if self.mode == 'rt':
            b = self.start + self.poll_s
            while b < self.end - 1e-6:
                bounds.append(b)
                b += self.poll_s
        bounds.append(self.end)
        for item_kind, t, data in media_items(self.frames, self.pcm, self.start, self.end):
            while bounds and t >= bounds[0] - 1e-9:
                await self._close_window(send, bounds.pop(0), last=not bounds)
            await self.clock.until(t)
            if item_kind == 'frame':
                await send(text=f'[t={t:.1f}s]')
                await send(video=_blob(data, 'image/jpeg'))
                self.frames_sent += 1
            else:
                await send(audio=_blob(data, f'audio/pcm;rate={AUDIO_RATE}'))
        while bounds:
            await self._close_window(send, bounds.pop(0), last=not bounds)
        if self.mode == 'rt':           # the realtime windows are closed: the question is a turn of its own,
            await self._quiet()         # asked once the cue answers have stopped, so later answers are its own
            await send(activity_start=_activity('start'))
            if self.pcm:                # Vertex refuses a turn without audio ("No audio present in the turn")
                await send(audio=_blob(b'\0\0' * int(AUDIO_RATE * 0.2), f'audio/pcm;rate={AUDIO_RATE}'))
        await send(text=self.prompts['final'])
        self.final_sent = True
        await send(activity_end=_activity('end'))

    async def _quiet(self, quiet_s=4.0, max_s=20.0):
        """Wait until no server message for ``quiet_s`` (and at least that long after the last cue), at most
        ``max_s``; both scale with the pace (tests)."""
        scale = max(self.clock.pace, 1.0)
        quiet, start = quiet_s / scale, time.monotonic()
        while time.monotonic() - start < max_s / scale:
            if time.monotonic() - start >= quiet and time.monotonic() - self._last_msg >= quiet:
                return
            await asyncio.sleep(min(0.1, quiet / 4))
        self.warnings.append(f'cue answers still arriving {max_s:g} s after the last cue; final question asked anyway')

    async def _close_window(self, send, bound, *, last):
        """rt: at each window's end send the cue and close the turn (the model answers). post: the turn stays open
        until the final question, so the model answers once per session."""
        await self.clock.until(bound)
        if self.mode == 'rt':
            await send(text=self.prompts['cue'])
            self.polls += 1
            await send(activity_end=_activity('end'))
            if not last:
                await send(activity_start=_activity('start'))

    def _add_items(self, occurrences, now, source):
        for occ in occurrences or ():
            text = json.dumps(occ, ensure_ascii=False)
            found, bad = le.parse_answer(text, said_at=round(now, 3), source=source)
            self.items += [dict(item, phase=source.split('.')[-2]) for item in found]
            self.rejected += bad

    async def _receive(self, session):
        current = dict(text=[], usage=None, first_at=None)
        while True:
            async for msg in session.receive():
                now = self.clock.clip_now()
                self._last_msg = time.monotonic()
                phase = self.phase()
                text = getattr(msg, 'text', None)
                content = getattr(msg, 'server_content', None)
                if not text and content is not None:
                    transcript = getattr(content, 'output_transcription', None)
                    text = getattr(transcript, 'text', None)
                if text:
                    current['text'].append((round(now, 3), text))
                    current['first_at'] = current['first_at'] if current['first_at'] is not None else round(now, 3)
                call = getattr(msg, 'tool_call', None)
                calls = []
                for fc in getattr(call, 'function_calls', None) or ():
                    args = dict(getattr(fc, 'args', None) or {})
                    calls.append(dict(name=fc.name, args=args))
                    self.tool_calls += 1
                    self._add_items(args.get('occurrences'), now, f's{self.index}.{phase}.tool')
                    current['first_at'] = current['first_at'] if current['first_at'] is not None else round(now, 3)
                if calls:
                    t = _types()
                    responses = [t.FunctionResponse(id=fc.id, name=fc.name, response={'result': 'ok'},
                                                    scheduling='SILENT') if t else
                                 {'id': fc.id, 'name': fc.name, 'response': {'result': 'ok'}, 'scheduling': 'SILENT'}
                                 for fc in call.function_calls]
                    await session.send_tool_response(function_responses=responses)
                usage = getattr(msg, 'usage_metadata', None)
                if usage is not None:
                    current['usage'] = _usage_row(usage, self.modality)
                if getattr(msg, 'go_away', None) is not None:
                    self.errors.append(f'go_away at clip {now:.1f}s: {msg.go_away}')
                done = bool(content is not None and getattr(content, 'turn_complete', False))
                self.log(dict(session=self.index, at=round(now, 3), phase=phase, text=text, tool_calls=calls or None,
                              turn_complete=done, usage=current['usage'] if usage is not None else None))
                if done:
                    joined = ''.join(c for _, c in current['text'])
                    if not calls and '{' in joined:          # JSON answered as text (TEXT modality)
                        found, bad = le.parse_answer(joined, source=f's{self.index}.{phase}.text')
                        for item in found:
                            item['said_at'], item['phase'] = round(now, 3), phase
                        if current['text']:
                            _said_at_by_chunk(found, current['text'], round(now, 3))
                        self.items += found
                        self.rejected += bad
                    if current['usage']:
                        self.usage_rows.append(current['usage'])
                    self.turns.append(dict(phase=phase, first_at=current['first_at'], done_at=round(now, 3),
                                           text=joined, usage=current['usage']))
                    current = dict(text=[], usage=None, first_at=None)


def _said_at_by_chunk(found, chunks, default):
    """Each JSON item gets the clip time its closing brace arrived."""
    text, ends = '', []
    for at, chunk in chunks:
        text += chunk
        ends.append((len(text), at))
    cursor = 0
    for item in found:
        cursor = text.find(item['raw'], cursor) + len(item['raw'])
        item['said_at'] = next((at for end, at in ends if end >= cursor), default)


async def run_condition(args, *, connect=None, photos=None, frames=None, pcm=None, out_dir=None):
    """Stream one clip in one mode and language; return the result dict (also written to ``out_dir``)."""
    info = le.CLIPS[args.clip]
    if frames is None:
        frames, pcm = prepare_media(args.clip, audio=args.audio)
    photos = photos if photos is not None else [p.read_bytes() for p in le.PHOTOS]
    out_dir = Path(out_dir or RUNS / condition_name(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / 'messages.jsonl').open('w')

    def log(row):
        log_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        log_file.flush()

    if connect is None:
        # Vertex with ADC only: an API key in the environment (unknown origin) must not be picked up by the SDK.
        for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
            os.environ.pop(name, None)
        from google import genai
        client = genai.Client(vertexai=True, project=args.project, location=args.location)
        connect = client.aio.live.connect
    poll_s = args.poll_s or info['poll_s']
    clock = Clock(args.pace)
    lead = 6.0                                  # clip seconds: a session connects this long before its media
    clock.start((lead + 2.0) / args.pace)       # wall seconds until clip time 0
    sessions, start, k = [], 0.0, 0
    while start < info['duration_s'] - 1e-6:
        end = min(start + args.session_s, info['duration_s'])
        prompts = dict(le.prompt_texts(args.clip, args.lang, args.mode, window=(start, end),
                                       answer='tool' if args.modality == 'AUDIO' else 'json'), lang=args.lang)
        sessions.append(SessionRun(k, start, end, prompts=prompts, frames=frames, pcm=pcm, photos=photos,
                                   mode=args.mode, poll_s=poll_s, clock=clock, modality=args.modality, log=log))
        start, k = end, k + 1
    wall0 = time.monotonic()
    labels = {'experiment': 'gamst-vs-gemini-live', 'condition': condition_name(args).replace('.', '-')}
    results = await asyncio.gather(*(
        s.run(connect, args.model, live_config(s.prompts['system'], modality=args.modality, labels=labels), lead)
        for s in sessions), return_exceptions=True)
    wall = time.monotonic() - wall0
    log_file.close()
    errors = [repr(r) for r in results if isinstance(r, BaseException)]
    realtime, post, rejected, usage_rows, turns = [], [], [], [], []
    for s in sessions:
        errors += [f'session {s.index}: {e}' for e in s.errors]
        usage_rows += s.usage_rows
        rejected += s.rejected
        realtime += [i for i in s.items if i['phase'] == 'rt']
        post += [i for i in s.items if i['phase'] == 'final']
        turns += [dict(t, session=s.index) for t in s.turns]
    frames_sent = sum(s.frames_sent for s in sessions)
    spent = le.usage_cost(usage_rows)
    result = dict(
        condition=condition_name(args), model=args.model, location=args.location, mode=args.mode, lang=args.lang,
        clip=args.clip, clip_s=info['duration_s'], sessions=len(sessions), session_s=args.session_s, poll_s=poll_s,
        audio=args.audio, pace=args.pace, wall_s=round(wall, 1), frames_sent=frames_sent,
        frames_per_clip_s=round(frames_sent / info['duration_s'], 3),
        estimate=le.gemini_estimate(args.clip, args.mode, session_s=args.session_s, poll_s=poll_s, audio=args.audio),
        spent=spent, turns_missing_usage=sum(1 for t in turns if not t['usage']),
        tool_calls=sum(s.tool_calls for s in sessions),
        warnings=[f'session {s.index}: {w}' for s in sessions for w in s.warnings],
        realtime=dict(items=realtime, score=le.score(args.clip, realtime) if args.mode == 'rt' else None),
        after=dict(items=post, score=le.score(args.clip, post)),
        rejected=rejected, errors=errors, turns=turns)
    (out_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    if out_dir.parent == RUNS:      # every hosted run's measured spend, kept across reruns (tests write elsewhere)
        with (RUNS / 'ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(condition=result['condition'], at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                         out=str(out_dir), **spent), ensure_ascii=False) + '\n')
    return result


# ---------------------------------------------------------------------------------------------------------- plan

PLAN = [  # (clip, mode, lang): Korean everywhere; English only where it is cheap (not 관제탑 realtime)
    ('gwan', 'rt', 'ko'), ('gwan', 'post', 'ko'), ('man', 'rt', 'ko'), ('man', 'post', 'ko'),
    ('shot', 'rt', 'ko'), ('shot', 'post', 'ko'),
    ('gwan', 'post', 'en'), ('man', 'rt', 'en'), ('man', 'post', 'en'), ('shot', 'rt', 'en'), ('shot', 'post', 'en'),
]
RUNS_EACH = 2
CLAUDE_PLAN = [('gwan', 2), ('gwan', 4), ('man', 2), ('man', 4), ('shot', 60)]


def plan_table(session_s=110.0, audio=True):
    rows, total = [], 0.0
    for clip, mode, lang in PLAN:
        est = le.gemini_estimate(clip, mode, session_s=session_s, audio=audio)
        cost = est['usd'] * RUNS_EACH
        total += cost
        rows.append((f'gemini {clip}.{mode}.{lang}', est['sessions'], est['turns'], est['tokens_in'], est['usd'], cost))
    claude_total = 0.0
    for clip, fps in CLAUDE_PLAN:
        est = le.claude_estimate(clip, fps)
        cost = est['usd'] * RUNS_EACH
        claude_total += cost
        rows.append((f'claude {clip}@{fps}fps', est['sheets'], est['turns'], est['tokens_new'], est['usd'], cost))
    return rows, round(total, 2), round(claude_total, 2)


def print_plan(args=None):
    rows, gemini_total, claude_total = plan_table(audio=True if args is None else args.audio)
    print(f'Model: {le.GEMINI_LIVE_MODEL} (Vertex AI Live API, GA). Pricing: {le.PRICING_URL}')
    print('  $/1M tokens: text in 0.75, video/image in 1.00, audio in 3.00, text out 4.50 (Standard PayGo, non-global)')
    print(f'  Billing rule: each turn bills every token in the session context. Audio {le.TOK_AUDIO_S} tok/s, '
          f'image {le.TOK_FRAME} tok/frame, 1 frame/s.')
    print('  Formula per turn k of a session: (450 text + 5 photos x 258) + w_k x (258 video + 8 text + 25 audio) tokens,'
          ' w_k = clip seconds streamed so far in the session; USD = sum_k sum_modality tokens x price + output x 4.50/1M')
    print(f'  Claude arm: {le.CLAUDE_MODEL} list price $4 in / $20 out per 1M (cache write $5, cache read $0.20); '
          'image ~ w x h / 750 tokens; 4x4 contact sheets 1536x936')
    print(f"\n{'condition':28} {'sess/sheets':>11} {'turns':>6} {'tok in/new':>11} {'USD/run':>9} {'USD x' + str(RUNS_EACH):>9}")
    for name, a, b, c, d, e in rows:
        print(f'{name:28} {a:>11} {b:>6} {c:>11,} {d:>9.3f} {e:>9.3f}')
    print(f'\nGemini Live total ({RUNS_EACH} runs each): ${gemini_total:.2f} (~{gemini_total * le.USD_KRW:,.0f} KRW at '
          f'{le.USD_KRW:.0f} KRW/USD)')
    print(f'Claude arm total ({RUNS_EACH} runs each): ${claude_total:.2f}')
    print(f'All: ${gemini_total + claude_total:.2f}')
    return gemini_total, claude_total


def check_gcp(project):
    """Read-only: gcloud account, ADC token, billing on ``project``, Vertex AI API enabled. Prints no secret."""
    out = {}

    def run(*cmd):
        r = subprocess.run([GCLOUD, *cmd], capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout.strip()
    if not Path(GCLOUD).exists():
        return dict(ok=False, problems=['gcloud not found'])
    out['account'] = run('config', 'get-value', 'account')[1]
    out['project'] = project or run('config', 'get-value', 'project')[1]
    out['adc'] = run('auth', 'application-default', 'print-access-token')[0] == 0
    code, billing = run('billing', 'projects', 'describe', out['project'], '--format=value(billingEnabled)')
    out['billing_enabled'] = billing.strip().lower() == 'true' if code == 0 else None
    code, services = run('services', 'list', '--enabled', '--project', out['project'], '--format=value(config.name)')
    out['aiplatform_enabled'] = 'aiplatform.googleapis.com' in services.split() if code == 0 else None
    problems = []
    if not out['adc']:
        problems.append('application-default credentials missing or expired')
    if out['billing_enabled'] is not True:
        problems.append(f"billing is not enabled on project {out['project']}")
    if out['aiplatform_enabled'] is False:
        problems.append(f"aiplatform.googleapis.com not enabled on {out['project']}")
    out['ok'] = not problems
    out['problems'] = problems
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip', choices=sorted(le.CLIPS), default='man')
    ap.add_argument('--mode', choices=('rt', 'post'), default='rt')
    ap.add_argument('--lang', choices=('ko', 'en'), default='ko')
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--model', default=le.GEMINI_LIVE_MODEL)
    ap.add_argument('--project', default=os.environ.get('GAMST_GCP_PROJECT'))
    ap.add_argument('--location', default=os.environ.get('GAMST_GCP_LOCATION', 'us-central1'))
    ap.add_argument('--session-s', type=float, default=110.0, help='clip seconds per Live session (limit 120 s)')
    ap.add_argument('--poll-s', type=float, default=None, help='realtime cue period (default per clip: 5 s, slam 2 s)')
    ap.add_argument('--no-audio', dest='audio', action='store_false')
    ap.add_argument('--pace', type=float, default=1.0, help='1.0 = the clip at its own pace (keep 1.0 for runs)')
    ap.add_argument('--modality', choices=('TEXT', 'AUDIO'), default='AUDIO',
                    help='AUDIO (+ report tool): what gemini-3.8-live supports on Vertex; TEXT is refused there')
    ap.add_argument('--max-usd', type=float, default=None, help='refuse --go if the estimate exceeds this')
    ap.add_argument('--plan', action='store_true', help='print every condition and the total, then exit')
    ap.add_argument('--check', action='store_true', help='read-only GCP auth and billing check, then exit')
    ap.add_argument('--go', action='store_true', help='actually stream to Google (costs money)')
    args = ap.parse_args(argv)
    if any(bad in args.model for bad in FORBIDDEN_MODELS) or 'live' not in args.model:
        ap.error(f'model {args.model!r} is not allowed here (Live models only; never Gemini 3.1 Pro)')
    if args.plan:
        print_plan(args)
        return 0
    if args.check:
        print(json.dumps(check_gcp(args.project), ensure_ascii=False, indent=1))
        return 0
    est = le.gemini_estimate(args.clip, args.mode, session_s=args.session_s, poll_s=args.poll_s, audio=args.audio)
    print(f'Condition {condition_name(args)}: {le.CLIPS[args.clip]["name"]}, {args.mode}, {args.lang}, '
          f'model {args.model} @ {args.location}')
    print(f"  estimate: {est['sessions']} sessions, {est['turns']} turns, {est['tokens_in']:,} input tokens billed, "
          f"{est['tokens_out']:,} output tokens, ${est['usd']:.3f} (~{est['usd'] * le.USD_KRW:,.0f} KRW). "
          f'Pricing: {le.PRICING_URL}')
    if not args.go:
        print('  (no --go: nothing sent)')
        return 0
    if args.max_usd is not None and est['usd'] > args.max_usd:
        print(f"  refused: estimate ${est['usd']:.3f} > --max-usd {args.max_usd}")
        return 2
    gcp = check_gcp(args.project)
    if not gcp['ok']:
        print('  refused: ' + '; '.join(gcp['problems']))
        return 2
    args.project = gcp['project']
    result = asyncio.run(run_condition(args))
    summary = {k: result[k] for k in ('condition', 'wall_s', 'frames_per_clip_s', 'spent', 'errors')}
    summary['realtime'] = result['realtime']['score']
    summary['after'] = result['after']['score']
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
