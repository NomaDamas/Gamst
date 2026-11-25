"""Find candidate intervals of a described movement in a recorded video. A VLM proposes, it never counts.

The movement is the owner's text. The video is read in bounded windows at a low frame rate; each window goes
to the semantic lane's client (``OpenAIStreamingBackend.locate``: local unless the owner opted in to a hosted
endpoint). The reply is untrusted: only a JSON object with ``intervals`` of exactly ``start_s``, ``end_s``
and ``score`` is accepted, and only intervals inside the window's own frames are kept. No free text from
the model is stored anywhere. Accepted intervals are *proposals* (:mod:`gamst.movement_proposals`).
"""
from __future__ import annotations

import asyncio
import json
import math

MAX_WINDOW_FRAMES = 16
MAX_INTERVALS = 8
MAX_REPLY = 16384
MAX_WINDOWS = 450
MAX_SIDE = 512
MAX_TEXT = 500


def movement_text(text):
    text = ' '.join(str(text or '').split())
    if not 1 <= len(text) <= MAX_TEXT:
        raise ValueError(f'the movement description has 1..{MAX_TEXT} characters')
    return text


def parse_intervals(reply, *, start_s, end_s):
    """(intervals, rejected) of one window's reply; ValueError when the reply is not exactly the contract.

    ``rejected`` counts well-formed intervals that are not inside the window the model was shown.
    """
    if not isinstance(reply, str) or len(reply) > MAX_REPLY:
        raise ValueError('reply is not bounded text')

    def unique(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError('duplicate keys')
        return dict(pairs)

    def constant(name):
        raise ValueError('not finite')

    try:
        value = json.loads(reply, object_pairs_hook=unique, parse_constant=constant)
    except (ValueError, RecursionError) as error:
        raise ValueError('reply is not the JSON contract') from error
    if type(value) is not dict or set(value) != {'intervals'} or type(value['intervals']) is not list \
            or len(value['intervals']) > MAX_INTERVALS:
        raise ValueError('reply is not the JSON contract')
    found, rejected = [], 0
    for item in value['intervals']:
        if type(item) is not dict or set(item) != {'start_s', 'end_s', 'score'} or any(
                type(number) not in (int, float) or not math.isfinite(number) for number in item.values()) \
                or not 0 <= item['score'] <= 1:
            raise ValueError('reply is not the JSON contract')
        if not start_s <= item['start_s'] < item['end_s'] <= end_s:
            rejected += 1
            continue
        found.append(dict(start_s=round(float(item['start_s']), 3), end_s=round(float(item['end_s']), 3),
                          score=round(float(item['score']), 4)))
    return found, rejected


def merge(candidates):
    """One candidate per place: those of different windows that overlap by half of the shorter are joined."""
    merged = []
    for item in sorted(candidates, key=lambda entry: (entry['start_s'], entry['end_s'])):
        for other in merged:
            overlap = min(item['end_s'], other['end_s']) - max(item['start_s'], other['start_s'])
            shorter = min(item['end_s'] - item['start_s'], other['end_s'] - other['start_s'])
            if item['window'] not in other['windows'] and 2 * overlap >= shorter:
                other.update(start_s=min(item['start_s'], other['start_s']), end_s=max(item['end_s'], other['end_s']),
                             score=max(item['score'], other['score']))
                other['windows'].add(item['window'])
                break
        else:
            merged.append(dict(start_s=item['start_s'], end_s=item['end_s'], score=item['score'],
                               windows={item['window']}))
    return [dict(start_s=item['start_s'], end_s=item['end_s'], score=item['score']) for item in merged]


def check_windows(*, fps, window_s, overlap_s):
    frames = int(round(fps * window_s))
    if not .2 <= fps <= 4 or not 2 <= frames <= MAX_WINDOW_FRAMES:
        raise ValueError(f'use .2..4 frames per second and 2..{MAX_WINDOW_FRAMES} frames per window')
    overlap = int(round(fps * overlap_s))
    if not 0 <= overlap < frames:
        raise ValueError('the window overlap must be shorter than the window')
    return frames, overlap


def _small(frame):
    from .crew.modules import bounded_image
    return bounded_image(frame, max_pixels=MAX_SIDE * MAX_SIDE, max_side=MAX_SIDE)


async def find(source, backend, text, *, fps=2., window_s=8., overlap_s=2., max_windows=MAX_WINDOWS, progress=None):
    """Candidate intervals (seconds from the start of the input) of ``text`` in a recorded video.

    ``source`` is a video input (``frames()``). Returns ``{"candidates", "windows", "rejected_replies",
    "rejected_intervals", "failed_windows", "truncated"}``; ``truncated`` when ``max_windows`` ended the search.
    """
    text = movement_text(text)
    size, overlap = check_windows(fps=fps, window_s=window_s, overlap_s=overlap_s)
    if not 1 <= max_windows <= MAX_WINDOWS:
        raise ValueError(f'max_windows must be 1..{MAX_WINDOWS}')
    report = dict(candidates=[], windows=0, rejected_replies=0, rejected_intervals=0, failed_windows=0,
                  truncated=False)
    candidates, frames, times = [], [], []
    last_error = None

    async def ask():
        nonlocal last_error
        report['windows'] += 1
        try:
            reply = await asyncio.to_thread(backend.locate, list(frames), list(times), text)
        except (OSError, KeyError, IndexError, TypeError, ValueError) as error:   # endpoint or protocol failure
            report['failed_windows'] += 1
            last_error = type(error).__name__
        else:
            try:
                found, rejected = parse_intervals(reply, start_s=times[0], end_s=times[-1])
            except ValueError:
                report['rejected_replies'] += 1
            else:
                report['rejected_intervals'] += rejected
                candidates.extend(dict(item, window=report['windows']) for item in found)
        if progress is not None:
            progress(report)

    first = fresh = None
    async for frame in source.frames():
        if first is None:
            origin = getattr(source, 'origin_ms', None)
            first = frame.timestamp_ms if origin is None else origin
        seconds = (frame.timestamp_ms - first) / 1000
        if times and seconds < times[-1] + 1 / fps - 1e-3:
            continue
        if report['windows'] >= max_windows:
            report['truncated'] = True
            break
        frames.append(_small(frame.data))
        times.append(round(seconds, 3))
        fresh = True
        if len(frames) == size:
            await ask()
            frames, times = frames[size - overlap:], times[size - overlap:]
            fresh = False
    if fresh and len(frames) >= 2 and report['windows'] < max_windows:
        await ask()
    if report['windows'] and report['failed_windows'] == report['windows']:
        raise RuntimeError(f'the VLM endpoint answered no window ({last_error})')
    report['candidates'] = merge(candidates)
    return report
