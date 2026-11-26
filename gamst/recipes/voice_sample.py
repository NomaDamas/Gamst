"""The person's voice sample for a run, taken from the same input the run counts on (owner decisions 27-28).

Microphones, rooms and recordings change a voice more than the speaker model does, so the sample should come from
the same input. ``gamst recipe run`` gets one of these, in this order:

- ``--voice FILE``: an enrolled voice (``gamst recipe enroll --kind voice``), as before;
- live input (``--microphone``, a camera's sound ...) with ``--record-voice-first SECONDS``: before counting starts,
  a short recording step on that same input: a countdown ("세려는 사람만 N초 동안 말해 주세요"), a level meter while it
  records, the consent line, and the choice to record again (interactive terminals). Nothing is counted meanwhile;
- a recording with ``--voice-range FROM-TO`` (seconds of that same recording), or ``--voice-wav FILE`` (a sample of
  the person as a WAV file);
- a recording with neither: the automatic "main speaker", the speaker who speaks alone longest in its first
  ``--main-speaker-seconds`` (default 15) per pyannoteAI Precision-3 (hosted: ``--allow-remote-diarization``).

Nothing changes unless one of these options is given. Each needs ``--voice-consent "who consented, to what"`` (the enrolment's consent record). The voice is enrolled into a
file readable by the owner only (``--voice-out``, else a temporary file removed after the run) and the run report says
where it came from (``voice_sample``): words inside the sample (the range, the main speaker's segments, or the
recorded seconds before counting) hold the sample's own sound and are not fairly scored.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time

PROMPT = '세려는 사람만 {seconds:g}초 동안 말해 주세요'
LIVE_SOURCES = ('microphone', 'iphone', 'camera')


def options(parser):
    parser.add_argument('--record-voice-first', type=float, metavar='SECONDS',
                        help='live input: record the person on the same input for SECONDS before counting starts')
    parser.add_argument('--voice-range', metavar='FROM-TO',
                        help='a recording: the person speaks alone in these seconds of the same recording')
    parser.add_argument('--main-speaker-seconds', type=float, default=15.,
                        help='a recording without --voice, --voice-range or --voice-wav: enroll the speaker who speaks '
                             'alone longest in its first SECONDS (pyannoteAI Precision-3, --allow-remote-diarization)')
    parser.add_argument('--voice-consent', help='who consented to the voice sample, and to what')
    parser.add_argument('--voice-out', help='keep the enrolled sample in this file (default: a temporary file)')


def parse_range(text):
    try:
        low, high = (float(part) for part in str(text).split('-', 1))
    except ValueError:
        raise SystemExit(f'--voice-range is FROM-TO in seconds, e.g. 31.0-38.5, not {text!r}') from None
    if not 0 <= low < high:
        raise SystemExit('--voice-range needs 0 <= FROM < TO')
    return low, high


def needed(recipe, args):
    """Whether the run enrolls its voice sample first (the recipe reads a voice and none was given as a file)."""
    if getattr(args, 'voice', None) or 'voice' in (getattr(args, 'exclude', None) or ()):
        return False
    asked = any(getattr(args, name, None) for name in ('record_voice_first', 'voice_range', 'voice_consent', 'voice_wav'))
    return asked and any(reference['kind'] == 'voice' for reference in recipe.get('references', ()))


def is_live(args):
    return any(getattr(args, name, None) not in (None, False) for name in LIVE_SOURCES)


def meter(dbfs, width=30):
    """A text level meter of ``dbfs`` (-60 .. 0 dBFS)."""
    filled = 0 if dbfs is None or math.isinf(dbfs) else max(0, min(width, round((dbfs + 60) / 60 * width)))
    return '[' + '#' * filled + '.' * (width - filled) + ']'


def _enroll_args(args, **extra):
    names = ('video', 'audio', 'microphone', 'camera', 'iphone', 'take', 'input', 'offset_ms', 'plugins', 'capture_fps',
             'profile', 'models', 'worker_python', 'device', 'sensor', 'allow_remote_diarization')
    given = {name: getattr(args, name, None) for name in names}
    return argparse.Namespace(**given, kind='voice', consent=args.voice_consent, **extra)


def _say(text, out):
    print(text, file=out, flush=True)


async def prepare(recipe, args, *, enroll_voice=None, open_source=None, ask=None, out=sys.stderr, countdown_s=3):
    """Enroll the run's voice sample (see the module doc); sets ``args.voice``; returns the report's ``voice_sample``
    (``kind``, where it came from) or None when a voice file was given. ``enroll_voice``, ``open_source`` and ``ask``
    stand in for the enrolment, the input and the terminal (tests)."""
    from . import _enroll_identity
    if not needed(recipe, args):
        return None
    if not getattr(args, 'voice_consent', None):
        raise SystemExit('the voice sample is biometric: say who consented, and to what: --voice-consent "..." (or pass '
                         'an enrolled --voice FILE)')
    target = args.voice_out or tempfile.mkstemp(prefix='gamst-voice-', suffix='.json')[1]
    temporary = not args.voice_out
    run = enroll_voice or (lambda enroll_args: _enroll_identity(recipe, enroll_args, 'voice'))
    if is_live(args):
        seconds = getattr(args, 'record_voice_first', None)
        if not seconds:
            raise SystemExit('a live run needs the voice sample from the same input: --record-voice-first SECONDS '
                             '(or an enrolled --voice FILE)')
        while True:
            _say(f'consent: {args.voice_consent}', out)
            _say(PROMPT.format(seconds=seconds), out)
            for left in range(countdown_s, 0, -1):
                _say(f'  {left}...', out)
                time.sleep(1 if enroll_voice is None else 0)
            levels = []

            def level(at_s, dbfs):
                levels.append(dbfs)
                print(f'\r  {at_s:4.1f}/{seconds:g} s {meter(dbfs)} {dbfs:6.1f} dBFS', end='', file=out, flush=True)

            saved = await run(_enroll_args(args, start=0., seconds=float(seconds), out=target, main_speaker=False,
                                           meter=level))
            loud = [value for value in levels if not math.isinf(value)]
            _say('', out)
            if loud and max(loud) < -45:
                _say('  very quiet: move closer to the microphone', out)
            again = (ask or _ask)('다시 녹음할까요? [y/N] ')
            if not again:
                break
        sample = dict(kind='recorded_first', seconds=float(seconds), input='same as the run',
                      note='recorded before counting started: no counted word is inside it')
    elif getattr(args, 'voice_range', None):
        low, high = parse_range(args.voice_range)
        saved = await run(_enroll_args(args, start=low, seconds=high - low, out=target, main_speaker=False))
        sample = dict(kind='range', from_s=low, to_s=high)
    elif getattr(args, 'voice_wav', None):
        wav = _enroll_args(args, start=0., seconds=300., out=target, main_speaker=False)
        for name in ('video', 'microphone', 'camera', 'iphone', 'take', 'input'):
            setattr(wav, name, None)
        wav.audio = args.voice_wav
        saved = await run(wav)
        sample = dict(kind='file', file=os.path.basename(args.voice_wav))
    else:
        if not getattr(args, 'allow_remote_diarization', False):
            raise SystemExit('no voice sample: give --voice-range FROM-TO (the person alone in this recording), '
                             '--voice-wav FILE, an enrolled --voice FILE, or let GAMST take the main speaker of the first '
                             f'{args.main_speaker_seconds:g} s (pyannoteAI, hosted): add --allow-remote-diarization')
        saved = await run(_enroll_args(args, start=0., seconds=float(args.main_speaker_seconds), out=target,
                                       main_speaker=True))
        sample = dict(kind='main_speaker')
    origin = (saved or {}).get('enrolled_from') or {}
    sample.update({key: value for key, value in origin.items() if key not in sample})
    args.voice = target
    if temporary:
        args._voice_temporary = target
    return sample


def _ask(question):
    if not sys.stdin.isatty():
        return False
    try:
        return input(question).strip().lower() in ('y', 'yes', '네', 'ㅇ')
    except EOFError:
        return False


def cleanup(args):
    path = getattr(args, '_voice_temporary', None)
    if path and os.path.exists(path):
        os.remove(path)
