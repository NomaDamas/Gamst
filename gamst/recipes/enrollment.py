"""Voice and person enrollment for ``gamst recipe enroll``: embeddings and a consent record, no media.

Audio and images live only in this process while the embeddings are computed; the saved file has the
embeddings, the model that made them, and who consented to what. ``gamst recipe run`` refuses a file
without that record.
"""
from __future__ import annotations

import json
from pathlib import Path
import time


def consent_record(statement):
    statement = (statement or '').strip()
    if not statement:
        raise SystemExit('enrolling a person needs --consent "who consented, to what, when"')
    return dict(statement=statement, recorded_ms=time.time_ns() // 1_000_000)


def _save(path, value):
    target = Path(path)
    target.touch(mode=0o600, exist_ok=True)
    target.chmod(0o600)   # biometric embeddings: readable by the owner only
    target.write_text(json.dumps(value))


async def _close(embedder):
    close = getattr(embedder, 'close', None)
    value = close() if callable(close) else None
    if hasattr(value, '__await__'):
        await value


def _item(recipe, use):
    uses = ('speaker-verify', 'speaker-tracks') if use == 'speaker-verify' else (use,)    # the same speaker model
    found = next((item for item in recipe['modules'] if item['use'] in uses), None)
    if found is None:
        raise SystemExit(f"{recipe['name']} has no {use} module to enroll for")
    return found


MAIN_SPEAKER_S = 15.    # --main-speaker: the first seconds whose most-speaking voice is enrolled


def choose_main_speaker(turns, start, end):
    """(speaker, [(start, end)] where that speaker alone speaks, seconds) of the speaker who speaks alone longest in
    [start, end): the "main speaker" of a recording. ``turns``: ``[(start, end, speaker)]``. None without speech."""
    from ..providers.word_verification import exclusive
    alone = exclusive(turns, start, end)
    if not alone:
        return None
    speaker = max(alone, key=lambda name: sum(high - low for low, high in alone[name]))
    return speaker, alone[speaker], round(sum(high - low for low, high in alone[speaker]), 2)


def _main_speaker(audio, args, offset_s):
    """Diarize the window with the verification lane's hosted diarizer (pyannoteAI Precision-3, its own consent) and
    keep the main speaker's own sound: (samples, enrolled_from)."""
    import numpy as np
    from ..providers.pyannoteai import PyannoteAI
    from ..providers.slot_sources import HostedBatch
    from ..providers.speaker_verify import SAMPLE_RATE
    from .perception_parts import VERIFY_DIARIZER, _registry_config
    config = dict(_registry_config('audio.diarization', VERIFY_DIARIZER) or {})
    config.pop('exclusive', None)
    if getattr(args, 'allow_remote_diarization', False):
        config['allow_remote'] = True
    try:
        client = PyannoteAI(**config)
    except PermissionError:
        raise SystemExit('--main-speaker sends the window to pyannoteAI (it leaves this device and is billed, 20 s '
                         'minimum): add --allow-remote-diarization to consent') from None
    turns = HostedBatch(client).diarize(audio)
    found = choose_main_speaker(turns, 0., len(audio) / SAMPLE_RATE)
    if found is None:
        raise SystemExit('--main-speaker: nobody speaks alone in the window')
    speaker, pieces, seconds = found
    sound = np.concatenate([audio[int(low * SAMPLE_RATE):int(high * SAMPLE_RATE)] for low, high in pieces])
    speakers = sorted({name for _, _, name in turns})
    return sound, dict(kind='main_speaker', diarizer=f'pyannoteai:{client.model}', speaker=speaker, speakers=len(speakers),
                       window_s=[round(offset_s, 2), round(offset_s + len(audio) / SAMPLE_RATE, 2)], speech_s=seconds,
                       segments_s=[[round(offset_s + low, 2), round(offset_s + high, 2)] for low, high in pieces],
                       cost_eur=round(client.spent_eur, 6))


async def enroll_voice(recipe, args, *, source, context, clip_s=3.):
    """Embeddings of the owner's voice in [start, start + seconds] of the input, one per ``clip_s``.

    ``--main-speaker``: not all of that window, only the sound of the speaker who speaks alone longest in it (pyannoteAI
    Precision-3; ``--allow-remote-diarization``), for a recording where nobody marked the person: the "main speaker"
    option (default window: the first 15 s). The saved voice says where it came from (``enrolled_from``: the range, or
    the main speaker's segments), so a score can leave out the words that are inside the reference.
    """
    import numpy as np
    from ..providers.audio_ring import mono_float, resample
    from ..providers.speaker_verify import SAMPLE_RATE, enroll
    from .perception_parts import speaker_embedder
    consent = consent_record(args.consent)
    embedder = await speaker_embedder(_item(recipe, 'speaker-verify'), context)
    chunks, seconds = [], 0.
    try:
        async for frame in source.frames():
            samples = mono_float(frame.data)
            begin, seconds = seconds, seconds + len(samples) / frame.sample_rate
            if begin >= args.start + args.seconds:
                break
            meter = getattr(args, 'meter', None)
            if callable(meter) and seconds > args.start:      # a live recording step shows its level
                from ..providers.audio_ring import rms_dbfs
                meter(min(seconds, args.start + args.seconds) - args.start, rms_dbfs(samples))
            if seconds > args.start:
                low = max(0, int((args.start - begin) * frame.sample_rate))
                high = min(len(samples), int((args.start + args.seconds - begin) * frame.sample_rate))
                chunks.append((samples[low:high], frame.sample_rate))
    finally:
        close = getattr(source, 'close', None)
        if callable(close):
            close()
    if not chunks:
        raise SystemExit('no audio in the enrollment window')
    rate = chunks[0][1]
    audio = np.concatenate([samples for samples, _ in chunks])
    audio = resample(audio, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else audio
    origin = dict(kind='range', from_s=round(args.start, 2), to_s=round(args.start + len(audio) / SAMPLE_RATE, 2))
    if getattr(args, 'main_speaker', False):
        audio, origin = _main_speaker(audio, args, args.start)
    size = int(clip_s * SAMPLE_RATE)
    clips = [audio[i:i + size] for i in range(0, len(audio), size)]
    clips = [clip for clip in clips if len(clip) >= SAMPLE_RATE] or [audio]
    try:
        saved = await enroll(embedder, clips)
    except ValueError as error:
        raise SystemExit(f'voice enrollment: {error}') from error
    finally:
        await _close(embedder)
    saved.update(recipe=recipe['name'], seconds=round(len(audio) / SAMPLE_RATE, 2), consent=consent,
                 enrolled_from=origin)
    _save(args.out, saved)
    print(json.dumps(dict(out=args.out, kind='voice', clips=len(saved['embeddings']), model=saved['model'],
                          enrolled_from={key: value for key, value in origin.items() if key != 'segments_s'})), flush=True)
    return saved


class _Joints:
    """Keeps the newest joints of the pose module in this process; commits nothing."""
    name, version = 'enrollment_joints', '1'

    def __init__(self):
        self.poses, self.timestamp_ms = [], None

    def process(self, frame):
        from ..private_payload import poses_of
        self.poses, self.timestamp_ms = poses_of(frame.data) or [], frame.timestamp_ms
        return []


async def enroll_person(recipe, args, *, build_pipeline, context, every_ms=200):
    """Face (and body) embeddings of the only visible person in the window; frames are not kept."""
    from .. import Gamst
    from ..providers.person_match import enroll_person as embed
    from .perception_parts import body_embedder, face_embedder
    consent = consent_record(args.consent)
    item = _item(recipe, 'target-person')
    face, body = await face_embedder(item, context), await body_embedder(item, context)
    upstream = [entry for entry in recipe['modules'] if entry['use'] in ('mediapipe-pose',)]
    found = dict(face=[], body=[])
    first, last, joints = {}, {}, _Joints()
    async with Gamst(consent=True) as human:
        pipeline = await build_pipeline(dict(recipe, references=[]), args, human, modules=upstream, lanes=False)
        pipeline.add_module('enrollment_joints', joints, consumes=[upstream[0]['joints']])

        async def on_frame(input_id, frame, records):
            first.setdefault('ms', frame.timestamp_ms)
            seconds = (frame.timestamp_ms - first['ms']) / 1000
            if seconds > args.start + args.seconds:
                return False
            if seconds < args.start or frame.timestamp_ms - last.get('ms', -every_ms) < every_ms:
                return True
            poses = joints.poses if joints.timestamp_ms == frame.timestamp_ms else []
            if len(poses) != 1:   # nobody, or it is unclear which person consented
                return True
            last['ms'] = frame.timestamp_ms
            try:
                result = await embed(face, body, [frame.data], poses)
            except ValueError:
                return True
            found['face'] += result['face']
            found['body'] += result['body']
            return True

        try:
            await pipeline.run(on_frame=on_frame, stop_on_failure=True)
        finally:
            await pipeline.close()
            await _close(face)
            await _close(body)
    if not found['face'] and not found['body']:
        raise SystemExit('no frame in the window showed exactly one person with a usable face')
    saved = dict(kind='person', recipe=recipe['name'], face_model=face.model_id,
                 body_model=getattr(body, 'model_id', None), consent=consent, **found)
    _save(args.out, saved)
    print(json.dumps(dict(out=args.out, kind='person', faces=len(found['face']), bodies=len(found['body']),
                          model=face.model_id)), flush=True)
    return saved


def saved_terms(path):
    """A terms file of ``gamst recipe enroll --kind terms`` (confirmed words and how they were given)."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get('kind') != 'terms' or not isinstance(value.get('terms'), dict) or not value['terms']:
        raise SystemExit(f'{path} is not a terms file')
    return value


def saved_reference(path, kind):
    """(value, consent) of an enrollment file; refused without a consent record."""
    value = json.loads(Path(path).read_text())
    if value.get('kind') != kind:
        raise SystemExit(f'{path} is not a {kind} enrollment')
    if not value.get('consent'):
        raise SystemExit(f'{path} has no consent record; enroll again with --consent')
    return value, value['consent']
