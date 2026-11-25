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
    found = next((item for item in recipe['modules'] if item['use'] == use), None)
    if found is None:
        raise SystemExit(f"{recipe['name']} has no {use} module to enroll for")
    return found


async def enroll_voice(recipe, args, *, source, context, clip_s=3.):
    """Embeddings of the owner's voice in [start, start + seconds] of the input, one per ``clip_s``."""
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
            if seconds > args.start:
                chunks.append((samples, frame.sample_rate))
    finally:
        close = getattr(source, 'close', None)
        if callable(close):
            close()
    if not chunks:
        raise SystemExit('no audio in the enrollment window')
    rate = chunks[0][1]
    audio = np.concatenate([samples for samples, _ in chunks])
    audio = resample(audio, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else audio
    size = int(clip_s * SAMPLE_RATE)
    clips = [audio[i:i + size] for i in range(0, len(audio), size)]
    clips = [clip for clip in clips if len(clip) >= SAMPLE_RATE] or [audio]
    try:
        saved = await enroll(embedder, clips)
    except ValueError as error:
        raise SystemExit(f'voice enrollment: {error}') from error
    finally:
        await _close(embedder)
    saved.update(recipe=recipe['name'], seconds=round(len(audio) / SAMPLE_RATE, 2), consent=consent)
    _save(args.out, saved)
    print(json.dumps(dict(out=args.out, kind='voice', clips=len(saved['embeddings']), model=saved['model'])), flush=True)
    return saved


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
    first, last = {}, {}
    async with Gamst(consent=True) as human:
        pipeline = await build_pipeline(dict(recipe, references=[]), args, human, modules=upstream)

        async def on_frame(input_id, frame, records):
            first.setdefault('ms', frame.timestamp_ms)
            seconds = (frame.timestamp_ms - first['ms']) / 1000
            if seconds > args.start + args.seconds:
                return False
            if seconds < args.start or frame.timestamp_ms - last.get('ms', -every_ms) < every_ms:
                return True
            poses = next((record['observation']['value'] for record in records
                          if record['observation'].get('name') == 'scene.poses'), [])
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
            await pipeline.run(on_frame=on_frame)
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


def saved_reference(path, kind):
    """(value, consent) of an enrollment file; refused without a consent record."""
    value = json.loads(Path(path).read_text())
    if value.get('kind') != kind:
        raise SystemExit(f'{path} is not a {kind} enrollment')
    if not value.get('consent'):
        raise SystemExit(f'{path} has no consent record; enroll again with --consent')
    return value, value['consent']
