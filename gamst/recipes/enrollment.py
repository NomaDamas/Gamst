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
    from ..voice_quality import self_consistency
    saved.update(recipe=recipe['name'], seconds=round(len(audio) / SAMPLE_RATE, 2), consent=consent,
                 enrolled_from=origin, quality=self_consistency(saved['embeddings']))
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


PERSON_PARTS = ('person-binding', 'target-person')


async def sample(face, body, image, pose, face_indices=None):
    """``(face embedding or None, body embedding or None)`` of one person in one image (nothing is kept)."""
    from ..providers.person_match import MEDIAPIPE_FACE_POINTS, align_face, body_crop, face_points, face_reading
    from ..providers.speaker_verify import embed
    height, width = image.shape[:2]
    found = []
    indices, visibility = face_reading(pose, face_indices or MEDIAPIPE_FACE_POINTS)
    crops = (align_face(image, face_points(pose.get('landmarks') or [], width, height, indices, min_visibility=visibility))
             if face is not None else None,
             body_crop(image, pose['bbox']) if body is not None and pose.get('bbox') else None)
    for embedder, crop in zip((face, body), crops):
        try:
            found.append(None if crop is None else await embed(embedder, crop))
        except ValueError:          # too small or unclear for the model
            found.append(None)
    return tuple(found)


def person_value(name, recipe, face, body, found, consent, *, session=None, origin=None, voice=None):
    """The saved person reference (embeddings only) with its check (:func:`gamst.person_reference.check`)."""
    from ..person_reference import MIN_BODY_AGREEMENT, MIN_FACE_AGREEMENT, check, cleaned, with_voice
    faces, dropped_faces = cleaned(found['face'], MIN_FACE_AGREEMENT)       # badly aligned crops (a turned head)
    bodies, dropped_bodies = cleaned(found['body'], MIN_BODY_AGREEMENT)
    value = dict(kind='person', name=name, recipe=recipe['name'] if recipe else None,
                 face_model=getattr(face, 'model_id', None), body_model=getattr(body, 'model_id', None), consent=consent,
                 face=faces, body=bodies, samples=found['samples'], session=session, enrolled_from=origin or {},
                 dropped=dict(face=dropped_faces, body=dropped_bodies))
    if voice is not None:
        value = with_voice(value, voice)
    value['quality'] = check(value, session=session)
    return value


async def enroll_person(recipe, args, *, build_pipeline, context, every_ms=200):
    """A person reference (few-shot, owner decision 44): face and whole-body embeddings of the one person in still images
    (``--image``, every image must show exactly one person), or in the frames of a window of a video or camera where
    exactly one person is visible; optionally with a voice enrollment (``--voice-file``). Refused, and nothing saved,
    when it fails :func:`gamst.person_reference.check`. Frames and images are not kept."""
    from .. import Gamst
    from ..inputs import ImageFilesInput
    from ..person_reference import refusal
    from ..providers.person_match import MEDIAPIPE_FACE_POINTS
    from . import run_session
    from .perception_parts import body_embedder, face_embedder
    consent = consent_record(args.consent)
    item = next((entry for entry in recipe['modules'] if entry['use'] in PERSON_PARTS), None)
    if item is None:
        raise SystemExit(f"{recipe['name']} has no person-binding or target-person module to enroll for")
    name = getattr(args, 'name', None) or Path(args.out).stem.split('.')[0]
    voice = None
    if getattr(args, 'voice_file', None):
        voice = json.loads(Path(args.voice_file).read_text())
    images = list(getattr(args, 'image', None) or ())
    upstream = [entry for entry in recipe['modules'] if entry['use'] in ('mediapipe-pose', 'rtmw-pose', 'sapiens2-pose')][:1]
    if not upstream:
        raise SystemExit(f"{recipe['name']} has no pose module to find the person with")
    face, body = await face_embedder(item, context), await body_embedder(item, context)
    found = dict(face=[], body=[], samples=[])
    crowded, first, last, joints = [], {}, {}, _Joints()
    sources = None
    if images:
        inputs = [entry for entry in recipe['inputs'] if entry['use'] == 'camera'][:1] or recipe['inputs'][:1]
        sources = {inputs[0]['id']: ImageFilesInput(images)}
    try:
        async with Gamst(consent=True) as human:
            pipeline = await build_pipeline(dict(recipe, references=[]), args, human, modules=upstream, lanes=False,
                                            sources=sources)
            pipeline.add_module('enrollment_joints', joints, consumes=[upstream[0]['joints']])
            counter = dict(index=0)

            async def on_frame(input_id, frame, records):
                first.setdefault('ms', frame.timestamp_ms)
                poses = joints.poses if joints.timestamp_ms == frame.timestamp_ms else []
                if images:
                    where = images[counter['index']]
                    counter['index'] += 1
                    if len(poses) != 1:      # nobody, or it is unclear which person consented
                        crowded.append(f'{Path(where).name}: {len(poses)} people')
                        return True
                else:
                    seconds = (frame.timestamp_ms - first['ms']) / 1000
                    if seconds > args.start + args.seconds:
                        return False
                    if seconds < args.start or frame.timestamp_ms - last.get('ms', -every_ms) < every_ms:
                        return True
                    if len(poses) != 1:
                        crowded.append(round(seconds, 2))
                        return True
                    last['ms'] = frame.timestamp_ms
                    where = round(seconds, 2)
                kind = poses[0].get('keypoint_set')
                face_points = None if kind else MEDIAPIPE_FACE_POINTS
                embeddings = await sample(face, body, frame.data, poses[0], face_points)
                for key, value in zip(('face', 'body'), embeddings):
                    if value is not None:
                        found[key].append(value)
                found['samples'].append(dict(source='image' if images else 'frame', at=where, persons=1,
                                             face=embeddings[0] is not None, body=embeddings[1] is not None))
                return True

            try:
                await pipeline.run(on_frame=on_frame, stop_on_failure=True)
            finally:
                await pipeline.close()
    finally:
        await _close(face)
        await _close(body)
    if images and crowded:
        raise SystemExit(refusal(name, ['every photo must show exactly one person: ' + ', '.join(crowded)]))
    session = None if images or getattr(args, 'camera', None) is not None else run_session(args)
    origin = dict(kind='images', files=[Path(path).name for path in images]) if images else dict(
        kind='camera' if getattr(args, 'camera', None) is not None else 'range',
        from_s=round(args.start, 2), to_s=round(args.start + args.seconds, 2), skipped_crowded=len(crowded))
    saved = person_value(name, recipe, face, body, found, consent, session=session, origin=origin, voice=voice)
    if not saved['quality']['ok']:
        raise SystemExit(refusal(name, saved['quality']['problems']))
    _save(args.out, saved)
    print(json.dumps(dict(out=args.out, kind='person', name=name, faces=len(saved['face']), bodies=len(saved['body']),
                          samples=len(found['samples']), skipped_frames=len(crowded), dropped=saved['dropped'],
                          basis=saved['quality']['basis'],
                          face_agreement=saved['quality']['face']['mean'], body_agreement=saved['quality']['body']['mean'],
                          model=face.model_id), ensure_ascii=False), flush=True)
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
