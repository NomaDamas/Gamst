"""A person reference (few-shot identity) and the check every use of ``who`` passes first (generic, no model).

Owner decision 44: a person is never identified by a name or a description. "Who" needs a few-shot reference of that
person, and GAMST refuses to start (a run, an instruction, an enrollment) when there is none or it fails this check. The
refusal says what is missing and how to give it, in the same words on the command line, over MCP and on the HUD
(:func:`refusal`, :func:`hud_line`).

A person reference is what ``gamst recipe enroll --kind person`` (or the MCP tool ``enroll_person``) saves, embeddings
only, never images::

    {"kind": "person", "name": "P", "consent": {...},
     "face_model": "lvface:...", "face": [[...], ...],          # aligned face crops (LVFace)
     "body_model": "osnet:...", "body": [[...], ...],           # whole-body crops (person re-ID)
     "voice": {"model", "embeddings", ...},                     # optional: a voice enrollment (gamst recipe enroll --kind voice)
     "samples": [{"source", "persons", "face", "body"}, ...],    # one entry per enrollment sample (photo or video frame)
     "session": "file:<digest>" | None,                         # where the body samples were seen
     "quality": {...}}                                          # this module's check at enrollment

What the check requires (:func:`check`):

- a consent record (who consented, to what, when);
- exactly one person in every sample (a sample with nobody or with several people is never enrolled);
- an identity that holds across sessions: at least :data:`MIN_FACES` usable face crops whose embeddings agree with each
  other (mean pairwise cosine >= :data:`MIN_FACE_AGREEMENT`), or an enrolled voice whose pieces agree
  (:mod:`gamst.voice_quality`);
- body (clothes, whole-body appearance) samples, when given, agree too (>= :data:`MIN_BODY_AGREEMENT`); appearance is
  trusted within one video or session only, because clothes change. A reference with body samples only is usable only
  in the session it was enrolled in (``session``); anywhere else it is refused (confirm by face or voice).

The agreement thresholds are parameters, not learnt values: they are the same kind of check as the voice reference's
self-consistency, and are not measured on labelled people yet (docs/person-identity.md says what was measured).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MIN_FACES = 3
MIN_BODIES = 3
MIN_FACE_AGREEMENT = .5
MIN_BODY_AGREEMENT = .5

HOW = ("give a few-shot person reference: CLI 'gamst recipe enroll RECIPE --kind person --name NAME --consent \"who "
       "consented, to what\" --out P.json' with --image a.jpg --image b.jpg, or --video FILE --start S --seconds N, or "
       "--camera 0 --seconds N (one person in view; add --voice-file V.json to include a voice), then run with --person "
       "P.json; MCP 'enroll_person' {name, consent, images | video+start_s+end_s | camera_seconds}")


def agreement(embeddings):
    """``{n, mean, min}``: mean and lowest pairwise cosine of the embeddings (None with fewer than two)."""
    from .voice_quality import self_consistency
    found = self_consistency(embeddings)
    return dict(n=found['pieces'], mean=found['mean'], min=found['min'])


def cleaned(embeddings, min_agreement):
    """``(kept, dropped)``: the embeddings without the few that disagree with the rest, when more than half of them agree
    (mean pairwise >= ``min_agreement``) once those are left out (:func:`gamst.voice_quality.consistent_majority`, the
    voice reference's rule). A crop the pose model aligned badly (a turned head) is one of those; two people never are a
    majority of one. Otherwise everything is kept and the check says the samples disagree."""
    from .voice_quality import consistent_majority
    embeddings = list(embeddings or ())
    found = agreement(embeddings)
    if found['mean'] is None or found['mean'] >= min_agreement:
        return embeddings, 0
    keep = consistent_majority(embeddings, min_agreement)
    if keep is None:
        return embeddings, 0
    return [embeddings[index] for index in keep], len(embeddings) - len(keep)


def session_of(path, *, block=1 << 22):
    """A short identity of one recording (its size, first and last 4 MB): body samples from it are trusted in it only."""
    file = Path(path)
    size = file.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with open(file, 'rb') as source:
        digest.update(source.read(block))
        if size > 2 * block:
            source.seek(size - block)
        digest.update(source.read(block))
    return 'file:' + digest.hexdigest()[:16]


def check(value, *, session=None, min_faces=MIN_FACES, min_bodies=MIN_BODIES, min_face=MIN_FACE_AGREEMENT,
          min_body=MIN_BODY_AGREEMENT):
    """``{ok, problems, basis, face, body, voice, session_only}`` of a person reference.

    ``problems``: what is missing or below the check, in words (empty when ``ok``). ``basis``: what identifies the
    person here (``face``, ``voice``; ``body`` only inside the session the body samples came from). ``session``: the
    run's own (:func:`session_of` of its video file; None for a camera or unknown)."""
    from .voice_quality import MIN_QUALITY as MIN_VOICE, self_consistency
    problems, basis = [], []
    if not isinstance(value, dict) or value.get('kind') != 'person':
        return dict(ok=False, problems=['this is not a person reference'], basis=[], face=None, body=None, voice=None,
                    session_only=False)
    if not value.get('consent'):
        problems.append('no consent record (who consented, to what, when)')
    crowded = [index for index, sample in enumerate(value.get('samples') or ()) if sample.get('persons') != 1]
    if crowded:
        problems.append(f'{len(crowded)} sample(s) do not show exactly one person')
    face, body = agreement(value.get('face') or ()), agreement(value.get('body') or ())
    if face['n'] >= min_faces and face['mean'] is not None and face['mean'] >= min_face:
        basis.append('face')
    voice = None
    if isinstance(value.get('voice'), dict) and value['voice'].get('embeddings'):
        voice = self_consistency(value['voice']['embeddings'])
        voice = dict(n=voice['pieces'], mean=voice['mean'], min=voice['min'])
        if voice['mean'] is None or voice['mean'] >= MIN_VOICE:     # one piece cannot be checked: not "low"
            basis.append('voice')
        else:
            problems.append(f"the voice samples disagree (mean pairwise {voice['mean']:.2f} < {MIN_VOICE:g}): "
                            'more than one voice, or noise')
    face_split = face['mean'] is not None and face['mean'] < min_face       # two people's faces, or poor crops
    body_split = body['mean'] is not None and body['mean'] < min_body
    if face_split:
        problems.append(f"the face crops disagree (mean pairwise {face['mean']:.2f} < {min_face:g}): more than one "
                        'person, or poor crops')
    if body_split:
        problems.append(f"the body crops disagree (mean pairwise {body['mean']:.2f} < {min_body:g}): more than one "
                        'person, or clothes changed')
    body_ok = body['n'] >= min_bodies and not body_split
    session_only = False
    if body_ok and value.get('session') is not None and value.get('session') == session:
        basis.append('body')            # clothes are trusted only in the video they were seen in
        session_only = basis == ['body']
    if not basis and not face_split and not body_split:
        if body_ok:
            problems.append('only whole-body appearance was enrolled: it is trusted only in the video it came from; '
                            'add face samples or a voice to confirm the person here')
        elif not face['n'] and not body['n'] and voice is None:
            problems.append('no face, body or voice samples')
        else:
            problems.append(f"only {face['n']} usable face crop(s), at least {min_faces} needed (a visible, frontal "
                            f"face)" + (f", and {body['n']} body crop(s), at least {min_bodies}" if body['n'] else ''))
    return dict(ok=not problems, problems=problems, basis=basis, face=face, body=body, voice=voice,
                session_only=session_only)


def refusal(who, problems=None):
    """The one sentence every surface (CLI, MCP, HUD) gives when ``who`` cannot be used."""
    what = '; '.join(problems) if problems else 'no person reference is enrolled for it'
    return f'who={who!r} refused before start: {what}. Few-shot identity is required: {HOW}'


def hud_line(who, problems=None):
    """The same refusal, short and ASCII (the HUD draws with cv2)."""
    what = '; '.join(problems) if problems else 'no person reference'
    text = f'WHO {who} REFUSED: {what} -> enroll a person (photos / video range / camera seconds)'
    return text.encode('ascii', 'replace').decode()


def name_of(value, default='person'):
    """The person's name in ``who``: what was enrolled, else ``default``."""
    name = value.get('name') if isinstance(value, dict) else None
    return name if isinstance(name, str) and name else default


def require(who, references, *, session=None):
    """``(reference name, value, check)`` of the person reference named ``who`` among ``references``
    (``{name: {kind, value}}``, a Pipeline's), else ValueError(:func:`refusal`)."""
    found = [(key, item) for key, item in (references or {}).items()
             if isinstance(item, dict) and item.get('kind') == 'person' and (name_of(item.get('value'), key) == who or key == who)]
    if not found:
        raise ValueError(refusal(who))
    key, item = found[0]
    verdict = check(item['value'], session=session)
    if not verdict['ok']:
        raise ValueError(refusal(who, verdict['problems']))
    return key, item['value'], verdict


def with_voice(person, voice):
    """The person reference with a voice enrollment (``gamst recipe enroll --kind voice``) as its ``voice``."""
    if not isinstance(voice, dict) or voice.get('kind') not in (None, 'voice') or not voice.get('embeddings'):
        raise ValueError('a voice is a file of gamst recipe enroll --kind voice')
    kept = {key: voice[key] for key in ('model', 'embeddings', 'quality', 'enrolled_from', 'seconds') if key in voice}
    return dict(person, voice=kept)
