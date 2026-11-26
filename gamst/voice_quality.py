"""How trustworthy a voice reference is, and what to use when it is not (generic; any enrolled voice).

A voice reference is a few embeddings of one person's speech (``gamst recipe enroll --kind voice``: one per 3 s piece,
whether the pieces came from an owner sample, a marked range or the automatic "main speaker" of a recording). If the
pieces are one person's, they agree with each other; if the reference mixes two people (a diarizer that merged two
voices into its main speaker), they do not, and everything decided against it inherits the mix.

- **quality** (:func:`self_consistency`): the mean pairwise cosine similarity of the reference's embeddings (and the
  lowest pair). One embedding cannot be checked: its quality is ``None`` (unmeasured), never "low".
- **check** (:func:`choose`): a reference below ``min_quality`` is not used as it is. In order: the run's own sample of
  the person (``--voice FILE``, ``--voice-range``, ``--voice-wav``, ``--record-voice-first``) is already preferred over
  the automatic main speaker (:mod:`gamst.recipes.voice_sample`); then the same reference without its disagreeing
  pieces when more than half of them agree (:func:`consistent_majority`, ``used: cleaned``: one off piece, e.g. music
  under the voice, is not a second person); then the first other reference of the same person the owner provided (``--voice-fallback FILE``, same speaker model) whose own quality passes, either ``combined`` with the
  low one (default: both with equal weight, the centroid of the two centroids) or ``replace``-ing it. Without a usable
  fallback the low reference is kept and flagged. Either way the run report says so (``voice_quality``) and the HUD
  shows "목소리 기준 품질 낮음".

Measured (owner's four clips, CAM++, automatic main speaker of the first 15 s): 뽀또 0.42 (two pieces; the reference
mixes the two men: no majority, so it falls back), 시발 0.45 (one of three pieces is off: the other two agree 0.67,
so it is cleaned, not replaced), 몇기야 0.63, 달 0.68; the person's 2-minute sample from
another recording 0.52 (40 pieces, so 0.5 is close to its own level: a parameter, not a learnt value). Offline on the owner's labelled words (v2 study cache), the other recording
alone loses every person word of 몇기야 (3 of 5) and 시발 (7 of 7) through the microphone change, while both combined keep
them all and cut 뽀또's other-voice words given to the person from 6 to 2: hence ``combined`` by default.
"""
from __future__ import annotations

from pathlib import Path

MIN_QUALITY = .5
MODES = ('combined', 'replace')


def self_consistency(embeddings):
    """``{pieces, mean, min}``: the mean and lowest pairwise cosine of the embeddings (None with fewer than two)."""
    from .providers.speaker_verify import cosine, l2_normalize
    vectors = [l2_normalize(item) for item in embeddings or ()]
    pairs = [cosine(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return dict(pieces=len(vectors), mean=round(sum(pairs) / len(pairs), 3) if pairs else None,
                min=round(min(pairs), 3) if pairs else None)


def consistent_majority(embeddings, min_quality=MIN_QUALITY):
    """Indices of more than half of the pieces that agree (mean pairwise >= ``min_quality``) once the pieces that
    disagree most with the others are left out, one at a time; None when no such majority exists (two pieces never
    have one: which of them would be the person?)."""
    from .providers.speaker_verify import cosine, l2_normalize
    vectors = [l2_normalize(item) for item in embeddings or ()]
    keep = list(range(len(vectors)))
    if len(keep) < 3:
        return None
    similar = {(i, j): cosine(vectors[i], vectors[j]) for i in keep for j in keep if i != j}
    mean = lambda items: sum(similar[i, j] for i in items for j in items if i < j) / (len(items) * (len(items) - 1) / 2)
    while len(keep) - 1 > len(vectors) / 2 and mean(keep) < min_quality:
        keep.remove(min(keep, key=lambda i: sum(similar[i, j] for j in keep if j != i)))
    return keep if mean(keep) >= min_quality and len(keep) < len(vectors) else None


def is_low(quality, min_quality=MIN_QUALITY):
    return quality.get('mean') is not None and quality['mean'] < min_quality


def combine(*values):
    """One voice reference of several, each with the same weight: the centroid of each becomes one embedding."""
    from .providers.speaker_verify import centroid
    models = {value.get('model') for value in values}
    if len(models) != 1:
        raise ValueError(f"voice references of different speaker models cannot be combined: {', '.join(map(str, models))}")
    first = values[0]
    combined = dict(first, embeddings=[centroid(value['embeddings']) for value in values],
                    combined_from=[value.get('enrolled_from') or dict(kind='file') for value in values])
    return combined


def _consent(*records):
    records = [record for record in records if record]
    if len(records) == 1:
        return records[0]
    return dict(statement=' + '.join(str(record.get('statement', record)) if isinstance(record, dict) else str(record)
                                     for record in records),
                recorded_ms=max((record.get('recorded_ms') or 0 for record in records if isinstance(record, dict)),
                                default=None))


def choose(value, consent, fallbacks=(), *, min_quality=None, mode=None, origin=None):
    """(value, consent, report) of the voice reference to use (see the module doc).

    ``fallbacks``: ``[(name, value, consent)]`` of other references of the same person, in the order given.
    ``report``: ``{quality, min_quality, low, used, fallback?}`` with ``used`` one of ``reference`` (the reference alone),
    ``cleaned`` (without its disagreeing pieces: ``kept``, ``dropped``, ``cleaned_quality``), ``combined``, ``replaced``
    or ``reference_low`` (low, nothing better to use).
    """
    min_quality = MIN_QUALITY if min_quality is None else float(min_quality)
    mode = mode or 'combined'
    if mode not in MODES:
        raise ValueError(f"voice fallback mode is one of {', '.join(MODES)}")
    if 'embeddings' not in (value or {}):        # samples embedded later: nothing to measure here
        return value, consent, dict(quality=None, min_quality=min_quality, low=False, used='reference')
    quality = self_consistency(value['embeddings'])
    report = dict(quality=quality, min_quality=min_quality, low=is_low(quality, min_quality), used='reference',
                  **({'origin': origin} if origin else {}))
    if not report['low']:
        return value, consent, report
    keep = consistent_majority(value['embeddings'], min_quality)
    if keep is not None:      # most pieces agree: the person's own sample without the pieces that do not
        report.update(used='cleaned', kept=len(keep), dropped=quality['pieces'] - len(keep),
                      cleaned_quality=self_consistency([value['embeddings'][i] for i in keep]))
        return dict(value, embeddings=[value['embeddings'][i] for i in keep]), consent, report
    report['used'] = 'reference_low'
    tried = []
    for name, other, other_consent in fallbacks:
        found = self_consistency(other.get('embeddings'))
        usable = 'embeddings' in other and other.get('model') == value.get('model') and not is_low(found, min_quality)
        tried.append(dict(name=name, quality=found, usable=usable,
                          **({} if other.get('model') == value.get('model') else {'why': 'another speaker model'})))
        if not usable:
            continue
        report['fallback'] = tried[-1]
        if mode == 'replace':
            report['used'] = 'replaced'
            return dict(other, replaced=value.get('enrolled_from') or dict(kind='file')), other_consent, report
        report['used'] = 'combined'
        return combine(value, other), _consent(consent, other_consent), report
    if tried:
        report['tried'] = tried
    return value, consent, report


def from_args(value, consent, args):
    """:func:`choose` with the command line's ``--voice-fallback``, ``--voice-min-quality`` and ``--voice-fallback-mode``."""
    from .recipes.enrollment import saved_reference
    fallbacks = []
    for path in getattr(args, 'voice_fallback', None) or ():
        other, other_consent = saved_reference(path, 'voice')
        fallbacks.append((Path(path).name, other, other_consent))
    return choose(value, consent, fallbacks, min_quality=getattr(args, 'voice_min_quality', None),
                  mode=getattr(args, 'voice_fallback_mode', None), origin=((value or {}).get('enrolled_from') or {}).get('kind'))


HUD_LOW = '목소리 기준 품질 낮음'
USED_WORD = {'combined': '다른 녹음 샘플과 합쳐 씀', 'replaced': '다른 녹음 샘플로 바꿈', 'reference_low': '대신 쓸 샘플 없음',
             'cleaned': '어긋난 조각 빼고 씀'}


def hud_line(report):
    """The HUD's Korean line for a low-quality reference (None when it is fine or unmeasured)."""
    if not isinstance(report, dict) or not report.get('low'):
        return None
    mean = (report.get('quality') or {}).get('mean')
    used = USED_WORD.get(report.get('used'), '')
    if report.get('used') == 'cleaned':
        used = f"어긋난 조각 {report['dropped']}개 빼고 씀"
    return f"{HUD_LOW} ({mean:.2f} < {report['min_quality']:.2f}) · {used}".rstrip(' ·')
