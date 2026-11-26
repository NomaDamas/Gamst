"""Was a word said by the enrolled person? The voice decides; lips are consulted only where the voice cannot.

Pipeline module :class:`SpeakerEvidence` sits between ``speaker_check`` (:class:`gamst.providers.speaker_verify.SpeakerVerify`
with ``words=True``: every word carries its voice score ``speaker_score``) and ``terms``
(:class:`gamst.providers.term_events.TermEvents`). It also reads the records of ``vision.active_speaker``
(:class:`gamst.providers.active_speaker.ActiveSpeaker`) and lines both up by capture time. The owner's rule, applied to
every word:

1. The voice score is the primary, absolute criterion. Below ``low`` the word is not the person's; at or above
   ``high`` it is. In between (the ambiguous band) the voice alone would decide by ``threshold``.
2. Only in the band are the faces consulted, and only faces that passed the active-speaker gate (size, turn, mouth in
   the picture, sharpness) in at least ``min_slots`` 40 ms slots of the word (``margin_ms`` added on each side):

   - *for*: the enrolled face (identity >= ``face_threshold``) moves its lips with the sound (mean speaking logit
     >= ``speaking_logit``), and no other face does;
   - *against*: another face (identity < ``face_reject``) moves its lips with the sound, and the enrolled face does not;
   - everything else is *unknown*: no readable face, a face of unclear identity, both speaking, or nobody speaking.
     A face that is not seen, turned away, too small or blurred is never evidence against.

   For or against decides the word; unknown leaves the voice's decision.
3. Every word says what decided it: ``decided_by`` is ``voice`` (outside the band, or the lips were unknown),
   ``lips`` (the lips decided against what the voice alone would have) or ``both`` (they agree), with the numbers that
   were used (``lips``: slots read and mean logit of the enrolled face and of the most speaking other face, and the
   verdict). No position, crop or point is in it.

A word waits until the active-speaker records cover its end plus ``margin_ms`` (they come later than the sound by
their lookahead); at the end of the input every waiting word is decided with what is there. Band edges and the gate
are parameters; see docs/measurements/av-speaker-2026-11-26.md for how the defaults were set and what they did on one clip.
"""
from __future__ import annotations

from ..capabilities import CapabilitySpec, OutputField

SLOT_MS = 40
# Default band around the voice threshold: [threshold - BELOW, threshold + ABOVE). Set on one clip (see the module doc).
BELOW, ABOVE = .20, .10


class SpeakerEvidence:
    def __init__(self, *, threshold=.35, low=None, high=None, margin_ms=100, min_slots=3, speaking_logit=0.,
                 face_threshold=.35, face_reject=.15, keep_ms=60000):
        low = threshold - BELOW if low is None else low
        high = threshold + ABOVE if high is None else high
        if not low <= threshold <= high:
            raise ValueError('require low <= threshold <= high')
        if not face_reject <= face_threshold:
            raise ValueError('require face_reject <= face_threshold')
        self.threshold, self.low, self.high = float(threshold), float(low), float(high)
        self.margin_ms, self.min_slots, self.speaking_logit = int(margin_ms), int(min_slots), float(speaking_logit)
        self.face_threshold, self.face_reject, self.keep_ms = float(face_threshold), float(face_reject), int(keep_ms)
        self.spec = CapabilitySpec(
            'speech.speaker_evidence', 'speech.speaker_evidence', 'audio', 'gamst.speaker_evidence', '1',
            (OutputField('kind', 'string'), OutputField('reference', 'string'), OutputField('start_ms', 'integer'),
             OutputField('end_ms', 'integer'), OutputField('score', 'number', required=False),
             OutputField('match', 'boolean'), OutputField('state', 'string'),
             OutputField('utterance_id', 'string', required=False), OutputField('status', 'string', required=False),
             OutputField('text', 'string', required=False), OutputField('words', 'array', required=False),
             OutputField('language', 'string', required=False), OutputField('speaker', 'object', required=False),
             OutputField('model', 'object'), OutputField('rule', 'object')),
            'native', stateful=True, freshness_ms=60000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.reset()

    def rule(self):
        return dict(threshold=self.threshold, low=self.low, high=self.high, margin_ms=self.margin_ms,
                    min_slots=self.min_slots, speaking_logit=self.speaking_logit,
                    face_threshold=self.face_threshold, face_reject=self.face_reject)

    def reset(self):
        self.slots = {}           # slot -> [(track, identity, logit or None)]
        self.until_ms = None      # the active-speaker records cover capture time up to here
        self.pending = []         # utterances waiting for their lips

    def process(self, frame):
        value = frame.data if isinstance(frame.data, dict) else {}
        if 'faces' in value and 'slot_ms' in value:
            self._lips(value)
        elif value.get('kind') == 'utterance':
            if value.get('status') != 'final':
                return None
            self.pending.append(value)
        else:
            return None
        return self._release(final=False)

    def finish(self):
        return self._release(final=True)

    def _lips(self, value):
        first = value['start_ms'] // SLOT_MS
        for face in value['faces']:
            for index, logit in enumerate(face.get('speaking') or ()):
                self.slots.setdefault(first + index, []).append((face.get('track'), face.get('identity'), logit))
        self.until_ms = max(self.until_ms or 0, value['end_ms'])
        oldest = (self.until_ms - self.keep_ms) // SLOT_MS
        for slot in [slot for slot in self.slots if slot < oldest]:
            del self.slots[slot]

    def _release(self, *, final):
        ready, waiting = [], []
        for value in self.pending:
            ends = [word['end_ms'] for word in value.get('words') or () if isinstance(word, dict)
                    and type(word.get('end_ms')) is int] or [value['end_ms']]
            covered = self.until_ms is not None and self.until_ms >= max(ends) + self.margin_ms
            (ready if final or covered else waiting).append(value)
        self.pending = waiting
        outputs = [self._decide(value) for value in ready]
        return outputs or None

    def lips(self, start_ms, end_ms):
        """(verdict, numbers) of the faces during [start_ms, end_ms]: verdict 'for', 'against' or 'unknown'."""
        first, last = (start_ms - self.margin_ms) // SLOT_MS, (end_ms + self.margin_ms) // SLOT_MS + 1
        tracks = {}
        for slot in range(first, last):
            for track, identity, logit in self.slots.get(slot, ()):
                entry = tracks.setdefault(track, dict(identity=identity, logits=[]))
                if identity is not None:
                    entry['identity'] = identity
                if logit is not None:
                    entry['logits'].append(logit)
        target, others = None, []
        for entry in tracks.values():
            if len(entry['logits']) < self.min_slots or entry['identity'] is None:
                continue          # not readable, or of unknown identity: no evidence either way
            mean = sum(entry['logits']) / len(entry['logits'])
            if entry['identity'] >= self.face_threshold:
                if target is None or len(entry['logits']) > target[0]:
                    target = (len(entry['logits']), mean)
            elif entry['identity'] < self.face_reject:
                others.append((len(entry['logits']), mean))
        other = max(others, key=lambda item: item[1]) if others else None
        target_speaks = target is not None and target[1] >= self.speaking_logit
        other_speaks = other is not None and other[1] >= self.speaking_logit
        verdict = 'for' if target_speaks and not other_speaks else 'against' if other_speaks and not target_speaks \
            else 'unknown'
        numbers = dict(verdict=verdict, faces=len(tracks))
        if target is not None:
            numbers.update(person_slots=target[0], person_logit=round(target[1], 3))
        if other is not None:
            numbers.update(other_slots=other[0], other_logit=round(other[1], 3))
        return verdict, numbers

    def word(self, item):
        """The word with its final ``match`` and what decided it."""
        score = item.get('speaker_score')
        voice = score is not None and score >= self.threshold
        if score is not None and not self.low <= score < self.high:
            return dict(item, match=voice, voice_match=voice, decided_by='voice')
        verdict, numbers = self.lips(item['start_ms'], item['end_ms'])
        if verdict == 'unknown':
            return dict(item, match=voice, voice_match=voice, decided_by='voice', lips=numbers)
        match = verdict == 'for'
        return dict(item, match=match, voice_match=voice, decided_by='both' if match == voice else 'lips', lips=numbers)

    def _decide(self, value):
        words = [self.word(item) if isinstance(item, dict) and type(item.get('start_ms')) is int
                 and type(item.get('end_ms')) is int else item for item in value.get('words') or ()]
        result = {key: item for key, item in value.items() if key in (
            'kind', 'reference', 'start_ms', 'end_ms', 'score', 'utterance_id', 'status', 'text', 'language', 'model')}
        checked = [item for item in words if isinstance(item, dict) and 'decided_by' in item]
        match = any(item['match'] for item in checked) if checked else bool(value.get('match'))
        speaker = dict(value.get('speaker') or {}, match=match)
        result.update(words=words, match=match, speaker=speaker, rule=self.rule(),
                      state='match' if match else value.get('state', 'other') if value.get('state') != 'match' else 'other')
        return self.spec.validate(result)
