"""The person's voice references and the voices known not to be the person, shared by the lanes of one run.

Speaker decision v2 (2026-11-26, measured with leave-one-clip-out on the owner's four clips:
``scripts/qa_speaker_decision_v2.py``). A fixed score of one short snippet against one reference cannot tell the
person from a similar voice standing next to them; a comparison can:

- **references** (``person``): the enrolment's embeddings, plus, with ``grow``, the person's own speech that the
  verification lane decided final (the diarized speaker it named the person in a window: never a provisional word);
  scored by ``combine``: ``centroid`` (cosine to the mean) or ``max`` (the closest reference);
- **negatives**: voices known not to be the person, taken automatically, without labels, from diarization: the other
  speakers of a window in which the verification lane clearly found the person;
- a sound is the person's when it is closer to the references than to every negative by a margin (:meth:`decide`).

The live lane (L1, :class:`gamst.providers.speaker_verify.SpeakerVerify` with ``relative``) uses what the bank holds
at the moment a word is committed; the verification lane (L4, :class:`gamst.providers.word_verification`
``decision: "relative"``) fills it. Embeddings stay in memory (at most ``max_person`` / ``max_negatives``, oldest
dropped); nothing is written to a record.
"""
from __future__ import annotations

import threading

from .speaker_verify import centroid, cosine, l2_normalize

COMBINE = ('centroid', 'max')


class VoiceBank:
    def __init__(self, embeddings, *, combine='centroid', grow=False, max_person=32, max_negatives=32):
        if combine not in COMBINE:
            raise ValueError(f"combine is one of {', '.join(COMBINE)}")
        self.combine, self.grow = combine, bool(grow)
        self.max_person, self.max_negatives = int(max_person), int(max_negatives)
        self.base = [l2_normalize(vector) for vector in embeddings]
        if not self.base:
            raise ValueError('a voice bank needs the enrolled embeddings')
        self.added, self.negatives = [], []
        self._lock = threading.Lock()
        self._centroid = centroid(self.base)

    def person(self):
        return self.base + [vector for _, vector in self.added]

    def pos(self, vector):
        """How close a sound is to the person: cosine to the references' mean, or to the closest reference."""
        with self._lock:
            if self.combine == 'max':
                return max(cosine(vector, item) for item in self.person())
            return cosine(vector, self._centroid)

    def neg(self, vector, extra=()):
        """Cosine to the closest negative (and ``extra`` vectors), None without any."""
        with self._lock:
            pool = [item for _, item in self.negatives] + [l2_normalize(item) for item in extra]
        return max((cosine(vector, item) for item in pool), default=None)

    def decide(self, vector, *, floor, margin, extra=()):
        """(person?, pos, neg): at least ``floor`` to the person and ``margin`` closer than to every negative."""
        pos, neg = self.pos(vector), self.neg(vector, extra)
        return pos >= floor and (neg is None or pos - neg >= margin), round(pos, 5), None if neg is None else round(neg, 5)

    def add_person(self, vector, key):
        if not self.grow:
            return False
        with self._lock:
            if any(old == key for old, _ in self.added):
                return False
            self.added.append((key, l2_normalize(vector)))
            del self.added[:-self.max_person]
            self._centroid = centroid(self.person())
        return True

    def add_negative(self, vector, key):
        with self._lock:
            if any(old == key for old, _ in self.negatives):
                return False
            self.negatives.append((key, l2_normalize(vector)))
            del self.negatives[:-self.max_negatives]
        return True

    def status(self):
        return dict(combine=self.combine, grow=self.grow, references=len(self.base) + len(self.added),
                    added=len(self.added), negatives=len(self.negatives))


def shared(pipeline, reference, embeddings, settings=None):
    """The one bank of ``reference`` in this pipeline (made by the first lane that asks)."""
    banks = getattr(pipeline, '_voice_banks', None)
    if banks is None:
        banks = pipeline._voice_banks = {}
    if reference not in banks:
        banks[reference] = VoiceBank(embeddings, **(settings or {}))
    return banks[reference]
