"""Owner-declared terms in final transcripts: one event per spoken occurrence (text math, no model).

Consumes ``speech.transcript_update`` values. Only ``final`` revisions count, once per utterance, so a
partial hypothesis that is later revised never adds to the count. Text is normalized before
matching: Unicode NFKC, lower case, whitespace and punctuation removed. Korean particles that follow
a word ("대박이", "대박을") still contain the term, so they match; a term is counted at each
non-overlapping position.

When the transcript carries word timings (``words: [{text, start_ms, end_ms}]``, e.g. from a forced
aligner) the event gets the matching words' times; otherwise the utterance's times.

Which voice a transcript belongs to is decided upstream: feed this module the transcript of a
derived input (one enrolled person's separated voice) to count what that person said.
"""
from __future__ import annotations

import re
import threading
import unicodedata

from ..capabilities import CapabilitySpec, OutputField

_DROP = re.compile(r'[\s\W_]+', re.UNICODE)


def normalize(text):
    return _DROP.sub('', unicodedata.normalize('NFKC', str(text)).lower())


def occurrences(text, term):
    """Start offsets (in normalized text) of non-overlapping occurrences of ``term``."""
    haystack, needle = normalize(text), normalize(term)
    found, start = [], 0
    while needle:
        index = haystack.find(needle, start)
        if index < 0:
            break
        found.append(index)
        start = index + len(needle)
    return found


def _word_times(words, offset, length):
    """Times of the words covering normalized characters [offset, offset+length)."""
    position, covered = 0, []
    for word in words:
        size = len(normalize(word.get('text', '')))
        if size and position < offset + length and position + size > offset:
            covered.append(word)
        position += size
    if not covered:
        return None
    return covered[0].get('start_ms'), covered[-1].get('end_ms')


class TermEvents:
    def __init__(self, *, reference='terms', terms=None, max_utterances=4096):
        self.reference_name, self.max_utterances = reference, max_utterances
        self.terms = {}
        if terms is not None:
            self.set_terms(terms)
        self.spec = CapabilitySpec(
            'speech.term_events', 'speech.term_events', 'audio', 'gamst.term_events', '1',
            (OutputField('count', 'integer'), OutputField('counts', 'object'), OutputField('events', 'array')),
            'native', stateful=True, freshness_ms=60000, interpretation='measurement', devices=('cpu',),
            license='Apache-2.0')
        self._lock = threading.RLock()
        self._clear()

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'terms':
            if not self.terms:
                raise ValueError(f'term_events needs a terms reference named {self.reference_name!r}')
            return
        self.set_terms(reference['value'])

    def set_terms(self, value):
        """A list of terms, or {term: [spellings ...]} so variants count as the same term."""
        mapping = {str(term): [str(term)] for term in value} if isinstance(value, (list, tuple)) else \
            {str(term): [str(item) for item in (spellings or [term])] for term, spellings in dict(value).items()}
        if not mapping or any(not normalize(item) for spellings in mapping.values() for item in spellings):
            raise ValueError('terms must be non-empty text')
        self.terms = mapping

    def _clear(self):
        self._counted, self._counts, self._total = {}, {term: 0 for term in self.terms}, 0

    def reset(self):
        with self._lock:
            self._clear()

    def process(self, frame):
        with self._lock:
            value = frame.data if isinstance(frame.data, dict) else {}
            if value.get('status') != 'final' or not value.get('text'):
                return None
            key = (value.get('utterance_id'), frame.subject_id)
            if key in self._counted:
                return None
            self._counted[key] = True
            while len(self._counted) > self.max_utterances:
                self._counted.pop(next(iter(self._counted)))
            events = []
            for term, spellings in self.terms.items():
                taken = []
                for spelling in sorted(spellings, key=lambda item: -len(normalize(item))):
                    size = len(normalize(spelling))
                    for offset in occurrences(value['text'], spelling):
                        if any(offset < end and offset + size > start for start, end in taken):
                            continue  # a longer spelling already counted these characters
                        taken.append((offset, offset + size))
                        times = _word_times(value['words'], offset, size) if value.get('words') else None
                        start_ms, end_ms = times if times else (value.get('start_ms'), value.get('end_ms'))
                        self._counts[term] = self._counts.get(term, 0) + 1
                        self._total += 1
                        events.append(dict(event='term.spoken', term=term, spoken=spelling,
                                           utterance_id=value.get('utterance_id'), start_ms=start_ms,
                                           end_ms=end_ms, word_timed=times is not None))
            if not events:
                return None
            return self.spec.validate(dict(count=self._total, counts=dict(self._counts), events=events))
