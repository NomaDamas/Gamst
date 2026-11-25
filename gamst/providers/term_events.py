"""Owner-declared terms in final transcripts: one event per spoken occurrence (text math, no model).

Consumes ``speech.transcript_update`` values, or plain ``text.transcript`` strings of a windowed ASR
(each window is a final utterance). Only ``final`` revisions count, once per utterance, so a
partial hypothesis that is later revised never adds to the count. Text is normalized before
matching: Unicode NFKC, lower case, whitespace and punctuation removed. Korean particles that follow
a word ("대박이", "대박을") still contain the term, so they match; a term is counted at each
non-overlapping position.

When the transcript carries word timings (``words: [{text, start_ms, end_ms}]``, e.g. from a forced
aligner) the event gets the matching words' times; otherwise the utterance's times.

Which voice a transcript belongs to is decided upstream: feed this module the transcript of a
derived input (one enrolled person's separated voice) to count what that person said. A transcript
that carries ``speaker: {"match": false}`` (re-checked by ``gamst.providers.speaker_verify``) is not
counted.

Each event says how its term was given (``method``: ``text``, or ``examples`` of ``gamst recipe enroll --kind
terms``, with ``confirmed``) and, when nothing upstream ties the sound to an enrolled voice, ``speaker:
"not_bound"``.

Text-pattern styles of speaking (a filler, a sentence ending) are terms too: a term written as
``re:<pattern>`` is a regular expression over the normalized text (lower case, no spaces or
punctuation), e.g. ``re:(거든요|잖아요)$``.
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
    return [offset for offset, _ in matches(text, term)]


def matches(text, term):
    """(offset, length) in normalized text of non-overlapping occurrences; ``re:`` terms are patterns."""
    haystack = normalize(text)
    if str(term).startswith('re:'):
        pattern = _pattern(term)
        return [(found.start(), found.end() - found.start()) for found in pattern.finditer(haystack)
                if found.end() > found.start()]
    needle = normalize(term)
    found, start = [], 0
    while needle:
        index = haystack.find(needle, start)
        if index < 0:
            break
        found.append((index, len(needle)))
        start = index + len(needle)
    return found


_PATTERNS = {}


def _pattern(term):
    if term not in _PATTERNS:
        _PATTERNS[term] = re.compile(unicodedata.normalize('NFKC', term[3:]), re.IGNORECASE)
    return _PATTERNS[term]


def _valid(spelling):
    if spelling.startswith('re:'):
        try:
            return bool(spelling[3:]) and _pattern(spelling) is not None
        except re.error as exc:
            raise ValueError(f'invalid term pattern {spelling!r}: {exc}') from exc
    return bool(normalize(spelling))


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
    def __init__(self, *, reference='terms', terms=None, max_utterances=4096, speaker_bound=True):
        self.reference_name, self.max_utterances = reference, max_utterances
        self.speaker_bound, self.origin = bool(speaker_bound), dict(method='text', confirmed=True)
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
        """A list of terms, or {term: [spellings ...]} so variants count as the same term, or a saved terms file."""
        if isinstance(value, dict) and value.get('kind') == 'terms' and isinstance(value.get('terms'), dict):
            self.origin = dict(method=str(value.get('method', 'examples')), confirmed=value.get('confirmed') is True)
            value = value['terms']
        mapping = {str(term): [str(term)] for term in value} if isinstance(value, (list, tuple)) else \
            {str(term): [str(item) for item in (spellings or [term])] for term, spellings in dict(value).items()}
        if not mapping or any(not _valid(item) for spellings in mapping.values() for item in spellings):
            raise ValueError('terms must be non-empty text')
        self.terms = mapping

    def _clear(self):
        self._counted, self._counts, self._total = {}, {term: 0 for term in self.terms}, 0

    def reset(self):
        with self._lock:
            self._clear()

    def process(self, frame):
        with self._lock:
            value = frame.data
            if isinstance(value, str):
                # A fixed-window transcript (e.g. ``text.transcript`` of a windowed ASR) is final
                # when emitted; each window is its own utterance.
                value = dict(utterance_id=f'window:{frame.timestamp_ms}', status='final', text=value,
                             end_ms=frame.timestamp_ms)
            value = value if isinstance(value, dict) else {}
            if value.get('status') != 'final' or not value.get('text'):
                return None
            speaker = value.get('speaker')
            if isinstance(speaker, dict) and speaker.get('match') is False:
                return None  # re-checked upstream (speaker_verify): not the enrolled person's voice
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
                    for offset, size in matches(value['text'], spelling):
                        if any(offset < end and offset + size > start for start, end in taken):
                            continue  # a longer spelling already counted these characters
                        taken.append((offset, offset + size))
                        times = _word_times(value['words'], offset, size) if value.get('words') else None
                        start_ms, end_ms = times if times else (value.get('start_ms'), value.get('end_ms'))
                        self._counts[term] = self._counts.get(term, 0) + 1
                        self._total += 1
                        spoken = normalize(value['text'])[offset:offset + size] if spelling.startswith('re:') \
                            else spelling
                        events.append(dict(event='term.spoken', term=term, spoken=spoken,
                                           utterance_id=value.get('utterance_id'), start_ms=start_ms,
                                           end_ms=end_ms, word_timed=times is not None, **self.origin,
                                           **({} if self.speaker_bound else {'speaker': 'not_bound'})))
            if not events:
                return None
            return self.spec.validate(dict(count=self._total, counts=dict(self._counts), events=events))
