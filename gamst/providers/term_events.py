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

The Pipeline may give other words while a run goes on (:meth:`TermEvents.prepare`, :meth:`TermEvents.schedule`): a word
belongs to the words that were asked for when it was said. Every event and count carries ``reference_id`` and
``reference_revision``; each revision has its own counts and its own records. In an utterance without word times that
goes on across the change, both the old and the new words are looked for, and such events say so (``across_change``).

Text-pattern styles of speaking (a filler, a sentence ending) are terms too: a term written as
``re:<pattern>`` is a regular expression over the normalized text (lower case, no spaces or
punctuation), e.g. ``re:(거든요|잖아요)$``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


def check_spelling(spelling):
    """Refuse a spelling that can never match (nothing but spaces and punctuation, or an invalid pattern)."""
    if not _valid(spelling):
        raise ValueError(f'{spelling!r} has nothing to match')


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


KEPT = 16       # revisions an utterance that arrives late may still belong to


@dataclass
class _Words:
    """One revision of the words to count, and what was counted under it."""
    terms: dict
    origin: dict
    identity: dict = field(default_factory=dict)
    from_ms: int = 0
    until_ms: int | None = None
    counts: dict = field(default_factory=dict)
    total: int = 0

    def clear(self):
        self.counts, self.total = {term: 0 for term in self.terms}, 0


def parse_terms(value):
    """({term: [spellings]}, origin) of a list of terms, {term: [spellings ...]} (variants count as the same term), or
    a saved terms file."""
    origin = dict(method='text', confirmed=True)
    if isinstance(value, dict) and value.get('kind') == 'terms' and isinstance(value.get('terms'), dict):
        origin = dict(method=str(value.get('method', 'examples')), confirmed=value.get('confirmed') is True)
        value = value['terms']
    mapping = {str(term): [str(term)] for term in value} if isinstance(value, (list, tuple)) else \
        {str(term): [str(item) for item in (spellings or [term])] for term, spellings in dict(value).items()}
    if not mapping or any(not _valid(item) for spellings in mapping.values() for item in spellings):
        raise ValueError('terms must be non-empty text')
    return mapping, origin


class TermEvents:
    def __init__(self, *, reference='terms', terms=None, max_utterances=4096, speaker_bound=True):
        self.reference_name, self.max_utterances = reference, max_utterances
        self.speaker_bound = bool(speaker_bound)
        self._lock = threading.RLock()
        self._words, self._counted = [], {}      # revisions, oldest first; a removed reference is one without words
        if terms is not None:
            self.set_terms(terms)
        self.spec = CapabilitySpec(
            'speech.term_events', 'speech.term_events', 'audio', 'gamst.term_events', '2',
            (OutputField('count', 'integer'), OutputField('counts', 'object'), OutputField('events', 'array'),
             OutputField('reference_id', 'string', required=False), OutputField('reference_revision', 'integer', required=False)),
            'native', stateful=True, freshness_ms=60000, interpretation='measurement', devices=('cpu',),
            license='Apache-2.0')

    @property
    def terms(self):
        return self._words[-1].terms if self._words else {}

    def bind(self, pipeline):
        reference = pipeline.references.get(self.reference_name)
        if reference is None or reference['kind'] != 'terms':
            if not self.terms:
                raise ValueError(f'term_events needs a terms reference named {self.reference_name!r}')
            return
        self._words = [self.prepare(reference)]
        self._words[0].from_ms = reference.get('from_ms', 0)

    def set_terms(self, value):
        """What to count, from the start (:func:`parse_terms`)."""
        words = _Words(*parse_terms(value))
        words.clear()
        self._words = [words]

    def prepare(self, reference):
        """Checked words to count from a stated time on; nothing of the running ones changes."""
        words = _Words(*parse_terms(reference['value']), identity={
            f'reference_{key}': reference[key] for key in ('id', 'revision') if reference.get(key) is not None})
        words.clear()
        return words

    def offline(self, reference):
        """A separate counter for records that were kept (nothing of it is committed)."""
        module = TermEvents(reference=self.reference_name, speaker_bound=self.speaker_bound)
        module._words = [self.prepare(reference)]
        return module

    def schedule(self, words, from_ms):
        """``words`` (None: nothing is counted) are counted from what is said at ``from_ms`` on."""
        with self._lock:
            words = _Words({}, {}) if words is None else words
            words.from_ms = from_ms
            self._words = [item for item in self._words if item.from_ms < from_ms][-KEPT + 1:]
            if self._words:
                self._words[-1].until_ms = from_ms
            self._words.append(words)

    def status(self):
        words = self._words[-1] if self._words else None
        return {} if words is None else dict(words.identity, terms=len(words.terms), count=words.total, from_ms=words.from_ms)

    def reset(self):
        with self._lock:
            self._counted = {}
            for words in self._words:
                words.clear()

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
            start, end = (value[key] if type(value.get(key)) in (int, float) else frame.timestamp_ms
                          for key in ('start_ms', 'end_ms'))
            found = [self._find(words, value, start, end) for words in self._words
                     if words.terms and words.from_ms <= end and (words.until_ms is None or start < words.until_ms)]
            found = [item for item in found if item is not None]
            return None if not found else found[0] if len(found) == 1 else found

    def _find(self, words, value, start, end):
        """One record of what this revision of the words has in an utterance, or None."""
        events = []
        for term, spellings in words.terms.items():
            taken = []
            for spelling in sorted(spellings, key=lambda item: -len(normalize(item))):
                for offset, size in matches(value['text'], spelling):
                    if any(offset < stop and offset + size > first for first, stop in taken):
                        continue  # a longer spelling already counted these characters
                    taken.append((offset, offset + size))
                    times = _word_times(value['words'], offset, size) if value.get('words') else None
                    start_ms, end_ms = times if times else (value.get('start_ms'), value.get('end_ms'))
                    across = not times and (start < words.from_ms or words.until_ms is not None and end >= words.until_ms)
                    if times and type(start_ms) in (int, float) and not (
                            words.from_ms <= start_ms and (words.until_ms is None or start_ms < words.until_ms)):
                        continue  # said when other words were asked for
                    words.counts[term] = words.counts.get(term, 0) + 1
                    words.total += 1
                    spoken = normalize(value['text'])[offset:offset + size] if spelling.startswith('re:') else spelling
                    events.append(dict(event='term.spoken', term=term, spoken=spoken, reference=self.reference_name,
                                       **words.identity, utterance_id=value.get('utterance_id'), start_ms=start_ms,
                                       end_ms=end_ms, word_timed=times is not None, **words.origin,
                                       **({'across_change': True} if across else {}),
                                       **({} if self.speaker_bound else {'speaker': 'not_bound'})))
        if not events:
            return None
        return self.spec.validate(dict(count=words.total, counts=dict(words.counts), events=events, **words.identity))
