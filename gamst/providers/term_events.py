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
counted. Where a word also says what decided it (``decided_by`` and ``lips`` of
:mod:`gamst.providers.speaker_evidence`), its event says so too.

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

Whole-word terms (:func:`spelling_parts`). ``word:<spelling>`` matches only a word of the recogniser that is exactly
the spelling once normalized (``word:18`` matches "18" and "18!", never "180", "2018" or "18살"); ``word:<spelling>*``
matches a word made only of repetitions of it and counts each repetition (``word:달*``: "달" 1, "달달" 2, "달달달" 3;
never "달라", "달걀", "달려"). Words are the recogniser's own: the whitespace-separated tokens of its text (Korean and
other space-delimited scripts; normalization removes spaces, so the joined text cannot say where a word ends; the
aligner's word list cannot either: Qwen3-ForcedAligner splits a Korean word into a stem and an ending, "달달달" into
"달달" + "달"). Nothing may follow the spelling inside the word, Korean particles included: a whole-word term is used
where the spelling is also part of other words, and the particle forms ("달이", "달도", "18을") are the other meaning's
(the moon, the number) as often as not, so allowing them would bring back the matches the term is there to refuse.
Several repetitions in one recogniser word share the word's time evenly, and each has its own word ID
(:func:`part_id`), so the verification lane decides each one. A term given as a list names itself without the
prefix (``word:달*`` counts as the term "달").

Flags. A spelling may end in ``#flag`` (``십팔#may_be_number``); its events say ``flags: [...]``. A spelling made
only of digits or other numerals (``18``, ``十八``) is flagged ``may_be_number`` by itself: the event may be the number,
not the word that sounds like it.

Revision states (:mod:`gamst.counting`). Every event has a stable ``event_id``. Where the words that decide it say
``state: "provisional"`` (:class:`gamst.providers.speaker_tracks.SpeakerTracks`), so does the event: it is counted at
once. A later ``verdicts`` value (``{"kind": "verdicts", "verdicts": [{"word_id", "match", "reason"?}]}``, from
:class:`gamst.providers.speaker_tracks.TrackVerification`) settles it: an ``event.revision`` marks it ``final``, or
``revoked`` with the reason, and ``count`` is provisional + final. A ``term.other_voice`` that the verdict gives to
the enrolled voice is revoked and a ``term.spoken`` that ``replaces`` it is committed as final.

Heard words (owner decision 26: count at once, correct later, visibly). A ``status: "partial"`` record whose words say
``state: "heard"`` (:class:`gamst.providers.speaker_verify.SpeakerVerify` with ``partials``: the counted words that are
stable in the streaming recogniser's partial hypothesis, each with its voice on the sound so far) makes a
``term.spoken`` event in state ``heard``, counted at once, with estimated times. A heard word whose voice is clearly
another's (``match: false``) is not counted; the final decides it. Without a voice module upstream, ``partials``
(:class:`gamst.providers.partial_terms.PartialSpotter` settings) finds them in the raw partial records here. When the
final of the turn arrives its occurrences are paired with the heard events by capture time
(:func:`gamst.providers.partial_terms.reconcile`): a paired one is revised to the state the final gives it
(``provisional``, ``uncertain``, ``final``) with the word's own times, the same event (never counted twice), and later
verdicts settle it under that ID; a final occurrence that is another voice's revokes its heard event (``other_voice``) and
is reported as before; a heard event the final does not hold is ``revoked`` (``partial_withdrawn``), as is one whose
turn's final never came (a later final of the same recogniser starts after it was heard, or the input ended).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import re
import threading
import unicodedata

from ..capabilities import CapabilitySpec, OutputField

_DROP = re.compile(r'[\s\W_]+', re.UNICODE)


def normalize(text):
    return _DROP.sub('', unicodedata.normalize('NFKC', str(text)).lower())


WORD = 'word:'
NUMBER = 'may_be_number'
_FLAG = re.compile(r'#([A-Za-z_][A-Za-z0-9_]*)$')


def spelling_parts(spelling):
    """What a spelling asks for: ``{kind: "text" | "re" | "word", body, repeat, flags}``.

    ``re:PATTERN``: a pattern over the normalized text. ``word:SPELLING`` (``word:SPELLING*``: repeated): whole words
    only. Otherwise text, found anywhere (inside other words too). ``#flag`` at the end (not for patterns) is kept on the
    events; numerals are flagged ``may_be_number`` by themselves.
    """
    spelling = str(spelling)
    if spelling.startswith('re:'):
        return dict(kind='re', body=spelling, repeat=False, flags=())
    flags = []
    while True:
        found = _FLAG.search(spelling)
        if found is None or not normalize(spelling[:found.start()]):
            break
        flags.insert(0, found.group(1))
        spelling = spelling[:found.start()]
    kind, repeat = 'text', False
    if spelling.startswith(WORD):
        kind, spelling = 'word', spelling[len(WORD):]
        if spelling.endswith('*'):
            repeat, spelling = True, spelling[:-1]
    body = normalize(spelling)
    if body and body.isnumeric() and NUMBER not in flags:
        flags.append(NUMBER)
    return dict(kind=kind, body=spelling.strip(), repeat=repeat, flags=tuple(flags))


def plain(spelling):
    """The spelling as a recogniser should be told it (no ``word:``, ``*`` or flags); None for a pattern."""
    parts = spelling_parts(spelling)
    return None if parts['kind'] == 're' else parts['body']


def term_name(spelling):
    """The name a term given as a list has: its spelling without ``word:``, ``*`` and flags (patterns as given)."""
    return plain(spelling) or str(spelling)


def part_id(word_id, index):
    """The ID of the ``index``-th repetition inside one recogniser word (``word:SPELLING*``)."""
    return f'{word_id}~{index}'


def _tokens(text):
    """(offset in the normalized text, normalized token) of each whitespace-separated word of ``text``."""
    position, found = 0, []
    for raw in str(text).split():
        token = normalize(raw)
        if token:
            found.append((position, token))
            position += len(token)
    return found


def spans(text, term):
    """``[(offset, length, part)]`` in normalized text of non-overlapping occurrences of one spelling. ``part`` is
    ``(index, repetitions, word offset, word length)`` for a repeated whole word of more than one repetition, else
    None."""
    parts = spelling_parts(term)
    if parts['kind'] != 'word':
        return [(offset, size, None) for offset, size in matches(text, term)]
    needle = normalize(parts['body'])
    found = []
    for offset, token in _tokens(text):
        if token == needle:
            found.append((offset, len(needle), None))
        elif parts['repeat'] and len(token) % len(needle) == 0 and token == needle * (len(token) // len(needle)):
            times = len(token) // len(needle)
            found += [(offset + index * len(needle), len(needle), (index, times, offset, len(token)))
                      for index in range(times)]
    return found


def occurrences(text, term):
    """Start offsets (in normalized text) of non-overlapping occurrences of ``term``."""
    return [offset for offset, _ in matches(text, term)]


def matches(text, term):
    """(offset, length) in normalized text of non-overlapping occurrences; ``re:`` terms are patterns, ``word:`` terms
    whole words (:func:`spelling_parts`)."""
    if spelling_parts(term)['kind'] == 'word':
        return [(offset, size) for offset, size, _ in spans(text, term)]
    haystack = normalize(text)
    if str(term).startswith('re:'):
        pattern = _pattern(term)
        return [(found.start(), found.end() - found.start()) for found in pattern.finditer(haystack)
                if found.end() > found.start()]
    needle = normalize(spelling_parts(term)['body'])
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
    return bool(normalize(spelling_parts(spelling)['body']))


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
    return covered[0].get('start_ms'), covered[-1].get('end_ms'), covered


def _part_times(words, offset, size, part):
    """:func:`_word_times`; a repetition inside a word gets its even share of the word's time."""
    if part is None:
        return _word_times(words, offset, size)
    index, repetitions, first, length = part
    times = _word_times(words, first, length)
    if times is None or type(times[0]) not in (int, float) or type(times[1]) not in (int, float):
        return times
    start, end = times[:2]
    return (start + (end - start) * index // repetitions, start + (end - start) * (index + 1) // repetitions, times[2])


def _whole_word(words, part):
    """``{word_start_ms, word_end_ms}`` of the recogniser word a repetition is part of (empty without word times)."""
    times = _word_times(words or (), part[2], part[3]) if words else None
    if not times or type(times[0]) not in (int, float) or type(times[1]) not in (int, float):
        return {}
    return dict(word_start_ms=int(times[0]), word_end_ms=int(times[1]))


def _found(text, spellings):
    """``[(offset, size, spelling, part)]``: occurrences of a term's spellings, longer spellings first, never two over
    the same characters."""
    found, taken = [], []
    for spelling in sorted(spellings, key=lambda item: -len(normalize(spelling_parts(item)['body']))):
        for offset, size, part in spans(text, spelling):
            if any(offset < stop and offset + size > first for first, stop in taken):
                continue  # a longer spelling already counted these characters
            taken.append((offset, offset + size))
            found.append((offset, size, spelling, part))
    return found


def _spoken(text, offset, size, spelling):
    return normalize(text)[offset:offset + size] if spelling.startswith('re:') else spelling_parts(spelling)['body']


def word_ids(covered, part):
    """The word IDs one occurrence is decided by (a repetition: its own :func:`part_id` of each word)."""
    ids = [word['word_id'] for word in covered if isinstance(word.get('word_id'), str)]
    return ids if part is None else [part_id(item, part[0]) for item in ids]


def term_occurrences(value, terms):
    """Where the counted words occur in one final utterance, as :class:`TermEvents` finds them: ``[{term, spoken,
    start_ms, end_ms, words, timed, word_ids, part, flags}]`` (``words``: the utterance's words covering it; ``timed``
    False: no word times, the utterance's own times; ``part``: ``[index, repetitions]`` of a repeated whole word).
    ``terms``: ``{term: [spellings]}`` (:func:`parse_terms`)."""
    found = []
    text, words = str(value.get('text') or ''), value.get('words') or []
    for term, spellings in terms.items():
        for offset, size, spelling, part in _found(text, spellings):
            times = _part_times(words, offset, size, part) if words else None
            start, end = times[:2] if times else (value.get('start_ms'), value.get('end_ms'))
            if type(start) not in (int, float) or type(end) not in (int, float):
                continue
            covered = list(times[2]) if times else []
            found.append(dict(term=term, spoken=_spoken(text, offset, size, spelling), start_ms=int(start),
                              end_ms=int(end), words=covered, timed=times is not None,
                              word_ids=word_ids(covered, part), part=None if part is None else list(part[:2]),
                              flags=list(spelling_parts(spelling)['flags'])))
    return sorted(found, key=lambda item: item['end_ms'])


KEPT = 16       # revisions an utterance that arrives late may still belong to
OPEN = 4096     # provisional events (and word verdicts) held until they are settled; the oldest is let go


def event_id(*parts):
    """A stable ID of one counted occurrence (same utterance, place and words: same ID)."""
    return 'te:' + hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:20]


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
    mapping = {term_name(term): [str(term)] for term in value} if isinstance(value, (list, tuple)) else \
        {str(term): [str(item) for item in (spellings or [term])] for term, spellings in dict(value).items()}
    if not mapping or any(not _valid(item) for spellings in mapping.values() for item in spellings):
        raise ValueError('terms must be non-empty text')
    return mapping, origin


WITHDRAWN = 'partial_withdrawn'
MATCH_MS = 4000     # a heard word and the final's occurrence of the same turn are the same word within this


class TermEvents:
    def __init__(self, *, reference='terms', terms=None, max_utterances=4096, speaker_bound=True, report_others=False,
                 partials=None, match_ms=MATCH_MS):
        self.report_others = bool(report_others)     # also say (never count) what another voice said
        self.partials, self.spotter, self.match_ms = None if partials is None else dict(partials), None, int(match_ms)
        self._heard = OrderedDict()        # heard events by ID: [event, words] (counted until reconciled)
        self.heard_stats = dict(heard=0, other_voice=0, confirmed=0, withdrawn=0, revoked_other_voice=0, late=0)
        self.reference_name, self.max_utterances = reference, max_utterances
        self.speaker_bound = bool(speaker_bound)
        self._lock = threading.RLock()
        self._words, self._counted = [], {}      # revisions, oldest first; a removed reference is one without words
        self._open, self._verdicts = OrderedDict(), OrderedDict()   # provisional events by ID; verdicts by word ID
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
        self._spot()

    def _spot(self):
        """The spotter of raw partials (``partials``) for the words counted now."""
        if self.partials is not None and self._words:
            from .partial_terms import PartialSpotter
            self.spotter = PartialSpotter(self._words[-1].terms, **self.partials)

    def set_terms(self, value):
        """What to count, from the start (:func:`parse_terms`)."""
        words = _Words(*parse_terms(value))
        words.clear()
        self._words = [words]
        self._spot()

    def prepare(self, reference):
        """Checked words to count from a stated time on; nothing of the running ones changes."""
        words = _Words(*parse_terms(reference['value']), identity={
            f'reference_{key}': reference[key] for key in ('id', 'revision') if reference.get(key) is not None})
        words.clear()
        return words

    def offline(self, reference):
        """A separate counter for records that were kept (nothing of it is committed)."""
        module = TermEvents(reference=self.reference_name, speaker_bound=self.speaker_bound,
                            report_others=self.report_others)
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
        heard = {'heard': dict(self.heard_stats, open=len(self._heard))} if any(self.heard_stats.values()) else {}
        return {} if words is None else dict(words.identity, terms=len(words.terms), count=words.total, from_ms=words.from_ms,
                                             **heard)

    def reset(self):
        with self._lock:
            self._counted = {}
            self._open.clear()
            self._verdicts.clear()
            self._heard.clear()
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
            if value.get('kind') == 'verdicts':
                return self._settle(value.get('verdicts') or ())
            if value.get('status') == 'partial':
                return self._partial(value, frame)
            if value.get('status') != 'final' or not value.get('text'):
                return None
            if self.spotter is not None:
                self.spotter.final(value.get('utterance_id'))
            speaker = value.get('speaker')
            whole = isinstance(speaker, dict) and speaker.get('match') is False
            if whole and not self.report_others and not any(
                    isinstance(word, dict) and word.get('match') for word in value.get('words') or ()):
                return None  # re-checked upstream (speaker_verify): not the enrolled person's voice
            key = (value.get('utterance_id'), frame.subject_id)
            if key in self._counted:
                return None
            self._counted[key] = True
            while len(self._counted) > self.max_utterances:
                self._counted.pop(next(iter(self._counted)))
            start, end = (value[key] if type(value.get(key)) in (int, float) else frame.timestamp_ms
                          for key in ('start_ms', 'end_ms'))
            stale = self._stale(value, start)
            found = [self._find(words, value, start, end) for words in self._words
                     if words.terms and words.from_ms <= end and (words.until_ms is None or start < words.until_ms)]
            found = [item for item in stale + found if item is not None]
            return None if not found else found[0] if len(found) == 1 else found

    def _find(self, words, value, start, end):
        """One record of what this revision of the words has in an utterance, or None."""
        events = []
        for term, spellings in words.terms.items():
            for offset, size, spelling, part in _found(value['text'], spellings):
                times = _part_times(value['words'], offset, size, part) if value.get('words') else None
                start_ms, end_ms = times[:2] if times else (value.get('start_ms'), value.get('end_ms'))
                across = not times and (start < words.from_ms or words.until_ms is not None and end >= words.until_ms)
                if times and type(start_ms) in (int, float) and not (
                        words.from_ms <= start_ms and (words.until_ms is None or start_ms < words.until_ms)):
                    continue  # said when other words were asked for
                speaker = value['speaker'] if isinstance(value.get('speaker'), dict) else {}
                other, score = speaker.get('match') is False, speaker.get('score')
                checked = [word for word in (times[2] if times else ()) if type(word.get('match')) is bool]
                decided = {}
                if checked:      # each word's own voice, where people share a phrase
                    other, score = not all(word['match'] for word in checked), checked[0].get('speaker_score')
                    decided = {key: checked[0][key] for key in ('decided_by', 'lips', 'basis', 'reason')
                               if key in checked[0]}
                ids = word_ids(times[2] if times else (), part)
                provisional = bool(ids) and any(word.get('state') == 'provisional' for word in times[2])
                uncertain = bool(ids) and not provisional and any(word.get('state') == 'uncertain' for word in times[2])
                if uncertain:       # neither the person's nor another's yet: shown, not counted, settled later
                    other = False
                if other and not self.report_others:
                    continue
                if not other and not uncertain:
                    words.total += 1
                    words.counts[term] = words.counts.get(term, 0) + 1
                spoken = _spoken(value['text'], offset, size, spelling)
                flags = spelling_parts(spelling)['flags']
                slot = {key: checked[0][key] for key in ('slot', 'slot_score') if checked and key in checked[0]}
                event = dict(event='term.other_voice' if other else 'term.spoken',
                             **({'state': 'provisional'} if provisional else {'state': 'uncertain'} if uncertain
                                else {}), term=term,
                             event_id=event_id(value.get('utterance_id'), offset, term, *words.identity.values()),
                             spoken=spoken, reference=self.reference_name, **({'flags': list(flags)} if flags else {}),
                             **({'repetition': list(part[:2])} if part else {}),
                             **(_whole_word(value.get('words'), part) if part else {}),
                             **words.identity, utterance_id=value.get('utterance_id'), start_ms=start_ms,
                             end_ms=end_ms, word_timed=times is not None, **words.origin,
                             **({'across_change': True} if across else {}),
                             **({} if score is None else {'speaker_score': score}), **decided, **slot,
                             **({'word_ids': ids} if provisional or uncertain else {}),
                             **({} if self.speaker_bound else {'speaker': 'not_bound'}))
                if provisional or uncertain:     # [event, words, counted in the total now]
                    self._open[event['event_id']] = [event, words, provisional and not other]
                    while len(self._open) > OPEN:
                        self._open.popitem(last=False)
                events.append(event)
        events = self._reconcile(words, value, start, end, events)
        if not events:
            return None
        return self.spec.validate(dict(count=words.total, counts=dict(words.counts), events=events, **words.identity))

    # ---------------------------------------------------------------- heard words (partial hypotheses)

    def _partial(self, value, frame):
        """Heard words of one partial record (module doc): ``term.spoken`` events in state ``heard``, counted at once."""
        if (value.get('utterance_id'), frame.subject_id) in self._counted:
            self.heard_stats['late'] += 1
            return None       # its final is here already
        heard = [word for word in value.get('words') or () if isinstance(word, dict) and word.get('state') == 'heard']
        if not heard and self.spotter is not None and value.get('text'):
            heard = [dict(text=item['spoken'], state='heard', term=item['term'], index=item['index'],
                          word_id=f"heard:{item['utterance_id']}:{item['term']}:{item['index']}",
                          start_ms=item['start_ms'], end_ms=item['end_ms'], heard_to_ms=item['heard_to_ms'],
                          word_timed=item['timed'], **({'flags': item['flags']} if item['flags'] else {}),
                          **({'part': item['part']} if item['part'] else {}))
                     for item in self.spotter.update(value)]
        changed = OrderedDict()
        for word in heard:
            start = word.get('start_ms')
            if type(start) not in (int, float) or type(word.get('end_ms')) not in (int, float):
                continue
            words = next((item for item in reversed(self._words) if word.get('term') in item.terms
                          and item.from_ms <= start and (item.until_ms is None or start < item.until_ms)), None)
            if words is None:
                continue
            key = event_id('heard', word.get('word_id'), *words.identity.values())
            if key in self._heard or key in self._open:
                continue
            if word.get('match') is False:     # clearly another voice already: the final decides it
                self.heard_stats['other_voice'] += 1
                continue
            flags, part = word.get('flags') or [], word.get('part')
            event = dict(event='term.spoken', state='heard', term=word['term'], event_id=key, spoken=word.get('text'),
                         reference=self.reference_name, **({'flags': list(flags)} if flags else {}),
                         **({'repetition': list(part)} if part else {}), **words.identity,
                         utterance_id=value.get('utterance_id'), start_ms=int(start), end_ms=int(word['end_ms']),
                         word_timed=bool(word.get('word_timed')), heard_to_ms=word.get('heard_to_ms'), **words.origin,
                         **({'speaker_score': word['speaker_score']} if 'speaker_score' in word else {}),
                         **({'l1': word['l1']} if 'l1' in word else {}),
                         **({} if self.speaker_bound else {'speaker': 'not_bound'}))
            self._heard[key] = [event, words]
            while len(self._heard) > OPEN:
                self._heard.popitem(last=False)
            words.total += 1
            words.counts[event['term']] = words.counts.get(event['term'], 0) + 1
            self.heard_stats['heard'] += 1
            changed.setdefault(id(words), (words, []))[1].append(event)
        records = [self.spec.validate(dict(count=words.total, counts=dict(words.counts), events=events, **words.identity))
                   for words, events in changed.values()]
        return None if not records else records[0] if len(records) == 1 else records

    def _withdraw(self, key, reason=WITHDRAWN):
        """The revision that takes a heard event out of the count (its final did not hold it)."""
        event, words = self._heard.pop(key)
        words.total -= 1
        words.counts[event['term']] = words.counts.get(event['term'], 0) - 1
        self.heard_stats['withdrawn' if reason == WITHDRAWN else 'revoked_other_voice'] += 1
        return words, dict(event='event.revision', state='revoked', revises=key, revised_event=event['event'],
                           term=event['term'], start_ms=event['start_ms'], end_ms=event['end_ms'], reason=reason,
                           by='l1', **words.identity)

    def _source(self, utterance_id):
        return str(utterance_id).rsplit(':', 1)[0]

    def _stale(self, value, start):
        """Heard events of the same recogniser whose turn's final never came: this later final starts after they were
        heard."""
        source, changed = self._source(value.get('utterance_id')), OrderedDict()
        for key in [key for key, (event, _) in self._heard.items()
                    if event['utterance_id'] != value.get('utterance_id') and self._source(event['utterance_id']) == source
                    and type(event.get('heard_to_ms')) in (int, float) and event['heard_to_ms'] <= start]:
            words, revision = self._withdraw(key)
            changed.setdefault(id(words), (words, []))[1].append(revision)
        return [self.spec.validate(dict(count=words.total, counts=dict(words.counts), events=events, **words.identity))
                for words, events in changed.values()]

    def _reconcile(self, words, value, start, end, events):
        """Pair this final's occurrences with the heard events of its turn (module doc); returns the record's events."""
        from .partial_terms import reconcile
        utterance = value.get('utterance_id')
        mine = [key for key, (event, owner) in self._heard.items() if owner is words and (
            event['utterance_id'] == utterance or self._source(event['utterance_id']) == self._source(utterance)
            and start - self.match_ms <= event['end_ms'] <= end + self.match_ms)]
        if not mine:
            return events
        out, paired = list(events), {}
        for term in {self._heard[key][0]['term'] for key in mine}:
            heard = [(self._heard[key][0]['end_ms'], key) for key in mine if self._heard[key][0]['term'] == term]
            final = [(event['end_ms'], index) for index, event in enumerate(events) if event.get('term') == term
                     and event.get('event') in ('term.spoken', 'term.other_voice') and type(event.get('end_ms')) is int]
            paired.update({index: key for key, index in reconcile(heard, final, max_ms=self.match_ms)})
        for index, key in paired.items():
            event = events[index]
            if event['event'] == 'term.other_voice':      # the final's voice check: clearly another's
                out.append(self._withdraw(key, 'other_voice')[1])
                continue
            heard, _ = self._heard.pop(key)
            state = event.get('state', 'final')
            # the heard event was counted, and the final's occurrence again if it counts (uncertain: not): once only
            words.total -= 1
            words.counts[event['term']] = words.counts.get(event['term'], 0) - 1
            if event['event_id'] in self._open:     # later verdicts settle it under the heard event's ID
                entry = self._open.pop(event['event_id'])
                entry[0] = dict(entry[0], event_id=key, heard_end_ms=heard['end_ms'])
                self._open[key] = entry
            self.heard_stats['confirmed'] += 1
            out[index] = dict(event='event.revision', state=state, revises=key, revised_event='term.spoken',
                              term=event['term'], start_ms=event['start_ms'], end_ms=event['end_ms'], by='l1',
                              **words.identity, **{name: event[name] for name in (
                                  'utterance_id', 'word_timed', 'word_ids', 'speaker_score', 'l1', 'reason', 'decided_by')
                                  if name in event})
        for key in mine:
            if key in self._heard:
                out.append(self._withdraw(key)[1])
        return out

    def finish(self):
        """End of input: heard words whose final never came leave the count."""
        with self._lock:
            changed = OrderedDict()
            for key in list(self._heard):
                words, revision = self._withdraw(key)
                changed.setdefault(id(words), (words, []))[1].append(revision)
            records = [self.spec.validate(dict(count=words.total, counts=dict(words.counts), events=events,
                                               **words.identity)) for words, events in changed.values()]
            return None if not records else records[0] if len(records) == 1 else records

    def _settle(self, verdicts):
        """Revisions of the open events whose words all have a verdict now (one record per revision of the words).

        A verdict with ``uncertain: true`` (overlapped speech whose voice is unclear) gives the event the state
        ``uncertain``: it is not counted, not revoked, and stays open, so a later verdict for its words (another lane's)
        settles it as ``final`` or ``revoked``; the newest revision decides.
        """
        received = set()
        for verdict in verdicts:
            if isinstance(verdict, dict) and isinstance(verdict.get('word_id'), str) and type(verdict.get('match')) is bool:
                self._verdicts[verdict['word_id']] = verdict
                self._verdicts.move_to_end(verdict['word_id'])
                received.add(verdict['word_id'])
        while len(self._verdicts) > OPEN:
            self._verdicts.popitem(last=False)
        changed = OrderedDict()
        for key in list(self._open):
            event, words, in_total = self._open[key]
            if not received & set(event['word_ids']):
                continue
            found = [self._verdicts.get(word) for word in event['word_ids']]
            if None in found:
                continue
            match = all(item['match'] for item in found)
            uncertain = not match and any(item.get('uncertain') for item in found) and \
                all(item['match'] or item.get('uncertain') for item in found)
            spoken = event['event'] == 'term.spoken'
            state = 'uncertain' if uncertain and spoken else 'revoked' if uncertain else \
                ('final' if match else 'revoked') if spoken else ('revoked' if match else 'final')
            if uncertain and spoken:
                self._open[key][2] = False
            else:
                del self._open[key]
            reason = next((item.get('reason') for item in found if item.get('reason')), None)
            revision = dict(event='event.revision', state=state, revises=key,
                            revised_event=event['event'], term=event['term'], start_ms=event['start_ms'],
                            end_ms=event['end_ms'], **words.identity,
                            **{name: found[0][name] for name in ('slot', 'slot_score', 'basis') if name in found[0]},
                            **({'by': found[0]['settled_by']} if isinstance(found[0].get('settled_by'), str) else {}))
            if state == 'revoked' or uncertain:
                revision['reason'] = reason or ('another_voice' if spoken else 'enrolled_voice')
            elif reason:        # unchanged, and the verdict says why (e.g. another voice's word)
                revision['reason'] = reason
            added = changed.setdefault(id(words), (words, []))[1]
            added.append(revision)
            now = spoken and state == 'final'
            if now != in_total:
                step = 1 if now else -1
                words.total += step
                words.counts[event['term']] = words.counts.get(event['term'], 0) + step
            if uncertain and not spoken:   # not another voice's after all, nor surely the person's: uncertain words are
                # always term.spoken in state uncertain (open: a later verdict for the same words settles it)
                doubt = dict(event='term.spoken', state='uncertain', **{
                    name: value for name, value in event.items() if name not in ('state', 'event', 'event_id')})
                doubt.update(event_id=event_id(key, 'uncertain'), replaces=key, reason=revision['reason'])
                self._open[doubt['event_id']] = [doubt, words, False]
                added.append(doubt)
            if match and not spoken:       # the enrolled voice after all: counted from now on, as final
                words.total += 1
                words.counts[event['term']] = words.counts.get(event['term'], 0) + 1
                promoted = dict(event='term.spoken', state='final', **{
                    name: value for name, value in event.items() if name not in ('word_ids', 'state', 'event', 'event_id')})
                promoted.update(event_id=event_id(key, 'final'), replaces=key,
                                **{name: found[0][name] for name in ('slot', 'slot_score') if name in found[0]})
                if 'slot_score' in found[0]:
                    promoted['speaker_score'] = found[0]['slot_score']
                added.append(promoted)
        records = [self.spec.validate(dict(count=words.total, counts=dict(words.counts), events=events, **words.identity))
                   for words, events in changed.values()]
        return None if not records else records[0] if len(records) == 1 else records
