"""Counted words in a streaming recogniser's partial hypotheses, at once (owner decision 26: count instantly, correct
later, visibly). Text math, no model; works with any streaming recogniser's ``status: "partial"`` records.

A streaming recogniser sends partial text of the open turn many times a second and revises it; its final (committed)
text of the turn comes later (for a recogniser the client commits, at the end of the turn: seconds after a word early in
a long turn). :class:`PartialSpotter` finds the counted words (:mod:`gamst.providers.term_events` spellings) in each
partial and says when one is *stable*: the ``k``-th occurrence of a term in an utterance's partial text has been there
in ``min_revisions`` partials in a row, or for ``min_ms`` of capture time (whichever comes first; ``min_revisions`` 1:
at once). Such an occurrence is *heard*: :class:`gamst.providers.speaker_verify.SpeakerVerify` checks its voice on the
audio so far and :class:`gamst.providers.term_events.TermEvents` counts it at once in state ``heard``; the final text of
the same turn later confirms it (``provisional``, same event) or withdraws it (``revoked``, ``partial_withdrawn``).
An occurrence that leaves the partial text before it is stable is a *flicker* (counted in :meth:`stats`, never
emitted); one that leaves after it was emitted is left to the final (it may come back).

Times. Where a partial carries word times (``words`` with ``start_ms``/``end_ms``), the occurrence has them. Otherwise
(ElevenLabs Scribe partials are text only; measured 2026-11-27) its end is estimated from when it was heard: the
capture time of the newest sound sent when the partial first held it (``heard_to_ms``), minus ``lag_ms`` (the
recogniser's partial delay) and ``ms_per_char`` for each normalized character after it in that partial, never before
the partial's start; its start is ``ms_per_char`` per character before that. The estimate only places the voice check
and the matching to the final; the final's word times replace it.

Trace (:meth:`trace`): for each utterance, each partial's ``heard_to_ms`` and how many occurrences of each term it held
(counts only, never text), so another debounce can be measured offline (``scripts/partial_flicker.py``).
"""
from __future__ import annotations

from collections import OrderedDict

from .term_events import _found, _part_times, _spoken, normalize, spelling_parts

DEFAULTS = dict(min_revisions=2, min_ms=None, lag_ms=400, ms_per_char=150)


class PartialSpotter:
    """``update(value)`` per partial record -> the occurrences that just became stable (each once per utterance)."""

    def __init__(self, terms, *, min_revisions=DEFAULTS['min_revisions'], min_ms=DEFAULTS['min_ms'],
                 lag_ms=DEFAULTS['lag_ms'], ms_per_char=DEFAULTS['ms_per_char'], max_utterances=64, max_trace=4000):
        if int(min_revisions) < 1:
            raise ValueError('min_revisions is at least 1')
        if min_ms is not None and float(min_ms) < 0:
            raise ValueError('min_ms is not negative')
        self.terms = dict(terms)      # {term: [spellings]} (term_events.parse_terms)
        self.min_revisions, self.min_ms = int(min_revisions), None if min_ms is None else float(min_ms)
        self.lag_ms, self.ms_per_char = float(lag_ms), float(ms_per_char)
        self.max_utterances, self.max_trace = int(max_utterances), int(max_trace)
        self._open = OrderedDict()    # utterance_id -> {(term, k): [first heard_to, streak, start_ms, end_ms, timed, ...]}
        self._emitted = OrderedDict()  # utterance_id -> {(term, k)}
        self._closed = OrderedDict()   # utterances whose final came: their late partials are ignored
        self._trace = []
        self._stats = dict(partials=0, appeared=0, flickered=0, emitted=0, left_after_emitted=0, late_partials=0)

    def settings(self):
        return dict(min_revisions=self.min_revisions, min_ms=self.min_ms, lag_ms=self.lag_ms, ms_per_char=self.ms_per_char)

    def update(self, value):
        """One partial record (``utterance_id``, ``text``, ``heard_to_ms``, ``start_ms``, optional ``words``)."""
        if not isinstance(value, dict) or value.get('status') != 'partial':
            return []
        key = str(value.get('utterance_id'))
        if key in self._closed:
            self._stats['late_partials'] += 1
            return []
        self._stats['partials'] += 1
        heard = value.get('heard_to_ms', value.get('end_ms'))
        if type(heard) not in (int, float):
            return []
        floor = value.get('start_ms') if type(value.get('start_ms')) in (int, float) else None
        text = str(value.get('text') or '')
        length = len(normalize(text))
        words = [word for word in value.get('words') or () if isinstance(word, dict)
                 and type(word.get('start_ms')) in (int, float) and type(word.get('end_ms')) in (int, float)]
        found = {}
        for term, spellings in self.terms.items():
            for index, (offset, size, spelling, part) in enumerate(sorted(_found(text, spellings))):
                times = _part_times(words, offset, size, part) if words else None
                if times is not None:
                    start, end, timed = int(times[0]), int(times[1]), True
                else:
                    end = heard - self.lag_ms - self.ms_per_char * max(0, length - offset - size)
                    start = end - self.ms_per_char * size
                    if floor is not None and start < floor:
                        start, end = floor, max(floor + self.ms_per_char * size, end)
                    start, end, timed = int(start), int(min(max(end, start + 1), heard)), False
                found[(term, index)] = dict(term=term, index=index, start_ms=start, end_ms=end, timed=timed,
                                            spoken=_spoken(text, offset, size, spelling),
                                            flags=list(spelling_parts(spelling)['flags']),
                                            part=None if part is None else list(part[:2]))
        self._remember(key, heard, found)
        held = self._open.setdefault(key, {})
        emitted = self._emitted.setdefault(key, set())
        self._open.move_to_end(key)
        while len(self._open) > self.max_utterances:
            old, _ = self._open.popitem(last=False)
            self._emitted.pop(old, None)
        for place in list(held):
            if place not in found:
                if place in emitted:
                    self._stats['left_after_emitted'] += 1
                else:
                    self._stats['flickered'] += 1
                del held[place]
        stable = []
        for place, occurrence in found.items():
            if place not in held:
                self._stats['appeared'] += place not in emitted
                held[place] = dict(occurrence, first_ms=heard, streak=0)
            item = held[place]
            item['streak'] += 1
            if place in emitted:
                continue
            if item['streak'] >= self.min_revisions or self.min_ms is not None and heard - item['first_ms'] >= self.min_ms:
                emitted.add(place)
                self._stats['emitted'] += 1
                stable.append(dict({name: item[name] for name in ('term', 'index', 'start_ms', 'end_ms', 'timed',
                                                                    'spoken', 'flags', 'part')},
                                   utterance_id=key, heard_to_ms=int(heard), first_heard_ms=int(item['first_ms']),
                                   revisions=item['streak']))
        return stable

    def final(self, utterance_id):
        """The final of ``utterance_id`` came: forget its partials (later partials of it are ignored)."""
        key = str(utterance_id)
        self._open.pop(key, None)
        self._emitted.pop(key, None)
        self._closed[key] = True
        while len(self._closed) > 4 * self.max_utterances:
            self._closed.popitem(last=False)

    def _remember(self, key, heard, found):
        if len(self._trace) < self.max_trace:
            counts = {}
            for term, _ in found:
                counts[term] = counts.get(term, 0) + 1
            self._trace.append([key, int(heard), counts])

    def stats(self):
        return dict(self._stats, **self.settings())

    def trace(self):
        """``[[utterance_id, heard_to_ms, {term: occurrences}], ...]`` of every partial (counts only, no text)."""
        return [list(item) for item in self._trace]


def reconcile(heard, final, *, max_ms=2500):
    """Pairs of heard occurrences and final ones of the same term, in time order, nearest first.

    ``heard`` / ``final``: lists of ``(end_ms, item)``; returns ``[(heard item, final item)]``: a monotone alignment
    (neither list's order is crossed) that pairs as many as possible within ``max_ms`` of each other, the nearest when
    there is a choice. Unpaired heard items were withdrawn by the final; unpaired final ones were not heard at once.
    """
    heard, final = sorted(heard, key=lambda item: item[0]), sorted(final, key=lambda item: item[0])
    rows, cols = len(heard), len(final)
    # best[i][j]: (pairs, -distance) over heard[i:], final[j:]
    best = [[(0, 0.)] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            options = [best[i + 1][j], best[i][j + 1]]
            gap = abs(heard[i][0] - final[j][0])
            if gap <= max_ms:
                pairs, distance = best[i + 1][j + 1]
                options.append((pairs + 1, distance - gap))
            best[i][j] = max(options)
    pairs, i, j = [], 0, 0
    while i < rows and j < cols:
        gap = abs(heard[i][0] - final[j][0])
        if gap <= max_ms and best[i][j] == (best[i + 1][j + 1][0] + 1, best[i + 1][j + 1][1] - gap):
            pairs.append((heard[i][1], final[j][1]))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return pairs
