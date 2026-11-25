"""Which retained records are current: append-only revisions, retractions and quality levels.

The Store never changes a record. A correction is a new record, with the fields ``gamst.temporal.TemporalEvent``
already defines: ``supersedes`` is the identity of what it corrects (the ID of the first record) and ``revision``
counts up. A retraction (``type='observation.retraction'``) is the last revision of something that turned out
not to be there. Utterances keep their own ``utterance_id`` / ``revision`` (:mod:`gamst.transcripts`).

``quality.level`` is ``provisional`` (reduced processing) or, when absent, final. Final evidence replaces
provisional evidence about the same capability, source, sensor and capture interval, whichever was learned
first; :meth:`gamst.Gamst.push_signal` also links it as a revision.
"""
from __future__ import annotations

import math

RETRACTION = 'observation.retraction'
LEVELS = ('provisional', 'final')
LATE_MS = 300_000       # how long after its interval an answer may be committed (transcripts allow five minutes)
SPAN_NAMES = (('start_ms', 'end_ms'), ('capture_start_ms', 'capture_end_ms'), ('window_start_ms', 'window_end_ms'))


def time_ms(value):
    return int(value) if type(value) in (int, float) and math.isfinite(value) else None


def span_of(value):
    """``(start_ms, end_ms)`` named by an object, or None."""
    if not isinstance(value, dict):
        return None
    for first, last in SPAN_NAMES:
        if first not in value:
            continue
        start, end = time_ms(value.get(first)), time_ms(value.get(last))
        if start is not None and end is not None and start <= end:
            return start, end
    return None


def record_span(record):
    """The interval a record is about (an utterance, a match, a window); None: it is about its capture time."""
    observation = record['observation']
    value = observation.get('value')
    if isinstance(value, dict):
        only = next(iter(value.values())) if len(value) == 1 else None
        return span_of(value) or span_of(value.get('window')) or span_of(only) or span_of(observation)
    return span_of(observation)


def interval(record):
    return record_span(record) or (record['timestamp_ms'], record['timestamp_ms'])


def level(record):
    return 'provisional' if (record.get('quality') or {}).get('level') == 'provisional' else 'final'


def identity(record):
    return record.get('supersedes') or record['id']


def series(record):
    """What a record is one answer of: the key ``get_context`` uses, with the subject."""
    observation = record['observation']
    sensor = observation.get('sensor') or {}
    return (record.get('subject_id'), (observation.get('capability') or {}).get('id') or observation.get('name'),
            sensor.get('source_id'), sensor.get('sensor_id'), sensor.get('clock_id'))


def depends_on(record, value=None):
    """Record IDs a result says it was derived from, as the record has them."""
    found = [item.get('record_id') for item in record.get('evidence') or () if isinstance(item, dict)]
    value = (record.get('observation') or {}).get('value') if value is None else value
    if isinstance(value, dict):
        for name in ('evidence_ids', 'evidence_id', 'record_id', 'record_ids', 'depends_on'):
            item = value.get(name)
            found += item if isinstance(item, list) else [item]
    return list(dict.fromkeys(item for item in found if isinstance(item, str)))


def current(records, superseded):
    """``records`` (stored order) without what a revision, a retraction or a final replaced."""
    kept = [record for record in records if record['id'] not in superseded]
    if any(level(record) == 'provisional' for record in kept):
        final = {(series(record), interval(record)) for record in kept if level(record) == 'final'}
        kept = [record for record in kept
                if level(record) == 'final' or (series(record), interval(record)) not in final]
    return kept


def _event_key(record, item):
    span = span_of(item) if isinstance(item, dict) else None
    return span and (series(record)[:2], item.get('event', item.get('type')), item.get('term'),
                     item.get('reference'), span)


def replaced_events(records):
    """``{(record ID, index)}`` of provisional events a final event of the same kind and interval replaced."""
    provisional, final = {}, set()
    for record in records:
        value = record['observation'].get('value')
        items = value.get('events') if isinstance(value, dict) else None
        for index, item in enumerate(items if isinstance(items, list) else ()):
            key = _event_key(record, item)
            if key is None:
                continue
            if level(record) == 'final':
                final.add(key)
            else:
                provisional[record['id'], index] = key
    return {place for place, key in provisional.items() if key in final}


def revision_fields(store, record, supersedes=None):
    """``supersedes`` / ``revision`` of a record about to be stored; ``{}`` for a first answer.

    ``supersedes``: the ID of the retained record it corrects. Without it, a final record revises the newest
    provisional one of the same series and capture interval.
    """
    if supersedes is not None:
        target = next(iter(store.get_many([supersedes])), None) if isinstance(supersedes, str) else None
        if target is None or target['subject_id'] != record['subject_id'] or target['type'] != 'observation.signal':
            raise ValueError('supersedes must name a retained signal of this subject')
        if record['type'] == 'observation.signal' and series(target)[1] != series(record)[1]:
            raise ValueError('a revision must be of the same capability or signal')
        root = identity(target)
    else:
        if level(record) != 'final':
            return {}
        wanted, (start, end), span = series(record), interval(record), record_span(record)
        # An interval answer may be committed after its interval; an instant is about its capture time.
        found = store.provisional(record['subject_id'], record['observation'].get('name'), start,
                                  end if span is None else max(end, record['timestamp_ms']) + LATE_MS)
        root = next((identity(item) for item in found if series(item) == wanted and interval(item) == (start, end)
                     and not any(row[2] > item.get('revision', 0)
                                 for row in store.revisions(record['subject_id'], of=identity(item)))), None)
        if root is None:
            return {}
    newest = max((row[2] for row in store.revisions(record['subject_id'], of=root)), default=0)
    return dict(supersedes=root, revision=newest + 1)
