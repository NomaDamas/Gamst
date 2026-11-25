"""Which retained records are current: append-only revisions and retractions.

The Store never changes a record. A correction is a new record, with the fields ``gamst.temporal.TemporalEvent``
already defines: ``supersedes`` is the identity of what it corrects (the ID of the first record) and ``revision``
counts up. A retraction (``type='observation.retraction'``) is the last revision of something that turned out
not to be there. Utterances keep their own ``utterance_id`` / ``revision`` (:mod:`gamst.transcripts`).
"""
from __future__ import annotations

import math

RETRACTION = 'observation.retraction'
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


def identity(record):
    return record.get('supersedes') or record['id']


def named(record):
    """The capability ID, or the name of an unregistered signal."""
    observation = record['observation']
    return (observation.get('capability') or {}).get('id') or observation.get('name')


def depends_on(record, value=None):
    """Record IDs a result says it was derived from, as the record has them."""
    found = [item.get('record_id') for item in record.get('evidence') or () if isinstance(item, dict)]
    value = (record.get('observation') or {}).get('value') if value is None else value
    if isinstance(value, dict):
        for name in ('evidence_ids', 'evidence_id', 'record_id', 'record_ids', 'depends_on'):
            item = value.get(name)
            found += item if isinstance(item, list) else [item]
    return list(dict.fromkeys(item for item in found if isinstance(item, str)))


def revision_fields(store, record, supersedes):
    """``supersedes`` / ``revision`` of a record that corrects the retained record ``supersedes``."""
    target = next(iter(store.get_many([supersedes])), None) if isinstance(supersedes, str) else None
    if target is None or target['subject_id'] != record['subject_id'] or target['type'] != 'observation.signal':
        raise ValueError('supersedes must name a retained signal of this subject')
    if record['type'] == 'observation.signal' and named(target) != named(record):
        raise ValueError('a revision must be of the same capability or signal')
    root = identity(target)
    newest = max((row[2] for row in store.revisions(record['subject_id'], of=root)), default=0)
    return dict(supersedes=root, revision=newest + 1)
