"""Count events across retained records: one query for any module's events.

A count is context, not a model output: it is computed from committed records each time, so it is
the same for an agent over MCP, an application over the SDK, and a replay of the same input.
Events are the ``events`` arrays inside observation values; an event's name is its ``event`` or
``type`` field (both conventions exist in providers); ``group_by='@event'`` groups by that name.

Revision states (any capability). An event may be committed at once as ``state: "provisional"`` with a stable
``event_id``; a later record of the same capability marks it with an ``event.revision`` event::

    {"event": "event.revision", "revises": "<event_id>", "state": "final" | "revoked" | "uncertain", "reason": "...",
     "start_ms": ..., "end_ms": ...}      # the revised event's own times, so a window that holds it holds this too

An event without ``state`` is final as committed. The newest revision of an ``event_id`` decides its state. A count
is of provisional and final events; revoked and uncertain ones are listed in ``states`` but not counted, unless ``where``
asks for a state (``{"state": "revoked"}``). ``uncertain`` is neither said nor not said (overlapped speech whose voice
is unclear): it is shown as such, never as revoked, and a later revision may settle it. Revisions are no events of their own (``event="event.revision"`` counts
them). Nothing is changed in the record that holds the revised event.
"""
from __future__ import annotations

import csv
import json

from .capabilities import bounded_json
from .revisions import depends_on

MAX_SCAN = 20000
GAP_CAPABILITY = 'pipeline.gaps'   # records of gamst.pipeline: captured frames that never reached a module
INSTRUCTION_CAPABILITY = 'pipeline.instructions'    # records of gamst.pipeline: other words or another movement
IDENTITY = ('reference_id', 'reference_revision')
REVISION = 'event.revision'        # marks an earlier event (``revises``: its ``event_id``) final or revoked
STATES = ('provisional', 'final', 'revoked', 'uncertain')
COUNTED = ('provisional', 'final')    # what a count holds unless ``where`` asks for a state


def event_name(item):
    return item.get('event', item.get('type'))


def revisions(rows):
    """``{event_id: (revision event, record)}``: the newest revision of each event in ``rows`` (oldest first)."""
    found = {}
    for record in rows:
        value = (record.get('observation') or {}).get('value')
        for item in (value.get('events') if isinstance(value, dict) else None) or ():
            if isinstance(item, dict) and event_name(item) == REVISION and isinstance(item.get('revises'), str) \
                    and item.get('state') in STATES[1:]:
                found[item['revises']] = item, record
    return found


def state_of(item, revised):
    """(state, (revision event, its record) or None) of one event under ``revised`` (:func:`revisions`)."""
    found = revised.get(item.get('event_id')) if isinstance(item.get('event_id'), str) else None
    if found is not None:
        return found[0]['state'], found
    state = item.get('state')
    return (state if state in STATES else 'final'), None


def with_state(item, revised):
    """The event as it stands now: its resolved ``state``, and what revised it (``revised_by``)."""
    if not isinstance(item.get('event_id'), str) and 'state' not in item:
        return item
    state, found = state_of(item, revised)
    if found is None:
        return item if item.get('state') == state else dict(item, state=state)
    revision, record = found
    return dict(item, state=state, revised_by=dict(
        record_id=record.get('id'), timestamp_ms=record.get('timestamp_ms'),
        **{key: revision[key] for key in ('reason', 'basis') if key in revision}))


def gaps_of(records, capability=None):
    """The gap records whose loss affects ``capability`` (None: all of them)."""
    return [record for record in records if capability in (None, GAP_CAPABILITY) or any(
        capability in event.get('capabilities', ()) for event in record['observation']['value']['events'])]


def lost(records):
    """``{gaps, frames, longest_ms}`` of gap records; one record may hold several gaps."""
    events = [event for record in records for event in record['observation']['value']['events']]
    return dict(gaps=sum(event['gaps'] for event in events), frames=sum(event['lost_frames'] for event in events),
                longest_ms=max((event['end_ms'] - event['start_ms'] for event in events), default=0))


def _lookup(value, path):
    for part in path.split('.'):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return False, None
    return True, value


def current_instructions(changes):
    """``{reference: ((id, revision), from_ms)}`` after ``pipeline.instructions`` events (oldest first); a removed one stays
    the last."""
    found = {}
    for change in changes:
        if change.get('event') == 'instruction.removed':
            found.setdefault(change['reference'], ((change.get('previous_id'), change.get('previous_revision')), 0))
        else:
            found[change['reference']] = (change.get('reference_id'), change.get('reference_revision')), change.get('start_ms', 0)
    return found


def _about(item):
    return item.get('reference') if isinstance(item.get('reference'), str) else None, tuple(map(item.get, IDENTITY))


def newest_instructions(rows, current=None):
    """``{reference: (id, revision)}`` that counts now: the last change (``current``), or that of an event after it."""
    seen = {name: (from_ms, which) for name, (which, from_ms) in (current or {}).items()}
    for record in rows:
        value = (record.get('observation') or {}).get('value')
        for item in (value.get('events') if isinstance(value, dict) else None) or ():
            if isinstance(item, dict) and any(key in item for key in IDENTITY):
                name, which = _about(item)
                at = item.get('start_ms')
                at = at if type(at) in (int, float) else record.get('timestamp_ms', 0)
                if name not in seen or at >= seen[name][0]:
                    seen[name] = at, which
    return {name: which for name, (_, which) in seen.items()}


def is_current(item, newest):
    """False for an event of an instruction that another one replaced."""
    name, which = _about(item)
    return not any(key in item for key in IDENTITY) or newest.get(name, which) == which


def count_events(rows, *, event=None, where=None, group_by=None, max_evidence=20, current=None):
    """``rows`` are current stored records (oldest first). Returns the count, groups and bounded evidence.

    Events of different revisions of one instruction (reference) are never counted together: the newest
    (``current``: :func:`current_instructions`, or that of a newer event) is counted, the others are listed in
    ``other_instructions``. ``where`` with ``reference_id`` or ``reference_revision`` chooses.

    Revision states (module docstring): revoked and uncertain events are not counted unless ``where`` names a ``state``;
    ``states`` (and with ``group_by``, ``group_states``) say how many of each there are, once any event has one.
    """
    where = where or {}
    count, groups, evidence = 0, {}, []
    others, of = {}, set()
    newest = {} if {*IDENTITY} & set(where) else newest_instructions(rows, current)
    revised = {} if event == REVISION else revisions(rows)
    states, group_states, stated = dict.fromkeys(STATES, 0), {}, bool(revised)
    for record in rows:
        observation = record.get('observation') or {}
        value = observation.get('value')
        events = value.get('events') if isinstance(value, dict) else None
        if not isinstance(events, list):
            continue
        for item in events:
            if not isinstance(item, dict):
                continue
            name = item.get('event', item.get('type'))
            if event is not None and name != event or event is None and name == REVISION:
                continue
            if event != REVISION:
                item = with_state(item, revised)
                stated = stated or 'state' in item
                if 'state' in where and 'state' not in item:
                    item = dict(item, state='final')     # as committed
            if any(_lookup(item, path) != (True, expected) for path, expected in where.items()):
                continue
            state = item.get('state', 'final') if event != REVISION else None
            if state is not None:
                states[state] += 1
                if group_by == '@event':
                    group_states.setdefault(name, dict.fromkeys(STATES, 0))[state] += 1
                if state not in COUNTED and 'state' not in where:
                    continue
            if any(key in item for key in IDENTITY):
                reference, which = _about(item)
                if not is_current(item, newest):
                    others[(reference, *which)] = others.get((reference, *which), 0) + 1
                    continue
                of.add((reference, *which))
            count += 1
            if group_by == '@event':
                groups[name] = groups.get(name, 0) + 1
            elif group_by is not None:
                found, key = _lookup(item, group_by)
                key = str(key) if found else None
                groups[key] = groups.get(key, 0) + 1
            if len(evidence) < max_evidence:
                evidence.append(dict(record_id=record.get('id'), timestamp_ms=record.get('timestamp_ms'),
                                     capability_id=(observation.get('capability') or {}).get('id'),
                                     event=item,
                                     **({'depends_on': sources[:8]} if (sources := depends_on(record, item)) else {})))
    result = dict(count=count, evidence=evidence, evidence_truncated=count > len(evidence))
    if group_by is not None:
        result['groups'] = groups
    if stated:      # provisional + final = count; revoked and uncertain ones are not in it
        for kept in (states, *group_states.values()):    # uncertain is said once there is one
            if not kept['uncertain']:
                del kept['uncertain']
        result['states'] = states
        if group_states:
            result['group_states'] = group_states
    for name, found in (('instructions', dict.fromkeys(sorted(of, key=str))), ('other_instructions', others)):
        if found:
            result[name] = [dict(zip(('reference', *IDENTITY), key), **({} if number is None else dict(count=number)))
                            for key, number in list(found.items())[:32]]
    return result


# ---------------------------------------------------------------- events as data (JSONL / CSV)

COLUMNS = ('capability', 'event', 'label', 'method', 'confirmed', 'reference', 'reference_id', 'reference_revision', 'index', 'example',
           'start_ms', 'end_ms', 'duration_ms', 'gap_frames', 'distance', 'best_distance', 'person', 'timestamp_ms', 'subject_id',
           'source_id', 'sensor_id', 'record_id')


def event_rows(records, *, capability=None, event=None):
    """One row per event of stored records (oldest first), as plain data.

    An event's own fields follow the record's capability, capture time and IDs. A match and its end are one
    row: ``exemplar.ended`` gives its ``exemplar.matched`` row (same capability, reference and start) the final
    ``end_ms`` and ``best_distance``. Events with raw or private fields are left out, never renamed. An event with a
    revision state has its current one (``state``, ``revised_by``); the revisions are rows too.
    """
    rows, open_matches, revised = [], {}, revisions(records)
    for record in records:
        observation = record.get('observation') or {}
        value = observation.get('value')
        events = value.get('events') if isinstance(value, dict) else None
        name = (observation.get('capability') or {}).get('id') or observation.get('name')
        if not isinstance(events, list) or capability not in (None, name):
            continue
        sensor = observation.get('sensor') or {}
        for item in events:
            if not isinstance(item, dict):
                continue
            try:
                item = bounded_json(item, forbid_raw=True)
            except ValueError:
                continue
            kind = item.get('event', item.get('type'))
            if kind != REVISION:
                item = with_state(item, revised)
            family, _, phase = str(kind).rpartition('.')
            key = (record.get('subject_id'), name, family, item.get('reference'), item.get('reference_id'),
                   item.get('reference_revision'), item.get('start_ms'))
            if phase == 'ended' and key in open_matches:
                row = open_matches.pop(key)
                for field in ('end_ms', 'best_distance', 'cut', 'gap_frames'):
                    if field in item:
                        row[field] = item[field]
                row['duration_ms'] = row['end_ms'] - row['start_ms']
                continue
            row = dict(item, event=kind, capability=name, record_id=record.get('id'),
                       timestamp_ms=record.get('timestamp_ms'), subject_id=record.get('subject_id'),
                       source_id=sensor.get('source_id'), sensor_id=sensor.get('sensor_id'))
            if type(row.get('start_ms')) is int and type(row.get('end_ms')) is int:
                row['duration_ms'] = row['end_ms'] - row['start_ms']
                if phase == 'matched':
                    open_matches[key] = row
            rows.append(row)
    return [row for row in rows if event in (None, row['event'])]


def _cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    # A spreadsheet runs a text cell that starts like a formula.
    return "'" + value if isinstance(value, str) and value[:1] in ('=', '+', '-', '@') else value


def write_rows(rows, out, format='jsonl'):
    """``rows`` to an open text file. CSV columns: the known ones, then any other field in order of appearance."""
    if format == 'jsonl':
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
        return
    if format != 'csv':
        raise ValueError("format must be 'jsonl' or 'csv'")
    extra = list(dict.fromkeys(key for row in rows for key in row if key not in COLUMNS))
    writer = csv.DictWriter(out, [*COLUMNS, *extra], restval='')
    writer.writeheader()
    writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
