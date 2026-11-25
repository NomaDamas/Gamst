"""Count events across retained records: one query for any module's events.

A count is context, not a model output: it is computed from committed records each time, so it is
the same for an agent over MCP, an application over the SDK, and a replay of the same input.
Events are the ``events`` arrays inside observation values; an event's name is its ``event`` or
``type`` field (both conventions exist in providers); ``group_by='@event'`` groups by that name.
"""
from __future__ import annotations

import csv
import json

from .capabilities import bounded_json
from .revisions import depends_on, level, replaced_events

MAX_SCAN = 20000


def _lookup(value, path):
    for part in path.split('.'):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return False, None
    return True, value


def count_events(rows, *, event=None, where=None, group_by=None, max_evidence=20):
    """``rows`` are current stored records (oldest first). Returns the count, groups and bounded evidence."""
    where = where or {}
    count, groups, evidence = 0, {}, []
    levels, replaced = dict(final=0, provisional=0), replaced_events(rows)
    for record in rows:
        observation = record.get('observation') or {}
        value = observation.get('value')
        events = value.get('events') if isinstance(value, dict) else None
        if not isinstance(events, list):
            continue
        for index, item in enumerate(events):
            if not isinstance(item, dict) or (record.get('id'), index) in replaced:
                continue
            name = item.get('event', item.get('type'))
            if event is not None and name != event:
                continue
            if any(_lookup(item, path) != (True, expected) for path, expected in where.items()):
                continue
            count += 1
            levels[level(record)] += 1
            if group_by == '@event':
                groups[name] = groups.get(name, 0) + 1
            elif group_by is not None:
                found, key = _lookup(item, group_by)
                key = str(key) if found else None
                groups[key] = groups.get(key, 0) + 1
            if len(evidence) < max_evidence:
                evidence.append(dict(record_id=record.get('id'), timestamp_ms=record.get('timestamp_ms'),
                                     capability_id=(observation.get('capability') or {}).get('id'),
                                     event=item, level=level(record),
                                     **({'depends_on': sources[:8]} if (sources := depends_on(record, item)) else {})))
    result = dict(count=count, **levels, evidence=evidence, evidence_truncated=count > len(evidence))
    if group_by is not None:
        result['groups'] = groups
    return result


# ---------------------------------------------------------------- events as data (JSONL / CSV)

COLUMNS = ('capability', 'event', 'label', 'method', 'confirmed', 'reference', 'reference_id', 'index', 'example',
           'start_ms', 'end_ms', 'duration_ms', 'distance', 'best_distance', 'person', 'timestamp_ms', 'subject_id',
           'source_id', 'sensor_id', 'record_id')


def event_rows(records, *, capability=None, event=None):
    """One row per event of stored records (oldest first), as plain data.

    An event's own fields follow the record's capability, capture time and IDs. A match and its end are one
    row: ``exemplar.ended`` gives its ``exemplar.matched`` row (same capability, reference and start) the final
    ``end_ms`` and ``best_distance``. Events with raw or private fields are left out, never renamed.
    """
    rows, open_matches = [], {}
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
            family, _, phase = str(kind).rpartition('.')
            key = (record.get('subject_id'), name, family, item.get('reference'), item.get('reference_id'),
                   item.get('start_ms'))
            if phase == 'ended' and key in open_matches:
                row = open_matches.pop(key)
                for field in ('end_ms', 'best_distance', 'cut'):
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
