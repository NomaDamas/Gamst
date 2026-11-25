"""Count events across retained records: one query for any module's events.

A count is context, not a model output: it is computed from committed records each time, so it is
the same for an agent over MCP, an application over the SDK, and a replay of the same input.
Events are the ``events`` arrays inside observation values; an event's name is its ``event`` or
``type`` field (both conventions exist in providers).
"""
from __future__ import annotations

MAX_SCAN = 20000


def _lookup(value, path):
    for part in path.split('.'):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return False, None
    return True, value


def count_events(rows, *, event=None, where=None, group_by=None, max_evidence=20):
    """``rows`` are stored records (oldest first). Returns the count, groups and bounded evidence."""
    where = where or {}
    count, groups, evidence = 0, {}, []
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
            if event is not None and name != event:
                continue
            if any(_lookup(item, path) != (True, expected) for path, expected in where.items()):
                continue
            count += 1
            if group_by is not None:
                found, key = _lookup(item, group_by)
                key = str(key) if found else None
                groups[key] = groups.get(key, 0) + 1
            if len(evidence) < max_evidence:
                evidence.append(dict(record_id=record.get('id'), timestamp_ms=record.get('timestamp_ms'),
                                     capability_id=(observation.get('capability') or {}).get('id'),
                                     event=item))
    result = dict(count=count, evidence=evidence, evidence_truncated=count > len(evidence))
    if group_by is not None:
        result['groups'] = groups
    return result
