"""Causal state chains: each state links the state before it, says what changed, and names the records it is attributed to.

Owner decision 47: a semantic state is never an independent snapshot. Each state of a chain carries ``chain``::

    {"state_id": "ss:…", "previous_state_id": "ss:…" (absent at the start of a chain),
     "changed": {"set": {path: value}, "changed": {path: {"from", "to"}}, "removed": [path], "truncated": bool},
     "caused_by": [{"record_id", "capability", "event", "start_ms", "end_ms", ...}],
     "since_ms": end of the previous state (or the start of this one), "attribution": "interval" | "none"}

``caused_by`` holds the records of the same or earlier time (movement events, words, person bindings, utterances) whose
end falls in the interval since the previous state, when the state changed. It is co-occurrence in that interval
("interval"), not a proven cause; nothing is attributed to a state that did not change ("none"). Computation only: no
model call, no media. Used by the semantic VLM lane (``vision.streaming_semantic`` records) and by the engine's
short-term chain (``push_semantic_state``); :func:`walk` reads a chain backwards, bounded.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

MAX_CHANGES, MAX_DEPTH, MAX_CAUSES, MAX_STEPS = 32, 3, 16, 32
CAUSE_KEYS = ('record_id', 'capability', 'event', 'start_ms', 'end_ms', 'label', 'term', 'who', 'track_id', 'text', 'state')


def state_id(*parts):
    """A short stable identity of one state: its model, its interval and its content."""
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()
    return 'ss:' + hashlib.sha256(encoded).hexdigest()[:16]


def diff(before, after):
    """What changed from ``before`` to ``after`` (state objects; ``before`` None: everything is set), by dotted path.

    Objects are compared field by field down to :data:`MAX_DEPTH` levels; other values are compared whole. At most
    :data:`MAX_CHANGES` entries; ``truncated`` says when there were more."""
    result, count = dict(set={}, changed={}, removed=[], truncated=False), [0]

    def add(kind, path, value):
        if count[0] >= MAX_CHANGES:
            result['truncated'] = True
            return
        count[0] += 1
        if kind == 'removed':
            result['removed'].append(path)
        else:
            result[kind][path] = deepcopy(value)

    def walk(old, new, prefix, depth):
        for key in sorted(set(old) | set(new), key=str):
            path = f'{prefix}{key}'
            if key not in new:
                add('removed', path, None)
            elif key not in old:
                add('set', path, new[key])
            elif isinstance(old[key], dict) and isinstance(new[key], dict) and depth < MAX_DEPTH:
                walk(old[key], new[key], path + '.', depth + 1)
            elif old[key] != new[key]:
                add('changed', path, {'from': old[key], 'to': new[key]})

    walk(before if isinstance(before, dict) else {}, after if isinstance(after, dict) else {}, '', 1)
    return result


def is_change(changed):
    return bool(changed and (changed.get('set') or changed.get('changed') or changed.get('removed')))


def attributed(candidates, since_ms, until_ms, *, start_ms=None, limit=MAX_CAUSES):
    """The candidates (``{record_id, start_ms?, end_ms, ...}``) that end in (``since_ms``, ``until_ms``]; at the start of a
    chain (``since_ms`` None) those that end in [``start_ms``, ``until_ms``]. Newest ``limit`` kept, oldest first."""
    found = []
    for item in candidates or ():
        if not isinstance(item, dict) or not isinstance(item.get('record_id'), str):
            continue
        end = item.get('end_ms', item.get('start_ms'))
        if type(end) is not int or end > until_ms:
            continue
        if since_ms is not None and end <= since_ms or since_ms is None and start_ms is not None and end < start_ms:
            continue
        found.append({key: item[key] for key in CAUSE_KEYS if item.get(key) is not None})
    found.sort(key=lambda item: (item.get('end_ms', item.get('start_ms', 0)), item['record_id']))
    return found[-limit:]


def link(previous, state, *, identity, start_ms, end_ms, candidates=()):
    """The ``chain`` of a new state. ``previous``: ``{state_id, state, end_ms}`` of the state before it, or None (a new
    chain: the first state, or after a reset or a gap). ``identity``: what makes the state id unique besides its
    content (model, interval). ``candidates``: records that may be attributed (:func:`attributed`)."""
    before = previous.get('state') if previous else None
    changed = diff(before, state)
    since = previous.get('end_ms') if previous else None
    causes = attributed(candidates, since, end_ms, start_ms=start_ms) if is_change(changed) else []
    chain = dict(state_id=state_id(identity, start_ms, end_ms, state, previous and previous.get('state_id')),
                 changed=changed, caused_by=causes, since_ms=since if since is not None else start_ms,
                 attribution='interval' if causes else 'none')
    if previous and previous.get('state_id'):
        chain['previous_state_id'] = previous['state_id']
    return chain


def walk(states, *, state_id=None, at_ms=None, steps=8, max_bytes=65536):
    """A chain read backwards from one state. ``states``: dicts with ``chain`` and ``end_ms`` (and whatever else they say:
    ``state``, ``summary``, ``record_id``). Starts at ``state_id``, else the newest state that ended at or before
    ``at_ms``, else the newest. Returns ``{states: [newest first], stopped: start | steps | not_retained | bytes,
    truncated}``; at most ``steps`` (<= :data:`MAX_STEPS`) states and about ``max_bytes``."""
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f'steps must be 1..{MAX_STEPS}')
    by_id = {}
    for item in states:
        chain = item.get('chain') if isinstance(item, dict) else None
        if isinstance(chain, dict) and isinstance(chain.get('state_id'), str):
            by_id.setdefault(chain['state_id'], item)
    if state_id is not None:
        current = by_id.get(state_id)
        if current is None:
            raise ValueError(f'no retained state {state_id}')
    else:
        ordered = sorted(by_id.values(), key=lambda item: item.get('end_ms') or 0)
        if at_ms is not None:
            ordered = [item for item in ordered if (item.get('end_ms') or 0) <= at_ms]
        current = ordered[-1] if ordered else None
    result, size, stopped = [], 2, 'start'
    while current is not None:
        encoded = len(json.dumps(current, ensure_ascii=False, default=str).encode())
        if result and size + encoded > max_bytes:
            stopped = 'bytes'
            break
        result.append(current)
        size += encoded + 1
        previous = current['chain'].get('previous_state_id')
        if previous is None:
            stopped = 'start'
            break
        if len(result) >= steps:
            stopped = 'steps'
            break
        current = by_id.get(previous)
        if current is None:
            stopped = 'not_retained'
    return dict(states=deepcopy(result), count=len(result), stopped=stopped, truncated=stopped in ('steps', 'bytes'))
