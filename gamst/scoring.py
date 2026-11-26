"""Score a recipe run against a person's ground truth (``gamst recipe run ... --truth truth.json``).

Truth file (times are seconds from the first frame of the input; an event that has an interval is compared
by its start)::

    {"capability": "temporal.exemplar_events", "event": "exemplar.matched",
     "count": 7,                                  # what a person counted
     "events": [{"t_s": 1.2}, {"t_s": 3.4, "value": 3}, ...],   # optional: when, and a value
     "value_key": "flips",                        # optional: event field compared with "value"
     "tolerance_s": 0.5,
     "at": "start"}                               # optional: compare by "start" (default) or "end"

The score reports only what was measured: the count error, and, when times are given, matched
events (greedy by time within the tolerance), precision, recall, mean time error, and per-event
value differences.
"""
from __future__ import annotations


def _event_time_ms(event, at='start'):
    for key in ('timestamp_ms',) + (('end_ms', 'start_ms') if at == 'end' else ('start_ms', 'end_ms')):
        if isinstance(event.get(key), (int, float)):
            return event[key]
    return None


def score(truth, events, *, start_ms):
    """``events``: the run's events of ``truth['capability']`` (dicts with times in ms)."""
    name = truth.get('event')
    found = [event for event in events if name is None or event.get('event', event.get('type')) == name]
    result = dict(capability=truth.get('capability'), event=name, counted=len(found))
    if 'count' in truth:
        result.update(truth_count=truth['count'], count_error=len(found) - truth['count'],
                      within_one=abs(len(found) - truth['count']) <= 1)
    truth_events = truth.get('events') or []
    if not truth_events:
        return result
    tolerance = float(truth.get('tolerance_s', .5)) * 1000
    key = truth.get('value_key')
    at = truth.get('at', 'start')
    timed = sorted(((_event_time_ms(event, at) - start_ms, event) for event in found
                    if _event_time_ms(event, at) is not None), key=lambda item: item[0])
    used, pairs = set(), []
    for expected in sorted(truth_events, key=lambda item: item['t_s']):
        target = expected['t_s'] * 1000
        best = None
        for index, (offset, event) in enumerate(timed):
            if index in used or abs(offset - target) > tolerance:
                continue
            if best is None or abs(offset - target) < abs(timed[best][0] - target):
                best = index
        if best is None:
            pairs.append(dict(t_s=expected['t_s'], matched=False, truth_value=expected.get('value')))
            continue
        used.add(best)
        offset, event = timed[best]
        pair = dict(t_s=expected['t_s'], matched=True, time_error_ms=round(offset - target))
        if key is not None and 'value' in expected:
            pair.update(truth_value=expected['value'], value=event.get(key))
            if isinstance(event.get(key), (int, float)):
                pair['value_error'] = round(event[key] - expected['value'], 4)
        pairs.append(pair)
    matched = [pair for pair in pairs if pair['matched']]
    result.update(
        events=pairs, matched=len(matched),
        precision=round(len(matched) / len(timed), 4) if timed else None,
        recall=round(len(matched) / len(truth_events), 4),
        mean_abs_time_error_ms=round(sum(abs(p['time_error_ms']) for p in matched) / len(matched))
        if matched else None)
    errors = [pair['value_error'] for pair in matched if 'value_error' in pair]
    if errors:
        result.update(mean_abs_value_error=round(sum(abs(e) for e in errors) / len(errors), 4),
                      values_within_one=sum(abs(e) <= 1 for e in errors))
    return result
