"""Evaluate paired perception/VLM traces against a frozen, caller-declared plan.

This computes evidence; it never runs a model or certifies trace authenticity.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random


def _number(value, label, minimum=0, maximum=1e18):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'invalid {label}')
    return value


def _text(value, label):
    if not isinstance(value, str) or not 1 <= len(value) <= 4096:
        raise ValueError(f'invalid {label}')
    return value


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('expected lowercase SHA256')
    return value


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)]


def compare(plan, rows, *, expected_plan_sha256):
    """All rows must match the same predeclared plan and complete paired tasks."""
    digest = plan_digest(plan)
    if digest != _digest(expected_plan_sha256):
        raise ValueError('plan differs from frozen digest')
    if plan.get('schema_version') != 1 or type(plan.get('schema_version')) is not int:
        raise ValueError('unsupported comparison schema')
    evidence = plan.get('evidence')
    if evidence not in ('synthetic', 'measured'):
        raise ValueError('declare synthetic or measured evidence')
    if type(plan.get('held_out')) is not bool:
        raise ValueError('declare held_out')
    for field in ('evaluation_id', 'scoring_protocol', 'latency_protocol', 'trace_provenance'):
        _text(plan.get(field), field)
    systems = plan.get('systems', {})
    if set(systems) != {'baseline', 'candidate'}:
        raise ValueError('declare baseline and candidate configuration')
    for config in systems.values():
        for field in ('model', 'revision', 'precision', 'hardware', 'policy', 'warm_state', 'sampling', 'window_lookahead'):
            _text(config.get(field), field)
    minimum = plan.get('min_pairs')
    if type(minimum) is not int or not 20 <= minimum <= 1000:
        raise ValueError('min_pairs must be 20..1000')
    repetitions = plan.get('bootstrap_repetitions')
    if type(repetitions) is not int or not 500 <= repetitions <= 5000:
        raise ValueError('bootstrap_repetitions must be 500..5000')
    seed = plan.get('bootstrap_seed')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('declare bootstrap_seed')
    tasks = plan.get('tasks')
    if type(tasks) is not list or not 1 <= len(tasks) <= 1000:
        raise ValueError('declare 1..1000 held-out tasks')
    task_map = {}
    for task in tasks:
        name = _text(task.get('id'), 'task ID')
        if name in task_map:
            raise ValueError('duplicate task ID')
        _digest(task.get('input_sha256'))
        facts = task.get('critical_facts')
        if type(facts) is not list or len(facts) > 128:
            raise ValueError('declare bounded critical fact IDs')
        for fact in facts:
            _text(fact, 'critical fact')
        if len(set(facts)) != len(facts):
            raise ValueError('duplicate critical fact')
        task_map[name] = task
    cost = plan.get('cost', {})
    mode = cost.get('mode')
    charges = ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')
    if mode == 'priced':
        _text(cost.get('currency'), 'currency')
        rates = cost.get('per_million', {})
        if set(rates) != set(charges):
            raise ValueError('declare all four exclusive token-bucket prices')
        for value in rates.values():
            _number(value, 'price', maximum=1e9)
    elif mode == 'resource':
        _text(cost.get('unit'), 'resource unit')
        _text(cost.get('definition'), 'measured total resource-cost definition')
    elif mode != 'unknown':
        raise ValueError('cost mode must be priced, resource or unknown')
    if type(rows) is not list or len(rows) != 2*len(tasks):
        raise ValueError('exactly one baseline and candidate row per task required')
    paired = {}
    for row in rows:
        task_id, system = row.get('task_id'), row.get('system')
        if task_id not in task_map or system not in systems or (task_id, system) in paired:
            raise ValueError('unknown or duplicate task/system')
        if row.get('plan_sha256') != digest or row.get('input_sha256') != task_map[task_id]['input_sha256']:
            raise ValueError('row does not match frozen plan/input')
        _number(row.get('latency_ms'), 'end-to-end task-answer-ready latency', minimum=1e-9)
        _number(row.get('quality_score'), 'quality score', maximum=1)
        for key in ('peak_memory_bytes', 'dropped_frames'):
            if type(row.get(key)) is not int or not 0 <= row[key] <= 10**15:
                raise ValueError(f'invalid {key}')
        usage = row.get('usage', {})
        if set(usage) != set(charges) | {'calls'}:
            raise ValueError('all usage buckets and calls are required')
        for value in usage.values():
            if type(value) is not int or not 0 <= value <= 10**12:
                raise ValueError('usage must be bounded nonnegative integers')
        facts = row.get('correct_critical_facts')
        if type(facts) is not list or len(facts) > 128 or any(not isinstance(f, str) for f in facts):
            raise ValueError('correct_critical_facts must be bounded fact IDs')
        if len(set(facts)) != len(facts) or set(facts)-set(task_map[task_id]['critical_facts']):
            raise ValueError('unknown/duplicate scored critical fact')
        if mode == 'resource':
            _number(row.get('resource_cost'), 'measured total resource cost')
        paired[task_id, system] = row
    pairs = [(paired[t['id'], 'baseline'], paired[t['id'], 'candidate']) for t in tasks]

    def row_cost(row):
        if mode == 'priced':
            return sum(Decimal(row['usage'][key])*Decimal(str(rates[key]))/Decimal(1000000)
                       for key in charges)
        return row.get('resource_cost') if mode == 'resource' else None

    def metrics(sample):
        baseline, candidate = zip(*sample)
        bcost = sum(row_cost(r) for r in baseline) if mode != 'unknown' else None
        ccost = sum(row_cost(r) for r in candidate) if mode != 'unknown' else None
        return dict(cost_ratio=float(ccost/bcost) if bcost else None,
            latency_ratio=_percentile([r['latency_ms'] for r in candidate], .95)/
                          _percentile([r['latency_ms'] for r in baseline], .95),
            quality_delta=sum(c['quality_score']-b['quality_score'] for b, c in sample)/len(sample))

    point = metrics(pairs)
    rng = random.Random(seed)
    draws = [metrics([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(repetitions)]
    intervals = {}
    for key in point:
        values = [draw[key] for draw in draws if draw[key] is not None]
        intervals[key] = ([_percentile(values, .025), _percentile(values, .975)]
                          if len(values) == repetitions else None)
    critical_count = sum(len(t['critical_facts']) for t in tasks)
    missing = [{'task_id': t['id'], 'fact': f} for t in tasks for f in t['critical_facts']
               if f not in paired[t['id'], 'candidate']['correct_critical_facts']]
    gates = dict(measured_held_out=evidence == 'measured' and plan['held_out'],
        sample_count=len(pairs) >= minimum, critical_facts=critical_count > 0 and not missing,
        cost=point['cost_ratio'] is not None and point['cost_ratio'] <= .8 and
             intervals['cost_ratio'] is not None and intervals['cost_ratio'][1] <= .8,
        latency=point['latency_ratio'] <= .9 and intervals['latency_ratio'][1] <= .9,
        quality=point['quality_delta'] >= 0 and intervals['quality_delta'][0] >= 0)
    summaries = {}
    for system in systems:
        selected = [paired[t['id'], system] for t in tasks]
        summaries[system] = dict(latency_p50_ms=_percentile([r['latency_ms'] for r in selected], .5),
            latency_p95_ms=_percentile([r['latency_ms'] for r in selected], .95),
            quality_mean=sum(r['quality_score'] for r in selected)/len(selected),
            cost=(str(sum(row_cost(r) for r in selected)) if mode == 'priced' else
                  sum(row_cost(r) for r in selected) if mode == 'resource' else None),
            usage={k: sum(r['usage'][k] for r in selected) for k in (*charges, 'calls')},
            peak_memory_bytes=max(r['peak_memory_bytes'] for r in selected),
            dropped_frames=sum(r['dropped_frames'] for r in selected))
    return dict(plan_sha256=digest, pairs=len(pairs), evidence=evidence, summaries=summaries,
        point=point, paired_bootstrap_95=intervals, gates=gates,
        meets_numeric_gates=all(gates.values()), missing_critical_facts=missing,
        certification='Trace authenticity, held-out provenance and scoring require independent audit; this is not an inference runner.')


def _read(path):
    with Path(path).open('rb') as stream:
        data = stream.read(16*1024*1024+1)
    if len(data) > 16*1024*1024:
        raise ValueError('evaluation file exceeds 16 MiB')
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--rows', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(compare(_read(args.plan), _read(args.rows),
        expected_plan_sha256=args.plan_sha256), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
