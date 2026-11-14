"""Frozen A/B/C numeric comparison; no execution, receipt or source certification."""
import argparse
import copy
import json
from pathlib import Path

from .perception_comparison import compare, plan_digest, validate_plan, _digest

COMMON = {'evidence', 'held_out', 'evaluation_id', 'scoring_protocol', 'latency_protocol',
          'trace_provenance', 'min_pairs', 'bootstrap_repetitions', 'bootstrap_seed', 'tasks', 'cost'}
CONFIG = {'model', 'revision', 'precision', 'hardware', 'policy', 'warm_state', 'sampling', 'window_lookahead'}
ROW = {'task_id', 'arm', 'plan_sha256', 'configuration_sha256', 'input_sha256', 'latency_ms',
       'quality_score', 'correct_critical_facts', 'peak_memory_bytes', 'dropped_frames', 'usage'}


def _keys(value, fields, label):
    if type(value) is not dict or set(value) != fields:
        raise ValueError('exact '+label+' fields required')


def _snapshot(value):
    # Bound both public in-memory APIs and the CLI; never retain caller dictionaries.
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    if len(encoded) > 16*1024*1024:
        raise ValueError('three-arm declaration/rows exceed 16 MiB')
    return json.loads(encoded)


def _project(plan, arm):
    pair = copy.deepcopy(plan['comparison'])
    pair['schema_version'] = 1
    pair['systems'] = {'baseline': copy.deepcopy(plan['systems']['A']),
                       'candidate': copy.deepcopy(plan['systems'][arm])}
    # Preserve the structured sampling declaration in each derived frozen plan.
    pair['three_arm_origin'] = {'plan_sha256': plan_digest(plan), 'candidate_arm': arm,
                               'baseline_sampling': copy.deepcopy(plan['baseline_sampling'])}
    return pair


def validate_three_arm_plan(plan, *, expected_plan_sha256):
    plan = _snapshot(plan)
    if plan_digest(plan) != _digest(expected_plan_sha256):
        raise ValueError('three-arm plan differs from frozen digest')
    _keys(plan, {'schema_version', 'primary_comparison', 'comparison', 'systems', 'baseline_sampling'}, 'plan')
    if type(plan['schema_version']) is not int or plan['schema_version'] != 1 or plan['primary_comparison'] != 'C_vs_A':
        raise ValueError('schema1 with C_vs_A primary comparison required')
    _keys(plan['comparison'], COMMON, 'common comparison')
    _keys(plan['systems'], {'A', 'B', 'C'}, 'systems')
    for config in plan['systems'].values():
        _keys(config, CONFIG, 'system configuration')
    sampling = plan['baseline_sampling']
    _keys(sampling, {'period_ms', 'phase_ms', 'time_origin_policy', 'audio_policy', 'task_policy'}, 'baseline sampling')
    if (type(sampling['period_ms']) is not int or sampling['period_ms'] != 1000 or
            type(sampling['phase_ms']) is not int or not 0 <= sampling['phase_ms'] < 1000 or
            sampling['time_origin_policy'] not in ('recording_start', 'capture_epoch') or
            any(type(sampling[k]) is not str or not 1 <= len(sampling[k]) <= 4096
                for k in ('audio_policy', 'task_policy')) or plan['systems']['A']['sampling'] != 'periodic_1fps'):
        raise ValueError('explicit periodic 1fps baseline, phase and audio/task policies required')
    for arm in ('B', 'C'):
        pair = _project(plan, arm)
        validate_plan(pair, expected_plan_sha256=plan_digest(pair))
    return plan


def project_pair_plan(plan, candidate_arm, *, expected_plan_sha256):
    if candidate_arm not in ('B', 'C'):
        raise ValueError('candidate arm must be B or C')
    return _project(validate_three_arm_plan(plan, expected_plan_sha256=expected_plan_sha256), candidate_arm)


def compare_three_arm(plan, rows, *, expected_plan_sha256, expected_rows_sha256):
    plan = validate_three_arm_plan(plan, expected_plan_sha256=expected_plan_sha256)
    rows = _snapshot(rows)
    if plan_digest(rows) != _digest(expected_rows_sha256):
        raise ValueError('three-arm rows differ from frozen digest')
    tasks = {task['id']: task for task in plan['comparison']['tasks']}
    if type(rows) is not list or len(rows) != 3*len(tasks):
        raise ValueError('exactly one A, B and C row per task required')
    mode = plan['comparison']['cost']['mode']
    fields = ROW | ({'resource_cost'} if mode == 'resource' else {'cost_quantities'} if mode == 'itemized' else set())
    selected = {}
    for row in rows:
        _keys(row, fields, 'three-arm row')
        arm, task = row['arm'], row['task_id']
        if (type(arm) is not str or arm not in ('A', 'B', 'C') or type(task) is not str or
                task not in tasks or (task, arm) in selected):
            raise ValueError('unknown or duplicate task/arm')
        if (row['plan_sha256'] != expected_plan_sha256 or row['input_sha256'] != tasks[task]['input_sha256'] or
                row['configuration_sha256'] != plan_digest(plan['systems'][arm])):
            raise ValueError('row plan/input/configuration binding mismatch')
        selected[task, arm] = row
    comparisons, projections = {}, {}
    for arm in ('B', 'C'):
        pair = _project(plan, arm); digest = plan_digest(pair)
        derived, mapping = [], []
        for task in tasks:
            for source_arm, system in (('A', 'baseline'), (arm, 'candidate')):
                original = selected[task, source_arm]
                row = {k: copy.deepcopy(v) for k, v in original.items() if k not in ('arm', 'configuration_sha256')}
                row.update(system=system, plan_sha256=digest)
                derived.append(row)
                mapping.append({'task_id': task, 'arm': source_arm, 'source_row_sha256': plan_digest(original),
                                'derived_row_sha256': plan_digest(row)})
        key = arm+'_vs_A'
        comparisons[key] = compare(pair, derived, expected_plan_sha256=digest)
        projections[key] = {'plan_sha256': digest, 'rows_sha256': plan_digest(derived), 'mapping': mapping}
    # Report each arm once. Never sum the two pair reports and count A twice.
    summaries = {'A': comparisons['C_vs_A']['summaries']['baseline'],
                 'B': comparisons['B_vs_A']['summaries']['candidate'],
                 'C': comparisons['C_vs_A']['summaries']['candidate']}
    return {'schema_version': 1, 'plan_sha256': expected_plan_sha256, 'rows_sha256': expected_rows_sha256,
            'primary_comparison': 'C_vs_A', 'diagnostic_only': ['B_vs_A'], 'summaries': summaries,
            'comparisons': comparisons, 'projections': projections,
            'primary_meets_numeric_gates': comparisons['C_vs_A']['meets_numeric_gates'],
            'actual_acceptance': 'incomplete', 'source_verified': False, 'receipts_verified': False,
            'sampling_execution_verified': False, 'recording_cluster_uncertainty_verified': False,
            'certification': 'Derived numeric rows only, not rewritten execution traces. '
                'Independent source/receipt/scoring audit, recording-cluster uncertainty and real held-out inference remain required.'}


def _read(path):
    with Path(path).open('rb') as source:
        data = source.read(16*1024*1024+1)
    if len(data) > 16*1024*1024:
        raise ValueError('evaluation file exceeds 16 MiB')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'rows', 'plan-sha256', 'rows-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_three_arm(_read(args.plan), _read(args.rows),
        expected_plan_sha256=args.plan_sha256, expected_rows_sha256=args.rows_sha256), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
