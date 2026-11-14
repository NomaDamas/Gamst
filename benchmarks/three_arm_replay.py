"""Collect original A/B/C command traces; no scores or real-inference certification."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from .paired_replay import _execute, _encoded, _keys, _write_new, validate_command, validate_command_declaration
from .perception_comparison import plan_digest, _digest
from .three_arm_comparison import _snapshot, _read, validate_three_arm_plan

ORDER = 'cycle_all_six_permutations_v1'
PERMUTATIONS = ('ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA')
MAX_TRACE_BYTES = 512*1024


def validate_collection_declaration(comparison_plan, collection_spec, inputs, *, expected_plan_sha256,
                             expected_collection_sha256):
    """Return detached validated declarations; preflight never starts a command."""
    plan = validate_three_arm_plan(comparison_plan, expected_plan_sha256=expected_plan_sha256)
    spec, inputs = _snapshot(collection_spec), _snapshot(inputs)
    if plan_digest(spec) != _digest(expected_collection_sha256):
        raise ValueError('collection differs from frozen digest')
    _keys(spec, ('schema_version', 'comparison_plan_sha256', 'runner', 'evidence_join'), 'collection spec')
    if (type(spec['schema_version']) is not int or spec['schema_version'] != 1 or
            spec['comparison_plan_sha256'] != expected_plan_sha256):
        raise ValueError('matching schema1 comparison plan required')
    runner = spec['runner']
    _keys(runner, ('timeout_ms', 'order', 'commands'), 'runner')
    if (type(runner['timeout_ms']) is not int or not 1 <= runner['timeout_ms'] <= 3600000 or
            runner['order'] != ORDER):
        raise ValueError('bounded timeout and fixed six-permutation order required')
    _keys(runner['commands'], ('A', 'B', 'C'), 'commands')
    for command in runner['commands'].values():
        validate_command_declaration(command)
    policy = spec['evidence_join']
    _keys(policy, ('schema_version', 'rubric_sha256', 'measurement_protocol_sha256', 'failure_policy'), 'evidence join')
    if (type(policy['schema_version']) is not int or policy['schema_version'] != 1 or
            policy['failure_policy'] != 'zero_quality_timeout_floor'):
        raise ValueError('schema1 frozen failure policy required')
    _digest(policy['rubric_sha256']); _digest(policy['measurement_protocol_sha256'])
    tasks = {task['id']: task for task in plan['comparison']['tasks']}
    if type(inputs) is not list or len(inputs) != len(tasks):
        raise ValueError('exactly one recorded input per task required')
    by_id = {}
    for item in inputs:
        _keys(item, ('task_id', 'input'), 'input')
        task = item['task_id']
        if type(task) is not str or task not in tasks or task in by_id or type(item['input']) is not dict:
            raise ValueError('unknown or duplicate task input')
        if plan_digest(item['input']) != tasks[task]['input_sha256']:
            raise ValueError('recorded input differs from frozen task')
        by_id[task] = item['input']
    return plan, spec, by_id


def validate_collection_spec(comparison_plan, collection_spec, inputs, *, expected_plan_sha256,
                             expected_collection_sha256):
    result = validate_collection_declaration(comparison_plan, collection_spec, inputs,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256)
    for command in result[1]['runner']['commands'].values():
        validate_command(command)
    return result


async def run(comparison_plan, collection_spec, inputs, output_directory, *, expected_plan_sha256,
              expected_collection_sha256):
    if os.name != 'posix':
        raise RuntimeError('command collection requires POSIX owned process groups')
    plan, spec, by_id = validate_collection_spec(comparison_plan, collection_spec, inputs,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256)
    tasks = plan['comparison']['tasks']
    root = Path(output_directory)
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_new(root/'comparison-plan.json', plan)
    _write_new(root/'collection-spec.json', spec)
    _write_new(root/'inputs.json', [{'task_id': task['id'], 'input': by_id[task['id']]} for task in tasks])
    identity = dict(schema_version=1, comparison_plan_sha256=expected_plan_sha256,
                    collection_sha256=expected_collection_sha256)
    _write_new(root/'collection.json', dict(identity, state='started', order=ORDER, scoring='unscored',
                                          started_unix_ns=time.time_ns()))
    count, previous = 0, '0'*64
    positions = {arm: [0, 0, 0] for arm in 'ABC'}
    with (root/'source-traces.jsonl').open('xb') as stream:
        os.chmod(root/'source-traces.jsonl', 0o600)
        for index, task in enumerate(tasks):
            for position, arm in enumerate(PERMUTATIONS[index % 6]):
                configuration = plan['systems'][arm]
                configuration_sha256 = plan_digest(configuration)
                request = dict(identity, task_id=task['id'], arm=arm,
                    configuration_sha256=configuration_sha256, configuration=configuration, input=by_id[task['id']])
                result = await _execute(spec['runner']['commands'][arm], request, spec['runner']['timeout_ms'])
                record = dict(ordinal=count, task_id=task['id'], arm=arm,
                    comparison_plan_sha256=expected_plan_sha256, collection_sha256=expected_collection_sha256,
                    configuration_sha256=configuration_sha256, input_sha256=task['input_sha256'],
                    request_sha256=plan_digest(request), previous_trace_sha256=previous, **result)
                digest = plan_digest(record)
                line = _encoded(dict(record, trace_sha256=digest))+b'\n'
                if len(line) > MAX_TRACE_BYTES:
                    raise ValueError('source trace exceeds 512 KiB')
                stream.write(line); stream.flush(); os.fsync(stream.fileno())
                count += 1; previous = digest; positions[arm][position] += 1
    summary = dict(identity, state='collected_unscored', records=count, last_trace_sha256=previous,
        order=ORDER, arm_position_counts=positions, complete_six_task_blocks=len(tasks)//6,
        remaining_tasks=len(tasks)%6, actual_acceptance='incomplete',
        certification='Original unscored command reports only. Source inference, sampling execution, receipts, '
                      'resource scope, independent scoring and recording-cluster uncertainty remain unverified.')
    _write_new(root/'complete.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'collection-spec', 'inputs', 'plan-sha256', 'collection-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(_read(args.plan), _read(args.collection_spec), _read(args.inputs), args.output,
        expected_plan_sha256=args.plan_sha256, expected_collection_sha256=args.collection_sha256))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
