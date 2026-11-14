"""Join original A/B/C sources to frozen assessments; never certify their truth."""
import argparse
import json
from pathlib import Path

from .paired_replay import _encoded, _keys, _write_new
from .perception_comparison import _digest, _number, _text, plan_digest
from .replay_evidence import ASSESSMENT_KEYS, TRACE_KEYS
from .three_arm_comparison import _read, _snapshot, compare_three_arm
from .three_arm_replay import ORDER, PERMUTATIONS, MAX_TRACE_BYTES, validate_collection_declaration

SOURCE_KEYS = tuple(k for k in TRACE_KEYS if k not in ('system', 'plan_sha256')) + (
    'arm', 'comparison_plan_sha256', 'collection_sha256', 'configuration_sha256')
COMPLETE_KEYS = ('schema_version', 'comparison_plan_sha256', 'collection_sha256', 'state',
    'records', 'last_trace_sha256', 'order', 'arm_position_counts', 'complete_six_task_blocks',
    'remaining_tasks', 'actual_acceptance', 'certification')


def _strict_decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON number')
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    _encoded(value)  # Reject exponent overflow as well as explicit NaN/Infinity.
    return value


def read_traces(path):
    with Path(path).open('rb') as stream:
        count = 0
        while True:
            line = stream.readline(MAX_TRACE_BYTES + 1)
            if not line:
                return
            count += 1
            if len(line) > MAX_TRACE_BYTES or count > 3000 or not line.endswith(b'\n'):
                raise ValueError('source trace line/count bound or incomplete line')
            yield _strict_decode(line)


def validate_source_trace(trace, *, task, arm, ordinal, previous, plan, by_id,
                          expected_plan_sha256, expected_collection_sha256):
    """Validate original unscored trace structure and lineage without assessment.

    The response digest is an original producer locator; raw stdout is not
    retained here, so this does not authenticate its contents independently.
    """
    if len(_encoded(trace)) + 1 > MAX_TRACE_BYTES:
        raise ValueError('source trace exceeds 512 KiB')
    identity = dict(schema_version=1, comparison_plan_sha256=expected_plan_sha256,
                    collection_sha256=expected_collection_sha256)
    trace = _snapshot(trace)
    _keys(trace, SOURCE_KEYS, 'original three-arm trace')
    key = _digest(trace['trace_sha256'])
    config = plan['systems'][arm]; config_digest = plan_digest(config)
    content = {k: v for k, v in trace.items() if k != 'trace_sha256'}
    if (plan_digest(content) != key or trace['previous_trace_sha256'] != previous or
            type(trace['ordinal']) is not int or trace['ordinal'] != ordinal or
            trace['task_id'] != task['id'] or trace['arm'] != arm or
            trace['comparison_plan_sha256'] != expected_plan_sha256 or
            trace['collection_sha256'] != expected_collection_sha256 or
            trace['configuration_sha256'] != config_digest or
            trace['input_sha256'] != task['input_sha256']):
        raise ValueError('source chain/order/input/plan/configuration mismatch')
    request = dict(identity, task_id=task['id'], arm=arm, configuration_sha256=config_digest,
                   configuration=config, input=by_id[task['id']])
    if trace['request_sha256'] != plan_digest(request):
        raise ValueError('source request mismatch')
    if trace['measurement_status'] != 'pipeline_reported_unverified' or trace['quality_status'] != 'unscored':
        raise ValueError('original unscored source required')
    status = trace['status']
    if status not in ('success', 'error', 'timeout'):
        raise ValueError('unknown source status')
    lifetime = _number(trace['process_lifetime_ms'], 'process lifetime', minimum=1e-9)
    ready = trace['answer_ready_ms']
    if ready is not None:
        _number(ready, 'answer ready', minimum=1e-9, maximum=lifetime)
    for field in ('response_sha256', 'stderr_sha256'):
        if trace[field] is not None:
            _digest(trace[field])
    if trace['returncode'] is not None and type(trace['returncode']) is not int:
        raise ValueError('invalid returncode')
    if trace['diagnostic'] is not None:
        _text(trace['diagnostic'], 'diagnostic')
    response = trace['response']
    if response is not None:
        _keys(response, ('schema_version', 'status', 'answer', 'measurements'), 'response')
        if (type(response['schema_version']) is not int or response['schema_version'] != 1 or
                response['status'] not in ('success', 'error') or type(response['answer']) is not str or
                type(response['measurements']) is not dict or trace['response_sha256'] is None or ready is None):
            raise ValueError('invalid source response')
    if status == 'success' and (response is None or response['status'] != 'success' or
            trace['returncode'] != 0 or trace['diagnostic'] is not None):
        raise ValueError('inconsistent source success')
    if status == 'timeout' and trace['diagnostic'] != 'deadline_exceeded':
        raise ValueError('inconsistent timeout')
    return trace, key, config_digest, status, lifetime, ready


def join(plan, spec, inputs, traces, completion, assessment, *, expected_plan_sha256,
         expected_collection_sha256, expected_trace_sha256, expected_assessment_sha256):
    """Audit portable declarations and hash lineage, not inference or measurement truth."""
    plan, spec, by_id = validate_collection_declaration(plan, spec, inputs,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256)
    completion, assessment = _snapshot(completion), _snapshot(assessment)
    _digest(expected_trace_sha256); _digest(expected_assessment_sha256)
    tasks = plan['comparison']['tasks']
    identity = dict(schema_version=1, comparison_plan_sha256=expected_plan_sha256,
                    collection_sha256=expected_collection_sha256)
    positions = {arm: [0, 0, 0] for arm in 'ABC'}
    expected = []
    for i, task in enumerate(tasks):
        for position, arm in enumerate(PERMUTATIONS[i % 6]):
            expected.append((task, arm))
            positions[arm][position] += 1
    _keys(completion, COMPLETE_KEYS, 'completion')
    for key, value in dict(identity, state='collected_unscored', records=len(expected),
            last_trace_sha256=expected_trace_sha256, order=ORDER, arm_position_counts=positions,
            complete_six_task_blocks=len(tasks)//6, remaining_tasks=len(tasks)%6,
            actual_acceptance='incomplete').items():
        if _encoded(completion[key]) != _encoded(value):
            raise ValueError('incomplete or mismatched collection: '+key)
    _text(completion['certification'], 'certification')
    if plan_digest(assessment) != expected_assessment_sha256:
        raise ValueError('assessment differs from externally frozen digest')
    _keys(assessment, ('schema_version', 'comparison_plan_sha256', 'collection_sha256',
        'last_trace_sha256', 'reviewer', 'rubric_sha256', 'measurement_protocol_sha256',
        'source_evidence_sha256', 'entries'), 'assessment')
    policy = spec['evidence_join']
    for key, value in dict(identity, last_trace_sha256=expected_trace_sha256,
            rubric_sha256=policy['rubric_sha256'],
            measurement_protocol_sha256=policy['measurement_protocol_sha256']).items():
        if _encoded(assessment[key]) != _encoded(value):
            raise ValueError('assessment source/rubric/protocol mismatch')
    _text(assessment['reviewer'], 'reviewer'); _digest(assessment['source_evidence_sha256'])
    entries = assessment['entries']
    if type(entries) is not list or len(entries) != len(expected):
        raise ValueError('assess every source trace exactly once')
    mode = plan['comparison']['cost']['mode']
    assessments = {}
    for entry in entries:
        _keys(entry, ASSESSMENT_KEYS + (('cost_quantities',) if mode == 'itemized' else ()), 'assessment entry')
        key = _digest(entry['trace_sha256'])
        if key in assessments:
            raise ValueError('duplicate assessed trace')
        assessments[key] = entry
    rows, lineage, previous = [], [], '0'*64
    for ordinal, trace in enumerate(traces):
        if ordinal >= len(expected) or ordinal >= 3000:
            raise ValueError('extra source trace')
        if len(_encoded(trace)) + 1 > MAX_TRACE_BYTES:
            raise ValueError('source trace exceeds 512 KiB')
        task, arm = expected[ordinal]
        trace, key, config_digest, status, lifetime, ready = validate_source_trace(
            trace, task=task, arm=arm, ordinal=ordinal, previous=previous,
            plan=plan, by_id=by_id, expected_plan_sha256=expected_plan_sha256,
            expected_collection_sha256=expected_collection_sha256)
        if key not in assessments:
            raise ValueError('missing trace-bound assessment')
        scored = assessments.pop(key)
        failed = status != 'success'
        if failed and (scored['quality_score'] != 0 or scored['correct_critical_facts'] != []):
            raise ValueError('failed tasks must retain zero quality and no correct critical facts')
        if mode != 'resource' and scored['resource_cost'] is not None:
            raise ValueError('resource_cost must be null outside resource mode')
        row = dict(task_id=task['id'], arm=arm, plan_sha256=expected_plan_sha256,
            configuration_sha256=config_digest, input_sha256=task['input_sha256'],
            latency_ms=max(lifetime, spec['runner']['timeout_ms']) if failed else ready,
            **{k: scored[k] for k in ('quality_score', 'correct_critical_facts', 'usage',
                                    'peak_memory_bytes', 'dropped_frames')})
        if mode == 'resource':
            row['resource_cost'] = scored['resource_cost']
        if mode == 'itemized':
            row['cost_quantities'] = scored['cost_quantities']
        rows.append(row)
        lineage.append(dict(ordinal=ordinal, task_id=task['id'], arm=arm, trace_sha256=key,
            assessment_entry_sha256=plan_digest(scored), row_sha256=plan_digest(row)))
        previous = key
    if len(rows) != len(expected) or previous != expected_trace_sha256 or assessments:
        raise ValueError('truncated source or unmatched assessment')
    rows_digest = plan_digest(rows)
    comparison = compare_three_arm(plan, rows, expected_plan_sha256=expected_plan_sha256,
                                   expected_rows_sha256=rows_digest)
    return dict(identity, rows=rows, lineage=lineage, rows_sha256=rows_digest,
        lineage_sha256=plan_digest(lineage), last_trace_sha256=previous,
        assessment_sha256=expected_assessment_sha256, comparison=comparison,
        actual_acceptance='incomplete', source_hash_integrity_verified=True,
        source_verified=False, live_assets_verified=False, receipts_verified=False,
        independent_scoring_verified=False, measurement_truth_verified=False,
        certification='Portable declaration and source hash lineage only. Real inference, sampling execution, '
        'independent scoring, receipt authenticity, attempt coverage, shared allocation and measurements remain unverified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'assessment', 'plan-sha256', 'collection-sha256', 'trace-sha256', 'assessment-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); root = Path(args.run)
    result = join(_read(root/'comparison-plan.json'), _read(root/'collection-spec.json'),
        _read(root/'inputs.json'), read_traces(root/'source-traces.jsonl'), _read(root/'complete.json'),
        _read(args.assessment), expected_plan_sha256=args.plan_sha256,
        expected_collection_sha256=args.collection_sha256, expected_trace_sha256=args.trace_sha256,
        expected_assessment_sha256=args.assessment_sha256)
    output = Path(args.output); output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_new(output/'rows.json', result.pop('rows'))
    _write_new(output/'lineage.json', result.pop('lineage'))
    _write_new(output/'join.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
