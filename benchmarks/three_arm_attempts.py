"""Native A/B/C attempt binding; hashes and recorded coverage are not authenticity."""
import argparse
import json
from pathlib import Path
import uuid

from .call_attempts import AttemptRecorder, _reduce_attempts
from .paired_replay import _encoded, _keys, _write_new
from .perception_comparison import _digest, _text, plan_digest
from .three_arm_comparison import _read, _snapshot
from .three_arm_evidence import join as join_evidence, read_traces
from .three_arm_replay import validate_collection_declaration

REPORT_KEYS = ('schema_version', 'comparison_plan_sha256', 'collection_sha256', 'cost_sha256',
    'task_id', 'arm', 'input_sha256', 'configuration_sha256', 'evidence', 'scope', 'max_attempts',
    'closed', 'rejected', 'attempts', 'clock', 'sha256')


def _scope(plan, spec, inputs, expected_plan_sha256, expected_collection_sha256, task_id, arm):
    plan, spec, _ = validate_collection_declaration(plan, spec, inputs,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256)
    tasks = {task['id']: task for task in plan['comparison']['tasks']}
    if (plan['comparison']['cost']['mode'] != 'itemized' or type(task_id) is not str or
            task_id not in tasks or type(arm) is not str or arm not in ('A', 'B', 'C')):
        raise ValueError('known original task/arm and itemized costs required')
    return plan, dict(schema_version=2, comparison_plan_sha256=expected_plan_sha256,
        collection_sha256=expected_collection_sha256, cost_sha256=plan_digest(plan['comparison']['cost']),
        task_id=task_id, arm=arm, input_sha256=tasks[task_id]['input_sha256'],
        configuration_sha256=plan_digest(plan['systems'][arm]), evidence=plan['comparison']['evidence'])


class ThreeArmAttemptRecorder(AttemptRecorder):
    """Native schema2 scope; inherited client instrumentation retains original A/B/C."""
    def __init__(self, plan, spec, inputs, *, expected_plan_sha256, expected_collection_sha256,
                 task_id, arm, max_attempts=128):
        plan, header = _scope(plan, spec, inputs, expected_plan_sha256,
                              expected_collection_sha256, task_id, arm)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 1024:
            raise ValueError('1..1024 attempts required')
        header.update(scope=uuid.uuid4().hex, max_attempts=max_attempts)
        self._initialize(plan['comparison']['cost']['rate_card'], header)


def _quantities(plan, report, header, expected_report_sha256):
    _keys(report, REPORT_KEYS, 'native three-arm attempt report')
    _digest(expected_report_sha256)
    if (report['sha256'] != expected_report_sha256 or
            plan_digest({k: v for k, v in report.items() if k != 'sha256'}) != expected_report_sha256):
        raise ValueError('native report differs from externally frozen digest')
    for key, value in header.items():
        if _encoded(report[key]) != _encoded(value):
            raise ValueError('native report differs from frozen task/arm/configuration: '+key)
    totals = _reduce_attempts(plan['comparison']['cost']['rate_card'], report)
    receipts = set()
    for attempt in report['attempts']:
        receipt = (attempt['provider'], attempt['revision'], attempt['receipt_sha256'])
        if receipt in receipts:
            raise ValueError('duplicate receipt identity; shared allocation unsupported')
        receipts.add(receipt)
    return totals


def cost_quantities(plan, spec, inputs, report, *, expected_plan_sha256, expected_collection_sha256,
                    expected_report_sha256, task_id, arm):
    """Reduce native terminal reports; zero recorded attempts never proves full coverage."""
    plan, header = _scope(plan, spec, inputs, expected_plan_sha256, expected_collection_sha256, task_id, arm)
    return _quantities(plan, _snapshot(report), header, expected_report_sha256)


def join(plan, spec, inputs, traces, completion, assessment, manifest, reports, *,
         expected_plan_sha256, expected_collection_sha256, expected_trace_sha256,
         expected_assessment_sha256, expected_manifest_sha256):
    """Bind original reports once, before unchanged numerical pair projections."""
    # Detached, bounded declarations are shared by both source and native report checks.
    plan, spec, inputs = _snapshot(plan), _snapshot(spec), _snapshot(inputs)
    assessment, manifest, reports = _snapshot(assessment), _snapshot(manifest), _snapshot(reports)
    result = join_evidence(plan, spec, inputs, traces, completion, assessment,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256,
        expected_trace_sha256=expected_trace_sha256, expected_assessment_sha256=expected_assessment_sha256)
    if plan['comparison']['cost']['mode'] != 'itemized':
        raise ValueError('native receipt binding requires itemized costs')
    if plan_digest(manifest) != _digest(expected_manifest_sha256):
        raise ValueError('manifest differs from externally frozen digest')
    _keys(manifest, ('schema_version', 'comparison_plan_sha256', 'collection_sha256',
        'last_trace_sha256', 'assessment_sha256', 'reviewer', 'coverage_evidence_sha256', 'entries'), 'manifest')
    for key, value in dict(schema_version=1, comparison_plan_sha256=expected_plan_sha256,
            collection_sha256=expected_collection_sha256, last_trace_sha256=expected_trace_sha256,
            assessment_sha256=expected_assessment_sha256).items():
        if _encoded(manifest[key]) != _encoded(value):
            raise ValueError('manifest differs from frozen source/assessment')
    _text(manifest['reviewer'], 'reviewer'); _digest(manifest['coverage_evidence_sha256'])
    count = len(result['rows'])
    entries = manifest['entries']
    if type(entries) is not list or len(entries) != count or type(reports) is not list or len(reports) != count:
        raise ValueError('exactly one manifest entry and report per original source trace required')
    bindings, by_digest = {}, {}
    for entry in entries:
        _keys(entry, ('trace_sha256', 'report_sha256', 'coverage_statement'), 'manifest entry')
        trace = _digest(entry['trace_sha256']); _digest(entry['report_sha256'])
        _text(entry['coverage_statement'], 'coverage statement')
        if trace in bindings:
            raise ValueError('duplicate source binding')
        bindings[trace] = entry
    for report in reports:
        _keys(report, REPORT_KEYS, 'native three-arm attempt report')
        digest = _digest(report['sha256'])
        if digest in by_digest:
            raise ValueError('duplicate report binding')
        by_digest[digest] = report
    scopes, attempt_ids, receipts = set(), set(), set()
    attempt_lineage = []
    card = plan['comparison']['cost']['rate_card']
    canonical_totals = {arm: {key: 0 for key in card} for arm in 'ABC'}
    tasks = {task['id']: task for task in plan['comparison']['tasks']}
    for row, source in zip(result['rows'], result['lineage']):
        binding = bindings.pop(source['trace_sha256'], None)
        report = None if binding is None else by_digest.pop(binding['report_sha256'], None)
        if report is None:
            raise ValueError('missing or reused original report binding')
        arm, task = row['arm'], row['task_id']
        header = dict(schema_version=2, comparison_plan_sha256=expected_plan_sha256,
            collection_sha256=expected_collection_sha256, cost_sha256=plan_digest(plan['comparison']['cost']),
            task_id=task, arm=arm, input_sha256=tasks[task]['input_sha256'],
            configuration_sha256=plan_digest(plan['systems'][arm]), evidence=plan['comparison']['evidence'])
        quantities = _quantities(plan, report, header, binding['report_sha256'])
        if quantities != row['cost_quantities']:
            raise ValueError('assessment quantities differ from complete recorded receipt totals')
        if report['scope'] in scopes:
            raise ValueError('duplicate recorder scope across original sources')
        scopes.add(report['scope'])
        for attempt in report['attempts']:
            receipt = (attempt['provider'], attempt['revision'], attempt['receipt_sha256'])
            if attempt['attempt_id'] in attempt_ids or receipt in receipts:
                raise ValueError('duplicate attempt/receipt across original sources; allocation unsupported')
            attempt_ids.add(attempt['attempt_id']); receipts.add(receipt)
        for key, quantity in quantities.items():
            canonical_totals[arm][key] += quantity
        attempt_lineage.append(dict(ordinal=source['ordinal'], task_id=task, arm=arm,
            trace_sha256=source['trace_sha256'], report_sha256=binding['report_sha256'],
            row_sha256=source['row_sha256']))
    if bindings or by_digest:
        raise ValueError('unmatched manifest binding or report')
    result.update(attempt_lineage=attempt_lineage, attempt_lineage_sha256=plan_digest(attempt_lineage),
        manifest_sha256=expected_manifest_sha256, canonical_cost_quantities=canonical_totals,
        recorded_attempts=len(attempt_ids), unique_recorded_receipts=len(receipts),
        attempt_binding_verified=True, recorded_receipt_totals_verified=True,
        instrumentation_coverage_verified=False, receipts_verified=False, actual_acceptance='incomplete',
        certification='Original source/report/assessment hash bindings and recorded itemized reduction only. '
        'Canonical A is counted once. Receipt authenticity, hidden or uninstrumented attempts, independent '
        'scoring, real inference, whole-stream resource scope and shared allocation remain unverified.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'assessment', 'manifest', 'reports', 'plan-sha256', 'collection-sha256',
                 'trace-sha256', 'assessment-sha256', 'manifest-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); root = Path(args.run)
    result = join(_read(root/'comparison-plan.json'), _read(root/'collection-spec.json'),
        _read(root/'inputs.json'), read_traces(root/'source-traces.jsonl'), _read(root/'complete.json'),
        _read(args.assessment), _read(args.manifest), _read(args.reports),
        expected_plan_sha256=args.plan_sha256, expected_collection_sha256=args.collection_sha256,
        expected_trace_sha256=args.trace_sha256, expected_assessment_sha256=args.assessment_sha256,
        expected_manifest_sha256=args.manifest_sha256)
    output = Path(args.output); output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_new(output/'rows.json', result.pop('rows'))
    _write_new(output/'lineage.json', result.pop('lineage'))
    _write_new(output/'attempt-lineage.json', result.pop('attempt_lineage'))
    _write_new(output/'join.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
