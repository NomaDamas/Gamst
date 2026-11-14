"""Native schema3 task/stream attempt scopes with explicit original-line claims.

These declarations bind recorded structure, never receipt or execution authenticity.
The full reconciliation protocol and line ownership are audited by the integration.
"""
from copy import deepcopy
import uuid

from .call_attempts import AttemptRecorder, _binding
from .paired_replay import _encoded, _keys
from .perception_comparison import _digest, _text, plan_digest
from .stream_cost_ledger import validate_protocol as validate_stream_protocol
from .three_arm_comparison import _snapshot
from .three_arm_replay import validate_collection_declaration

MAX_LINE_CLAIMS = 10000
REPORT_KEYS = ('schema_version', 'comparison_plan_sha256', 'collection_sha256',
    'stream_protocol_sha256', 'reconciliation_protocol_sha256', 'cost_sha256', 'scope_kind',
    'task_id', 'arm', 'stream_id', 'input_sha256', 'recording_sha256', 'configuration_sha256',
    'evidence', 'scope', 'max_attempts', 'closed', 'rejected', 'attempts', 'clock', 'sha256')
ATTEMPT_KEYS = ('attempt_id', 'provider', 'revision', 'role', 'phase', 'evidence_sha256',
    'prompt_sha256', 'execution_sha256', 'predecessor', 'relationship', 'started_ns', 'ended_ns',
    'status', 'cost_quantities', 'line_claims', 'execution_instance_sha256')
PROTOCOL_KEYS = ('schema_version', 'comparison_plan_sha256', 'collection_sha256', 'stream_protocol_sha256',
    'allocation_protocol_sha256', 'currency_atomic_per_comparison_unit', 'rate_map', 'lane_owners',
    'provider_counter_groups', 'vlm_counter_bindings', 'recording_groups', 'whole_stream_cost_policy')


def _scope(plan, spec, inputs, stream_protocol, reconciliation_protocol, *, expected_plan_sha256,
           expected_collection_sha256, expected_stream_protocol_sha256, expected_reconciliation_protocol_sha256,
           scope_kind, task_id, arm, stream_id):
    plan, _, _ = validate_collection_declaration(plan, spec, inputs,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256)
    stream_protocol = validate_stream_protocol(stream_protocol, expected_protocol_sha256=expected_stream_protocol_sha256)
    protocol = _snapshot(reconciliation_protocol)
    if plan_digest(protocol) != _digest(expected_reconciliation_protocol_sha256):
        raise ValueError('reconciliation protocol differs from frozen digest')
    _keys(protocol, PROTOCOL_KEYS, 'reconciliation protocol identity')
    if type(protocol['schema_version']) is not int or protocol['schema_version'] != 1:
        raise ValueError('reconciliation protocol schema1 required')
    for field, expected in (('comparison_plan_sha256', expected_plan_sha256),
                            ('collection_sha256', expected_collection_sha256)):
        if protocol[field] != expected or stream_protocol[field] != expected:
            raise ValueError('original plan/collection declaration mismatch')
    if protocol['stream_protocol_sha256'] != expected_stream_protocol_sha256:
        raise ValueError('reconciliation belongs to another stream protocol')
    _digest(protocol['allocation_protocol_sha256'])
    if plan['comparison']['cost']['mode'] != 'itemized':
        raise ValueError('native line-claim scopes require original itemized cost plan')
    streams = {stream['id']: stream for stream in stream_protocol['streams']}
    if type(stream_id) is not str or stream_id not in streams or type(arm) is not str or arm not in ('A','B','C'):
        raise ValueError('known original stream and arm required')
    stream = streams[stream_id]
    if stream['arm'] != arm:
        raise ValueError('scope stream belongs to another original arm')
    tasks = {task['id']: task for task in plan['comparison']['tasks']}
    if scope_kind == 'task':
        if type(task_id) is not str or task_id not in tasks:
            raise ValueError('task scope requires original task ID')
        input_digest = tasks[task_id]['input_sha256']
    elif scope_kind == 'stream':
        if task_id is not None:
            raise ValueError('stream scope cannot acquire a task owner')
        input_digest = None
    else:
        raise ValueError('native scope_kind must be task or stream')
    header = dict(schema_version=3, comparison_plan_sha256=expected_plan_sha256,
        collection_sha256=expected_collection_sha256, stream_protocol_sha256=expected_stream_protocol_sha256,
        reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,
        cost_sha256=plan_digest(plan['comparison']['cost']), scope_kind=scope_kind, task_id=task_id,
        arm=arm, stream_id=stream_id, input_sha256=input_digest, recording_sha256=stream['recording_sha256'],
        configuration_sha256=plan_digest(plan['systems'][arm]), evidence=plan['comparison']['evidence'])
    return plan, header


def _claims(claims):
    if type(claims) is not list or not 1 <= len(claims) <= MAX_LINE_CLAIMS:
        raise ValueError('nonempty bounded original ledger line claims required')
    seen = set()
    for claim in claims:
        key = _digest(claim)
        if key in seen:
            raise ValueError('duplicate original ledger line claim')
        seen.add(key)
    return seen


class StreamAttemptRecorder(AttemptRecorder):
    """Original task or stream-owned callbacks, compatible with RecordedContextClient.

    Complete lines may arrive after close or before callback completion. Snapshots
    remain unresolved until callbacks terminate and every attempt has explicit lines.
    """
    def __init__(self, plan, spec, inputs, stream_protocol, reconciliation_protocol, *,
                 expected_plan_sha256, expected_collection_sha256, expected_stream_protocol_sha256,
                 expected_reconciliation_protocol_sha256, scope_kind, task_id, arm, stream_id, max_attempts=128):
        plan, header = _scope(plan, spec, inputs, stream_protocol, reconciliation_protocol,
            expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256,
            expected_stream_protocol_sha256=expected_stream_protocol_sha256,
            expected_reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,
            scope_kind=scope_kind, task_id=task_id, arm=arm, stream_id=stream_id)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 1024:
            raise ValueError('1..1024 attempts required')
        header.update(scope=uuid.uuid4().hex, max_attempts=max_attempts)
        self._initialize(plan['comparison']['cost']['rate_card'], header)

    def report(self, attempt_id, *, quantities, receipt_sha256):
        raise ValueError('native schema3 requires report_lines; whole-receipt reporting is unsupported')

    def report_lines(self, attempt_id, *, quantities, line_claims, execution_instance_sha256):
        supplied = _snapshot(dict(quantities=quantities, line_claims=line_claims,
                                  execution_instance_sha256=execution_instance_sha256))
        quantities, line_claims = supplied['quantities'], supplied['line_claims']
        claims = _claims(line_claims)
        instance = _digest(supplied['execution_instance_sha256'])
        with self._lock:
            row = self._rows.get(attempt_id)
            if row is None:
                raise ValueError('unknown attempt')
            if row['cost_quantities'] is not None:
                raise ValueError('original line claims already supplied')
            required = _binding(self._card, row['provider'], row['revision'])
            if type(quantities) is not dict or set(quantities) != required:
                raise ValueError('every provider/revision rate requires an explicit quantity')
            if any(type(v) is not int or not 0 <= v <= 10**12 for v in quantities.values()):
                raise ValueError('bounded exact nonnegative quantities required')
            existing = set()
            for previous in self._rows.values():
                existing.update(previous.get('line_claims') or ())
                if previous.get('execution_instance_sha256') == instance:
                    raise ValueError('actual execution instance already owned in this scope')
            if existing & claims:
                raise ValueError('original ledger line already owned in this scope')
            if len(existing)+len(claims) > MAX_LINE_CLAIMS:
                raise ValueError('total line claims exceed 10000')
            row.update(cost_quantities=dict(quantities), line_claims=list(line_claims),
                       execution_instance_sha256=instance)

    def snapshot(self):
        with self._lock:
            rows = deepcopy(list(self._rows.values()))
            for row in rows:
                row.pop('receipt_sha256', None)
                row.setdefault('line_claims', None)
                row.setdefault('execution_instance_sha256', None)
            result = dict(self._header, closed=self._closed, rejected=self._rejected,
                attempts=rows, clock='monotonic_callback_duration_not_task_latency')
            result['sha256'] = plan_digest(result)
            return _snapshot(result)


def _validate_body(card, report):
    """Native body checks only; no synthetic receipt digest or relabeled saved report."""
    if report['closed'] is not True or type(report['rejected']) is not int or report['rejected'] != 0:
        raise ValueError('closed native scope without rejected admissions required')
    if report['clock'] != 'monotonic_callback_duration_not_task_latency':
        raise ValueError('unsupported callback clock scope')
    scope, limit, rows = report['scope'], report['max_attempts'], report['attempts']
    if type(scope) is not str or len(scope) != 32 or any(c not in '0123456789abcdef' for c in scope):
        raise ValueError('invalid native recorder scope')
    if type(limit) is not int or not 1 <= limit <= 1024 or type(rows) is not list or len(rows) > limit:
        raise ValueError('bounded native attempt rows required')
    totals, seen, claimed, instances, last_started = {key:0 for key in card}, {}, set(), set(), -1
    for index, row in enumerate(rows):
        _keys(row, ATTEMPT_KEYS, 'native schema3 attempt')
        if row['attempt_id'] != scope+':'+str(index):
            raise ValueError('ordered original native attempt IDs required')
        if (row['role'] not in ('perception','small_vlm','strong_vlm','text_llm') or
                row['phase'] not in ('cold','warm','unknown') or row['status'] not in ('success','error','cancelled')):
            raise ValueError('terminal supported native attempt required')
        _digest(row['evidence_sha256'])
        for field in ('prompt_sha256','execution_sha256'):
            if row[field] is not None: _digest(row[field])
        instance = _digest(row['execution_instance_sha256']); claims = _claims(row['line_claims'])
        if instance in instances or claims & claimed:
            raise ValueError('duplicate actual instance/original ledger line ownership')
        if len(claimed)+len(claims) > MAX_LINE_CLAIMS:
            raise ValueError('total native line claims exceed 10000')
        instances.add(instance); claimed.update(claims)
        start, end = row['started_ns'], row['ended_ns']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= 2**63-1 or start < last_started:
            raise ValueError('ordered bounded monotonic callback timings required')
        last_started = start
        if row['predecessor'] is not None and type(row['predecessor']) is not str:
            raise ValueError('predecessor must be original native attempt ID')
        parent = seen.get(row['predecessor'])
        if row['relationship'] == 'root':
            if row['predecessor'] is not None: raise ValueError('root cannot have predecessor')
        elif row['relationship'] in ('retry','escalation'):
            if parent is None or parent['ended_ns'] > start or parent['evidence_sha256'] != row['evidence_sha256']:
                raise ValueError('completed same-evidence native predecessor required')
            if row['relationship'] == 'retry' and any(parent[k] != row[k] for k in ('provider','revision','role')):
                raise ValueError('native retry binding mismatch')
        else:
            raise ValueError('unsupported native attempt lineage')
        _text(row['provider'], 'provider'); _text(row['revision'], 'revision')
        required = _binding(card, row['provider'], row['revision']); quantities = row['cost_quantities']
        if not required or type(quantities) is not dict or set(quantities) != required:
            raise ValueError('complete bound native provider quantities required')
        for key, value in quantities.items():
            if type(value) is not int or not 0 <= value <= 10**12:
                raise ValueError('bounded exact native cost quantity required')
            totals[key] += value
            if totals[key] > 10**12: raise ValueError('native scope quantity exceeds evaluator bound')
        seen[row['attempt_id']] = row
    return totals


def validate_report(plan, spec, inputs, stream_protocol, reconciliation_protocol, report, *,
                    expected_plan_sha256, expected_collection_sha256, expected_stream_protocol_sha256,
                    expected_reconciliation_protocol_sha256, expected_report_sha256,
                    scope_kind, task_id, arm, stream_id):
    plan, header = _scope(plan, spec, inputs, stream_protocol, reconciliation_protocol,
        expected_plan_sha256=expected_plan_sha256, expected_collection_sha256=expected_collection_sha256,
        expected_stream_protocol_sha256=expected_stream_protocol_sha256,
        expected_reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,
        scope_kind=scope_kind, task_id=task_id, arm=arm, stream_id=stream_id)
    report = _snapshot(report)
    _keys(report, REPORT_KEYS, 'native schema3 report')
    _digest(expected_report_sha256)
    if (report['sha256'] != expected_report_sha256 or
            plan_digest({k:v for k,v in report.items() if k != 'sha256'}) != expected_report_sha256):
        raise ValueError('native report differs from externally frozen digest')
    for key, value in header.items():
        if _encoded(report[key]) != _encoded(value):
            raise ValueError('native report differs from original scope binding: '+key)
    return _validate_body(plan['comparison']['cost']['rate_card'], report)
