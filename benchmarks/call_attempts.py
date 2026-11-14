"""Opt-in evaluation instrumentation; never certifies provider or billing truth."""
import asyncio
from copy import deepcopy
import threading
import time
import uuid

from .perception_comparison import _digest, _text, plan_digest, validate_plan


def _binding(card, provider, revision):
    return {key for key, rate in card.items()
            if (rate['provider'], rate['revision']) == (provider, revision)}


class AttemptRecorder:
    """One frozen task/system. Wrap every attempt explicitly; no automatic retries.

    Callback duration excludes scheduling before invocation and is not task latency.
    Unknown quantities remain unknown after errors/cancellation until late reporting.
    """
    def __init__(self, plan, *, expected_plan_sha256, task_id, system, max_attempts=128):
        digest, tasks, _, mode = validate_plan(plan, expected_plan_sha256=expected_plan_sha256)
        if mode != 'itemized' or task_id not in tasks or system not in ('baseline','candidate'):
            raise ValueError('known frozen task/system and itemized costs required')
        if type(max_attempts) is not int or not 1 <= max_attempts <= 1024:
            raise ValueError('1..1024 attempts required')
        header = dict(schema_version=1, plan_sha256=digest, cost_sha256=plan_digest(plan['cost']),
            task_id=task_id, system=system, input_sha256=tasks[task_id]['input_sha256'],
            configuration_sha256=plan_digest(plan['systems'][system]), evidence=plan['evidence'],
            scope=uuid.uuid4().hex, max_attempts=max_attempts)
        self._initialize(plan['cost']['rate_card'], header)

    def _initialize(self, card, header):
        self._card = deepcopy(card)
        self._header = deepcopy(header)
        self._rows = {}
        self._closed = False
        self._rejected = 0
        self._lock = threading.RLock()
        self._origin = time.perf_counter_ns()

    def _now(self):
        return time.perf_counter_ns() - self._origin

    async def invoke(self, callback, *, provider, revision, role, phase,
                     evidence_sha256, prompt_sha256=None, execution_sha256=None,
                     predecessor=None, relationship='root'):
        """Await callback(attempt_id); return its result without retaining it.

        The adapter may call report() inside the callback or after completion.
        An exception/cancellation propagates unchanged, after its attempt is retained.
        """
        if not callable(callback):
            raise ValueError('explicit async callback required')
        _text(provider, 'provider'); _text(revision, 'revision'); _digest(evidence_sha256)
        for value in (prompt_sha256, execution_sha256):
            if value is not None: _digest(value)
        if role not in ('perception','small_vlm','strong_vlm','text_llm') or phase not in ('cold','warm','unknown'):
            raise ValueError('explicit supported role and declared warm state required')
        if predecessor is not None and type(predecessor) is not str:
            raise ValueError('predecessor must be an attempt ID')
        if relationship not in ('root','retry','escalation'):
            raise ValueError('explicit supported relationship required')
        with self._lock:
            if self._closed:
                self._rejected += 1
                raise RuntimeError('attempt recorder closed')
            if len(self._rows) >= self._header['max_attempts']:
                self._rejected += 1
                raise RuntimeError('attempt capacity exhausted; callback not invoked')
            if not _binding(self._card, provider, revision):
                raise ValueError('provider/revision missing from frozen rate card')
            if relationship == 'root':
                if predecessor is not None: raise ValueError('root cannot have predecessor')
            else:
                previous = self._rows.get(predecessor)
                if previous is None or previous['status'] == 'running':
                    raise ValueError('completed predecessor in this scope required')
                if previous['evidence_sha256'] != evidence_sha256:
                    raise ValueError('linked attempts require the same evidence digest')
                if relationship == 'retry' and (previous['provider'],previous['revision'],previous['role']) != (provider,revision,role):
                    raise ValueError('retry must retain provider/revision/role')
            token = self._header['scope'] + ':' + str(len(self._rows))
            self._rows[token] = dict(attempt_id=token, provider=provider, revision=revision,
                role=role, phase=phase, evidence_sha256=evidence_sha256,
                prompt_sha256=prompt_sha256, execution_sha256=execution_sha256,
                predecessor=predecessor, relationship=relationship, started_ns=self._now(),
                ended_ns=None, status='running', cost_quantities=None, receipt_sha256=None)
        status = 'error'
        try:
            result = await callback(token)
            status = 'success'
            return result
        except asyncio.CancelledError:
            status = 'cancelled'
            raise
        finally:
            with self._lock:
                self._rows[token].update(status=status, ended_ns=self._now())

    def report(self, attempt_id, *, quantities, receipt_sha256):
        """One complete externally verified receipt; no unknown-to-zero conversion."""
        _digest(receipt_sha256)
        with self._lock:
            row = self._rows.get(attempt_id)
            if row is None: raise ValueError('unknown attempt')
            if row['cost_quantities'] is not None: raise ValueError('receipt already supplied')
            keys = _binding(self._card, row['provider'], row['revision'])
            if type(quantities) is not dict or set(quantities) != keys:
                raise ValueError('every provider/revision rate requires an explicit quantity')
            if any(type(v) is not int or not 0 <= v <= 10**12 for v in quantities.values()):
                raise ValueError('bounded exact nonnegative quantities required')
            row.update(cost_quantities=dict(quantities), receipt_sha256=receipt_sha256)

    def close(self):
        """Stop admission; running work and late receipts remain accounted for."""
        with self._lock: self._closed = True

    def snapshot(self):
        with self._lock:
            rows = deepcopy(list(self._rows.values()))
            result = dict(self._header, closed=self._closed, rejected=self._rejected,
                          attempts=rows, clock='monotonic_callback_duration_not_task_latency')
            result['sha256'] = plan_digest(result)
            return result


def cost_quantities(plan, report, *, expected_plan_sha256, expected_report_sha256, task_id, system):
    """Validate frozen attempt records and reduce receipts, not their truth.

    Unknown/inflight accounting or rejected admissions cannot certify completeness.
    The expected report digest must come from the independently reviewed artifact.
    """
    digest, tasks, _, mode = validate_plan(plan, expected_plan_sha256=expected_plan_sha256)
    _digest(expected_report_sha256)
    keys = {'schema_version','plan_sha256','cost_sha256','task_id','system','input_sha256',
            'configuration_sha256','evidence','scope','max_attempts','closed','rejected',
            'attempts','clock','sha256'}
    if type(report) is not dict or set(report) != keys:
        raise ValueError('exact attempt report fields required')
    if report['sha256'] != expected_report_sha256 or plan_digest({k:v for k,v in report.items() if k!='sha256'}) != expected_report_sha256:
        raise ValueError('attempt report differs from reviewed digest')
    if type(report['task_id']) is not str or type(report['system']) is not str:
        raise ValueError('task/system identifiers required')
    if report['task_id'] != task_id or report['system'] != system:
        raise ValueError('attempt report belongs to another task/system')
    task = tasks.get(report['task_id'])
    if (mode != 'itemized' or task is None or system not in ('baseline','candidate') or
            type(report['schema_version']) is not int or report['schema_version'] != 1 or
            report['plan_sha256'] != digest or report['cost_sha256'] != plan_digest(plan['cost']) or
            report['input_sha256'] != task['input_sha256'] or report['evidence'] != plan['evidence'] or
            report['configuration_sha256'] != plan_digest(plan['systems'][system])):
        raise ValueError('attempt report differs from frozen task/configuration')
    return _reduce_attempts(plan['cost']['rate_card'], report)


def _reduce_attempts(card, report):
    """Shared attempt-body validation; callers validate their native frozen header."""
    if report['closed'] is not True or type(report['rejected']) is not int or report['rejected'] != 0:
        raise ValueError('closed scope without rejected admissions required')
    if report['clock'] != 'monotonic_callback_duration_not_task_latency':
        raise ValueError('unsupported clock scope')
    scope = report['scope']
    if type(scope) is not str or len(scope) != 32 or any(c not in '0123456789abcdef' for c in scope):
        raise ValueError('invalid recorder scope')
    limit = report['max_attempts']; rows = report['attempts']
    if type(limit) is not int or not 1 <= limit <= 1024 or type(rows) is not list or len(rows) > limit:
        raise ValueError('invalid attempt bounds')
    totals = {key:0 for key in card}; seen = {}; last_started = -1
    fields = {'attempt_id','provider','revision','role','phase','evidence_sha256','prompt_sha256',
              'execution_sha256','predecessor','relationship','started_ns','ended_ns','status',
              'cost_quantities','receipt_sha256'}
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != fields or row['attempt_id'] != scope+':'+str(index):
            raise ValueError('ordered complete attempt rows required')
        if row['role'] not in ('perception','small_vlm','strong_vlm','text_llm') or row['phase'] not in ('cold','warm','unknown') or row['status'] not in ('success','error','cancelled'):
            raise ValueError('terminal supported attempt required')
        _digest(row['evidence_sha256']); _digest(row['receipt_sha256'])
        for field in ('prompt_sha256','execution_sha256'):
            if row[field] is not None: _digest(row[field])
        start, end = row['started_ns'], row['ended_ns']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= 2**63-1 or start < last_started:
            raise ValueError('ordered bounded monotonic timings required')
        last_started = start
        if row['predecessor'] is not None and type(row['predecessor']) is not str:
            raise ValueError('predecessor must be an attempt ID')
        parent = seen.get(row['predecessor'])
        if row['relationship'] == 'root':
            if row['predecessor'] is not None: raise ValueError('root cannot have predecessor')
        elif row['relationship'] in ('retry','escalation'):
            if parent is None or parent['ended_ns'] > start or parent['evidence_sha256'] != row['evidence_sha256']:
                raise ValueError('completed same-evidence predecessor required')
            if row['relationship'] == 'retry' and any(parent[k] != row[k] for k in ('provider','revision','role')):
                raise ValueError('retry binding mismatch')
        else: raise ValueError('unsupported lineage')
        _text(row['provider'],'provider'); _text(row['revision'],'revision')
        required = _binding(card,row['provider'],row['revision']); quantities = row['cost_quantities']
        if not required or type(quantities) is not dict or set(quantities) != required:
            raise ValueError('complete bound provider receipt required')
        for key, value in quantities.items():
            if type(value) is not int or not 0 <= value <= 10**12:
                raise ValueError('bounded exact cost quantity required')
            totals[key] += value
            if totals[key] > 10**12: raise ValueError('task quantity exceeds evaluator bound')
        seen[row['attempt_id']] = row
    return totals
