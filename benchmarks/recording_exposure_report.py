"""Original paired quality/latency and recording-exposure reconciliation v2."""
from dataclasses import dataclass
from decimal import Decimal,Context,localcontext
from fractions import Fraction
import hashlib
import math
import random
from gamst.capabilities import identifier

from .recording_financial import digest,encode,fraction,integer,keys,ratio,sha
from .recording_lifetime_ledger import ValidatedLifetimeLedger
from .matched_source_exposure import compare_costs
from .perception_comparison import _percentile
from .three_arm_comparison import validate_three_arm_plan
from .perception_comparison import plan_digest

_QUALITY_ISSUER = object()


@dataclass(frozen=True,init=False)
class ValidatedQualityLatency:
    comparison_plan_sha256: str
    assessment_sha256: str
    report_sha256: str
    reports: object
    runtime_policy_sha256: str
    collection_sha256: str
    execution_inventory_sha256: str
    execution_receipts: tuple

    def __init__(self,plan_sha256,assessment_sha256,reports,*,runtime_policy_sha256,
                 collection_sha256,execution_inventory_sha256,execution_receipts,_issuer=None):
        if _issuer is not _QUALITY_ISSUER: raise TypeError('validated original quality/latency report required')
        object.__setattr__(self,'comparison_plan_sha256',plan_sha256)
        object.__setattr__(self,'assessment_sha256',assessment_sha256)
        object.__setattr__(self,'runtime_policy_sha256',runtime_policy_sha256)
        object.__setattr__(self,'collection_sha256',collection_sha256)
        object.__setattr__(self,'execution_inventory_sha256',execution_inventory_sha256)
        object.__setattr__(self,'execution_receipts',execution_receipts)
        # Keep immutable canonical bytes, not mutable caller dictionaries.
        object.__setattr__(self,'reports',encode(reports,16*1024**2))
        object.__setattr__(self,'report_sha256',hashlib.sha256(self.reports).hexdigest())


def compare_quality_latency(plan,assessment,receipts,*,expected_assessment_sha256):
    # Preserve the original comparator's precision and isolation from caller
    # decimal settings, including the full accepted finite-float exponent range.
    with localcontext(Context(prec=400)):
        return _compare_quality_latency(plan,assessment,receipts,expected_assessment_sha256=expected_assessment_sha256)


def _compare_quality_latency(plan,assessment,receipts,*,expected_assessment_sha256):
    """Same original task bootstrap/critical facts and p95 target, no fake cost.

    Original exact parent receipts are supplied in original task/arm order.
    This helper never calls the old priced comparator with missing usage set0.
    Runtime receipt/source lineage is validated before invoking this helper.
    """
    plan = validate_three_arm_plan(plan,expected_plan_sha256=plan_digest(plan))
    # Assessment legitimately includes finite quality decimals; financial JSON
    # forbids floats, so validate and hash the original assessment separately.
    if plan_digest(assessment) != sha(expected_assessment_sha256): raise ValueError('assessment differs from external reviewed digest')
    keys(assessment,('schema_version','kind','comparison_plan_sha256','runtime_policy_sha256',
        'collection_sha256','execution_inventory_sha256','reviewer','rubric_sha256',
        'measurement_protocol_sha256','source_evidence_sha256','rows'),'recording assessment')
    if (type(assessment['schema_version']) is not int or assessment['schema_version'] != 1 or
            assessment['kind'] != 'recording_assessment' or assessment['comparison_plan_sha256'] != plan_digest(plan)):
        raise ValueError('original recording assessment schema required')
    for field in ('runtime_policy_sha256','collection_sha256','execution_inventory_sha256',
                  'rubric_sha256','measurement_protocol_sha256','source_evidence_sha256'): sha(assessment[field])
    identifier(assessment['reviewer'])
    if type(assessment['rows']) is not list or len(assessment['rows']) > 3000: raise ValueError('bounded original assessment rows required')
    tasks = plan['comparison']['tasks']; taskmap = {t['id']:t for t in tasks}
    if len(assessment['rows']) != 3*len(tasks) or type(receipts) is not list or len(receipts) != len(assessment['rows']):
        raise ValueError('all original task/arm rows and parent receipts required')
    paired = {}; receipt_groups = {}
    for row,pair in zip(assessment['rows'],receipts):
        keys(row,('task_id','input_sha256','invocation_ordinal','parent_receipt_sha256','answer_sha256',
                  'quality_score','correct_critical_facts'),'original scored task')
        if type(pair) is not tuple or len(pair) != 2: raise ValueError('original arm and parent receipt required')
        arm,receipt = pair
        if type(arm) is not str or len(arm) != 1 or arm not in 'ABC': raise ValueError('original ABC arm required')
        tid = row['task_id']
        keys(receipt,('schema_version','kind','execution_sha256','task_id','input_sha256','result_sha256',
            'received_ns','failed_observed_ns','status','failure_reason','answer_sha256','latency_ns'),'parent task receipt')
        if type(receipt['schema_version']) is not int or receipt['schema_version'] != 1 or receipt['kind'] != 'recording_parent_task_receipt':
            raise ValueError('original parent receipt schema required')
        sha(receipt['execution_sha256']); sha(receipt['input_sha256'])
        for key in ('result_sha256','answer_sha256'):
            if receipt[key] is not None: sha(receipt[key])
        for key in ('received_ns','failed_observed_ns'):
            if receipt[key] is not None: integer(receipt[key])
        if receipt['status'] not in ('answer','late_answer','failed','not_executed'):
            raise ValueError('original receipt status required')
        if receipt['failure_reason'] not in ('completed','not_executed','snapshot_failed','oversize','busy','stale',
                'admission_refused','timeout','cancelled','backend_failed','source_failed','clock_discontinuity',
                'transport_failed','writer_failed','no_receipt','child_failed','protocol_failed'):
            raise ValueError('finite original child/parent failure reason required')
        if receipt['status'] == 'answer' and (receipt['answer_sha256'] is None or receipt['result_sha256'] is None or
                receipt['received_ns'] is None or receipt['failure_reason'] != 'completed'):
            raise ValueError('accepted original receipt must retain its actual result/answer/time')
        if type(tid) is not str or tid not in taskmap or (tid,arm) in paired: raise ValueError('unknown/duplicate original task/arm')
        if row['input_sha256'] != taskmap[tid]['input_sha256'] or receipt['task_id'] != tid or receipt['input_sha256'] != row['input_sha256']:
            raise ValueError('assessment changed original task/input')
        if row['parent_receipt_sha256'] != digest(receipt): raise ValueError('assessment parent receipt binding differs')
        invocation = integer(row['invocation_ordinal'],maximum=95)
        identity = (invocation,arm,receipt['execution_sha256'])
        receipt_groups.setdefault(identity,[]).append(dict(receipt))
        latency = integer(receipt['latency_ns'],1)
        quality = row['quality_score']
        if type(quality) not in (int,float) or not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError('bounded finite original quality score required')
        facts = row['correct_critical_facts']
        if type(facts) is not list or len(facts) > 128 or any(type(f) is not str for f in facts) or len(set(facts)) != len(facts) or set(facts)-set(taskmap[tid]['critical_facts']):
            raise ValueError('unique original critical facts required')
        success = receipt['status'] == 'answer'
        if (success and row['answer_sha256'] != receipt['answer_sha256']) or (not success and (quality != 0 or facts or row['answer_sha256'] is not None)):
            raise ValueError('late/missing/failed task must retain zero quality and no accepted answer')
        paired[tid,arm] = (Decimal(latency)/Decimal(10**6),Decimal(str(quality)),frozenset(facts))
    common = plan['comparison']; repetitions = common['bootstrap_repetitions']; n = len(tasks)
    if repetitions*n > 1000000: raise ValueError('original task bootstrap exceeds processing bound')
    result = {}
    for candidate in 'BC':
        pairs = [(paired[t['id'],'A'],paired[t['id'],candidate]) for t in tasks]
        def metrics(sample):
            baseline = [b for b,c in sample]; selected = [c for b,c in sample]
            latency_ratio = _percentile([r[0] for r in selected],.95)/_percentile([r[0] for r in baseline],.95)
            quality_delta = sum(c[1]-b[1] for b,c in sample)/len(sample)
            return latency_ratio,quality_delta
        point = metrics(pairs); rng = random.Random(common['bootstrap_seed'])
        draws = [metrics([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(repetitions)]
        intervals = [(_percentile([d[k] for d in draws],.025),_percentile([d[k] for d in draws],.975)) for k in range(2)]
        missing = [dict(task_id=t['id'],fact=f) for t in tasks for f in t['critical_facts'] if f not in paired[t['id'],candidate][2]]
        gates = dict(measured_held_out=common['evidence'] == 'measured' and common['held_out'],
            sample_count=n >= common['min_pairs'],critical_facts=bool(sum(len(t['critical_facts']) for t in tasks)) and not missing,
            latency=point[0] <= Decimal('.9') and intervals[0][1] <= Decimal('.9'),
            quality=point[1] >= 0 and intervals[1][0] >= 0)
        result[candidate+'_vs_A'] = dict(exact_point=dict(latency_ratio=str(point[0]),quality_delta=str(point[1])),
            exact_paired_bootstrap_95=dict(latency_ratio=list(map(str,intervals[0])),quality_delta=list(map(str,intervals[1]))),
            gates=gates,meets_quality_latency_gates=all(gates.values()),missing_critical_facts=missing)
    return ValidatedQualityLatency(plan_digest(plan),expected_assessment_sha256,result,
        runtime_policy_sha256=assessment['runtime_policy_sha256'],collection_sha256=assessment['collection_sha256'],
        execution_inventory_sha256=assessment['execution_inventory_sha256'],
        execution_receipts=tuple(sorted((*identity,digest(sorted(group,key=lambda r:r['task_id'])))
            for identity,group in receipt_groups.items())),_issuer=_QUALITY_ISSUER)


def reconcile_exposure(plan,exposure,recordings,ledger,quality_latency,*,runtime_policy_sha256,
        collection_sha256,scope_protocol_sha256,instrumentation_protocol_sha256,
        execution_inventory_sha256,allocation_sha256=None):
    """J2 exact output over complete immutable typed lifetime/assessment joins."""
    if type(ledger) is not ValidatedLifetimeLedger or type(quality_latency) is not ValidatedQualityLatency:
        raise TypeError('typed validated full lifetime and original task reports required')
    if (quality_latency.comparison_plan_sha256 != plan_digest(plan) or ledger.exposure_sha256 != digest(exposure) or
            ledger.policy_sha256 != exposure['accounting_policy_sha256'] or
            exposure['comparison_plan_sha256'] != plan_digest(plan) or exposure['recordings_sha256'] != digest(recordings)):
        raise ValueError('reconciliation original policy/assessment bindings differ')
    if allocation_sha256 is not None: raise ValueError('allocation adapter not yet reviewed for lifetime2; cannot accept an old allocation')
    for value in (runtime_policy_sha256,collection_sha256,scope_protocol_sha256,
                  instrumentation_protocol_sha256,execution_inventory_sha256): sha(value)
    for name,value in (('runtime_policy_sha256',runtime_policy_sha256),('collection_sha256',collection_sha256),
            ('scope_protocol_sha256',scope_protocol_sha256),('instrumentation_protocol_sha256',instrumentation_protocol_sha256),
            ('execution_inventory_sha256',execution_inventory_sha256)):
        if getattr(ledger,name) != value: raise ValueError('reconciliation crosses original validated ledger '+name)
    expected_receipts = tuple(row for row in ledger.execution_receipts if row[3] != digest([]))
    if quality_latency.execution_receipts != expected_receipts:
        raise ValueError('quality receipts cross original validated execution/parent receipt inventory')
    if (quality_latency.runtime_policy_sha256 != runtime_policy_sha256 or
            quality_latency.collection_sha256 != collection_sha256 or
            quality_latency.execution_inventory_sha256 != execution_inventory_sha256):
        raise ValueError('quality/latency assessment crosses the original validated runtime collection')
    from .recording_financial import decode
    quality = decode(quality_latency.reports,16*1024**2)
    by_group = {}; original = {g['recording_id']:g for g in recordings['groups']}
    for total in ledger.totals:
        run = total.run
        if run.recording_id not in original or run.arm in by_group.setdefault(run.recording_id,{}): raise ValueError('duplicate/unknown original recording total')
        by_group[run.recording_id][run.arm] = total
    if set(by_group) != set(original) or any(set(arms) != set('ABC') for arms in by_group.values()):
        raise ValueError('all original canonical recording triples required')
    arm_values = {rid:{arm:dict(full_monetary_atomic=t.full_monetary_atomic,
        source_comparable=t.run.source_comparable,cost_complete=t.full_monetary_atomic is not None and not t.run.violations) for arm,t in arms.items()}
        for rid,arms in by_group.items()}
    compared = compare_costs(exposure,recordings,arm_values)
    groups = []; counts = dict(source_incomplete=0,cost_incomplete=0,invalidated=0,quality_latency_failed=0,runtime_policy_failed=0)
    for group in recordings['groups']:
        values = by_group[group['recording_id']]; arms = []
        for arm in 'ABC':
            total = values[arm]; run = total.run
            if run.source_manifest_sha256 != group['source_manifest_sha256'] or total.exposure_duration_ns != group['exposure_duration_ns']:
                raise ValueError('reconciliation source/exposure changed')
            counts['source_incomplete'] += int(not run.source_comparable)
            counts['cost_incomplete'] += int(total.full_monetary_atomic is None)
            counts['runtime_policy_failed'] += int(bool(run.violations))
            arms.append(dict(arm=arm,execution_sha256=run.execution_sha256,run_inventory_sha256=run.run_inventory_sha256,
                source_coverage_sha256=run.source_coverage_sha256,lifetime_sha256=total.lifetime_sha256,
                actual_elapsed_ns=total.actual_elapsed_ns,known_monetary_atomic=total.known_monetary_atomic,
                full_monetary_atomic=total.full_monetary_atomic,closure_sha256=total.closure_sha256,
                task_row_inventory_sha256=run.task_row_inventory_sha256))
        groups.append(dict(recording_id=group['recording_id'],source_manifest_sha256=group['source_manifest_sha256'],
            exposure_duration_ns=group['exposure_duration_ns'],arms=arms,
            source_comparable=all(t.run.source_comparable for t in values.values()),
            cost_complete=all(t.full_monetary_atomic is not None for t in values.values())))
    invalidated = ledger.validity['state'] == 'invalidated'; counts['invalidated'] = int(invalidated)
    for name,comparison in compared.items():
        qpass = quality[name]['meets_quality_latency_gates']; counts['quality_latency_failed'] += int(not qpass)
        comparison.update(quality_latency_report_sha256=quality_latency.report_sha256,
            overall_gate=bool(comparison['cost_gate'] and qpass and not invalidated))
        if invalidated: comparison['cost_gate'] = False
    totals = {}
    for arm in 'ABC':
        selected = [arms[arm] for arms in by_group.values()]
        known = integer(sum(t.known_monetary_atomic for t in selected)); duration = integer(sum(t.exposure_duration_ns for t in selected),1)
        full = integer(sum(t.full_monetary_atomic for t in selected)) if all(t.full_monetary_atomic is not None for t in selected) else None
        totals[arm] = dict(known_monetary_atomic=known,full_monetary_atomic=full,exposure_duration_ns=duration,
            full_cost_per_exposure_hour=None if full is None else ratio(fraction(Fraction(full*3600*10**9,duration))))
    state = 'invalidated' if invalidated else 'incomplete' if counts['source_incomplete'] or counts['cost_incomplete'] or counts['runtime_policy_failed'] else 'complete'
    result = dict(schema_version=2,kind='recording_reconciliation',comparison_plan_sha256=plan_digest(plan),
        exposure_policy_sha256=ledger.exposure_sha256,accounting_policy_sha256=ledger.policy_sha256,
        runtime_policy_sha256=runtime_policy_sha256,collection_sha256=collection_sha256,
        scope_protocol_sha256=scope_protocol_sha256,instrumentation_protocol_sha256=instrumentation_protocol_sha256,
        execution_inventory_sha256=execution_inventory_sha256,assessment_sha256=quality_latency.assessment_sha256,
        ledger_sha256=ledger.ledger_sha256,allocation_sha256=None,
        validity_tail_sha256=ledger.validity['validity_tail_sha256'],validity_checkpoint_sha256=ledger.validity['validity_checkpoint_sha256'],
        valid_as_of_ns=ledger.validity['valid_as_of_ns'],groups=groups,canonical_arm_totals=totals,comparison=compared,
        state=state,evidence_class=ledger.evidence_class,
        truth_flags=dict(structural_integrity_verified=True,measurement_truth_verified=False,receipt_authenticity_verified=False,
            isolation_verified=False,worker_backend_coverage_verified=False,actual_model_quality_verified=False),incomplete_counts=counts)
    encode(result,16*1024**2,newline=True)
    return result
