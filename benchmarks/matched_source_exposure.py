"""Versioned original-exposure policy and exact recording bootstrap.

The denominator is original replay opportunity, never a rewritten SDK lifetime.
All complete lifetime costs are retained. Legacy elapsed-hour gates are untouched.
"""
from decimal import Decimal
from fractions import Fraction
import itertools
import random
import re

from gamst.capabilities import identifier
from .recording_financial import (N63, array, digest, enum, fraction, integer, keys,
                                  nullable, ratio, rational, sha, snapshot)
from .stream_cost_ledger import CATEGORIES
from .three_arm_comparison import validate_three_arm_plan
from .perception_comparison import plan_digest

EXPOSURE_LIMITS = dict(max_groups=32,max_arm_runs=96,max_original_tasks=1000,
    max_cost_lines=131072,max_cost_pages=1024,max_financial_metadata_bytes=67108864,
    max_source_document_bytes=67108864,max_output_bytes=16777216,max_total_bytes=4294967296)
SOURCE_POLICY = dict(ownership='owned_recorded_replay_v1',clock='fixed_1x_original_offset',
    coverage='every_supplied_record_acquired',availability='exact_original_known_partitions',
    counter_boundary='verified_acquisition',require_terminal_offset_ns='original_duration',
    physical_device_truth='unverified',downstream_delivery='separate')
FAILURE_POLICY = dict(recording='fail_entire_gate',task='original_zero_quality_timeout_floor',
    rerun='separate_original_run_never_replace',subset='forbidden')
TARGETS = dict(cost_ratio_upper=dict(numerator=4,denominator=5),
    latency_ratio_upper=dict(numerator=9,denominator=10),quality='unchanged_original_P')


def _frozen(value,expected,limit=16*1024**2):
    value = snapshot(value,limit)
    if digest(value) != sha(expected):
        raise ValueError('declaration differs from frozen digest')
    return value


def validate_recording_opportunities(value,*,expected_sha256,comparison_plan_sha256):
    value = _frozen(value,expected_sha256)
    keys(value,('schema_version','kind','comparison_plan_sha256','groups'),'recording opportunities')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['kind'] != 'matched_recording_opportunities' or
            value['comparison_plan_sha256'] != sha(comparison_plan_sha256)):
        raise ValueError('original matched recording opportunities required')
    ids = set(); manifests = set()
    for group in array(value['groups'],32,1):
        keys(group,('recording_id','source_manifest_sha256','tasks_sha256','exposure_duration_ns',
                    'sources','availability_sha256','arm_order'),'original recording group')
        identifier(group['recording_id'])
        if group['recording_id'] in ids or sha(group['source_manifest_sha256']) in manifests:
            raise ValueError('unique original recording identity/digest required')
        ids.add(group['recording_id']); manifests.add(group['source_manifest_sha256'])
        sha(group['tasks_sha256']); sha(group['availability_sha256'])
        integer(group['exposure_duration_ns'],1,3600*10**9)
        if type(group['arm_order']) is not list or tuple(group['arm_order']) not in tuple(itertools.permutations('ABC')):
            raise ValueError('one original ABC permutation required')
        source_ids = set(); sensors = set()
        for source in array(group['sources'],16,1):
            keys(source,('source_id','sensor_id','original_stream_epoch','format','calibration_id',
                         'record_count','record_inventory_sha256','counter_contract'),'original source')
            for key in ('source_id','sensor_id','original_stream_epoch'):
                identifier(source[key])
            nullable(source['calibration_id'],identifier)
            if source['source_id'] in source_ids or (source['source_id'],source['sensor_id']) in sensors:
                raise ValueError('unique original source identity required')
            source_ids.add(source['source_id']); sensors.add((source['source_id'],source['sensor_id']))
            enum(source['format'],('bgr8_hwc','pcm_s16le_mono'))
            integer(source['record_count'],maximum=262144); sha(source['record_inventory_sha256'])
            if source['counter_contract'] != 'attempted_equals_captured_plus_dropped':
                raise ValueError('known original capture counter contract required')
        if not any(s['format'] == 'bgr8_hwc' and s['record_count'] for s in group['sources']):
            raise ValueError('nonempty original primary visual opportunity required')
    return value


def validate_exposure_policy(value,plan,recordings,*,expected_sha256,expected_recordings_sha256):
    plan = validate_three_arm_plan(plan,expected_plan_sha256=plan_digest(plan))
    value = _frozen(value,expected_sha256)
    keys(value,('schema_version','kind','comparison_plan_sha256','recordings_sha256',
                'latency_policy_sha256','accounting_policy_sha256','financial_adapter_inventory_sha256',
                'source_policy','bootstrap','targets','failure_policy','limits'),'exposure policy')
    if (type(value['schema_version']) is not int or value['schema_version'] != 2 or
            value['kind'] != 'matched_source_exposure_full_lifetime' or value['comparison_plan_sha256'] != plan_digest(plan) or
            value['recordings_sha256'] != sha(expected_recordings_sha256)):
        raise ValueError('new upstream exposure2 declaration required')
    for key in ('latency_policy_sha256','accounting_policy_sha256','financial_adapter_inventory_sha256'):
        sha(value[key])
    recordings = validate_recording_opportunities(recordings,expected_sha256=expected_recordings_sha256,
                                                 comparison_plan_sha256=plan_digest(plan))
    policy = value['source_policy']
    keys(policy,tuple(SOURCE_POLICY)+('max_acquisition_lateness_ns',),'source coverage policy')
    if any(policy[k] != v for k,v in SOURCE_POLICY.items()):
        raise ValueError('only exact owned original replay acquisition is supported')
    integer(policy['max_acquisition_lateness_ns'])
    expected_bootstrap = dict(unit='original_recording',minimum_recordings=plan['comparison']['min_pairs'],
        seed=plan['comparison']['bootstrap_seed'],repetitions=plan['comparison']['bootstrap_repetitions'],
        order='source_manifest_sha256_ascending',interval='nearest_rank_95',undefined_draw='interval_undefined')
    keys(value['bootstrap'],expected_bootstrap,'recording bootstrap')
    if value['bootstrap'] != expected_bootstrap or any(type(value['bootstrap'][k]) is not int for k in ('minimum_recordings','seed','repetitions')):
        raise ValueError('original paired-plan bootstrap/minimum required')
    if value['targets'] != TARGETS or value['failure_policy'] != FAILURE_POLICY or value['limits'] != EXPOSURE_LIMITS:
        raise ValueError('unchanged numeric targets and bounded failure policy required')
    # Exact nested field sets and strict scalar types (True cannot stand for 1).
    keys(value['targets'],TARGETS); rational(value['targets']['cost_ratio_upper']); rational(value['targets']['latency_ratio_upper'])
    keys(value['failure_policy'],FAILURE_POLICY); keys(value['limits'],EXPOSURE_LIMITS)
    for key,limit in EXPOSURE_LIMITS.items():
        if integer(value['limits'][key]) != limit: raise ValueError('fixed joint limit required')
    n = len(recordings['groups'])
    if n < expected_bootstrap['minimum_recordings']:
        raise ValueError('frozen statistical minimum must fit before activation')
    if n*expected_bootstrap['repetitions'] > 1000000:
        raise ValueError('recording bootstrap exceeds one million draws')
    return value


def validate_accounting_policy(value,*,expected_sha256,comparison_plan_sha256,recordings):
    value = _frozen(value,expected_sha256)
    keys(value,('schema_version','kind','comparison_plan_sha256','recordings_sha256',
                'financial_adapter_inventory_sha256','boundary','currency','currency_unit','rate_revision',
                'invoice_rounding','isolation','components','rates','streams','closure','limits'),'lifetime accounting')
    if (type(value['schema_version']) is not int or value['schema_version'] != 2 or
            value['kind'] != 'recording_lifetime_accounting' or
            value['comparison_plan_sha256'] != sha(comparison_plan_sha256) or
            value['recordings_sha256'] != digest(recordings) or
            value['boundary'] != 'full_attributable_arm_lifetime' or
            value['invoice_rounding'] != 'line_atomic_half_even' or value['isolation'] != 'isolated_arms'):
        raise ValueError('new full-lifetime accounting2 declaration required')
    sha(value['financial_adapter_inventory_sha256'])
    for key in ('currency','currency_unit','rate_revision'): identifier(value[key])
    rates = value['rates']
    if type(rates) is not dict or len(rates) > 256: raise ValueError('bounded frozen rate table required')
    for rate_id,rate in rates.items():
        identifier(rate_id); keys(rate,('provider','revision','unit','per_unit'),'rate')
        for key in ('provider','revision','unit'): identifier(rate[key])
        price = rate['per_unit']
        if type(price) is not str or len(price) > 64 or re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9]+)?',price) is None:
            raise ValueError('bounded exact nonnegative decimal rate required')
        fraction(Fraction(Decimal(price)))
    components = {}; categories = set()
    for component in array(value['components'],256,1):
        keys(component,('id','category','kind','unit','source_unit','atomic_per_source','coverage','rate_id'),'component')
        cid = identifier(component['id'])
        if cid in components or component['category'] not in CATEGORIES: raise ValueError('unique required component category required')
        categories.add(component['category'])
        enum(component['kind'],('monetary','proxy')); enum(component['coverage'],('interval','event'))
        identifier(component['unit']); identifier(component['source_unit'])
        conversion = rational(component['atomic_per_source'])
        if conversion <= 0: raise ValueError('positive exact source-unit conversion required')
        # Preserve ledger1's N63 conversion fields.
        integer(component['atomic_per_source']['numerator'],1); integer(component['atomic_per_source']['denominator'],1)
        rate_id = component['rate_id']
        if component['kind'] == 'monetary':
            if type(rate_id) is not str or rate_id not in rates or rates[rate_id]['unit'] != component['unit']:
                raise ValueError('matching frozen monetary rate required')
        elif rate_id is not None: raise ValueError('proxy cannot become a monetary rate')
        components[cid] = component
    if categories != set(CATEGORIES): raise ValueError('all ten cost boundary categories required')
    groups = {g['recording_id']:g for g in recordings['groups']}
    streams = set(); pairs = set(); namespaces = set(); lanes = set(); resource_arms = {}
    for stream in array(value['streams'],96,3):
        keys(stream,('stream_id','recording_id','arm','source_manifest_sha256','exposure_duration_ns',
                     'execution_namespace','cache_namespace','requirements','isolation_policy_sha256'),'lifetime stream')
        sid = identifier(stream['stream_id']); rid = stream['recording_id']; arm = enum(stream['arm'],'ABC')
        if sid in streams or type(rid) is not str or rid not in groups or (rid,arm) in pairs:
            raise ValueError('one immutable original ABC triple required')
        streams.add(sid); pairs.add((rid,arm)); group = groups[rid]
        if stream['source_manifest_sha256'] != group['source_manifest_sha256'] or stream['exposure_duration_ns'] != group['exposure_duration_ns']:
            raise ValueError('original exposure/manifest binding changed')
        integer(stream['exposure_duration_ns'],1); sha(stream['isolation_policy_sha256'])
        for field in ('execution_namespace','cache_namespace'):
            namespace = identifier(stream[field])
            if (field,namespace) in namespaces: raise ValueError('namespace reuse unsupported')
            namespaces.add((field,namespace))
        keys(stream['requirements'],components,'all component requirements')
        for cid,requirement in stream['requirements'].items():
            keys(requirement,('state','reason_code','justification_sha256','lanes'),'component requirement')
            state = enum(requirement['state'],('required','not_applicable')); identifier(requirement['reason_code'])
            nullable(requirement['justification_sha256'],sha)
            declared = array(requirement['lanes'],256)
            if bool(declared) != (state == 'required') or (state == 'not_applicable' and requirement['justification_sha256'] is None):
                raise ValueError('required lanes or independent nonapplicability evidence required')
            for lane in declared:
                keys(lane,('resource_id','lane_id','coverage','lifecycle_binding_id'),'resource lane')
                resource = identifier(lane['resource_id']); lid = identifier(lane['lane_id'])
                if (resource,lid) in lanes: raise ValueError('resource measurement lane counted twice')
                lanes.add((resource,lid))
                if resource in resource_arms and resource_arms[resource] != arm: raise ValueError('cross-arm resource reuse unsupported')
                resource_arms[resource] = arm
                enum(lane['coverage'],('whole_lifetime','observed_lifecycle'))
                nullable(lane['lifecycle_binding_id'],identifier)
                if (lane['lifecycle_binding_id'] is None) != (lane['coverage'] == 'whole_lifetime'):
                    raise ValueError('explicit lifecycle binding required')
    if pairs != {(rid,arm) for rid in groups for arm in 'ABC'}: raise ValueError('every original canonical stream required')
    closure = value['closure']
    expected_closure = dict(admission_end='source_duration_and_original_task_deadlines',cost_boundary_end='all_attributable_work_closed',
        seal='external_finality_or_incomplete',late_invoice='immutable_invalidate_no_rewrite',amendments='unsupported',
        deadline_clock='same_host_monotonic_ns')
    keys(closure,tuple(expected_closure)+('settlement_wait_ms',),'closure policy')
    if any(closure[k] != v for k,v in expected_closure.items()): raise ValueError('authoritative closure and immutable late-cost policy required')
    integer(closure['settlement_wait_ms'],maximum=86400000)
    # K2 limits use the same exact global ledger envelope as E2, avoiding an
    # independently reset capacity hidden behind each stream.
    keys(value['limits'],EXPOSURE_LIMITS,'joint accounting limits')
    for key,limit in EXPOSURE_LIMITS.items():
        if integer(value['limits'][key]) != limit: raise ValueError('fixed joint accounting limit required')
    return value


def compare_costs(exposure_policy,recordings,arm_totals):
    """Pure exact statistic over validated full totals, preserving ALL groups.

    `arm_totals` is a map by original recording_id, then ABC with exact
    {full_monetary_atomic,source_comparable,cost_complete}; it is produced only
    by the lifetime join. This helper deliberately makes no evidence claims.
    """
    keys(arm_totals,{g['recording_id'] for g in recordings['groups']},'all original totals')
    groups = []; complete = True
    for group in sorted(recordings['groups'],key=lambda g:g['source_manifest_sha256']):
        values = arm_totals[group['recording_id']]
        keys(values,'ABC','canonical ABC totals')
        row = {}
        for arm in 'ABC':
            value = values[arm]
            keys(value,('full_monetary_atomic','source_comparable','cost_complete'),'validated arm total')
            if type(value['source_comparable']) is not bool or type(value['cost_complete']) is not bool:
                raise ValueError('explicit source/cost completeness required')
            amount = value['full_monetary_atomic']
            if amount is not None: integer(amount)
            complete &= amount is not None and value['source_comparable'] and value['cost_complete']
            row[arm] = amount
        groups.append(row)
    n = len(groups); bootstrap = exposure_policy['bootstrap']; repetitions = bootstrap['repetitions']
    if not 1 <= n <= 32 or n*repetitions > 1000000: raise ValueError('bounded original recording bootstrap required')
    totals = {arm:sum(g[arm] for g in groups) if all(g[arm] is not None for g in groups) else None for arm in 'ABC'}
    for total in totals.values():
        if total is not None: integer(total)
    points = {arm:None if totals['A'] in (None,0) or totals[arm] is None else fraction(Fraction(totals[arm],totals['A'])) for arm in 'BC'}
    samples = {arm:[] for arm in 'BC'}; undefined = 0
    if all(total is not None for total in totals.values()):
        rng = random.Random(bootstrap['seed'])
        for _ in range(repetitions):
            indices = [rng.randrange(n) for _ in range(n)]
            sums = {arm:sum(groups[i][arm] for i in indices) for arm in 'ABC'}
            if sums['A'] == 0: undefined += 1; continue
            for arm in 'BC': samples[arm].append(fraction(Fraction(sums[arm],sums['A'])))
    else:
        undefined = repetitions
    lower = max(0,(repetitions+39)//40-1); upper = max(0,(39*repetitions+39)//40-1)
    result = {}
    for arm in 'BC':
        interval = None
        if not undefined:
            values = sorted(samples[arm]); interval = [ratio(values[lower]),ratio(values[upper])]
        passed = bool(complete and n >= bootstrap['minimum_recordings'] and points[arm] is not None and
                      interval is not None and points[arm] <= Fraction(4,5) and rational(interval[1]) <= Fraction(4,5))
        result[arm+'_vs_A'] = dict(point_ratio=None if points[arm] is None else ratio(points[arm]),
                                  interval95=interval,cost_gate=passed)
    return result
