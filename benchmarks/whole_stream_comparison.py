"""Exact matched-recording whole-stream cost gate; no measurement certification."""
from fractions import Fraction
import random

from .paired_replay import _encoded, _keys
from .perception_comparison import _digest, plan_digest
from .stream_cost_ledger import evaluate, _fraction, _ratio, MAX_REPORT_BYTES
from .three_arm_comparison import _snapshot, validate_three_arm_plan

MAX_BOOTSTRAP_DRAWS = 1_000_000


def compare_whole_stream(plan, stream_protocol, ledger, ledger_report, recording_groups, policy):
    """Require original full ledger totals and bootstrap recordings, never tasks.

    The caller binds these declarations to external frozen digests. This helper
    independently recomputes the ledger report rather than trusting supplied totals.
    Comparability failures produce a failed gate, not a selective subset of groups.
    """
    plan = validate_three_arm_plan(plan, expected_plan_sha256=plan_digest(plan))
    stream_protocol, ledger = _snapshot(stream_protocol), _snapshot(ledger)
    ledger_report, recording_groups, policy = map(_snapshot, (ledger_report, recording_groups, policy))
    recomputed = evaluate(stream_protocol, ledger, expected_protocol_sha256=plan_digest(stream_protocol),
                         expected_ledger_sha256=plan_digest(ledger))
    if ledger_report != recomputed:
        raise ValueError('whole-stream gate requires recomputed original ledger report')
    if stream_protocol['comparison_plan_sha256'] != plan_digest(plan):
        raise ValueError('whole-stream comparison plan binding mismatch')
    _keys(policy, ('version', 'threshold', 'interval', 'bootstrap_seed',
                   'bootstrap_repetitions', 'min_recordings'), 'whole-stream policy')
    _keys(policy['threshold'], ('numerator', 'denominator'), 'whole-stream threshold')
    if (policy['version'] != 'matched_recording_elapsed_hour_v1' or
            policy['interval'] != 'nearest_rank_95' or
            any(type(policy['threshold'][k]) is not int for k in ('numerator', 'denominator')) or
            policy['threshold'] != dict(numerator=4, denominator=5)):
        raise ValueError('frozen whole-stream 20 percent gate and nearest-rank interval required')
    for key, original in (('bootstrap_seed', 'bootstrap_seed'),
                          ('bootstrap_repetitions', 'bootstrap_repetitions'), ('min_recordings', 'min_pairs')):
        if type(policy[key]) is not int or policy[key] != plan['comparison'][original]:
            raise ValueError('whole-stream bootstrap policy differs from frozen comparison')
    if type(recording_groups) is not list or not 1 <= len(recording_groups) <= 333:
        raise ValueError('bounded nonempty matched recording groups required')
    n, repetitions = len(recording_groups), policy['bootstrap_repetitions']
    if repetitions*n > MAX_BOOTSTRAP_DRAWS:
        raise ValueError('whole-stream bootstrap processing exceeds one million recording draws')
    frozen = {s['id']: s for s in stream_protocol['streams']}
    observations = {s['id']: s for s in ledger['streams']}
    streams = ledger_report['streams']
    seen_streams, seen_recordings, reasons, groups = set(), set(), [], []

    def reject(index, reason):
        reasons.append(dict(recording_index=index, reason=reason))

    for index, group in enumerate(recording_groups):
        _keys(group, ('recording_sha256', 'streams'), 'matched recording group')
        recording = _digest(group['recording_sha256'])
        _keys(group['streams'], ('A', 'B', 'C'), 'matched recording streams')
        if recording in seen_recordings:
            raise ValueError('recording may appear in only one matched group')
        seen_recordings.add(recording)
        ids = group['streams']
        for arm, sid in ids.items():
            if type(sid) is not str or sid not in frozen or sid in seen_streams:
                raise ValueError('each original stream required exactly once')
            seen_streams.add(sid)
            if frozen[sid]['arm'] != arm or frozen[sid]['recording_sha256'] != recording:
                raise ValueError('matched stream arm/recording binding mismatch')
        source_contracts, partitions, durations, counts = [], [], [], []
        for arm in 'ABC':
            sid = ids[arm]; declared, observed, report = frozen[sid], observations[sid], streams[sid]
            source_contracts.append({s['id']: (s['semantics'], tuple(sorted(s['required_counters'])))
                                     for s in declared['sources']})
            start = observed['start_ns']
            partitions.append([(a['start_ns']-start, a['end_ns']-start, a['state'])
                               for a in observed['availability']])
            durations.append((declared['requested_duration_ns'], report['elapsed_ns']))
            capture = {source: tuple(counter[c] for c in ('captured', 'attempted', 'dropped'))
                       for source, counter in report['counters'].items()}
            counts.append(capture)
            if any(value is None for values in capture.values() for value in values):
                reject(index, 'unknown captured/attempted/dropped opportunity counter: '+arm)
            if not report['complete_accounting']:
                reject(index, 'incomplete whole-stream accounting: '+arm)
            if any(c['kind'] == 'proxy' and declared['requirements'][c['id']]['state'] == 'required'
                   for c in stream_protocol['components']):
                reject(index, 'required proxy cannot establish monetary whole-stream cost: '+arm)
        for values, reason in ((source_contracts, 'capture source identities/semantics/contracts differ'),
                               (partitions, 'relative availability partitions differ'),
                               (durations, 'requested/observed elapsed opportunities differ'),
                               (counts, 'capture opportunity counters differ')):
            if values[1:] != values[:1]*2:
                reject(index, reason)
        groups.append({arm: (streams[ids[arm]]['known_monetary_atomic'], streams[ids[arm]]['elapsed_ns'])
                       for arm in 'ABC'})
    if seen_streams != set(frozen):
        raise ValueError('matched recording groups must cover every original stream')

    totals = {arm: (sum(g[arm][0] for g in groups), sum(g[arm][1] for g in groups)) for arm in 'ABC'}
    def ratio(values, arm):
        base, candidate = values['A'], values[arm]
        return None if base[0] == 0 else _fraction(Fraction(candidate[0]*base[1], candidate[1]*base[0]))

    comparable = not reasons
    enough = n >= policy['min_recordings']
    points = {arm: ratio(totals, arm) for arm in 'BC'}
    samples = {arm: [] for arm in 'BC'}
    undefined = 0
    rng = random.Random(policy['bootstrap_seed'])
    # Stable group order makes a reordered declaration produce the same resamples.
    groups = [g for _, g in sorted(zip((r['recording_sha256'] for r in recording_groups), groups))]
    for _ in range(repetitions):
        draw = [groups[rng.randrange(n)] for _ in range(n)]
        values = {arm: (sum(g[arm][0] for g in draw), sum(g[arm][1] for g in draw)) for arm in 'ABC'}
        if values['A'][0] == 0:
            undefined += 1
            continue
        for arm in 'BC':
            samples[arm].append(ratio(values, arm))
    # Never manufacture an interval from only the defined subset of bootstrap draws.
    lower_index = max(0, (repetitions+39)//40-1)
    upper_index = max(0, (39*repetitions+39)//40-1)
    comparisons = {}
    for arm in 'BC':
        interval = None
        if not undefined:
            ordered = sorted(samples[arm])
            interval = dict(lower=_ratio(ordered[lower_index]), upper=_ratio(ordered[upper_index]))
        passed = (comparable and enough and points[arm] is not None and not undefined and
                  points[arm] <= Fraction(4, 5) and ordered[upper_index] <= Fraction(4, 5))
        comparisons[arm+'_vs_A'] = dict(cost_ratio=None if points[arm] is None else _ratio(points[arm]),
            interval_95=interval, undefined_baseline_draws=undefined,
            meets_whole_stream_cost_gate=bool(passed))
    result = dict(schema_version=1, policy=policy,
        comparison_plan_sha256=plan_digest(plan), stream_protocol_sha256=plan_digest(stream_protocol),
        ledger_sha256=plan_digest(ledger), recording_groups_sha256=plan_digest(recording_groups),
        recording_count=n, minimum_recordings=policy['min_recordings'],
        enough_recordings=enough, comparable_source_opportunities=comparable, comparability_reasons=reasons,
        bootstrap_repetitions=repetitions, bootstrap_seed=policy['bootstrap_seed'],
        nearest_rank_indices=dict(lower=lower_index, upper=upper_index),
        bootstrap_order='recording_sha256 ascending; Python Random.randrange with replacement',
        denominator='pooled observed elapsed hours; matched recording resampling',
        canonical_arms={arm:dict(monetary_atomic=totals[arm][0], elapsed_ns=totals[arm][1],
            known_cost_per_elapsed_hour=_ratio(Fraction(totals[arm][0]*3600*10**9, totals[arm][1]))) for arm in 'ABC'},
        comparisons=comparisons,
        primary_meets_whole_stream_cost_gate=comparisons['C_vs_A']['meets_whole_stream_cost_gate'],
        recording_independence_verified=False, receipt_authenticity_verified=False,
        measurement_truth_verified=False, whole_pipeline_cost_verified=False, actual_acceptance='incomplete')
    if len(_encoded(result))+1 > MAX_REPORT_BYTES:
        raise ValueError('whole-stream comparison report exceeds 16 MiB')
    return result
