"""Frozen within-stream attribution; conservation does not certify actual savings."""
import argparse
from decimal import Decimal
from fractions import Fraction
from math import gcd
from pathlib import Path

from .paired_replay import _encoded, _keys, _write_new
from .perception_comparison import _digest, plan_digest
from .three_arm_comparison import _read, _snapshot, validate_three_arm_plan
from .stream_cost_ledger import (evaluate, _fraction, _identifier, _int, _list, _ratio,
                                 MAX_REPORT_BYTES)

MAX_GRAPH = 10000
MAX_WORK = 100000
BASE_RULE = ('stream_id', 'component_id', 'resource_id', 'lane_id', 'method')


class _Budget:
    def __init__(self):
        self.work = 0
        self.edges = 0
        self.bytes = 0

    def spend(self, count):
        if self.work + count > MAX_WORK:
            raise ValueError('allocation processing budget exceeds 100000 steps')
        self.work += count

    def edge(self, count):
        if self.edges + count > MAX_WORK:
            raise ValueError('allocation edge budget exceeds 100000')
        self.edges += count
        self.spend(count)

    def output(self, value):
        self.bytes += len(_encoded(value))
        if self.bytes + 1 > MAX_REPORT_BYTES:
            raise ValueError('allocation output exceeds 16 MiB')


def _key(rule):
    return tuple(rule[k] for k in BASE_RULE[:-1])


def _protocol(plan, stream_protocol, protocol, expected_sha):
    if plan_digest(protocol) != _digest(expected_sha):
        raise ValueError('allocation protocol differs from frozen digest')
    _keys(protocol, ('schema_version', 'comparison_plan_sha256', 'stream_protocol_sha256',
        'rounding', 'tie_break', 'duration_model', 'targets', 'rules'), 'allocation protocol')
    if (type(protocol['schema_version']) is not int or protocol['schema_version'] != 1 or
            protocol['comparison_plan_sha256'] != plan_digest(plan) or
            protocol['stream_protocol_sha256'] != plan_digest(stream_protocol) or
            protocol['rounding'] != 'largest_remainder_v1' or
            protocol['tie_break'] != 'unicode_codepoint_target_id' or
            protocol['duration_model'] != 'uniform_line_quantity_over_interval'):
        raise ValueError('frozen allocation version/bindings/rounding contract required')
    streams = {s['id']: s for s in stream_protocol['streams']}
    components = {c['id']: c for c in stream_protocol['components']}
    tasks = {t['id'] for t in plan['comparison']['tasks']}
    targets, task_arms, buckets = {}, set(), set()
    windows = 0
    for target in _list(protocol['targets'], 'allocation targets', MAX_GRAPH):
        _keys(target, ('id', 'stream_id', 'kind', 'task_id', 'windows'), 'target')
        tid = _identifier(target['id'], 'target ID'); sid = target['stream_id']
        if tid in targets or type(sid) is not str or sid not in streams:
            raise ValueError('unique target and known original stream required')
        kind = target['kind']
        if kind not in ('task', 'startup', 'idle', 'overhead'):
            raise ValueError('explicit allocation target kind required')
        intervals = _list(target['windows'], 'target windows', MAX_GRAPH)
        windows += len(intervals)
        if windows > MAX_GRAPH:
            raise ValueError('total frozen windows exceed 10000')
        if kind == 'task':
            task = target['task_id']
            if type(task) is not str or task not in tasks or (task, streams[sid]['arm']) in task_arms:
                raise ValueError('exactly one original task/arm target required')
            task_arms.add((task, streams[sid]['arm']))
        else:
            if target['task_id'] is not None or intervals or (sid, kind) in buckets:
                raise ValueError('one explicit empty-window startup/idle/overhead bucket per stream required')
            buckets.add((sid, kind))
        previous = 0
        for window in intervals:
            _keys(window, ('start_offset_ns', 'end_offset_ns'), 'frozen window')
            start = _int(window['start_offset_ns'], 'window start')
            end = _int(window['end_offset_ns'], 'window end', 1)
            if not previous <= start < end <= streams[sid]['requested_duration_ns']:
                raise ValueError('sorted non-overlapping bounded frozen windows required')
            previous = end
        targets[tid] = target
    if task_arms != {(task, arm) for task in tasks for arm in 'ABC'}:
        raise ValueError('include every original task and arm, even when its attributed cost is zero')
    if buckets != {(sid, kind) for sid in streams for kind in ('startup', 'idle', 'overhead')}:
        raise ValueError('explicit startup/idle/overhead buckets required for all streams')
    expected = {(sid, cid, lane['resource_id'], lane['lane_id'])
        for sid, stream in streams.items() for cid, requirement in stream['requirements'].items()
        for lane in requirement['lanes']}
    rules = {}
    references = 0
    def same_stream(tid, sid):
        if type(tid) is not str or tid not in targets or targets[tid]['stream_id'] != sid:
            raise ValueError('target must belong to original source stream; cross-arm reuse unsupported')
    for rule in _list(protocol['rules'], 'allocation rules', MAX_GRAPH):
        if type(rule) is not dict or rule.get('method') not in ('direct','fixed','duration'):
            raise ValueError('explicit supported allocation method required')
        method = rule['method']
        extra = ('target_id',) if method == 'direct' else ('shares',) if method == 'fixed' else ('task_target_ids','idle_target_id')
        _keys(rule, BASE_RULE+extra, 'allocation rule')
        for field in BASE_RULE[:-1]: _identifier(rule[field], field)
        key, sid = _key(rule), rule['stream_id']
        if key not in expected or key in rules:
            raise ValueError('exactly one rule per original required resource lane')
        references += (1 if method == 'direct' else len(_list(rule['shares'], 'fixed shares', MAX_GRAPH))
                       if method == 'fixed' else len(_list(rule['task_target_ids'], 'duration task roster', MAX_GRAPH))+1)
        if references > MAX_WORK:
            raise ValueError('frozen rule references exceed 100000')
        if method == 'direct':
            same_stream(rule['target_id'], sid)
        elif method == 'fixed':
            seen, total = set(), Fraction(0)
            for share in _list(rule['shares'], 'fixed shares', MAX_GRAPH):
                _keys(share, ('target_id','numerator','denominator'), 'fixed share')
                tid = share['target_id']; same_stream(tid, sid)
                n = _int(share['numerator'], 'share numerator'); d = _int(share['denominator'], 'share denominator', 1)
                if tid in seen or gcd(n,d) != 1:
                    raise ValueError('unique reduced exact fixed shares required')
                seen.add(tid); total = _fraction(total+Fraction(n,d))
            if total != 1: raise ValueError('fixed shares must conserve exactly one')
        else:
            if components[rule['component_id']]['coverage'] != 'interval':
                raise ValueError('duration allocation requires interval source, not startup/event')
            roster = _list(rule['task_target_ids'], 'duration task roster', MAX_GRAPH)
            seen = set()
            for tid in roster:
                same_stream(tid, sid)
                if tid in seen or targets[tid]['kind'] != 'task' or not targets[tid]['windows']:
                    raise ValueError('unique frozen task-window roster required')
                seen.add(tid)
            if not seen: raise ValueError('nonempty duration task roster required')
            same_stream(rule['idle_target_id'], sid)
            if targets[rule['idle_target_id']]['kind'] not in ('idle','overhead'):
                raise ValueError('duration gaps require explicit idle/overhead target')
        rules[key] = rule
    if set(rules) != expected:
        raise ValueError('allocation rule missing for required lane')
    return streams, components, targets, rules


def _weights(entry, rule, targets, observed_start, budget):
    budget.spend(1)
    if rule['method'] == 'direct':
        weights = {rule['target_id']: Fraction(1)}
    elif rule['method'] == 'fixed':
        budget.spend(len(rule['shares']))
        weights = {s['target_id']: Fraction(s['numerator'],s['denominator']) for s in rule['shares']}
    else:
        start, end = entry['start_ns'], entry['end_ns']
        events = {start: [], end: []}
        for tid in rule['task_target_ids']:
            budget.spend(len(targets[tid]['windows']))
            for window in targets[tid]['windows']:
                a = max(start, observed_start+window['start_offset_ns'])
                b = min(end, observed_start+window['end_offset_ns'])
                if a < b:
                    events.setdefault(a, []).append((tid, 1))
                    events.setdefault(b, []).append((tid, -1))
        weights, active = {}, set()
        budget.spend(len(events))
        points = sorted(events)
        for index, point in enumerate(points[:-1]):
            # Apply all coincident boundaries before the next half-open segment.
            for tid, direction in events[point]:
                if direction < 0: active.discard(tid)
            for tid, direction in events[point]:
                if direction > 0: active.add(tid)
            budget.spend(len(active) if active else 1)
            recipients = sorted(active) if active else [rule['idle_target_id']]
            amount = _fraction(Fraction(points[index+1]-point, (end-start)*len(recipients)))
            for tid in recipients:
                weights[tid] = _fraction(weights.get(tid, Fraction(0))+amount)
    weights = {tid:value for tid,value in weights.items() if value > 0}
    weight_sum = Fraction(0)
    for weight in weights.values():
        weight_sum = _fraction(weight_sum+weight)
    if weight_sum != 1:
        raise ValueError('computed source weights do not conserve one')
    budget.edge(len(weights))
    return weights


def _largest(total, weights):
    """Python string ordering is Unicode code-point order; IDs stay unchanged."""
    exact = {tid:_fraction(total*weight) for tid,weight in weights.items()}
    result = {tid:value.numerator//value.denominator for tid,value in exact.items()}
    remaining = total-sum(result.values())
    if not 0 <= remaining < len(weights):
        raise ValueError('invalid bounded largest-remainder residual')
    order = sorted(weights, key=lambda tid: (-(exact[tid]-result[tid]), tid))
    for tid in order[:remaining]: result[tid] += 1
    if sum(result.values()) != total:
        raise ValueError('allocated quantity conservation failed')
    return result


def allocate(comparison_plan, stream_protocol, ledger, allocation_protocol, *,
             expected_comparison_plan_sha256, expected_stream_protocol_sha256,
             expected_ledger_sha256, expected_allocation_protocol_sha256):
    plan = validate_three_arm_plan(comparison_plan, expected_plan_sha256=expected_comparison_plan_sha256)
    stream_protocol, ledger, protocol = _snapshot(stream_protocol), _snapshot(ledger), _snapshot(allocation_protocol)
    source_report = evaluate(stream_protocol, ledger, expected_protocol_sha256=expected_stream_protocol_sha256,
                             expected_ledger_sha256=expected_ledger_sha256)
    if stream_protocol['comparison_plan_sha256'] != expected_comparison_plan_sha256:
        raise ValueError('stream protocol belongs to another original comparison plan')
    streams, components, targets, rules = _protocol(plan, stream_protocol, protocol, expected_allocation_protocol_sha256)
    budget = _Budget()
    target_ids, component_ids, rule_keys = sorted(targets), sorted(components), sorted(rules)
    target_index = {tid:i for i,tid in enumerate(target_ids)}
    component_index = {cid:i for i,cid in enumerate(component_ids)}
    rule_index = {key:i for i,key in enumerate(rule_keys)}
    target_table = [{k:targets[tid][k] for k in ('id','stream_id','kind','task_id')} for tid in target_ids]
    component_table = [components[cid] for cid in component_ids]
    rule_table = [dict(rule_sha256=plan_digest(rules[key]), **{k:rules[key][k] for k in BASE_RULE}) for key in rule_keys]
    for table in (target_table,component_table,rule_table): budget.output(table)
    observed = {s['id']:s for s in ledger['streams']}
    totals = [dict(target_index=i, known_quantity_by_component={}, known_billed_atomic=0,
        known_rate_derived_atomic=Fraction(0), unknown_source_indices=[],
        source_stream_incomplete=not source_report['streams'][targets[tid]['stream_id']]['complete_accounting'])
        for i,tid in enumerate(target_ids)]
    sources = []
    for ordinal, entry in enumerate(ledger['entries']):
        key = tuple(entry[k] for k in BASE_RULE[:-1]); rule = rules[key]
        weights = _weights(entry, rule, targets, observed[entry['stream_id']]['start_ns'], budget)
        known = entry['status'] in ('settled','measured')
        component = components[entry['component_id']]
        monetary = component['kind'] == 'monetary'
        quantity = _largest(entry['quantity'], weights) if known else None
        billed = _largest(entry['billed_amount'], weights) if known and monetary else None
        record = dict(ordinal=ordinal, entry_sha256=plan_digest(entry), identity_sha256=plan_digest(entry['identity']),
            rule_index=rule_index[key], component_index=component_index[entry['component_id']],
            status=entry['status'], shares=[], allocations=[] if known else None)
        for tid in sorted(weights):
            index = target_index[tid]; total = totals[index]
            record['shares'].append(dict(target_index=index, **_ratio(weights[tid])))
            if not known:
                total['unknown_source_indices'].append(ordinal)
                continue
            q = quantity[tid]; cid = str(component_index[entry['component_id']])
            total['known_quantity_by_component'][cid] = total['known_quantity_by_component'].get(cid,0)+q
            rate_cost = (_fraction(q*Fraction(Decimal(stream_protocol['rates'][component['rate_id']]['per_unit'])))
                         if monetary else None)
            money = billed[tid] if monetary else None
            if monetary:
                total['known_billed_atomic'] += money
                total['known_rate_derived_atomic'] = _fraction(total['known_rate_derived_atomic']+rate_cost)
            record['allocations'].append(dict(target_index=index, quantity=q, billed_amount=money,
                rate_derived_atomic=None if rate_cost is None else _ratio(rate_cost),
                billed_minus_rate_derived_atomic=None if rate_cost is None else _ratio(money-rate_cost)))
        budget.output(record)
        sources.append(record)
    stream_totals = {sid:dict(known_quantity_by_component={}, known_billed_atomic=0) for sid in streams}
    for index, total in enumerate(totals):
        sid = targets[target_ids[index]]['stream_id']; stream_total = stream_totals[sid]
        stream_total['known_billed_atomic'] += total['known_billed_atomic']
        for cid,q in total['known_quantity_by_component'].items():
            stream_total['known_quantity_by_component'][cid] = stream_total['known_quantity_by_component'].get(cid,0)+q
        total['known_rate_derived_atomic'] = _ratio(total['known_rate_derived_atomic'])
        complete = not total['source_stream_incomplete'] and not total['unknown_source_indices']
        total['declared_complete_cost_atomic'] = total['known_billed_atomic'] if complete else None
    for sid, total in stream_totals.items():
        original = source_report['streams'][sid]
        expected = {str(component_index[cid]):value['known_quantity'] for cid,value in original['component_totals'].items()
                    if value['known_quantity'] or str(component_index[cid]) in total['known_quantity_by_component']}
        if total['known_quantity_by_component'] != expected or total['known_billed_atomic'] != original['known_monetary_atomic']:
            raise ValueError('target/stream source-total conservation failed')
    result = dict(schema_version=1, comparison_plan_sha256=expected_comparison_plan_sha256,
        stream_protocol_sha256=expected_stream_protocol_sha256, ledger_sha256=expected_ledger_sha256,
        allocation_protocol_sha256=expected_allocation_protocol_sha256, source_ledger_report_sha256=plan_digest(source_report),
        currency=stream_protocol['currency'], currency_unit=stream_protocol['currency_unit'],
        rate_revision=stream_protocol['rate_revision'],
        targets=target_table, components=component_table, rules=rule_table, sources=sources, target_totals=totals,
        stream_totals=stream_totals, canonical_arms=source_report['canonical_arms'],
        source_ledger_declared_complete=source_report['declared_accounting_complete'],
        declared_allocation_complete=source_report['declared_accounting_complete'],
        known_line_conservation_verified=True, processing_steps=budget.work, allocation_edges=budget.edges,
        receipt_authenticity_verified=False, measurement_truth_verified=False, isolation_verified=False,
        instrumentation_coverage_verified=False, whole_pipeline_cost_verified=False,
        numeric_integration_implemented=False, actual_acceptance='incomplete',
        certification='Exact attribution of declared known quantities and billed amounts only. Unknown source lines '
        'retain exposure shares and null allocations. No allocated amount is a new invoice. No real savings, '
        'independent measurement, receipt authenticity, cache isolation, source coverage or numeric integration is certified.')
    if len(_encoded(result))+1 > MAX_REPORT_BYTES:
        raise ValueError('serialized allocation report exceeds 16 MiB')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','stream-protocol','ledger','allocation-protocol','plan-sha256',
                 'stream-protocol-sha256','ledger-sha256','allocation-protocol-sha256','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = allocate(_read(args.plan), _read(args.stream_protocol), _read(args.ledger), _read(args.allocation_protocol),
        expected_comparison_plan_sha256=args.plan_sha256, expected_stream_protocol_sha256=args.stream_protocol_sha256,
        expected_ledger_sha256=args.ledger_sha256, expected_allocation_protocol_sha256=args.allocation_protocol_sha256)
    output = Path(args.output); output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_new(output/'report.json', result)
    print(_encoded(result).decode('ascii'))


if __name__ == '__main__':
    main()
