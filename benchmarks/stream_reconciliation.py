"""Strict declared stream/attempt reconciliation; never proof of actual savings."""
import argparse
from decimal import Decimal
from fractions import Fraction
from math import gcd
from pathlib import Path

from .paired_replay import _encoded, _keys, _write_new
from .perception_comparison import _digest, _text, plan_digest
from .three_arm_comparison import _read, _snapshot
from .three_arm_evidence import join as join_evidence, read_traces
from .three_arm_attempts import _quantities
from .stream_attempts import _validate_body, REPORT_KEYS
from .stream_cost_ledger import evaluate, _fraction, _identifier, _int, _list, MAX_REPORT_BYTES
from .stream_cost_allocation import allocate, BASE_RULE
from .whole_stream_comparison import compare_whole_stream

MAX_ITEMS = 10000
MAX_REFERENCES = 100000
BINDINGS = ('comparison_plan_sha256', 'collection_sha256', 'stream_protocol_sha256',
            'allocation_protocol_sha256', 'reconciliation_protocol_sha256')
PROTOCOL_KEYS = BINDINGS[:-1] + ('schema_version', 'currency_atomic_per_comparison_unit',
    'rate_map', 'lane_owners', 'provider_counter_groups', 'vlm_counter_bindings',
    'recording_groups', 'whole_stream_cost_policy')
MANIFEST_KEYS = BINDINGS + ('schema_version', 'last_trace_sha256', 'ledger_sha256',
    'assessment_sha256', 'reviewer', 'coverage_evidence_sha256', 'trace_reports',
    'stream_reports', 'attempt_instances', 'line_owners', 'provider_counters')


def _positive_ratio(value, label):
    _keys(value, ('numerator', 'denominator'), label)
    n = _int(value['numerator'], label, 1)
    d = _int(value['denominator'], label, 1)
    if gcd(n, d) != 1:
        raise ValueError('reduced conversion required: '+label)
    return _fraction(Fraction(n, d))


def _unique(items, keys, label):
    result = {}
    for item in _list(items, label, MAX_ITEMS):
        key = tuple(item.get(k) for k in keys) if type(item) is dict else None
        if key is None or any(type(k) is not str for k in key) or key in result:
            raise ValueError('unique string identity required: '+label)
        result[key] = item
    return result


def _protocol(plan, spec, stream, allocation, protocol, bindings):
    _keys(protocol, PROTOCOL_KEYS, 'reconciliation protocol')
    if type(protocol['schema_version']) is not int or protocol['schema_version'] != 1:
        raise ValueError('reconciliation protocol schema1 required')
    for key in BINDINGS[:-1]:
        if protocol[key] != bindings[key]:
            raise ValueError('reconciliation protocol frozen binding mismatch: '+key)
    if stream['collection_sha256'] != bindings['collection_sha256']:
        raise ValueError('stream collection differs from original source')
    cost = plan['comparison']['cost']
    if cost['mode'] != 'itemized' or stream['currency'] != cost['currency']:
        raise ValueError('same-currency itemized accounting required; no implicit FX')
    scale = _positive_ratio(protocol['currency_atomic_per_comparison_unit'], 'currency conversion')
    components = {c['id']: c for c in stream['components']}
    streams = {s['id']: s for s in stream['streams']}
    card = cost['rate_card']
    mapping = _unique(protocol['rate_map'], ('component_id',), 'rate map')
    if set(k[0] for k in mapping) != {cid for cid,c in components.items() if c['kind'] == 'monetary'}:
        raise ValueError('exactly one mapping per monetary component required')
    rates, covered = {}, set()
    for (cid,), item in mapping.items():
        _keys(item, ('component_id','comparison_rate_id','atomic_quantity_per_comparison_unit'), 'rate mapping')
        rid = item['comparison_rate_id']
        if type(rid) is not str or rid not in card:
            raise ValueError('known comparison rate required')
        q = _positive_ratio(item['atomic_quantity_per_comparison_unit'], 'quantity conversion')
        source = stream['rates'][components[cid]['rate_id']]; target = card[rid]
        if (source['provider'], source['revision']) != (target['provider'], target['revision']):
            raise ValueError('source provider/revision must match comparison rate')
        # Both unit names are already frozen at the map endpoints; conversion is explicit.
        if source['unit'] == target['unit'] and q != 1:
            raise ValueError('identical units require identity conversion')
        if _fraction(Fraction(Decimal(source['per_unit']))*q) != _fraction(Fraction(Decimal(str(target['per_unit'])))*scale):
            raise ValueError('exact unit/currency rate conversion mismatch')
        rates[cid] = (rid, q, _fraction(Decimal(str(target['per_unit'])))); covered.add(rid)
    if covered != set(card):
        raise ValueError('every comparison rate must have an explicit source mapping')
    expected_lanes = {(s['id'],cid,l['resource_id'],l['lane_id']) for s in streams.values()
        for cid,req in s['requirements'].items() for l in req['lanes']}
    owners = _unique(protocol['lane_owners'], BASE_RULE[:-1], 'lane owners')
    if set(owners) != expected_lanes:
        raise ValueError('exactly one owner kind per required lane')
    for key, owner in owners.items():
        _keys(owner, BASE_RULE[:-1]+('owner_kind',), 'lane owner')
        kind = owner['owner_kind']; component = components[key[1]]
        if component['kind'] != 'monetary':
            raise ValueError('required resource proxy prevents numeric money reconciliation')
        if kind not in ('task_attempt','stream_attempt','non_attempt') or (kind == 'non_attempt' and component['category'] in ('vlm','text_fallback')):
            raise ValueError('explicit supported attempt ownership required for VLM/text lanes')
    groups = _unique(protocol['provider_counter_groups'], ('id',), 'provider counter groups')
    tuples = {}
    for (gid,), group in groups.items():
        _keys(group, ('id','stream_id','provider','revision','role'), 'counter group')
        _identifier(gid, 'counter ID')
        key = tuple(group[k] for k in ('stream_id','provider','revision','role'))
        if any(type(v) is not str for v in key) or key in tuples or key[0] not in streams:
            raise ValueError('unique stream/provider/revision/role counter tuple required')
        if key[3] not in ('perception','small_vlm','strong_vlm','text_llm') or not any((r['provider'],r['revision']) == key[1:3] for r in card.values()):
            raise ValueError('known counter provider/revision/role required')
        tuples[key] = gid
    vlm = _unique(protocol['vlm_counter_bindings'], ('stream_id',), 'VLM counter bindings')
    if {k[0] for k in vlm} != set(streams):
        raise ValueError('one VLM counter binding per stream required')
    for (sid,), binding in vlm.items():
        _keys(binding, ('stream_id','source_id','counter_ids'), 'VLM counter binding')
        sources = {s['id']:s for s in streams[sid]['sources']}
        source = binding['source_id']; ids = _list(binding['counter_ids'], 'VLM groups', MAX_ITEMS)
        expected = {gid for key,gid in tuples.items() if key[0] == sid and key[3] in ('small_vlm','strong_vlm')}
        if (type(source) is not str or source not in sources or 'vlm_calls' not in sources[source]['required_counters']
                or any(type(v) is not str for v in ids) or len(set(ids)) != len(ids) or set(ids) != expected):
            raise ValueError('exact original VLM counter group coverage required')
    return streams, components, scale, rates, owners, tuples, vlm


def join(plan, spec, inputs, traces, completion, assessment, stream_protocol, ledger,
         allocation_protocol, reconciliation_protocol, manifest, reports, *,
         expected_plan_sha256, expected_collection_sha256, expected_trace_sha256,
         expected_assessment_sha256, expected_stream_protocol_sha256, expected_ledger_sha256,
         expected_allocation_protocol_sha256, expected_reconciliation_protocol_sha256,
         expected_manifest_sha256):
    plan,spec,inputs,assessment,stream_protocol,ledger,allocation_protocol,reconciliation_protocol,manifest,reports = [
        _snapshot(v) for v in (plan,spec,inputs,assessment,stream_protocol,ledger,allocation_protocol,reconciliation_protocol,manifest,reports)]
    bindings = dict(zip(BINDINGS, (expected_plan_sha256,expected_collection_sha256,expected_stream_protocol_sha256,
                                  expected_allocation_protocol_sha256,expected_reconciliation_protocol_sha256)))
    for value,digest in ((reconciliation_protocol,expected_reconciliation_protocol_sha256),(manifest,expected_manifest_sha256)):
        if plan_digest(value) != _digest(digest):
            raise ValueError('reconciliation declaration differs from frozen digest')
    _keys(manifest, MANIFEST_KEYS, 'integration manifest')
    expected = dict(bindings, schema_version=2, last_trace_sha256=expected_trace_sha256,
                    ledger_sha256=expected_ledger_sha256, assessment_sha256=expected_assessment_sha256)
    if any(_encoded(manifest[k]) != _encoded(v) for k,v in expected.items()):
        raise ValueError('manifest original frozen binding mismatch')
    _text(manifest['reviewer'], 'reviewer'); _digest(manifest['coverage_evidence_sha256'])
    # Validate original source/failure policies before assigning any cost to its task.
    result = join_evidence(plan,spec,inputs,traces,completion,assessment,
        expected_plan_sha256=expected_plan_sha256,expected_collection_sha256=expected_collection_sha256,
        expected_trace_sha256=expected_trace_sha256,expected_assessment_sha256=expected_assessment_sha256)
    ledger_report = evaluate(stream_protocol,ledger,expected_protocol_sha256=expected_stream_protocol_sha256,
                             expected_ledger_sha256=expected_ledger_sha256)
    allocation = allocate(plan,stream_protocol,ledger,allocation_protocol,
        expected_comparison_plan_sha256=expected_plan_sha256,expected_stream_protocol_sha256=expected_stream_protocol_sha256,
        expected_ledger_sha256=expected_ledger_sha256,expected_allocation_protocol_sha256=expected_allocation_protocol_sha256)
    streams,components,scale,rates,lane_owners,groups,vlm = _protocol(plan,spec,stream_protocol,allocation_protocol,reconciliation_protocol,bindings)
    if not ledger_report['declared_accounting_complete'] or not allocation['declared_allocation_complete']:
        raise ValueError('unknown/pending/missing stream accounting prevents numeric reconciliation')
    # Bounds count original inventories plus repeated cross-references before expansion.
    reports = _list(reports, 'reports', MAX_ITEMS)
    count = sum(len(_list(r.get('attempts'), 'attempts', 1024)) for r in reports if type(r) is dict)
    if count > MAX_ITEMS:
        raise ValueError('total canonical attempts exceed 10000')
    refs = count
    claims_count = 0
    for key in ('trace_reports','stream_reports','attempt_instances','line_owners','provider_counters'):
        refs += len(_list(manifest[key], key, MAX_ITEMS))
    for report in reports:
        if type(report) is not dict: raise ValueError('report object required')
        for attempt in report['attempts']:
            if type(attempt) is not dict: raise ValueError('attempt object required')
            claims_count += len(_list(attempt.get('line_claims', []), 'line claims', MAX_ITEMS))
            quantities = attempt.get('cost_quantities')
            if type(quantities) is not dict:
                raise ValueError('complete attempt quantities required before numeric reconciliation')
            refs += len(quantities)
    if claims_count > MAX_ITEMS:
        raise ValueError('total canonical line claims exceed 10000')
    refs += claims_count
    refs += allocation['allocation_edges'] + len(result['rows'])*len(plan['comparison']['cost']['rate_card'])
    if refs > MAX_REFERENCES:
        raise ValueError('combined reconciliation cross-references exceed 100000')
    by_report = _unique(reports, ('sha256',), 'reports')
    trace_bindings = _unique(manifest['trace_reports'], ('trace_sha256',), 'trace reports')
    stream_bindings = _unique(manifest['stream_reports'], ('stream_id',), 'stream reports')
    task_targets = {(t['task_id'],streams[t['stream_id']]['arm']):t for t in allocation['targets'] if t['kind']=='task'}
    for task in {key[0] for key in task_targets}:
        if len({streams[task_targets[task,arm]['stream_id']]['recording_sha256'] for arm in 'ABC'}) != 1:
            raise ValueError('original task ABC targets must retain one matched recording')
    expected_reports = {}
    for row,source in zip(result['rows'],result['lineage']):
        binding = trace_bindings.pop((source['trace_sha256'],), None)
        if binding is None: raise ValueError('missing original trace report')
        _keys(binding, ('trace_sha256','report_sha256'), 'trace report')
        digest = _digest(binding['report_sha256']); target = task_targets[row['task_id'],row['arm']]
        if digest in expected_reports: raise ValueError('report reused across original traces')
        expected_reports[digest] = ('task',row['task_id'],row['arm'],target['stream_id'])
    if trace_bindings or set(k[0] for k in stream_bindings) != set(streams):
        raise ValueError('exact source trace and stream report inventories required')
    for (sid,),binding in stream_bindings.items():
        _keys(binding, ('stream_id','report_sha256'), 'stream report')
        digest = _digest(binding['report_sha256'])
        if digest in expected_reports: raise ValueError('report scope reuse')
        expected_reports[digest] = ('stream',None,streams[sid]['arm'],sid)
    if {k[0] for k in by_report} != set(expected_reports):
        raise ValueError('extra or missing report')
    attempts, scopes, ids = {}, set(), set()
    counter_totals = {gid:0 for gid in groups.values()}
    task_inputs = {t['id']:t['input_sha256'] for t in plan['comparison']['tasks']}
    configuration_hashes = {arm:plan_digest(plan['systems'][arm]) for arm in 'ABC'}
    for digest,context in expected_reports.items():
        report = by_report[digest,]; kind,task,arm,sid = context
        # Declarations were validated once above. Audit each native header/body
        # without revalidating every frozen stream and task for every report.
        header = dict(schema_version=2, comparison_plan_sha256=expected_plan_sha256,
            collection_sha256=expected_collection_sha256, cost_sha256=plan_digest(plan['comparison']['cost']),
            task_id=task,arm=arm,input_sha256=None if task is None else task_inputs[task],
            configuration_sha256=configuration_hashes[arm],evidence=plan['comparison']['evidence'])
        if report.get('schema_version') == 2 and type(report.get('schema_version')) is int:
            if kind != 'task': raise ValueError('native v2 is task scoped only')
            _quantities(plan,report,header,digest)
        else:
            header.update(schema_version=3,stream_protocol_sha256=expected_stream_protocol_sha256,
                reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,
                scope_kind=kind,stream_id=sid,recording_sha256=streams[sid]['recording_sha256'])
            _keys(report, REPORT_KEYS, 'native schema3 report')
            if (report['sha256'] != digest or plan_digest({k:v for k,v in report.items() if k != 'sha256'}) != digest
                    or any(_encoded(report[k]) != _encoded(v) for k,v in header.items())):
                raise ValueError('native report differs from original scope binding or frozen hash')
            _validate_body(plan['comparison']['cost']['rate_card'],report)
        if report['scope'] in scopes: raise ValueError('duplicate recorder scope')
        scopes.add(report['scope'])
        for attempt in report['attempts']:
            aid = attempt['attempt_id']
            if aid in ids: raise ValueError('duplicate canonical attempt')
            ids.add(aid); key = (sid,attempt['provider'],attempt['revision'],attempt['role'])
            if key not in groups: raise ValueError('attempt lacks exact provider counter group')
            counter_totals[groups[key]] += 1
            attempts[digest,aid] = (attempt,context,report['schema_version'])
    instances = _unique(manifest['attempt_instances'], ('report_sha256','attempt_id'), 'attempt instances')
    if set(instances) != set(attempts): raise ValueError('exact attempt instance inventory required')
    instance_ids = set()
    for key,item in instances.items():
        _keys(item, ('report_sha256','attempt_id','execution_instance_sha256','evidence_sha256'), 'attempt instance')
        identity = _digest(item['execution_instance_sha256']); _digest(item['evidence_sha256'])
        attempt,_,version = attempts[key]
        if identity in instance_ids or (version == 3 and identity != attempt['execution_instance_sha256']):
            raise ValueError('duplicate or changed execution instance')
        instance_ids.add(identity)
    counters = _unique(manifest['provider_counters'], ('counter_id',), 'provider counters')
    if {k[0] for k in counters} != set(counter_totals): raise ValueError('exact provider counter inventory required')
    for (gid,),item in counters.items():
        _keys(item, ('counter_id','count','evidence_sha256'), 'provider counter')
        _digest(item['evidence_sha256'])
        if _int(item['count'], 'provider count') != counter_totals[gid]:
            raise ValueError('provider counter differs from complete original attempt count')
    for (sid,),binding in vlm.items():
        observed = ledger_report['streams'][sid]['counters'][binding['source_id']]['vlm_calls']
        if observed is None or observed != sum(counter_totals[gid] for gid in binding['counter_ids']):
            raise ValueError('known source VLM delta must equal all bound attempt counters')
    entries = {plan_digest(e):e for e in ledger['entries']}
    owners = _unique(manifest['line_owners'], ('entry_sha256',), 'line owners')
    if {k[0] for k in owners} != set(entries): raise ValueError('exact ledger line owner inventory required')
    owned = {key:[] for key in attempts}
    for (digest,),owner in owners.items():
        entry = entries[digest]; lane = tuple(entry[k] for k in BASE_RULE[:-1])
        kind = lane_owners[lane]['owner_kind']
        if owner.get('owner_kind') != kind: raise ValueError('line owner differs from frozen lane kind')
        if kind == 'non_attempt':
            _keys(owner, ('entry_sha256','owner_kind','reason','evidence_sha256'), 'non-attempt owner')
            _text(owner['reason'],'non-attempt reason'); _digest(owner['evidence_sha256'])
        else:
            _keys(owner, ('entry_sha256','owner_kind','report_sha256','attempt_id'), 'attempt owner')
            _digest(owner['report_sha256']); _text(owner['attempt_id'], 'attempt ID')
            key = (owner['report_sha256'],owner['attempt_id'])
            if key not in attempts: raise ValueError('line owner references unknown attempt')
            attempt,context,version = attempts[key]
            if context[0]+'_attempt' != kind or context[3] != entry['stream_id']:
                raise ValueError('original task/stream owner cannot migrate across scope or stream')
            identity = entry['identity']
            if identity['kind'] != 'invoice' or (identity['provider'],identity['revision']) != (attempt['provider'],attempt['revision']):
                raise ValueError('attempt line requires matching provider invoice identity')
            owned[key].append(digest)
    documents = {}
    for digest,entry in entries.items():
        if entry['identity']['kind'] == 'invoice':
            documents.setdefault(entry['identity']['document_sha256'],set()).add(digest)
    v2_documents = set()
    for key,(attempt,context,version) in attempts.items():
        claims = set(owned[key])
        if not claims: raise ValueError('every original attempt needs explicit invoice lines, including zero cost')
        if version == 3:
            if claims != set(attempt['line_claims']): raise ValueError('native line claims differ from ownership manifest')
        else:
            receipt = attempt['receipt_sha256']
            if receipt in v2_documents or claims != documents.get(receipt,set()):
                raise ValueError('native v2 whole-document exclusivity violated')
            v2_documents.add(receipt)
        quantities = {rid:0 for rid,r in plan['comparison']['cost']['rate_card'].items()
                      if (r['provider'],r['revision']) == (attempt['provider'],attempt['revision'])}
        for digest in claims:
            entry = entries[digest]; rid,q,_ = rates[entry['component_id']]
            converted = _fraction(Fraction(entry['quantity'])/q)
            if converted.denominator != 1 or converted > 10**12:
                raise ValueError('original line must exactly fit comparison quantity units')
            quantities[rid] += int(converted)
        if quantities != attempt['cost_quantities']:
            raise ValueError('attempt quantities differ from exact owned source lines')
    task_quantities = {key:{rid:0 for rid in plan['comparison']['cost']['rate_card']} for key in task_targets}
    for source in allocation['sources']:
        digest = source['entry_sha256']; entry = entries[digest]; owner = owners[digest,]
        rid,q,price = rates[entry['component_id']]
        # Original line discrepancies cannot cancel after attribution either.
        if _fraction(Fraction(entry['billed_amount'])/scale) != _fraction(Fraction(entry['quantity'])/q*price):
            raise ValueError('original billed line differs from exact comparison charge')
        if owner['owner_kind'] == 'task_attempt':
            context = attempts[owner['report_sha256'],owner['attempt_id']][1]
            target = task_targets[context[1],context[2]]
            if (allocation['rules'][source['rule_index']]['method'] != 'direct' or len(source['shares']) != 1
                    or allocation['targets'][source['shares'][0]['target_index']]['id'] != target['id']):
                raise ValueError('task-owned line cannot be shared or moved to overhead after execution')
        for item in source['allocations']:
            converted = _fraction(Fraction(item['quantity'])/q)
            if converted.denominator != 1 or converted > 10**12:
                raise ValueError('allocated line must exactly fit comparison quantity units')
            if _fraction(Fraction(item['billed_amount'])/scale) != _fraction(converted*price):
                raise ValueError('allocated per-line billed/rate discrepancy; cancellation forbidden')
            target = allocation['targets'][item['target_index']]
            if target['kind'] == 'task':
                key = (target['task_id'],streams[target['stream_id']]['arm'])
                task_quantities[key][rid] += int(converted)
                if task_quantities[key][rid] > 10**12: raise ValueError('final task quantity exceeds 1e12')
    for row in result['rows']:
        if row['cost_quantities'] != task_quantities[row['task_id'],row['arm']]:
            raise ValueError('assessment must equal allocated quantities exactly once')
    whole = compare_whole_stream(plan,stream_protocol,ledger,ledger_report,
        reconciliation_protocol['recording_groups'],reconciliation_protocol['whole_stream_cost_policy'])
    # Compact indices bind each original attempt and source once; never duplicate canonical A.
    report_ids = sorted(expected_reports); report_indices = {v:i for i,v in enumerate(report_ids)}
    attempt_keys = sorted(attempts); attempt_indices = {v:i for i,v in enumerate(attempt_keys)}
    lineage = dict(reports=[dict(sha256=d,scope_kind=expected_reports[d][0],task_id=expected_reports[d][1],
        arm=expected_reports[d][2],stream_id=expected_reports[d][3]) for d in report_ids],
        attempts=[dict(report_index=report_indices[k[0]],attempt_id=k[1],
            execution_instance_sha256=instances[k]['execution_instance_sha256']) for k in attempt_keys],
        lines=[dict(entry_sha256=s['entry_sha256'],attempt_index=None if owners[s['entry_sha256'],]['owner_kind']=='non_attempt'
            else attempt_indices[owners[s['entry_sha256'],]['report_sha256'],owners[s['entry_sha256'],]['attempt_id']]) for s in allocation['sources']])
    result.update(schema_version=2,stream_protocol_sha256=expected_stream_protocol_sha256,
        ledger_sha256=expected_ledger_sha256,allocation_protocol_sha256=expected_allocation_protocol_sha256,
        reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,manifest_sha256=expected_manifest_sha256,
        allocation_report_sha256=plan_digest(allocation),ledger_report_sha256=plan_digest(ledger_report),
        reconciliation_lineage=lineage,whole_stream_comparison=whole,
        task_comparison_scope='attributed_task_quality_latency_and_cost_diagnostic',
        primary_meets_reconciled_numeric_gates=(result['comparison']['primary_meets_numeric_gates'] and whole['primary_meets_whole_stream_cost_gate']),
        declared_reconciliation_complete=True,canonical_stream_costs=ledger_report['canonical_arms'],
        receipt_authenticity_verified=False,instrumentation_coverage_verified=False,isolation_verified=False,
        whole_pipeline_cost_verified=False,actual_acceptance='incomplete',
        certification='Frozen declared source, attempt, invoice-line and allocation reconciliation only. '
        'Whole-stream declared money includes startup, idle and overhead once; matched recording resampling '
        'does not certify recording independence or task correlations. Receipt authenticity, real inference, '
        'measurement truth, coverage and actual savings remain unverified.')
    if len(_encoded(result))+1 > MAX_REPORT_BYTES:
        raise ValueError('serialized reconciliation output exceeds 16 MiB')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','assessment','stream-protocol','ledger','allocation-protocol','reconciliation-protocol',
                 'manifest','reports','plan-sha256','collection-sha256','trace-sha256','assessment-sha256',
                 'stream-protocol-sha256','ledger-sha256','allocation-protocol-sha256',
                 'reconciliation-protocol-sha256','manifest-sha256','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); root = Path(args.run)
    values = {name:_read(getattr(args,name)) for name in ('assessment','stream_protocol','ledger',
              'allocation_protocol','reconciliation_protocol','manifest','reports')}
    digests = {'expected_'+name:getattr(args,name) for name in ('plan_sha256','collection_sha256','trace_sha256',
        'assessment_sha256','stream_protocol_sha256','ledger_sha256','allocation_protocol_sha256',
        'reconciliation_protocol_sha256','manifest_sha256')}
    result = join(_read(root/'comparison-plan.json'),_read(root/'collection-spec.json'),_read(root/'inputs.json'),
        read_traces(root/'source-traces.jsonl'),_read(root/'complete.json'),**values,**digests)
    output = Path(args.output); output.mkdir(mode=0o700,parents=False,exist_ok=False)
    _write_new(output/'report.json',result)
    print(_encoded(result).decode('ascii'))


if __name__ == '__main__':
    main()
