"""Frozen observed stream accounting; declarations never certify measurement truth."""
import argparse
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import re

from .paired_replay import _encoded, _keys, _write_new
from .perception_comparison import _digest, _text, plan_digest
from .three_arm_comparison import _read, _snapshot

CATEGORIES = ('perception', 'resident_idle', 'startup', 'capture_sensor', 'media_preparation',
              'storage', 'transport', 'vlm', 'text_fallback', 'local_workers')
COUNTERS = ('captured', 'sampled', 'vlm_calls', 'attempted', 'dropped')
LIMIT = 2**63-1
MAX_ENTRIES = 10000
MAX_REPORT_BYTES = 16*1024*1024
MAX_IDENTIFIER_CHARS = 256


def _int(value, label, minimum=0):
    if type(value) is not int or not minimum <= value <= LIMIT:
        raise ValueError('bounded exact integer required: '+label)
    return value


def _fraction(value):
    value = Fraction(value)
    if abs(value.numerator).bit_length() > 256 or value.denominator.bit_length() > 256:
        raise ValueError('exact rational exceeds 256-bit bound')
    return value


def _ratio(value):
    value = _fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def _rounded(value):
    """Frozen nearest integer atomic currency unit, ties to even, per invoice line."""
    value = _fraction(value)
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2*remainder > value.denominator or
                          (2*remainder == value.denominator and quotient % 2 == 1))


def _list(value, label, maximum=MAX_ENTRIES):
    if type(value) is not list or len(value) > maximum:
        raise ValueError('bounded list required: '+label)
    return value


def _identifier(value, label):
    _text(value, label)
    if len(value) > MAX_IDENTIFIER_CHARS:
        raise ValueError('identifier/unit label exceeds 256 characters: '+label)
    return value


def validate_protocol(protocol, *, expected_protocol_sha256):
    protocol = _snapshot(protocol)
    if plan_digest(protocol) != _digest(expected_protocol_sha256):
        raise ValueError('protocol differs from frozen digest')
    _keys(protocol, ('schema_version', 'comparison_plan_sha256', 'collection_sha256',
        'measurement_protocol_sha256', 'boundary', 'currency', 'currency_unit', 'rate_revision',
        'invoice_rounding', 'isolation', 'components', 'rates', 'streams'), 'stream protocol')
    if type(protocol['schema_version']) is not int or protocol['schema_version'] != 1:
        raise ValueError('stream protocol schema1 required')
    for field in ('comparison_plan_sha256', 'collection_sha256', 'measurement_protocol_sha256'):
        _digest(protocol[field])
    for field in ('boundary', 'currency', 'currency_unit', 'rate_revision'):
        _text(protocol[field], field)
    if protocol['invoice_rounding'] != 'line_atomic_half_even' or protocol['isolation'] != 'isolated_arms':
        raise ValueError('frozen line rounding and isolated arms required')
    rates = protocol['rates']
    if type(rates) is not dict or len(rates) > 256:
        raise ValueError('bounded rate card required')
    for key, rate in rates.items():
        _identifier(key, 'rate ID')
        _keys(rate, ('provider', 'revision', 'unit', 'per_unit'), 'rate')
        for field in ('provider', 'revision'):
            _text(rate[field], field)
        _identifier(rate['unit'], 'rate unit')
        price = rate['per_unit']
        if (type(price) is not str or len(price) > 64 or
                re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9]+)?', price) is None):
            raise ValueError('bounded nonnegative decimal rate string required')
        _fraction(Decimal(price))
    components, categories = {}, set()
    for component in _list(protocol['components'], 'components', 256):
        _keys(component, ('id', 'category', 'kind', 'unit', 'source_unit', 'atomic_per_source', 'coverage', 'rate_id'), 'component')
        key = _identifier(component['id'], 'component ID')
        if key in components or component['category'] not in CATEGORIES:
            raise ValueError('unique component and supported category required')
        if component['kind'] not in ('monetary', 'proxy') or component['coverage'] not in ('interval', 'event'):
            raise ValueError('explicit component kind/coverage required')
        _identifier(component['unit'], 'component unit'); _identifier(component['source_unit'], 'source unit')
        conversion = component['atomic_per_source']
        _keys(conversion, ('numerator', 'denominator'), 'atomic conversion')
        _int(conversion['numerator'], 'conversion numerator', 1)
        _int(conversion['denominator'], 'conversion denominator', 1)
        _fraction(Fraction(conversion['numerator'], conversion['denominator']))
        rate = component['rate_id']
        if component['kind'] == 'monetary':
            if type(rate) is not str or rate not in rates or rates[rate]['unit'] != component['unit']:
                raise ValueError('explicit matching frozen monetary rate required')
        elif rate is not None:
            raise ValueError('proxy cannot silently become monetary')
        components[key] = component; categories.add(component['category'])
    if categories != set(CATEGORIES):
        raise ValueError('declare every required boundary category, including not-applicable components')
    streams, namespaces, resources = {}, {}, {}
    declared_lanes = 0
    for stream in _list(protocol['streams'], 'streams', 1000):
        _keys(stream, ('id', 'arm', 'recording_sha256', 'clock_domain', 'requested_duration_ns',
            'execution_namespace', 'cache_namespace', 'isolation_evidence_sha256', 'sources', 'requirements'), 'stream')
        sid = _identifier(stream['id'], 'stream ID')
        if sid in streams or stream['arm'] not in ('A', 'B', 'C'):
            raise ValueError('unique original A/B/C stream required')
        _digest(stream['recording_sha256']); _digest(stream['isolation_evidence_sha256'])
        _text(stream['clock_domain'], 'clock domain'); _int(stream['requested_duration_ns'], 'requested duration', 1)
        for field in ('execution_namespace', 'cache_namespace'):
            namespace = _identifier(stream[field], field)
            if (field, namespace) in namespaces:
                raise ValueError('execution/cache namespace reuse unsupported')
            namespaces[field, namespace] = sid
        sources = set()
        for source in _list(stream['sources'], 'sources', 256):
            _keys(source, ('id', 'semantics', 'required_counters'), 'capture source')
            key = _identifier(source['id'], 'capture source ID')
            required = source['required_counters']
            if (key in sources or source['semantics'] not in ('captured_only', 'attempted_equals_captured_plus_dropped') or
                    type(required) is not list or any(type(c) is not str or c not in COUNTERS for c in required) or
                    len(set(required)) != len(required) or 'captured' not in required):
                raise ValueError('unique capture source and explicit counter contract required')
            if source['semantics'] == 'attempted_equals_captured_plus_dropped' and not {'attempted','dropped'} <= set(required):
                raise ValueError('attempted/drop counters required by capture semantics')
            sources.add(key)
        if not sources:
            raise ValueError('at least one capture source required')
        requirements = stream['requirements']
        _keys(requirements, components, 'component requirements')
        for key, requirement in requirements.items():
            _keys(requirement, ('state', 'reason', 'lanes'), 'requirement')
            if requirement['state'] not in ('required', 'not_applicable'):
                raise ValueError('explicit component requirement required')
            _text(requirement['reason'], 'requirement reason')
            lanes = _list(requirement['lanes'], 'resource lanes', 256)
            declared_lanes += len(lanes)
            if declared_lanes > MAX_ENTRIES:
                raise ValueError('total declared resource lanes exceed 10000')
            if bool(lanes) != (requirement['state'] == 'required'):
                raise ValueError('required components need lanes; not-applicable components have none')
            seen = set()
            for lane in lanes:
                _keys(lane, ('resource_id', 'lane_id'), 'resource lane')
                resource = _identifier(lane['resource_id'], 'resource ID')
                lane_id = _identifier(lane['lane_id'], 'lane ID')
                if (resource, lane_id) in seen:
                    raise ValueError('duplicate component lane')
                seen.add((resource, lane_id))
                # Distinct components may meter the same resource but cannot claim the same lane.
                if (resource, lane_id) in resources:
                    raise ValueError('resource measurement lane counted more than once')
                resources[resource, lane_id] = sid
        streams[sid] = stream
    if not streams or {s['arm'] for s in streams.values()} != set('ABC'):
        raise ValueError('declare canonical streams for all three arms')
    # A resource cannot be shared across arms even under differently named meter lanes.
    owners = {}
    for (resource, lane), sid in resources.items():
        arm = streams[sid]['arm']
        if resource in owners and owners[resource] != arm:
            raise ValueError('cross-arm resource reuse unsupported')
        owners[resource] = arm
    return protocol


def evaluate(protocol, ledger, *, expected_protocol_sha256, expected_ledger_sha256):
    protocol = validate_protocol(protocol, expected_protocol_sha256=expected_protocol_sha256)
    ledger = _snapshot(ledger)
    if plan_digest(ledger) != _digest(expected_ledger_sha256):
        raise ValueError('ledger differs from externally reviewed digest')
    _keys(ledger, ('schema_version', 'protocol_sha256', 'comparison_plan_sha256', 'collection_sha256',
        'reviewer', 'source_evidence_sha256', 'streams', 'entries'), 'ledger')
    if (type(ledger['schema_version']) is not int or ledger['schema_version'] != 1 or
            ledger['protocol_sha256'] != expected_protocol_sha256 or
            any(ledger[k] != protocol[k] for k in ('comparison_plan_sha256', 'collection_sha256'))):
        raise ValueError('ledger frozen protocol/source mismatch')
    _text(ledger['reviewer'], 'reviewer'); _digest(ledger['source_evidence_sha256'])
    frozen = {s['id']: s for s in protocol['streams']}
    components = {c['id']: c for c in protocol['components']}
    observations, reports, issues = {}, {}, {}
    entry_count = 0
    def incomplete(sid, reason):
        key = (sid, reason)
        if key not in issues:
            issues[key] = dict(stream_id=sid, reason=reason, occurrences=0)
        issues[key]['occurrences'] += 1
    for stream in _list(ledger['streams'], 'observed streams', 1000):
        _keys(stream, ('id', 'clock_domain', 'start_ns', 'end_ns', 'utc_anchor', 'availability', 'counters'), 'observed stream')
        sid = stream['id']
        if type(sid) is not str or sid not in frozen or sid in observations:
            raise ValueError('unknown or duplicate observed stream')
        if stream['clock_domain'] != frozen[sid]['clock_domain']:
            raise ValueError('observed clock domain differs from frozen contract')
        start = _int(stream['start_ns'], 'start'); end = _int(stream['end_ns'], 'end', 1)
        if end <= start:
            raise ValueError('positive observed elapsed duration required')
        anchor = stream['utc_anchor']
        _keys(anchor, ('monotonic_ns', 'unix_ns', 'evidence_sha256'), 'UTC anchor')
        _int(anchor['monotonic_ns'], 'anchor monotonic'); _int(anchor['unix_ns'], 'anchor UTC')
        _digest(anchor['evidence_sha256'])
        durations = dict(available=0, unavailable=0, unknown=0); previous = start
        for interval in _list(stream['availability'], 'availability'):
            entry_count += 1
            _keys(interval, ('start_ns', 'end_ns', 'state', 'reason', 'evidence_sha256'), 'availability interval')
            begin = _int(interval['start_ns'], 'availability start'); finish = _int(interval['end_ns'], 'availability end')
            if begin != previous or not begin < finish <= end or interval['state'] not in durations:
                raise ValueError('sorted gap-free non-overlapping availability partition required')
            _text(interval['reason'], 'availability reason')
            if interval['evidence_sha256'] is not None:
                _digest(interval['evidence_sha256'])
            elif interval['state'] != 'unknown':
                raise ValueError('known availability requires evidence reference')
            durations[interval['state']] += finish-begin; previous = finish
        if previous != end:
            raise ValueError('availability must cover full observed interval')
        if durations['unknown']:
            incomplete(sid, 'unknown availability duration')
        if end-start != frozen[sid]['requested_duration_ns']:
            incomplete(sid, 'observed duration differs from frozen requested duration')
        counters, source_specs = {}, {s['id']: s for s in frozen[sid]['sources']}
        for counter in _list(stream['counters'], 'capture counters', 256):
            entry_count += 1
            _keys(counter, ('source_id', 'start', 'end'), 'capture counter')
            source = counter['source_id']
            if type(source) is not str or source not in source_specs or source in counters:
                raise ValueError('unknown or duplicate capture counters')
            _keys(counter['start'], COUNTERS, 'counter start'); _keys(counter['end'], COUNTERS, 'counter end')
            delta = {}
            for key in COUNTERS:
                a, b = counter['start'][key], counter['end'][key]
                if a is not None: _int(a, 'counter start')
                if b is not None: _int(b, 'counter end')
                if a is None or b is None:
                    delta[key] = None
                    if key in source_specs[source]['required_counters']:
                        incomplete(sid, 'unknown required counter: '+source+'/'+key)
                else:
                    if b < a: raise ValueError('counter reset/wrap requires separate stream')
                    delta[key] = b-a
            if delta['sampled'] is not None and delta['captured'] is not None and delta['sampled'] > delta['captured']:
                raise ValueError('unique sampled frames exceed captured frames')
            if source_specs[source]['semantics'] == 'attempted_equals_captured_plus_dropped':
                values = [delta[k] for k in ('attempted', 'captured', 'dropped')]
                if None not in values and values[0] != values[1]+values[2]:
                    raise ValueError('capture attempt/drop counter discrepancy')
            delta['captured_per_elapsed_second'] = (None if delta['captured'] is None else
                _ratio(Fraction(delta['captured']*10**9, end-start)))
            delta['captured_per_available_second'] = (None if delta['captured'] is None or durations['unknown'] or not durations['available'] else
                _ratio(Fraction(delta['captured']*10**9, durations['available'])))
            if delta['captured'] and not durations['available'] and not durations['unknown']:
                raise ValueError('captured frames with no capture availability')
            counters[source] = delta
        if set(counters) != set(source_specs):
            raise ValueError('explicit counter records required for every source; use null for unknown')
        observations[sid] = stream
        reports[sid] = dict(arm=frozen[sid]['arm'], elapsed_ns=end-start, availability_ns=durations,
            counters=counters, known_monetary_atomic=0, known_rate_derived_atomic=_ratio(0),
            known_proxy_quantities={}, component_totals={}, complete_accounting=False)
    if set(observations) != set(frozen):
        raise ValueError('exactly one observed record per frozen stream required')
    lines, documents, lanes, rate_totals = set(), {}, {}, {sid: Fraction(0) for sid in frozen}
    for entry in _list(ledger['entries'], 'cost entries'):
        entry_count += 1
        _keys(entry, ('stream_id', 'component_id', 'resource_id', 'lane_id', 'start_ns', 'end_ns',
            'event_ns', 'identity', 'source_quantity', 'quantity', 'status', 'billed_amount', 'evidence_sha256'), 'cost entry')
        sid, cid = entry['stream_id'], entry['component_id']
        if type(sid) is not str or sid not in frozen or type(cid) is not str or cid not in components:
            raise ValueError('unknown stream/component')
        component, requirement = components[cid], frozen[sid]['requirements'][cid]
        lane = dict(resource_id=entry['resource_id'], lane_id=entry['lane_id'])
        if requirement['state'] != 'required' or lane not in requirement['lanes']:
            raise ValueError('entry requires frozen applicable resource lane')
        source = observations[sid]; start, end = source['start_ns'], source['end_ns']
        key = (sid, cid, entry['resource_id'], entry['lane_id'])
        if component['coverage'] == 'interval':
            begin = _int(entry['start_ns'], 'measurement start'); finish = _int(entry['end_ns'], 'measurement end')
            if entry['event_ns'] is not None or not start <= begin < finish <= end:
                raise ValueError('measurement interval outside observed stream')
            lanes.setdefault(key, []).append((begin, finish))
        else:
            event = _int(entry['event_ns'], 'event time')
            if entry['start_ns'] is not None or entry['end_ns'] is not None or not start <= event < end:
                raise ValueError('measurement event outside observed stream')
            lanes.setdefault(key, []).append((event, event))
        identity = entry['identity']
        if type(identity) is not dict or identity.get('kind') not in ('invoice', 'local'):
            raise ValueError('explicit invoice/local measurement identity required')
        if identity['kind'] == 'invoice':
            _keys(identity, ('kind', 'provider', 'revision', 'account_scope', 'invoice_id', 'document_sha256', 'line_id'), 'invoice identity')
            for field in ('provider','revision','account_scope','invoice_id','line_id'):
                _text(identity[field], field)
            _digest(identity['document_sha256'])
            doc = tuple(identity[k] for k in ('provider','account_scope','invoice_id'))
            document_binding = (identity['revision'], identity['document_sha256'])
            if doc in documents and documents[doc] != document_binding:
                raise ValueError('same logical invoice with changed document/revision')
            documents[doc] = document_binding
            line_id = ('invoice',)+doc+(identity['line_id'],)
            if component['kind'] != 'monetary':
                raise ValueError('invoice cannot masquerade as local resource proxy')
            rate = protocol['rates'][component['rate_id']]
            if (identity['provider'],identity['revision']) != (rate['provider'],rate['revision']):
                raise ValueError('invoice provider/revision differs from frozen rate')
        else:
            _keys(identity, ('kind', 'meter_id', 'measurement_id'), 'local identity')
            _text(identity['meter_id'], 'meter ID'); _text(identity['measurement_id'], 'measurement ID')
            line_id = ('local', identity['meter_id'], identity['measurement_id'])
        if line_id in lines:
            raise ValueError('duplicate actual invoice line/local measurement')
        lines.add(line_id)
        quantity, billed, status = entry['quantity'], entry['billed_amount'], entry['status']
        if quantity is not None: _int(quantity, 'atomic quantity')
        original = entry['source_quantity']
        if original is not None:
            _keys(original, ('numerator', 'denominator'), 'original source quantity')
            _int(original['numerator'], 'source quantity numerator')
            _int(original['denominator'], 'source quantity denominator', 1)
            conversion = component['atomic_per_source']
            converted = _fraction(Fraction(original['numerator'], original['denominator'])*
                                  Fraction(conversion['numerator'], conversion['denominator']))
            if quantity is None or converted.denominator != 1 or converted.numerator != quantity:
                raise ValueError('source quantity does not exactly normalize to atomic quantity')
        elif quantity is not None:
            raise ValueError('known atomic quantity requires original quantity')
        if billed is not None: _int(billed, 'atomic billed amount')
        if entry['evidence_sha256'] is not None: _digest(entry['evidence_sha256'])
        allowed = ('settled','pending','unknown') if component['kind'] == 'monetary' else ('measured','unknown')
        if status not in allowed:
            raise ValueError('explicit supported measurement/billing status required')
        known = status in ('settled','measured')
        if known and (quantity is None or entry['evidence_sha256'] is None):
            raise ValueError('known measurements require quantity and evidence reference')
        if not known:
            incomplete(sid, 'pending/unknown component: '+cid)
        total = reports[sid]['component_totals'].setdefault(cid, dict(known_quantity=0, unknown_entries=0))
        if known: total['known_quantity'] += quantity
        else: total['unknown_entries'] += 1
        if component['kind'] == 'proxy':
            if billed is not None:
                raise ValueError('local resource proxy cannot contain billed money')
            if known:
                proxies = reports[sid]['known_proxy_quantities']
                proxies[component['unit']] = proxies.get(component['unit'],0)+quantity
        else:
            if not known and billed is not None:
                raise ValueError('unsettled bills cannot be treated as known money')
            if known:
                if billed is None: raise ValueError('settled monetary line requires atomic billed amount')
                derived = _fraction(quantity*Fraction(Decimal(protocol['rates'][component['rate_id']]['per_unit'])))
                rate_totals[sid] = _fraction(rate_totals[sid]+derived)
                reports[sid]['known_monetary_atomic'] += billed
                if _rounded(derived) != billed:
                    incomplete(sid, 'invoice differs from frozen line rounding: '+cid)
    if entry_count > MAX_ENTRIES:
        raise ValueError('total observation/cost entries exceed 10000')
    for sid, stream in frozen.items():
        source = observations[sid]
        for cid, requirement in stream['requirements'].items():
            for lane in requirement['lanes']:
                key = (sid, cid, lane['resource_id'], lane['lane_id'])
                intervals = sorted(lanes.get(key, []))
                if components[cid]['coverage'] == 'event':
                    if not intervals: incomplete(sid, 'missing required event component: '+cid)
                    continue
                previous = source['start_ns']
                for begin, end in intervals:
                    if begin < previous:
                        raise ValueError('overlapping measurement intervals count a resource lane twice')
                    if begin != previous:
                        incomplete(sid, 'unmeasured resource interval: '+cid)
                    previous = end
                if previous != source['end_ns']:
                    incomplete(sid, 'unmeasured resource interval: '+cid)
        report = reports[sid]
        report['known_rate_derived_atomic'] = _ratio(rate_totals[sid])
        report['complete_accounting'] = not any(i['stream_id'] == sid for i in issues.values())
        report['known_subtotal_per_elapsed_hour'] = _ratio(Fraction(report['known_monetary_atomic']*3600*10**9, report['elapsed_ns']))
        report['declared_complete_cost_per_elapsed_hour'] = (report['known_subtotal_per_elapsed_hour'] if report['complete_accounting'] else None)
        report['known_proxy_per_elapsed_hour'] = {unit:_ratio(Fraction(q*3600*10**9, report['elapsed_ns']))
                                                 for unit,q in report['known_proxy_quantities'].items()}
    canonical = {arm:dict(stream_ids=[], known_monetary_atomic=0) for arm in 'ABC'}
    for sid, report in reports.items():
        arm = canonical[report['arm']]
        arm['stream_ids'].append(sid); arm['known_monetary_atomic'] += report['known_monetary_atomic']
    result = dict(schema_version=1, protocol_sha256=expected_protocol_sha256, ledger_sha256=expected_ledger_sha256,
        comparison_plan_sha256=protocol['comparison_plan_sha256'], collection_sha256=protocol['collection_sha256'],
        currency=protocol['currency'], currency_unit=protocol['currency_unit'], rate_revision=protocol['rate_revision'],
        streams=reports, canonical_arms=canonical, incomplete_reasons=list(issues.values()),
        declared_accounting_complete=not issues, structural_hash_integrity_verified=True,
        receipt_authenticity_verified=False, measurement_truth_verified=False, isolation_verified=False,
        instrumentation_coverage_verified=False, whole_pipeline_cost_verified=False,
        allocation_implemented=False, actual_acceptance='incomplete',
        certification='Frozen declarations and exact observed-ledger arithmetic only. Declared completeness is not '
        'proof of whole-pipeline measurement, provider authenticity, isolation, real inference or missing-call coverage. '
        'No allocation or daily extrapolation is performed; task numerical gates are unchanged.')
    # Bound compact bytes including the newline used by both output artifacts and stdout.
    # Never truncate reasons or suppress incomplete state to fit the bound.
    if len(_encoded(result)) + 1 > MAX_REPORT_BYTES:
        raise ValueError('serialized stream report exceeds 16 MiB')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','ledger','protocol-sha256','ledger-sha256','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = evaluate(_read(args.protocol), _read(args.ledger), expected_protocol_sha256=args.protocol_sha256,
                      expected_ledger_sha256=args.ledger_sha256)
    output = Path(args.output); output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_new(output/'report.json', result)
    print(_encoded(result).decode('ascii'))


if __name__ == '__main__':
    main()
