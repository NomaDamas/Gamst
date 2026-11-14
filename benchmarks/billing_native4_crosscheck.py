"""Original recording-native4 lane to concrete transport; no monetary inference."""
from pathlib import Path

from .billing_aggregate import build_report, encoded, sha, MAX_REPORT, Diagnostics, validate_manifest
from .billing_transport_crosscheck import _query_relation, _BINDING_KEYS, _NATIVE_KEYS
from .recording_native_schema import decode, MAX_DECLARATIONS, sha as checked_sha
from .recording_native_validation import (ValidatedRecordingIndex, _read,
    validate_recording_native_inventory, canonical_lane_inventory, iter_canonical_operations)
from humansense.native_entry import validate_owner
from humansense.openai_responses import TRANSPORT
from humansense.openai_responses_codec import PROJECTION
from humansense.responses_evidence_reader import read_responses_evidence


def _owned(value, depth=0):
    """Copy only the finite selection shape, before any hash or external read."""
    if type(value) is dict and depth < 3 and len(value) <= 8:
        pairs = list(value.items())
        if any(type(k) is not str or len(k) > 64 for k, _ in pairs):
            raise ValueError('bounded selection keys required')
        return {k: _owned(v, depth+1) for k, v in pairs}
    if type(value) is str and 0 < len(value) <= 256 and len(value.encode()) <= 1024:
        return value
    if type(value) is int and 0 <= value <= 2**63-1:
        return value
    raise ValueError('bounded original recording selection required')


def _original(native_index, declarations_path, declarations_sha256, terminal_sha256):
    if type(native_index) is not ValidatedRecordingIndex or not getattr(native_index, '_sealed', False):
        raise TypeError('original reader-issued native4 index required')
    checked_sha(declarations_sha256); checked_sha(terminal_sha256)
    path = Path(declarations_path)
    if not path.is_absolute() or not path.name:
        raise ValueError('absolute original declaration file required')
    raw = _read(path.parent, path.name, MAX_DECLARATIONS, declarations_sha256)
    declarations = decode(raw)
    # Reopen and validate the original directory once. Supplied hashes are never
    # merely attached to an older in-memory index or an arbitrary terminal.
    fresh = validate_recording_native_inventory(declarations, native_index.directory,
        terminal_sha256, source_inventory=declarations['source_manifest'])
    if encoded(fresh.terminal) != encoded(native_index.terminal) or encoded(fresh.declarations) != encoded(native_index.declarations):
        raise ValueError('original files differ from supplied native index')
    return fresh


def crosscheck_recording_transport(manifest, asset_directory, transport_directory, *,
        expected_inventory_sha256, native_index, expected_binding,
        declarations_path, declarations_sha256, terminal_sha256):
    selection = _owned(expected_binding)
    if type(selection) is not dict or set(selection) != _BINDING_KEYS:
        raise ValueError('exact independent transport selection required')
    if selection['owner_kind'] != 'recording_native4':
        raise ValueError('explicit recording-native4 selection required')
    validate_owner(selection['owner_kind'], selection['owner'])
    if selection['projection_version'] != PROJECTION or selection['transport_version'] != TRANSPORT:
        raise ValueError('supported concrete transport required')
    index = _original(native_index, declarations_path, declarations_sha256, terminal_sha256)
    owner = selection['owner']; si = owner['scope_index']; bi = owner['binding_index']
    lane = canonical_lane_inventory(index, si, bi)
    for key in ('owner_kind', 'owner', 'provider', 'model_revision', 'role'):
        if encoded(selection[key]) != encoded(lane[key]):
            raise ValueError('selection differs from original declared lane')
    selected = {}
    for row in iter_canonical_operations(index, scope_index=si, binding_index=bi):
        ordinal = row['entry']['entry_ordinal']
        if ordinal in selected or len(selected) >= 16384:
            raise ValueError('duplicate or excessive original native entries')
        selected[ordinal] = dict(schema_version=1,
            **{key: row[key] for key in _NATIVE_KEYS if key != 'schema_version'},
            original_chunk=row['source_chunk'], original_event=row['entry_locator'])
    if len(selected) != lane['counts']['entered']:
        raise ValueError('original whole-lane inventory count differs')
    manifest = validate_manifest(manifest)
    aggregate, _ = build_report(manifest, asset_directory)
    transport = read_responses_evidence(transport_directory,
        expected_inventory_sha256=expected_inventory_sha256)
    declared = transport.owner_declaration()
    if declared['owner_kind'] != 'recording_native4' or declared['owner_sha256'] != sha(encoded(owner)):
        raise ValueError('transport owner differs from original selected lane')
    canonical = {view['query_id'] for view in aggregate['views']}
    queries = [q for q in manifest['queries'] if q['id'] in canonical]
    diagnostics = Diagnostics(); rows = []; matched = set(); referenced = set(); row_bytes = 0
    counts = {key: 0 for key in ('preparations', 'native_entries_matched', 'possible_sends',
        'responses_observed', 'model_responses_completed', 'not_entered', 'cancelled', 'timeout', 'error',
        'native_entries_without_transport')}
    for original in transport.entries():
        m = original['manifest']; binding = m['entry_binding']; pointer = str(original['ordinal'])
        counts['preparations'] += 1
        if counts['preparations'] > 1024: raise ValueError('transport inventory exceeds bound')
        events = {mark['event'] for mark in m['milestones']}
        possible = 'possible_send' in events; observed = 'headers' in events
        counts['possible_sends'] += int(possible); counts['responses_observed'] += int(observed)
        counts['model_responses_completed'] += int(m['terminal'] == 'completed')
        if m['terminal'] in ('not_entered', 'cancelled', 'timeout', 'error'): counts[m['terminal']] += 1
        state = 'not_entered'; identity = None; source = None
        if (m['provider'], m['requested_model']) != (selection['provider'], selection['model_revision']):
            state = 'conflict'; diagnostics.add('native_entry_conflict', pointer)
        if binding is not None:
            if binding['owner_kind'] != 'recording_native4': raise ValueError('wrong original native variant')
            identity = binding['entry']; ordinal = identity['entry_ordinal']
            if ordinal in referenced: raise ValueError('native entry reused by transport preparations')
            referenced.add(ordinal); actual = selected.get(ordinal)
            if actual is None:
                state = 'missing'; diagnostics.add('native_entry_missing', pointer)
            elif state == 'conflict' or encoded(binding) != encoded({k: actual[k] for k in _NATIVE_KEYS}):
                state = 'conflict'; diagnostics.add('native_entry_conflict', pointer)
            else:
                state = 'matched'; matched.add(ordinal); counts['native_entries_matched'] += 1
                source = dict(chunk=actual['original_chunk'], event=actual['original_event'])
        row = dict(ordinal=original['ordinal'], financial_token=m['opaque_financial_token'], origin=m['origin'],
            client_request_id=m['client_request_id'], server_request_id=m['server_request_id'], response_id=m['response_id'],
            native_entry_identity=identity, original_native_locator=source,
            original_manifest_sha256=original['original_manifest_sha256'], raw_response_sha256=m['raw_response']['sha256'],
            terminal=m['terminal'], possible_send=possible, response_observed=observed, native_match=state,
            query_relations=[_query_relation(q, m, manifest['account_scope'], diagnostics, pointer+'/'+str(n))
                for n, q in enumerate(queries)], per_attempt_cost=None)
        row_bytes += len(encoded(row))+1
        if row_bytes > MAX_REPORT: raise ValueError('crosscheck report exceeds bound')
        rows.append(row)
    if counts['preparations'] != declared['entry_count']: raise ValueError('transport inventory count differs')
    uncovered = [row['entry'] for ordinal, row in selected.items() if ordinal not in matched]
    counts['native_entries_without_transport'] = len(uncovered)
    for row in uncovered: diagnostics.add('native_inventory_uncovered', str(row['entry_ordinal']))
    for view in aggregate['views']:
        if view['state'] == 'revision_quarantined': diagnostics.add('nonadditive_query_revision', view['query_id'])
    diagnostics.add('account_isolation_unverified', 'account')
    diagnostics.add('request_count_semantics_unverified', 'usage')
    complete = not (lane['unobserved_entry_inventory_possible'] or lane['counts']['outstanding_tickets']
        or lane['outstanding_entries'] or uncovered)
    report = dict(schema_version=2, kind='original_transport_aggregate_crosscheck',
        aggregate_report_sha256=sha(encoded(aggregate)), transport_inventory_sha256=expected_inventory_sha256,
        transport_selection_sha256=sha(encoded(selection)),
        native_sources=dict(reader_version=4, owner_kind='recording_native4', owner=owner,
            declarations_file_sha256=declarations_sha256, terminal_sha256=terminal_sha256,
            original_index_tail_sha256=index.terminal['index_tail_sha256'],
            source_inventory_locator=lane['source_inventory_locator'], structural_native_verified=True),
        native_lane=lane, local_counts=counts, entries=rows, native_entries_without_transport=uncovered,
        query_views=[dict(query_id=v['query_id'], state=v['state']) for v in aggregate['views']],
        local_binding_complete=complete and all(r['native_match'] in ('matched','not_entered') for r in rows),
        transport_selection_preexecution_verified=False, original_channel_frames_verified=False,
        instrumentation_coverage_verified=False, account_isolation_verified=False, count_bucket_verified=False,
        request_costs_verified=False, receipt_authenticity_verified=False, whole_pipeline_cost_verified=False,
        actual_acceptance='incomplete', diagnostics=diagnostics.values)
    if len(encoded(report)) > MAX_REPORT: raise ValueError('encoded crosscheck exceeds16MiB')
    return report
