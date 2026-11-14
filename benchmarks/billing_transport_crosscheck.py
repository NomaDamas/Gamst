"""Original native/transport linkage; account aggregates never become request bills."""
from .billing_aggregate import build_report, encoded, sha, MAX_REPORT, Diagnostics, validate_manifest
from .native3_evidence import ValidatedNative3Index
from humansense.responses_evidence_reader import read_responses_evidence
from humansense.openai_responses import TRANSPORT
from humansense.openai_responses_codec import PROJECTION
from humansense.native_entry import validate_owner

_BINDING_KEYS = {'owner_kind','owner','provider','model_revision','role','projection_version','transport_version'}
_NATIVE_KEYS = ('schema_version','owner_kind','owner','entry','provider','model_revision','role','prompt_sha256')


def _selection(value, index):
    if type(value) is not dict or set(value)!=_BINDING_KEYS:
        raise ValueError('exact independent transport selection required')
    if value['owner_kind']!='task_native3' or type(index) is not ValidatedNative3Index:
        raise ValueError('original collector native3 index required; other readers must be explicitly integrated')
    validate_owner(value['owner_kind'],value['owner'])
    if value['projection_version']!=PROJECTION or value['transport_version']!=TRANSPORT:
        raise ValueError('supported concrete transport selection required')
    if value['role'] not in ('perception','small_vlm','strong_vlm','text_llm'):
        raise ValueError('known original native role required')
    data=index.snapshot()
    if encoded(value['owner'])!=encoded(data['owner']): raise ValueError('selection owner differs from original native scope')
    provider=dict(provider=value['provider'],model_revision=value['model_revision'])
    if provider not in data['declared_provider_revisions']:
        raise ValueError('selection provider/model absent from original declaration')
    selection=encoded(value)
    if len(selection)>8192: raise ValueError('selection declaration exceeds bound')
    return data,selection


def _query_relation(query, manifest, account_scope, diagnostics, pointer):
    equal=manifest['declared_account_scope']==account_scope
    if not equal: diagnostics.add('declared_account_mismatch',pointer)
    filter_state='compatible'
    # These are comparisons of declared values only, not account isolation.
    available={'project_ids':manifest['declared_project'],'models':manifest['requested_model']}
    for key, values in query['filters'].items():
        if key not in available:
            if filter_state!='different': filter_state='unknown'
        elif available[key] not in values: filter_state='different'
    if filter_state!='compatible':diagnostics.add('declared_filter_'+('mismatch' if filter_state=='different' else 'unknown'),pointer)
    send=next((mark['clock'] for mark in manifest['milestones'] if mark['event']=='possible_send'),None)
    nominal='not_sent' if send is None else 'unknown'
    if send is not None:
        diagnostics.add('unverified_host_clock',pointer)
        if manifest['clock_jump_detected']:diagnostics.add('host_clock_jump',pointer)
        else:
            # Descriptive host timestamp only. Outside does not prove exclusion:
            # no absolute UTC accuracy bound is available in this transport.
            utc=send['utc_ns']
            nominal='intersects' if query['start_time']*10**9<=utc<query['end_time']*10**9 else 'outside'
    return dict(query_id=query['id'],declared_account_relation='equal' if equal else 'different',
        known_filter_relation=filter_state,nominal_host_time_relation=nominal,
        time_assignment_verified=False,request_count_comparable=False)


def crosscheck_transport(manifest, asset_directory, transport_directory, *,
                         expected_inventory_sha256, native_index, expected_binding):
    """Re-read actual aggregate/transport originals and compare the complete lane.

    native3 currently supplies the concrete original reader. native4 support is
    pending integration of its separately reviewed reader, never duck-typed.
    No caller verification callback or hand-authored native dictionary is accepted.
    """
    native,selection_bytes=_selection(expected_binding,native_index)
    manifest=validate_manifest(manifest)
    aggregate,_=build_report(manifest,asset_directory)
    transport=read_responses_evidence(transport_directory,expected_inventory_sha256=expected_inventory_sha256)
    owner=transport.owner_declaration()
    if owner['owner_kind']!=expected_binding['owner_kind'] or owner['owner_sha256']!=sha(encoded(expected_binding['owner'])):
        raise ValueError('original transport owner differs from independent selection')
    lane=[row for row in native['entries'] if all(row[key]==expected_binding[key] for key in ('provider','model_revision','role'))]
    if len(lane)>10000: raise ValueError('selected native inventory exceeds bound')
    native_by_id={row['entry']['operation_id']:row for row in lane}
    if len(native_by_id)!=len(lane): raise ValueError('duplicate original native operation')
    diagnostics=Diagnostics(); matched=set(); referenced=set(); rows=[]; row_bytes=0
    counts={key:0 for key in ('preparations','native_entries_matched','possible_sends','responses_observed',
        'model_responses_completed','not_entered','cancelled','timeout','error','native_entries_without_transport')}
    canonical={view['query_id']:view for view in aggregate['views']}
    queries=[query for query in manifest['queries'] if query['id'] in canonical]
    for original in transport.entries():
        m=original['manifest']; binding=m['entry_binding']; pointer=str(original['ordinal'])
        counts['preparations']+=1
        if counts['preparations']>1024: raise ValueError('transport preparation bound exceeded')
        events={mark['event'] for mark in m['milestones']}
        possible='possible_send' in events; observed='headers' in events
        counts['possible_sends']+=int(possible);counts['responses_observed']+=int(observed)
        counts['model_responses_completed']+=int(m['terminal']=='completed')
        if m['terminal'] in ('not_entered','cancelled','timeout','error'):counts[m['terminal']]+=1
        state='not_entered'; identity=None
        if (m['provider'],m['requested_model'])!=(expected_binding['provider'],expected_binding['model_revision']):
            state='conflict';diagnostics.add('native_entry_conflict',pointer)
        if binding is not None:
            if binding['owner_kind']!='task_native3': raise ValueError('native variant differs from selected reader')
            identity=binding['entry']; operation=identity['operation_id']
            if operation in referenced: raise ValueError('native entry reused by multiple transport preparations')
            referenced.add(operation); actual=native_by_id.get(operation)
            if actual is None:
                state='missing';diagnostics.add('native_entry_missing',pointer)
            elif encoded(binding)!=encoded({key:actual[key] for key in _NATIVE_KEYS}):
                state='conflict';diagnostics.add('native_entry_conflict',pointer)
            else:
                state='matched';matched.add(operation);counts['native_entries_matched']+=1
        relations=[_query_relation(query,m,manifest['account_scope'],diagnostics,pointer+'/'+str(qi)) for qi,query in enumerate(queries)]
        row=dict(ordinal=original['ordinal'],financial_token=m['opaque_financial_token'],origin=m['origin'],
            client_request_id=m['client_request_id'],server_request_id=m['server_request_id'],response_id=m['response_id'],
            owner_kind=expected_binding['owner_kind'],native_entry_identity=identity,
            original_manifest_sha256=original['original_manifest_sha256'],raw_response_sha256=m['raw_response']['sha256'],
            terminal=m['terminal'],possible_send=possible,response_observed=observed,native_match=state,
            query_relations=relations,per_attempt_cost=None)
        row_bytes+=len(encoded(row))+1
        if row_bytes>MAX_REPORT: raise ValueError('expanded crosscheck exceeds report bound')
        rows.append(row)
    if counts['preparations']!=owner['entry_count']: raise ValueError('transport inventory count differs from original reader')
    uncovered=[row['entry'] for row in lane if row['entry']['operation_id'] not in matched]
    counts['native_entries_without_transport']=len(uncovered)
    for row in uncovered:diagnostics.add('native_inventory_uncovered',row['operation_id'])
    for view in aggregate['views']:
        if view['state']=='revision_quarantined':diagnostics.add('nonadditive_query_revision',view['query_id'])
    diagnostics.add('account_isolation_unverified','account')
    diagnostics.add('request_count_semantics_unverified','usage')
    incomplete_native=bool(native['incomplete_reasons'] or native['dispatched_without_invocation'] or native['original_incomplete_reason']
        or not native['producer_reported_instrumentation_complete'] or not native['producer_reported_collection_complete'])
    report=dict(schema_version=1,kind='original_transport_aggregate_crosscheck',
        aggregate_report_sha256=sha(encoded(aggregate)),transport_inventory_sha256=expected_inventory_sha256,
        transport_selection_sha256=sha(selection_bytes),native_sources=dict(reader_version=native['reader_version'],
            owner_kind='task_native3',original_files=native['original_files'],sidecar_ordinal=native['sidecar_ordinal'],
            original_sidecar_sha256=native['original_sidecar_sha256'],original_source_trace_sha256=native['original_source_trace_sha256'],
            original_native_journal_sha256=native['original_native_journal_sha256'],structural_native_verified=True,
            original_channel_frames_verified=False),local_counts=counts,entries=rows,native_entries_without_transport=uncovered,
        query_views=[dict(query_id=view['query_id'],state=view['state']) for view in aggregate['views']],
        local_binding_complete=not incomplete_native and not uncovered and all(row['native_match'] in ('matched','not_entered') for row in rows),
        transport_selection_preexecution_verified=False,original_channel_frames_verified=False,
        instrumentation_coverage_verified=False,account_isolation_verified=False,count_bucket_verified=False,
        request_costs_verified=False,receipt_authenticity_verified=False,whole_pipeline_cost_verified=False,
        actual_acceptance='incomplete',diagnostics=diagnostics.values)
    if len(encoded(report))>MAX_REPORT:raise ValueError('encoded crosscheck exceeds16MiB')
    return report
