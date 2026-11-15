"""Full observed-lifetime ledger v2 and immutable financial closure.

Only a complete typed recording/native join can supply operation ownership.
All actual entries are required, including failures, cancellations and discarded
publication. Synthetic F1 can establish fixture arithmetic, never real money.
"""
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
from types import MappingProxyType

from gamst.capabilities import identifier
from .recording_financial import (FinancialAdapters,FinancialRunView,N63,PAGE_BYTES,array,
    claim_sha256,digest,encode,enum,fraction,integer,iter_financial_rows,keys,nullable,
    operation_key,ratio,rational,reference,sha,snapshot,validate_adapters,
    validate_finality,validate_financial_evidence,validate_invoice)
from .matched_source_exposure import validate_accounting_policy,validate_exposure_policy
from .perception_comparison import plan_digest

LIFETIME_FIELDS = ('schema_version','kind','execution_sha256','run_inventory_sha256','stream_id',
    'clock_domain','activation_started_ns','child_spawn_started_ns','playback_anchor_ns','source_finished_ns',
    'child_cleanup_finished_ns','process_reaped_ns','parent_resources_closed_ns','utc_anchor',
    'external_work_closed','worker_inventory_sha256','resource_intervals','coverage_sha256','state')
COST_FIELDS = ('stream_id','component_id','resource_id','lane_id','start_ns','end_ns','event_ns','identity',
    'source_quantity','quantity','status','billed_amount','evidence_sha256','operation_instance',
    'lifecycle_work_ref','received_ns','financial_evidence_sha256')
CLOSURE_FIELDS = ('schema_version','kind','execution_sha256','stream_id','accounting_policy_sha256',
    'financial_adapter_inventory_sha256','operational_end_ns','cutoff_ns','sealed_ns','operation_count',
    'known_settled_operations','unresolved_operations','required_lane_count','closed_lane_count',
    'pending_line_count','unknown_component_count','external_finality_refs','financial_inventory_sha256','state')
SETTLEMENT_FIELDS = ('schema_version','kind','execution_sha256','instance','controller_id','request_id',
    'financial_evidence_sha256','unit','pricing_revision','components','charged','observed_ns','controller_transition_sha256')
_TOTAL_ISSUER = object()


def _schema(value,version,kind):
    if type(value['schema_version']) is not int or value['schema_version'] != version or value['kind'] != kind:
        raise ValueError('explicit '+kind+' schema required')


def _rounded(value):
    value = fraction(value)
    quotient,remainder = divmod(value.numerator,value.denominator)
    return quotient+int(2*remainder > value.denominator or (2*remainder == value.denominator and quotient%2))


def _artifact(ref,value):
    reference(ref)
    data = encode(value,128*1024**2,newline=True)
    if ref['bytes'] != len(data) or ref['file_sha256'] != hashlib.sha256(data).hexdigest():
        raise ValueError('artifact reference differs from exact canonical file bytes')


def validate_lifetime(value,run):
    if type(run) is not FinancialRunView: raise TypeError('complete trusted recording join required')
    value = snapshot(value)
    keys(value,LIFETIME_FIELDS,'observed lifetime')
    _schema(value,2,'recording_observed_lifetime')
    for key in ('execution_sha256','run_inventory_sha256','stream_id','clock_domain',
                'activation_started_ns','child_spawn_started_ns','playback_anchor_ns','source_finished_ns',
                'child_cleanup_finished_ns','process_reaped_ns','parent_resources_closed_ns',
                'external_work_closed','worker_inventory_sha256'):
        if value[key] != getattr(run,key) or type(value[key]) is not type(getattr(run,key)):
            raise ValueError('lifetime differs from actual parent-owned observation: '+key)
    if value['coverage_sha256'] != run.source_coverage_sha256:
        raise ValueError('source coverage binding differs')
    reference(value['resource_intervals'])
    keys(value['utc_anchor'],('monotonic_ns','unix_ns','evidence_sha256'),'actual UTC anchor')
    integer(value['utc_anchor']['monotonic_ns']); integer(value['utc_anchor']['unix_ns']); sha(value['utc_anchor']['evidence_sha256'])
    previous = value['activation_started_ns']
    for key in ('child_spawn_started_ns','playback_anchor_ns','source_finished_ns',
                'child_cleanup_finished_ns','process_reaped_ns','parent_resources_closed_ns'):
        current = value[key]
        if current is not None:
            integer(current)
            if current < previous: raise ValueError('ordered actual parent lifetime milestones required')
            previous = current
    enum(value['state'],('closed','incomplete'))
    if value['state'] == 'closed' and (not run.operational_closed or not run.external_work_closed or
            not run.clock_consistent or run.parent_resources_closed_ns is None):
        raise ValueError('child terminal/cancellation cannot certify full work closure')
    if run.parent_resources_closed_ns is not None and run.parent_resources_closed_ns <= run.activation_started_ns:
        raise ValueError('positive observed lifetime required')
    return value


def validate_controller_settlement(value,*,run,operation,evidence,policy):
    value = snapshot(value)
    keys(value,SETTLEMENT_FIELDS,'actual controller settlement')
    _schema(value,1,'recording_controller_settlement')
    if (value['execution_sha256'] != run.execution_sha256 or value['instance'] != operation['instance'] or
            value['financial_evidence_sha256'] != digest(evidence) or
            value['unit'] != policy['currency_unit'] or value['pricing_revision'] != policy['rate_revision'] or
            value['charged'] is not True):
        raise ValueError('actual controller settlement owner/unit/evidence differs')
    identifier(value['controller_id']); identifier(value['request_id']); sha(value['controller_transition_sha256'])
    integer(value['observed_ns'])
    components = value['components']
    if type(components) is not dict or not components or not set(components) <= {'input','output','cache_read','cache_write','media','text_consumer','local_compute','other'}:
        raise ValueError('exact known controller billing components required')
    if sum(integer(amount,maximum=10**18) for amount in components.values()) != evidence['billed_atomic']:
        raise ValueError('actual controller settlement differs from original billed charge')
    # The origin of this object must also be retained by the runtime's SDK-owned
    # recorder. A hash of arbitrary JSON never upgrades an unknown controller.
    return value


def validate_validity_chain(entries,checkpoint,*,expected_checkpoint_sha256,exposure_policy_sha256,
                            ledger_sha256,sealed_ns,notices):
    """Validate a pinned external as-of checkpoint; never infer no later files."""
    checkpoint = snapshot(checkpoint)
    if digest(checkpoint) != sha(expected_checkpoint_sha256):
        raise ValueError('current validity checkpoint differs from externally reviewed digest')
    keys(checkpoint,('schema_version','kind','exposure_policy_sha256','ledger_sha256','tail_sha256',
                     'tail_sequence','reviewed_as_of_ns','reviewer','review_method_sha256'),'validity checkpoint')
    _schema(checkpoint,1,'recording_validity_checkpoint')
    if checkpoint['exposure_policy_sha256'] != exposure_policy_sha256 or checkpoint['ledger_sha256'] != ledger_sha256:
        raise ValueError('validity checkpoint frozen ledger binding differs')
    identifier(checkpoint['reviewer']); sha(checkpoint['review_method_sha256']); sha(checkpoint['tail_sha256'])
    integer(checkpoint['tail_sequence'],maximum=131071); integer(checkpoint['reviewed_as_of_ns'])
    if type(entries) is not list or not 1 <= len(entries) <= 131072 or len(entries) != checkpoint['tail_sequence']+1:
        raise ValueError('complete bounded genesis-to-reviewed-tail chain required')
    previous = None; previous_time = sealed_ns; state = 'sealed'; known = set()
    for sequence,pair in enumerate(entries):
        if type(pair) is not tuple or len(pair) != 2: raise ValueError('retained validity reference/object pair required')
        ref,value = pair; _artifact(ref,value)
        keys(value,('schema_version','kind','exposure_policy_sha256','ledger_sha256','sequence',
                    'previous_sha256','notice','state','observed_ns'),'validity entry')
        _schema(value,1,'recording_validity_entry')
        if (value['exposure_policy_sha256'] != exposure_policy_sha256 or value['ledger_sha256'] != ledger_sha256 or
                type(value['sequence']) is not int or value['sequence'] != sequence or
                value['previous_sha256'] != previous or value['observed_ns'] < previous_time or
                integer(value['observed_ns']) > checkpoint['reviewed_as_of_ns']):
            raise ValueError('validity chain identity/order/time mismatch')
        if sequence == 0:
            if value['notice'] is not None or value['state'] != 'sealed': raise ValueError('mandatory zero-notice sealed genesis required')
        else:
            notice_ref = reference(value['notice'])
            if value['state'] != 'invalidated' or notice_ref['file_sha256'] not in notices:
                raise ValueError('late evidence requires immutable invalidation, never reopening')
            notice = notices[notice_ref['file_sha256']]; _artifact(notice_ref,notice)
            keys(notice,('schema_version','kind','sealed_ledger_sha256','closure_sha256','stream_id',
                         'noticed_ns','financial_evidence','reason'),'late cost notice')
            _schema(notice,1,'recording_late_cost_notice')
            if notice['sealed_ledger_sha256'] != ledger_sha256 or integer(notice['noticed_ns']) > value['observed_ns']:
                raise ValueError('late notice ledger/time mismatch')
            sha(notice['closure_sha256']); identifier(notice['stream_id']); reference(notice['financial_evidence'])
            enum(notice['reason'],('new_charge','correction','duplicate_conflict','finality_breach'))
            if notice_ref['file_sha256'] in known: raise ValueError('late notice cannot be counted twice')
            known.add(notice_ref['file_sha256']); state = 'invalidated'
        previous = ref['file_sha256']; previous_time = value['observed_ns']
    if previous != checkpoint['tail_sha256'] or set(notices) != known:
        raise ValueError('current checkpoint omits known late notice or pins another tail')
    return dict(state=state,validity_tail_sha256=previous,validity_checkpoint_sha256=expected_checkpoint_sha256,
                valid_as_of_ns=checkpoint['reviewed_as_of_ns'])


@dataclass(frozen=True)
class ArmLifetimeTotal:
    run: FinancialRunView
    lifetime_sha256: str
    closure_sha256: str
    exposure_duration_ns: int
    actual_elapsed_ns: object
    known_monetary_atomic: int
    full_monetary_atomic: object
    reasons: tuple


@dataclass(frozen=True,init=False)
class ValidatedLifetimeLedger:
    policy_sha256: str
    exposure_sha256: str
    ledger_sha256: str
    runtime_policy_sha256: str
    collection_sha256: str
    scope_protocol_sha256: str
    instrumentation_protocol_sha256: str
    execution_inventory_sha256: str
    execution_receipts: tuple
    run_inventory: tuple
    totals: tuple
    evidence_class: str
    validity: object

    def __init__(self,*,policy_sha256,exposure_sha256,ledger_sha256,runtime_policy_sha256,collection_sha256,
                 scope_protocol_sha256,instrumentation_protocol_sha256,execution_inventory_sha256,
                 execution_receipts,run_inventory,totals,evidence_class,validity,_issuer=None):
        if _issuer is not _TOTAL_ISSUER: raise TypeError('validated full lifetime ledger required')
        validity = MappingProxyType(dict(validity))
        for name,value in locals().copy().items():
            if name not in ('self','_issuer'): object.__setattr__(self,name,value)


def _native_inventory(indices,runs,expected_context,*,budget=None):
    from .recording_native_validation import ValidatedRecordingIndex,iter_canonical_inventory
    from .recording_native_schema import encoded as native_encoded
    if len(indices) != len(runs): raise ValueError('one validated native inventory per original run required')
    operations = {}; work = {}; keys_to_instance = {}; ordinal = {}; run_by_invocation = {r.invocation_ordinal:r for r in runs}
    for index,run in zip(indices,runs):
        if run.native_terminal_sha256 is None:
            if index is not None: raise ValueError('missing original terminal cannot accept a substituted native inventory')
            # This is an explicitly unknown prefix, never an empty verified run.
            # Its frozen arm remains in all financial inventories and totals.
            continue
        if type(index) is not ValidatedRecordingIndex: raise TypeError('native4 typed validated inventory required')
        if budget is not None:
            budget.reserve_index((index.declarations,index.entries,index.hashes,index.flags,index.settlements,index.chunk_refs))
        if hashlib.sha256(native_encoded(index.terminal)+b'\n').hexdigest() != run.native_terminal_sha256:
            raise ValueError('financial join substituted another actual native terminal')
        context = index.terminal['context']
        if context['invocation_ordinal'] != run.invocation_ordinal or any(context[name] != expected for name,expected in expected_context.items()):
            raise ValueError('native terminal crosses original frozen run')
        for value in iter_canonical_inventory(index):
            if budget is not None: budget.reserve_index(value)
            context = value['owner']['context'] if value['kind'] == 'operation' else value['context']
            if context['invocation_ordinal'] != run.invocation_ordinal: raise ValueError('native inventory crosses original run')
            if any(context[name] != expected for name,expected in expected_context.items()):
                raise ValueError('native inventory crosses frozen collection/scope/instrumentation')
            if value['kind'] == 'operation':
                if value['source_inventory_locator']['source_manifest_sha256'] != run.source_manifest_sha256:
                    raise ValueError('native source locator crosses original recording')
                instance = sha(value['instance']); binding = integer(value['binding_index'],maximum=63)
                if instance in operations: raise ValueError('duplicate original native operation')
                lane = (run.invocation_ordinal,binding); rank = ordinal.get(lane,0); ordinal[lane] = rank+1
                key = lane+(rank,)
                operations[instance] = value; keys_to_instance[key] = instance
            elif value['kind'] == 'worker_interval':
                key = (run.invocation_ordinal,value['worker_ordinal'],value['work_ordinal'])
                if key in work: raise ValueError('duplicate original work interval')
                work[key] = value
            else: raise ValueError('unknown canonical native inventory item')
        actual_work = sorted((row for key,row in work.items() if key[0] == run.invocation_ordinal),
                             key=lambda row:(row['work_ordinal'],row['worker_ordinal']))
        if digest(actual_work) != run.worker_inventory_sha256:
            raise ValueError('actual complete worker inventory digest differs')
    return operations,work,keys_to_instance


def _financial_inventory(store,value,*,adapters,policy,exposure,collection_sha256,operations,keys_to_instance,runs):
    keys(value,('schema_version','kind','exposure_policy_sha256','accounting_policy_sha256',
                'financial_adapter_inventory_sha256','collection_sha256','source_documents','evidence_pages',
                'operation_pages','lane_closure_pages','finality','state'),'financial inventory')
    _schema(value,1,'recording_financial_inventory')
    if (value['exposure_policy_sha256'] != digest(exposure) or value['accounting_policy_sha256'] != digest(policy) or
            value['financial_adapter_inventory_sha256'] != adapters.digest or value['collection_sha256'] != collection_sha256):
        raise ValueError('financial inventory frozen declaration mismatch')
    enum(value['state'],('complete','incomplete'))
    global_rows = global_pages = 0
    for name in ('source_documents','evidence_pages','operation_pages','lane_closure_pages','finality'):
        index = value[name]
        keys(index,('row_count','page_count','index_page_count','index_tail'),'financial index')
        global_rows += integer(index['row_count'],maximum=131072)
        global_pages += integer(index['page_count'],maximum=1024)+integer(index['index_page_count'],maximum=1024)
    if global_rows > 131072 or global_pages > 1024: raise ValueError('global financial inventory capacity exceeded')
    documents = {}; logical_documents = {}; original_lines = {}; source_bytes = 0
    for row in iter_financial_rows(store,'source',value['source_documents']):
        keys(row,('source_document',),'original source document row'); ref = reference(row['source_document'])
        if ref['file_sha256'] in documents: raise ValueError('duplicate original source document')
        document = validate_invoice(store.read(ref))
        if any(document[k]!=policy[k] for k in ('currency','currency_unit')):
            raise ValueError('original invoice denomination differs from frozen accounting policy')
        documents[ref['file_sha256']] = (ref,document)
        store.reserve_index((ref,document))
        source_bytes += ref['bytes']
        if source_bytes > 64*1024**2: raise ValueError('original financial source byte cap exceeded')
        logical = tuple(document[k] for k in ('provider','account_scope','invoice_id'))
        binding = (document['revision'],ref['file_sha256'])
        if logical in logical_documents and logical_documents[logical] != binding:
            raise ValueError('same logical invoice changed original document/revision')
        logical_documents[logical] = binding
        for line in document['lines']:
            claim = claim_sha256(document['provider'],document['account_scope'],document['invoice_id'],line['line_id'])
            if claim in original_lines: raise ValueError('original invoice line cannot acquire another owner')
            original_lines[claim] = (ref,document,line)
    evidence = {}; claims = set(); used_lines = set(); unverified_instances = {}
    runmap = {run.invocation_ordinal:run for run in runs}
    for row in iter_financial_rows(store,'evidence',value['evidence_pages']):
        ref = reference(row['source_document'])
        if ref['file_sha256'] not in documents: raise ValueError('financial evidence omits original document')
        original_ref,invoice = documents[ref['file_sha256']]
        row = validate_financial_evidence(row,adapters=adapters,invoice=invoice,source_reference=original_ref)
        store.reserve_index(row)
        claim = row['claim_sha256']; object_hash = digest(row)
        if claim in claims or object_hash in evidence: raise ValueError('original claim/evidence counted more than once')
        claims.add(claim); evidence[object_hash] = row; used_lines.add(claim)
        line = original_lines[claim][2]
        key = nullable(line['operation_key'],operation_key)
        if key is not None:
            if key not in keys_to_instance:
                run = runmap.get(key[0])
                if run is None or run.native_terminal_sha256 is not None or row['stream_id'] != run.stream_id:
                    raise ValueError('fixture operation key does not resolve bijectively to actual entered native operation')
                instance = sha(row['owner_instance'])
                if instance in unverified_instances or instance in operations:
                    raise ValueError('unknown-prefix original charge owner counted twice')
                unverified_instances[instance] = run.invocation_ordinal
            elif row['owner_instance'] != keys_to_instance[key]:
                raise ValueError('fixture operation key crosses actual entered native operation')
        if key is None and row['owner_instance'] is not None: raise ValueError('lifecycle meter line cannot pretend to own a semantic operation')
    if used_lines != set(original_lines): raise ValueError('orphan or omitted original invoice line')
    financial_operations = {}; runmap = {run.invocation_ordinal:run for run in runs}; actual_settlements = {}
    for row in iter_financial_rows(store,'operations',value['operation_pages']):
        keys(row,('instance','invocation_ordinal','binding_index','entry_ordinal','status','basis','evidence',
                  'controller_settlement','meter_lane'),'financial operation')
        instance = sha(row['instance'])
        if instance not in operations or instance in financial_operations: raise ValueError('every actual native entry required exactly once')
        operation = operations[instance]
        if any(row[key] != operation[key] or type(row[key]) is not int for key in ('invocation_ordinal','binding_index','entry_ordinal')):
            raise ValueError('financial operation crosses original native entry')
        enum(row['status'],('settled','pending','unknown')); enum(row['basis'],('original_line','meter_lane','unknown'))
        if row['basis'] == 'unknown':
            if any(row[k] is not None for k in ('evidence','controller_settlement','meter_lane')) or row['status'] == 'settled':
                raise ValueError('unknown financial operation cannot contain an optimistic charge')
        elif row['basis'] == 'original_line':
            ref = reference(row['evidence']); obj = store.read(ref)
            if digest(obj) not in evidence or evidence[digest(obj)] != obj or obj['owner_instance'] != instance or row['meter_lane'] is not None:
                raise ValueError('financial operation requires its single original line')
            semantic = operation['role'] in ('small_vlm','strong_vlm','text_llm')
            if semantic:
                settlement = store.read(reference(row['controller_settlement']))
                validate_controller_settlement(settlement,run=runmap[row['invocation_ordinal']],operation=operation,evidence=obj,policy=policy)
                actual_settlements.setdefault(row['invocation_ordinal'],[]).append(settlement)
            elif row['controller_settlement'] is not None:
                raise ValueError('nonsemantic callback cannot manufacture controller settlement')
        else:
            if row['evidence'] is not None or row['controller_settlement'] is not None or operation['role'] in ('small_vlm','strong_vlm','text_llm'):
                raise ValueError('meter lane cannot conceal semantic controller liability')
            keys(row['meter_lane'],('stream_id','component_id','resource_id','lane_id'),'operation meter lane')
            for item in row['meter_lane'].values(): identifier(item)
            if row['meter_lane']['stream_id'] != runmap[row['invocation_ordinal']].stream_id:
                raise ValueError('operation meter crosses original run stream')
        financial_operations[instance] = row
        store.reserve_index(row)
    if set(financial_operations) != set(operations): raise ValueError('omitted failed/cancelled/discarded native operation')
    for run in runs:
        if run.native_terminal_sha256 is None:
            # The missing native join cannot certify absent settlement rows.
            # All original evidence remains known-prefix/incomplete below.
            continue
        actual = sorted(actual_settlements.get(run.invocation_ordinal,[]),key=lambda row:row['instance'])
        if digest(actual) != run.controller_settlement_inventory_sha256:
            raise ValueError('controller settlement attachment differs from actual SDK-issued inventory')
    lanes = {}; locator_pages = locator_rows = 0
    for row in iter_financial_rows(store,'lanes',value['lane_closure_pages']):
        keys(row,('stream_id','component_id','resource_id','lane_id','window_start_ns','window_end_ns','status',
                  'cost_line_inventory','work_inventory_sha256'),'financial lane')
        lane = tuple(identifier(row[key]) for key in ('stream_id','component_id','resource_id','lane_id'))
        if lane in lanes: raise ValueError('duplicate financial resource lane')
        integer(row['window_start_ns']); integer(row['window_end_ns'])
        if row['window_start_ns'] >= row['window_end_ns']: raise ValueError('positive actual resource window required')
        enum(row['status'],('closed','pending','unknown')); sha(row['work_inventory_sha256'])
        index = row['cost_line_inventory']; locator_pages += integer(index['page_count'])+integer(index['index_page_count']); locator_rows += integer(index['row_count'])
        lanes[lane] = row
        store.reserve_index(row)
    if global_pages+locator_pages > 1024 or global_rows+locator_rows > 131072:
        raise ValueError('financial nested locator pages exceed global capacity')
    finality = {}; used_finality = set()
    for row in iter_financial_rows(store,'finality',value['finality']):
        keys(row,('finality',),'financial finality row'); ref = reference(row['finality'])
        obj = validate_finality(store.read(ref)); h = digest(obj)
        if h in finality or obj['stream_id'] in used_finality: raise ValueError('unique original stream finality required')
        finality[h] = (ref,obj); used_finality.add(obj['stream_id'])
        store.reserve_index((ref,obj))
    return dict(documents=documents,original_lines=original_lines,evidence=evidence,operations=financial_operations,
                lanes=lanes,finality=finality,unverified_instances=unverified_instances,settlements=actual_settlements,state=value['state'])


def _validate_costs(store,page_refs,*,policy,runs,lifetimes,financial,operations,work,collection=None):
    """Exact original-line arithmetic plus all required event/interval lanes."""
    components = {c['id']:c for c in policy['components']}; streams = {s['stream_id']:s for s in policy['streams']}
    runmap = {r.stream_id:r for r in runs}; runordinal = {r.invocation_ordinal:r for r in runs}
    totals = {sid:0 for sid in streams}; reasons = {sid:set() for sid in streams}
    lines = {}; claims = set(); identities = set(); measured = {}; operation_lines = {}; work_lines = {}
    locator_positions = {}; lane_positions = {}; component_entries = {}; stream_entries = {}
    operation_work = {}
    for (invocation,_,_),interval in work.items():
        if interval['parent_entry_ordinal'] is not None:
            operation_work.setdefault((invocation,interval['binding'],interval['parent_entry_ordinal']),[]).append(interval)
    previous = None; entries = 0
    for ordinal,ref in enumerate(array(page_refs,1024)):
        if reference(ref)['path'] != 'ledger/cost-%06d.json'%ordinal: raise ValueError('fixed contiguous cost page path required')
        page = store.read(ref)
        keys(page,('schema_version','kind','page_ordinal','previous_page_sha256','entries'),'cost page')
        _schema(page,2,'recording_cost_lines')
        if type(page['page_ordinal']) is not int or page['page_ordinal'] != ordinal or page['previous_page_sha256'] != previous:
            raise ValueError('cost page identity/hash chain differs')
        for entry_index,line in enumerate(array(page['entries'],128,1)):
            entries += 1
            if entries > 131072: raise ValueError('full original cost-line inventory exceeds bound')
            keys(line,COST_FIELDS,'cost line2')
            sid,cid = line['stream_id'],line['component_id']
            if type(sid) is not str or sid not in streams or type(cid) is not str or cid not in components:
                raise ValueError('unknown frozen stream/component')
            run = runmap[sid]; component = components[cid]; requirement = streams[sid]['requirements'][cid]
            key = (sid,cid,identifier(line['resource_id']),identifier(line['lane_id']))
            frozen_lane = next((lane for lane in requirement['lanes'] if (lane['resource_id'],lane['lane_id']) == key[2:]),None)
            if requirement['state'] != 'required' or frozen_lane is None: raise ValueError('cost requires a frozen applicable lane')
            declared = financial['lanes'].get(key)
            if declared is None:
                raise ValueError('all cost lines require explicit retained financial lane inventory')
            start,end = declared['window_start_ns'],declared['window_end_ns']
            if component['coverage'] == 'interval':
                begin = integer(line['start_ns']); finish = integer(line['end_ns'])
                if line['event_ns'] is not None or not start <= begin < finish <= end: raise ValueError('interval outside actual observed resourcewindow')
                measured.setdefault(key,[]).append((begin,finish))
            else:
                event = integer(line['event_ns'])
                if line['start_ns'] is not None or line['end_ns'] is not None or not start <= event < end:
                    raise ValueError('event outside actual observed resourcewindow')
                measured.setdefault(key,[]).append((event,event))
            instance = nullable(line['operation_instance'],sha)
            lifecycle = line['lifecycle_work_ref']
            if instance is not None and lifecycle is not None: raise ValueError('one original operation or lifecycle owner only')
            if instance is not None:
                operation_invocation = (operations[instance]['invocation_ordinal'] if instance in operations else
                    financial.get('unverified_instances',{}).get(instance))
                if operation_invocation != run.invocation_ordinal:
                    raise ValueError('cost line crosses original actual operation/run')
                if instance not in operations: reasons[sid].add('missing_native_prefix')
                operation_lines.setdefault(instance,[]).append((ordinal,entry_index))
            elif lifecycle is not None:
                keys(lifecycle,('invocation_ordinal','worker_ordinal','work_ordinal'),'actual lifecycle owner')
                lifecycle_key = tuple(integer(lifecycle[k]) for k in ('invocation_ordinal','worker_ordinal','work_ordinal'))
                if lifecycle_key not in work or lifecycle_key[0] != run.invocation_ordinal:
                    raise ValueError('unknown/cross-run actual lifecycle owner')
                work_lines.setdefault(lifecycle_key,[]).append((ordinal,entry_index))
            # Whole-lifetime overhead can have no callback owner only under its
            # explicit frozen meter lane. Event work may never vanish this way.
            elif frozen_lane['coverage'] != 'whole_lifetime' or component['coverage'] != 'interval':
                raise ValueError('unowned cost only allowed for frozen whole-lifetime interval meter')
            identity = line['identity']
            if type(identity) is not dict: raise ValueError('original invoice/local identity required')
            kind = enum(identity.get('kind'),('invoice','local'))
            claim = None
            if kind == 'invoice':
                keys(identity,('kind','provider','revision','account_scope','invoice_id','document_sha256','line_id'),'original invoice identity')
                for field in ('provider','revision','account_scope','invoice_id','line_id'): identifier(identity[field])
                sha(identity['document_sha256'])
                claim = claim_sha256(*(identity[k] for k in ('provider','account_scope','invoice_id','line_id')))
                if claim in claims: raise ValueError('one original line cannot have two cost owners')
                claims.add(claim)
                if claim not in financial['original_lines'] or component['kind'] != 'monetary':
                    raise ValueError('recognized original monetary line required')
                original_ref,document,original_line = financial['original_lines'][claim]
                if identity['document_sha256'] != original_ref['file_sha256'] or identity['revision'] != document['revision']:
                    raise ValueError('original invoice document/revision rebound')
                rate = policy['rates'][component['rate_id']]
                if (identity['provider'],identity['revision']) != (rate['provider'],rate['revision']):
                    raise ValueError('original invoice differs from frozen rate identity')
                for field in ('stream_id','component_id','resource_id','lane_id','source_quantity','quantity'):
                    if line[field] != original_line[field]: raise ValueError('cost line changed original invoice field')
                for target,source in (('start_ns','service_start_ns'),('end_ns','service_end_ns'),('event_ns','service_event_ns')):
                    if line[target] != original_line[source]: raise ValueError('receipt arrival cannot replace service time')
            else:
                keys(identity,('kind','meter_id','measurement_id'),'local meter identity')
                marker = (identifier(identity['meter_id']),identifier(identity['measurement_id']))
                if marker in identities: raise ValueError('local meter measurement counted twice')
                identities.add(marker)
            status = line['status']; monetary = component['kind'] == 'monetary'
            enum(status,('settled','pending','unknown') if monetary else ('measured','unknown'))
            known = status in ('settled','measured')
            quantity = nullable(line['quantity'],integer); billed = nullable(line['billed_amount'],integer)
            source_quantity = None if line['source_quantity'] is None else rational(line['source_quantity'])
            nullable(line['evidence_sha256'],sha); nullable(line['financial_evidence_sha256'],sha); nullable(line['received_ns'],integer)
            if source_quantity is not None:
                converted = fraction(source_quantity*rational(component['atomic_per_source']))
                if quantity is None or converted.denominator != 1 or converted.numerator != quantity:
                    raise ValueError('source-unit normalization must remain exact')
            elif quantity is not None: raise ValueError('known atomic quantity needs original source quantity')
            if known and (quantity is None or line['evidence_sha256'] is None): raise ValueError('known quantity needs retained evidence')
            if not known:
                reasons[sid].add('pending_or_unknown_component')
                if billed is not None: raise ValueError('unknown charge cannot contain known billed money')
            if not monetary:
                if billed is not None or kind != 'local': raise ValueError('proxy cannot contain original billed money')
                reasons[sid].add('required_proxy_component')
            elif known:
                if billed is None or kind != 'invoice' or line['financial_evidence_sha256'] not in financial['evidence']:
                    raise ValueError('settled money requires original supported financial evidence')
                evidence = financial['evidence'][line['financial_evidence_sha256']]
                if (evidence['claim_sha256'] != claim or evidence['owner_instance'] != instance or
                        evidence['observed_received_ns'] != line['received_ns'] or billed != evidence['billed_atomic']):
                    raise ValueError('cost/receipt/owner/arrival original line mismatch')
                if _rounded(fraction(quantity*Fraction(Decimal(policy['rates'][component['rate_id']]['per_unit'])))) != billed:
                    reasons[sid].add('frozen_line_rounding_mismatch')
                totals[sid] = integer(totals[sid]+billed)
            lines[ordinal,entry_index] = (ref,line)
            store.reserve_index((ref,line))
            locator_positions[ref['file_sha256'],entry_index] = (ordinal,entry_index)
            lane_positions.setdefault(key,set()).add((ordinal,entry_index))
            component_entries.setdefault((sid,cid),[]).append(line)
            stream_entries.setdefault(sid,[]).append(line)
        previous = ref['file_sha256']
    if claims != set(financial['original_lines']): raise ValueError('omitted or extra original invoice cost line')
    expected_lanes = set()
    for sid,stream in streams.items():
        run = runmap[sid]; lifetime = lifetimes[sid]
        for cid,requirement in stream['requirements'].items():
            for lane in requirement['lanes']:
                key = (sid,cid,lane['resource_id'],lane['lane_id']); expected_lanes.add(key)
                declared = financial['lanes'].get(key)
                if declared is None:
                    reasons[sid].add('missing_required_lane'); continue
                start,end = declared['window_start_ns'],declared['window_end_ns']
                if lane['coverage'] == 'whole_lifetime':
                    if start != run.activation_started_ns or end != run.parent_resources_closed_ns:
                        reasons[sid].add('incomplete_full_lifetime_window')
                    if declared['work_inventory_sha256'] != run.worker_inventory_sha256:
                        reasons[sid].add('unverified_lifecycle_window')
                else:
                    if collection is None: raise ValueError('frozen provider binding table required for lifecycle lanes')
                    bindings = next(a['provider_bindings'] for a in collection['arms'] if a['arm'] == run.arm)
                    matching = [i for i,b in enumerate(bindings) if b['binding_id'] == lane['lifecycle_binding_id']]
                    if len(matching) != 1: raise ValueError('lifecycle lane must resolve one exact frozen provider binding')
                    selected = sorted((row for key,row in work.items() if key[0] == run.invocation_ordinal and
                        row['binding'] == matching[0]),key=lambda row:(row['work_ordinal'],row['worker_ordinal']))
                    if digest(selected) != declared['work_inventory_sha256'] or run.native_terminal_sha256 is None:
                        reasons[sid].add('unverified_lifecycle_window')
                    if not selected: reasons[sid].add('missing_actual_lifecycle_work')
                    for row in selected:
                        if row['clock_domain'] != 'host_monotonic_ns' or row['ended_ns'] is None:
                            reasons[sid].add('unclosed_actual_lifecycle_work')
                        elif not start <= row['started_ns'] <= row['ended_ns'] <= end:
                            raise ValueError('financial lifecycle window excludes actual selected work')
                if declared['status'] != 'closed': reasons[sid].add('unclosed_required_lane')
                locators = iter_financial_rows(store,'line-locators',declared['cost_line_inventory'])
                owned = set()
                for locator in locators:
                    keys(locator,('cost_page','entry_index'),'cost line locator')
                    ref = reference(locator['cost_page']); ix = integer(locator['entry_index'],maximum=127)
                    position = locator_positions.get((ref['file_sha256'],ix))
                    if position is None: raise ValueError('cost locator must resolve one original immutable entry')
                    original_ref,line = lines[position]
                    if original_ref != ref: raise ValueError('cost locator path/byte alias differs')
                    if position in owned or tuple(line[k] for k in ('stream_id','component_id','resource_id','lane_id')) != key:
                        raise ValueError('cost locator duplicated or crosses resource lane')
                    owned.add(position)
                actual = lane_positions.get(key,set())
                if owned != actual: raise ValueError('financial lane omits original cost line')
                intervals = sorted(measured.get(key,[]))
                if components[cid]['coverage'] == 'interval':
                    previous = start
                    for begin,finish in intervals:
                        if begin < previous: raise ValueError('resource interval billed twice')
                        if begin != previous: reasons[sid].add('unmeasured_resource_interval')
                        previous = finish
                    if previous != end: reasons[sid].add('unmeasured_resource_interval')
                elif not intervals:
                    reasons[sid].add('missing_required_event')
    if set(financial['lanes']) != expected_lanes: raise ValueError('extra unplanned financial lane')
    for instance,operation in operations.items():
        row = financial['operations'][instance]; sid = runordinal[operation['invocation_ordinal']].stream_id
        if row['status'] != 'settled' or row['basis'] == 'unknown': reasons[sid].add('unsettled_actual_operation')
        if row['basis'] == 'original_line':
            if len(operation_lines.get(instance,[])) != 1: raise ValueError('each original percall line requires exactly one cost owner')
        elif row['basis'] == 'meter_lane':
            key = tuple(row['meter_lane'][k] for k in ('stream_id','component_id','resource_id','lane_id'))
            if key[0] != sid: raise ValueError('operation meter crosses original run stream')
            requirement = streams[sid]['requirements'].get(key[1])
            lane = next((lane for lane in requirement['lanes'] if
                (lane['resource_id'],lane['lane_id']) == key[2:]),None) if requirement is not None else None
            if (lane is None or requirement['state'] != 'required' or components[key[1]]['kind'] != 'monetary' or
                    components[key[1]]['coverage'] != 'interval' or lane['coverage'] != 'observed_lifecycle'):
                raise ValueError('operation meter requires an explicit monetary lifecycle binding, not overhead or event coverage')
            if collection is None: raise ValueError('operation meter requires frozen provider bindings')
            bindings = next(a['provider_bindings'] for a in collection['arms'] if a['arm'] == runordinal[operation['invocation_ordinal']].arm)
            matching = [i for i,b in enumerate(bindings) if b['binding_id'] == lane['lifecycle_binding_id']]
            if matching != [operation['binding_index']]: raise ValueError('operation meter crosses frozen provider binding/resource')
            declared = financial['lanes'].get(key)
            if declared is None: raise ValueError('operation meter requires its original financial resource lane')
            selected = operation_work.get((operation['invocation_ordinal'],operation['binding_index'],operation['entry_ordinal']),())
            if not selected: reasons[sid].add('unmeasured_operation_work')
            for interval in selected:
                if interval['clock_domain'] != 'host_monotonic_ns' or interval['ended_ns'] is None:
                    reasons[sid].add('unclosed_operation_work')
                elif not declared['window_start_ns'] <= interval['started_ns'] <= interval['ended_ns'] <= declared['window_end_ns']:
                    raise ValueError('operation meter excludes actual operation work interval')
            if declared['status'] != 'closed': reasons[sid].add('unclosed_operation_meter')
            if operation_lines.get(instance): raise ValueError('original call cannot be billed by meter AND percall line')
    return totals,reasons,lines


def validate_lifetime_ledger(policy,exposure,runtime,collection,scope_protocol,instrumentation,observed_runs,
        native_indices,ledger,financial_inventory,validity_checkpoint,*,plan,recordings,adapters_document,store,
        expected_policy_sha256,expected_exposure_sha256,expected_ledger_sha256,expected_checkpoint_sha256,
        validity_entries,late_notices):
    """Recompute every original source/operation/line/window; no selective groups.

    The explicit expected checkpoint digest is an external review trust input.
    Only the typed runtime/native producers may supply `observed_runs/indices`.
    """
    policy = validate_accounting_policy(policy,expected_sha256=expected_policy_sha256,
        comparison_plan_sha256=plan_digest(plan),recordings=recordings)
    exposure = validate_exposure_policy(exposure,plan,recordings,expected_sha256=expected_exposure_sha256,
        expected_recordings_sha256=digest(recordings))
    if exposure['accounting_policy_sha256'] != expected_policy_sha256 or policy['financial_adapter_inventory_sha256'] != exposure['financial_adapter_inventory_sha256']:
        raise ValueError('upstream E2/K2/F inventory mismatch')
    adapters = validate_adapters(adapters_document,expected_sha256=policy['financial_adapter_inventory_sha256'])
    # Numeric versions are prospective and explicit. Null-R/Q1 cannot become
    # eligible after execution by presenting a new E/K alongside old journals.
    if (type(runtime.get('schema_version')) is not int or runtime['schema_version'] != 2 or
            runtime.get('evaluation_mode') != 'matched_source_exposure_full_lifetime_v2' or
            runtime.get('numeric_policy_sha256') != expected_exposure_sha256 or
            runtime.get('accounting_policy_sha256') != expected_policy_sha256 or
            collection.get('runtime_policy_sha256') != plan_digest(runtime) or
            type(scope_protocol.get('schema_version')) is not int or scope_protocol['schema_version'] != 2 or
            scope_protocol.get('recording_reconciliation_sha256') != expected_exposure_sha256 or
            scope_protocol.get('stream_protocol_sha256') != expected_policy_sha256 or
            scope_protocol.get('collection_sha256') != digest(collection) or
            type(instrumentation.get('schema_version')) is not int or instrumentation['schema_version'] != 3 or
            instrumentation.get('scope_protocol_sha256') != digest(scope_protocol) or
            instrumentation.get('collection_sha256') != digest(collection)):
        raise ValueError('prospectively frozen U2/C2/Q2/I3 numeric declarations required')
    ledger = snapshot(ledger,128*1024**2)
    if digest(ledger) != sha(expected_ledger_sha256): raise ValueError('externally reviewed ledger hash differs')
    keys(ledger,('schema_version','kind','accounting_policy_sha256','exposure_policy_sha256','runtime_policy_sha256',
        'collection_sha256','scope_protocol_sha256','comparison_plan_sha256','reviewer','source_evidence_sha256',
        'lifetimes','coverage','entry_pages','financial_inventory','closure'),'lifetime ledger2')
    _schema(ledger,2,'recording_lifetime_ledger')
    for key,expected in (('accounting_policy_sha256',expected_policy_sha256),('exposure_policy_sha256',expected_exposure_sha256),
            ('runtime_policy_sha256',plan_digest(runtime)),('collection_sha256',digest(collection)),
            ('scope_protocol_sha256',digest(scope_protocol)),('comparison_plan_sha256',plan_digest(plan))):
        if ledger[key] != expected: raise ValueError('ledger frozen upstream binding differs')
    identifier(ledger['reviewer']); sha(ledger['source_evidence_sha256'])
    runs = tuple(observed_runs)
    if not 3 <= len(runs) <= 96 or any(type(run) is not FinancialRunView for run in runs):
        raise TypeError('all original typed runtime arm joins required')
    if len({r.invocation_ordinal for r in runs}) != len(runs) or len({r.execution_sha256 for r in runs}) != len(runs):
        raise ValueError('all original invocation and execution identities must be unique')
    execution_inventories = {r.execution_inventory_sha256 for r in runs}
    if len(execution_inventories) != 1: raise ValueError('all original runs must bind the same actual execution inventory')
    runmap = {r.stream_id:r for r in runs}; streams = {s['stream_id']:s for s in policy['streams']}
    if len(runmap) != len(runs) or set(runmap) != set(streams): raise ValueError('every frozen original arm run required once')
    for run in runs:
        stream = streams[run.stream_id]
        if (run.recording_id,run.arm,run.source_manifest_sha256) != (stream['recording_id'],stream['arm'],stream['source_manifest_sha256']):
            raise ValueError('run crosses original recording/arm/source')
    lifetimes = {}; lifetime_hashes = {}; coverage = set()
    for ref in array(ledger['lifetimes'],96):
        value = store.read(reference(ref)); sid = value.get('stream_id')
        if sid not in runmap or sid in lifetimes: raise ValueError('every original observed lifetime required once')
        lifetimes[sid] = validate_lifetime(value,runmap[sid]); lifetime_hashes[sid] = digest(value)
    for ref in array(ledger['coverage'],96):
        value = store.read(reference(ref)); h = digest(value)
        matches = [run.stream_id for run in runs if run.source_coverage_sha256 == h]
        if len(matches) != 1 or matches[0] in coverage: raise ValueError('exact original strict source coverage per run required')
        coverage.add(matches[0])
    if set(lifetimes) != set(streams) or coverage != set(streams): raise ValueError('omitted lifetime or source coverage')
    operations,work,operation_keys = _native_inventory(native_indices,runs,dict(collection_sha256=digest(collection),
        scope_protocol_sha256=digest(scope_protocol),instrumentation_protocol_sha256=digest(instrumentation)),budget=store)
    if store.read(reference(ledger['financial_inventory'])) != financial_inventory: raise ValueError('financial inventory artifact differs')
    financial = _financial_inventory(store,financial_inventory,adapters=adapters,policy=policy,exposure=exposure,
        collection_sha256=digest(collection),operations=operations,keys_to_instance=operation_keys,runs=runs)
    totals,reasons,lines = _validate_costs(store,ledger['entry_pages'],policy=policy,runs=runs,lifetimes=lifetimes,
        financial=financial,operations=operations,work=work,collection=collection)
    if financial['state']!='complete':
        for sid in reasons:reasons[sid].add('financial_inventory_incomplete')
    closures = store.read(reference(ledger['closure']))
    keys(closures,('schema_version','kind','closures'),'financial closure inventory')
    _schema(closures,1,'recording_financial_closures')
    closure_map = {}; final_seal = 0
    for ref in array(closures['closures'],96):
        obj = store.read(reference(ref)); keys(obj,CLOSURE_FIELDS,'financial closure'); _schema(obj,1,'recording_financial_closure')
        sid = obj['stream_id']
        if type(sid) is not str or sid not in runmap or sid in closure_map: raise ValueError('one original financial closure perstream required')
        run = runmap[sid]; stream = streams[sid]
        if (obj['execution_sha256'] != run.execution_sha256 or obj['accounting_policy_sha256'] != expected_policy_sha256 or
                obj['financial_adapter_inventory_sha256'] != adapters.digest or obj['financial_inventory_sha256'] != digest(financial_inventory) or
                obj['operational_end_ns'] != run.parent_resources_closed_ns): raise ValueError('closure original observed boundary differs')
        if obj['operational_end_ns'] is None:
            if obj['cutoff_ns'] is not None: raise ValueError('unclosed work cannot start optimistic settlement cutoff')
            reasons[sid].add('operational_work_not_closed')
        elif obj['cutoff_ns'] != obj['operational_end_ns']+policy['closure']['settlement_wait_ms']*10**6:
            raise ValueError('financial cutoff differs from frozen wait after actual closure')
        sealed = integer(obj['sealed_ns']); final_seal = max(final_seal,sealed)
        if obj['operational_end_ns'] is not None and sealed < obj['operational_end_ns']: raise ValueError('seal precedes actual operational closure')
        expected_operations = [o for o in financial['operations'].values() if o['invocation_ordinal'] == run.invocation_ordinal]
        required_lanes = [lane for key,lane in financial['lanes'].items() if key[0] == sid]
        receipt_times=[row['observed_received_ns'] for row in financial['evidence'].values() if row['stream_id']==sid]
        receipt_times.extend(row['observed_ns'] for row in financial['settlements'].get(run.invocation_ordinal,[]))
        if any(stamp>sealed for stamp in receipt_times):
            raise ValueError('sealed financial inventory contains a future receipt or settlement')
        settled_count = sum(o['status'] == 'settled' and o['basis'] != 'unknown' for o in expected_operations)
        expected_counts = dict(operation_count=len(expected_operations),known_settled_operations=settled_count,
            unresolved_operations=len(expected_operations)-settled_count,required_lane_count=len(required_lanes),
            closed_lane_count=sum(lane['status'] == 'closed' for lane in required_lanes),
            pending_line_count=sum(line['status'] == 'pending' for ref,line in lines.values() if line['stream_id'] == sid),
            unknown_component_count=sum(1 for cid,req in stream['requirements'].items() if req['state'] == 'required' and
                (any(line['status'] == 'unknown' for ref,line in lines.values() if (line['stream_id'],line['component_id']) == (sid,cid)) or
                 not any((line['stream_id'],line['component_id']) == (sid,cid) for ref,line in lines.values()))))
        for key,count in expected_counts.items():
            if integer(obj[key]) != count: raise ValueError('financial closure counter does not recompute: '+key)
        finality_refs = array(obj['external_finality_refs'],64); valid_finality = []
        for finality_ref in finality_refs:
            finality_obj = validate_finality(store.read(reference(finality_ref))); h = digest(finality_obj)
            if h not in financial['finality'] or financial['finality'][h] != (finality_ref,finality_obj): raise ValueError('closure finality not in complete frozen F1 inventory')
            if finality_obj['stream_id'] != sid or finality_obj['execution_sha256'] != run.execution_sha256 or finality_obj['accounting_policy_sha256'] != expected_policy_sha256:
                raise ValueError('finality crosses original stream/execution/policy')
            if not any(a['closure_capability'] == 'explicit_original_lines_and_finality' for a in adapters.adapters):
                raise ValueError('frozen adapter inventory has no supported original finality capability')
            docs = sorted(ref['file_sha256'] for ref,document in financial['documents'].values() if any(line['stream_id'] == sid for line in document['lines']))
            if finality_obj['source_document_sha256'] != docs: raise ValueError('finality omits original invoice documents')
            operation_rows = [o for o in financial['operations'].values() if o['invocation_ordinal'] == run.invocation_ordinal]
            lane_rows = [lane for key,lane in financial['lanes'].items() if key[0] == sid]
            if (finality_obj['operation_inventory_sha256'] != digest(sorted(operation_rows,key=lambda o:o['entry_ordinal'])) or
                    finality_obj['lane_closure_inventory_sha256'] != digest(sorted(lane_rows,key=lambda o:(o['component_id'],o['resource_id'],o['lane_id'])))):
                raise ValueError('finality does not cover complete operation/lane inventories')
            if run.parent_resources_closed_ns is None or not run.parent_resources_closed_ns <= finality_obj['service_end_ns'] <= finality_obj['issued_ns'] <= sealed:
                raise ValueError('finality issued before all attributable work closed')
            if any(lane['window_end_ns']>finality_obj['service_end_ns'] for lane in required_lanes):
                raise ValueError('finality excludes an original resource service window')
            if any(stamp>finality_obj['issued_ns'] for stamp in receipt_times):
                raise ValueError('finality precedes its original receipt or settlement')
            valid_finality.append(h)
        if len(valid_finality) != 1: reasons[sid].add('missing_authoritative_fixture_finality')
        # Original F1 receipts precede finality and retain their nullable pointer
        # as null. The downstream finality above covers their exact original
        # documents and complete operation/lane inventories without a cycle.
        for key,count in expected_counts.items():
            if key in ('unresolved_operations','pending_line_count','unknown_component_count') and count: reasons[sid].add(key)
        if expected_counts['required_lane_count'] != expected_counts['closed_lane_count']: reasons[sid].add('required_lane_not_closed')
        if not run.clock_consistent: reasons[sid].add('clock_discontinuity')
        if not run.operational_closed or not run.external_work_closed or lifetimes[sid]['state'] != 'closed': reasons[sid].add('work_closure_unknown')
        # Source opportunity failure and known overrun do not erase actual known
        # money. They remain immutable runtime gate failures in the report.
        reasons[sid].update(v for v in run.violations if v not in ('source_incomplete','reservation_overrun'))
        state = enum(obj['state'],('closed','incomplete'))
        if state == 'closed' and reasons[sid]: raise ValueError('financial closure falsely claims closed despite required unknowns/failures')
        if state != 'closed': reasons[sid].add('financial_closure_incomplete')
        closure_map[sid] = digest(obj)
    if set(closure_map) != set(streams): raise ValueError('omitted failed original closure')
    for ref,obj in validity_entries:
        if store.read(reference(ref)) != obj: raise ValueError('retained validity file differs from pinned chain')
    notice_refs=[]
    for _,entry in validity_entries:
        if entry['notice'] is not None:
            ref=reference(entry['notice']);notice_refs.append(ref)
            if ref['file_sha256'] not in late_notices or store.read(ref)!=late_notices[ref['file_sha256']]:
                raise ValueError('retained original late notice differs from current reviewed chain')
    for noticed_hash,notice in late_notices.items():
        if notice['stream_id'] not in closure_map or notice['closure_sha256'] != closure_map[notice['stream_id']]:
            raise ValueError('late notice crosses original sealed stream closure')
        evidence = store.read(reference(notice['financial_evidence']))
        source_ref = reference(evidence['source_document']); invoice = validate_invoice(store.read(source_ref))
        validate_financial_evidence(evidence,adapters=adapters,invoice=invoice,source_reference=source_ref)
        if evidence['stream_id'] != notice['stream_id'] or evidence['observed_received_ns'] > notice['noticed_ns']:
            raise ValueError('late notice does not retain actual received original evidence')
    validity = validate_validity_chain(validity_entries,validity_checkpoint,expected_checkpoint_sha256=expected_checkpoint_sha256,
        exposure_policy_sha256=expected_exposure_sha256,ledger_sha256=expected_ledger_sha256,sealed_ns=final_seal,notices=late_notices)
    store.assert_financial_inventory()
    store.assert_validity_inventory(notice_refs,[ref for ref,_ in validity_entries])
    result = []
    for run in runs:
        sid = run.stream_id; stream = streams[sid]
        elapsed = None if run.parent_resources_closed_ns is None else run.parent_resources_closed_ns-run.activation_started_ns
        result.append(ArmLifetimeTotal(run,lifetime_hashes[sid],closure_map[sid],stream['exposure_duration_ns'],elapsed,
            totals[sid],None if reasons[sid] else totals[sid],tuple(sorted(reasons[sid]))))
    return ValidatedLifetimeLedger(policy_sha256=expected_policy_sha256,exposure_sha256=expected_exposure_sha256,
        ledger_sha256=expected_ledger_sha256,runtime_policy_sha256=plan_digest(runtime),collection_sha256=digest(collection),
        scope_protocol_sha256=digest(scope_protocol),instrumentation_protocol_sha256=digest(instrumentation),
        execution_inventory_sha256=next(iter(execution_inventories)),
        execution_receipts=tuple(sorted((r.invocation_ordinal,r.arm,r.execution_sha256,r.parent_receipt_inventory_sha256) for r in runs)),
        run_inventory=tuple(sorted((r.invocation_ordinal,r.execution_sha256,r.stream_id,r.run_inventory_sha256,
            r.task_row_inventory_sha256,r.native_terminal_sha256,r.controller_settlement_inventory_sha256) for r in runs)),
        totals=tuple(result),evidence_class='fixture',validity=validity,_issuer=_TOTAL_ISSUER)
