"""Original collector-native3 structural index, never a channel/billing attestation."""
from dataclasses import dataclass
import json
from pathlib import Path

from .native3_files import DECLARATIONS, OriginalNativeFiles
from .paired_replay import _encoded, _keys
from .perception_comparison import plan_digest, _digest
from .three_arm_evidence import _strict_decode, COMPLETE_KEYS, validate_source_trace
from .three_arm_replay import ORDER, PERMUTATIONS
from .runtime_channel import HASH_FIELDS, PATH_HASHES, RuntimeValidationState, validate_bootstrap
from .instrumented_replay import validate

_OWNER_KEYS = ('comparison_plan_sha256','collection_sha256','stream_protocol_sha256',
    'reconciliation_protocol_sha256','scope_kind','task_id','arm','stream_id',
    'input_sha256','recording_sha256','configuration_sha256','cost_sha256')
_READER = object()


def _decode(data):
    # Bound parser expansion before decoding, while preserving original JSON
    # numeric types required by the existing collector validators.
    text = data.decode('utf-8', errors='strict')
    depth = nodes = 0; quoted = escaped = primitive = False
    for char in text:
        if quoted:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
            continue
        if char == '"': quoted = True; primitive = False; nodes += 1
        elif char in '[{': depth += 1; nodes += 1; primitive = False
        elif char in ']}': depth -= 1; primitive = False
        elif char in ',: \t\r\n': primitive = False
        elif not primitive: nodes += 1; primitive = True
        if depth > 64 or nodes > 1000000: raise ValueError('original native JSON expansion exceeds bound')
    return _strict_decode(text)


def _same(actual, expected, reason):
    if _encoded(actual) != _encoded(expected): raise ValueError(reason)


@dataclass(frozen=True, init=False)
class ValidatedNative3Index:
    """Reader-issued immutable structural evidence, not an unforgeable capability."""
    _data: bytes
    directory: Path
    def __init__(self, value, *, directory=None, _reader=None):
        if _reader is not _READER: raise TypeError('read original collector files to construct an index')
        object.__setattr__(self, '_data', _encoded(value))
        object.__setattr__(self, 'directory', Path(directory).absolute())
    def snapshot(self): return json.loads(self._data)
    def entries(self): return self.snapshot()['entries']
    def lookup(self, operation_id):
        for entry in self.entries():
            if entry['entry']['operation_id'] == operation_id: return entry
        raise KeyError('original native operation absent')


def _source_inventory(files, plan, spec, by_id, hashes):
    tasks = plan['comparison']['tasks']; expected = [(t, arm) for i,t in enumerate(tasks) for arm in PERMUTATIONS[i%6]]
    previous = '0'*64; refs = []; positions = {arm:[0,0,0] for arm in 'ABC'}
    for ordinal, line in enumerate(files.source_lines()):
        if ordinal >= len(expected): raise ValueError('extra original source trace')
        task, arm = expected[ordinal]
        trace, digest, _, _, _, _ = validate_source_trace(_decode(line), task=task, arm=arm,
            ordinal=ordinal, previous=previous, plan=plan, by_id=by_id,
            expected_plan_sha256=hashes['comparison_plan_sha256'], expected_collection_sha256=hashes['collection_sha256'])
        refs.append({key:trace[key] for key in ('trace_sha256','response_sha256','request_sha256','task_id','arm','status')})
        positions[arm][ordinal%3] += 1; previous = digest
    if len(refs) != len(expected): raise ValueError('truncated original source trace inventory')
    complete = _decode(files.read('complete.json')); _keys(complete, COMPLETE_KEYS, 'source completion')
    wanted = dict(schema_version=1, comparison_plan_sha256=hashes['comparison_plan_sha256'],
        collection_sha256=hashes['collection_sha256'], state='collected_unscored', records=len(refs),
        last_trace_sha256=previous, order=ORDER, arm_position_counts=positions,
        complete_six_task_blocks=len(tasks)//6, remaining_tasks=len(tasks)%6, actual_acceptance='incomplete')
    for key,value in wanted.items(): _same(complete[key],value,'original source completion mismatch')
    if type(complete['certification']) is not str or not 1<=len(complete['certification'])<=4096:
        raise ValueError('bounded original source certification required')
    return refs, previous


def _sidecar_inventory(files, refs, hashes, mapping, selected_ordinal):
    raw = files.read('runtime-sidecars.jsonl'); previous = '0'*64; selected = None
    fields = ('schema_version','ordinal',*HASH_FIELDS,'task_id','arm','stream_id','request_sha256','nonce',
        'original_trace_sha256','original_response_sha256','source_status','previous_sidecar_sha256','runtime','sidecar_sha256')
    count = 0
    for count,line in enumerate(raw.splitlines(keepends=True),1):
        ordinal = count-1
        if count>3000 or ordinal>=len(refs) or not line.endswith(b'\n'): raise ValueError('bounded complete original sidecar lines required')
        side = _decode(line); _keys(side, fields, 'original runtime sidecar')
        _same(side['schema_version'],1,'sidecar schema1 required'); _same(side['ordinal'],ordinal,'sidecar ordinal differs')
        for key,value in hashes.items(): _same(side[key],value,'sidecar declaration differs')
        if side['previous_sidecar_sha256']!=previous or side['sidecar_sha256']!=plan_digest({k:v for k,v in side.items() if k!='sidecar_sha256'}):
            raise ValueError('original sidecar chain differs')
        ref=refs[ordinal]
        wanted = dict(task_id=ref['task_id'],arm=ref['arm'],stream_id=mapping[(ref['task_id'],ref['arm'])],
            request_sha256=ref['request_sha256'],original_trace_sha256=ref['trace_sha256'],
            original_response_sha256=ref['response_sha256'],source_status=ref['status'])
        for key,value in wanted.items(): _same(side[key],value,'sidecar/source binding differs')
        bootstrap=side['runtime'].get('bootstrap') if type(side['runtime']) is dict else None
        validate_bootstrap(bootstrap)
        for key in (*HASH_FIELDS,'task_id','arm','stream_id','request_sha256','nonce'):
            _same(bootstrap[key],side[key],'original bootstrap differs from sidecar')
        # bootstrap.paths is provenance only. Never resolve/open its strings.
        previous = side['sidecar_sha256']
        if ordinal==selected_ordinal: selected=side
    if count!=len(refs) or selected is None: raise ValueError('truncated or missing original selected sidecar')
    return selected, previous, count


def _snapshot_index(side, declarations):
    runtime=side['runtime']
    _keys(runtime,('schema_version','bootstrap','hello_verified','eof','records','journal','native_reports','terminal',
        'validated_prefix_sha256','native_journal_sha256','instrumentation_complete','incomplete_reason',
        'actual_acceptance','receipt_authenticity_verified','measurement_truth_verified','instrumentation_coverage_verified',
        'isolation_verified'),'original native snapshot')
    _same(runtime['schema_version'],1,'snapshot schema1 required')
    for key in ('hello_verified','eof','instrumentation_complete'):
        if type(runtime[key]) is not bool: raise ValueError('producer-reported status must be boolean')
    for key in ('receipt_authenticity_verified','measurement_truth_verified','instrumentation_coverage_verified','isolation_verified'):
        if runtime[key] is not False: raise ValueError('native snapshot cannot certify truth')
    if runtime['actual_acceptance']!='incomplete': raise ValueError('original acceptance must remain incomplete')
    if type(runtime['records']) is not int or not 0<=runtime['records']<=8195: raise ValueError('bounded original frame count required')
    _digest(runtime['validated_prefix_sha256'])
    if runtime['incomplete_reason'] not in (None,'cancelled','deadline_exceeded','invalid_channel','incomplete_channel','session_accounting_incomplete'):
        raise ValueError('known original incomplete reason required')
    bootstrap=runtime['bootstrap']; policy=declarations['instrumentation_protocol']
    if bootstrap['hello_timeout_ms']!=policy['hello_timeout_ms']: raise ValueError('original hello deadline differs')
    validator=RuntimeValidationState(bootstrap,declarations)
    events=runtime['journal']; reports=runtime['native_reports']; terminal=runtime['terminal']
    if type(events) is not list or len(events)>8192 or type(reports) is not list or len(reports)!=2:
        raise ValueError('bounded original events and exactly two reports required')
    for event in events: validator._event(event)
    for report in reports: validator._report(report)
    if runtime['native_journal_sha256']!=validator.native_previous: raise ValueError('original journal tail differs')
    _keys(terminal,('report_sha256s','summary'),'original runtime terminal')
    _same(terminal['report_sha256s'],[r['sha256'] for r in reports],'original terminal report inventory differs')
    validator._summary(terminal['summary'])
    task=next(report for report in reports if report['scope_kind']=='task')
    owner={key:task[key] for key in _OWNER_KEYS}; owner['instrumentation_protocol_sha256']=bootstrap['instrumentation_protocol_sha256']
    attempts={attempt['attempt_id']:attempt for attempt in task['attempts']}; operations={}; entries=[]
    for event in events:
        name=event['event']; operation=event['operation_id']
        if name=='call_capacity_reserved': operations[operation]={}
        if operation in operations: operations[operation][name]=event
        if name!='backend_invocation_finished' or event['fields']['attempt_id'] is None: continue
        attempt=attempts[event['fields']['attempt_id']]; marks=operations[operation]
        admission=marks['backend_entry_admission']; reservation=marks['call_capacity_reserved']['fields']
        entries.append(dict(schema_version=1,owner_kind='task_native3',owner=owner,
            entry=dict(operation_id=operation,attempt_id=attempt['attempt_id'],
                execution_instance_sha256=admission['fields']['execution_instance_sha256'],execution_sha256=attempt['execution_sha256']),
            provider=reservation['provider'],model_revision=reservation['revision'],role=reservation['role'],
            prompt_sha256=reservation['prompt_sha256'],evidence_sha256=reservation['evidence_sha256'],
            original_admission_sha256=admission['sha256'],original_report_sha256=task['sha256']))
    summary=terminal['summary']
    return dict(owner=owner,entries=entries,original_native_journal_sha256=validator.native_previous,
        producer_reported_channel_prefix_sha256=runtime['validated_prefix_sha256'],
        dispatched_without_invocation=summary['dispatched_without_invocation'],incomplete_reasons=summary['incomplete_reasons'],
        structural_native_verified=True,original_channel_frames_verified=False,
        producer_reported_instrumentation_complete=runtime['instrumentation_complete'],
        original_incomplete_reason=runtime['incomplete_reason'])


def read_native3_evidence(directory, *, expected_files, sidecar_ordinal):
    """Read actual instrumented_replay outputs; standalone snapshots are unsupported."""
    if type(sidecar_ordinal) is not int or not 0<=sidecar_ordinal<3000: raise ValueError('bounded selected sidecar required')
    with OriginalNativeFiles(directory,expected_files) as files:
        declarations={key:_decode(files.read(name)) for key,name in DECLARATIONS.items()}
        hashes={PATH_HASHES[key]:plan_digest(value) for key,value in declarations.items()}
        _,_,_,_,_,_,_,by_id,mapping=validate(*declarations.values(),hashes=hashes,portable=True)
        refs, source_tail=_source_inventory(files,declarations['plan'],declarations['collection'],by_id,hashes)
        side,side_tail,count=_sidecar_inventory(files,refs,hashes,mapping,sidecar_ordinal)
        complete=_decode(files.read('runtime-complete.json'))
        _keys(complete,('schema_version',*HASH_FIELDS,'state','records','last_sidecar_sha256','last_trace_sha256',
            'received_runtime_bytes','admitted_runtime_frame_bytes','instrumentation_complete','actual_acceptance',
            'receipt_authenticity_verified','measurement_truth_verified','instrumentation_coverage_verified',
            'isolation_verified','whole_pipeline_cost_verified'),'original runtime completion')
        for key,value in dict(schema_version=1,**hashes,state='collected_unscored',records=count,
                              last_sidecar_sha256=side_tail,last_trace_sha256=source_tail,actual_acceptance='incomplete').items():
            _same(complete[key],value,'original runtime completion differs')
        for key in ('received_runtime_bytes','admitted_runtime_frame_bytes'):
            if type(complete[key]) is not int or not 0<=complete[key]<=16*1024**2: raise ValueError('bounded producer byte accounting required')
        if type(complete['instrumentation_complete']) is not bool: raise ValueError('producer completion boolean required')
        for key in ('receipt_authenticity_verified','measurement_truth_verified','instrumentation_coverage_verified','isolation_verified','whole_pipeline_cost_verified'):
            if complete[key] is not False: raise ValueError('original runtime truth flags must be false')
        index=_snapshot_index(side,declarations)
        providers = sorted({(row['provider'],row['revision']) for row in declarations['plan']['comparison']['cost']['rate_card'].values()})
        index['declared_provider_revisions'] = [dict(provider=provider,model_revision=revision) for provider,revision in providers]
        index.update(reader_version='original_collector_native3_v1',sidecar_ordinal=sidecar_ordinal,
            original_sidecar_sha256=side['sidecar_sha256'],original_source_trace_sha256=side['original_trace_sha256'],
            original_files=files.inventory,transport_selection_preexecution_verified=False,
            producer_reported_collection_complete=complete['instrumentation_complete'],
            receipt_authenticity_verified=False,instrumentation_coverage_verified=False,actual_acceptance='incomplete')
        if files.verified!=set(files.inventory): raise ValueError('incomplete original file verification')
        return ValidatedNative3Index(index,directory=directory,_reader=_READER)
