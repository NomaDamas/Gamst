"""Owned, bounded POSIX evaluation side channel; never an inference attestation."""
import asyncio
import os
from pathlib import Path
import secrets
import socket
import stat
import struct
import time

from .paired_replay import _encoded, _keys
from .perception_comparison import _digest, plan_digest
from .three_arm_evidence import _strict_decode

FD_ENVIRONMENT_KEY = 'GAMST_EVALUATION_FD'
MAX_FRAME = 512*1024
MAX_BYTES = 16*1024*1024
MAX_EVENTS = 8192
MAX_QUEUE_FRAMES = 64
MAX_QUEUE_BYTES = 1024*1024
CONTROL_EVENTS = 32
CONTROL_BYTES = 64*1024
FINAL_BYTES = 3*(MAX_FRAME+4)
CLOSE_TIMEOUT_SECONDS = 5
MAX_CALL_RESERVATIONS = 1024
HASH_FIELDS = ('comparison_plan_sha256','collection_sha256','inputs_sha256','stream_protocol_sha256',
    'allocation_protocol_sha256','reconciliation_protocol_sha256','instrumentation_protocol_sha256')
PATH_HASHES = dict(zip(('plan','collection','inputs','stream_protocol','allocation_protocol',
    'reconciliation_protocol','instrumentation_protocol'),HASH_FIELDS))
BOOTSTRAP_KEYS = ('schema_version','type','nonce',*HASH_FIELDS,'task_id','arm','stream_id',
    'request_sha256','deadline_ns','hello_timeout_ms','paths')
FRAME_KEYS = ('schema_version','type','nonce','sequence','previous_sha256','payload','sha256')
EVENT_FIELDS = {
    'call_capacity_reserved':{'provider','revision','role','prompt_sha256','evidence_sha256'},
    'controller_dispatched':{'liability','execution_sha256','predecessor','relationship'},
    'backend_entry_admission':{'execution_instance_sha256'},
    'backend_invocation_finished':{'status','attempt_id','reported_usage','usage_verified'},
    'call_publication_finished':{'outcome','invoked'},
    'call_capacity_released':{'dispatched','invoked','reason'},
    'work_capacity_reserved':{'kind','worker_id','profile','configuration_sha256','operation','parent_operation_id'},
    'work_finished':{'kind','outcome'},
    'private_mappings_invalidated':{'reason','affected'},
    'external_billing_attached':{'operation_id','evidence_sha256'},
}
CALL_OUTCOMES={'completed','published','discarded','error','cancelled','timeout','failed','success'}
WORK_OUTCOMES={'closed','completed','error','cancelled','timeout'}
WORK_OPERATIONS={'start','process','process_regions','finish','endpoint','reset','close'}


def _attempt_id(value):
    if type(value) is not str:return False
    scope,separator,index=value.partition(':')
    return (len(scope)==32 and all(c in '0123456789abcdef' for c in scope) and separator==':' and
            index.isascii() and index.isdecimal() and len(index)<=3 and str(int(index))==index and int(index)<128)


class RuntimeChannelError(RuntimeError):
    """Invalid side-channel data requires prompt owned invocation cleanup."""
for _milestone in ('spawn_begin','handle_acquired','handshake_ready','handshake_failed',
                   'reap_confirmed','operation_started','operation_finished','invalidated'):
    EVENT_FIELDS[_milestone]={'returncode','status','operation','parent_operation_id'}


def validate_instrumentation_protocol(protocol, *, expected_sha256):
    protocol=_strict_decode(_encoded(protocol))
    _keys(protocol,('schema_version','comparison_plan_sha256','collection_sha256',
        'stream_protocol_sha256','allocation_protocol_sha256','reconciliation_protocol_sha256',
        'transport','hello_timeout_ms','fd_environment_key','retention_policy','phase'),'instrumentation protocol')
    if plan_digest(protocol)!=_digest(expected_sha256): raise ValueError('frozen instrumentation digest mismatch')
    for key in HASH_FIELDS:
        if key in protocol: _digest(protocol[key])
    if (type(protocol['schema_version']) is not int or protocol['schema_version']!=1 or
            protocol['transport']!='socketpair_be32_json_v1' or protocol['fd_environment_key']!=FD_ENVIRONMENT_KEY or
            protocol['retention_policy']!='external_owner_retains_task_and_evidence_hashes' or
            protocol['phase'] not in ('cold','warm','unknown') or type(protocol['hello_timeout_ms']) is not int or
            not 1<=protocol['hello_timeout_ms']<=3600000):
        raise ValueError('unsupported frozen instrumentation transport')
    return protocol


def validate_bootstrap(value):
    _keys(value,BOOTSTRAP_KEYS,'runtime bootstrap')
    if type(value['schema_version']) is not int or value['schema_version']!=1 or value['type']!='bootstrap':
        raise ValueError('bootstrap version required')
    for key in (*HASH_FIELDS,'nonce','request_sha256'): _digest(value[key])
    for key in ('task_id','stream_id'):
        if type(value[key]) is not str or not 1<=len(value[key])<=256: raise ValueError('bounded original ID required')
    if value['arm'] not in ('A','B','C'): raise ValueError('original arm required')
    if (type(value['deadline_ns']) is not int or not 0<value['deadline_ns']<=2**63-1 or
            type(value['hello_timeout_ms']) is not int or not 1<=value['hello_timeout_ms']<=3600000):
        raise ValueError('bounded original deadline required')
    _keys(value['paths'],PATH_HASHES,'bootstrap artifact paths')
    for path in value['paths'].values():
        if type(path) is not str or not Path(path).is_absolute() or len(path)>4096 or '\0' in path:
            raise ValueError('absolute bounded declaration path required')
    return value


def read_declaration_paths(paths):
    """Read only bounded regular data files, with one aggregate allocation cap."""
    _keys(paths,PATH_HASHES,'runtime declaration paths')
    result={};total=0
    for key in PATH_HASHES:
        descriptor=os.open(paths[key],os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        with os.fdopen(descriptor,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_BYTES-total:
                raise ValueError('bounded regular runtime declaration required')
            data=stream.read(MAX_BYTES-total+1)
        total+=len(data)
        if total>MAX_BYTES: raise ValueError('runtime declarations exceed16MiB')
        result[key]=_strict_decode(data.decode('utf-8'))
    return result


def load_declarations(bootstrap):
    """Read data only, bounded before allocation; authenticate canonical identities."""
    validate_bootstrap(bootstrap)
    result=read_declaration_paths(bootstrap['paths'])
    for key,field in PATH_HASHES.items():
        if plan_digest(result[key])!=bootstrap[field]: raise ValueError('runtime declaration digest mismatch')
    protocol=validate_instrumentation_protocol(result['instrumentation_protocol'],
        expected_sha256=bootstrap['instrumentation_protocol_sha256'])
    if any(protocol[key]!=bootstrap[key] for key in HASH_FIELDS if key in protocol):
        raise ValueError('runtime protocol belongs to another bootstrap')
    if protocol['hello_timeout_ms']!=bootstrap['hello_timeout_ms']: raise ValueError('changed hello deadline')
    return result


def frame(value):
    data=_encoded(value)
    if not 1<=len(data)<=MAX_FRAME: raise ValueError('runtime frame exceeds512KiB')
    return struct.pack('!I',len(data))+data


async def receive(sock, budget=None):
    loop=asyncio.get_running_loop()
    async def exact(count,allow_eof=False):
        data=bytearray()
        while len(data)<count:
            wanted=count-len(data)
            if budget is not None:
                wanted=min(wanted,budget.maximum-budget.received)
                if wanted<=0:
                    budget.exhausted=True;raise ValueError('collection received byte limit exhausted')
            part=await loop.sock_recv(sock,wanted)
            if not part:
                if not data and allow_eof:return None
                raise ValueError('partial runtime frame at EOF')
            data.extend(part)
            if budget is not None:budget.received+=len(part)
        return bytes(data)
    header=await exact(4,True)
    if header is None:return None
    size=struct.unpack('!I',header)[0]
    if not 1<=size<=MAX_FRAME: raise ValueError('runtime frame length exceeds bound')
    if budget is not None: budget.consume(size+4)
    data=await exact(size)
    return _strict_decode(data.decode('utf-8'))


async def _until(awaitable, deadline):
    remaining=(deadline-time.monotonic_ns())/1e9
    if remaining<=0:
        if hasattr(awaitable,'close'): awaitable.close()
        raise asyncio.TimeoutError()
    return await asyncio.wait_for(awaitable,remaining)


class CollectionBudget:
    def __init__(self,maximum=MAX_BYTES):
        if type(maximum) is not int or not 1<=maximum<=MAX_BYTES: raise ValueError('bounded collection bytes required')
        self.maximum=maximum;self.used=0;self.received=0;self.overhead=0;self.exhausted=False
    def reserve_artifact_overhead(self,size):
        if self.exhausted or self.used+self.overhead+size>self.maximum:
            self.exhausted=True;raise ValueError('runtime artifact capacity exhausted before work')
        self.overhead+=size
    def consume(self,size):
        if self.exhausted or self.used+self.overhead+size>self.maximum:
            self.exhausted=True;raise ValueError('collection runtime bytes exhausted')
        self.used+=size


class RuntimeValidationState:
    """Pure original native journal/report validation, with no socket or EOF claim.

    Callers must supply independently validated frozen declarations. The live
    channel and offline reader share these exact accounting checks.
    """
    def __init__(self, bootstrap, declarations):
        self.bootstrap = _strict_decode(_encoded(validate_bootstrap(bootstrap)))
        self.declarations = declarations
        self.events = []
        self.reports = []
        self.session_id = None
        self.native_previous = '0' * 64

    def _event(self,item):
        _keys(item,('schema_version','session_id','clock_domain','sequence','previous_sha256',
            'monotonic_ns','operation_id','event','fields','sha256'),'native journal event')
        if len(_encoded(item))>4096 or len(self.events)>=MAX_EVENTS: raise ValueError('native event bound')
        if self.session_id is None:self.session_id=item['session_id']
        if (item['session_id']!=self.session_id or type(self.session_id) is not str or len(self.session_id)!=32 or
                any(c not in '0123456789abcdef' for c in self.session_id) or
                item['clock_domain']!='host-monotonic:'+self.session_id or type(item['schema_version']) is not int or item['schema_version']!=1 or
                type(item['sequence']) is not int or item['sequence']!=len(self.events) or
                item['previous_sha256']!=self.native_previous or
                item['sha256']!=plan_digest({k:v for k,v in item.items() if k!='sha256'}) or
                type(item['monotonic_ns']) is not int or not 0<=item['monotonic_ns']<=2**63-1 or
                (self.events and item['monotonic_ns']<self.events[-1]['monotonic_ns']) or
                type(item['event']) is not str or len(item['event'])>128 or type(item['fields']) is not dict):
            raise ValueError('native journal identity/chain mismatch')
        fields=item['fields']
        if item['event'] not in EVENT_FIELDS or set(fields)-EVENT_FIELDS[item['event']]:
            raise ValueError('unknown native event metadata')
        if item['operation_id'] is not None and (type(item['operation_id']) is not str or len(item['operation_id'])!=32 or
                any(c not in '0123456789abcdef' for c in item['operation_id'])):
            raise ValueError('opaque native operation ID required')
        for key,value in fields.items():
            if key=='reported_usage' and value is not None:
                _keys(value,('input_tokens','output_tokens','cached_input_tokens'),'reported usage')
                if any(type(v) is not int or not 0<=v<=10**12 for v in value.values()):raise ValueError('bounded usage counts required')
            elif key.endswith('_sha256') and value is not None:_digest(value)
            elif value is not None and not (type(value) is bool or type(value) is int and abs(value)<=2**63-1 or type(value) is str and len(value)<=256):
                raise ValueError('primitive bounded event metadata required')
        if item['event']=='call_capacity_reserved':
            from .call_attempts import _binding
            if (set(fields)!=EVENT_FIELDS[item['event']] or fields['role'] not in ('perception','small_vlm','strong_vlm','text_llm') or
                    not _binding(self.declarations['plan']['comparison']['cost']['rate_card'],fields['provider'],fields['revision'])):
                raise ValueError('observed provider outside frozen rate card')
        if item['event']=='work_capacity_reserved' and (
                not {'kind','worker_id'}<=set(fields) or fields['kind'] not in ('worker_lifecycle','worker_operation') or
                type(fields['worker_id']) is not str or not 1<=len(fields['worker_id'])<=128):
            raise ValueError('bounded original worker identity required')
        name=item['event']
        if name in ('call_capacity_reserved','controller_dispatched','backend_entry_admission',
                    'backend_invocation_finished','call_publication_finished','call_capacity_released',
                    'work_finished','private_mappings_invalidated','external_billing_attached') and set(fields)!=EVENT_FIELDS[name]:
            raise ValueError('complete structural event metadata required')
        if name=='controller_dispatched' and (fields['liability']!='unknown_until_external_billing' or
                fields['relationship'] not in ('root','retry','escalation') or
                (fields['predecessor'] is not None and not _attempt_id(fields['predecessor'])) or
                (fields['relationship']=='root') is not (fields['predecessor'] is None)):
            raise ValueError('known native dispatch relationship required')
        for key in ({'controller_dispatched':('execution_sha256',),
                     'backend_entry_admission':('execution_instance_sha256',),
                     'external_billing_attached':('evidence_sha256',),
                     'call_capacity_reserved':('evidence_sha256','prompt_sha256')}.get(name,())):
            _digest(fields[key])
        if name=='backend_invocation_finished' and (fields['status'] not in ('success','error','cancelled') or
                fields['usage_verified'] is not False or
                fields['attempt_id'] is not None and not _attempt_id(fields['attempt_id']) or
                fields['reported_usage'] is not None and fields['reported_usage']['cached_input_tokens']>fields['reported_usage']['input_tokens']):
            raise ValueError('closed native callback metadata required')
        if name=='call_publication_finished' and (fields['outcome'] not in CALL_OUTCOMES or type(fields['invoked']) is not bool):
            raise ValueError('known call publication outcome required')
        if name=='call_capacity_released' and (fields['reason'] not in CALL_OUTCOMES|{'unadmitted'} or
                type(fields['invoked']) is not bool or type(fields['dispatched']) is not bool):
            raise ValueError('known call release reason required')
        if name=='work_finished' and (fields['kind'] not in ('worker_lifecycle','worker_operation') or fields['outcome'] not in WORK_OUTCOMES):
            raise ValueError('known worker terminal outcome required')
        if name=='private_mappings_invalidated' and (item['operation_id'] is not None or
                fields['reason'] not in ('delete','delete_data','reset','reset_processing','pause','consent_revoked','consent','close') or
                type(fields['affected']) is not int or not 0<=fields['affected']<=128):
            raise ValueError('known bounded private invalidation required')
        if name=='external_billing_attached' and (item['operation_id'] is not None or type(fields['operation_id']) is not str or
                len(fields['operation_id'])!=32 or any(c not in '0123456789abcdef' for c in fields['operation_id'])):
            raise ValueError('original opaque billing operation required')
        if name not in ('private_mappings_invalidated','external_billing_attached') and item['operation_id'] is None:
            raise ValueError('observed operation identity required')
        if name=='work_capacity_reserved' or name in ('spawn_begin','handle_acquired','handshake_ready','handshake_failed',
                'reap_confirmed','operation_started','operation_finished','invalidated'):
            if 'status' in fields and fields['status'] not in WORK_OUTCOMES|{'success','failed','ready'}:
                raise ValueError('known worker milestone status required')
            if 'operation' in fields and fields['operation'] not in WORK_OPERATIONS:raise ValueError('known worker operation required')
            if 'returncode' in fields and fields['returncode'] is not None and type(fields['returncode']) is not int:
                raise ValueError('integer worker return code required')
            parent=fields.get('parent_operation_id')
            if parent is not None and (type(parent) is not str or len(parent)!=32 or any(c not in '0123456789abcdef' for c in parent)):
                raise ValueError('opaque original parent operation required')
        self.events.append(item);self.native_previous=item['sha256']
    def _report(self,item):
        from .stream_attempts import REPORT_KEYS,ATTEMPT_KEYS,_scope
        from .call_attempts import _binding
        _keys(item,REPORT_KEYS,'native runtime report')
        scope=item['scope_kind']
        if (type(item['schema_version']) is not int or item['schema_version']!=3 or item['closed'] is not True or
                scope not in ('task','stream') or any(r['scope_kind']==scope or r['scope']==item['scope'] for r in self.reports) or
                item['sha256']!=plan_digest({k:v for k,v in item.items() if k!='sha256'}) or
                type(item['attempts']) is not list or len(item['attempts'])>128 or
                item['task_id']!=(self.bootstrap['task_id'] if scope=='task' else None)):
            raise ValueError('bounded closed native task/stream report required')
        data=self.declarations;b=self.bootstrap
        _,header=_scope(data['plan'],data['collection'],data['inputs'],data['stream_protocol'],data['reconciliation_protocol'],
            expected_plan_sha256=b['comparison_plan_sha256'],expected_collection_sha256=b['collection_sha256'],
            expected_stream_protocol_sha256=b['stream_protocol_sha256'],expected_reconciliation_protocol_sha256=b['reconciliation_protocol_sha256'],
            scope_kind=scope,task_id=item['task_id'],arm=b['arm'],stream_id=b['stream_id'])
        if any(_encoded(item[key])!=_encoded(value) for key,value in header.items()):raise ValueError('report original native header mismatch')
        if (type(item['scope']) is not str or len(item['scope'])!=32 or any(c not in '0123456789abcdef' for c in item['scope']) or
                type(item['max_attempts']) is not int or not len(item['attempts'])<=item['max_attempts']<=128 or
                type(item['rejected']) is not int or item['rejected']!=0 or item['clock']!='monotonic_callback_duration_not_task_latency'):
            raise ValueError('bounded unrejected native scope required')
        card=data['plan']['comparison']['cost']['rate_card'];seen={};last=-1;claims=set();instances=set()
        for index,attempt in enumerate(item['attempts']):
            _keys(attempt,ATTEMPT_KEYS,'native runtime attempt')
            if (attempt['attempt_id']!=item['scope']+':'+str(index) or attempt['role'] not in ('perception','small_vlm','strong_vlm','text_llm') or
                    attempt['phase']!=data['instrumentation_protocol']['phase'] or attempt['status'] not in ('success','error','cancelled')):
                raise ValueError('ordered terminal native attempt required')
            required=_binding(card,attempt['provider'],attempt['revision'])
            if not required:raise ValueError('native provider outside frozen rate card')
            for key in ('evidence_sha256','prompt_sha256','execution_sha256','execution_instance_sha256'):
                if attempt[key] is not None:_digest(attempt[key])
            if attempt['evidence_sha256'] is None:raise ValueError('native evidence digest required')
            start,end=attempt['started_ns'],attempt['ended_ns']
            if type(start) is not int or type(end) is not int or not max(0,last)<=start<=end<=2**63-1:
                raise ValueError('ordered completed native callback required')
            last=start;parent=seen.get(attempt['predecessor']) if type(attempt['predecessor']) is str else None
            if attempt['relationship']=='root':
                if attempt['predecessor'] is not None:raise ValueError('root native predecessor unsupported')
            elif attempt['relationship'] in ('retry','escalation'):
                if parent is None or parent['ended_ns']>start or parent['evidence_sha256']!=attempt['evidence_sha256']:
                    raise ValueError('native completed predecessor mismatch')
                if attempt['relationship']=='retry' and any(parent[k]!=attempt[k] for k in ('provider','revision','role')):
                    raise ValueError('native retry changed provider')
            else:raise ValueError('unknown native relationship')
            quantity=attempt['cost_quantities'];lines=attempt['line_claims'];instance=attempt['execution_instance_sha256']
            if quantity is None or lines is None:
                if quantity is not None or lines is not None or instance is not None:raise ValueError('partial native billing declaration')
            else:
                if type(quantity) is not dict or set(quantity)!=required or any(type(v) is not int or not 0<=v<=10**12 for v in quantity.values()):
                    raise ValueError('bounded native quantity declaration required')
                if type(lines) is not list or not 1<=len(lines)<=10000 or len(set(lines))!=len(lines):raise ValueError('bounded original line claims required')
                for line in lines:_digest(line)
                if set(lines)&claims or instance is None or instance in instances:raise ValueError('duplicate native line/instance ownership')
                claims.update(lines);instances.add(instance)
            seen[attempt['attempt_id']]=attempt
        self.reports.append(item)

    def _summary(self,summary):
        keys=('schema_version','comparison_plan_sha256','collection_sha256','stream_protocol_sha256',
            'reconciliation_protocol_sha256','instrumentation_protocol_sha256','task_id','arm','stream_id',
            'session_id','clock_domain','started_ns','ended_ns','closed','terminal','retention',
            'sdk_deletion_erases_evaluation_artifacts','private_subject_mappings','provider_counters',
            'worker_measurements','dispatched_without_invocation','incomplete_reasons','unresolved_attempt_billing',
            'unresolved_billing','monetary_cost','physical_capture_counters','required_boundary_lanes','source_availability',
            'continuous_recording_verified','measurement_status','actual_acceptance','instrumentation_coverage_verified',
            'receipt_authenticity_verified','measurement_truth_verified','isolation_verified','whole_pipeline_cost_verified')
        _keys(summary,keys,'runtime summary')
        for key in ('comparison_plan_sha256','collection_sha256','stream_protocol_sha256',
                    'reconciliation_protocol_sha256','instrumentation_protocol_sha256','task_id','arm','stream_id'):
            if summary[key]!=self.bootstrap[key]:raise ValueError('terminal identity mismatch')
        sid=summary['session_id']
        if (type(summary['schema_version']) is not int or summary['schema_version']!=1 or
                type(sid) is not str or len(sid)!=32 or any(c not in '0123456789abcdef' for c in sid) or
                summary['clock_domain']!='host-monotonic:'+sid or self.events and sid!=self.session_id or
                summary['closed'] is not True or summary['terminal'] is not True or
                summary['retention']!='explicit_external_owner_retains_task_and_evidence_hashes' or
                type(summary['private_subject_mappings']) is not int or summary['private_subject_mappings']!=0 or
                summary['monetary_cost'] is not None or summary['physical_capture_counters'] is not None or
                summary['actual_acceptance']!='incomplete' or summary['measurement_status']!='observed_host_lifecycle_only' or
                summary['source_availability']!='unknown; SDK lifecycle is not physical capture availability'):
            raise ValueError('bounded unknown-evidence terminal required')
        for key in ('sdk_deletion_erases_evaluation_artifacts','continuous_recording_verified','instrumentation_coverage_verified',
                    'receipt_authenticity_verified','measurement_truth_verified','isolation_verified','whole_pipeline_cost_verified'):
            if summary[key] is not False:raise ValueError('runtime observation cannot assert acceptance')
        start,end=summary['started_ns'],summary['ended_ns']
        if (type(start) is not int or type(end) is not int or not 0<=start<=end<=2**63-1 or
                any(not start<=e['monotonic_ns']<=end for e in self.events)):raise ValueError('summary journal clock mismatch')
        counters={};rows=[a for r in self.reports for a in r['attempts']];attempts={a['attempt_id']:a for a in rows}
        claims=[line for a in rows for line in (a['line_claims'] or [])]
        instances=[a['execution_instance_sha256'] for a in rows if a['execution_instance_sha256'] is not None]
        if len(attempts)!=len(rows) or len(claims)!=len(set(claims)) or len(instances)!=len(set(instances)):
            raise ValueError('unique native attempt/line/instance ownership across both scopes required')
        task=next(r for r in self.reports if r['scope_kind']=='task')
        for attempt in task['attempts']:
            key=(attempt['provider'],attempt['revision'],attempt['role']);counters[key]=counters.get(key,0)+1
        expected=[dict(provider=p,revision=r,role=role,count=n) for (p,r,role),n in sorted(counters.items())]
        if _encoded(summary['provider_counters'])!=_encoded(expected):raise ValueError('native observed provider counter mismatch')
        dispatched,invoked=self._calls(attempts,summary)
        if type(summary['dispatched_without_invocation']) is not int or summary['dispatched_without_invocation']!=len(dispatched-invoked):
            raise ValueError('dispatched unknown liability count mismatch')
        declaration=self.declarations['stream_protocol'];components={c['id']:c for c in declaration['components']}
        stream=next(s for s in declaration['streams'] if s['id']==self.bootstrap['stream_id'])
        lanes=[dict(component_id=cid,kind=components[cid]['kind'],unit=components[cid]['unit'],**lane,
            status='unknown',quantity=None,billed_amount=None,
            reason='runtime instrumentation does not measure this frozen resource/billing lane')
            for cid,req in stream['requirements'].items() for lane in req['lanes']]
        if _encoded(summary['required_boundary_lanes'])!=_encoded(lanes):raise ValueError('unknown required lane inventory mismatch')
        unresolved=any(a['cost_quantities'] is None for a in attempts.values())
        if summary['unresolved_attempt_billing'] is not unresolved or summary['unresolved_billing'] is not (unresolved or bool(dispatched-invoked) or any(l['kind']=='monetary' for l in lanes)):
            raise ValueError('unknown billing status mismatch')
        reasons=summary['incomplete_reasons']
        allowed={'control_event_capacity_exhausted','invalidation_journal_failed','invocations_not_drained',
            'journal_capacity_exhausted','journal_reservation_failed','journal_sink_failed','reserved_event_capacity_exhausted',
            'terminal_observation_failed','unfinished_owned_operation','worker_reap_unverified','native_admission_failed_before_callback',
            'native_attempt_capacity_exhausted','call_reservation_capacity_exhausted'}
        if type(reasons) is not list or any(type(r) is not str or r not in allowed for r in reasons) or reasons!=sorted(set(reasons)):
            raise ValueError('known incomplete reasons required')
        self._workers(summary['worker_measurements'],summary['incomplete_reasons'])

    def _calls(self,attempts,summary):
        operations={};finishes={};dispatched=set();invoked=set();billed=set();instances=set()
        for event in self.events:
            name=event['event'];ident=event['operation_id']
            if name=='external_billing_attached':
                target_id=event['fields']['operation_id'];target=operations.get(target_id,{})
                if 'backend_entry_admission' not in target or target_id in billed:
                    raise ValueError('unique billing requires original observed callback admission')
                billed.add(target_id)
            if name not in ('call_capacity_reserved','controller_dispatched','backend_entry_admission',
                            'backend_invocation_finished','call_publication_finished','call_capacity_released'):continue
            if name=='call_capacity_reserved':
                if ident in operations or len(operations)>=MAX_CALL_RESERVATIONS:raise ValueError('duplicate/excess observed call reservation')
                operations[ident]={}
            marks=operations.get(ident)
            if marks is None or name in marks or 'call_publication_finished' in marks or 'call_capacity_released' in marks:
                raise ValueError('unique ordered observed call milestones required')
            if name=='controller_dispatched':
                dispatched.add(ident)
            if name=='backend_entry_admission' and 'controller_dispatched' not in marks:
                raise ValueError('backend admission requires original dispatch')
            if name=='backend_entry_admission':
                instance=event['fields']['execution_instance_sha256']
                if instance in instances:raise ValueError('unique observed callback execution instance required')
                instances.add(instance)
            if name=='backend_invocation_finished':
                if 'backend_entry_admission' not in marks:raise ValueError('callback finish requires original admission')
                aid=event['fields']['attempt_id']
                if aid is None:
                    if 'native_admission_failed_before_callback' not in summary['incomplete_reasons']:
                        raise ValueError('missing callback must remain explicitly incomplete')
                else:
                    if aid in finishes or aid not in attempts:raise ValueError('unique observed native callback required')
                    attempt=attempts[aid];reserve=marks['call_capacity_reserved']['fields'];dispatch=marks['controller_dispatched']['fields']
                    # Native timestamps are relative perf-counter values; the
                    # journal uses absolute monotonic values. Only their elapsed
                    # durations share this explicitly host-local interpretation.
                    if (any(attempt[k]!=reserve[k] for k in ('provider','revision','role','evidence_sha256','prompt_sha256')) or
                            any(attempt[k]!=dispatch[k] for k in ('execution_sha256','predecessor','relationship')) or
                            attempt['status']!=event['fields']['status'] or
                            attempt['ended_ns']-attempt['started_ns']>event['monotonic_ns']-marks['backend_entry_admission']['monotonic_ns'] or
                            attempt['execution_instance_sha256'] is not None and
                            attempt['execution_instance_sha256']!=marks['backend_entry_admission']['fields']['execution_instance_sha256']):
                        raise ValueError('native callback differs from its observed operation')
                    finishes[aid]=event;invoked.add(ident)
            if name in ('call_publication_finished','call_capacity_released'):
                fields=event['fields'];entered=ident in invoked
                if (fields['invoked'] is not entered or 'backend_entry_admission' in marks and 'backend_invocation_finished' not in marks or
                        name=='call_publication_finished' and ident not in dispatched or
                        name=='call_capacity_released' and fields['dispatched'] is not (ident in dispatched)):
                    raise ValueError('call terminal differs from original dispatch/callback')
            marks[name]=event
        if set(finishes)!=set(attempts):raise ValueError('native attempt inventory differs from observed invocation journal')
        if not billed<=invoked or any((event['operation_id'] in billed) is not (attempts[aid]['cost_quantities'] is not None)
                for aid,event in finishes.items()):
            raise ValueError('native billing inventory differs from observed attachment journal')
        for marks in operations.values():
            if not ({'call_publication_finished','call_capacity_released'}&set(marks)) and 'unfinished_owned_operation' not in summary['incomplete_reasons']:
                raise ValueError('missing observed call terminal')
        return dispatched,invoked

    def _workers(self,workers,reasons):
        tickets={};marks={};worker_ids=set();calls=set();finished=set()
        for event in self.events:
            ident=event['operation_id'];name=event['event']
            if name=='call_capacity_reserved':
                if ident in tickets:raise ValueError('operation identity cannot be reused across call/work')
                calls.add(ident)
            if name=='work_capacity_reserved':
                if ident in tickets or ident in calls:raise ValueError('duplicate observed worker operation')
                fields=event['fields'];worker=fields['worker_id'];parent=fields.get('parent_operation_id')
                if parent is not None and parent not in calls:raise ValueError('worker parent must be an observed original call')
                if fields['kind']=='worker_lifecycle':
                    if worker in worker_ids:raise ValueError('unique observed worker lifetime required')
                    worker_ids.add(worker)
                elif worker not in worker_ids or 'operation' not in fields:
                    raise ValueError('worker operation requires an admitted original worker')
                tickets[ident]=fields;marks[ident]={}
            worker_names={'spawn_begin','handle_acquired','handshake_ready','handshake_failed','reap_confirmed',
                          'operation_started','operation_finished','invalidated','work_finished'}
            if name in worker_names:
                if ident not in tickets or name in marks[ident] or ident in finished:
                    raise ValueError('unique ordered worker milestones required')
                fields=event['fields'];ticket=tickets[ident];seen=marks[ident]
                if name in ('operation_started','operation_finished') and ticket['kind']!='worker_operation' or name in (
                        'spawn_begin','handle_acquired','handshake_ready','handshake_failed','reap_confirmed') and ticket['kind']!='worker_lifecycle':
                    raise ValueError('worker milestone belongs to another operation kind')
                if name in ('handshake_ready','handshake_failed') and ({'handshake_ready','handshake_failed'}&set(seen)):
                    raise ValueError('one observed worker handshake outcome required')
                if (name=='spawn_begin' and len(seen)!=1 or name=='handle_acquired' and 'spawn_begin' not in seen or
                        name=='handshake_ready' and 'handle_acquired' not in seen or
                        name=='reap_confirmed' and 'handle_acquired' not in seen or
                        name=='operation_finished' and 'operation_started' not in seen):
                    raise ValueError('worker milestone lacks original preceding ownership')
                if any(fields[key]!=ticket.get(key) for key in ('operation','parent_operation_id') if key in fields):
                    raise ValueError('worker milestone changed original operation identity')
                if name=='work_finished':
                    if fields['kind']!=ticket['kind']:raise ValueError('worker terminal changed original kind')
                    finished.add(ident)
            if ident in marks:marks[ident][name]=event['monotonic_ns']
        if set(tickets)-finished and 'unfinished_owned_operation' not in reasons:
            raise ValueError('unfinished worker requires explicit incomplete accounting')
        if any(ticket['kind']=='worker_lifecycle' and 'handle_acquired' in marks[ident] and 'reap_confirmed' not in marks[ident]
                for ident,ticket in tickets.items()) and 'worker_reap_unverified' not in reasons:
            raise ValueError('unreaped worker requires explicit incomplete accounting')
        lifetimes=[(ident,item) for ident,item in tickets.items() if item['kind']=='worker_lifecycle']
        if len(lifetimes)>32 or len(tickets)-len(lifetimes)>1024 or type(workers) is not list or len(workers)!=len(lifetimes):
            raise ValueError('bounded original worker inventory required')
        expected=[]
        for ident,item in lifetimes:
            milestone=marks[ident];begin=milestone.get('handle_acquired');end=milestone.get('reap_confirmed')
            a=milestone.get('spawn_begin');b=milestone.get('handshake_ready',milestone.get('handshake_failed'))
            operations=[dict(operation_id=key,operation=value['operation'],parent_operation_id=value.get('parent_operation_id'),
                started_ns=marks[key].get('operation_started'),ended_ns=marks[key].get('work_finished'))
                for key,value in tickets.items() if value['kind']=='worker_operation' and value['worker_id']==item['worker_id']]
            busy=idle=None
            if begin is not None and end is not None and all(o['started_ns'] is None or o['ended_ns'] is not None for o in operations):
                intervals=sorted((max(begin,o['started_ns']),min(end,o['ended_ns'])) for o in operations if o['started_ns'] is not None and max(begin,o['started_ns'])<min(end,o['ended_ns']))
                cursor=begin;busy=0
                for first,last in intervals:busy+=max(0,last-max(cursor,first));cursor=max(cursor,last)
                idle=end-begin-busy
            expected.append(dict(worker_id=item['worker_id'],startup_elapsed_ns=None if a is None or b is None else b-a,
                residency_elapsed_ns=None if begin is None or end is None else end-begin,reap_confirmed=end is not None,
                observed_exchange_union_ns=busy,observed_no_exchange_gap_ns=idle,operations=operations,
                interpretation='host elapsed interval proxy; overlapping intervals must not be summed as CPU or money'))
        if _encoded(workers)!=_encoded(expected):raise ValueError('worker summary differs from observed milestones')


class ParentChannel(RuntimeValidationState):
    """One invocation's sockets; the collector alone certifies final source binding."""
    def __init__(self,bootstrap,*,budget=None):
        self.bootstrap=_strict_decode(_encoded(bootstrap))
        super().__init__(self.bootstrap, load_declarations(self.bootstrap))
        self.budget=budget if budget is not None else CollectionBudget()
        self.parent,self.child=socket.socketpair()
        self.parent.setblocking(False);self.child.set_inheritable(False)
        self.terminal=None;self.hello=False;self.eof=False
        self.sequence=0;self.previous='0'*64;self.reason=None;self.closed=False
    def prepare(self,environment,remaining_seconds):
        if FD_ENVIRONMENT_KEY in environment: raise ValueError('evaluation FD must be an explicit transport delta')
        self.bootstrap['deadline_ns']=time.monotonic_ns()+int(remaining_seconds*1e9)
        validate_bootstrap(self.bootstrap)
        frame(self.bootstrap)
        return dict(env=dict(environment,**{FD_ENVIRONMENT_KEY:str(self.child.fileno())}),
                    pass_fds=(self.child.fileno(),),close_fds=True)
    def spawned(self):
        self.child.close()
    async def collect(self):
        try:
            hello_deadline=min(self.bootstrap['deadline_ns'],time.monotonic_ns()+self.bootstrap['hello_timeout_ms']*1000000)
            await _until(asyncio.get_running_loop().sock_sendall(self.parent,frame(self.bootstrap)),hello_deadline)
            hello=await _until(receive(self.parent,self.budget),hello_deadline)
            if (type(hello) is not dict or type(hello.get('schema_version')) is not int or
                    _encoded(hello)!=_encoded({'schema_version':1,'type':'hello','bootstrap':self.bootstrap})): raise ValueError('runtime hello identity mismatch')
            self.hello=True
            while True:
                item=await _until(receive(self.parent,self.budget),self.bootstrap['deadline_ns'])
                if item is None:
                    self.eof=True
                    if self.terminal is None: raise ValueError('missing runtime terminal')
                    return
                if self.terminal is not None: raise ValueError('trailing runtime frame after terminal')
                _keys(item,FRAME_KEYS,'runtime record')
                if (type(item['schema_version']) is not int or item['schema_version']!=1 or item['nonce']!=self.bootstrap['nonce'] or
                        type(item['sequence']) is not int or item['sequence']!=self.sequence or
                        item['previous_sha256']!=self.previous or
                        item['sha256']!=plan_digest({k:v for k,v in item.items() if k!='sha256'})):
                    raise ValueError('runtime binding/sequence/hash mismatch')
                kind=item['type'];payload=item['payload']
                if kind=='event': self._event(payload)
                elif kind=='report': self._report(payload)
                elif kind=='terminal':
                    _keys(payload,('report_sha256s','summary'),'runtime terminal')
                    if (len(self.reports)!=2 or payload['report_sha256s']!=[r['sha256'] for r in self.reports] or
                            type(payload['summary']) is not dict): raise ValueError('runtime terminal inventory mismatch')
                    summary=payload['summary']
                    self._summary(summary)
                    self.terminal=payload
                else: raise ValueError('unknown runtime record type')
                self.sequence+=1;self.previous=item['sha256']
        except asyncio.CancelledError:
            self.reason='cancelled';raise
        except Exception as error:
            self.reason='deadline_exceeded' if isinstance(error,asyncio.TimeoutError) else 'invalid_channel'
            if isinstance(error,asyncio.TimeoutError):raise
            raise RuntimeChannelError('invalid runtime side channel') from None
        finally:self.parent.close()
    def close(self):
        self.closed=True;self.parent.close();self.child.close()
        if self.reason is None and (not self.hello or not self.eof or self.terminal is None): self.reason='incomplete_channel'
    def snapshot(self):
        complete=self.reason is None and self.eof and self.terminal is not None and not self.terminal['summary']['incomplete_reasons']
        return dict(schema_version=1,bootstrap=self.bootstrap,hello_verified=self.hello,eof=self.eof,
            records=self.sequence,journal=self.events,native_reports=self.reports,terminal=self.terminal,
            validated_prefix_sha256=self.previous,native_journal_sha256=self.native_previous,
            instrumentation_complete=complete,
            incomplete_reason=self.reason if self.reason is not None else (
                'session_accounting_incomplete' if self.terminal and self.terminal['summary']['incomplete_reasons'] else None),
            actual_acceptance='incomplete',receipt_authenticity_verified=False,
            measurement_truth_verified=False,instrumentation_coverage_verified=False,isolation_verified=False)


class ChildChannel:
    def __init__(self,sock,bootstrap):
        self.socket=sock;self.bootstrap=bootstrap;self.queue=asyncio.Queue(MAX_QUEUE_FRAMES)
        self.queued_bytes=0;self.used=0;self.events=0;self.reserved_events=0;self.reserved_bytes=0
        self.control_events=0;self.control_bytes=0;self.sequence=0;self.previous='0'*64
        self.failure=None;self.closed=False;self.writer=asyncio.create_task(self._write())
        self.close_task=None;self.abort_task=None
    @classmethod
    async def from_environment(cls):
        raw=os.environ.get(FD_ENVIRONMENT_KEY)
        if raw is None or not raw.isdecimal(): raise ValueError('explicit inherited evaluation FD required')
        descriptor=int(raw);os.set_inheritable(descriptor,False)
        sock=socket.socket(fileno=descriptor);sock.setblocking(False)
        try:
            # Parent enforces the frozen hello deadline; this initial read is
            # independently bounded, then the original deadline is authoritative.
            bootstrap=validate_bootstrap(await asyncio.wait_for(receive(sock),30))
            declarations=load_declarations(bootstrap)
            await _until(asyncio.get_running_loop().sock_sendall(sock,frame(dict(schema_version=1,
                type='hello',bootstrap=bootstrap))),bootstrap['deadline_ns'])
            self=cls(sock,bootstrap);self.declarations=declarations
            return self
        except BaseException:sock.close();raise
    def _admit(self):
        if self.closed or self.failure is not None or self.writer.done(): raise RuntimeError('runtime journal writer unavailable')
    def reserve(self,events,byte_count):
        self._admit()
        if type(events) is not int or type(byte_count) is not int or events<1 or byte_count<1: raise ValueError('positive journal reservation required')
        wire=self.reserved_bytes+byte_count+(self.reserved_events+events)*1024
        if (self.events+self.reserved_events+events>MAX_EVENTS-CONTROL_EVENTS or
                self.queue.qsize()+self.reserved_events+events>MAX_QUEUE_FRAMES-1 or
                self.queued_bytes+wire>MAX_QUEUE_BYTES-4096 or self.used+wire+CONTROL_BYTES+FINAL_BYTES>MAX_BYTES):
            self.failure='capacity_exhausted';raise RuntimeError('runtime channel capacity exhausted before admission')
        self.reserved_events+=events;self.reserved_bytes+=byte_count
    def release(self,unused_events,unused_bytes):
        if (type(unused_events) is not int or type(unused_bytes) is not int or
                not 0<=unused_events<=self.reserved_events or not 0<=unused_bytes<=self.reserved_bytes):
            raise ValueError('invalid unused channel reservation')
        self.reserved_events-=unused_events;self.reserved_bytes-=unused_bytes
    def _enqueue(self,kind,payload):
        item=dict(schema_version=1,type=kind,nonce=self.bootstrap['nonce'],sequence=self.sequence,
                  previous_sha256=self.previous,payload=payload)
        item['sha256']=plan_digest(item);data=frame(item)
        if (self.queue.full() or self.queued_bytes+len(data)>MAX_QUEUE_BYTES or self.used+len(data)>MAX_BYTES):
            self.failure='capacity_exhausted';raise RuntimeError('runtime channel bounded queue exhausted')
        self.queue.put_nowait(data);self.queued_bytes+=len(data);self.used+=len(data)
        self.sequence+=1;self.previous=item['sha256']
    def emit(self,item):
        self._admit();item=_strict_decode(_encoded(item));size=len(_encoded(item))+1
        if size>4096 or self.events>=MAX_EVENTS: raise ValueError('bounded runtime event required')
        if item.get('operation_id') is None:
            if self.control_events>=CONTROL_EVENTS or self.control_bytes+size>CONTROL_BYTES:
                self.failure='control_capacity_exhausted';raise RuntimeError('runtime control credit exhausted')
            self.control_events+=1;self.control_bytes+=size
        else:
            if self.reserved_events<1 or self.reserved_bytes<size: raise RuntimeError('runtime event lacks reserved capacity')
            self.reserved_events-=1;self.reserved_bytes-=size
        self._enqueue('event',item);self.events+=1
    async def _write(self):
        try:
            # A completed wait_for socket write can consume cancellation on
            # Python 3.10. Abort revokes the FD first: never enter an idle queue
            # wait after that revocation, even if cancellation was consumed.
            # self.closed is not suitable here: normal close must drain first.
            while self.socket.fileno()>=0:
                data=await self.queue.get()
                try:
                    if data is None:return
                    await _until(asyncio.get_running_loop().sock_sendall(self.socket,data),self.bootstrap['deadline_ns'])
                    self.queued_bytes-=len(data)
                finally:self.queue.task_done()
        except BaseException as error:
            self.failure='writer_cancelled' if isinstance(error,asyncio.CancelledError) else 'writer_failed'
            self.socket.close()
    async def close(self,*,reports,summary):
        from .paired_replay import _finish_owned
        if self.abort_task is not None:
            try:return await asyncio.shield(self.abort_task)
            except asyncio.CancelledError:
                await _finish_owned(self.abort_task);raise
        if self.close_task is None:self.close_task=asyncio.create_task(self._close(reports,summary))
        try:return await asyncio.shield(self.close_task)
        except asyncio.CancelledError:
            await _finish_owned(self.close_task);raise
    async def abort(self):
        """Reap this journal's writer/socket without emitting a success terminal."""
        from .paired_replay import _finish_owned
        if self.abort_task is None:self.abort_task=asyncio.create_task(self._abort())
        try:return await asyncio.shield(self.abort_task)
        except asyncio.CancelledError:
            await _finish_owned(self.abort_task);raise
    async def _abort(self):
        self.closed=True;self.failure='aborted'
        # Revoke the FD before joining cancellation: Python 3.10 wait_for can
        # consume cancellation racing a completed socket write. A later writer
        # iteration must fail immediately instead of using the original deadline.
        self.socket.close()
        try:
            if self.close_task is not None and not self.close_task.done():self.close_task.cancel()
            if self.close_task is not None:await asyncio.gather(self.close_task,return_exceptions=True)
        finally:
            self.socket.close()
            if not self.writer.done():self.writer.cancel()
            await asyncio.gather(self.writer,return_exceptions=True)
            self.socket.close()
    async def _close(self,reports,summary):
        if self.closed:return
        self.closed=True
        deadline=min(self.bootstrap['deadline_ns'],time.monotonic_ns()+int(CLOSE_TIMEOUT_SECONDS*1e9))
        try:
            if self.failure is not None:raise RuntimeError('runtime writer incomplete')
            if type(reports) is not list or len(reports)!=2:raise ValueError('two native scope reports required')
            await _until(self.queue.join(),deadline)
            for report in reports:
                self._enqueue('report',report)
                await _until(self.queue.join(),deadline)
            self._enqueue('terminal',dict(report_sha256s=[r['sha256'] for r in reports],summary=summary))
            await _until(self.queue.join(),deadline)
            self.queue.put_nowait(None)
            await _until(asyncio.shield(self.writer),deadline)
            self.socket.shutdown(socket.SHUT_WR)
        finally:
            self.socket.close()
            if not self.writer.done():self.writer.cancel()
            await asyncio.gather(self.writer,return_exceptions=True)
            self.socket.close()


def build_bootstrap(*,paths,hashes,task_id,arm,stream_id,request_sha256,hello_timeout_ms):
    return dict(schema_version=1,type='bootstrap',nonce=secrets.token_hex(32),**hashes,
        task_id=task_id,arm=arm,stream_id=stream_id,request_sha256=request_sha256,
        deadline_ns=1,hello_timeout_ms=hello_timeout_ms,paths=paths)
