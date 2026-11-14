"""Opt-in owned SDK execution observations; measured clocks are not monetary bills."""
import asyncio
from copy import deepcopy
import hashlib
import time
import uuid

from .paired_replay import _encoded
from .perception_comparison import _digest, plan_digest
from .stream_attempts import StreamAttemptRecorder
from .three_arm_comparison import _snapshot

MAX_EVENTS = 8192
MAX_BYTES = 16*1024*1024
MAX_OPERATIONS = 1024
MAX_WORKERS = 32


class EvaluationSession:
    """Explicit external artifact owner; SDK sensory deletion does not erase artifacts.

    Retained task/evidence hashes are correlation data. No prompts, answers, pixels,
    raw subject IDs or exception messages belong in this observer's journal.
    """
    def __init__(self, plan, spec, inputs, stream_protocol, reconciliation_protocol, *,
                 expected_plan_sha256, expected_collection_sha256, expected_stream_protocol_sha256,
                 expected_reconciliation_protocol_sha256, task_id, arm, stream_id,
                 retain_evaluation_hashes=False, phase='unknown', journal=None,
                 max_events=MAX_EVENTS, max_bytes=MAX_BYTES, max_attempts=128):
        if retain_evaluation_hashes is not True:
            raise PermissionError('explicit separate evaluation hash retention consent required')
        if phase not in ('cold','warm','unknown'):
            raise ValueError('explicit observed/declaration phase required')
        if type(max_events) is not int or not 8 <= max_events <= MAX_EVENTS:
            raise ValueError('8..8192 journal events required')
        if type(max_bytes) is not int or not 4096 <= max_bytes <= MAX_BYTES:
            raise ValueError('bounded evaluation byte budget required')
        if type(max_attempts) is not int or not 1 <= max_attempts <= 128:
            raise ValueError('1..128 native attempts per scope required')
        args=dict(expected_plan_sha256=expected_plan_sha256,expected_collection_sha256=expected_collection_sha256,
            expected_stream_protocol_sha256=expected_stream_protocol_sha256,
            expected_reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,
            arm=arm,stream_id=stream_id,max_attempts=max_attempts)
        self._task=StreamAttemptRecorder(plan,spec,inputs,stream_protocol,reconciliation_protocol,
            scope_kind='task',task_id=task_id,**args)
        self._stream=StreamAttemptRecorder(plan,spec,inputs,stream_protocol,reconciliation_protocol,
            scope_kind='stream',task_id=None,**args)
        self.id=uuid.uuid4().hex; self.clock_domain='host-monotonic:'+self.id
        self._stream_declaration=next(s for s in _snapshot(stream_protocol)['streams'] if s['id']==stream_id)
        self._components={c['id']:c for c in _snapshot(stream_protocol)['components']}
        self._phase=phase;self._journal=journal;self._max_events=max_events;self._max_bytes=max_bytes
        self._limit=max_attempts;self._events=[];self._bytes=0;self._reserved=0;self._reserved_bytes=0
        self._closed=False;self._terminal=False;self._revoked=False;self._incomplete=set()
        self._terminal_result=None;self._close_task=None
        self._tickets={};self._workers={};self._links={};self._operations=0;self._calls=0;self._running=set()
        self._started=time.monotonic_ns();self._ended=None
        self._identity=dict(comparison_plan_sha256=expected_plan_sha256,collection_sha256=expected_collection_sha256,
            stream_protocol_sha256=expected_stream_protocol_sha256,reconciliation_protocol_sha256=expected_reconciliation_protocol_sha256,
            task_id=task_id,arm=arm,stream_id=stream_id)

    @classmethod
    def from_channel(cls,channel):
        """Load only the parent-owned bounded declaration artifacts, never code."""
        from .runtime_channel import load_declarations
        bootstrap=channel.bootstrap
        paths=bootstrap['paths']
        names=('plan','collection','inputs','stream_protocol','allocation_protocol',
               'reconciliation_protocol','instrumentation_protocol')
        if type(paths) is not dict or set(paths)!=set(names):
            raise ValueError('exact parent declaration path inventory required')
        values=deepcopy(channel.declarations) if hasattr(channel,'declarations') else load_declarations(bootstrap)
        if type(values) is not dict or set(values)!=set(names):
            raise ValueError('exact verified parent declaration payloads required')
        for name,key in (('plan','comparison_plan_sha256'),('collection','collection_sha256'),('inputs','inputs_sha256'),
                ('stream_protocol','stream_protocol_sha256'),('allocation_protocol','allocation_protocol_sha256'),
                ('reconciliation_protocol','reconciliation_protocol_sha256'),
                ('instrumentation_protocol','instrumentation_protocol_sha256')):
            if plan_digest(values[name])!=_digest(bootstrap[key]):
                raise ValueError('parent declaration path content differs from bootstrap digest')
        policy=values['instrumentation_protocol']
        if policy.get('retention_policy')!='external_owner_retains_task_and_evidence_hashes':
            raise PermissionError('explicit external evaluation artifact retention required')
        result=cls(values['plan'],values['collection'],values['inputs'],values['stream_protocol'],values['reconciliation_protocol'],
            expected_plan_sha256=bootstrap['comparison_plan_sha256'],expected_collection_sha256=bootstrap['collection_sha256'],
            expected_stream_protocol_sha256=bootstrap['stream_protocol_sha256'],
            expected_reconciliation_protocol_sha256=bootstrap['reconciliation_protocol_sha256'],
            task_id=bootstrap['task_id'],arm=bootstrap['arm'],stream_id=bootstrap['stream_id'],
            retain_evaluation_hashes=True,phase=policy['phase'],journal=channel)
        result._identity['instrumentation_protocol_sha256']=bootstrap['instrumentation_protocol_sha256']
        return result

    def _mark(self, reason):
        # Finite reason vocabulary, never exception content.
        self._incomplete.add(reason)

    def _reserve(self, events, size):
        if self._closed or self._terminal or self._revoked or self._incomplete:
            raise RuntimeError('evaluation observer is closed, invalidated or incomplete')
        # One bounded terminal-control event is reserved outside regular admission.
        if len(self._events)+self._reserved+events+1 > self._max_events or self._bytes+self._reserved_bytes+size+1024 > self._max_bytes:
            self._mark('journal_capacity_exhausted')
            raise RuntimeError('evaluation journal capacity exhausted before admission')
        if self._journal is not None and hasattr(self._journal,'reserve'):
            try:self._journal.reserve(events,size)
            except BaseException:
                self._mark('journal_reservation_failed')
                raise
        self._reserved+=events;self._reserved_bytes+=size

    def _emit(self, ticket, event, fields=None):
        if self._terminal: raise RuntimeError('immutable evaluation terminal already frozen')
        fields={} if fields is None else _snapshot(fields)
        now=time.monotonic_ns()
        item=dict(schema_version=1,session_id=self.id,clock_domain=self.clock_domain,
            sequence=len(self._events),previous_sha256=self._events[-1]['sha256'] if self._events else '0'*64,
            monotonic_ns=now,operation_id=None if ticket is None else ticket.id,event=event,fields=fields)
        item['sha256']=plan_digest(item);size=len(_encoded(item))+1
        if ticket is None and (len(self._events)+self._reserved+1>=self._max_events or
                self._bytes+self._reserved_bytes+size+1024>self._max_bytes):
            self._mark('control_event_capacity_exhausted')
            raise RuntimeError('evaluation control event exceeds unreserved capacity')
        if size>4096 or (ticket is not None and (ticket._events_left<1 or ticket._bytes_left<size)):
            self._mark('reserved_event_capacity_exhausted');raise RuntimeError('reserved evaluation capacity exhausted')
        if ticket is not None:
            ticket._events_left-=1;ticket._bytes_left-=size
            self._reserved-=1;self._reserved_bytes-=size
        if len(self._events)>=self._max_events or self._bytes+size>self._max_bytes:
            self._mark('journal_capacity_exhausted');raise RuntimeError('evaluation journal bound exceeded')
        self._events.append(item);self._bytes+=size
        if self._journal is not None:
            try:self._journal.emit(item)
            except BaseException:
                self._mark('journal_sink_failed')
                raise
        return now

    def _release_capacity(self,ticket):
        self._reserved-=ticket._events_left;self._reserved_bytes-=ticket._bytes_left
        if self._journal is not None and hasattr(self._journal,'release'):
            self._journal.release(ticket._events_left,ticket._bytes_left)
        ticket._events_left=0;ticket._bytes_left=0

    def link_call(self,request_id,*,predecessor_operation_id,relationship):
        """Explicit next retry/escalation metadata; never an automatic new call."""
        if self._closed or self._terminal or self._revoked:
            raise RuntimeError('evaluation admission closed')
        previous=self._tickets.get(predecessor_operation_id)
        if (type(request_id) is not str or not 1<=len(request_id)<=256 or request_id in self._links or
                len(self._links)>=128 or relationship not in ('retry','escalation') or
                not isinstance(previous,_Call) or previous.attempt_id is None or not previous._finished):
            raise ValueError('bounded explicit completed predecessor required')
        self._links[request_id]=(previous.attempt_id,relationship)

    def reserve_call(self, controller, request_id, *, provider, model_revision, prompt_sha256):
        if self._calls>=self._limit: raise RuntimeError('native task attempt admission bound reached')
        state=controller.snapshot()
        pending=next((r for r in state['requests'] if r['id']==request_id),None)
        if state['closed'] or pending is None or pending['state']!='pending':
            raise ValueError('original pending controller request required')
        evidence=pending['decision']['evidence'];role=pending['decision']['desired_route']
        if role not in ('text_llm','small_vlm','strong_vlm'):
            raise ValueError('explicit semantic provider route required')
        _digest(prompt_sha256)
        # _binding checks every provider/revision against the frozen native card.
        from .call_attempts import _binding
        if not _binding(self._task._card,provider,model_revision):
            raise ValueError('provider is outside frozen evaluation rate card')
        self._reserve(8,32768);self._calls+=1
        ticket=_Call(self,provider,model_revision,role,prompt_sha256,evidence['record_sha256'],evidence['subject_id'],controller,request_id)
        ticket._predecessor,ticket._relationship=self._links.pop(request_id,(None,'root'))
        self._tickets[ticket.id]=ticket
        try:self._emit(ticket,'call_capacity_reserved',dict(provider=provider,revision=model_revision,role=role,
            prompt_sha256=prompt_sha256,evidence_sha256=evidence['record_sha256']))
        except BaseException:
            ticket._scrub();self._release_capacity(ticket);ticket._finished=True
            raise
        return ticket

    def reserve_work(self,kind,metadata):
        metadata=_snapshot(metadata)
        if kind not in ('worker_lifecycle','worker_operation') or type(metadata) is not dict:
            raise ValueError('explicit owned worker lifecycle/operation required')
        allowed={'worker_id','profile','configuration_sha256','operation','parent_operation_id'}
        if set(metadata)-allowed or type(metadata.get('worker_id')) is not str or len(metadata['worker_id'])>128:
            raise ValueError('bounded explicit worker identity required')
        for key,value in metadata.items():
            if value is not None and (type(value) is not str or len(value)>256):
                raise ValueError('bounded worker metadata required')
        if 'configuration_sha256' in metadata:_digest(metadata['configuration_sha256'])
        parent=metadata.get('parent_operation_id')
        if parent is not None and parent not in self._tickets:
            raise ValueError('worker operation parent must be an original observed call')
        if kind=='worker_lifecycle':
            if metadata['worker_id'] in self._workers or len(self._workers)>=MAX_WORKERS:
                raise ValueError('unique bounded worker lifetime required')
        elif self._operations>=MAX_OPERATIONS or metadata['worker_id'] not in self._workers:
            raise ValueError('bounded operation on an observed worker required')
        self._reserve(8,32768)
        ticket=_Work(self,kind,metadata)
        if kind=='worker_lifecycle':self._workers[metadata['worker_id']]=ticket
        else:self._operations+=1
        self._tickets[ticket.id]=ticket
        try:self._emit(ticket,'work_capacity_reserved',dict(kind=kind,**metadata))
        except BaseException:
            self._release_capacity(ticket);ticket._finished=True
            raise
        return ticket

    def invalidate(self,reason,subject_id=None):
        if reason not in ('delete','delete_data','reset','reset_processing','pause','consent_revoked','consent','close'):
            raise ValueError('known privacy/lifecycle invalidation required')
        affected=0
        self._links.clear()
        for ticket in self._tickets.values():
            if isinstance(ticket,_Call) and (subject_id is None or ticket._subject==subject_id):
                ticket._invalidated=True;ticket._scrub();affected+=1
        if subject_id is None:self._revoked=True
        if not self._terminal:
            try:self._emit(None,'private_mappings_invalidated',dict(reason=reason,affected=affected))
            except BaseException:self._mark('invalidation_journal_failed')

    def attach_declared_lines(self,operation_id,*,quantities,line_claims,evidence_sha256):
        """Optional explicit external billing evidence; never infer it from usage.

        Real provider extraction is a separate adapter. Tests can supply synthetic
        declarations, which still do not authenticate invoices.
        """
        if self._terminal:raise RuntimeError('late billing cannot mutate an immutable terminal bundle')
        _digest(evidence_sha256)
        ticket=self._tickets.get(operation_id)
        if not isinstance(ticket,_Call) or ticket.attempt_id is None:
            raise ValueError('observed actual invocation required for declared billing')
        self._task.report_lines(ticket.attempt_id,quantities=quantities,line_claims=line_claims,
            execution_instance_sha256=ticket.instance_id)
        # Billing is separate from public sensory mappings and may arrive after
        # redaction; the opaque operation token does not revive those mappings.
        self._emit(None,'external_billing_attached',dict(operation_id=operation_id,evidence_sha256=evidence_sha256))

    def _unmeasured_boundary(self):
        # Observing an SDK invocation does not measure its hardware/cloud bill,
        # physical sensor availability or all declared resource lanes.
        return [dict(component_id=cid,kind=self._components[cid]['kind'],unit=self._components[cid]['unit'],resource_id=lane['resource_id'],lane_id=lane['lane_id'],
            status='unknown',quantity=None,billed_amount=None,
            reason='runtime instrumentation does not measure this frozen resource/billing lane')
            for cid,requirement in self._stream_declaration['requirements'].items()
            for lane in requirement['lanes']]

    def snapshot(self):
        if self._terminal_result is not None:return deepcopy(self._terminal_result)
        reports=[self._task.snapshot(),self._stream.snapshot()]
        provider_counts={}
        for attempt in reports[0]['attempts']:
            key=(attempt['provider'],attempt['revision'],attempt['role'])
            provider_counts[key]=provider_counts.get(key,0)+1
        workers=[]
        for worker in self._workers.values():
            marks=worker._marks
            begin=marks.get('handle_acquired');end=marks.get('reap_confirmed')
            startup_begin=marks.get('spawn_begin')
            startup_end=marks.get('handshake_ready',marks.get('handshake_failed'))
            operations=[t for t in self._tickets.values() if isinstance(t,_Work) and t.kind=='worker_operation'
                        and t.metadata['worker_id']==worker.metadata['worker_id']]
            intervals=[(t._marks.get('operation_started'),t._marks.get('work_finished')) for t in operations]
            busy=idle=None
            if begin is not None and end is not None and all(a is None or b is not None for a,b in intervals):
                covered=[]
                for a,b in intervals:
                    if a is not None and max(begin,a)<min(end,b):covered.append((max(begin,a),min(end,b)))
                cursor=begin;busy=0
                for a,b in sorted(covered):
                    busy+=max(0,b-max(cursor,a));cursor=max(cursor,b)
                idle=end-begin-busy
            workers.append(dict(worker_id=worker.metadata['worker_id'],
                startup_elapsed_ns=None if startup_begin is None or startup_end is None else startup_end-startup_begin,
                residency_elapsed_ns=None if begin is None or end is None else end-begin,
                reap_confirmed=end is not None,observed_exchange_union_ns=busy,
                observed_no_exchange_gap_ns=idle,
                operations=[dict(operation_id=t.id,operation=t.metadata['operation'],
                    parent_operation_id=t.metadata.get('parent_operation_id'),
                    started_ns=t._marks.get('operation_started'),ended_ns=t._marks.get('work_finished')) for t in operations],
                interpretation='host elapsed interval proxy; overlapping intervals must not be summed as CPU or money'))
        unresolved=any(a['cost_quantities'] is None for r in reports for a in r['attempts'])
        result=dict(schema_version=1,**self._identity,session_id=self.id,clock_domain=self.clock_domain,
            started_ns=self._started,ended_ns=self._ended,closed=self._closed,terminal=self._terminal,
            retention='explicit_external_owner_retains_task_and_evidence_hashes',
            sdk_deletion_erases_evaluation_artifacts=False,private_subject_mappings=sum(
                isinstance(t,_Call) and t._subject is not None for t in self._tickets.values()),
            journal=deepcopy(self._events),reports=reports,
            provider_counters=[dict(provider=p,revision=r,role=role,count=n) for (p,r,role),n in sorted(provider_counts.items())],
            worker_measurements=workers,
            dispatched_without_invocation=sum(isinstance(t,_Call) and t._dispatched and t.attempt_id is None for t in self._tickets.values()),
            incomplete_reasons=sorted(self._incomplete),unresolved_attempt_billing=unresolved,
            unresolved_billing=(unresolved or any(lane['kind']=='monetary' for lane in self._unmeasured_boundary())),
            monetary_cost=None,physical_capture_counters=None,
            required_boundary_lanes=self._unmeasured_boundary(),
            source_availability='unknown; SDK lifecycle is not physical capture availability',
            continuous_recording_verified=False,
            measurement_status='observed_host_lifecycle_only',actual_acceptance='incomplete',
            instrumentation_coverage_verified=False,receipt_authenticity_verified=False,
            measurement_truth_verified=False,isolation_verified=False,whole_pipeline_cost_verified=False)
        if len(_encoded(result))+1>self._max_bytes:
            raise ValueError('runtime evaluation bundle exceeds byte bound')
        return result

    async def close(self,timeout=5):
        from .paired_replay import _finish_owned
        if self._close_task is None:
            self._close_task=asyncio.create_task(self._close(timeout))
        try:return await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish_owned(self._close_task)
            raise

    async def _close(self,timeout):
        if self._terminal:return self.snapshot()
        if type(timeout) not in (int,float) or not 0<timeout<=30:
            raise ValueError('bounded session close timeout required')
        self._closed=True
        tasks={t for t in self._running if t is not asyncio.current_task() and not t.done()}
        if tasks:
            done,pending=await asyncio.wait(tasks,timeout=timeout)
            if pending:
                self._mark('invocations_not_drained')
                # SDK adapters remain the owners of callback cancellation/reap.
                # Do not claim drained scopes or block forever on foreign clients.
        for ticket in self._tickets.values():
            if isinstance(ticket,_Call):ticket._scrub()
            if not ticket._finished:self._mark('unfinished_owned_operation')
        self._task.close();self._stream.close();self._ended=time.monotonic_ns()
        if any('reap_confirmed' not in w._marks for w in self._workers.values()):self._mark('worker_reap_unverified')
        self._terminal=True
        result=self.snapshot()
        self._terminal_result=deepcopy(result)
        if self._journal is not None and hasattr(self._journal,'close'):
            await self._journal.close(reports=result['reports'],summary={k:v for k,v in result.items() if k not in ('reports','journal')})
        return result


class _Ticket:
    def __init__(self,session):
        self.session=session;self.id=uuid.uuid4().hex
        self._events_left=8;self._bytes_left=32768;self._finished=False

    def _finish(self,event,fields):
        if self._finished:return
        try:return self.session._emit(self,event,fields)
        finally:
            self._finished=True;self.session._release_capacity(self)


class _Call(_Ticket):
    def __init__(self,session,provider,revision,role,prompt,evidence,subject,controller,request_id):
        super().__init__(session)
        self._provider=provider;self._revision=revision;self._role=role
        self._prompt=prompt;self._evidence=evidence;self._subject=subject
        self._controller=controller;self._request_id=request_id;self._execution=None
        self._predecessor=None;self._relationship='root'
        self._dispatched=False;self._entered=False;self._invalidated=False
        self.attempt_id=None;self.instance_id=hashlib.sha256(uuid.uuid4().bytes+uuid.uuid4().bytes).hexdigest()

    def _scrub(self):
        self._subject=None;self._controller=None;self._request_id=None

    def dispatched(self):
        if self._dispatched or self._finished or self._invalidated:
            raise RuntimeError('observed dispatch cannot repeat or use invalidated capacity')
        pending=next((r for r in self._controller.snapshot()['requests'] if r['id']==self._request_id),None)
        execution=None if pending is None else pending['execution']
        if (pending is None or pending['state']!='dispatched' or execution is None or
                (execution['provider'],execution['model_revision'],execution['route'],execution['prompt_sha256']) !=
                (self._provider,self._revision,self._role,self._prompt)):
            raise ValueError('observer dispatch differs from actual controller execution')
        self._execution=plan_digest(execution)
        self._controller=None;self._request_id=None
        self._dispatched=True
        self.session._emit(self,'controller_dispatched',dict(liability='unknown_until_external_billing',execution_sha256=self._execution,predecessor=self._predecessor,relationship=self._relationship))

    async def invoke(self,callback):
        if not self._dispatched or self._entered or self._finished or self._invalidated:
            raise RuntimeError('live unique observed dispatch required at backend entry')
        # No await precedes native admission and the immediate callback invocation.
        # Cancellation before this coroutine entry never fabricates a native row.
        self.session._emit(self,'backend_entry_admission',dict(execution_instance_sha256=self.instance_id))
        self._entered=True;task=asyncio.current_task();self.session._running.add(task)
        async def call(token):
            self.attempt_id=token
            return await callback()
        outcome='error';reported_usage=None
        try:
            value=await self.session._task.invoke(call,provider=self._provider,revision=self._revision,
                role=self._role,phase=self.session._phase,evidence_sha256=self._evidence,
                prompt_sha256=self._prompt,execution_sha256=self._execution,predecessor=self._predecessor,relationship=self._relationship)
            usage=value.get('usage') if type(value) is dict else None
            if (type(usage) is dict and set(usage)=={'input_tokens','output_tokens','cached_input_tokens'} and
                    all(type(v) is int and 0<=v<=10**12 for v in usage.values()) and
                    usage['cached_input_tokens']<=usage['input_tokens']):
                reported_usage=dict(usage)
            outcome='success';return value
        except asyncio.CancelledError:
            outcome='cancelled';raise
        finally:
            self.session._running.discard(task)
            try:self.session._emit(self,'backend_invocation_finished',dict(status=outcome,attempt_id=self.attempt_id,reported_usage=reported_usage,usage_verified=False))
            except BaseException:self.session._mark('terminal_observation_failed')

    def finish(self,outcome):
        if outcome not in ('completed','published','discarded','error','cancelled','timeout','failed','success'):
            raise ValueError('bounded call outcome required')
        try:self._finish('call_publication_finished',dict(outcome=outcome,invoked=self._entered))
        finally:self._scrub()

    def release(self,reason):
        if self._finished:return
        # Reasons are structural SDK states, not third-party exception messages.
        if reason not in ('unadmitted','completed','published','discarded','error','cancelled','timeout','failed','success'):
            raise ValueError('known structural release reason required')
        try:self._finish('call_capacity_released',dict(dispatched=self._dispatched,invoked=self._entered,reason=reason))
        finally:self._scrub()


class _Work(_Ticket):
    def __init__(self,session,kind,metadata):
        super().__init__(session);self.kind=kind;self.metadata=metadata;self._marks={}

    def event(self,name,fields=None):
        if self._finished or name not in ('spawn_begin','handle_acquired','handshake_ready','handshake_failed',
                'reap_confirmed','operation_started','operation_finished','invalidated'):
            raise ValueError('bounded original worker milestone required')
        if name in self._marks:raise ValueError('worker milestone repeated')
        # SDK must provide only structural fields, never frame/config/response bodies.
        fields={} if fields is None else fields
        if type(fields) is not dict or set(fields)-{'returncode','status','operation','parent_operation_id'}:
            raise ValueError('worker milestone metadata is not allowlisted')
        self._marks[name]=self.session._emit(self,name,fields)

    def finish(self,outcome):
        if outcome not in ('closed','completed','error','cancelled','timeout'):
            raise ValueError('known structural worker outcome required')
        if self._finished:return
        end=self._finish('work_finished',dict(kind=self.kind,outcome=outcome))
        self._marks['work_finished']=end
