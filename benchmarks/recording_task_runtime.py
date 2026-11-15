"""Timed original Context snapshots and actual persistent text answer lanes."""
import asyncio
from collections import Counter
from dataclasses import dataclass,asdict
import hashlib
import heapq
import inspect
import json
import time

from gamst.call_controller import CallController
from gamst.capabilities import bounded_json
from gamst.evaluation_snapshot import EvaluationSnapshotPublisher,evaluation_snapshot_implementation_sha256
from gamst.native_entry import NativeEntryBinding
from gamst.schema import timestamp
from gamst.streams import _finish
from gamst.text_context_call import TextContextCalls
from .recorded_source import digest,encoded,validate_manifest
from .replay_clock import ReplayClock,validate_tasks
from .recording_evaluation import RecordingEvaluationSession


_FAILURES=frozenset(('not_executed','snapshot_failed','oversize','busy','stale','admission_refused',
    'timeout','cancelled','backend_failed','source_failed','clock_discontinuity','transport_failed','writer_failed'))


class _ClockDiscontinuity(RuntimeError):pass
class _WriterFailed(RuntimeError):pass
class _TransportFailed(RuntimeError):pass


@dataclass(frozen=True)
class _Lane:
    name:str
    adapter:TextContextCalls
    completion:object
    controller:CallController
    binding_index:int
    maximum_charge:object
    ttl_ms:int


class RecordingTaskRuntime:
    """Own task operations only; caller owns SDK, adapters, clients and writers.

    A scheduler creates one source Context per original snapshot time. Waiting
    answers use earliest original deadline then frozen task index. No task asks
    a visual model, retries its answer, or refreshes queued evidence.
    """
    def __init__(self,human,owner,clock,manifest,tasks,subject_id,publisher,lanes,
                 text_lane_ids,queue_capacity,artifacts,answer_ready,watermarks,
                 receipt_router=None):
        from .recording_runtime_artifacts import RecordingRuntimeArtifacts
        from .recording_transports import FixtureCompletion,LocalChatCompletions
        if type(owner) is not RecordingEvaluationSession or type(clock) is not ReplayClock:
            raise TypeError('actual recording owner and replay clock required')
        if type(publisher) is not EvaluationSnapshotPublisher or publisher.subject_id!=subject_id:
            raise ValueError('same-subject SDK snapshot publisher required')
        publisher.context_view._exclusion(human,subject_id)
        if type(artifacts) is not RecordingRuntimeArtifacts:raise TypeError('owned runtime artifacts required')
        if not callable(answer_ready) or not callable(watermarks):raise TypeError('trusted runtime callbacks required')
        manifest=validate_manifest(manifest);tasks=validate_tasks(tasks,manifest['duration_ns'])
        if digest(manifest)!=digest(owner.declarations['source_manifest']) or tasks!=owner.declarations['tasks']:
            raise ValueError('exact original source/tasks required')
        if clock.mapping['source_manifest_sha256']!=digest(manifest):raise ValueError('clock belongs to another source')
        execution=artifacts.execution
        if any(execution[name]!=owner.context[name] for name in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal')):
            raise ValueError('task artifacts belong to another original native run')
        u=owner.declarations['runtime_policy'];arm=next(a for a in u['arms'] if a['arm']==execution['arm'])
        policy=next(p for p in arm['recording_policies'] if p['recording_id']==manifest['recording_id'])
        if type(text_lane_ids) not in (list,tuple) or list(text_lane_ids)!=policy['text_lane_ids']:
            raise ValueError('exact frozen answer lane order required')
        if type(queue_capacity) is not int or queue_capacity!=arm['task_queue']['capacity']:
            raise ValueError('exact frozen answer queue capacity required')
        if type(lanes) is not dict or not set(text_lane_ids)<=set(lanes):raise ValueError('persistent answer lanes required')
        selected=[]
        for name in text_lane_ids:
            value=lanes[name];original=next(l for l in arm['lanes'] if l['lane_id']==name)
            if set(value)!=set(('adapter','completion','controller','policy','binding_index')) or value['policy']!=original:
                raise ValueError('explicit frozen original answer lane required')
            adapter=value['adapter'];controller=value['controller'];completion=value['completion'];binding=value['binding_index']
            if type(adapter) is not TextContextCalls or type(controller) is not CallController or adapter.human is not human or adapter.controller is not controller or adapter._completion is not completion:
                raise ValueError('actual persistent SDK/controller/client identity required')
            if type(completion) not in (FixtureCompletion,LocalChatCompletions):
                raise TypeError('allowlisted actual-entry-retaining completion required')
            adapter_policy=next(a for a in u['adapters'] if a['adapter_id']==original['adapter_id'])
            controller_policy=next(c for c in arm['controllers'] if c['controller_id']==original['controller_id'])
            if completion.adapter!=adapter_policy or human._call_controllers.get(original['controller_id']) is not controller or asdict(controller._policy)!=controller_policy['policy'] or controller._max_requests!=controller_policy['max_requests']:
                raise ValueError('actual answer adapter/controller differs from original policy')
            scope=owner.stream_scope(binding_index=binding);owner.validate_backend(scope,completion)
            frozen=owner._binding(scope)
            if original['role']!='text_answer' or frozen['binding_id']!=original['binding_id']:
                raise ValueError('original text binding required')
            selected.append(_Lane(name,adapter,completion,controller,binding,original['maximum_charge'],original['ttl_ms']))
        if len({id(l.adapter) for l in selected})!=len(selected) or len({id(l.completion) for l in selected})!=len(selected):
            raise ValueError('each persistent lane requires its own SDK adapter and client')
        self.human=human;self.owner=owner;self.clock=clock;self.manifest=manifest;self.tasks=tasks
        self.subject_id=subject_id;self.publisher=publisher;self.artifacts=artifacts
        self._lanes=tuple(selected);self._idle={lane.name:lane for lane in selected}
        self._answer_ready=answer_ready;self._watermarks=watermarks;self._router=receipt_router
        self._queue_capacity=queue_capacity;self._queue=[];self._queued=set();self._active={}
        self._states=[dict(done=False,opened=False,snapshot=None,request_id=None,decision=None,entry=None,evidence=None,result=None,finalization=None) for _ in tasks]
        self._results={};self._result_values={};self._failures=Counter();self._changed=asyncio.Event()
        self._run_task=None;self._close_task=None;self._stopping=None;self._closed=False;self._fatal=None
        self._builder=evaluation_snapshot_implementation_sha256();self._manifest_sha=digest(manifest)

    @property
    def results(self):return tuple(json.loads(encoded(self._results[n])) for n in sorted(self._results))
    @property
    def result_values(self):return tuple(json.loads(encoded(self._result_values[n])) for n in sorted(self._result_values))
    @property
    def failure_counts(self):return dict(self._failures)

    def _now(self):
        try:return self.clock.check()
        except RuntimeError as error:raise _ClockDiscontinuity(str(error)) from error

    def _deadline(self,n):return self.clock.monotonic_ns+self.tasks[n]['deadline_ns']

    async def _wait(self,deadline):
        while self._stopping is None:
            self._changed.clear();remaining=deadline-self._now()
            if remaining<=0:return
            try:await asyncio.wait_for(self._changed.wait(),remaining/1e9)
            except asyncio.TimeoutError:return
            # Completion may make a scheduled deadline irrelevant.
            return

    async def run(self):
        if self._run_task is not None or self._closed:raise RuntimeError('original task scheduler is single-use')
        self._run_task=asyncio.create_task(self._run())
        try:return await asyncio.shield(self._run_task)
        except asyncio.CancelledError:
            self.stop('cancelled');self._run_task.cancel()
            await _finish(asyncio.gather(self._run_task,return_exceptions=True));raise

    async def _run(self):
        events=[]
        for n,task in enumerate(self.tasks):
            for kind,field in ((0,'opens_ns'),(1,'snapshot_ns'),(2,'deadline_ns')):
                heapq.heappush(events,(self.clock.monotonic_ns+task[field],kind,n))
        try:
            while events and self._stopping is None:
                due,kind,n=events[0];state=self._states[n]
                if state['done']:
                    heapq.heappop(events);continue
                if self._now()<due:
                    await self._wait(due);continue
                heapq.heappop(events);self.human._check()
                if kind==0:state['opened']=True
                elif kind==1:
                    await self._snapshot(n)
                elif n not in self._active:
                    self._queued.discard(n);await self._finalize(n,'timeout')
            while self._active and self._stopping is None:
                self._changed.clear();await self._changed.wait()
            if self._stopping is not None:await self._cancel_remaining(self._stopping)
            if self._fatal is not None:raise self._fatal
            return self.results
        except _ClockDiscontinuity:
            self.stop('clock_discontinuity');await self._cancel_remaining('clock_discontinuity');return self.results
        except asyncio.CancelledError:
            self.stop(self._stopping or 'cancelled')
            await _finish(asyncio.create_task(self._cancel_remaining(self._stopping)));raise
        except _WriterFailed:
            self.stop('writer_failed');await self._cancel_remaining('writer_failed');raise
        except _TransportFailed:
            self.stop('transport_failed');await self._cancel_remaining('transport_failed');raise
        except BaseException:
            self.stop('source_failed');await self._cancel_remaining('source_failed');raise

    async def _snapshot(self,n):
        state=self._states[n];task=self.tasks[n]
        try:
            begin=self._now()
            if begin>=self._deadline(n):await self._finalize(n,'timeout');return
            # Capture watermark BEFORE Context assembly, never infer processing
            # from queued/accepted counts or refresh it at answer dispatch.
            marks=bounded_json(self._watermarks(),max_bytes=8192)
            context=await self.human.get_context(self.subject_id,task['window'],context_view=self.publisher.context_view)
            end=self._now()
            envelope=dict(kind='frozen_context_bundle_not_new_sensor_measurement',task_id=task['task_id'],
                input_sha256=task['input_sha256'],task_sha256=digest(task),source_manifest_sha256=self._manifest_sha,
                mapping_sha256=self.clock.sha256,context_sha256=digest(context),snapshot_timestamp_ms=context['timestamp_ms'],
                scheduled_offset_ns=task['snapshot_ns'],observed_begin_ns=begin,observed_end_ns=end,watermarks=marks,context=context)
            # Publisher validates complete original Context/envelope bounds; no
            # trim, flattened request, timestamp replacement or retry is allowed.
            record=await self.publisher.publish(envelope)
            try:
                publication=self.artifacts.record_publication(record,binding_index=None,builder_sha256=self._builder,
                    operation_reference=None,capture_refs=[],role='task_snapshot',origin_result_sha256=None,mapping_sha256=self.clock.sha256)
                await self.artifacts.publication_ready(publication)
            except Exception as error:raise _WriterFailed('task publication write failed') from error
            refs={}
            for kind,value in (('context',context),('envelope',envelope),('record',record)):
                refs[kind]=await self._write_task(n,kind,value)
            state['snapshot']=dict(context=context,envelope=envelope,record=record,refs=refs)
            if self._now()>=self._deadline(n):await self._finalize(n,'timeout');return
            await self._enqueue(n)
        except (_ClockDiscontinuity,_WriterFailed,_TransportFailed):raise
        except asyncio.CancelledError:raise
        except ValueError as error:
            await self._finalize(n,'oversize' if any(word in str(error).lower() for word in ('exceeds','bound','size','depth','capacity')) else 'snapshot_failed')
        except Exception:
            await self._finalize(n,'snapshot_failed')

    async def _enqueue(self,n):
        if self._stopping is not None:return
        if self._idle and not self._queued:
            lane=next(l for l in self._lanes if l.name in self._idle);self._start(n,lane)
        elif len(self._queued)>=self._queue_capacity:
            await self._finalize(n,'busy')
        else:
            self._queued.add(n);heapq.heappush(self._queue,(self._deadline(n),n))

    def _start(self,n,lane):
        self._idle.pop(lane.name,None);self._queued.discard(n)
        job=asyncio.create_task(self._answer(n,lane));self._active[n]=job
        job.add_done_callback(lambda done:self._answer_finished(n,lane,done))

    def _answer_finished(self,n,lane,job):
        if not job.cancelled():
            error=job.exception()
            if error is not None:
                self._record_fatal(error,'backend_failed');self.stop('backend_failed')
        # Catch failures in cleanup itself, with no unobserved task exception or
        # scheduler waiting forever for a lane whose coroutine already exited.
        if self._active.get(n) is job:
            self._active.pop(n,None);self._idle[lane.name]=lane
        self._changed.set()

    def _retain_entry(self,n,lane,entry):
        if type(entry) is not NativeEntryBinding:raise ValueError('actual original native entry required')
        ref=self.owner.operation_reference(entry);value=ref.snapshot()
        if value['scope_index']!=n+1 or value['binding_index']!=lane.binding_index:
            raise ValueError('actual answer entry belongs to another original task/lane')
        self._states[n]['entry']=entry
        return ref

    async def _dispatch(self):
        while self._idle and self._queue and self._stopping is None:
            _,n=heapq.heappop(self._queue)
            if n not in self._queued or self._states[n]['done']:continue
            lane=next(l for l in self._lanes if l.name in self._idle);self._start(n,lane)

    async def _answer(self,n,lane):
        state=self._states[n];task=self.tasks[n];source=None;decision_sha=None
        reason='backend_failed';answer=None;operation=None
        try:
            self.human._check();now=self._now()
            if now>=self._deadline(n):reason='timeout';return
            # Canonical detached order makes the actual SDK user-message bytes
            # reproducible from the immutable retained original record. Values,
            # record digest and capture freshness are unchanged.
            record=json.loads(encoded(state['snapshot']['record']));wall=timestamp()
            if (wall-record['timestamp_ms']>min(1000,lane.controller._policy.max_age_ms) or wall<record['timestamp_ms']):reason='stale';return
            source=lane.controller.open_source(record,now_ms=wall)
            decision=lane.controller.submit_next(source,record,now_ms=wall,query=True,needs_visual=False)
            if decision['status']!='queued':reason='admission_refused';return
            rid=decision['request_id'];state['request_id']=rid;state['decision']=decision['decision']
            decision_sha=decision['decision']['decision_sha256']
            if self._router is not None:self._router.register_request(self.owner.context['invocation_ordinal'],lane.binding_index,rid,decision_sha)
            scope=self.owner.task_scope(task['task_id'],binding_index=lane.binding_index)
            call=asyncio.create_task(lane.adapter.execute(rid,record,task['input']['question'],maximum_charge=lane.maximum_charge,
                ttl_ms=lane.ttl_ms,evaluation_scope=scope))
            try:
                remaining=max(0,self._deadline(n)-self._now())/1e9
                result=await asyncio.wait_for(call,remaining)
            except BaseException:
                if not call.done():call.cancel()
                await _finish(asyncio.gather(call,return_exceptions=True));raise
            if result.get('accepted') is True:state['sdk_result']=bounded_json(result,max_bytes=65536)
            if not result.get('accepted'):reason='admission_refused';return
            if self._now()>self._deadline(n):reason='timeout';return
            entry=lane.completion.entry_for_decision(decision_sha)
            if type(entry) is not NativeEntryBinding:raise ValueError('accepted answer lacks actual original native entry')
            operation=self._retain_entry(n,lane,entry).instance_sha256;answer=result['text'];reason='completed'
        except _ClockDiscontinuity:
            reason='clock_discontinuity';self.stop(reason)
        except asyncio.TimeoutError:reason='timeout'
        except asyncio.CancelledError:reason=self._stopping or 'cancelled'
        except (PermissionError,RuntimeError):reason='cancelled' if self._stopping is not None else 'backend_failed'
        except Exception:reason='backend_failed'
        finally:
            if decision_sha is not None:
                entry=lane.completion.entry_for_decision(decision_sha)
                if type(entry) is NativeEntryBinding:self._retain_entry(n,lane,entry)
            if source is not None:
                try:lane.controller.close_source(source,now_ms=timestamp())
                except (KeyError,ValueError,RuntimeError):pass
            try:
                await self._finalize(n,reason,answer=answer,operation=operation)
            except BaseException as error:
                failure='writer_failed' if isinstance(error,_WriterFailed) else 'transport_failed'
                self._record_fatal(error,failure);self.stop(failure)
            finally:
                if decision_sha is not None and lane.completion.entry_for_decision(decision_sha) is not None:
                    lane.completion.release_entry(decision_sha)
                self._active.pop(n,None);self._idle[lane.name]=lane;self._changed.set()
                await self._dispatch()

    async def _write_task(self,n,kind,value):
        try:return await self.artifacts.write_task(n,kind,value)
        except asyncio.CancelledError:raise
        except Exception as error:raise _WriterFailed('task artifact write failed') from error

    async def _finalize(self,n,reason,*,answer=None,operation=None):
        state=self._states[n]
        if state['finalization'] is None:
            state['finalization']=asyncio.create_task(self._seal_result(n,reason,answer,operation))
        # This task is the sole writer for immutable task paths. Repeated caller
        # cancellation cannot interrupt or duplicate the owned finalization.
        return await _finish(state['finalization'])

    async def _seal_result(self,n,reason,answer,operation):
        state=self._states[n];task=self.tasks[n]
        if reason=='completed' and time.monotonic_ns()>=self._deadline(n):
            reason='timeout';answer=None;operation=None
        evidence=None;snapshot=state['snapshot'];entry=state['entry']
        if snapshot is not None:
            value=None if entry is None else entry.snapshot()
            evidence=dict(schema_version=1,kind='recording_task_evidence',execution_sha256=self.artifacts.execution_sha256,
                task_id=task['task_id'],input_sha256=task['input_sha256'],scope_index=n+1,mapping_sha256=self.clock.sha256,
                context=snapshot['refs']['context'],context_object_sha256=digest(snapshot['context']),
                snapshot_envelope=snapshot['refs']['envelope'],snapshot_record=snapshot['refs']['record'],snapshot_record_sha256=digest(snapshot['record']),
                snapshot_observed_begin_ns=snapshot['envelope']['observed_begin_ns'],snapshot_observed_end_ns=snapshot['envelope']['observed_end_ns'],
                controller_request_id=state['request_id'],request_evidence_sha256=None if state['decision'] is None else state['decision']['evidence']['record_sha256'],
                prompt_sha256=None if value is None else value['prompt_sha256'],operation_instance=None if value is None else value['entry']['instance_sha256'])
            evidence=await self._write_task(n,'evidence',evidence)
        sdk_result=state.get('sdk_result')
        if sdk_result is not None:
            if entry is None:raise _WriterFailed('actual accepted SDK result missing original native entry')
            response=dict(schema_version=1,kind='recording_semantic_response',execution_sha256=self.artifacts.execution_sha256,
                task_id=task['task_id'],input_sha256=task['input_sha256'],scope_index=n+1,
                operation_instance=entry.snapshot()['entry']['instance_sha256'],
                native_raw_result_sha256=digest(dict(text=sdk_result['text'],usage=sdk_result['usage'])),sdk_result=sdk_result)
            try:
                if len(encoded(response))+1>65536:raise ValueError('actual SDK response sidecar exceeds64KiB')
                await self.artifacts.write_request(n,response)
            except Exception as error:raise _WriterFailed('actual SDK response evidence write failed') from error
        if reason=='completed' and time.monotonic_ns()>=self._deadline(n):reason='timeout';answer=None;operation=None
        status='answer' if reason=='completed' else 'failed'
        if status=='failed':answer=None;operation=None;self._failures[reason]+=1
        result=dict(schema_version=1,kind='recording_task_result',execution_sha256=self.artifacts.execution_sha256,
            task_id=task['task_id'],input_sha256=task['input_sha256'],scope_index=n+1,status=status,reason=reason,evidence=evidence,
            answer_text=answer,answer_sha256=None if answer is None else hashlib.sha256(answer.encode()).hexdigest(),
            operation_instance=operation,child_finished_ns=time.monotonic_ns(),cleanup_finished=True)
        ref=await self._write_task(n,'result',result)
        state['done']=True;state['result']=ref;self._results[n]=ref;self._result_values[n]=result;self._changed.set()
        try:
            response=self._answer_ready(task['task_id'],ref,ref['file_sha256'])
            if not inspect.isawaitable(response):raise TypeError('async sealed result callback required')
            await response
        except BaseException as error:raise _TransportFailed('sealed task result delivery failed') from error

    def _record_fatal(self,error,reason):
        if self._fatal is None:
            self._fatal=error;self._failures[reason]+=1

    def stop(self,reason='cancelled'):
        if reason not in _FAILURES:raise ValueError('finite task stop reason required')
        if self._stopping is None:self._stopping=reason
        self._changed.set()
        for job in list(self._active.values()):
            if job is not asyncio.current_task():job.cancel()

    async def _cancel_remaining(self,reason):
        jobs=list(self._active.values())
        for job in jobs:job.cancel()
        if jobs:await _finish(asyncio.gather(*jobs,return_exceptions=True))
        self._queued.clear();self._queue.clear()
        for n,state in enumerate(self._states):
            if not state['done'] and state['finalization'] is None:
                try:await self._finalize(n,reason)
                except Exception as error:self._record_fatal(error,'writer_failed' if isinstance(error,_WriterFailed) else 'transport_failed')

    async def _close(self):
        self.stop('cancelled')
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel();await _finish(asyncio.gather(self._run_task,return_exceptions=True))
        await self._cancel_remaining(self._stopping);self._closed=True
    async def close(self):
        if self._close_task is None:
            self.stop('cancelled');self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise
