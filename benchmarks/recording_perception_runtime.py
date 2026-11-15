"""Owned original-tick VisualContextCalls and actual conditional fill execution.

JSON selects only the finite policy already validated before construction. The
source owner, clients, SDK, writers and financial ledger have separate lifetimes.
"""
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path
import time

from gamst.capabilities import CapabilitySpec,OutputField,SensorMetadata
from gamst.fill_evaluation import FillEvaluationContext
from gamst.schema import timestamp
from gamst.smolvlm_call import smolvlm_frame_sha256
from gamst.streams import _finish
from .recorded_source import digest
from .recorded_ticks import RecordedTickSchedule
from .recording_capture_router import capture_reference,packet_frame


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


from .recording_runtime_contract import validate_visual_prompt as visual_prompt


class RecordingPerceptionRuntime:
    def __init__(self,*,human,owner,clock,manifest,subject_id,source_bindings,router,
                 policy,lanes,artifacts,prompt=None,fill_driver=None,fill_binding_index=None,
                 receipt_router=None):
        self.human=human;self.owner=owner;self.clock=clock;self.manifest=manifest
        self.subject_id=subject_id;self.router=router;self.policy=policy;self.lanes=lanes
        self.artifacts=artifacts;self.receipt_router=receipt_router;self.fill_driver=fill_driver
        self.fill_binding_index=fill_binding_index;self.prompt=prompt
        self.source_bindings={b['source_id']:b for b in source_bindings}
        self.sources={s['source_id']:(i,s) for i,s in enumerate(manifest['sources'])}
        self.jobs=set();self.closed=False;self.failures=[];self._sources={};self._schedulers=[]
        self._builder=implementation_sha256()
        self.descriptor=CapabilitySpec('evaluation.original_visual.v1','evaluation.original_visual.v1',
            'video','gamst.recording_visual',self._builder,(OutputField('capture','object'),),'frame',
            freshness_ms=60000,interpretation='estimate')
        self.semantic=CapabilitySpec('evaluation.visual_semantics.v1','evaluation.visual_semantics.v1',
            'video','gamst.recording_visual',self._builder,(OutputField('interpretation','object'),),'frame',
            freshness_ms=60000,interpretation='estimate')
        human.register_capability(self.descriptor);human.register_capability(self.semantic)
        periodic=policy['periodic']
        if periodic is not None:
            self.question=visual_prompt(prompt,periodic['prompt_policy_sha256'])
            lane=lanes[periodic['lane_id']]
            self.smol=lane['declaration']['implementation']=='smolvlm_process_v1'
            if self.smol:
                from gamst.providers.smolvlm_native import PROMPT
                if periodic['scheduler']!='smolvlm_sentinel_v1' or self.question!=PROMPT:
                    raise ValueError('fixed original Smol prompt/sentinel policy required')
            elif periodic['scheduler']!='recorded_visual_v1':raise ValueError('explicit Visual scheduler required')
        else:self.question=None
        if policy['fill_chain'] is not None and (fill_driver is None or fill_binding_index is None):
            raise ValueError('actual owned FillConflictRuntime and direct binding required')

    async def _until(self,offset):
        while True:
            now=self.clock.check();remaining=self.clock.monotonic_ns+offset-now
            if remaining<=0:return now
            await asyncio.sleep(min(remaining/1e9,.05))

    def _publish(self,record,packet,role,operation=None,result=None):
        return self.artifacts.record_publication(record,binding_index=None if operation is None else operation.binding_index,
            builder_sha256=self._builder if operation is None else None,operation_reference=operation,
            capture_refs=[capture_reference(packet)],role=role,origin_result_sha256=result,mapping_sha256=self.clock.sha256)

    def _decision(self,lane,record,packet):
        controller=lane['controller'];key=(lane['policy']['lane_id'],record['observation']['capability']['id'])
        if key not in self._sources:self._sources[key]=controller.open_source(record,now_ms=timestamp())
        visual=dict(source_id=packet.source_id,sensor_id=packet.sensor_id,
            stream_epoch=record['observation']['sensor']['stream_epoch'],capture_ms=packet.sdk_timestamp_ms,
            sha256=smolvlm_frame_sha256(packet_frame(packet,self.sources[packet.source_id][1],
                self.source_bindings[packet.source_id],subject_id=self.subject_id).data))
        return controller.submit_next(self._sources[key],record,now_ms=timestamp(),query=True,
            needs_visual=True,visual_reference=visual)

    def _register(self,lane,rid,packet):
        state=next(r for r in lane['controller'].snapshot()['requests'] if r['id']==rid)
        decision=state['decision']['decision_sha256'];client=lane['completion']
        if lane['declaration']['implementation']!='smolvlm_process_v1':client.register_capture(decision,packet)
        if self.receipt_router is not None:
            self.receipt_router.register_request(self.owner.context['invocation_ordinal'],lane['binding_index'],rid,decision)
        return decision

    async def _periodic(self,packet,lane):
        record=await self.human.push_observation(self.descriptor.id,dict(capture=dict(
            source_manifest_sha256=packet.source_manifest_sha256,record_sha256=packet.record_sha256,
            record_ordinal=packet.record_ordinal,payload_sha256=packet.payload_sha256,
            original_capture_ns=packet.original_capture_ns,mapping_sha256=packet.mapping_sha256)),
            sensor=packet_frame(packet,self.sources[packet.source_id][1],self.source_bindings[packet.source_id],
                subject_id=self.subject_id).sensor,subject_id=self.subject_id,timestamp_ms=packet.sdk_timestamp_ms)
        self._publish(record,packet,'raw_capture_descriptor')
        decision=self._decision(lane,record,packet)
        if decision['status']!='queued':return None,None,'financial_refused'
        rid=decision['request_id'];sha=self._register(lane,rid,packet);operation=None
        try:
            frame=packet_frame(packet,self.sources[packet.source_id][1],self.source_bindings[packet.source_id],subject_id=self.subject_id)
            args=(rid,record,frame) if self.smol else (rid,record,frame,self.question)
            response=await lane['adapter'].execute(*args,
                maximum_charge=lane['policy']['maximum_charge'],ttl_ms=lane['policy']['ttl_ms'],
                evaluation_scope=self.owner.stream_scope(lane['binding_index']))
            entry=lane['completion'].entry_for_decision(sha)
            operation=None if entry is None else self.owner.operation_reference(entry)
            if response.get('accepted') is True:
                if operation is None:raise RuntimeError('accepted semantic response missing actual entry')
                if self.smol:
                    observation=response['observation'];raw=response['observation']['observation']['value']
                    self._publish(observation,packet,'semantic_description',operation,digest(raw))
                    return rid,operation.instance_sha256,'selected'
                interpreted=await self.human.push_observation(self.semantic.id,dict(interpretation=dict(
                    text=response['text'],trust='untrusted_model_interpretation',request_sha256=operation.snapshot()['request_sha256'],
                    prompt_sha256=operation.snapshot()['prompt_sha256'],response_sha256=digest(dict(text=response['text'],usage=response['usage'])),
                    original_record_sha256=packet.record_sha256)),sensor=frame.sensor,
                    subject_id=self.subject_id,timestamp_ms=packet.sdk_timestamp_ms)
                self._publish(interpreted,packet,'semantic_description',operation,digest(dict(text=response['text'],usage=response['usage'])))
            return rid,None if operation is None else operation.instance_sha256,'selected'
        finally:
            # These objects retain the original actual entry only until its
            # publication/export reservation; never fabricate a replacement.

            if lane['completion'].entry_for_decision(sha) is not None:lane['completion'].release_entry(sha)
            if not self.smol:lane['completion'].release_capture(sha)

    async def _attempt(self,kind,opportunity,packet,lane,row):
        context=None
        try:
            if kind=='periodic':
                rid,instance,outcome=await self._periodic(packet,lane)
                row.update(request_id=rid,operation_instance=instance,outcome=outcome)
            else:
                context=_FillContext(self,packet,opportunity.index)
                frame=packet_frame(packet,self.sources[packet.source_id][1],self.source_bindings[packet.source_id],subject_id=self.subject_id)
                await self.fill_driver.inspect(frame,evaluation_context=context)
                row['operation_instance']=context.last_operation.instance_sha256 if context.last_operation else None
        except asyncio.CancelledError:
            row['outcome']='closed';raise
        except Exception as error:
            if len(self.failures)<64:self.failures.append(type(error).__name__)
            row['outcome']='observer_refused' if isinstance(error,(ValueError,TypeError)) else 'financial_refused'
        finally:
            if context is not None and context.case is not None and all(k in context.publications for k in ('fill_measurement','fill_descriptor')):
                export=asyncio.create_task(context.export())
                try:row['case']=await asyncio.shield(export)
                except asyncio.CancelledError:row['case']=await _finish(export)
            row['observed_end_ns']=time.monotonic_ns();self.artifacts.record_tick(row)

    async def _schedule(self,kind,definition,source_id):
        i,_=self.sources[source_id]
        schedule=RecordedTickSchedule(duration_ns=self.manifest['duration_ns'],
            period_ns=definition['period_ns'] if kind=='periodic' else definition['selection_period_ns'],
            phase_ns=definition['phase_ns'],time_origin_policy=definition.get('time_origin_policy','recording_start'),
            origin_capture_ns=self.manifest['origin_capture_ns'],max_age_ms=definition.get('max_age_ms',1000),
            deadline_tolerance_ns=definition.get('deadline_tolerance_ns',definition.get('max_lateness_ns')))
        last=-1;active=None
        lane=self.lanes[definition['lane_id']] if kind=='periodic' else None
        while not self.closed and schedule.next_due_offset() is not None:
            await self._until(schedule.next_due_offset());now=self.clock.check()
            for opportunity in schedule.advance(now-self.clock.monotonic_ns):
                row=dict(tick_index=opportunity.index,count=opportunity.count,
                    first_due_offset_ns=opportunity.first_due_offset_ns,period_ns=opportunity.period_ns,
                    policy=kind,source_index=i,observed_begin_ns=now,observed_end_ns=now,
                    outcome='late',capture_ref=None,request_id=None,operation_instance=None,case=None)
                if opportunity.count>1:self.artifacts.record_tick(row);continue
                selected=schedule.select(opportunity,self.router.retained(i),anchor_ns=self.clock.monotonic_ns,
                    observed_ns=now,source_index=i,last_selected_ordinal=last,busy=active is not None and not active.done(),closed=self.closed)
                row['outcome']=selected.outcome
                if selected.packet is not None:row['capture_ref']=capture_reference(selected.packet)
                if selected.outcome!='selected':self.artifacts.record_tick(row);continue
                packet=selected.packet;last=packet.record_ordinal
                active=asyncio.create_task(self._attempt(kind,opportunity,packet,lane,row));self.jobs.add(active)
                active.add_done_callback(self.jobs.discard)
        if active is not None:await active

    async def run(self,primary_source_id):
        if self.policy['periodic'] is not None:self._schedulers.append(asyncio.create_task(self._schedule('periodic',self.policy['periodic'],primary_source_id)))
        if self.policy['fill_chain'] is not None:
            definition=self.policy['fill_chain'];self._schedulers.append(asyncio.create_task(self._schedule('fill_chain',definition,definition['source_id'])))
        await asyncio.gather(*self._schedulers)

    async def close(self):
        self.closed=True
        for task in [*self._schedulers,*self.jobs]:
            if not task.done():task.cancel()
        await _finish(asyncio.gather(*self._schedulers,*self.jobs,return_exceptions=True))


class _FillContext(FillEvaluationContext):
    def __init__(self,runtime,packet,case_ordinal):
        self.r=runtime;self.packet=packet;self.case_ordinal=case_ordinal;self.ticket=None;self.last_entry=None;self.last_operation=None
        self.publications={};self.stages=[];self.pending=None;self.case=None;self.outcome='cancelled'
    def validate(self,driver,frame):
        if driver is not self.r.fill_driver:raise ValueError('foreign fill driver')
        ref=capture_reference(self.packet);b=self.r.fill_binding_index
        self.ticket=self.r.owner.reserve_operation(b,self.r.owner.stream_scope(b),'process',[ref],
            digest(dict(binding=b,operation_kind='process',capture_refs=[ref])),ref['record_sha256'])
    def measure(self,callback):return self.ticket.invoke_sync(callback)
    def published(self,role,record):
        operation=self.ticket.entry_reference() if role!='fill_comparison' or self.last_operation is None else self.last_operation
        if operation is None:raise RuntimeError('fill publication missing actual operation')
        self.publications[role]=self.r._publish(record,self.packet,role,operation)
        if role=='fill_measurement':self.ticket.published(True)
    def stage_arguments(self,stage,request_id,adapter):
        policy=self.r.policy['fill_chain'];lid=policy['strong_lane_id'] if stage=='strong' else policy['small_lane_id']
        lane=self.r.lanes[lid]
        if adapter._completion is not lane['completion']:raise ValueError('fill stage differs from frozen client')
        decision=self.r._register(lane,request_id,self.packet)
        self.pending=(stage,request_id,lane,decision,self.last_entry)
        return dict(evaluation_scope=self.r.owner.stream_scope(lane['binding_index']),
            evaluation_relationship={'initial':'root','reinspection':'retry','strong':'escalation'}[stage],
            evaluation_predecessor=None if stage=='initial' else self.last_entry)
    def stage_finished(self,stage,request_id,response,error):
        actual,rid,lane,decision,previous=self.pending
        if (stage,request_id)!=(actual,rid):raise RuntimeError('fill stage ownership changed')
        entry=lane['completion'].entry_for_decision(decision)
        operation=None if entry is None else self.r.owner.operation_reference(entry)
        prompt=None if operation is None else operation.snapshot()['prompt_sha256']
        if prompt is not None:
            self.stages.append(dict(stage=stage,binding_index=lane['binding_index'],controller_request_id=rid,
                operation_instance=None if operation is None else operation.instance_sha256,prompt_sha256=prompt,
                raw_response_sha256=None if response is None or response.get('accepted') is not True else digest(dict(text=response['text'],usage=response['usage'])),parsed_claim_sha256=None,
                rule_outcome='stage_failed' if error or response is None else 'indeterminate_claim',
                predecessor_instance=None if previous is None else previous.snapshot()['entry']['instance_sha256'],
                status='failed' if error or response is None else 'completed'))
        if entry is not None:self.last_entry=entry;self.last_operation=operation

        if entry is not None:lane['completion'].release_entry(decision)
        lane['completion'].release_capture(decision);self.pending=None
    def completed(self,case,record):self.case=case
    def partial(self,case,outcome):
        if self.case is None:
            self.case=case;self.case['outcome']='cancelled' if outcome=='cancelled' else 'stage_failed'
    def closed(self,outcome):self.outcome=outcome
    async def export(self):
        if self.case is None:raise RuntimeError('no completed actual fill case')
        case=self.case
        # Rule outcomes/parsed claims come from the actual private chain export,
        # rather than interpretation of model text by the coordinator.
        attempts={a.get('request_id'):a for a in case['attempts']}
        for row in self.stages:
            attempt=attempts.get(row['controller_request_id'],{})
            row['rule_outcome']=attempt.get('comparison','stage_failed' if row['status']=='failed' else 'indeterminate_claim')
            row['parsed_claim_sha256']=None if attempt.get('claim') is None else digest(attempt['claim'])
        refs={role:await self.r.artifacts.publication_ready(handle) for role,handle in self.publications.items()}
        value=dict(schema_version=1,kind='recording_fill_case',execution_sha256=self.r.artifacts.execution_sha256,
            case_id=case['id'],source_manifest_sha256=self.packet.source_manifest_sha256,mapping_sha256=self.packet.mapping_sha256,
            capture_ref=capture_reference(self.packet),image_sha256=case['source_sha256'],configuration_sha256=case['capture']['configuration_sha256'],
            rule_sha256=case['rule_sha256'],measurement_publication=refs['fill_measurement'],descriptor_publication=refs['fill_descriptor'],
            stages=self.stages,comparison_publication=refs.get('fill_comparison'),outcome=case['outcome'])
        return await self.r.artifacts.write_fill_case(self.case_ordinal,value)
