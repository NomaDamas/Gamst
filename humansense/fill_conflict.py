"""Owned, opt-in same-capture fill/VLM conditional inspection chain."""
import asyncio
from collections import deque
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import math
import uuid

from .capabilities import CapabilitySpec,OutputField,SensorMetadata,bounded_json,identifier
from .call_controller import _digest
from .fill_conflict_rule import RULE_ID,RULE_SHA,parse_claim,compare,prompt
from .plugins import InputFrame
from .providers.fill import CalibratedColorFill
from .schema import timestamp
from .smolvlm_call import smolvlm_frame_sha256
from .streams import _finish
from .visual_context_call import VisualContextCalls


def _client(human,controller_id,value,timeout,observer):
    if type(value) is not dict or set(value)!={'completion','provider','model_revision'}:raise ValueError('explicit application client/provider/revision required')
    return VisualContextCalls(human,controller_id,value['completion'],provider=value['provider'],model_revision=value['model_revision'],timeout=timeout,evaluation_observer=observer)


class FillConflictRuntime:
    """One real OpenCV measurement and at most three explicitly budgeted calls."""
    def __init__(self,human,driver_id,controller_id,*,fill_config,small_client,strong_client=None,
                 strong_opt_in=False,charge_ceilings,chain_ceiling,pricing_revision,unit,timeout=10,retention_ms=60000,evaluation_observer=None):
        human._check();identifier(driver_id);identifier(controller_id)
        if not hasattr(human,'_register_call_driver'):raise RuntimeError('owned call-driver lifecycle support required')
        if controller_id not in human._call_controllers:raise ValueError('registered call controller required')
        controller=human._call_controllers[controller_id]
        if controller._policy.mode!='escalation':raise ValueError('explicit escalation policy required; no policy changes are implicit')
        if type(strong_opt_in) is not bool or strong_opt_in and strong_client is None:raise ValueError('explicit strong client/opt-in required')
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not .01<=timeout<=10:raise ValueError('stage timeout0.01..10s required')
        if type(retention_ms) is not int or not 1<=retention_ms<=60000:raise ValueError('history retention1..60000ms required')
        if type(charge_ceilings) is not dict or set(charge_ceilings)!={'initial','reinspection','strong'}:raise ValueError('three explicit stage ceilings required')
        ceilings=dict(charge_ceilings)
        for amount in ceilings.values():
            if amount is not None and (type(amount) is not int or not 0<=amount<=10**12):raise ValueError('bounded integer quote or explicit unknown required')
        if type(chain_ceiling) is not int or not 0<=chain_ceiling<=3*10**12:raise ValueError('bounded chain authorization required')
        if sum(x for x in ceilings.values() if x is not None)>chain_ceiling:raise ValueError('stage authorization exceeds chain ceiling')
        budget=controller.snapshot()['budget']
        if budget['unit']!=unit or budget['pricing_revision']!=pricing_revision:raise ValueError('chain pricing units/revision differ from authoritative ledger')
        self.human,self.driver_id,self.controller=human,driver_id,controller
        self.fill=CalibratedColorFill(**fill_config);self.config=self.fill.config;self._config_sha=_digest(self.config)
        try:
            self.small=_client(human,controller_id,small_client,timeout,evaluation_observer)
            self.strong=None if strong_client is None else _client(human,controller_id,strong_client,timeout,evaluation_observer)
        except BaseException:
            self.fill.close();raise
        self.strong_opt_in=strong_opt_in;self.ceilings=ceilings;self.chain_ceiling=chain_ceiling
        self.timeout=timeout;self.retention=retention_ms;self.closed=False;self._generation=0
        self._operation=self._native=self._close_task=None;self._case=None;self._history=deque(maxlen=32)
        self._source=self._owner=self._last=self._superseded_order=None;self._delivery=0;self._reason=None
        version='1.'+_digest(dict(config=self.config,rule=RULE_SHA))
        self.descriptor=CapabilitySpec('vision.fill_inspection_input','vision.fill_inspection_input','video','gamst.fill_conflict',version,
            (OutputField('capture','object'),),'frame',freshness_ms=1000,interpretation='estimate')
        self.spec=CapabilitySpec('vision.fill_comparison','vision.fill_comparison','video','gamst.fill_conflict',version,
            (OutputField('comparison','object'),),'frame',freshness_ms=1000,interpretation='estimate')
        # Registration does not start capture, network or models. The actual
        # OpenCV instance is already dependency-checked and owned here.
        try:
            human.register_capability(self.fill.spec);human.register_capability(self.descriptor);human.register_capability(self.spec)
            human._register_call_driver('context_fill',driver_id,self)
        except BaseException:self.fill.close();raise

    def owns_task(self,task):
        return task is not None and (task in (self._operation,self._native,self._close_task) or
            any(task in (adapter._active,adapter._backend_task,adapter._close_task) for adapter in (self.small,self.strong) if adapter))

    def invalidate(self,reason):
        if self._close_task is not None:return
        self.closed=True;self._generation+=1;self._reason=reason;self._case=None;self._history.clear();self._last=None
        if self._operation is not None:self._operation.cancel()
        if self._source is not None:
            try:self.controller.close_source(self._source,now_ms=timestamp())
            except (ValueError,RuntimeError,PermissionError):pass
        self._source=self._owner=None
        self._close_task=asyncio.create_task(self._cleanup())
        self.human._track_call_driver_cleanup('context_fill',self.driver_id,self,self._close_task)

    async def _cleanup(self):
        errors=[]
        for adapter in (self.small,self.strong):
            if adapter is not None:
                try:await adapter.close()
                except BaseException as error:errors.append(error)
        if self._operation is not None:await asyncio.gather(self._operation,return_exceptions=True)
        if self._native is not None:await asyncio.gather(self._native,return_exceptions=True)
        self.fill.close()
        if errors:raise RuntimeError('inspection adapter cleanup failed') from errors[0]

    async def close(self):
        if self.owns_task(asyncio.current_task()):
            self.invalidate('self_close');raise RuntimeError('owned inspection cannot join itself')
        self.invalidate('close')
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task);raise

    def _check(self,generation):
        self.human._check()
        if _digest(self.fill.config)!=self._config_sha:raise RuntimeError('frozen fill configuration changed')
        if self.closed or self.human.paused or generation!=(self._generation,self.human._generation):raise RuntimeError('inspection invalidated')

    def _fresh(self,frame):
        return 0<=timestamp()-frame.timestamp_ms<=min(1000,self.controller._policy.max_age_ms)

    def snapshot(self):
        now=timestamp()
        while self._history and now-self._history[0]['retained_ms']>self.retention:self._history.popleft()
        case=None if self._case is None else {k:v for k,v in self._case.items() if k!='pixels'}
        return bounded_json(dict(closed=self.closed,reason=self._reason,active=case,history=list(self._history)[-2:],omitted_history=max(0,len(self._history)-2),
            original_evidence_age_ms=None if case is None else now-case['capture_ms'],cleanup_pending=self._close_task is not None and not self._close_task.done(),
            history_limit=32,rule_sha256=RULE_SHA,executes_model=False),max_bytes=65536)

    async def inspect(self,frame):
        self.human._check()
        if self.closed:raise RuntimeError('inspection driver closed')
        import numpy as np
        c=self.config
        if (not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None or
            not isinstance(frame.sensor,SensorMetadata) or type(frame.data) is not np.ndarray or frame.data.dtype!=np.uint8 or
            frame.data.shape!=(c['height'],c['width'],3)):raise ValueError('bounded configured plain BGR capture required')
        if frame.sensor.stream_epoch is None or frame.sensor.capture_sequence is None or frame.sensor.channels!=('B','G','R'):
            raise ValueError('declared BGR capture epoch/sequence required')
        if (frame.sensor.calibration_id,frame.sensor.coordinate_frame)!=(c['calibration_id'],c['coordinate_frame']):raise ValueError('calibration/coordinate mismatch')
        identifier(frame.subject_id)
        if type(frame.timestamp_ms) is not int or not 0<=frame.timestamp_ms<=2**53-1:raise ValueError('bounded capture timestamp required')
        if frame.timestamp_ms>timestamp():raise ValueError('future capture is not admissible')
        owner=(frame.subject_id,frame.sensor.source_id,frame.sensor.sensor_id,frame.sensor.stream_epoch,c['calibration_id'],c['container_id'],c['coordinate_frame'])
        order=(frame.timestamp_ms,frame.sensor.capture_sequence)
        if self._owner is not None and owner!=self._owner:
            raise ValueError('source/configuration differs; explicit new driver required')
        if self._operation is not None:
            previous=self._superseded_order or self._last
            if self._fresh(frame) and previous is not None and order[0]>previous[0] and order[1]>previous[1]:
                self._generation+=1;self._case=None;self._superseded_order=order;self._operation.cancel()
            raise RuntimeError('one active case; newer capture invalidates it but is not queued')
        if self._last is not None and (order[0]<=self._last[0] or order[1]<=self._last[1]):raise ValueError('new case requires newer original capture')
        if self._superseded_order is not None and (order[0]<self._superseded_order[0] or order[1]<self._superseded_order[1]):
            raise ValueError('capture predates observed superseding capture')
        pixels=frame.data.copy();pixels.setflags(write=False);frame=replace(frame,data=pixels)
        self._superseded_order=None;self._owner=owner;self._last=order
        generation=(self._generation,self.human._generation)
        task=asyncio.create_task(self._run_case(frame,generation));self._operation=task
        try:return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel();await _finish(asyncio.gather(task,return_exceptions=True));raise
        finally:
            if self._operation is task:self._operation=None
            self._case=None

    async def _run_case(self,frame,generation):
        requests=[]
        try:return await self._inspect(frame,generation,requests)
        finally:
            # Release only this case's never-dispatched work. Dispatched unknown
            # bills retain their controller-owned liability and receipt identity.
            pending={row['id']:row for row in self.controller.snapshot()['requests']}
            for request_id in requests:
                row=pending.get(request_id)
                if row is not None and row['state']!='dispatched':
                    self.controller.cancel(request_id,now_ms=timestamp())

    async def _execute_stage(self,adapter,args,kwargs,deadline):
        # One absolute case deadline; cancellation joins the real adapter's
        # owned cleanup, which can outlive the admission/response deadline.
        remaining=deadline-asyncio.get_running_loop().time()
        if remaining<=0:raise TimeoutError('case deadline exhausted')
        task=asyncio.create_task(adapter.execute(*args,**kwargs))
        try:
            done,_=await asyncio.wait((task,),timeout=remaining)
            if not done:raise TimeoutError('case deadline exhausted')
            return task.result()
        finally:
            if not task.done():
                task.cancel();await _finish(asyncio.gather(task,return_exceptions=True))

    async def _inspect(self,frame,generation,requests):
        self._check(generation);case_id=uuid.uuid4().hex;started=asyncio.get_running_loop().time()
        image_sha=smolvlm_frame_sha256(frame.data)
        case=dict(id=case_id,capture_ms=frame.timestamp_ms,source_sha256=image_sha,stage='measurement',attempts=[],rule_sha256=RULE_SHA,
            capture=dict(subject_id=frame.subject_id,sensor=dict(asdict(frame.sensor),channels=list(frame.sensor.channels)),
                shape=list(frame.data.shape),roi_px=list(self.config['roi']),container_id=self.config['container_id'],
                configuration_sha256=self._config_sha,processing_epoch=f'{self.human._session_id}:{self.human._generation}'))
        self._case=case
        self._native=asyncio.create_task(asyncio.to_thread(self.fill.process,frame))
        try:
            try:numeric=await asyncio.shield(self._native)
            except asyncio.CancelledError:
                await _finish(asyncio.gather(self._native,return_exceptions=True));raise
        finally:self._native=None
        self._check(generation)
        case['fill_processing_ms']=round((asyncio.get_running_loop().time()-started)*1000,3)
        measurement=await self.human.push_observation(self.fill.spec.id,numeric,sensor=frame.sensor,subject_id=frame.subject_id,timestamp_ms=frame.timestamp_ms)
        self._check(generation)
        c=self.config
        # No numeric result/prior model answer goes to any stage's model record.
        descriptor=await self.human.push_observation(self.descriptor.id,dict(capture=dict(
            source_sha256=image_sha,roi_px=list(c['roi']),container_id=c['container_id'],coordinate_frame=c['coordinate_frame'],
            shape=list(frame.data.shape),config_sha256=_digest(c),producer='owned_opencv_input',identity_verified=False)),
            sensor=frame.sensor,subject_id=frame.subject_id,timestamp_ms=frame.timestamp_ms)
        self._check(generation)
        case.update(measurement_record_id=measurement['id'],measurement_sha256=_digest(measurement),descriptor_record_id=descriptor['id'],descriptor_sha256=_digest(descriptor))
        if self._source is None:self._source=self.controller.open_source(descriptor,now_ms=timestamp())
        visual=dict(source_id=frame.sensor.source_id,sensor_id=frame.sensor.sensor_id,stream_epoch=frame.sensor.stream_epoch,capture_ms=frame.timestamp_ms,sha256=image_sha)
        outcome='indeterminate_fill';initial_conflict=False
        if numeric['fill']['status']=='estimated':
            for stage in ('initial','reinspection','strong'):
                self._check(generation)
                if stage=='reinspection' and not initial_conflict:break
                if stage=='strong':
                    if outcome in ('consistent_under_rule','resolved_under_rule'):break
                    if not self.strong_opt_in:outcome='strong_opt_out';break
                if not self._fresh(frame):outcome='blocked_stale_evidence';break
                if asyncio.get_running_loop().time()-started>=30:outcome='blocked_case_deadline';break
                case['stage']=stage
                rules={'query':True,'needs_visual':True} if stage=='initial' else ({'unknown_candidate':True} if stage=='reinspection' else {'conflict_rule':RULE_ID})
                decision=self.controller.submit_next(self._source,descriptor,now_ms=timestamp(),visual_reference=visual,**rules)
                entry=dict(stage=stage,parent_request_id=None if not case['attempts'] else case['attempts'][-1].get('request_id'),
                    status=decision['status'],maximum_charge=self.ceilings[stage],desired_route=decision.get('decision',{}).get('desired_route'),
                    allowed_route=decision.get('decision',{}).get('recommended_route'),dispatched_route=None,rule_sha256=RULE_SHA,evidence_sha256=_digest(dict(case=case_id,measurement=case['measurement_sha256'],descriptor=case['descriptor_sha256'],pixels=image_sha)))
                case['attempts'].append(entry)
                if decision['status']!='queued':outcome='blocked_admission';entry['blockers']=decision.get('decision',{}).get('blockers',[]);break
                rid=decision['request_id'];requests.append(rid);entry['request_id']=rid
                adapter=self.strong if stage=='strong' else self.small
                question=prompt(stage,width=c['width'],height=c['height'],roi=c['roi'],container_id=c['container_id'],coordinate_frame=c['coordinate_frame'])
                try:
                    response=await self._execute_stage(adapter,(rid,descriptor,frame,question),
                        dict(maximum_charge=self.ceilings[stage],ttl_ms=1000),started+30)
                    self._check(generation)
                except (RuntimeError,ValueError,TimeoutError) as error:
                    self._check(generation);entry.update(status='failed',error=type(error).__name__);outcome='stage_failed';break
                state=self.controller.snapshot()
                actual=next((row for row in state['requests']+state['recent'] if row['id']==rid),None)
                if actual is not None and (actual.get('last_admission') or {}).get('stage')=='dispatch' and actual['last_admission'].get('status')=='dispatched':
                    entry['dispatched_route']=actual['execution']['route']
                entry.update(status=response['status'],provider=adapter._provider,model_revision=adapter._revision,
                    prompt_sha256=response.get('prompt_sha256'),result_sha256=_digest(response))
                if not response.get('accepted'):
                    entry['blockers']=response.get('blockers',[]);outcome='blocked_stage';break
                if not self._fresh(frame):outcome='blocked_stale_evidence';entry['status']='historical_response';break
                try:claim,interval=parse_claim(response['text']);result=compare(numeric['fill'],interval)
                except ValueError:claim=None;result='indeterminate_invalid_claim'
                entry.update(claim=claim,claim_id=case_id+':'+stage,claim_version=1,
                    supersedes_claim_id=None if len(case['attempts'])==1 else case['attempts'][-2].get('claim_id'),
                    comparison=result,conditional=True)
                outcome=result
                if stage=='initial':
                    initial_conflict=result=='conditional_image_model_conflict'
                    if not initial_conflict:break
                elif result=='consistent_under_rule':outcome='resolved_under_rule';break
        self._check(generation)
        if not self._fresh(frame):outcome='blocked_stale_evidence'
        case.update(stage='complete',outcome=outcome,conditional=True,current_evidence=self._fresh(frame),
            measurement_verification='not_verified',claim_verification='not_verified',budget_authorization_ceiling=self.chain_ceiling,
            accounting='authoritative_controller_ledger',actual_cost=None,local_overhead_cost=None,
            whole_chain_cost_complete=False,pipeline_elapsed_ms=round((asyncio.get_running_loop().time()-started)*1000,3))
        value=bounded_json(dict(comparison=case),max_bytes=15*1024)
        self._delivery=max(self._delivery+1,frame.sensor.sequence)
        result=await self.human.push_observation(self.spec.id,value,sensor=replace(frame.sensor,sequence=self._delivery),
            subject_id=frame.subject_id,timestamp_ms=frame.timestamp_ms,quality='sufficient' if case['current_evidence'] else 'insufficient')
        self._check(generation)
        self._history.append(dict(retained_ms=timestamp(),value=value))
        return result
