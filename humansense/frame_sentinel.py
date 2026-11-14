"""Opt-in fresh-frame sampling through the existing controller/Smol executor.

Capture timestamps are Unix milliseconds; scheduling uses host monotonic ns.
No camera, detector, consumer LLM or receipt extraction is created here.
"""
import asyncio
from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import time

from .capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from .call_controller import _digest
from .plugins import InputFrame
from .schema import timestamp
from .smolvlm_call import SmolVLMCalls, smolvlm_frame_sha256
from .streams import _finish


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('bounded integer sentinel configuration required')
    return value


class FreshFrameSentinel:
    """One source, one selected worker, one active and one latest candidate.

    All calls and lifecycle changes belong to the HumanSense event loop. A
    supplied capture is evidence of bytes, not evidence of camera authenticity,
    recognition quality or continuous coverage. Invalidated instances cannot
    restart; construct a new explicit binding using the existing controller.
    """
    @classmethod
    async def open(cls, human, sentinel_id, controller_id, python, config, *,
                   subject_id='local-user', sensor, frame_shape, maximum_charge=None,
                   interval_ms=5000, phase_ms=0, deadline_tolerance_ms=250,
                   max_age_ms=1000, max_owned_bytes=16*1024*1024, history_limit=128,
                   ttl_ms=1000, capture_clock='unix_ms', evaluation_observer=None,
                   startup_timeout=120, inference_timeout=30):
        human._check();identifier(sentinel_id);identifier(controller_id);identifier(subject_id)
        if controller_id not in human._call_controllers:
            raise ValueError('registered controller required')
        if not isinstance(sensor, SensorMetadata) or sensor.stream_epoch is None:
            raise ValueError('explicit original sensor and capture epoch required')
        if (type(frame_shape) is not tuple or len(frame_shape)!=3 or
                any(type(v) is not int for v in frame_shape) or frame_shape[2]!=3 or
                min(frame_shape[:2])<32 or max(frame_shape[:2])>2160 or frame_shape[0]*frame_shape[1]>1920*1080):
            raise ValueError('bounded original HWC BGR shape required')
        interval_ms=_integer(interval_ms,1,3600000);phase_ms=_integer(phase_ms,0,interval_ms-1)
        deadline_tolerance_ms=_integer(deadline_tolerance_ms,0,interval_ms-1)
        max_age_ms=_integer(max_age_ms,1,60000);max_owned_bytes=_integer(max_owned_bytes,1,16*1024*1024)
        history_limit=_integer(history_limit,1,128);ttl_ms=_integer(ttl_ms,1,3600000)
        if capture_clock!='unix_ms':raise ValueError('capture timestamp domain must be unix_ms; map other clocks explicitly before submission')
        if maximum_charge is not None:_integer(maximum_charge,0,2**53-1)
        frame_bytes=frame_shape[0]*frame_shape[1]*3
        if frame_bytes>max_owned_bytes:raise ValueError('one frame exceeds configured retained pixel limit')
        self=cls();self.human=human;self.id=sentinel_id;self.controller_id=controller_id
        self.controller=human._call_controllers[controller_id];self.subject=subject_id;self.sensor=sensor
        self.shape=frame_shape;self.frame_bytes=frame_bytes;self.maximum_charge=maximum_charge;self.ttl_ms=ttl_ms
        self._config=dict(interval_ms=interval_ms,phase_ms=phase_ms,deadline_tolerance_ms=deadline_tolerance_ms,
            max_age_ms=max_age_ms,max_owned_bytes=max_owned_bytes,history_limit=history_limit,
            capture_clock=capture_clock,scheduler_clock='host_monotonic_ns',
            selection='latest_received_unique_capture_at_due_tick',skipped_ticks='skip_without_catchup')
        self._epoch=(human._session_id,human._generation);self._loop=asyncio.get_running_loop()
        self._candidate=None;self._active_frame=None;self._active_meta=None;self._active_task=None
        self._scheduler=None;self._startup_task=None;self._close_task=None;self.adapter=None
        self._closed=False;self._invalidated=False;self._source=None;self._reason=None
        self._last_index=-1;self._last_capture=None;self._last_wall=None;self._origin=None;self._next_tick=0
        self._events=deque(maxlen=history_limit);self._event_count=0;self._counts=Counter()
        self._last_success=None;self._unavailable_since=None
        freshness=min(max_age_ms,self.controller._policy.max_age_ms)
        self.spec=CapabilitySpec('sdk.capture.available.'+_digest({'freshness_ms':freshness})[:12],
            'capture.available','video','gamst.capture_availability','1',
            (OutputField('capture_available','boolean'),OutputField('capture_index','integer'),
             OutputField('width','integer'),OutputField('height','integer'),OutputField('pixels_sha256','string')),
            'frame',freshness_ms=freshness,interpretation='measurement')
        human.register_capability(self.spec)
        human._register_call_driver('frame_sentinel',sentinel_id,self)
        options=dict(startup_timeout=startup_timeout,inference_timeout=inference_timeout)
        if evaluation_observer is not None:options['evaluation_observer']=evaluation_observer
        try:
            self._startup_task=asyncio.create_task(SmolVLMCalls.open(human,controller_id,python,config,**options))
            self.adapter=await asyncio.shield(self._startup_task)
            self._check();return self
        except BaseException:
            self.invalidate('startup_failed')
            await _finish(self._close_task)
            raise

    @property
    def config(self):
        """Detached frozen-declaration view; changes require a new explicit open."""
        return dict(self._config)

    def _check(self):
        if asyncio.get_running_loop() is not self._loop:raise RuntimeError('sentinel belongs to its original event loop')
        self.human._check()
        if self._closed or self._invalidated or self._epoch!=(self.human._session_id,self.human._generation):
            raise RuntimeError('sentinel binding invalidated; explicitly reopen')

    def _event(self,kind,**fields):
        self._event_count+=1;self._counts[kind]+=1
        self._events.append(dict(kind=kind,observed_monotonic_ns=time.monotonic_ns(),**fields))

    @staticmethod
    def _scrub(item):
        if item is not None:
            frame=item.pop('frame',None)
            if frame is not None:frame.data.fill(0)
            item.clear()

    def _wall(self):
        now=timestamp()
        if self._last_wall is not None and now<self._last_wall:
            self.invalidate('capture_clock_regressed')
            raise ValueError('host Unix capture clock regressed; explicit new binding required')
        self._last_wall=now;return now

    def offer(self,frame):
        """Own latest bytes synchronously; never waits for model/queue/future input."""
        import numpy as np
        self._check();now=self._wall();arrival=time.monotonic_ns()
        if not isinstance(frame,InputFrame) or not isinstance(frame.sensor,SensorMetadata):raise ValueError('typed original camera frame required')
        sensor=frame.sensor;binding=lambda s:{k:v for k,v in asdict(s).items() if k not in ('sequence','capture_sequence')}
        if (frame.subject_id!=self.subject or frame.modality!='video' or frame.sample_rate is not None or
                binding(sensor)!=binding(self.sensor) or (sensor.capture_sequence is None)!=(self.sensor.capture_sequence is None)):
            self.invalidate('source_binding_changed');raise ValueError('capture owner/source binding changed; explicitly reopen')
        data=frame.data
        if type(data) is not np.ndarray or data.dtype!=np.uint8 or tuple(data.shape)!=self.shape:
            self.invalidate('capture_format_changed');raise ValueError('original plain uint8 HWC BGR shape changed')
        index=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        if (type(frame.timestamp_ms) is not int or not 0<=frame.timestamp_ms<=now or
                self._last_capture is not None and frame.timestamp_ms<self._last_capture or index<=self._last_index):
            raise ValueError('original Unix capture timestamp/index must advance without future or replay')
        if now-frame.timestamp_ms>self.config['max_age_ms']:
            self._event('stale_capture',capture_ms=frame.timestamp_ms,capture_index=index);return dict(accepted=False,reason='stale_capture')
        retained=self.frame_bytes*(self._active_frame is not None)
        if retained+self.frame_bytes>self.config['max_owned_bytes']:
            self._event('pixel_capacity',capture_ms=frame.timestamp_ms,capture_index=index);return dict(accepted=False,reason='pixel_capacity')
        # A memoryview pins allocation/geometry while copying. The owned snapshot,
        # rather than mutable caller metadata, determines every subsequent hash.
        with memoryview(data) as view:
            if view.format!='B' or view.itemsize!=1 or tuple(view.shape)!=self.shape or view.nbytes!=self.frame_bytes:
                raise ValueError('bounded original camera buffer required')
            if self._candidate is not None:
                self._event('candidate_replaced',capture_ms=self._candidate['frame'].timestamp_ms,
                    capture_index=self._candidate['index'],meaning='unadmitted_capture_unobserved')
                self._scrub(self._candidate);self._candidate=None
            copied=np.asarray(view).copy(order='C')
        owned=replace(frame,data=copied);digest=smolvlm_frame_sha256(copied)
        if self._last_index>=0 and index!=self._last_index+1:
            self._event('capture_index_gap',missing_indices=index-self._last_index-1,
                previous_capture_ms=self._last_capture,capture_ms=frame.timestamp_ms,meaning='unobserved_not_physical_capture_loss_proof')
        self._candidate=dict(frame=owned,index=index,sha256=digest,arrival_monotonic_ns=arrival)
        self._last_index=index;self._last_capture=frame.timestamp_ms
        if self._unavailable_since is not None:
            self._event('capture_returned',unavailable_since_monotonic_ns=self._unavailable_since)
            self._unavailable_since=None
        self._counts['captures_received']+=1
        return dict(accepted=True,capture_ms=frame.timestamp_ms,capture_index=index,sha256=digest,
                    owned_pixel_bytes=self.frame_bytes*(1+(self._active_frame is not None)))

    def start(self):
        """Explicit timer activation. First due point is start + frozen phase."""
        self._check()
        if self._scheduler is not None:raise RuntimeError('sentinel already started')
        self._origin=time.monotonic_ns();self._scheduler=asyncio.create_task(self._run(),name='humansense-frame-sentinel')
        return self

    async def _run(self):
        try:
            while not self._invalidated:
                due=self._origin+(self.config['phase_ms']+self._next_tick*self.config['interval_ms'])*1000000
                await asyncio.sleep(max(0,(due-time.monotonic_ns())/1e9))
                await self._tick()
        except asyncio.CancelledError:raise
        except Exception:
            self._event('scheduler_failed');self.invalidate('scheduler_failed')

    async def _tick(self):
        self._check();now=time.monotonic_ns();wall=self._wall()
        step=self.config['interval_ms']*1000000;first=self._origin+self.config['phase_ms']*1000000
        if now<first+self._next_tick*step:return
        tick=(now-first)//step;skipped=tick-self._next_tick;due=first+tick*step;self._next_tick=tick+1
        if skipped:self._counts['skipped_ticks']+=skipped;self._event('ticks_skipped',count=skipped,meaning='uninspected')
        self._event('due',tick=tick,due_monotonic_ns=due,lateness_ns=now-due)
        if now-due>self.config['deadline_tolerance_ms']*1000000:
            self._event('deadline_missed',tick=tick,meaning='uninspected');return
        if self._active_task is not None:
            self._refresh_dispatch();self._event('busy',tick=tick,meaning='uninspected');return
        item=self._candidate;self._candidate=None
        if item is None:
            if self._unavailable_since is None:self._unavailable_since=now
            self._event('capture_unavailable',tick=tick,meaning='uninspected');return
        frame=item['frame']
        if (not 0<=wall-frame.timestamp_ms<=self.config['max_age_ms'] or
                not 0<=now-item['arrival_monotonic_ns']<=self.config['max_age_ms']*1000000):
            self._event('stale_selected_capture',tick=tick,capture_ms=frame.timestamp_ms,capture_index=item['index'],meaning='uninspected')
            self._scrub(item);return
        self._active_frame=item;self._active_meta=dict(capture_ms=frame.timestamp_ms,capture_index=item['index'],
            frame_sha256=item['sha256'],tick=tick,due_monotonic_ns=due,selected_monotonic_ns=now,
            processing_epoch=f'{self._epoch[0]}:{self._epoch[1]}',request_id=None,dispatched=False)
        self._event('selected',**self._active_meta)
        self._active_task=asyncio.create_task(self._execute(item,self._active_meta),name='humansense-frame-inspection')

    def _refresh_dispatch(self):
        meta=self._active_meta
        if meta is None or meta['dispatched'] or meta['request_id'] is None:return
        state=self.controller.snapshot();rid=meta['request_id']
        row=next((r for r in state['requests']+state['recent'] if r['id']==rid),None)
        admission=None if row is None else row.get('last_admission')
        if rid in state['detached_financial_requests'] or admission and admission['stage']=='dispatch' and admission['status']=='dispatched':
            meta['dispatched']=True
            self._event('dispatched',request_id=rid,capture_ms=meta['capture_ms'],capture_index=meta['capture_index'],
                controller_dispatch_ms=None if admission is None else admission['evaluated_ms'],
                meaning='controller_dispatch_not_backend_entry')

    async def _execute(self,item,meta):
        rid=None
        try:
            self._check();frame=item['frame']
            record=await self.human.push_observation(self.spec.id,dict(capture_available=True,capture_index=item['index'],
                width=self.shape[1],height=self.shape[0],pixels_sha256=item['sha256']),sensor=frame.sensor,
                subject_id=frame.subject_id,timestamp_ms=frame.timestamp_ms,
                quality={'status':'sufficient','flags':['capture_bytes_valid_only_not_scene_or_detection_quality']})
            self._check()
            if self._source is None:self._source=self.controller.open_source(record,now_ms=timestamp())
            reference=dict(source_id=frame.sensor.source_id,sensor_id=frame.sensor.sensor_id,
                stream_epoch=frame.sensor.stream_epoch,capture_ms=frame.timestamp_ms,sha256=item['sha256'])
            # Other application callers share this controller's admission order;
            # our request/evidence pair is always the one submitted here.
            with self.controller._lock:
                submission=self.controller.submit(self._source,record,sequence=self.controller._submission_seq+1,
                    now_ms=timestamp(),visual_reference=reference)
            rid=submission.get('request_id');meta['request_id']=rid
            if rid is None:
                self._event('not_admitted',policy_status=submission['status'],capture_ms=frame.timestamp_ms,
                    capture_index=item['index'],reason=submission.get('reason',submission['decision']['reason']),meaning='uninspected')
                return
            pending=next(r for r in self.controller.snapshot()['requests'] if r['id']==rid)
            if pending['decision']['evidence']['record_sha256']!=_digest(record) or pending['decision']['visual_reference']!=reference:
                raise ValueError('sentinel request/evidence join changed')
            result=await self.adapter.execute(rid,record,frame,maximum_charge=self.maximum_charge,ttl_ms=self.ttl_ms,
                dispatch_deadline_ns=meta['due_monotonic_ns']+self.config['deadline_tolerance_ms']*1000000)
            self._refresh_dispatch()
            if 'dispatch_deadline_exceeded' in result.get('blockers',[]):
                self._event('dispatch_deadline_missed',capture_ms=meta['capture_ms'],capture_index=meta['capture_index'],meaning='uninspected')
            self._event('completed',request_id=rid,status=result['status'],capture_ms=meta['capture_ms'],
                capture_index=meta['capture_index'],actual_cost=None)
            if result.get('accepted') is True and result['status']=='published' and not self._invalidated:
                if self._last_success is not None and self._last_success['capture_ms']<meta['capture_ms']:
                    self._event('between_sample_points',start_capture_ms=self._last_success['capture_ms'],
                        end_capture_ms=meta['capture_ms'],meaning='open_interval_unobserved_no_continuous_coverage')
                output=result['observation'];self._last_success=dict(meta,observation_id=output['id'],
                    model=deepcopy(output['models']),completed_monotonic_ns=time.monotonic_ns(),
                    prompt_sha256=output['observation']['value']['model']['prompt_sha256'],
                    meaning='inspected_capture_point_untrusted_description_not_continuous_coverage')
                self._event('accepted',**self._last_success)
            else:self._event('uninspected',request_id=rid,reason=result['status'],capture_ms=meta['capture_ms'],capture_index=meta['capture_index'])
        except asyncio.CancelledError:
            self._refresh_dispatch();self._counts['cancelled']+=1
            raise
        except Exception:
            self._refresh_dispatch();self._event('execution_failed',meaning='uninspected')
        finally:
            if rid is not None:
                state=self.controller.snapshot()
                pending=next((r for r in state['requests'] if r['id']==rid),None)
                if pending is not None and pending['state']!='dispatched':self.controller.cancel(rid,now_ms=timestamp())
            self._scrub(item)
            if self._active_frame is item:self._active_frame=None
            if self._active_meta is meta:self._active_meta=None
            self._active_task=None
            if self.adapter is not None and self.adapter.worker.closed and not self._invalidated:self.invalidate('worker_unavailable')

    def invalidate(self,reason):
        """Synchronous privacy/lifecycle barrier, before any cleanup await."""
        if reason not in ('startup_failed','capture_clock_regressed','source_binding_changed','capture_format_changed',
                          'scheduler_failed','worker_unavailable','close','consent_revoked','pause','reset','delete'):
            raise ValueError('known sentinel lifecycle reason required')
        if self._invalidated:return
        self._refresh_dispatch();self._invalidated=True;self._reason=reason
        self._scrub(self._candidate);self._scrub(self._active_frame)
        self._candidate=None;self._active_frame=None;self._active_meta=None
        if self._source is not None:
            if self._source in self.controller.snapshot()['sources']:
                self.controller.close_source(self._source,now_ms=timestamp())
            self._source=None
        self._last_success=None;self._events.clear();self.subject=None;self.sensor=None
        self._last_capture=None;self._last_index=-1;self._unavailable_since=None
        for task in (self._startup_task,self._scheduler,self._active_task):
            if task is not None and task is not asyncio.current_task() and not task.done():task.cancel()
        if self._close_task is None:
            self._close_task=asyncio.create_task(self._close())
            self.human._track_call_driver_cleanup('frame_sentinel',self.id,self,self._close_task)

    def owns_task(self,task):
        return task in (self._startup_task,self._scheduler,self._active_task,self._close_task) or (
            self.adapter is not None and task is self.adapter._active)

    async def _close(self):
        self._closed=True
        owned=[t for t in (self._startup_task,self._scheduler,self._active_task) if t is not None and t is not asyncio.current_task()]
        if owned:await asyncio.gather(*owned,return_exceptions=True)
        self._active_task=None;self._scheduler=None
        adapter=self.adapter
        if adapter is None and self._startup_task is not None and not self._startup_task.cancelled():
            try:adapter=self._startup_task.result()
            except Exception:pass
        if adapter is not None:await adapter.close()

    async def close(self):
        self.invalidate('close')
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task);raise

    async def __aenter__(self):self._check();return self
    async def __aexit__(self,*args):await self.close()

    def snapshot(self):
        """Read-only bounded diagnostics. Polling never schedules or executes."""
        return deepcopy(dict(id=self.id,controller_id=self.controller_id,closed=self._closed,
            invalidated=self._invalidated,invalidation_reason=self._reason,configuration=self.config,
            sampling_origin_monotonic_ns=self._origin,next_tick=self._next_tick,
            binding=None if self.sensor is None else dict(subject_id=self.subject,sensor=asdict(self.sensor),shape=list(self.shape)),
            pending=self._candidate is not None,inflight=self._active_task is not None,
            owned_pixel_bytes=self.frame_bytes*((self._candidate is not None)+(self._active_frame is not None)),
            counters=dict(self._counts),events=list(self._events),total_events=self._event_count,
            omitted_events=self._event_count-len(self._events),last_accepted_capture_point=self._last_success,
            unavailable_since_monotonic_ns=self._unavailable_since,continuous_coverage_verified=False,
            semantic_quality_verified=False,actual_cost=None,physical_capture_verified=False,
            interpretation='sampled_capture_points_and_unobserved_gaps_only'))
