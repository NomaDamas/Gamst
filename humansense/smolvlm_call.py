"""Explicit single-frame small-VLM execution through an owned local worker."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import struct
import time

from .capabilities import SensorMetadata, bounded_json, identifier
from .call_controller import _digest
from .plugins import InputFrame
from .process_provider import ProcessProvider
from .providers.smolvlm_native import configuration, prompt_text
from .schema import timestamp
from .streams import _finish


def smolvlm_frame_sha256(image):
    """Versioned shape-bound digest of plain uint8 HWC BGR pixels, C-order.

    Use this when creating the controller visual_reference for SmolVLMCalls.
    """
    import numpy as np
    if type(image) is not np.ndarray or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<32 or max(image.shape[:2])>2160 or image.shape[0]*image.shape[1]>1920*1080:
        raise ValueError('bounded plain uint8 BGR pixels required')
    header=b'gamst.smolvlm.uint8.HWC.BGR.v1\0'+struct.pack('>III',*image.shape)
    return hashlib.sha256(header+image.tobytes(order='C')).hexdigest()


class SmolVLMCalls:
    """Application-owned adapter: explicit open/execute/close, no automatic routing.

    One in-flight call; raw pixels are retained only for that call. HumanSense owns
    admission and output lifecycle, while the application owns this worker lifetime.
    """
    @classmethod
    async def open(cls, human, controller_id, python, config, *, startup_timeout=120, inference_timeout=30, evaluation_observer=None):
        human._check(); identifier(controller_id)
        if controller_id not in human._call_controllers:
            raise ValueError('registered HumanSense controller required')
        if evaluation_observer is not None:human._register_evaluation_observer(evaluation_observer)
        native = configuration(config)
        worker = await ProcessProvider.open(python,'smolvlm2.500m.local',dict(native,window_frames=1),
            startup_timeout=startup_timeout,inference_timeout=inference_timeout,
            **({"evaluation_observer":evaluation_observer} if evaluation_observer is not None else {}))
        try:
            human._check()
            human.register_capability(worker.spec)
            self=cls();self.human=human;self.controller=human._call_controllers[controller_id]
            self._evaluation_observer=evaluation_observer
            self.worker=worker;self.native=native;self._active=None;self._closed=False;self._close_task=None
            return self
        except BaseException:
            await _finish(asyncio.create_task(worker.close()))
            raise

    async def __aenter__(self):
        if self._closed:raise RuntimeError('adapter closed')
        return self

    async def __aexit__(self,*args):await self.close()

    def _capture(self, request_id, record, frame):
        import numpy as np
        record=bounded_json(record,max_bytes=131072)
        pending=next((r for r in self.controller.snapshot()['requests'] if r['id']==request_id),None)
        if pending is None or pending['state']!='pending' or pending['decision']['desired_route']!='small_vlm':
            raise ValueError('pending small_vlm request required')
        decision=pending['decision'];evidence=decision['evidence']
        if _digest(record)!=evidence['record_sha256']:
            raise ValueError('record differs from retained request evidence')
        if not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None or not isinstance(frame.sensor,SensorMetadata):
            raise ValueError('camera InputFrame required')
        data=frame.data
        if type(data) is not np.ndarray or data.dtype!=np.uint8 or data.ndim!=3 or data.shape[2]!=3 or min(data.shape[:2])<32 or max(data.shape[:2])>2160 or data.shape[0]*data.shape[1]>1920*1080:
            raise ValueError('bounded plain uint8 BGR pixels required')
        sensor=asdict(frame.sensor);sensor['channels']=list(frame.sensor.channels)
        if (type(frame.timestamp_ms) is not int or frame.timestamp_ms!=record['timestamp_ms'] or
                frame.subject_id!=record['subject_id'] or sensor!=record['observation']['sensor']):
            raise ValueError('frame ownership/capture metadata differs from retained observation')
        copied=replace(frame,data=data.copy())
        digest=hashlib.sha256(copied.data.tobytes()).hexdigest()
        reference=decision['visual_reference']
        if reference is None or reference['sha256']!=smolvlm_frame_sha256(copied.data):
            raise ValueError('retained pixel bytes differ from visual reference')
        return copied,dict(evidence,_source_handle=pending['source']),digest

    def _complete(self, request_id, evidence, *, success, result_sha256):
        # Receipt and result can arrive in either order. Hold the controller's
        # reentrant lock across completion and the conditional unknown receipt.
        with self.controller._lock:
            state=self.controller.snapshot()
            pending=next((r for r in state['requests'] if r['id']==request_id),None)
            if pending is not None:
                charged=pending['charged']
                completion=self.controller.finish(request_id,success=success,
                    evidence_sha256=evidence['record_sha256'],result_sha256=result_sha256,now_ms=timestamp())
                if not charged:self.controller.settle(request_id,components=None,now_ms=timestamp())
                return completion
            if request_id in state['detached_financial_requests']:
                self.controller.settle(request_id,components=None,now_ms=timestamp())
                return {'accepted':False}
            if evidence['_source_handle'] not in state['sources']:
                # Deletion may already have erased a financially settled request.
                return {'accepted':False}
            raise ValueError('completion request is no longer available')

    def _validate_result(self,value,frame,pixels_sha256):
        value=self.worker.spec.validate(value)
        semantic,window,model=value['semantic'],value['window'],value['model']
        sequence=frame.sensor.capture_sequence if frame.sensor.capture_sequence is not None else frame.sensor.sequence
        capture=[dict(timestamp_ms=frame.timestamp_ms,capture_sequence=sequence,pixels_sha256=pixels_sha256)]
        expected=dict(subject_id=frame.subject_id,source_id=frame.sensor.source_id,sensor_id=frame.sensor.sensor_id,
            stream_epoch=frame.sensor.stream_epoch,coordinate_frame=frame.sensor.coordinate_frame,
            calibration_id=frame.sensor.calibration_id,channels=list(frame.sensor.channels),shape=list(frame.data.shape),
            sequence_kind='capture' if frame.sensor.capture_sequence is not None else 'sensor')
        if (any(window.get(k)!=v for k,v in expected.items()) or window.get('captures')!=capture or
                window.get('capture_start_ms')!=frame.timestamp_ms or window.get('capture_end_ms')!=frame.timestamp_ms or
                window.get('required_frames')!=1 or window.get('observed_frames')!=1 or window.get('complete') is not True):
            raise ValueError('worker result differs from retained capture')
        if (model.get('checkpoint_sha256')!=self.native['checkpoint_sha256'] or
                model.get('tokenizer_sha256')!=self.native['tokenizer_sha256'] or
                model.get('prompt_sha256')!=hashlib.sha256(prompt_text([frame.timestamp_ms]).encode()).hexdigest()):
            raise ValueError('worker result model/prompt binding mismatch')
        if (semantic.get('status') not in ('described','empty_generation') or type(semantic.get('text')) is not str or
                len(semantic['text'].encode())>8192 or semantic.get('quality_validated') is not False or
                semantic.get('interpretation')!='untrusted_semantic_hypothesis'):
            raise ValueError('bounded unvalidated semantic result required')
        return value

    async def execute(self, request_id, record, frame, *, maximum_charge, ttl_ms=1000, dispatch_deadline_ns=None):
        """Execute exactly one pending small_vlm request, then publish if still valid.

        Callers supply the original registered record/pixels and a verified budget
        ceiling. Actual charges remain unknown until a separate controller.settle.
        Optional dispatch_deadline_ns is an absolute reading of this host's
        time.monotonic_ns clock, never a capture timestamp or remote clock.
        """
        self.human._check()
        if self._closed or self.worker.closed:raise RuntimeError('adapter worker closed')
        if self._active is not None:raise RuntimeError('adapter busy; no hidden queue')
        if dispatch_deadline_ns is not None and (type(dispatch_deadline_ns) is not int or not 1<=dispatch_deadline_ns<=2**63-1):
            raise ValueError('bounded absolute host monotonic dispatch deadline required')
        copied,evidence,pixels_sha=self._capture(request_id,record,frame)
        task=asyncio.create_task(self._execute(request_id,copied,evidence,pixels_sha,maximum_charge,ttl_ms,dispatch_deadline_ns))
        self._active=task
        try:
            return await task
        finally:
            if self._active is task:self._active=None

    async def _execute(self,request_id,copied,evidence,pixels_sha,maximum_charge,ttl_ms,dispatch_deadline_ns):
        dispatched=False;completed=False
        ticket=None;outcome="unadmitted"
        try:
            # Explicit retained frames need not be consecutive in the camera stream.
            await self.worker.reset()
            if dispatch_deadline_ns is not None and time.monotonic_ns()>dispatch_deadline_ns:
                return dict(status='blocked',blockers=['dispatch_deadline_exceeded'])
            if self._evaluation_observer is not None:
                ticket=self._evaluation_observer.reserve_call(self.controller,request_id,provider=self.worker.spec.provider,
                    model_revision=self.worker.spec.version,
                    prompt_sha256=hashlib.sha256(prompt_text([copied.timestamp_ms]).encode()).hexdigest())
            reserved=self.controller.reserve(request_id,provider=self.worker.spec.provider,
                model_revision=self.worker.spec.version,
                prompt_sha256=hashlib.sha256(prompt_text([copied.timestamp_ms]).encode()).hexdigest(),
                maximum_charge=maximum_charge,ttl_ms=ttl_ms,now_ms=timestamp())
            if reserved['status']!='reserved':return reserved
            if dispatch_deadline_ns is not None and time.monotonic_ns()>dispatch_deadline_ns:
                self.controller.cancel(request_id,now_ms=timestamp())
                return dict(status='blocked',blockers=['dispatch_deadline_exceeded'])
            admission=self.controller.dispatch(request_id,now_ms=timestamp())
            if admission['status']!='dispatched':return admission
            dispatched=True;outcome="error"
            if ticket is not None:ticket.dispatched()
            callback=lambda:self.worker.process(copied,**({"evaluation_operation_id":ticket.id} if ticket is not None else {}))
            raw=await (ticket.invoke(callback) if ticket is not None else callback())
            value=self._validate_result(raw,copied,pixels_sha)
            success=bool(value['semantic']['text'].strip()) and value['semantic']['status']=='described'
            completion=self._complete(request_id,evidence,success=success,result_sha256=_digest(value));completed=True
            if not completion['accepted']:
                outcome='discarded'
                return dict(status='discarded',accepted=False,actual_cost=None)
            # No await between accepted epoch validation and push_signal's initial
            # synchronous store emission. As with HumanSense, use one event loop.
            if evidence['processing_epoch']!=f'{self.human._session_id}:{self.human._generation}':
                outcome='discarded'
                return dict(status='discarded',accepted=False,actual_cost=None)
            self.human._check()
            observation=await self.human.push_observation(self.worker.spec.id,value,sensor=copied.sensor,
                subject_id=copied.subject_id,timestamp_ms=copied.timestamp_ms)
            outcome='published'
            return dict(status='published',accepted=True,observation=observation,actual_cost=None)
        except BaseException as exc:
            outcome="cancelled" if isinstance(exc,asyncio.CancelledError) else "timeout" if isinstance(exc,asyncio.TimeoutError) else "error"
            if dispatched and not completed:
                self._complete(request_id,evidence,success=False,result_sha256=hashlib.sha256(b'failed_or_cancelled').hexdigest())
            raise
        finally:
            if ticket is not None:
                try:
                    if dispatched:ticket.finish(outcome)
                finally:ticket.release(outcome)

    async def _close(self):
        self._closed=True
        task=self._active
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        await self.worker.close()

    async def close(self):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise
