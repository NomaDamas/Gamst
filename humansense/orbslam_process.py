"""Opt-in bounded session rotation for the concrete ORB-SLAM3 worker."""
import asyncio
from collections import deque
from dataclasses import replace
import hashlib
import json
import time

from .process_provider import ProcessProvider
from .provider_wire import encode_frame, decode_frame
from .providers.orbslam_context import configuration, ORBSLAMContext
from .streams import _finish


class ORBSLAMProcess:
    """One child and one operation; errors stop, successful budgets may rotate."""
    @classmethod
    async def open(cls,python,config,*,startup_timeout=120,inference_timeout=30,
                   max_starts_per_minute=6):
        if type(max_starts_per_minute) is not int or not 1<=max_starts_per_minute<=60:
            raise ValueError('max_starts_per_minute must be1..60')
        for value in (startup_timeout,inference_timeout):
            if type(value) not in (int,float) or not .01<=value<=600:
                raise ValueError('bounded worker timeout required')
        self=cls();self.config=configuration(**config)
        self.python=python;self.startup_timeout=startup_timeout;self.inference_timeout=inference_timeout
        self.max_starts=max_starts_per_minute;self._starts=deque()
        base=ORBSLAMContext.capability(**self.config)
        digest=hashlib.sha256(json.dumps([base.version,max_starts_per_minute,startup_timeout,inference_timeout]).encode()).hexdigest()
        self.spec=replace(base,provider='orbslam3.rotating',version='1.'+digest)
        self._worker=self._active=self._close_task=None;self.closed=False
        self.sessions=self.attempts=self.completed=self.session_attempts=0
        self.start_attempts=0;self.total_startup_ms=self.total_rotation_ms=self.total_request_ms=0.
        self._last_owner=self._last_capture=None
        self.rotation_ms=self.startup_ms=0.;self.stop_reason=None
        await self._start('open')
        return self

    async def _new_worker(self):
        now=time.monotonic()
        while self._starts and now-self._starts[0]>=60:self._starts.popleft()
        if len(self._starts)>=self.max_starts:
            raise RuntimeError('ORB session startup rate exhausted; explicitly reopen later')
        if self.sessions>=2**53-1:raise RuntimeError('session counter exhausted')
        self._starts.append(now);self.start_attempts+=1
        self._worker=await ProcessProvider.open(self.python,'orbslam3.monocular',self.config,
            startup_timeout=self.startup_timeout,inference_timeout=self.inference_timeout)
        self.startup_ms=(time.monotonic()-now)*1000
        self.total_startup_ms+=self.startup_ms
        self.sessions+=1;self.session_attempts=0

    async def _stop_worker(self):
        worker=self._worker
        if worker is not None:
            try:await worker.close()
            finally:self._worker=None

    def _check(self):
        if self.closed:raise RuntimeError('ORB rotating worker closed; explicitly reopen')
        if self._active is not None:raise RuntimeError('ORB rotating worker busy; no hidden queue')

    async def _start(self,kind,frame=None):
        self._check()
        task=asyncio.create_task(self._operate(kind,frame));self._active=task
        try:
            result=await asyncio.shield(task)
            if self.closed:raise RuntimeError('ORB result discarded after close')
            return result
        except asyncio.CancelledError:
            task.cancel()
            await _finish(asyncio.gather(task,return_exceptions=True))
            raise
        finally:
            if self._active is task:self._active=None

    async def _operate(self,kind,frame):
        try:
            if kind=='open':
                await self._new_worker();return
            if kind=='reset':
                await self._worker.reset();self._last_owner=self._last_capture=None;return
            sensor=frame.sensor
            seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
            owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,
                   sensor.calibration_id,sensor.coordinate_frame,sensor.capture_sequence is not None)
            if self._last_owner==owner and self._last_capture is not None and (seq<=self._last_capture[0] or frame.timestamp_ms<=self._last_capture[1]):
                raise ValueError('capture order must advance across session rotation')
            if self.attempts>=2**53-1:raise RuntimeError('request counter exhausted')
            rotated=False;self.rotation_ms=0.
            if self.session_attempts>=self.config['max_frames']:
                start=time.monotonic()
                await self._stop_worker()  # Must reap before starting another child.
                await self._new_worker()
                self.rotation_ms=(time.monotonic()-start)*1000;rotated=True
                self.total_rotation_ms+=self.rotation_ms
            self.attempts+=1;self.session_attempts+=1
            self._last_owner,self._last_capture=owner,(seq,frame.timestamp_ms)
            started=time.monotonic()
            try:result=await self._worker.process(frame)
            finally:self.total_request_ms+=(time.monotonic()-started)*1000
            if self.closed:raise RuntimeError('ORB result discarded after close')
            if result is None:raise RuntimeError('ORB worker omitted per-capture context')
            result['provenance']['rotation']=dict(session_number=self.sessions,
                request_attempt=self.attempts,session_request_attempt=self.session_attempts,
                rotated_before_capture=rotated,rotation_ms=self.rotation_ms,startup_ms=self.startup_ms,
                continuous_coverage_verified=False,actual_native_attempt_count='unverified',
                actual_cost='unavailable')
            result=self.spec.validate(result);self.completed+=1
            return result
        except BaseException as exc:
            self.closed=True;self.stop_reason=type(exc).__name__
            await _finish(asyncio.create_task(self._stop_worker()))
            raise

    async def process(self,frame):
        self._check()
        # Snapshot before rotation awaits: no caller-owned pixels retained.
        header,payload=encode_frame(frame)
        owned=decode_frame(header,payload)
        return await self._start('process',owned)

    async def reset(self):
        # Does not replenish the native or wrapper lifetime request budget.
        return await self._start('reset')

    def diagnostics(self):
        return dict(closed=self.closed,busy=self._active is not None,
            sessions_started=self.sessions,startup_attempts=self.start_attempts,request_attempts=self.attempts,
            total_successful_startup_ms=self.total_startup_ms,total_successful_rotation_ms=self.total_rotation_ms,
            total_request_elapsed_ms=self.total_request_ms,
            completed_observations=self.completed,session_request_attempts=self.session_attempts,
            last_rotation_ms=self.rotation_ms,last_startup_ms=self.startup_ms,
            pending_frames=int(self._active is not None),hidden_queue=0,
            stop_reason=self.stop_reason,native_attempt_count_verified=False,
            continuous_coverage_verified=False,actual_cost=None)

    async def _close(self):
        self.closed=True
        task=self._active
        if task is not None and not task.done():
            task.cancel()
            await _finish(asyncio.gather(task,return_exceptions=True))
        await self._stop_worker()

    async def close(self):
        self.closed=True
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise
