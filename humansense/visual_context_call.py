"""Explicit retained-image context for application-owned async VLM clients."""
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json

from .capabilities import SensorMetadata
from .plugins import InputFrame
from .smolvlm_call import smolvlm_frame_sha256
from .text_context_call import TextContextCalls
from .streams import _finish
from .recording_scope import resolve_evaluation_observer, validate_evaluation_backend


class VisualContextCalls(TextContextCalls):
    """Evidence-bound small/strong VLM calls; no model or network owned here."""

    async def execute(self,request_id,record,frame,question,*,maximum_charge,ttl_ms=1000,evaluation_scope=None):
        self.human._check()
        if self._closed:raise RuntimeError('adapter closed')
        if self._active is not None:raise RuntimeError('adapter busy; no hidden queue')
        observer=resolve_evaluation_observer(self._evaluation_observer,evaluation_scope)
        backend=self._completion;validate_evaluation_backend(observer,backend)
        request,evidence=self._prepare(request_id,record,question,routes=('small_vlm','strong_vlm'))
        if (not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None
                or not isinstance(frame.sensor,SensorMetadata)):
            raise ValueError('camera InputFrame required')
        # Reuse the existing versioned shape/format-bound visual reference contract.
        smolvlm_frame_sha256(frame.data)
        pixels=frame.data.copy()
        source_sha=smolvlm_frame_sha256(pixels)
        sensor=asdict(frame.sensor);sensor['channels']=list(frame.sensor.channels)
        if (type(frame.timestamp_ms) is not int or frame.timestamp_ms!=record['timestamp_ms'] or
                frame.subject_id!=record['subject_id'] or sensor!=record['observation']['sensor']):
            raise ValueError('frame ownership/capture metadata differs from retained observation')
        pending=next((r for r in self.controller.snapshot()['requests'] if r['id']==request_id),None)
        reference=None if pending is None else pending['decision']['visual_reference']
        if reference is None or reference['sha256']!=source_sha:
            raise ValueError('retained pixels differ from visual reference')
        request['_pixels']=pixels
        request['_capture_ms']=frame.timestamp_ms
        request['_source_sha']=source_sha
        result=await self._start(request_id,request,evidence,maximum_charge,ttl_ms,observer,backend)
        if result.get('accepted'):
            result.update(interpretation='untrusted_visual_response',source_frame_sha256=source_sha,
                          image_sha256=request['images'][0]['sha256'],capture_ms=frame.timestamp_ms)
        return result

    @staticmethod
    def _encode(request):
        pixels=request.pop('_pixels');capture=request.pop('_capture_ms');source_sha=request.pop('_source_sha')
        import cv2
        ok,encoded=cv2.imencode('.png',pixels,[cv2.IMWRITE_PNG_COMPRESSION,1])
        if not ok:raise ValueError('PNG encoding failed')
        png=encoded.tobytes()
        if len(png)>8*1024*1024:raise ValueError('PNG exceeds 8 MiB')
        request['images']=[dict(mime_type='image/png',data_base64=base64.b64encode(png).decode('ascii'),
            sha256=hashlib.sha256(png).hexdigest(),source_frame_sha256=source_sha,
            capture_ms=capture,shape=list(pixels.shape))]
        if len(json.dumps(request,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode())>12*1024*1024:
            raise ValueError('serialized visual request exceeds 12 MiB')
        return request

    async def _prepare_execution(self,request):
        # Thread work cannot be forcibly cancelled. Retain and join it before
        # releasing the owned operation, including repeated caller cancellation.
        preparation=asyncio.create_task(asyncio.to_thread(self._encode,request))
        try:return await asyncio.shield(preparation)
        except asyncio.CancelledError:
            await _finish(asyncio.gather(preparation,return_exceptions=True))
            raise
