"""Owned two-runtime detector/ROI-embedding pipeline for SensorStreams."""
import asyncio
from dataclasses import replace
import hashlib
import json

from .process_provider import ProcessProvider
from .provider_wire import spec_value
from .providers.region_appearance import RegionAppearance, capture_binding, region_evidence
from .streams import _finish


class ObjectAppearancePipeline:
    def __init__(self,detector,regions):
        if detector.spec.id!='vision.rfdetr.objects' or regions.spec.id!='vision.dinov3.regions':
            raise ValueError('explicit RF-DETR and DINOv3 region providers required')
        self.detector,self.regions=detector,regions
        self.spec=self.combined_capability(detector.spec,regions.spec)
        self._lock=asyncio.Lock();self._close_task=None;self.closed=False;self._owner=self._last=None

    @staticmethod
    def combined_capability(detector_spec,region_spec):
        digest=hashlib.sha256(json.dumps([spec_value(detector_spec),spec_value(region_spec)],sort_keys=True).encode()).hexdigest()
        return replace(region_spec,id='vision.object_appearance',provider='gamst.rfdetr_dinov3',
                       version='v1.sha256.'+digest,license='RF-DETR:Apache-2.0;DINOv3:verify-local-artifact')

    @classmethod
    def capability(cls,detector_config,region_config):
        return cls.combined_capability(ProcessProvider.capability('rfdetr.nano',detector_config),
                                       ProcessProvider.capability('dinov3.regions',region_config))

    @classmethod
    async def open(cls,detector_python,detector_config,region_python,region_config,**timeouts):
        cls.capability(detector_config,region_config)
        detector=regions=None
        try:
            detector=await ProcessProvider.open(detector_python,'rfdetr.nano',detector_config,**timeouts)
            regions=await ProcessProvider.open(region_python,'dinov3.regions',region_config,**timeouts)
            return cls(detector,regions)
        except BaseException:
            for provider in (regions,detector):
                if provider is not None:
                    try:await _finish(asyncio.create_task(provider.close()))
                    except Exception:pass
            raise

    async def _dispose(self):
        self.closed=True;self._owner=self._last=None
        # Attempt both independently; a failing close cannot leak the other worker.
        results=await asyncio.gather(self.regions.close(),self.detector.close(),return_exceptions=True)
        for result in results:
            if isinstance(result,BaseException):raise result

    async def _close(self):
        async with self._lock:
            if not self.closed:await self._dispose()

    async def close(self):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task);raise

    async def reset(self):
        async with self._lock:
            if self.closed:raise RuntimeError('object appearance pipeline closed')
            try:
                await self.detector.reset();await self.regions.reset()
                self._owner=self._last=None
            except BaseException:
                await _finish(asyncio.create_task(self._dispose()));raise

    async def process(self,frame):
        async with self._lock:
            if self.closed:raise RuntimeError('object appearance pipeline closed')
            # Snapshot before the first await. Each worker receives the same owned pixels.
            binding=capture_binding(frame)
            owned=replace(frame,data=frame.data.copy())
            owner=(frame.subject_id,frame.sensor.source_id,frame.sensor.sensor_id,frame.sensor.stream_epoch,
                   frame.sensor.coordinate_frame,frame.sensor.calibration_id)
            if owner==self._owner and self._last and (frame.timestamp_ms<=self._last[0] or frame.sensor.sequence<=self._last[1]):
                raise ValueError('capture order must increase')
            try:
                if self._owner is not None and owner!=self._owner:
                    await self.detector.reset();await self.regions.reset()
                elif self._last and frame.sensor.sequence!=self._last[1]+1:
                    # Detector continuity is gap-sensitive; appearance history is not.
                    await self.detector.reset()
                detections=await self.detector.process(replace(owned,data=owned.data.copy()))
                detections=self.detector.spec.validate(detections)
                evidence=region_evidence(owned,detections['objects'],detector_id=self.detector.spec.id,
                    detector_version=self.detector.spec.version,complete_snapshot=detections['summary']['complete_snapshot'])
                if evidence['capture']!=binding:raise ValueError('capture changed before ROI extraction')
                result=await self.regions.process_regions(owned,evidence)
                result=self.spec.validate(result)
                self._owner=owner;self._last=(frame.timestamp_ms,frame.sensor.sequence)
                return result
            except BaseException:
                await _finish(asyncio.create_task(self._dispose()));raise
