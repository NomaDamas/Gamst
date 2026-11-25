"""Owned RF-DETR seed detection plus EfficientTAM tracking for SensorStreams."""
import asyncio
from dataclasses import replace
import hashlib
import json

from .object_appearance import ObjectAppearancePipeline
from .process_provider import ProcessProvider
from .provider_wire import spec_value
from .providers.efficienttam_stream import _image
from .providers.region_appearance import region_evidence
from .streams import _finish


class MaskTrackingPipeline(ObjectAppearancePipeline):
    """Reuses the two-worker ownership/lifecycle implementation, not appearance matching."""
    def __init__(self, detector, tracker):
        if detector.spec.id != 'vision.rfdetr.objects' or tracker.spec.id != 'vision.efficienttam.tracking':
            raise ValueError('RF-DETR detector and EfficientTAM tracker required')
        self.detector, self.regions = detector, tracker
        self.spec = self.combined_capability(detector.spec, tracker.spec)
        self._lock = asyncio.Lock()
        self._close_task = None
        self.closed = False
        self._owner = self._last = None

    @staticmethod
    def combined_capability(detector_spec, tracker_spec):
        digest = hashlib.sha256(json.dumps([spec_value(detector_spec),spec_value(tracker_spec)],sort_keys=True).encode()).hexdigest()
        return replace(tracker_spec, id='vision.mask_tracking', provider='gamst.rfdetr_efficienttam',
            version='1.'+digest, license='RF-DETR-code:Apache-2.0;EfficientTAM-code:Apache-2.0;checkpoints:verify-separately')

    @classmethod
    def capability(cls, detector_config, tracker_config):
        return cls.combined_capability(ProcessProvider.capability('rfdetr.nano',detector_config),
            ProcessProvider.capability('efficienttam.tiny512',tracker_config))

    @classmethod
    async def open(cls, detector_python, detector_config, tracker_python, tracker_config, **timeouts):
        cls.capability(detector_config, tracker_config)
        detector = tracker = None
        try:
            detector = await ProcessProvider.open(detector_python,'rfdetr.nano',detector_config,**timeouts)
            tracker = await ProcessProvider.open(tracker_python,'efficienttam.tiny512',tracker_config,**timeouts)
            return cls(detector, tracker)
        except BaseException:
            for provider in (tracker,detector):
                if provider is not None:
                    try: await _finish(asyncio.create_task(provider.close()))
                    except Exception: pass
            raise

    async def process(self, frame):
        async with self._lock:
            if self.closed: raise RuntimeError('mask tracking pipeline closed')
            try:
                owned = replace(frame, data=_image(frame.data))
                result = self.regions.spec.validate(await self.regions.process(owned))
                if result['summary']['status'] == 'needs_prompt':
                    # Detection is sampled only on seed acquisition, so reset its
                    # tracker rather than implying uninterrupted detector state.
                    await self.detector.reset()
                    detections = self.detector.spec.validate(await self.detector.process(
                        replace(owned, data=owned.data.copy())))
                    rows = detections['objects']
                    h,w = owned.data.shape[:2]
                    eligible = [row for row in rows if row['location']['bbox']['width']*w >= 1
                                and row['location']['bbox']['height']*h >= 1]
                    if eligible:
                        evidence = region_evidence(owned, eligible[:4], detector_id=self.detector.spec.id,
                            detector_version=self.detector.spec.version, complete_snapshot=False)
                        result = await self.regions.process_regions(owned, evidence)
                    result['summary']['seed_candidates_omitted'] = max(0, len(eligible)-4)
                    result['summary']['seed_candidates_unseedable'] = len(rows)-len(eligible)
                    result['summary']['seed_detection_performed'] = True
                else:
                    result['summary']['seed_detection_performed'] = False
                return self.spec.validate(result)
            except BaseException:
                await _finish(asyncio.create_task(self._dispose()))
                raise
