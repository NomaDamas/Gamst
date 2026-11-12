"""Local trusted Ultralytics detection weights; no implicit weight downloads.

Verify both model-weight and Ultralytics software licensing for your deployment.
.pt files are executable/trusted artifacts, not safe untrusted user uploads.
This module does not claim compatibility with untested latest model releases.
"""
from __future__ import annotations
import asyncio
import hashlib
import math
from pathlib import Path
import threading
from .plugins import PerceptionPlugin, _signal, _version
from .tracking import PersonTracker


class UltralyticsYOLOPlugin(PerceptionPlugin):
    """Person boxes/tracks from the Ultralytics YOLO detection API.

    Supply an existing trusted local .pt or .onnx file. Model class numbering is
    explicit: person_class_id defaults to COCO's zero, but other datasets may
    differ. Input must be decoded HWC BGR pixels. The adapter calls predict with
    save=False/show=False; neither raw frames nor prediction files are persisted.
    """
    name='ultralytics_yolo'
    modalities=('video',)

    def __init__(self, model_path, *, person_class_id=0, score_threshold=.5, device='cpu'):
        path=Path(model_path)
        if not path.is_file() or path.suffix.lower() not in ('.pt','.onnx'):
            raise ValueError('Supply an existing trusted local .pt or .onnx detection model')
        if isinstance(person_class_id,bool) or not isinstance(person_class_id,int) or person_class_id<0:
            raise ValueError('person_class_id must be a nonnegative integer')
        if isinstance(score_threshold,bool) or not math.isfinite(score_threshold) or not 0<=score_threshold<=1:
            raise ValueError('score_threshold must be in [0,1]')
        if not isinstance(device,str) or not device.strip():
            raise ValueError('device must explicitly name the local inference device')
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            while block:=handle.read(1024*1024): digest.update(block)
        self.model_hash=digest.hexdigest()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[yolo] and verify upstream/model licenses') from exc
        self.model=YOLO(str(path),task='detect')
        self.person_class_id=person_class_id
        self.score_threshold=score_threshold
        self.device=device
        self.tracker=PersonTracker()
        self._native_lock=threading.Lock()
        self._generation=0
        self._closed=False

    @staticmethod
    def _tolist(value):
        if hasattr(value,'detach'): value=value.detach()
        if hasattr(value,'cpu'): value=value.cpu()
        return value.tolist() if hasattr(value,'tolist') else list(value)

    def _predict(self,image):
        if getattr(image,'ndim',None)!=3 or len(image.shape)!=3 or image.shape[2]!=3 or min(image.shape[:2])<=0:
            raise ValueError('Video must be non-empty HWC BGR pixels')
        if str(getattr(image,'dtype',''))!='uint8':
            raise ValueError('Video must use uint8 BGR pixels')
        height,width=image.shape[:2]
        results=list(self.model.predict(source=image,verbose=False,save=False,show=False,
                                       device=self.device,conf=self.score_threshold,classes=[self.person_class_id]))
        if len(results)!=1:
            raise ValueError('Expected exactly one result for a single image')
        detections=results[0].boxes
        if detections is None: return [],[]
        coordinates=self._tolist(detections.xyxy)
        confidences=self._tolist(detections.conf)
        classes=self._tolist(detections.cls)
        if len(coordinates)>10000 or len(coordinates)!=len(confidences) or len(coordinates)!=len(classes):
            raise ValueError('Model box/score/class arrays have inconsistent lengths')
        boxes,scores=[],[]
        for xyxy,score,category in zip(coordinates,confidences,classes):
            if len(xyxy)!=4: raise ValueError('Expected four xyxy coordinates per detection')
            x1,y1,x2,y2=[float(v) for v in xyxy]
            score,category=float(score),float(category)
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2,score,category)):
                raise ValueError('Detection outputs must be finite')
            if not 0<=score<=1 or category<0 or not category.is_integer():
                raise ValueError('Detection score/class are invalid')
            if score<self.score_threshold or category!=self.person_class_id: continue
            if x2<=x1 or y2<=y1: raise ValueError('Selected detection boxes must have positive area')
            x1,x2=max(0.,min(1.,x1/width)),max(0.,min(1.,x2/width))
            y1,y2=max(0.,min(1.,y1/height)),max(0.,min(1.,y2/height))
            if x2>x1 and y2>y1:
                boxes.append(dict(x=x1,y=y1,width=x2-x1,height=y2-y1)); scores.append(score)
        return boxes,scores

    async def process(self,frame):
        generation=self._generation
        def run():
            with self._native_lock:
                if self._closed: raise RuntimeError('UltralyticsYOLOPlugin is closed')
                if generation!=self._generation: return None
                return self._predict(frame.data)
        result=await asyncio.to_thread(run)
        if result is None or generation!=self._generation: return []
        boxes,scores=result
        tracks=self.tracker.update(boxes,frame.timestamp_ms,stream_id=frame.subject_id)
        version=f"{_version('ultralytics')}:sha256:{self.model_hash}"
        return [_signal(frame,name,value,self.name,version) for name,value in (
            ('scene.person_count',len(boxes)),('scene.person_boxes',boxes),
            ('scene.person_detection_scores',dict(model_scores=scores,probability=None,calibrated=False)),
            ('scene.person_tracks',tracks))]

    def reset(self):
        self._generation+=1
        self.tracker.reset()

    async def close(self):
        self.reset()
        def release():
            with self._native_lock:
                self._closed=True
                self.model=None
        await asyncio.to_thread(release)
