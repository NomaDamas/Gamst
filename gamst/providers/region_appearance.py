"""Exact-capture ROI embedding and bounded historical appearance candidates."""
from collections import deque
from dataclasses import asdict
import hashlib
import math
import threading
import time
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from .visual_retrieval import DINOv3Retrieval


def capture_binding(frame):
    import numpy as np
    data=frame.data
    if (frame.modality!='video' or not isinstance(frame.sensor,SensorMetadata) or
            not frame.sensor.coordinate_frame or not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or
            data.ndim!=3 or data.shape[2]!=3 or min(data.shape[:2])<32 or data.shape[0]>2160 or
            data.shape[1]>3840 or data.shape[0]*data.shape[1]>1920*1080):
        raise ValueError('regions require bounded uint8 BGR with declared image coordinates')
    identifier(frame.subject_id,'subject_id')
    if type(frame.timestamp_ms) is not int or not 0<=frame.timestamp_ms<=2**53-1:
        raise ValueError('invalid capture timestamp')
    sensor=asdict(frame.sensor);sensor['channels']=list(sensor['channels'])
    return dict(subject_id=frame.subject_id,timestamp_ms=frame.timestamp_ms,sensor=sensor,
                shape=list(data.shape),sha256=hashlib.sha256(data.tobytes(order='C')).hexdigest())


def region_evidence(frame,objects,*,detector_id,detector_version,complete_snapshot):
    """Bind trusted caller-supplied detector output to its exact capture, not an attestation."""
    identifier(detector_id);identifier(detector_version)
    if type(complete_snapshot) is not bool: raise ValueError('explicit snapshot completeness required')
    return bounded_json(dict(capture=capture_binding(frame),objects=objects,
        detector_id=detector_id,detector_version=detector_version,complete_snapshot=complete_snapshot))


class RegionAppearance:
    def __init__(self,encoder,*,sha256,capacity=128,retention_ms=600000,max_regions=8,top_k=2):
        self.spec=self.capability(sha256,capacity=capacity,retention_ms=retention_ms,max_regions=max_regions,top_k=top_k)
        if not callable(getattr(encoder,'predictor',None)) or not callable(getattr(encoder,'close',None)):
            raise ValueError('owned RGB encoder with close required')
        self.encoder=encoder;self.capacity=capacity;self.retention_ms=retention_ms
        self.max_regions=max_regions;self.top_k=top_k
        self._lock=threading.RLock();self.closed=False;self._entries=deque();self._owner=self._last=None

    @staticmethod
    def capability(sha256,*,capacity=128,retention_ms=600000,max_regions=8,top_k=2):
        DINOv3Retrieval.capability(sha256,capacity=capacity,retention_ms=retention_ms,top_k=top_k)
        if type(max_regions) is not int or not 1<=max_regions<=8 or not 1<=top_k<=2:
            raise ValueError('regions1..8 and top_k1..2 required')
        digest=hashlib.sha256(f'{sha256}:{capacity}:{retention_ms}:{max_regions}:{top_k}'.encode()).hexdigest()
        return CapabilitySpec('vision.dinov3.regions','vision.object_appearance','video','dinov3.vits16.regions',
            'v1.sha256.'+digest,(OutputField('objects','array'),OutputField('summary','object')),
            'frame',stateful=True,reset_on_gap=False,freshness_ms=1000,interpretation='estimate',
            devices=('cpu',),sensor_requirements=('uint8-bgr-camera','normalized-image-coordinate-frame'),
            license='DINOv3-license;verify-local-artifact')

    @classmethod
    def from_local(cls,path,sha256,**options):
        cls.capability(sha256,**options)
        encoder=DINOv3Retrieval.from_local(path,sha256)
        try:return cls(encoder,sha256=sha256,**options)
        except BaseException:
            encoder.close();raise

    def process(self,frame):
        raise ValueError('region provider requires process_regions with exact capture evidence')

    def reset(self):
        with self._lock:
            if self.closed:raise RuntimeError('region provider closed')
            self._entries.clear();self._owner=self._last=None

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed=True;self._entries.clear();self._owner=self._last=None
                encoder,self.encoder=self.encoder,None
                encoder.close()

    def process_regions(self,frame,evidence):
        with self._lock:
            if self.closed:raise RuntimeError('region provider closed')
            import numpy as np
            evidence=bounded_json(evidence)
            if (type(evidence) is not dict or set(evidence)!={'capture','objects','detector_id','detector_version','complete_snapshot'} or
                    evidence['capture']!=capture_binding(frame) or type(evidence['complete_snapshot']) is not bool):
                raise ValueError('region evidence must match the exact capture')
            identifier(evidence['detector_id']);identifier(evidence['detector_version'])
            rows=evidence['objects']
            if type(rows) is not list or len(rows)>64:raise ValueError('at most64 detector objects required')
            sensor=frame.sensor;ts=frame.timestamp_ms;h,w=frame.data.shape[:2]
            owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.coordinate_frame,
                   sensor.calibration_id,evidence['detector_id'],evidence['detector_version'])
            if owner==self._owner and self._last is not None and (ts<=self._last[0] or sensor.sequence<=self._last[1]):
                raise ValueError('capture sequence/timestamp must increase')
            validated=[];tracks=set()
            for row in rows:
                if type(row) is not dict or not {'track_id','label','location'}<=set(row):raise ValueError('object identity/label/box required')
                track=identifier(row['track_id']);label=row['label'];location=row['location']
                if track in tracks or type(label) is not str or not 1<=len(label)<=256:raise ValueError('unique tracks and bounded labels required')
                tracks.add(track)
                if type(location) is not dict or location.get('unit')!='normalized_image':raise ValueError('normalized image location required')
                box=location.get('bbox')
                if type(box) is not dict or set(box)!={'x','y','width','height'}:raise ValueError('normalized xywh box required')
                x,y,bw,bh=(box[k] for k in ('x','y','width','height'))
                if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y,bw,bh)) or
                    min(x,y)<0 or min(bw,bh)<=0 or x+bw>1.0000001 or y+bh>1.0000001):raise ValueError('out-of-image region')
                validated.append((track,label,dict(box), (max(0,math.floor(x*w)),max(0,math.floor(y*h)),
                    min(w,math.ceil((x+bw)*w)),min(h,math.ceil((y+bh)*h)))))
            outputs=[];pending=[];now=time.monotonic()
            prior=[] if owner!=self._owner else [item for item in self._entries
                if ts-item['timestamp_ms']<=self.retention_ms and (now-item['inserted'])*1000<=self.retention_ms]
            for track,label,box,(x0,y0,x1,y1) in validated[:self.max_regions]:
                reference=uuid.uuid4().hex
                item=dict(track_id=track,label=label[:64],location=dict(bbox=box,unit='normalized_image'),
                    attributes=dict(reference_id=reference,label_truncated=len(label)>64,identity_verified=False,
                                    appearance_status='unavailable_crop',appearance_candidates=[]))
                outputs.append(item)
                ch,cw=y1-y0,x1-x0
                if min(ch,cw)<16 or ch*cw>1024*1024 or max(ch,cw)/min(ch,cw)>4:continue
                vector=np.asarray(self.encoder.predictor(frame.data[y0:y1,x0:x1,::-1].copy()))
                if vector.shape!=(384,) or vector.dtype.kind not in 'fi' or not np.isfinite(vector).all() or (np.abs(vector)>1e6).any():
                    raise ValueError('encoder must return384 finite bounded features')
                vector=vector.astype(np.float64,copy=True);norm=float(np.linalg.norm(vector))
                if norm<=1e-12:raise ValueError('zero appearance descriptor')
                vector/=norm
                scores=sorted(((float(np.clip(np.dot(vector,old['embedding']),-1,1)),old) for old in prior),
                    key=lambda pair:(-pair[0],-pair[1]['timestamp_ms'],pair[1]['reference_id']))
                item['attributes'].update(appearance_status='estimated',appearance_candidates=[dict(
                    reference_id=old['reference_id'],track_id=old['track_id'],label=old['label'],
                    timestamp_ms=old['timestamp_ms'],capture_sequence=old['capture_sequence'],label_truncated=old['label_truncated'],
                    bbox=old['bbox'],cosine_similarity=score,current_location=None,identity_confirmed=False)
                    for score,old in scores[:self.top_k]])
                pending.append(dict(reference_id=reference,track_id=track,label=label[:64],label_truncated=len(label)>64,timestamp_ms=ts,
                    capture_sequence=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence,
                    bbox=box,inserted=now,embedding=vector))
                bounded_json(item,max_bytes=4096,forbid_raw=True)
            result=self.spec.validate(dict(objects=outputs,summary=dict(
                complete_snapshot=evidence['complete_snapshot'] and len(rows)<=self.max_regions,
                regions_omitted=max(0,len(rows)-self.max_regions),returned_count=len(outputs),
                capture_sha256=evidence['capture']['sha256'],detector_id=evidence['detector_id'],
                detector_version=evidence['detector_version'],comparison_scope='historical_object_crop_appearance',
                same_frame_comparisons=False,identity_merging=False,current_location_inferred=False,
                content_role='untrusted_observed_labels',
                retained_regions=min(self.capacity,len(prior)+len(pending)),retention_ms=self.retention_ms)))
            self._entries=deque((prior+pending)[-self.capacity:]);self._owner=owner;self._last=(ts,sensor.sequence)
            return result
