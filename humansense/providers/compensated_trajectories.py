"""Owned RF-DETR/background similarity path with separate anchor projections."""
from dataclasses import replace
import hashlib
import json
import re
import struct
import threading
import uuid

from ..camera_similarity import BackgroundSimilarity,background_roi
from ..capabilities import CapabilitySpec,OutputField,SensorMetadata,identifier,bounded_json
from ..plugins import InputFrame
from .detection import RFDETRObjects

ASSUMPTION='approximately-similarity-static-background'


def configuration(value):
    value=bounded_json(value,max_bytes=8192)
    required={'path','sha256','background_roi','operating_assumption'}
    if type(value) is not dict or not required<=set(value) or set(value)-required-{'device','max_gap_ms'}:
        raise ValueError('exact explicit compensated trajectory configuration required')
    if type(value['path']) is not str or not value['path'] or type(value['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',value['sha256']) is None:
        raise ValueError('local checkpoint path/hash required')
    result=dict(value,background_roi=list(background_roi(value['background_roi'])))
    if result['operating_assumption']!=ASSUMPTION:raise ValueError('explicit conditional operating assumption required')
    result.setdefault('device','cpu');result.setdefault('max_gap_ms',1000)
    if result['device'] not in ('cpu','cuda','mps') or type(result['max_gap_ms']) is not int or not 1<=result['max_gap_ms']<=1000:
        raise ValueError('bounded device/gap policy required')
    return result


class CompensatedTrajectories:
    @staticmethod
    def capability(config):
        config=configuration(config)
        digest=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return CapabilitySpec('vision.rfdetr.compensated_trajectories','vision.trajectories','video',
            'gamst.rfdetr.compensated_trajectories','1.'+digest,
            (OutputField('detections','array'),OutputField('objects','array'),OutputField('geometry','object')),
            'frame',stateful=True,reset_on_gap=True,freshness_ms=config['max_gap_ms'],interpretation='estimate',
            devices=('cpu','cuda','mps'),sensor_requirements=('uint8-bgr-camera','declared-static-background'),license='Apache-2.0')

    def __init__(self,predictor,*,version,config):
        self.config=configuration(config);self.spec=self.capability(self.config)
        self.detector=RFDETRObjects(predictor,version=version,max_objects=8,max_age_ms=self.config['max_gap_ms'])
        self.motion=BackgroundSimilarity(self.config['background_roi'])
        self.lock=threading.RLock();self.closed=False;self._clear()

    @classmethod
    def from_local(cls,config):
        config=configuration(config)
        detector=RFDETRObjects.from_local(config['path'],sha256=config['sha256'],device=config['device'],
            max_objects=8,max_age_ms=config['max_gap_ms'])
        return cls(detector.predictor,version=detector.version,config=config)

    def _clear(self):
        self.detector.reset()
        self.owner=self.last=self.previous=self.anchor=self.to_anchor=None
        self.boxes=[];self.paths={};self.mappings=[];self.links=0

    def reset(self):
        with self.lock:self._clear()

    def close(self):
        with self.lock:
            if not self.closed:self._clear();self.detector.close();self.closed=True

    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('compensated trajectory provider closed')
            try:return self._process(frame)
            except BaseException:self._clear();raise

    def _project_boxes(self,matrix,size):
        np=self.motion.np;w,h=size;result={}
        for key,track in self.detector.tracks.items():
            if track['last_seen_ms']!=self.last[0]:continue
            box=track['bbox'];x,y=box['x']*w,box['y']*h
            bw,bh=box['width']*w,box['height']*h
            points=np.array([[x,y,1],[x+bw,y,1],[x+bw,y+bh,1],[x,y+bh,1]])@matrix.T
            points=points[:,:2]/np.array([w,h]);low=points.min(axis=0);high=points.max(axis=0)
            if np.isfinite(points).all() and (low>=0).all() and (high<=1).all() and (high>low).all():
                result[key]=dict(x=float(low[0]),y=float(low[1]),width=float(high[0]-low[0]),height=float(high[1]-low[1]))
        return result

    def _process(self,frame):
        np=self.motion.np
        if (not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None
                or not isinstance(frame.sensor,SensorMetadata)):
            raise ValueError('explicit video frame and sensor required')
        sensor=frame.sensor;identifier(frame.subject_id)
        if sensor.coordinate_frame is None or sensor.stream_epoch is None:raise ValueError('image coordinate frame/epoch required')
        ts=frame.timestamp_ms
        if type(ts) is not int or not 0<=ts<=2**53-1:raise ValueError('bounded capture timestamp required')
        data=frame.data
        if (type(data) is not np.ndarray or data.dtype!=np.uint8 or data.ndim!=3 or data.shape[2]!=3
                or min(data.shape[:2])<16 or max(data.shape[:2])>4096 or data.nbytes>16*1024*1024):
            raise ValueError('bounded plain original BGR required')
        owned=data.copy(order='C');gray,size=self.motion.prepare(owned)
        digest=hashlib.sha256(b'gamst.compensated.trajectory.uint8.HWC.BGR.v1\0'+struct.pack('>III',*owned.shape)+owned.tobytes()).hexdigest()
        seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.coordinate_frame,
               sensor.calibration_id,sensor.channels,owned.shape,sensor.capture_sequence is not None)
        reason='initial' if self.last is None else None
        if self.owner!=owner:
            reason='source_or_geometry_change' if self.owner is not None else 'initial';self._clear()
        elif self.last is not None:
            if ts<=self.last[0] or seq<=self.last[1]:raise ValueError('capture time/sequence regressed')
            if seq!=self.last[1]+1 or ts-self.last[0]>self.config['max_gap_ms']:
                reason='capture_gap';self._clear()
        batch=self.detector._predict_rows(replace(frame,data=owned))
        rows,count,_,_,_=batch;complete=count<=8
        current_boxes=[row['bbox'] for row in rows]
        transform=None;diagnostics={};candidate=None
        if not complete:reason='incomplete_detection_snapshot'
        elif self.previous is not None:
            if self.links>=60 or ts-self.anchor['timestamp_ms']>2000:reason='reference_horizon'
            else:
                fit=self.motion.estimate(self.previous,gray,size,self.boxes,current_boxes)
                diagnostics=fit.diagnostics
                if fit.previous_to_current is None:reason=diagnostics['reason']
                else:
                    transform=np.asarray(fit.previous_to_current)
                    candidate=self.to_anchor@np.linalg.inv(transform)
                    if self.motion.geometry(candidate,size,cumulative=True) is None:reason='reference_transform_limit'
                    else:
                        for box in current_boxes:
                            point=candidate@np.array([(box['x']+box['width']/2)*size[0],(box['y']+box['height']/2)*size[1],1.])
                            normalized=point[:2]/np.array(size)
                            if not np.isfinite(normalized).all() or (normalized < -4).any() or (normalized > 5).any():
                                reason='reference_projection_limit';break
        valid=complete and transform is not None and reason is None
        projected_boxes=self._project_boxes(transform,size) if valid else {}
        if not valid:
            self.detector.reset();self.paths={};self.mappings=[];self.links=0
            self.to_anchor=np.eye(3)
            self.anchor=dict(id=uuid.uuid4().hex,timestamp_ms=ts,capture_sequence=seq,evidence_sha256=digest)
        else:self.to_anchor=candidate;self.links+=1
        detected=self.detector._associate_rows(*batch,association_boxes=projected_boxes,visible_only=True)
        mapping=dict(capture_sequence=seq,timestamp_ms=ts,evidence_sha256=digest,
                     current_to_anchor_pixels=self.to_anchor.tolist())
        mappings=(self.mappings+[mapping])[-8:]
        pending={};objects=[]
        for item in detected['objects'] if complete else []:
            key=item['track_id'];box=item['location']['bbox']
            raw=[box['x']+box['width']/2,box['y']+box['height']/2]
            point=self.to_anchor@np.array([raw[0]*size[0],raw[1]*size[1],1.])
            projection=[float(point[0]/size[0]),float(point[1]/size[1])]
            sample=dict(timestamp_ms=ts,capture_sequence=seq,raw_centroid=raw,anchor_projection=projection,
                evidence_sha256=digest,
                kind='observed_with_estimated_projection')
            prior=self.paths.get(key) if valid else None
            segment=prior['segment_id'] if prior else uuid.uuid4().hex
            history=(prior['observations']+[sample])[-8:] if prior else [sample]
            velocity=interval=None
            if prior:
                last=prior['observations'][-1];dt=(ts-last['timestamp_ms'])/1000
                velocity=[(a-b)/dt for a,b in zip(projection,last['anchor_projection'])]
                interval=[last['timestamp_ms'],ts]
            pending[key]=dict(segment_id=segment,observations=history)
            if valid:objects.append(dict(track_id=key,segment_id=segment,anchor_id=self.anchor['id'],
                observations=history,velocity_reference_image_s=velocity,velocity_interval_ms=interval,
                identity_verified=False,visibility='observed',predictions=[]))
        geometry=dict(status='estimated' if valid else 'warmup' if complete else 'unavailable',reason=reason,
            timestamp_ms=ts,capture_sequence=seq,source_id=sensor.source_id,sensor_id=sensor.sensor_id,
            stream_epoch=sensor.stream_epoch,coordinate_frame=sensor.coordinate_frame,calibration_id=sensor.calibration_id,
            image_size=list(size),input_sha256=digest,evidence_verified=True,evidence_storage='caller_owned',
            evidence_hash_format='gamst.compensated.trajectory.uint8.HWC.BGR.v1',complete_snapshot=complete,
            detection_summary=detected['summary'],detector_version=self.detector.version,
            camera_motion_compensated=valid,camera_motion_verified=False,metric_geometry=False,
            reference_units='anchor_image_width_height',anchor=self.anchor,composition_links=self.links,
            current_to_anchor_pixels=self.to_anchor.tolist() if valid else None,
            projection_records=mappings if valid else [],
            projection_record_join='same_anchor_capture_sequence_timestamp_evidence_sha256',
            previous_to_current_pixels=transform.tolist() if valid else None,
            previous_evidence_sha256=self.last[2] if valid else None,
            previous_capture_timestamp_ms=self.last[0] if valid else None,
            previous_capture_sequence=self.last[1] if valid else None,
            sequence_kind='capture_sequence' if sensor.capture_sequence is not None else 'sequence',
            diagnostics=diagnostics,operating_assumption=self.config['operating_assumption'],
            background_roi=self.config['background_roi'],events_status='unavailable_reference_rules_not_bound',
            opencv_version=self.motion.cv.__version__,config_version=self.spec.version)
        result=self.spec.validate(dict(detections=detected['objects'],objects=objects,geometry=geometry))
        if complete:
            self.paths=pending;self.mappings=mappings;self.previous=gray;self.boxes=current_boxes
            self.owner=owner;self.last=(ts,seq,digest)
        else:self._clear()
        return result
