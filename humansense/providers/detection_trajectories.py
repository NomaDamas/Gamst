"""Owned RF-DETR pixel inference and conservative observed trajectory context."""
from dataclasses import replace
import hashlib
import json
import re
import struct
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame
from .detection import RFDETRObjects
from .trajectories import ImageTrajectories


def configuration(config):
    keys = {'path','sha256','device','camera_motion','max_gap_ms','max_samples','event_rules'}
    if (type(config) is not dict or not {'path','sha256'} <= set(config) or set(config)-keys
            or type(config['path']) is not str or not config['path']
            or type(config['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',config['sha256']) is None):
        raise ValueError('explicit local checkpoint path/hash and bounded trajectory options required')
    args = dict(device='cpu',camera_motion='unknown',max_gap_ms=1000,max_samples=8)
    args.update(config)
    RFDETRTrajectories.capability('1.12.1.eager.sha256.'+args['sha256'],
                                 **{k:v for k,v in args.items() if k not in ('path','sha256')})
    return args


class RFDETRTrajectories:
    """One serialized pixel stream; owns detector and trajectory lifecycle."""
    @staticmethod
    def capability(version, *, device='cpu', camera_motion='unknown', max_gap_ms=1000, max_samples=8, event_rules=None):
        identifier(version)
        if device not in ('cpu','cuda','mps') or camera_motion not in ('unknown','stationary'):
            raise ValueError('declare supported device and unknown/stationary camera motion')
        paths = ImageTrajectories(max_gap_ms=max_gap_ms,max_samples=max_samples,event_rules=event_rules)
        options = [version,device,camera_motion,max_gap_ms,max_samples]
        if paths._rules is not None: options.append(paths._rules.describe())
        digest = hashlib.sha256(json.dumps(options).encode()).hexdigest()
        return CapabilitySpec('vision.rfdetr.trajectories','vision.trajectories','video',
            'gamst.rfdetr.trajectories','1.'+digest,
            (OutputField('objects','array'),OutputField('geometry','object'),OutputField('detections','array')) +
            (() if event_rules is None else (OutputField('events','array'),OutputField('event_summary','object'))),
            'frame',stateful=True,reset_on_gap=True,freshness_ms=max_gap_ms,interpretation='estimate',
            devices=('cpu','cuda','mps'),sensor_requirements=('uint8-bgr-camera','declared-camera-motion'),license='Apache-2.0')

    def __init__(self, predictor, *, version, device='cpu', camera_motion='unknown', max_gap_ms=1000, max_samples=8, event_rules=None):
        self.spec = self.capability(version,device=device,camera_motion=camera_motion,max_gap_ms=max_gap_ms,max_samples=max_samples,event_rules=event_rules)
        self._detector = RFDETRObjects(predictor,version=version,max_objects=8,max_age_ms=max_gap_ms)
        self._paths = ImageTrajectories(max_gap_ms=max_gap_ms,max_samples=max_samples,event_rules=event_rules)
        self._motion,self._gap = camera_motion,max_gap_ms
        self._lock = threading.RLock()
        self._closed = False
        self._owner = self._last = None

    @classmethod
    def from_local(cls, **config):
        args = configuration(config)
        detector = RFDETRObjects.from_local(args['path'],sha256=args['sha256'],device=args['device'],
                                           max_objects=8,max_age_ms=args['max_gap_ms'])
        return cls(detector.predictor,version=detector.version,
                   **{k:v for k,v in args.items() if k not in ('path','sha256')})

    def reset(self):
        with self._lock:
            self._detector.reset(); self._paths.reset()
            self._owner = self._last = None

    def close(self):
        with self._lock:
            if not self._closed:
                self.reset(); self._detector.close(); self._paths.close()
                self._closed = True

    def process(self, frame):
        with self._lock:
            if self._closed:
                raise RuntimeError('RF-DETR trajectory provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self.reset()
                raise

    def _process(self, frame):
        import numpy as np
        if (not isinstance(frame,InputFrame) or frame.modality != 'video' or frame.sample_rate is not None
                or not isinstance(frame.sensor,SensorMetadata)):
            raise ValueError('video frame and sensor metadata required')
        identifier(frame.subject_id)
        sensor = frame.sensor
        if sensor.coordinate_frame is None or sensor.stream_epoch is None:
            raise ValueError('explicit image frame and stream epoch required')
        data,ts = frame.data,frame.timestamp_ms
        if (type(data) is not np.ndarray or data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3
                or min(data.shape[:2]) < 16 or max(data.shape[:2]) > 4096 or data.nbytes > 16*1024*1024):
            raise ValueError('bounded plain uint8 HWC BGR image required')
        if type(ts) is not int or not 0 <= ts <= 2**53-1:
            raise ValueError('integer capture timestamp required')
        owned = data.copy(order='C')
        digest = hashlib.sha256(b'gamst.rfdetr.trajectory.uint8.HWC.BGR.v1\0'
                                +struct.pack('>III',*owned.shape)+owned.tobytes()).hexdigest()
        owner = (frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.coordinate_frame,
                 sensor.calibration_id,sensor.channels,owned.shape,sensor.capture_sequence is not None)
        seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        same = owner == self._owner
        if same and self._last and (ts <= self._last[0] or seq <= self._last[1]):
            raise ValueError('capture timestamp and sequence must strictly increase')
        if (not same or self._last and (seq != self._last[1]+1 or ts-self._last[0] > self._gap)
                or self._motion != 'stationary'):
            self.reset()
        detected = self._detector.process(replace(frame,data=owned))
        complete = detected['summary']['complete_snapshot']
        rows = []
        for item in detected['objects']:
            box = item['location']['bbox']
            rows.append(dict(track_id=item['track_id'],centroid=[box['x']+box['width']/2,box['y']+box['height']/2],
                             association_confidence=None,evidence_sha256=digest))
        structured = dict(units='normalized_image',image_size=[owned.shape[1],owned.shape[0]],
                          tracker_id='rfdetr.class_iou',tracker_version=self.spec.version,
                          camera_motion=self._motion,objects=rows if complete else [])
        if not complete:
            self._paths.reset()
        paths = self._paths.process(replace(frame,data=structured))
        paths['geometry'].update(trajectory_status='observed' if complete else 'incomplete_detection_snapshot',
            detection_summary=detected['summary'],detector_version=self._detector.version,
            input_sha256=digest,evidence_verified=True,evidence_hash_format='gamst.rfdetr.trajectory.uint8.HWC.BGR.v1',
            evidence_storage='caller_owned',source_id=sensor.source_id,sensor_id=sensor.sensor_id,
            calibration_id=sensor.calibration_id,capture_sequence=seq,
            sequence_kind='capture_sequence' if sensor.capture_sequence is not None else 'sequence')
        # The empty fallback is not a claim of complete scene coverage.
        paths['geometry']['complete_snapshot'] = complete
        if 'event_summary' in paths:
            paths['event_summary']['input_snapshot_complete'] = complete
        result = self.spec.validate(dict(paths,detections=detected['objects']))
        if not complete:
            self.reset()
        self._owner,self._last = owner,(ts,seq)
        return result
