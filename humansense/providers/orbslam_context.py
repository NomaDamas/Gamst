"""Owned monocular camera-pose context; scale and map continuity unverified."""
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame
from .orbslam_native import ORBSLAMNative, SOURCE_REVISION


def configuration(*,library_path,vocabulary_path,settings_path,library_sha256,
                  vocabulary_sha256,settings_sha256,calibration_id,max_frames=300,
                  max_gap_ms=1000,sequence_step=1):
    values=locals().copy()
    for key in ('library_path','vocabulary_path','settings_path'):
        if type(values[key]) is not str or len(values[key])>4096 or not Path(values[key]).is_absolute():
            raise ValueError('explicit absolute local asset paths required')
    for key in ('library_sha256','vocabulary_sha256','settings_sha256'):
        if type(values[key]) is not str or not re.fullmatch('[0-9a-f]{64}',values[key]):
            raise ValueError('explicit lowercase asset SHA256 required')
    identifier(calibration_id)
    for value,limit in ((max_frames,3000),(max_gap_ms,10000),(sequence_step,1000)):
        if type(value) is not int or not 1<=value<=limit:raise ValueError('invalid bounded session policy')
    return values


class ORBSLAMContext:
    def __init__(self,native,**config):
        self.config=configuration(**config)
        self.spec=self.capability(**self.config)
        if not isinstance(native,ORBSLAMNative):raise TypeError('concrete ORBSLAMNative required')
        self.native=native;self.lock=threading.RLock();self.closed=False
        self.session_id=uuid.uuid4().hex;self.segment=uuid.uuid4().hex
        self.resets=0;self.map_changes=0;self.owner=self.last=self.origin=None
        self.needs_reset=False

    @staticmethod
    def capability(**config):
        config=configuration(**config)
        digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('vision.orbslam3.monocular','vision.camera_pose','video',
            'orbslam3.monocular','1.'+digest,
            (OutputField('pose','object'),OutputField('provenance','object')),
            'native',freshness_ms=1000,stateful=True,reset_on_gap=False,
            interpretation='estimate',devices=('cpu',),
            sensor_requirements=('calibrated-uint8-bgr-camera','capture-clock','owned-native-worker'),
            license='adapter:Apache-2.0;ORB-SLAM3:GPL-3.0;dependencies:verify-separately')

    @classmethod
    def from_local(cls,**config):
        config=configuration(**config)
        native=ORBSLAMNative(**{k:v for k,v in config.items() if k not in ('calibration_id','max_gap_ms','sequence_step')})
        try:return cls(native,**config)
        except BaseException:native.close();raise

    def reset(self):
        with self.lock:
            if self.closed:raise RuntimeError('ORB context closed')
            self.needs_reset=True
            self.native.reset()
            self.resets+=1;self.segment=uuid.uuid4().hex
            self.owner=self.last=self.origin=None;self.needs_reset=False

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.owner=self.last=self.origin=None
            self.native.close()

    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('ORB context closed')
            try:return self._process(frame)
            except BaseException:
                self.needs_reset=True
                raise

    def _process(self,frame):
        import numpy as np
        import cv2
        if (not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None or
                not isinstance(frame.sensor,SensorMetadata) or type(frame.timestamp_ms) is not int or
                not 0<=frame.timestamp_ms<=2**53-1):
            raise ValueError('video frame with bounded capture timestamp and sensor provenance required')
        identifier(frame.subject_id)
        sensor=frame.sensor
        if sensor.calibration_id!=self.config['calibration_id'] or not sensor.coordinate_frame:
            raise ValueError('matching calibration ID and camera coordinate frame required')
        pixels=frame.data
        if (type(pixels) is not np.ndarray or pixels.dtype!=np.uint8 or pixels.ndim!=3 or
                pixels.shape[2]!=3 or pixels.shape[:2]!=self.native._shape):
            raise ValueError('uint8 BGR must match native calibration dimensions')
        seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,
               sensor.coordinate_frame,sensor.calibration_id,sensor.channels,sensor.capture_sequence is not None)
        if self.owner==owner and self.last and (seq<=self.last[0] or frame.timestamp_ms<=self.last[1]):
            raise ValueError('capture sequence and timestamp must strictly increase')
        gap=(self.last is not None and (self.owner!=owner or
             seq-self.last[0]!=self.config['sequence_step'] or
             frame.timestamp_ms-self.last[1]>self.config['max_gap_ms']))
        if self.needs_reset or gap:self.reset()
        if self.origin is None:self.origin=frame.timestamp_ms
        # Own the BGR snapshot before conversion; never resize calibration geometry.
        gray=cv2.cvtColor(pixels.copy(),cv2.COLOR_BGR2GRAY)
        result=self.native.track(gray,(frame.timestamp_ms-self.origin)/1000)
        if result['map_changed']:self.map_changes+=1
        value=self.spec.validate(dict(pose=dict(tracking_state=result['state'],
            tracked=result['state']==2,world_to_camera=result['world_to_camera'],
            transform='T_cw',units='arbitrary_monocular_scale',metric=False,
            coordinate_continuity_verified=False,map_changed_hint=result['map_changed'],
            object_motion_available=False,quality_validated=False),provenance=dict(
            capture_ms=frame.timestamp_ms,capture_sequence=seq,source_id=sensor.source_id,
            sensor_id=sensor.sensor_id,stream_epoch=sensor.stream_epoch,
            calibration_id=sensor.calibration_id,settings_sha256=self.config['settings_sha256'],
            coordinate_frame=sensor.coordinate_frame,session_id=self.session_id,
            reset_segment=self.segment,reset_count=self.resets,map_change_count=self.map_changes,
            source_revision=SOURCE_REVISION,pixel_conversion='opencv_BGR2GRAY_no_resize',
            input_width=gray.shape[1],input_height=gray.shape[0],
            map_identity='unavailable',continuous_world_motion_available=False)))
        self.owner,self.last=owner,(seq,frame.timestamp_ms)
        return value
