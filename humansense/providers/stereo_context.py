"""Owned stereo + RF-DETR tracks to bounded metric observations."""
from dataclasses import replace
import hashlib
import json
import math
import re
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from ..stereo_packet import unpack_stereo
from .kinematics import MetricKinematics
from .stereo_metric import StereoGeometry
from .detection_trajectories import RFDETRTrajectories


def configuration(config):
    c = bounded_json(config, max_bytes=16384)
    required = {'path','sha256','calibration','left_sensor','right_sensor','clock_id',
                'alignment_id','rectified_frame','max_skew_ms','camera_motion'}
    if type(c) is not dict or not required <= set(c) or set(c)-required-{'disparities','block_size','max_gap_ms'}:
        raise ValueError('explicit stereo model, calibration, sensors and clock configuration required')
    if type(c['path']) is not str or not c['path'] or len(c['path']) > 4096:
        raise ValueError('explicit detector checkpoint path required')
    if type(c['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',c['sha256']) is None:
        raise ValueError('detector checkpoint SHA256 required')
    if c['camera_motion'] != 'stationary':
        raise ValueError('this profile requires a declared fixed stereo rig')
    for key in ('clock_id','alignment_id','rectified_frame'):
        identifier(c[key])
    for key in ('left_sensor','right_sensor'):
        v = c[key]
        if type(v) is not dict or set(v) != {'source_id','sensor_id','coordinate_frame'}:
            raise ValueError('exact source/sensor/optical-frame binding required')
        for value in v.values(): identifier(value)
    if tuple(c['left_sensor'][k] for k in ('source_id','sensor_id')) == tuple(c['right_sensor'][k] for k in ('source_id','sensor_id')):
        raise ValueError('distinct stereo sensors required')
    if len({c['rectified_frame'],c['left_sensor']['coordinate_frame'],c['right_sensor']['coordinate_frame']}) != 3:
        raise ValueError('distinct original-left, original-right and rectified coordinate names required')
    if type(c['max_skew_ms']) is not int or not 0 <= c['max_skew_ms'] <= 100:
        raise ValueError('explicit 0..100ms clock skew policy required')
    if type(c['calibration']) is not dict or 'calibration_id' not in c['calibration']:
        raise ValueError('explicit stereo calibration required')
    identifier(c['calibration']['calibration_id'])
    c.setdefault('disparities',64); c.setdefault('block_size',5); c.setdefault('max_gap_ms',1000)
    if type(c['max_gap_ms']) is not int or not 1 <= c['max_gap_ms'] <= 10000:
        raise ValueError('bounded stereo gap required')
    # Full geometric validation occurs before selected worker construction. No
    # NumPy/OpenCV/model import or source access is needed for discovery.
    if type(c['disparities']) is not int or c['disparities'] not in range(16,129,16):
        raise ValueError('bounded disparity count required')
    if type(c['block_size']) is not int or c['block_size'] not in (3,5,7,9):
        raise ValueError('bounded matching block required')
    return c


class StereoMetricContext:
    @staticmethod
    def capability(**config):
        c = configuration(config)
        digest = hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return CapabilitySpec('vision.stereo.rfdetr.metric','vision.kinematics','video',
            'gamst.stereo.rfdetr','1.'+digest,
            (OutputField('objects','array'),OutputField('geometry','object'),OutputField('stereo','object')),
            'frame',stateful=True,reset_on_gap=True,freshness_ms=c['max_gap_ms'],
            interpretation='estimate',devices=('cpu',),
            sensor_requirements=('paired-bgr-stereo','fixed-camera','declared-calibration','capture-clock-alignment'),
            license='Apache-2.0')

    def __init__(self, detector, **config):
        self.config = configuration(config)
        self.spec = self.capability(**self.config)
        c = self.config
        self._detector = detector
        self._geometry = StereoGeometry(c['calibration'],disparities=c['disparities'],block_size=c['block_size'])
        self._motion = MetricKinematics(calibration_id=self._geometry.calibration_id,
            input_frame=c['rectified_frame'],output_frame=c['rectified_frame'],max_gap_ms=c['max_gap_ms'])
        self._lock = threading.RLock()
        self._closed = False
        self._owner = self._last = None

    @classmethod
    def from_local(cls, **config):
        c = configuration(config)
        # Validate selected geometry before loading a potentially large model.
        check = StereoGeometry(c['calibration'],disparities=c['disparities'],block_size=c['block_size'])
        check.close()
        detector = RFDETRTrajectories.from_local(path=c['path'],sha256=c['sha256'],device='cpu',
            camera_motion='stationary',max_gap_ms=c['max_gap_ms'],max_samples=2)
        try:
            return cls(detector, **c)
        except BaseException:
            detector.close()
            raise

    def reset(self):
        with self._lock:
            self._detector.reset(); self._motion.reset()
            self._owner = self._last = None

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                try: self._detector.close()
                finally:
                    self._motion.close(); self._geometry.close()
                    self._owner = self._last = None

    def process(self, frame):
        with self._lock:
            if self._closed: raise RuntimeError('stereo context closed')
            try: return self._process(frame)
            except BaseException:
                self.reset()
                raise

    def _process(self, frame):
        import cv2
        import numpy as np
        if (not isinstance(frame,InputFrame) or frame.modality != 'video' or frame.sample_rate is not None
                or not isinstance(frame.sensor,SensorMetadata)):
            raise ValueError('stereo video packet and outer sensor required')
        h, lb, rb = unpack_stereo(frame.data)
        c, sensor = self.config, frame.sensor
        if (h['width'],h['height']) != (self._geometry.width,self._geometry.height):
            raise ValueError('stereo geometry changed from calibrated dimensions')
        if (h['clock_id'],h['alignment_id']) != (c['clock_id'],c['alignment_id']):
            raise ValueError('stereo clock alignment identity mismatch')
        for name in ('left','right'):
            s = h[name]['sensor']
            if (any(s[k] != v for k,v in c[name+'_sensor'].items())
                    or s['calibration_id'] != self._geometry.calibration_id):
                raise ValueError('stereo source differs from configured calibration binding')
        lm, rm = h['left'], h['right']
        ls, rs = lm['sensor'],rm['sensor']
        if any(abs(a-b) > c['max_skew_ms'] for a,b in zip(lm['interval_ms'],rm['interval_ms'])):
            raise ValueError('stereo captures exceed declared time skew')
        seq = lambda s: s['capture_sequence'] if s['capture_sequence'] is not None else s['sequence']
        outerseq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        if (frame.subject_id != lm['subject_id'] or type(frame.timestamp_ms) is not int
                or frame.timestamp_ms != lm['interval_ms'][1] or sensor.stream_epoch is None
                or outerseq != seq(ls) or any(getattr(sensor,k) != ls[k]
                    for k in ('source_id','sensor_id','coordinate_frame','calibration_id'))):
            raise ValueError('outer frame does not identify captured left image')
        owner = (frame.subject_id,sensor.stream_epoch,ls['stream_epoch'],rs['stream_epoch'],
                 ls['capture_sequence'] is not None,rs['capture_sequence'] is not None)
        now = (frame.timestamp_ms,rm['interval_ms'][1],seq(ls),seq(rs))
        if owner == self._owner and self._last:
            if any(a <= b for a,b in zip(now,self._last)):
                raise ValueError('both stereo capture times and sequences must increase')
            if (now[2] != self._last[2]+1 or now[3] != self._last[3]+1
                    or max(now[0]-self._last[0],now[1]-self._last[1]) > c['max_gap_ms']):
                self.reset()
        elif self._owner is not None:
            self.reset()
        shape = (h['height'],h['width'],3)
        left = np.frombuffer(lb,np.uint8).reshape(shape).copy()
        right = np.frombuffer(rb,np.uint8).reshape(shape).copy()
        detected = self._detector.process(replace(frame,data=left))
        points, valid = self._geometry.compute(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),
                                               cv2.cvtColor(right,cv2.COLOR_BGR2GRAY))
        complete = detected['geometry']['complete_snapshot'] is True
        positions, support = [], []
        if complete:
            for item in detected['detections']:
                box = item['location']['bbox']
                # Central half reduces obvious surrounding pixels; it does not
                # prove foreground ownership or identify center-of-mass.
                x0 = max(0,math.floor((box['x']+box['width']*.25)*h['width']))
                x1 = min(h['width'],math.ceil((box['x']+box['width']*.75)*h['width']))
                y0 = max(0,math.floor((box['y']+box['height']*.25)*h['height']))
                y1 = min(h['height'],math.ceil((box['y']+box['height']*.75)*h['height']))
                mask = np.zeros(valid.shape,np.uint8); mask[y0:y1,x0:x1] = 255
                roi = self._geometry.rectified_mask(mask)
                selected = points[roi & valid]
                count, total = len(selected),int(roi.sum())
                row = dict(track_id=item['track_id'],valid_pixels=count,roi_pixels=total,
                           original_left_bbox=dict(box),
                           position_semantics='visible_surface_median',status='insufficient_support')
                if count >= 16 and count >= total*.5:
                    z = selected[:,2]
                    relative_iqr = float((np.percentile(z,75)-np.percentile(z,25))/np.median(z))
                    row['relative_depth_iqr'] = relative_iqr
                    point = np.median(selected,axis=0)
                    if relative_iqr <= .25 and np.max(np.abs(point)) <= 10000:
                        positions.append(dict(track_id=item['track_id'],position_m=point.tolist()))
                        row['status'] = 'estimated'
                    else: row['status'] = 'ambiguous_depth'
                support.append(row)
        else:
            self._motion.reset()
        metric_frame = replace(frame,data=dict(units='m',tracker_id='rfdetr.class_iou.stereo',
            tracker_version=self._detector.spec.version,objects=positions),
            sensor=replace(sensor,coordinate_frame=c['rectified_frame']))
        result = self._motion.process(metric_frame)
        detections = {item['track_id']:item for item in detected['detections']}
        for item in result['objects']:
            observed = detections[item['track_id']]
            item['detection'] = dict(label=observed['label'],**observed['attributes'])
        result['geometry'].update(complete_snapshot=complete and all(s['status']=='estimated' for s in support),
            detection_snapshot_complete=complete,physical_accuracy_verified=False,
            camera_stationarity_verified=False,object_center_of_mass=False)
        result['stereo'] = dict(left=lm,right=rm,clock_id=h['clock_id'],alignment_id=h['alignment_id'],
            max_skew_ms=c['max_skew_ms'],clock_alignment_verified=False,
            calibration_sha256=self._geometry.calibration_sha256,opencv_version=self._geometry.runtime_version,
            detector_version=self._detector.spec.version,disparities=c['disparities'],block_size=c['block_size'],
            detector_input_sha256=detected['geometry']['input_sha256'],
            rectification=dict(self._geometry.provenance(),
                original_left_frame=c['left_sensor']['coordinate_frame'],
                rectified_left_frame=c['rectified_frame']),
            support=support,rectified_coordinate_frame=c['rectified_frame'],
            evidence_storage='caller_owned',depth_units='m',calibration_verified=False)
        result = self.spec.validate(result)
        if not complete: self.reset()
        self._owner,self._last = owner,now
        return result
