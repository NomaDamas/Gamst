"""Explicit camera/model geometry + native face landmarks to a pose estimate."""
from dataclasses import replace
import hashlib
import json
import math

from ..capabilities import SensorMetadata, bounded_json, identifier
from .face_context import MediaPipeFaceContext
from .landmarker_video import MediaPipeVideoContext


def _number(value, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('finite bounded geometry required')
    return float(value)


def configuration(config):
    c = bounded_json(config, max_bytes=16384)
    required = {'path', 'sha256', 'camera', 'sensor', 'model_frame', 'model_points', 'max_reprojection_error_px'}
    if type(c) is not dict or not required <= set(c) or set(c)-required-{'max_faces','max_gap_ms'}:
        raise ValueError('explicit local face model, camera, sensor and 3D correspondence geometry required')
    if type(c['path']) is not str or not 1 <= len(c['path']) <= 4096:
        raise ValueError('local face task path required')
    c.setdefault('max_faces', 1); c.setdefault('max_gap_ms', 1000)
    MediaPipeVideoContext.capability('face', c['sha256'], max_entities=c['max_faces'], max_gap_ms=c['max_gap_ms'])
    camera = c['camera']
    if type(camera) is not dict or set(camera) != {'width','height','fx','fy','cx','cy','distortion','mirrored'}:
        raise ValueError('explicit calibrated image dimensions, pinhole intrinsics and five distortion coefficients required')
    if camera['mirrored'] is not False:
        raise ValueError('explicit unmirrored calibrated pixels required')
    for key,limit in (('width',3840),('height',2160)):
        if type(camera[key]) is not int or not 32 <= camera[key] <= limit:
            raise ValueError('bounded calibrated image geometry required')
    if camera['width']*camera['height'] > 1920*1080:
        raise ValueError('calibrated image exceeds pixel bound')
    for key in ('fx','fy'): camera[key] = _number(camera[key], 1, 100000)
    camera['cx'] = _number(camera['cx'], 0, camera['width']-1)
    camera['cy'] = _number(camera['cy'], 0, camera['height']-1)
    if type(camera['distortion']) is not list or len(camera['distortion']) != 5:
        raise ValueError('OpenCV k1,k2,p1,p2,k3 distortion coefficients required')
    camera['distortion'] = [_number(v,-2,2) for v in camera['distortion']]
    sensor = c['sensor']
    if type(sensor) is not dict or set(sensor) != {'source_id','sensor_id','calibration_id','coordinate_frame'}:
        raise ValueError('explicit camera source/sensor/calibration/optical-frame binding required')
    for value in sensor.values(): identifier(value)
    identifier(c['model_frame'])
    if c['model_frame'] == sensor['coordinate_frame']:
        raise ValueError('distinct model and camera coordinate frames required')
    rows = c['model_points']
    if type(rows) is not list or not 6 <= len(rows) <= 32:
        raise ValueError('six to 32 noncoplanar landmark correspondences required')
    indices = set(); points = []
    for row in rows:
        if type(row) is not dict or set(row) != {'landmark_index','xyz_m'}:
            raise ValueError('each correspondence needs landmark_index and explicit xyz_m')
        index = row['landmark_index']
        if type(index) is not int or not 0 <= index < 478 or index in indices:
            raise ValueError('unique MediaPipe face landmark indices required')
        indices.add(index)
        if type(row['xyz_m']) is not list or len(row['xyz_m']) != 3:
            raise ValueError('explicit model coordinates in metres required')
        row['xyz_m'] = [_number(v,-1,1) for v in row['xyz_m']]
        points.append(row['xyz_m'])
    if len({tuple(p) for p in points}) != len(points):
        raise ValueError('duplicate model points cannot constrain orientation')
    # Pure metadata conditioning check; discovery imports no numerical runtime.
    vectors = [[x-y for x,y in zip(p,points[0])] for p in points[1:]]
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    a = max(vectors,key=lambda v:dot(v,v)); scale = math.sqrt(dot(a,a))
    cross = lambda a,b: [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    normal = max((cross(a,b) for b in vectors),key=lambda v:dot(v,v))
    if scale < .0001 or max(abs(dot(normal,v)) for v in vectors) <= scale**3*1e-5:
        raise ValueError('nondegenerate noncoplanar model points required')
    c['max_reprojection_error_px'] = _number(c['max_reprojection_error_px'], .01, 20)
    return c


class HeadOrientationContext:
    @staticmethod
    def capability(**config):
        c = configuration(config)
        sha = hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        base = MediaPipeVideoContext.capability('face',c['sha256'],max_entities=c['max_faces'],max_gap_ms=c['max_gap_ms'])
        return replace(base,id='vision.mediapipe.head_orientation',output_name='vision.head_orientation',
            provider='mediapipe.head_orientation.video',version='1.'+sha,
            sensor_requirements=('uint8-bgr-camera','declared-camera-intrinsics','explicit-3d-face-correspondences'))

    def __init__(self, video, **config):
        self.config = configuration(config); self.spec = self.capability(**self.config)
        if not isinstance(video,MediaPipeVideoContext) or video.kind != 'face':
            raise TypeError('owned native face VIDEO context required')
        expected = MediaPipeVideoContext.capability('face',self.config['sha256'],
            max_entities=self.config['max_faces'],max_gap_ms=self.config['max_gap_ms'])
        if video.spec != expected:
            raise ValueError('native face configuration differs from calibrated profile')
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('install the vision extra in the explicitly selected runtime') from exc
        self.cv, self.np, self.video = cv2, np, video
        self.result = None
        camera = self.config['camera']
        self.matrix = np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]],dtype=np.float64)
        self.distortion = np.array(camera['distortion'],dtype=np.float64)
        self.points = np.array([p['xyz_m'] for p in self.config['model_points']],dtype=np.float64)
        self.indices = [p['landmark_index'] for p in self.config['model_points']]
        self.video.formatter.detector.detect = self._detect

    @classmethod
    def from_local(cls, **config):
        c = configuration(config)
        video = MediaPipeVideoContext.from_local('face',c['path'],c['sha256'],
            max_entities=c['max_faces'],max_gap_ms=c['max_gap_ms'])
        try: return cls(video,**c)
        except BaseException: video.close(); raise

    def _detect(self,image):
        self.result = self.video._detect(image)
        return self.result

    def reset(self):
        with self.video.lock:
            self.result = None; self.video.reset()

    def close(self):
        with self.video.lock:
            self.result = None; self.video.close()

    def _orientation(self,raw):
        np,cv = self.np,self.cv
        points = MediaPipeFaceContext._points(raw)
        selected = [points[i] for i in self.indices]
        missing = dict(status='unavailable',reason='landmarks_outside_image',rotation_model_to_camera=None,
            rodrigues_model_to_camera_rad=None,reprojection_rmse_px=None)
        if not all(MediaPipeFaceContext._in_image(p) for p in selected): return missing
        camera = self.config['camera']
        image = np.array([[p['x']*camera['width'],p['y']*camera['height']] for p in selected],dtype=np.float64)
        singular = np.linalg.svd(image-image.mean(axis=0),compute_uv=False)
        if singular[-1] < 1 or singular[-1]/singular[0] < .01:
            return dict(missing,reason='degenerate_image_correspondences')
        try:
            ok,rvec,tvec = cv.solvePnP(self.points,image,self.matrix,self.distortion,flags=cv.SOLVEPNP_ITERATIVE)
            if not ok: return dict(missing,reason='solver_failed')
            if (rvec.shape != (3,1) or tvec.shape != (3,1) or
                    not np.isfinite(rvec).all() or not np.isfinite(tvec).all()):
                return dict(missing,reason='invalid_pose')
            rotation,_ = cv.Rodrigues(rvec)
            in_camera = (rotation@self.points.T+tvec).T
            if (rotation.shape != (3,3) or not np.isfinite(rotation).all() or
                    not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or
                    abs(float(np.linalg.det(rotation))-1)>1e-6 or np.any(in_camera[:,2] <= 1e-6)):
                return dict(missing,reason='invalid_pose_or_cheirality')
            projected,_ = cv.projectPoints(self.points,rvec,tvec,self.matrix,self.distortion)
            rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-image)**2,axis=1))))
            if not math.isfinite(rmse) or rmse > self.config['max_reprojection_error_px']:
                return dict(missing,reason='reprojection_error')
            return dict(status='estimated',reason=None,rotation_model_to_camera=rotation.tolist(),
                rodrigues_model_to_camera_rad=rvec.reshape(3).tolist(),reprojection_rmse_px=rmse)
        except cv.error:
            return dict(missing,reason='solver_failed')

    def process(self,frame):
        with self.video.lock:
            if self.video.closed: raise RuntimeError('head orientation provider closed')
            try:
                sensor = frame.sensor; camera = self.config['camera']
                if (not isinstance(sensor,SensorMetadata) or
                        any(getattr(sensor,k) != value for k,value in self.config['sensor'].items())):
                    raise ValueError('capture source/calibration differs from configured camera')
                if getattr(frame.data,'shape',None) != (camera['height'],camera['width'],3):
                    raise ValueError('capture dimensions differ from camera calibration')
                result = self.video.process(frame)
                for face,raw in zip(result['faces'],self.result.face_landmarks):
                    face['orientation'] = self._orientation(raw)
                result['summary'].update(model_frame=self.config['model_frame'],
                    camera_frame=self.config['sensor']['coordinate_frame'],calibration_id=self.config['sensor']['calibration_id'],
                    camera_axes='x_right_y_down_z_forward',transform='X_camera=R_model_to_camera*X_model+t',
                    units='rotation_matrix_and_rodrigues_radians',translation_published=False,
                    mirror_status='caller_declared_unmirrored',mirrored_input_supported=False,
                    anatomical_axes='caller_defined_not_inferred',
                    calibration_verified=False,model_geometry_verified=False,pose_accuracy_verified=False,
                    opencv_version=self.cv.__version__,pose_configuration_sha256=self.spec.version[2:])
                return self.spec.validate(result)
            except BaseException:
                self.video.reset(); raise
            finally: self.result = None
