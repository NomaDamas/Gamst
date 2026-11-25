"""Bounded MediaPipe face geometry context; no identity or mental-state inference."""
import hashlib
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField


# MediaPipe Face Landmarker coefficient names; not emotion or gaze probabilities.
BLENDSHAPES = frozenset("""_neutral browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight
eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight eyeLookOutLeft eyeLookOutRight
eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight
jawForward jawLeft jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight
mouthFrownLeft mouthFrownRight mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight
mouthPressLeft mouthPressRight mouthPucker mouthRight mouthRollLower mouthRollUpper
mouthShrugLower mouthShrugUpper mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight
mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight""".split())


class MediaPipeFaceContext:
    def __init__(self, detector, image_api, *, sha256, max_faces=4):
        self.spec = self.capability(sha256, max_faces=max_faces)
        if not callable(getattr(detector, 'detect', None)) or not callable(getattr(detector, 'close', None)):
            raise ValueError('explicit MediaPipe image-mode detector required')
        self.detector, self.image_api = detector, image_api
        self.max_faces, self.closed = max_faces, False

    @staticmethod
    def capability(sha256, *, max_faces=4):
        if type(sha256) is not str or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('lowercase task model SHA256 required')
        if type(max_faces) is not int or not 1 <= max_faces <= 4:
            raise ValueError('max_faces must be 1..4')
        return CapabilitySpec('vision.mediapipe.face_context', 'vision.face_movements', 'video',
            'mediapipe.face.image', '0.10.35.sha256.'+sha256 +'.f'+str(max_faces),
            (OutputField('faces','array'),OutputField('summary','object')), 'frame',
            freshness_ms=1000, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera',), license='code:Apache-2.0;weights:unspecified')

    @classmethod
    def from_local(cls, path, sha256, *, max_faces=4):
        cls.capability(sha256,max_faces=max_faces)
        path=Path(path).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError('absolute existing local face task path required')
        with path.open('rb') as source:
            payload=source.read(64*1024*1024+1)
        if not 0 < len(payload) <= 64*1024*1024 or hashlib.sha256(payload).hexdigest()!=sha256:
            raise ValueError('face task must match SHA256 and be at most 64 MiB')
        try:
            import importlib.metadata
            if importlib.metadata.version('mediapipe')!='0.10.35':
                raise RuntimeError('select a runtime with mediapipe==0.10.35')
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install mediapipe==0.10.35 in the explicitly selected runtime') from exc
        detector=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=payload,delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,num_faces=max_faces,
            output_face_blendshapes=True,output_facial_transformation_matrixes=False))
        try:
            return cls(detector,mp,sha256=sha256,max_faces=max_faces)
        except BaseException:
            detector.close()
            raise

    @staticmethod
    def _points(raw):
        if not isinstance(raw,(list,tuple)) or len(raw)!=478:
            raise ValueError('face detector must return 478 points per face')
        points=[]
        for point in raw:
            values={key:getattr(point,key,None) for key in ('x','y','z')}
            if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>100 for v in values.values()):
                raise ValueError('invalid face coordinate')
            points.append({key:float(value) for key,value in values.items()})
        return points

    @staticmethod
    def _in_image(point):
        return 0<=point['x']<=1 and 0<=point['y']<=1

    @classmethod
    def _ratio(cls, points, gap, width, image_width, image_height):
        if not all(cls._in_image(points[i]) for i in (*gap,*width)): return None
        def distance(ids):
            a,b=(points[i] for i in ids)
            return math.hypot((a['x']-b['x'])*image_width,(a['y']-b['y'])*image_height)
        denominator=distance(width)
        return round(distance(gap)/denominator,6) if denominator>=1. else None

    @staticmethod
    def _blendshapes(categories):
        if not isinstance(categories,(list,tuple)) or len(categories)>52:
            raise ValueError('face blendshapes must have at most 52 coefficients')
        output={}
        for category in categories:
            name,score=getattr(category,'category_name',None),getattr(category,'score',None)
            if (type(name) is not str or name not in BLENDSHAPES or name in output or
                    type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1):
                raise ValueError('invalid face blendshape coefficient')
            output[name]=float(score)
        return output

    def process(self, frame):
        if self.closed: raise RuntimeError('face provider is closed')
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('install numpy in the selected vision runtime') from exc
        data=frame.data
        if (frame.modality!='video' or not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or
                data.ndim!=3 or data.shape[2]!=3 or not 1<=data.shape[0]<=2160 or
                not 1<=data.shape[1]<=3840 or data.shape[0]*data.shape[1]>1920*1080):
            raise ValueError('supply a uint8 BGR image, at most 1920*1080 pixels and 3840x2160 dimensions')
        height,width=data.shape[:2]
        image=self.image_api.Image(image_format=self.image_api.ImageFormat.SRGB,data=data[:,:,::-1].copy())
        result=self.detector.detect(image)
        raw=result.face_landmarks
        coefficients=result.face_blendshapes
        if (not isinstance(raw,(list,tuple)) or len(raw)>self.max_faces or
                not isinstance(coefficients,(list,tuple)) or len(coefficients)!=len(raw)):
            raise ValueError('face result count exceeds bound or blendshape association is malformed')
        faces=[]
        for index,face in enumerate(raw):
            points=self._points(face)
            geometry={
                'left_eye_gap_over_width':self._ratio(points,(386,374),(263,362),width,height),
                'right_eye_gap_over_width':self._ratio(points,(159,145),(33,133),width,height),
                'mouth_gap_over_width':self._ratio(points,(13,14),(61,291),width,height),
            }
            in_image=[p for p in points if self._in_image(p)]
            bbox=None
            if len(in_image)>=3:
                x0,y0=min(p['x'] for p in in_image),min(p['y'] for p in in_image)
                x1,y1=max(p['x'] for p in in_image),max(p['y'] for p in in_image)
                if x1>x0 and y1>y0: bbox=dict(x=x0,y=y0,width=x1-x0,height=y1-y0)
            scores=self._blendshapes(coefficients[index])
            faces.append(dict(frame_face_index=index,bbox=bbox,geometry=geometry,
                blendshapes=scores,blendshapes_available=bool(scores),joint_confidence='unavailable'))
        return self.spec.validate(dict(faces=faces,summary=dict(count=len(faces),max_faces=self.max_faces,
            coordinate_frame='normalized_image_xy',geometry_space='pixel_plane',
            image_width=width,image_height=height,mirror_status='unspecified',
            identity='unavailable',temporal_tracking=False,metric_3d=False,
            gaze='unavailable',emotion='unavailable')))

    def close(self):
        if not self.closed:
            self.closed=True
            self.detector.close()
