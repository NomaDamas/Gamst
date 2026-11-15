"""Bounded MediaPipe body geometry context; no identity or mental-state inference."""
import hashlib
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField
from .pose import MediaPipePosePlugin


class MediaPipeBodyContext:
    def __init__(self, detector, image_api, *, sha256, max_poses=4):
        self.spec = self.capability(sha256, max_poses=max_poses)
        if not callable(getattr(detector, 'detect', None)) or not callable(getattr(detector, 'close', None)):
            raise ValueError('explicit MediaPipe image-mode detector required')
        self.detector, self.image_api = detector, image_api
        self.max_poses, self.closed = max_poses, False

    @staticmethod
    def capability(sha256, *, max_poses=4):
        if type(sha256) is not str or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('lowercase task model SHA256 required')
        if type(max_poses) is not int or not 1 <= max_poses <= 4:
            raise ValueError('max_poses must be 1..4')
        return CapabilitySpec('vision.mediapipe.body_context', 'vision.body_geometry', 'video',
            'mediapipe.pose.image', '0.10.35.sha256.'+sha256+'.p'+str(max_poses),
            (OutputField('poses','array'),OutputField('summary','object')), 'frame',
            freshness_ms=1000, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera',), license='code:Apache-2.0;weights:unspecified')

    @classmethod
    def from_local(cls, path, sha256, *, max_poses=4):
        cls.capability(sha256,max_poses=max_poses)
        path=Path(path).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError('absolute existing local pose task path required')
        with path.open('rb') as source:
            payload=source.read(64*1024*1024+1)
        if not 0 < len(payload) <= 64*1024*1024 or hashlib.sha256(payload).hexdigest()!=sha256:
            raise ValueError('pose task must match SHA256 and be at most 64 MiB')
        try:
            import importlib.metadata
            if importlib.metadata.version('mediapipe')!='0.10.35':
                raise RuntimeError('select a runtime with mediapipe==0.10.35')
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install mediapipe==0.10.35 in the explicitly selected runtime') from exc
        detector=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=payload,delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,num_poses=max_poses,output_segmentation_masks=False))
        try:
            return cls(detector,mp,sha256=sha256,max_poses=max_poses)
        except BaseException:
            detector.close()
            raise

    @staticmethod
    def _points(raw):
        if not isinstance(raw,(list,tuple)) or len(raw)!=33:
            raise ValueError('pose detector must return 33 points per pose')
        points=[]
        for point in raw:
            values={}
            for key in ('x','y','z','visibility','presence'):
                value=getattr(point,key,None)
                if value is None and key in ('visibility','presence'): value=0.
                if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>100:
                    raise ValueError('invalid pose coordinate or score')
                if key in ('visibility','presence') and not 0<=value<=1:
                    raise ValueError('pose score outside 0..1')
                values[key]=float(value)
            points.append(values)
        return points

    @staticmethod
    def _angle(points, ids, width, height):
        selected=[points[i] for i in ids]
        if any(min(p['visibility'],p['presence'])<.5 or not 0<=p['x']<=1 or not 0<=p['y']<=1 for p in selected):
            return None
        a,b,c=selected
        u=((a['x']-b['x'])*width,(a['y']-b['y'])*height)
        v=((c['x']-b['x'])*width,(c['y']-b['y'])*height)
        norm=math.hypot(*u)*math.hypot(*v)
        if norm<1e-9: return None
        cosine=max(-1.,min(1.,sum(x*y for x,y in zip(u,v))/norm))
        return round(math.degrees(math.acos(cosine)),3)

    def process(self, frame):
        if self.closed: raise RuntimeError('body provider is closed')
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
        raw=result.pose_landmarks
        if not isinstance(raw,(list,tuple)) or len(raw)>self.max_poses:
            raise ValueError('pose detector exceeded selected output bound')
        poses=[]
        for index,body in enumerate(raw):
            points=self._points(body)
            geometry=MediaPipePosePlugin._gestures(points)
            # Off-image joints are not enough evidence for projected comparisons.
            for side,shoulder,wrist in (('left',11,15),('right',12,16)):
                if any(not 0<=points[i]['x']<=1 or not 0<=points[i]['y']<=1 for i in (shoulder,wrist)):
                    geometry[side+'_wrist_above_shoulder']=None
            for name,ids in (('left_elbow',(11,13,15)),('right_elbow',(12,14,16)),
                             ('left_knee',(23,25,27)),('right_knee',(24,26,28))):
                geometry[name+'_projected_angle_deg']=self._angle(points,ids,width,height)
            poses.append(dict(frame_pose_index=index,bbox=MediaPipePosePlugin._bbox(points),
                geometry=geometry,visible_joint_count=sum(min(p['visibility'],p['presence'])>=.5 for p in points)))
        return self.spec.validate(dict(poses=poses,summary=dict(count=len(poses),max_poses=self.max_poses,
            coordinate_frame='normalized_image_xy',angle_space='pixel_plane',
            image_width=width,image_height=height,mirror_status='unspecified',
            identity='unavailable',temporal_tracking=False,metric_3d=False)))

    def close(self):
        if not self.closed:
            self.closed=True
            self.detector.close()
