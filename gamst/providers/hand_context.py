"""Bounded MediaPipe hand geometry context; no identity or mental-state inference."""
import hashlib
import math
from pathlib import Path
import re

from ..capabilities import CapabilitySpec, OutputField


class MediaPipeHandContext:
    def __init__(self, detector, image_api, *, sha256, max_hands=4):
        self.spec = self.capability(sha256, max_hands=max_hands)
        if not callable(getattr(detector, 'detect', None)) or not callable(getattr(detector, 'close', None)):
            raise ValueError('explicit MediaPipe image-mode detector required')
        self.detector, self.image_api = detector, image_api
        self.max_hands, self.closed = max_hands, False

    @staticmethod
    def capability(sha256, *, max_hands=4):
        if type(sha256) is not str or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('lowercase task model SHA256 required')
        if type(max_hands) is not int or not 1 <= max_hands <= 4:
            raise ValueError('max_hands must be 1..4')
        return CapabilitySpec('vision.mediapipe.hand_context', 'vision.hand_geometry', 'video',
            'mediapipe.hand.image', '0.10.35.sha256.'+sha256+'.h'+str(max_hands),
            (OutputField('hands','array'),OutputField('summary','object')), 'frame',
            freshness_ms=1000, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera',), license='code:Apache-2.0;weights:unspecified')

    @classmethod
    def from_local(cls, path, sha256, *, max_hands=4):
        cls.capability(sha256,max_hands=max_hands)
        path=Path(path).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError('absolute existing local hand task path required')
        with path.open('rb') as source:
            payload=source.read(64*1024*1024+1)
        if not 0 < len(payload) <= 64*1024*1024 or hashlib.sha256(payload).hexdigest()!=sha256:
            raise ValueError('hand task must match SHA256 and be at most 64 MiB')
        try:
            import importlib.metadata
            if importlib.metadata.version('mediapipe')!='0.10.35':
                raise RuntimeError('select a runtime with mediapipe==0.10.35')
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install mediapipe==0.10.35 in the explicitly selected runtime') from exc
        detector=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=payload,delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,num_hands=max_hands))
        try:
            return cls(detector,mp,sha256=sha256,max_hands=max_hands)
        except BaseException:
            detector.close()
            raise

    @staticmethod
    def _points(raw):
        if not isinstance(raw,(list,tuple)) or len(raw)!=21:
            raise ValueError('hand detector must return 21 points per hand')
        points=[]
        for point in raw:
            values={key:getattr(point,key,None) for key in ('x','y','z')}
            if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>100 for v in values.values()):
                raise ValueError('invalid hand coordinate')
            points.append({key:float(value) for key,value in values.items()})
        return points

    @staticmethod
    def _in_image(point):
        return 0<=point['x']<=1 and 0<=point['y']<=1

    @classmethod
    def _angle(cls, points, ids, width, height):
        selected=[points[i] for i in ids]
        if not all(cls._in_image(p) for p in selected): return None
        a,b,c=selected
        u=((a['x']-b['x'])*width,(a['y']-b['y'])*height)
        v=((c['x']-b['x'])*width,(c['y']-b['y'])*height)
        norm=math.hypot(*u)*math.hypot(*v)
        if norm<1e-9: return None
        return round(math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(u,v))/norm)))),3)

    @staticmethod
    def _handedness(categories):
        if not isinstance(categories,(list,tuple)) or len(categories)>2:
            raise ValueError('handedness must have at most two candidates')
        output=[]; seen=set()
        for category in categories:
            name,score=category.category_name,category.score
            if name not in ('Left','Right') or name in seen or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('invalid handedness candidate')
            seen.add(name); output.append(dict(label=name,score=float(score)))
        return output

    def process(self, frame):
        if self.closed: raise RuntimeError('hand provider is closed')
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
        raw=result.hand_landmarks
        classifications=result.handedness
        if (not isinstance(raw,(list,tuple)) or len(raw)>self.max_hands or
                not isinstance(classifications,(list,tuple)) or len(classifications)!=len(raw)):
            raise ValueError('hand result count exceeds bound or handedness association is malformed')
        hands=[]
        for index,hand in enumerate(raw):
            points=self._points(hand)
            geometry={}
            for name,ids in (('thumb_ip',(2,3,4)),('index_pip',(5,6,7)),('middle_pip',(9,10,11)),
                             ('ring_pip',(13,14,15)),('pinky_pip',(17,18,19))):
                geometry[name+'_projected_angle_deg']=self._angle(points,ids,width,height)
            ratio=None
            if all(self._in_image(points[i]) for i in (4,8,5,17)):
                def distance(a,b):
                    return math.hypot((points[a]['x']-points[b]['x'])*width,
                                      (points[a]['y']-points[b]['y'])*height)
                palm=distance(5,17)
                # Near-zero projected palm widths make a ratio unstable; abstain.
                if palm>=1.: ratio=round(distance(4,8)/palm,6)
            geometry['thumb_index_distance_over_palm_width']=ratio
            visible=[p for p in points if self._in_image(p)]
            bbox=None
            if len(visible)>=3:
                x0,y0=min(p['x'] for p in visible),min(p['y'] for p in visible)
                x1,y1=max(p['x'] for p in visible),max(p['y'] for p in visible)
                if x1>x0 and y1>y0: bbox=dict(x=x0,y=y0,width=x1-x0,height=y1-y0)
            tip=points[8]
            hands.append(dict(frame_hand_index=index,handedness=self._handedness(classifications[index]),
                bbox=bbox,index_tip_xy={key:tip[key] for key in ('x','y')} if self._in_image(tip) else None,
                geometry=geometry,joint_confidence='unavailable'))
        return self.spec.validate(dict(hands=hands,summary=dict(count=len(hands),max_hands=self.max_hands,
            coordinate_frame='normalized_image_xy',angle_space='pixel_plane',
            image_width=width,image_height=height,mirror_status='unspecified',
            identity='unavailable',temporal_tracking=False,metric_3d=False,
            gesture_classification='unavailable')))

    def close(self):
        if not self.closed:
            self.closed=True
            self.detector.close()
