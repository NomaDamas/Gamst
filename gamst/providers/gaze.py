"""Local L2CS gaze heads with bounded MediaPipe face crops; no target inference."""
import hashlib
import io
import math
from pathlib import Path
import re
import threading

from ..capabilities import CapabilitySpec, OutputField
from .face_context import MediaPipeFaceContext


class L2CSGaze:
    def __init__(self, face_provider, predictor, *, sha256, face_sha256, arch='ResNet50', max_faces=4):
        self.spec = self.capability(sha256, face_sha256, arch=arch, max_faces=max_faces)
        if not callable(getattr(face_provider, 'process', None)) or not callable(getattr(face_provider, 'close', None)) or not callable(predictor):
            raise ValueError('owned face provider and RGB crop predictor required')
        self.face_provider, self.predictor = face_provider, predictor
        self.max_faces, self.closed = max_faces, False
        self._lock = threading.RLock()

    @staticmethod
    def capability(sha256, face_sha256, *, arch='ResNet50', max_faces=4):
        for value in (sha256, face_sha256):
            if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('lowercase local gaze and face checkpoint SHA256 required')
        MediaPipeFaceContext.capability(face_sha256, max_faces=max_faces)
        if arch not in ('ResNet18', 'ResNet50'):
            raise ValueError('gaze architecture must be ResNet18 or ResNet50')
        digest = hashlib.sha256((sha256+face_sha256+arch+str(max_faces)).encode()).hexdigest()
        return CapabilitySpec('vision.l2cs.gaze', 'vision.gaze', 'video', 'l2cs.mediapipe.image',
            'v1.sha256.'+digest, (OutputField('faces','array'), OutputField('summary','object')),
            'frame', freshness_ms=1000, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera',), license='code:MIT+Apache-2.0;weights:unspecified')

    @classmethod
    def from_local(cls, path, sha256, face_path, face_sha256, *, arch='ResNet50', max_faces=4):
        cls.capability(sha256, face_sha256, arch=arch, max_faces=max_faces)
        # Verify all selected artifacts before importing native inference packages.
        blobs = []
        for name, expected, bound in ((path, sha256, 256*1024*1024), (face_path, face_sha256, 64*1024*1024)):
            file = Path(name).expanduser()
            if not file.is_absolute() or not file.is_file():
                raise ValueError('absolute existing local gaze and face checkpoint paths required')
            with file.open('rb') as source:
                payload = source.read(bound+1)
            if not 0 < len(payload) <= bound or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('local gaze/face checkpoint exceeds size bound or differs from SHA256')
            blobs.append(payload)
        try:
            import importlib.metadata
            for package, version in (('torch','2.8.0'), ('torchvision','0.23.0'), ('mediapipe','0.10.35')):
                if importlib.metadata.version(package).split('+')[0] != version:
                    raise RuntimeError('select torch==2.8.0, torchvision==0.23.0 and mediapipe==0.10.35')
            import torch
            import torchvision
            import cv2
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install the documented gaze dependencies in the selected local runtime') from exc
        # L2CS uses torchvision's residual blocks and the same stem/layers/pool.
        # Identity removes the ImageNet head. Keep the checkpoint's vestigial
        # finetune layer for strict key validation; it is never used in inference.
        model = getattr(torchvision.models, arch.lower())(weights=None)
        features = model.fc.in_features
        model.fc = torch.nn.Identity()
        model.fc_yaw_gaze = torch.nn.Linear(features,90)
        model.fc_pitch_gaze = torch.nn.Linear(features,90)
        model.fc_finetune = torch.nn.Linear(features+3,3)
        state = torch.load(io.BytesIO(blobs[0]), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(), torchvision.transforms.Resize(448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        def predict(crops):
            # Official image path first resizes each RGB face to224, then448.
            batch = torch.stack([transform(cv2.resize(crop,(224,224))) for crop in crops])
            with torch.inference_mode():
                features = model(batch)
                return (model.fc_yaw_gaze(features).cpu().numpy(),
                        model.fc_pitch_gaze(features).cpu().numpy())
        # Rechecks the face artifact and passes verified bytes to MediaPipe.
        # Changing its file between checks fails instead of mixing provenance.
        face = MediaPipeFaceContext.from_local(face_path,face_sha256,max_faces=max_faces)
        try:
            return cls(face,predict,sha256=sha256,face_sha256=face_sha256,arch=arch,max_faces=max_faces)
        except BaseException:
            face.close()
            raise

    @staticmethod
    def _angles(logits, count):
        import numpy as np
        values = np.asarray(logits)
        if values.shape != (count,90) or values.dtype.kind not in 'fi' or not np.isfinite(values).all() or ((values>1e6)|(values < -1e6)).any():
            raise ValueError('gaze model must return finite numeric Nx90 logits for each head')
        values = values.astype(np.float64)
        weights = np.exp(values-values.max(axis=1,keepdims=True))
        probabilities = weights/weights.sum(axis=1,keepdims=True)
        return (probabilities@np.arange(90)*4-180).tolist()

    def process(self, frame):
        with self._lock:
            if self.closed: raise RuntimeError('gaze provider is closed')
            import numpy as np
            data = frame.data
            if (frame.modality!='video' or not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or
                    data.ndim!=3 or data.shape[2]!=3 or not 1<=data.shape[0]<=2160 or
                    not 1<=data.shape[1]<=3840 or data.shape[0]*data.shape[1]>1920*1080):
                raise ValueError('supply a bounded uint8 BGR image (at most1920*1080 pixels)')
            height,width = data.shape[:2]
            observed = self.face_provider.process(frame)['faces']
            if not isinstance(observed,list) or len(observed)>self.max_faces:
                raise ValueError('face count exceeds selected bound')
            faces, crops, indices = [], [], []
            for index, face in enumerate(observed):
                bbox = face.get('bbox')
                result = dict(frame_face_index=index,bbox=bbox,status='unavailable_crop',
                              yaw_deg=None,pitch_deg=None,direction=None,confidence='unavailable')
                faces.append(result)
                if bbox is None: continue
                if (type(bbox) is not dict or set(bbox)!= {'x','y','width','height'} or
                        any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in bbox.values()) or
                        bbox['x']+bbox['width']>1.0000001 or bbox['y']+bbox['height']>1.0000001):
                    raise ValueError('face crop must be normalized and within image')
                x0,y0 = math.floor(bbox['x']*width), math.floor(bbox['y']*height)
                x1,y1 = min(width,math.ceil((bbox['x']+bbox['width'])*width)), min(height,math.ceil((bbox['y']+bbox['height'])*height))
                if x1-x0<16 or y1-y0<16: continue
                crops.append(data[y0:y1,x0:x1,::-1].copy())
                indices.append(index)
            if crops:
                heads = self.predictor(crops)
                if not isinstance(heads,(list,tuple)) or len(heads)!=2:
                    raise ValueError('gaze predictor must return yaw and pitch heads')
                yaw, pitch = (self._angles(head,len(crops)) for head in heads)
                for index,y,p in zip(indices,yaw,pitch):
                    yr,pr = math.radians(y),math.radians(p)
                    faces[index].update(status='estimated',yaw_deg=y,pitch_deg=p,
                        direction=dict(x=-math.cos(pr)*math.sin(yr),y=-math.sin(pr),z=-math.cos(pr)*math.cos(yr)))
            return self.spec.validate(dict(faces=faces,summary=dict(count=len(faces),
                estimated_count=len(crops),coordinate_frame='l2cs_model_convention',
                direction_units='unit_vector',angle_units='degrees',crop_source='mediapipe_face_landmarks_bbox',
                image_width=width,image_height=height,mirror_status='unspecified',
                calibration='unavailable',gaze_target='unavailable',identity='unavailable',
                temporal_tracking=False,smoothing=False)))

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed=True
                self.predictor=None
                self.face_provider.close()
