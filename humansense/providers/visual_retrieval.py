"""Local DINOv3 frame embeddings and bounded appearance-history candidates."""
from collections import deque
import hashlib
from pathlib import Path
import re
import threading
import time
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier


class DINOv3Retrieval:
    def __init__(self, predictor, *, sha256, capacity=128, retention_ms=600000, top_k=5):
        self.spec=self.capability(sha256,capacity=capacity,retention_ms=retention_ms,top_k=top_k)
        if not callable(predictor): raise ValueError('explicit RGB embedding predictor required')
        self.predictor=predictor
        self.capacity,self.retention_ms,self.top_k=capacity,retention_ms,top_k
        self._lock=threading.RLock(); self.closed=False
        self._entries=deque(); self._owner=None; self._last=None

    @staticmethod
    def capability(sha256, *, capacity=128, retention_ms=600000, top_k=5):
        if type(sha256) is not str or not re.fullmatch('[0-9a-f]{64}',sha256):
            raise ValueError('lowercase local safetensors SHA256 required')
        for value,low,high in ((capacity,1,1024),(retention_ms,1,86400000),(top_k,1,20)):
            if type(value) is not int or not low<=value<=high: raise ValueError('invalid retrieval bounds')
        version='v1.sha256.'+sha256+f'.c{capacity}.r{retention_ms}.k{top_k}'
        return CapabilitySpec('vision.dinov3.retrieval','vision.appearance_candidates','video',
            'dinov3.vits16.local',version,(OutputField('candidates','array'),OutputField('summary','object')),
            'frame',freshness_ms=1000,stateful=True,reset_on_gap=False,interpretation='estimate',
            devices=('cpu',),sensor_requirements=('uint8-bgr-camera',),license='DINOv3-license;verify-local-artifact')

    @classmethod
    def from_local(cls,path,sha256,*,capacity=128,retention_ms=600000,top_k=5):
        cls.capability(sha256,capacity=capacity,retention_ms=retention_ms,top_k=top_k)
        file=Path(path).expanduser()
        if not file.is_absolute() or not file.is_file(): raise ValueError('absolute existing local HF Small safetensors required')
        with file.open('rb') as source: payload=source.read(128*1024*1024+1)
        if not 0<len(payload)<=128*1024*1024 or hashlib.sha256(payload).hexdigest()!=sha256:
            raise ValueError('checkpoint must match SHA256 and be at most128MiB')
        try:
            import importlib.metadata
            for package,version in (('torch','2.8.0'),('torchvision','0.23.0'),('transformers','4.57.6')):
                if importlib.metadata.version(package).split('+')[0]!=version:
                    raise RuntimeError('select the pinned isolated humansense-sdk[retrieval] runtime')
            import torch
            from safetensors.torch import load
            from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast
        except (ImportError,importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('select a compatible isolated humansense-sdk[retrieval] runtime') from exc
        config=DINOv3ViTConfig(patch_size=16,hidden_size=384,intermediate_size=1536,
            num_hidden_layers=12,num_attention_heads=6,hidden_act='gelu',num_register_tokens=4,
            use_gated_mlp=False,proj_bias=True)
        with torch.device('cpu'): model=DINOv3ViTModel(config)
        model.to(device='cpu',dtype=torch.float32)
        model.load_state_dict(load(payload),strict=True); model.eval()
        processor=DINOv3ViTImageProcessorFast(do_resize=True,size={'height':224,'width':224},
            resample=2,do_rescale=True,rescale_factor=1/255,do_normalize=True,
            image_mean=[.485,.456,.406],image_std=[.229,.224,.225],do_center_crop=False)
        def predict(rgb):
            pixels=processor(images=rgb,return_tensors='pt',input_data_format='channels_last',device='cpu')['pixel_values']
            if tuple(pixels.shape)!=(1,3,224,224) or not torch.isfinite(pixels).all():
                raise ValueError('embedding preprocessor must return one finite224x224 tensor')
            with torch.inference_mode(): output=model(pixel_values=pixels).pooler_output
            if tuple(output.shape)!=(1,384): raise ValueError('Small model must return384 CLS features')
            return output[0].cpu().numpy()
        return cls(predict,sha256=sha256,capacity=capacity,retention_ms=retention_ms,top_k=top_k)

    def reset(self):
        with self._lock:
            if self.closed: raise RuntimeError('retrieval provider is closed')
            self._entries.clear(); self._owner=self._last=None

    def close(self):
        with self._lock:
            self._entries.clear(); self._owner=self._last=None
            self.predictor=None; self.closed=True

    def process(self,frame):
        with self._lock:
            if self.closed: raise RuntimeError('retrieval provider is closed')
            import numpy as np
            sensor,ts,data=frame.sensor,frame.timestamp_ms,frame.data
            if not isinstance(sensor,SensorMetadata): raise ValueError('capture SensorMetadata required')
            identifier(frame.subject_id,'subject_id')
            if type(ts) is not int or not 0<=ts<=2**53-1: raise ValueError('invalid capture timestamp')
            if (frame.modality!='video' or not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or
                data.ndim!=3 or data.shape[2]!=3 or min(data.shape[:2])<32 or
                data.shape[0]>2160 or data.shape[1]>3840 or data.shape[0]*data.shape[1]>1920*1080 or
                max(data.shape[:2])/min(data.shape[:2])>4):
                raise ValueError('retrieval requires bounded uint8 BGR,32px minimum,2MP and4:1 aspect maximum')
            owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,
                   sensor.coordinate_frame,sensor.calibration_id)
            if self._owner==owner and self._last is not None and (ts<=self._last[0] or sensor.sequence<=self._last[1]):
                raise ValueError('capture timestamp and sequence must increase within an owner')
            embedding=np.asarray(self.predictor(data[:,:,::-1].copy()))
            if (embedding.shape!=(384,) or embedding.dtype.kind not in 'fi' or
                not np.isfinite(embedding).all() or (np.abs(embedding)>1e6).any()):
                raise ValueError('predictor must return384 finite bounded features')
            embedding=embedding.astype(np.float64,copy=True)
            norm=float(np.linalg.norm(embedding))
            if norm<=1e-12: raise ValueError('zero embedding cannot support similarity')
            embedding/=norm
            # Mutation starts only after input and inference validation. No image is retained.
            now=time.monotonic()
            changed=self._owner is not None and owner!=self._owner
            if changed: self._entries.clear()
            self._owner=owner
            while self._entries and (ts-self._entries[0]['timestamp_ms']>self.retention_ms or
                    (now-self._entries[0]['inserted'])*1000>self.retention_ms):
                self._entries.popleft()
            scored=[(float(np.clip(np.dot(embedding,item['embedding']),-1,1)),item) for item in self._entries]
            scored.sort(key=lambda pair:(-pair[0],-pair[1]['timestamp_ms']))
            candidates=[dict(reference_id=item['reference_id'],timestamp_ms=item['timestamp_ms'],
                sequence=item['sequence'],age_ms=ts-item['timestamp_ms'],cosine_similarity=score,
                current_location='unknown',identity_confirmed=False) for score,item in scored[:self.top_k]]
            reference=uuid.uuid4().hex
            evicted=len(self._entries)>=self.capacity
            if evicted: self._entries.popleft()
            self._entries.append(dict(reference_id=reference,timestamp_ms=ts,sequence=sensor.sequence,
                                      inserted=now,embedding=embedding))
            self._last=(ts,sensor.sequence)
            return self.spec.validate(dict(candidates=candidates,summary=dict(
                reference_id=reference,retained_frames=len(self._entries),capacity=self.capacity,
                retention_ms=self.retention_ms,top_k=self.top_k,owner_changed=changed,capacity_evicted=evicted,
                descriptor='normalized_cls_384',comparison_scope='whole_frame_appearance',
                score_kind='cosine_similarity_not_probability',threshold_applied=False,
                object_identity=False,current_location_inferred=False,temporal_encoder=False,
                source_id=sensor.source_id,sensor_id=sensor.sensor_id,stream_epoch=sensor.stream_epoch)))
