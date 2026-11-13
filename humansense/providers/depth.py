"""Bounded frame-local relative inverse-depth context from local DA-V2 Small."""
import hashlib
from pathlib import Path
import re
import threading

from ..capabilities import CapabilitySpec, OutputField


class DepthAnythingContext:
    def __init__(self, predictor, *, sha256, grid_size=3):
        self.spec=self.capability(sha256,grid_size=grid_size)
        if not callable(predictor): raise ValueError('explicit RGB depth predictor required')
        self.predictor=predictor
        self.grid_size=grid_size
        self.closed=False
        self._lock=threading.RLock()

    @staticmethod
    def capability(sha256, *, grid_size=3):
        if type(sha256) is not str or not re.fullmatch('[0-9a-f]{64}',sha256):
            raise ValueError('lowercase safetensors SHA256 required')
        if type(grid_size) is not int or not 1<=grid_size<=8:
            raise ValueError('depth grid_size must be1..8')
        return CapabilitySpec('vision.depth_anything_v2.small', 'vision.relative_depth','video',
            'depth_anything_v2.small.hf','v1.sha256.'+sha256+'.g'+str(grid_size),
            (OutputField('grid','object'),OutputField('summary','object')),
            'frame',freshness_ms=1000,interpretation='estimate',devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera',),license='Apache-2.0')

    @classmethod
    def from_local(cls,path,sha256,*,grid_size=3):
        cls.capability(sha256,grid_size=grid_size)
        file=Path(path).expanduser()
        if not file.is_absolute() or not file.is_file():
            raise ValueError('absolute existing local HF Small model.safetensors required')
        with file.open('rb') as source: payload=source.read(128*1024*1024+1)
        if not 0<len(payload)<=128*1024*1024 or hashlib.sha256(payload).hexdigest()!=sha256:
            raise ValueError('depth checkpoint must match SHA256 and be at most128MiB')
        try:
            import importlib.metadata
            for package,version in (('transformers','4.57.6'),('torch','2.8.0')):
                if importlib.metadata.version(package).split('+')[0]!=version:
                    raise RuntimeError('select transformers==4.57.6 and torch==2.8.0 in the depth runtime')
            import torch
            from safetensors.torch import load
            from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
        except (ImportError,importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('select a compatible isolated depth runtime; see humansense-sdk[depth]') from exc
        # Fixed HF Small config at5426e4f0; no AutoClass, Hub call, external
        # config, pickle loading, metric head or pretrained-backbone download.
        backbone=Dinov2Config(hidden_size=384,image_size=518,num_attention_heads=6,
            num_hidden_layers=12,patch_size=14,out_features=['stage3','stage6','stage9','stage12'],
            out_indices=[3,6,9,12],reshape_hidden_states=False)
        config=DepthAnythingConfig(backbone_config=backbone,backbone=None,
            use_pretrained_backbone=False,use_timm_backbone=False,patch_size=14,
            reassemble_hidden_size=384,reassemble_factors=[4,2,1,.5],
            neck_hidden_sizes=[48,96,192,384],fusion_hidden_size=64,
            head_hidden_size=32,head_in_index=-1,depth_estimation_type='relative')
        with torch.device('cpu'):
            model=DepthAnythingForDepthEstimation(config)
        model.to(device='cpu',dtype=torch.float32)
        model.load_state_dict(load(payload),strict=True)
        model.eval()
        processor=DPTImageProcessor(do_resize=True,size={'height':518,'width':518},
            keep_aspect_ratio=True,ensure_multiple_of=14,resample=3,do_rescale=True,
            rescale_factor=1/255,do_normalize=True,image_mean=[.485,.456,.406],
            image_std=[.229,.224,.225],do_pad=False)
        def predict(rgb):
            inputs=processor(images=rgb,return_tensors='pt',input_data_format='channels_last')
            pixels=inputs['pixel_values']
            if (pixels.ndim!=4 or tuple(pixels.shape[:2])!=(1,3) or
                    min(pixels.shape[-2:])<14 or max(pixels.shape[-2:])>1036 or
                    pixels.shape[-2]*pixels.shape[-1]>518*1036 or
                    any(side%14 for side in pixels.shape[-2:])):
                raise ValueError('depth preprocessor tensor exceeds selected bounds')
            with torch.inference_mode():
                output=model(pixel_values=pixels).predicted_depth
            if output.ndim!=3 or output.shape[0]!=1 or tuple(output.shape[-2:])!=tuple(pixels.shape[-2:]):
                raise ValueError('depth model must return one scalar map')
            return output[0].cpu().numpy()
        return cls(predict,sha256=sha256,grid_size=grid_size)

    def process(self,frame):
        with self._lock:
            if self.closed: raise RuntimeError('depth provider is closed')
            import numpy as np
            data=frame.data
            if (frame.modality!='video' or not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or
                    data.ndim!=3 or data.shape[2]!=3 or not 32<=data.shape[0]<=2160 or
                    not 32<=data.shape[1]<=3840 or data.shape[0]*data.shape[1]>1920*1080 or
                    max(data.shape[:2])/min(data.shape[:2])>2):
                raise ValueError('depth requires uint8 BGR,32px minimum, at most2MP and2:1 aspect ratio')
            height,width=data.shape[:2]
            depth=np.asarray(self.predictor(data[:,:,::-1].copy()))
            if (depth.ndim!=2 or depth.dtype.kind not in 'fi' or min(depth.shape)<self.grid_size or
                    max(depth.shape)>1036 or depth.size>518*1036 or not np.isfinite(depth).all() or
                    (depth<0).any() or (depth>1e6).any()):
                raise ValueError('depth model must return a bounded finite nonnegative scalar map')
            depth=depth.astype(np.float64)
            low,high=np.percentile(depth,[5,95])
            usable=high-low>max(1e-6,abs(high)*1e-6)
            scores=[]
            rows=np.linspace(0,depth.shape[0],self.grid_size+1,dtype=int)
            columns=np.linspace(0,depth.shape[1],self.grid_size+1,dtype=int)
            if usable:
                normalized=np.clip((depth-low)/(high-low),0,1)
                for row in range(self.grid_size):
                    for column in range(self.grid_size):
                        scores.append(float(np.median(normalized[rows[row]:rows[row+1],columns[column]:columns[column+1]])))
            else: scores=[None]*(self.grid_size**2)
            return self.spec.validate(dict(grid=dict(rows=self.grid_size,columns=self.grid_size,
                order='row_major_top_left',relative_nearness=scores,
                x_edges=(columns/depth.shape[1]).tolist(),y_edges=(rows/depth.shape[0]).tolist()),summary=dict(
                status='estimated' if usable else 'unavailable_low_spread',
                semantics='larger_is_relatively_nearer',units='dimensionless',
                normalization='per_frame_p05_p95_clipped',aggregation='cell_median',
                confidence='unavailable',metric_depth=False,temporal_comparability=False,
                temporal_tracking=False,coordinate_frame='normalized_image_xy',
                image_width=width,image_height=height,model_width=int(depth.shape[1]),
                model_height=int(depth.shape[0]),calibration='unavailable')))

    def close(self):
        with self._lock:
            self.closed=True
            self.predictor=None
