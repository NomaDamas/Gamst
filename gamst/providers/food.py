"""Task-trained SegFormer-B0 food semantic masks as bounded numerical context."""
import hashlib
import json
import math
from pathlib import Path
import re
import threading

from ..capabilities import CapabilitySpec,OutputField,bounded_json,identifier


def digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('explicit lowercase SHA256 required')
    return value


def configuration(config):
    config=bounded_json(config)
    required={'checkpoint_path','checkpoint_sha256','labels','background_id','unknown_id','training_dataset','target_cuisine','model_source','checkpoint_license','input_size'}
    optional={'threshold','max_labels','device'}
    if type(config) is not dict or not required<=set(config) or set(config)-required-optional:raise ValueError('explicit food checkpoint, label head and training/preprocessing provenance required')
    path=config['checkpoint_path']
    if type(path) is not str or len(path)>4096 or '\x00' in path or not Path(path).is_absolute():raise ValueError('absolute local safetensors checkpoint path required')
    digest(config['checkpoint_sha256'])
    labels=config['labels']
    if type(labels) is not list or not 2<=len(labels)<=104 or any(type(v) is not str or not 1<=len(v)<=64 or v.strip()!=v or not v.isprintable() for v in labels) or len(set(labels))!=len(labels):raise ValueError('2..104 distinct bounded exact checkpoint labels required')
    bg,unknown=config['background_id'],config['unknown_id']
    if type(bg) is not int or not 0<=bg<len(labels) or (unknown is not None and (type(unknown) is not int or not 0<=unknown<len(labels) or unknown==bg)):raise ValueError('explicit distinct background/optional unknown class IDs required')
    for key in ('training_dataset','target_cuisine'):identifier(config[key])
    for key in ('model_source','checkpoint_license'):
        v=config[key]
        if type(v) is not str or not 1<=len(v)<=(128 if key=='checkpoint_license' else 256) or not v.isprintable():raise ValueError('bounded declared model source and weight license required')
    if type(config['input_size']) is not int or config['input_size'] not in (256,512):raise ValueError('explicit square resize256/512 required')
    config.setdefault('threshold',.5);config.setdefault('max_labels',16);config.setdefault('device','cpu')
    t=config['threshold'];k=config['max_labels']
    if type(t) not in (int,float) or not math.isfinite(t) or not 0<t<1 or type(k) is not int or not 1<=k<=16 or config['device'] not in ('cpu','cuda'):raise ValueError('bounded threshold/max_labels and explicit cpu/cuda required')
    return config


def preprocess(bgr,size):
    import numpy as np
    from PIL import Image
    if type(bgr) is not np.ndarray or bgr.dtype!=np.uint8 or bgr.ndim!=3 or bgr.shape[2]!=3 or min(bgr.shape[:2])<32 or max(bgr.shape[:2])>2160 or bgr.shape[0]*bgr.shape[1]>1920*1080:raise ValueError('plain uint8 BGR image32..2160axes,at most1080p required')
    if type(size) is not int or size not in (256,512):raise ValueError('resize256/512 required')
    rgb=bgr[...,::-1].copy()
    scaled=np.asarray(Image.fromarray(rgb).resize((size,size),resample=Image.Resampling.BILINEAR),dtype=np.float32)/255
    scaled=(scaled-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return scaled.transpose(2,0,1)[None].copy()


def dependencies():
    from importlib.metadata import version,PackageNotFoundError
    try:
        for name,expected in (('transformers','4.57.6'),('torch','2.8.0'),('huggingface-hub','0.36.0'),('safetensors','0.7.0')):
            if version(name).split('+')[0]!=expected:raise RuntimeError('food runtime requires '+name+'=='+expected+'; use the documented dedicated environment')
        import torch
        from transformers import SegformerConfig,SegformerForSemanticSegmentation
        from safetensors.torch import load
    except (ImportError,PackageNotFoundError) as exc:raise RuntimeError('food SegFormer runtime unavailable/incompatible; no automatic installation') from exc
    return torch,SegformerConfig,SegformerForSemanticSegmentation,load


def model_config(cls,labels):
    return cls(num_labels=len(labels),id2label=dict(enumerate(labels)),label2id={name:i for i,name in enumerate(labels)},
        num_channels=3,num_encoder_blocks=4,depths=[2,2,2,2],sr_ratios=[8,4,2,1],hidden_sizes=[32,64,160,256],
        patch_sizes=[7,3,3,3],strides=[4,2,2,2],num_attention_heads=[1,2,5,8],mlp_ratios=[4,4,4,4],
        hidden_act='gelu',hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,classifier_dropout_prob=.1,
        drop_path_rate=.1,layer_norm_eps=1e-6,decoder_hidden_size=256,semantic_loss_ignore_index=255)


class SegFormerFoodNative:
    @classmethod
    def from_local(cls,config):
        config=configuration(config)
        path=Path(config['checkpoint_path'])
        if not path.is_file():raise ValueError('existing local task-trained safetensors required')
        with path.open('rb') as source:payload=source.read(64*1024*1024+1)
        if not 1<=len(payload)<=64*1024*1024 or hashlib.sha256(payload).hexdigest()!=config['checkpoint_sha256']:raise ValueError('food checkpoint SHA256/64MiB limit mismatch')
        torch,Config,Model,load=dependencies()
        with torch.device('cpu'),torch.random.fork_rng(devices=[]):model=Model(model_config(Config,config['labels'])).float().eval()
        state=load(payload);expected=model.state_dict()
        if set(state)!=set(expected):raise ValueError('food checkpoint exact B0 head/key mismatch')
        for key,value in state.items():
            if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not torch.isfinite(value).all():raise ValueError('food checkpoint tensor shape/dtype/finite mismatch')
        model.load_state_dict(state,strict=True)
        if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('explicit CUDA runtime unavailable')
        self=cls();self.config=config;self.model=model.to(config['device']);self.closed=False
        return self

    def predict(self,bgr):
        if self.closed:raise RuntimeError('food native model closed')
        import torch
        pixels=torch.from_numpy(preprocess(bgr,self.config['input_size'])).to(self.config['device'])
        with torch.inference_mode():out=self.model(pixel_values=pixels,return_dict=True).logits
        expected=(1,len(self.config['labels']),self.config['input_size']//4,self.config['input_size']//4)
        if tuple(out.shape)!=expected or out.dtype!=torch.float32 or not torch.isfinite(out).all():raise ValueError('food logits shape/dtype/finite mismatch')
        return out[0].detach().cpu().numpy().copy()
    def close(self):self.closed=True;self.model=None


class FoodContext:
    def __init__(self,backend,**config):
        self.config=configuration(config);self.spec=self.capability(**config);self.backend=backend
        self.lock=threading.RLock();self.closed=False
    @staticmethod
    def capability(**config):
        config=configuration(config);public={k:v for k,v in config.items() if k!='checkpoint_path'}
        version=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('vision.food.segformer_b0','vision.food','video','food.segformer_b0','1.'+version,
            (OutputField('regions','array'),OutputField('summary','object')),'frame',freshness_ms=1000,
            interpretation='classification',devices=(config['device'],),sensor_requirements=('uint8-bgr-camera','task-trained-food-label-head'),
            license='adapter:Apache-2.0;HF-code:Apache-2.0;checkpoint:'+config['checkpoint_license'])
    @classmethod
    def from_local(cls,**config):return cls(SegFormerFoodNative.from_local(config),**config)
    def close(self):
        with self.lock:
            if not self.closed:self.closed=True;self.backend.close()
    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('food context closed')
            import numpy as np
            data=frame.data
            # Validate and copy before invoking the backend, retaining source pixels.
            if frame.modality!='video' or type(data) is not np.ndarray or data.dtype!=np.uint8 or data.ndim!=3 or data.shape[2]!=3 or min(data.shape[:2])<32 or max(data.shape[:2])>2160 or data.shape[0]*data.shape[1]>1920*1080:raise ValueError('bounded plain uint8 BGR video required')
            data=data.copy();config=self.config;size=config['input_size']//4
            logits=self.backend.predict(data)
            if type(logits) is not np.ndarray or logits.dtype!=np.float32 or logits.shape!=(len(config['labels']),size,size) or not np.isfinite(logits).all():raise ValueError('exact finite float32 C,H,W food logits required')
            logits=logits.copy();classes=np.argmax(logits,axis=0)
            stable=logits.astype(np.float64)-np.max(logits,axis=0).astype(np.float64)
            scores=1/np.exp(stable).sum(axis=0)
            accepted=scores>=config['threshold'];total=size*size
            counts=np.bincount(classes[accepted],minlength=len(config['labels']))
            bg,unknown=config['background_id'],config['unknown_id']
            food_ids=[i for i,count in enumerate(counts) if count and i not in (bg,unknown)]
            food_ids.sort(key=lambda i:(-int(counts[i]),i));chosen=food_ids[:config['max_labels']]
            regions=[]
            for label in chosen:
                mask=(classes==label)&accepted;y,x=np.nonzero(mask)
                # Both supported output sizes divide16, so every cell is equal.
                coverage=mask.reshape(16,size//16,16,size//16).mean(axis=(1,3))
                regions.append(dict(class_id=label,label=config['labels'][label],semantic_pixel_count=int(counts[label]),
                    image_area_fraction=float(counts[label]/total),mean_softmax_score=float(scores[mask].mean()),
                    box_xyxy_normalized=[float(x.min()/size),float(y.min()/size),float((x.max()+1)/size),float((y.max()+1)/size)],
                    mask=dict(width=16,height=16,order='row-major',coverage=coverage.reshape(-1).tolist())))
            low=int(np.count_nonzero(~accepted));background=int(counts[bg]);unknown_count=int(counts[unknown]) if unknown is not None else 0
            omitted=sum(int(counts[i]) for i in food_ids[config['max_labels']:])
            return self.spec.validate(dict(regions=regions,summary=dict(
                status='labels_estimated' if regions else 'no_accepted_food_labels',segmentation_type='semantic_not_instances',
                source_width=data.shape[1],source_height=data.shape[0],semantic_width=size,semantic_height=size,
                geometry='square-resized native logits; normalized coordinates approximate source pixels',
                background_fraction=background/total,declared_unknown_class_fraction=unknown_count/total,
                below_threshold_fraction=low/total,omitted_food_fraction=omitted/total,truncated=len(food_ids)>len(chosen),
                threshold=config['threshold'],scores_calibrated=False,unknown_detection_validated=False,
                model_quality_validated=False,checkpoint_training_verified=False,training_dataset=config['training_dataset'],
                target_cuisine=config['target_cuisine'],model_source=config['model_source'],checkpoint_sha256=config['checkpoint_sha256'],
                label_map_sha256=hashlib.sha256(json.dumps(config['labels'],ensure_ascii=True).encode()).hexdigest(),
                capture_sha256=hashlib.sha256(data.tobytes()).hexdigest(),preprocessing='RGB.PIL-bilinear.square.rescale255.ImageNet-mean-std',
                mass_grams=None,calories_kcal=None,nutrients_available=False,temperature_available=False,complete_snapshot=False)))
