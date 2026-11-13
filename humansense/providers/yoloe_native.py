"""Explicit local YOLOE runtime. This module does not import ML during discovery."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

from .yoloe import YOLOEContext, ULTRALYTICS_VERSION, _hash, prompt_labels, visual_examples

CLIP_REVISION = '7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40'


def configuration(config):
    """Validate discovery without opening a file, importing ML or spawning a child."""
    common = {'checkpoint_path','checkpoint_sha256','labels','mode','runtime_root'}
    optional = {'device','threshold','max_objects','imgsz'}
    mode = config.get('mode')
    extra = {'encoder_path','encoder_sha256','encoder_family'} if mode == 'text' else {
        'reference_path','reference_sha256','boxes','class_ids'}
    if not common|extra <= set(config) or set(config)-(common|optional|extra):
        raise ValueError('explicit YOLOE model/runtime and mode-specific local assets required')
    labels = prompt_labels(config['labels'])
    for key in ('checkpoint_path','runtime_root')+(('encoder_path',) if mode == 'text' else ('reference_path',)):
        if type(config[key]) is not str or not 1 <= len(config[key]) <= 4096 or '\x00' in config[key]:
            raise ValueError('bounded explicit local path required')
    _hash(config['checkpoint_sha256'])
    provenance = dict(mode=mode,labels=list(labels),checkpoint_sha256=config['checkpoint_sha256'])
    if mode == 'text':
        _hash(config['encoder_sha256'])
        if config['encoder_family'] not in ('mobileclip:blt','mobileclip2:b'):
            raise ValueError('explicit matching MobileCLIP family required')
        provenance.update(encoder_sha256=config['encoder_sha256'],encoder_family=config['encoder_family'],
                          tokenizer_revision=CLIP_REVISION)
    elif mode == 'exemplar':
        import math
        _hash(config['reference_sha256'])
        boxes,ids = config['boxes'],config['class_ids']
        if (type(boxes) is not list or type(ids) is not list or not 1 <= len(boxes) <= 32 or
                len(boxes) != len(ids) or any(type(i) is not int or not 0 <= i < len(labels) for i in ids) or
                set(ids) != set(range(len(labels)))):
            raise ValueError('bounded exemplar boxes and sequential class mapping required')
        for box in boxes:
            if (type(box) is not list or len(box) != 4 or any(type(v) not in (int,float) or
                    not math.isfinite(v) or not 0 <= v <= 2048 for v in box) or
                    box[2]-box[0] < 1 or box[3]-box[1] < 1):
                raise ValueError('bounded nonempty pixel xyxy reference boxes required')
        provenance.update(reference_sha256=config['reference_sha256'],boxes=boxes,class_ids=ids)
    else:
        raise ValueError('text or exemplar mode required')
    device = config.get('device','cpu')
    imgsz = config.get('imgsz',640)
    if device not in ('cpu','cuda') or type(imgsz) is not int or imgsz not in (320,480,640):
        raise ValueError('cpu/cuda device and imgsz320/480/640 required')
    provenance.update(imgsz=imgsz,runtime=ULTRALYTICS_VERSION)
    args = dict(labels=labels,prompt_sha256=hashlib.sha256(json.dumps(provenance,sort_keys=True,
        separators=(',',':')).encode()).hexdigest(),checkpoint_sha256=config['checkpoint_sha256'],mode=mode,
        threshold=config.get('threshold',.25),max_objects=config.get('max_objects',16))
    YOLOEContext.capability(**args)
    return args,device,imgsz


def _read(path, digest, limit):
    _hash(digest)
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or not 1 <= path.stat().st_size <= limit:
        raise ValueError('local asset missing or exceeds size limit')
    with path.open('rb') as handle:
        data = handle.read(limit+1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('local asset SHA256 or size mismatch')
    return data


def _dependencies(mode):
    from importlib.metadata import version, distribution, PackageNotFoundError
    try:
        for name,expected in (('ultralytics',ULTRALYTICS_VERSION),('torch','2.8.0'),('torchvision','0.23.0')):
            if version(name).split('+')[0] != expected:
                raise RuntimeError(f'YOLOE requires {name}=={expected} in its dedicated runtime')
        if mode == 'text':
            dist = distribution('clip')
            provenance = json.loads(dist.read_text('direct_url.json') or '{}')
            if dist.version != '1.0' or provenance.get('vcs_info',{}).get('commit_id') != CLIP_REVISION:
                raise RuntimeError('install the explicitly pinned ultralytics/CLIP revision in the dedicated runtime')
    except PackageNotFoundError as exc:
        raise RuntimeError('YOLOE optional runtime is unavailable; install documented dependencies explicitly') from exc


def _offline(*args,**kwargs):
    raise RuntimeError('network access is disabled in the local YOLOE worker')


def _bootstrap(directory):
    """Only for the owned one-provider child; this is not an untrusted-code sandbox."""
    if os.environ.get('GAMST_PROVIDER_WORKER') != '1' or 'ultralytics' in sys.modules:
        raise RuntimeError('YOLOE native loading requires a fresh owned ProcessProvider worker')
    os.environ.update(YOLO_AUTOINSTALL='False',YOLO_OFFLINE='True',ULTRALYTICS_SAFE_LOAD='True',
                      YOLO_CONFIG_DIR=str(directory))
    # Keep these guards for the child's lifetime. Native artifacts are trusted;
    # this stops accidental Python download/telemetry paths, not hostile code.
    import socket
    socket.getaddrinfo = _offline
    socket.create_connection = _offline
    socket.socket.connect = _offline
    socket.socket.connect_ex = _offline
    socket.socket.sendto = _offline
    if hasattr(socket.socket,'sendmsg'): socket.socket.sendmsg = _offline


class NativeYOLOE:
    def __init__(self, model, *, device, imgsz, threshold, max_objects, reference=None, prompts=None):
        self.model,self.device,self.imgsz = model,device,imgsz
        self.threshold,self.max_objects = threshold,max_objects
        self.reference,self.prompts = reference,prompts
        self.runtime = None

    def predict(self,image):
        import numpy as np
        import torch
        kwargs = dict(source=image,device=self.device,imgsz=self.imgsz,conf=self.threshold,
            max_det=self.max_objects,retina_masks=True,half=False,save=False,save_txt=False,
            save_crop=False,show=False,verbose=False,stream=False,batch=1,augment=False,
            rect=False,agnostic_nms=True,iou=.7)
        if self.reference is not None:
            kwargs.update(refer_image=self.reference,visual_prompts=self.prompts)
        with torch.inference_mode(): results = self.model.predict(**kwargs)
        if type(results) is not list or len(results) != 1:
            raise ValueError('YOLOE must return exactly one result')
        result = results[0]
        if tuple(result.orig_shape) != image.shape[:2] or result.boxes is None:
            raise ValueError('YOLOE original image geometry mismatch')
        shape = tuple(result.boxes.data.shape)
        if len(shape) != 2 or shape[1:] != (6,) or shape[0] > self.max_objects:
            raise ValueError('native YOLOE detection bound exceeded before host transfer')
        boxes = result.boxes.data.detach().cpu().float().numpy().copy()
        if boxes.ndim != 2 or boxes.shape[1:] != (6,) or len(boxes) > self.max_objects:
            raise ValueError('native YOLOE detection bound exceeded')
        if not len(boxes):
            masks = np.empty((0,*image.shape[:2]),dtype=np.uint8)
        elif result.masks is None or tuple(result.masks.data.shape) != (len(boxes),*image.shape[:2]):
            raise ValueError('YOLOE must restore one original-image mask per box')
        else:
            masks = result.masks.data.detach().cpu().numpy().copy()
        # The model now owns the fixed visual embeddings; do not re-encode on
        # every capture. Preserve reference until a full successful prediction.
        self.reference = self.prompts = None
        return boxes,masks

    def close(self):
        self.model = self.reference = self.prompts = None
        if self.runtime is not None:
            runtime,self.runtime = self.runtime,None
            runtime.cleanup()


def load_worker(config):
    args,device,imgsz = configuration(config)
    if os.environ.get('GAMST_PROVIDER_WORKER') != '1' or 'ultralytics' in sys.modules:
        raise RuntimeError('use a fresh owned ProcessProvider worker for YOLOE')
    _dependencies(config['mode'])
    checkpoint = _read(config['checkpoint_path'],config['checkpoint_sha256'],256*1024*1024)
    other = _read(config['encoder_path'],config['encoder_sha256'],512*1024*1024) if config['mode']=='text' else _read(
        config['reference_path'],config['reference_sha256'],4*1024*1024)
    root = Path(config['runtime_root']).expanduser().resolve(strict=True)
    if not root.is_dir(): raise ValueError('existing owned runtime_root directory required')
    runtime = tempfile.TemporaryDirectory(prefix='gamst-yoloe-',dir=root)
    backend = None
    try:
        directory = Path(runtime.name)
        _bootstrap(directory)
        model_path = directory/'checkpoint.pt'
        model_path.write_bytes(checkpoint); del checkpoint
        import torch
        if not hasattr(torch.serialization,'get_unsafe_globals_in_checkpoint'):
            raise RuntimeError('restricted native checkpoint loading is unavailable')
        from ultralytics import YOLOE
        from ultralytics.nn.tasks import YOLOESegModel
        from ultralytics.utils import SETTINGS, SAFE_LOAD, AUTOINSTALL, ONLINE
        if not SAFE_LOAD or AUTOINSTALL or ONLINE:
            raise RuntimeError('YOLOE offline/restricted-load bootstrap did not take effect')
        SETTINGS.update({key:False for key in ('sync','clearml','comet','dvc','mlflow','raytune','tensorboard','wandb','vscode_msg')})
        model = YOLOE(str(model_path),task='segment',verbose=False)
        if not isinstance(model.model,YOLOESegModel) or model.task != 'segment':
            raise ValueError('promptable YOLOE segmentation checkpoint required')
        head = model.model.model[-1]
        if head.is_fused or hasattr(head,'lrpc') or head.embed != 512:
            raise ValueError('unfused promptable512-dimensional YOLOE head required')
        parameters = list(model.model.parameters())
        if not 0 < sum(p.numel() for p in parameters) <= 80000000:
            raise ValueError('YOLOE parameter count exceeds selected runtime limit')
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError('nonfinite YOLOE model parameters')
        model.model.float().eval().to(device)
        model.ckpt = None  # Drop training/optimizer checkpoint references.
        if hasattr(model.model,'clip_model'): delattr(model.model,'clip_model')
        del parameters
        model.overrides = {'model':str(model_path),'task':'segment'}
        reference = prompts = None
        if config['mode'] == 'text':
            if getattr(model.model,'text_model','mobileclip:blt') != config['encoder_family']:
                raise ValueError('declared text encoder family does not match checkpoint')
            import clip
            tokens = clip.tokenize(list(args['labels']),truncate=False).to(device)
            encoder = torch.jit.load(io.BytesIO(other),map_location=device).float().eval()
            del other
            with torch.inference_mode():
                features = encoder(tokens)
                if (not isinstance(features,torch.Tensor) or tuple(features.shape) != (len(args['labels']),512) or
                        not torch.isfinite(features).all()):
                    raise ValueError('local MobileCLIP must emit finite Nx512 features')
                embeddings = head.get_tpe(features.float().reshape(1,len(args['labels']),512))
                if tuple(embeddings.shape) != (1,len(args['labels']),512) or not torch.isfinite(embeddings).all():
                    raise ValueError('invalid checkpoint-specific text embeddings')
                model.set_classes(list(args['labels']),embeddings)
            del encoder,features,tokens
        else:
            from PIL import Image
            import numpy as np
            with Image.open(io.BytesIO(other)) as source:
                if source.format != 'PNG' or source.mode != 'RGB' or min(source.size)<32 or max(source.size)>2048 or source.width*source.height>1048576:
                    raise ValueError('bounded RGB PNG reference required')
                reference = np.asarray(source).copy()[:,:,::-1].copy()
            reference,boxes,ids,_ = visual_examples(reference,config['boxes'],config['class_ids'],args['labels'])
            prompts = dict(bboxes=boxes,cls=ids)
            del other
        backend = NativeYOLOE(model,device=device,imgsz=imgsz,threshold=args['threshold'],
            max_objects=args['max_objects'],reference=reference,prompts=prompts)
        backend.runtime = runtime
        return YOLOEContext(backend,**args,artifact_verified=True)
    except BaseException:
        if backend is not None: backend.close()
        else: runtime.cleanup()
        raise
