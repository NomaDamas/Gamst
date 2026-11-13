"""Local PP-OCRv5 mobile ONNX text context with explicit bounded crops."""
import hashlib
import math
from pathlib import Path
import re
import threading

from ..capabilities import CapabilitySpec, OutputField


class _Config(dict):
    def __getattr__(self,name):
        try: return self[name]
        except KeyError as exc: raise AttributeError(name) from exc


class _BoundedSession:
    def __init__(self,session,kind,classes=None): self.session,self.kind,self.classes=session,kind,classes
    def __getattr__(self,name): return getattr(self.session,name)
    def __call__(self,data):
        import numpy as np
        if (not isinstance(data,np.ndarray) or data.dtype!=np.float32 or data.ndim!=4 or
                tuple(data.shape[:2])!=(1,3) or not np.isfinite(data).all()):
            raise ValueError('OCR inference requires one finite float32 NCHW input')
        h,w=data.shape[-2:]
        if self.kind=='det':
            if not 32<=h<=960 or not 32<=w<=960 or h%32 or w%32:
                raise ValueError('OCR detection tensor exceeds selected bounds')
        elif h!=48 or not 1<=w<=1536:
            raise ValueError('OCR recognition tensor exceeds selected bounds')
        result=np.asarray(self.session(data))
        valid=(result.ndim==4 and result.shape[:2]==(1,1) and
               1<=min(result.shape[-2:]) and max(result.shape[-2:])<=960) if self.kind=='det' else (
               result.ndim==3 and result.shape[0]==1 and 1<=result.shape[1]<=512 and result.shape[2]==self.classes)
        if (not valid or result.dtype.kind!='f' or not np.isfinite(result).all() or
                (result<0).any() or (result>1).any()):
            raise ValueError('OCR model must return bounded finite probability outputs')
        return result


class PPOCRContext:
    def __init__(self,detector,recognizer,cropper,*,det_sha256,rec_sha256,language='en',max_regions=16):
        self.spec=self.capability(det_sha256,rec_sha256,language=language,max_regions=max_regions)
        if not all(callable(x) for x in (detector,recognizer,cropper)):
            raise ValueError('explicit detector, crop recognizer and rectifier required')
        self.detector,self.recognizer,self.cropper=detector,recognizer,cropper
        self.language,self.max_regions=language,max_regions
        self.closed=False; self._lock=threading.RLock()

    @staticmethod
    def capability(det_sha256,rec_sha256,*,language='en',max_regions=16):
        for digest in (det_sha256,rec_sha256):
            if type(digest) is not str or not re.fullmatch('[0-9a-f]{64}',digest):
                raise ValueError('lowercase local detector/recognizer SHA256 required')
        if language not in ('en','ch','korean') or type(max_regions) is not int or not 1<=max_regions<=32:
            raise ValueError('OCR language must be en/ch/korean and max_regions1..32')
        digest=hashlib.sha256((det_sha256+rec_sha256+language+str(max_regions)).encode()).hexdigest()
        return CapabilitySpec('vision.ppocr.mobile','vision.text','video','ppocr.v5.mobile.onnx',
            'v1.sha256.'+digest,(OutputField('lines','array'),OutputField('summary','object')),
            'frame',freshness_ms=1000,interpretation='estimate',devices=('cpu',),
            sensor_requirements=('uint8-bgr-camera',),license='code:Apache-2.0;weights:verify-selected-exports')

    @classmethod
    def from_local(cls,det_path,det_sha256,rec_path,rec_sha256,*,language='en',max_regions=16):
        cls.capability(det_sha256,rec_sha256,language=language,max_regions=max_regions)
        blobs=[]
        for name,digest in ((det_path,det_sha256),(rec_path,rec_sha256)):
            path=Path(name).expanduser()
            if not path.is_absolute() or not path.is_file(): raise ValueError('absolute existing local OCR ONNX files required')
            with path.open('rb') as source: payload=source.read(64*1024*1024+1)
            if not 0<len(payload)<=64*1024*1024 or hashlib.sha256(payload).hexdigest()!=digest:
                raise ValueError('OCR files must match SHA256 and be at most64MiB each')
            blobs.append(payload)
        try:
            import importlib.metadata
            for package,version in (('rapidocr','3.10.0'),('onnxruntime','1.23.2')):
                if importlib.metadata.version(package)!=version:
                    raise RuntimeError('select rapidocr==3.10.0 and onnxruntime==1.23.2')
            import onnxruntime as ort
            from rapidocr.ch_ppocr_det import TextDetector
            from rapidocr.ch_ppocr_rec import TextRecognizer, TextRecInput
            from rapidocr.utils.process_img import get_rotate_crop_image
            from rapidocr.utils.typings import EngineType
        except (ImportError,importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install humansense-sdk[ocr] in the selected isolated runtime') from exc
        options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
        sessions=[ort.InferenceSession(blob,sess_options=options,providers=['CPUExecutionProvider']) for blob in blobs]
        for session in sessions:
            inputs=session.get_inputs()
            if len(inputs)!=1 or inputs[0].type!='tensor(float)' or len(inputs[0].shape)!=4:
                raise ValueError('OCR exports must accept one float NCHW input')
            if len(session.get_outputs())!=1: raise ValueError('OCR exports must have one probability output')
        characters=sessions[1].get_modelmeta().custom_metadata_map.get('character')
        if type(characters) is not str or not 0<len(characters.encode('utf-8'))<=512*1024:
            raise ValueError('recognizer must embed a bounded character dictionary; no fallback download')
        chars=characters.splitlines()
        if not 1<=len(chars)<=20000 or any(not c or len(c.encode('utf-8'))>32 for c in chars):
            raise ValueError('invalid embedded recognition dictionary')
        # Direct constructors with provided sessions bypass model resolution;
        # embedded dictionary validation above prevents dictionary fallback.
        det=TextDetector(_Config(session=sessions[0],engine_type=EngineType.ONNXRUNTIME,
            limit_side_len=960,limit_type='max',mean=[.485,.456,.406],std=[.229,.224,.225],
            thresh=.3,box_thresh=.5,max_candidates=128,unclip_ratio=1.6,use_dilation=True,score_mode='fast'))
        rec=TextRecognizer(_Config(session=sessions[1],engine_type=EngineType.ONNXRUNTIME,
            lang_type=language,rec_batch_num=1,rec_img_shape=[3,48,320],rec_keys_path=None,font_path=None))
        det.session=_BoundedSession(det.session,'det')
        rec.session=_BoundedSession(rec.session,'rec',len(chars)+2)
        def detect(bgr): return det(bgr).boxes
        def recognize(crop):
            result=rec(TextRecInput(img=crop,return_word_box=False))
            if len(result.txts)!=1 or len(result.scores)!=1: raise ValueError('recognizer result count differs from crop count')
            return result.txts[0],result.scores[0]
        return cls(detect,recognize,get_rotate_crop_image,det_sha256=det_sha256,
                   rec_sha256=rec_sha256,language=language,max_regions=max_regions)

    def process(self,frame):
        with self._lock:
            if self.closed: raise RuntimeError('OCR provider is closed')
            import numpy as np
            data=frame.data
            if (frame.modality!='video' or not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or
                    data.ndim!=3 or data.shape[2]!=3 or min(data.shape[:2])<32 or
                    data.shape[0]>2160 or data.shape[1]>3840 or data.shape[0]*data.shape[1]>1920*1080 or
                    max(data.shape[:2])/min(data.shape[:2])>8):
                raise ValueError('OCR requires bounded uint8 BGR, minimum32px and aspect at most8:1')
            h,w=data.shape[:2]
            # Own the frame so optional native implementations cannot mutate capture.
            image=data.copy(); raw=self.detector(image)
            boxes=np.empty((0,4,2)) if raw is None else np.asarray(raw)
            if (boxes.ndim!=3 or boxes.shape[1:]!=(4,2) or len(boxes)>128 or boxes.dtype.kind not in 'fi' or
                    not np.isfinite(boxes).all() or (boxes<0).any() or
                    (boxes[:,:,0]>w).any() or (boxes[:,:,1]>h).any()):
                raise ValueError('detector must return at most128 in-image quadrilaterals')
            lines=[]; remaining=4096; truncated=False
            for index,box in enumerate(boxes[:self.max_regions]):
                quad=box.astype(np.float32)
                lengths=[float(np.linalg.norm(quad[a]-quad[b])) for a,b in ((0,1),(2,3),(0,3),(1,2))]
                cw,ch=int(max(lengths[:2])),int(max(lengths[2:]))
                line=dict(frame_region_index=index,quad=[[round(float(x)/w,6),round(float(y)/h,6)] for x,y in box],
                          text=None,recognition_score=None,status='unavailable_crop',text_truncated=False)
                lines.append(line)
                edges=np.roll(quad,-1,axis=0)-quad
                following=np.roll(edges,-1,axis=0)
                cross=edges[:,0]*following[:,1]-edges[:,1]*following[:,0]
                if (min(cw,ch)<2 or max(cw,ch)>2048 or cw*ch>1024*1024 or
                        max(cw,ch)/min(cw,ch)>32 or not ((cross>0).all() or (cross<0).all())): continue
                crop=self.cropper(image,quad.copy())
                if (not isinstance(crop,np.ndarray) or crop.dtype!=np.uint8 or crop.ndim!=3 or crop.shape[2]!=3 or
                        min(crop.shape[:2])<1 or max(crop.shape[:2])>2048 or crop.shape[0]*crop.shape[1]>1024*1024 or
                        crop.shape[1]/crop.shape[0]>32):
                    raise ValueError('rectified crop exceeds recognition bounds')
                text,score=self.recognizer(crop)
                if type(text) is not str or len(text)>16384 or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                    raise ValueError('recognizer must return text and a finite score0..1')
                encoded=text.encode('utf-8')
                limit=min(512,remaining)
                value=encoded[:limit].decode('utf-8',errors='ignore')
                lost=len(value.encode('utf-8'))<len(encoded)
                remaining-=len(value.encode('utf-8')); truncated|=lost
                line.update(text=value,recognition_score=float(score),status='recognized' if text else 'empty',text_truncated=lost)
            return self.spec.validate(dict(lines=lines,summary=dict(count=len(lines),detected_count=len(boxes),
                region_limit=self.max_regions,regions_omitted=max(0,len(boxes)-self.max_regions),
                candidate_limit=128,detection_exhaustive=False,text_truncated=truncated,
                text_budget_utf8_bytes=4096,coordinate_frame='normalized_image_xy',
                image_width=w,image_height=h,configured_language=self.language,language_verified=False,
                reading_order='detector_geometric_not_document_layout',orientation='quad_rectified_no_180_classifier',
                confidence_calibrated=False,temporal_tracking=False,content_role='untrusted_observed_text')))

    def close(self):
        with self._lock:
            self.closed=True
            self.detector=self.recognizer=self.cropper=None
