"""Bounded YOLOE prompted detection context; native runtime is loaded explicitly."""
import hashlib
import json
import math
import re
import threading
import unicodedata

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame

ULTRALYTICS_VERSION = '8.4.212'
SOURCE_REVISION = '24f58a21f979eb90d502ac0f9400bf07c7429f6b'


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('lowercase SHA256 required')
    return value


def prompt_labels(labels):
    """Stable ordered labels, not a closed-vocabulary correctness assertion."""
    if type(labels) not in (list,tuple) or not 1 <= len(labels) <= 16:
        raise ValueError('1..16 prompt labels required')
    result = []
    for value in labels:
        if (type(value) is not str or not 1 <= len(value) <= 128 or value != value.strip() or
                any(unicodedata.category(c).startswith('C') for c in value)):
            raise ValueError('prompt labels must be bounded visible text without edge whitespace')
        label = unicodedata.normalize('NFC',value)
        if label in result:
            raise ValueError('duplicate normalized prompt label')
        result.append(label)
    return tuple(result)


def visual_examples(image, boxes, class_ids, labels):
    """Own a bounded BGR reference and explicit sequential class mapping."""
    import numpy as np
    image = _image(image)
    labels = prompt_labels(labels)
    if (type(boxes) not in (list,tuple) or type(class_ids) not in (list,tuple) or
            not 1 <= len(boxes) <= 32 or len(boxes) != len(class_ids) or
            any(type(c) is not int or not 0 <= c < len(labels) for c in class_ids) or
            set(class_ids) != set(range(len(labels)))):
        raise ValueError('1..32 example boxes with every sequential class represented required')
    h,w = image.shape[:2]
    values = []
    for box in boxes:
        if (type(box) not in (list,tuple) or len(box) != 4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in box)):
            raise ValueError('finite pixel xyxy example boxes required')
        x0,y0,x1,y1 = box
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1-x0 >= 1 and y1-y0 >= 1):
            raise ValueError('example boxes must be nonempty and inside the reference image')
        values.append([float(v) for v in box])
    meta = dict(labels=list(labels),shape=list(image.shape),boxes=values,class_ids=list(class_ids),
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return image, np.asarray(values,dtype=np.float32), np.asarray(class_ids,dtype=np.int64), digest


def _image(image):
    import numpy as np
    if (type(image) is not np.ndarray or image.dtype != np.uint8 or image.ndim != 3 or
            image.shape[2] != 3 or min(image.shape[:2]) < 32 or max(image.shape[:2]) > 2048 or
            image.shape[0]*image.shape[1] > 1048576):
        raise ValueError('plain uint8 BGR image32..2048px and at most1048576pixels required')
    return image.copy(order='C')


def mask_context(mask):
    """16x16 row-major coverage bytes, using integer partitions of original pixels.

    Each byte is round(255*foreground/cell_area). This is lossy occupancy, not a
    resized full-resolution mask or confidence map. Exact total area is separate.
    """
    import numpy as np
    if (type(mask) is not np.ndarray or mask.ndim != 2 or min(mask.shape) < 32 or
            max(mask.shape) > 2048 or mask.size > 1048576 or mask.dtype not in
            (np.dtype('bool'),np.dtype('uint8'),np.dtype('float32')) or
            not np.isfinite(mask).all() or np.any((mask != 0)&(mask != 1))):
        raise ValueError('binary original-image mask required')
    h,w = mask.shape
    # uint32 is sufficient for <=1048576 foreground pixels; no float accumulation.
    integral = np.pad(mask.astype(np.uint32).cumsum(0,dtype=np.uint32).cumsum(1,dtype=np.uint32),((1,0),(1,0)))
    ys = np.arange(17,dtype=np.int64)*h//16
    xs = np.arange(17,dtype=np.int64)*w//16
    corners = integral[np.ix_(ys,xs)].astype(np.int64)
    counts = corners[1:,1:]-corners[:-1,1:]-corners[1:,:-1]+corners[:-1,:-1]
    areas = np.diff(ys)[:,None]*np.diff(xs)[None,:]
    coverage = np.rint(counts*255/areas).astype(np.uint8)
    foreground = int(integral[-1,-1])
    return dict(grid_shape=[16,16],encoding='uint8-coverage-row-major-hex',coverage_hex=coverage.tobytes().hex(),
                partition='floor-index-original-image',foreground_pixels=foreground,
                area_fraction=foreground/(h*w),original_shape=[h,w],lossy=True)


class YOLOEContext:
    """One immutable prompt set per instance; no implicit tracking/identity.

    Native backend predict returns Nx6 pixel xyxy/score/class and NxHxW binary
    masks restored to the original input shape. A test backend is not model
    execution evidence. Worker registration is provided only with a native loader.
    """
    def __init__(self, backend, *, labels, prompt_sha256, checkpoint_sha256, mode,
                 threshold=.25, max_objects=16, artifact_verified=False):
        self.spec = self.capability(labels=labels,prompt_sha256=prompt_sha256,
            checkpoint_sha256=checkpoint_sha256,mode=mode,threshold=threshold,max_objects=max_objects)
        if not callable(getattr(backend,'predict',None)) or not callable(getattr(backend,'close',None)):
            raise ValueError('owned YOLOE backend with predict/close required')
        if type(artifact_verified) is not bool:
            raise ValueError('artifact_verified must be boolean')
        self.backend, self.labels, self.mode = backend, prompt_labels(labels), mode
        self.prompt_hash, self.checkpoint_hash = prompt_sha256, checkpoint_sha256
        self.threshold, self.max_objects, self.verified = threshold,max_objects,artifact_verified
        self._lock = threading.RLock()
        self._closed = False
        self._owner = self._last = None

    @staticmethod
    def capability(*, labels, prompt_sha256, checkpoint_sha256, mode, threshold=.25, max_objects=16):
        labels = prompt_labels(labels); _hash(prompt_sha256); _hash(checkpoint_sha256)
        if mode not in ('text','exemplar'):
            raise ValueError('prompt mode must be text or exemplar')
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('confidence threshold must be0..1 exclusive of0')
        if type(max_objects) is not int or not 1 <= max_objects <= 16:
            raise ValueError('max_objects must be1..16')
        version = hashlib.sha256(json.dumps([labels,prompt_sha256,checkpoint_sha256,mode,threshold,max_objects]).encode()).hexdigest()
        return CapabilitySpec('vision.yoloe.prompted','vision.prompted_objects','video','yoloe.local','1.'+version,
            (OutputField('objects','array'),OutputField('summary','object')),'frame',stateful=True,
            reset_on_gap=True,freshness_ms=1000,interpretation='estimate',devices=('cpu','cuda'),
            sensor_requirements=('uint8-bgr-camera','declared-image-coordinate-frame'),
            license='adapter:Apache-2.0;Ultralytics:AGPL-3.0-or-commercial;weights:verify-separately')

    def reset(self):
        with self._lock:
            self._owner = self._last = None

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._owner = self._last = None
                backend,self.backend = self.backend,None
                backend.close()

    def process(self, frame):
        with self._lock:
            if self._closed: raise RuntimeError('YOLOE provider closed')
            try: return self._process(frame)
            except Exception:
                self._owner = self._last = None
                raise

    def _process(self, frame):
        import numpy as np
        if (not isinstance(frame,InputFrame) or frame.modality != 'video' or
                not isinstance(frame.sensor,SensorMetadata) or not frame.sensor.coordinate_frame or
                type(frame.timestamp_ms) is not int or not 0 <= frame.timestamp_ms <= 2**53-1):
            raise ValueError('video capture time and declared image coordinates required')
        identifier(frame.subject_id)
        image = _image(frame.data)
        h,w = image.shape[:2]
        sensor,ts = frame.sensor,frame.timestamp_ms
        seq = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner = (frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,
                 sensor.coordinate_frame,sensor.calibration_id,sensor.capture_sequence is not None)
        if owner == self._owner and self._last and (ts <= self._last[0] or seq <= self._last[1]):
            raise ValueError('capture sequence and timestamp must advance')
        boxes,masks = self.backend.predict(image)
        if (type(boxes) is not np.ndarray or boxes.dtype != np.float32 or boxes.ndim != 2 or
                boxes.shape[1:] != (6,) or len(boxes) > self.max_objects or
                not np.isfinite(boxes).all()):
            raise ValueError('bounded Nx6 float32 YOLOE boxes required')
        if (type(masks) is not np.ndarray or masks.shape != (len(boxes),h,w) or
                masks.dtype not in (np.dtype('bool'),np.dtype('uint8'),np.dtype('float32'))):
            raise ValueError('one binary original-image mask per detection required')
        objects = []
        for index, (x0,y0,x1,y1,score,cls) in enumerate(boxes):
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and 0 <= score <= 1 and
                    cls == int(cls) and 0 <= cls < len(self.labels)):
                raise ValueError('invalid YOLOE coordinates, score or class mapping')
            mask = mask_context(masks[index])  # Validate even below-threshold native outputs.
            if score < self.threshold: continue
            objects.append(dict(label=self.labels[int(cls)],class_id=int(cls),score=float(score),
                bbox=dict(x=float(x0/w),y=float(y0/h),width=float((x1-x0)/w),height=float((y1-y0)/h)),
                mask=mask,identity_verified=False))
        objects.sort(key=lambda item:-item['score'])
        result = self.spec.validate(dict(objects=objects,summary=dict(
            prompt_mode=self.mode,labels=list(self.labels),prompt_sha256=self.prompt_hash,
            checkpoint_sha256=self.checkpoint_hash,artifact_integrity_verified=self.verified,
            model_quality_verified=False,identity_verified=False,tracked=False,
            capture_timestamp_ms=ts,coordinate_frame=sensor.coordinate_frame,
            input_shape=[h,w],box_units='normalized_image',returned_count=len(objects),
            native_returned_count=len(boxes),max_objects=self.max_objects,
            native_limit_reached=len(boxes)==self.max_objects,complete_scene=False)))
        self._owner,self._last = owner,(ts,seq)
        return result
