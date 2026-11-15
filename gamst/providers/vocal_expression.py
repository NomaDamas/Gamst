"""Trailing-window emotion2vec seed ONNX features plus declared checkpoint head."""
import hashlib
import json
import math
from pathlib import Path
import re
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame

SOURCE_SHA256='327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3'
LABELS=('angry','disgusted','fearful','happy','neutral','other','sad','surprised','unknown')


def _digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('lowercase SHA256 required')
    return value


def _read(path,limit,digest):
    _digest(digest)
    with Path(path).expanduser().open('rb') as source:payload=source.read(limit+1)
    if not 0<len(payload)<=limit or hashlib.sha256(payload).hexdigest()!=digest:
        raise ValueError('local artifact size or SHA256 mismatch')
    return payload


class Emotion2VecContext:
    def __init__(self,predictor,*,model_sha256,head_sha256,window_ms=2000,hop_ms=500):
        self.spec=self.capability(model_sha256=model_sha256,head_sha256=head_sha256,window_ms=window_ms,hop_ms=hop_ms)
        if not callable(predictor):raise ValueError('score predictor required')
        import numpy as np
        self.np,self.predictor=np,predictor
        self.window_ms,self.hop_ms=window_ms,hop_ms
        self._lock=threading.RLock();self.closed=False;self._clear()

    @staticmethod
    def capability(*,model_sha256,head_sha256,window_ms=2000,hop_ms=500):
        _digest(model_sha256);_digest(head_sha256)
        if type(window_ms) is not int or not 1000<=window_ms<=4000:
            raise ValueError('window_ms must be1000..4000')
        if type(hop_ms) is not int or not 100<=hop_ms<=window_ms:
            raise ValueError('hop_ms must be100..window_ms')
        version='1.'+hashlib.sha256(json.dumps([model_sha256,head_sha256,window_ms,hop_ms]).encode()).hexdigest()
        return CapabilitySpec('audio.emotion2vec.seed','audio.vocal_expression','audio','emotion2vec.seed.onnx',version,
            (OutputField('classification','object'),),'window',window_ms=window_ms,stateful=True,reset_on_gap=True,
            freshness_ms=window_ms,interpretation='classification',devices=('cpu',),
            sensor_requirements=('16000hz-mono-pcm','verified-local-feature-graph-and-head'),
            license='weights:model-license-unverified;adapter:Apache-2.0;ONNXRuntime:MIT')

    @classmethod
    def from_local(cls,model_path,head_path,**config):
        cls.capability(**config)
        model=_read(model_path,512*1024*1024,config['model_sha256'])
        head_bytes=_read(head_path,512*1024,config['head_sha256'])
        import numpy as np
        head=json.loads(head_bytes)
        if type(head) is not dict or set(head)!={'source_checkpoint_sha256','labels','weight','bias'} or head['source_checkpoint_sha256']!=SOURCE_SHA256 or head['labels']!=list(LABELS):
            raise ValueError('head must declare pinned source checkpoint and exact seed label order')
        if type(head['weight']) is not list or len(head['weight'])!=9 or any(type(row) is not list or len(row)!=768 for row in head['weight']) or type(head['bias']) is not list or len(head['bias'])!=9:
            raise ValueError('head must be9x768 weights and9 biases')
        if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for row in [*head['weight'],head['bias']] for v in row):
            raise ValueError('head parameters must be finite bounded numbers')
        weight=np.asarray(head['weight'],dtype=np.float64);bias=np.asarray(head['bias'],dtype=np.float64)
        try:
            from importlib.metadata import version
            if version('onnxruntime')!='1.23.2':raise RuntimeError('requires onnxruntime==1.23.2')
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('Install optional humansense-sdk[vocal-expression] in selected runtime') from exc
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
        ins,outs=session.get_inputs(),session.get_outputs()
        if (len(ins)!=1 or ins[0].name!='input' or ins[0].type!='tensor(float)' or len(ins[0].shape)!=2 or
            ins[0].shape[0] not in (1,None,'batch_size') or isinstance(ins[0].shape[1],int) or
            len(outs)!=1 or outs[0].name!='output' or outs[0].type!='tensor(float)' or len(outs[0].shape)!=3 or outs[0].shape[2]!=768):
            raise ValueError('expected dynamic batch1 input and frame features output[1,time,768]')
        metadata=session.get_modelmeta().custom_metadata_map
        if metadata.get('gamst.source_checkpoint_sha256')!=SOURCE_SHA256 or metadata.get('gamst.export_contract')!='emotion2vec_seed_features_v1':
            raise ValueError('ONNX must declare pinned checkpoint and emotion2vec_seed_features_v1 export contract')
        def predict(samples):
            # Official feature export includes whole-window layer normalization; do not normalize twice.
            features=session.run(['output'],{'input':samples.reshape(1,-1).copy()})[0]
            if type(features) is not np.ndarray or features.dtype!=np.float32 or features.ndim!=3 or features.shape[0]!=1 or not 1<=features.shape[1]<=256 or features.shape[2]!=768 or not np.isfinite(features).all() or np.max(np.abs(features))>1e6:
                raise ValueError('invalid emotion2vec feature grid')
            logits=weight@np.mean(features[0],axis=0,dtype=np.float64)+bias
            logits-=np.max(logits);scores=np.exp(logits);scores/=scores.sum()
            return scores
        return cls(predict,**config)

    def _clear(self):
        self._buffer=self.np.empty(0,dtype=self.np.float32)
        self._owner=self._last=self._expected_end=None;self._since=0;self._emitted=False

    def reset(self):
        with self._lock:self._clear()

    def close(self):
        with self._lock:
            self._clear();self.predictor=None;self.closed=True

    def process(self,frame):
        with self._lock:
            if self.closed:raise RuntimeError('vocal expression provider is closed')
            try:return self._process(frame)
            except Exception:
                self._clear();raise

    def _process(self,frame):
        np=self.np
        if not isinstance(frame,InputFrame) or frame.modality!='audio' or type(frame.sample_rate) is not int or frame.sample_rate!=16000 or not isinstance(frame.sensor,SensorMetadata):
            raise ValueError('explicit16000Hz mono audio InputFrame and metadata required')
        identifier(frame.subject_id);sensor=frame.sensor;ts=frame.timestamp_ms
        if type(ts) is not int or not 0<=ts<=2**53-1 or len(sensor.channels)>1:
            raise ValueError('integer capture end timestamp and mono channel metadata required')
        data=frame.data
        if type(data) is bytes:
            if not 0<len(data)<=32000 or len(data)%2:raise ValueError('PCM16 block must be1..16000 mono samples')
            samples=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768
            clipped=bool(np.any(np.frombuffer(data,dtype='<i2')==-32768) or np.any(np.frombuffer(data,dtype='<i2')==32767))
        else:
            if type(data) is not np.ndarray or data.dtype not in (np.dtype('int16'),np.dtype('float32')) or data.ndim not in (1,2) or data.ndim==2 and data.shape[1]!=1 or not 1<=len(data)<=16000:
                raise ValueError('plain mono int16/float32 array of at most16000 samples required')
            clipped=bool(np.any(data==-32768) or np.any(data==32767)) if data.dtype==np.int16 else bool(np.any(np.abs(data)>=1))
            samples=data.reshape(-1).astype(np.float32,copy=True)
            if data.dtype==np.int16:samples/=32768
        if not np.isfinite(samples).all() or np.max(np.abs(samples))>1:raise ValueError('PCM must be finite and normalized')
        if len(samples)%16:raise ValueError('PCM blocks must contain whole milliseconds (multiples of16 samples)')
        duration=len(samples)//16
        if ts<duration:raise ValueError('capture end precedes block duration')
        seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.channels,
               sensor.calibration_id,sensor.coordinate_frame,sensor.capture_sequence is not None)
        same=owner==self._owner
        if same and self._last and (ts<=self._last[0] or seq<=self._last[1]):raise ValueError('capture time and sequence must increase')
        contiguous=bool(same and self._last and seq==self._last[1]+1 and ts-duration==self._expected_end)
        if not contiguous:self._clear()
        expected_end=self._expected_end+duration if contiguous else ts
        size=self.window_ms*16
        self._buffer=np.concatenate((self._buffer,samples))[-size:].copy()
        self._since+=len(samples)
        self._owner=owner;self._last=(ts,seq);self._expected_end=expected_end
        if len(self._buffer)<size:return None
        if self._emitted and self._since<self.hop_ms*16:return None
        window=self._buffer.copy();rms=float(np.sqrt(np.mean(window.astype(np.float64)**2)))
        result=dict(status='insufficient_energy' if rms<1e-4 else 'classified',scores=[],
            start_ms=ts-self.window_ms,end_ms=ts,window_ms=self.window_ms,hop_ms=self.hop_ms,
            sample_rate=16000,rms=rms,last_block_clipped=clipped,inner_emotion_verified=False,
            speaker_identity_verified=False,source_checkpoint_sha256=SOURCE_SHA256,
            source_provenance='artifact-metadata-declaration',export_parity_verified=False,
            normalization='whole-window-in-graph',native_recurrent_state=False)
        if rms>=1e-4:
            scores=np.asarray(self.predictor(window))
            if scores.shape!=(9,) or scores.dtype.kind not in 'fi' or not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1) or abs(float(scores.sum())-1)>1e-4:
                raise ValueError('predictor must return9 normalized probabilities')
            result['scores']=[dict(label=label,score=float(score)) for label,score in zip(LABELS,scores)]
        output=self.spec.validate({'classification':result})
        self._since=0;self._emitted=True
        return output
