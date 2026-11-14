"""Capture-bound, bounded semantic windows for the explicit local SmolVLM engine."""
import hashlib
import json
import threading
import time
import uuid

from ..capabilities import CapabilitySpec,OutputField,SensorMetadata,bounded_json,identifier
from ..plugins import InputFrame
from .smolvlm_native import SmolVLMNative,configuration as native_configuration,MODEL_ID,REVISION,PROMPT,prompt_text


def configuration(config):
    config=bounded_json(config)
    if type(config) is not dict:raise ValueError('SmolVLM window configuration must be an object')
    window={k:config[k] for k in ('window_frames','max_gap_ms','sequence_step') if k in config}
    native=native_configuration({k:v for k,v in config.items() if k not in window})
    window.setdefault('window_frames',4);window.setdefault('max_gap_ms',2000);window.setdefault('sequence_step',1)
    for key,upper in (('window_frames',4),('max_gap_ms',10000),('sequence_step',10000)):
        if type(window[key]) is not int or not 1<=window[key]<=upper:raise ValueError('invalid bounded SmolVLM '+key)
    return native,window


class SmolVLMContext:
    def __init__(self,backend,**config):
        self.native,self.options=configuration(config);self.spec=self.capability(**config)
        if any(not callable(getattr(backend,k,None)) for k in ('predict','close')):raise ValueError('owned predictor and close required')
        self.backend=backend;self.lock=threading.RLock();self.closed=False;self._clear()

    @staticmethod
    def capability(**config):
        native,options=configuration(config)
        public={k:v for k,v in native.items() if not k.endswith('_path')}
        digest=hashlib.sha256(json.dumps([public,options,MODEL_ID,REVISION,prompt_text([0,1000])],sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('vision.smolvlm2_500m','vision.semantic_window','video','smolvlm2.500m.local','1.'+digest,
            (OutputField('semantic','object'),OutputField('window','object'),OutputField('model','object')),
            'window',window_ms=10000,lookahead_ms=0,freshness_ms=1000,stateful=True,reset_on_gap=False,
            interpretation='estimate',devices=(native['device'],),sensor_requirements=('uint8-bgr-camera','capture-time-source-metadata'),
            license='adapter:Apache-2.0;reference-model:Apache-2.0;substitute-weights:verify-separately')

    @classmethod
    def from_local(cls,**config):
        native,_=configuration(config);backend=SmolVLMNative.from_local(**native)
        try:return cls(backend,**config)
        except BaseException:backend.close();raise

    def _clear(self):
        self.owner=self.last=self.segment=None
        self.images=[];self.captures=[];self.provenance=None;self.announced=False

    def reset(self):
        with self.lock:
            if self.closed:raise RuntimeError('SmolVLM context closed')
            self._clear()

    def close(self):
        with self.lock:
            if not self.closed:self.closed=True;self._clear();self.backend.close()

    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('SmolVLM context closed')
            try:return self._process(frame)
            except BaseException:self._clear();raise

    def _process(self,frame):
        import numpy as np
        if not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None or not isinstance(frame.sensor,SensorMetadata):raise ValueError('video frame with source metadata required')
        identifier(frame.subject_id)
        ts=frame.timestamp_ms;sensor=frame.sensor;data=frame.data
        if type(ts) is not int or not 0<=ts<=2**53-1 or sensor.coordinate_frame is None:raise ValueError('capture milliseconds and image coordinate frame required')
        if type(data) is not np.ndarray or data.dtype!=np.uint8 or data.ndim!=3 or data.shape[2]!=3 or min(data.shape[:2])<32 or max(data.shape[:2])>2160 or data.shape[0]*data.shape[1]>1920*1080:raise ValueError('bounded plain uint8 BGR camera pixels required')
        seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        provenance=dict(subject_id=frame.subject_id,source_id=sensor.source_id,sensor_id=sensor.sensor_id,
            stream_epoch=sensor.stream_epoch,coordinate_frame=sensor.coordinate_frame,calibration_id=sensor.calibration_id,
            channels=list(sensor.channels),sequence_kind='capture' if sensor.capture_sequence is not None else 'sensor',shape=list(data.shape))
        owner=json.dumps(provenance,sort_keys=True)
        if self.owner==owner and self.last and (ts<=self.last[0] or seq<=self.last[1]):raise ValueError('capture time/sequence must strictly increase')
        changed=self.owner!=owner or (self.last is not None and (seq-self.last[1]!=self.options['sequence_step'] or ts-self.last[0]>self.options['max_gap_ms']))
        if self.captures and ts-self.captures[0]['timestamp_ms']>10000:changed=True
        if changed:self._clear()
        if self.segment is None:self.segment=uuid.uuid4().hex
        self.owner,self.last,self.provenance=owner,(ts,seq),provenance
        data=data.copy()
        self.images.append(data);self.captures.append(dict(timestamp_ms=ts,capture_sequence=seq,pixels_sha256=hashlib.sha256(data.tobytes()).hexdigest()))
        if len(self.images)==self.options['window_frames']:return self._emit(False)
        if not self.announced:
            self.announced=True
            return self._observation(None,False,None)
        return None

    def _observation(self,result,partial,elapsed):
        captures=[dict(x) for x in self.captures]
        evidence=dict(provenance=self.provenance,captures=captures)
        return self.spec.validate(dict(
            semantic=dict(status='collecting' if result is None else ('described' if result['text'] else 'empty_generation'),
                text=None if result is None else result['text'],interpretation='untrusted_semantic_hypothesis',
                quality_validated=False,confidence=None,complete_snapshot=False),
            window=dict(**self.provenance,segment_id=self.segment,captures=captures,
                capture_start_ms=captures[0]['timestamp_ms'],capture_end_ms=captures[-1]['timestamp_ms'],
                required_frames=self.options['window_frames'],observed_frames=len(captures),
                complete=len(captures)==self.options['window_frames'] and result is not None,final_partial=partial,
                max_gap_ms=self.options['max_gap_ms'],sequence_step=self.options['sequence_step'],
                evidence_sha256=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                clock_verified=False,native_stateful_video=False,additional_lookahead_ms=0),
            model=dict(id=MODEL_ID,reference_revision=REVISION,checkpoint_sha256=self.native['checkpoint_sha256'],
                tokenizer_sha256=self.native['tokenizer_sha256'],prompt_sha256=hashlib.sha256(prompt_text([c['timestamp_ms'] for c in captures]).encode()).hexdigest(),
                prompt_hash_stage='before-image-token-expansion',instruction_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                preprocessing='BGR-RGB.Lanczos.longest512.square512.no-split.mean-std0.5',
                device=self.native['device'],precision=self.native['precision'],
                input_tokens=None if result is None else result['input_tokens'],output_tokens=None if result is None else result['output_tokens'],
                output_limit_reached=None if result is None else result['output_limit_reached'],
                inference_ms=elapsed,cache_reused_across_windows=False,cost=None)))

    def _emit(self,partial):
        started=time.monotonic()
        result=bounded_json(self.backend.predict(self.images,[x['timestamp_ms'] for x in self.captures]))
        expected={'text','input_tokens','output_tokens','output_limit_reached','cache_reused_across_windows','interpretation','quality_validated'}
        if type(result) is not dict or set(result)!=expected or type(result['text']) is not str or len(result['text'].encode())>8192:raise ValueError('invalid bounded semantic result')
        if (type(result['input_tokens']) is not int or not 1<=result['input_tokens']<=1024 or
            type(result['output_tokens']) is not int or not 0<=result['output_tokens']<=self.native['max_new_tokens'] or
            type(result['output_limit_reached']) is not bool or result['cache_reused_across_windows'] is not False or
            result['quality_validated'] is not False or result['interpretation']!='untrusted_semantic_hypothesis'):raise ValueError('invalid semantic result accounting/status')
        value=self._observation(result,partial,round((time.monotonic()-started)*1000,3))
        self.images=[];self.captures=[];self.announced=True
        return value

    def finish(self):
        with self.lock:
            if self.closed:raise RuntimeError('SmolVLM context closed')
            try:return self._emit(True) if self.images else None
            finally:self._clear()
