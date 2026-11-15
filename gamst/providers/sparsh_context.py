"""Capture-bound Sparsh slip windows; no contact or physical force measurement."""
from collections import deque
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

from ..capabilities import CapabilitySpec, OutputField, bounded_json, identifier
from .sparsh_native import SparshSlipNative, GRAPH, REVISION
from .tactile import TactileWindow, _bound_frame, _shape, _sha
from .vocal_expression import _read

PREPROCESS_SOURCE_SHA256 = '8bfc5e3c255ec7617236e41ad6112e001d59badf7da94b2fe81dbf20d6b5103c'

PROFILE_FIELDS = frozenset(('profile_id', 'source_id', 'sensor_id', 'sensor_model', 'serial',
    'width', 'height', 'color_order', 'calibration_id', 'illumination_id', 'coordinate_frame',
    'clock_id', 'reference_sha256', 'orientation', 'training_config_sha256', 'training_domain_id',
    'min_interval_ms', 'max_interval_ms', 'min_pair_span_ms', 'max_pair_span_ms', 'reference_consent'))
BIND_FIELDS = ('source_id', 'sensor_id', 'sensor_model', 'serial', 'width', 'height', 'color_order',
               'calibration_id', 'illumination_id', 'coordinate_frame', 'clock_id', 'reference_sha256')


def profile_value(profile):
    value = bounded_json(profile, max_bytes=8192)
    if type(value) is not dict or set(value) != PROFILE_FIELDS:
        raise ValueError('exact tactile task profile required')
    for key in PROFILE_FIELDS - {'width','height','min_interval_ms','max_interval_ms',
        'min_pair_span_ms','max_pair_span_ms','reference_consent'}:
        identifier(value[key], key)
    _sha(value['reference_sha256']); _sha(value['training_config_sha256'])
    _shape(value['width'],value['height'])
    w,h=value['width'],value['height']
    orientation='landscape_cw' if w>h else 'portrait'
    if max(w,h)*3 != min(w,h)*4 or value['orientation'] != orientation:
        raise ValueError('only declared 4:3 landscape-clockwise or 3:4 portrait is supported')
    if value['sensor_model'] not in ('digit','gelsight_mini') or value['color_order'] not in ('RGB','BGR'):
        raise ValueError('unsupported tactile sensor or color order')
    if value['reference_consent'] is not True:
        raise ValueError('explicit local reference-use consent required')
    for key in ('min_interval_ms','max_interval_ms','min_pair_span_ms','max_pair_span_ms'):
        if type(value[key]) is not int or not 1 <= value[key] <= 5000:
            raise ValueError('bounded positive cadence interval required')
    lo,hi=value['min_interval_ms'],value['max_interval_ms']
    if lo>hi or hi>1000 or not 5*lo <= value['min_pair_span_ms'] <= value['max_pair_span_ms'] <= 5*hi:
        raise ValueError('pair span must lie within five declared adjacent intervals')
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configuration(config):
    required={'source_root','encoder_path','head_path','encoder_sha256','head_sha256','reference_path','profile'}
    if type(config) is not dict or set(config)!=required:
        raise ValueError('exact local Sparsh context configuration required')
    for key in ('source_root','encoder_path','head_path','reference_path'):
        if type(config[key]) is not str or not Path(config[key]).is_absolute():
            raise ValueError('absolute supplied local paths required')
    return {key:config[key] for key in ('encoder_sha256','head_sha256','profile')}


def preprocess_pair(current, previous, reference, profile):
    """Official training helper parity for verified aspect ratios, no enhancement."""
    import numpy as np
    from PIL import Image
    p=profile_value(profile)
    count=_shape(p['width'],p['height'])
    if any(type(item) is not bytes or len(item)!=count for item in (current,previous,reference)):
        raise ValueError('immutable geometry-bound raw image/reference bytes required')
    if hashlib.sha256(reference).hexdigest()!=p['reference_sha256']:
        raise ValueError('reference hash mismatch')
    shape=(p['height'],p['width'],3)
    ref=np.frombuffer(reference,dtype=np.uint8).reshape(shape)
    if p['color_order']=='BGR':ref=ref[:,:,::-1]
    outputs=[]
    for payload in (current,previous):
        pixels=np.frombuffer(payload,dtype=np.uint8).reshape(shape)
        if p['color_order']=='BGR':pixels=pixels[:,:,::-1]
        diff=np.clip((pixels.astype(np.int32)-ref.astype(np.int32))/255.0+0.5,0,1)
        diff=(diff*255.0).astype(np.uint8)
        if p['orientation']=='landscape_cw':diff=np.rot90(diff,k=-1)
        # torchvision Resize on a PIL RGB image uses PIL bilinear antialiasing.
        resized=Image.fromarray(diff).resize((240,320),resample=Image.Resampling.BILINEAR)
        outputs.append(np.asarray(resized,dtype=np.float32).transpose(2,0,1)/255.0)
    return np.concatenate(outputs,axis=0)[None].copy()


class SparshSlipContext:
    def __init__(self, native, reference, *, encoder_sha256, head_sha256, profile):
        self.spec=self.capability(encoder_sha256=encoder_sha256,head_sha256=head_sha256,profile=profile)
        self._profile=profile_value(profile)
        if type(reference) is not bytes or len(reference)!=_shape(self._profile['width'],self._profile['height']) or hashlib.sha256(reference).hexdigest()!=self._profile['reference_sha256']:
            raise ValueError('profile-bound immutable local reference required')
        if any(not callable(getattr(native,k,None)) for k in ('predict','close')):
            raise ValueError('owned native predictor required')
        self._native=native;self._reference=reference
        self._model=dict(graph=GRAPH,source_revision=REVISION,encoder_sha256=encoder_sha256,head_sha256=head_sha256)
        self._profile_sha=_hash(self._profile)
        self._verified=False
        self._lock=threading.RLock();self.closed=False
        self.window=TactileWindow(max_gap_ms=self._profile['max_interval_ms'])
        self._clear()

    @staticmethod
    def capability(*, encoder_sha256, head_sha256, profile):
        _sha(encoder_sha256);_sha(head_sha256);p=profile_value(profile)
        digest=_hash([REVISION,GRAPH,encoder_sha256,head_sha256,p,'training-rgbdiff-pil-bilinear-v1',0.4,3])
        return CapabilitySpec('tactile.sparsh.slip','tactile.slip_context','tactile','sparsh.slip.local','1.'+digest,
            (OutputField('classification','object'),OutputField('evidence','object')),
            'window',window_ms=p['max_pair_span_ms'],lookahead_ms=0,
            freshness_ms=p['max_pair_span_ms'],stateful=True,reset_on_gap=True,
            interpretation='classification',devices=('cpu',),
            sensor_requirements=('bound-tactile-packets','declared-training-domain','hash-bound-reference'),
            license='adapter:Apache-2.0;upstream:CC-BY-NC-4.0-and-per-file;weights:caller-supplied')

    @classmethod
    def from_local(cls, source_root, encoder_path, head_path, *, encoder_sha256, head_sha256, reference_path, profile):
        cls.capability(encoder_sha256=encoder_sha256,head_sha256=head_sha256,profile=profile)
        configuration(dict(source_root=source_root,encoder_path=encoder_path,head_path=head_path,
            encoder_sha256=encoder_sha256,head_sha256=head_sha256,reference_path=reference_path,profile=profile))
        p=profile_value(profile)
        reference=_read(reference_path,_shape(p['width'],p['height']),p['reference_sha256'])
        if len(reference)!=_shape(p['width'],p['height']):raise ValueError('reference size mismatch')
        native=SparshSlipNative.from_local(source_root,encoder_path,head_path,
            encoder_sha256=encoder_sha256,head_sha256=head_sha256)
        try:
            result=cls(native,reference,encoder_sha256=encoder_sha256,head_sha256=head_sha256,profile=p)
            result._verified=True
            return result
        except BaseException:
            native.close()
            raise

    def _clear(self):
        self.window.reset();self._epoch=None;self._segment=uuid.uuid4().hex
        self._decisions=deque([False]*3,maxlen=3)
        self._decision_times=deque(maxlen=3)
        self._decision_evidence=deque(maxlen=3)

    def reset(self):
        with self._lock:
            if self.closed:raise RuntimeError('Sparsh context closed')
            self._clear()

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed=True;self._clear();self._reference=b''
                self._native.close()

    def process(self, frame):
        with self._lock:
            if self.closed:raise RuntimeError('Sparsh context closed')
            try:return self._process(frame)
            except BaseException:
                self._clear()
                raise

    def _process(self, frame):
        import numpy as np
        meta,_=_bound_frame(frame)
        if any(meta[k]!=self._profile[k] for k in BIND_FIELDS):
            raise ValueError('frame/reference/training profile misalignment')
        epoch=frame.sensor.stream_epoch
        reset_reason=None
        if self.window.frames and epoch!=self._epoch:
            self._clear();reset_reason='runtime_epoch_change'
        if self.window.frames:
            last=self.window.frames[-1]
            interval=frame.timestamp_ms-last.timestamp_ms
            # Changed source identity is handled by TactileWindow. Cadence may
            # invalidate a same-owner history, never resample or pad it.
            if interval>0 and interval<self._profile['min_interval_ms']:
                self._clear();reset_reason='cadence_outside_profile'
        meta,pair,window_reason=self.window.push(frame)
        self._epoch=epoch
        reason=reset_reason or window_reason
        if reason:
            self._segment=uuid.uuid4().hex
            self._decisions=deque([False]*3,maxlen=3);self._decision_times.clear();self._decision_evidence.clear()
        if pair and not self._profile['min_pair_span_ms'] <= pair[0].timestamp_ms-pair[1].timestamp_ms <= self._profile['max_pair_span_ms']:
            self._clear();self.window.push(frame);self._epoch=epoch
            pair=None;reason='pair_span_outside_profile'
        evidence=dict(self._model,profile_sha256=self._profile_sha,profile_id=self._profile['profile_id'],
            training_config_sha256=self._profile['training_config_sha256'],training_domain_id=self._profile['training_domain_id'],
            source_id=meta['source_id'],sensor_id=meta['sensor_id'],source_epoch=meta['source_epoch'],runtime_epoch=epoch,
            state_segment=self._segment,clock_id=meta['clock_id'],calibration_id=meta['calibration_id'],illumination_id=meta['illumination_id'],
            reference_sha256=meta['reference_sha256'],current_timestamp_ms=meta['timestamp_ms'],
            current_capture_sequence=meta['sequence'],current_pixels_sha256=meta['pixels_sha256'],
            pair_timestamps_ms=[],pair_capture_sequences=[],pair_pixels_sha256=[],
            capture_pair_span_ms=None,preprocessing_ms=None,inference_ms=None,
            buffered_captures=len(self.window.frames),required_captures=6,reset_reason=reason,
            source_stride_frames=5,frame_order='current-previous',preprocessing='rgbdiff-pil-bilinear-v1',
            preprocessing_source_sha256=PREPROCESS_SOURCE_SHA256,
            input_binding={key:self._profile[key] for key in (*BIND_FIELDS,'orientation')},
            acquisition=meta['acquisition'],backend=meta['backend'],backend_version=meta['backend_version'],
            backend_revision=meta['backend_revision'],pair_read_intervals_ms=[],
            native_artifacts_verified=self._verified,trained_quality_verified=False,
            reference_no_contact_verified=False,physical_calibration_verified=False,
            training_profile_verified=False,sensor_domain_verified=False,clock_alignment_verified=False,
            complete_snapshot=False,debounce_evidence=[])
        classification=dict(status='warming',labels=['no_slip','slip'],slip_logits=None,slip_probability=None,
            confidence_calibrated=False,candidate_slip=None,debounced_slip=None,
            threshold=0.4,comparison='strict_greater',required_consecutive_decisions=3,
            decision_timestamps_ms=[],normalized_delta_force=None,force_unit='unitless-normalized-delta',
            contact_state='not_assessed')
        if pair:
            decoded=[_bound_frame(item) for item in pair]
            started=time.perf_counter_ns()
            data=preprocess_pair(decoded[0][1],decoded[1][1],self._reference,self._profile)
            ready=time.perf_counter_ns()
            output=self._native.predict(data)
            finished=time.perf_counter_ns()
            if type(output) is not dict or set(output)!={'slip_logits','normalized_delta_force'}:
                raise ValueError('invalid Sparsh native result')
            for key,shape in (('slip_logits',(2,)),('normalized_delta_force',(3,))):
                values=output[key]
                if type(values) is not np.ndarray or values.dtype!=np.float32 or values.shape!=shape or not np.isfinite(values).all():
                    raise ValueError('invalid Sparsh native values')
            if np.max(np.abs(output['normalized_delta_force']))>1:raise ValueError('normalized force outside bounds')
            logits=output['slip_logits'].astype(np.float64)
            probs=np.exp(logits-logits.max());probs/=probs.sum()
            candidate=bool(probs[1]>0.4)
            self._decisions.append(candidate);self._decision_times.append(meta['timestamp_ms'])
            self._decision_evidence.append(dict(
                pair_timestamps_ms=[m['timestamp_ms'] for m,_ in decoded],
                pair_capture_sequences=[m['sequence'] for m,_ in decoded],
                pair_pixels_sha256=[m['pixels_sha256'] for m,_ in decoded],
                candidate_slip=candidate,slip_probability=float(probs[1])))
            classification.update(status='estimated',slip_logits=logits.tolist(),slip_probability=float(probs[1]),
                candidate_slip=candidate,debounced_slip=all(self._decisions),
                decision_timestamps_ms=list(self._decision_times),normalized_delta_force=output['normalized_delta_force'].tolist())
            evidence.update(pair_timestamps_ms=[m['timestamp_ms'] for m,_ in decoded],
                pair_capture_sequences=[m['sequence'] for m,_ in decoded],
                pair_pixels_sha256=[m['pixels_sha256'] for m,_ in decoded],
                pair_read_intervals_ms=[[m['read_start_ms'],m['read_end_ms']] for m,_ in decoded],
                capture_pair_span_ms=pair[0].timestamp_ms-pair[1].timestamp_ms,
                preprocessing_ms=(ready-started)/1e6,inference_ms=(finished-ready)/1e6,
                debounce_evidence=list(self._decision_evidence))
        return self.spec.validate(dict(classification=classification,evidence=evidence))
