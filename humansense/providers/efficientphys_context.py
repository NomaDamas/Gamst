"""Capture-bound 30fps face windows for local EfficientPhys context."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
import uuid

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from .efficientphys_native import EfficientPhysNative, REVISION
from .region_appearance import capture_binding
from .vocal_expression import _digest

MAGIC = b'GRPP001\n'
HEADER_LIMIT = 8192
PIXELS = 72*72*3


def _provenance(frame):
    if not isinstance(frame, InputFrame): raise ValueError('InputFrame required')
    sensor = frame.sensor
    if not isinstance(frame, InputFrame) or frame.modality != 'video' or not isinstance(sensor, SensorMetadata):
        raise ValueError('video InputFrame and sensor provenance required')
    identifier(frame.subject_id)
    if type(frame.timestamp_ms) is not int or not 0 <= frame.timestamp_ms <= 2**53-1:
        raise ValueError('bounded capture timestamp required')
    if not sensor.coordinate_frame: raise ValueError('declared image coordinate frame required')
    return dict(subject_id=frame.subject_id, timestamp_ms=frame.timestamp_ms,
        source_id=sensor.source_id, sensor_id=sensor.sensor_id,
        coordinate_frame=sensor.coordinate_frame, calibration_id=sensor.calibration_id,
        channels=list(sensor.channels),
        sequence_kind='capture' if sensor.capture_sequence is not None else 'sensor',
        capture_sequence=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence)


def face_frame(frame, *, track_id, box, detector_id, detector_version, clock_id):
    """Bind one caller-selected face (or explicit loss) to original BGR capture.

    box is integer original-image xyxy. No automatic detection/identity claim.
    Returned bytes can be submitted directly to SensorStreams/ProcessProvider.
    """
    import numpy as np
    identifier(detector_id); identifier(detector_version); identifier(clock_id)
    # Validate before copying a potentially oversized array.
    capture_binding(frame)
    frame = replace(frame, data=np.array(frame.data, copy=True, order='C'))
    binding = capture_binding(frame)
    meta = dict(provenance=_provenance(frame), track_id=track_id, box=box,
        detector_id=detector_id, detector_version=detector_version, clock_id=clock_id,
        shape=binding['shape'], capture_sha256=binding['sha256'])
    if box is None:
        if track_id is not None: raise ValueError('face loss must clear track and box')
        payload = b''
    else:
        identifier(track_id)
        h,w = frame.data.shape[:2]
        if (type(box) is not list or len(box) != 4 or any(type(v) is not int for v in box) or
                not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h or
                min(box[2]-box[0],box[3]-box[1]) < 16):
            raise ValueError('integer in-image xyxy face at least16px required')
        import cv2
        rgb = frame.data[box[1]:box[3],box[0]:box[2],::-1]
        payload = cv2.resize(rgb,(72,72),interpolation=cv2.INTER_AREA).tobytes()
    meta['crop_sha256'] = hashlib.sha256(payload).hexdigest()
    header = json.dumps(bounded_json(meta,max_bytes=HEADER_LIMIT),separators=(',',':')).encode()
    return replace(frame,data=MAGIC+struct.pack('!I',len(header))+header+payload)


def decode(frame):
    import numpy as np
    provenance = _provenance(frame)
    packet = frame.data
    if type(packet) is not bytes or not 12 <= len(packet) <= 12+HEADER_LIMIT+PIXELS or packet[:8] != MAGIC:
        raise ValueError('bounded gamst rPPG face packet required')
    n = struct.unpack('!I',packet[8:12])[0]
    if not 1 <= n <= HEADER_LIMIT or len(packet) < 12+n: raise ValueError('invalid face header size')
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('duplicate face metadata')
            result[key] = value
        return result
    try: meta = bounded_json(json.loads(packet[12:12+n],object_pairs_hook=pairs),max_bytes=HEADER_LIMIT)
    except (UnicodeError,RecursionError,json.JSONDecodeError) as exc: raise ValueError('invalid face metadata') from exc
    keys = {'provenance','track_id','box','detector_id','detector_version','clock_id','shape','capture_sha256','crop_sha256'}
    if type(meta) is not dict or set(meta) != keys or json.dumps(meta['provenance'],sort_keys=True) != json.dumps(provenance,sort_keys=True):
        raise ValueError('face provenance disagrees with outer capture')
    for key in ('detector_id','detector_version','clock_id'): identifier(meta[key])
    for key in ('capture_sha256','crop_sha256'): _digest(meta[key])
    shape = meta['shape']
    if (type(shape) is not list or len(shape) != 3 or any(type(x) is not int for x in shape) or
            shape[2] != 3 or min(shape[:2]) < 32 or shape[0] > 2160 or shape[1] > 3840 or shape[0]*shape[1] > 1920*1080):
        raise ValueError('invalid original image geometry')
    payload = packet[12+n:]
    if hashlib.sha256(payload).hexdigest() != meta['crop_sha256']: raise ValueError('face payload hash mismatch')
    box = meta['box']
    if box is None:
        if meta['track_id'] is not None or payload: raise ValueError('face loss must contain no track or pixels')
        return meta,None
    identifier(meta['track_id'])
    if (type(box) is not list or len(box) != 4 or any(type(v) is not int for v in box) or
            not 0 <= box[0] < box[2] <= shape[1] or not 0 <= box[1] < box[3] <= shape[0] or
            min(box[2]-box[0],box[3]-box[1]) < 16 or len(payload) != PIXELS):
        raise ValueError('invalid face geometry or pixel count')
    return meta,np.frombuffer(payload,np.uint8).reshape(72,72,3).copy()


def configuration(config):
    required = {'source_path','checkpoint_path','checkpoint_sha256'}
    if (type(config) is not dict or not required <= set(config) or set(config)-required-{'device'} or
            any(type(config[k]) is not str or not Path(config[k]).is_absolute() for k in ('source_path','checkpoint_path'))):
        raise ValueError('rPPG requires absolute local source/checkpoint paths and hash')
    return {k:v for k,v in config.items() if k not in ('source_path','checkpoint_path')}


def spectral_candidate(values, fps):
    """Window-only exploratory spectral feature, NOT validated pulse rate.

    Integrate model differences, remove linear trend, Hann taper, select largest
    FFT-bin power within0.75..2.5Hz. No zero padding or precision interpolation.
    This is intentionally distinct from upstream full-clip smoothness-prior DSP.
    """
    import numpy as np
    values = np.asarray(values,dtype=np.float64)
    if values.shape != (180,) or not np.isfinite(values).all() or not 29 <= fps <= 31:
        raise ValueError('180 finite rPPG estimates and30fps cadence required')
    signal = np.cumsum(values)
    x = np.arange(180,dtype=np.float64);x -= x.mean()
    signal -= signal.mean()+x*np.dot(x,signal)/np.dot(x,x)
    power = np.abs(np.fft.rfft(signal*np.hanning(180)))**2
    freq = np.fft.rfftfreq(180,d=1/fps)
    selected = (freq >= .75)&(freq <= 2.5)
    band = power[selected]
    total = float(band.sum())
    if not math.isfinite(total): raise ValueError('rPPG spectral overflow')
    if total <= 1e-12: return None
    i = int(band.argmax())
    return dict(frequency_hz=float(freq[selected][i]), equivalent_bpm=float(freq[selected][i]*60),
        bin_width_bpm=fps/180*60, band_power_fraction=float(band[i]/total),
        interpretation='unvalidated_spectral_candidate', confidence=None)


class EfficientPhysContext:
    def __init__(self, backend, *, checkpoint_sha256, device='cpu'):
        self.spec = self.capability(checkpoint_sha256=checkpoint_sha256,device=device)
        if any(not callable(getattr(backend,k,None)) for k in ('predict','close')):
            raise ValueError('owned native predictor and close required')
        self.backend = backend
        self.verified = bool(getattr(backend,'artifacts_verified',False))
        self.lock = threading.RLock(); self.closed = False
        self._clear()

    @staticmethod
    def capability(*,checkpoint_sha256,device='cpu'):
        _digest(checkpoint_sha256)
        if type(device) is not str or device not in ('cpu','cuda'): raise ValueError('cpu or cuda required')
        digest = hashlib.sha256(json.dumps([REVISION,checkpoint_sha256,device,'window-standardized-180-30-v1']).encode()).hexdigest()
        return CapabilitySpec('vision.efficientphys','vision.rppg_context','video','efficientphys.local','1.'+digest,
            (OutputField('signal','object'),OutputField('summary','object')),
            'window',window_ms=6000,lookahead_ms=0,freshness_ms=6000,stateful=True,reset_on_gap=True,
            interpretation='estimate',devices=(device,),sensor_requirements=('capture-bound-face-packet','30fps-declared-same-clock'),
            license='adapter:Apache-2.0;upstream:Responsible-AI-Source-Code-License1.1;weights:verify-separately')

    @classmethod
    def from_local(cls,**config):
        args = configuration(config); cls.capability(**args)
        backend = EfficientPhysNative.from_local(**config)
        try: return cls(backend,**args)
        except BaseException: backend.close(); raise

    def _clear(self):
        self.owner = self.last = self.start = self.segment = None
        self.images = []; self.hashes = []

    def reset(self):
        with self.lock:
            if self.closed: raise RuntimeError('rPPG context closed')
            self._clear()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed=True; self._clear(); self.backend.close()

    def process(self,frame):
        with self.lock:
            if self.closed: raise RuntimeError('rPPG context closed')
            try: return self._process(frame)
            except BaseException: self._clear(); raise

    def _process(self,frame):
        import numpy as np
        meta,rgb = decode(frame)
        p = meta['provenance']; ts,seq = p['timestamp_ms'],p['capture_sequence']
        owner = (p['subject_id'],p['source_id'],p['sensor_id'],p['coordinate_frame'],p['calibration_id'],tuple(p['channels']),
            frame.sensor.stream_epoch,p['sequence_kind'],meta['track_id'],meta['clock_id'],meta['detector_id'],meta['detector_version'],tuple(meta['shape']))
        if self.owner == owner and self.last and (ts <= self.last[0] or seq <= self.last[1]):
            raise ValueError('rPPG capture timestamp and sequence must advance')
        changed = self.owner != owner
        if self.last and not changed:
            changed = seq != self.last[1]+1 or abs((ts-self.last[0])*30-1000) > 60
        if self.start is not None and not changed:
            changed = abs((ts-self.start)*30-len(self.images)*1000) > 60
        if changed: self._clear()
        self.owner,self.last = owner,(ts,seq)
        signal = dict(pulse_bpm=None,spectral_candidate=None,physiological_quality='not_assessed')
        status = 'face_unavailable' if rgb is None else 'collecting'
        observed_start = ts
        count = 0; window_hash = None; effective_fps = None
        if rgb is None:
            self._clear(); self.owner,self.last = owner,(ts,seq)
        else:
            if not self.images:
                self.start = ts; self.segment = uuid.uuid4().hex
            self.images.append(rgb)
            self.hashes.append(meta['capture_sha256']+meta['crop_sha256']+str(seq)+':'+str(ts))
            count = len(self.images); observed_start = self.start
            if count == 180:
                raw = np.stack(self.images).astype(np.float64)
                mean,std = float(raw.mean()),float(raw.std())
                window_hash = hashlib.sha256(json.dumps(self.hashes,separators=(',',':')).encode()).hexdigest()
                effective_fps = 179000/(ts-self.start)
                if std <= 1e-8 or float(np.max(np.abs(np.diff(raw,axis=0)))) == 0:
                    status = 'insufficient_variation'
                else:
                    data = ((raw-mean)/std).astype(np.float32)
                    estimates = self.backend.predict(data)
                    if (type(estimates) is not np.ndarray or estimates.dtype != np.float32 or
                            estimates.shape != (180,) or not np.isfinite(estimates).all()):
                        raise ValueError('invalid rPPG model signal')
                    signal['spectral_candidate'] = spectral_candidate(estimates,effective_fps)
                    signal['model_signal_rms'] = float(np.sqrt(np.mean(estimates.astype(np.float64)**2)))
                    status = 'estimated' if signal['spectral_candidate'] is not None else 'insufficient_signal'
                self.images=[];self.hashes=[];self.start=None
        return self.spec.validate(dict(signal=signal,summary=dict(status=status,track_id=meta['track_id'],
            state_segment=self.segment,capture_start_ms=observed_start,capture_end_ms=ts,
            accumulated_frames=count,required_frames=180,nominal_fps=30,effective_fps=effective_fps,
            window_evidence_sha256=window_hash,current_capture_sha256=meta['capture_sha256'],
            normalization='per-window-global-zscore',offline_preprocessing_parity=False,
            identity_verified=False,clock_verified=False,model_quality_validated=False,
            artifacts_verified=self.verified,window_complete=count==180,causal_per_frame=False,
            additional_lookahead_ms=0,terminal_frame='duplicate',complete_snapshot=False)))
