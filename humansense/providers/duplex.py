"""Ten-millisecond WebRTC AEC3 with separate unchanged ambient PCM."""
from dataclasses import dataclass, replace
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier


@dataclass(frozen=True)
class DuplexBlock:
    ambient: object
    speech: object  # None when playback reference is unavailable.
    context: dict


class WebRTCDuplex:
    def __init__(self,factory,*,sample_rate=16000,stream_delay_ms=0,
                 ambient_sensor_id='ambient',speech_sensor_id='speech'):
        self.spec=self.capability(sample_rate=sample_rate,stream_delay_ms=stream_delay_ms,
            ambient_sensor_id=ambient_sensor_id,speech_sensor_id=speech_sensor_id)
        if not callable(factory):raise ValueError('explicit fresh AEC factory required')
        self.factory=factory;self.rate=sample_rate;self.delay=stream_delay_ms
        self.ambient_sensor_id=ambient_sensor_id;self.speech_sensor_id=speech_sensor_id
        self._lock=threading.RLock();self.closed=False;self.backend=None;self.resets=0
        self.owner=self.last=self.expected_end=None

    @staticmethod
    def capability(*,sample_rate=16000,stream_delay_ms=0,ambient_sensor_id='ambient',speech_sensor_id='speech'):
        if type(sample_rate) is not int or sample_rate not in (16000,32000,48000):raise ValueError('AEC sample_rate must be16000/32000/48000')
        if type(stream_delay_ms) is not int or not 0<=stream_delay_ms<=500:raise ValueError('configured delay must be0..500ms')
        identifier(ambient_sensor_id);identifier(speech_sensor_id)
        if ambient_sensor_id==speech_sensor_id:raise ValueError('ambient and speech sensor IDs must differ')
        import hashlib
        version='v1.'+hashlib.sha256(f'{sample_rate}:{stream_delay_ms}:{ambient_sensor_id}:{speech_sensor_id}'.encode()).hexdigest()
        return CapabilitySpec('audio.webrtc.aec3','audio.duplex','audio','pywebrtc_audio.0.2.0',version,
            (OutputField('processing','object'),),'native',window_ms=10,stateful=True,reset_on_gap=True,
            freshness_ms=1000,interpretation='estimate',devices=('cpu',),
            sensor_requirements=('microphone-and-render-pcm','declared-clock-pairing'),
            license='wrapper:Apache-2.0;WebRTC:vendored-notices')

    @classmethod
    def from_installed(cls,**options):
        cls.capability(**options)
        try:
            import importlib.metadata
            if importlib.metadata.version('pywebrtc-audio')!='0.2.0':raise RuntimeError('select pywebrtc-audio==0.2.0 in the AEC runtime')
            from pywebrtc_audio import EchoCanceller
        except (ImportError,importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('install humansense-sdk[duplex] in an explicitly selected runtime') from exc
        return cls(lambda:EchoCanceller(sample_rate=options.get('sample_rate',16000),num_channels=1,
                    stream_delay_ms=options.get('stream_delay_ms',0)),**options)

    def _reset(self):
        backend,self.backend=self.backend,None
        self.owner=self.last=self.expected_end=None
        if backend is not None:
            closer=getattr(backend,'close',None)
            if callable(closer):closer()
        # Native binding has no close method: dropping the last owned reference
        # destroys its RAII state, including filter and AudioBuffer histories.

    def reset(self):
        with self._lock:
            if self.closed:raise RuntimeError('duplex provider closed')
            try:self._reset()
            except BaseException:self.closed=True;self.factory=None;raise
            self.resets+=1

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed=True
                try:self._reset()
                finally:self.factory=None

    def transform(self,frame):
        with self._lock:
            if self.closed:raise RuntimeError('duplex provider closed')
            import numpy as np
            try:
                data,sensor,end=frame.data,frame.sensor,frame.timestamp_ms
                if (frame.modality!='audio' or not isinstance(sensor,SensorMetadata) or
                    type(frame.sample_rate) is not int or frame.sample_rate!=self.rate or
                    not isinstance(data,np.ndarray) or data.dtype not in (np.dtype('int16'),np.dtype('float32')) or
                    data.ndim!=2 or data.shape[0]!=self.rate//100 or data.shape[1] not in (1,2)):
                    raise ValueError('AEC requires exactly10ms of native int16 or float32 Nx1/Nx2 PCM')
                identifier(frame.subject_id)
                if type(end) is not int or not 10<=end<=2**53-1:raise ValueError('capture end timestamp must be bounded integer ms')
                present=data.shape[1]==2
                if sensor.channels!=(('microphone','render') if present else ('microphone',)):
                    raise ValueError('declare exact microphone/render channel roles')
                if present and sensor.calibration_id is None:raise ValueError('declare clock-pairing calibration ID')
                if sensor.sensor_id in (self.ambient_sensor_id,self.speech_sensor_id):raise ValueError('derived sensor IDs must differ from input')
                values=data.astype(np.float32,copy=True)
                if data.dtype==np.int16:values/=32768
                if not np.isfinite(values).all() or (np.abs(values)>1).any():raise ValueError('PCM must be finite normalized samples')
                owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,
                       sensor.coordinate_frame,sensor.calibration_id,sensor.channels,self.rate)
                if self.owner==owner and self.last and (sensor.sequence<=self.last[0] or end<=self.last[1]):
                    raise ValueError('capture sequence and end timestamp must increase')
                if self.owner is not None and (owner!=self.owner or sensor.sequence!=self.last[0]+1 or
                                               abs(end-(self.expected_end+10))>1):
                    self._reset();self.resets+=1
                ambient=replace(frame,data=values[:,0].copy(),
                    sensor=replace(sensor,sensor_id=self.ambient_sensor_id,channels=('microphone',)))
                speech=None;clipped=0;out_rms=out_peak=None
                if present:
                    if self.backend is None:
                        backend=self.factory()
                        if not callable(getattr(backend,'process',None)):
                            closer=getattr(backend,'close',None)
                            if callable(closer):closer()
                            raise ValueError('factory must return a fresh EchoCanceller-compatible backend')
                        self.backend=backend
                    # Contiguous independent vectors avoid upstream raw-pointer stride
                    # assumptions and prevent optional native code mutating ambient PCM.
                    result=self.backend.process(values[:,0].copy(),values[:,1].copy())
                    if (not isinstance(result,np.ndarray) or result.dtype!=np.float32 or result.shape!=(self.rate//100,) or
                            not np.isfinite(result).all() or (np.abs(result)>16).any()):
                        raise ValueError('AEC returned invalid normalized float32 block')
                    out_peak=float(np.max(np.abs(result)));clipped=int(np.count_nonzero(np.abs(result)>1))
                    clean=np.clip(result,-1,1).astype(np.float32,copy=True)
                    out_rms=float(np.sqrt(np.mean(clean.astype(np.float64)**2)))
                    speech=replace(frame,data=clean,sensor=replace(sensor,sensor_id=self.speech_sensor_id,channels=('aec_speech',)))
                elif self.backend is not None:
                    self._reset();self.resets+=1
                self.owner=owner;self.last=(sensor.sequence,end)
                self.expected_end=end if self.expected_end is None else self.expected_end+10
                processing=dict(status='processed_estimate' if present else 'unavailable_reference',
                    reference_available=present,reference_silent=bool(not np.any(values[:,1])) if present else None,
                    start_ms=end-10,end_ms=end,sample_rate=self.rate,block_ms=10,
                    input_sensor_id=sensor.sensor_id,ambient_sensor_id=self.ambient_sensor_id,speech_sensor_id=self.speech_sensor_id,
                    pairing_id=sensor.calibration_id,clock_pairing_verified=False,configured_delay_ms=self.delay,
                    input_rms=float(np.sqrt(np.mean(values[:,0].astype(np.float64)**2))),
                    render_rms=float(np.sqrt(np.mean(values[:,1].astype(np.float64)**2))) if present else None,
                    speech_rms=out_rms,speech_peak_before_clip=out_peak,output_samples_clipped=clipped,
                    input_clipping_present=bool(np.any((data[:,0]==-32768)|(data[:,0]==32767))) if data.dtype==np.int16
                        else bool(np.any(np.abs(values[:,0])>=1)),ambient_preserved=True,
                    high_pass_filter_applied=present,noise_suppression=False,automatic_gain_control=False,
                    continuity_resets=self.resets,echo_reduction_verified=False,recognition_improvement_verified=False)
                return DuplexBlock(ambient,speech,self.spec.validate(dict(processing=processing)))
            except Exception:
                try:self._reset()
                except BaseException:self.closed=True;self.factory=None;raise
                self.resets+=1
                raise

    def process(self,frame):
        """Context-only provider mode. Use transform to route PCM to speech/ambient consumers."""
        return self.transform(frame).context
