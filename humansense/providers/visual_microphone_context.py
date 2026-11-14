"""Strict timed ROI windows -> owned offline reconstruction -> numeric context."""
import hashlib
import json
import re
import struct
import threading
import uuid
import zlib

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from .visual_microphone import ALGORITHM, recover

MAGIC=b'GVMC001\n'
PREFIX=struct.Struct('<8sI')
LIMIT=4*1024*1024


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError('bounded integer required')
    return value


def configuration(config):
    config=bounded_json(config,max_bytes=2048)
    keys={'fps_numerator','fps_denominator','jitter_ns','native_origin_ns','capture_origin_ms','calibration_id'}
    if type(config) is not dict or set(config)!=keys:raise ValueError('explicit rational fps and clock configuration required')
    a=integer(config['fps_numerator'],1,1000000);b=integer(config['fps_denominator'],1,1000000)
    if not 100*b<=a<=100000*b:raise ValueError('true capture rate must be100..100000Hz')
    integer(config['jitter_ns'],0,(1000000000*b)//(100*a))
    native(config['native_origin_ns']);integer(config['capture_origin_ms'],0,2**53-1);identifier(config['calibration_id'])
    return config


def native(value):
    if type(value) is not str or not re.fullmatch(r'0|[1-9][0-9]{0,18}',value) or int(value)>2**63-1:
        raise ValueError('nonnegative int64 native timestamp decimal string required')
    return int(value)


def _arrays(pixels,times,indices):
    import numpy as np
    if (type(pixels) is not np.ndarray or pixels.dtype!=np.uint8 or pixels.ndim!=3 or
        not 128<=pixels.shape[0]<=4096 or any(not 32<=v<=128 for v in pixels.shape[1:]) or
        pixels.nbytes+pixels.shape[0]*16+8208>LIMIT):raise ValueError('bounded plain uint8 grayscale window required')
    for value in (times,indices):
        if type(value) is not np.ndarray or value.dtype.kind not in 'iu' or value.dtype.itemsize!=8 or value.shape!=(pixels.shape[0],):
            raise ValueError('plain int64 per-frame timestamps and capture indices required')


def _timing(times,indices,config):
    a,b=config['fps_numerator'],config['fps_denominator'];tolerance=config['jitter_ns']*a
    first=int(times[0]);start_index=int(indices[0]);origin=native(config['native_origin_ns'])
    integer(first,origin,2**63-1);integer(start_index,0,2**53-1)
    previous=first
    for i,(stamp,index) in enumerate(zip(times,indices)):
        stamp,index=int(stamp),int(index);integer(stamp,origin,2**63-1);integer(index,0,2**53-1)
        if index!=start_index+i:raise ValueError('missing/reordered original capture index')
        if abs((stamp-first)*a-i*1000000000*b)>tolerance:raise ValueError('capture clock cumulative drift')
        if i and (stamp<=previous or abs((stamp-previous)*a-1000000000*b)>tolerance):raise ValueError('capture interval jitter/gap')
        previous=stamp
    mapped=config['capture_origin_ms']+(previous-origin)//1000000
    integer(mapped,0,2**53-1)
    return mapped


def _metadata(meta,pixels,times,indices):
    keys={'config','subject_id','source_id','sensor_id','coordinate_frame','window_sequence',
          'roi_id','roi_box','source_shape','source_sha256','shutter','exposure_ns','shape'}
    if type(meta) is not dict or set(meta)!=keys:raise ValueError('exact visual window metadata required')
    config=configuration(meta['config']);_arrays(pixels,times,indices)
    for key in ('subject_id','source_id','sensor_id','coordinate_frame','roi_id'):identifier(meta[key])
    integer(meta['window_sequence'],0,2**53-1)
    if type(meta['shape']) is not list or meta['shape']!=list(pixels.shape) or any(type(v) is not int for v in meta['shape']):raise ValueError('window shape mismatch')
    shape,box=meta['source_shape'],meta['roi_box']
    if type(shape) is not list or len(shape)!=2 or type(box) is not list or len(box)!=4:raise ValueError('source geometry and fixed xyxy ROI required')
    for v in shape:integer(v,32,8192)
    for v in box:integer(v,0,8192)
    x0,y0,x1,y1=box
    if not 0<=x0<x1<=shape[1] or not 0<=y0<y1<=shape[0] or (y1-y0,x1-x0)!=pixels.shape[1:]:raise ValueError('unresized ROI must match source coordinates')
    if type(meta['source_sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',meta['source_sha256']):raise ValueError('declared original source evidence SHA256 required')
    if meta['shutter'] not in ('global','unknown'):raise ValueError('rolling-shutter reconstruction unsupported')
    if meta['exposure_ns'] is not None:integer(meta['exposure_ns'],1,1000000000*config['fps_denominator']//config['fps_numerator'])
    return _timing(times,indices,config)


def window_frame(pixels,native_times_ns,capture_indices,*,config,subject_id='local-user',source_id,sensor_id,
                 coordinate_frame,window_sequence,roi_id,roi_box,source_shape,source_sha256,shutter,exposure_ns=None):
    """Snapshot a caller-declared contiguous, fixed, unresized grayscale ROI window."""
    import numpy as np
    _arrays(pixels,native_times_ns,capture_indices)
    pixels=pixels.copy();times=native_times_ns.copy();indices=capture_indices.copy()
    meta=bounded_json(dict(config=configuration(config),subject_id=subject_id,source_id=source_id,sensor_id=sensor_id,
        coordinate_frame=coordinate_frame,window_sequence=window_sequence,roi_id=roi_id,roi_box=roi_box,source_shape=source_shape,
        source_sha256=source_sha256,shutter=shutter,exposure_ns=exposure_ns,shape=list(pixels.shape)),max_bytes=8192)
    timestamp=_metadata(meta,pixels,times,indices)
    head=json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    payload=PREFIX.pack(MAGIC,len(head))+head+times.astype('<i8').tobytes()+indices.astype('<i8').tobytes()+pixels.tobytes()
    payload+=struct.pack('<I',zlib.crc32(payload))
    if len(payload)>LIMIT:raise ValueError('window packet exceeds4MiB')
    return InputFrame(payload,'video',timestamp,subject_id,sensor=SensorMetadata(source_id,sensor_id,window_sequence,
        coordinate_frame=coordinate_frame,calibration_id=config['calibration_id'],capture_sequence=window_sequence))


def decode(packet):
    import numpy as np
    if type(packet) is not bytes or not PREFIX.size+4<=len(packet)<=LIMIT:raise ValueError('bounded window packet required')
    magic,size=PREFIX.unpack(packet[:PREFIX.size])
    if magic!=MAGIC or not 1<=size<=8192 or PREFIX.size+size+4>len(packet) or zlib.crc32(packet[:-4])!=struct.unpack('<I',packet[-4:])[0]:raise ValueError('window header/CRC mismatch')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate window metadata key')
            out[k]=v
        return out
    try:meta=bounded_json(json.loads(packet[PREFIX.size:PREFIX.size+size],object_pairs_hook=pairs),max_bytes=8192)
    except (UnicodeError,RecursionError,json.JSONDecodeError) as exc:raise ValueError('invalid window JSON') from exc
    shape=meta.get('shape') if type(meta) is dict else None
    if type(shape) is not list or len(shape)!=3:raise ValueError('window shape required')
    n=integer(shape[0],128,4096);h=integer(shape[1],32,128);w=integer(shape[2],32,128)
    offset=PREFIX.size+size
    if len(packet)!=offset+n*16+n*h*w+4:raise ValueError('window exact length mismatch')
    times=np.frombuffer(packet,dtype='<i8',count=n,offset=offset).copy()
    indices=np.frombuffer(packet,dtype='<i8',count=n,offset=offset+n*8).copy()
    pixels=np.frombuffer(packet,dtype=np.uint8,count=n*h*w,offset=offset+n*16).reshape(n,h,w).copy()
    timestamp=_metadata(meta,pixels,times,indices)
    return meta,pixels,times,indices,timestamp


class VisualMicrophoneContext:
    def __init__(self,**config):
        self.config=configuration(config);self.spec=self.capability(**config)
        self.lock=threading.RLock();self.closed=False;self._clear()

    @staticmethod
    def capability(**config):
        config=configuration(config)
        digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        return CapabilitySpec('vision.visual_microphone','vision.vibration','video','visual_microphone.phase','1.'+digest,
            (OutputField('vibration','object'),OutputField('summary','object')),'offline',freshness_ms=1000,
            stateful=True,reset_on_gap=True,interpretation='estimate',devices=('cpu',),
            sensor_requirements=('timed-high-speed-fixed-grayscale-roi-window','declared-acquisition-clock'),license='adapter:Apache-2.0;algorithm:MIT')

    def _clear(self):self.owner=self.last=self.segment=None
    def reset(self):
        with self.lock:
            if self.closed:raise RuntimeError('visual microphone closed')
            self._clear()
    def close(self):
        with self.lock:self.closed=True;self._clear()
    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('visual microphone closed')
            try:return self._process(frame)
            except BaseException:self._clear();raise

    def _process(self,frame):
        if not isinstance(frame,InputFrame) or frame.modality!='video' or frame.sample_rate is not None or not isinstance(frame.sensor,SensorMetadata):raise ValueError('video window InputFrame required')
        meta,pixels,times,indices,mapped=decode(frame.data);sensor=frame.sensor
        seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        if (meta['config']!=self.config or type(frame.timestamp_ms) is not int or frame.timestamp_ms!=mapped or
            frame.subject_id!=meta['subject_id'] or sensor.source_id!=meta['source_id'] or sensor.sensor_id!=meta['sensor_id'] or
            sensor.coordinate_frame!=meta['coordinate_frame'] or sensor.calibration_id!=self.config['calibration_id'] or
            sensor.channels or seq!=meta['window_sequence']):raise ValueError('outer source/clock/sequence binding mismatch')
        owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.coordinate_frame,sensor.calibration_id,
               meta['roi_id'],tuple(meta['roi_box']),tuple(meta['source_shape']),meta['source_sha256'],meta['shutter'],meta['exposure_ns'],sensor.capture_sequence is not None)
        first,last=int(times[0]),int(times[-1]);i0,i1=int(indices[0]),int(indices[-1]);continuous=False
        if self.owner==owner and self.last:
            previous_sequence,previous_time,previous_index=self.last
            if seq<=previous_sequence or first<=previous_time or i0<=previous_index:raise ValueError('duplicate/overlapping window')
            a,b=self.config['fps_numerator'],self.config['fps_denominator']
            continuous=(seq==previous_sequence+1 and i0==previous_index+1 and
                abs((first-previous_time)*a-1000000000*b)<=self.config['jitter_ns']*a)
        if not continuous:self.segment=uuid.uuid4().hex
        result=recover(pixels,capture_fps=self.config['fps_numerator']/self.config['fps_denominator'])
        detail=result['metadata']
        value=self.spec.validate(dict(vibration=dict(status=detail['status'],peak_hz=detail['peak_hz'],
            peak_fraction=detail.get('peak_fraction'),frequency_bin_hz=detail['frequency_bin_hz'],
            nyquist_hz=detail['nyquist_hz'],highpass_cutoff_hz=detail['highpass_cutoff_hz'],quality='not_assessed',
            waveform_available_locally=False,speech_available=False,sound_pressure_calibrated=False),
            summary=dict(algorithm=ALGORITHM,processing_mode='offline',full_window_lookahead=True,
                frames=len(times),native_start_ns=str(first),native_end_ns=str(last),capture_end_ms=mapped,
                original_first_frame_index=i0,original_last_frame_index=i1,window_sequence=seq,
                fps_numerator=self.config['fps_numerator'],fps_denominator=self.config['fps_denominator'],jitter_ns=self.config['jitter_ns'],
                roi_id=meta['roi_id'],roi_box=meta['roi_box'],source_shape=meta['source_shape'],coordinate_frame=meta['coordinate_frame'],
                shutter=meta['shutter'],exposure_ns=meta['exposure_ns'],source_sha256=meta['source_sha256'],
                declared_source_hash_verified=False,pixel_sha256=detail['input_sha256'],packet_sha256=hashlib.sha256(frame.data).hexdigest(),
                state_segment=self.segment,continuous=continuous,clock_calibration_verified=False,
                capture_completeness='not_verified',hardware_validated=False,official_matlab_parity_verified=False,
                complete_snapshot=False)))
        self.owner,self.last=owner,(seq,last,i1)
        return value
