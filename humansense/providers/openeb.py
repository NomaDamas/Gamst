"""Bounded OpenEB EventCD packets and numerical activity context, not objects."""
import hashlib
import json
import re
import struct
import threading
import uuid
import zlib

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame

REVISION = '9003b5416676e78ba994d912087486cfa94fae73'
MAGIC = b'GECD001\n'
HEADER = struct.Struct('<8sHHQQI')
MAX_EVENTS = 8192
ROW_BYTES = 13
MAX_PACKET = HEADER.size+MAX_EVENTS*ROW_BYTES+4


def _geometry(width,height):
    if any(type(v) is not int or not 1 <= v <= 4096 for v in (width,height)) or width*height > 4096*2160:
        raise ValueError('bounded EventCD image geometry required')


def _interval(start_us,end_us):
    if any(type(v) is not int or not 0 <= v <= 2**63-1 for v in (start_us,end_us)) or not start_us <= end_us <= start_us+1000000:
        raise ValueError('ordered native microsecond interval of at most1second required')


def _dtype():
    import numpy as np
    return np.dtype([('x','<u2'),('y','<u2'),('p','u1'),('t','<u8')])


def _events(events,width,height,start_us,end_us):
    import numpy as np
    _geometry(width,height);_interval(start_us,end_us)
    if (type(events) is not np.ndarray or events.ndim != 1 or len(events) > MAX_EVENTS or
            events.dtype.names != ('x','y','p','t') or
            any(events.dtype.fields[k][0].kind not in 'iu' or events.dtype.fields[k][0].subdtype is not None for k in ('x','y','p','t'))):
        raise ValueError('plain bounded structured EventCD integer x/y/p/t array required')
    if len(events) and (np.any(events['x'] < 0) or np.any(events['x'] >= width) or
            np.any(events['y'] < 0) or np.any(events['y'] >= height) or
            np.any(events['p'] < 0) or np.any(events['p'] > 1) or
            np.any(events['t'] < start_us) or np.any(events['t'] > end_us) or
            np.any(events['t'][1:] < events['t'][:-1])):
        raise ValueError('invalid EventCD coordinate/polarity/time/order')


def event_packet(events,*,width,height,start_us,end_us):
    """Copy native EventCD fields explicitly; never serialize native struct padding."""
    import numpy as np
    _events(events,width,height,start_us,end_us)
    events=events.copy()
    _events(events,width,height,start_us,end_us)
    value=np.empty(len(events),dtype=_dtype())
    for field in ('x','y','p','t'):value[field]=events[field]
    # Revalidate the owned snapshot in case the source changed while being copied.
    _events(value,width,height,start_us,end_us)
    header=HEADER.pack(MAGIC,width,height,start_us,end_us,len(value))
    packet=header+value.tobytes()
    return packet+struct.pack('<I',zlib.crc32(packet))


def decode_event_packet(packet):
    import numpy as np
    if type(packet) is not bytes or not HEADER.size+4 <= len(packet) <= MAX_PACKET:
        raise ValueError('bounded EventCD packet required')
    magic,width,height,start,end,count=HEADER.unpack(packet[:HEADER.size])
    if magic != MAGIC or count > MAX_EVENTS or len(packet) != HEADER.size+ROW_BYTES*count+4:
        raise ValueError('invalid EventCD packet header/count')
    if zlib.crc32(packet[:-4]) != struct.unpack('<I',packet[-4:])[0]:raise ValueError('EventCD CRC mismatch')
    events=np.frombuffer(packet[HEADER.size:-4],dtype=_dtype()).copy()
    _events(events,width,height,start,end)
    return width,height,start,end,events


def _clock(origin_us,capture_origin_ms,calibration_id):
    identifier(calibration_id)
    if type(origin_us) is not str or not re.fullmatch(r'0|[1-9][0-9]{0,18}',origin_us) or int(origin_us)>2**63-1:
        raise ValueError('sensor_origin_us must be a nonnegative int64 decimal string')
    if type(capture_origin_ms) is not int or not 0 <= capture_origin_ms <= 2**53-1:
        raise ValueError('capture_origin_ms must be bounded integer milliseconds')


class OpenEBContext:
    def __init__(self,*,width,height,sensor_origin_us,capture_origin_ms,calibration_id):
        self.spec=self.capability(width=width,height=height,sensor_origin_us=sensor_origin_us,
            capture_origin_ms=capture_origin_ms,calibration_id=calibration_id)
        self.width,self.height=width,height
        self.origin,self.capture_origin=int(sensor_origin_us),capture_origin_ms
        self.calibration_id=calibration_id
        self.lock=threading.RLock();self.closed=False;self._clear()

    @staticmethod
    def capability(*,width,height,sensor_origin_us,capture_origin_ms,calibration_id):
        _geometry(width,height);_clock(sensor_origin_us,capture_origin_ms,calibration_id)
        digest=hashlib.sha256(json.dumps([REVISION,width,height,sensor_origin_us,capture_origin_ms,calibration_id]).encode()).hexdigest()
        return CapabilitySpec('events.openeb.activity','events.activity','events','openeb.eventcd','1.'+digest,
            (OutputField('activity','object'),OutputField('summary','object')),
            'native',freshness_ms=1000,stateful=True,reset_on_gap=True,interpretation='measurement',devices=('cpu',),
            sensor_requirements=('openeb-eventcd-x-y-p-t','mapped-capture-clock','pixel-coordinate-frame'),
            license='adapter:Apache-2.0;OpenEB:Apache-2.0;device-plugin:verify-separately')

    def _clear(self):self.owner=self.last=self.segment=None

    def reset(self):
        with self.lock:
            if self.closed:raise RuntimeError('EventCD context closed')
            self._clear()

    def close(self):
        with self.lock:self.closed=True;self._clear()

    def process(self,frame):
        with self.lock:
            if self.closed:raise RuntimeError('EventCD context closed')
            try:return self._process(frame)
            except BaseException:self._clear();raise

    def _process(self,frame):
        import numpy as np
        if not isinstance(frame,InputFrame) or frame.modality!='events' or not isinstance(frame.sensor,SensorMetadata):
            raise ValueError('events InputFrame with sensor provenance required')
        identifier(frame.subject_id)
        sensor=frame.sensor
        if sensor.calibration_id!=self.calibration_id or not sensor.coordinate_frame:
            raise ValueError('matching clock calibration and declared pixel coordinates required')
        width,height,start,end,events=decode_event_packet(frame.data)
        mapped=self.capture_origin+(end-self.origin)//1000
        if ((width,height)!=(self.width,self.height) or start<self.origin or not 0<=mapped<=2**53-1 or
                type(frame.timestamp_ms) is not int or frame.timestamp_ms!=mapped):
            raise ValueError('EventCD geometry/native clock disagrees with outer capture')
        seq=sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner=(frame.subject_id,sensor.source_id,sensor.sensor_id,sensor.stream_epoch,sensor.coordinate_frame,
            sensor.calibration_id,sensor.channels,sensor.capture_sequence is not None)
        continuous=False
        if self.owner==owner and self.last:
            if seq<=self.last[0] or start<self.last[1]:raise ValueError('EventCD capture sequence must advance and intervals must not overlap')
            continuous=seq==self.last[0]+1 and start==self.last[1]
        if not continuous:self.segment=uuid.uuid4().hex
        n=len(events);positive=int(np.count_nonzero(events['p']))
        grid=np.zeros((2,64),dtype=np.int64)
        if n:
            x,y=events['x'].astype(np.int64),events['y'].astype(np.int64)
            cells=(y*8//height)*8+x*8//width
            for polarity in (0,1):grid[polarity]=np.bincount(cells[events['p']==polarity],minlength=64)
            active=int(len(np.unique(y*width+x)))
            centroid=dict(x=float(x.mean()),y=float(y.mean()),unit='pixel',meaning='event_weighted_activity_centroid')
        else:active=0;centroid=None
        duration=end-start
        value=self.spec.validate(dict(activity=dict(event_count=n,positive_count=positive,negative_count=n-positive,
            duration_us=duration,event_rate_hz=n*1e6/duration if duration else None,
            positive_fraction=positive/n if n else None,active_pixel_count=active,
            active_pixel_fraction=active/(width*height),activity_centroid=centroid,
            polarity_grid=dict(width=8,height=8,order='row-major',negative=grid[0].tolist(),positive=grid[1].tolist(),unit='event_count')),
            summary=dict(native_start_us=str(start),native_end_us=str(end),capture_end_ms=mapped,
                capture_sequence=seq,coordinate_frame=sensor.coordinate_frame,width=width,height=height,
                state_segment=self.segment,continuous=continuous,clock_calibration_verified=False,
                packet_sha256=hashlib.sha256(frame.data).hexdigest(),events_complete='not_verified',
                object_detection_available=False,physical_motion_available=False,complete_snapshot=False)))
        self.owner,self.last=owner,(seq,end)
        return value
