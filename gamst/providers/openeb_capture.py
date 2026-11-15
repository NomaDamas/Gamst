"""Explicit installed OpenEB5.2 RAW/camera source. Native I/O is synchronous.

Run capture in a supervised process when hard cancellation/deadlines are needed.
No camera, SDK or native dependency is loaded during module import/discovery.
"""
import hashlib
import importlib
from numbers import Integral
from pathlib import Path
import re
import threading

from ..capabilities import SensorMetadata, identifier
from ..plugins import InputFrame
from .openeb import OpenEBContext, _clock, _geometry, event_packet

SOURCES={
    'events_iterator':'f4ada10870fc14488e4fd14df9a8d394f4fd37607b6830cf103c5c7b918991b8',
    'raw_reader':'d9976c236951968546780ed1563cd46e0e7d06d718104c979f4609712ca8480d',
}


def _installed_iterator():
    modules={}
    try:
        for name,digest in SOURCES.items():
            module=importlib.import_module('metavision_core.event_io.'+name)
            path=Path(module.__file__)
            if path.suffix!='.py':raise RuntimeError('OpenEB Python sources must be available for revision check')
            with path.open('rb') as source:blob=source.read(131073)
            if len(blob)>131072 or hashlib.sha256(blob).hexdigest()!=digest:
                raise RuntimeError('selected OpenEB5.2.0 Python source mismatch: '+name)
            modules[name]=module
    except ImportError as exc:
        raise RuntimeError('explicit capture requires an installed OpenEB5.2.0 SDK and device plugin') from exc
    # Hashes check installed wrapper source identity, not native binary attestation.
    return modules['events_iterator'].EventsIterator


class OpenEBCapture:
    """Own one native iterator. Obtain at most8192events or10ms per slice.

    Source selection is explicit: an existing absolute RAW file, or named serial.
    Closing is serialized with reads; it cannot interrupt a blocked driver call.
    """
    @classmethod
    def recording(cls,path,**kwargs):
        if type(path) is not str or not Path(path).is_absolute() or not Path(path).is_file() or Path(path).suffix.lower()!='.raw':
            raise ValueError('existing absolute local .raw recording required')
        return cls._open(path,**kwargs)

    @classmethod
    def camera(cls,serial,**kwargs):
        if type(serial) is not str or not re.fullmatch('[a-zA-Z0-9_-]{1,64}',serial):
            raise ValueError('explicit camera serial required; implicit first-camera selection disabled')
        return cls._open(serial,**kwargs)

    @classmethod
    def _open(cls,source,*,source_id,sensor_id,subject_id='local-user',sensor_origin_us,
              capture_origin_ms,calibration_id,coordinate_frame):
        _clock(sensor_origin_us,capture_origin_ms,calibration_id)
        for value in (source_id,sensor_id,subject_id,coordinate_frame):identifier(value)
        origin=int(sensor_origin_us)
        if origin%10000:raise ValueError('OpenEB capture start must align to10000microseconds')
        factory=_installed_iterator()
        reader=factory(source,start_ts=origin,mode='mixed',delta_t=10000,n_events=8192,relative_timestamps=False)
        try:
            height,width=reader.get_size();_geometry(width,height)
            config=dict(width=width,height=height,sensor_origin_us=sensor_origin_us,
                capture_origin_ms=capture_origin_ms,calibration_id=calibration_id)
            OpenEBContext.capability(**config)
            result=cls()
            result.reader=reader;result.iterator=iter(reader)
            result.width,result.height=width,height
            result.origin,result.capture_origin=int(sensor_origin_us),capture_origin_ms
            result.source_id,result.sensor_id,result.subject_id=source_id,sensor_id,subject_id
            result.calibration_id,result.coordinate_frame=calibration_id,coordinate_frame
            result.config=config;result.sequence=0;result.previous_us=origin
            result.lock=threading.RLock();result.closed=False
            return result
        except BaseException:
            reader.reader.__exit__(None,None,None)
            raise

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def next_frame(self):
        with self.lock:
            if self.closed:raise RuntimeError('OpenEB capture closed')
            try:
                events=next(self.iterator)
                end=self.reader.get_current_time()
                if isinstance(end,bool) or not isinstance(end,Integral):raise ValueError('integer native boundary required')
                end=int(end)
                start=self.previous_us
                # Reader's microsecond boundary is independent of first/last event,
                # including empty slices and count-limited equal-time slices.
                packet=event_packet(events,width=self.width,height=self.height,start_us=start,end_us=end)
                mapped=self.capture_origin+(end-self.origin)//1000
                if start<self.origin or not 0<=mapped<=2**53-1:raise ValueError('native source predates declared clock origin')
                sensor=SensorMetadata(self.source_id,self.sensor_id,self.sequence,
                    coordinate_frame=self.coordinate_frame,calibration_id=self.calibration_id)
                result=InputFrame(packet,'events',mapped,self.subject_id,sensor=sensor)
                self.sequence+=1;self.previous_us=end
                return result
            except BaseException:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            iterator,reader=self.iterator,self.reader
            self.iterator=self.reader=None
            try:iterator.close()
            finally:
                # Also closes constructor-started RAW acquisition before first next.
                # Pinned RawReaderBase.__exit__/__del__ are idempotent by attribute checks.
                reader.reader.__exit__(None,None,None)
