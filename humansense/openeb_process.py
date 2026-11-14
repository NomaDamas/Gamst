"""Async supervision of one explicitly owned native event-camera source."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import struct
import uuid

from .capabilities import bounded_json, identifier
from .process_provider import ProcessProvider
from .provider_wire import envelope, decode_frame
from .providers.openeb import OpenEBContext, _clock, decode_event_packet, MAX_PACKET
from .streams import _finish

HEADER_LIMIT=16384
PROTOCOL='gamst-openeb-capture-v1'


def configuration(config):
    config=bounded_json(config,max_bytes=HEADER_LIMIT)
    required={'kind','source','source_id','sensor_id','sensor_origin_us','capture_origin_ms','calibration_id','coordinate_frame'}
    if type(config) is not dict or not required<=set(config) or set(config)-required-{'subject_id'}:
        raise ValueError('explicit OpenEB source selection, identity and clock mapping required')
    for key in ('source_id','sensor_id','calibration_id','coordinate_frame'):identifier(config[key])
    config.setdefault('subject_id','local-user');identifier(config['subject_id'])
    _clock(config['sensor_origin_us'],config['capture_origin_ms'],config['calibration_id'])
    if int(config['sensor_origin_us'])%10000:raise ValueError('capture origin must align to10000microseconds')
    source=config['source']
    if config['kind']=='camera':
        if type(source) is not str or not re.fullmatch('[a-zA-Z0-9_-]{1,64}',source):raise ValueError('explicit camera serial required')
    elif config['kind']=='recording':
        if type(source) is not str or not Path(source).is_absolute() or not Path(source).is_file() or Path(source).suffix.lower()!='.raw':
            raise ValueError('existing absolute RAW recording required')
    else:raise ValueError('source kind must be camera or recording')
    return config


def header(data):
    if type(data) is not bytes or not 1<=len(data)<=HEADER_LIMIT:raise ValueError('invalid capture header length')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate capture header key')
            result[key]=value
        return result
    try:return bounded_json(json.loads(data,object_pairs_hook=pairs),max_bytes=HEADER_LIMIT)
    except (UnicodeError,RecursionError,json.JSONDecodeError) as exc:raise ValueError('invalid capture header') from exc


def prefix(data):
    size,payload=struct.unpack('!II',data)
    if not 1<=size<=HEADER_LIMIT or payload>MAX_PACKET:raise ValueError('capture response exceeds bounds')
    return size,payload


class _Transport(ProcessProvider):
    """Reuse owned-process reaping; capture responses can carry bounded packets."""
    async def _stop(self):
        if self._stop_task is None:self._stop_task=asyncio.create_task(super()._stop())
        await _finish(self._stop_task)

    async def _exchange(self,operation,payload,fields):
        if self._sequence>=2**53-1:raise ValueError('capture request sequence exhausted')
        self._sequence+=1;number=self._sequence;process=self._process
        process.stdin.write(envelope(dict(id=number,operation=operation,**fields)))
        await process.stdin.drain()
        size,count=prefix(await process.stdout.readexactly(8))
        response=header(await process.stdout.readexactly(size))
        data=await process.stdout.readexactly(count)
        if (type(response) is not dict or type(response.get('id')) is not int or response['id']!=number or
                type(response.get('ok')) is not bool):raise ValueError('capture response ID/status mismatch')
        if not response['ok']:
            if set(response)!={'id','ok','error_type'} or type(response['error_type']) is not str or not re.fullmatch('[A-Za-z0-9_]{1,128}',response['error_type']) or data:
                raise ValueError('malformed capture failure')
            raise RuntimeError('OpenEB capture failed ('+response['error_type']+'); verify explicit source/runtime and clock')
        if set(response)!={'id','ok','result'}:raise ValueError('invalid capture response fields')
        return response['result'],data


class OpenEBProcess:
    """One owned source child, one in-flight read, no auto restart or device choice.

    Completed next_frame calls carry a source epoch. Use managed=True only when
    submitting to SensorStreams, which assigns its own independently owned epoch.
    """
    @classmethod
    async def open(cls,python,config,*,startup_timeout=30,read_timeout=5,close_timeout=2):
        for value in (startup_timeout,read_timeout,close_timeout):
            if type(value) not in (int,float) or not .01<=value<=600:raise ValueError('timeouts must be0.01..600seconds')
        config=configuration(config)
        python=str(Path(python).expanduser().absolute())
        if not os.path.isfile(python) or not os.access(python,os.X_OK):raise ValueError('existing executable Python required')
        self=cls();self.selection=config;self.epoch=uuid.uuid4().hex
        self._closing=False;self._inflight=None;self._close_task=None;self.close_timeout=close_timeout
        self.stop_reason=None;self.graceful_acknowledged=False;self.frames_received=0
        self._last_end=int(config['sensor_origin_us'])
        transport=_Transport();self._transport=transport
        transport.profile='openeb.capture';transport.timeout=read_timeout;transport._lock=asyncio.Lock()
        transport._sequence=0;transport._process=None;transport._close_task=None;transport._stop_task=None;transport.closed=False
        self._child=None
        loop=asyncio.get_running_loop();deadline=loop.time()+startup_timeout
        spawn=asyncio.create_task(asyncio.create_subprocess_exec(python,'-u','-m','humansense.openeb_capture_worker',
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,limit=32768))
        try:
            self._child=transport._process=await asyncio.wait_for(asyncio.shield(spawn),startup_timeout)
            remaining=deadline-loop.time()
            if remaining<=0:raise asyncio.TimeoutError()
            result,data=await transport._request('start',config=config,timeout=remaining)
            if type(result) is not dict or set(result)!={'protocol','config'} or result['protocol']!=PROTOCOL or data:
                raise ValueError('capture startup contract mismatch')
            context=result['config']
            expected={k:config[k] for k in ('sensor_origin_us','capture_origin_ms','calibration_id')}
            if type(context) is not dict or set(context)!=set(expected)|{'width','height'} or any(context[k]!=v for k,v in expected.items()):
                raise ValueError('capture clock contract mismatch')
            OpenEBContext.capability(**context)
            self.config=bounded_json(context)
            return self
        except BaseException as exc:
            self.stop_reason=self._reason(exc,'startup');self._closing=True
            if self._child is None:
                try:self._child=transport._process=await _finish(spawn)
                except Exception:pass
            await _finish(asyncio.create_task(transport._stop()))
            raise

    @staticmethod
    def _reason(exc,stage):
        if isinstance(exc,asyncio.CancelledError):return stage+'_cancelled'
        if isinstance(exc,asyncio.TimeoutError):return stage+'_timeout'
        return stage+'_error'

    @property
    def closed(self):return self._closing or self._transport.closed

    def diagnostics(self):
        child=self._child
        return dict(epoch=self.epoch,closed=self.closed,frames_received=self.frames_received,
            stop_reason=self.stop_reason,graceful_acknowledged=self.graceful_acknowledged,
            child_returncode=child.returncode if child else None,reaped=bool(child and child.returncode is not None),
            native_cleanup_verified=False,stderr_retained_bytes=0)

    def _frame(self,metadata,data):
        if (type(metadata) is not dict or set(metadata)!={'metadata','layout'} or metadata['layout']!={'kind':'bytes'} or
                type(metadata['metadata']) is not dict or set(metadata['metadata'])!={'modality','timestamp_ms','subject_id','sample_rate','sensor'}):
            raise ValueError('capture frame layout mismatch')
        sensor_meta=metadata['metadata']['sensor']
        if (type(sensor_meta) is not dict or set(sensor_meta)!={'source_id','sensor_id','sequence','coordinate_frame','calibration_id','channels','stream_epoch','capture_sequence'} or
                type(sensor_meta['channels']) is not list or sensor_meta['channels']):
            raise ValueError('capture sensor metadata mismatch')
        frame=decode_frame(metadata,data);sensor=frame.sensor;selected=self.selection
        if (frame.modality!='events' or frame.sample_rate is not None or frame.subject_id!=selected['subject_id'] or
                sensor.source_id!=selected['source_id'] or sensor.sensor_id!=selected['sensor_id'] or
                sensor.coordinate_frame!=selected['coordinate_frame'] or sensor.calibration_id!=selected['calibration_id'] or
                sensor.sequence!=self.frames_received or sensor.capture_sequence is not None or sensor.stream_epoch is not None or sensor.channels):
            raise ValueError('capture source/sequence provenance mismatch')
        width,height,start,end,_=decode_event_packet(data)
        mapped=selected['capture_origin_ms']+(end-int(selected['sensor_origin_us']))//1000
        if ((width,height)!=(self.config['width'],self.config['height']) or start!=self._last_end or
                type(frame.timestamp_ms) is not int or frame.timestamp_ms!=mapped or not 0<=mapped<=2**53-1):
            raise ValueError('capture event interval/clock mismatch')
        self._last_end=end;self.frames_received+=1
        return replace(frame,sensor=replace(sensor,stream_epoch=self.epoch,capture_sequence=sensor.sequence))

    async def _next(self,managed):
        try:
            result,data=await self._transport._request('next')
            if self._closing:raise RuntimeError('capture closed before publication')
            if type(result) is not dict:raise ValueError('invalid capture result')
            if result=={'eof':True} and type(result.get('eof')) is bool and not data:
                self.stop_reason='eof';self._closing=True
                await self._transport._stop()
                raise StopAsyncIteration
            if set(result)!={'frame'}:raise ValueError('invalid capture frame result')
            frame=self._frame(result['frame'],data)
            return replace(frame,sensor=replace(frame.sensor,stream_epoch=None)) if managed else frame
        except StopAsyncIteration:raise
        except BaseException as exc:
            if self.stop_reason is None:self.stop_reason=self._reason(exc,'read')
            self._closing=True
            await _finish(asyncio.create_task(self._transport._stop()))
            raise

    async def next_frame(self,*,managed=False):
        if type(managed) is not bool:raise ValueError('managed must be boolean')
        if self.closed:raise RuntimeError('capture is closed; explicitly reopen source and stream')
        if self._inflight is not None and not self._inflight.done():raise RuntimeError('only one in-flight capture read allowed')
        task=asyncio.create_task(self._next(managed));self._inflight=task
        try:
            frame=await asyncio.shield(task)
            if self.closed:raise RuntimeError('capture closed before frame delivery')
            return frame
        except asyncio.CancelledError:
            if self.stop_reason is None:self.stop_reason='read_cancelled'
            self._closing=True;task.cancel()
            try:await _finish(task)
            except BaseException:pass
            await _finish(asyncio.create_task(self._transport._stop()))
            raise
        finally:
            if self._inflight is task and task.done():self._inflight=None

    async def _close(self):
        self._closing=True
        if self.stop_reason is None:self.stop_reason='closed'
        task=self._inflight
        try:
            if task is not None and not task.done():
                task.cancel()
                try:await _finish(task)
                except BaseException:pass
            if not self._transport.closed:
                result,data=await self._transport._request('close',timeout=self.close_timeout)
                if result is not None or data:raise ValueError('invalid capture close acknowledgement')
                self.graceful_acknowledged=True
        except Exception as exc:
            self.stop_reason=self._reason(exc,'close')
        finally:await _finish(asyncio.create_task(self._transport._stop()))

    async def close(self):
        self._closing=True  # Invalidate publication before yielding or waiting.
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _finish(self._close_task)
            raise

    async def __aenter__(self):return self
    async def __aexit__(self,*args):await self.close()
