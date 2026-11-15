"""Owned two-packet replay retention and independent actual SDK delivery.

Acquisition never awaits a provider. Replacement is a downstream mailbox event;
it cannot alter original acquisition counters. Active SDK calls own their own
separately reserved snapshots, outside this two-packet history.
"""
import asyncio
from collections import deque
from dataclasses import replace
import time

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.streams import _finish
from .owned_capture import CapturePacket,CaptureSink,DeliveryAdmission
from .recorded_source import digest,validate_manifest


def capture_reference(packet):
    if type(packet) is not CapturePacket:raise TypeError('owned original capture required')
    return dict(kind='capture',source_index=packet.source_index,capture_ordinal=packet.record_ordinal,
        record_sha256=packet.record_sha256,mapping_sha256=packet.mapping_sha256)


def packet_frame(packet,source,binding,*,subject_id,stream_epoch=None):
    """Decode original owned bytes only; no resampling, channel conversion or IO."""
    if type(packet) is not CapturePacket:raise TypeError('owned original capture required')
    if packet.source_id!=source['source_id'] or packet.sensor_id!=source['sensor_id']:
        raise ValueError('original packet/source identity mismatch')
    if packet.format=='bgr8_hwc':
        import numpy as np
        data=np.frombuffer(packet.payload,dtype=np.uint8).reshape(packet.shape)
    else:data=packet.payload
    sensor=SensorMetadata(packet.source_id,packet.sensor_id,packet.capture_sequence,
        binding['coordinate_frame'],() if binding['channels'] is None else tuple(binding['channels']),
        source['calibration_id'],packet.original_stream_epoch if stream_epoch is None else stream_epoch,packet.capture_sequence)
    return InputFrame(data,source['modality'],packet.sdk_timestamp_ms,sample_rate=packet.sample_rate,
        sensor=sensor,subject_id=subject_id)


class RecordingCaptureRouter(CaptureSink):
    def __init__(self,manifest,*,streams,source_bindings,subject_id,delivery_sources,clock=None,artifacts=None,delivery_bindings=None):
        self.manifest=validate_manifest(manifest);self._source_sha=digest(self.manifest)
        self.streams=streams;self.subject_id=subject_id;self.clock=clock;self.artifacts=artifacts
        self._delivery_bindings={} if delivery_bindings is None else {k:tuple(v) for k,v in delivery_bindings.items()}
        self._delivery_ordinal=0
        self.sources=self.manifest['sources'];self._bindings={b['source_id']:b for b in source_bindings}
        if set(self._bindings)!={s['source_id'] for s in self.sources}:
            raise ValueError('complete frozen original source bindings required')
        self._delivery=set(delivery_sources)
        if not self._delivery<=set(self._bindings):raise ValueError('unknown delivery source')
        self._packets=[deque(maxlen=2) for _ in self.sources]
        self._pending=[deque(maxlen=2) for _ in self.sources]
        self._events=[asyncio.Event() for _ in self.sources];self._tasks=[]
        self._closed=False;self._ordinal=0;self._failure=None
        self._stats=[dict(offered=0,retained=0,mailbox_replaced=0,submitted=0,submit_errors=0,
            closed_before_submit=0,last_acquired_ordinal=None,last_submitted_sequence=None) for _ in self.sources]
        self._tasks=[asyncio.create_task(self._deliver(i)) for i in range(len(self.sources)) if self.sources[i]['source_id'] in self._delivery]

    def _record(self,event,packet,reason,binding=None):
        if self.artifacts is None:return
        self.artifacts.record_delivery(dict(delivery_ordinal=self._delivery_ordinal,event=event,
            observed_ns=time.monotonic_ns(),source_index=packet.source_index,capture_ordinal=packet.record_ordinal,
            binding_index=binding,reason=reason,payload_bytes=len(packet.payload),operation_instance=None,publication_ordinal=None))
        self._delivery_ordinal+=1

    def offer(self,packet):
        if type(packet) is not CapturePacket:raise TypeError('owned immutable packet required')
        if self._closed:
            self._record('mailbox_refused',packet,'closed');return DeliveryAdmission(False,'closed',None)
        i=packet.source_index
        if not 0<=i<len(self.sources) or packet.source_id!=self.sources[i]['source_id']:
            raise ValueError('packet source outside original inventory')
        if packet.source_manifest_sha256!=self._source_sha or not 0<=packet.record_ordinal<len(self.manifest['records']):
            raise ValueError('foreign original capture inventory')
        original=self.manifest['records'][packet.record_ordinal]
        if packet.record_sha256!=digest(original) or packet.payload_sha256!=original['payload_sha256']:
            raise ValueError('capture differs from original record/payload identity')
        expected=(original['source_id'],original['sensor_id'],original['stream_epoch'],original['sequence'],
            original['capture_ns'],original['offset_ns'],original['format'],
            tuple(original['shape']) if 'shape' in original else None,original.get('sample_rate'),original.get('sample_count'))
        observed=(packet.source_id,packet.sensor_id,packet.original_stream_epoch,packet.capture_sequence,
            packet.original_capture_ns,packet.offset_ns,packet.format,packet.shape,packet.sample_rate,packet.sample_count)
        if observed!=expected:raise ValueError('packet capture metadata differs from frozen original')
        if self.clock is not None:
            mapped=self.clock.capture(packet.offset_ns)
            if (packet.mapping_sha256!=self.clock.sha256 or packet.sdk_timestamp_ms!=mapped['sdk_timestamp_ms']
                    or packet.submillisecond_remainder_ns!=mapped['submillisecond_remainder_ns']
                    or packet.scheduled_monotonic_ns!=mapped['scheduled_monotonic_ns']
                    or packet.acquired_monotonic_ns<packet.scheduled_monotonic_ns):
                raise ValueError('packet differs from actual replay mapping')
        stats=self._stats[i]
        if stats['last_acquired_ordinal'] is not None and packet.record_ordinal<=stats['last_acquired_ordinal']:
            raise ValueError('source acquisition ordinal repeated or regressed')
        stats['offered']+=1;stats['last_acquired_ordinal']=packet.record_ordinal
        # Pending packets and selection history share references to the SAME two
        # byte buffers. Evict pending before retaining a third unique capture.
        history=self._packets[i]
        if len(history)==2:
            evicted=history[0]
            pending=self._pending[i]
            if pending and pending[0] is evicted:
                pending.popleft();stats['mailbox_replaced']+=1
                self._record('mailbox_replaced',evicted,'capacity')
        history.append(packet);stats['retained']=len(history)
        if packet.source_id in self._delivery:
            self._pending[i].append(packet);self._events[i].set()
        self._record('mailbox_accepted',packet,'completed')
        ordinal=self._ordinal;self._ordinal+=1
        return DeliveryAdmission(True,'accepted',ordinal)

    def retained(self,source_index):return tuple(self._packets[source_index])

    async def _deliver(self,i):
        event=self._events[i];pending=self._pending[i];source=self.sources[i]
        try:
            while True:
                await event.wait();event.clear()
                while pending:
                    packet=pending.popleft()
                    if self._closed:
                        self._stats[i]['closed_before_submit']+=1;self._record('mailbox_refused',packet,'closed');continue
                    try:
                        frame=packet_frame(packet,source,self._bindings[packet.source_id],subject_id=self.subject_id)
                        # SensorStreams owns its delivery epoch; original epoch remains in packet+capture locator.
                        frame=replace(frame,sensor=replace(frame.sensor,stream_epoch=None))
                        result=await self.streams.submit(frame,evaluation_capture_ref=capture_reference(packet))
                        if result.get('accepted') is not True:raise RuntimeError('SDK refused recorded submission')
                        for bi in self._delivery_bindings.get(packet.source_id,(None,)):self._record('provider_submitted',packet,'completed',bi)
                        self._stats[i]['submitted']+=1
                        self._stats[i]['last_submitted_sequence']=packet.capture_sequence
                    except asyncio.CancelledError:raise
                    except Exception as error:
                        for bi in self._delivery_bindings.get(packet.source_id,(None,)):self._record('provider_refused',packet,'backend_error',bi)
                        self._stats[i]['submit_errors']+=1
                        if self._failure is None:self._failure=type(error).__name__
                if self._closed:return
        finally:
            self._stats[i]['closed_before_submit']+=len(pending)
            for packet in pending:self._record('mailbox_refused',packet,'closed')
            pending.clear()

    def watermarks(self):
        return [dict(source_index=i,last_acquired_ordinal=s['last_acquired_ordinal'],
            last_submitted_sequence=s['last_submitted_sequence'],processed_sequence=None,published_sequence=None,
            processed_known=False,published_known=False) for i,s in enumerate(self._stats)]

    def counters(self):return [dict(s) for s in self._stats]

    @property
    def failure(self):return self._failure

    def stop_admission(self):
        self._closed=True
        for event in self._events:event.set()

    async def close(self):
        self.stop_admission()
        await _finish(asyncio.gather(*self._tasks,return_exceptions=True))
        for history in self._packets:history.clear()
        for s in self._stats:s['retained']=0
