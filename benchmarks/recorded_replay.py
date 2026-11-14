"""Owned 1x source replay and timed real SDK Context snapshots (not ABC scoring)."""
import asyncio
import time

from humansense import HumanSense
from humansense.capabilities import CapabilitySpec, OutputField, SensorMetadata
from humansense.plugins import InputFrame
from humansense.streams import SensorStreams, _finish
from .recorded_source import validate_manifest, read_capture, digest, encoded, _integer
from .replay_clock import ReplayClock, validate_tasks
from .recorded_profiles import profiles_for


MAPPING = CapabilitySpec('replay.mapping.v1', 'replay.provenance', 'text',
    'gamst.recorded_source', '1', (OutputField('mapping', 'object'),), 'frame',
    freshness_ms=3600000, interpretation='measurement')


class RecordedSourceReplay:
    def __init__(self, manifest, tasks, profiles, asset_directory, journal, *,
                 max_lateness_ns=50_000_000, max_drift_ns=50_000_000):
        self.manifest = validate_manifest(manifest)
        self.tasks = validate_tasks(tasks, self.manifest['duration_ns'])
        self.profiles, self.admission = profiles_for(self.manifest, profiles)
        self.asset_directory = asset_directory
        self.journal = journal
        self.max_lateness = _integer(max_lateness_ns, 1, 1_000_000_000)
        self.max_drift = _integer(max_drift_ns, 1, 1_000_000_000)
        self.human = self.streams = self.clock = None
        self.jobs = set(); self.reads = set(); self.closed = False
        self.lane_refs = []; self.provider_outcomes = []
        self.close_task = self.cleanup_task = self.run_task = None; self.snapshots = {}; self.snapshot_bytes = 0
        self.counters = {s['source_id']:dict(declared=sum(r['source_id']==s['source_id'] for r in self.manifest['records']),
            attempted=0,captured=0,dropped=0,verified_reads=0,submitted=0,submit_refused=0,
            last_submitted_sequence=None) for s in self.manifest['sources']}
        self.manifest_sha256 = digest(self.manifest)
        self.records_by_source = {s['source_id']:[] for s in self.manifest['sources']}
        for ordinal, record in enumerate(self.manifest['records']):
            self.records_by_source[record['source_id']].append((ordinal,record))
        self.started_ns = time.monotonic_ns()

    async def _read(self, record):
        task = asyncio.create_task(asyncio.to_thread(read_capture, record, self.asset_directory))
        self.reads.add(task)
        try:
            try:return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _finish(task)
                raise
        finally:self.reads.discard(task)

    async def _until(self, offset):
        target = self.clock.monotonic_ns+offset
        delay = target-time.monotonic_ns()
        if delay > 0: await asyncio.sleep(delay/1e9)
        if self.closed: raise asyncio.CancelledError()
        return self.clock.check()

    async def _source(self, source):
        sid = source['source_id']; counters = self.counters[sid]
        for ordinal, record in self.records_by_source[sid]:
            payload = None; failure = None
            try:
                payload = await self._read(record)
                counters['verified_reads'] += 1
            except (OSError, ValueError): failure = 'payload_verification_failed'
            now = await self._until(record['offset_ns'])
            counters['attempted'] += 1
            timing = self.clock.capture(record['offset_ns'])
            late = now-timing['scheduled_monotonic_ns'] > self.max_lateness
            if failure or late:
                counters['dropped'] += 1
                self.journal.emit('capture_drop',dict(source_id=sid,ordinal=ordinal,
                    sequence=record['sequence'],reason=failure or 'acquisition_late',**timing))
                if failure: raise ValueError('recording source changed or unavailable')
                payload = None
                await asyncio.sleep(0)
                continue
            counters['captured'] += 1
            self.journal.emit('capture_acquired',dict(source_id=sid,ordinal=ordinal,
                sequence=record['sequence'],payload_sha256=record['payload_sha256'],**timing))
            if source['modality']=='video':
                import numpy as np
                data = np.frombuffer(payload,dtype=np.uint8).reshape(record['shape'])
            else:data = payload
            sensor = SensorMetadata(sid,source['sensor_id'],record['sequence'],
                calibration_id=source['calibration_id'],stream_epoch=None,
                capture_sequence=record['sequence'])
            frame = InputFrame(data,source['modality'],timing['sdk_timestamp_ms'],
                sample_rate=record.get('sample_rate'),sensor=sensor)
            try:
                await self.streams.submit(frame)
            except (ValueError, RuntimeError):
                counters['submit_refused'] += 1
                self.journal.emit('delivery_refused',dict(source_id=sid,ordinal=ordinal,sequence=record['sequence']))
            else:
                counters['submitted'] += 1; counters['last_submitted_sequence']=record['sequence']
                self.journal.emit('delivery_admitted',dict(source_id=sid,ordinal=ordinal,sequence=record['sequence']))
            data = payload = frame = None
            await asyncio.sleep(0)

    async def _snapshot(self, task):
        await self._until(task['opens_ns'])
        self.journal.emit('task_opened',dict(task_id=task['task_id'],input_sha256=task['input_sha256']))
        begin = await self._until(task['snapshot_ns'])
        deadline = self.clock.monotonic_ns+task['deadline_ns']
        if begin >= deadline:
            self.journal.emit('snapshot_failed',dict(task_id=task['task_id'],reason='deadline_before_snapshot'))
            return
        watermark = {sid:c['last_submitted_sequence'] for sid,c in self.counters.items()}
        context = await self.human.get_context(window=task['window'])
        end = self.clock.check(); raw = encoded(context)
        if end >= deadline or len(raw)>65536 or self.snapshot_bytes+len(raw)>4*1024**2:
            self.journal.emit('snapshot_failed',dict(task_id=task['task_id'],reason='deadline_or_byte_bound'))
            return
        import json
        context=json.loads(raw);self.snapshot_bytes+=len(raw)
        included = [dict(evidence_id=o['evidence_id'],sensor=o.get('sensor')) for o in context['observations'].values()]
        value = dict(task_id=task['task_id'],input_sha256=task['input_sha256'],
            task_sha256=digest(task),source_manifest_sha256=self.manifest_sha256,
            mapping_sha256=self.clock.sha256,scheduled_snapshot_ns=self.clock.monotonic_ns+task['snapshot_ns'],
            observed_begin_ns=begin,observed_end_ns=end,submitted_watermark=watermark,
            included_evidence=included,context_sha256=digest(context),context=context)
        self.snapshots[task['task_id']]=value
        self.journal.emit('snapshot_taken',dict(task_id=task['task_id'],context_sha256=value['context_sha256'],
            observed_begin_ns=begin,observed_end_ns=end))

    async def run(self):
        if self.closed or self.run_task is not None:raise RuntimeError('recording runtime is single-use')
        self.run_task=asyncio.create_task(self._run())
        try:return await asyncio.shield(self.run_task)
        except asyncio.CancelledError:
            self.run_task.cancel()
            await _finish(asyncio.create_task(_drain(self.run_task)))
            raise

    async def _run(self):
        self.journal.emit('runtime_starting',dict(admission=self.admission))
        try:
            # Actual bytes are preflighted before providers activate and verified
            # again at acquisition. Neither read is hidden as free model work.
            verified = set()
            for record in self.manifest['records']:
                if record['payload_path'] not in verified:
                    await self._read(record); verified.add(record['payload_path'])
            self.human = HumanSense(consent=True)
            self.streams = SensorStreams(self.human,capacity=2,max_streams=16,
                max_packet_bytes=8*1024**2,max_buffer_bytes=256*1024**2)
            for spec,factory in self.profiles.values():
                if spec.id not in self.streams.factories:self.streams.register(spec,factory)
            for source in self.manifest['sources']:
                await self.streams.open(source['source_id'],source['sensor_id'],[self.profiles[source['source_id']][0].id])
            self.lane_refs = [(key,lane) for key,stream in self.streams.streams.items() for lane in stream['lanes']]
            self.human.register_capability(MAPPING)
            self.clock = ReplayClock(self.manifest,max_drift_ns=self.max_drift)
            for source in self.manifest['sources']:
                lanes=next(s['providers'] for s in self.streams.status()['streams'] if s['source_id']==source['source_id'])
                value=dict(**self.clock.mapping,mapping_sha256=self.clock.sha256,
                    original_source=source,provider_epochs=[dict(capability_id=l['capability_id'],stream_epoch=l['stream_epoch']) for l in lanes])
                await self.human.push_observation(MAPPING.id,dict(mapping=value),
                    sensor=SensorMetadata(source['source_id'],source['sensor_id'],0,stream_epoch=source['stream_epoch']),
                    timestamp_ms=self.clock.mapping['sdk_anchor_ms'])
                self.journal.emit('mapping_published',dict(source_id=source['source_id'],mapping_sha256=self.clock.sha256))
            self.jobs={asyncio.create_task(self._source(s)) for s in self.manifest['sources']}
            self.jobs.update(asyncio.create_task(self._snapshot(t)) for t in self.tasks)
            await asyncio.gather(*self.jobs)
            await self._until(self.manifest['duration_ns'])
            self.clock.check()
            self.last_status=self.streams.status()
            for sid,counters in self.counters.items():
                self.journal.emit('acquisition_finished',dict(source_id=sid,counters=counters))
            for source in self.manifest['sources']:
                await self.streams.finish(source['source_id'],source['sensor_id'])
            self.clock.check()
        finally:
            await self._cleanup()

    async def close(self):
        self.closed=True
        if self.close_task is None:self.close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self.close_task)
        except asyncio.CancelledError:
            await _finish(self.close_task)
            raise

    async def _close(self):
        if self.run_task is not None and not self.run_task.done():self.run_task.cancel()
        if self.run_task is not None:await _drain(self.run_task)
        await self._cleanup()

    async def _cleanup(self):
        if self.cleanup_task is None:self.cleanup_task=asyncio.create_task(self._cleanup_impl())
        try:await asyncio.shield(self.cleanup_task)
        except asyncio.CancelledError:
            await _finish(self.cleanup_task)
            raise

    async def _cleanup_impl(self):
        self.closed=True
        for task in self.jobs:
            if not task.done():task.cancel()
        if self.jobs:await asyncio.gather(*self.jobs,return_exceptions=True)
        for task in tuple(self.reads):await _finish(task)
        if self.human is not None:await self.human.close()
        self.provider_outcomes=[dict(source_id=key[0],sensor_id=key[1],capability_id=lane['spec'].id,
            stream_epoch=lane['epoch'],**{k:lane[k] for k in ('accepted','processed','dropped','errors','resets')})
            for key,lane in self.lane_refs]
        self.lane_refs.clear()
        self.ended_ns=time.monotonic_ns()


async def _drain(task):
    await asyncio.gather(task,return_exceptions=True)
