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
        self.lane_refs = []; self.provider_outcomes = []; self.source_mappings = []
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
            # Freeze provenance before publication can await or fail.
            for source in self.manifest['sources']:
                lanes=next(s['providers'] for s in self.streams.status()['streams'] if s['source_id']==source['source_id'])
                value=dict(**self.clock.mapping,mapping_sha256=self.clock.sha256,
                    original_source=source,provider_epochs=[dict(capability_id=l['capability_id'],stream_epoch=l['stream_epoch']) for l in lanes])
                self.source_mappings.append(value)
            for value in self.source_mappings:
                source=value['original_source']
                await self.human.push_observation(MAPPING.id,dict(mapping=value),
                    sensor=SensorMetadata(source['source_id'],source['sensor_id'],0,stream_epoch=source['stream_epoch']),
                    timestamp_ms=self.clock.mapping['sdk_anchor_ms'])
                self.journal.emit('mapping_published',dict(source_id=source['source_id'],mapping_sha256=self.clock.sha256,
                    source_mapping_sha256=digest(value)))
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
        before = {id(lane):dict(queued=lane['queue'].qsize(),
            unfinished=lane['queue']._unfinished_tasks,
            active=lane['queue']._unfinished_tasks-lane['queue'].qsize())
            for _,lane in self.lane_refs}
        try:
            if self.human is not None:await self.human.close()
        finally:
            self.provider_outcomes=[]
            for key,lane in self.lane_refs:
                counts={k:lane[k] for k in ('accepted','processed','dropped','errors','resets')}
                unresolved=counts['accepted']-counts['processed']-counts['dropped']-counts['errors']
                task=lane['task']
                self.provider_outcomes.append(dict(source_id=key[0],sensor_id=key[1],
                    capability_id=lane['spec'].id,stream_epoch=lane['epoch'],**counts,
                    preclose_work=before[id(lane)],final_queued=lane['queue'].qsize(),
                    final_unfinished=lane['queue']._unfinished_tasks,
                    provider_task_done=task is None or task.done(),
                    provider_task_cancelled=False if task is None else task.cancelled(),
                    unresolved_delivery=unresolved,disposition_truth_verified=False,
                    unresolved_semantics='unknown_discarded_or_cancelled_not_source_loss'))
            self.lane_refs.clear()
            self.ended_ns=time.monotonic_ns()


async def _drain(task):
    await asyncio.gather(task,return_exceptions=True)


async def run_source_context_recording(manifest, tasks, profiles, output, *,
        asset_directory, retain_evaluation_artifacts=False,
        max_lateness_ns=50_000_000, max_drift_ns=50_000_000):
    """Record one owned source runtime; no semantic client or model is implicit.

    Artifacts retain source paths, original task inputs and sensory Context under
    a separate explicit owner. SDK deletion does not erase this evaluation copy.
    """
    if retain_evaluation_artifacts is not True:
        raise PermissionError('explicit separate evaluation artifact retention required')
    from .recording_artifacts import RecordingArtifacts
    import json
    profiles=json.loads(encoded(profiles))
    replay=RecordedSourceReplay(manifest,tasks,profiles,asset_directory,None,
        max_lateness_ns=max_lateness_ns,max_drift_ns=max_drift_ns)
    # Bound declarations + maximum detached snapshot envelopes before activating
    # any source/provider. These artifacts are not the old16MiB collector format.
    declarations={'source-manifest.json':replay.manifest,'tasks.json':replay.tasks,
        'profiles.json':profiles,'execution.json':dict(schema_version=1,
            asset_directory=str(asset_directory),max_lateness_ns=max_lateness_ns,
            max_wall_monotonic_drift_ns=max_drift_ns,source_schema='recorded-source-v1',
            retain_evaluation_artifacts=True, sdk_deletion_erases_artifacts=False)}
    predicted=2*len(replay.manifest['records'])+2*len(replay.tasks)+2*len(replay.manifest['sources'])+16
    if sum(len(encoded(v))+1 for v in declarations.values())+len(replay.tasks)*128*1024+1024*1024>80*1024**2:
        raise ValueError('joint declarations and snapshot artifact admission exceeds80MiB')
    artifacts=await RecordingArtifacts.open(output,declarations=declarations,predicted_events=predicted)
    replay.journal=artifacts; failure=None
    try:
        await replay.run()
    except BaseException as error:
        failure=error
    try:
        # run() joins every owned source/read/provider on all exits. Do not turn
        # an interrupted recording into success merely because cleanup completed.
        await replay.close()
        if replay.clock is not None:
            try:replay.clock.check()
            except RuntimeError as clock_error:
                if failure is None:failure=clock_error
        if replay.clock is not None:
            await artifacts.write_json('mapping.json',replay.clock.mapping,max_bytes=65536)
            await artifacts.write_json('source-mappings.json',replay.source_mappings,max_bytes=65536)
        for index, result in enumerate(replay.snapshots.values()):
            await artifacts.write_json(f'context-{index:04d}.json',result,max_bytes=128*1024)
        counters={sid:dict(value,uninspected_suffix=value['declared']-value['attempted'])
                  for sid,value in replay.counters.items()}
        summary=dict(schema_version=1,state='failed' if failure else 'replayed_unscored',
            failure_category=None if failure is None else type(failure).__name__,
            source_manifest_sha256=replay.manifest_sha256,tasks_sha256=digest(replay.tasks),
            mapping_sha256=None if replay.clock is None else replay.clock.sha256,
            source_mappings_sha256=digest(replay.source_mappings),
            observed_start_ns=replay.started_ns,observed_end_ns=replay.ended_ns,
            observed_elapsed_ns=replay.ended_ns-replay.started_ns,
            sources=counters,provider_outcomes=replay.provider_outcomes,
            snapshots=len(replay.snapshots),requested_snapshots=len(replay.tasks),
            source_availability_truth_verified=False,physical_device_losses=None,
            actual_acceptance='incomplete',quality_verified=False,monetary_cost=None,
            whole_pipeline_cost_verified=False,isolation_verified=False,
            cleanup_finished=True,execution_kind='source_context_replay_not_abc')
        await artifacts.close(summary,cleanup_finished=True)
    except BaseException as cleanup:
        await artifacts.abort()
        if failure is not None:raise failure from cleanup
        raise
    if failure is not None:raise failure
    return summary


def main():
    import argparse
    import json
    import os
    import stat
    from .recorded_source import _pairs, MAX_MANIFEST_BYTES
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','tasks','profiles','assets','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--retain-evaluation-artifacts',action='store_true',
        help='Explicitly retain original inputs, source paths and sensory Context outside SDK deletion')
    parser.add_argument('--max-lateness-ms',type=int,default=50)
    parser.add_argument('--max-clock-drift-ms',type=int,default=50)
    args=parser.parse_args()

    def load(path,limit):
        fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=limit:
                raise ValueError('bounded regular declaration file required')
            chunks=[];remaining=info.st_size
            while remaining:
                chunk=os.read(fd,min(65536,remaining))
                if not chunk:raise ValueError('declaration truncated during read')
                chunks.append(chunk);remaining-=len(chunk)
            if os.read(fd,1):raise ValueError('declaration grew during read')
            def invalid(_):raise ValueError('nonfinite declaration JSON')
            return json.loads(b''.join(chunks),object_pairs_hook=_pairs,parse_constant=invalid)
        finally:os.close(fd)

    summary=asyncio.run(run_source_context_recording(load(args.manifest,MAX_MANIFEST_BYTES),
        load(args.tasks,4*1024**2),load(args.profiles,65536),args.output,
        asset_directory=args.assets,retain_evaluation_artifacts=args.retain_evaluation_artifacts,
        max_lateness_ns=args.max_lateness_ms*1_000_000,max_drift_ns=args.max_clock_drift_ms*1_000_000))
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
