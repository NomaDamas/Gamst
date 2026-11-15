"""Concrete persistent original-recording source, SDK and semantic lane owner."""
import asyncio
import hashlib
import os
from pathlib import Path
import sysconfig
import time

from gamst import Gamst,SensorStreams,TextContextCalls,VisualContextCalls
from gamst.call_controller import CallController
from gamst.call_budget import CallBudget
from gamst.call_policy import CallPolicy
from gamst.evaluation_publication import AcceptedObservationSink
from gamst.fill_conflict import FillConflictRuntime
from gamst.streams import _finish
from .owned_capture import OwnedCaptureSource,CaptureCapacity
from .recorded_source import digest,encoded
from .recording_native_validation import open_original_directory
from .recording_artifacts import RecordingArtifacts
from .recording_capture_router import RecordingCaptureRouter
from .recording_dependencies import verify_dependencies,VERIFICATION_WORKSPACE_BYTES
from .recording_runtime_inputs import RuntimeInputReceipt
from .recording_runtime_native import validate_runtime_native_inventory
from .recording_native_validation import canonical_lane_inventory,iter_canonical_inventory
from .recording_runtime_profiles import builtin_profile
from .recording_perception_runtime import RecordingPerceptionRuntime
from .recording_transports import (FixtureCompletion,LocalChatCompletions,
    TransportCapacity,TransportMemoryBudget,TransportOutcomeSink)
from .replay_clock import ReplayClock


def numeric_inputs(documents):
    if documents['runtime/policy.json']['schema_version']==1:return None
    return dict(opportunities=documents['runtime/opportunities.json'],exposure_policy=documents['runtime/exposure.json'],
        financial_adapter_inventory=documents['runtime/financial-adapters.json'])


def source_ref(ref):
    return dict(path='source/'+ref['path'],file_sha256=ref['sha256'],bytes=ref['bytes'])


class _TransportSink(TransportOutcomeSink):
    def __init__(self,artifacts):self.artifacts=artifacts
    def record(self,row):self.artifacts.record_transport(row)


class _PublicationSink(AcceptedObservationSink):
    def __init__(self,artifacts):self.artifacts=artifacts;self.mapping_sha256=None
    def record(self,record,operation_reference):
        if self.mapping_sha256 is None:raise RuntimeError('no mapped source before publication')
        from .recording_native_validation import iter_canonical_operations
        # Actual SDK SensorStreams passes an issued reference for this operation;
        # its original capture refs are held by the immutable session ticket.
        refs=self.owner.capture_references(operation_reference)
        self.artifacts.record_publication(record,binding_index=operation_reference.binding_index,
            builder_sha256=None,operation_reference=operation_reference,capture_refs=refs,
            role='specialist_result',origin_result_sha256=None,mapping_sha256=self.mapping_sha256)


class RecordingArmRuntime:
    @classmethod
    async def open(cls,inputs,*,owner,artifacts,receipt_router=None):
        """Startup actual resources before playback; no recording clock exists yet."""
        if type(inputs.receipt) is not RuntimeInputReceipt:raise TypeError('owned original input authority required')
        self=cls();self.inputs=inputs;self.receipt=inputs.receipt;self.owner=owner;self.artifacts=artifacts
        self.docs=self.receipt.documents();self.declarations=self.receipt.declarations();self.bundles=self.receipt.source_bundles()
        self.execution=artifacts.execution;self.manifest=self.declarations['source_manifest'];self.tasks=self.declarations['tasks']
        self.u=self.docs['runtime/policy.json'];self.arm=next(a for a in self.u['arms'] if a['arm']==self.execution['arm'])
        self.recording=next(r for r in self.u['recordings'] if r['recording_id']==self.manifest['recording_id'])
        self.policy=next(r for r in self.arm['recording_policies'] if r['recording_id']==self.manifest['recording_id'])
        arm_binding=next(a for a in self.declarations['collection']['arms'] if a['arm']==self.execution['arm'])
        self.bindings=arm_binding['provider_bindings'];self.binding_index={b['binding_id']:i for i,b in enumerate(self.bindings)}
        self.receipt_router=receipt_router;self.finance=None;self.lanes={};self.controllers={};self.clock=None;self.mapping_ref=None;self.source_mappings_ref=None
        self.human=self.streams=self.source=self.router=self.perception=self.task_runtime=self.journal=self.fill_driver=None
        self._close_task=None;self.jobs=[];self.failures={};self.clock_state='unknown';self.cleanup_finished_ns=None
        self._lane_opened={};self._lane_closed={};self._delivery_lanes=[];self._source_terminal=None;self.capture_inventory=None
        try:
            await self._startup();return self
        except BaseException:
            self._fail('startup_failed')
            await self.close();raise

    def _fail(self,reason):self.failures[reason]=self.failures.get(reason,0)+1

    def _document(self,sha):
        matches=[v for p,v in self.docs.items() if p.startswith(('configurations/','responses/','budgets/')) and digest(v)==sha]
        if len(matches)!=1:raise ValueError('one exact frozen configuration document required')
        return matches[0]

    async def _startup(self):
        # Admission includes temporary dependency verification and retained parse
        # workspace before importing/constructing selected providers or clients.
        maxima={s['source_id']:max(r['payload_bytes'] for r in self.manifest['records'] if r['source_id']==s['source_id']) for s in self.manifest['sources']}
        source_bytes=4*sum(maxima.values())
        delivery_bytes=8*sum(maxima[d['source_id']] for d in self.policy['deliveries'] if d['profile']!='acquire_only')
        active_bytes=sum(max(maxima.values())*4 for _ in self.arm['lanes'])
        metadata=self.inputs.workspace_bytes+16*1024**2+8*65536+4*1024**2
        if self.u['schema_version']==2:metadata+=1024**2+65536
        if source_bytes+delivery_bytes+active_bytes+metadata>self.u['limits']['max_owned_media_bytes']:
            raise ValueError('joint source/SDK/active/context/runtime memory cannot fit')
        if self.inputs.workspace_bytes+VERIFICATION_WORKSPACE_BYTES>256*1024**2:
            raise ValueError('dependency verification plus retained metadata exceeds owned workspace')
        assets=self.docs['runtime/assets.json'];self.asset_inventory=assets
        roots=(Path(sysconfig.get_paths()['data']),);metadata_roots=(Path(sysconfig.get_paths()['purelib']),)
        job=asyncio.create_task(asyncio.to_thread(verify_dependencies,assets['dependencies'],allowed_roots=roots,metadata_roots=metadata_roots))
        try:self.dependencies=await asyncio.shield(job)
        except asyncio.CancelledError:await _finish(job);raise
        # Only the installed fixed implementation is activated. JSON hashes are
        # declarations checked against its actual bytes, never import paths.
        from . import recording_transports
        transport_sha=hashlib.sha256(Path(recording_transports.__file__).read_bytes()).hexdigest()
        adapters={a['adapter_id']:a for a in self.u['adapters']}
        for lane in self.arm['lanes']:
            adapter=adapters[lane['adapter_id']]
            actual_sha=transport_sha
            if adapter['implementation']=='smolvlm_process_v1':
                from .recording_smol_runtime import implementation_sha256
                actual_sha=implementation_sha256()
            if adapter['implementation_sha256']!=actual_sha or adapter['dependency_inventory_sha256']!=self.dependencies.sha256:
                raise ValueError('actual installed completion implementation/dependencies differ from frozen adapter')
            if lane['role']!='text_answer':
                self.dependencies.require('numpy');self.dependencies.require('opencv-python')
        self.journal=await RecordingArtifacts.open(self.artifacts.output/'source',declarations={
            'source-manifest.json':self.manifest,'tasks.json':self.tasks,'profiles.json':self.declarations['profiles'],
            'execution.json':self.execution},predicted_events=2*len(self.manifest['records'])+2*len(self.manifest['sources'])+16)
        fd=self.inputs.duplicate_payload_root(self.manifest['recording_id'])
        source_lateness=50_000_000 if self.u['schema_version']==1 else self.docs['runtime/exposure.json']['source_policy']['max_acquisition_lateness_ns']
        try:self.source=await OwnedCaptureSource.preflight(self.manifest,fd,CaptureCapacity(source_bytes,max_lateness_ns=source_lateness),journal=self.journal)
        finally:os.close(fd)
        self.human=Gamst(consent=True)
        from .recorded_replay import MAPPING
        self.human.register_capability(MAPPING)
        sink=_PublicationSink(self.artifacts);sink.owner=self.owner;self.publication_sink=sink
        self.streams=SensorStreams(self.human,capacity=2,max_streams=16,max_packet_bytes=max(maxima.values()),
            max_buffer_bytes=max(1,delivery_bytes),evaluation_publication_sink=sink)
        for delivery in self.policy['deliveries']:
            if delivery['profile']=='acquire_only':continue
            if delivery['profile'] in ('opencv_lk','opencv_fill'):
                self.dependencies.require('numpy');self.dependencies.require('opencv-python')
            spec,factory,config=builtin_profile(delivery,self._document(delivery['config_sha256']))
            if spec.id not in self.streams.factories:self.streams.register(spec,factory)
            source=next(s for s in self.manifest['sources'] if s['source_id']==delivery['source_id'])
            bi=self.binding_index[delivery['binding_id']]
            await self.streams.open(source['source_id'],source['sensor_id'],[spec.id],
                evaluation_bindings={spec.id:self.owner.stream_scope(bi)},evaluation_configurations={spec.id:config})
            stream=self.streams.streams[(source['source_id'],source['sensor_id'])]
            self._delivery_lanes.extend((self.manifest['sources'].index(source),bi,lane) for lane in stream['lanes'])
        self._provider_epochs={s['source_id']:tuple(dict(capability_id=l['spec'].id,stream_epoch=l['epoch'])
            for l in self.streams.streams.get((s['source_id'],s['sensor_id']),{}).get('lanes',())) for s in self.manifest['sources']}
        for original in self.arm['controllers']:
            controller=CallController(policy=CallPolicy(**original['policy']),budget=CallBudget(**self._document(original['budget_sha256'])),max_requests=original['max_requests'])
            self.human.register_call_controller(original['controller_id'],controller);self.controllers[original['controller_id']]=controller
        if self.u['schema_version']==2:
            if self.receipt_router is not None:raise ValueError('U2 requires the owned original financial router')
            from .recording_runtime_finance import open_fixture_receipts
            self.finance=open_fixture_receipts(self.docs,self.execution,self.controllers,
                {lane['lane_id']:self.binding_index[lane['binding_id']] for lane in self.arm['lanes']},
                dict(self.owner.context),self.artifacts.output,self.artifacts.budget,
                retain_financial_documents=self.declarations['scope_protocol']['retain_financial_documents'])
            self.receipt_router=self.finance.router
        self.memory=TransportMemoryBudget(self.u['limits']['max_owned_media_bytes']-source_bytes-delivery_bytes-active_bytes-metadata)
        for lane in self.arm['lanes']:
            declaration=adapters[lane['adapter_id']];bi=self.binding_index[lane['binding_id']]
            controller=self.controllers[lane['controller_id']]
            self.owner.bind_semantic_lane(controller,bi)
            if declaration['implementation']=='smolvlm_process_v1':
                from .recording_smol_runtime import open_smol_lane
                client=await open_smol_lane(self.human,lane['controller_id'],declaration,self.asset_inventory,self.docs,self.owner,bi,self.dependencies,
                    scheduler=self.policy['periodic']['scheduler'])
                self._lane_opened[lane['lane_id']]=time.monotonic_ns()
                self.lanes[lane['lane_id']]=dict(completion=client,controller=controller,policy=lane,binding_index=bi,adapter=client.adapter,declaration=declaration)
                continue
            visual=lane['role']!='text_answer'
            visual_records=[r for r in self.manifest['records'] if r['format']=='bgr8_hwc']
            decoded=max(r['payload_bytes'] for r in visual_records)
            png=min(8*1024**2,decoded+max(r['shape'][0] for r in visual_records)*32+65536)
            request=min(12*1024**2,262144+4*((png+2)//3)+8192)
            cap=TransportCapacity(request if visual else 262144,png if visual else 0,decoded if visual else 0)
            args=dict(invocation_ordinal=self.execution['invocation_ordinal'],lane_id=lane['lane_id'],binding_index=bi,
                context=dict(self.owner.context),capacity=cap,memory=self.memory)
            if declaration['implementation']=='fixture_completion_v1':
                client=await FixtureCompletion.open(declaration,self._document(declaration['config']['responses_sha256']),receipt_router=self.receipt_router,**args)
            else:client=await LocalChatCompletions.open(declaration,dependency_inventory=self.dependencies,sink=_TransportSink(self.artifacts),**args)
            self._lane_opened[lane['lane_id']]=time.monotonic_ns()
            controller=self.controllers[lane['controller_id']]
            value=dict(completion=client,controller=controller,policy=lane,binding_index=bi,adapter=None,declaration=declaration)
            self.lanes[lane['lane_id']]=value
            self.owner.bind_backend(bi,client,'callback')
            if lane['role'] not in ('fill_small','fill_strong'):
                value['adapter']=(VisualContextCalls if visual else TextContextCalls)(self.human,lane['controller_id'],client,
                    provider=declaration['provider'],model_revision=declaration['model_revision'],evaluation_observer=self.owner,
                    timeout=declaration['config'].get('timeout_ms',30000)/1000)
        fill=self.policy['fill_chain']
        self.fill_binding_index=None
        if fill is not None:
            small=self.lanes[fill['small_lane_id']];strong=None if fill['strong_lane_id'] is None else self.lanes[fill['strong_lane_id']]
            def client(value):return dict(completion=value['completion'],provider=value['declaration']['provider'],model_revision=value['declaration']['model_revision'])
            self.fill_driver=FillConflictRuntime(self.human,'recording-fill',small['policy']['controller_id'],
                fill_config=self._document(fill['fill_config_sha256']),small_client=client(small),strong_client=None if strong is None else client(strong),
                strong_opt_in=fill['strong_opt_in'],charge_ceilings=fill['charge_ceilings'],chain_ceiling=fill['chain_ceiling'],
                pricing_revision=fill['pricing_revision'],unit=fill['unit'],timeout=fill['stage_timeout_ms']/1000,
                retention_ms=fill['retention_ms'],evaluation_observer=self.owner)
            small['adapter']=self.fill_driver.small
            if strong is not None:strong['adapter']=self.fill_driver.strong
            matches=[i for i,b in enumerate(self.bindings) if b['operation_kinds']==['process'] and b['entry_boundary']=='in_process_backend' and b['config_sha256']==fill['fill_config_sha256'] and fill['source_id'] in b['source_ids']]
            if len(matches)!=1:raise ValueError('one frozen actual direct fill binding required')
            self.fill_binding_index=matches[0];self.owner.bind_backend(matches[0],self.fill_driver.fill,'in_process')

    async def start(self,*,playback_started,answer_ready):
        try:return await self._start(playback_started=playback_started,answer_ready=answer_ready)
        except BaseException as error:
            if not self.failures:self._fail('clock_discontinuity' if 'clock discontinuity' in str(error) else 'source_failed')
            raise
        finally:await self.close()

    async def _start(self,*,playback_started,answer_ready):
        """Called only after the fixed parent START; anchor is never refreshed."""
        if self.clock is not None:raise RuntimeError('recording playback is single use')
        self.clock=ReplayClock(self.manifest);self.owner.bind_mapping(self.clock.mapping,self.clock.sha256)
        self.publication_sink.mapping_sha256=self.clock.sha256
        self.source_mappings=[]
        for source in self.manifest['sources']:
            epochs=list(self._provider_epochs[source['source_id']])
            self.source_mappings.append(dict(**self.clock.mapping,mapping_sha256=self.clock.sha256,original_source=source,provider_epochs=epochs))
        self.router=RecordingCaptureRouter(self.manifest,streams=self.streams,source_bindings=self.recording['source_bindings'],subject_id=self.recording['subject_id'],
            delivery_sources=[d['source_id'] for d in self.policy['deliveries'] if d['profile']!='acquire_only'],
            clock=self.clock,artifacts=self.artifacts,delivery_bindings={s['source_id']:[self.binding_index[d['binding_id']]
                for d in self.policy['deliveries'] if d['source_id']==s['source_id'] and d['profile']!='acquire_only'] for s in self.manifest['sources']})
        self.jobs=[self.source.start(self.clock,self.router)]
        self.mapping_ref=await self.artifacts.write_metadata('mapping',self.clock.mapping)
        self.source_mappings_ref=await self.artifacts.write_metadata('source-mappings',self.source_mappings)
        await self.journal.write_json('mapping.json',self.clock.mapping)
        await self.journal.write_json('source-mappings.json',self.source_mappings)
        from .recorded_replay import MAPPING
        from gamst.capabilities import SensorMetadata
        mapping_records=[]
        for value in self.source_mappings:
            original=value['original_source']
            accepted=await self.human.push_observation(MAPPING.id,dict(mapping=value),
                sensor=SensorMetadata(original['source_id'],original['sensor_id'],0,stream_epoch=original['stream_epoch']),
                subject_id=self.recording['subject_id'],timestamp_ms=self.clock.mapping['sdk_anchor_ms'])
            mapping_records.append(accepted)
        mapping_records_ref=await self.journal.write_json('mapping-records.json',mapping_records,max_bytes=65536)
        for ordinal,(value,accepted) in enumerate(zip(self.source_mappings,mapping_records)):
            self.journal.emit('mapping_published',dict(source_id=value['original_source']['source_id'],mapping_sha256=self.clock.sha256,
                source_mapping_sha256=digest(value),record_sha256=digest(accepted),record_ordinal=ordinal,
                mapping_records_sha256=mapping_records_ref['sha256']))
        await playback_started(self.mapping_ref,self.source_mappings_ref,self.clock)
        self.perception=RecordingPerceptionRuntime(human=self.human,owner=self.owner,clock=self.clock,manifest=self.manifest,
            subject_id=self.recording['subject_id'],source_bindings=self.recording['source_bindings'],router=self.router,policy=self.policy,
            lanes=self.lanes,artifacts=self.artifacts,prompt=None if self.policy['periodic'] is None else self._document(self.policy['periodic']['prompt_policy_sha256']),
            fill_driver=self.fill_driver,fill_binding_index=self.fill_binding_index,receipt_router=self.receipt_router)
        from .recording_task_runtime import RecordingTaskRuntime
        publisher=self.human.create_evaluation_snapshot_publisher(self.recording['subject_id'],
            implementation_sha256=__import__('gamst.evaluation_snapshot',fromlist=['evaluation_snapshot_implementation_sha256']).evaluation_snapshot_implementation_sha256())
        self.task_runtime=RecordingTaskRuntime(self.human,self.owner,self.clock,self.manifest,self.tasks,self.recording['subject_id'],publisher,
            {lid:{k:v for k,v in value.items() if k!='declaration'} for lid,value in self.lanes.items()},self.policy['text_lane_ids'],self.arm['task_queue']['capacity'],self.artifacts,answer_ready,self.router.watermarks,receipt_router=self.receipt_router)
        self.jobs.extend([asyncio.create_task(self.perception.run(self.recording['primary_visual_source_id'])),asyncio.create_task(self.task_runtime.run())])
        try:
            await asyncio.gather(*self.jobs)
            if self.router.failure:raise RuntimeError('recorded SDK delivery failed: '+self.router.failure)
            self.clock.check();self.clock_state='consistent'
        except BaseException as error:
            self._fail('clock_discontinuity' if 'clock discontinuity' in str(error) else 'source_failed');raise
        finally:await self.close()
        if self.failures:raise RuntimeError('recording runtime incomplete')

    async def close(self):
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        try:await asyncio.shield(self._close_task)
        except asyncio.CancelledError:await _finish(self._close_task);raise

    async def _close(self):
        if self.source is not None:self.source.stop_admission('runtime_cleanup')
        for job in self.jobs:
            if not job.done():job.cancel()
        if self.jobs:await _finish(asyncio.gather(*self.jobs,return_exceptions=True))
        cleanup=[]
        async def finish(obj):
            if obj is None:return
            try:await obj.close()
            except BaseException as error:cleanup.append(error)
        for obj in (self.task_runtime,self.perception,self.source,self.router):await finish(obj)
        before={id(lane):(lane['queue'].qsize(),lane['queue']._unfinished_tasks) for _,_,lane in self._delivery_lanes}
        await finish(self.fill_driver)
        for lid,lane in self.lanes.items():
            await finish(lane['adapter']);await finish(lane['completion'])
            if lane['declaration']['implementation']!='smolvlm_process_v1':
                try:self.owner.backend_closed(lane['binding_index'],lane['completion'])
                except Exception as error:cleanup.append(error)
            self._lane_closed[lid]=time.monotonic_ns()
        if self.fill_driver is not None:
            try:self.owner.backend_closed(self.fill_binding_index,self.fill_driver.fill)
            except Exception as error:cleanup.append(error)
        await finish(self.human)
        for si,bi,lane in self._delivery_lanes:
            queued,unfinished=before[id(lane)];residual=lane['accepted']-lane['processed']-lane['errors']-lane['dropped']
            if residual<0:
                self._fail('cleanup_incomplete')
                self.journal.emit('delivery_counter_invalid',dict(source_index=si,binding_index=bi,accepted=lane['accepted'],
                    processed=lane['processed'],errors=lane['errors'],evicted=lane['dropped'],residual=residual))
                continue
            self.artifacts.record_delivery_summary(dict(source_index=si,binding_index=bi,accepted=lane['accepted'],
                preclose_queued=queued,preclose_unfinished=unfinished,preclose_active=unfinished-queued,
                postclose_processed=lane['processed'],postclose_errors=lane['errors'],postclose_evicted=lane['dropped'],
                postclose_queued=lane['queue'].qsize(),postclose_unfinished=lane['queue']._unfinished_tasks,
                worker_task_cancelled=bool(lane['task'] is not None and lane['task'].cancelled()),
                residual_unresolved=residual,disposition_truth_verified=False))
        if cleanup:self._fail('cleanup_incomplete')
        if self.task_runtime is not None:
            for reason,count in self.task_runtime.failure_counts.items():self.failures[reason]=self.failures.get(reason,0)+count
        if self.perception is not None and self.perception.failures:self.failures['backend_failed']=len(self.perception.failures)
        if self.clock is not None:
            try:
                self.clock.check()
                if 'clock_discontinuity' not in self.failures:self.clock_state='consistent'
            except RuntimeError:self.clock_state='discontinuous';self._fail('clock_discontinuity')
        settlements=[];settlement_sha=None;financial_refs=[]
        if self.finance is not None:
            try:
                settlements=list(self.finance.actual_settlements())
                if len(settlements)>256:raise ValueError('actual SDK settlement inventory exceeds256 rows')
                settlement_sha=digest(settlements);financial_refs=list(self.finance.retained_references())
                settlement_ref=await self.journal.write_json('controller-settlements.json',settlements,max_bytes=1024**2)
                self.journal.emit('controller_settlement_inventory',dict(inventory_sha256=settlement_sha,
                    inventory_file_sha256=settlement_ref['sha256'],count=len(settlements)))
            except BaseException as error:
                self._fail('cleanup_incomplete');cleanup.append(error)
            finally:self.finance.close()
        self.cleanup_finished_ns=time.monotonic_ns()
        if self.journal is not None:
            # Acquisition is never started before a successful source preflight.
            # These are uninspected declared records, not verified zero work for
            # a source whose replay actually started.
            counters=self.source.counters() if self.source is not None else [dict(
                source_index=n,source_id=s['source_id'],declared=count,attempted=0,captured=0,dropped=0,
                verified_reads=0,offered=0,admitted=0,refused=0,uninspected_suffix=count)
                for n,s in enumerate(self.manifest['sources'])
                for count in [sum(r['source_id']==s['source_id'] for r in self.manifest['records'])]]
            self._source_terminal=await self.journal.close(dict(state='failed' if self.failures else 'replayed_unscored',
                failure_category=next(iter(self.failures),None),mapping_sha256=None if self.clock is None else self.clock.sha256,
                counters=counters,controller_settlement_inventory_sha256=settlement_sha,controller_settlement_count=len(settlements),
                financial_references=financial_refs,cleanup_finished_ns=self.cleanup_finished_ns,terminal_reached_ns=None if self.source is None else self.source.terminal_reached_ns,
                router_counters=[] if self.router is None else self.router.counters()),cleanup_finished=True)

    async def finalize(self,native_directory):
        """After owned SDK cleanup and native owner close, seal finite runtime output."""
        index=validate_runtime_native_inventory(self.declarations,native_directory,source_inventory=self.manifest,
            latency_policy=self.docs['runtime/latency.json'],source_bundles=self.bundles,numeric_inputs=numeric_inputs(self.docs))
        for lid,lane in self.lanes.items():
            inventories=[canonical_lane_inventory(index,si,lane['binding_index']) for si in range(len(self.tasks)+1)]
            counts={k:sum(r['counts'][k] for r in inventories) for k in ('entered','finished','published')}
            process=lane['declaration']['implementation']=='smolvlm_process_v1'
            workers={row['worker_ordinal'] for row in iter_canonical_inventory(index,binding_index=lane['binding_index']) if row['kind']=='worker_interval'}
            self.artifacts.record_adapter(dict(lane_id=lid,adapter_id=lane['policy']['adapter_id'],binding_index=lane['binding_index'],controller_id=lane['policy']['controller_id'],
                state='closed' if lid in self._lane_closed else 'incomplete',opened_ns=self._lane_opened.get(lid),closed_ns=self._lane_closed.get(lid),
                entry_count=counts['entered'],finished_count=counts['finished'],published_count=counts['published'],
                unknown_liability_count=sum(r['unknown_liability_count']+r['unknown_dispatch_liability_count'] for r in inventories),worker_ordinal=next(iter(workers)) if process and len(workers)==1 else None,
                attachment='owned_process' if process else 'external_async_callback',external_backend_closed='not_applicable' if process else 'unknown'))
        # Seal source evidence before the summary. Exact original268 JSON bytes
        # have no added LF and are referenced through their original reader.
        indices=await self.artifacts.seal_inventory()
        if self.mapping_ref is not None and self.source_mappings_ref is not None and self._source_terminal is not None:
            journal=self.journal.status()
            job=asyncio.create_task(asyncio.to_thread(_journal_ref,self.artifacts.output,journal['valid_prefix_bytes']))
            try:events=await asyncio.shield(job)
            except asyncio.CancelledError:await _finish(job);raise
            self.capture_inventory=await self.artifacts.write_metadata('capture-inventory',dict(schema_version=1,kind='recording_capture_inventory',
                execution_sha256=self.artifacts.execution_sha256,source_manifest_sha256=digest(self.manifest),mapping_sha256=self.clock.sha256,
                source_mappings=self.source_mappings_ref,source_terminal=source_ref(self._source_terminal),source_events=[events],
                source_counters=source_ref(self._source_terminal),delivery_outcomes=indices['deliveries']))
        results=[] if self.task_runtime is None else list(self.task_runtime.results)
        summary=dict(schema_version=1,kind='recording_child_summary',execution_sha256=self.artifacts.execution_sha256,
            mapping=self.mapping_ref,capture_inventory=self.capture_inventory,task_results=results,tick_inventory=indices['ticks'],
            adapter_outcomes=indices['adapters'],delivery_outcomes=indices['deliveries'],publication_inventory=indices['publications'],mapping_updates=indices['mapping-updates'],
            child_cleanup_finished=self.cleanup_finished_ns is not None,clock_state=self.clock_state,state='closed' if not self.failures else 'failed',failure_counts=dict(self.failures))
        ref=await self.artifacts.write_metadata('child-summary',summary)
        await self.artifacts.close(cleanup_finished=True)
        return ref,summary,index


def _journal_ref(directory,size):
    fd=open_original_directory(Path(directory)/'source')
    try:child=os.open('journal.jsonl',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    finally:os.close(fd)
    try:
        import stat
        before=os.fstat(child)
        if not stat.S_ISREG(before.st_mode) or before.st_size!=size or not 1<=size<=1024**3:
            raise ValueError('original bounded journal prefix required')
        sha=hashlib.sha256();count=0
        while count<=size:
            block=os.read(child,min(65536,size+1-count))
            if not block:break
            count+=len(block);sha.update(block)
        after=os.fstat(child)
        if count!=size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('original journal changed after closure')
        return dict(path='source/journal.jsonl',file_sha256=sha.hexdigest(),bytes=count)
    finally:os.close(child)
