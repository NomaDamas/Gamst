import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from benchmarks.recording_runtime_artifacts import (RecordingRuntimeArtifacts,
    iter_outcome_rows, read_ref, preflight_runtime_artifacts, KINDS, _line,
    validate_task_artifact,read_publication)
from benchmarks.recording_native_writer import ArtifactBudget
from benchmarks.recording_native_schema import digest
from gamst import Gamst,SensorMetadata
from gamst.capabilities import CapabilitySpec,OutputField

H='a'*64


def execution():
    return dict(schema_version=1,kind='recording_execution',runtime_policy_sha256=H,
        collection_sha256=H,scope_protocol_sha256=H,instrumentation_protocol_sha256=H,
        invocation_ordinal=0,recording_index=0,arm='A',stream_id='stream',
        execution_namespace='exec',cache_namespace='cache',asset_inventory_sha256=H)


def tick(n=0):
    return dict(tick_index=n,count=1,first_due_offset_ns=n*10,period_ns=10,policy='periodic',
        source_index=0,observed_begin_ns=n*10,observed_end_ns=n*10,outcome='no_frame',
        capture_ref=None,request_id=None,operation_instance=None,case=None)


class RuntimeArtifacts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='.');self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'runtime'
        self.budget=ArtifactBudget();self.writer=await RecordingRuntimeArtifacts.open(
            self.path,execution(),budget=self.budget,retain_evaluation_artifacts=True)
        self.addAsyncCleanup(self.writer.abort)

    async def test_pages_canonical_hashchain_task_failure_and_exclusive_outputs(self):
        for n in range(260):
            self.writer.record_tick(tick(n))
            if n%16==15:await asyncio.sleep(.002)
        value=dict(schema_version=1,kind='recording_task_result',execution_sha256=digest(execution()),
            task_id='task',input_sha256=H,scope_index=1,status='failed',reason='admission_refused',
            evidence=None,answer_text=None,answer_sha256=None,operation_instance=None,
            child_finished_ns=time.monotonic_ns(),cleanup_finished=False)
        result=await self.writer.write_task(0,'result',value)
        self.assertEqual(read_ref(self.path,result),value)
        with self.assertRaises(ValueError):await self.writer.write_task(0,'result',value)
        refs=await self.writer.seal_inventory()
        rows=list(iter_outcome_rows(self.path,refs['ticks'],execution_sha256=digest(execution()),kind='ticks'))
        self.assertEqual(rows,[tick(n) for n in range(260)])
        index=read_ref(self.path,refs['ticks']);self.assertEqual(index['page_count'],3)
        self.assertEqual(index['state'],'complete')
        self.assertEqual(list(iter_outcome_rows(self.path,refs['adapters'],execution_sha256=digest(execution()),kind='adapters')),[])
        # Final child summary is sealed after indices, never overwrites them.
        summary_value=dict(schema_version=1,kind='recording_child_summary',execution_sha256=digest(execution()),
            mapping=None,capture_inventory=None,task_results=[result],tick_inventory=refs['ticks'],
            adapter_outcomes=refs['adapters'],delivery_outcomes=refs['deliveries'],publication_inventory=refs['publications'],
            mapping_updates=refs['mapping-updates'],child_cleanup_finished=True,clock_state='unknown',state='failed',failure_counts={'admission_refused':1})
        summary=await self.writer.write_metadata('child-summary',summary_value)
        await self.writer.close(cleanup_finished=True)
        self.assertEqual(read_ref(self.path,summary),summary_value)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode),0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in self.path.rglob('*.json')))
        with self.assertRaises(FileExistsError):await RecordingRuntimeArtifacts.open(self.path,execution(),budget=self.budget,retain_evaluation_artifacts=True)

    async def test_capture_inventory_can_follow_seal_but_must_bind_sealed_deliveries(self):
        refs=await self.writer.seal_inventory()
        source=dict(path='source/terminal.json',file_sha256=H,bytes=1)
        value=dict(schema_version=1,kind='recording_capture_inventory',execution_sha256=digest(execution()),
            source_manifest_sha256=H,mapping_sha256=H,source_mappings=dict(path='source-mappings.json',file_sha256=H,bytes=1),
            source_terminal=source,source_events=[],source_counters=source,delivery_outcomes=refs['ticks'])
        with self.assertRaises(ValueError):await self.writer.write_metadata('capture-inventory',value)
        value['delivery_outcomes']=refs['deliveries']
        ref=await self.writer.write_metadata('capture-inventory',value)
        self.assertEqual(read_ref(self.path,ref),value)
        await self.writer.close(cleanup_finished=True)

    async def test_actual_sdk_accepted_record_snapshot_mutation_alias_and_external_retention(self):
        human=Gamst(consent=True);self.addAsyncCleanup(human.close)
        human.register_capability(CapabilitySpec('source.fill','source.fill','video','test.source','1',
            (OutputField('fraction','number'),),'frame'))
        record=await human.push_observation('source.fill',{'fraction':.4},sensor=SensorMetadata('camera','video',1,stream_epoch='original'))
        original=copy.deepcopy(record)
        handle=self.writer.record_publication(record,binding_index=None,builder_sha256=H,
            operation_reference=None,capture_refs=[],role='raw_capture_descriptor',origin_result_sha256=None,mapping_sha256=H)
        record['observation']['value']['fraction']=.9
        ref=await self.writer.publication_ready(handle)
        row=read_ref(self.path,ref);stored=read_ref(self.path,row['record'],maximum=128*1024)
        self.assertEqual(stored,original);self.assertEqual(handle.record_id,record['id'])
        with self.assertRaises(ValueError):self.writer.record_publication(original,binding_index=None,builder_sha256=H,
            operation_reference=None,capture_refs=[],role='task_snapshot',origin_result_sha256=None,mapping_sha256=H)
        await human.delete_data()
        # External retained artifact remains explicitly separate from SDK deletion.
        self.assertEqual(read_ref(self.path,row['record'],maximum=128*1024),original)
        refs=await self.writer.close(cleanup_finished=True)
        self.assertEqual(list(iter_outcome_rows(self.path,refs['publications'],execution_sha256=digest(execution()),kind='publications')),[ref])
        with self.assertRaises(ValueError):await RecordingRuntimeArtifacts.open(self.path.parent/'denied',execution(),budget=self.budget)

    async def test_sync_queue_capacity_includes_inflight_and_preserves_durable_prefix(self):
        entered=threading.Event();release=threading.Event();real=self.writer._store
        def held(path,data):
            if path.startswith('tasks/'):
                entered.set();release.wait(3)
            return real(path,data)
        self.writer._store=held
        writing=asyncio.create_task(self.writer.write_task(0,'context',{'context':1}))
        await asyncio.to_thread(entered.wait,1)
        for n in range(63):self.writer.record_tick(tick(n))
        with self.assertRaises(RuntimeError):self.writer.record_tick(tick(63))
        self.assertEqual(self.writer.status()['queued_frames'],64)
        release.set();await writing
        with self.assertRaises(RuntimeError):await self.writer.close(cleanup_finished=True)
        self.assertTrue((self.path/'tasks/task-0000-context.json').exists())
        self.assertEqual(self.writer._directory,None)

    async def test_repeated_cancellation_joins_owned_write_and_no_success_on_failure(self):
        entered=threading.Event();release=threading.Event();real=self.writer._store
        def held(path,data):
            if path.startswith('ticks/page'):
                entered.set();release.wait(3)
            return real(path,data)
        self.writer._store=held;self.writer.record_tick(tick())
        close=asyncio.create_task(self.writer.close(cleanup_finished=True))
        await asyncio.to_thread(entered.wait,1)
        close.cancel();await asyncio.sleep(.01);close.cancel();release.set()
        with self.assertRaises(asyncio.CancelledError):await close
        self.assertIsNone(self.writer._directory);self.assertFalse(self.writer._threads)
        self.assertTrue((self.path/'ticks/page-000000.json').exists())

    async def test_writer_error_retains_prior_files_and_no_replacement(self):
        prior=await self.writer.write_task(0,'context',{'original':True})
        real=self.writer._store
        def fail(path,data):
            if path.startswith('ticks/page'):raise OSError('fixture disk error')
            return real(path,data)
        self.writer._store=fail;self.writer.record_tick(tick())
        with self.assertRaises(RuntimeError):await self.writer.close(cleanup_finished=True)
        self.assertEqual(read_ref(self.path,prior),{'original':True})
        self.assertEqual(self.writer.status()['failure'],'owned_write_failed')
        index=json.loads((self.path/'ticks-index.json').read_bytes())
        self.assertEqual(index['state'],'incomplete');self.assertEqual(index['row_count'],0)

    async def test_strict_reader_rejects_rehash_alias_fifo_symlink_and_noncanonical(self):
        self.writer.record_tick(tick());refs=await self.writer.close(cleanup_finished=True)
        ref=refs['ticks'];original=(self.path/ref['path']).read_bytes()
        changed=json.loads(original);changed['row_count']=2
        data=_line(changed,65536);(self.path/ref['path']).write_bytes(data)
        forged=dict(ref,file_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        with self.assertRaises(ValueError):list(iter_outcome_rows(self.path,forged,execution_sha256=digest(execution()),kind='ticks'))
        fifo=self.path/'handshake.json';os.mkfifo(fifo)
        with self.assertRaises(ValueError):read_ref(self.path,dict(path='handshake.json',file_sha256=H,bytes=1))
        link=self.path/'mapping.json';link.symlink_to('ticks-index.json')
        with self.assertRaises(OSError):read_ref(self.path,dict(path='mapping.json',file_sha256=forged['file_sha256'],bytes=len(data)))

    async def test_strict_rows_preflight_unknown_cost_and_nonfinite(self):
        counts=dict.fromkeys(KINDS,0);counts['ticks']=3600
        self.assertLess(preflight_runtime_artifacts(counts)['maximum_bytes'],16*1024**2)
        counts['deliveries']=262144
        with self.assertRaises(ValueError):preflight_runtime_artifacts(counts)
        for mutate in (lambda r:r.update(extra=1),lambda r:r.update(count=True),lambda r:r.update(observed_end_ns=-1),lambda r:r.update(period_ns=float('nan'))):
            row=tick();mutate(row)
            with self.assertRaises(ValueError):self.writer.record_tick(row)
        value=dict(schema_version=1,kind='recording_task_result',execution_sha256=digest(execution()),
            task_id='task',input_sha256=H,scope_index=1,status='failed',reason='timeout',evidence=None,
            answer_text='late answer',answer_sha256=H,operation_instance=H,child_finished_ns=1,cleanup_finished=True)
        with self.assertRaises(ValueError):validate_task_artifact('result',value,digest(execution()))

    async def test_actual_native_operation_binds_accepted_sdk_record_and_foreign_execution_rejects(self):
        from recording_native_fixtures import declarations,rehash
        from benchmarks.recording_native_schema import run_context
        from benchmarks.recording_native_writer import RecordingNativeWriter
        from benchmarks.recording_evaluation import RecordingEvaluationSession
        from benchmarks.recording_native_validation import validate_recording_native_inventory
        from benchmarks.replay_clock import ReplayClock
        root=Path(self.tmp.name);d=declarations(root,role='signal')
        for arm in d['collection']['arms']:arm['provider_bindings'][0].update(operation_kinds=['process'],semantic_lane=None)
        rehash(d);context=run_context(d,'b'*64,0)
        native=await RecordingNativeWriter.open(root/'native',context)
        observer=RecordingEvaluationSession.from_frozen(d,native,run_nonce='b'*64,execution_ordinal=0,retain_evaluation_hashes=True)
        self.addAsyncCleanup(observer.close)
        human=Gamst(consent=True);self.addAsyncCleanup(human.close)
        human.register_capability(CapabilitySpec('source.level','source.level','audio','fixture','1',(OutputField('level','number'),),'frame'))
        async def callback():return {'level':.5}
        observer.bind_backend(0,callback,'callback');clock=ReplayClock(d['source_manifest']);observer.bind_mapping(clock.mapping,clock.sha256)
        capture=dict(kind='capture',source_index=0,capture_ordinal=0,record_sha256=digest(d['source_manifest']['records'][0]),mapping_sha256=clock.sha256)
        ticket=observer.reserve_operation(0,observer.stream_scope(0),'process',[capture],digest(dict(binding=0,operation_kind='process',capture_refs=[capture])),capture['record_sha256'])
        value=await ticket.invoke(callback)
        record=await human.push_observation('source.level',value,sensor=SensorMetadata('mic','audio',1,stream_epoch='original'))
        ticket.published(True);actual=ticket.entry_reference()
        params=dict(binding_index=0,builder_sha256=None,operation_reference=actual,capture_refs=[capture],role='specialist_result',origin_result_sha256=digest(value),mapping_sha256=clock.sha256)
        with self.assertRaises(ValueError):self.writer.record_publication(record,**params)
        ex=execution()
        for key in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal'):ex[key]=context[key]
        artifacts=await RecordingRuntimeArtifacts.open(root/'bound-runtime',ex,budget=self.budget,retain_evaluation_artifacts=True)
        self.addAsyncCleanup(artifacts.abort)
        handle=artifacts.record_publication(record,**params);ref=await artifacts.publication_ready(handle)
        row,original=read_publication(artifacts.output,ref,execution_sha256=digest(ex))
        self.assertEqual(row['operation_instance'],actual.instance_sha256);self.assertEqual(original,record)
        await artifacts.close(cleanup_finished=True)
        observer.backend_closed(0,callback);await observer.close()
        index=validate_recording_native_inventory(d,native.output,source_inventory=d['source_manifest'])
        self.assertEqual(index.terminal['entry_count'],1)

class ParentImport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import sys
        from recording_native_fixtures import declarations
        from benchmarks.recording_native_channel import ParentNativeChannel
        from benchmarks.recording_native_schema import run_context,encoded
        self.tmp=tempfile.TemporaryDirectory(dir='.');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();d=declarations(self.root);paths={}
        for k,v in d.items():
            p=self.root/(k+'.json');p.write_bytes(encoded(v)+b'\n');paths[k]=str(p)
        context=run_context(d,'a'*64,0)
        native=ParentNativeChannel(context,d,paths=paths,output=self.root/'native',deadline_ns=time.monotonic_ns()+5_000_000_000)
        self.addAsyncCleanup(native.abort)
        child=await asyncio.create_subprocess_exec(sys.executable,'-c','pass',**native.prepare(dict(os.environ)))
        native.spawned(child);await child.wait();await native.record_reaped(child)
        self.inventory=native.snapshot();self.execution=execution()
        for k in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256','invocation_ordinal'):self.execution[k]=context[k]
        self.parent=await RecordingRuntimeArtifacts.open(self.root/'parent',self.execution,budget=ArtifactBudget(),retain_evaluation_artifacts=True)
        self.addAsyncCleanup(self.parent.abort)
        self.child=await RecordingRuntimeArtifacts.open(self.root/'child',self.execution,budget=ArtifactBudget(),retain_evaluation_artifacts=True)
        self.addAsyncCleanup(self.child.abort)
        self.child.record_tick(tick());self.refs=await self.child.close(cleanup_finished=True)

    async def test_original_copy_has_identical_hashes_and_parent_alone_writes_inventory(self):
        copied=await self.parent.ingest_child(self.root/'child',reaped_inventory=self.inventory,expected_refs=[self.refs['ticks']])
        self.assertEqual(copied['ticks-index.json'],self.refs['ticks'])
        self.assertEqual((self.root/'child/ticks/page-000000.json').read_bytes(),(self.root/'parent/ticks/page-000000.json').read_bytes())
        ref=await self.parent.write_native_inventory(self.inventory)
        self.assertTrue(read_ref(self.root/'parent',ref)['parent_reaped'])
        self.assertNotIn('native-inventory.json',copied)
        sealed=await self.parent.close(cleanup_finished=True)
        self.assertEqual(list(iter_outcome_rows(self.root/'parent',sealed['ticks'],execution_sha256=digest(self.execution),kind='ticks')),[tick()])
        with self.assertRaises(TypeError):await self.parent.write_native_inventory(dict(self.inventory))

    async def test_changed_accepted_file_rejected_before_copy(self):
        path=self.root/'child/ticks-index.json';path.write_bytes(path.read_bytes().replace(b'complete',b'incomplete'))
        with self.assertRaises(ValueError):await self.parent.ingest_child(self.root/'child',reaped_inventory=self.inventory,expected_refs=[self.refs['ticks']])
        self.assertFalse((self.root/'parent/ticks-index.json').exists())

    async def test_child_parent_namespace_and_unapproved_financial_tree_reject(self):
        (self.root/'child/native-inventory.json').write_bytes(b'{}\n')
        with self.assertRaises(ValueError):await self.parent.ingest_child(self.root/'child',reaped_inventory=self.inventory)
        # Separate child tree tests no accidental financial authority in U1.
        other=self.root/'other';other.mkdir();(other/'financial').mkdir()
        with self.assertRaises(ValueError):await self.parent.ingest_child(other,reaped_inventory=self.inventory)
