"""Actual bytes and installed identity admission for persistent ABC fixtures."""
import asyncio
import hashlib
import importlib.util
import os
from itertools import permutations
from pathlib import Path
import tempfile
import unittest

from benchmarks.recording_native_schema import digest
from benchmarks.recording_runtime_inputs import OwnedRuntimeInputs
from benchmarks.recording_runtime_native import preflight_runtime_collection
from benchmarks.recorded_profiles import FixtureBGR
from benchmarks.owned_capture import OwnedCaptureSource, CaptureCapacity
from benchmarks.replay_clock import ReplayClock
import test_owned_capture as capture_fixtures
from recording_abc_fixtures import build_runtime_fixture, installed_dependencies


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('numpy','cv2')),
                     'existing optional visual dependencies required; never installed by tests')
class ActualInputs(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = installed_dependencies()

    def build(self, **kwargs):
        temporary = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(temporary.cleanup)
        return build_runtime_fixture(Path(temporary.name), dependency_evidence=self.dependencies, **kwargs)

    async def test_twenty_four_recordings_actual_inputs_and_joint_capacity(self):
        fixture = self.build()
        d=fixture.documents; c=d['runtime/collection.json']; u=d['runtime/policy.json']
        self.assertEqual(len(c['recordings']),24)
        self.assertEqual(len(c['invocations']),72)
        instrumentation=d['runtime/instrumentation.json']
        self.assertEqual(instrumentation['hello_timeout_ms'],10_000)
        for inv in c['invocations']:
            execution=d[f"invocations/{inv['ordinal']:02}/execution.json"]
            self.assertEqual(execution['instrumentation_protocol_sha256'],digest(instrumentation))
            self.assertEqual(fixture.execution_sha256s[inv['ordinal']],digest(execution))
        self.assertEqual(d['runtime/plan.json']['comparison']['min_pairs'],20)
        self.assertFalse(d['runtime/plan.json']['comparison']['held_out'])
        self.assertEqual(d['runtime/plan.json']['comparison']['evidence'],'synthetic')
        orders=[tuple(row['arm'] for row in c['invocations'] if row['recording_index']==n) for n in range(24)]
        for order in permutations('ABC'):self.assertEqual(orders.count(order),4)
        self.assertEqual(len({r['source_manifest_sha256'] for r in c['recordings']}),24)
        for n,bundle in enumerate(fixture.source_bundles.values()):
            for record in bundle['manifest']['records']:
                raw=(fixture.root/f'recordings/{n:02}/payload'/record['payload_path']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),record['payload_sha256'])
                self.assertEqual(len(raw),record['payload_bytes'])
        from benchmarks import recording_transports
        self.assertEqual(u['adapters'][0]['implementation_sha256'],hashlib.sha256(Path(recording_transports.__file__).read_bytes()).hexdigest())
        self.assertEqual(u['adapters'][0]['dependency_inventory_sha256'],fixture.dependency_inventory.sha256)
        capability=FixtureBGR.capability(0)
        for arm in c['arms'][1:]:
            provider=next(b for b in arm['provider_bindings'] if b['binding_id']=='vision')
            self.assertEqual((provider['provider'],provider['model_revision']),(capability.provider,capability.version))
        async with await OwnedRuntimeInputs.open(str(fixture.root), expected_execution_sha256=fixture.execution_sha256s[0], invocation_ordinal=0) as owned:
            self.assertEqual(len(owned.receipt.source_bundles()),24)
            self.assertLess(owned.workspace_bytes,256*1024**2)
            manifest=fixture.source_bundles['original-0']['manifest']
            fd=owned.duplicate_payload_root('original-0')
            try:
                source=await OwnedCaptureSource.preflight(manifest,fd,
                    CaptureCapacity(4*32*32*3,1_000_000_000),journal=capture_fixtures.Journal())
            finally:
                os.close(fd)
            sink=capture_fixtures.Sink()
            try:
                await source.start(ReplayClock(manifest),sink)
                self.assertEqual(len(sink.packets),3)
                self.assertEqual(source.counters()[0]['uninspected_suffix'],0)
                for packet in sink.packets:
                    self.assertEqual(packet.payload_sha256,hashlib.sha256(packet.payload).hexdigest())
                    self.assertGreaterEqual(packet.acquired_monotonic_ns,packet.scheduled_monotonic_ns)
            finally:
                await source.close()
        result=preflight_runtime_collection([fixture.declarations(n) for n in range(24)],
            latency_policy=d['runtime/latency.json'],source_bundles=fixture.source_bundles)
        self.assertTrue(result['sample_gate_eligible'])
        self.assertLess(result['predicted_artifact_bytes'],4*1024**3)
        self.assertEqual(result['original_task_count'],24)
        self.assertEqual(result['actual_acceptance'],'incomplete')

    async def test_overlapping_tasks_phase_zero_and_no_wildcard_responses(self):
        fixture=self.build(count=20,tasks_per_recording=3,duration_ns=300_000_000,phase_ns=0)
        d=fixture.documents;c=d['runtime/collection.json']
        tasks=next(iter(fixture.source_bundles.values()))['tasks']
        self.assertEqual(len(tasks),3)
        self.assertEqual(len({t['snapshot_ns'] for t in tasks}),1)
        self.assertTrue(all(t['opens_ns']<t['snapshot_ns']<t['deadline_ns'] for t in tasks))
        rows=d['responses/000.json']['rows']; keys={(r['invocation_ordinal'],r['lane_id'],r['call_ordinal']) for r in rows}
        self.assertEqual(len(rows),len(keys))
        for inv in c['invocations']:
            for call in range(3):self.assertIn((inv['ordinal'],'answer',call),keys)
            self.assertEqual((inv['ordinal'],'periodic',0) in keys, inv['arm']!='B')
        self.assertTrue(all(r['synthetic_receipt_sha256'] is None for r in rows))
        # Select a non-zero invocation to catch accidental ordinal0-only hashing.
        ordinal=7
        async with await OwnedRuntimeInputs.open(str(fixture.root), expected_execution_sha256=fixture.execution_sha256s[ordinal], invocation_ordinal=ordinal) as owned:
            self.assertEqual(owned.receipt.identities['execution_sha256'],fixture.execution_sha256s[ordinal])
        result=preflight_runtime_collection([fixture.declarations(n) for n in range(20)],
            latency_policy=d['runtime/latency.json'],source_bundles=fixture.source_bundles)
        self.assertEqual(result['original_task_count'],60)
        self.assertLess(result['predicted_artifact_bytes'],4*1024**3)

    def test_never_overwrite_existing_input_or_shrink_sample_floor(self):
        temporary=tempfile.TemporaryDirectory(dir='.');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);marker=root/'keep';marker.write_bytes(b'original')
        with self.assertRaises(ValueError):build_runtime_fixture(root,dependency_evidence=self.dependencies)
        self.assertEqual(marker.read_bytes(),b'original')
        with self.assertRaises(ValueError):build_runtime_fixture(root,count=1,dependency_evidence=self.dependencies)

    async def test_actual_pcm_context_continues_while_vision_is_pending(self):
        from gamst import Gamst,SensorStreams
        from benchmarks.recording_capture_router import RecordingCaptureRouter
        from benchmarks.recording_runtime_profiles import builtin_profile
        from benchmarks.recording_evaluation import RecordingEvaluationSession
        from benchmarks.recording_native_writer import RecordingNativeWriter
        from benchmarks.recording_native_schema import run_context
        fixture=self.build(count=20,duration_ns=300_000_000,phase_ns=0,
                           include_audio=True,vision_delay_ms=500)
        d=fixture.declarations();u=d['runtime_policy'];arm=u['arms'][1]
        expected=preflight_runtime_collection([fixture.declarations(n) for n in range(20)],
            latency_policy=fixture.documents['runtime/latency.json'],source_bundles=fixture.source_bundles)
        self.assertLess(expected['predicted_artifact_bytes'],4*1024**3)
        output=tempfile.TemporaryDirectory(dir='.');self.addCleanup(output.cleanup)
        context=run_context(d,'a'*64,1)
        writer=await RecordingNativeWriter.open(Path(output.name).resolve()/'native',context)
        owner=RecordingEvaluationSession.from_runtime_frozen(d,writer,
            latency_policy=fixture.documents['runtime/latency.json'],source_bundles=fixture.source_bundles,
            run_nonce='a'*64,execution_ordinal=1,retain_evaluation_hashes=True)
        human=Gamst(consent=True);streams=SensorStreams(human);source=None;router=None
        try:
            bindings=d['collection']['arms'][1]['provider_bindings']
            for delivery in arm['recording_policies'][0]['deliveries']:
                config={'delay_ms':500} if delivery['source_id']=='camera' else {}
                spec,factory,config=builtin_profile(delivery,config)
                streams.register(spec,factory)
                bi=next(i for i,b in enumerate(bindings) if b['binding_id']==delivery['binding_id'])
                sensor=next(s for s in d['source_manifest']['sources'] if s['source_id']==delivery['source_id'])
                await streams.open(sensor['source_id'],sensor['sensor_id'],[spec.id],
                    evaluation_bindings={spec.id:owner.stream_scope(bi)},evaluation_configurations={spec.id:config})
            router=RecordingCaptureRouter(d['source_manifest'],streams=streams,
                source_bindings=u['recordings'][0]['source_bindings'],subject_id='local-user',
                delivery_sources=['camera','mic'])
            async with await OwnedRuntimeInputs.open(str(fixture.root),
                    expected_execution_sha256=fixture.execution_sha256s[1],invocation_ordinal=1) as owned:
                fd=owned.duplicate_payload_root('original-0')
                try:source=await OwnedCaptureSource.preflight(d['source_manifest'],fd,
                    CaptureCapacity(4*(32*32*3+1600),1_000_000_000),journal=capture_fixtures.Journal())
                finally:os.close(fd)
                clock=ReplayClock(d['source_manifest']);owner.bind_mapping(clock.mapping,clock.sha256)
                await source.start(clock,router)
                current=await human.get_context()
                names={o['capability']['id'] for o in current['observations'].values()}
                self.assertIn('audio.replay.level',names)
                self.assertNotIn('vision.replay.fixture',names)
                counters={r['source_id']:r for r in source.counters()}
                self.assertEqual(counters['mic']['captured'],6)
                self.assertEqual(counters['camera']['captured'],3)
                self.assertTrue(all(r['uninspected_suffix']==0 for r in counters.values()))
                await router.close();await streams.drain()
                current=await human.get_context()
                self.assertIn('vision.replay.fixture',{o['capability']['id'] for o in current['observations'].values()})
        finally:
            if source is not None:await source.close()
            if router is not None:await router.close()
            await human.close();await owner.close()
