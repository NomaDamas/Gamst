"""Original source/native/parent ownership joins; no actual model claims."""
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recording_abc_fixtures import build_runtime_fixture, installed_dependencies
from benchmarks.recording_coordinator import _preflight, _run_one
from benchmarks.recording_runtime_inputs import OwnedRuntimeInputs
from benchmarks.recording_runtime_validation import (ValidatedRecordingRun, validate_recording_run,
    preflight_recording_run_validation)


class JoinAdmission(unittest.TestCase):
    def test_joint_peak_and_retained_bounds_without_activation(self):
        execution=dict(schema_version=1,kind='recording_execution',runtime_policy_sha256='a'*64,
            collection_sha256='b'*64,scope_protocol_sha256='c'*64,instrumentation_protocol_sha256='d'*64,
            invocation_ordinal=0,recording_index=0,arm='A',stream_id='stream-A',execution_namespace='run-A',
            cache_namespace='cache-A',asset_inventory_sha256='e'*64)
        declaration=dict(collection=dict(capacity=dict(max_entries_per_run=16,max_events_per_run=128,
            max_chunks_per_run=8,max_reservations_per_run=32),arms=[dict(arm='A',provider_bindings=[{}])]),
            source_manifest=dict(records=[{},{}]),tasks=[{}])
        bounds=preflight_recording_run_validation(declaration,execution,publication_rows=8)
        self.assertGreater(bounds['peak_workspace_bytes'],bounds['retained_workspace_bytes'])
        larger=preflight_recording_run_validation(declaration,execution,publication_rows=9)
        self.assertGreater(larger['peak_workspace_bytes'],bounds['peak_workspace_bytes'])
        with self.assertRaises(ValueError):
            preflight_recording_run_validation(declaration,execution,publication_rows=262144)


class OriginalRunJoin(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = installed_dependencies()

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        for name in ('inputs', 'stage', 'runs'):
            (self.root / name).mkdir()
        self.fixture = build_runtime_fixture(self.root / 'inputs', count=20,
            duration_ns=2_000_000_000, dependency_evidence=self.dependencies)
        self.inputs = await OwnedRuntimeInputs.open(str(self.fixture.root),
            expected_execution_sha256=self.fixture.execution_sha256s[0], invocation_ordinal=0)
        self.addAsyncCleanup(self.inputs.close)
        self.docs, self.bundles, self.declarations, _, self.budget = await _preflight(self.inputs)

    async def run_one(self, *, spawn_failure=False):
        args = (str(self.fixture.root), self.root / 'stage/00', self.root / 'runs/00',
            self.docs, self.bundles, self.declarations, self.budget,
            self.docs['runtime/collection.json']['invocations'][0])
        if spawn_failure:
            with patch('asyncio.create_subprocess_exec', side_effect=OSError('fixture spawn refusal')):
                self.result = await _run_one(*args,retain_financial_documents=False)
        else:
            self.child_output=[]
            async def capture(stream,maximum):
                data=bytearray()
                while True:
                    block=await stream.read(65536)
                    if not block:break
                    if len(data)<maximum:data.extend(block[:maximum-len(data)])
                self.child_output.append(bytes(data).decode('utf-8','replace'))
            with patch('benchmarks.recording_coordinator._drain',capture):
                self.result = await _run_one(*args,retain_financial_documents=False)
        return self.result

    def validate(self, *, inventory=None, documents=None):
        _, _, token, activation = self.result
        return validate_recording_run(self.root / 'runs/00', documents=self.docs if documents is None else documents,
            source_bundles=self.bundles, invocation_ordinal=0,
            parent_native_inventory=token if inventory is None else inventory,
            activation_started_ns=activation)

    async def test_failed_spawn_keeps_original_arm_unknown_prefix_and_receipt_floor(self):
        await self.run_one(spawn_failure=True)
        joined = self.validate()
        self.assertIsNone(joined.native_index)
        self.assertIsNone(joined.financial_values['native_terminal_sha256'])
        self.assertEqual(len(joined.task_receipts), len(self.declarations[0]['tasks']))
        for key in ('source_comparable', 'operational_closed', 'external_work_closed'):
            self.assertFalse(joined.financial_values[key])
        self.assertIn('missing_native_prefix', joined.financial_values['violations'])
        self.assertEqual(joined.controller_settlements, [])
        copy = joined.execution
        copy['arm'] = 'B'
        self.assertEqual(joined.execution['arm'], 'A')
        with self.assertRaises(TypeError):
            self.validate(inventory=dict(self.result[2]))
        with self.assertRaises(TypeError):
            ValidatedRecordingRun({}, {}, [], {}, {}, [], None)
        changed = deepcopy(self.docs)
        changed['invocations/00/execution.json']['execution_namespace'] = 'different'
        with self.assertRaises(ValueError):
            self.validate(documents=changed)

    async def test_mapping_startup_failures_retain_failed_prefix_and_next_arm(self):
        # Test-only child wrapper: the production command and runtime have no
        # injectable implementation or environment-controlled failure hook.
        script = '''
import asyncio, sys
from benchmarks.recording_arm_runtime import RecordingArmRuntime
from benchmarks.recording_runtime_artifacts import RecordingRuntimeArtifacts
from benchmarks.recording_artifacts import RecordingArtifacts
from benchmarks.recording_control_channel import ChildCoordinatorChannel
from gamst import Gamst
phase=sys.argv.pop(1)
def fail(): raise OSError('synthetic startup boundary '+phase)
original_metadata=RecordingRuntimeArtifacts.write_metadata
async def metadata(self, kind, value):
    if phase=='metadata-'+kind:
        if kind=='mapping': await asyncio.sleep(.1)
        fail()
    return await original_metadata(self, kind, value)
RecordingRuntimeArtifacts.write_metadata=metadata
original_json=RecordingArtifacts.write_json
async def write_json(self, name, value, **kw):
    if phase=='journal-'+name: fail()
    return await original_json(self, name, value, **kw)
RecordingArtifacts.write_json=write_json
original_emit=RecordingArtifacts.emit
def emit(self, name, value):
    if phase=='mapping-event' and name=='mapping_published': fail()
    return original_emit(self, name, value)
RecordingArtifacts.emit=emit
original_push=Gamst.push_observation
async def push(self, capability_id, *args, **kw):
    if phase=='publication' and capability_id=='replay.mapping.v1': fail()
    return await original_push(self, capability_id, *args, **kw)
Gamst.push_observation=push
original_send=ChildCoordinatorChannel.send_wait
async def send(self, kind, fields):
    if phase=='playback' and kind=='playback_started': fail()
    return await original_send(self, kind, fields)
ChildCoordinatorChannel.send_wait=send
from benchmarks.recording_arm import main
raise SystemExit(main())
'''
        real_spawn=asyncio.create_subprocess_exec
        phases=('metadata-mapping','metadata-source-mappings','journal-mapping.json',
            'journal-source-mappings.json','journal-mapping-records.json','publication','playback','mapping-event')
        from benchmarks.recording_runtime_artifacts import read_ref
        for ordinal,phase in enumerate((*phases,None)):
            async def spawn(*args, **kw):
                return await real_spawn(*(args if phase is None else (args[0],'-c',script,phase,*args[3:])), **kw)
            with self.subTest(phase=phase),patch('asyncio.create_subprocess_exec',spawn):
                result=await _run_one(str(self.fixture.root),self.root/f'stage/{ordinal:02}',self.root/f'runs/{ordinal:02}',
                    self.docs,self.bundles,self.declarations,self.budget,
                    self.docs['runtime/collection.json']['invocations'][ordinal],retain_financial_documents=False)
                _,run,token,activation=result
                directory=self.root/f'runs/{ordinal:02}'
                joined=validate_recording_run(directory,documents=self.docs,source_bundles=self.bundles,
                    invocation_ordinal=ordinal,parent_native_inventory=token,activation_started_ns=activation)
                self.assertIsNotNone(run['process_reaped_ns'])
                self.assertIsNotNone(run['child_summary'])
                if phase is None:
                    self.assertEqual(run['state'],'closed')
                    self.assertTrue(joined.financial_values['source_comparable'])
                else:
                    self.assertEqual(run['state'],'failed')
                    self.assertEqual(read_ref(directory,run['child_summary'])['state'],'failed')
                    self.assertIsNone(joined.financial_values['playback_anchor_ns'])
                    for key in ('source_comparable','operational_closed','clock_consistent'):
                        self.assertFalse(joined.financial_values[key])
                    for receipt,task in zip(joined.task_receipts,self.declarations[ordinal//3]['tasks']):
                        self.assertEqual(receipt['status'],'not_executed')
                        self.assertEqual(receipt['latency_ns'],task['deadline_ns']-task['opens_ns'])

    async def test_actual_child_source_sdk_native_response_and_parent_receipt_join(self):
        _, run, token, _ = await self.run_one()
        self.assertEqual(run['state'], 'closed', (run,self.child_output))
        joined = self.validate()
        self.assertIsNotNone(joined.native_index)
        self.assertTrue(joined.financial_values['source_comparable'])
        self.assertTrue(joined.financial_values['operational_closed'])
        self.assertEqual(joined.financial_values['native_terminal_sha256'], token['terminal_sha256'])
        self.assertEqual(joined.task_receipts[0]['status'], 'answer')
        self.assertEqual(joined.controller_settlements, [])
        # Selected ticks finish after later busy/late rows. Joining originals
        # must be independent of completion order without accepting omissions.
        from benchmarks import recording_runtime_validation as validation
        original_outcomes=validation._outcomes
        def outcomes(directory,ref,execution,kind,violations):
            rows=list(original_outcomes(directory,ref,execution,kind,violations))
            if kind=='ticks':
                self.assertGreater(len(rows),1)
                rows.reverse()
                if mutation=='duplicate':rows.append(rows[0])
                elif mutation=='missing':rows.pop()
            return iter(rows)
        for mutation in ('reordered','duplicate','missing'):
            with self.subTest(tick_inventory=mutation),patch.object(validation,'_outcomes',outcomes):
                if mutation=='reordered':self.assertTrue(self.validate().financial_values['source_comparable'])
                else:
                    with self.assertRaises(ValueError):self.validate()
        # Answer text cannot stand in for a different actual SDK/native result.
        import json
        path = self.root / 'runs/00/requests/request-000000.json'
        original = path.read_bytes()
        changed = json.loads(original)
        changed['sdk_result']['usage'] = dict(input_tokens=1,output_tokens=1,cached_input_tokens=0)
        from benchmarks.recording_native_schema import encoded
        path.write_bytes(encoded(changed) + b'\n')
        with self.assertRaises(ValueError):
            self.validate()
        path.write_bytes(original)
        source = self.root / 'runs/00/source/mapping-records.json'
        original = source.read_bytes()
        changed = json.loads(original)
        changed[0]['timestamp_ms'] += 1
        source.write_bytes(encoded(changed))
        with self.assertRaises(ValueError):
            self.validate()
        source.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
