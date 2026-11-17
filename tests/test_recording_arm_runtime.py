"""Actual persistent source/SDK/fixture-completion composition, no model claims."""
import asyncio
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
from recording_abc_fixtures import build_runtime_fixture,installed_dependencies
from benchmarks.recording_runtime_inputs import OwnedRuntimeInputs
from benchmarks.recording_runtime_artifacts import RecordingRuntimeArtifacts,iter_outcome_rows
from benchmarks.recording_native_writer import RecordingNativeWriter,ArtifactBudget
from benchmarks.recording_native_schema import run_context
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_arm_runtime import RecordingArmRuntime


class Arms(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):cls.dependencies=installed_dependencies()

    async def test_actual_startup_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp).resolve();(root/'inputs').mkdir()
            # Mapping metadata is deliberately delayed below.  Keep a full
            # second between the frozen snapshot and its deadline so the test
            # exercises startup/cleanup ownership without making success
            # depend on hosted-run scheduler jitter.  The runtime still
            # enforces the declared deadline; this only gives the fixture a
            # bounded admission margin after the clock anchor is established.
            f=build_runtime_fixture(root/'inputs',count=20,duration_ns=2_000_000_000,task_snapshot_ns=1_000_000_000,dependency_evidence=self.dependencies)
            for invocation in (0,1,2,4):
                i=await OwnedRuntimeInputs.open(str(f.root),expected_execution_sha256=f.execution_sha256s[invocation],invocation_ordinal=invocation)
                d=i.receipt.declarations();a=await RecordingRuntimeArtifacts.open(root/f'arm-{invocation}',f.documents[f'invocations/{invocation:02}/execution.json'],budget=ArtifactBudget(),retain_evaluation_artifacts=True)
                w=await RecordingNativeWriter.open(a.output/'native',run_context(d,'a'*64,invocation))
                o=RecordingEvaluationSession.from_runtime_frozen(d,w,latency_policy=f.documents['runtime/latency.json'],source_bundles=f.source_bundles,run_nonce='a'*64,execution_ordinal=invocation,retain_evaluation_hashes=True)
                r=None
                try:
                    r=await RecordingArmRuntime.open(i,owner=o,artifacts=a)
                    self.assertEqual(len(r.lanes),1 if r.execution['arm']=='B' else 2)
                    self.assertIsNone(r.clock)
                    received=[];captured_while_mapping_waited=[]
                    async def playback(*args):pass
                    async def answer(*args):received.append(args)
                    original_write=a.write_metadata
                    async def delayed_metadata(kind,value):
                        if invocation==0 and kind=='mapping':
                            await asyncio.sleep(.25)
                            captured_while_mapping_waited.extend(r.source.counters())
                        if invocation==4 and kind=='mapping':
                            async def actual_publication():
                                while not any(lane['processed'] for _,_,lane in r._delivery_lanes):await asyncio.sleep(.001)
                            await asyncio.wait_for(actual_publication(),2)
                            raise OSError('owned metadata write failed after actual provider work')
                        return await original_write(kind,value)
                    try:
                        with patch.object(a,'write_metadata',delayed_metadata):await r.start(playback_started=playback,answer_ready=answer)
                    except Exception as error:
                        if invocation!=4:self.fail(f"{error}: failures={r.failures}, perception={r.perception.failures}, results={r.task_runtime.result_values}")
                        self.assertIsInstance(error,OSError)
                        self.assertGreaterEqual(r.source.counters()[0]['captured'],1)
                        self.assertTrue(any(lane['processed'] for _,_,lane in r._delivery_lanes))
                        self.assertTrue(r.human.closed);self.assertTrue(all(job.done() for job in r.jobs))
                        self.assertTrue(r.failures);self.assertFalse(received)
                        self.assertGreater(r.journal.status()['valid_prefix_bytes'],0)
                        continue
                    self.assertNotEqual(invocation,4)
                    self.assertEqual(len(received),1)
                    self.assertEqual(r.task_runtime.result_values[0]['status'],'answer')
                    self.assertEqual(r.source.counters()[0]['attempted'],3)
                    self.assertEqual(r.source.counters()[0]['captured']+r.source.counters()[0]['dropped'],3)
                    if invocation==0:self.assertGreaterEqual(captured_while_mapping_waited[0]['captured'],1)
                    self.assertFalse(r.failures)
                finally:
                    if r:await r.close()
                    await o.close();await a.abort();await i.close()
                self.assertTrue(r.human.closed)
                self.assertTrue(all(lane['adapter']._closed for lane in r.lanes.values()))


if __name__=='__main__':unittest.main()
