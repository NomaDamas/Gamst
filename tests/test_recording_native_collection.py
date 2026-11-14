"""Original ABC ownership fixtures, never real independence or numeric proof."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_schema import run_context
from benchmarks.recording_native_writer import RecordingNativeWriter, ArtifactBudget
from benchmarks.recording_native_validation import validate_recording_native_collection
from benchmarks.replay_clock import ReplayClock
from recording_native_fixtures import declarations


class CollectionInventory(unittest.IsolatedAsyncioTestCase):
    async def test_all_original_runs_once_and_no_missing_zero_cost_run(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory);d=declarations(root);clock=ReplayClock(d['source_manifest'])
            budget=ArtifactBudget();runs=[]
            for ordinal in range(3):
                nonce=('%064x'%(ordinal+1))
                writer=await RecordingNativeWriter.open(root/('run-'+str(ordinal)),run_context(d,nonce,ordinal),budget=budget)
                owner=RecordingEvaluationSession.from_frozen(d,writer,run_nonce=nonce,execution_ordinal=ordinal,retain_evaluation_hashes=True)
                owner.bind_mapping(clock.mapping,clock.sha256)
                await owner.close()
                runs.append(dict(declarations=d,directory=writer.output,
                    terminal_sha256=hashlib.sha256((writer.output/'terminal.json').read_bytes()).hexdigest(),
                    source_inventory=d['source_manifest']))
            report=validate_recording_native_collection(runs)
            self.assertEqual(report['entry_count'],0)
            self.assertEqual(len(report['runs']),3)
            self.assertEqual(report['actual_acceptance'],'incomplete')
            with self.assertRaises(ValueError):validate_recording_native_collection(runs[:2])
            with self.assertRaises(ValueError):validate_recording_native_collection([runs[0],runs[0],runs[2]])
            altered=[dict(v) for v in runs];altered[0]['terminal_sha256']=None
            with self.assertRaises(ValueError):validate_recording_native_collection(altered)
