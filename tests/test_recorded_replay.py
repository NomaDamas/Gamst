import asyncio
import hashlib
from pathlib import Path
import tempfile
import unittest
import importlib.util
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.recorded_source import digest
from benchmarks.recorded_replay import RecordedSourceReplay, run_source_context_recording


class JournalFixture:
    def __init__(self):self.events=[]
    def emit(self, event, value):self.events.append((event,value))


class ReplaySDKTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self, root):
        payload=b'\x00\x10'*400
        (root/'pcm.bin').write_bytes(payload)
        source=dict(source_id='mic',sensor_id='mic-1',stream_epoch='old-epoch',
            format='pcm_s16le_mono',modality='audio',calibration_id=None)
        records=[dict(source_id='mic',sensor_id='mic-1',stream_epoch='old-epoch',
            sequence=i,capture_ns=1000000000+i*50000000,offset_ns=i*50000000,
            format='pcm_s16le_mono',sample_rate=8000,sample_count=400,
            payload_path='pcm.bin',payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest()) for i in range(5)]
        manifest=dict(schema_version=1,recording_id='old-recording',origin_capture_ns=1000000000,
            duration_ns=400000000,sources=[source],records=records,
            availability=[dict(source_id='mic',intervals=[dict(start_ns=0,end_ns=400000000,state='available')])])
        task=dict(task_id='first',input={'question':'audio level'},input_sha256=digest({'question':'audio level'}),
            opens_ns=0,snapshot_ns=100000000,deadline_ns=300000000,window='1h')
        tasks=[task,dict(task,task_id='second',opens_ns=50000000,snapshot_ns=200000000,deadline_ns=350000000)]
        return manifest,tasks,{'mic':dict(profile='pcm_level',delay_ms=0)}

    async def test_real_sdk_old_capture_mapping_and_overlapping_snapshots(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            journal=JournalFixture()
            replay=RecordedSourceReplay(manifest,tasks,profiles,root,journal,max_lateness_ns=1000000000)
            await replay.run()
            self.assertTrue(replay.human.closed)
            self.assertEqual(replay.counters['mic']['captured'],5)
            self.assertEqual(replay.counters['mic']['attempted'],5)
            self.assertEqual(replay.counters['mic']['dropped'],0)
            self.assertEqual(set(replay.snapshots),{'first','second'})
            for snapshot in replay.snapshots.values():
                observations=list(snapshot['context']['observations'].values())
                level=next(o for o in observations if o['capability']['id']=='audio.replay.level')
                self.assertTrue(level['fresh'])
                self.assertEqual(level['value']['level']['audio.rms'],.125)
                self.assertEqual(snapshot['mapping_sha256'],replay.clock.sha256)
                self.assertEqual(snapshot['context_sha256'],digest(snapshot['context']))
            self.assertEqual(replay.clock.mapping['origin_capture_ns'],1000000000)
            self.assertTrue(all(t.done() for t in replay.jobs))
            self.assertFalse(replay.reads)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional BGR array runtime unavailable')
    async def test_slow_video_keeps_actual_pcm_context_running_and_counts_delivery_loss(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            pixels=bytes([20,40,80])*(16*16); (root/'bgr.bin').write_bytes(pixels)
            source=dict(source_id='camera',sensor_id='cam',stream_epoch='camera-original',
                format='bgr8_hwc',modality='video',calibration_id=None)
            manifest['sources'].append(source)
            manifest['availability'].append(dict(source_id='camera',intervals=[dict(start_ns=0,end_ns=400000000,state='available')]))
            for i in range(5):
                manifest['records'].append(dict(source_id='camera',sensor_id='cam',stream_epoch='camera-original',
                    sequence=i,capture_ns=1000000000+i*20000000,offset_ns=i*20000000,
                    format='bgr8_hwc',shape=[16,16,3],payload_path='bgr.bin',payload_bytes=len(pixels),
                    payload_sha256=hashlib.sha256(pixels).hexdigest()))
            manifest['records'].sort(key=lambda r:(r['offset_ns'],r['source_id'],r['sequence']))
            profiles['camera']=dict(profile='fixture_bgr',delay_ms=300)
            replay=RecordedSourceReplay(manifest,tasks,profiles,root,JournalFixture(),max_lateness_ns=1000000000)
            await replay.run()
            for snapshot in replay.snapshots.values():
                values=list(snapshot['context']['observations'].values())
                self.assertTrue(any(o['capability']['id']=='audio.replay.level' for o in values))
                self.assertFalse(any(o['capability']['id']=='vision.replay.fixture' for o in values))
            self.assertEqual(replay.counters['camera']['captured'],5)
            self.assertEqual(replay.counters['camera']['dropped'],0)
            camera=next(p for p in replay.provider_outcomes if p['source_id']=='camera')
            self.assertGreater(camera['dropped'],0)
            self.assertEqual(camera['accepted'],camera['processed']+camera['dropped'])
            mic=next(p for p in replay.provider_outcomes if p['source_id']=='mic')
            self.assertEqual(mic['processed'],5)

    async def test_changed_source_stops_admission_and_preserves_actual_prefix(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            journal=JournalFixture()
            replay=RecordedSourceReplay(manifest,tasks,profiles,root,journal,max_lateness_ns=1000000000)
            running=asyncio.create_task(replay.run())
            await asyncio.sleep(.08)
            (root/'pcm.bin').write_bytes(b'\x01'*800)
            with self.assertRaisesRegex(ValueError,'changed or unavailable'):
                await running
            self.assertTrue(replay.human.closed)
            count=replay.counters['mic']
            self.assertEqual(count['attempted'],count['captured']+count['dropped'])
            self.assertEqual(count['dropped'],1)
            self.assertLess(count['attempted'],count['declared'])
            self.assertTrue(any(e=='capture_drop' and v['reason']=='payload_verification_failed' for e,v in journal.events))

    async def test_external_close_joins_actual_runtime_and_prevents_snapshots(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            replay=RecordedSourceReplay(manifest,tasks,profiles,root,JournalFixture())
            running=asyncio.create_task(replay.run())
            await asyncio.sleep(.03)
            await replay.close()
            with self.assertRaises(asyncio.CancelledError):await running
            self.assertFalse(replay.snapshots)
            self.assertTrue(replay.human is None or replay.human.closed)
            self.assertTrue(all(t.done() for t in replay.jobs))
            self.assertFalse(replay.reads)

    async def test_artifact_wrapper_retains_original_mapping_context_and_terminal(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            output=root/'result'
            with self.assertRaises(PermissionError):
                await run_source_context_recording(manifest,tasks,profiles,output,asset_directory=root)
            self.assertFalse(output.exists())
            result=await run_source_context_recording(manifest,tasks,profiles,output,
                asset_directory=root,retain_evaluation_artifacts=True,max_lateness_ns=1000000000)
            terminal=json.loads((output/'terminal.json').read_text())
            self.assertEqual(terminal['state'],'replayed_unscored')
            self.assertEqual(terminal['summary'],result)
            self.assertFalse(terminal['whole_pipeline_cost_verified'])
            self.assertIsNone(result['monetary_cost'])
            self.assertEqual(json.loads((output/'source-manifest.json').read_text()),manifest)
            snapshot=json.loads((output/'context-0000.json').read_text())
            self.assertEqual(snapshot['context_sha256'],digest(snapshot['context']))
            self.assertEqual(snapshot['mapping_sha256'],result['mapping_sha256'])
            rows=[json.loads(line) for line in (output/'journal.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            with self.assertRaises(FileExistsError):
                await run_source_context_recording(manifest,tasks,profiles,output,
                    asset_directory=root,retain_evaluation_artifacts=True)

    async def test_cancelled_wrapper_keeps_failed_terminal_and_uninspected_suffix(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();manifest,tasks,profiles=self.inputs(root)
            output=root/'cancelled'
            job=asyncio.create_task(run_source_context_recording(manifest,tasks,profiles,output,
                asset_directory=root,retain_evaluation_artifacts=True,max_lateness_ns=1000000000))
            await asyncio.sleep(.08)
            job.cancel()
            with self.assertRaises(asyncio.CancelledError):await job
            terminal=json.loads((output/'terminal.json').read_text())
            self.assertEqual(terminal['state'],'failed')
            self.assertEqual(terminal['summary']['failure_category'],'CancelledError')
            self.assertGreater(terminal['summary']['sources']['mic']['uninspected_suffix'],0)
            self.assertTrue(terminal['cleanup_finished'])

    async def test_wall_jump_aborts_without_reanchoring_or_relabeling_capture(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            offset=[0]
            clock=SimpleNamespace(time_ns=lambda:time.time_ns()+offset[0],monotonic_ns=time.monotonic_ns)
            replay=RecordedSourceReplay(manifest,tasks,profiles,root,JournalFixture(),max_lateness_ns=1000000000)
            with patch('benchmarks.replay_clock.time',clock):
                running=asyncio.create_task(replay.run())
                await asyncio.sleep(.07)
                original_mapping=replay.clock.sha256
                offset[0]=60_000_000
                with self.assertRaisesRegex(RuntimeError,'discontinuity'):await running
            self.assertEqual(replay.clock.sha256,original_mapping)
            self.assertTrue(replay.human.closed)
            self.assertLess(replay.counters['mic']['attempted'],5)

    async def test_late_opportunities_are_drops_not_burst_submission(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,profiles=self.inputs(root)
            replay=RecordedSourceReplay(manifest,tasks,profiles,root,JournalFixture(),max_lateness_ns=1)
            await replay.run()
            counts=replay.counters['mic']
            self.assertEqual(counts['attempted'],5)
            self.assertEqual(counts['dropped'],5)
            self.assertEqual(counts['captured'],0)
            self.assertEqual(counts['submitted'],0)
            self.assertTrue(all(p['accepted']==0 for p in replay.provider_outcomes))

    async def test_cleanup_clock_jump_fails_and_keeps_independent_epoch_bridge(self):
        from humansense import HumanSense
        original_close=HumanSense.close
        offset=[0]
        async def jumping_close(human):
            await original_close(human)
            offset[0]=1000000000
        fake=SimpleNamespace(time_ns=lambda:time.time_ns()+offset[0],monotonic_ns=time.monotonic_ns)
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve();manifest,tasks,profiles=self.inputs(root)
            with patch('benchmarks.replay_clock.time',fake), patch.object(HumanSense,'close',jumping_close):
                with self.assertRaisesRegex(RuntimeError,'discontinuity'):
                    await run_source_context_recording(manifest,tasks,profiles,root/'output',
                        asset_directory=root,retain_evaluation_artifacts=True,max_lateness_ns=1000000000)
            terminal=json.loads((root/'output'/'terminal.json').read_text())
            bridges=json.loads((root/'output'/'source-mappings.json').read_text())
            self.assertEqual(bridges[0]['original_source']['stream_epoch'],'old-epoch')
            self.assertTrue(bridges[0]['provider_epochs'][0]['stream_epoch'])
            self.assertEqual(terminal['summary']['source_mappings_sha256'],digest(bridges))
            self.assertEqual(terminal['summary']['failure_category'],'RuntimeError')
            self.assertEqual(terminal['summary']['state'],'failed')

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional BGR array runtime unavailable')
    async def test_cancelled_provider_work_retains_unknown_disposition_and_epoch_bridge(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory).resolve(); manifest,tasks,_=self.inputs(root)
            pixels=bytes([20,40,80])*(16*16);(root/'bgr.bin').write_bytes(pixels)
            manifest['sources'][0].update(format='bgr8_hwc',modality='video')
            manifest['records']=[]
            for i in range(3):
                manifest['records'].append(dict(source_id='mic',sensor_id='mic-1',stream_epoch='old-epoch',
                    sequence=i,capture_ns=1000000000+i*20000000,offset_ns=i*20000000,
                    format='bgr8_hwc',shape=[16,16,3],payload_path='bgr.bin',payload_bytes=len(pixels),
                    payload_sha256=hashlib.sha256(pixels).hexdigest()))
            from humansense.streams import SensorStreams
            original_submit=SensorStreams.submit
            admitted=asyncio.Event(); count=[0]
            async def observed_submit(streams,frame):
                result=await original_submit(streams,frame)
                count[0]+=1
                if count[0]==3:admitted.set()
                return result
            with patch.object(SensorStreams,'submit',observed_submit):
                running=asyncio.create_task(run_source_context_recording(manifest,tasks,
                    {'mic':dict(profile='fixture_bgr',delay_ms=500)},root/'output',asset_directory=root,
                    retain_evaluation_artifacts=True,max_lateness_ns=1000000000))
                try:await asyncio.wait_for(admitted.wait(),2)
                finally:
                    running.cancel()
                    with self.assertRaises(asyncio.CancelledError):await running
            terminal=json.loads((root/'output'/'terminal.json').read_text())
            summary=terminal['summary'];lane=summary['provider_outcomes'][0]
            self.assertEqual(lane['accepted'],3)
            self.assertEqual(lane['unresolved_delivery'],3)
            self.assertEqual(lane['preclose_work'],dict(queued=2,unfinished=3,active=1))
            self.assertEqual(lane['final_unfinished'],0)
            self.assertTrue(lane['provider_task_done'])
            self.assertTrue(lane['provider_task_cancelled'])
            self.assertFalse(lane['disposition_truth_verified'])
            self.assertEqual(summary['sources']['mic']['dropped'],0)
            # Snapshot scheduling is independent of the third source admission.
            # The durable bridge must exist regardless of how many snapshots ran.
            bridges=json.loads((root/'output'/'source-mappings.json').read_text())
            self.assertEqual(summary['source_mappings_sha256'],digest(bridges))
            self.assertEqual(bridges[0]['provider_epochs'][0]['stream_epoch'],lane['stream_epoch'])
