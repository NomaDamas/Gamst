import asyncio
import hashlib
from pathlib import Path
import tempfile
import unittest

from benchmarks.recorded_source import digest
from benchmarks.recorded_replay import RecordedSourceReplay


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
