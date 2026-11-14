import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from benchmarks.recording_artifacts import (RecordingArtifacts, preflight, MAX_EVENT_BYTES,
    MAX_JOURNAL_BYTES, MAX_QUEUE_FRAMES)
from benchmarks.perception_comparison import plan_digest


class Artifacts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    async def open(self, name='out', predicted=100):
        return await RecordingArtifacts.open(self.root / name,
            declarations={'source.json': {'schema_version': 1, 'id': 'source'}}, predicted_events=predicted)

    async def test_exact_hash_chain_detached_fields_permissions_and_immutable_terminal(self):
        owner = await self.open()
        fields = {'sequence': 3}; first = owner.emit('capture_acquired', fields); fields['sequence'] = 999
        second = owner.emit('snapshot_taken', {'task_id': 'task'})
        result = await owner.write_json('context-0001.json', {'data': [1, 2, 3]})
        terminal = await owner.close({'state': 'replayed_unscored'}, cleanup_finished=True)
        rows = [json.loads(line) for line in (owner.output/'journal.jsonl').read_bytes().splitlines()]
        self.assertEqual(rows[0]['fields']['sequence'], 3)
        self.assertEqual(rows[0]['previous_sha256'], '0'*64)
        self.assertEqual(rows[1]['previous_sha256'], first['sha256'])
        for index, row in enumerate(rows):
            self.assertEqual(row['sequence'], index)
            self.assertEqual(row['sha256'], plan_digest({k:v for k,v in row.items() if k!='sha256'}))
        summary = json.loads((owner.output/'terminal.json').read_bytes())
        self.assertEqual(summary['journal']['valid_prefix_sha256'], second['sha256'])
        self.assertEqual(summary['state'], 'replayed_unscored')
        self.assertEqual(summary['actual_acceptance'], 'incomplete')
        self.assertEqual(result['sha256'], hashlib.sha256((owner.output/result['path']).read_bytes()).hexdigest())
        self.assertEqual(terminal['sha256'], hashlib.sha256((owner.output/'terminal.json').read_bytes()).hexdigest())
        self.assertEqual(os.stat(owner.output).st_mode & 0o777, 0o700)
        for path in owner.output.iterdir():
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertEqual(await owner.close({'state':'failed'}, cleanup_finished=True), terminal)
        with self.assertRaises(RuntimeError): owner.emit('late_event', {})
        with self.assertRaises(FileExistsError): await self.open()

    async def test_preflight_event_size_nonfinite_and_prediction_bounds_fail_closed(self):
        with self.assertRaises(ValueError): preflight(MAX_JOURNAL_BYTES//MAX_EVENT_BYTES+1)
        with self.assertRaises(ValueError): preflight(True)
        owner = await self.open(predicted=1)
        owner.emit('first', {})
        with self.assertRaises(RuntimeError): owner.emit('extra', {})
        with self.assertRaises(RuntimeError): await owner.close({'state':'completed'}, cleanup_finished=True)
        self.assertEqual(json.loads((owner.output/'terminal.json').read_bytes())['state'], 'failed')
        for name, fields in (('oversized', {'value':'x'*2048}), ('nonfinite', {'value':float('nan')}), ('nonstring', {1:'x'})):
            owner = await self.open(name)
            with self.assertRaises(ValueError): owner.emit('invalid', fields)
            await owner.abort()
            self.assertFalse((owner.output/'terminal.json').exists())

    async def test_queue_including_inflight_is_bounded_and_emit_never_writes(self):
        owner = await self.open(); entered = threading.Event(); release = threading.Event()
        original = owner._append
        def blocked(rows, offset):
            entered.set(); release.wait(2); return original(rows, offset)
        with patch.object(owner, '_append', blocked):
            owner.emit('first', {})
            for _ in range(100):
                if entered.is_set(): break
                await asyncio.sleep(.001)
            self.assertTrue(entered.is_set())
            for _ in range(MAX_QUEUE_FRAMES-1): owner.emit('next', {})
            with self.assertRaises(RuntimeError): owner.emit('overflow', {})
            self.assertEqual(owner.status()['queued_frames'], MAX_QUEUE_FRAMES)
            self.assertEqual(owner.status()['written_events'], 0)
            release.set()
            with self.assertRaises(RuntimeError): await owner.close({'state':'completed'}, cleanup_finished=True)
        self.assertEqual(owner.status()['written_events'], MAX_QUEUE_FRAMES)
        self.assertEqual(owner.status()['failure'], 'journal_capacity_exhausted')

    async def test_partial_write_failure_preserves_last_durable_prefix_and_failed_terminal(self):
        owner = await self.open(); first = owner.emit('first', {})
        for _ in range(100):
            if owner.status()['written_events']==1: break
            await asyncio.sleep(.001)
        prefix = (owner.output/'journal.jsonl').read_bytes()
        real_write = os.write
        def broken(fd, data):
            if fd == owner._fd:
                real_write(fd, bytes(data)[:7]); raise OSError('synthetic failure')
            return real_write(fd, data)
        with patch('benchmarks.recording_artifacts.os.write', broken):
            owner.emit('second', {})
            for _ in range(100):
                if owner.status()['failure']: break
                await asyncio.sleep(.001)
        await owner.write_json('failure-details.json', {'reason':'write_failed'})
        with self.assertRaises(RuntimeError): await owner.close({'state':'completed'}, cleanup_finished=True)
        self.assertEqual((owner.output/'journal.jsonl').read_bytes(), prefix)
        report = json.loads((owner.output/'terminal.json').read_bytes())
        self.assertEqual(report['state'], 'failed')
        self.assertEqual(report['journal']['valid_prefix_sha256'], first['sha256'])
        self.assertEqual(report['journal']['unwritten_events'], 1)

    async def test_repeated_close_cancellation_joins_owned_thread_before_fd_close(self):
        owner = await self.open(); entered = threading.Event(); release = threading.Event()
        original = owner._append
        def blocked(rows, offset):
            entered.set(); release.wait(2); return original(rows, offset)
        with patch.object(owner, '_append', blocked):
            owner.emit('first', {})
            while not entered.is_set(): await asyncio.sleep(.001)
            closing = asyncio.create_task(owner.close({'state':'completed'}, cleanup_finished=True))
            await asyncio.sleep(0); closing.cancel(); await asyncio.sleep(0); closing.cancel()
            self.assertFalse(closing.done()); self.assertIsNotNone(owner._fd)
            release.set()
            with self.assertRaises(asyncio.CancelledError): await closing
        self.assertIsNone(owner._fd); self.assertEqual(owner.status()['owned_write_jobs'],0)
        self.assertTrue(owner.status()['writer_done']); self.assertTrue((owner.output/'terminal.json').exists())

    async def test_cleanup_required_and_failed_encoding_still_aborts_writer(self):
        owner = await self.open()
        with self.assertRaises(ValueError): await owner.close({'state':'completed'}, cleanup_finished=False)
        with self.assertRaises(ValueError): await owner.write_json('../escape.json', {})
        with self.assertRaises(ValueError): await owner.write_json('bad.json', {'value':float('inf')})
        await owner.abort(); await owner.abort()
        self.assertIsNone(owner._fd); self.assertTrue(owner.status()['writer_done'])
        self.assertFalse((owner.output/'terminal.json').exists())

    async def test_close_joins_pending_file_jobs_and_no_duplicate_overwrite(self):
        owner = await self.open(); entered=threading.Event(); release=threading.Event()
        original=owner._save_json
        def blocked(name,value,limit):
            if name.startswith('context'):
                entered.set(); release.wait(2)
            return original(name,value,limit)
        with patch.object(owner,'_save_json',blocked):
            jobs=[asyncio.create_task(owner.write_json(f'context-{i}.json', {'i':i})) for i in range(6)]
            while not entered.is_set(): await asyncio.sleep(.001)
            closing=asyncio.create_task(owner.close({'state':'completed'}, cleanup_finished=True))
            await asyncio.sleep(.01); self.assertFalse(closing.done())
            release.set(); await asyncio.gather(*jobs); await closing
        report=json.loads((owner.output/'terminal.json').read_bytes())
        self.assertEqual(len(report['files']),7)
        with self.assertRaises(RuntimeError): await owner.write_json('context-0.json', {'changed':True})


    async def test_terminal_encoding_failure_closes_all_owned_descriptors(self):
        owner=await self.open(); owner.emit('first', {})
        with self.assertRaises(ValueError):
            await owner.close({'state':'completed','bad':float('nan')},cleanup_finished=True)
        self.assertIsNone(owner._fd); self.assertIsNone(owner._directory_fd)
        self.assertTrue(owner.status()['writer_done']); self.assertEqual(owner.status()['written_events'],1)
        self.assertFalse((owner.output/'terminal.json').exists())
        await owner.abort()

    async def test_replaced_output_directory_cannot_redirect_artifact_writes(self):
        owner=await self.open(); retained=self.root/'retained'; foreign=self.root/'foreign';foreign.mkdir()
        owner.output.rename(retained); owner.output.symlink_to(foreign,target_is_directory=True)
        with self.assertRaises(ValueError): await owner.write_json('context.json',{})
        await owner.abort()
        self.assertEqual(list(foreign.iterdir()),[])
        self.assertTrue((retained/'source.json').exists())
        self.assertIsNone(owner._fd); self.assertIsNone(owner._directory_fd)

    async def test_partial_open_failure_preserves_exclusive_directory(self):
        with self.assertRaises(ValueError):
            await RecordingArtifacts.open(self.root/'partial', declarations={'bad.json':{'nan':float('nan')}},predicted_events=1)
        self.assertTrue((self.root/'partial').is_dir())
        self.assertEqual((self.root/'partial/journal.jsonl').read_bytes(),b'')
        with self.assertRaises(FileExistsError):
            await RecordingArtifacts.open(self.root/'partial',declarations={},predicted_events=0)


if __name__ == '__main__': unittest.main()
