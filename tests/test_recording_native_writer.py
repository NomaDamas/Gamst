import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from unittest import mock

from benchmarks.recording_native_writer import ArtifactBudget, RecordingNativeWriter, _line


def context():
    return dict(schema_version=4, kind='recording_native', nonce='a'*64,
        invocation_ordinal=0, collection_sha256='b'*64, scope_protocol_sha256='c'*64,
        instrumentation_protocol_sha256='d'*64)


def summary(**extra):
    value = dict(mapping_sha256=None, reservation_count=0, released_count=0,
        outstanding_tickets=0, outstanding_entries=0, outstanding_work=0,
        registered_workers=0, unclosed_workers=0, unsettled_entries=0,
        counter_pages=[], incomplete_counts={})
    value.update(extra)
    return value


class WriterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    async def open(self, name='run', **kwargs):
        return await RecordingNativeWriter.open(self.base/name, context(), **kwargs)

    async def test_exact_chain_rotation_detached_data_and_permissions(self):
        writer = await self.open()
        for batch in range(6):
            token = writer.reserve(50)
            for i in range(50):
                fields = dict(reason='settlement_unknown', count=batch*50+i)
                writer.emit('incomplete', fields, reservation=token)
                fields['count'] = -1
            writer.release(token)
            await writer.drain()
        terminal = await writer.close(summary())
        status = writer.status()
        self.assertEqual(len(status['chunks']), 2)
        self.assertEqual([r['event_count'] for r in status['chunks']], [256,44])
        previous = hashlib.sha256(json.dumps(['gamst.recording.events.v1',context()],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        seq = 0
        for ref in status['chunks']:
            data = (writer.output/ref['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), ref['sha256'])
            lines = data.splitlines(keepends=True)
            footer = json.loads(lines[-1])
            self.assertEqual(footer['prefix_sha256'],hashlib.sha256(b''.join(lines[:-1])).hexdigest())
            for line in lines[1:-1]:
                row = json.loads(line); digest=row.pop('sha256')
                self.assertEqual(row['sequence'],seq)
                self.assertEqual(row['previous_sha256'],previous)
                self.assertEqual(row['fields']['count'],seq)
                self.assertEqual(hashlib.sha256(_line(row,2048)[:-1]).hexdigest(),digest)
                previous=digest;seq+=1
        self.assertEqual(seq,300)
        self.assertEqual(stat.S_IMODE(writer.output.stat().st_mode),0o700)
        for path in writer.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
        self.assertEqual(terminal['path'],'terminal.json')
        with self.assertRaises(RuntimeError):writer.emit('incomplete',{})
        with self.assertRaises(FileExistsError):await self.open()

    async def test_thread_commit_and_entry_count_rotation(self):
        writer=await self.open()
        for base in range(0,130,10):
            token=writer.reserve(10)
            def commit():
                for i in range(base,base+10):writer.emit('entered',dict(entry_ordinal=i),reservation=token)
            await asyncio.to_thread(commit)
            writer.release(token);await writer.drain()
        await writer.close(summary(reservation_count=130,unsettled_entries=130))
        self.assertEqual([r['entry_count'] for r in writer.status()['chunks']],[128,2])
        self.assertEqual(writer.status()['entry_count'],130)

    async def test_inflight_counts_backpressure_and_repeated_cancel_cleanup(self):
        writer=await self.open()
        entered=threading.Event();release=threading.Event();original=writer._append
        def held(rows):
            entered.set();release.wait(3);return original(rows)
        writer._append=held
        writer.emit('incomplete',dict(reason='settlement_unknown',count=1))
        await asyncio.to_thread(entered.wait,1)
        for i in range(63):writer.emit('incomplete',dict(reason='settlement_unknown',count=i))
        self.assertEqual(writer.status()['pending_frames'],64)
        with self.assertRaises(RuntimeError):writer.emit('incomplete',dict(reason='settlement_unknown',count=0))
        close=asyncio.create_task(writer.close(summary()))
        await asyncio.sleep(0);close.cancel();await asyncio.sleep(0);close.cancel()
        release.set()
        with self.assertRaises((asyncio.CancelledError,RuntimeError)):await asyncio.wait_for(close,2)
        self.assertTrue(writer._writer.done());self.assertIsNone(writer._directory_fd)
        terminal=json.loads((writer.output/'terminal.json').read_bytes())
        self.assertEqual(terminal['state'],'incomplete')

    async def test_partial_write_preserves_previous_durable_prefix(self):
        writer=await self.open()
        writer.emit('incomplete',dict(reason='settlement_unknown',count=1));await writer.drain()
        before=writer.status()['prefix']
        original=os.write;called=False
        def broken(fd,data):
            nonlocal called
            if fd==writer._fd:
                if not called:
                    called=True;return original(fd,bytes(data[:9]))
                raise OSError('fixture partial write')
            return original(fd,data)
        with mock.patch('benchmarks.recording_native_writer.os.write',side_effect=broken):
            writer.emit('incomplete',dict(reason='settlement_unknown',count=2))
            with self.assertRaises(RuntimeError):await writer.drain()
        with self.assertRaises(RuntimeError):await writer.close(summary())
        actual=(writer.output/'native-000000.jsonl').read_bytes()
        self.assertEqual(len(actual),before['valid_bytes'])
        self.assertEqual(hashlib.sha256(actual).hexdigest(),before['sha256'])
        self.assertEqual(writer.status()['prefix'],before)
        self.assertFalse(writer.status()['chunks'])

    async def test_shared_budget_reservation_refusal_and_return_unused(self):
        budget=ArtifactBudget(limit=400000,initial_bytes=1000)
        one=await self.open('one',budget=budget,max_chunks=1)
        two=await self.open('two',budget=budget,max_chunks=1)
        token=one.reserve(60)
        with self.assertRaises(RuntimeError):two.reserve(60)
        used=budget.used;one.release(token);one.release(token)
        self.assertEqual(budget.used,used-60*2048)
        other=two.reserve(60);two.release(other)
        await one.close(summary());await two.close(summary())
        actual=1000+sum(p.stat().st_size for out in (one.output,two.output) for p in out.iterdir())
        self.assertEqual(budget.used,actual)

    async def test_invalid_terminal_still_closes_owned_nonempty_writer(self):
        writer=await self.open()
        writer.emit('incomplete',dict(reason='settlement_unknown',count=1))
        with self.assertRaises(ValueError):await writer.close(dict(summary(),context={}))
        self.assertTrue(writer._writer.done());self.assertIsNone(writer._directory_fd)
        self.assertFalse((writer.output/'terminal.json').exists())

    async def test_abort_preserves_prefix_without_terminal(self):
        writer=await self.open()
        writer.emit('incomplete',dict(reason='settlement_unknown',count=1));await writer.drain()
        await writer.abort();await writer.abort()
        self.assertTrue((writer.output/'native-000000.jsonl').exists())
        self.assertFalse((writer.output/'terminal.json').exists())
        self.assertIsNone(writer._fd);self.assertIsNone(writer._directory_fd)

    async def test_financial_consent_and_original_byte_identity(self):
        writer=await self.open()
        with self.assertRaises(PermissionError):await writer.write_receipt_source(b'raw')
        await writer.abort()
        writer=await self.open('financial',retain_financial_documents=True)
        source=await writer.write_receipt_source(b'raw\r\n')
        self.assertEqual(source['sha256'],hashlib.sha256(b'raw\r\n').hexdigest())
        claim=hashlib.sha256(_line(['gamst.recording.claim.v1','provider','account','invoice','line'],65536)[:-1]).hexdigest()
        receipt=await writer.write_receipt_attachment(dict(schema_version=1,kind='recording_declared_receipt',
            origin='synthetic_fixture',money_basis='unknown',provider='provider',revision='v1',account_scope='account',
            invoice_id='invoice',source_document_sha256=source['sha256'],source_document_bytes=5,
            lines=[dict(line_id='line',quantities={'rate':1},claim_sha256=claim)]))
        instance=hashlib.sha256(_line(['gamst.recording.instance.v1',context()['nonce'],0],65536)[:-1]).hexdigest()
        claims=await writer.write_claims_page(dict(schema_version=1,kind='recording_declared_claims',
            receipt_sha256=receipt['sha256'],owner_instance=instance,claims=[dict(line_index=0,claim_sha256=claim)]))
        writer.emit('entered',dict(entry_ordinal=0))
        writer.emit('settlement_ref',dict(entry_ordinal=0,receipt_sha256=receipt['sha256'],claims_page_sha256=claims['sha256']))
        await writer.close(summary(reservation_count=1,unsettled_entries=1))
        self.assertEqual((writer.output/source['path']).read_bytes(),b'raw\r\n')

    async def test_index_pages_rotate_at_128_chunks_without_event_history(self):
        writer=await self.open()
        total=128*256+1
        for start in range(0,total,64):
            for index in range(start,min(total,start+64)):
                writer.emit('incomplete',dict(reason='settlement_unknown',count=index))
            await writer.drain()
        await writer.close(summary())
        first=json.loads((writer.output/'index-000000.json').read_bytes())
        last=json.loads((writer.output/'index-000001.json').read_bytes())
        self.assertEqual(len(first['chunks']),128)
        self.assertEqual(len(last['chunks']),1)
        self.assertEqual(last['previous_page_sha256'],hashlib.sha256((writer.output/'index-000000.json').read_bytes()).hexdigest())
        self.assertEqual(writer.status()['written_events'],total)
        self.assertEqual(len(writer.status()['chunks']),129)
        self.assertFalse(writer._queue)
        self.assertLess((writer.output/'index-000000.json').stat().st_size,65536)

    async def test_directory_replacement_cannot_redirect_owned_writes(self):
        writer=await self.open()
        old=self.base/'original'
        writer.output.rename(old)
        writer.output.mkdir()
        writer.emit('incomplete',dict(reason='settlement_unknown',count=1))
        with self.assertRaises(RuntimeError):await writer.drain()
        with self.assertRaises(RuntimeError):await writer.close(summary())
        self.assertEqual(list(writer.output.iterdir()),[])
        self.assertIsNone(writer._directory_fd)

    async def test_max_chunks_and_foreign_reservations_fail_before_admission(self):
        writer=await self.open(max_chunks=1)
        other=await self.open('other')
        foreign=other.reserve(1)
        with self.assertRaises(ValueError):writer.emit('incomplete',{},reservation=foreign)
        other.release(foreign);await other.abort()
        for start in range(0,128,32):
            for index in range(start,start+32):writer.emit('entered',dict(entry_ordinal=index))
            await writer.drain()
        with self.assertRaises(RuntimeError):writer.emit('entered',dict(entry_ordinal=128))
        self.assertEqual(writer.status()['entry_count'],128)
        with self.assertRaises(RuntimeError):await writer.close(summary(reservation_count=128,unsettled_entries=128))

    async def test_orphan_receipt_cannot_get_closed_terminal(self):
        writer=await self.open(retain_financial_documents=True)
        await writer.write_receipt_source(b'original')
        with self.assertRaises(RuntimeError):await writer.close(summary())
        terminal=json.loads((writer.output/'terminal.json').read_bytes())
        self.assertEqual(terminal['state'],'incomplete')
        self.assertIn('journal_failed',terminal['incomplete_counts'])

    def test_declared_lines_are_bounded_before_unclaimed_inventory_storage(self):
        budget=ArtifactBudget()
        for index in range(4096):
            budget.admit_attachment(100,('p','a',str(index)),('v','a'*64),64)
        with self.assertRaises(RuntimeError):
            budget.admit_attachment(100,('p','a','extra'),('v','a'*64),1)
        self.assertEqual(budget._financial['declared_lines'],262144)

    async def test_nonfinite_and_event_size_are_rejected_before_write(self):
        writer=await self.open()
        with self.assertRaises(ValueError):writer.emit('incomplete',dict(value=float('nan')))
        with self.assertRaises(ValueError):writer.emit('incomplete',dict(value='x'*2048))
        self.assertEqual(writer.status()['admitted_events'],0)
        await writer.close(summary())


if __name__=='__main__':unittest.main()
