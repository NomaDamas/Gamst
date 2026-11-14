import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

from benchmarks.recording_native_channel import (ParentNativeChannel,ChildNativeChannel,NativeChannelError,
    FD_ENVIRONMENT_KEY,frame,receive,load_declarations,_hello,MAX_FRAMES,MAX_WIRE_MULTIPLIER,MAX_WIRE_OVERHEAD)
from benchmarks.recording_native_schema import encoded,run_context
from recording_native_fixtures import declarations,rehash


CHILD='''
import asyncio,os,sys
from pathlib import Path
from benchmarks.recording_native_channel import ChildNativeChannel
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.replay_clock import ReplayClock
async def main():
    channel=await ChildNativeChannel.from_environment()
    assert not os.get_inheritable(channel.socket.fileno())
    writer=await RecordingNativeWriter.open(Path(sys.argv[1]),channel.context)
    channel.follow_writer(writer)
    observer=RecordingEvaluationSession.from_frozen(channel.declarations,writer,
        run_nonce=channel.context['nonce'],execution_ordinal=channel.context['invocation_ordinal'],retain_evaluation_hashes=True)
    clock=ReplayClock(channel.declarations['source_manifest'])
    observer.bind_mapping(clock.mapping,clock.sha256)
    for i in range(260):
        writer.emit('incomplete',dict(reason='settlement_unknown',count=i))
        if i%32==31:await writer.drain()
    await writer.drain()
    await asyncio.sleep(.12)
    await observer.close()
    await channel.close()
asyncio.run(main())
'''


class ChannelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).absolute()
        self.d=declarations(self.root)
        self.paths={}
        for name,value in self.d.items():
            path=self.root/(name+'.json');path.write_bytes(encoded(value)+b'\n');self.paths[name]=str(path)
        self.context=run_context(self.d,'a'*64,0)

    def parent(self,name='received',seconds=5):
        return ParentNativeChannel(self.context,self.d,paths=self.paths,output=self.root/name,
            deadline_ns=time.monotonic_ns()+int(seconds*1e9),
            retain_financial_documents=self.d['scope_protocol']['retain_financial_documents'])

    async def process(self,parent,script=CHILD,*arguments):
        process=await asyncio.create_subprocess_exec(sys.executable,'-c',script,str(self.root/'staging'),*map(str,arguments),
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,**parent.prepare(os.environ))
        parent.spawned(process)
        return process

    async def reap(self,process):
        if process.returncode is None:process.kill()
        await process.wait()

    async def test_parent_snapshot_and_declarations_are_issued_immutable_state(self):
        from benchmarks.recording_native_channel import NativeRunInventory
        with self.assertRaises(TypeError):NativeRunInventory({'parent_reaped':True})
        parent=ParentNativeChannel(self.context,self.d,paths=self.paths,output=self.root/'guarded-parent',deadline_ns=time.monotonic_ns()+10**9)
        try:
            with self.assertRaises(TypeError):parent.context['nonce']='b'*64
            with self.assertRaises(TypeError):parent.declarations['collection']['capacity']['max_entries_per_run']=999999
            value=parent.snapshot()
            self.assertIs(type(value),NativeRunInventory)
            with self.assertRaises(TypeError):value['parent_reaped']=True
        finally:await parent.abort()

    async def test_actual_subprocess_bootstrap_live_chunks_and_parent_reap(self):
        parent=self.parent();process=await self.process(parent)
        collection=asyncio.create_task(parent.collect())
        try:
            for _ in range(100):
                if parent.snapshot()['validated_prefix']['chunks']:break
                await asyncio.sleep(.01)
            self.assertGreater(parent.snapshot()['validated_prefix']['chunks'],0)
            self.assertFalse(parent.parent_reaped)
            result=await collection
            stdout,stderr=await process.communicate()
            self.assertEqual(process.returncode,0,stderr.decode())
            self.assertTrue(result['hello_verified']);self.assertTrue(result['eof'])
            self.assertFalse(result['parent_reaped'])
            await parent.record_reaped(process);self.assertTrue(parent.snapshot()['parent_reaped'])
            self.assertEqual(parent.validated_index.count,0)
            self.assertEqual(parent.terminal['event_count'],261)
            self.assertEqual(parent.terminal['actual_acceptance'],'incomplete')
            self.assertEqual((self.root/'staging'/'native-000000.jsonl').read_bytes(),(parent.output/'native-000000.jsonl').read_bytes())
        finally:
            await parent.abort();await self.reap(process)

    async def test_wrong_hello_causes_prompt_owned_caller_cleanup(self):
        marker=self.root/'forbidden-marker'
        script='''
import asyncio,os,socket,sys
from pathlib import Path
from benchmarks.recording_native_channel import receive,frame,_hello,FD_ENVIRONMENT_KEY
async def main():
 s=socket.socket(fileno=int(os.environ[FD_ENVIRONMENT_KEY]));s.setblocking(False)
 b=await receive(s);h=_hello(b);h['nonce']='f'*64
 await asyncio.get_running_loop().sock_sendall(s,frame(h))
 await asyncio.sleep(.3);Path(sys.argv[2]).write_text('late side effect');await asyncio.sleep(30)
asyncio.run(main())
'''
        parent=self.parent(seconds=3);process=await self.process(parent,script,marker)
        start=time.monotonic()
        try:
            with self.assertRaises(NativeChannelError):await parent.collect()
            await self.reap(process)
            self.assertLess(time.monotonic()-start,1)
            self.assertFalse(marker.exists())
            self.assertEqual(parent.reason,'invalid_channel')
            self.assertFalse(parent.parent_reaped)
        finally:await parent.abort();await self.reap(process)

    async def test_declared_snapshots_changed_duplicate_and_fd_collision_reject(self):
        parent=self.parent()
        with self.assertRaises(ValueError):parent.prepare(dict(os.environ,**{FD_ENVIRONMENT_KEY:'3'}))
        changed=Path(self.paths['plan']);original=changed.read_bytes()
        changed.write_bytes(b'{"schema_version":1,"schema_version":2}')
        with self.assertRaises(ValueError):load_declarations(parent.bootstrap)
        changed.write_bytes(original+b'garbage')
        with self.assertRaises(ValueError):load_declarations(parent.bootstrap)
        await parent.abort()

    async def test_partial_length_payload_and_duplicate_keys_reject(self):
        for data in (b'\x00\x00',b'\x00\x00\x00\x05{}',b'\x00\x00\x00\x0d{"x":1,"x":2}'):
            a,b=socket.socketpair();a.setblocking(False);b.sendall(data);b.close()
            try:
                with self.assertRaises((ValueError,NativeChannelError)):await receive(a)
            finally:a.close()

    async def test_fixed_frame_and_received_wire_amplification_caps(self):
        self.assertEqual(MAX_FRAMES,262144)
        parent=self.parent();a,b=socket.socketpair();child=ChildNativeChannel(a,parent.bootstrap,self.d)
        try:
            child._sequence=MAX_FRAMES
            with self.assertRaises(NativeChannelError):child.send('terminal',dict(terminal={}))
            self.assertEqual(child._pending_frames,0)
            parent.received=MAX_WIRE_MULTIPLIER*parent.budget.limit+MAX_WIRE_OVERHEAD-1
            a2,b2=socket.socketpair();a2.setblocking(False)
            try:
                b2.sendall(b'\x00\x00');b2.close()
                with self.assertRaises(ValueError):await receive(a2,consume=parent._consume)
                self.assertGreater(parent.received,MAX_WIRE_MULTIPLIER*parent.budget.limit+MAX_WIRE_OVERHEAD)
            finally:a2.close();b2.close()
        finally:b.close();await child.abort();await parent.abort()

    async def test_blocked_writer_repeated_abort_joins_socket_and_tasks(self):
        parent=self.parent();a,b=socket.socketpair();a.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4096)
        child=ChildNativeChannel(a,parent.bootstrap,self.d)
        data=base64.b64encode(b'x'*(256*1024)).decode()
        child.send('chunk_data',dict(chunk_ordinal=0,offset=0,data_base64=data))
        await asyncio.sleep(.01)
        abort=asyncio.create_task(child.abort());await asyncio.sleep(0);abort.cancel();await asyncio.sleep(0);abort.cancel()
        try:
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(abort,1)
            self.assertTrue(child._writer.done());self.assertEqual(child.socket.fileno(),-1)
        finally:b.close();await parent.abort();await child.abort()

    async def test_forged_sequence_preserves_live_prefix_and_rejects_terminal(self):
        script=CHILD.replace('    await observer.close()\n    await channel.close()', """    from benchmarks.recording_native_channel import frame
    await channel.drain()
    bad=dict(schema_version=2,type='chunk_data',nonce=channel.context['nonce'],frame_sequence=999999,
        fields=dict(chunk_ordinal=1,offset=0,data_base64='eA=='))
    await asyncio.get_running_loop().sock_sendall(channel.socket,frame(bad))
    await asyncio.sleep(30)""")
        parent=self.parent();process=await self.process(parent,script)
        try:
            with self.assertRaises(NativeChannelError):await parent.collect()
            await self.reap(process)
            self.assertEqual(parent.snapshot()['validated_prefix']['chunks'],1)
            self.assertTrue((parent.output/'native-000000.jsonl').exists())
            self.assertFalse((parent.output/'terminal.json').exists())
            self.assertEqual(parent.reason,'invalid_channel')
        finally:await parent.abort();await self.reap(process)

    async def test_trailing_frame_after_terminal_keeps_prefix_without_publishing_terminal(self):
        script=CHILD.replace('    await channel.close()', """    await channel._follow_task
    terminal=await channel.export_run(writer.output)
    await channel._send_wait('terminal',dict(terminal=terminal))
    await channel.drain()
    from benchmarks.recording_native_channel import frame
    extra=dict(schema_version=2,type='counter_page',nonce=channel.context['nonce'],
        frame_sequence=channel._sequence,fields=dict(ordinal=0,page={}))
    await asyncio.get_running_loop().sock_sendall(channel.socket,frame(extra))
    await asyncio.sleep(30)""")
        parent=self.parent();process=await self.process(parent,script)
        try:
            with self.assertRaises(NativeChannelError) as caught:await parent.collect()
            self.assertIn('trailing channel frame after terminal',str(caught.exception.__cause__))
            self.assertGreater(parent.snapshot()['validated_prefix']['chunks'],0)
            self.assertFalse((parent.output/'terminal.json').exists())
        finally:await parent.abort();await self.reap(process)

    async def test_source_frames_without_consent_fail_before_source_artifact(self):
        script=CHILD[:CHILD.index('    writer=')]+"""    from benchmarks.recording_native_channel import frame
    data=dict(schema_version=2,type='receipt_source_data',nonce=channel.context['nonce'],frame_sequence=0,
        fields=dict(source_sha256='f'*64,offset=0,data_base64='eA=='))
    await asyncio.get_running_loop().sock_sendall(channel.socket,frame(data))
    await asyncio.sleep(30)
asyncio.run(main())
"""
        parent=self.parent();process=await self.process(parent,script)
        try:
            with self.assertRaises(NativeChannelError):await parent.collect()
            self.assertEqual(list(parent.output.iterdir()),[])
        finally:await parent.abort();await self.reap(process)

    async def test_hello_timeout_and_repeated_parent_abort_close_all_descriptors(self):
        script='import time;time.sleep(30)'
        parent=self.parent(seconds=.05);process=await self.process(parent,script)
        try:
            with self.assertRaises(asyncio.TimeoutError):await parent.collect()
            jobs=[asyncio.create_task(parent.abort()) for _ in range(3)]
            await asyncio.gather(*jobs)
            self.assertEqual(parent.parent.fileno(),-1);self.assertEqual(parent.child.fileno(),-1)
            self.assertIsNone(parent._directory_fd)
            with self.assertRaises(ValueError):await parent.record_reaped(process)
        finally:await parent.abort();await self.reap(process)

    async def test_declared_receipt_transport_preserves_original_bytes_and_unknown_money(self):
        from benchmarks.recording_native_schema import digest
        card={'request':dict(provider='test.client',revision='v1',unit='request',category='request',per_unit=1)}
        self.d['plan']['comparison']['cost']=dict(mode='itemized',currency='USD',scope='full_pipeline',
            accounting_protocol='Synthetic declared fixture only',rate_card=card)
        self.d['cost_card']=card;self.d['scope_protocol']['cost_card_sha256']=digest(card)
        self.d['scope_protocol']['retain_financial_documents']=True;rehash(self.d)
        for name,value in self.d.items():
            path=self.root/(name+'.json');path.write_bytes(encoded(value)+b'\n');self.paths[name]=str(path)
        self.context=run_context(self.d,'a'*64,0)
        script=CHILD[:CHILD.index('    clock=')].replace('writer=await RecordingNativeWriter.open(Path(sys.argv[1]),channel.context)',
            'writer=await RecordingNativeWriter.open(Path(sys.argv[1]),channel.context,retain_financial_documents=True)').replace(
            'retain_evaluation_hashes=True)','retain_evaluation_hashes=True,retain_financial_documents=True)')+r"""    import hashlib
    from benchmarks.recording_native_schema import digest
    sys.path.insert(0,'tests')
    from test_text_context_call import Calls
    from humansense.schema import timestamp
    fixture=Calls();await fixture.asyncSetUp()
    bound=[]
    async def actual(binding):bound.append(binding);return dict(answer='synthetic')
    observer.bind_semantic_lane(fixture.c,0)
    observer.bind_backend(0,actual,'callback')
    clock=ReplayClock(channel.declarations['source_manifest']);observer.bind_mapping(clock.mapping,clock.sha256)
    scope=observer.task_scope(channel.declarations['tasks'][0]['task_id'],0)
    rid,record=await fixture.queued()
    ticket=observer.reserve_call(fixture.c,rid,provider='test.client',model_revision='v1',prompt_sha256='b'*64,evaluation_scope=scope)
    fixture.c.reserve(rid,provider='test.client',model_revision='v1',prompt_sha256='b'*64,maximum_charge=2,ttl_ms=1000,now_ms=timestamp())
    fixture.c.dispatch(rid,now_ms=timestamp());ticket.dispatched()
    await ticket.invoke_bound(actual);ticket.finish('completed')
    source=b'original financial fixture\r\n'
    claim=digest(['gamst.recording.claim.v1','test.client','account','invoice','line'])
    receipt=dict(schema_version=1,kind='recording_declared_receipt',origin='synthetic_fixture',money_basis='unknown',
        provider='test.client',revision='v1',account_scope='account',invoice_id='invoice',
        source_document_sha256=hashlib.sha256(source).hexdigest(),source_document_bytes=len(source),
        lines=[dict(line_id='line',quantities={'request':1},claim_sha256=claim)])
    await observer.attach_declared_receipt(bound[0],receipt,source)
    observer.backend_closed(0,actual);await observer.close();await channel.close();await fixture.asyncTearDown()
asyncio.run(main())
"""
        parent=self.parent();process=await self.process(parent,script)
        try:
            try:await parent.collect()
            except BaseException:
                await self.reap(process)
                _,stderr=await process.communicate()
                if stderr:raise AssertionError(stderr.decode())
                raise
            _,stderr=await process.communicate()
            self.assertEqual(process.returncode,0,stderr.decode())
            self.assertEqual(parent.validated_index.count,1)
            self.assertEqual(parent.terminal['unsettled_entries'],1)
            self.assertFalse(parent.terminal['receipt_authenticity_verified'])
            sources=list(parent.output.glob('receipt-source-*.bin'))
            self.assertEqual(len(sources),1)
            self.assertEqual(sources[0].read_bytes(),b'original financial fixture\r\n')
        finally:await parent.abort();await self.reap(process)

    async def test_valid_partial_prefix_survives_child_writer_failure(self):
        script=CHILD[:CHILD.index('    for i in range(260):')]+"""    from benchmarks.recording_native_writer import SUMMARY_KEYS
    await writer.drain()
    counters=[dict(scope=n,binding=0,entered=0,finished=0,published=0,unsettled=0) for n in (0,1)]
    ref=await writer.write_counter_page(dict(schema_version=1,kind='recording_counters',nonce=channel.context['nonce'],ordinal=0,counters=counters))
    def fail(rows):raise OSError('owned fixture sink failure')
    writer._append=fail
    writer.emit('incomplete',dict(reason='settlement_unknown',count=1))
    try:await writer.drain()
    except RuntimeError:pass
    summary=dict(mapping_sha256=clock.sha256,reservation_count=0,released_count=0,outstanding_tickets=0,
        outstanding_entries=0,outstanding_work=0,registered_workers=0,unclosed_workers=0,unsettled_entries=0,
        counter_pages=[ref],incomplete_counts={'journal_failed':1})
    try:await writer.close(summary)
    except RuntimeError:pass
    await channel.close()
asyncio.run(main())
"""
        parent=self.parent();process=await self.process(parent,script)
        try:
            result=await parent.collect();_,stderr=await process.communicate()
            self.assertEqual(process.returncode,0,stderr.decode())
            self.assertEqual(result['failure_category'],'native_incomplete')
            self.assertEqual(parent.terminal['state'],'incomplete')
            prefix=parent.snapshot()['validated_prefix']['partial']
            self.assertEqual(prefix['valid_events'],1)
            data=(parent.output/'native-000000.jsonl').read_bytes()
            self.assertEqual(len(data),prefix['valid_bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(),prefix['sha256'])
        finally:await parent.abort();await self.reap(process)

    async def test_parent_copy_is_immutable_after_child_staging_changes(self):
        parent=self.parent();process=await self.process(parent)
        collection=asyncio.create_task(parent.collect())
        try:
            await collection
            await process.communicate()
            self.assertGreater(parent.snapshot()['validated_prefix']['chunks'],0)
            # The independently persisted prefix cannot be modified by child
            # staging mutations after close.
            saved=(parent.output/'native-000000.jsonl').read_bytes()
            (self.root/'staging'/'native-000000.jsonl').write_bytes(b'changed')
            self.assertEqual((parent.output/'native-000000.jsonl').read_bytes(),saved)
        finally:await parent.abort();await self.reap(process)


if __name__=='__main__':unittest.main()
