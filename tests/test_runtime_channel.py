import asyncio
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from benchmarks.paired_replay import _encoded
from benchmarks.paired_replay import _execute
from benchmarks.perception_comparison import plan_digest
from benchmarks.runtime_channel import (ParentChannel,ChildChannel,CollectionBudget,frame,receive,
    build_bootstrap,load_declarations,read_declaration_paths,validate_instrumentation_protocol,HASH_FIELDS,MAX_FRAME,MAX_BYTES,FD_ENVIRONMENT_KEY)


def declarations(root):
    values=dict(plan={},collection={},inputs=[],stream_protocol={},allocation_protocol={},reconciliation_protocol={})
    hashes=dict(zip(HASH_FIELDS[:-1],map(plan_digest,values.values())))
    policy=dict(schema_version=1,**{k:v for k,v in hashes.items() if k!='inputs_sha256'},
        transport='socketpair_be32_json_v1',hello_timeout_ms=500,fd_environment_key=FD_ENVIRONMENT_KEY,
        retention_policy='external_owner_retains_task_and_evidence_hashes',phase='unknown')
    values['instrumentation_protocol']=policy;hashes['instrumentation_protocol_sha256']=plan_digest(policy)
    paths={}
    for name,value in values.items():
        path=root/(name+'.json');path.write_bytes(_encoded(value));paths[name]=str(path)
    return build_bootstrap(paths=paths,hashes=hashes,task_id='0',arm='A',stream_id='stream-A',
        request_sha256='a'*64,hello_timeout_ms=500)


def event(index=0,previous='0'*64,**changes):
    value=dict(schema_version=1,session_id='a'*32,clock_domain='host-monotonic:'+'a'*32,
        sequence=index,previous_sha256=previous,monotonic_ns=time.monotonic_ns(),operation_id=None,
        event='private_mappings_invalidated',fields=dict(reason='close',affected=0))
    value.update(changes);value['sha256']=plan_digest(value);return value


class Channel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='.');self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name).resolve();self.bootstrap=declarations(self.root)
    async def peer(self,parent):
        parent.prepare({},2);peer=parent.child.dup();peer.setblocking(False);parent.spawned()
        reader=asyncio.create_task(parent.collect())
        bootstrap=await receive(peer)
        await asyncio.get_running_loop().sock_sendall(peer,frame(dict(schema_version=1,type='hello',bootstrap=bootstrap)))
        return peer,reader
    async def test_strict_frames_partial_duplicate_nonfinite_and_oversized(self):
        utf16=b'{"a":1}'.decode().encode('utf-16')
        for data in (b'\x00\x00',struct.pack('!I',MAX_FRAME+1),struct.pack('!I',13)+b'{"a":1,"a":2}',
                     struct.pack('!I',9)+b'{"a":NaN}',struct.pack('!I',10)+b'{}',struct.pack('!I',len(utf16))+utf16):
            left,right=socket.socketpair();left.setblocking(False);right.setblocking(False)
            try:
                await asyncio.get_running_loop().sock_sendall(right,data);right.shutdown(socket.SHUT_WR)
                with self.assertRaises((ValueError,UnicodeError)):await receive(left)
            finally:left.close();right.close()
    async def test_valid_prefix_preserved_on_wrong_chain_missing_terminal_and_trailing(self):
        parent=ParentChannel(self.bootstrap);peer,reader=await self.peer(parent)
        item=dict(schema_version=1,type='event',nonce=parent.bootstrap['nonce'],sequence=0,
            previous_sha256='0'*64,payload=event());item['sha256']=plan_digest(item)
        await asyncio.get_running_loop().sock_sendall(peer,frame(item))
        bad=dict(item,sequence=2);bad.pop('sha256');bad['sha256']=plan_digest(bad)
        await asyncio.get_running_loop().sock_sendall(peer,frame(bad));peer.close();await reader;parent.close()
        self.assertEqual(len(parent.events),1);self.assertEqual(parent.sequence,1)
        self.assertFalse(parent.snapshot()['instrumentation_complete']);self.assertEqual(parent.reason,'invalid_channel')
    async def test_global_budget_and_environment_collision_close_owned_fds(self):
        parent=ParentChannel(self.bootstrap,budget=CollectionBudget(16))
        with self.assertRaises(ValueError):parent.prepare({FD_ENVIRONMENT_KEY:'2'},1)
        parent.close();self.assertEqual(parent.parent.fileno(),-1);self.assertEqual(parent.child.fileno(),-1)
        parent=ParentChannel(self.bootstrap,budget=CollectionBudget(16));peer,reader=await self.peer(parent)
        await reader;peer.close();parent.close();self.assertTrue(parent.budget.exhausted)
        self.assertFalse(parent.snapshot()['instrumentation_complete'])
    async def test_child_reservation_control_credit_and_blocked_writer_are_bounded(self):
        left,right=socket.socketpair();left.setblocking(False);right.setblocking(False)
        bootstrap=dict(self.bootstrap,deadline_ns=time.monotonic_ns()+100000000)
        child=ChildChannel(left,bootstrap)
        try:
            child.reserve(8,32768)
            child.release(8,32768)
            for index in range(32):child.emit(event(index))
            with self.assertRaises(RuntimeError):child.emit(event(32))
            with self.assertRaises(RuntimeError):child.reserve(1,1)
            with self.assertRaises(RuntimeError):await child.close(reports=[],summary={})
            self.assertTrue(child.writer.done());self.assertEqual(left.fileno(),-1)
        finally:right.close()
    async def test_blocked_writer_deadline_and_repeated_close_cancel_leave_no_task(self):
        left,right=socket.socketpair();left.setblocking(False);right.setblocking(False)
        left.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4096)
        child=ChildChannel(left,dict(self.bootstrap,deadline_ns=time.monotonic_ns()+30000000000))
        try:
            child.reserve(63,63*4096)
            for index in range(63):child.emit(event(index,operation_id='b'*32))
            with patch('benchmarks.runtime_channel.CLOSE_TIMEOUT_SECONDS',.05):
                started=time.monotonic()
                closing=asyncio.create_task(child.close(reports=[{'sha256':'a'*64}]*2,summary={}))
                await asyncio.sleep(.01);closing.cancel();await asyncio.sleep(0);closing.cancel()
                with self.assertRaises((asyncio.TimeoutError,asyncio.CancelledError)):await closing
                self.assertLess(time.monotonic()-started,.5)
            self.assertTrue(child.writer.done());self.assertEqual(left.fileno(),-1)
            with self.assertRaises(RuntimeError):child.reserve(1,1)
        finally:right.close()
    async def test_spawn_failure_and_bounded_missing_hello_close_owned_descriptors(self):
        parent=ParentChannel(self.bootstrap)
        result=await _execute(dict(argv=['/no/such/runtime-owned-command'],cwd=str(self.root),environment={}),{},100,runtime_channel=parent)
        self.assertEqual(result['status'],'error');self.assertEqual(parent.child.fileno(),-1);self.assertEqual(parent.parent.fileno(),-1)
        parent=ParentChannel(self.bootstrap);parent.prepare({},.02)
        peer=parent.child.dup();peer.setblocking(False);parent.spawned();started=time.monotonic()
        await parent.collect();parent.close();peer.close()
        self.assertLess(time.monotonic()-started,.5);self.assertEqual(parent.reason,'deadline_exceeded')
    async def test_declarations_are_bounded_regular_immutable_identity(self):
        self.assertEqual(load_declarations(self.bootstrap)['inputs'],[])
        path=Path(self.bootstrap['paths']['inputs']);path.write_text('[1]')
        with self.assertRaises(ValueError):load_declarations(self.bootstrap)
        path.unlink();os.mkfifo(path)
        with self.assertRaises(ValueError):load_declarations(self.bootstrap)
        policy=self.bootstrap['instrumentation_protocol_sha256']
        with self.assertRaises(ValueError):validate_instrumentation_protocol({},expected_sha256=policy)
    async def test_cli_declaration_reader_checks_aggregate_and_size_before_decode(self):
        total=sum(Path(path).stat().st_size for path in self.bootstrap['paths'].values())
        with patch('benchmarks.runtime_channel.MAX_BYTES',total-1):
            with self.assertRaises(ValueError):read_declaration_paths(self.bootstrap['paths'])
        first=Path(self.bootstrap['paths']['plan'])
        with first.open('wb') as stream:stream.truncate(MAX_BYTES+1)
        with patch('benchmarks.runtime_channel._strict_decode',side_effect=AssertionError('oversize data was decoded')):
            with self.assertRaises(ValueError):read_declaration_paths(self.bootstrap['paths'])


if __name__=='__main__':unittest.main()
