import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from benchmarks.recording_native_schema import digest,encoded
from benchmarks.recording_native_validation import (validate_recording_native_inventory,
    iter_canonical_operations,lookup_canonical_operation)
from test_recording_sdk_scopes import NativeFixture
import test_text_context_call as text_tests


class NativeValidation(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    adapter=text_tests.Calls.adapter

    async def artifact(self):
        observer,writer,clock=await self.recording()
        async def callback(request):return dict(text='original answer',usage=None)
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,callback,'callback')
        observer.bind_mapping(clock.mapping,clock.sha256)
        adapter=self.adapter(callback,evaluation_observer=observer)
        rid,record=await self.queued()
        scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)
        await adapter.execute(rid,record,'question',maximum_charge=1,evaluation_scope=scope)
        await adapter.close();observer.backend_closed(0,callback);await observer.close()
        return observer,writer

    def validate(self,observer,path):
        return validate_recording_native_inventory(observer.declarations,path,
            source_inventory=observer.declarations['source_manifest'])

    async def test_actual_sdk_inventory_and_compact_original_iterator(self):
        observer,writer=await self.artifact();index=self.validate(observer,writer.output)
        rows=list(iter_canonical_operations(index))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['scope_index'],1)
        self.assertTrue(rows[0]['unknown_liability']);self.assertIsNone(rows[0]['declared_settlement_ref'])
        self.assertEqual(len(index.entries),128);self.assertEqual(index.actual_acceptance,'incomplete')
        with self.assertRaises(TypeError):index.declarations['collection']['arms'][0]['provider_bindings'][0]['provider']='forged'
        with self.assertRaises(AttributeError):index.entries=b''
        lookup=lookup_canonical_operation(index,0)
        reserved=next(e['fields'] for e in self.events(writer) if e['event']=='reserved')
        self.assertEqual(lookup['entry']['request_decision_sha256'],reserved['request_sha256'])
        self.assertEqual(lookup['prompt_sha256'],reserved['prompt_sha256'])
        self.assertEqual(lookup['provider'],'test.client')
        self.assertEqual(lookup['owner']['scope_index'],1)
        self.assertEqual(lookup['source_chunk']['sha256'],index.chunk_refs[0]['sha256'])

    async def test_missing_truncated_or_changed_original_chunk_rejects(self):
        observer,writer=await self.artifact()
        for mode in ('missing','truncated','changed'):
            with tempfile.TemporaryDirectory(dir='.') as d:
                copy=Path(d)/'copy';shutil.copytree(writer.output,copy)
                path=copy/'native-000000.jsonl';body=path.read_bytes()
                if mode=='missing':path.rename(copy/'task-owned-removed-chunk.jsonl')
                elif mode=='truncated':path.write_bytes(body[:-8])
                else:path.write_bytes(body.replace(b'accepted',b'discarded'))
                with self.assertRaises((ValueError,FileNotFoundError)):self.validate(observer,copy)

    async def test_rehashed_terminal_cannot_hide_unsettled_or_pending_work(self):
        observer,writer=await self.artifact()
        for field,value in (('unsettled_entries',0),('outstanding_tickets',1),('entry_count',0),('registered_workers',0)):
            with tempfile.TemporaryDirectory(dir='.') as d:
                copy=Path(d)/'copy';shutil.copytree(writer.output,copy)
                path=copy/'terminal.json';terminal=json.loads(path.read_bytes());terminal[field]=value
                path.write_bytes(encoded(terminal)+b'\n')
                with self.assertRaises(ValueError):self.validate(observer,copy)

    async def test_rehashed_entry_before_dispatch_rejects(self):
        observer,writer=await self.artifact()
        with tempfile.TemporaryDirectory(dir='.') as d:
            copy=Path(d)/'copy';shutil.copytree(writer.output,copy)
            path=copy/'native-000000.jsonl';lines=[json.loads(line) for line in path.read_bytes().splitlines()]
            header,footer=lines[0],lines[-1];events=lines[1:-1]
            a=next(n for n,e in enumerate(events) if e['event']=='dispatched')
            b=next(n for n,e in enumerate(events) if e['event']=='entered')
            events[a],events[b]=events[b],events[a]
            last=digest(['gamst.recording.events.v1',header['context']]);now=events[0]['monotonic_ns']
            for n,event in enumerate(events):
                event.update(sequence=n,monotonic_ns=now+n,previous_sha256=last)
                event['sha256']=digest({k:v for k,v in event.items() if k!='sha256'});last=event['sha256']
            prefix=b''.join(encoded(row)+b'\n' for row in [header]+events)
            footer.update(prefix_bytes=len(prefix),prefix_sha256=hashlib.sha256(prefix).hexdigest())
            body=prefix+encoded(footer)+b'\n';path.write_bytes(body)
            p=copy/'index-000000.json';page=json.loads(p.read_bytes());page['chunks'][0].update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            data=encoded(page)+b'\n';p.write_bytes(data)
            p=copy/'terminal.json';term=json.loads(p.read_bytes());term['index_tail_sha256']=hashlib.sha256(data).hexdigest();p.write_bytes(encoded(term)+b'\n')
            with self.assertRaisesRegex(ValueError,'actual entry'):self.validate(observer,copy)

    async def test_cli_audits_actual_sdk_bundle_exclusively_and_refusal_creates_nothing(self):
        observer,writer=await self.artifact()
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in writer.output.iterdir()}
        with tempfile.TemporaryDirectory(dir='.') as d:
            root=Path(d);declaration=root/'declarations.json';output=root/'audit.json'
            declaration.write_bytes(encoded(observer.declarations)+b'\n')
            command=[sys.executable,'-m','benchmarks.recording_native_cli','--declarations',str(declaration),
                '--artifacts',str(writer.output),'--terminal-sha256',before['terminal.json'],'--output',str(output)]
            first=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(first.returncode,0,first.stderr)
            report=json.loads(output.read_bytes());self.assertEqual(report['unknown_liability_count'],1)
            self.assertEqual(report['actual_acceptance'],'incomplete')
            self.assertNotEqual(subprocess.run(command,capture_output=True,timeout=15).returncode,0)
            refused=root/'refused.json';command[-1]=str(refused);command[-3]='f'*64
            self.assertNotEqual(subprocess.run(command,capture_output=True,timeout=15).returncode,0)
            self.assertFalse(refused.exists())
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in writer.output.iterdir()})

    async def test_failed_prefix_is_validated_without_fabricated_completion(self):
        observer,writer=await self.artifact()
        with tempfile.TemporaryDirectory(dir='.') as d:
            copy=Path(d)/'copy';shutil.copytree(writer.output,copy)
            path=copy/'native-000000.jsonl';lines=path.read_bytes().splitlines(keepends=True)
            body=b''.join(lines[:-1]);path.write_bytes(body+b'{"partial')
            terminal=json.loads((copy/'terminal.json').read_bytes())
            terminal.update(state='incomplete',chunk_count=0,page_count=0,index_tail_sha256=None,
                prefix=dict(chunk_ordinal=0,valid_bytes=len(body),valid_events=len(lines)-2,
                    sha256=hashlib.sha256(body).hexdigest()))
            (copy/'terminal.json').write_bytes(encoded(terminal)+b'\n')
            index=self.validate(observer,copy)
            self.assertEqual(index.count,1)
            self.assertEqual(lookup_canonical_operation(index,0)['entry']['entry_ordinal'],0)
            path.write_bytes(body.replace(b'accepted',b'discarded')+b'{"partial')
            with self.assertRaises(ValueError):self.validate(observer,copy)

    async def test_reader_issued_index_and_nonblocking_original_paths(self):
        from benchmarks.recording_native_validation import ValidatedRecordingIndex
        with self.assertRaises(TypeError):ValidatedRecordingIndex(None,None,None,None,None,None,None,None,None)
        observer,writer=await self.artifact()
        with tempfile.TemporaryDirectory(dir='.') as d:
            root=Path(d);copy=root/'copy';shutil.copytree(writer.output,copy)
            path=copy/'native-000000.jsonl';path.rename(copy/'original-owned-chunk.jsonl');os.mkfifo(path)
            code="from benchmarks.recording_native_validation import _read; import sys; _read(sys.argv[1],'native-000000.jsonl',1048576)"
            result=subprocess.run([sys.executable,'-c',code,str(copy)],capture_output=True,timeout=2)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'bounded regular artifact required',result.stderr)
            alias=root/'alias';alias.symlink_to(writer.output.absolute(),target_is_directory=True)
            with self.assertRaises(OSError):self.validate(observer,alias)
