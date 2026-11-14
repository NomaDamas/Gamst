"""Original collector subprocess + actual SDK + HTTP + aggregate fixture join."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from benchmarks.billing_transport_crosscheck import crosscheck_transport
from benchmarks.billing_crosscheck_bundle import persist_crosscheck
from benchmarks.billing_aggregate import encoded, sha
from benchmarks.native3_evidence import read_native3_evidence
from benchmarks.native3_files import LIMITS
from humansense.openai_responses import TRANSPORT
from humansense.openai_responses_codec import PROJECTION
from humansense.responses_evidence import ResponsesEvidenceSession
from test_instrumented_replay import setup, collect
from test_openai_responses import response

SCRIPT='''import asyncio,json,sys
from pathlib import Path
from benchmarks.runtime_channel import ChildChannel
from benchmarks.runtime_instrumentation import EvaluationSession
from benchmarks.perception_comparison import plan_digest
from humansense import TextContextCalls,VisualContextCalls
from humansense.openai_responses import OpenAIResponsesClient
from humansense.responses_evidence import ResponsesEvidenceSession
async def main():
    channel=await ChildChannel.from_environment()
    request=json.loads(sys.stdin.readline())
    assert plan_digest(request)==channel.bootstrap['request_sha256']
    observer=EvaluationSession.from_channel(channel)
    visual=request['input']['visual']
    if visual:from test_visual_context_call import Calls
    else:from test_text_context_call import Calls
    helper=Calls();await helper.asyncSetUp()
    sink=ResponsesEvidenceSession(Path(request['input']['evidence_root'])/('transport-'+request['arm']),
        owner_kind='task_native3',expected_owner=observer.binding_owner(),retain_response_bytes=True,retain_financial_hashes=True)
    client=OpenAIResponsesClient._loopback_fixture(port=request['input']['port'],api_key='fixture-key',
        model_revision='fixture-model',account_scope='fixture-account',project_id='fixture-project',evidence_session=sink,
        timeout=.2 if request['input']['mode']=='connect_failure' else 30)
    status='success'
    try:
        adapter=(VisualContextCalls if visual else TextContextCalls)(helper.h,'main',client,
            provider='openai',model_revision='fixture-model',evaluation_observer=observer)
        helper.adapters.append(adapter)
        queued=await helper.queued()
        args=(*queued,'fixture question')
        mode=request['input']['mode']
        if mode=='cancel_after_send':
            pending=asyncio.create_task(adapter.execute(*args,maximum_charge=5))
            marker=Path(request['input']['evidence_root'])/('received-'+str('ABC'.index(request['arm'])+1))
            async def received():
                while not marker.exists():await asyncio.sleep(.001)
            await asyncio.wait_for(received(),5)
            pending.cancel()
            try:await pending
            except asyncio.CancelledError:status='error'
        else:
            try:
                result=await adapter.execute(*args,maximum_charge=None if mode=='denied' else 5)
                assert result['status'] in ('completed','blocked')
            except (OSError,ValueError):
                assert mode=='connect_failure'
                status='error'
    finally:
        await helper.asyncTearDown();await client.close();await sink.close();await observer.close()
    print(json.dumps(dict(schema_version=1,status=status,answer='fixture',measurements={'request':request})),flush=True)
asyncio.run(main())
'''


class Crosscheck(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='.');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.calls=[];self.connections=set();self.hold_after_receive=False;self.port_override=None
        async def serve(reader,writer):
            task=asyncio.current_task();self.connections.add(task)
            try:
                header=await reader.readuntil(b'\r\n\r\n')
                size=int(next(line.split(b':',1)[1] for line in header.split(b'\r\n') if line.lower().startswith(b'content-length:')))
                self.calls.append(json.loads(await reader.readexactly(size)))
                if self.hold_after_receive:
                    (self.root/('received-'+str(len(self.calls)))).write_text('actual HTTP body received')
                    await reader.read()
                    return
                value=response();value['id']='resp_fixture_'+str(len(self.calls));raw=json.dumps(value).encode()
                writer.write(b'HTTP/1.1 200 OK\r\nX-Request-Id: req_fixture_'+str(len(self.calls)).encode()+
                    b'\r\nContent-Type: application/json\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw)
                await writer.drain()
            finally:
                writer.close();await writer.wait_closed();self.connections.discard(task)
        self.server=await asyncio.start_server(serve,'127.0.0.1',0)

    async def asyncTearDown(self):
        self.server.close();await self.server.wait_closed()
        for task in tuple(self.connections):task.cancel()
        if self.connections:await asyncio.gather(*tuple(self.connections),return_exceptions=True)

    async def collect(self,visual=False,mode='ok'):
        values=setup(self.root,mode=mode,script=SCRIPT,provider='openai',revision='fixture-model',timeout_ms=8000,
            extra_input=dict(evidence_root=str(self.root),port=self.port_override or self.server.sockets[0].getsockname()[1],visual=visual))
        output=self.root/'collector';await collect(values,output)
        files={name:dict(bytes=(output/name).stat().st_size,sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()) for name in LIMITS}
        self.index=read_native3_evidence(output,expected_files=files,sidecar_ordinal=0)
        self.selection=dict(owner_kind='task_native3',owner=self.index.snapshot()['owner'],provider='openai',
            model_revision='fixture-model',role='small_vlm' if visual else 'text_llm',projection_version=PROJECTION,transport_version=TRANSPORT)
        self.transport=self.root/'transport-A'
        now=int(time.time());page=dict(object='page',data=[dict(object='bucket',start_time=now-60,end_time=now,
            results=[dict(object='organization.costs.result',project_id='fixture-project',amount=dict(value=.1,currency='usd'))])],
            has_more=False,next_page=None)
        raw=json.dumps(page).encode();(self.root/'costs.json').write_bytes(raw)
        self.manifest=dict(schema_version=1,kind='openai_aggregate_acquisition',account_scope='fixture-account',
            origin='synthetic_fixture',acquired_at=now,queries=[dict(id='costs',endpoint='costs',start_time=now-60,end_time=now,
                bucket_width='1d',group_by=['project_id'],filters={},pages=[dict(request_cursor=None,path='costs.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])])

    def check(self):
        return crosscheck_transport(self.manifest,self.root,self.transport,
            expected_inventory_sha256=hashlib.sha256((self.transport/'inventory.json').read_bytes()).hexdigest(),
            native_index=self.index,expected_binding=self.selection)

    async def test_original_text_sdk_collector_and_actual_http_join_without_cost_claim(self):
        await self.collect();report=self.check()
        self.assertEqual(len(self.calls),3)
        self.assertTrue(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['native_entries_matched'],1)
        self.assertEqual(report['local_counts']['possible_sends'],1)
        self.assertEqual(report['entries'][0]['native_match'],'matched')
        self.assertFalse(report['count_bucket_verified']);self.assertFalse(report['request_costs_verified'])
        self.assertIsNone(report['entries'][0]['per_attempt_cost'])
        self.manifest['account_scope']='different-declared-account'
        changed=self.check()
        self.assertTrue(changed['local_binding_complete'])
        self.assertFalse(changed['account_isolation_verified'])
        self.assertEqual(changed['entries'][0]['query_relations'][0]['declared_account_relation'],'different')
        self.selection['owner']=dict(self.selection['owner'],arm='B')
        with self.assertRaisesRegex(ValueError,'selection owner differs'):self.check()

    async def test_original_visual_sdk_and_png_transport_join(self):
        try:import cv2;import numpy
        except ImportError:self.skipTest('optional vision dependencies absent')
        await self.collect(visual=True);report=self.check()
        self.assertTrue(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['native_entries_matched'],1)
        self.assertTrue(self.calls[0]['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,'))

    async def test_whole_lane_omission_is_not_zero_cost_or_complete(self):
        await self.collect()
        self.transport=self.root/'empty-original-session'
        sink=ResponsesEvidenceSession(self.transport,owner_kind='task_native3',expected_owner=self.selection['owner'],
            retain_response_bytes=True,retain_financial_hashes=True)
        await sink.close();report=self.check()
        self.assertFalse(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['preparations'],0)
        self.assertEqual(report['local_counts']['native_entries_without_transport'],1)

    async def test_original_denied_preparation_preserved_without_native_send(self):
        await self.collect(mode='denied');report=self.check()
        self.assertEqual(self.calls,[])
        self.assertEqual(report['local_counts']['preparations'],1)
        self.assertEqual(report['local_counts']['not_entered'],1)
        self.assertEqual(report['local_counts']['native_entries_matched'],0)
        self.assertEqual(report['local_counts']['possible_sends'],0)
        self.assertIsNone(report['entries'][0]['per_attempt_cost'])

    async def test_actual_native_entry_with_connection_failure_is_not_a_send(self):
        import socket
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1',0));self.port_override=reserved.getsockname()[1]
            await self.collect(mode='connect_failure')
        report=self.check()
        self.assertEqual(self.calls,[])
        self.assertEqual(report['local_counts']['native_entries_matched'],1)
        self.assertEqual(report['local_counts']['possible_sends'],0)
        self.assertEqual(report['local_counts']['error']+report['local_counts']['timeout'],1)
        self.assertTrue(report['local_binding_complete'])
        self.assertIsNone(report['entries'][0]['per_attempt_cost'])

    async def test_actual_cancel_after_server_received_body_preserves_unknown_liability(self):
        self.hold_after_receive=True
        await self.collect(mode='cancel_after_send');report=self.check()
        self.assertEqual(len(self.calls),3)
        self.assertEqual(report['local_counts']['native_entries_matched'],1)
        self.assertEqual(report['local_counts']['possible_sends'],1)
        self.assertEqual(report['local_counts']['responses_observed'],0)
        self.assertEqual(report['local_counts']['cancelled'],1)
        self.assertTrue(report['local_binding_complete'])
        self.assertIsNone(report['entries'][0]['per_attempt_cost'])

    async def test_crosscheck_bundle_reuse_rejects_jointly_rehashed_false_claim(self):
        await self.collect();output=self.root/'joined-evidence'
        def persist():
            return persist_crosscheck(self.manifest,self.root,self.transport,
                expected_inventory_sha256=sha((self.transport/'inventory.json').read_bytes()),
                native_index=self.index,expected_binding=self.selection,output_directory=output)
        originals={path:path.read_bytes() for path in self.transport.iterdir()}
        report=persist();self.assertEqual(persist(),report)
        self.assertEqual(output.stat().st_mode & 0o777,0o700)
        self.assertTrue(all(path.stat().st_mode & 0o777==0o600 for path in output.iterdir()))
        report['request_costs_verified']=True;payload=encoded(report)
        (output/'crosscheck.json').write_bytes(payload)
        completion=json.loads((output/'completion.json').read_bytes())
        completion['files']['crosscheck.json']=dict(bytes=len(payload),sha256=sha(payload))
        (output/'completion.json').write_bytes(encoded(completion))
        with self.assertRaises(ValueError):persist()
        self.assertEqual((output/'crosscheck.json').read_bytes(),payload)
        self.assertTrue(all(path.read_bytes()==data for path,data in originals.items()))

    async def test_offline_cli_reloads_original_collector_and_transport_files(self):
        import subprocess
        import sys
        await self.collect()
        inputs={'manifest':self.manifest,'selection':self.selection,'native-files':self.index.snapshot()['original_files']}
        for name,value in inputs.items():(self.root/(name+'.json')).write_bytes(encoded(value))
        command=[sys.executable,'-m','benchmarks.billing_crosscheck_bundle',
            '--manifest',str(self.root/'manifest.json'),'--assets',str(self.root),
            '--transport',str(self.transport),'--transport-inventory-sha256',sha((self.transport/'inventory.json').read_bytes()),
            '--native-root',str(self.root/'collector'),'--native-files',str(self.root/'native-files.json'),
            '--selection',str(self.root/'selection.json'),'--sidecar-ordinal','0','--output',str(self.root/'cli-crosscheck')]
        result=await asyncio.to_thread(subprocess.run,command,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('actual acceptance remains incomplete',result.stdout)
        self.assertEqual(json.loads((self.root/'cli-crosscheck'/'crosscheck.json').read_bytes()),self.check())

    async def test_mutated_caller_selection_cannot_hide_an_uncovered_original_lane(self):
        from unittest.mock import patch
        from benchmarks.billing_aggregate import build_report
        await self.collect()
        frozen_selection=encoded(self.selection)
        self.transport=self.root/'empty-for-mutation'
        sink=ResponsesEvidenceSession(self.transport,owner_kind='task_native3',expected_owner=self.selection['owner'],
            retain_response_bytes=True,retain_financial_hashes=True)
        await sink.close()
        def change_after_freeze(*args,**kwargs):
            self.selection['role']='strong_vlm'
            self.selection['owner']['arm']='B'
            return build_report(*args,**kwargs)
        with patch('benchmarks.billing_transport_crosscheck.build_report',side_effect=change_after_freeze):report=self.check()
        self.assertFalse(report['local_binding_complete'])
        self.assertEqual(report['local_counts']['native_entries_without_transport'],1)
        self.assertEqual(report['transport_selection_sha256'],sha(frozen_selection))
