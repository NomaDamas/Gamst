"""Real loopback HTTP transport. All provider identities/results are synthetic."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from humansense.native_entry import _issue_entry_binding
from humansense.openai_responses import OpenAIResponsesClient
from humansense.openai_responses_codec import normalize,strict_json
from humansense.responses_evidence import ResponsesEvidenceSession,RESERVATION
from test_native_entry import owner,binding


def request():return dict(provider='openai',model_revision='fixture-model',messages=[dict(role='system',content='fixture instructions'),dict(role='user',content='fixture question')])

def response():return dict(id='resp_fixture_one',object='response',model='fixture-model',status='completed',created_at=1700000000,
    output=[dict(type='message',role='assistant',status='completed',content=[dict(type='output_text',text='fixture answer')])],
    usage=dict(input_tokens=10,output_tokens=2,input_tokens_details=dict(cached_tokens=3,cache_write_tokens=4),total_tokens=12))


class Transport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='.');self.sink=ResponsesEvidenceSession(Path(self.tmp.name)/'evidence',owner_kind='task_native3',expected_owner=owner(),retain_response_bytes=True,retain_financial_hashes=True)
        self.calls=[];self.connections=set();self.entered=asyncio.Event();self.hold=None;self.reply=response();self.status=200;self.raw_override=None;self.chunked=False
        async def serve(reader,writer):
            task=asyncio.current_task();self.connections.add(task)
            try:
                head=await reader.readuntil(b'\r\n\r\n');size=int(next(x.split(b':',1)[1] for x in head.split(b'\r\n') if x.lower().startswith(b'content-length:')))
                body=await reader.readexactly(size);self.calls.append((head,body));self.entered.set()
                if self.hold is not None:await self.hold.wait()
                raw=self.raw_override if self.raw_override is not None else json.dumps(self.reply).encode()
                framing=b'Transfer-Encoding: chunked' if self.chunked else f'Content-Length: {len(raw)}'.encode()
                writer.write(f'HTTP/1.1 {self.status} fixture\r\n'.encode()+b'X-Request-Id: req_fixture_one\r\nContent-Type: application/json\r\n'+framing+b'\r\n\r\n')
                if self.chunked:writer.write(f'{len(raw):x}\r\n'.encode()+raw+b'\r\n0\r\n\r\n')
                else:writer.write(raw)
                await writer.drain()
            except (asyncio.CancelledError,ConnectionError):pass
            finally:writer.close();await writer.wait_closed();self.connections.discard(task)
        self.server=await asyncio.start_server(serve,'127.0.0.1',0)
        self.client=OpenAIResponsesClient._loopback_fixture(port=self.server.sockets[0].getsockname()[1],api_key='fixture-key',model_revision='fixture-model',account_scope='fixture-account',project_id='fixture-project',evidence_session=self.sink)
    async def asyncTearDown(self):
        await self.client.close();await self.sink.close();self.server.close();await self.server.wait_closed()
        for task in tuple(self.connections):task.cancel()
        if self.connections:await asyncio.gather(*tuple(self.connections),return_exceptions=True)
        self.tmp.cleanup()
    def prepare(self):
        prepared=self.client.prepare(request());return prepared,_issue_entry_binding(binding(prepared.sdk_hash))
    def manifest(self):return json.loads((self.sink.directory/'call-0000.json').read_bytes())

    async def test_actual_wire_single_send_raw_usage_and_ids(self):
        prepared,entry=self.prepare();result=await prepared.invoke(entry);await prepared.release()
        self.assertEqual(result,dict(text='fixture answer',usage=dict(input_tokens=10,output_tokens=2,cached_input_tokens=3)))
        head,body=self.calls[0];wire=json.loads(body)
        self.assertEqual(wire['model'],'fixture-model');self.assertEqual(wire['input'][0]['content'][0]['text'],'fixture question')
        self.assertEqual((wire['stream'],wire['background'],wire['store']),(False,False,False))
        m=self.manifest();self.assertEqual(m['wire_sha256'],hashlib.sha256(body).hexdigest());self.assertEqual(m['entry_binding'],entry.snapshot())
        self.assertNotEqual(m['sdk_request_sha256'],m['wire_sha256']);self.assertEqual(m['server_request_id'],'req_fixture_one')
        self.assertEqual(m['response_id'],'resp_fixture_one');self.assertFalse(m['count_bucket_verified']);self.assertIsNone(m['actual_cost'])
        original=json.loads((self.sink.directory/m['raw_response']['path']).read_bytes())
        self.assertEqual(original['usage']['input_tokens_details']['cache_write_tokens'],4)
        self.assertNotIn('fixture-key',json.dumps(m));self.assertEqual(len(self.calls),1)
        await self.sink.close()
        with self.assertRaises(RuntimeError):self.client.prepare(request())

    async def test_chunked_real_wire(self):
        self.chunked=True;p,b=self.prepare();await p.invoke(b);await p.release();self.assertTrue(self.manifest()['raw_response']['complete'])

    async def test_model_mismatch_is_accounted_failure_with_original_bytes(self):
        self.reply['model']='fixture-wrong';p,b=self.prepare()
        with self.assertRaises(ValueError):await p.invoke(b)
        await p.release();m=self.manifest();self.assertEqual(m['returned_model'],'fixture-wrong');self.assertEqual(m['terminal'],'error')
        self.assertEqual(len(self.calls),1);self.assertIsNone(m['actual_cost'])

    async def test_http_error_keeps_request_id_and_body(self):
        self.status=429;self.reply={'error':{'message':'fixture refusal'}};p,b=self.prepare()
        with self.assertRaises(ValueError):await p.invoke(b)
        await p.release();m=self.manifest();self.assertEqual(m['server_request_id'],'req_fixture_one');self.assertTrue(m['raw_response']['complete'])

    async def test_oversize_and_malformed_remain_incomplete_or_failed(self):
        self.raw_override=b'x'*(1048576+1);p,b=self.prepare()
        with self.assertRaises(ValueError):await p.invoke(b)
        await p.release();m=self.manifest();self.assertTrue(m['raw_response']['truncated']);self.assertFalse(m['raw_response']['complete'])
        self.assertEqual(m['raw_response']['bytes'],1048576)

    async def test_cancel_after_actual_send_and_concurrent_close_join(self):
        self.hold=asyncio.Event();p,b=self.prepare();task=asyncio.create_task(p.invoke(b));await self.entered.wait()
        await asyncio.gather(self.client.close(),self.client.close())
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(p._task.done());self.assertIsNone(p.writer);self.assertEqual(len(self.calls),1)
        m=self.manifest();self.assertEqual(m['terminal'],'cancelled');self.assertIsNone(m['server_request_id'])
        self.assertIn('possible_send',[x['event'] for x in m['milestones']])

    async def test_session_closes_unused_lease_and_prevents_late_send(self):
        p,b=self.prepare();inventory=await self.sink.close()
        with self.assertRaises(RuntimeError):await p.invoke(b)
        self.assertEqual(self.calls,[]);self.assertEqual(len(inventory['entries']),1);self.assertEqual(self.manifest()['terminal'],'not_entered')

    async def test_binding_mismatch_never_sends(self):
        p,b=self.prepare();row=b.snapshot();row['owner']['arm']='A'
        with self.assertRaises(ValueError):await p.invoke(_issue_entry_binding(row))
        await p.release();self.assertEqual(self.calls,[])

    async def test_sink_capacity_failure_prevents_send(self):
        self.sink._max=1024
        with self.assertRaises(RuntimeError):self.client.prepare(request())
        self.assertEqual(self.calls,[])

    async def test_artifact_failure_after_send_marks_incomplete(self):
        p,b=self.prepare()
        with patch.object(self.sink,'_write',side_effect=OSError('fixture disk failure')):
            with self.assertRaises(OSError):await p.invoke(b)
        await p.release();self.assertTrue(self.sink._incomplete);self.assertEqual(len(self.calls),1)
        with self.assertRaises(RuntimeError):self.client.prepare(request())

    async def test_reader_verifies_exact_originals_without_certifying_native_money(self):
        from humansense.responses_evidence_reader import read_responses_evidence
        p,b=self.prepare();await p.invoke(b);await p.release();await self.sink.close()
        inventory=(self.sink.directory/'inventory.json').read_bytes();sha=hashlib.sha256(inventory).hexdigest()
        evidence=read_responses_evidence(self.sink.directory,expected_inventory_sha256=sha)
        row=next(evidence.entries());self.assertEqual(json.loads(row['raw_response'])['id'],'resp_fixture_one')
        self.assertEqual(row['ordinal'],0)
        self.assertEqual(row['original_manifest_sha256'],hashlib.sha256((self.sink.directory/'call-0000.json').read_bytes()).hexdigest())
        self.assertEqual(evidence.owner_declaration()['entry_count'],1)
        detached=evidence.owner_declaration();detached['owner_kind']='changed'
        self.assertEqual(evidence.owner_declaration()['owner_kind'],'task_native3')
        self.assertFalse(row['native_binding_verified']);self.assertFalse(row['receipt_authenticity_verified'])
        (self.sink.directory/'raw-0000.bin').write_bytes(b'{}')
        with self.assertRaises(ValueError):read_responses_evidence(self.sink.directory,expected_inventory_sha256=sha)

    async def test_reader_rejects_impossible_states_and_allows_late_cancel(self):
        from humansense.responses_evidence_reader import read_responses_evidence
        p,b=self.prepare();await p.invoke(b);await p.release();await self.sink.close()
        directory=self.sink.directory
        original=json.loads((directory/'call-0000.json').read_bytes())
        inventory=json.loads((directory/'inventory.json').read_bytes())
        def read_changed(change):
            m=json.loads(json.dumps(original));change(m)
            raw=json.dumps(m).encode();(directory/'call-0000.json').write_bytes(raw)
            inventory['entries'][0]['manifest_sha256']=hashlib.sha256(raw).hexdigest()
            inventory['entries'][0]['manifest_bytes']=len(raw)
            raw=json.dumps(inventory).encode();(directory/'inventory.json').write_bytes(raw)
            return read_responses_evidence(directory,expected_inventory_sha256=hashlib.sha256(raw).hexdigest())
        invalid=[
            lambda m:m.update(milestones=[x for x in m['milestones'] if x['event'] not in ('possible_send','headers')]),
            lambda m:m.update(terminal='not_entered'),
            lambda m:m.update(failure_category='error'),
            lambda m:m.update(terminal='error',failure_category='error',normalized_usage={'input_tokens':-1}),
            lambda m:m.update(terminal='cancelled',failure_category='cancelled',consumption_basis='unknown'),
            lambda m:m.update(created_at=True),
        ]
        for change in invalid:
            with self.assertRaises(ValueError):read_changed(change)
        evidence=read_changed(lambda m:m.update(terminal='cancelled',failure_category='cancelled'))
        self.assertEqual(next(evidence.entries())['manifest']['normalized_usage']['input_tokens'],10)

    async def test_explicit_retention_and_symlink_boundaries(self):
        with self.assertRaises(PermissionError):ResponsesEvidenceSession(Path(self.tmp.name)/'no-consent',owner_kind='task_native3',expected_owner=owner())
        self.assertFalse((Path(self.tmp.name)/'no-consent').exists())
        target=Path(self.tmp.name)/'linked';target.symlink_to(self.sink.directory.resolve(),target_is_directory=True)
        with self.assertRaises(OSError):ResponsesEvidenceSession(target/'child',owner_kind='task_native3',expected_owner=owner(),retain_response_bytes=True,retain_financial_hashes=True)

    async def test_repeated_cancel_of_session_close_joins_live_send(self):
        self.hold=asyncio.Event();p,b=self.prepare();task=asyncio.create_task(p.invoke(b));await self.entered.wait()
        close=asyncio.create_task(self.sink.close());await asyncio.sleep(0);close.cancel();close.cancel()
        try:await close
        except asyncio.CancelledError:pass
        with self.assertRaises(asyncio.CancelledError):await task
        result=await self.sink.close();self.assertFalse(self.sink._leases);self.assertIsNone(p.writer)
        self.assertEqual(len(result['entries']),1)


    async def test_shared_deadline_after_actual_send_owns_socket_cleanup(self):
        self.hold=asyncio.Event();p,b=self.prepare();real_wait=asyncio.wait
        async def controlled_wait(tasks,**kwargs):
            values=tuple(tasks)
            if len(values)==1 and values[0].get_coro().__name__=='_exchange':
                await self.entered.wait()
                return set(),set(values)
            return await real_wait(tasks,**kwargs)
        with patch('humansense.openai_responses.asyncio.wait',side_effect=controlled_wait):
            with self.assertRaises(TimeoutError):await p.invoke(b)
        await p.release();self.assertEqual(len(self.calls),1);self.assertIsNone(p.writer)
        self.assertEqual(self.manifest()['terminal'],'timeout')


    async def test_cancel_before_owned_transport_task_first_step_preserves_native_entry(self):
        p,b=self.prepare();real_create=asyncio.create_task
        def cancelled_before_start(coro,*args,**kwargs):
            task=real_create(coro,*args,**kwargs)
            if coro.__name__=='_run':task.cancel()
            return task
        with patch('humansense.openai_responses.asyncio.create_task',side_effect=cancelled_before_start):
            with self.assertRaises(asyncio.CancelledError):await p.invoke(b)
        await p.release();m=self.manifest()
        self.assertEqual(m['terminal'],'cancelled');self.assertIsNotNone(m['entry_binding'])
        self.assertNotIn('possible_send',[x['event'] for x in m['milestones']]);self.assertEqual(self.calls,[])


    async def test_session_closes_all_leases_when_one_unused_artifact_fails(self):
        self.hold=asyncio.Event();p,b=self.prepare();active=asyncio.create_task(p.invoke(b));await self.entered.wait()
        other=OpenAIResponsesClient._loopback_fixture(port=self.server.sockets[0].getsockname()[1],api_key='fixture-other-key',
            model_revision='fixture-model',account_scope='fixture-account',project_id='fixture-project',evidence_session=self.sink)
        unused=other.prepare(request());real_write=self.sink._write
        def fail_one(name,raw):
            if name=='call-0001.json':raise OSError('fixture manifest failure')
            return real_write(name,raw)
        with patch.object(self.sink,'_write',side_effect=fail_one):result=await self.sink.close()
        with self.assertRaises(asyncio.CancelledError):await active
        self.assertTrue(result['incomplete']);self.assertFalse(self.sink._leases)
        self.assertTrue(p._task.done());self.assertIsNone(p.writer);self.assertIsNone(other._lease)
        await other.close();self.assertIsNone(other._key)



class Projection(unittest.TestCase):
    def test_projection_owns_request_before_hash_and_wire(self):
        from humansense import openai_responses_codec as codec
        caller=request();original=json.loads(json.dumps(caller));encode=codec.encoded
        def mutate(value):
            result=encode(value)
            caller['messages'][1]['content']='mutated after original hash'
            return result
        with patch.object(codec,'encoded',side_effect=mutate):
            wire,sha=codec.prepare_body(caller,'openai','fixture-model',16)
        self.assertEqual(sha,hashlib.sha256(encode(original)).hexdigest())
        self.assertEqual(json.loads(wire)['input'][0]['content'][0]['text'],original['messages'][1]['content'])
        caller=request();caller['messages'][0]['content']=['unexpected']
        with patch.object(codec,'encoded',side_effect=AssertionError('must preflight before encoding')):
            with self.assertRaises(ValueError):codec.prepare_body(caller,'openai','fixture-model',16)

    def test_json_counts_primitive_nodes_before_decode(self):
        for scalar in (b'0',b'true',b'null'):
            raw=b'['+b','.join([scalar]*32768)+b']'
            with patch('humansense.openai_responses_codec.json.loads',side_effect=AssertionError('must bound before allocation')):
                with self.assertRaises(ValueError):strict_json(raw)
        self.assertEqual(strict_json(b'[1,true,null,"a,0"]'),[1,True,None,'a,0'])

    def test_missing_details_are_unknown_not_cached_zero(self):
        value=response();value['usage'].pop('input_tokens_details');self.assertIsNone(normalize(value,'fixture-model')['usage'])
        value['usage']=None;self.assertIsNone(normalize(value,'fixture-model')['usage'])
    def test_json_duplicate_nonfinite_and_depth(self):
        for raw in (b'{"id":1,"id":2}',b'{"x":NaN}',b'['*33+b'0'+b']'*33,b'{"x":1e999}'):
            with self.assertRaises(ValueError):strict_json(raw)

class OriginalEntry(unittest.IsolatedAsyncioTestCase):
    async def test_native3_bound_callback_has_one_original_attempt_and_owner(self):
        from test_runtime_instrumentation import session
        from test_text_context_call import Calls
        from humansense.text_context_call import TextContextCalls
        from humansense.call_controller import _digest
        from humansense.schema import timestamp
        harness=Calls();await harness.asyncSetUp()
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            observer=session(Path(tmp).resolve(),provider='openai',revision='fixture-model')
            async def unused(request):raise AssertionError('request-only callback must not run')
            adapter=TextContextCalls(harness.h,'main',unused,provider='openai',model_revision='fixture-model')
            rid,record=await harness.queued();request,_=adapter._prepare(rid,record,'fixture question')
            ticket=observer.reserve_call(harness.c,rid,provider='openai',model_revision='fixture-model',prompt_sha256=_digest(request))
            harness.c.reserve(rid,provider='openai',model_revision='fixture-model',prompt_sha256=_digest(request),maximum_charge=1,now_ms=timestamp())
            harness.c.dispatch(rid,now_ms=timestamp());ticket.dispatched();seen=[]
            async def callback(entry):seen.append(entry.snapshot());return dict(text='fixture result',usage=None)
            await ticket.invoke_bound(callback);ticket.finish('completed');report=await observer.close()
            attempt=report['reports'][0]['attempts'][0]
            self.assertEqual(len(report['reports'][0]['attempts']),1)
            self.assertEqual(seen[0]['entry']['attempt_id'],attempt['attempt_id'])
            self.assertEqual(seen[0]['owner'],observer.binding_owner())
            self.assertEqual(seen[0]['prompt_sha256'],_digest(request))
            await adapter.close()
        await harness.asyncTearDown()

class SDKTransport(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=Transport.asyncSetUp
    manifest=Transport.manifest
    async def asyncTearDown(self):
        if hasattr(self,'adapter'):
            await self.adapter.close();await self.harness.asyncTearDown();await self.observer.close()
        await Transport.asyncTearDown(self)

    async def sdk(self,visual=False,journal=None):
        from test_runtime_instrumentation import session
        from test_text_context_call import Calls
        from humansense.text_context_call import TextContextCalls
        from humansense.visual_context_call import VisualContextCalls
        self.harness=Calls();await self.harness.asyncSetUp()
        self.observer=session(Path(self.tmp.name).resolve(),provider='openai',revision='fixture-model',journal=journal)
        await self.client.close();await self.sink.close()
        self.sink=ResponsesEvidenceSession(Path(self.tmp.name)/'sdk-evidence',owner_kind='task_native3',expected_owner=self.observer.binding_owner(),retain_response_bytes=True,retain_financial_hashes=True)
        self.client=OpenAIResponsesClient._loopback_fixture(port=self.server.sockets[0].getsockname()[1],api_key='fixture-key',model_revision='fixture-model',account_scope='fixture-account',project_id='fixture-project',evidence_session=self.sink)
        self.adapter=(VisualContextCalls if visual else TextContextCalls)(self.harness.h,'main',self.client,provider='openai',model_revision='fixture-model',evaluation_observer=self.observer)
        if visual:
            from test_visual_context_call import Calls as Visual
            return await Visual.queued(self.harness)
        return await self.harness.queued()

    async def test_actual_text_sdk_native_entry_and_http_identity(self):
        rid,record=await self.sdk()
        result=await self.adapter.execute(rid,record,'fixture SDK question',maximum_charge=5)
        self.assertTrue(result['accepted']);self.assertEqual(result['text'],'fixture answer')
        self.assertIsNone(result['actual_cost']);self.assertFalse(result['usage_verified'])
        self.assertEqual(len(self.calls),1)
        report=await self.observer.close();m=self.manifest();attempts=report['reports'][0]['attempts']
        self.assertEqual(len(attempts),1);self.assertEqual(m['entry_binding']['entry']['attempt_id'],attempts[0]['attempt_id'])
        self.assertEqual(m['sdk_request_sha256'],result['prompt_sha256']);self.assertEqual(m['entry_binding']['entry']['execution_sha256'],attempts[0]['execution_sha256'])
        self.assertEqual(self.harness.c.snapshot()['budget']['reserved_liability'],5)
        self.assertNotIn('fixture SDK question',json.dumps(report))
        wire=json.loads(self.calls[0][1]);self.assertEqual(json.loads(wire['input'][0]['content'][0]['text'])['observation_record'],record)

    async def test_actual_visual_sdk_png_unchanged_and_one_native_call(self):
        try:import cv2;import numpy as np
        except ImportError:self.skipTest('optional vision unavailable')
        import base64
        rid,record,frame=await self.sdk(visual=True)
        result=await self.adapter.execute(rid,record,frame,'fixture visual question',maximum_charge=5)
        wire=json.loads(self.calls[0][1]);uri=wire['input'][0]['content'][1]['image_url']
        png=base64.b64decode(uri.split(',',1)[1],validate=True)
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR),frame.data)
        self.assertTrue(result['accepted']);self.assertEqual(len(self.calls),1)
        self.assertEqual(self.manifest()['sdk_request_sha256'],result['prompt_sha256'])
        report=await self.observer.close();self.assertEqual(report['provider_counters'][0]['count'],1)

    async def test_unknown_quote_releases_unused_evidence_with_no_native_or_http(self):
        rid,record=await self.sdk()
        result=await self.adapter.execute(rid,record,'fixture question',maximum_charge=None)
        self.assertEqual(result['status'],'blocked');self.assertEqual(self.calls,[])
        self.assertEqual(self.manifest()['terminal'],'not_entered');self.assertIsNone(self.client._lease)
        report=await self.observer.close();self.assertEqual(report['reports'][0]['attempts'],[])

    async def test_sink_full_refuses_before_controller_dispatch(self):
        rid,record=await self.sdk();self.sink._max=1024
        with self.assertRaises(RuntimeError):await self.adapter.execute(rid,record,'fixture question',maximum_charge=5)
        self.assertEqual(self.calls,[]);self.assertEqual(self.harness.c.snapshot()['budget']['dispatched'],0)
        report=await self.observer.close();self.assertEqual(report['reports'][0]['attempts'],[])

    async def test_cancel_after_dispatch_before_entry_has_no_http_or_native(self):
        class Cancel:
            def emit(self,event):
                if event['event']=='controller_dispatched':raise asyncio.CancelledError()
        rid,record=await self.sdk(journal=Cancel())
        with self.assertRaises(asyncio.CancelledError):await self.adapter.execute(rid,record,'fixture question',maximum_charge=5)
        self.assertEqual(self.calls,[]);self.assertEqual(self.manifest()['terminal'],'not_entered')
        report=await self.observer.close();self.assertEqual(report['reports'][0]['attempts'],[])
        self.assertEqual(self.harness.c.snapshot()['budget']['reserved_liability'],5)

    async def test_sensory_delete_discards_late_result_but_preserves_external_financial_evidence(self):
        rid,record=await self.sdk();self.hold=asyncio.Event()
        task=asyncio.create_task(self.adapter.execute(rid,record,'fixture private question',maximum_charge=5))
        await self.entered.wait();await self.harness.h.delete_data();self.hold.set()
        result=await task;self.assertFalse(result['accepted'])
        self.assertEqual(self.harness.c.snapshot()['budget']['reserved_liability'],5)
        self.assertEqual(self.observer.snapshot()['private_subject_mappings'],0)
        self.assertIsNotNone(self.manifest()['entry_binding']);self.assertEqual(len(self.calls),1)
        self.assertIsNone(self.client._lease)
