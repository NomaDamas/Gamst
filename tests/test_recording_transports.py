"""Actual SDK/native4/owned socket transport tests; synthetic source and models."""
import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from benchmarks.recorded_source import digest, encoded
from benchmarks.recording_dependencies import distribution_descriptor, verify_dependencies
from benchmarks.recording_transports import (LocalChatCompletions, FixtureCompletion,
    FixtureReceiptRouter, TransportCapacity, TransportMemoryBudget, TransportOutcomeSink,
    _response, _validate_png)
from benchmarks.owned_capture import CapturePacket
from gamst import TextContextCalls, VisualContextCalls
from gamst.call_controller import _digest
from gamst.native_entry import NativeEntryBinding
from test_recording_sdk_scopes import NativeFixture
import test_text_context_call as text_tests
import test_visual_context_call as visual_tests
from gamst.schema import timestamp

import sysconfig
METADATA_ROOTS = (Path(sysconfig.get_paths()['purelib']),)
ROOTS = (Path(sysconfig.get_paths()['data']),)


HTTP_DEPS=('httpx','httpcore','h11','anyio','certifi','idna','sniffio')

def dependencies(names=HTTP_DEPS):
    entries=[]
    for name in names:
        descriptor=distribution_descriptor(name,allowed_roots=ROOTS,metadata_roots=METADATA_ROOTS)
        entries.append(dict(distribution=name,version=descriptor['version'],installation_sha256=digest(descriptor)))
    return verify_dependencies(entries,allowed_roots=ROOTS,metadata_roots=METADATA_ROOTS)


def success():
    return dict(id='server/request/one',object='chat.completion',created=1,model='local/model',
        choices=[dict(index=0,message=dict(role='assistant',content='owned answer'),finish_reason='stop')],
        usage=dict(prompt_tokens=3,completion_tokens=2,total_tokens=5),system_fingerprint=None)


class Sink(TransportOutcomeSink):
    def __init__(self):self.rows=[]
    def record(self,row):
        if len(self.rows)>=256:raise ValueError('test bounded outcome sink full')
        self.rows.append(deepcopy(row))


class Server:
    async def __aenter__(self):
        self.requests=[];self.tasks=set();self.arrived=asyncio.Event();self.release=asyncio.Event()
        self.block=False;self.body=encoded(success());self.status=200;self.headers={};self.partial=False
        self.server=await asyncio.start_server(self.handle,'127.0.0.1',0)
        self.endpoint='http://127.0.0.1:'+str(self.server.sockets[0].getsockname()[1])+'/v1/chat/completions'
        return self
    async def handle(self,reader,writer):
        task=asyncio.current_task();self.tasks.add(task)
        try:
            header=await reader.readuntil(b'\r\n\r\n')
            if len(header)>16384:raise ValueError('test server request head bound')
            fields={}
            for line in header.split(b'\r\n')[1:]:
                if b':' in line:
                    key,value=line.split(b':',1);fields[key.lower()]=value.strip()
            count=int(fields[b'content-length'])
            if not 0<=count<=16*1024**2:raise ValueError('test server request body bound')
            body=await reader.readexactly(count);self.requests.append((header,body));self.arrived.set()
            if self.block:await self.release.wait()
            headers=dict(self.headers)
            if 'Content-Length' not in headers:headers['Content-Length']=str(len(self.body))
            headers.setdefault('Connection','close')
            raw=('HTTP/1.1 '+str(self.status)+' Test\r\n'+''.join(k+': '+v+'\r\n' for k,v in headers.items())+'\r\n').encode()
            writer.write(raw+(self.body[:1] if self.partial else self.body));await writer.drain()
            if self.partial:await self.release.wait()
        except (ConnectionError,asyncio.IncompleteReadError):pass
        finally:
            writer.close();await writer.wait_closed();self.tasks.discard(task)
    async def __aexit__(self,*args):
        self.release.set();self.server.close();await self.server.wait_closed()
        for task in list(self.tasks):task.cancel()
        await asyncio.gather(*list(self.tasks),return_exceptions=True)


def real_adapter(server,inventory,**changes):
    config=dict(endpoint=server.endpoint,api_key_env=None,timeout_ms=1000,response_bytes=65536,
        temperature_millionths=0,retries=0,redirects=False,trust_env=False,
        server_revision_sha256='1'*64,pricing_identity_sha256=None)
    config.update(changes)
    return dict(adapter_id='http',implementation='local_chat_completions_v1',evidence_class='real',
        provider='test.client',model_id='local/model',model_revision='v1',
        implementation_sha256='2'*64,dependency_inventory_sha256=inventory.sha256,config=config)


def packet(frame):
    raw=frame.data.tobytes()
    return CapturePacket(source_manifest_sha256='1'*64,record_sha256='2'*64,record_ordinal=0,
        source_index=0,source_id=frame.sensor.source_id,sensor_id=frame.sensor.sensor_id,
        original_stream_epoch='original',capture_sequence=1,original_capture_ns=0,offset_ns=0,
        mapping_sha256='3'*64,scheduled_monotonic_ns=1,acquired_monotonic_ns=2,
        sdk_timestamp_ms=frame.timestamp_ms,submillisecond_remainder_ns=0,format='bgr8_hwc',
        shape=frame.data.shape,sample_rate=None,sample_count=None,payload_sha256=hashlib.sha256(raw).hexdigest(),payload=raw)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in HTTP_DEPS), 'optional HTTP provider dependencies absent')
class HTTPTests(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    @classmethod
    def setUpClass(cls):cls.inventory=dependencies()

    async def setup_client(self,server,*,visual=False,**changes):
        if visual and (importlib.util.find_spec('cv2') is None or importlib.util.find_spec('numpy') is None):
            self.skipTest('optional vision dependencies absent')
        inventory=dependencies(HTTP_DEPS+('numpy','opencv-python')) if visual else self.inventory
        spec=real_adapter(server,inventory,**changes)
        observer,writer,clock=await self.recording(role='small_vlm' if visual else 'text_llm',config=spec['config'])
        sink=Sink();memory=TransportMemoryBudget(64*1024**2)
        client=await LocalChatCompletions.open(spec,invocation_ordinal=0,lane_id='text',binding_index=0,
            context=observer.context,dependency_inventory=inventory,sink=sink,
            capacity=TransportCapacity(262144,65536,32*48*3) if visual else TransportCapacity(262144),memory=memory)
        self.addAsyncCleanup(client.close)
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,client,'callback')
        observer.bind_mapping(clock.mapping,clock.sha256)
        adapter=(VisualContextCalls if visual else TextContextCalls)(self.h,'main',client,
            provider='test.client',model_revision='v1',evaluation_observer=observer)
        self.adapters.append(adapter)
        scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)
        return adapter,client,sink,memory,observer,writer,scope

    async def test_actual_text_sdk_one_post_exact_hashes_and_body_id_only(self):
        async with Server() as server:
            server.headers={'X-Request-ID':'never-use-header'}
            adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server)
            rid,record=await self.queued();request,_=adapter._prepare(rid,record,'question')
            decision=next(r for r in self.c.snapshot()['requests'] if r['id']==rid)['decision']['decision_sha256']
            self.assertIsNone(client.entry_for_decision(decision))
            result=await adapter.execute(rid,record,'question',maximum_charge=1,evaluation_scope=scope)
            self.assertTrue(result['accepted']);self.assertEqual(len(server.requests),1)
            body=server.requests[0][1];wire=json.loads(body)
            self.assertEqual(wire,dict(model='local/model',messages=request['messages'],temperature=0,stream=False))
            detached=client.adapter;detached['config']['endpoint']='http://remote.invalid'
            self.assertEqual(client.adapter['config']['endpoint'],server.endpoint)
            row=sink.rows[0]
            self.assertEqual(row['prompt_sha256'],digest(request));self.assertEqual(row['wire_request_sha256'],hashlib.sha256(body).hexdigest())
            self.assertNotEqual(row['prompt_sha256'],row['wire_request_sha256'])
            self.assertEqual(row['provider_request_id'],'server/request/one');self.assertEqual(row['outcome'],'ok')
            self.assertEqual(result['usage'],dict(input_tokens=3,output_tokens=2,cached_input_tokens=0))
            self.assertIsNone(result['actual_cost']);self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],1)
            entry=client.entry_for_decision(decision);self.assertIs(type(entry),NativeEntryBinding)
            self.assertEqual(entry.snapshot()['owner']['scope_index'],1)
            client.release_entry(decision);self.assertIsNone(client.entry_for_decision(decision))
            self.assertGreater(memory.reserved_bytes,0)
            await adapter.close();await client.close();observer.backend_closed(0,client);await observer.close()
            self.assertEqual(memory.reserved_bytes,0)
            self.assertEqual((len(client._entries),len(client._captures),len(client._instances),len(client._prompts)),(0,0,0,0))
            events=self.events(writer);self.assertEqual(sum(e['event']=='entered' for e in events),1)

    async def test_actual_visual_sdk_png_wire_and_owned_packet_join(self):
        async with Server() as server:
            adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server,visual=True)
            rid,record,frame=await visual_tests.Calls.queued(self)
            decision=next(r for r in self.c.snapshot()['requests'] if r['id']==rid)['decision']['decision_sha256']
            client.register_capture(decision,packet(frame))
            result=await adapter.execute(rid,record,frame,'question',maximum_charge=1,evaluation_scope=scope)
            self.assertTrue(result['accepted']);wire=json.loads(server.requests[0][1]);content=wire['messages'][1]['content']
            self.assertEqual([v['type'] for v in content],['text','image_url'])
            image=base64.b64decode(content[1]['image_url']['url'].split(',',1)[1],validate=True)
            self.assertEqual(hashlib.sha256(image).hexdigest(),result['image_sha256'])
            self.assertEqual(content[1]['image_url']['detail'],'auto');self.assertEqual(sink.rows[0]['prompt_sha256'],result['prompt_sha256'])
            self.assertEqual(len(server.requests),1)
            await adapter.close();await client.close();observer.backend_closed(0,client)

    async def test_wrong_owned_pixels_reject_after_entry_without_post(self):
        async with Server() as server:
            adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server,visual=True)
            rid,record,frame=await visual_tests.Calls.queued(self)
            decision=next(r for r in self.c.snapshot()['requests'] if r['id']==rid)['decision']['decision_sha256']
            original=packet(frame);different=bytes([original.payload[0]^1])+original.payload[1:]
            client.register_capture(decision,replace(original,payload=different,payload_sha256=hashlib.sha256(different).hexdigest()))
            with self.assertRaisesRegex(ValueError,'owned original capture'):
                await adapter.execute(rid,record,frame,'q',maximum_charge=1,evaluation_scope=scope)
            self.assertEqual(server.requests,[]);self.assertEqual(sink.rows[0]['outcome'],'invalid_response')
            self.assertIsNotNone(client.entry_for_decision(decision));self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],1)
            await adapter.close();await client.close();observer.backend_closed(0,client)

    async def test_http_error_invalid_response_and_timeout_never_retry(self):
        cases=[('http',503,encoded(success()),{},RuntimeError,'http_error'),
            ('duplicate',200,b'{"model":"local/model","model":"local/model","choices":[]}',{},ValueError,'invalid_response'),
            ('model',200,encoded(dict(success(),model='other')),{},ValueError,'invalid_response'),
            ('compressed',200,encoded(success()),{'Content-Encoding':'gzip'},ValueError,'invalid_response'),
            ('oversize',200,b'x'*1025,{},ValueError,'invalid_response'),
            ('timeout',200,encoded(success()),{},asyncio.TimeoutError,'timeout')]
        for name,status,body,headers,error,outcome in cases:
            with self.subTest(name=name):
                async with Server() as server:
                    server.status=status;server.body=body;server.headers=headers;server.block=name=='timeout'
                    adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server,
                        response_bytes=1024,timeout_ms=500 if name=='timeout' else 1000)
                    rid,record=await self.queued()
                    with self.assertRaises(error):await adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope)
                    self.assertEqual(len(server.requests),1);self.assertEqual(sink.rows[0]['outcome'],outcome)
                    self.assertEqual(sink.rows[0]['attempts'],1)
                    self.c.settle(rid,components={'other':0},now_ms=timestamp())
                    await adapter.close();await client.close();observer.backend_closed(0,client)

    async def test_complete_oversized_and_excessive_headers_reject_before_parser(self):
        for fields in ({'x'+str(n):'a' for n in range(12000)}, {'x'+str(n):'a' for n in range(129)}):
            async with Server() as server:
                server.headers=fields
                adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server)
                rid,record=await self.queued()
                with self.assertRaisesRegex(ValueError,'headers|header fields|workspace'):
                    await adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope)
                self.assertEqual(len(server.requests),1);self.assertEqual(sink.rows[0]['outcome'],'invalid_response')
                self.assertIsNone(sink.rows[0]['response_body_sha256'])
                self.c.settle(rid,components={'other':0},now_ms=timestamp())
                await adapter.close();await client.close();observer.backend_closed(0,client)
                self.assertEqual(memory.reserved_bytes,0)

    async def test_cancelled_actual_http_owned_close_and_unknown_liability(self):
        async with Server() as server:
            server.block=True
            adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server)
            rid,record=await self.queued()
            task=asyncio.create_task(adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope))
            await asyncio.wait_for(server.arrived.wait(),1);task.cancel();task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertEqual(sink.rows[0]['outcome'],'cancelled');self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],1)
            await adapter.close();await client.close();observer.backend_closed(0,client)
            self.assertEqual(memory.reserved_bytes,0)

    async def test_repeated_cancel_joins_decoder_before_memory_release(self):
        started=threading.Event();release=threading.Event();finished=threading.Event()
        original=_validate_png
        def decode(*args):
            started.set()
            try:
                if not release.wait(5):raise TimeoutError('test barrier missing')
                return original(*args)
            finally:finished.set()
        async with Server() as server:
            adapter,client,sink,memory,observer,writer,scope=await self.setup_client(server,visual=True)
            rid,record,frame=await visual_tests.Calls.queued(self)
            decision=next(r for r in self.c.snapshot()['requests'] if r['id']==rid)['decision']['decision_sha256']
            client.register_capture(decision,packet(frame))
            with patch('benchmarks.recording_transports._validate_png',decode):
                task=asyncio.create_task(adapter.execute(rid,record,frame,'q',maximum_charge=1,evaluation_scope=scope))
                async def entered():
                    while not started.is_set():await asyncio.sleep(.001)
                await asyncio.wait_for(entered(),2)
                task.cancel();closing=asyncio.create_task(client.close());task.cancel()
                for _ in range(10):await asyncio.sleep(0)
                self.assertFalse(task.done());self.assertFalse(closing.done());self.assertGreater(memory.reserved_bytes,0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):await task
                await closing
            self.assertTrue(finished.is_set());self.assertEqual(memory.reserved_bytes,0);self.assertEqual(server.requests,[])
            await adapter.close();observer.backend_closed(0,client)

    async def test_workspace_refused_before_open_and_descriptor_detached(self):
        async with Server() as server:
            spec=real_adapter(server,self.inventory);observer,writer,clock=await self.recording(config=spec['config'])
            with self.assertRaisesRegex(ValueError,'workspace'):
                await LocalChatCompletions.open(spec,invocation_ordinal=0,lane_id='text',binding_index=0,
                    context=observer.context,dependency_inventory=self.inventory,sink=Sink(),
                    capacity=TransportCapacity(262144),memory=TransportMemoryBudget(1))
            self.assertEqual(server.requests,[])


class ValidationTests(unittest.TestCase):
    def test_exact_response_types_and_bounded_tree(self):
        good=success();self.assertEqual(_response(encoded(good),'local/model')[1],'server/request/one')
        wrong=[dict(good,id=[]),dict(good,created=True),dict(good,object='chat.completion.chunk'),
            dict(good,system_fingerprint={'x':1}),dict(good,extra=1),
            dict(good,usage=dict(prompt_tokens=3,completion_tokens=2,total_tokens=4)),
            dict(good,choices=[dict(index=0,message=dict(role='assistant',content='a'*8193),finish_reason='stop')]),
            dict(good,choices=[dict(index=0,message=dict(role='assistant',content='ok'),finish_reason='length')])]
        for value in wrong:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):_response(encoded(value),'local/model')
        for raw in (b'{"x":NaN}',b'{"x":1234567890123456789012345}',b'['*20+b'0'+b']'*20,b'\xff'):
            with self.assertRaises(ValueError):_response(raw,'local/model')

    def test_large_rejected_json_stops_before_object_conversion(self):
        raw=b'['+b'"ab",'*200000+b'"ab"]'
        with patch('benchmarks.recording_transports.json.loads',side_effect=AssertionError('unbounded object conversion')):
            with self.assertRaisesRegex(ValueError,'node bound'):_response(raw,'local/model')

    @unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('numpy'), 'optional vision dependencies absent')
    def test_png_wrong_valid_source_digest_and_mismatched_pixels(self):
        import numpy as np
        from gamst.smolvlm_call import smolvlm_frame_sha256
        pixels=np.arange(32*48*3,dtype=np.uint8).reshape(32,48,3)
        request=VisualContextCalls._encode(dict(_pixels=pixels,_capture_ms=100,_source_sha=smolvlm_frame_sha256(pixels)))
        image=request['images'][0];capacity=TransportCapacity(262144,65536,pixels.nbytes)
        capture=dict(pixels_sha256=image['source_frame_sha256'],capture_ms=100,shape=list(pixels.shape))
        _validate_png(image,capacity,capture)
        wrong=dict(image,source_frame_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'source pixel digest'):_validate_png(wrong,capacity,capture)
        changed=pixels.copy();changed[0,0,0]^=1
        other=VisualContextCalls._encode(dict(_pixels=changed,_capture_ms=100,_source_sha=image['source_frame_sha256']))['images'][0]
        with self.assertRaisesRegex(ValueError,'source pixel digest'):_validate_png(other,capacity,capture)


    @unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('numpy'), 'optional vision dependencies absent')
    def test_png_ancillary_and_inflation_rejected_before_native_decoder(self):
        import struct,zlib,numpy as np
        from gamst.smolvlm_call import smolvlm_frame_sha256
        pixels=np.zeros((32,48,3),dtype=np.uint8)
        image=VisualContextCalls._encode(dict(_pixels=pixels,_capture_ms=100,_source_sha=smolvlm_frame_sha256(pixels)))['images'][0]
        original=base64.b64decode(image['data_base64'])
        def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
        payloads=[original[:33]+chunk(b'iCCP',b'not-an-admitted-profile')+original[33:],
            original[:33]+chunk(b'IDAT',zlib.compress(b'x'*100000))+chunk(b'IEND',b'')]
        for png in payloads:
            bad=dict(image,data_base64=base64.b64encode(png).decode(),sha256=hashlib.sha256(png).hexdigest())
            with patch('cv2.imdecode',side_effect=AssertionError('native decoder before admission')):
                with self.assertRaises(ValueError):_validate_png(bad,TransportCapacity(262144,65536,pixels.nbytes),None)


class FixtureTests(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    async def test_fixture_actual_sdk_exact_ordinal_noentry_release_and_unknown(self):
        rows=dict(schema_version=1,kind='fixture_completions',rows=[dict(invocation_ordinal=0,lane_id='text',
            call_ordinal=0,text='fixture answer',usage=None,synthetic_receipt_sha256=None)])
        config=dict(responses_sha256=digest(rows),delay_ms=0,settlement='unknown')
        spec=dict(adapter_id='fixture',implementation='fixture_completion_v1',evidence_class='fixture',
            provider='test.client',model_id='fixture/model',model_revision='v1',implementation_sha256='2'*64,
            dependency_inventory_sha256='3'*64,config=config)
        observer,writer,clock=await self.recording(config=config)
        memory=TransportMemoryBudget(32*1024**2)
        client=await FixtureCompletion.open(spec,rows,invocation_ordinal=0,lane_id='text',binding_index=0,
            context=observer.context,capacity=TransportCapacity(262144),memory=memory)
        self.addAsyncCleanup(client.close)
        self.assertEqual(json.loads(client._context), dict(observer.context))
        with self.assertRaises(TypeError): observer.context['nonce']='b'*64
        for context in (dict(observer.context,extra=True), dict(observer.context,nonce='b'*4097),
                        dict(observer.context,invocation_ordinal=1)):
            with self.subTest(context_keys=tuple(context)),self.assertRaises(ValueError):
                await FixtureCompletion.open(spec,rows,invocation_ordinal=0,lane_id='text',binding_index=0,
                    context=context,capacity=TransportCapacity(262144),memory=memory)
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,client,'callback');observer.bind_mapping(clock.mapping,clock.sha256)
        adapter=TextContextCalls(self.h,'main',client,provider='test.client',model_revision='v1',evaluation_observer=observer)
        self.adapters.append(adapter);scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)
        rid,record=await self.queued();request,_=adapter._prepare(rid,record,'q')
        prepared=client.prepare(request);await prepared.release()
        self.assertEqual((await adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope))['text'],'fixture answer')
        rid,record=await self.queued()
        with self.assertRaisesRegex(ValueError,'exhausted'):await adapter.execute(rid,record,'q',maximum_charge=1,evaluation_scope=scope)
        self.assertEqual(client._call_ordinal,2);self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],2)
        await adapter.close();await client.close();observer.backend_closed(0,client);self.assertEqual(memory.reserved_bytes,0)


    # (FixtureTests methods below exercise only synthetic routing, no receipts.)
