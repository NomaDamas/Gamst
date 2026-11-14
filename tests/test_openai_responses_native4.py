"""Actual native4 SDK -> loopback HTTP -> original readers; synthetic inputs only."""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import unittest

import test_openai_responses as transport_tests
from recording_native_fixtures import session
from humansense.openai_responses import OpenAIResponsesClient
from humansense.responses_evidence import ResponsesEvidenceSession
from humansense.responses_evidence_reader import read_responses_evidence
from benchmarks.recording_native_validation import validate_recording_native_inventory,lookup_canonical_operation


class Native4Transport(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=transport_tests.Transport.asyncSetUp

    async def asyncTearDown(self):
        if hasattr(self,'adapter'):
            await self.adapter.close()
            await self.client.close()
            if not getattr(self,'_backend_detached',False):
                self.observer.backend_closed(0,self.client);self._backend_detached=True
            await self.observer.close()
            await self.harness.asyncTearDown()
        await transport_tests.Transport.asyncTearDown(self)

    async def sdk(self,visual=False):
        from test_text_context_call import Calls
        from humansense.text_context_call import TextContextCalls
        from humansense.visual_context_call import VisualContextCalls
        self.harness=Calls();await self.harness.asyncSetUp()
        self.observer,self.writer,clock=await session(Path(self.tmp.name).resolve(),
            provider='openai',revision='fixture-model',role='small_vlm' if visual else 'text_llm')
        self.scope=self.observer.task_scope(self.observer.declarations['tasks'][0]['task_id'],binding_index=0)
        await self.client.close();await self.sink.close()
        self.sink=ResponsesEvidenceSession(Path(self.tmp.name)/'native4-http',owner_kind='recording_native4',
            expected_owner=self.scope.binding_owner(),retain_response_bytes=True,retain_financial_hashes=True)
        self.client=OpenAIResponsesClient._loopback_fixture(port=self.server.sockets[0].getsockname()[1],
            api_key='fixture-key',model_revision='fixture-model',account_scope='fixture-account',
            project_id='fixture-project',evidence_session=self.sink)
        self.observer.bind_semantic_lane(self.harness.c,0)
        self.observer.bind_backend(0,self.client,'callback')
        self.observer.bind_mapping(clock.mapping,clock.sha256)
        self.adapter=(VisualContextCalls if visual else TextContextCalls)(self.harness.h,'main',self.client,
            provider='openai',model_revision='fixture-model',evaluation_observer=self.observer)
        if visual:
            from test_visual_context_call import Calls as Visual
            return await Visual.queued(self.harness)
        return await self.harness.queued()

    async def originals(self):
        await self.adapter.close();await self.client.close()
        self.observer.backend_closed(0,self.client);self._backend_detached=True
        await self.observer.close();await self.sink.close()
        native=validate_recording_native_inventory(self.observer.declarations,self.writer.output,
            source_inventory=self.observer.declarations['source_manifest'])
        data=(self.sink.directory/'inventory.json').read_bytes()
        transport=read_responses_evidence(self.sink.directory,expected_inventory_sha256=hashlib.sha256(data).hexdigest())
        return native,list(transport.entries())

    def assert_join(self,native,row,result=None):
        manifest=row['manifest'];binding=manifest['entry_binding']
        original=lookup_canonical_operation(native,binding['entry']['entry_ordinal'])
        for key in ('owner_kind','owner','entry','provider','model_revision','role','prompt_sha256'):
            self.assertEqual(binding[key],original[key],key)
        self.assertEqual(original['owner']['scope_index'],1)
        self.assertEqual(original['owner']['binding_index'],0)
        self.assertEqual(original['entry']['entry_ordinal'],0)
        self.assertNotIn('attempt_id',binding['entry'])
        self.assertEqual(manifest['wire_sha256'],hashlib.sha256(self.calls[0][1]).hexdigest())
        if result is not None:self.assertEqual(binding['prompt_sha256'],result['prompt_sha256'])
        self.assertTrue(original['unknown_liability']);self.assertEqual(original['actual_acceptance'],'incomplete')
        self.assertIsNone(manifest['actual_cost']);self.assertEqual(manifest['money_basis'],'unknown')
        self.assertFalse(manifest['receipt_authenticity_verified']);self.assertFalse(manifest['count_bucket_verified'])
        self.assertFalse(row['native_binding_verified']);self.assertFalse(row['receipt_authenticity_verified'])
        self.assertEqual(native.terminal['entry_count'],1);self.assertEqual(native.terminal['unsettled_entries'],1)
        self.assertEqual(native.terminal['unclosed_workers'],0)
        self.assertEqual(self.harness.c.snapshot()['budget']['reserved_liability'],5)

    async def test_text_actual_original_entry_matches_wire_and_both_original_readers(self):
        rid,record=await self.sdk()
        result=await self.adapter.execute(rid,record,'native4 fixture question',maximum_charge=5,evaluation_scope=self.scope)
        self.assertTrue(result['accepted']);self.assertEqual(len(self.calls),1)
        native,rows=await self.originals();self.assertEqual(len(rows),1);self.assert_join(native,rows[0],result)
        self.assertEqual(rows[0]['manifest']['server_request_id'],'req_fixture_one')
        self.assertEqual(json.loads(rows[0]['raw_response'])['id'],'resp_fixture_one')

    async def test_visual_actual_owned_png_and_original_task_binding(self):
        try:import cv2;import numpy as np
        except ImportError:self.skipTest('optional vision unavailable')
        rid,record,frame=await self.sdk(visual=True)
        result=await self.adapter.execute(rid,record,frame,'native4 visual question',maximum_charge=5,evaluation_scope=self.scope)
        self.assertTrue(result['accepted']);self.assertEqual(len(self.calls),1)
        wire=json.loads(self.calls[0][1]);uri=wire['input'][0]['content'][1]['image_url']
        png=base64.b64decode(uri.split(',',1)[1],validate=True)
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR),frame.data)
        native,rows=await self.originals();self.assert_join(native,rows[0],result)
        self.assertEqual(rows[0]['manifest']['entry_binding']['role'],'small_vlm')

    async def test_unknown_quote_has_zero_native_entry_and_no_http(self):
        rid,record=await self.sdk()
        result=await self.adapter.execute(rid,record,'q',maximum_charge=None,evaluation_scope=self.scope)
        self.assertEqual(result['status'],'blocked');self.assertEqual(self.calls,[])
        native,rows=await self.originals()
        self.assertEqual(native.terminal['entry_count'],0);self.assertEqual(native.terminal['released_count'],1)
        self.assertEqual(rows[0]['manifest']['terminal'],'not_entered')
        self.assertIsNone(rows[0]['manifest']['entry_binding'])
        self.assertEqual(self.harness.c.snapshot()['budget']['reserved_liability'],0)

    async def test_cancellation_after_actual_http_preserves_one_unknown_native_bill(self):
        rid,record=await self.sdk();self.hold=asyncio.Event()
        task=asyncio.create_task(self.adapter.execute(rid,record,'q',maximum_charge=5,evaluation_scope=self.scope))
        await self.entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        native,rows=await self.originals();self.assert_join(native,rows[0])
        self.assertEqual(rows[0]['manifest']['terminal'],'cancelled')
        self.assertIsNone(self.client._lease);self.assertFalse(self.sink._leases)

    async def test_delete_invalidates_publication_but_retains_explicit_external_originals(self):
        rid,record=await self.sdk();self.hold=asyncio.Event()
        task=asyncio.create_task(self.adapter.execute(rid,record,'private fixture q',maximum_charge=5,evaluation_scope=self.scope))
        await self.entered.wait();await self.harness.h.delete_data();self.hold.set()
        result=await task;self.assertFalse(result['accepted']);self.assertNotIn('text',result)
        native,rows=await self.originals();self.assert_join(native,rows[0])
        self.assertEqual(len(self.calls),1)
        self.assertEqual(rows[0]['manifest']['terminal'],'completed')
