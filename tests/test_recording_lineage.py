"""Actual SDK entry lineage and direct callback operation locators."""
import unittest
from benchmarks.recording_native_schema import digest
from gamst.native_operation import NativeOperationRef
from gamst.recording_scope import evaluation_lineage
from gamst.schema import timestamp
from test_recording_sdk_scopes import NativeFixture
import test_recording_native_guards as guard_tests
import test_text_context_call as text_tests


class TextLineage(NativeFixture, unittest.IsolatedAsyncioTestCase):
    asyncSetUp = text_tests.Calls.asyncSetUp
    asyncTearDown = text_tests.Calls.asyncTearDown
    queued = text_tests.Calls.queued
    adapter = text_tests.Calls.adapter

    async def test_actual_retry_binding_and_foreign_scope_rejection_before_prepare(self):
        owner, writer, clock = await self.recording()
        tickets = []
        original = owner.reserve_call
        def reserve(*args, **kwargs):
            ticket = original(*args, **kwargs); tickets.append(ticket); return ticket
        owner.reserve_call = reserve
        async def callback(request): return dict(text='answer', usage=None)
        owner.bind_semantic_lane(self.c, 0); owner.bind_backend(0, callback, 'callback')
        owner.bind_mapping(clock.mapping, clock.sha256)
        scope = owner.task_scope(owner.declarations['tasks'][0]['task_id'], binding_index=0)
        adapter = self.adapter(callback, evaluation_observer=owner)
        rid, record = await self.queued()
        self.assertTrue((await adapter.execute(rid, record, 'q', maximum_charge=1, evaluation_scope=scope))['accepted'])
        first = tickets[0]
        binding = first.semantic_entry_binding()
        reference = first.entry_reference()
        self.assertIs(type(reference), NativeOperationRef)
        self.assertEqual(reference.snapshot()['entered_event_sha256'], binding.snapshot()['entry']['entered_event_sha256'])
        detached = reference.snapshot(); detached['binding_index'] = 42
        self.assertEqual(reference.binding_index, 0)
        with self.assertRaises(TypeError): NativeOperationRef(reference.snapshot())
        with self.assertRaises(ValueError): evaluation_lineage(owner.stream_scope(0), 'retry', binding)
        with self.assertRaises(ValueError): evaluation_lineage(None, 'retry', binding)
        with self.assertRaises(ValueError): evaluation_lineage(scope, 'root', binding)
        self.c.settle(rid, components={'other':0}, now_ms=timestamp())
        handle = next(iter(self.c.snapshot()['sources']))
        second = self.c.submit_next(handle, record, now_ms=timestamp(), query=True)['request_id']
        self.assertTrue((await adapter.execute(second, record, 'q2', maximum_charge=1,
            evaluation_scope=scope, evaluation_relationship='retry', evaluation_predecessor=binding))['accepted'])
        # Writer buffering may delay physical event files until close.
        await adapter.close(); owner.backend_closed(0, callback); await owner.close()
        fields = [row['fields'] for row in self.events(writer) if row['event']=='reserved']
        self.assertEqual([row['relationship'] for row in fields], ['root', 'retry'])
        self.assertEqual(fields[1]['predecessor'], reference.instance_sha256)


class DirectReference(unittest.IsolatedAsyncioTestCase):
    recording = guard_tests.Guards.recording
    ticket = guard_tests.Guards.ticket

    async def test_direct_callback_reference_is_absent_until_actual_entry(self):
        owner, writer, callback, calls, good, bad = await self.recording()
        ticket = self.ticket(owner, good)
        self.assertIsNone(ticket.entry_reference()); self.assertIsNone(ticket.semantic_entry_binding())
        captured = []
        def actual_callback():
            captured.append(ticket.entry_reference())
            return {'actual':'result'}
        ticket.invoke_sync(actual_callback)
        self.assertEqual(captured[0].snapshot()['evidence_sha256'], good['record_sha256'])
        self.assertIsNone(captured[0].snapshot()['prompt_sha256'])
        with self.assertRaises(ValueError): ticket.semantic_entry_binding()
        ticket.published(True)
        self.assertEqual(ticket.entry_reference().snapshot(), captured[0].snapshot())
        owner.backend_closed(0, callback); await owner.close()
