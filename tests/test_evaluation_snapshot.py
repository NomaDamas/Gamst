import asyncio
import copy
import json
import time
import unittest

from gamst import (Gamst, SensorMetadata, TextContextCalls, SourceContextView,
                        evaluation_snapshot_implementation_sha256)
from gamst.capabilities import CapabilitySpec, OutputField, bounded_json
from gamst.call_controller import CallController, _digest
from gamst.call_policy import CallPolicy
from gamst.call_budget import CallBudget
from gamst.schema import timestamp


class SnapshotContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = Gamst(consent=True)
        self.publisher = self.h.create_evaluation_snapshot_publisher(
            implementation_sha256=evaluation_snapshot_implementation_sha256())
        self.view = self.publisher.context_view
        self.h.register_capability(CapabilitySpec('source.fill', 'source.fill', 'video',
            'test.source', '1', (OutputField('fraction', 'number'),), 'frame'))
        self.source = await self.h.push_observation('source.fill', {'fraction': .4},
            sensor=SensorMetadata('camera', 'video', 1, stream_epoch='original'),
            timestamp_ms=timestamp())
        # A scalar legacy signal exercises the NULL-safe exclusion and trends.
        await self.h.push_signal('ambient.level', 3, 'audio')
        self.adapters = []

    async def asyncTearDown(self):
        for adapter in self.adapters:
            await adapter.close()
        await self.h.close()

    async def snapshot(self, task, *, publisher=None):
        publisher = publisher or self.publisher
        begin = time.monotonic_ns()
        context = await self.h.get_context(publisher.subject_id, context_view=publisher.context_view)
        end = time.monotonic_ns()
        return await publisher.publish(dict(
            kind='frozen_context_bundle_not_new_sensor_measurement', task_id=task,
            input_sha256=_digest({'question': 'What is the fill?'}), task_sha256='a'*64,
            source_manifest_sha256='b'*64, mapping_sha256='c'*64,
            context_sha256=_digest(context), snapshot_timestamp_ms=context['timestamp_ms'],
            scheduled_offset_ns=0, observed_begin_ns=begin, observed_end_ns=end,
            watermarks=[], context=context))

    def consumer(self, complete):
        name = 'consumer' + str(len(self.adapters))
        controller = CallController(policy=CallPolicy(mode='escalation', cooldown_ms=0,
            max_age_ms=60000), budget=CallBudget(unit='USD.micro', pricing_revision='fixture1',
            ceiling=1000), max_requests=32)
        self.h.register_call_controller(name, controller)
        adapter = TextContextCalls(self.h, name, complete, provider='fixture.answer',
                                   model_revision='fixture1')
        self.adapters.append(adapter)
        return controller, adapter

    async def answer(self, task, controller, adapter):
        record = await self.snapshot(task)
        now = timestamp()
        source = controller.open_source(record, now_ms=now)
        decision = controller.submit(source, record, sequence=record['observation']['sensor']['sequence'],
                                     now_ms=now, query=True)
        result = await adapter.execute(decision['request_id'], record, 'What is the fill?',
                                       maximum_charge=10)
        self.assertTrue(result['accepted'])
        # Explicit synthetic fixture settlement, never a real invoice assertion.
        controller.settle(decision['request_id'], components={'other': 1}, now_ms=timestamp())
        return record, result

    async def test_eight_actual_answers_and_overlap_do_not_reconsume_wrappers(self):
        sent = []
        async def complete(request):
            sent.append(copy.deepcopy(request))
            return dict(text='The observed fraction is 0.4.', usage=None)
        controller, adapter = self.consumer(complete)
        first_context = await self.h.get_context(context_view=self.view)
        source_count = first_context['summary']['record_count']
        for i in range(8):
            record, _ = await self.answer('sequential' + str(i), controller, adapter)
            context = record['observation']['value']['context']
            self.assertEqual(context['summary']['record_count'], source_count)
            self.assertEqual(context['trends'], first_context['trends'])
            self.assertNotIn('gamst.recording_snapshot', json.dumps(context['models']))
            self.assertNotIn('evaluation.context_snapshot', json.dumps(context['observations']))
            self.assertIn(self.source['id'], json.dumps(context['observations']))
            bounded_json(record, max_bytes=131072)
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        def callback(index):
            async def overlapping(request):
                sent.append(copy.deepcopy(request)); entered[index].set()
                await release.wait()
                return dict(text='overlap fixture answer', usage=None)
            return overlapping
        c1, a1 = self.consumer(callback(0)); c2, a2 = self.consumer(callback(1))
        t1 = asyncio.create_task(self.answer('overlap1', c1, a1))
        await entered[0].wait()
        t2 = asyncio.create_task(self.answer('overlap2', c2, a2))
        await entered[1].wait()
        during = await self.h.get_context(context_view=self.view)
        self.assertEqual(during['summary']['record_count'], source_count)
        release.set(); await asyncio.gather(t1, t2)
        self.assertEqual(len(sent), 10)
        for request in sent:
            record = json.loads(request['messages'][1]['content'])['observation_record']
            self.assertEqual(record['observation']['value']['context']['summary']['record_count'], source_count)
        default = await self.h.get_context()
        self.assertEqual(default['summary']['record_count'], source_count + 10)
        self.assertNotIn('view', default['context_policy'])
        self.assertIn('gamst.recording_snapshot', json.dumps(default['models']))
        self.assertIn('evaluation.context_snapshot', json.dumps(default['observations']))
        retained = self.h.store.get_many([self.source['id']])[0]
        self.assertEqual(retained, self.source)

    async def test_view_cannot_cross_host_subject_or_privacy_generation(self):
        with self.assertRaises(TypeError): SourceContextView(self.publisher)
        with self.assertRaises(TypeError): await self.h.get_context(context_view={})
        with self.assertRaises(ValueError): await self.h.get_context('someone-else', context_view=self.view)
        other = Gamst(consent=True)
        try:
            with self.assertRaises(ValueError): await other.get_context(context_view=self.view)
        finally:
            await other.close()
        await self.snapshot('before-reset')
        await self.h.reset_processing()
        with self.assertRaises(RuntimeError): await self.h.get_context(context_view=self.view)
        publisher = self.h.create_evaluation_snapshot_publisher(
            implementation_sha256=evaluation_snapshot_implementation_sha256())
        context = await self.h.get_context(context_view=publisher.context_view)
        self.assertNotIn('evaluation.context_snapshot', json.dumps(context['observations']))
        await self.h.pause()
        with self.assertRaises(RuntimeError): await self.h.get_context(context_view=publisher.context_view)

    async def test_snapshot_preserves_original_time_and_rejects_changed_or_duplicate_bundle(self):
        record = await self.snapshot('original')
        envelope = record['observation']['value']
        self.assertEqual(record['timestamp_ms'], envelope['context']['timestamp_ms'])
        with self.assertRaises(ValueError): await self.publisher.publish(envelope)
        changed = copy.deepcopy(envelope); changed['task_id'] = 'changed'
        changed['context']['timestamp_ms'] += 1
        with self.assertRaises(ValueError): await self.publisher.publish(changed)
        with self.assertRaises(ValueError):
            self.h.create_evaluation_snapshot_publisher(implementation_sha256='f'*64)

    async def test_privacy_actions_invalidate_issued_views(self):
        for action in ('delete', 'revoke', 'close'):
            human = Gamst(consent=True)
            publisher = human.create_evaluation_snapshot_publisher(
                implementation_sha256=evaluation_snapshot_implementation_sha256())
            try:
                if action == 'delete': await human.delete_data()
                elif action == 'revoke': human.consent = False
                else: await human.close()
                with self.assertRaises((RuntimeError, PermissionError)):
                    await human.get_context(context_view=publisher.context_view)
                with self.assertRaises((RuntimeError, PermissionError)):
                    await human.get_trends(context_view=publisher.context_view)
                self.assertEqual(human._evaluation_snapshot_publishers, {})
                self.assertIsNone(publisher.subject_id)
                self.assertEqual(publisher._task_ids, set())
            finally:
                await human.close()

    async def test_deleted_subjects_do_not_exhaust_active_registry_or_retain_task_ids(self):
        old_views = []
        for i in range(20):
            subject = 'private-subject-' + str(i)
            publisher = self.h.create_evaluation_snapshot_publisher(subject,
                implementation_sha256=evaluation_snapshot_implementation_sha256())
            await self.snapshot('private-task-' + str(i), publisher=publisher)
            old_views.append(publisher.context_view)
            await self.h.delete_data(subject)
            self.assertNotIn(subject, self.h._evaluation_snapshot_publishers)
            self.assertIsNone(publisher.subject_id)
            self.assertFalse(publisher._task_ids)
        for view in old_views:
            with self.assertRaises(RuntimeError): await self.h.get_context(context_view=view)


if __name__ == '__main__':
    unittest.main()
