"""Journal capacity is separate from actual SDK unknown-financial admission.

The 3,601-entry fixture observes real local callbacks through the session/writer.
It is journal-only evidence, not SDK budget capacity, model quality, elapsed-hour
execution, original-recording replay, or a verified monetary result.
"""
from pathlib import Path
import tempfile
import unittest

from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_schema import digest, run_context
from benchmarks.recording_native_validation import (validate_recording_native_inventory,
    iter_canonical_operations, lookup_canonical_operation)
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.replay_clock import ReplayClock
from humansense import HumanSense, TextContextCalls, CapabilitySpec, OutputField, SensorMetadata
from humansense.call_budget import CallBudget
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.schema import timestamp
from recording_native_fixtures import declarations, rehash, session


class Capacity(unittest.IsolatedAsyncioTestCase):
    def directory(self):
        directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(directory.cleanup)
        return Path(directory.name).resolve()

    def validate(self,observer,writer):
        return validate_recording_native_inventory(observer.declarations,writer.output,
            source_inventory=observer.declarations['source_manifest'])

    async def test_journal_only_3601_actual_entries_preserve_original_entry_across_chunk_rotation(self):
        root=self.directory();d=declarations(root,role='signal')
        for arm in d['collection']['arms']:
            arm['provider_bindings'][0].update(operation_kinds=['process'],semantic_lane=None)
        rehash(d)
        writer=await RecordingNativeWriter.open(root/'native',run_context(d,'a'*64,0))
        observer=RecordingEvaluationSession.from_frozen(d,writer,run_nonce='a'*64,execution_ordinal=0,
            retain_evaluation_hashes=True)
        self.addAsyncCleanup(observer.close)
        calls=0
        async def actual_callback():
            nonlocal calls
            calls+=1
            return {'callback_sequence':calls}
        observer.bind_backend(0,actual_callback,'callback')
        clock=ReplayClock(d['source_manifest']);observer.bind_mapping(clock.mapping,clock.sha256)
        scope=observer.stream_scope(0);record=d['source_manifest']['records'][0]
        ref=dict(kind='capture',source_index=0,capture_ordinal=0,record_sha256=digest(record),mapping_sha256=clock.sha256)
        request=digest(dict(binding=0,operation_kind='process',capture_refs=[ref]))
        first=None
        for ordinal in range(3601):
            ticket=observer.reserve_operation(0,scope,'process',[ref],request,ref['record_sha256'])
            result=await ticket.invoke(actual_callback)
            self.assertEqual(result['callback_sequence'],ordinal+1)
            if first is None:
                first=ticket
            else:
                ticket.published(True)
            if ordinal%8==7:
                await writer.drain()
        await writer.drain()
        # Original entry0 stays explicitly active until after many original
        # chunks have sealed; no rotation can retire its unknown disposition.
        self.assertIs(observer._tickets[first.ordinal],first)
        self.assertGreater(len(writer.status()['chunks']),2)
        self.assertEqual(observer._entries,3601)
        first.published(True)
        observer.backend_closed(0,actual_callback);await observer.close()
        index=self.validate(observer,writer)
        rows=list(iter_canonical_operations(index))
        self.assertEqual(calls,3601);self.assertEqual(len(rows),3601)
        self.assertEqual([row['entry_ordinal'] for row in rows],list(range(3601)))
        self.assertEqual(len({row['instance'] for row in rows}),3601)
        self.assertTrue(all(row['unknown_liability'] and row['declared_settlement_ref'] is None for row in rows))
        self.assertEqual(rows[0]['result_sha256'],digest({'callback_sequence':1}))
        self.assertEqual(rows[-1]['result_sha256'],digest({'callback_sequence':3601}))
        original=lookup_canonical_operation(index,0)
        self.assertEqual(original['entry']['instance_sha256'],rows[0]['instance'])
        self.assertEqual(original['entry']['request_decision_sha256'],request)
        self.assertLess(rows[0]['entry_locator']['chunk_ordinal'],rows[-1]['entry_locator']['chunk_ordinal'])
        self.assertTrue(all(ref['entry_count']<=128 and ref['event_count']<=256 for ref in index.chunk_refs))
        self.assertEqual(index.terminal['state'],'closed');self.assertEqual(index.terminal['unsettled_entries'],3601)
        self.assertEqual(index.terminal['outstanding_tickets'],0)
        self.assertEqual(index.actual_acceptance,'incomplete');self.assertFalse(index.money_verified)

    async def test_actual_sdk_default128_and_max256_unknown_requests_refuse_more_without_resetting_debt(self):
        for maximum in (None,256):
            with self.subTest(maximum=maximum):
                root=self.directory();observer,writer,clock=await session(root)
                self.addAsyncCleanup(observer.close)
                human=HumanSense(consent=True);self.addAsyncCleanup(human.close)
                budget=CallBudget(unit='fixture',pricing_revision='1',ceiling=10000,max_calls=3601,max_inflight=1024)
                policy=CallPolicy(mode='escalation',cooldown_ms=0,max_age_ms=60000)
                options={} if maximum is None else {'max_requests':maximum}
                controller=CallController(policy=policy,budget=budget,**options)
                human.register_call_controller('capacity',controller)
                spec=CapabilitySpec('capacity.signal','capacity.signal','audio','fixture.source','1',
                    (OutputField('level','number'),),'frame',freshness_ms=60000)
                human.register_capability(spec)
                called=[]
                async def actual_client(request):called.append(True);return dict(text='answer',usage=None)
                observer.bind_semantic_lane(controller,0);observer.bind_backend(0,actual_client,'callback')
                observer.bind_mapping(clock.mapping,clock.sha256)
                scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)
                adapter=TextContextCalls(human,'capacity',actual_client,provider='test.client',model_revision='v1',
                    evaluation_observer=observer)
                self.addAsyncCleanup(adapter.close)
                source=None;sequence=0
                async def submit():
                    nonlocal source,sequence
                    sequence+=1;now=timestamp()
                    record=await human.push_observation(spec.id,{'level':.5},
                        sensor=SensorMetadata('mic','mic-1',sequence,'samples',stream_epoch='capacity'),timestamp_ms=now)
                    if source is None:source=controller.open_source(record,now_ms=now)
                    return controller.submit(source,record,sequence=sequence,now_ms=now,query=True),record
                limit=128 if maximum is None else maximum
                for ordinal in range(limit):
                    pending,record=await submit();self.assertEqual(pending['status'],'queued')
                    result=await adapter.execute(pending['request_id'],record,'q',maximum_charge=1,evaluation_scope=scope)
                    self.assertTrue(result['accepted'])
                    if ordinal%8==7:await writer.drain()
                ledger_scope=budget.snapshot()['scope']
                for lifecycle in ('full','reset','delete'):
                    if lifecycle=='reset':await human.reset_processing();source=None
                    elif lifecycle=='delete':await human.delete_data();source=None
                    rejected,_=await submit()
                    self.assertEqual((rejected['status'],rejected['reason']),('rejected','request_capacity'))
                    state=controller.snapshot();financial=state['budget']
                    self.assertEqual(financial['scope'],ledger_scope)
                    self.assertEqual(financial['accepted'],limit);self.assertEqual(financial['dispatched'],limit)
                    self.assertEqual(financial['unresolved_dispatched'],limit)
                    self.assertEqual(financial['reserved_liability'],limit)
                    self.assertEqual(financial['unknown_reports'],limit);self.assertEqual(financial['spent_known'],0)
                    self.assertEqual(len(state['requests'])+len(state['detached_financial_requests']),limit)
                self.assertEqual(len(called),limit);self.assertLess(limit,3600)
                self.assertEqual(len(controller.snapshot()['detached_financial_requests']),limit)
                await adapter.close();observer.backend_closed(0,actual_client)
                await human.close();await observer.close()
                index=self.validate(observer,writer)
                self.assertEqual(index.count,limit);self.assertEqual(index.terminal['unsettled_entries'],limit)
                self.assertEqual(index.terminal['state'],'closed')
        with self.assertRaises(ValueError):
            CallController(policy=CallPolicy(),budget=CallBudget(unit='fixture',pricing_revision='1',ceiling=10000),
                max_requests=3600)


if __name__=='__main__':unittest.main()
