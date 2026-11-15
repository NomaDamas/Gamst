"""Actual SDK entry/settlement/committed native inventory, no model weights."""
import asyncio
import hashlib
import unittest

from gamst.native_entry import BoundCompletion,PreparedCall
from gamst.recording_settlement import RecordingSettlementRecorder
from gamst.schema import timestamp
from benchmarks.recording_financial import digest,encode
from benchmarks.recording_lifetime_ledger import _native_inventory
from benchmarks.recording_native_validation import validate_recording_native_inventory,iter_canonical_inventory
from test_recording_sdk_scopes import NativeFixture
from test_recording_lifetime_ledger import runview
from test_recording_settlement import scenario
from test_recording_financial import adapters
import test_text_context_call as text_tests


class ActualFinance(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued
    adapter=text_tests.Calls.adapter

    async def execute_fixture(self,mode='ok'):
        observer,writer,clock=await self.recording()
        recorder=RecordingSettlementRecorder(self.c,execution_sha256='1'*64,controller_id='main',
            unit='USD.micro',pricing_revision='v1')
        value=scenario();value['operation_key'].update(binding_index=0)
        a=adapters();inventory=dict(schema_version=1,kind='recording_financial_adapters',adapters=list(a.adapters))
        entered=asyncio.Event();release=asyncio.Event()
        class Prepared(PreparedCall):
            async def invoke(inner,binding):
                rid=recorder.bind_entry(binding)
                receipt=recorder.generate_synthetic_receipt(rid,value,expected_scenario_sha256=digest(value),
                    adapter_inventory=encode(inventory,newline=True),expected_adapter_inventory_sha256=a.digest)
                recorder.settle(rid,receipt,now_ms=timestamp())
                entered.set()
                if mode=='cancel':await release.wait()
                if mode=='bad':return dict(text=None,usage=None)
                return dict(text='actual synthetic callback',usage=None)
            async def release(inner):pass
        class Completion(BoundCompletion):
            def prepare(inner,request):return Prepared()
        callback=Completion()
        observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,callback,'callback')
        observer.bind_mapping(clock.mapping,clock.sha256)
        adapter=self.adapter(callback,evaluation_observer=observer)
        rid,record=await self.queued()
        decision=next(r['decision'] for r in self.c.snapshot()['requests'] if r['id']==rid)
        recorder.register_request(rid,decision['decision_sha256'])
        task=asyncio.create_task(adapter.execute(rid,record,'question',maximum_charge=5,
            evaluation_scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0)))
        if mode=='cancel':
            await asyncio.wait_for(entered.wait(),1);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        elif mode=='bad':
            with self.assertRaises(ValueError):await task
        else:self.assertTrue((await task)['accepted'])
        await adapter.close();observer.backend_closed(0,callback);await observer.close()
        index=validate_recording_native_inventory(observer.declarations,writer.output,
            source_inventory=observer.declarations['source_manifest'])
        rows=list(iter_canonical_inventory(index));works=sorted((row for row in rows if row['kind']=='worker_interval'),
            key=lambda row:(row['work_ordinal'],row['worker_ordinal']))
        source_sha=observer.declarations['collection']['recordings'][0]['source_manifest_sha256']
        view=runview(execution_sha256='1'*64,source_manifest_sha256=source_sha,worker_inventory_sha256=digest(works),
            controller_settlement_inventory_sha256=digest(sorted(recorder.receipts(),key=lambda row:row['instance'])),
            native_terminal_sha256=hashlib.sha256((writer.output/'terminal.json').read_bytes()).hexdigest())
        context={key:index.terminal['context'][key] for key in ('collection_sha256','scope_protocol_sha256','instrumentation_protocol_sha256')}
        operations,work,keys=_native_inventory([index],[view],context)
        self.assertEqual(len(operations),1);self.assertEqual(len(recorder.receipts()),1)
        original=next(iter(operations.values()))
        self.assertEqual(recorder.receipts()[0]['instance'],original['instance'])
        self.assertEqual(keys[(0,0,0)],original['instance'])
        self.assertEqual(self.c.snapshot()['budget']['spent_known'],4)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],0)
        # Native historical unknown does not get rewritten by authoritative SDK settlement.
        self.assertTrue(original['unknown_liability'])
        return original

    async def test_actual_entered_callback_and_controller_settle_join_original_native_instance(self):
        self.assertEqual((await self.execute_fixture())['outcome'],'ok')

    async def test_bad_response_preserves_actual_entered_charge(self):
        self.assertEqual((await self.execute_fixture('bad'))['outcome'],'ok')

    async def test_cancelled_callback_preserves_actual_entered_charge_and_original_prefix(self):
        self.assertEqual((await self.execute_fixture('cancel'))['outcome'],'cancelled')

    def test_missing_terminal_is_unknown_prefix_never_a_verified_empty_run(self):
        failed=runview(native_terminal_sha256=None,source_comparable=False,operational_closed=False,
            external_work_closed=False,violations=('missing_native_prefix',))
        self.assertEqual(_native_inventory([None],[failed],{}),({},{},{}))
        with self.assertRaises(ValueError):runview(native_terminal_sha256=None)
        with self.assertRaises(ValueError):_native_inventory([object()],[failed],{})


if __name__=='__main__':unittest.main()
