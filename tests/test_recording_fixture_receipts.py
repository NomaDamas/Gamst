import asyncio
from pathlib import Path
import tempfile
import unittest

from gamst import TextContextCalls
from gamst.recording_settlement import RecordingSettlementRecorder
from benchmarks.recording_financial import FinancialArtifacts,digest
from benchmarks.recording_fixture_receipts import FinancialFixtureReceiptRouter
from benchmarks.recording_transports import FixtureCompletion,TransportCapacity,TransportMemoryBudget
from test_recording_sdk_scopes import NativeFixture
from test_recording_settlement import scenario
from test_recording_financial import adapters
import test_text_context_call as text_tests


class FixtureReceipts(NativeFixture,unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown
    queued=text_tests.Calls.queued

    async def test_actual_fixture_unknown_prefix_then_known_charge_has_exact_transport_ordinal(self):
        value=scenario();value['operation_key'].update(binding_index=0,call_ordinal=1)
        rows=dict(schema_version=1,kind='fixture_completions',rows=[dict(invocation_ordinal=0,lane_id='text',
            call_ordinal=n,text='actual synthetic answer',usage=None,synthetic_receipt_sha256=digest(value) if n else None) for n in range(2)])
        config=dict(responses_sha256=digest(rows),delay_ms=0,settlement='synthetic_explicit')
        spec=dict(adapter_id='fixture',implementation='fixture_completion_v1',evidence_class='fixture',provider='test.client',
            model_id='fixture/model',model_revision='v1',implementation_sha256='2'*64,dependency_inventory_sha256='3'*64,config=config)
        observer,writer,clock=await self.recording(config=config)
        recorder=RecordingSettlementRecorder(self.c,execution_sha256='1'*64,controller_id='main',unit='USD.micro',pricing_revision='v1')
        f=adapters();inventory=dict(schema_version=1,kind='recording_financial_adapters',adapters=list(f.adapters))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            with FinancialArtifacts(root) as store:
                router=FinancialFixtureReceiptRouter(scenarios=[value],adapter_inventory=inventory,recorders={(0,0):recorder},
                    contexts={0:observer.context},artifacts=store,retain_financial_documents=True)
                memory=TransportMemoryBudget(32*1024**2)
                client=await FixtureCompletion.open(spec,rows,invocation_ordinal=0,lane_id='text',binding_index=0,
                    context=dict(observer.context),capacity=TransportCapacity(262144),memory=memory,receipt_router=router)
                observer.bind_semantic_lane(self.c,0);observer.bind_backend(0,client,'callback');observer.bind_mapping(clock.mapping,clock.sha256)
                adapter=TextContextCalls(self.h,'main',client,provider='test.client',model_revision='v1',evaluation_observer=observer)
                self.adapters.append(adapter)
                try:
                    for n in range(2):
                        rid,record=await self.queued()
                        decision=next(r['decision'] for r in self.c.snapshot()['requests'] if r['id']==rid)
                        router.register_request(0,0,rid,decision['decision_sha256'])
                        result=await adapter.execute(rid,record,'q',maximum_charge=5,
                            evaluation_scope=observer.task_scope(observer.declarations['tasks'][0]['task_id'],0))
                        self.assertTrue(result['accepted'])
                    self.assertEqual(self.c.snapshot()['budget']['spent_known'],4)
                    self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],5)
                    refs=router.retained_references();self.assertEqual(len(refs),1)
                    settlement=store.read(refs[0]['controller_settlement']);evidence=store.read(refs[0]['evidence'])
                    source=store.read(refs[0]['source_document'])
                    self.assertEqual(source['lines'][0]['operation_key']['call_ordinal'],1)
                    self.assertEqual(settlement['instance'],evidence['owner_instance'])
                    self.assertEqual(router.actual_settlement_inventory_sha256(0),digest([settlement]))
                    changed=router.actual_settlements(0)[0];changed['charged']=False
                    self.assertTrue(router.actual_settlements(0)[0]['charged'])
                finally:
                    await adapter.close();await client.close();observer.backend_closed(0,client)
                self.assertEqual(memory.reserved_bytes,0)

    def test_missing_retention_cannot_activate_financial_router(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            with FinancialArtifacts(root) as store:
                with self.assertRaisesRegex(ValueError,'retention'):
                    FinancialFixtureReceiptRouter(scenarios=[],adapter_inventory={},recorders={},contexts={},
                        artifacts=store,retain_financial_documents=False)


if __name__=='__main__':unittest.main()
