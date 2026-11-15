"""Actual SDK ownership for the finite runtime finance bridge, no real money."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from gamst import TextContextCalls
from benchmarks.recording_financial import decode,digest,encode
from benchmarks.recording_native_schema import digest as original_digest,run_context
from benchmarks.recording_native_writer import ArtifactBudget,RecordingNativeWriter
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_runtime_native import validate_runtime_native_inventory
from benchmarks.recording_transports import FixtureCompletion,LocalChatCompletions,TransportCapacity,TransportMemoryBudget
from benchmarks.recording_runtime_finance import (SCENARIOS_PATH,open_fixture_receipts,
    validate_runtime_charge_scenarios)
from benchmarks.replay_clock import ReplayClock
from recording_numeric_fixtures import prospective
import test_text_context_call as text_tests
from test_recording_transports import Server,Sink,dependencies,real_adapter,HTTP_DEPS


def fixture(unknown=False,real_spec=None):
    ds,l,bundles,numeric,responses,scenarios=prospective();d=ds[0]
    budget=dict(unit='USD.micro',pricing_revision='v1',ceiling=100,max_calls=256,max_inflight=4,receipt_limit=128)
    if unknown:
        responses['rows'][0]['synthetic_receipt_sha256']=None;scenarios.pop(0)
    u=d['runtime_policy'];c=d['collection'];q=d['scope_protocol'];i=d['instrumentation_protocol']
    for arm in u['arms']:
        for controller in arm['controllers']:controller['budget_sha256']=original_digest(budget)
    if real_spec is None:u['adapters'][0]['config']['responses_sha256']=original_digest(responses)
    else:u['adapters'][0]={**deepcopy(real_spec),'adapter_id':'fixture'}
    c['runtime_policy_sha256']=original_digest(u)
    for arm in c['arms']:
        for binding in arm['provider_bindings']:
            if binding['operation_kinds']==['semantic']:
                binding.update(config_sha256=original_digest(u['adapters'][0]['config']),
                    provider=u['adapters'][0]['provider'],model_revision=u['adapters'][0]['model_revision'])
    q['collection_sha256']=original_digest(c)
    i.update(collection_sha256=original_digest(c),scope_protocol_sha256=original_digest(q))
    execution=dict(schema_version=1,kind='recording_execution',runtime_policy_sha256=original_digest(u),
        collection_sha256=original_digest(c),scope_protocol_sha256=original_digest(q),instrumentation_protocol_sha256=original_digest(i),
        invocation_ordinal=0,recording_index=c['invocations'][0]['recording_index'],arm=c['invocations'][0]['arm'],
        stream_id=c['invocations'][0]['stream_id'],execution_namespace='original-execution',cache_namespace='original-cache',asset_inventory_sha256='1'*64)
    docs={'runtime/policy.json':u,'runtime/collection.json':c,'runtime/scopes.json':q,'runtime/instrumentation.json':i,
        'runtime/accounting.json':d['stream_protocol'],'runtime/financial-adapters.json':numeric['financial_adapter_inventory'],
        'responses/000.json':responses,'budgets/000.json':budget,SCENARIOS_PATH:dict(schema_version=1,
            kind='recording_synthetic_charge_scenarios',scenarios=scenarios)}
    if real_spec is not None:docs.pop('responses/000.json');docs[SCENARIOS_PATH]['scenarios']=[]
    return d,l,bundles,numeric,docs,execution


def validate(docs):
    return validate_runtime_charge_scenarios(docs[SCENARIOS_PATH],runtime_policy=docs['runtime/policy.json'],
        collection=docs['runtime/collection.json'],responses=[docs['responses/000.json']],
        accounting_policy=docs['runtime/accounting.json'],financial_adapter_inventory=docs['runtime/financial-adapters.json'])


class RuntimeChargeInputs(unittest.TestCase):
    def test_all_original_seventy_two_scenarios_fit_joint_limit_and_are_immutable(self):
        *_,docs,execution=fixture();checked=validate(docs)
        self.assertEqual(len(checked),72);self.assertTrue(all(type(value) is bytes for value in checked))
        self.assertLessEqual(len(encode(docs[SCENARIOS_PATH],newline=True)),65536)
        docs[SCENARIOS_PATH]['scenarios'][0]['billed_atomic']=0
        self.assertTrue(all(decode(value)['billed_atomic']>0 for value in checked))

    def test_missing_extra_duplicate_unbound_and_oversized_scenarios_refuse_before_activation(self):
        for mutation in ('missing','duplicate','extra','operation','resource','unit','u1','oversize','reused_body'):
            with self.subTest(mutation=mutation):
                *_,docs,execution=fixture();values=docs[SCENARIOS_PATH]['scenarios']
                if mutation=='missing':values.pop()
                elif mutation=='duplicate':values.append(deepcopy(values[0]))
                elif mutation=='extra':values.append({**deepcopy(values[0]),'invoice_id':'another-unreferenced-invoice'})
                elif mutation=='operation':values[0]['operation_key']['binding_index']=1
                elif mutation=='resource':values[0]['resource_id']='other-resource'
                elif mutation=='unit':values[0]['currency_unit']='other-unit'
                elif mutation=='u1':docs['runtime/policy.json']['schema_version']=1
                elif mutation=='reused_body':
                    docs['responses/000.json']['rows'][-1]['synthetic_receipt_sha256']=digest(values[0]);values.pop()
                else:values.extend(deepcopy(values)*2)
                if mutation in ('operation','resource','unit','reused_body'):
                    if mutation!='reused_body':docs['responses/000.json']['rows'][0]['synthetic_receipt_sha256']=digest(values[0])
                    docs['runtime/policy.json']['adapters'][0]['config']['responses_sha256']=original_digest(docs['responses/000.json'])
                    docs['runtime/collection.json']['runtime_policy_sha256']=original_digest(docs['runtime/policy.json'])
                    for arm in docs['runtime/collection.json']['arms']:
                        for binding in arm['provider_bindings']:
                            if binding['operation_kinds']==['semantic']:binding['config_sha256']=original_digest(docs['runtime/policy.json']['adapters'][0]['config'])
                with self.assertRaises(ValueError):validate(docs)


class ActualRuntimeFinance(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_refusal_never_binds_a_recorder_to_an_active_or_wrong_lane_controller(self):
        for mutation in ('binding','active','unit'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                d,l,bundles,numeric,docs,execution=fixture();case=text_tests.Calls();await case.asyncSetUp()
                try:
                    bindings={'answer':0,'periodic':1}
                    if mutation=='binding':bindings['periodic']=0
                    elif mutation=='active':await case.queued()
                    else:docs['budgets/000.json']['unit']='another-unit'
                    with self.assertRaises(ValueError):
                        open_fixture_receipts(docs,execution,{'main':case.c},bindings,run_context(d,'1'*64,0),
                            Path(directory).resolve(),ArtifactBudget(),retain_financial_documents=True)
                    self.assertIsNone(case.c._recording_settlement_recorder)
                    self.assertEqual(list(Path(directory).iterdir()),[])
                finally:await case.asyncTearDown()

    async def execute_fixture(self,root,mode,*,real_spec=None,dependency_inventory=None):
        asyncio.get_running_loop().slow_callback_duration=2
        d,l,bundles,numeric,docs,execution=fixture(unknown=mode=='unknown',real_spec=real_spec)
        case=text_tests.Calls();await case.asyncSetUp();observer=None;client=None;owner=None;adapter=None
        budget=ArtifactBudget();memory=TransportMemoryBudget(32*1024**2)
        writer=await RecordingNativeWriter.open(root/'native',run_context(d,'1'*64,0))
        try:
            observer=RecordingEvaluationSession.from_runtime_frozen(d,writer,latency_policy=l,source_bundles=bundles,
                numeric_inputs=numeric,run_nonce='1'*64,execution_ordinal=0,retain_evaluation_hashes=True,retain_financial_documents=True)
            finance=root/'finance';finance.mkdir()
            owner=open_fixture_receipts(docs,execution,{'main':case.c},{'answer':0,'periodic':1},observer.context,finance,budget,
                retain_financial_documents=True)
            self.assertEqual(len(owner.recorders),1)
            self.assertEqual(owner.actual_settlements(),());self.assertEqual(list(finance.iterdir()),[])
            spec=docs['runtime/policy.json']['adapters'][0]
            if real_spec is None:
                client=await FixtureCompletion.open(spec,docs['responses/000.json'],invocation_ordinal=0,lane_id='answer',binding_index=0,
                    context=dict(observer.context),capacity=TransportCapacity(262144),memory=memory,receipt_router=owner.router)
            else:
                client=await LocalChatCompletions.open(spec,invocation_ordinal=0,lane_id='answer',binding_index=0,context=dict(observer.context),
                    dependency_inventory=dependency_inventory,sink=Sink(),capacity=TransportCapacity(262144),memory=memory)
            observer.bind_semantic_lane(case.c,0);observer.bind_backend(0,client,'callback')
            clock=ReplayClock(d['source_manifest']);observer.bind_mapping(clock.mapping,clock.sha256)
            adapter=TextContextCalls(case.h,'main',client,provider=spec['provider'],model_revision=spec['model_revision'],evaluation_observer=observer)
            rid,record=await case.queued();decision=next(row['decision'] for row in case.c.snapshot()['requests'] if row['id']==rid)
            owner.register_request(0,rid,decision['decision_sha256'])
            if mode=='cancel':
                entered=asyncio.Event();release=asyncio.Event();original=owner.router.settle
                async def settlement(*args,**kwargs):
                    await original(*args,**kwargs);entered.set();await release.wait()
                owner.router.settle=settlement
            task=asyncio.create_task(adapter.execute(rid,record,'question',maximum_charge=10,
                evaluation_scope=observer.task_scope(d['tasks'][0]['task_id'],0)))
            if mode=='cancel':
                await asyncio.wait_for(entered.wait(),5);task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
            else:self.assertTrue((await task)['accepted'])
            if mode in ('unknown','real_unknown'):
                self.assertEqual(owner.actual_settlements(),());self.assertEqual(owner.retained_references(),())
                self.assertEqual(case.c.snapshot()['budget']['reserved_liability'],10)
            else:
                self.assertEqual(case.c.snapshot()['budget']['spent_known'],10)
                actual=owner.actual_settlements();self.assertEqual(len(actual),1)
                self.assertEqual(actual[0]['execution_sha256'],original_digest(execution))
                refs=owner.retained_references();self.assertEqual(len(refs),1)
                self.assertEqual(owner.artifacts.read(refs[0]['controller_settlement']),actual[0])
                self.assertEqual(owner.controller_settlement_inventory_sha256,digest(list(actual)))
                self.assertGreater(budget.used,0)
            return d,l,bundles,numeric,writer
        finally:
            if adapter is not None:await adapter.close()
            if client is not None:
                await client.close()
                if observer is not None:observer.backend_closed(0,client)
            if observer is not None:await observer.close()
            if owner is not None:owner.close()
            await case.asyncTearDown()
            self.assertEqual(memory.reserved_bytes,0)

    async def test_actual_sdk_original_receipt_only_after_callback_and_budget_settlement(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            d,l,bundles,numeric,writer=await self.execute_fixture(Path(directory).resolve(),'ok')
            index=validate_runtime_native_inventory(d,writer.output,source_inventory=d['source_manifest'],latency_policy=l,source_bundles=bundles,numeric_inputs=numeric)
            self.assertEqual(index.count,1)

    async def test_cancellation_retains_original_entered_known_charge(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:await self.execute_fixture(Path(directory).resolve(),'cancel')

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in HTTP_DEPS),'optional owned HTTP dependencies absent')
    async def test_real_owned_http_callback_never_becomes_synthetic_settled_money(self):
        async with Server() as server:
            inventory=dependencies()
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                await self.execute_fixture(Path(directory).resolve(),'real_unknown',real_spec=real_adapter(server,inventory),dependency_inventory=inventory)
                self.assertEqual(len(server.requests),1)

    async def test_unknown_row_keeps_actual_liability_and_creates_no_invoice(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:await self.execute_fixture(Path(directory).resolve(),'unknown')


if __name__=='__main__':unittest.main()
