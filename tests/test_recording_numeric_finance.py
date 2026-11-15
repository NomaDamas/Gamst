"""Actual prospective Q2 SDK entries and complete synthetic financial ledger.

This is a financial fixture, not an ABC source/quality efficacy experiment.
The typed runtime source/parent observations are explicit fixture inputs;
no real camera, invoice authority, model weights or backend entry is claimed.
"""
import asyncio
import dataclasses
import copy
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

from gamst import TextContextCalls
from gamst.recording_settlement import RecordingSettlementRecorder
from benchmarks.recording_financial import FinancialArtifacts,_issue_financial_run_view,decode,digest,ratio
from benchmarks.recording_native_schema import run_context,digest as upstream_digest
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.recording_native_validation import iter_canonical_inventory
from benchmarks.recording_runtime_native import validate_runtime_native_inventory,preflight_runtime_collection
from benchmarks.recording_transports import FixtureCompletion,TransportCapacity,TransportMemoryBudget
from benchmarks.recording_fixture_receipts import FinancialFixtureReceiptRouter
from benchmarks.recording_lifetime_ledger import validate_lifetime_ledger
from benchmarks.recording_exposure_report import compare_quality_latency,reconcile_exposure
from benchmarks.replay_clock import ReplayClock
from benchmarks.perception_comparison import plan_digest
from recording_numeric_fixtures import prospective
from test_recording_lifetime_ledger import runview,lifetime
import test_text_context_call as text_tests


async def complete_collection(root,store,*,financial_state='complete'):
    ds,l,bundles,numeric,responses,scenarios=prospective();d=ds[0]
    admission=preflight_runtime_collection(ds,latency_policy=l,source_bundles=bundles,numeric_inputs=numeric)
    if not admission['sample_gate_eligible'] or admission['predicted_artifact_bytes']>4*1024**3:
        raise AssertionError('original 24-group joint minimum does not fit before activation')
    p=d['plan'];c=d['collection'];q=d['scope_protocol'];i=d['instrumentation_protocol'];u=d['runtime_policy']
    k=d['stream_protocol'];g=numeric['opportunities'];e=numeric['exposure_policy'];f=numeric['financial_adapter_inventory']
    source_rows=[];evidence_rows=[];operation_rows=[];lane_rows=[];finality_rows=[]
    runs=[];indices=[];costs=[];lifes=[];coverage_refs=[];receipts=[];assessment_rows=[]
    for inv in c['invocations']:
        ordinal=inv['ordinal'];record=c['recordings'][inv['recording_index']];current=ds[inv['recording_index']]
        start=time.monotonic_ns();case=text_tests.Calls();await case.asyncSetUp()
        nonce=hashlib.sha256(('original-invocation-'+str(ordinal)).encode()).hexdigest()
        writer=await RecordingNativeWriter.open(root/('native-%03d'%ordinal),run_context(current,nonce,ordinal))
        observer=RecordingEvaluationSession.from_runtime_frozen(current,writer,latency_policy=l,source_bundles=bundles,
            numeric_inputs=numeric,run_nonce=nonce,execution_ordinal=ordinal,retain_evaluation_hashes=True,retain_financial_documents=True)
        recorder=RecordingSettlementRecorder(case.c,execution_sha256=nonce,controller_id='main',unit='USD.micro',pricing_revision='v1')
        router=FinancialFixtureReceiptRouter(scenarios=[scenarios[ordinal]],adapter_inventory=f,recorders={(ordinal,0):recorder},
            contexts={ordinal:observer.context},artifacts=store,retain_financial_documents=True)
        client=await FixtureCompletion.open(u['adapters'][0],responses,invocation_ordinal=ordinal,lane_id='answer',binding_index=0,
            context=dict(observer.context),capacity=TransportCapacity(262144),memory=TransportMemoryBudget(32*1024**2),receipt_router=router)
        observer.bind_semantic_lane(case.c,0);observer.bind_backend(0,client,'callback')
        clock=ReplayClock(current['source_manifest']);observer.bind_mapping(clock.mapping,clock.sha256)
        adapter=TextContextCalls(case.h,'main',client,provider='test.fixture',model_revision='1',evaluation_observer=observer)
        case.adapters.append(adapter)
        try:
            rid,record_value=await case.queued();decision=next(r['decision'] for r in case.c.snapshot()['requests'] if r['id']==rid)
            router.register_request(ordinal,0,rid,decision['decision_sha256'])
            task=current['tasks'][0]
            result=await adapter.execute(rid,record_value,task['input']['question'],maximum_charge=10,
                evaluation_scope=observer.task_scope(task['task_id'],0))
            if not result['accepted']:raise AssertionError('actual fixture callback not admitted')
            received=time.monotonic_ns()
        finally:
            await adapter.close();await client.close();observer.backend_closed(0,client);await observer.close();await case.asyncTearDown()
        end=time.monotonic_ns()
        index=validate_runtime_native_inventory(current,writer.output,source_inventory=current['source_manifest'],
            latency_policy=l,source_bundles=bundles,numeric_inputs=numeric);indices.append(index)
        native=list(iter_canonical_inventory(index));operation=next(row for row in native if row['kind']=='operation')
        works=sorted((row for row in native if row['kind']=='worker_interval'),key=lambda row:(row['work_ordinal'],row['worker_ordinal']))
        refs=router.retained_references()[0];document=store.read(refs['source_document']);evidence=store.read(refs['evidence'])
        source_rows.append(dict(source_document=refs['source_document']));evidence_rows.append(evidence)
        operation_rows.append(dict(instance=operation['instance'],invocation_ordinal=ordinal,binding_index=0,entry_ordinal=0,
            status='settled',basis='original_line',evidence=refs['evidence'],controller_settlement=refs['controller_settlement'],meter_lane=None))
        coverage=dict(stream_id=inv['stream_id'],source_manifest_sha256=record['source_manifest_sha256'],fixture_source_join=True)
        covref=store.write('ledger/coverage-%03d.json'%ordinal,coverage);coverage_refs.append(covref)
        run=runview(inv['arm'],execution_sha256=nonce,invocation_ordinal=ordinal,stream_id=inv['stream_id'],recording_id=record['recording_id'],
            source_manifest_sha256=record['source_manifest_sha256'],source_coverage_sha256=digest(coverage),
            activation_started_ns=start,child_spawn_started_ns=None,playback_anchor_ns=None,source_finished_ns=None,
            child_cleanup_finished_ns=end,process_reaped_ns=None,parent_resources_closed_ns=end,worker_inventory_sha256=digest(works),
            controller_settlement_inventory_sha256=router.actual_settlement_inventory_sha256(ordinal),
            native_terminal_sha256=hashlib.sha256((writer.output/'terminal.json').read_bytes()).hexdigest())
        runs.append(run);life=lifetime(run);life.update(resource_intervals=covref)
        lifes.append(store.write('ledger/lifetime-%03d.json'%ordinal,life))
        line=document['lines'][0]
        costs.append(dict(stream_id=run.stream_id,component_id='vlm',resource_id=line['resource_id'],lane_id='vlm',start_ns=None,end_ns=None,
            event_ns=line['service_event_ns'],identity=dict(kind='invoice',provider=document['provider'],revision=document['revision'],
                account_scope=document['account_scope'],invoice_id=document['invoice_id'],document_sha256=refs['source_document']['file_sha256'],line_id=line['line_id']),
            source_quantity=line['source_quantity'],quantity=line['quantity'],status='settled',billed_amount=line['billed_atomic'],
            evidence_sha256=operation['evidence_sha256'],operation_instance=operation['instance'],lifecycle_work_ref=None,
            received_ns=evidence['observed_received_ns'],financial_evidence_sha256=digest(evidence)))
        receipt=dict(schema_version=1,kind='recording_parent_task_receipt',execution_sha256=nonce,task_id=task['task_id'],
            input_sha256=task['input_sha256'],result_sha256=digest(result),received_ns=received,failed_observed_ns=None,status='answer',
            failure_reason='completed',answer_sha256=digest(result['text']),latency_ns=100000000 if inv['arm']=='A' else 70000000)
        runs[-1]=_issue_financial_run_view(**{**dataclasses.asdict(run),
            'parent_receipt_inventory_sha256':digest([receipt])})
        receipts.append((inv['arm'],receipt));assessment_rows.append(dict(task_id=task['task_id'],input_sha256=task['input_sha256'],
            invocation_ordinal=ordinal,parent_receipt_sha256=digest(receipt),answer_sha256=receipt['answer_sha256'],quality_score=1,
            correct_critical_facts=p['comparison']['tasks'][inv['recording_index']]['critical_facts']))
    costref=store.write('ledger/cost-000000.json',dict(schema_version=2,kind='recording_cost_lines',page_ordinal=0,previous_page_sha256=None,entries=costs))
    for ordinal,run in enumerate(runs):
        lane_rows.append(dict(stream_id=run.stream_id,component_id='vlm',resource_id='resource-'+run.stream_id,lane_id='vlm',
            window_start_ns=run.activation_started_ns,window_end_ns=run.parent_resources_closed_ns,status='closed',
            cost_line_inventory=store.write_index('line-locators',[dict(cost_page=costref,entry_index=ordinal)]),work_inventory_sha256=run.worker_inventory_sha256))
    for ordinal,run in enumerate(runs):
        finality=dict(schema_version=1,kind='synthetic_recording_finality',evidence_class='fixture',authority='gamst.synthetic_invoice_fixture_v1',
            execution_sha256=run.execution_sha256,stream_id=run.stream_id,accounting_policy_sha256=digest(k),
            source_document_sha256=[source_rows[ordinal]['source_document']['file_sha256']],operation_inventory_sha256=digest([operation_rows[ordinal]]),
            lane_closure_inventory_sha256=digest([lane_rows[ordinal]]),service_end_ns=run.parent_resources_closed_ns,issued_ns=time.monotonic_ns(),state='final_no_further_charges')
        finality_rows.append(dict(finality=store.write('financial/finality/finality-%06d.json'%ordinal,finality)))
    fin=dict(schema_version=1,kind='recording_financial_inventory',exposure_policy_sha256=digest(e),accounting_policy_sha256=digest(k),
        financial_adapter_inventory_sha256=digest(f),collection_sha256=upstream_digest(c),source_documents=store.write_index('source',source_rows),
        evidence_pages=store.write_index('evidence',evidence_rows),operation_pages=store.write_index('operations',operation_rows),
        lane_closure_pages=store.write_index('lanes',lane_rows),finality=store.write_index('finality',finality_rows),state=financial_state)
    finref=store.write('ledger/financial-inventory.json',fin);closures=[];seal=0
    for ordinal,run in enumerate(runs):
        seal=time.monotonic_ns()
        closure=dict(schema_version=1,kind='recording_financial_closure',execution_sha256=run.execution_sha256,stream_id=run.stream_id,
            accounting_policy_sha256=digest(k),financial_adapter_inventory_sha256=digest(f),operational_end_ns=run.parent_resources_closed_ns,
            cutoff_ns=run.parent_resources_closed_ns,sealed_ns=seal,operation_count=1,known_settled_operations=1,unresolved_operations=0,
            required_lane_count=1,closed_lane_count=1,pending_line_count=0,unknown_component_count=0,
            external_finality_refs=[finality_rows[ordinal]['finality']],financial_inventory_sha256=digest(fin),state='closed')
        closures.append(store.write('ledger/closure-%03d.json'%ordinal,closure))
    closure_ref=store.write('ledger/closures.json',dict(schema_version=1,kind='recording_financial_closures',closures=closures))
    ledger=dict(schema_version=2,kind='recording_lifetime_ledger',accounting_policy_sha256=digest(k),exposure_policy_sha256=digest(e),
        runtime_policy_sha256=upstream_digest(u),collection_sha256=upstream_digest(c),scope_protocol_sha256=upstream_digest(q),comparison_plan_sha256=plan_digest(p),
        reviewer='fixture',source_evidence_sha256='a'*64,lifetimes=lifes,coverage=coverage_refs,entry_pages=[costref],financial_inventory=finref,closure=closure_ref)
    genesis=dict(schema_version=1,kind='recording_validity_entry',exposure_policy_sha256=digest(e),ledger_sha256=digest(ledger),sequence=0,
        previous_sha256=None,notice=None,state='sealed',observed_ns=seal)
    genref=store.write('ledger/validity-000000.json',genesis)
    checkpoint=dict(schema_version=1,kind='recording_validity_checkpoint',exposure_policy_sha256=digest(e),ledger_sha256=digest(ledger),
        tail_sha256=genref['file_sha256'],tail_sequence=0,reviewed_as_of_ns=time.monotonic_ns(),reviewer='external-fixture',review_method_sha256='b'*64)
    args=dict(policy=k,exposure=e,runtime=u,collection=c,scope_protocol=q,instrumentation=i,observed_runs=runs,native_indices=indices,
        ledger=ledger,financial_inventory=fin,validity_checkpoint=checkpoint,plan=p,recordings=g,adapters_document=f,store=store,
        expected_policy_sha256=digest(k),expected_exposure_sha256=digest(e),expected_ledger_sha256=digest(ledger),
        expected_checkpoint_sha256=digest(checkpoint),validity_entries=[(genref,genesis)],late_notices={})
    assessment=dict(schema_version=1,kind='recording_assessment',comparison_plan_sha256=plan_digest(p),runtime_policy_sha256=upstream_digest(u),
        collection_sha256=upstream_digest(c),execution_inventory_sha256='c'*64,reviewer='fixture',rubric_sha256='d'*64,
        measurement_protocol_sha256='e'*64,source_evidence_sha256='f'*64,rows=assessment_rows)
    return args,assessment,receipts


class ActualNumericFinance(unittest.IsolatedAsyncioTestCase):
    async def test_explicitly_incomplete_original_inventory_cannot_claim_closed_finance(self):
        asyncio.get_running_loop().slow_callback_duration=2
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory).resolve()
            with FinancialArtifacts(root) as store:
                args,_,_=await complete_collection(root,store,financial_state='incomplete')
                with self.assertRaisesRegex(ValueError,'falsely claims closed'):
                    validate_lifetime_ledger(**args)

    async def test_prospective_twenty_four_groups_actual_sdk_settlement_and_full_financial_reconciliation(self):
        asyncio.get_running_loop().slow_callback_duration=2
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory).resolve()
            with FinancialArtifacts(root) as store:
                args,assessment,receipts=await complete_collection(root,store)
                validated=validate_lifetime_ledger(**args)
                self.assertEqual(len(validated.totals),72);self.assertTrue(all(not t.reasons for t in validated.totals))
                quality=compare_quality_latency(args['plan'],assessment,receipts,expected_assessment_sha256=plan_digest(assessment))
                report=reconcile_exposure(args['plan'],args['exposure'],args['recordings'],validated,quality,
                    runtime_policy_sha256=upstream_digest(args['runtime']),collection_sha256=upstream_digest(args['collection']),
                    scope_protocol_sha256=upstream_digest(args['scope_protocol']),instrumentation_protocol_sha256=upstream_digest(args['instrumentation']),
                    execution_inventory_sha256='c'*64)
                self.assertEqual(report['state'],'complete');self.assertEqual(report['canonical_arm_totals']['A']['full_monetary_atomic'],240)
                self.assertTrue(report['comparison']['B_vs_A']['cost_gate']);self.assertTrue(report['comparison']['C_vs_A']['cost_gate'])
                # Original synthetic-P measured/heldout gate is preserved.
                self.assertFalse(report['comparison']['B_vs_A']['overall_gate']);self.assertFalse(report['comparison']['C_vs_A']['overall_gate'])
                self.assertEqual(report['evidence_class'],'fixture');self.assertFalse(report['truth_flags']['receipt_authenticity_verified'])
                # Caller strings and a newly reviewed assessment cannot retarget
                # a validated monetary ledger to another runtime or receipt set.
                original=dict(runtime_policy_sha256=upstream_digest(args['runtime']),collection_sha256=upstream_digest(args['collection']),
                    scope_protocol_sha256=upstream_digest(args['scope_protocol']),instrumentation_protocol_sha256=upstream_digest(args['instrumentation']),
                    execution_inventory_sha256='c'*64)
                forged_plan=copy.deepcopy(args['plan'])
                forged_plan['comparison'].update(evidence='measured',held_out=True)
                forged_assessment=copy.deepcopy(assessment)
                forged_assessment['comparison_plan_sha256']=plan_digest(forged_plan)
                alternate=compare_quality_latency(forged_plan,forged_assessment,receipts,
                    expected_assessment_sha256=plan_digest(forged_assessment))
                with self.assertRaisesRegex(ValueError,'original policy/assessment'):
                    reconcile_exposure(forged_plan,args['exposure'],args['recordings'],validated,alternate,**original)
                forged_recordings=copy.deepcopy(args['recordings']);forged_recordings['groups'].reverse()
                with self.assertRaisesRegex(ValueError,'original policy/assessment'):
                    reconcile_exposure(args['plan'],args['exposure'],forged_recordings,validated,quality,**original)
                for field in original:
                    changed=dict(original);changed[field]='9'*64
                    forged=copy.deepcopy(assessment)
                    if field in forged:forged[field]=changed[field]
                    alternate=compare_quality_latency(args['plan'],forged,receipts,expected_assessment_sha256=plan_digest(forged))
                    with self.subTest(crossed=field),self.assertRaisesRegex(ValueError,'original validated ledger'):
                        reconcile_exposure(args['plan'],args['exposure'],args['recordings'],validated,alternate,**changed)
                for mutation in ('execution','receipt','invocation'):
                    forged=copy.deepcopy(assessment);foreign=copy.deepcopy(receipts)
                    if mutation=='execution':foreign[0][1]['execution_sha256']='9'*64
                    elif mutation=='receipt':foreign[0][1]['received_ns']+=1
                    else:forged['rows'][0]['invocation_ordinal']=1
                    forged['rows'][0]['parent_receipt_sha256']=digest(foreign[0][1])
                    alternate=compare_quality_latency(args['plan'],forged,foreign,expected_assessment_sha256=plan_digest(forged))
                    with self.subTest(crossed=mutation),self.assertRaisesRegex(ValueError,'execution/parent receipt inventory'):
                        reconcile_exposure(args['plan'],args['exposure'],args['recordings'],validated,alternate,**original)

            # A syntactically valid attachment cannot replace the actual SDK inventory.
            values=dataclasses.asdict(args['observed_runs'][0]);values['controller_settlement_inventory_sha256']='a'*64
            args['observed_runs']=[_issue_financial_run_view(**values),*args['observed_runs'][1:]]
            with FinancialArtifacts(root) as reader:
                args['store']=reader
                with self.assertRaisesRegex(ValueError,'actual SDK-issued inventory'):validate_lifetime_ledger(**args)


if __name__=='__main__':unittest.main()
