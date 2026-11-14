import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.paired_replay import _encoded
from benchmarks.perception_comparison import plan_digest
from benchmarks.stream_attempts import StreamAttemptRecorder
from benchmarks.stream_reconciliation import join
from benchmarks.three_arm_attempts import ThreeArmAttemptRecorder
from benchmarks.three_arm_evidence import read_traces
from test_stream_cost_ledger import fixture as ledger_fixture
from test_three_arm_replay import setup, collect


def rehash(report):
    report['sha256']=plan_digest({k:v for k,v in report.items() if k!='sha256'})
    return report


@unittest.skipUnless(os.name=='posix','actual collector requires POSIX')
class StreamReconciliation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name).resolve()
        self.plan,self.spec,self.inputs=setup(self.root,('ok','error'))
        self.plan['comparison']['cost']=dict(mode='itemized',currency='USD',scope='full_pipeline',
            accounting_protocol='Explicit synthetic all-component fixture',rate_card={'request':dict(
                provider='remote',revision='frozen-v1',unit='attempt',category='request',per_unit=1)})
        self.spec['comparison_plan_sha256']=plan_digest(self.plan)
        self.stream,self.ledger=ledger_fixture()
        self.stream.update(comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec),
            rates={'request':dict(provider='remote',revision='frozen-v1',unit='attempt',per_unit='1')})
        for component in self.stream['components']:
            component.update(kind='monetary',unit='attempt',source_unit='attempt',rate_id='request',
                             coverage='event' if component['id'] in ('perception','startup') else 'interval')
        self.ledger.update(comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec),entries=[])
        targets,rules,owners,groups,vlm=[],[],[],[],[]
        for frozen,observed in zip(self.stream['streams'],self.ledger['streams']):
            arm,sid=frozen['arm'],frozen['id']
            for cid,requirement in frozen['requirements'].items():
                required=cid in ('perception','resident_idle','startup')
                requirement.update(state='required' if required else 'not_applicable',reason='Explicit fixture boundary',
                    lanes=[dict(resource_id=arm+'-device',lane_id='task-'+str(i)) for i in range(2)] if cid=='perception' else
                          [dict(resource_id=arm+'-device',lane_id=cid)] if required else [])
            observed['counters'][0]['end']['vlm_calls']=2
            targets.extend(dict(id=arm+'-task-'+str(i),stream_id=sid,kind='task',task_id=str(i),windows=[]) for i in range(2))
            targets.extend(dict(id=arm+'-'+kind,stream_id=sid,kind=kind,task_id=None,windows=[]) for kind in ('startup','idle','overhead'))
            for cid,requirement in frozen['requirements'].items():
                for lane in requirement['lanes']:
                    kind='task_attempt' if cid=='perception' else 'stream_attempt' if cid=='startup' else 'non_attempt'
                    target=arm+'-'+lane['lane_id'] if cid=='perception' else arm+('-startup' if cid=='startup' else '-idle')
                    rules.append(dict(stream_id=sid,component_id=cid,**lane,method='direct',target_id=target))
                    owners.append(dict(stream_id=sid,component_id=cid,**lane,owner_kind=kind))
                    quantity=1 if cid=='perception' else 2 if cid=='startup' else 3
                    event=cid in ('perception','startup')
                    self.ledger['entries'].append(dict(stream_id=sid,component_id=cid,**lane,
                        start_ns=None if event else observed['start_ns'],end_ns=None if event else observed['end_ns'],
                        event_ns=observed['start_ns'] if event else None,
                        identity=dict(kind='invoice',provider='remote',revision='frozen-v1',account_scope='fixture',
                            invoice_id='invoice-'+arm,document_sha256=plan_digest(['doc',arm]),line_id=lane['lane_id']),
                        source_quantity=dict(numerator=quantity,denominator=1),quantity=quantity,status='settled',
                        billed_amount=quantity,evidence_sha256='b'*64))
            for role in ('strong_vlm','perception'):
                groups.append(dict(id=sid+'-'+role,stream_id=sid,provider='remote',revision='frozen-v1',role=role))
            vlm.append(dict(stream_id=sid,source_id='camera',counter_ids=[sid+'-strong_vlm']))
        self.ledger['protocol_sha256']=plan_digest(self.stream)
        self.allocation=dict(schema_version=1,comparison_plan_sha256=plan_digest(self.plan),stream_protocol_sha256=plan_digest(self.stream),
            rounding='largest_remainder_v1',tie_break='unicode_codepoint_target_id',duration_model='uniform_line_quantity_over_interval',
            targets=targets,rules=rules)
        self.protocol=dict(schema_version=1,comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec),
            stream_protocol_sha256=plan_digest(self.stream),allocation_protocol_sha256=plan_digest(self.allocation),
            currency_atomic_per_comparison_unit=dict(numerator=1,denominator=1),
            rate_map=[dict(component_id=c['id'],comparison_rate_id='request',atomic_quantity_per_comparison_unit=dict(numerator=1,denominator=1))
                      for c in self.stream['components']],lane_owners=owners,provider_counter_groups=groups,vlm_counter_bindings=vlm,
            recording_groups=[dict(recording_sha256='5'*64,streams={arm:'stream-'+arm for arm in 'ABC'})],
            whole_stream_cost_policy=dict(version='matched_recording_elapsed_hour_v1',threshold=dict(numerator=4,denominator=5),
                interval='nearest_rank_95',bootstrap_seed=self.plan['comparison']['bootstrap_seed'],
                bootstrap_repetitions=self.plan['comparison']['bootstrap_repetitions'],min_recordings=self.plan['comparison']['min_pairs']))
        self.run_path=self.root/'run';self.completion=await collect(self.plan,self.spec,self.inputs,self.run_path)
        self.traces=list(read_traces(self.run_path/'source-traces.jsonl'))
        self.reports=[];self.task_reports={};self.stream_reports={}
        for trace in self.traces:
            rec=self.recorder('task',trace['task_id'],trace['arm'])
            line=next(e for e in self.ledger['entries'] if e['stream_id']=='stream-'+trace['arm'] and e['lane_id']=='task-'+trace['task_id'])
            async def callback(token):
                rec.report_lines(token,quantities={'request':1},line_claims=[plan_digest(line)],execution_instance_sha256=plan_digest(['instance',trace['arm'],trace['task_id']]))
                if trace['status']!='success':raise RuntimeError('fixture failure still costs')
            try:await rec.invoke(callback,provider='remote',revision='frozen-v1',role='strong_vlm',phase='cold',evidence_sha256=trace['input_sha256'])
            except RuntimeError:pass
            rec.close();report=rec.snapshot();self.reports.append(report);self.task_reports[trace['task_id'],trace['arm']]=report
        for arm in 'ABC':
            rec=self.recorder('stream',None,arm);line=next(e for e in self.ledger['entries'] if e['stream_id']=='stream-'+arm and e['component_id']=='startup')
            async def startup(token):rec.report_lines(token,quantities={'request':2},line_claims=[plan_digest(line)],execution_instance_sha256=plan_digest(['stream-instance',arm]))
            await rec.invoke(startup,provider='remote',revision='frozen-v1',role='perception',phase='cold',evidence_sha256='5'*64)
            rec.close();report=rec.snapshot();self.reports.append(report);self.stream_reports[arm]=report
        self.assessment=dict(schema_version=1,comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec),
            last_trace_sha256=self.completion['last_trace_sha256'],reviewer='fixture scorer',rubric_sha256='1'*64,
            measurement_protocol_sha256='2'*64,source_evidence_sha256='3'*64,entries=[dict(trace_sha256=t['trace_sha256'],
                quality_score=int(t['status']=='success'),correct_critical_facts=['fact'] if t['status']=='success' else [],
                usage=dict(calls=1,input_tokens=1,output_tokens=1,cache_read_tokens=0,cache_write_tokens=0),
                peak_memory_bytes=100,dropped_frames=0,resource_cost=None,cost_quantities={'request':1}) for t in self.traces])
        self.manifest=dict(schema_version=2,comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec),
            stream_protocol_sha256=plan_digest(self.stream),allocation_protocol_sha256=plan_digest(self.allocation),
            reconciliation_protocol_sha256=plan_digest(self.protocol),last_trace_sha256=self.completion['last_trace_sha256'],
            ledger_sha256=plan_digest(self.ledger),assessment_sha256=plan_digest(self.assessment),reviewer='fixture owner audit',
            coverage_evidence_sha256='4'*64,
            trace_reports=[dict(trace_sha256=t['trace_sha256'],report_sha256=self.task_reports[t['task_id'],t['arm']]['sha256']) for t in self.traces],
            stream_reports=[dict(stream_id='stream-'+a,report_sha256=self.stream_reports[a]['sha256']) for a in 'ABC'],
            attempt_instances=[dict(report_sha256=r['sha256'],attempt_id=a['attempt_id'],execution_instance_sha256=a['execution_instance_sha256'],evidence_sha256='6'*64)
                               for r in self.reports for a in r['attempts']],line_owners=[],
            provider_counters=[dict(counter_id=g['id'],count=2 if g['role']=='strong_vlm' else 1,evidence_sha256='7'*64) for g in groups])
        for entry in self.ledger['entries']:
            arm=entry['stream_id'][-1];cid=entry['component_id'];owner=dict(entry_sha256=plan_digest(entry))
            if cid=='resident_idle':owner.update(owner_kind='non_attempt',reason='Frozen meter outside callback',evidence_sha256='8'*64)
            else:
                report=self.task_reports[entry['lane_id'][-1],arm] if cid=='perception' else self.stream_reports[arm]
                owner.update(owner_kind='task_attempt' if cid=='perception' else 'stream_attempt',report_sha256=report['sha256'],attempt_id=report['attempts'][0]['attempt_id'])
            self.manifest['line_owners'].append(owner)

    def recorder(self,kind,task,arm):
        return StreamAttemptRecorder(self.plan,self.spec,self.inputs,self.stream,self.protocol,
            expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            expected_stream_protocol_sha256=plan_digest(self.stream),expected_reconciliation_protocol_sha256=plan_digest(self.protocol),
            scope_kind=kind,task_id=task,arm=arm,stream_id='stream-'+arm)

    def joined(self,**overrides):
        args=dict(plan=self.plan,spec=self.spec,inputs=self.inputs,traces=self.traces,completion=self.completion,
            assessment=self.assessment,stream_protocol=self.stream,ledger=self.ledger,allocation_protocol=self.allocation,
            reconciliation_protocol=self.protocol,manifest=self.manifest,reports=self.reports,
            expected_plan_sha256=plan_digest(self.plan),expected_collection_sha256=plan_digest(self.spec),
            expected_trace_sha256=self.completion['last_trace_sha256'],expected_assessment_sha256=plan_digest(self.assessment),
            expected_stream_protocol_sha256=plan_digest(self.stream),expected_ledger_sha256=plan_digest(self.ledger),
            expected_allocation_protocol_sha256=plan_digest(self.allocation),expected_reconciliation_protocol_sha256=plan_digest(self.protocol),
            expected_manifest_sha256=plan_digest(self.manifest))
        args.update(overrides);return join(**args)

    def refresh(self):
        self.stream.update(comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec))
        self.ledger.update(protocol_sha256=plan_digest(self.stream),comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec))
        self.allocation.update(comparison_plan_sha256=plan_digest(self.plan),stream_protocol_sha256=plan_digest(self.stream))
        self.protocol.update(comparison_plan_sha256=plan_digest(self.plan),collection_sha256=plan_digest(self.spec),
            stream_protocol_sha256=plan_digest(self.stream),allocation_protocol_sha256=plan_digest(self.allocation))
        changed={}
        for report in self.reports:
            old=report['sha256']
            if report['schema_version']==3:report.update(stream_protocol_sha256=plan_digest(self.stream),reconciliation_protocol_sha256=plan_digest(self.protocol))
            rehash(report);changed[old]=report['sha256']
        for field in ('trace_reports','stream_reports','attempt_instances','line_owners'):
            for item in self.manifest[field]:
                if 'report_sha256' in item:item['report_sha256']=changed.get(item['report_sha256'],item['report_sha256'])
        self.manifest.update(stream_protocol_sha256=plan_digest(self.stream),allocation_protocol_sha256=plan_digest(self.allocation),
            reconciliation_protocol_sha256=plan_digest(self.protocol),ledger_sha256=plan_digest(self.ledger),assessment_sha256=plan_digest(self.assessment))

    def changed_lines(self,before):
        changed={plan_digest(old):plan_digest(new) for old,new in zip(before,self.ledger['entries'])}
        for report in self.reports:
            for attempt in report['attempts']:
                if 'line_claims' in attempt:attempt['line_claims']=[changed.get(d,d) for d in attempt['line_claims']]
        for owner in self.manifest['line_owners']:owner['entry_sha256']=changed.get(owner['entry_sha256'],owner['entry_sha256'])
        self.refresh()

    async def replace_v2(self,index=0,exclusive=False):
        old=self.reports[index];arm=old['arm'];task=old['task_id']
        entry=next(e for e in self.ledger['entries'] if e['stream_id']=='stream-'+arm and e['lane_id']=='task-'+task)
        if exclusive:
            before=copy.deepcopy(self.ledger['entries']);entry['identity'].update(invoice_id='exclusive-'+arm+'-'+task,document_sha256='f'*64)
            self.changed_lines(before);old=self.reports[index]
        rec=ThreeArmAttemptRecorder(self.plan,self.spec,self.inputs,expected_plan_sha256=plan_digest(self.plan),
            expected_collection_sha256=plan_digest(self.spec),task_id=task,arm=arm)
        async def callback(token):rec.report(token,quantities={'request':1},receipt_sha256=entry['identity']['document_sha256'])
        await rec.invoke(callback,provider='remote',revision='frozen-v1',role='strong_vlm',phase='cold',evidence_sha256=old['input_sha256'])
        rec.close();new=rec.snapshot();self.reports[index]=new
        for field in ('trace_reports','stream_reports','attempt_instances','line_owners'):
            for item in self.manifest[field]:
                if item.get('report_sha256')==old['sha256']:
                    item['report_sha256']=new['sha256']
                    if 'attempt_id' in item:item['attempt_id']=new['attempts'][0]['attempt_id']
        self.refresh()

    async def test_actual_collector_native_reports_canonical_costs_failures_and_immutability(self):
        before=copy.deepcopy((self.plan,self.spec,self.inputs,self.traces,self.completion,self.assessment,self.stream,self.ledger,self.allocation,self.protocol,self.manifest,self.reports))
        originals={p.name:p.read_bytes() for p in self.run_path.iterdir()}
        result=self.joined()
        self.assertTrue(result['declared_reconciliation_complete'])
        self.assertEqual(result['actual_acceptance'],'incomplete')
        self.assertEqual(len(result['rows']),6)
        self.assertEqual(len(result['reconciliation_lineage']['attempts']),9)
        self.assertEqual(len(result['reconciliation_lineage']['lines']),12)
        self.assertEqual(result['canonical_stream_costs']['A']['known_monetary_atomic'],7)
        self.assertEqual(sum(v['known_monetary_atomic'] for v in result['canonical_stream_costs'].values()),21)
        for row,trace in zip(result['rows'],self.traces):
            self.assertEqual(row['cost_quantities'],{'request':1})
            if trace['status']=='error':
                self.assertEqual(row['quality_score'],0);self.assertEqual(row['correct_critical_facts'],[])
                self.assertGreaterEqual(row['latency_ms'],self.spec['runner']['timeout_ms'])
        self.assertFalse(result['primary_meets_reconciled_numeric_gates'])
        for field in ('receipt_authenticity_verified','instrumentation_coverage_verified','isolation_verified','whole_pipeline_cost_verified'):
            self.assertFalse(result[field])
        self.assertEqual(before,(self.plan,self.spec,self.inputs,self.traces,self.completion,self.assessment,self.stream,self.ledger,self.allocation,self.protocol,self.manifest,self.reports))
        self.assertEqual(originals,{p.name:p.read_bytes() for p in self.run_path.iterdir()})

    async def test_all_frozen_digest_bindings_and_manifest_inventory(self):
        for name in ('plan','collection','trace','assessment','stream_protocol','ledger','allocation_protocol','reconciliation_protocol','manifest'):
            with self.subTest(name=name),self.assertRaises(ValueError):self.joined(**{'expected_'+name+'_sha256':'f'*64})
        for field in ('trace_reports','stream_reports','attempt_instances','line_owners','provider_counters'):
            for mode in ('missing','duplicate'):
                manifest=copy.deepcopy(self.manifest)
                if mode=='missing':manifest[field].pop()
                else:manifest[field].append(copy.deepcopy(manifest[field][0]))
                with self.subTest(field=field,mode=mode),self.assertRaises(ValueError):self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        for reports in (self.reports[:-1],self.reports+[self.reports[0]]):
            with self.assertRaises(ValueError):self.joined(reports=reports)

    async def test_provider_counter_and_execution_instance_fail_closed(self):
        for value in (None,True,0,3):
            manifest=copy.deepcopy(self.manifest);manifest['provider_counters'][0]['count']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        for field in ('execution_instance_sha256','evidence_sha256'):
            manifest=copy.deepcopy(self.manifest);manifest['attempt_instances'][0][field]=None
            with self.assertRaises(ValueError):self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        manifest=copy.deepcopy(self.manifest);manifest['attempt_instances'][1]['execution_instance_sha256']=manifest['attempt_instances'][0]['execution_instance_sha256']
        with self.assertRaises(ValueError):self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        self.ledger['streams'][0]['counters'][0]['end']['vlm_calls']=3;self.refresh()
        with self.assertRaisesRegex(ValueError,'VLM delta'):self.joined()

    async def test_task_ownership_cannot_move_to_overhead_or_cross_arm(self):
        manifest=copy.deepcopy(self.manifest)
        first=next(o for o in manifest['line_owners'] if o['owner_kind']=='task_attempt')
        other=next(r for r in self.reports if r['schema_version']==3 and r['scope_kind']=='task' and r['arm']=='B')
        first.update(report_sha256=other['sha256'],attempt_id=other['attempts'][0]['attempt_id'])
        with self.assertRaisesRegex(ValueError,'migrate'):self.joined(manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        rule=next(r for r in self.allocation['rules'] if r['stream_id']=='stream-A' and r['lane_id']=='task-1')
        rule['target_id']='A-overhead';self.refresh()
        with self.assertRaisesRegex(ValueError,'task-owned line'):self.joined()

    async def test_task_owned_fixed_weight_one_is_not_a_frozen_direct_rule(self):
        rule=next(r for r in self.allocation['rules'] if r['stream_id']=='stream-A' and r['lane_id']=='task-0')
        target=rule.pop('target_id');rule.update(method='fixed',shares=[dict(target_id=target,numerator=1,denominator=1)])
        self.refresh()
        with self.assertRaisesRegex(ValueError,'task-owned line'):self.joined()

    async def test_explicit_empty_stream_scopes_still_require_reports_and_counters(self):
        empty_hashes={r['sha256'] for r in self.reports if r.get('scope_kind')=='stream'}
        for report in self.reports:
            if report['sha256'] in empty_hashes:report['attempts']=[]
        self.manifest['attempt_instances']=[i for i in self.manifest['attempt_instances'] if i['report_sha256'] not in empty_hashes]
        for owner in self.manifest['line_owners']:
            if owner['owner_kind']=='stream_attempt':
                owner.pop('report_sha256');owner.pop('attempt_id')
                owner.update(owner_kind='non_attempt',reason='Explicit frozen startup meter, no wrapped callback',evidence_sha256='9'*64)
        for lane in self.protocol['lane_owners']:
            if lane['owner_kind']=='stream_attempt':lane['owner_kind']='non_attempt'
        for counter in self.manifest['provider_counters']:
            if counter['counter_id'].endswith('-perception'):counter['count']=0
        self.refresh();result=self.joined()
        self.assertEqual(len(result['reconciliation_lineage']['attempts']),6)
        self.assertEqual(len(result['reconciliation_lineage']['reports']),9)
        self.assertEqual(result['canonical_stream_costs']['A']['known_monetary_atomic'],7)
        self.assertFalse(result['instrumentation_coverage_verified'])
        self.manifest['stream_reports'].pop()
        with self.assertRaises(ValueError):self.joined()

    async def test_native_scope_headers_and_failed_credit_are_revalidated(self):
        for field,value in (('task_id','1'),('scope_kind','stream'),('arm','B'),('stream_id','stream-B'),
                            ('input_sha256','f'*64),('configuration_sha256','f'*64),('recording_sha256','f'*64),
                            ('reconciliation_protocol_sha256','f'*64)):
            reports=copy.deepcopy(self.reports);manifest=copy.deepcopy(self.manifest)
            old=reports[0]['sha256'];reports[0][field]=value;rehash(reports[0])
            for name in ('trace_reports','attempt_instances','line_owners'):
                for item in manifest[name]:
                    if item.get('report_sha256')==old:item['report_sha256']=reports[0]['sha256']
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.joined(reports=reports,manifest=manifest,expected_manifest_sha256=plan_digest(manifest))
        assessment=copy.deepcopy(self.assessment);assessment['entries'][-1]['quality_score']=1
        manifest=copy.deepcopy(self.manifest);manifest['assessment_sha256']=plan_digest(assessment)
        with self.assertRaisesRegex(ValueError,'failed tasks'):
            self.joined(assessment=assessment,manifest=manifest,expected_assessment_sha256=plan_digest(assessment),
                        expected_manifest_sha256=plan_digest(manifest))
        with self.assertRaises(ValueError):self.joined(traces=self.traces[:-1])

    async def test_inventory_limits_precede_expensive_report_validation(self):
        reports=[copy.deepcopy(self.reports[0]) for _ in range(11)]
        for report in reports:report['attempts']*=1024
        self.assertLess(len(_encoded(reports)),16*1024*1024)
        with self.assertRaisesRegex(ValueError,'total canonical attempts'):
            self.joined(reports=reports)
        with patch('benchmarks.stream_reconciliation.MAX_REFERENCES',1):
            with self.assertRaisesRegex(ValueError,'cross-references'):self.joined()

    async def test_unknown_pending_missing_cost_and_assessment_totals_fail_closed(self):
        for state in ('pending','unknown'):
            ledger=copy.deepcopy(self.ledger);ledger['entries'][0].update(status=state,quantity=None,source_quantity=None,billed_amount=None,evidence_sha256=None)
            manifest=copy.deepcopy(self.manifest);manifest['ledger_sha256']=plan_digest(ledger)
            with self.assertRaisesRegex(ValueError,'unknown/pending'):
                self.joined(ledger=ledger,manifest=manifest,expected_ledger_sha256=plan_digest(ledger),expected_manifest_sha256=plan_digest(manifest))
        assessment=copy.deepcopy(self.assessment);assessment['entries'][0]['cost_quantities']['request']=2
        manifest=copy.deepcopy(self.manifest);manifest['assessment_sha256']=plan_digest(assessment)
        with self.assertRaisesRegex(ValueError,'exactly once'):self.joined(assessment=assessment,manifest=manifest,
            expected_assessment_sha256=plan_digest(assessment),expected_manifest_sha256=plan_digest(manifest))
        reports=copy.deepcopy(self.reports);reports[0]['closed']=False;old=reports[0]['sha256'];rehash(reports[0])
        manifest=copy.deepcopy(self.manifest)
        for field in ('trace_reports','attempt_instances','line_owners'):
            for item in manifest[field]:
                if item.get('report_sha256')==old:item['report_sha256']=reports[0]['sha256']
        with self.assertRaisesRegex(ValueError,'closed native'):self.joined(reports=reports,manifest=manifest,expected_manifest_sha256=plan_digest(manifest))

    async def test_native2_exclusive_document_compatibility_and_shared_document_rejection(self):
        await self.replace_v2(exclusive=True)
        self.assertTrue(self.joined()['declared_reconciliation_complete'])
        # A distinct zero-charge line in the same document still belongs to that
        # whole-document v2 owner; another owner cannot hide it as free overhead.
        source=next(e for e in self.ledger['entries'] if e['identity']['document_sha256']=='f'*64)
        extra=copy.deepcopy(next(e for e in self.ledger['entries'] if e['stream_id']=='stream-A' and e['component_id']=='startup'))
        extra['identity']=dict(source['identity'],line_id='zero-other-owner')
        extra.update(quantity=0,source_quantity=dict(numerator=0,denominator=1),billed_amount=0)
        self.ledger['entries'].append(extra)
        stream_report=next(r for r in self.reports if r.get('scope_kind')=='stream' and r['arm']=='A')
        stream_report['attempts'][0]['line_claims'].append(plan_digest(extra))
        self.manifest['line_owners'].append(dict(entry_sha256=plan_digest(extra),owner_kind='stream_attempt',
            report_sha256=stream_report['sha256'],attempt_id=stream_report['attempts'][0]['attempt_id']))
        self.refresh()
        with self.assertRaisesRegex(ValueError,'whole-document exclusivity'):self.joined()

    async def test_native2_whole_receipt_never_becomes_shared_line_claim(self):
        await self.replace_v2(exclusive=False)
        with self.assertRaisesRegex(ValueError,'whole-document exclusivity'):self.joined()

    async def test_per_line_fractional_attribution_cannot_cancel_discrepancies(self):
        before=copy.deepcopy(self.ledger['entries'])
        self.stream['rates']['request']['per_unit']='2'
        self.protocol['currency_atomic_per_comparison_unit']=dict(numerator=2,denominator=1)
        for entry in self.ledger['entries']:
            if entry['stream_id']=='stream-A' and entry['component_id']=='resident_idle':
                entry.update(quantity=1,source_quantity=dict(numerator=1,denominator=1))
            entry['billed_amount']=entry['quantity']*2
        rule=next(r for r in self.allocation['rules'] if r['stream_id']=='stream-A' and r['component_id']=='resident_idle')
        del rule['target_id'];rule.update(method='fixed',shares=[dict(target_id='A-task-'+str(i),numerator=1,denominator=2) for i in range(2)])
        self.changed_lines(before)
        with self.assertRaisesRegex(ValueError,'per-line billed/rate discrepancy'):self.joined()

    async def test_rate_map_currency_proxy_and_frozen_lane_contracts(self):
        for mode in ('missing_rate','unknown_rate','wrong_conversion','wrong_currency','forbidden_nonattempt'):
            protocol=copy.deepcopy(self.protocol);stream=copy.deepcopy(self.stream)
            if mode=='missing_rate':protocol['rate_map'].pop()
            if mode=='unknown_rate':protocol['rate_map'][0]['comparison_rate_id']='unknown'
            if mode=='wrong_conversion':protocol['rate_map'][0]['atomic_quantity_per_comparison_unit']['numerator']=2
            if mode=='wrong_currency':stream['currency']='EUR'
            if mode=='forbidden_nonattempt':
                stream['components'][0]['category']='vlm'
                # Preserve category coverage by changing an unused component's category.
                next(c for c in stream['components'] if c['id']=='vlm')['category']='perception'
                protocol['lane_owners'][0]['owner_kind']='non_attempt'
            old_stream,old_protocol=self.stream,self.protocol
            self.stream,self.protocol=stream,protocol;self.refresh()
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.joined()
            self.stream,self.protocol=old_stream,old_protocol;self.refresh()
        before=copy.deepcopy(self.ledger['entries'])
        component=next(c for c in self.stream['components'] if c['id']=='resident_idle');component.update(kind='proxy',rate_id=None)
        self.protocol['rate_map']=[m for m in self.protocol['rate_map'] if m['component_id']!='resident_idle']
        for entry in self.ledger['entries']:
            if entry['component_id']=='resident_idle':entry.update(status='measured',billed_amount=None,
                identity=dict(kind='local',meter_id=entry['stream_id'],measurement_id='resident'))
        self.changed_lines(before)
        with self.assertRaisesRegex(ValueError,'proxy'):self.joined()

    async def test_compact_cli_immutable_exclusive_and_fail_before_output(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory);paths={}
            for name,value in (('assessment',self.assessment),('stream-protocol',self.stream),('ledger',self.ledger),
                               ('allocation-protocol',self.allocation),('reconciliation-protocol',self.protocol),('manifest',self.manifest),('reports',self.reports)):
                paths[name]=root/(name+'.json');paths[name].write_bytes(_encoded(value)+b'\n')
            originals={str(p):p.read_bytes() for p in list(paths.values())+list(self.run_path.iterdir())}
            cmd=[sys.executable,'-m','benchmarks.stream_reconciliation','--run',str(self.run_path)]
            for name,path in paths.items():cmd.extend(['--'+name,str(path)])
            for name,digest in (('plan',plan_digest(self.plan)),('collection',plan_digest(self.spec)),('trace',self.completion['last_trace_sha256']),
                ('assessment',plan_digest(self.assessment)),('stream-protocol',plan_digest(self.stream)),('ledger',plan_digest(self.ledger)),
                ('allocation-protocol',plan_digest(self.allocation)),('reconciliation-protocol',plan_digest(self.protocol)),('manifest',plan_digest(self.manifest))):
                cmd.extend(['--'+name+'-sha256',digest])
            output=root/'out';cmd.extend(['--output',str(output)])
            run=await asyncio.to_thread(subprocess.run,cmd,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            report=(output/'report.json').read_bytes();self.assertEqual(run.stdout.encode('ascii'),report)
            self.assertEqual(json.loads(report),self.joined())
            again=await asyncio.to_thread(subprocess.run,cmd,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(again.returncode,0);self.assertEqual((output/'report.json').read_bytes(),report)
            self.assertEqual(originals,{p:Path(p).read_bytes() for p in originals})
            for i,payload in enumerate((b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b' '*(16*1024*1024+1))):
                paths['manifest'].write_bytes(payload);bad=root/('bad-'+str(i))
                result=await asyncio.to_thread(subprocess.run,cmd[:-1]+[str(bad)],capture_output=True,text=True,timeout=20)
                self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())
        result=self.joined();size=len(_encoded(result))+1
        with patch('benchmarks.stream_reconciliation.MAX_REPORT_BYTES',size):self.joined()
        with patch('benchmarks.stream_reconciliation.MAX_REPORT_BYTES',size-1):
            with self.assertRaisesRegex(ValueError,'serialized reconciliation'):self.joined()
