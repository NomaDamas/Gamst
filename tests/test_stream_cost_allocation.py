import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.paired_replay import _encoded
from benchmarks.perception_comparison import plan_digest
from benchmarks.stream_cost_allocation import allocate
from test_stream_cost_ledger import fixture as ledger_fixture
from test_three_arm_comparison import fixture as comparison_fixture


def fixture():
    plan,_=comparison_fixture('synthetic')
    plan['comparison']['tasks']=plan['comparison']['tasks'][:2]
    stream,ledger=ledger_fixture()
    stream['comparison_plan_sha256']=plan_digest(plan)
    ledger['comparison_plan_sha256']=plan_digest(plan)
    targets,rules=[],[]
    for frozen,observed in zip(stream['streams'],ledger['streams']):
        arm,sid=frozen['arm'],frozen['id']
        frozen['requested_duration_ns']=8
        observed.update(start_ns=0,end_ns=8,utc_anchor=dict(monotonic_ns=0,unix_ns=100,evidence_sha256='7'*64),
            availability=[dict(start_ns=0,end_ns=8,state='available',reason='Fixture available',evidence_sha256='8'*64)])
        targets.extend([dict(id=arm+'-task-0',stream_id=sid,kind='task',task_id='0',
                windows=[dict(start_offset_ns=0,end_offset_ns=6)]),
            dict(id=arm+'-task-1',stream_id=sid,kind='task',task_id='1',
                windows=[dict(start_offset_ns=2,end_offset_ns=4)])])
        targets.extend(dict(id=arm+'-'+kind,stream_id=sid,kind=kind,task_id=None,windows=[])
                       for kind in ('startup','idle','overhead'))
        for cid,req in frozen['requirements'].items():
            for lane in req['lanes']:
                rule=dict(stream_id=sid,component_id=cid,**lane)
                if cid=='perception':
                    rule.update(method='duration',task_target_ids=[arm+'-task-0',arm+'-task-1'],idle_target_id=arm+'-idle')
                elif cid=='capture_sensor':
                    rule.update(method='fixed',shares=[dict(target_id=arm+'-task-'+str(i),numerator=1,denominator=2) for i in range(2)])
                else:
                    rule.update(method='direct',target_id=arm+('-startup' if cid=='startup' else '-idle'))
                rules.append(rule)
    for entry in ledger['entries']:
        if entry['event_ns'] is not None:entry['event_ns']=0
        else:entry.update(start_ns=0,end_ns=8)
        if entry['component_id']=='perception':
            entry.update(quantity=7,source_quantity=dict(numerator=7,denominator=1),billed_amount=4)
    ledger['protocol_sha256']=plan_digest(stream)
    allocation=dict(schema_version=1,comparison_plan_sha256=plan_digest(plan),stream_protocol_sha256=plan_digest(stream),
        rounding='largest_remainder_v1',tie_break='unicode_codepoint_target_id',duration_model='uniform_line_quantity_over_interval',
        targets=targets,rules=rules)
    return plan,stream,ledger,allocation


class StreamCostAllocation(unittest.TestCase):
    def setUp(self):self.plan,self.stream,self.ledger,self.allocation=fixture()

    def allocated(self,plan=None,stream=None,ledger=None,allocation=None,**kwargs):
        plan=copy.deepcopy(self.plan if plan is None else plan)
        stream=copy.deepcopy(self.stream if stream is None else stream)
        ledger=copy.deepcopy(self.ledger if ledger is None else ledger)
        allocation=copy.deepcopy(self.allocation if allocation is None else allocation)
        stream['comparison_plan_sha256']=plan_digest(plan)
        ledger['comparison_plan_sha256']=plan_digest(plan);ledger['protocol_sha256']=plan_digest(stream)
        allocation['comparison_plan_sha256']=plan_digest(plan);allocation['stream_protocol_sha256']=plan_digest(stream)
        args=dict(expected_comparison_plan_sha256=plan_digest(plan),expected_stream_protocol_sha256=plan_digest(stream),
                  expected_ledger_sha256=plan_digest(ledger),expected_allocation_protocol_sha256=plan_digest(allocation))
        args.update(kwargs)
        return allocate(plan,stream,ledger,allocation,**args)

    def test_frozen_digests_and_exact_schema(self):
        for field in ('expected_comparison_plan_sha256','expected_stream_protocol_sha256','expected_ledger_sha256',
                      'expected_allocation_protocol_sha256'):
            with self.subTest(field=field),self.assertRaises(ValueError):self.allocated(**{field:'f'*64})
        for field,value in (('schema_version',True),('rounding','nearest'),('tie_break','input_order'),
                            ('duration_model','outcome_selected'),('extra',1)):
            allocation=copy.deepcopy(self.allocation);allocation[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.allocated(allocation=allocation)

    def test_missing_extra_duplicate_targets_and_rules(self):
        for mode in ('missing_task','missing_idle','duplicate_target','unknown_task','cross_stream','missing_rule','duplicate_rule','extra_rule'):
            allocation=copy.deepcopy(self.allocation)
            if mode=='missing_task':allocation['targets'].pop(0)
            if mode=='missing_idle':allocation['targets'].pop(3)
            if mode=='duplicate_target':allocation['targets'].append(copy.deepcopy(allocation['targets'][0]))
            if mode=='unknown_task':allocation['targets'][0]['task_id']='unknown'
            if mode=='cross_stream':allocation['rules'][0]['task_target_ids'][0]='B-task-0'
            if mode=='missing_rule':allocation['rules'].pop()
            if mode=='duplicate_rule':allocation['rules'].append(copy.deepcopy(allocation['rules'][0]))
            if mode=='extra_rule':allocation['rules'][0]['component_id']='storage'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.allocated(allocation=allocation)

    def test_fixed_shares_windows_and_event_contracts_fail_closed(self):
        for mode in ('sum','negative','zero_denominator','unreduced','bool','duplicate_share','event_duration',
                     'empty_windows','window_overlap','window_outside','negative_window'):
            allocation=copy.deepcopy(self.allocation);rule=next(r for r in allocation['rules'] if r['method']=='fixed')
            if mode=='sum':rule['shares'][0]['numerator']=0
            if mode=='negative':rule['shares'][0]['numerator']=-1
            if mode=='zero_denominator':rule['shares'][0]['denominator']=0
            if mode=='unreduced':rule['shares'][0].update(numerator=2,denominator=4)
            if mode=='bool':rule['shares'][0]['numerator']=True
            if mode=='duplicate_share':rule['shares'][1]=copy.deepcopy(rule['shares'][0])
            if mode=='event_duration':
                event=next(r for r in allocation['rules'] if r['component_id']=='startup')
                del event['target_id'];event.update(method='duration',task_target_ids=['A-task-0'],idle_target_id='A-idle')
            if mode=='empty_windows':allocation['targets'][0]['windows']=[]
            if mode=='window_overlap':allocation['targets'][0]['windows'].append(dict(start_offset_ns=5,end_offset_ns=7))
            if mode=='window_outside':allocation['targets'][0]['windows'][0]['end_offset_ns']=9
            if mode=='negative_window':allocation['targets'][0]['windows'][0]['start_offset_ns']=-1
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.allocated(allocation=allocation)

    @staticmethod
    def source_values(result,ordinal=0):
        source=result['sources'][ordinal]
        target_ids={i:t['id'] for i,t in enumerate(result['targets'])}
        shares={target_ids[s['target_index']]:Fraction(s['numerator'],s['denominator']) for s in source['shares']}
        allocations=None if source['allocations'] is None else {target_ids[a['target_index']]:
            {k:v for k,v in a.items() if k!='target_index'} for a in source['allocations']}
        return shares,allocations

    def test_independent_overlap_idle_oracle_and_separate_money_rounding(self):
        before=copy.deepcopy((self.plan,self.stream,self.ledger,self.allocation))
        result=self.allocated();shares,values=self.source_values(result)
        self.assertEqual(shares,{'A-task-0':Fraction(5,8),'A-task-1':Fraction(1,8),'A-idle':Fraction(1,4)})
        self.assertEqual({k:v['quantity'] for k,v in values.items()},{'A-task-0':4,'A-task-1':1,'A-idle':2})
        self.assertEqual({k:v['billed_amount'] for k,v in values.items()},{'A-task-0':3,'A-task-1':0,'A-idle':1})
        for tid,derived,difference in (('A-task-0',Fraction(2),Fraction(1)),
                ('A-task-1',Fraction(1,2),Fraction(-1,2)),('A-idle',Fraction(1),Fraction(0))):
            self.assertEqual(values[tid]['rate_derived_atomic'],dict(numerator=derived.numerator,denominator=derived.denominator))
            self.assertEqual(values[tid]['billed_minus_rate_derived_atomic'],dict(numerator=difference.numerator,denominator=difference.denominator))
        self.assertEqual(sum(v['quantity'] for v in values.values()),7)
        self.assertEqual(sum(v['billed_amount'] for v in values.values()),4)
        self.assertEqual(result['canonical_arms']['A']['known_monetary_atomic'],10)
        self.assertEqual(sum(a['known_monetary_atomic'] for a in result['canonical_arms'].values()),30)
        self.assertEqual(result['stream_totals']['stream-A']['known_billed_atomic'],10)
        self.assertEqual(before,(self.plan,self.stream,self.ledger,self.allocation))
        self.assertTrue(result['known_line_conservation_verified'])
        self.assertTrue(result['declared_allocation_complete'])
        self.assertEqual(result['actual_acceptance'],'incomplete')
        for flag in ('receipt_authenticity_verified','measurement_truth_verified','isolation_verified',
                     'instrumentation_coverage_verified','whole_pipeline_cost_verified','numeric_integration_implemented'):
            self.assertFalse(result[flag])

    def test_direct_startup_proxy_fixed_and_zero_quantities(self):
        result=self.allocated()
        shares,values=self.source_values(result,2)
        self.assertEqual(shares,{'A-startup':Fraction(1)})
        self.assertEqual(values['A-startup']['quantity'],7)
        self.assertEqual(values['A-startup']['billed_amount'],4)
        shares,values=self.source_values(result,3)
        self.assertEqual(shares,{'A-task-0':Fraction(1,2),'A-task-1':Fraction(1,2)})
        self.assertEqual([v['quantity'] for v in values.values()],[10,10])
        self.assertTrue(all(v['billed_amount'] is None and v['rate_derived_atomic'] is None for v in values.values()))
        ledger=copy.deepcopy(self.ledger);ledger['entries'][0].update(quantity=0,billed_amount=0,source_quantity=dict(numerator=0,denominator=1))
        result=self.allocated(ledger=ledger);shares,values=self.source_values(result)
        self.assertEqual(len(shares),3)
        self.assertTrue(all(v['quantity']==0 and v['billed_amount']==0 for v in values.values()))

    def test_nonuniform_fixed_shares_and_fraction_overflow_fail_closed(self):
        allocation=copy.deepcopy(self.allocation)
        rule=allocation['rules'][0];del rule['task_target_ids'];del rule['idle_target_id']
        rule.update(method='fixed',shares=[dict(target_id='A-task-0',numerator=1,denominator=3),
                                          dict(target_id='A-task-1',numerator=2,denominator=3)])
        shares,values=self.source_values(self.allocated(allocation=allocation))
        self.assertEqual(shares,{'A-task-0':Fraction(1,3),'A-task-1':Fraction(2,3)})
        self.assertEqual({k:v['quantity'] for k,v in values.items()},{'A-task-0':2,'A-task-1':5})
        self.assertEqual({k:v['billed_amount'] for k,v in values.items()},{'A-task-0':1,'A-task-1':3})
        rule['shares']=[dict(target_id=tid,numerator=1,denominator=2**61-offset) for tid,offset in zip(
            ('A-task-0','A-task-1','A-startup','A-idle','A-overhead'),(1,3,5,7,9))]
        with self.assertRaisesRegex(ValueError,'256-bit'):
            self.allocated(allocation=allocation)

    def test_unicode_codepoint_ties_no_normalization_and_input_order_invariance(self):
        for left,right,winner in (('\U00010000','\ue000','\ue000'),('é','e\u0301','e\u0301')):
            allocation=copy.deepcopy(self.allocation);ledger=copy.deepcopy(self.ledger)
            rename={'A-task-0':left,'A-task-1':right}
            for target in allocation['targets']:target['id']=rename.get(target['id'],target['id'])
            for rule in allocation['rules']:
                if rule['method']=='fixed':
                    for share in rule['shares']:share['target_id']=rename.get(share['target_id'],share['target_id'])
                if rule['method']=='duration':rule['task_target_ids']=[rename.get(t,t) for t in rule['task_target_ids']]
            rule=allocation['rules'][0]
            del rule['task_target_ids'];del rule['idle_target_id']
            rule.update(method='fixed',shares=[dict(target_id=t,numerator=1,denominator=2) for t in (left,right)])
            ledger['entries'][0].update(quantity=1,billed_amount=0,source_quantity=dict(numerator=1,denominator=1))
            result=self.allocated(ledger=ledger,allocation=allocation)
            shares,values=self.source_values(result)
            self.assertEqual(set(shares),{left,right})
            self.assertEqual(values[winner]['quantity'],1)
            self.assertEqual(sum(v['quantity'] for v in values.values()),1)
            allocation['targets'].reverse();allocation['rules'].reverse()
            for rule in allocation['rules']:
                if rule['method']=='fixed':rule['shares'].reverse()
                if rule['method']=='duration':rule['task_target_ids'].reverse()
            other=self.allocated(ledger=ledger,allocation=allocation)
            self.assertEqual((shares,values),self.source_values(other))
            ledger['entries'].reverse()
            reordered=self.allocated(ledger=ledger,allocation=allocation)
            old={s['entry_sha256']:self.source_values(result,i) for i,s in enumerate(result['sources'])}
            new={s['entry_sha256']:self.source_values(reordered,i) for i,s in enumerate(reordered['sources'])}
            self.assertEqual(old,new)

    def test_adjacent_windows_clipping_and_idle_gaps(self):
        allocation=copy.deepcopy(self.allocation)
        allocation['targets'][0]['windows']=[dict(start_offset_ns=0,end_offset_ns=2),dict(start_offset_ns=2,end_offset_ns=6)]
        self.assertEqual(self.source_values(self.allocated()),self.source_values(self.allocated(allocation=allocation)))
        # Source intervals can be finer than frozen windows; only overlapping time
        # contributes and absolute observed start must not shift frozen offsets.
        ledger=copy.deepcopy(self.ledger)
        original=ledger['entries'].pop(0)
        first=copy.deepcopy(original);second=copy.deepcopy(original)
        first.update(end_ns=4,quantity=3,billed_amount=2,source_quantity=dict(numerator=3,denominator=1))
        second.update(start_ns=4,quantity=4,billed_amount=2,source_quantity=dict(numerator=4,denominator=1))
        second['identity']['line_id']='perception-second'
        ledger['entries']=[first,second]+ledger['entries']
        result=self.allocated(ledger=ledger)
        self.assertEqual(self.source_values(result,0)[0],{'A-task-0':Fraction(3,4),'A-task-1':Fraction(1,4)})
        self.assertEqual(self.source_values(result,1)[0],{'A-task-0':Fraction(1,2),'A-idle':Fraction(1,2)})
        for observed in ledger['streams']:
            observed['start_ns']+=100;observed['end_ns']+=100
            for part in observed['availability']:part['start_ns']+=100;part['end_ns']+=100
        for entry in ledger['entries']:
            for field in ('start_ns','end_ns','event_ns'):
                if entry[field] is not None:entry[field]+=100
        shifted=self.allocated(ledger=ledger)
        self.assertEqual(self.source_values(result,0),self.source_values(shifted,0))
        self.assertEqual(self.source_values(result,1),self.source_values(shifted,1))

    def test_unknown_lines_exposure_and_conservative_stream_completeness(self):
        for state in ('unknown','pending'):
            ledger=copy.deepcopy(self.ledger)
            ledger['entries'][0].update(status=state,quantity=None,source_quantity=None,billed_amount=None,evidence_sha256=None)
            result=self.allocated(ledger=ledger);shares,values=self.source_values(result)
            self.assertEqual(shares,{'A-task-0':Fraction(5,8),'A-task-1':Fraction(1,8),'A-idle':Fraction(1,4)})
            self.assertIsNone(values)
            self.assertFalse(result['declared_allocation_complete'])
            for target,total in zip(result['targets'],result['target_totals']):
                if target['stream_id']=='stream-A':
                    self.assertTrue(total['source_stream_incomplete'])
                    self.assertIsNone(total['declared_complete_cost_atomic'])
                    self.assertEqual(total['unknown_source_indices'],[0] if target['id'] in shares else [])
                else:self.assertFalse(total['source_stream_incomplete'])
            self.assertEqual(result['stream_totals']['stream-A']['known_billed_atomic'],6)
        ledger=copy.deepcopy(self.ledger);ledger['entries'].pop(0)
        result=self.allocated(ledger=ledger)
        self.assertFalse(result['declared_allocation_complete'])
        self.assertTrue(all(t['declared_complete_cost_atomic'] is None for target,t in zip(result['targets'],result['target_totals']) if target['stream_id']=='stream-A'))

    def test_original_invoice_rounding_discrepancy_stays_incomplete(self):
        ledger=copy.deepcopy(self.ledger);ledger['entries'][0]['billed_amount']=5
        result=self.allocated(ledger=ledger)
        self.assertFalse(result['declared_allocation_complete'])
        self.assertEqual(sum(v['billed_amount'] for v in self.source_values(result)[1].values()),5)
        self.assertTrue(all(t['declared_complete_cost_atomic'] is None for target,t in zip(result['targets'],result['target_totals']) if target['stream_id']=='stream-A'))
        ledger=copy.deepcopy(self.ledger);ledger['entries'][1]['identity']=copy.deepcopy(ledger['entries'][0]['identity'])
        with self.assertRaises(ValueError):self.allocated(ledger=ledger)

    def test_graph_label_nonfinite_and_exact_output_bounds(self):
        allocation=copy.deepcopy(self.allocation);allocation['targets']*=667
        with self.assertRaises(ValueError):self.allocated(allocation=allocation)
        allocation=copy.deepcopy(self.allocation);allocation['targets'][0]['windows']*=10001
        with self.assertRaises(ValueError):self.allocated(allocation=allocation)
        allocation=copy.deepcopy(self.allocation);allocation['targets'][0]['id']='x'*257
        with self.assertRaises(ValueError):self.allocated(allocation=allocation)
        allocation=copy.deepcopy(self.allocation);allocation['targets'][0]['windows'][0]['start_offset_ns']=float('nan')
        with self.assertRaises(ValueError):self.allocated(allocation=allocation)
        result=self.allocated();size=len(_encoded(result))+1
        with patch('benchmarks.stream_cost_allocation.MAX_REPORT_BYTES',size):self.allocated()
        with patch('benchmarks.stream_cost_allocation.MAX_REPORT_BYTES',size-1):
            with self.assertRaisesRegex(ValueError,'allocation report'):self.allocated()

    def test_unknown_lines_cannot_bypass_actual_processing_budget(self):
        plan=copy.deepcopy(self.plan)
        template=plan['comparison']['tasks'][0]
        plan['comparison']['tasks']=[dict(template,id=str(i)) for i in range(1000)]
        stream=copy.deepcopy(self.stream);ledger=copy.deepcopy(self.ledger);allocation=copy.deepcopy(self.allocation)
        for frozen,observed in zip(stream['streams'],ledger['streams']):
            frozen['requested_duration_ns']=100
            observed['end_ns']=100;observed['availability'][0]['end_ns']=100
            for i in range(2,1000):allocation['targets'].append(dict(id=frozen['arm']+'-task-'+str(i),
                stream_id=frozen['id'],kind='task',task_id=str(i),windows=[]))
        for entry in ledger['entries']:
            if entry['end_ns'] is not None:entry['end_ns']=100
        source=ledger['entries'].pop(0);expanded=[]
        for i in range(100):
            entry=copy.deepcopy(source);entry.update(start_ns=i,end_ns=i+1,status='unknown',
                quantity=None,source_quantity=None,billed_amount=None,evidence_sha256=None)
            entry['identity']['line_id']='unsettled-'+str(i);expanded.append(entry)
        ledger['entries']=expanded+ledger['entries']
        rule=allocation['rules'][0];del rule['task_target_ids'];del rule['idle_target_id']
        rule.update(method='fixed',shares=[dict(target_id='A-task-'+str(i),numerator=1,denominator=1000) for i in range(1000)])
        self.assertLess(len(_encoded(allocation)),16*1024*1024)
        with self.assertRaisesRegex(ValueError,'processing budget'):
            self.allocated(plan=plan,stream=stream,ledger=ledger,allocation=allocation)

    def test_cli_compact_exclusive_immutable_and_strict_json(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory);paths={}
            for name,value in (('plan',self.plan),('stream-protocol',self.stream),('ledger',self.ledger),('allocation-protocol',self.allocation)):
                paths[name]=root/(name+'.json');paths[name].write_bytes(_encoded(value)+b'\n')
            original={name:path.read_bytes() for name,path in paths.items()};output=root/'out'
            cmd=[sys.executable,'-m','benchmarks.stream_cost_allocation']
            for name,path in paths.items():cmd.extend(['--'+name,str(path)])
            for name,value in (('plan',self.plan),('stream-protocol',self.stream),('ledger',self.ledger),('allocation-protocol',self.allocation)):
                cmd.extend(['--'+name+'-sha256',plan_digest(value)])
            cmd.extend(['--output',str(output)])
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            report=(output/'report.json').read_bytes()
            self.assertEqual(run.stdout.encode('ascii'),report)
            self.assertEqual(json.loads(report),self.allocated())
            self.assertNotEqual(subprocess.run(cmd,capture_output=True,text=True,timeout=20).returncode,0)
            self.assertEqual((output/'report.json').read_bytes(),report)
            self.assertEqual({name:path.read_bytes() for name,path in paths.items()},original)
            for i,payload in enumerate((b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b' '*(16*1024*1024+1))):
                paths['allocation-protocol'].write_bytes(payload);bad=root/('bad-'+str(i))
                run=subprocess.run(cmd[:-1]+[str(bad)],capture_output=True,text=True,timeout=20)
                self.assertNotEqual(run.returncode,0)
                self.assertFalse(bad.exists())
