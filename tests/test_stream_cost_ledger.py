import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.perception_comparison import plan_digest
from benchmarks.stream_cost_ledger import CATEGORIES, COUNTERS, evaluate


def fixture():
    components = [dict(id=c, category=c, kind='proxy' if c=='capture_sensor' else 'monetary',
        unit='frame_tick' if c=='capture_sensor' else 'cpu_tick', source_unit='source_tick',
        atomic_per_source=dict(numerator=1,denominator=1),
        coverage='event' if c=='startup' else 'interval', rate_id=None if c=='capture_sensor' else 'cpu')
        for c in CATEGORIES]
    protocol = dict(schema_version=1, comparison_plan_sha256='1'*64, collection_sha256='2'*64,
        measurement_protocol_sha256='3'*64, boundary='Fixture full declared boundary; measurement truth unaudited',
        currency='USD', currency_unit='microUSD', rate_revision='fixture-v1',
        invoice_rounding='line_atomic_half_even', isolation='isolated_arms', components=components,
        rates={'cpu':dict(provider='meter-provider',revision='meter-v1',unit='cpu_tick',per_unit='0.5')}, streams=[])
    ledger = dict(schema_version=1, protocol_sha256='', comparison_plan_sha256='1'*64,
        collection_sha256='2'*64, reviewer='fixture reviewer', source_evidence_sha256='4'*64,
        streams=[], entries=[])
    required = ('perception','resident_idle','startup','capture_sensor')
    start, end = 100, 2_000_000_100
    for arm in 'ABC':
        sid='stream-'+arm
        protocol['streams'].append(dict(id=sid,arm=arm,recording_sha256='5'*64,
            clock_domain='clock-'+arm,requested_duration_ns=end-start,execution_namespace='exec-'+arm,
            cache_namespace='cache-'+arm,isolation_evidence_sha256='6'*64,
            sources=[dict(id='camera',semantics='attempted_equals_captured_plus_dropped',required_counters=list(COUNTERS))],
            requirements={c:dict(state='required' if c in required else 'not_applicable',
                reason='Fixture measured lane' if c in required else 'Fixture has no such component; external assertion only',
                lanes=[dict(resource_id=arm+'-device',lane_id=c)] if c in required else []) for c in CATEGORIES}))
        ledger['streams'].append(dict(id=sid,clock_domain='clock-'+arm,start_ns=start,end_ns=end,
            utc_anchor=dict(monotonic_ns=start,unix_ns=1_700_000_000_000_000_000,evidence_sha256='7'*64),
            availability=[dict(start_ns=start,end_ns=start+1_500_000_000,state='available',reason='capture available',evidence_sha256='8'*64),
                          dict(start_ns=start+1_500_000_000,end_ns=end,state='unavailable',reason='capture unavailable',evidence_sha256='9'*64)],
            counters=[dict(source_id='camera',start={c:0 for c in COUNTERS},
                end=dict(captured=6,sampled=2,vlm_calls=3,attempted=8,dropped=2))]))
        for c,quantity,billed in (('perception',10,5),('resident_idle',4,2),('startup',7,4),('capture_sensor',20,None)):
            identity=(dict(kind='local',meter_id='meter-'+arm,measurement_id=c) if c=='capture_sensor' else
                dict(kind='invoice',provider='meter-provider',revision='meter-v1',account_scope='fixture',
                     invoice_id='invoice-'+arm,document_sha256='a'*64,line_id=c))
            ledger['entries'].append(dict(stream_id=sid,component_id=c,resource_id=arm+'-device',lane_id=c,
                start_ns=None if c=='startup' else start,end_ns=None if c=='startup' else end,
                event_ns=start if c=='startup' else None, identity=identity,source_quantity=dict(numerator=quantity,denominator=1),quantity=quantity,
                status='measured' if c=='capture_sensor' else 'settled',billed_amount=billed,evidence_sha256='b'*64))
    ledger['protocol_sha256']=plan_digest(protocol)
    return protocol,ledger


class StreamCostLedger(unittest.TestCase):
    def setUp(self):
        self.protocol,self.ledger=fixture()

    def evaluate(self,protocol=None,ledger=None,**kwargs):
        protocol=copy.deepcopy(self.protocol if protocol is None else protocol)
        ledger=copy.deepcopy(self.ledger if ledger is None else ledger)
        ledger['protocol_sha256']=plan_digest(protocol)
        args=dict(expected_protocol_sha256=plan_digest(protocol),expected_ledger_sha256=plan_digest(ledger))
        args.update(kwargs)
        return evaluate(protocol,ledger,**args)

    def test_exact_observed_denominators_startup_idle_proxy_and_canonical_a(self):
        before=copy.deepcopy((self.protocol,self.ledger))
        result=self.evaluate()
        self.assertTrue(result['declared_accounting_complete'])
        report=result['streams']['stream-A']
        self.assertEqual(report['elapsed_ns'],2_000_000_000)
        self.assertEqual(report['availability_ns'],dict(available=1_500_000_000,unavailable=500_000_000,unknown=0))
        counters=report['counters']['camera']
        self.assertEqual(counters['captured_per_elapsed_second'],dict(numerator=3,denominator=1))
        self.assertEqual(counters['captured_per_available_second'],dict(numerator=4,denominator=1))
        self.assertEqual((counters['captured'],counters['sampled'],counters['vlm_calls']),(6,2,3))
        self.assertEqual(report['known_monetary_atomic'],11)
        self.assertEqual(report['known_rate_derived_atomic'],dict(numerator=21,denominator=2))
        self.assertEqual(report['declared_complete_cost_per_elapsed_hour'],dict(numerator=19800,denominator=1))
        self.assertEqual(report['known_proxy_quantities'],{'frame_tick':20})
        self.assertEqual(report['known_proxy_per_elapsed_hour'],{'frame_tick':dict(numerator=36000,denominator=1)})
        self.assertEqual(result['canonical_arms']['A'],dict(stream_ids=['stream-A'],known_monetary_atomic=11))
        self.assertEqual(sum(a['known_monetary_atomic'] for a in result['canonical_arms'].values()),33)
        self.assertEqual(before,(self.protocol,self.ledger))
        self.assertEqual(result['actual_acceptance'],'incomplete')
        self.assertFalse(result['allocation_implemented'])
        for key in ('receipt_authenticity_verified','measurement_truth_verified','isolation_verified',
                    'instrumentation_coverage_verified','whole_pipeline_cost_verified'):
            self.assertFalse(result[key])
        self.assertNotIn('daily_cost',report)

    def test_all_not_applicable_declarations_never_certify_whole_pipeline(self):
        protocol=copy.deepcopy(self.protocol); ledger=copy.deepcopy(self.ledger)
        for stream in protocol['streams']:
            for req in stream['requirements'].values():
                req.update(state='not_applicable',reason='Externally asserted absent, not proven by software',lanes=[])
        ledger['entries']=[]
        result=self.evaluate(protocol,ledger)
        self.assertTrue(result['declared_accounting_complete'])
        self.assertFalse(result['whole_pipeline_cost_verified'])
        self.assertFalse(result['measurement_truth_verified'])
        self.assertEqual(result['actual_acceptance'],'incomplete')

    def test_pending_unknown_and_missing_components_remain_explicit(self):
        for state in ('pending','unknown'):
            ledger=copy.deepcopy(self.ledger)
            ledger['entries'][0].update(status=state,source_quantity=None,quantity=None,billed_amount=None,evidence_sha256=None)
            result=self.evaluate(ledger=ledger)
            report=result['streams']['stream-A']
            self.assertFalse(result['declared_accounting_complete'])
            self.assertEqual(report['known_monetary_atomic'],6)
            self.assertIsNone(report['declared_complete_cost_per_elapsed_hour'])
            self.assertEqual(report['component_totals']['perception']['unknown_entries'],1)
            self.assertTrue(any('pending/unknown' in i['reason'] for i in result['incomplete_reasons']))
        for index,reason in ((0,'unmeasured resource interval'),(2,'missing required event')):
            ledger=copy.deepcopy(self.ledger);ledger['entries'].pop(index)
            result=self.evaluate(ledger=ledger)
            self.assertFalse(result['streams']['stream-A']['complete_accounting'])
            self.assertTrue(any(reason in i['reason'] for i in result['incomplete_reasons']))
        ledger=copy.deepcopy(self.ledger);ledger['entries'][3].update(status='unknown',source_quantity=None,quantity=None,evidence_sha256=None)
        result=self.evaluate(ledger=ledger)
        self.assertEqual(result['streams']['stream-A']['known_proxy_quantities'],{})
        self.assertFalse(result['declared_accounting_complete'])

    def test_availability_partition_and_unknown_duration(self):
        for mutate in ('gap','overlap','reverse','outside','empty'):
            ledger=copy.deepcopy(self.ledger);parts=ledger['streams'][0]['availability']
            if mutate=='gap':parts[1]['start_ns']+=1
            if mutate=='overlap':parts[1]['start_ns']-=1
            if mutate=='reverse':parts.reverse()
            if mutate=='outside':parts[-1]['end_ns']+=1
            if mutate=='empty':parts.clear()
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        ledger=copy.deepcopy(self.ledger)
        ledger['streams'][0]['availability'][1].update(state='unknown',evidence_sha256=None)
        result=self.evaluate(ledger=ledger)
        self.assertEqual(result['streams']['stream-A']['availability_ns']['unknown'],500_000_000)
        self.assertIsNone(result['streams']['stream-A']['counters']['camera']['captured_per_available_second'])
        self.assertIsNone(result['streams']['stream-A']['declared_complete_cost_per_elapsed_hour'])

    def test_counter_resets_reconciliation_required_unknown_and_zero_availability(self):
        for key,value in (('captured',-1),('captured',True),('sampled',7),('attempted',7),('vlm_calls',2**63)):
            ledger=copy.deepcopy(self.ledger);ledger['streams'][0]['counters'][0]['end'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        ledger=copy.deepcopy(self.ledger);ledger['streams'][0]['counters'][0]['start']['captured']=10
        with self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        ledger=copy.deepcopy(self.ledger);ledger['streams'][0]['counters'][0]['end']['vlm_calls']=None
        result=self.evaluate(ledger=ledger)
        self.assertIsNone(result['streams']['stream-A']['counters']['camera']['vlm_calls'])
        self.assertFalse(result['declared_accounting_complete'])
        ledger=copy.deepcopy(self.ledger);ledger['streams'][0]['counters']=[]
        with self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        ledger=copy.deepcopy(self.ledger)
        for part in ledger['streams'][0]['availability']:part['state']='unavailable'
        with self.assertRaises(ValueError):self.evaluate(ledger=ledger)

    def test_interval_coverage_gaps_overlap_and_concurrent_distinct_lanes(self):
        ledger=copy.deepcopy(self.ledger);ledger['entries'][0]['start_ns']+=100
        result=self.evaluate(ledger=ledger)
        self.assertFalse(result['streams']['stream-A']['complete_accounting'])
        ledger=copy.deepcopy(self.ledger);extra=copy.deepcopy(ledger['entries'][0])
        extra['identity']['line_id']='another-line';ledger['entries'].append(extra)
        with self.assertRaisesRegex(ValueError,'overlapping'):self.evaluate(ledger=ledger)
        self.assertTrue(self.evaluate()['streams']['stream-A']['complete_accounting'])
        for field,value in (('start_ns',99),('end_ns',2_000_000_101),('event_ns',100)):
            ledger=copy.deepcopy(self.ledger);ledger['entries'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.evaluate(ledger=ledger)

    def test_duplicate_invoice_line_changed_document_and_local_identity(self):
        for mode in ('line','changed_document','changed_revision','local'):
            ledger=copy.deepcopy(self.ledger)
            if mode=='line':ledger['entries'][1]['identity']['line_id']=ledger['entries'][0]['identity']['line_id']
            if mode=='changed_document':ledger['entries'][1]['identity']['document_sha256']='c'*64
            if mode=='changed_revision':ledger['entries'][1]['identity']['revision']='other'
            if mode=='local':ledger['entries'][7]['identity']=copy.deepcopy(ledger['entries'][3]['identity'])
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        # Distinct invoice lines in a shared document are legitimate charges.
        self.assertTrue(self.evaluate()['declared_accounting_complete'])

    def test_exact_half_even_rounding_and_discrepancy_incomplete(self):
        for quantity,billed in ((1,0),(3,2),(5,2),(7,4)):
            ledger=copy.deepcopy(self.ledger);ledger['entries'][0].update(quantity=quantity,source_quantity=dict(numerator=quantity,denominator=1),billed_amount=billed)
            with self.subTest(quantity=quantity):self.assertTrue(self.evaluate(ledger=ledger)['declared_accounting_complete'])
        ledger=copy.deepcopy(self.ledger);ledger['entries'][0]['billed_amount']=6
        result=self.evaluate(ledger=ledger)
        self.assertFalse(result['declared_accounting_complete'])
        self.assertEqual(result['streams']['stream-A']['known_monetary_atomic'],12)
        self.assertIsNone(result['streams']['stream-A']['declared_complete_cost_per_elapsed_hour'])
        self.assertTrue(any('rounding' in i['reason'] for i in result['incomplete_reasons']))
        protocol=copy.deepcopy(self.protocol);protocol['rates']['cpu']['per_unit']='0.1'
        ledger=copy.deepcopy(self.ledger)
        for entry in ledger['entries']:
            if entry['status']=='settled':entry['billed_amount']={'perception':1,'resident_idle':0,'startup':1}[entry['component_id']]
        result=self.evaluate(protocol,ledger)
        self.assertEqual(result['streams']['stream-A']['known_rate_derived_atomic'],dict(numerator=21,denominator=10))

    def test_exact_source_unit_normalization_and_arithmetic_overflow(self):
        protocol=copy.deepcopy(self.protocol);ledger=copy.deepcopy(self.ledger)
        protocol['components'][0]['atomic_per_source']=dict(numerator=4,denominator=1)
        for entry in ledger['entries']:
            if entry['component_id']=='perception':entry['source_quantity']=dict(numerator=5,denominator=2)
        self.assertTrue(self.evaluate(protocol,ledger)['declared_accounting_complete'])
        for source in (dict(numerator=1,denominator=3),dict(numerator=3,denominator=1),
                       dict(numerator=True,denominator=1),dict(numerator=1,denominator=0)):
            changed=copy.deepcopy(ledger);changed['entries'][0]['source_quantity']=source
            with self.subTest(source=source),self.assertRaises(ValueError):self.evaluate(protocol,changed)
        protocol=copy.deepcopy(self.protocol);protocol['rates']['cpu']['per_unit']='1'+'0'*63
        ledger=copy.deepcopy(self.ledger);ledger['entries'][0].update(quantity=2**63-1,
            source_quantity=dict(numerator=2**63-1,denominator=1))
        with self.assertRaisesRegex(ValueError,'256-bit'):self.evaluate(protocol,ledger)
        ledger=copy.deepcopy(self.ledger);ledger['streams'][0]['clock_domain']='another-clock'
        with self.assertRaises(ValueError):self.evaluate(ledger=ledger)

    def test_frozen_identity_and_schema_changes_fail_closed(self):
        for field in ('expected_protocol_sha256','expected_ledger_sha256'):
            with self.subTest(field=field),self.assertRaises(ValueError):self.evaluate(**{field:'f'*64})
        for field,value in (('comparison_plan_sha256','f'*64),('collection_sha256','f'*64),('schema_version',True),('reviewer','')):
            ledger=copy.deepcopy(self.ledger);ledger[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        for mutate in ('missing_category','extra_field','float_price','unit','currency','isolation'):
            protocol=copy.deepcopy(self.protocol)
            if mutate=='missing_category':protocol['components'].pop()
            if mutate=='extra_field':protocol['unfrozen']=1
            if mutate=='float_price':protocol['rates']['cpu']['per_unit']=.5
            if mutate=='unit':protocol['rates']['cpu']['unit']='other'
            if mutate=='currency':protocol['currency']=''
            if mutate=='isolation':protocol['isolation']='shared_arms'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.evaluate(protocol=protocol)

    def test_cache_execution_resource_isolation_and_missing_original_arm(self):
        for field in ('cache_namespace','execution_namespace'):
            protocol=copy.deepcopy(self.protocol);protocol['streams'][1][field]=protocol['streams'][0][field]
            with self.subTest(field=field),self.assertRaises(ValueError):self.evaluate(protocol=protocol)
        protocol=copy.deepcopy(self.protocol)
        protocol['streams'][1]['requirements']['perception']['lanes'][0].update(resource_id='A-device',lane_id='new-meter')
        with self.assertRaisesRegex(ValueError,'cross-arm'):self.evaluate(protocol=protocol)
        protocol=copy.deepcopy(self.protocol);protocol['streams'].pop()
        with self.assertRaises(ValueError):self.evaluate(protocol=protocol)
        ledger=copy.deepcopy(self.ledger);ledger['streams'].append(copy.deepcopy(ledger['streams'][0]))
        with self.assertRaises(ValueError):self.evaluate(ledger=ledger)

    def test_unknown_money_proxy_mixing_and_unexplained_short_run(self):
        for mode in ('pending_known_bill','proxy_money','settled_unknown','known_no_evidence','invoice_proxy'):
            ledger=copy.deepcopy(self.ledger)
            if mode=='pending_known_bill':ledger['entries'][0]['status']='pending'
            if mode=='proxy_money':ledger['entries'][3]['billed_amount']=1
            if mode=='settled_unknown':ledger['entries'][0]['quantity']=None
            if mode=='known_no_evidence':ledger['entries'][0]['evidence_sha256']=None
            if mode=='invoice_proxy':ledger['entries'][3]['identity']=copy.deepcopy(ledger['entries'][0]['identity'])
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        protocol=copy.deepcopy(self.protocol);protocol['streams'][0]['requested_duration_ns']+=1
        result=self.evaluate(protocol=protocol)
        self.assertFalse(result['streams']['stream-A']['complete_accounting'])
        self.assertEqual(result['streams']['stream-A']['elapsed_ns'],2_000_000_000)

    def test_bounds_and_nonfinite_in_memory(self):
        for value in (float('nan'),float('inf'),True,-1,2**63):
            ledger=copy.deepcopy(self.ledger);ledger['entries'][0]['quantity']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.evaluate(ledger=ledger)
        # Each interval is valid and contiguous; only the combined cross-list bound fails.
        ledger=copy.deepcopy(self.ledger);start=ledger['streams'][0]['start_ns']
        ledger['streams'][0]['availability']=[dict(start_ns=start+i*200000,
            end_ns=start+(i+1)*200000,state='available',reason='bounded fixture',evidence_sha256='8'*64)
            for i in range(10000)]
        with self.assertRaisesRegex(ValueError,'total observation/cost entries'):
            self.evaluate(ledger=ledger)
        protocol=copy.deepcopy(self.protocol);protocol['boundary']='x'*(16*1024*1024)
        with self.assertRaises(ValueError):self.evaluate(protocol=protocol)
        protocol=copy.deepcopy(self.protocol);protocol['rates']['cpu']['per_unit']='1'+'0'*64
        with self.assertRaises(ValueError):self.evaluate(protocol=protocol)

    def test_cli_exclusive_immutable_strict_json(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            root=Path(directory);protocol=root/'protocol.json';ledger=root/'ledger.json';output=root/'out'
            protocol.write_text(json.dumps(self.protocol));ledger.write_text(json.dumps(self.ledger))
            original=(protocol.read_bytes(),ledger.read_bytes())
            cmd=[sys.executable,'-m','benchmarks.stream_cost_ledger','--protocol',str(protocol),'--ledger',str(ledger),
                 '--protocol-sha256',plan_digest(self.protocol),'--ledger-sha256',plan_digest(self.ledger),'--output',str(output)]
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads((output/'report.json').read_text()),self.evaluate())
            before=(output/'report.json').read_bytes()
            self.assertNotEqual(subprocess.run(cmd,capture_output=True,text=True,timeout=20).returncode,0)
            self.assertEqual(before,(output/'report.json').read_bytes())
            self.assertEqual(original,(protocol.read_bytes(),ledger.read_bytes()))
            for i,payload in enumerate((b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}',b' '*(16*1024*1024+1))):
                ledger.write_bytes(payload);bad=root/('bad-'+str(i));command=cmd[:-1]+[str(bad)]
                result=subprocess.run(command,capture_output=True,text=True,timeout=20)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(bad.exists())
