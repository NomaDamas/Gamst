import copy
import hashlib
import time
import unittest

from gamst.native_entry import _issue_entry_binding
from gamst.recording_settlement import FrozenRecordingReceipt,RecordingSettlementRecorder
from benchmarks.recording_financial import claim_sha256,digest,encode
from test_call_controller import controller,finish,record,reserve,submit
from test_recording_financial import adapters,invoice


def setup():
    c = controller(); h = c.open_source(record(),now_ms=1000)
    decision = submit(c,h,1); request = decision['request_id']
    reserve(c,request); dispatched = c.dispatch(request,now_ms=1000)
    execution = dispatched['execution']
    context = dict(schema_version=4,kind='recording_native',nonce='a'*64,invocation_ordinal=0,
        collection_sha256='b'*64,scope_protocol_sha256='c'*64,instrumentation_protocol_sha256='d'*64)
    binding = _issue_entry_binding(dict(schema_version=1,owner_kind='recording_native4',
        owner=dict(context=context,scope_index=0,binding_index=3),
        entry=dict(ticket_ordinal=0,entry_ordinal=0,instance_sha256='e'*64,
            request_decision_sha256=execution['decision_sha256'],entered_event_sha256='f'*64),
        provider=execution['provider'],model_revision=execution['model_revision'],role=execution['route'],
        prompt_sha256=execution['prompt_sha256']))
    recorder = RecordingSettlementRecorder(c,execution_sha256='1'*64,controller_id='main',
        unit='USD.micro',pricing_revision='v1')
    recorder.bind_request(request,binding)
    return c,decision,request,recorder,binding


def scenario():
    document = invoice(); line = document['lines'][0]
    value = {key:document[key] for key in ('provider','revision','account_scope','invoice_id','currency','currency_unit')}
    value.update({key:line[key] for key in ('line_id','stream_id','component_id','resource_id','lane_id','operation_key','source_quantity','quantity','billed_atomic')})
    value.update(schema_version=1,kind='synthetic_recording_charge_scenario',evidence_class='fixture',authority=document['authority'],
        adapter_id='fixture',currency_unit='USD.micro',pricing_revision='v1',components={'other':4})
    return value


def receipt(recorder,request,billed=4):
    document = invoice(); document['currency_unit'] = 'USD.micro'; document['lines'][0]['billed_atomic'] = billed
    source = encode(document,newline=True); source_hash = hashlib.sha256(source).hexdigest()
    adapter = adapters(); inventory = dict(schema_version=1,kind='recording_financial_adapters',adapters=list(adapter.adapters))
    evidence = dict(schema_version=1,kind='recording_financial_evidence',adapter_id='fixture',
        adapter_inventory_sha256=adapter.digest,origin='synthetic_fixture',
        source_document=dict(path='financial/source/source-'+source_hash+'.json',file_sha256=source_hash,bytes=len(source)),
        provider='fixture',revision='v1',account_scope='study',invoice_id='invoice-1',line_id='line-1',
        claim_sha256=claim_sha256('fixture','study','invoice-1','line-1'),owner_instance='e'*64,
        stream_id='stream-A',quantities=dict(vlm=dict(numerator=2,denominator=1)),billed_atomic=billed,
        currency='USD',currency_unit='USD.micro',service_start_ns=None,service_end_ns=None,service_event_ns=101,
        observed_received_ns=time.monotonic_ns(),finality_evidence_sha256=None)
    return recorder.synthetic_receipt(request,source_document=source,financial_evidence=encode(evidence,newline=True),
        adapter_inventory=encode(inventory,newline=True),expected_adapter_inventory_sha256=adapter.digest,components={'other':billed})


class RecordingSettlement(unittest.TestCase):
    def test_frozen_scenario_generates_downstream_actual_time_without_digest_cycle(self):
        c,d,t,recorder,binding = setup(); value = scenario(); adapter = adapters()
        inventory = dict(schema_version=1,kind='recording_financial_adapters',adapters=list(adapter.adapters))
        before = time.monotonic_ns()
        issued = recorder.generate_synthetic_receipt(t,value,expected_scenario_sha256=digest(value),
            adapter_inventory=encode(inventory,newline=True),expected_adapter_inventory_sha256=adapter.digest)
        after = time.monotonic_ns()
        from benchmarks.recording_financial import decode
        source,evidence = map(decode,recorder.generated_artifacts(t))
        event = source['lines'][0]['service_event_ns']
        self.assertLessEqual(before,event); self.assertLessEqual(event,after)
        self.assertIsNone(evidence['finality_evidence_sha256'])
        self.assertEqual(source['lines'][0]['operation_key'],value['operation_key'])
        recorder.settle(t,issued,now_ms=1001)
        self.assertEqual(c.snapshot()['budget']['spent_known'],4)

    def test_actual_transition_captured_before_finished_request_retirement(self):
        c,d,t,recorder,binding = setup(); finish(c,t,d,1001)
        issued = receipt(recorder,t)
        result = recorder.settle(t,issued,now_ms=1002)
        self.assertEqual(result['state'],'settled')
        self.assertFalse(c.snapshot()['requests'])
        observed = recorder.receipts()[0]
        self.assertEqual(observed['instance'],'e'*64)
        self.assertEqual(observed['components'],{'other':4})
        self.assertEqual(observed['controller_transition_sha256'],digest(result))
        self.assertEqual(recorder.transition(t),result)
        self.assertEqual(c.snapshot()['budget']['spent_known'],4)
        with self.assertRaises(ValueError): recorder.settle(t,issued,now_ms=1003)

    def test_privacy_detach_preserves_opaque_binding_and_late_settlement(self):
        c,d,t,recorder,binding = setup(); issued = receipt(recorder,t)
        c.forget_subject(now_ms=1001)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        self.assertEqual(c.snapshot()['detached_financial_requests'],[t])
        recorder.settle(t,issued,now_ms=1002)
        self.assertFalse(c.snapshot()['detached_financial_requests'])
        self.assertEqual(recorder.receipts()[0]['request_id'],t)

    def test_early_settlement_and_overrun_remain_actual_policy_failure(self):
        c,d,t,recorder,binding = setup(); issued = receipt(recorder,t,billed=25)
        recorder.settle(t,issued,now_ms=1001)
        self.assertTrue(c.snapshot()['requests'][0]['charged'])
        self.assertTrue(c.snapshot()['budget']['overrun'])
        self.assertTrue(recorder.transition(t)['reservation_exceeded'])
        finish(c,t,d,1002)
        self.assertFalse(c.snapshot()['requests'])

    def test_ordinary_unknown_and_declared_attachment_do_not_clear_debt(self):
        c,d,t,recorder,binding = setup(); finish(c,t,d,1001)
        c.settle(t,components=None,now_ms=1002)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        self.assertEqual(recorder.receipts(),())
        with self.assertRaises(TypeError): FrozenRecordingReceipt(None,None,None,None,None,None,None)
        with self.assertRaises(ValueError): c.settle(t,components={'other':4},now_ms=1003,recording_receipt={})
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)

    def test_changed_components_wrong_controller_and_forged_entry_reject(self):
        c,d,t,recorder,binding = setup(); issued = receipt(recorder,t)
        with self.assertRaises(ValueError): c.settle(t,components={'other':0},now_ms=1001,recording_receipt=issued)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        c2 = controller()
        with self.assertRaises(ValueError): c2.settle(t,components={'other':4},now_ms=1001,recording_receipt=issued)
        with self.assertRaises(TypeError): recorder.bind_request(t,binding.snapshot())
        self.assertFalse(recorder.receipts())


if __name__ == '__main__': unittest.main()
