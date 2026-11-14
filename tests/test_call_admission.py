import json
import unittest
from humansense.call_budget import CallBudget, BudgetDenied
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from test_call_controller import record, visual


def make(**kwargs):
    return CallController(policy=CallPolicy(mode='sentinel',cooldown_ms=0),
        budget=CallBudget(unit='USD.micro',pricing_revision='v1',ceiling=20,**kwargs))


def queued(c,seq=1,ts=1000,handle=None,**rules):
    h=handle or c.open_source(record(ts),now_ms=ts)
    d=c.submit(h,record(ts),sequence=seq,now_ms=ts,visual_reference=visual(ts),**rules)
    return h,d['request_id']


def reserve(c,t,charge=10,now=1000):
    return c.reserve(t,provider='test.vlm',model_revision='1',prompt_sha256='a'*64,
        maximum_charge=charge,now_ms=now)


class Admission(unittest.TestCase):
    def test_recommendation_reservation_dispatch_are_distinct(self):
        c=make();h,t=queued(c,unknown_candidate=True)
        a=c.snapshot()['requests'][0]
        self.assertEqual(a['allowed_route'],'none');self.assertIn('reservation_required',a['admission_blockers'])
        reserve(c,t);a=c.snapshot()['requests'][0]
        self.assertEqual(a['allowed_route'],'small_vlm');self.assertTrue(a['admission_advisory'])
        self.assertTrue(a['semantic_audit_due']);self.assertTrue(a['unknown_candidate_unresolved'])
        d=c.dispatch(t,now_ms=1000);a=c.snapshot()['requests'][0]
        self.assertEqual(a['allowed_route'],'none');self.assertEqual(a['dispatched_route'],'small_vlm')
        self.assertTrue(a['unknown_candidate_unresolved'])
        c.finish(t,success=True,evidence_sha256=d['evidence']['record_sha256'],result_sha256='b'*64,now_ms=1001)
        a=c.snapshot()['requests'][0]
        self.assertFalse(a['unknown_candidate_unresolved']);self.assertFalse(a['semantic_audit_due'])
        self.assertEqual(a['last_semantic_capture_ms'],1000)
        self.assertTrue(c.snapshot(now_ms=6000)['requests'][0]['semantic_audit_due'])

    def test_unknown_price_and_exhaustion_preserve_pending_event(self):
        c=make();h,t=queued(c,unknown_candidate=True)
        for quote,reason in ((None,'unknown_charge_ceiling'),(21,'budget_exhausted')):
            result=reserve(c,t,quote)
            self.assertEqual(result,{'status':'blocked','blockers':[reason]})
            a=c.snapshot()['requests'][0]
            self.assertEqual(a['state'],'pending');self.assertEqual(a['allowed_route'],'none')
            self.assertEqual(a['last_admission']['reasons'],[reason])
            self.assertEqual(a['last_admission']['candidate']['provider'],'test.vlm')
            self.assertTrue(a['unknown_candidate_unresolved'])
        self.assertEqual(c.snapshot()['budget']['accepted'],0)
        self.assertEqual(c.snapshot()['admission_denials'],{'unknown_charge_ceiling':1,'budget_exhausted':1})
        reserve(c,t,1);self.assertEqual(c.snapshot()['requests'][0]['last_admission']['status'],'reserved')
        self.assertEqual(c.snapshot()['admission_denials']['budget_exhausted'],1)

    def test_call_and_inflight_limits_have_stable_reasons(self):
        for opts,reason in (({'max_calls':1},'call_limit'),({'max_inflight':1},'inflight_limit')):
            c=make(**opts);h,a=queued(c);_,b=queued(c,2,1001,h,query=True)
            reserve(c,a,1,1001)
            self.assertEqual(reserve(c,b,1,1001)['blockers'],[reason])
            self.assertEqual(c.snapshot()['budget']['accepted'],1)
            self.assertEqual(c.snapshot()['requests'][1 if c.snapshot()['requests'][0]['id']==a else 0]['state'],'pending')

    def test_reservation_expiry_and_overrun_remove_advisory_permission(self):
        c=make();h,a=queued(c);_,b=queued(c,2,1001,h,query=True)
        reserve(c,a,1,1001);reserve(c,b,1,1001)
        c.dispatch(a,now_ms=1001);c.settle(a,components={'input':2},now_ms=1001)
        view=next(r for r in c.snapshot()['requests'] if r['id']==b)
        self.assertEqual(view['allowed_route'],'none');self.assertIn('reservation_overrun',view['admission_blockers'])
        self.assertEqual(c.dispatch(b,now_ms=1001)['blockers'],['reservation_overrun'])
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],1)
        c.cancel(b,now_ms=1001)
        self.assertEqual(c.snapshot()['recent'][-1]['last_admission']['reasons'],['reservation_overrun'])
        c=make();h,t=queued(c);reserve(c,t)
        self.assertEqual(c.snapshot(now_ms=2000)['requests'][0]['allowed_route'],'none')
        self.assertIn('reservation_expired',c.snapshot()['requests'][0]['admission_blockers'])

    def test_text_query_does_not_claim_visual_coverage(self):
        c=make();h=c.open_source(record(),now_ms=1000)
        d=c.submit(h,record(),sequence=1,now_ms=1000,query=True)
        t=d['request_id'];reserve(c,t);a=c.dispatch(t,now_ms=1000)
        c.finish(t,success=True,evidence_sha256=a['evidence']['record_sha256'],result_sha256='b'*64,now_ms=1000)
        self.assertTrue(c.snapshot()['requests'][0]['semantic_audit_due'])

    def test_overdue_source_and_expired_unknown_receipt_stay_unresolved(self):
        c=make();h,t=queued(c,unknown_candidate=True)
        c.expire(now_ms=61000)
        self.assertTrue(c.snapshot()['sources'][h]['semantic_audit_due'])
        self.assertTrue(c.snapshot()['recent'][0]['unknown_candidate_unresolved'])
        self.assertEqual(c.snapshot()['recent'][0]['trigger_reason'],'unknown_candidate')

    def test_budget_denial_type_and_invalid_inputs(self):
        b=CallBudget(unit='u',pricing_revision='v',ceiling=0)
        with self.assertRaises(BudgetDenied) as exc:
            b.reserve(sequence=1,decision_sha256='a'*64,evidence_sha256='b'*64,
                provider='p',model_revision='1',maximum_charge=1,now_ms=0)
        self.assertIsInstance(exc.exception,RuntimeError);self.assertEqual(exc.exception.reason,'budget_exhausted')
        c=make();h,t=queued(c)
        for invalid in (-1,True,float('nan')):
            with self.assertRaises(ValueError):reserve(c,t,invalid)
        self.assertEqual(c.snapshot()['admission_denials'],{})


class Wire(unittest.IsolatedAsyncioTestCase):
    async def test_actual_mcp_denial_provenance_and_delete_redaction(self):
        from humansense import HumanSense,SensorMetadata
        from humansense.capabilities import CapabilitySpec,OutputField
        from humansense.schema import timestamp
        from humansense.mcp import MCPServer
        async with HumanSense(consent=True) as human:
            spec=CapabilitySpec('test.c','test.c','video','test.provider','1',(OutputField('text','string'),),'frame',freshness_ms=1000)
            human.register_capability(spec);c=make();human.register_call_controller('main',c)
            r=await human.push_observation(spec.id,{'text':'PRIVATE-DATA'},subject_id='alice',sensor=SensorMetadata('cam','video',0,coordinate_frame='pixels',stream_epoch='epoch'))
            h=c.open_source(r,now_ms=timestamp())
            d=c.submit(h,r,sequence=1,now_ms=timestamp(),visual_reference=visual(r['timestamp_ms']),unknown_candidate=True)
            reserve(c,d['request_id'],None,timestamp())
            server=MCPServer(human)
            response=await server.call_tool('get_call_status',{'controller_id':'main'})
            request=next(v for v in response['entries'] if v['kind']=='request')
            self.assertEqual(request['allowed_route'],'none');self.assertTrue(request['unknown_candidate_unresolved'])
            self.assertEqual(request['last_admission']['reasons'],['unknown_charge_ceiling'])
            self.assertNotIn('PRIVATE-DATA',json.dumps(response))
            await human.delete_data('alice')
            response=await server.call_tool('get_call_status',{})
            self.assertNotIn('alice',json.dumps(response));self.assertNotIn('test.vlm',json.dumps(response))
            self.assertEqual(response['entries'][0]['admission_denials']['unknown_charge_ceiling'],1)
            await server.close()
