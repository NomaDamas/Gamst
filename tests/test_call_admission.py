import json
import unittest
from gamst.call_controller import CallController
from gamst.call_policy import CallPolicy
from test_call_controller import record, visual


def make(**kwargs):
    return CallController(policy=CallPolicy(mode='sentinel',cooldown_ms=0), **kwargs)


def queued(c,seq=1,ts=1000,handle=None,**rules):
    h=handle or c.open_source(record(ts),now_ms=ts)
    d=c.submit(h,record(ts),sequence=seq,now_ms=ts,visual_reference=visual(ts),**rules)
    return h,d['request_id']


def reserve(c,t,now=1000):
    return c.reserve(t,provider='test.vlm',model_revision='1',prompt_sha256='a'*64,now_ms=now)


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
        a=c.snapshot()['recent'][0]
        self.assertFalse(a['unknown_candidate_unresolved']);self.assertTrue(a['outcome']['accepted'])
        self.assertEqual(a['outcome']['semantic_capture_ms'],1000)
        self.assertFalse(c.snapshot()['sources'][h]['semantic_audit_due'])
        self.assertTrue(c.snapshot(now_ms=6000)['sources'][h]['semantic_audit_due'])

    def test_reservation_expiry_removes_advisory_permission(self):
        c=make();h,t=queued(c);reserve(c,t)
        self.assertEqual(c.snapshot(now_ms=2000)['requests'][0]['allowed_route'],'none')
        self.assertIn('reservation_expired',c.snapshot()['requests'][0]['admission_blockers'])

    def test_text_query_does_not_claim_visual_coverage(self):
        c=make();h=c.open_source(record(),now_ms=1000)
        d=c.submit(h,record(),sequence=1,now_ms=1000,query=True)
        t=d['request_id'];reserve(c,t);a=c.dispatch(t,now_ms=1000)
        c.finish(t,success=True,evidence_sha256=a['evidence']['record_sha256'],result_sha256='b'*64,now_ms=1000)
        self.assertTrue(c.snapshot()['sources'][h]['semantic_audit_due'])

    def test_overdue_source_and_expired_unknown_receipt_stay_unresolved(self):
        c=make();h,t=queued(c,unknown_candidate=True)
        c.expire(now_ms=61000)
        self.assertTrue(c.snapshot()['sources'][h]['semantic_audit_due'])
        self.assertTrue(c.snapshot()['recent'][0]['unknown_candidate_unresolved'])
        self.assertEqual(c.snapshot()['recent'][0]['trigger_reason'],'unknown_candidate')


class Wire(unittest.IsolatedAsyncioTestCase):
    async def test_actual_mcp_admission_provenance_and_delete_redaction(self):
        from gamst import Gamst,SensorMetadata
        from gamst.capabilities import CapabilitySpec,OutputField
        from gamst.schema import timestamp
        from gamst.mcp import MCPServer
        async with Gamst(consent=True) as human:
            spec=CapabilitySpec('test.c','test.c','video','test.provider','1',(OutputField('text','string'),),'frame',freshness_ms=1000)
            human.register_capability(spec);c=make();human.register_call_controller('main',c)
            r=await human.push_observation(spec.id,{'text':'PRIVATE-DATA'},subject_id='alice',sensor=SensorMetadata('cam','video',0,coordinate_frame='pixels',stream_epoch='epoch'))
            h=c.open_source(r,now_ms=timestamp())
            d=c.submit(h,r,sequence=1,now_ms=timestamp(),visual_reference=visual(r['timestamp_ms']),unknown_candidate=True)
            reserve(c,d['request_id'],timestamp())
            server=MCPServer(human)
            response=await server.call_tool('get_call_status',{'controller_id':'main'})
            request=next(v for v in response['entries'] if v['kind']=='request')
            self.assertEqual(request['allowed_route'],'small_vlm');self.assertTrue(request['unknown_candidate_unresolved'])
            self.assertEqual(request['last_admission']['status'],'reserved')
            self.assertNotIn('PRIVATE-DATA',json.dumps(response))
            await human.delete_data('alice')
            response=await server.call_tool('get_call_status',{})
            self.assertNotIn('alice',json.dumps(response));self.assertNotIn('test.vlm',json.dumps(response))
            self.assertEqual(response['entries'][0]['admission_denials'],{})
            await server.close()
