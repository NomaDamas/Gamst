from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import unittest

from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.call_budget import CallBudget
from test_call_policy import record as _record, visual


def record(ts=1000):
    r=_record(ts)
    r["models"]=[{"id":"test.provider","version":"1","backend":"plugin"}]
    return r


def controller(**kwargs):
    return CallController(policy=CallPolicy(mode='sentinel',cooldown_ms=100),
        budget=CallBudget(unit='USD.micro',pricing_revision='v1',ceiling=100,max_inflight=4),**kwargs)


def submit(c,h,seq,ts=1000,**rules):
    return c.submit(h,record(ts),sequence=seq,now_ms=ts,visual_reference=visual(ts),**rules)


def reserve(c,t,now=1000,amount=20,ttl=1000):
    return c.reserve(t,provider='test.provider',model_revision='model-1',prompt_sha256='c'*64,
        maximum_charge=amount,now_ms=now,ttl_ms=ttl)


def finish(c,t,d,now=1001,success=True):
    return c.finish(t,success=success,evidence_sha256=d['decision']['evidence']['record_sha256'],result_sha256='d'*64,now_ms=now)


class ControllerTests(unittest.TestCase):
    def test_success_and_charges_are_separate_and_capture_bound(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000)
        self.assertIsNone(c.snapshot()['sources'][h]['last_semantic'])
        result=finish(c,t,d,1100);self.assertEqual(result['semantic_capture_ms'],1000)
        c.settle(t,components=None,now_ms=1100)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        c.settle(t,components={'input':10},now_ms=1101)
        self.assertFalse(c.snapshot()['requests'])
        self.assertEqual(submit(c,h,2,1200)['status'],'context_only')
        with self.assertRaises(ValueError):finish(c,t,d,1201)

    def test_cooldown_retains_short_event_and_no_hysteresis_loss(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000);finish(c,t,d,1001)
        c.settle(t,components={'input':1},now_ms=1001)
        event=submit(c,h,2,1010,change_score=1,needs_visual=True)
        t=event['request_id'];self.assertEqual(reserve(c,t,1010)['blockers'],['cooldown'])
        self.assertEqual(submit(c,h,3,1020,change_score=.5)['status'],'context_only')
        self.assertEqual(reserve(c,t,1100)['status'],'reserved')
        evidence=c.dispatch(t,now_ms=1100)['evidence']
        self.assertEqual(evidence['capture_ms'],1010)

    def test_exact_coalescing_priority_and_explicit_overflow(self):
        c=controller(max_requests=1);h=c.open_source(record(),now_ms=1000)
        a=submit(c,h,1); self.assertEqual(submit(c,h,2)['request_id'],a['request_id'])
        rare=submit(c,h,3,1001,unknown_candidate=True)
        self.assertEqual(rare['status'],'queued')
        self.assertEqual(c.snapshot()['recent'][0]['reason'],'preempted_uninspected')
        refused=submit(c,h,4,1002,unknown_candidate=True)
        self.assertEqual(refused['reason'],'request_capacity')
        self.assertEqual(c.snapshot()['requests'][0]['decision']['evidence']['capture_ms'],1001)
        self.assertEqual(c.snapshot()['overflow_rejections'],1)

    def test_epoch_reset_discards_output_preserves_liability(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000);c.close_source(h,now_ms=1001)
        r=record(1002);r['observation']['sensor']['stream_epoch']='epoch2'
        h2=c.open_source(r,now_ms=1002)
        self.assertFalse(finish(c,t,d,1003)['accepted'])
        self.assertIsNone(c.snapshot()['sources'][h2]['last_semantic'])
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        c.settle(t,components={'input':9},now_ms=1004)
        self.assertEqual(c.snapshot()['budget']['spent_known'],9)
        with self.assertRaises(ValueError):submit(c,h,2,1005)

    def test_stale_missing_evidence_and_reservation_expiry(self):
        c=controller();h=c.open_source(record(),now_ms=1000)
        bad=c.submit(h,record(),sequence=1,now_ms=1000)
        self.assertEqual(bad['status'],'rejected')
        d=submit(c,h,2);t=d['request_id'];reserve(c,t,ttl=10)
        self.assertIn('reservation_expired',c.dispatch(t,now_ms=1010)['blockers'])
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],0)
        self.assertEqual(reserve(c,t,2001)['blockers'],['stale_capture'])
        self.assertEqual(c.expire(now_ms=61000),1)
        self.assertEqual(c.snapshot()['recent'][0]['reason'],'expired_uninspected')
        self.assertIsNone(c.snapshot()['sources'][h]['last_semantic'])

    def test_close_cancel_and_late_charge(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000)
        c.cancel(t,now_ms=1001);c.close(now_ms=1002)
        self.assertFalse(finish(c,t,d,1003)['accepted'])
        c.settle(t,components=None,now_ms=1004)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        c.settle(t,components={'other':25},now_ms=1005)
        self.assertTrue(c.snapshot()['budget']['overrun'])
        self.assertFalse(c.snapshot()['requests'])

    def test_settlement_before_result_and_failure_does_not_cover_audit(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000);c.settle(t,components={'other':0},now_ms=1001)
        self.assertTrue(c.snapshot()['requests'][0]['charged'])
        finish(c,t,d,1002,False)
        self.assertIsNone(c.snapshot()['sources'][h]['last_semantic'])
        self.assertEqual(submit(c,h,2,1100)['status'],'queued')

    def test_owner_replay_validation_and_copied_snapshot(self):
        c=controller();h=c.open_source(record(),now_ms=1000)
        with self.assertRaises(ValueError):c.open_source(record(),now_ms=1000)
        r=record();r['observation']['processing_epoch']='new'
        with self.assertRaises(ValueError):c.submit(h,r,sequence=1,now_ms=1000)
        d=submit(c,h,1)
        with self.assertRaises(ValueError):submit(c,h,1)
        with self.assertRaises(ValueError):c.submit(h,record(),sequence=2,now_ms=1000,last_dispatch_ms=None)
        s=c.snapshot();s['sources'][h]['last_semantic']=99999
        self.assertIsNone(c.snapshot()['sources'][h]['last_semantic'])
        t=d['request_id'];reserve(c,t);c.dispatch(t,now_ms=1000)
        with self.assertRaises(ValueError):c.finish(t,success=True,evidence_sha256='f'*64,result_sha256='d'*64,now_ms=1001)

    def test_expiry_sweep_releases_reservation_without_erasing_event(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t,ttl=10)
        self.assertEqual(c.expire(now_ms=1010),1)
        self.assertEqual(c.snapshot()['requests'][0]['state'],'pending')
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],0)
        self.assertEqual(reserve(c,t,1011)['status'],'reserved')

    def test_retry_while_failed_attempt_has_unknown_bill(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000);finish(c,t,d,1001,False)
        c.settle(t,components=None,now_ms=1001)
        retry=c.submit(h,record(),sequence=2,now_ms=1100,visual_reference=visual())
        self.assertEqual(retry['status'],'queued');self.assertNotEqual(retry['request_id'],t)
        reserve(c,retry['request_id'],1100);c.dispatch(retry['request_id'],now_ms=1100)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],40)

    def test_failed_threshold_retries_preserve_newer_hysteresis_and_billing(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id']
        reserve(c,t);c.dispatch(t,now_ms=1000);finish(c,t,d,1001)
        c.settle(t,components={'input':1},now_ms=1001)
        event=submit(c,h,2,1100,change_score=1,needs_visual=True);old=event['request_id']
        reserve(c,old,1100);c.dispatch(old,now_ms=1100);finish(c,old,event,1101,False)
        c.settle(old,components=None,now_ms=1101)
        submit(c,h,3,1150,change_score=0)
        retry=c.submit(h,record(1100),sequence=4,now_ms=1200,visual_reference=visual(1100),change_score=1,needs_visual=True)
        self.assertEqual(retry['status'],'queued')
        self.assertFalse(c.snapshot()['sources'][h]['active'])
        self.assertEqual(c.snapshot()['sources'][h]['last_capture'],1150)
        new=retry['request_id'];reserve(c,new,1200);c.dispatch(new,now_ms=1200)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],40)
        finish(c,new,retry,1201,False);c.settle(new,components={'input':1},now_ms=1201)
        again=c.submit(h,record(1100),sequence=5,now_ms=1300,visual_reference=visual(1100),change_score=1,needs_visual=True)
        self.assertEqual(again['status'],'queued')

    def test_provider_version_and_channel_change_require_new_source(self):
        c=controller();h=c.open_source(record(),now_ms=1000)
        for mutate in ('model','channels'):
            r=record()
            if mutate=='model':r['models'][0]['version']='2'
            else:r['observation']['sensor']['channels']=['other']
            with self.assertRaises(ValueError):c.submit(h,r,sequence=1,now_ms=1000,visual_reference=visual())
        r=record();r.pop('models')
        with self.assertRaises(ValueError):c.open_source(r,now_ms=1000)

    def test_dispatched_observation_copy_is_not_retained_for_billing(self):
        c=controller();r=record();r['observation']['value']={'text':'private-lived-context'}
        h=c.open_source(r,now_ms=1000)
        d=c.submit(h,r,sequence=1,now_ms=1000,visual_reference=visual())
        t=d['request_id'];reserve(c,t)
        admitted=c.dispatch(t,now_ms=1000)
        self.assertEqual(admitted['observation'],r)
        self.assertNotIn('private-lived-context',repr(c._requests))
        admitted['observation']['observation']['value']['text']='mutated'
        c.cancel(t,now_ms=1001);c.close_source(h,now_ms=1002)
        self.assertEqual(c.snapshot()['budget']['reserved_liability'],20)
        self.assertNotIn('private-lived-context',repr(c._requests))

    def test_concurrent_duplicate_dispatch_and_completion(self):
        c=controller();h=c.open_source(record(),now_ms=1000);d=submit(c,h,1);t=d['request_id'];reserve(c,t)
        def dispatch(_):
            try:return c.dispatch(t,now_ms=1000)['status']=='dispatched'
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=8) as pool:self.assertEqual(sum(pool.map(dispatch,range(32))),1)
        def done(_):
            try:finish(c,t,d);return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=8) as pool:self.assertEqual(sum(pool.map(done,range(32))),1)
        self.assertEqual(c.snapshot()['budget']['dispatched'],1)


class RegisteredIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_registered_store_context_remains_independent(self):
        from humansense import HumanSense, SensorMetadata
        from humansense.capabilities import CapabilitySpec, OutputField
        from humansense.schema import timestamp
        spec=CapabilitySpec('test.measured','test.measured','video','test.provider','1',
            (OutputField('count','integer'),),'frame',freshness_ms=1000)
        async with HumanSense(consent=True) as human:
            human.register_capability(spec); now=timestamp()
            await human.push_observation(spec.id,{'count':2},sensor=SensorMetadata('cam','video',0,
                coordinate_frame='pixels',stream_epoch='epoch'),timestamp_ms=now)
            records=human.store.latest_signals('local-user',0,now);r=records[0]
            c=controller();h=c.open_source(r,now_ms=now)
            d=c.submit(h,r,sequence=1,now_ms=now,visual_reference=visual(now))
            t=d['request_id'];reserve(c,t,now);c.dispatch(t,now_ms=now)
            finish(c,t,d,now);c.settle(t,components={'local_compute':2},now_ms=now)
            self.assertEqual(human.store.latest_signals('local-user',0,now),records)
            self.assertEqual(c.snapshot()['sources'][h]['last_semantic'],now)
            c.close(now_ms=now)
