from concurrent.futures import ThreadPoolExecutor
import unittest

from humansense.call_budget import CallBudget


def ledger(**kwargs):
    return CallBudget(unit='USD.micro', pricing_revision='rates-1', ceiling=100, **kwargs)


def reserve(b, seq=1, amount=30, now=0, **kwargs):
    return b.reserve(sequence=seq, decision_sha256='a'*64, evidence_sha256='b'*64,
        provider='test.provider', model_revision='model-1', maximum_charge=amount,
        now_ms=now, **kwargs)['token']


class BudgetTests(unittest.TestCase):
    def test_chain_prices_and_exact_boundary(self):
        b=ledger(); t=reserve(b, amount=100); b.dispatch(t, now_ms=1)
        self.assertEqual(b.snapshot()['available'],0)
        b.settle(t,components={'input':10,'output':20,'cache_read':3,'cache_write':2,'media':5,'local_compute':10},now_ms=2)
        t2=reserve(b,2,50,3); b.dispatch(t2,now_ms=4)
        b.settle(t2,components={'text_consumer':30,'other':20},now_ms=5)
        s=b.snapshot(); self.assertEqual(s['spent_known'],100);self.assertEqual(s['reserved_liability'],0)
        self.assertEqual(sum(s['known_components'].values()),100)
        with self.assertRaises(RuntimeError):reserve(b,3,1,6)

    def test_unknown_cancel_and_late_settlement(self):
        b=ledger(max_inflight=1); t=reserve(b); b.dispatch(t,now_ms=1)
        b.settle(t,components=None,now_ms=2);b.settle(t,components=None,now_ms=3)
        self.assertEqual(b.snapshot()['unknown_reports'],1)
        with self.assertRaises(ValueError):b.release(t,now_ms=4)
        self.assertEqual(b.expire(now_ms=2000),0)
        with self.assertRaises(RuntimeError):reserve(b,2,0,2001)
        b.close(now_ms=2002);self.assertEqual(b.snapshot()['reserved_liability'],30)
        b.settle(t,components={'other':25},now_ms=2003)
        self.assertEqual(b.snapshot()['spent_known'],25)
        self.assertEqual(b.snapshot()['unresolved_dispatched'],0)
        with self.assertRaises(RuntimeError):reserve(b,2,1,2004)

    def test_expiry_and_release_do_not_refund_dispatched(self):
        b=ledger();a=reserve(b,ttl_ms=5);c=reserve(b,2)
        with self.assertRaises(ValueError):b.dispatch(a,now_ms=5)
        b.dispatch(c,now_ms=5);self.assertEqual(b.expire(now_ms=1001),0)
        with self.assertRaises(ValueError):b.release(c,now_ms=1002)
        self.assertEqual(b.snapshot()['reserved_liability'],30)
        d=reserve(b,3,10,1003);b.close(now_ms=1004)
        self.assertEqual(b.snapshot()['released'],2)
        with self.assertRaises(ValueError):b.dispatch(d,now_ms=1005)

    def test_overspend_is_visible_and_blocks_admission_and_dispatch(self):
        b=ledger();a=reserve(b);c=reserve(b,2)
        b.dispatch(a,now_ms=1);b.settle(a,components={'other':110},now_ms=2)
        s=b.snapshot();self.assertTrue(s['overrun']);self.assertEqual(s['spent_known'],110)
        self.assertEqual(s['reserved_liability'],30)
        with self.assertRaises(RuntimeError):reserve(b,3,0,3)
        with self.assertRaises(RuntimeError):b.dispatch(c,now_ms=4)
        b.release(c,now_ms=5)
        self.assertTrue(b.snapshot()['overrun'])

    def test_bounded_receipts_keep_aggregates_and_reject_replay(self):
        b=ledger(receipt_limit=1,max_calls=3)
        old=[]
        for seq in (1,2,3):
            t=reserve(b,seq,10); old.append(t); b.dispatch(t,now_ms=0)
            b.settle(t,components={'input':10},now_ms=0)
        s=b.snapshot();self.assertEqual(len(s['recent']),1);self.assertEqual(s['omitted_receipts'],2)
        self.assertEqual(s['spent_known'],30)
        for t in old:
            with self.assertRaises(ValueError):b.settle(t,components={'input':0},now_ms=0)
        with self.assertRaises(ValueError):reserve(b,1)
        with self.assertRaises(RuntimeError):reserve(b,4)
        other=ledger()
        with self.assertRaises(ValueError):other.dispatch(old[0],now_ms=0)

    def test_concurrent_reservation_and_settlement(self):
        b=ledger(max_inflight=100)
        def attempt(seq):
            try:return reserve(b,seq,10)
            except (ValueError,RuntimeError):return None
        with ThreadPoolExecutor(max_workers=16) as pool:
            tokens=[t for t in pool.map(attempt,range(1,101)) if t]
        self.assertLessEqual(len(tokens),10)
        self.assertEqual(b.snapshot()['reserved_liability'],len(tokens)*10)
        t=tokens[0];b.dispatch(t,now_ms=0)
        def settle(_):
            try:b.settle(t,components={'input':7},now_ms=0);return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=16) as pool:self.assertEqual(sum(pool.map(settle,range(32))),1)
        self.assertEqual(b.snapshot()['spent_known'],7)

    def test_validation_and_returned_snapshot_ownership(self):
        b=ledger(); t=reserve(b)
        s=b.snapshot();s['active'][0]['maximum_charge']=0
        self.assertEqual(b.snapshot()['reserved_liability'],30)
        with self.assertRaises(AttributeError):b.ceiling=999999
        for amount in (None,True,-1,1.5,float('nan')):
            with self.assertRaises(ValueError):reserve(b,2,amount)
        with self.assertRaises(ValueError):b.settle(t,components={'input':1},now_ms=1)
        b.dispatch(t,now_ms=2)
        with self.assertRaises(ValueError):b.dispatch(t,now_ms=2)
        for components in ({},{'unknown':1},{'input':None},{'input':True}):
            with self.assertRaises(ValueError):b.settle(t,components=components,now_ms=2)
        with self.assertRaises(ValueError):b.expire(now_ms=1)
        self.assertEqual(b.snapshot()['spent_known'],0)
