"""Protocol fixtures only: controlled Smol worker, no trained model/device proof."""
import asyncio
from dataclasses import replace
import json
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense, FreshFrameSentinel, SensorMetadata
from humansense.call_budget import CallBudget
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.mcp import MCPServer
from humansense.schema import timestamp
from humansense.smolvlm_call import smolvlm_frame_sha256
import test_smolvlm_context as smol_tests
from test_smolvlm_context import frame, np, WORKER
from test_smolvlm_native import CONFIG


@unittest.skipIf(np is None,'optional numpy absent; no installation')
class Sentinel(unittest.IsolatedAsyncioTestCase):
    fixture=smol_tests.Managed.fixture
    async def asyncSetUp(self):
        self.children=[];self.spawn=asyncio.create_subprocess_exec;self.humans=[]
    async def asyncTearDown(self):
        for human in self.humans:await human.close()
        self.assertTrue(all(child.returncode is not None for child in self.children))
    def human(self,budget=None,**policy):
        h=HumanSense(consent=True);self.humans.append(h)
        settings=dict(mode='sentinel',semantic_interval_ms=1,cooldown_ms=0);settings.update(policy)
        c=CallController(policy=CallPolicy(**settings),
            budget=budget or CallBudget(unit='fixture',pricing_revision='unpriced-test',ceiling=10000,max_inflight=128))
        h.register_call_controller('main',c);return h,c
    def frame(self,index=0,ts=None,**changes):
        f=frame(index,timestamp() if ts is None else ts)
        return replace(f,sensor=replace(f.sensor,stream_epoch='camera-epoch',capture_sequence=index),**changes)
    async def open(self,h,**options):
        f=self.frame()
        settings=dict(maximum_charge=2,interval_ms=30,deadline_tolerance_ms=29);settings.update(options)
        return await FreshFrameSentinel.open(h,'audit','main',sys.executable,CONFIG,sensor=f.sensor,
            frame_shape=f.data.shape,**settings)
    async def until(self,predicate,timeout=5):
        async def wait():
            while not predicate():await asyncio.sleep(.001)
        await asyncio.wait_for(wait(),timeout)
    async def tick(self,s):
        import time
        s._origin=time.monotonic_ns()-s._next_tick*s.config['interval_ms']*1000000-s.config['phase_ms']*1000000
        await s._tick()

    async def test_real_timer_without_any_detector_original_context_and_readonly_mcp(self):
        with self.fixture():
            h,c=self.human();s=await self.open(h)
            f=self.frame();expected=smolvlm_frame_sha256(f.data);s.offer(f);f.data[:]=97;s.start()
            await self.until(lambda:s.snapshot()['counters'].get('accepted',0)==1)
            point=s.snapshot()['last_accepted_capture_point']
            self.assertEqual(point['capture_ms'],f.timestamp_ms);self.assertEqual(point['frame_sha256'],expected)
            self.assertIsNone(point.get('actual_cost'));self.assertEqual(point['capture_index'],0)
            context=await h.get_context();self.assertIn('worker description',str(context))
            rows=h.store.query('local-user',kind='observation.signal',name=s.adapter.worker.spec.output_name)
            self.assertEqual(rows[0]['timestamp_ms'],f.timestamp_ms)
            self.assertEqual(rows[0]['observation']['value']['window']['captures'][0]['capture_sequence'],0)
            await self.until(lambda:s.snapshot()['counters'].get('capture_unavailable',0)>=2)
            self.assertEqual(c.snapshot()['budget']['dispatched'],1)
            before=s.snapshot()['counters'].copy()
            reply=await MCPServer(h).call_tool('get_sentinel_status',{})
            self.assertIn('sampled_capture_points',str(reply));self.assertNotIn('worker description',json.dumps(reply))
            self.assertEqual(s.snapshot()['counters'],before)
            with self.assertRaises(ValueError):s.offer(f)
            self.assertFalse(s.snapshot()['continuous_coverage_verified'])

    async def test_slow_worker_latest_slot_does_not_block_audio_or_context(self):
        with self.fixture(delay=.2):
            h,c=self.human();s=await self.open(h);f=self.frame();digest=smolvlm_frame_sha256(f.data)
            s.offer(f);s.start()
            await self.until(lambda:c.snapshot()['budget']['dispatched']==1)
            raw=s._active_frame['frame'].data
            for i in range(1,8):
                candidate=self.frame(i);s.offer(candidate);candidate.data[:]=200
                await h.push_signal('audio.independent',i,'audio')
                await h.get_context()
                self.assertLessEqual(s.snapshot()['owned_pixel_bytes'],2*f.data.nbytes)
            self.assertEqual(s.snapshot()['counters']['candidate_replaced'],6)
            await self.until(lambda:s.snapshot()['counters'].get('busy',0)>=1)
            await self.until(lambda:s.snapshot()['counters'].get('accepted',0)>=1)
            self.assertTrue(np.all(raw==0))
            rows=h.store.query('local-user',kind='observation.signal',name=s.adapter.worker.spec.output_name)
            self.assertEqual(rows[0]['observation']['value']['window']['captures'][0]['pixels_sha256'],
                __import__('hashlib').sha256(f.data.tobytes()).hexdigest())
            self.assertEqual(s.snapshot()['last_accepted_capture_point']['frame_sha256'],digest)

    async def test_lifecycle_scrubs_before_reap_and_preserves_unknown_or_early_bill(self):
        for action in ('pause','reset','delete','revoke','close','early_delete'):
            with self.subTest(action=action),self.fixture(delay=60):
                h,c=self.human();s=await self.open(h);s.offer(self.frame());s.start()
                await self.until(lambda:c.snapshot()['budget']['dispatched']==1)
                rid=s._active_meta['request_id'];active=s._active_frame['frame'].data
                s.offer(self.frame(1));pending=s._candidate['frame'].data
                if action=='early_delete':c.settle(rid,components={'local_compute':1},now_ms=timestamp())
                if action=='revoke':h.consent=False
                else:
                    method={'pause':h.pause,'reset':h.reset_processing,'delete':h.delete_data,
                            'early_delete':h.delete_data,'close':h.close}[action]
                    task=asyncio.create_task(method());await asyncio.sleep(0)
                self.assertTrue(np.all(active==0));self.assertTrue(np.all(pending==0))
                self.assertIsNone(s.snapshot()['binding']);self.assertIsNone(s.snapshot()['last_accepted_capture_point'])
                if action!='revoke':await task
                await s.close()
                self.assertEqual(s.snapshot()['owned_pixel_bytes'],0)
                self.assertTrue(all(child.returncode is not None for child in self.children))
                state=c.snapshot()['budget']
                if action=='early_delete':self.assertEqual(state['spent_known'],1)
                else:
                    self.assertEqual(state['reserved_liability'],2)
                    c.settle(rid,components={'local_compute':1},now_ms=timestamp());self.assertEqual(c.snapshot()['budget']['spent_known'],1)
                if not h.closed:self.assertFalse(h.store.query('local-user',kind='observation.signal',name=s.adapter.worker.spec.output_name))

    async def test_binding_change_and_startup_failure_release_only_after_owned_reap(self):
        with self.fixture():
            h,c=self.human();s=await self.open(h);s.offer(self.frame());owned=s._candidate['frame'].data
            changed=self.frame(1);changed=replace(changed,sensor=replace(changed.sensor,calibration_id='replacement'))
            with self.assertRaises(ValueError):s.offer(changed)
            self.assertTrue(np.all(owned==0));self.assertTrue(s.snapshot()['invalidated'])
            await s.close();self.assertFalse(h._call_drivers)
            second=await self.open(h);await second.close();self.assertFalse(h._call_drivers)
        with self.fixture(code='raise RuntimeError("controlled startup failure")'):
            with self.assertRaises(Exception):await self.open(h)
            self.assertFalse(h._call_drivers);self.assertFalse(h._call_driver_cleanup)

    async def test_capture_clock_shape_index_freshness_and_pixel_admission_before_copy(self):
        with self.fixture():
            h,c=self.human();s=await self.open(h,max_age_ms=100)
            now=timestamp()
            for f in (self.frame(0,now+5000),self.frame(0,-1)):
                with self.assertRaises(ValueError):s.offer(f)
            with patch('numpy.asarray',side_effect=AssertionError('stale frame copied')):
                self.assertFalse(s.offer(self.frame(0,now-200))['accepted'])
            a=self.frame(0);s.offer(a);b=self.frame(1,a.timestamp_ms);self.assertTrue(s.offer(b)['accepted'])
            with self.assertRaises(ValueError):s.offer(b)
            stale=self.frame(2,a.timestamp_ms-1)
            with self.assertRaises(ValueError):s.offer(stale)
            with patch('humansense.frame_sentinel.timestamp',return_value=a.timestamp_ms-1):
                with self.assertRaises(ValueError):s.offer(self.frame(2))
            await s.close()
            with self.assertRaises(ValueError):await self.open(h,capture_clock='arbitrary_stream_ms')
            self.assertFalse(h._call_drivers)

    async def test_skipped_ticks_and_capture_gaps_never_burst_catchup(self):
        import time
        with self.fixture():
            h,c=self.human();s=await self.open(h);s.offer(self.frame())
            s._origin=time.monotonic_ns()-9*s.config['interval_ms']*1000000
            await s._tick();await self.until(lambda:s._active_task is None)
            self.assertEqual(s.snapshot()['counters']['skipped_ticks'],9)
            self.assertEqual(c.snapshot()['budget']['dispatched'],1)
            s.offer(self.frame(10));await self.tick(s);await self.until(lambda:s._active_task is None)
            self.assertEqual(s.snapshot()['counters']['capture_index_gap'],1)
            self.assertTrue(any(e['kind']=='between_sample_points' for e in s.snapshot()['events']))
            self.assertEqual(c.snapshot()['budget']['dispatched'],2)

    async def test_short_event_phase_aliasing_reports_sampled_points_and_known_input_misses(self):
        # The worker always returns a fixture description: this measures only
        # whether rare input pixels reach sampling, never recognition accuracy.
        with self.fixture():
            h,c=self.human();misses=0;hits=0
            for duration in (5,20,70):
                for phase in (30,105,190):
                    s=await self.open(h,interval_ms=100,deadline_tolerance_ms=99)
                    s._origin=1000000000;selected=[];event_indices=set()
                    try:
                        for t in range(0,301,5):
                            index=t//5;present=phase<=t<phase+duration
                            if present:event_indices.add(index)
                            f=self.frame(index);f.data[:]=255 if present else 0
                            with patch('humansense.frame_sentinel.time.monotonic_ns',return_value=s._origin+t*1000000):
                                s.offer(f)
                                if t%100==0:
                                    await s._tick();await self.until(lambda:s._active_task is None)
                                    point=s.snapshot()['last_accepted_capture_point'];self.assertIsNotNone(point)
                                    selected.append(point['capture_index'])
                        observed=bool(set(selected)&event_indices)
                        self.assertEqual(observed,any(phase<=t<phase+duration for t in (0,100,200,300)))
                        if observed:hits+=1
                        else:misses+=1
                        self.assertFalse(s.snapshot()['continuous_coverage_verified'])
                        self.assertGreater(s.snapshot()['counters']['candidate_replaced'],0)
                    finally:await s.close()
            self.assertGreater(misses,0);self.assertGreater(hits,0)

    async def test_unknown_quote_empty_error_and_context_only_never_advance_inspection(self):
        cases=('quote','budget','empty','error','context_only')
        for case in cases:
            code=WORKER.replace('DELAY','0')
            if case=='empty':code=code.replace("text='worker description'","text=''")
            if case=='error':code=code.replace("time.sleep(0)","raise RuntimeError('PRIVATE BACKEND FAILURE')")
            with self.subTest(case=case),self.fixture(code=code):
                h,c=self.human(mode='context_only' if case=='context_only' else 'sentinel')
                charge=None if case=='quote' else 10001 if case=='budget' else 2
                s=await self.open(h,maximum_charge=charge);s.offer(self.frame());await self.tick(s)
                await self.until(lambda:s._active_task is None)
                status=s.snapshot();self.assertIsNone(status['last_accepted_capture_point'])
                self.assertEqual(status['counters'].get('accepted',0),0)
                self.assertNotIn('PRIVATE',json.dumps(status));self.assertFalse(status['semantic_quality_verified'])
                self.assertEqual(c.snapshot()['budget']['dispatched'],1 if case in ('empty','error') else 0)
                self.assertTrue(all(source['last_semantic'] is None for source in c.snapshot()['sources'].values()))
                await s.close()

    async def test_repeated_cancel_close_joins_worker_and_preserves_debt(self):
        with self.fixture(delay=60):
            h,c=self.human();s=await self.open(h);s.offer(self.frame());s.start()
            await self.until(lambda:c.snapshot()['budget']['dispatched']==1)
            close=asyncio.create_task(s.close());await asyncio.sleep(0);close.cancel();await asyncio.sleep(0);close.cancel()
            with self.assertRaises(asyncio.CancelledError):await close
            await asyncio.gather(s.close(),s.close())
            self.assertTrue(all(child.returncode is not None for child in self.children))
            self.assertFalse(h._call_drivers);self.assertEqual(c.snapshot()['budget']['reserved_liability'],2)


class OwnedDrivers(unittest.IsolatedAsyncioTestCase):
    async def test_shared_bounded_driver_registry_revoke_retire_identity_and_self_join(self):
        h=HumanSense(consent=True);gate=asyncio.Event();drivers=[]
        class Driver:
            def __init__(self,kind,ident):
                self.kind=kind;self.id=ident;self.buffer=bytearray(b'private');self.task=None;self.close_task=None
            def owns_task(self,task):return task is self.task or task is self.close_task
            def invalidate(self,reason):
                self.buffer[:]=b'\0'*len(self.buffer)
                if self.close_task is None:
                    async def close():await gate.wait()
                    self.close_task=asyncio.create_task(close())
                    h._track_call_driver_cleanup(self.kind,self.id,self,self.close_task)
        try:
            for kind in ('frame_sentinel','context_fill'):
                for i in range(4):
                    d=Driver(kind,str(i));h._register_call_driver(kind,d.id,d);drivers.append(d)
            with self.assertRaises(RuntimeError):h._register_call_driver('context_fill','overflow',Driver('context_fill','overflow'))
            h.consent=False
            self.assertTrue(all(not any(d.buffer) for d in drivers));self.assertEqual(len(h._call_drivers),8)
            h.consent=True
            with self.assertRaises(RuntimeError):h._register_call_driver('context_fill','overflow',Driver('context_fill','overflow'))
            gate.set();await h._join_call_drivers();await asyncio.sleep(0)
            self.assertFalse(h._call_drivers);self.assertFalse(h._call_driver_cleanup)
            d=Driver('context_fill','new');h._register_call_driver(d.kind,d.id,d)
            d.task=asyncio.current_task()
            with self.assertRaisesRegex(RuntimeError,'own host lifecycle'):await h.pause()
            self.assertFalse(any(d.buffer));d.task=None
            await h._join_call_drivers()
        finally:gate.set();await h.close()


if __name__=='__main__':unittest.main()
