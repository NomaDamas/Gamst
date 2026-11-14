"""Actual worker rotation using the compiled bridge/test-double SLAM fixture."""
import asyncio
from dataclasses import replace
import shutil
import sys
import unittest
from unittest.mock import patch
import test_orbslam_context as fixture
from humansense.orbslam_process import ORBSLAMProcess
from humansense import HumanSense,SensorStreams
from humansense.mcp import MCPServer
from humansense.schema import timestamp


@unittest.skipIf(fixture.np is None or shutil.which('c++') is None,'requires numpy/OpenCV/compiler')
class Rotation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.children=[];self.spawn=asyncio.create_subprocess_exec
        async def spawn(*args,**kwargs):
            self.assertFalse(any(c.returncode is None for c in self.children),'overlapping native children')
            child=await self.spawn(*args,**kwargs);self.children.append(child);return child
        p=patch('asyncio.create_subprocess_exec',spawn);p.start();self.addCleanup(p.stop)
    async def asyncTearDown(self):
        leaked=[c for c in self.children if c.returncode is None]
        for c in leaked:c.kill();await c.wait()
        self.assertFalse(leaked,'owned native child leaked')

    async def test_budgets_reset_original_capture_and_mcp(self):
        config=dict(fixture.assets(self),max_frames=2)
        p=await ORBSLAMProcess.open(sys.executable,config,max_starts_per_minute=10)
        try:
            a=await p.process(fixture.frame(0));await p.reset()
            b=await p.process(fixture.frame(1));c=await p.process(fixture.frame(2))
            self.assertEqual(c['provenance']['rotation']['session_number'],2)
            self.assertNotEqual(a['provenance']['session_id'],c['provenance']['session_id'])
            self.assertEqual(b['provenance']['rotation']['session_request_attempt'],2)
            self.assertEqual(c['provenance']['capture_ms'],1200)
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);streams.register(p.spec,lambda:p)
                await streams.open('cam','video',[p.spec.id]);now=timestamp()-1000
                try:
                    for i in range(3,7):
                        f=fixture.frame(i,now+i*100);f=replace(f,sensor=replace(f.sensor,stream_epoch=None))
                        await streams.submit(f);await streams.drain()
                    reply=await MCPServer(human).call_tool('get_observations',{'capability_id':p.spec.id})
                    values=[r['observation']['value'] for r in reply['items']]
                    self.assertTrue(any(v['provenance']['rotation']['session_number']==4 for v in values))
                    self.assertEqual(p.diagnostics()['request_attempts'],7)
                    self.assertEqual(p.diagnostics()['completed_observations'],7)
                finally:await streams.clear()
        finally:await p.close()
        self.assertEqual(len(self.children),4)

    async def test_rate_exhaustion_and_failed_rotation_stop_without_retry(self):
        p=await ORBSLAMProcess.open(sys.executable,dict(fixture.assets(self),max_frames=1),max_starts_per_minute=1)
        try:
            await p.process(fixture.frame(0))
            with self.assertRaisesRegex(RuntimeError,'rate exhausted'):await p.process(fixture.frame(1))
            self.assertEqual(p.diagnostics()['request_attempts'],1)
            self.assertTrue(p.closed)
        finally:await p.close()
        p=await ORBSLAMProcess.open(sys.executable,dict(fixture.assets(self),max_frames=1))
        try:
            await p.process(fixture.frame(0))
            with patch('humansense.orbslam_process.ProcessProvider.open',side_effect=RuntimeError('startup failed')):
                with self.assertRaisesRegex(RuntimeError,'startup failed'):await p.process(fixture.frame(1))
            self.assertTrue(p.closed);self.assertEqual(p.diagnostics()['request_attempts'],1)
        finally:await p.close()

    async def test_blocked_inference_close_cancellation_and_busy_rejection(self):
        p=await ORBSLAMProcess.open(sys.executable,fixture.assets(self,slow=True))
        await p.process(fixture.frame(0))
        task=asyncio.create_task(p.process(fixture.frame(1,color=253)))
        await asyncio.sleep(.1)
        with self.assertRaisesRegex(RuntimeError,'busy'):await p.process(fixture.frame(2))
        close=asyncio.create_task(p.close());await asyncio.sleep(0);close.cancel()
        with self.assertRaises(asyncio.CancelledError):await close
        with self.assertRaises(asyncio.CancelledError):await task
        await p.close();self.assertTrue(p.closed)
        self.assertEqual(p.diagnostics()['request_attempts'],2)

    async def test_duplicate_capture_cannot_replay_after_rotation(self):
        p=await ORBSLAMProcess.open(sys.executable,dict(fixture.assets(self),max_frames=1))
        try:
            await p.process(fixture.frame(0))
            with self.assertRaisesRegex(ValueError,'capture order'):await p.process(fixture.frame(0))
            self.assertEqual(len(self.children),1);self.assertEqual(p.attempts,1)
        finally:await p.close()

    async def test_stalled_rotation_startup_timeout_and_close_reap(self):
        config=dict(fixture.assets(self),max_frames=1)
        for cancel in (False,True):
            p=await ORBSLAMProcess.open(sys.executable,config,startup_timeout=5)
            await p.process(fixture.frame(0))
            p.startup_timeout=30 if cancel else .15
            owned_spawn=asyncio.create_subprocess_exec
            async def stalled(*args,**kwargs):
                return await owned_spawn(sys.executable,'-u','-c','import time; time.sleep(60)',**kwargs)
            try:
                with patch('asyncio.create_subprocess_exec',stalled):
                    if not cancel:
                        with self.assertRaises(asyncio.TimeoutError):await p.process(fixture.frame(1))
                    else:
                        count=len(self.children);task=asyncio.create_task(p.process(fixture.frame(1)))
                        for _ in range(300):
                            if len(self.children)>count:break
                            await asyncio.sleep(.01)
                        self.assertGreater(len(self.children),count)
                        await p.close()
                        with self.assertRaises(asyncio.CancelledError):await task
                self.assertEqual(p.attempts,1)
            finally:await p.close()

    async def test_stalled_old_shutdown_reaped_and_rotation_stops(self):
        header=fixture.fixture.HEADER.replace('#include <stdexcept>','#include <stdexcept>\n#include <thread>\n#include <chrono>')
        header=header.replace('void Shutdown(){}','void Shutdown(){std::this_thread::sleep_for(std::chrono::seconds(60));}')
        with patch.object(fixture.fixture,'HEADER',header):config=dict(fixture.assets(self),max_frames=1)
        p=await ORBSLAMProcess.open(sys.executable,config)
        try:
            await p.process(fixture.frame(0))
            with self.assertRaises(asyncio.TimeoutError):await p.process(fixture.frame(1))
            self.assertTrue(p.closed);self.assertEqual(len(self.children),1)
            self.assertIsNotNone(self.children[0].returncode)
        finally:await p.close()

    async def test_pause_and_delete_during_rotation_suppress_late_context(self):
        config=dict(fixture.assets(self),max_frames=1)
        for method in ('pause','delete_data'):
            p=await ORBSLAMProcess.open(sys.executable,config)
            async with HumanSense(consent=True) as human:
                streams=SensorStreams(human);streams.register(p.spec,lambda:p)
                await streams.open('cam','video',[p.spec.id]);now=timestamp()-1000
                def frame(i):
                    f=fixture.frame(i,now+i*100)
                    return replace(f,sensor=replace(f.sensor,stream_epoch=None))
                await streams.submit(frame(0));await streams.drain()
                count=len(self.children);owned_spawn=asyncio.create_subprocess_exec
                async def stalled(*args,**kwargs):
                    return await owned_spawn(sys.executable,'-u','-c','import time; time.sleep(60)',**kwargs)
                try:
                    with patch('asyncio.create_subprocess_exec',stalled):
                        await streams.submit(frame(1))
                        for _ in range(300):
                            if len(self.children)>count:break
                            await asyncio.sleep(.01)
                        self.assertGreater(len(self.children),count)
                        await getattr(human,method)()
                    self.assertTrue(p.closed)
                    rows=human.store.latest_signals('local-user',0,timestamp())
                    self.assertFalse(any(r['timestamp_ms']==now+100 for r in rows))
                finally:await streams.clear();await p.close()

    async def test_channel_declaration_owner_change_inside_and_across_budget(self):
        config=fixture.assets(self)
        for budget in (1,3):
            p=await ORBSLAMProcess.open(sys.executable,dict(config,max_frames=budget))
            try:
                await p.process(fixture.frame(10,2000))
                f=fixture.frame(0,100);f=replace(f,sensor=replace(f.sensor,channels=('B','G','R')))
                value=await p.process(f)
                self.assertEqual(value['provenance']['capture_ms'],100)
                self.assertEqual(value['provenance']['rotation']['request_attempt'],2)
                self.assertEqual(value['provenance']['rotation']['session_number'],2 if budget==1 else 1)
                self.assertEqual(value['provenance']['reset_count'],0 if budget==1 else 1)
            finally:await p.close()
