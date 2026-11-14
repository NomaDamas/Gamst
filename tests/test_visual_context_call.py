import asyncio
import base64
import copy
from dataclasses import replace
import hashlib
import json
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2=np=None

from humansense import VisualContextCalls,SensorMetadata,smolvlm_frame_sha256
from humansense.plugins import InputFrame
from humansense.call_controller import _digest
from humansense.schema import timestamp
from humansense.mcp import MCPServer
import test_text_context_call as text_tests


@unittest.skipIf(np is None,'optional vision dependencies absent')
class Calls(unittest.IsolatedAsyncioTestCase):
    asyncSetUp=text_tests.Calls.asyncSetUp
    asyncTearDown=text_tests.Calls.asyncTearDown

    async def queued(self,route='small'):
        self.seq+=1;now=timestamp()
        sensor=SensorMetadata('cam'+str(self.seq),'video',self.seq,'pixels',stream_epoch='e')
        pixels=np.arange(32*48*3,dtype=np.uint8).reshape(32,48,3)
        frame=InputFrame(modality='video',data=pixels,sensor=sensor,timestamp_ms=now)
        r=await self.h.push_observation(self.spec.id,dict(objects=[],note='PRIVATE OBSERVATION'),sensor=sensor,timestamp_ms=now)
        source=self.c.open_source(r,now_ms=now)
        visual=dict(source_id=sensor.source_id,sensor_id=sensor.sensor_id,stream_epoch='e',capture_ms=now,sha256=smolvlm_frame_sha256(pixels))
        rules=dict(unknown_candidate=True,visual_reference=visual)
        if route=='strong':rules['conflict_rule']='test.conflict'
        if route=='text':rules=dict(query=True)
        decision=self.c.submit(source,r,sequence=self.seq,now_ms=now,**rules)
        return decision['request_id'],r,frame

    def adapter(self,callback,**kwargs):
        a=VisualContextCalls(self.h,'main',callback,provider='test.client',model_revision='v1',**kwargs)
        self.adapters.append(a);return a

    async def test_exact_png_and_prompt_for_small_and_strong(self):
        captured=[]
        async def complete(request):captured.append(copy.deepcopy(request));return dict(text='PRIVATE ANSWER',usage=None)
        a=self.adapter(complete)
        for route in ('small','strong'):
            rid,r,f=await self.queued(route);before=self.h.store.status()['records']
            result=await a.execute(rid,r,f,'What changed?',maximum_charge=10)
            self.assertTrue(result['accepted']);sent=captured[-1];im=sent['images'][0]
            png=base64.b64decode(im['data_base64'],validate=True)
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png,dtype=np.uint8),cv2.IMREAD_COLOR),f.data)
            self.assertEqual(hashlib.sha256(png).hexdigest(),im['sha256'])
            self.assertEqual(smolvlm_frame_sha256(f.data),im['source_frame_sha256'])
            self.assertEqual(_digest(sent),result['prompt_sha256'])
            self.assertEqual(json.loads(sent['messages'][1]['content'])['observation_record'],r)
            self.assertEqual(self.h.store.status()['records'],before)
            source=self.c.snapshot()['sources'][next(x for x in self.c.snapshot()['sources'] if self.c.snapshot()['sources'][x]['last_semantic']==f.timestamp_ms)]
            self.assertEqual(source['last_semantic'],f.timestamp_ms)
            self.assertEqual(result['capture_ms'],f.timestamp_ms);self.assertFalse(result['quality_validated'])
            self.c.settle(rid,components={'other':1},now_ms=timestamp())
        status=await MCPServer(self.h).call_tool('get_call_status',{})
        self.assertNotIn('PRIVATE',json.dumps(status));self.assertNotIn('data_base64',json.dumps(status))

    async def test_mismatch_and_budget_never_call_client(self):
        calls=[]
        async def complete(request):calls.append(request);return dict(text='ok',usage=None)
        a=self.adapter(complete);rid,r,f=await self.queued()
        bad=f.data.copy();bad[0,0]=0
        for changed in (replace(f,data=bad),replace(f,data=f.data.reshape(48,32,3)),
                        replace(f,timestamp_ms=f.timestamp_ms+1),replace(f,sensor=replace(f.sensor,sequence=99)),
                        replace(f,subject_id='other')):
            with self.assertRaises(ValueError):await a.execute(rid,r,changed,'q',maximum_charge=10)
        with self.assertRaises(ValueError):await a.execute(rid,dict(r,id='changed'),f,'q',maximum_charge=10)
        self.assertEqual((await a.execute(rid,r,f,'q',maximum_charge=None))['status'],'blocked')
        tr,rr,ff=await self.queued('text')
        with self.assertRaises(ValueError):await a.execute(tr,rr,ff,'q',maximum_charge=10)
        self.assertEqual(calls,[])

    async def test_copy_and_early_receipt_lifecycle_discard(self):
        for action in ('normal','pause','reset','delete','delete_then_bill','close'):
            started=asyncio.Event();release=asyncio.Event();captured=[]
            async def complete(request):
                captured.append(copy.deepcopy(request));started.set();await release.wait();return dict(text='LATE PRIVATE',usage=None)
            a=self.adapter(complete);rid,r,f=await self.queued();original=f.data.copy()
            task=asyncio.create_task(a.execute(rid,r,f,'q',maximum_charge=10));await started.wait()
            f.data[:]=0
            if action=='delete_then_bill':await self.h.delete_data()
            self.c.settle(rid,components={'other':1},now_ms=timestamp())
            if action=='pause':await self.h.pause()
            elif action=='reset':await self.h.reset_processing()
            elif action=='delete':await self.h.delete_data()
            elif action=='close':await self.h.close()
            release.set();result=await task
            png=base64.b64decode(captured[0]['images'][0]['data_base64'])
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png,dtype=np.uint8),cv2.IMREAD_COLOR),original)
            self.assertEqual(result['status'],'completed' if action=='normal' else 'discarded')
            if action!='normal':self.assertNotIn('text',result)
            self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],0)
            if action=='pause':await self.h.resume()
        self.assertEqual(self.c.snapshot()['budget']['spent_known'],6)

    async def test_timeout_and_repeated_cancel_join_cleanup(self):
        async def delayed(request):await asyncio.sleep(60)
        a=self.adapter(delayed,timeout=.02);rid,r,f=await self.queued()
        with self.assertRaises(asyncio.TimeoutError):await a.execute(rid,r,f,'q',maximum_charge=10)
        self.assertEqual(self.c.snapshot()['budget']['reserved_liability'],10)
        started=asyncio.Event();cleaning=asyncio.Event();release=asyncio.Event();done=asyncio.Event()
        async def complete(request):
            started.set()
            try:await asyncio.Event().wait()
            finally:cleaning.set();await release.wait();done.set()
        b=self.adapter(complete);rid,r,f=await self.queued()
        task=asyncio.create_task(b.execute(rid,r,f,'q',maximum_charge=10));await started.wait()
        task.cancel();await cleaning.wait();task.cancel();closing=asyncio.create_task(b.close())
        for _ in range(5):await asyncio.sleep(0)
        self.assertFalse(task.done());self.assertFalse(closing.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        await closing;self.assertTrue(done.is_set());self.assertIsNone(b._backend_task)

    async def test_encoder_does_not_block_loop_and_close_joins_it(self):
        import threading
        from unittest.mock import patch
        started=threading.Event();release=threading.Event();finished=threading.Event();calls=[]
        original=VisualContextCalls._encode
        def encode(request):
            started.set()
            try:
                if not release.wait(5):raise TimeoutError('test encoder release missing')
                return original(request)
            finally:finished.set()
        async def complete(request):calls.append(request);return dict(text='ok',usage=None)
        a=self.adapter(complete);rid,r,f=await self.queued()
        with patch.object(VisualContextCalls,'_encode',staticmethod(encode)):
            task=asyncio.create_task(a.execute(rid,r,f,'q',maximum_charge=10))
            async def running():
                while not started.is_set():await asyncio.sleep(.001)
            try:
                await asyncio.wait_for(running(),1)
                with self.assertRaisesRegex(RuntimeError,'busy'):await a.execute(rid,r,f,'q',maximum_charge=10)
                task.cancel();closing=asyncio.create_task(a.close())
                for _ in range(10):await asyncio.sleep(0)
                task.cancel()
                self.assertFalse(task.done());self.assertFalse(closing.done());self.assertFalse(finished.is_set())
            finally:release.set()
            with self.assertRaises(asyncio.CancelledError):await task
            await closing
        self.assertTrue(finished.is_set());self.assertEqual(calls,[])
        self.assertEqual(self.c.snapshot()['budget']['dispatched'],0)
