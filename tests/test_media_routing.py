import asyncio
import json
import time
import unittest
from humansense import HumanSense
from humansense.media import media_router, decode_media
from humansense.plugins import AudioLevelPlugin
from humansense.server import handle_connection
from humansense.mcp import MCPServer
from test_server import Socket


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_runs_while_video_is_slow_and_video_is_bounded(self):
        gate,entered=asyncio.Event(),asyncio.Event()
        class Vision:
            name,modalities='slow-vision',('video',)
            async def process(self,frame):
                entered.set();await gate.wait();return []
        async with HumanSense(consent=True) as h:
            h.register(Vision()).register(AudioLevelPlugin())
            r=media_router(h)
            await r.submit('video',None,source='test')
            await entered.wait()
            for i in range(20): await r.submit('video',None,source='test')
            await r.submit('audio',[.1]*100,source='test',sample_rate=16000)
            await asyncio.wait_for(r.queues['audio'].join(),.2)
            self.assertEqual(r.status()['video']['dropped'],19)
            self.assertAlmostEqual((await h.get_context())['observations']['audio.rms']['value'],.1)
            await asyncio.wait_for(h.pause(),.2)
            self.assertEqual(r.status()['video']['queued'],0)

    async def test_sequence_subject_validation_and_audio_gap_reset(self):
        class Audio:
            name,modalities='audio-fixture',('audio',)
            resets=0
            async def process(self,frame): return []
            def reset(self): self.resets+=1
        async with HumanSense(consent=True) as h:
            p=Audio();h.register(p);r=media_router(h)
            async def submit(seq,**kw):
                return await r.submit('audio',[0],source='a',sample_rate=16000,sequence=seq,**kw)
            await submit(0);await r.drain()
            await submit(4);await r.drain()
            self.assertEqual(p.resets,1)
            with self.assertRaises(ValueError): await submit(4)
            with self.assertRaises(ValueError): await submit(5,subject_id='other')
            await r.clear('a')
            await r.submit('audio',[0],source='b',sample_rate=16000,sequence=0)
            await h.delete_data()
            self.assertEqual(r.status()['audio']['queued'],0)
            self.assertEqual(await h.get_history(),[])

    async def test_websocket_consent_media_ack_metrics_and_cleanup(self):
        async with HumanSense(consent=True) as h:
            h.register(AudioLevelPlugin());server=MCPServer(h);socket=Socket()
            task=asyncio.create_task(handle_connection(server,socket))
            async def rpc(n,method,params):
                await socket.incoming.put(dict(jsonrpc='2.0',id=n,method=method,params=params))
                while True:
                    response=await asyncio.wait_for(socket.outgoing.get(),1)
                    if response.get('id')==n:return response
            data=dict(modality='audio',samples=[.2]*100,sample_rate=16000,sequence=0)
            self.assertIn('error',await rpc(1,'humansense/media',data))
            self.assertTrue((await rpc(2,'humansense/stream/start',{'consent':True}))['result']['streaming'])
            self.assertTrue((await rpc(3,'humansense/media',data))['result']['accepted'])
            await media_router(h).drain()
            self.assertIn('audio.rms',(await h.get_context())['observations'])
            self.assertIn('error',await rpc(4,'humansense/media',data))
            status=(await rpc(5,'humansense/stream/status',{}))['result']['routing']
            self.assertEqual(status['audio']['processed'],1)
            await socket.incoming.put(None);await task
            self.assertIsNone(media_router(h).status()['audio']['source'])

    async def test_existing_bridge_uses_shared_router_after_conversion(self):
        from humansense.integrations import PipecatBridge
        async with HumanSense(consent=True) as h:
            h.register(AudioLevelPlugin())
            bridge=PipecatBridge(h,subject_id='participant',routed=True)
            ack=await bridge.on_audio([.1,.3]*100,sample_rate=16000,channels=2)
            self.assertTrue(ack['accepted'])
            await media_router(h).drain()
            context=await h.get_context(subject_id='participant')
            self.assertAlmostEqual(context['observations']['audio.rms']['value'],.2)
            await bridge.close()
            self.assertIsNone(media_router(h).status()['audio']['source'])

    def test_wire_rejects_nonfinite_samples_and_invalid_rate(self):
        for samples,rate in [([float('nan')],16000),([2],16000),([0],True),([0],16000.0)]:
            with self.assertRaises(ValueError):decode_media(dict(modality='audio',samples=samples,sample_rate=rate))
