import json
import unittest
from unittest.mock import patch
from humansense import HumanSense
from humansense.plugins import InputFrame
from humansense.realtime import LocalScenePlugin, QueuedPlugin, configure_scene
from humansense.mcp import MCPServer


class TemporalSceneTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_shared_by_mcp_and_dashboard_and_validated(self):
        from humansense.dashboard import Dashboard
        import asyncio
        async with HumanSense(consent=True) as human:
            plugin = LocalScenePlugin('http://127.0.0.1:18767', 'fixture', human=human)
            human.register(QueuedPlugin(human, plugin, 1))
            mcp = MCPServer(human)
            await mcp.call_tool('configure_vlm', {'enabled': False, 'window_seconds': 10, 'sample_fps': 4})
            state = await Dashboard(human, asyncio.get_running_loop()).dispatch({'op': 'state'})
            self.assertFalse(state['vlm']['enabled'])
            self.assertEqual(state['vlm']['effective_sample_fps'], .8)
            self.assertIsNone(await plugin.prepare(InputFrame(None, 'video', 100)))
            for settings in ({'interval': 0}, {'sample_fps': float('nan')}, {'enabled': 1}, {'bogus': 1}):
                with self.assertRaises(ValueError):
                    await configure_scene(human, settings)
            self.assertEqual(plugin.interval, 3)

    async def test_disable_discards_inflight_output(self):
        import asyncio
        entered = asyncio.Event()
        gate = asyncio.Event()
        class SlowScene(LocalScenePlugin):
            async def prepare(self, frame):
                return frame
            async def process(self, frame):
                entered.set()
                await gate.wait()
                return [{'name': 'scene.description', 'value': 'late result'}]
        async with HumanSense(consent=True) as human:
            plugin = SlowScene('http://127.0.0.1:18767', 'fixture')
            worker = QueuedPlugin(human, plugin, 1)
            human.register(worker)
            await human.push_video(None)
            await entered.wait()
            await human.push_video(None)
            await configure_scene(human, {'enabled': False})
            gate.set()
            self.assertIsNone(worker.task)
            self.assertEqual(worker.queue.qsize(), 0)
            self.assertNotIn('scene.description', (await human.get_context())['observations'])

    async def test_window_sampling_alignment_and_no_media_in_context(self):
        # cv2 is optional for base installs; mock the encoder, test window semantics.
        import sys
        from types import SimpleNamespace
        encoder = SimpleNamespace(resize=lambda data, size: data, imencode=lambda *a: (True, b'fixture-jpeg'), IMWRITE_JPEG_QUALITY=1)
        async with HumanSense(consent=True) as human:
            plugin = LocalScenePlugin('http://127.0.0.1:18767', 'fixture', human=human)
            plugin.interval = 3
            import time
            base = int(time.time()*1000)-15000
            image = SimpleNamespace(shape=(360, 640))
            with patch.dict(sys.modules, {'cv2': encoder}):
                first = await plugin.prepare(InputFrame(image, 'video', base+10000))
                self.assertEqual(len(first.data['frames']), 1)
                self.assertIsNone(await plugin.prepare(InputFrame(image, 'video', base+10100)))
                job = None
                for t in range(base+10600, base+14201, 600):
                    await human.push_signal('audio.rms', .2, 'audio', timestamp_ms=t)
                    result = await plugin.prepare(InputFrame(image, 'video', t))
                    job = result or job
                self.assertLessEqual(len(plugin.frames), 8)
                self.assertGreater(len(job.data['frames']), 1)
                sample = job.data['frames'][-1]
                self.assertEqual(sample['measurements']['audio.rms']['timestamp_ms'], sample['timestamp_ms'])
                self.assertEqual(sample['measurements']['audio.rms']['value'], .2)
                body = {}
                class Response:
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                    def read(self, n): return b'{"choices":[{"message":{"content":"A person moves."}}]}'
                def request(req, **kw):
                    body.update(json.loads(req.data))
                    return Response()
                plugin.transport._opener = request
                output = plugin._describe(job)
                self.assertEqual(sum(x['type']=='image_url' for x in body['messages'][0]['content']), len(job.data['frames']))
                # Use current timestamps for the context retention window.
                import time
                for signal in output:
                    signal['timestamp_ms'] = int(time.time()*1000)
                await human.ingest_plugin_output(output, job, plugin)
                context = await human.get_context()
                window = context['observations']['scene.window']['value']
                self.assertEqual(window['frame_count'], len(job.data['frames']))
                self.assertEqual(len(window['measurements_series']), window['frame_count'])
                self.assertIn('audio.rms', window['measurements_series'][-1]['measurements'])
                self.assertNotIn('fixture-jpeg', json.dumps(context))
                self.assertNotIn('jpeg', json.dumps(window))
                self.assertGreater(window['actual_seconds'], 0)
                await plugin.prepare(InputFrame(image, 'video', 20000, 'another-subject'))
                self.assertEqual(len(plugin.frames), 1)
                plugin.reset()
                self.assertEqual(len(plugin.frames), 0)
