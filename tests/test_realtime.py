import asyncio
import time
import unittest
from humansense import HumanSense
from humansense.plugins import InputFrame
from humansense.realtime import QueuedPlugin, LocalScenePlugin


class RealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_throttle_keeps_signals_and_speech_transitions(self):
        async with HumanSense(consent=True, sensor_inference_interval_ms=5000) as human:
            await human.push_signal('speech.activity', True, 'audio')
            await human.push_signal('speech.activity', False, 'audio')
            for i in range(20):
                await human.push_signal('audio.rms', i / 100, 'audio')
            inferences = await human.get_history(type='human.inference')
            self.assertEqual(len(inferences), 1)
            events = await human.get_history(type='observation.event')
            self.assertEqual({e['event'] for e in events}, {'speech.started', 'speech.stopped'})
            await human.push_text('Please explain again')
            self.assertEqual(len(await human.get_history(type='human.inference')), 2)
            self.assertEqual((await human.get_context())['observations']['audio.rms']['value'], .19)

    async def test_video_drops_do_not_reset_scene_rate_limit(self):
        gate = asyncio.Event()
        class Video:
            name, modalities = 'video-fixture', ('video',)
            resets = 0
            async def process(self, frame):
                await gate.wait()
                return []
            def reset(self):
                self.resets += 1
        async with HumanSense(consent=True) as human:
            plugin = Video()
            worker = QueuedPlugin(human, plugin, capacity=1)
            human.register(worker)
            for _ in range(4):
                await human.push_video(None)
            self.assertEqual(plugin.resets, 0)
            self.assertEqual(worker.dropped, 3)
            gate.set()
            await worker.task

    async def test_context_contains_compact_faces_but_no_raw_media(self):
        async with HumanSense(consent=True) as human:
            await human.push_signal('scene.faces', [{'frame_face_index': 0, 'landmarks': [{'x': 1}] * 478,
                'blendshapes': {'mouthSmileLeft': .8, 'eyeBlinkLeft': .1}}], 'video')
            await human.push_signal('scene.image', 'secret-image', 'video')
            context = await human.get_context()
            self.assertNotIn('scene.image', context['observations'])
            face = context['observations']['scene.faces']
            self.assertNotIn('landmarks', face['value'][0])
            self.assertTrue(face['fresh'])
            self.assertTrue(face['evidence_id'])

    async def test_slow_asr_does_not_block_capture_and_pause_discards_results(self):
        entered = asyncio.Event()
        class Slow:
            name, modalities = 'slow', ('audio',)
            async def process(self, frame):
                entered.set()
                await asyncio.sleep(10)
                return [{'name': 'text.transcript', 'value': 'late'}]
        async with HumanSense(consent=True) as human:
            worker = QueuedPlugin(human, Slow(), capacity=2)
            human.register(worker)
            await asyncio.wait_for(human.push_audio([0] * 512), .1)
            await entered.wait()
            for _ in range(6):
                await human.push_audio([0] * 512)
            self.assertLessEqual(worker.queue.qsize(), 2)
            self.assertGreater(worker.dropped, 0)
            await asyncio.wait_for(human.pause(), .2)
            self.assertEqual(worker.queue.qsize(), 0)
            self.assertEqual(await human.get_history(), [])

    async def test_background_transcript_reaches_conversation(self):
        class ASR:
            name, modalities = 'asr-fixture', ('audio',)
            async def process(self, frame):
                return [{'name': 'text.transcript', 'value': 'Please explain again'}]
        async with HumanSense(consent=True) as human:
            worker = QueuedPlugin(human, ASR())
            human.register(worker)
            await human.push_audio([0] * 512)
            await worker.task
            context = await human.get_context()
            self.assertEqual(context['observations']['text.utterance']['value'], 'Please explain again')
            self.assertIn('interaction.clarification_requested', context['summary']['event_counts'])

    async def test_scene_endpoint_cannot_be_remote(self):
        with self.assertRaises(PermissionError):
            LocalScenePlugin('https://example.com/v1/chat/completions', 'test')

    async def test_stale_measurements_are_marked(self):
        async with HumanSense(consent=True) as human:
            await human.push_signal('audio.rms', .1, 'audio', timestamp_ms=int(time.time()*1000)-6000)
            context = await human.get_context()
            self.assertFalse(context['observations']['audio.rms']['fresh'])

    async def test_worker_error_reported_without_stopping_text(self):
        class Broken:
            name, modalities = 'broken', ('audio',)
            async def process(self, frame):
                raise RuntimeError('fixture failure')
        async with HumanSense(consent=True) as human:
            worker = QueuedPlugin(human, Broken())
            human.register(worker)
            await human.push_audio([0] * 512)
            await worker.task
            await human.push_text('still works')
            self.assertEqual(human.plugin_errors[-1]['error'], 'fixture failure')

if __name__ == '__main__':
    unittest.main()
