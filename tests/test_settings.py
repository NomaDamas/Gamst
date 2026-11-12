import asyncio
import unittest
from humansense import HumanSense
from humansense.plugins import AudioLevelPlugin
from humansense.realtime import QueuedPlugin
from humansense.mcp import MCPServer
from humansense.dashboard import Dashboard


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_switches_stop_actual_inference_and_are_shared_with_mcp(self):
        async with HumanSense(consent=True) as human:
            human.register(AudioLevelPlugin())
            app = Dashboard(human, asyncio.get_running_loop())
            mcp = MCPServer(human)
            await human.pause()
            await app.dispatch({'op': 'configure_feature', 'name': 'audio_level', 'enabled': False})
            self.assertFalse((await mcp.call_tool('get_features'))[0]['enabled'])
            await human.resume()
            await human.push_audio([.2] * 512)
            self.assertNotIn('audio.rms', (await human.get_context())['observations'])
            with self.assertRaises(ValueError):
                await mcp.call_tool('configure_feature', {'name': 'audio_level', 'enabled': True})
            await human.pause()
            await mcp.call_tool('configure_feature', {'name': 'audio_level', 'enabled': True})
            await human.resume()
            await human.push_audio([.2] * 512)
            self.assertAlmostEqual((await human.get_context())['observations']['audio.rms']['value'], .2)

    async def test_disabled_queued_and_text_features(self):
        async with HumanSense(consent=True) as human:
            human.register(QueuedPlugin(human, AudioLevelPlugin()))
            mcp = MCPServer(human)
            await human.pause()
            for name in ('audio_level', 'text_companion'):
                await mcp.call_tool('configure_feature', {'name': name, 'enabled': False})
            await human.resume()
            await human.push_audio([.2] * 512)
            self.assertIsNone(human.plugins[0].task)
            with self.assertRaises(ValueError):
                await mcp.call_tool('push_text', {'text': 'disabled'})
            self.assertEqual((await human.get_context())['observations'], {})

    async def test_capabilities_and_unavailable_vlm(self):
        async with HumanSense(consent=True) as human:
            app = Dashboard(human, asyncio.get_running_loop())
            result = await app.dispatch({'op': 'capabilities'})
            self.assertIn('configure_feature', [t['name'] for t in result['tools']])
            with self.assertRaises(ValueError):
                await app.dispatch({'op': 'test_vlm'})
