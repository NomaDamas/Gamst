import asyncio
import unittest

from humansense import HumanSense
from humansense.cadence import ContextCadenceProfile
from humansense.mcp import MCPServer
from humansense.server import handle_connection
from test_perception_api import Socket


class Cadence(unittest.TestCase):
    def test_unknown_then_bounded_recommendation_never_changes_user_interval(self):
        p = ContextCadenceProfile(50)
        self.assertIsNone(p.snapshot()['recommended_interval_ms'])
        for i in range(100):
            p.record(200, 1000+i*250)
        s = p.snapshot()
        self.assertEqual((s['retained_samples'], s['completed']), (64, 100))
        self.assertEqual(s['recommended_interval_ms'], 200)
        self.assertEqual((s['requested_interval_ms'], s['applied_interval_ms']), (50, 50))
        self.assertEqual(s['observed_completion_intervals']['p95_ms'], 250)
        self.assertEqual(s['consumer_comprehension'], 'unknown')
        self.assertFalse(s['automatic_application'])
        for args in ((float('nan'), 1), (-1, 1), (1, 0)):
            with self.assertRaises(ValueError):
                p.record(*args)
        for value in (True, 0, 60001):
            with self.assertRaises(ValueError):
                ContextCadenceProfile(value)

    def test_excessive_work_is_explicitly_clamped(self):
        p = ContextCadenceProfile(500)
        for i in range(8):
            p.record(70000, i*71000)
        self.assertEqual(p.snapshot()['recommended_interval_ms'], 60000)
        self.assertTrue(p.snapshot()['recommendation_exceeds_supported_range'])


class ConnectionProfile(unittest.IsolatedAsyncioTestCase):
    async def receive_id(self, socket, ident):
        while True:
            message = await asyncio.wait_for(socket.outgoing.get(), 2)
            if message.get('id') == ident:
                return message

    async def test_profiles_are_passive_reset_and_preserve_invalid_subscription(self):
        async with HumanSense(consent=True) as human:
            socket = Socket()
            task = asyncio.create_task(handle_connection(MCPServer(human), socket))
            async def request(ident, method, params={}):
                await socket.incoming.put(dict(jsonrpc='2.0', id=ident, method=method, params=params))
                return await self.receive_id(socket, ident)
            try:
                initial = (await request(1, 'humansense/context_profile'))['result']
                self.assertFalse(initial['configured'])
                self.assertEqual(initial['probes_sent'], 0)
                await request(2, 'humansense/subscribe', {'types': ['context'], 'interval_ms': 50})
                for _ in range(8):
                    await asyncio.wait_for(socket.outgoing.get(), 2)
                active = (await request(3, 'humansense/context_profile'))['result']
                self.assertGreaterEqual(active['completed'], 8)
                self.assertEqual(active['applied_interval_ms'], 50)
                self.assertIsNotNone(active['recommended_interval_ms'])
                self.assertTrue(active['running'])
                invalid = await request(4, 'humansense/subscribe', {'types': ['context'], 'interval_ms': 0})
                self.assertIn('error', invalid)
                after = (await request(5, 'humansense/context_profile'))['result']
                self.assertGreaterEqual(after['completed'], active['completed'])
                await request(6, 'humansense/subscribe', {'types': []})
                stopped = (await request(7, 'humansense/context_profile'))['result']
                self.assertFalse(stopped['configured'])
                self.assertEqual(stopped['completed'], 0)
                self.assertIn('error', await request(8, 'humansense/context_profile', {'probe': True}))
            finally:
                await socket.incoming.put(None)
                await task
