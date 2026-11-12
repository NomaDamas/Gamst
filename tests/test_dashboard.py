"""Exercise dispatch and HTTP boundaries without requiring network socket access."""
import asyncio
import io
import json
import unittest
from types import SimpleNamespace
from humansense import HumanSense
from humansense.dashboard import Dashboard, handler_for
from humansense.plugins import AudioLevelPlugin


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)
        self.human.register(AudioLevelPlugin())
        self.app = Dashboard(self.human, asyncio.get_running_loop())
        await self.human.pause()

    async def asyncTearDown(self):
        await self.human.close()

    async def test_consent_lifecycle_and_three_outputs(self):
        with self.assertRaises(ValueError):
            await self.app.dispatch({"op": "text", "text": "help me"})
        with self.assertRaises(ValueError):
            await self.app.dispatch({"op": "start", "consent": False})
        await self.app.dispatch({"op": "start", "consent": True})
        result = await self.app.dispatch({"op": "text", "text": "help me"})
        self.assertEqual(set(result["predictions"]), {"conversation", "behavior", "consistency"})
        await self.app.dispatch({"op": "audio", "samples": [.1] * 512, "sample_rate": 16000})
        state = await self.app.dispatch({"op": "state"})
        self.assertEqual(state["state"]["modalities"]["audio"], "available")
        self.assertEqual(state["state"]["modalities"]["video"], "unavailable")
        await self.app.dispatch({"op": "stop"})
        self.assertTrue(self.human.paused)
        await self.app.dispatch({"op": "delete"})
        self.assertEqual(await self.human.get_history(), [])

    async def test_media_size_boundaries(self):
        await self.app.dispatch({"op": "start", "consent": True})
        for body in ({"op": "audio", "samples": [0] * 96001},
                     {"op": "audio", "samples": [0], "sample_rate": 0},
                     {"op": "video", "jpeg": "a" * 1000001}):
            with self.assertRaises(ValueError):
                await self.app.dispatch(body)


class HTTPBoundaryTests(unittest.TestCase):
    def make_handler(self, host="localhost:8766", token="secret", path="/api"):
        app = SimpleNamespace(token="secret")
        handler = handler_for(app).__new__(handler_for(app))
        handler.server = SimpleNamespace(server_port=8766)
        handler.headers = {"Host": host, "X-HumanSense-Token": token}
        handler.path = path
        handler.rfile = io.BytesIO(b"{}")
        handler.response = None
        handler.respond = lambda status, payload, *args: setattr(handler, "response", (status, payload))
        return handler

    def test_foreign_host_and_missing_token_rejected(self):
        handler = self.make_handler(host="evil.example:8766")
        handler.do_POST()
        self.assertEqual(handler.response[0], 404)
        handler = self.make_handler(token="")
        handler.do_POST()
        self.assertEqual(handler.response[0], 403)

    def test_foreign_origin_rejected(self):
        handler = self.make_handler()
        handler.headers["Origin"] = "https://evil.example"
        handler.do_POST()
        self.assertEqual(handler.response[0], 403)

    def test_index_contains_session_token_but_no_remote_assets(self):
        handler = self.make_handler(path="/")
        handler.do_GET()
        self.assertEqual(handler.response[0], 200)
        self.assertIn("const token='secret'", handler.response[1])
        self.assertNotIn("__SESSION_TOKEN__", handler.response[1])
        self.assertNotIn("<script src=", handler.response[1])


class ModelCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_does_not_wait_for_long_model_call(self):
        from humansense.companion import Companion
        entered, release = asyncio.Event(), asyncio.Event()
        class Backend:
            model_id, model_version = 'fixture', '1'
            async def reply(self, records, context, inference):
                entered.set(); await release.wait(); return 'late response'
        async with HumanSense(consent=True) as human:
            app = Dashboard(human, asyncio.get_running_loop())
            app.companion = Companion(human, Backend())
            await app.dispatch({'op':'start','consent':True})
            task = asyncio.create_task(app.dispatch({'op':'chat','text':'help me'}))
            await entered.wait()
            await asyncio.wait_for(app.dispatch({'op':'stop'}), .2)
            self.assertTrue(human.paused)
            release.set()
            with self.assertRaises(RuntimeError): await task
            history = await human.get_history()
            self.assertFalse(any(r.get('observation',{}).get('value') == 'late response' for r in history))
