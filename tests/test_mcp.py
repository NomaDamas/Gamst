"""Exercise wire protocol and shared runtime instead of mocking the adapter."""
import asyncio
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from humansense.engine import HumanSense
from humansense.mcp import MCPServer


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)
        await self.human.__aenter__()
        self.server = MCPServer(self.human)

    async def asyncTearDown(self):
        await self.server.close()
        await self.human.__aexit__(None, None, None)

    async def request(self, name, arguments):
        response = await self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        self.assertEqual(response["id"], 1)
        return response["result"]

    async def test_initialize_and_notifications(self):
        for version in ("2024-11-05", "2025-03-26", "2025-06-18"):
            result = await self.server.handle({"jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {"protocolVersion": version}})
            self.assertEqual(result["result"]["protocolVersion"], version)
        self.assertIsNone(await self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    async def test_validation_and_consent(self):
        cases = [("start_capture", {}), ("start_capture", {"consent": False}),
                 ("push_text", {"text": "hello", "timestamp_ms": True}),
                 ("push_text", {"text": "hello", "unknown": 3}),
                 ("inspect_interval", {"start_ms": 20, "end_ms": 10}),
                 ("get_events", {"limit": -1}), ("get_state", []),
                 ("resume_capture", {})]
        for name, args in cases:
            with self.subTest(name=name, arguments=args):
                self.assertTrue((await self.request(name, args))["isError"])
        self.assertIsNone(self.server.capture)

    async def test_shared_runtime_and_deletion(self):
        result = await self.request("push_text", {"text": "다시 설명해 주세요", "subject_id": "alice"})
        self.assertFalse(result["isError"], result)
        state = await self.request("get_state", {"subject_id": "alice"})
        self.assertFalse(state["isError"], state)
        sdk_state = await self.human.get_state(subject_id="alice")
        adapter_state = json.loads(state["content"][0]["text"])
        # Snapshot IDs/times are minted per request; measured contents must match.
        self.assertEqual(adapter_state["observation"], sdk_state["observation"])
        self.assertEqual(adapter_state["subject_id"], sdk_state["subject_id"])
        events = await self.request("get_events", {"subject_id": "alice"})
        self.assertFalse(events["isError"], events)
        self.assertGreater(len(json.loads(events["content"][0]["text"])), 0)
        await self.request("delete_data", {"subject_id": "alice"})
        self.assertEqual(await self.human.get_history(subject_id="alice"), [])

    async def test_invalid_rpc(self):
        for request in ([], {}, {"jsonrpc": "1.0", "method": "ping"},
                        {"jsonrpc": "2.0", "method": "ping", "id": None},
                        {"jsonrpc": "2.0", "method": "notifications/initialized", "id": 2}):
            result = await self.server.handle(request)
            self.assertEqual(result["error"]["code"], -32600)
        result = await self.server.handle({"jsonrpc": "2.0", "method": "missing", "id": 1})
        self.assertEqual(result["error"]["code"], -32601)

    async def test_request_methods_without_id_have_no_side_effects(self):
        await self.human.push_text("existing conversation", subject_id="alice")
        original = await self.human.get_history(subject_id="alice")
        for name, arguments in (
                ("push_text", {"text": "unwanted insertion", "subject_id": "alice"}),
                ("push_claim", {"key": "unwanted", "value": 1, "subject_id": "alice"}),
                ("delete_data", {"subject_id": "alice"}),
                ("start_capture", {"consent": True, "video": None, "audio": True})):
            result = await self.server.handle({"jsonrpc": "2.0", "method": "tools/call",
                                               "params": {"name": name, "arguments": arguments}})
            self.assertIsNone(result)
        self.assertEqual(await self.human.get_history(subject_id="alice"), original)
        self.assertIsNone(self.server.capture)

    async def test_capture_lifecycle_single_shared_session(self):
        created = []

        class Capture:
            def __init__(self, human, **kwargs):
                self.human, self.args = human, kwargs
                self.running = self.paused = False
                created.append(self)

            async def start(self):
                await asyncio.sleep(0)
                self.running = True

            async def stop(self):
                self.running = False

            async def pause(self):
                self.paused = True

            async def resume(self):
                self.paused = False

            def status(self):
                return {"running": self.running, "paused": self.paused}

        with patch("humansense.capture.CaptureSession", Capture):
            results = await asyncio.gather(
                self.request("start_capture", {"consent": True, "video": None, "audio": True}),
                self.request("start_capture", {"consent": True, "video": None, "audio": True}))
            self.assertEqual(sum(r["isError"] for r in results), 1)
            self.assertEqual(len(created), 1)
            self.assertIs(created[0].human, self.human)
            self.assertIsNone(created[0].args["video"])
            self.assertFalse((await self.request("pause_capture", {}))["isError"])
            self.assertTrue(created[0].paused)
            self.assertFalse((await self.request("resume_capture", {}))["isError"])
            self.assertFalse(created[0].paused)
            self.assertFalse((await self.request("delete_data", {}))["isError"])
            self.assertFalse(created[0].running)
            self.assertIsNone(self.server.capture)

    async def test_local_model_selection_rollback_reuse_and_subject_binding(self):
        from humansense.plugins import PerceptionPlugin
        created = []

        class Model(PerceptionPlugin):
            def __init__(self, *signature):
                self.signature, self.closed = signature, False
                self.resets = 0
                created.append(self)

            def close(self):
                self.closed = True

            def reset(self):
                self.resets += 1

        class Capture:
            def __init__(self, human, **args):
                self.args = args

            async def start(self):
                pass

            async def stop(self):
                pass

            def status(self):
                return {"running": True}

        args = {"consent": True, "video": None, "audio": True,
                "subject_id": "alice", "vad_model_path": "/tmp/vad.jit",
                "asr_model_path": "/tmp/asr"}
        with patch("humansense.capture.CaptureSession", Capture), \
                patch("humansense.plugins.AudioLevelPlugin", Model), \
                patch("humansense.plugins.SileroVADPlugin", Model), \
                patch("humansense.plugins.FasterWhisperPlugin", side_effect=RuntimeError("missing local ASR")):
            failed = await self.request("start_capture", args)
            self.assertTrue(failed["isError"])
            self.assertEqual(self.human.plugins, [])
            self.assertEqual(self.server._capture_plugins, {})
            self.assertTrue(all(model.closed for model in created))
        with patch("humansense.capture.CaptureSession", Capture), \
                patch("humansense.plugins.AudioLevelPlugin", Model), \
                patch("humansense.plugins.SileroVADPlugin", Model), \
                patch("humansense.plugins.FasterWhisperPlugin", Model):
            result = await self.request("start_capture", args)
            self.assertFalse(result["isError"], result)
            self.assertNotIn("vad_model_path", self.server.capture.args)
            models = list(self.human.plugins)
            await self.request("stop_capture", {})
            self.assertTrue(all(model.resets == 1 for model in models))
            result = await self.request("start_capture", args)
            self.assertFalse(result["isError"], result)
            self.assertEqual(self.human.plugins, models)
            await self.request("stop_capture", {})
            result = await self.request("start_capture", {**args, "subject_id": "bob"})
            self.assertTrue(result["isError"])
            result = await self.request("start_capture", {**args, "asr_model_path": "/tmp/other"})
            self.assertTrue(result["isError"])
            self.assertIsNone(self.server.capture)


class StdioTests(unittest.TestCase):
    def test_real_stdio_protocol(self):
        messages = ["{broken", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}),
                    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                    json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "push_text", "arguments": {"text": "Please explain again"}}}),
                    json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_events", "arguments": {}}})]
        proc = subprocess.run([sys.executable, "-m", "humansense.mcp"], input="\n".join(messages) + "\n", text=True,
                              capture_output=True, timeout=15, env=os.environ.copy())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        output = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(len(output), 5, output)
        self.assertEqual(output[0]["error"]["code"], -32700)
        self.assertEqual(output[1]["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("get_context", [t["name"] for t in output[2]["result"]["tools"]])
        self.assertFalse(output[3]["result"]["isError"], output[3])
        self.assertGreater(len(json.loads(output[4]["result"]["content"][0]["text"])), 0)


if __name__ == "__main__":
    unittest.main()
