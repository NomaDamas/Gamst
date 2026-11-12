import asyncio
import io
import json
import os
import unittest
from unittest.mock import patch

from humansense import HumanSense
from humansense.companion import Companion, LocalChatBackend, companion_backend_from_env


class CompanionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)

    async def asyncTearDown(self):
        await self.human.close()

    async def test_no_backend_does_not_ingest_or_fake_chat(self):
        with self.assertRaisesRegex(RuntimeError, "No companion backend"):
            await Companion(self.human).reply("hello")
        self.assertEqual(await self.human.get_history(), [])

    async def test_adapter_receives_actual_context_and_subject_separated_turns(self):
        calls = []

        class Backend:
            model_id, model_version = "local-chat", "test"

            async def reply(self, records, context, inference):
                calls.append((records, context, inference))
                return "어느 부분부터 다시 설명할까요?"

        await self.human.push_text("Bob's private information", subject_id="bob")
        # Scalar measurements cannot crowd conversation out of the named query.
        for index in range(45):
            await self.human.push_signal("audio.rms", .1, "audio", subject_id="alice", model_id="test", model_version="1")
        companion = Companion(self.human, Backend())
        output = await companion.reply("다시 설명해 주세요", subject_id="alice")
        self.assertEqual(output["assistant_text"], "어느 부분부터 다시 설명할까요?")
        self.assertEqual(output["model"]["id"], "local-chat")
        self.assertFalse(output["verified"])
        self.assertTrue(calls[0][0])
        self.assertTrue(all(record["subject_id"] == "alice" for record in calls[0][0]))
        self.assertNotIn("Bob", json.dumps(calls, ensure_ascii=False))
        self.assertIsNotNone(calls[0][2]["recommended_action"])
        self.assertIn("interaction.clarification_requested", calls[0][1]["summary"]["event_counts"])
        turns = self.human.store.query("alice", kind="observation.signal", name="text.utterance")
        self.assertEqual([turn["observation"]["role"] for turn in turns], ["user", "assistant"])
        await self.human.delete_data(subject_id="alice")
        self.assertEqual(self.human.store.query("alice"), [])
        self.assertNotIn("history", companion.__dict__)

    async def test_late_response_discarded_on_deletion_pause_or_close(self):
        for operation in ("delete", "pause", "close"):
            with self.subTest(operation=operation):
                async with HumanSense(consent=True) as human:
                    started, release = asyncio.Event(), asyncio.Event()

                    class Backend:
                        model_id, model_version = "local", "test"

                        async def reply(self, records, context, inference):
                            started.set()
                            await release.wait()
                            return "stale private reply"

                    task = asyncio.create_task(Companion(human, Backend()).reply("hello"))
                    await started.wait()
                    if operation == "delete":
                        await human.delete_data()
                    elif operation == "pause":
                        await human.pause()
                    else:
                        await human.close()
                    release.set()
                    with self.assertRaisesRegex(RuntimeError, "discarded"):
                        await task
                    if not human.closed:
                        turns = human.store.query(kind="observation.signal", name="text.utterance")
                        self.assertTrue(all(r["observation"]["role"] == "user" for r in turns))

    async def test_cancellation_and_pause_do_not_save_assistant(self):
        started = asyncio.Event()

        class Backend:
            model_id, model_version = "local", "test"

            async def reply(self, records, context, inference):
                started.set()
                await asyncio.Event().wait()

        companion = Companion(self.human, Backend())
        task = asyncio.create_task(companion.reply("hello"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.human.pause()
        with self.assertRaisesRegex(RuntimeError, "paused"):
            await companion.reply("no collection while paused")
        turns = self.human.store.query(kind="observation.signal", name="text.utterance")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["observation"]["role"], "user")

    async def test_local_http_prompt_and_bounds(self):
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": "Could I clarify the last step?"}}]}).encode())
        backend = LocalChatBackend("http://127.0.0.1:8000/v1/chat/completions", "local-chat", opener=opener)
        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)
        for index in range(43):
            await self.human.push_text("turn-" + str(index) + " " + "x" * 450, role="user")
        await self.human.push_signal("scene.image", "private image not a turn", "video", model_id="test", model_version="1")
        with patch("humansense.companion.asyncio.to_thread", inline):
            output = await Companion(self.human, backend).reply("Please explain again")
        self.assertEqual(output["assistant_text"], "Could I clarify the last step?")
        body = json.loads(requests[0].data)
        messages = body["messages"]
        self.assertLessEqual(len(messages), 42)
        self.assertIn("UNVERIFIED", messages[0]["content"])
        self.assertIn("Never diagnose", messages[0]["content"])
        self.assertIn("recommended_strategy", messages[1]["content"])
        self.assertNotIn("private image", json.dumps(body))
        self.assertLessEqual(sum(len(m["content"]) for m in messages[2:]), 16000)
        self.assertEqual(messages[-1]["content"], "Please explain again")
        self.assertEqual(body["max_tokens"], 700)


class CompanionConfigurationTests(unittest.TestCase):
    def test_remote_requires_explicit_permission(self):
        with self.assertRaises(PermissionError):
            LocalChatBackend("https://provider.example/v1/chat/completions", "chat")
        with self.assertRaises(ValueError):
            LocalChatBackend("http://provider.example/v1/chat/completions", "chat", allow_remote=True)

    def test_env_factory_and_reasoning_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(companion_backend_from_env())
            os.environ.update(HUMANSENSE_REASONING_URL="http://127.0.0.1/v1/chat/completions", HUMANSENSE_REASONING_MODEL="reasoning")
            self.assertEqual(companion_backend_from_env().model_id, "reasoning")
            os.environ.update(HUMANSENSE_COMPANION_URL="https://provider.example/v1/chat/completions", HUMANSENSE_COMPANION_MODEL="chat")
            with self.assertRaises(PermissionError):
                companion_backend_from_env()
            os.environ["HUMANSENSE_COMPANION_ALLOW_REMOTE"] = "true"
            self.assertEqual(companion_backend_from_env().model_id, "chat")

    def test_malformed_response_rejected(self):
        for raw in (b"oops", b'{"choices":[]}', b'{"choices":[{"message":{"content":""}}]}'):
            backend = LocalChatBackend("http://localhost/v1/chat/completions", "m", opener=lambda *a, **k: io.BytesIO(raw))
            with self.assertRaises(ValueError):
                backend._reply_sync([{"role": "user", "text": "hi"}], {})


if __name__ == "__main__":
    unittest.main()
