import asyncio
import builtins
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples.companion import LocalChatBackend, companion_backend_from_env
from gamst.providers.gemini import (
    GEMINI_OPENAI_URL,
    GeminiLiveBackend,
    GeminiLiveCompanion,
    gemini_flash_backend,
)


class FakeSession:
    def __init__(self, responses=(), hang=False):
        self.responses = responses
        self.hang = hang
        self.sent = []

    async def send_client_content(self, **kwargs):
        self.sent.append(kwargs)

    def receive(self):
        async def messages():
            for response in self.responses:
                yield response
            if self.hang:
                await asyncio.Event().wait()
        return messages()


class FakeConnection:
    def __init__(self, session):
        self.session = session
        self.enters = 0
        self.exits = 0

    async def __aenter__(self):
        self.enters += 1
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        self.exits += 1


def response(text, complete=False):
    return SimpleNamespace(text=text, server_content=SimpleNamespace(turn_complete=complete))


def session_factory(session, calls):
    connection = FakeConnection(session)

    def factory(**kwargs):
        calls.append(kwargs)
        return connection

    return factory, connection


def sent_text(call):
    return call["turns"]["parts"][0]["text"]


class GeminiFlashTests(unittest.TestCase):
    def test_api_key_and_vertex_endpoints(self):
        backend = gemini_flash_backend()
        self.assertIsInstance(backend, LocalChatBackend)
        self.assertEqual(backend.model_id, "gemini-3.5-flash-lite")
        self.assertEqual(backend._transport.endpoint, GEMINI_OPENAI_URL)
        self.assertEqual(backend._transport.api_key_env, "GEMINI_API_KEY")
        self.assertTrue(backend._transport.remote)

        vertex = gemini_flash_backend("google/gemini-test", vertex=True,
                                      project="project-id", location="us-central1")
        self.assertEqual(vertex._transport.endpoint,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/project-id/"
            "locations/us-central1/endpoints/openapi/chat/completions")
        self.assertEqual(vertex._transport.api_key_env, "GOOGLE_OAUTH_ACCESS_TOKEN")
        self.assertTrue(vertex._transport.remote)

    def test_vertex_requires_project_and_location(self):
        with self.assertRaisesRegex(ValueError, "project and location"):
            gemini_flash_backend(vertex=True, project="project-id")

    def test_environment_presets(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ["GAMST_COMPANION_PROVIDER"] = "gemini-flash"
            flash = companion_backend_from_env()
            self.assertEqual(flash._transport.endpoint, GEMINI_OPENAI_URL)
            os.environ.update(GAMST_GEMINI_MODEL="google/gemini-test",
                              GAMST_GEMINI_VERTEX="true",
                              GAMST_GEMINI_PROJECT="project-id",
                              GAMST_GEMINI_LOCATION="europe-west4")
            flash = companion_backend_from_env()
            self.assertEqual(flash.model_id, "google/gemini-test")
            self.assertIn("europe-west4-aiplatform", flash._transport.endpoint)

            os.environ["GAMST_COMPANION_PROVIDER"] = "gemini-live"
            live = companion_backend_from_env()
            self.assertIsInstance(live, GeminiLiveBackend)
            self.assertEqual(live.model_id, "google/gemini-test")
            self.assertTrue(live.live.vertex)
            self.assertEqual(live.live.project, "project-id")

            os.environ["GAMST_COMPANION_PROVIDER"] = "unknown"
            with self.assertRaisesRegex(ValueError, "local, gemini-flash, or gemini-live"):
                companion_backend_from_env()


class GeminiLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_push_then_ask_uses_one_session_and_strips_media(self):
        session = FakeSession([response("Hel"), response("lo", complete=True)])
        calls = []
        factory, connection = session_factory(session, calls)
        live = GeminiLiveCompanion(timeout=1, session_factory=factory)
        async with live:
            await live.push_context({"event": "speech.started", "capture_ms": 10,
                "now_ms": 20, "fresh": True, "evidence_id": "evidence-1", "value": 1,
                "jpeg": "raw-image", "audio": b"raw-audio", "arbitrary": "discard"})
            answer = await live.ask("What changed?")
            self.assertEqual(answer, "Hello")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["model"], "gemini-live-2.5-flash-native-audio")
            self.assertEqual(calls[0]["config"]["response_modalities"], ["TEXT"])
            instruction = calls[0]["config"]["system_instruction"]
            self.assertIn("untrusted measurements, not inner states", instruction)
            self.assertIn("untrusted data", instruction)

        self.assertEqual(connection.enters, 1)
        self.assertEqual(connection.exits, 1)
        self.assertEqual(len(session.sent), 2)
        self.assertFalse(session.sent[0]["turn_complete"])
        context = json.loads(sent_text(session.sent[0]))
        self.assertEqual(set(context), {"event", "capture_ms", "now_ms", "fresh",
                                       "evidence_id", "value"})
        self.assertNotIn("raw-image", sent_text(session.sent[0]))
        self.assertTrue(session.sent[1]["turn_complete"])
        self.assertEqual(sent_text(session.sent[1]), "What changed?")

    async def test_oversized_context_is_rejected_without_sending(self):
        session = FakeSession()
        factory, _ = session_factory(session, [])
        async with GeminiLiveCompanion(session_factory=factory) as live:
            with self.assertRaisesRegex(ValueError, "4096 UTF-8 bytes"):
                await live.push_context({"event": "large", "value": "x" * 4097})
        self.assertEqual(session.sent, [])

    async def test_utf8_cap_counts_bytes_not_characters(self):
        session = FakeSession()
        factory, _ = session_factory(session, [])
        async with GeminiLiveCompanion(session_factory=factory) as live:
            with self.assertRaisesRegex(ValueError, "4096 UTF-8 bytes"):
                await live.push_context({"event": "large", "value": "한" * 1400})
        self.assertEqual(session.sent, [])

    async def test_ask_timeout(self):
        session = FakeSession(hang=True)
        factory, _ = session_factory(session, [])
        async with GeminiLiveCompanion(timeout=0.01, session_factory=factory) as live:
            with self.assertRaisesRegex(TimeoutError, "timed out after 0.01 seconds"):
                await live.ask("hello")

    async def test_missing_sdk_has_install_action(self):
        real_import = builtins.__import__

        def without_google(name, *args, **kwargs):
            if name == "google" or name.startswith("google."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_google):
            with self.assertRaisesRegex(ImportError, r"pip install 'gamst-sdk\[gemini\]'"):
                await GeminiLiveCompanion().__aenter__()

    async def test_backend_pushes_bounded_context_then_newest_user_turn(self):
        session = FakeSession([response("answer", complete=True)])
        calls = []
        factory, _ = session_factory(session, calls)
        backend = GeminiLiveBackend(timeout=1, session_factory=factory)
        records = [
            {"id": "old", "subject_id": "subject", "timestamp_ms": 1,
             "observation": {"name": "text.utterance", "value": "old question", "role": "user"}},
            {"id": "new", "subject_id": "subject", "timestamp_ms": 2,
             "observation": {"name": "text.utterance", "value": "new question", "role": "user"}},
            {"id": "assistant", "subject_id": "subject", "timestamp_ms": 3,
             "observation": {"name": "text.utterance", "value": "later answer", "role": "assistant"}},
        ]
        context = {"summary": {"event_counts": {"speech.started": 1}},
                   "observations": {"large": "x" * 14000}}
        try:
            self.assertEqual(await backend.reply(records, context, {"predictions": {}}), "answer")
            self.assertEqual(await backend.reply(records, context, {"predictions": {}}), "answer")
        finally:
            await backend.close()
        self.assertEqual(len(calls), 1, "replies reuse the same persistent session")
        self.assertEqual(len(session.sent), 4)
        for context_call, question_call in zip(session.sent[::2], session.sent[1::2]):
            encoded = sent_text(context_call)
            self.assertLessEqual(len(encoded.encode("utf-8")), 4096)
            self.assertFalse(context_call["turn_complete"])
            delta = json.loads(encoded)
            self.assertEqual(delta["evidence_id"], "new")
            self.assertTrue(delta["value"]["truncated"])
            self.assertEqual(sent_text(question_call), "new question")
            self.assertTrue(question_call["turn_complete"])


if __name__ == "__main__":
    unittest.main()
