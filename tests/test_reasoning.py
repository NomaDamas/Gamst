import asyncio
from copy import deepcopy
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from humansense.reasoning import (ChatCompletionsBackend, backend_from_env,
    parse_analysis, prepare_records, validate_analysis, _normalize)


def records():
    return [
        {"id": "a", "subject_id": "alice", "timestamp_ms": 10, "type": "observation.signal",
         "observation": {"name": "text.utterance", "value": "I live in Seoul", "role": "user", "modality": "text"}},
        {"id": "b", "subject_id": "alice", "timestamp_ms": 20, "type": "observation.signal",
         "observation": {"name": "text.utterance", "value": "I live in Busan. Please explain again", "role": "user", "modality": "text"}}]


def payload():
    return {"expressions": [{"name": "confusion", "model_score": .7, "evidence_ids": ["b"]}],
            "interaction": [{"name": "needs_clarification", "hypothesis": "User may want clarification", "evidence_ids": ["b"]}],
            "consistency": [{"hypothesis": "The statements may differ about current residence", "evidence": [
                {"record_id": "a", "excerpt": "I live in Seoul"}, {"record_id": "b", "excerpt": "I live in Busan"}]}]}


class ReasoningValidationTests(unittest.TestCase):
    def test_canonical_hypotheses_and_recommendation(self):
        prepared = prepare_records(records())
        result = _normalize(parse_analysis(payload(), prepared))
        self.assertEqual(validate_analysis(result, records()), result)
        hypothesis = result["predictions"]["conversation"]["hypotheses"][0]
        self.assertIsNone(hypothesis["probability"])
        self.assertFalse(hypothesis["verified"])
        self.assertFalse(result["predictions"]["consistency"]["deception_assessed"])
        self.assertEqual(result["recommended_action"]["type"], "ask_clarifying_question")
        self.assertFalse(result["recommended_action"]["automatic_execution"])

    def test_malicious_or_incomplete_provider_fields_rejected(self):
        modifications = [
            lambda p: p["interaction"][0].update(probability=.8),
            lambda p: p["interaction"][0].update(calibrated=True),
            lambda p: p["interaction"][0].update(confidence=.9),
            lambda p: p["interaction"][0].update(evidence_ids=[]),
            lambda p: p["interaction"][0].update(evidence_ids=["other-user"]),
            lambda p: p["expressions"][0].update(model_score=float("nan")),
            lambda p: p["expressions"][0].update(model_score=True),
            lambda p: p["expressions"][0].update(name="depression"),
            lambda p: p["consistency"][0]["evidence"][0].update(excerpt="invented words"),
            lambda p: p["consistency"][0]["evidence"].pop(),
            lambda p: p["consistency"][0].update(hypothesis="The person is lying"),
            lambda p: p.update(recommended_action={"type": "send_email"}),
        ]
        for change in modifications:
            invalid = payload()
            change(invalid)
            with self.subTest(output=invalid), self.assertRaises(ValueError):
                parse_analysis(invalid, prepare_records(records()))

    def test_runtime_boundary_rejects_tampered_custom_backend(self):
        result = _normalize(parse_analysis(payload(), prepare_records(records())))
        result["predictions"]["conversation"]["hypotheses"][0]["verified"] = True
        with self.assertRaises(ValueError):
            validate_analysis(result, records())

    def test_preparation_excludes_raw_media_and_mixed_subjects(self):
        supplied = records() + [{"id": "raw", "subject_id": "alice", "type": "observation.signal",
                               "observation": {"name": "camera.image", "value": b"private image bytes", "modality": "video"}}]
        prepared = prepare_records(supplied)
        self.assertEqual({item["id"] for item in prepared}, {"a", "b"})
        supplied[0]["subject_id"] = "bob"
        with self.assertRaises(ValueError):
            prepare_records(supplied)
        limited = prepare_records(records(), max_text_chars=5)
        self.assertEqual(sum(len(item.get("text", "")) for item in limited), 5)

    def test_remote_requires_opt_in_https_and_no_embedded_credentials(self):
        with self.assertRaises(PermissionError):
            ChatCompletionsBackend("https://provider.example/v1/chat/completions", "model")
        with self.assertRaises(ValueError):
            ChatCompletionsBackend("http://provider.example/v1/chat/completions", "model", allow_remote=True)
        with self.assertRaises(ValueError):
            ChatCompletionsBackend("http://user:secret@127.0.0.1/v1/chat/completions", "model")
        with self.assertRaises(ValueError):
            ChatCompletionsBackend("http://127.0.0.1/v1/chat/completions", "model", timeout=float("nan"))
        self.assertTrue(ChatCompletionsBackend("https://provider.example/v1/chat/completions", "model", allow_remote=True).remote)

    def test_fake_http_transport_and_no_secret_leak(self):
        calls = []
        def opener(request, timeout):
            calls.append((request, timeout))
            content = json.dumps({"choices": [{"message": {"content": json.dumps(payload())}}]}).encode()
            return io.BytesIO(content)
        with patch.dict(os.environ, {"TEST_REASONING_SECRET": "private-key"}):
            backend = ChatCompletionsBackend("http://127.0.0.1:8000/v1/chat/completions", "local-model",
                api_key_env="TEST_REASONING_SECRET", timeout=4, opener=opener)
            result = backend._analyze_sync(prepare_records(records()))
        request, timeout = calls[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer private-key")
        self.assertEqual(timeout, 4)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "local-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("private-key", json.dumps(result))
        self.assertEqual(len(json.loads(body["messages"][1]["content"])["records"]), 2)

    def test_bad_http_response_errors(self):
        for response in (b"not json", b'{"choices":[]}', b"x" * 1_048_577):
            backend = ChatCompletionsBackend("http://localhost/v1/chat/completions", "m", opener=lambda *a, **k: io.BytesIO(response))
            with self.assertRaises(ValueError):
                backend._analyze_sync(prepare_records(records()))
        def denied(*args, **kwargs):
            raise HTTPError("https://secret-url.example?secret=key", 401, "secret text", {}, None)
        backend = ChatCompletionsBackend("http://localhost/v1/chat/completions", "m", opener=denied)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            backend._analyze_sync(prepare_records(records()))
        self.assertNotIn("secret", str(caught.exception))

    def test_env_factory_and_remote_consent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(backend_from_env())
            os.environ["HUMANSENSE_REASONING_URL"] = "http://127.0.0.1:8000/v1/chat/completions"
            with self.assertRaises(ValueError):
                backend_from_env()
            os.environ["HUMANSENSE_REASONING_MODEL"] = "local-model"
            self.assertEqual(backend_from_env().model_id, "local-model")
            os.environ["HUMANSENSE_REASONING_URL"] = "https://provider.example/v1/chat/completions"
            with self.assertRaises(PermissionError):
                backend_from_env()
            os.environ["HUMANSENSE_REASONING_ALLOW_REMOTE"] = "true"
            self.assertTrue(backend_from_env().remote)


class AsyncReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_records_do_not_contact_provider(self):
        def no_network(*args, **kwargs):
            raise AssertionError("Should not contact provider with no observations")
        backend = ChatCompletionsBackend("http://127.0.0.1:8000/v1/chat/completions", "local", opener=no_network)
        output = await backend.analyze([])
        self.assertEqual(validate_analysis(output, []), output)
        self.assertIsNone(output["recommended_action"])

    async def test_async_analyze_and_runtime_integration(self):
        from humansense import HumanSense
        backend = ChatCompletionsBackend("http://127.0.0.1:8000/v1/chat/completions", "local")
        # Exercise async adapter and runtime without sockets or real model weights.
        def fake_provider(prepared):
            return parse_analysis({"expressions": [], "interaction": [{"name": "needs_clarification",
                "hypothesis": "The user may need another explanation", "evidence_ids": [prepared[0]["id"]]}], "consistency": []}, prepared)
        backend._analyze_sync = fake_provider
        async def inline_worker(function, *args, **kwargs):
            # CI sandbox restricts event-loop wakeup sockets. Test provider IO
            # separately above; retain actual threading in production code.
            return function(*args, **kwargs)
        with patch("humansense.reasoning.asyncio.to_thread", inline_worker):
            async with HumanSense(consent=True) as human:
                await human.push_text("Please explain again")
                human.set_analysis_backend(backend)
                output = await human.analyze()
                self.assertEqual(output["type"], "human.inference")
                self.assertEqual(output["recommended_action"]["type"], "ask_clarifying_question")
                self.assertIn("local", [model["id"] for model in output["models"]])


if __name__ == "__main__":
    unittest.main()
