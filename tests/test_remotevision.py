import asyncio
import io
import json
import os
import types
import unittest
from unittest.mock import patch

from humansense.plugins import InputFrame
from humansense.remotevision import VisionDescriptionPlugin, parse_description, vision_from_env


def description():
    return {"observations": [{"name": "person_count", "value": 1, "evidence_description": "One visible person"},
            {"name": "visible_action", "value": "Looking toward the camera", "evidence_description": "Head faces camera"}],
            "expressions": [{"name": "smile", "model_score": .6}]}


async def inline(function, *args, **kwargs):
    return function(*args, **kwargs)


class VisionDescriptionTests(unittest.TestCase):
    def test_model_score_not_probability(self):
        result = parse_description(description())
        self.assertFalse(result["calibrated"])
        self.assertIsNone(result["probability"])
        self.assertEqual(result["interpretation"], "unverified_model_hypotheses")

    def test_unsupported_and_malicious_fields_rejected(self):
        changes = [
            lambda p: p.update(identity="person-name"),
            lambda p: p["expressions"][0].update(probability=.8),
            lambda p: p["expressions"][0].update(model_score=float("nan")),
            lambda p: p["expressions"][0].update(model_score=True),
            lambda p: p["expressions"][0].update(name="hidden_intent"),
            lambda p: p["observations"][0].update(value=True),
            lambda p: p["observations"][1].update(value="This person is lying"),
            lambda p: p["observations"][1].update(name="diagnosis"),
            lambda p: p["observations"][1].update(evidence_description=""),
        ]
        for change in changes:
            invalid = description()
            change(invalid)
            with self.subTest(output=invalid), self.assertRaises(ValueError):
                parse_description(invalid)

    def test_remote_opt_in(self):
        with self.assertRaises(PermissionError):
            VisionDescriptionPlugin("https://provider.example/v1/chat/completions", "vision")
        with self.assertRaises(ValueError):
            VisionDescriptionPlugin("http://provider.example/v1/chat/completions", "vision", allow_remote=True)

    def test_env_selection_and_explicit_remote_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(vision_from_env())
            os.environ["HUMANSENSE_VISION_URL"] = "http://127.0.0.1/v1/chat/completions"
            with self.assertRaises(ValueError):
                vision_from_env()
            os.environ["HUMANSENSE_VISION_MODEL"] = "vision-model"
            plugin = vision_from_env()
            self.assertEqual(plugin.model_id, "vision-model")
            self.assertEqual(plugin.interval_ms, 1000)
            os.environ["HUMANSENSE_VISION_INTERVAL_MS"] = "5"
            with self.assertRaises(ValueError):
                vision_from_env()
            os.environ["HUMANSENSE_VISION_INTERVAL_MS"] = "1200"
            os.environ["HUMANSENSE_VISION_URL"] = "https://provider.example/v1/chat/completions"
            with self.assertRaises(PermissionError):
                vision_from_env()
            os.environ["HUMANSENSE_VISION_ALLOW_REMOTE"] = "true"
            self.assertEqual(vision_from_env().interval_ms, 1200)

    def test_actual_request_shape_via_fake_opener(self):
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(description())}}]}).encode())
        plugin = VisionDescriptionPlugin("http://127.0.0.1:8000/v1/chat/completions", "vision", opener=opener)
        plugin._encode_jpeg = lambda image: "aW1hZ2U="
        result = plugin._describe_sync(object())
        payload = json.loads(requests[0].data)
        self.assertEqual(payload["model"], "vision")
        image = payload["messages"][1]["content"][1]
        self.assertEqual(image["image_url"]["url"], "data:image/jpeg;base64,aW1hZ2U=")
        self.assertIn("Never identify people", payload["messages"][0]["content"])
        self.assertFalse(result["identity_assessed"])
        self.assertNotIn("aW1hZ2U", json.dumps(result))

    def test_resize_and_encode_without_real_opencv(self):
        sizes = []
        class Image:
            shape = (1080, 1920, 3)
        class Encoded:
            def tobytes(self):
                return b"jpeg"
        cv2 = types.SimpleNamespace(IMWRITE_JPEG_QUALITY=1,
            resize=lambda image, size: sizes.append(size) or image,
            imencode=lambda *args: (True, Encoded()))
        plugin = VisionDescriptionPlugin("http://localhost/v1/chat/completions", "vision", max_side=640)
        with patch.dict("sys.modules", {"cv2": cv2}):
            self.assertEqual(plugin._encode_jpeg(Image()), "anBlZw==")
        self.assertEqual(sizes, [(640, 360)])

    def test_invalid_completion(self):
        plugin = VisionDescriptionPlugin("http://localhost/v1/chat/completions", "vision", opener=lambda *a, **k: io.BytesIO(b"bad"))
        plugin._encode_jpeg = lambda image: "encoded"
        with self.assertRaises(ValueError):
            plugin._describe_sync(object())


class AsyncVisionDescriptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_subject_rate_limit_reset_and_quality(self):
        plugin = VisionDescriptionPlugin("http://localhost/v1/chat/completions", "vision", interval_ms=1000)
        calls = []
        plugin._describe_sync = lambda image: calls.append(image) or parse_description(description())
        with patch("humansense.remotevision.asyncio.to_thread", inline):
            output = await plugin.process(InputFrame("frame1", "video", 1000, "alice"))
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]["quality"]["status"], "insufficient")
            self.assertIn("model_interpretation", output[0]["quality"]["flags"])
            self.assertEqual(await plugin.process(InputFrame("frame2", "video", 1500, "alice")), [])
            self.assertEqual(len(await plugin.process(InputFrame("frame3", "video", 1500, "bob"))), 1)
            self.assertEqual(len(await plugin.process(InputFrame("frame4", "video", 2000, "alice"))), 1)
            plugin.reset()
            self.assertEqual(len(await plugin.process(InputFrame("frame5", "video", 2100, "alice"))), 1)
        self.assertEqual(calls, ["frame1", "frame3", "frame4", "frame5"])

    async def test_reset_discards_inflight_image_result(self):
        plugin = VisionDescriptionPlugin("http://localhost/v1/chat/completions", "vision")
        started, release = asyncio.Event(), asyncio.Event()
        async def pending(function, *args):
            started.set()
            await release.wait()
            return parse_description(description())
        with patch("humansense.remotevision.asyncio.to_thread", pending):
            task = asyncio.create_task(plugin.process(InputFrame("raw", "video", 1000)))
            await started.wait()
            plugin.reset()
            release.set()
            self.assertEqual(await task, [])
            self.assertEqual(plugin._last, {})

    async def test_cancelled_call_does_not_publish_or_throttle(self):
        plugin = VisionDescriptionPlugin("http://localhost/v1/chat/completions", "vision")
        started = asyncio.Event()
        async def pending(function, *args):
            started.set()
            await asyncio.Event().wait()
        with patch("humansense.remotevision.asyncio.to_thread", pending):
            task = asyncio.create_task(plugin.process(InputFrame("raw", "video", 1000)))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(plugin._last, {})


if __name__ == "__main__":
    unittest.main()
