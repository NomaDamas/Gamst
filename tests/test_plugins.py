import asyncio
import math
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humansense.plugins import AudioLevelPlugin, InputFrame, MediaPipeFacePlugin, FasterWhisperPlugin, SileroVADPlugin
from humansense.capture import CaptureSession


class AudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_rms_matches_normalized_pcm_without_claiming_speech(self):
        plugin = AudioLevelPlugin()
        frame = InputFrame([.5, -.5], "audio", 123, "speaker")
        signals = await plugin.process(frame)
        self.assertEqual(signals[0]["value"], .5)
        self.assertAlmostEqual(signals[1]["value"], 20 * math.log10(.5))
        self.assertEqual(signals[0]["subject_id"], "speaker")
        self.assertEqual(signals[0]["timestamp_ms"], 123)
        self.assertNotIn("speech", str(signals))

    async def test_pcm16_and_silence(self):
        signals = await AudioLevelPlugin().process(InputFrame(struct.pack("<hh", 16384, -16384), "audio", 0))
        self.assertEqual(signals[0]["value"], .5)
        silence = await AudioLevelPlugin().process(InputFrame([0, 0], "audio", 1))
        self.assertFalse(silence[2]["value"])

    async def test_rejects_non_finite_or_unnormalized(self):
        for data in ([float("nan")], [2], b"x"):
            with self.assertRaises(ValueError):
                await AudioLevelPlugin().process(InputFrame(data, "audio", 1))

    def test_optional_models_require_local_artifacts(self):
        with self.assertRaises(ValueError):
            MediaPipeFacePlugin("/does-not-exist.task")
        with self.assertRaises(ValueError):
            FasterWhisperPlugin("/does-not-exist-model")


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Deterministic fake devices need no worker thread. Real device I/O
        # remains off the event loop in production CaptureSession.
        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)
        self.worker = patch("humansense.capture.asyncio.to_thread", inline)
        self.worker.start()

    async def asyncTearDown(self):
        self.worker.stop()

    async def test_consent_required_before_device_access(self):
        with self.assertRaises(PermissionError):
            await CaptureSession(object(), video=0).start()

    async def test_empty_capture_rejected(self):
        with self.assertRaises(ValueError):
            await CaptureSession(object(), consent=True).start()

    async def test_queue_is_bounded_and_pause_discards_audio(self):
        session = CaptureSession(object(), consent=True)
        session._running = True
        for timestamp in range(20):
            session._enqueue_audio([0], timestamp, None)
        self.assertEqual(session._audio_queue.qsize(), 8)
        self.assertEqual(session.status()["dropped_audio_windows"], 12)
        await session.pause()
        session._enqueue_audio([0], 21, None)
        self.assertEqual(session._audio_queue.qsize(), 0)
        await session.stop()
        self.assertFalse(session.status()["running"])

    async def test_camera_is_released_on_failure(self):
        class Camera:
            released = False
            def isOpened(self): return False
            def release(self): self.released = True
        camera = Camera()
        class CV:
            @staticmethod
            def VideoCapture(device): return camera
        with patch.dict("sys.modules", {"cv2": CV}):
            session = CaptureSession(object(), video=0, consent=True)
            with self.assertRaisesRegex(RuntimeError, "Cannot open camera"):
                await session.start()
        self.assertTrue(camera.released)
        self.assertFalse(session.status()["running"])

    async def test_stop_releases_camera_and_cancels_read_loop(self):
        class Runtime:
            async def push_video(self, *a, **kw): pass
        class Camera:
            released = False
            def isOpened(self): return True
            def read(self): return True, object()
            def release(self): self.released = True
        camera = Camera()
        class CV:
            @staticmethod
            def VideoCapture(device): return camera
        with patch.dict("sys.modules", {"cv2": CV}):
            session = CaptureSession(Runtime(), video=0, consent=True)
            await session.start()
            await asyncio.sleep(.02)
            await session.stop()
        self.assertTrue(camera.released)
        self.assertFalse(session.status()["running"])

    async def test_background_camera_failure_is_visible_and_releases_device(self):
        class Camera:
            released = False
            def isOpened(self): return True
            def read(self): return False, None
            def release(self): self.released = True
        camera = Camera()
        class CV:
            @staticmethod
            def VideoCapture(device): return camera
        with patch.dict("sys.modules", {"cv2": CV}):
            session = CaptureSession(object(), video=0, consent=True)
            await session.start()
            await asyncio.sleep(.02)
            self.assertIn("Camera frame read failed", " ".join(session.status()["errors"]))
            self.assertFalse(session.status()["running"])
            self.assertTrue(camera.released)
            with self.assertRaises(RuntimeError):
                await session.wait()

    async def test_pause_and_stop_reset_processing_without_pausing_text_runtime(self):
        class Runtime:
            resets = 0
            async def reset_processing(self): self.resets += 1
        runtime = Runtime()
        session = CaptureSession(runtime, consent=True)
        session._running = True
        await session.pause()
        self.assertEqual(runtime.resets, 1)
        await session.resume()
        # A callback produced before resume must never enter the new segment.
        session._enqueue_audio([0], session._accept_after_ms - 1, None)
        self.assertEqual(session._audio_queue.qsize(), 0)
        await session.stop()
        self.assertEqual(runtime.resets, 2)

    async def test_concurrent_stop_releases_device_once(self):
        class Device:
            releases = 0
            def release(self): self.releases += 1
        device = Device()
        session = CaptureSession(object(), consent=True)
        session._camera = device
        session._running = True
        await asyncio.gather(session.stop(), session.stop())
        self.assertEqual(device.releases, 1)

    async def test_bad_mic_close_still_releases_camera(self):
        class Mic:
            closed = False
            def stop(self): raise RuntimeError("bad stop")
            def close(self): self.closed = True
        class Camera:
            released = False
            def release(self): self.released = True
        mic, camera = Mic(), Camera()
        session = CaptureSession(object(), consent=True)
        session._mic, session._camera = mic, camera
        with self.assertRaisesRegex(RuntimeError, "bad stop"):
            await session.stop()
        self.assertTrue(mic.closed)
        self.assertTrue(camera.released)

    async def test_audio_callback_backlog_is_bounded_before_copy(self):
        callbacks = []
        class Loop:
            def call_soon_threadsafe(self, fn, *args): callbacks.append((fn, args))
        class Data:
            copies = 0
            def copy(self): self.copies += 1; return self
            def reshape(self, *args): return self
            def tolist(self): return [0.]
        session = CaptureSession(object(), consent=True)
        session._running = True
        data = Data()
        for _ in range(20):
            session._post_audio(Loop(), data, None)
        self.assertEqual(data.copies, 8)
        self.assertEqual(len(callbacks), 8)
        self.assertEqual(session.status()["dropped_audio_windows"], 12)
        for function, args in callbacks:
            function(*args)
        self.assertEqual(session._audio_queue.qsize(), 8)
        session._post_audio(Loop(), data, None)
        self.assertEqual(data.copies, 9)


class WhisperTests(unittest.IsolatedAsyncioTestCase):
    def make_plugin(self, directory):
        class Model:
            calls = 0
            def __init__(self, *args, **kwargs): pass
            def transcribe(self, chunk, **kwargs):
                self.calls += 1
                return [SimpleNamespace(text=f"utterance {self.calls}")], None
        fake_whisper = SimpleNamespace(WhisperModel=Model)
        with patch.dict("sys.modules", {"faster_whisper": fake_whisper}):
            return FasterWhisperPlugin(directory, window_seconds=.032)

    async def test_all_complete_windows_are_processed_without_silent_drop(self):
        async def inline(function, *args, **kwargs): return function(*args, **kwargs)
        fake_numpy = SimpleNamespace(float32="float32", asarray=lambda values, **kwargs: values)
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            with patch.dict("sys.modules", {"numpy": fake_numpy}), patch("humansense.plugins.asyncio.to_thread", inline):
                output = await plugin.process(InputFrame([0.] * 1024, "audio", 1000, "alice", 16000))
                transcripts = [p for p in output if p["name"] == "text.transcript"]
                self.assertEqual([p["value"] for p in transcripts], ["utterance 1", "utterance 2"])
                rates = [p["value"] for p in output if p["name"] == "speech.transcribed_words_per_minute"]
                self.assertEqual(rates, [2*60/.032]*2)
                self.assertEqual([p["timestamp_ms"] for p in transcripts], [968, 1000])
                self.assertEqual(plugin.buffers["alice"], [])
                self.assertEqual(await plugin.process(InputFrame([0.] * 256, "audio", 1100, "bob", 16000)), [])
                self.assertEqual(await plugin.process(InputFrame([0.] * 256, "audio", 1100, "alice", 16000)), [])
                self.assertEqual(len(plugin.buffers["alice"]), 256)
                self.assertEqual(len(plugin.buffers["bob"]), 256)

    async def test_reset_discards_inflight_transcription_and_buffers(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(function, *args, **kwargs):
            entered.set()
            await release.wait()
            return function(*args, **kwargs)
        fake_numpy = SimpleNamespace(float32="float32", asarray=lambda values, **kwargs: values)
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            with patch.dict("sys.modules", {"numpy": fake_numpy}), patch("humansense.plugins.asyncio.to_thread", delayed):
                task = asyncio.create_task(plugin.process(InputFrame([0.] * 512, "audio", 1000, "alice", 16000)))
                await entered.wait()
                plugin.reset()
                release.set()
                self.assertEqual(await task, [])
                self.assertEqual(plugin.model.calls, 0)
                self.assertEqual(plugin.buffers, {})

    async def test_transcription_integrates_with_runtime_and_preserves_provenance(self):
        from humansense import HumanSense
        async def inline(function, *args, **kwargs): return function(*args, **kwargs)
        fake_numpy = SimpleNamespace(float32="float32", asarray=lambda values, **kwargs: values)
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            with patch.dict("sys.modules", {"numpy": fake_numpy}), patch("humansense.plugins.asyncio.to_thread", inline):
                async with HumanSense(consent=True) as runtime:
                    runtime.register(plugin)
                    await runtime.push_audio([0.] * 512, subject_id="alice", sample_rate=16000)
                    records = await runtime.get_history(subject_id="alice")
                    transcripts = [r for r in records if r.get("observation", {}).get("name") == "text.transcript"]
                    self.assertEqual(len(transcripts), 1)
                    self.assertEqual(transcripts[0]["models"][0]["id"], "faster_whisper")
                    self.assertIn("sha256:", transcripts[0]["models"][0]["version"])

    async def test_flush_transcribes_residual_once_and_uses_actual_duration(self):
        async def inline(function, *args, **kwargs): return function(*args, **kwargs)
        fake_numpy = SimpleNamespace(float32="float32", asarray=lambda values, **kwargs: values)
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            plugin.window_seconds = 3
            with patch.dict("sys.modules", {"numpy": fake_numpy}), patch("humansense.plugins.asyncio.to_thread", inline):
                self.assertEqual(await plugin.process(InputFrame([0.] * 8000, "audio", 2000, "alice", 16000)), [])
                result = await plugin.flush("alice")
                values = {signal["name"]: signal["value"] for signal in result}
                self.assertEqual(values["text.transcript"], "utterance 1")
                self.assertEqual(values["speech.transcribed_word_count"], 2)
                self.assertEqual(values["speech.transcribed_words_per_minute"], 240)
                self.assertTrue(all(signal["timestamp_ms"] == 2000 for signal in result))
                self.assertEqual(await plugin.flush("alice"), [])
                self.assertEqual(plugin.model.calls, 1)

    async def test_flush_removes_only_selected_subject_and_never_replays_full_windows(self):
        async def inline(function, *args, **kwargs): return function(*args, **kwargs)
        fake_numpy = SimpleNamespace(float32="float32", asarray=lambda values, **kwargs: values)
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            plugin.window_seconds = .25
            with patch.dict("sys.modules", {"numpy": fake_numpy}), patch("humansense.plugins.asyncio.to_thread", inline):
                full = await plugin.process(InputFrame([0.] * 6000, "audio", 3000, "alice", 16000))
                self.assertEqual(next(signal["value"] for signal in full if signal["name"] == "text.transcript"), "utterance 1")
                await plugin.process(InputFrame([0.] * 2000, "audio", 3100, "bob", 16000))
                flushed = await plugin.flush("alice", timestamp_ms=3200)
                self.assertEqual(next(signal["value"] for signal in flushed if signal["name"] == "text.transcript"), "utterance 2")
                self.assertEqual(len(plugin.buffers["bob"]), 2000)
                self.assertNotIn("alice", plugin.buffers)
                self.assertTrue(all(signal["timestamp_ms"] == 3200 for signal in flushed))
                self.assertEqual(await plugin.flush("alice"), [])
                await plugin.flush("bob")
                self.assertEqual(plugin.model.calls, 3)
                self.assertEqual(plugin.buffers, {})

    async def test_flush_discards_very_short_residual_without_running_model(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            plugin.window_seconds = 3
            await plugin.process(InputFrame([0.] * 1599, "audio", 2000, "alice", 16000))
            self.assertEqual(await plugin.flush("alice"), [])
            self.assertEqual(plugin.model.calls, 0)
            self.assertNotIn("alice", plugin.buffers)

    async def test_reset_discards_inflight_flush_result(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(function, *args, **kwargs):
            entered.set()
            await release.wait()
            return function(*args, **kwargs)
        fake_numpy = SimpleNamespace(float32="float32", asarray=lambda values, **kwargs: values)
        with tempfile.TemporaryDirectory() as directory:
            plugin = self.make_plugin(directory)
            plugin.window_seconds = 3
            await plugin.process(InputFrame([0.] * 2000, "audio", 2000, "alice", 16000))
            with patch.dict("sys.modules", {"numpy": fake_numpy}), patch("humansense.plugins.asyncio.to_thread", delayed):
                task = asyncio.create_task(plugin.flush("alice"))
                await entered.wait()
                plugin.reset()
                release.set()
                self.assertEqual(await task, [])
                self.assertEqual(plugin.model.calls, 0)
                self.assertEqual(plugin.buffers, {})


class SileroTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)
        self.worker = patch("humansense.plugins.asyncio.to_thread", inline)
        self.worker.start()

    async def asyncTearDown(self):
        self.worker.stop()

    async def test_recurrent_state_is_bound_and_reset_and_hysteresis_applies(self):
        from contextlib import nullcontext
        class Score:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        class Model:
            def __init__(self):
                self.values = iter([.8, .45, .1])
                self.did_reset = False
            def eval(self): pass
            def __call__(self, values, rate): return Score(next(self.values))
            def reset_states(self): self.did_reset = True
        model = Model()
        class JIT:
            @staticmethod
            def load(*a, **kw): return model
        class Torch:
            jit = JIT
            float32 = "float32"
            no_grad = staticmethod(nullcontext)
            tensor = staticmethod(lambda value, **kw: value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silero.jit"
            path.write_bytes(b"mock model artifact")
            with patch.dict("sys.modules", {"torch": Torch}):
                plugin = SileroVADPlugin(str(path), subject_id="alice")
                signals = await plugin.process(InputFrame([0.] * 1536, "audio", 1000, "alice", 16000))
                activity = [signal for signal in signals if signal["name"] == "speech.activity"]
                self.assertEqual([p["value"] for p in activity], [True, True, False])
                self.assertEqual([p["timestamp_ms"] for p in activity], [936, 968, 1000])
                self.assertIn("sha256:", activity[0]["model_version"])
                with self.assertRaisesRegex(ValueError, "bound to one subject"):
                    await plugin.process(InputFrame([0.] * 512, "audio", 1001, "bob", 16000))
                await plugin.reset()
                self.assertTrue(model.did_reset)
                self.assertFalse(plugin.active)


if __name__ == "__main__":
    unittest.main()
