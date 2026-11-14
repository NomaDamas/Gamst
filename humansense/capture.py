"""Explicit-consent local camera/microphone capture with bounded queues."""
from __future__ import annotations

import asyncio
import inspect
import threading
import time
from typing import Any


class CaptureSession:
    """Capture is opt-in and does not save raw images or recordings.

    ``subject_id`` is a caller-assigned stream owner, not a verified face identity.
    All people in camera images remain scene observations. Background errors are
    exposed by status() and wait(); they never silently claim capture success.
    """

    def __init__(self, human: Any, video: int | str | None = None,
                 audio: bool = False, consent: bool = False,
                 sample_rate: int = 16000, fps: float = 10,
                 subject_id: str = "local-user"):
        if sample_rate <= 0 or fps <= 0 or fps > 60:
            raise ValueError("sample_rate must be positive and fps in (0, 60]")
        self.human, self.video, self.audio = human, video, audio
        self.consent, self.sample_rate, self.fps = consent, sample_rate, fps
        self.subject_id = subject_id
        self._camera = self._mic = None
        self._tasks: list[asyncio.Task] = []
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._running = self._paused = False
        self._errors: list[str] = []
        self._start_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._camera_io_lock = threading.Lock()
        self._audio_delivery_slots = threading.BoundedSemaphore(8)
        self._capture_generation = 0
        self._accept_after_ms = 0
        self._counts = {"video_frames": 0, "audio_windows": 0, "dropped_audio_windows": 0}

    async def start(self):
        # Do not replace a previous generation's handles while its owned
        # cleanup is still releasing devices or resetting provider state.
        async with self._start_lock:
            if self._cleanup_task is not None:
                await self._join_cleanup(self._cleanup_task)
            if self._running:
                return self
            self._cleanup_task = None
            return await self._start()

    async def _start(self):
        if not self.consent:
            raise PermissionError("Explicit capture consent=True is required")
        if self._running:
            return self
        if self.video is None and not self.audio:
            raise ValueError("Choose a camera and/or microphone")
        self._errors.clear()
        self._capture_generation += 1
        self._accept_after_ms = time.time_ns() // 1_000_000
        try:
            if self.video is not None:
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError('Install humansense-sdk[vision] for camera capture') from exc
                self._camera = await asyncio.to_thread(cv2.VideoCapture, self.video)
                if not self._camera.isOpened():
                    raise RuntimeError(f"Cannot open camera {self.video!r}")
            if self.audio:
                try:
                    import sounddevice as sd
                except ImportError as exc:
                    raise RuntimeError('Install humansense-sdk[audio] for microphone capture') from exc
                loop = asyncio.get_running_loop()
                def callback(indata, frames, timing, status):
                    self._post_audio(loop, indata, str(status) if status else None)
                self._mic = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                           dtype="float32", blocksize=int(self.sample_rate * .1),
                                           callback=callback)
                await asyncio.to_thread(self._mic.start)
            self._running = True
            self._paused = False
            if self._camera is not None:
                self._tasks.append(asyncio.create_task(self._camera_loop(), name="humansense-camera"))
            if self._mic is not None:
                self._tasks.append(asyncio.create_task(self._microphone_loop(), name="humansense-microphone"))
            for task in self._tasks:
                task.add_done_callback(self._task_done)
            return self
        except BaseException:
            # Cancelled device startup must also release already-opened handles.
            await self.stop()
            raise

    def _task_done(self, task):
        if task not in self._tasks:
            return
        if not task.cancelled() and task.exception() is not None:
            self._errors.append(f"{task.get_name()}: {task.exception()}")
            self._errors[:] = self._errors[-20:]
            self._begin_cleanup()

    def _begin_cleanup(self):
        if self._cleanup_task is None:
            self._running = False
            self._capture_generation += 1
            self._cleanup_task = asyncio.create_task(self._cleanup(), name="humansense-capture-cleanup")
            # Automatic failure cleanup can finish before any public waiter.
            # Retain its result for wait/stop, while consuming background errors.
            self._cleanup_task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception())
        return self._cleanup_task

    @staticmethod
    async def _join_cleanup(task):
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        if cancelled:
            # Cleanup errors stay recorded on the session. Caller cancellation
            # wins, but only after the owned operation has finished.
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    def _post_audio(self, loop, data, status):
        # A bounded asyncio.Queue alone does not bound callbacks already queued
        # by call_soon_threadsafe while the event loop is busy running a model.
        # Reserve a slot before copying the callback-owned raw audio buffer.
        if not self._running or self._paused:
            return
        if not self._audio_delivery_slots.acquire(blocking=False):
            self._counts["dropped_audio_windows"] += 1
            return
        try:
            payload = data.copy().reshape(-1).tolist()
            timestamp = time.time_ns() // 1_000_000
            loop.call_soon_threadsafe(self._deliver_audio, payload, timestamp, status)
        except Exception:
            self._audio_delivery_slots.release()
            raise

    def _deliver_audio(self, payload, timestamp, status):
        try:
            self._enqueue_audio(payload, timestamp, status)
        finally:
            self._audio_delivery_slots.release()

    def _enqueue_audio(self, payload, timestamp, status):
        if not self._running or self._paused or timestamp < self._accept_after_ms:
            return
        if status:
            self._errors.append(f"microphone: {status}")
            self._errors[:] = self._errors[-20:]
        if self._audio_queue.full():
            self._audio_queue.get_nowait()
            self._counts["dropped_audio_windows"] += 1
        self._audio_queue.put_nowait((payload, timestamp))

    async def _camera_loop(self):
        while self._running:
            started = asyncio.get_running_loop().time()
            if self._paused:
                await asyncio.sleep(.05)
                continue
            generation = self._capture_generation
            ok, image = await asyncio.to_thread(self._read_camera, self._camera)
            if not ok:
                raise RuntimeError("Camera frame read failed")
            if self._running and not self._paused and generation == self._capture_generation:
                await self.human.push_video(image, subject_id=self.subject_id,
                                            timestamp_ms=time.time_ns() // 1_000_000)
                self._counts["video_frames"] += 1
            await asyncio.sleep(max(0, 1 / self.fps - (asyncio.get_running_loop().time() - started)))

    async def _microphone_loop(self):
        while self._running:
            payload, timestamp = await self._audio_queue.get()
            if not self._paused:
                await self.human.push_audio(payload, subject_id=self.subject_id,
                                            timestamp_ms=timestamp, sample_rate=self.sample_rate)
                self._counts["audio_windows"] += 1

    async def pause(self):
        self._paused = True
        self._capture_generation += 1
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        # Invalidate in-flight inference before awaiting the device shutdown.
        await self._reset_processing()
        # Do not leave the microphone delivering data while paused.
        if self._mic is not None:
            await asyncio.to_thread(self._mic.stop)

    async def resume(self):
        if not self._running:
            raise RuntimeError("Capture is not running")
        self._accept_after_ms = time.time_ns() // 1_000_000
        if self._mic is not None:
            await asyncio.to_thread(self._mic.start)
        self._paused = False

    async def stop(self):
        """Join device release/reset, even if this waiter is cancelled repeatedly."""
        cleanup = self._begin_cleanup()
        if asyncio.current_task() in self._tasks:
            raise RuntimeError("Capture task cannot join its own cleanup; stop requested")
        await self._join_cleanup(cleanup)

    async def _cleanup(self):
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        mic, self._mic = self._mic, None
        camera, self._camera = self._camera, None
        cleanup_errors = []
        for device, operations in ((mic, ("stop", "close")), (camera, ("release",))):
            if device is not None:
                for operation in operations:
                    try:
                        function = self._release_camera if operation == "release" else getattr(device, operation)
                        if operation == "release":
                            await asyncio.to_thread(function, device)
                        else:
                            await asyncio.to_thread(function)
                    except Exception as exc:
                        cleanup_errors.append(f"{operation}: {exc}")
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        try:
            await self._reset_processing()
        except Exception as exc:
            cleanup_errors.append(f"reset_processing: {exc}")
        if cleanup_errors:
            self._errors.extend(cleanup_errors)
            self._errors[:] = self._errors[-20:]
            raise RuntimeError("; ".join(cleanup_errors))

    async def _reset_processing(self):
        reset = getattr(self.human, "reset_processing", None)
        if reset is not None:
            result = reset()
            if inspect.isawaitable(result):
                await result

    def _read_camera(self, camera):
        # Cancelling asyncio.to_thread does not interrupt native OpenCV I/O.
        # Serialize release with any still-running read to avoid using a
        # camera handle after it has been freed in another worker thread.
        with self._camera_io_lock:
            return camera.read()

    def _release_camera(self, camera):
        with self._camera_io_lock:
            camera.release()

    async def wait(self):
        """Join capture and owned cleanup before reporting capture/cleanup errors.

        Cancelling a waiter stops capture and still joins release before raising
        CancelledError. Native device I/O cannot be interrupted by asyncio.
        """
        if asyncio.current_task() in self._tasks:
            raise RuntimeError("Capture task cannot wait for its own completion")
        tasks = tuple(self._tasks)
        try:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if tasks or self._cleanup_task is not None:
                try:
                    await self._join_cleanup(self._begin_cleanup())
                except Exception:
                    # Preserve both the original failure and cleanup diagnostics.
                    if not self._errors:
                        raise
        except asyncio.CancelledError:
            try:
                await self.stop()
            except Exception:
                pass  # recorded cleanup errors remain visible; preserve cancellation
            raise
        if self._errors:
            raise RuntimeError("; ".join(self._errors))

    def status(self):
        return {"running": self._running, "paused": self._paused,
                "cleanup_pending": self._cleanup_task is not None and not self._cleanup_task.done(),
                "camera": self._camera is not None and self._running and not self._paused,
                "microphone": self._mic is not None and self._running and not self._paused,
                "consent": self.consent, "subject_id": self.subject_id,
                "identity_verified": False, "raw_media_saved": False,
                "errors": list(self._errors), **self._counts}

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.stop()
