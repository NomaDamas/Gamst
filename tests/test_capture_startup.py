"""Native startup ownership using fake devices and event-controlled workers."""
import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from humansense.capture import CaptureSession


class Device:
    def __init__(self):
        self.calls = []
    def isOpened(self): return True
    def read(self): return True, object()
    def start(self): self.calls.append('start')
    def stop(self): self.calls.append('stop')
    def close(self): self.calls.append('close')
    def release(self): self.calls.append('release')


class Gate:
    def __init__(self, result=None, error=None):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.allow = threading.Event()
        self.result, self.error = result, error
    def __call__(self, *args, **kwargs):
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.allow.wait(5): raise RuntimeError('fixture gate timeout')
        if self.error: raise self.error
        return self.result


class CaptureStartupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop_errors = []
        self.loop = asyncio.get_running_loop()
        self.previous_handler = self.loop.get_exception_handler()
        self.loop.set_exception_handler(lambda loop, context: self.loop_errors.append(context))

    async def asyncTearDown(self):
        await asyncio.sleep(0)
        self.loop.set_exception_handler(self.previous_handler)
        self.assertEqual(self.loop_errors, [], 'Owned startup must not leak background exceptions')

    async def test_cancel_or_stop_owns_late_camera_and_microphone_constructors(self):
        for audio in (False, True):
            for cancel in (False, True):
                with self.subTest(audio=audio, cancel=cancel):
                    device = Device(); gate = Gate(device)
                    modules = {'sounddevice': SimpleNamespace(InputStream=gate)} if audio else {'cv2': SimpleNamespace(VideoCapture=gate)}
                    session = CaptureSession(object(), audio=audio, video=None if audio else 0, consent=True)
                    pending = []
                    with patch.dict('sys.modules', modules):
                        try:
                            start = asyncio.create_task(session.start()); pending.append(start)
                            await asyncio.wait_for(gate.entered.wait(), 2)
                            if cancel:
                                start.cancel()
                                await asyncio.sleep(0)
                                start.cancel()
                            stop = asyncio.create_task(session.stop()); pending.append(stop)
                            await asyncio.sleep(0)
                            self.assertFalse(start.done()); self.assertFalse(stop.done())
                            self.assertEqual(device.calls, [])
                            gate.allow.set()
                            results = await asyncio.gather(*pending, return_exceptions=True)
                            self.assertIsInstance(results[0], asyncio.CancelledError if cancel else RuntimeError)
                            self.assertIsNone(results[1])
                            self.assertEqual(device.calls, ['stop', 'close'] if audio else ['release'])
                            self.assertFalse(session.status()['running'])
                            self.assertIsNone(session._camera); self.assertIsNone(session._mic)
                            await session.stop()
                            self.assertEqual(len(device.calls), 2 if audio else 1)
                        finally:
                            gate.allow.set(); await asyncio.gather(*pending, return_exceptions=True)
                            await session.stop()

    async def test_cancel_during_microphone_start_joins_before_stop_close(self):
        mic = Device(); gate = Gate(); mic.start = gate
        session = CaptureSession(object(), audio=True, consent=True)
        with patch.dict('sys.modules', {'sounddevice': SimpleNamespace(InputStream=lambda **kw: mic)}):
            start = asyncio.create_task(session.start())
            try:
                await asyncio.wait_for(gate.entered.wait(), 2)
                start.cancel(); await asyncio.sleep(0); start.cancel(); await asyncio.sleep(0)
                self.assertFalse(start.done()); self.assertEqual(mic.calls, [])
                gate.allow.set()
                with self.assertRaises(asyncio.CancelledError): await start
                self.assertEqual(mic.calls, ['stop', 'close'])
                self.assertFalse(session.status()['running'])
            finally:
                gate.allow.set(); await asyncio.gather(start, return_exceptions=True); await session.stop()

    async def test_dual_partial_failure_releases_camera_and_preserves_error(self):
        for failing_start in (False, True):
            camera, mic = Device(), Device()
            gate = Gate(error=ValueError('fixture startup failure'))
            constructor = (lambda **kw: mic) if failing_start else gate
            if failing_start: mic.start = gate
            session = CaptureSession(object(), video=0, audio=True, consent=True)
            with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera), 'sounddevice': SimpleNamespace(InputStream=constructor)}):
                start = asyncio.create_task(session.start())
                try:
                    await asyncio.wait_for(gate.entered.wait(), 2); gate.allow.set()
                    with self.assertRaisesRegex(ValueError, 'fixture startup failure'): await start
                    self.assertEqual(camera.calls, ['release'])
                    self.assertEqual(mic.calls, ['stop', 'close'] if failing_start else [])
                    self.assertFalse(session.status()['running'])
                finally:
                    gate.allow.set(); await asyncio.gather(start, return_exceptions=True); await session.stop()

    async def test_cancel_queued_start_does_not_cancel_active_start_or_cleanup(self):
        device = Device(); gate = Gate(device)
        session = CaptureSession(object(), video=0, consent=True)
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=gate)}):
            active = asyncio.create_task(session.start()); queued = stop = None
            try:
                await asyncio.wait_for(gate.entered.wait(), 2)
                queued = asyncio.create_task(session.start()); await asyncio.sleep(0); queued.cancel()
                with self.assertRaises(asyncio.CancelledError): await queued
                self.assertFalse(active.done()); self.assertIsNone(session._cleanup_task)
                stop = asyncio.create_task(session.stop()); await asyncio.sleep(0)
                queued = asyncio.create_task(session.start()); await asyncio.sleep(0); queued.cancel()
                with self.assertRaises(asyncio.CancelledError): await queued
                self.assertFalse(session._cleanup_task.done()); self.assertFalse(session._startup_task.cancelled())
                gate.allow.set()
                with self.assertRaisesRegex(RuntimeError, 'interrupted by stop'): await active
                await stop
                self.assertEqual(device.calls, ['release'])
            finally:
                gate.allow.set()
                await asyncio.gather(*(t for t in (active, queued, stop) if t is not None), return_exceptions=True)
                await session.stop()

    async def test_stopped_startup_can_restart_with_fresh_handle(self):
        camera, fresh = Device(), Device()
        gate = Gate(camera)
        class Runtime:
            async def push_video(self, *args, **kwargs): pass
        session = CaptureSession(Runtime(), video=0, consent=True)
        module = SimpleNamespace(VideoCapture=gate)
        with patch.dict('sys.modules', {'cv2': module}):
            start = asyncio.create_task(session.start()); stop = None
            try:
                await asyncio.wait_for(gate.entered.wait(), 2)
                stop = asyncio.create_task(session.stop()); await asyncio.sleep(0)
                gate.allow.set()
                with self.assertRaisesRegex(RuntimeError, 'interrupted by stop'): await start
                await stop
                module.VideoCapture = lambda _: fresh
                await session.start()
                self.assertTrue(session.status()['running'])
                self.assertIs(session._camera, fresh)
                await session.stop()
                self.assertEqual(camera.calls, ['release'])
                self.assertEqual(fresh.calls, ['release'])
            finally:
                gate.allow.set()
                await asyncio.gather(*(t for t in (start, stop) if t is not None), return_exceptions=True)
                await session.stop()
