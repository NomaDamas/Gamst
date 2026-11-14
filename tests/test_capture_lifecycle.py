"""Owned fake-device cleanup; never opens physical capture devices."""
import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from humansense.capture import CaptureSession


class GateCamera:
    def __init__(self, loop, *, fail_read=True, fail_release=False):
        self.loop = loop
        self.entered = asyncio.Event()
        self.release_gate = threading.Event()
        self.releases = 0
        self.released = False
        self.fail_read = fail_read
        self.fail_release = fail_release

    def isOpened(self):
        return True

    def read(self):
        return (False, None) if self.fail_read else (True, object())

    def release(self):
        self.releases += 1
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.release_gate.wait(5):
            raise RuntimeError('test release gate was not opened')
        self.released = True
        if self.fail_release:
            raise RuntimeError('fixture release failure')


class CaptureLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_joins_release_before_and_after_background_failure(self):
        for early_wait in (True, False):
            camera = GateCamera(asyncio.get_running_loop())
            session = CaptureSession(object(), video=0, consent=True)
            waiter = None
            with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
                try:
                    await session.start()
                    if early_wait:
                        waiter = asyncio.create_task(session.wait())
                    await asyncio.wait_for(camera.entered.wait(), 2)
                    if waiter is None:
                        waiter = asyncio.create_task(session.wait())
                    await asyncio.sleep(0)
                    self.assertFalse(waiter.done())
                    self.assertFalse(camera.released)
                    self.assertFalse(session.status()['running'])
                    self.assertTrue(session.status()['cleanup_pending'])
                    self.assertIn('Camera frame read failed', ' '.join(session.status()['errors']))
                    camera.release_gate.set()
                    with self.assertRaisesRegex(RuntimeError, 'Camera frame read failed'):
                        await waiter
                    self.assertTrue(camera.released)
                    self.assertEqual(camera.releases, 1)
                    self.assertFalse(session.status()['cleanup_pending'])
                finally:
                    camera.release_gate.set()
                    if waiter is not None:
                        await asyncio.gather(waiter, return_exceptions=True)
                    await session.stop()

    async def test_concurrent_stop_and_repeated_cancel_join_one_cleanup(self):
        camera = GateCamera(asyncio.get_running_loop())
        session = CaptureSession(object(), video=0, consent=True)
        pending = []
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            try:
                await session.start()
                await asyncio.wait_for(camera.entered.wait(), 2)
                cancelled = asyncio.create_task(session.stop())
                other = asyncio.create_task(session.stop())
                waiter = asyncio.create_task(session.wait())
                pending = [cancelled, other, waiter]
                await asyncio.sleep(0)
                for _ in range(3):
                    cancelled.cancel()
                    waiter.cancel()
                    await asyncio.sleep(0)
                self.assertTrue(all(not task.done() for task in pending))
                self.assertFalse(session._cleanup_task.cancelled())
                camera.release_gate.set()
                with self.assertRaises(asyncio.CancelledError):
                    await cancelled
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                await other
                await session.stop()
                self.assertEqual(camera.releases, 1)
                self.assertTrue(camera.released)
                self.assertTrue(session._cleanup_task.done())
                self.assertEqual(session._tasks, [])
            finally:
                camera.release_gate.set()
                await asyncio.gather(*pending, return_exceptions=True)
                await session.stop()

    async def test_restart_waits_for_release_and_reset_before_new_handle(self):
        loop = asyncio.get_running_loop()
        first, second = GateCamera(loop), GateCamera(loop, fail_read=False)
        second.release_gate.set()
        reset_entered, reset_gate = asyncio.Event(), asyncio.Event()
        class Runtime:
            resets = 0
            async def reset_processing(self):
                self.resets += 1
                if self.resets == 1:
                    reset_entered.set()
                    await reset_gate.wait()
            async def push_video(self, *args, **kwargs):
                pass
        runtime = Runtime()
        acquired = []
        def open_camera(_):
            camera = first if not acquired else second
            acquired.append(camera)
            return camera
        session = CaptureSession(runtime, video=0, consent=True)
        restart = old_waiter = None
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=open_camera)}):
            try:
                await session.start()
                await asyncio.wait_for(first.entered.wait(), 2)
                restart = asyncio.create_task(session.start())
                old_waiter = asyncio.create_task(session.wait())
                await asyncio.sleep(0)
                self.assertEqual(len(acquired), 1)
                first.release_gate.set()
                await asyncio.wait_for(reset_entered.wait(), 2)
                self.assertFalse(restart.done())
                self.assertEqual(len(acquired), 1)
                reset_gate.set()
                with self.assertRaisesRegex(RuntimeError, 'Camera frame read failed'):
                    await old_waiter
                self.assertIs(await restart, session)
                self.assertTrue(session.status()['running'])
                self.assertTrue(first.released)
                self.assertFalse(second.released)
                self.assertEqual(session.status()['errors'], [])
                await session.stop()
                self.assertEqual((first.releases, second.releases), (1, 1))
                self.assertEqual(runtime.resets, 2)
            finally:
                first.release_gate.set()
                reset_gate.set()
                if old_waiter is not None:
                    await asyncio.gather(old_waiter, return_exceptions=True)
                if restart is not None:
                    await asyncio.gather(restart, return_exceptions=True)
                await session.stop()

    async def test_release_and_reset_failures_are_owned_and_block_restart(self):
        camera = GateCamera(asyncio.get_running_loop(), fail_release=True)
        camera.release_gate.set()
        class Runtime:
            async def reset_processing(self):
                raise RuntimeError('fixture reset failure')
        class Mic:
            stops = closes = 0
            def stop(self):
                self.stops += 1
                raise RuntimeError('fixture mic stop failure')
            def close(self):
                self.closes += 1
        mic = Mic()
        session = CaptureSession(Runtime(), video=0, consent=True)
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            await session.start()
            session._mic = mic
            with self.assertRaises(RuntimeError) as raised:
                await session.wait()
            for message in ('Camera frame read failed', 'fixture release failure',
                            'fixture reset failure', 'fixture mic stop failure'):
                self.assertIn(message, str(raised.exception))
            self.assertEqual((mic.stops, mic.closes, camera.releases), (1, 1, 1))
            with self.assertRaisesRegex(RuntimeError, 'fixture release failure'):
                await session.stop()
            with self.assertRaisesRegex(RuntimeError, 'fixture release failure'):
                await session.start()
            self.assertEqual(camera.releases, 1)
            self.assertFalse(session.status()['cleanup_pending'])

    async def test_wait_cancellation_stops_active_capture_and_joins_release(self):
        camera = GateCamera(asyncio.get_running_loop(), fail_read=False)
        class Runtime:
            async def push_video(self, *args, **kwargs):
                pass
        session = CaptureSession(Runtime(), video=0, consent=True)
        waiter = None
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            try:
                await session.start()
                waiter = asyncio.create_task(session.wait())
                await asyncio.sleep(0)
                waiter.cancel()
                await asyncio.wait_for(camera.entered.wait(), 2)
                waiter.cancel()
                await asyncio.sleep(0)
                self.assertFalse(waiter.done())
                camera.release_gate.set()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                self.assertTrue(camera.released)
                self.assertFalse(session.status()['running'])
            finally:
                camera.release_gate.set()
                if waiter is not None:
                    await asyncio.gather(waiter, return_exceptions=True)
                await session.stop()

    async def test_stop_waits_for_native_read_before_release_and_discards_frame(self):
        loop = asyncio.get_running_loop()
        entered, unblock = asyncio.Event(), threading.Event()
        camera = GateCamera(loop, fail_read=False)
        camera.release_gate.set()
        def read():
            loop.call_soon_threadsafe(entered.set)
            if not unblock.wait(5):
                raise RuntimeError('test read gate was not opened')
            return True, object()
        camera.read = read
        class Runtime:
            pushed = 0
            async def push_video(self, *args, **kwargs):
                self.pushed += 1
        runtime = Runtime()
        session = CaptureSession(runtime, video=0, consent=True)
        stopper = None
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            try:
                await session.start()
                await asyncio.wait_for(entered.wait(), 2)
                stopper = asyncio.create_task(session.stop())
                await asyncio.sleep(0)
                stopper.cancel()
                await asyncio.sleep(0)
                self.assertFalse(stopper.done())
                self.assertEqual(camera.releases, 0)
                unblock.set()
                with self.assertRaises(asyncio.CancelledError):
                    await stopper
                self.assertTrue(camera.released)
                self.assertEqual(runtime.pushed, 0)
            finally:
                unblock.set()
                if stopper is not None:
                    await asyncio.gather(stopper, return_exceptions=True)
                await session.stop()

    async def test_multiple_waiters_share_cleanup_after_one_waiter_cancels(self):
        camera = GateCamera(asyncio.get_running_loop(), fail_read=False)
        class Runtime:
            async def push_video(self, *args, **kwargs):
                pass
        session = CaptureSession(Runtime(), video=0, consent=True)
        waiters = []
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            try:
                await session.start()
                waiters = [asyncio.create_task(session.wait()) for _ in range(2)]
                await asyncio.sleep(0)
                waiters[0].cancel()
                await asyncio.wait_for(camera.entered.wait(), 2)
                self.assertTrue(all(not waiter.done() for waiter in waiters))
                camera.release_gate.set()
                with self.assertRaises(asyncio.CancelledError):
                    await waiters[0]
                await waiters[1]
                self.assertTrue(camera.released)
                self.assertEqual(camera.releases, 1)
            finally:
                camera.release_gate.set()
                await asyncio.gather(*waiters, return_exceptions=True)
                await session.stop()

    async def test_stop_from_capture_callback_fails_without_self_join_deadlock(self):
        camera = GateCamera(asyncio.get_running_loop(), fail_read=False)
        camera.release_gate.set()
        class Runtime:
            async def push_video(self, *args, **kwargs):
                await session.stop()
        session = CaptureSession(Runtime(), video=0, consent=True)
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            try:
                await session.start()
                with self.assertRaisesRegex(RuntimeError, 'cannot join its own cleanup'):
                    await asyncio.wait_for(session.wait(), 2)
                self.assertTrue(camera.released)
                self.assertEqual(camera.releases, 1)
            finally:
                await session.stop()

    async def test_cleanup_records_failure_before_delayed_done_callback(self):
        camera = GateCamera(asyncio.get_running_loop(), fail_read=False)
        camera.release_gate.set()
        class Runtime:
            async def push_video(self, *args, **kwargs):
                await session.stop()
        session = CaptureSession(Runtime(), video=0, consent=True)
        deferred = []
        original = session._task_done
        # Hold callbacks until ownership has been cleared. This exercises the
        # scheduler ordering deterministically on every supported Python version.
        session._task_done = deferred.append
        with patch.dict('sys.modules', {'cv2': SimpleNamespace(VideoCapture=lambda _: camera)}):
            try:
                await session.start()
                with self.assertRaisesRegex(RuntimeError, 'cannot join its own cleanup'):
                    await asyncio.wait_for(session.wait(), 2)
                self.assertTrue(camera.released)
                self.assertEqual(session._tasks, [])
                self.assertEqual(len(deferred), 1)
                before = session.status()['errors']
                self.assertEqual(len(before), 1)
                for task in deferred:
                    original(task)
                    original(task)
                self.assertEqual(session.status()['errors'], before)
                self.assertEqual(camera.releases, 1)
            finally:
                await session.stop()

    async def test_cleanup_callback_cannot_restart_or_join_itself(self):
        for operation in ('start', 'stop', 'wait'):
            class Runtime:
                async def reset_processing(self):
                    await getattr(session, operation)()
            session = CaptureSession(Runtime(), video=0, consent=True)
            with self.assertRaisesRegex(RuntimeError, 'cannot'):
                await asyncio.wait_for(session.stop(), 2)
            self.assertFalse(session.status()['cleanup_pending'])
            self.assertIn('reset_processing', ' '.join(session.status()['errors']))
