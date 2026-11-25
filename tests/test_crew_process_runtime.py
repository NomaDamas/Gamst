import asyncio
import json
import sys
import unittest
from gamst.crew import RuntimeManager
from gamst.crew.packet import EventPacket, RuntimeDescriptor
from gamst.crew.process_runtime import ProcessRuntime

SCRIPT = "import sys,json; print(json.dumps({'type':'ready'}),flush=True); [print(json.dumps({'accepted':True}),flush=True) for line in sys.stdin]"
OVERSIZED_RESPONSE_SCRIPT = "import sys,json; print(json.dumps({'type':'ready'}),flush=True); next(sys.stdin); print('x'*1100,flush=True)"


def packet():
    return EventPacket(
        "e",
        "s",
        "u",
        "vision",
        "event",
        "mono",
        1,
        1,
        None,
        None,
        1,
        1,
        1,
        False,
        0,
        "fixture",
        "1",
        {"x": 1},
    )


class ProcessRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_submit_stop(self):
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", SCRIPT),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
        )
        await runtime.start()
        try:
            self.assertEqual(await runtime.submit(packet()), {"accepted": True})
            self.assertTrue(runtime.health()["running"])
        finally:
            await runtime.stop()
        self.assertIsNone(runtime.process)

    async def test_stderr_is_drained_and_exposed_as_diagnostics(self):
        script = "import sys,json; print(json.dumps({'type':'ready'}),flush=True); print('provider warning',file=sys.stderr,flush=True); [print(json.dumps({'accepted':True}),flush=True) for line in sys.stdin]"
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", script),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.01)
            self.assertIn("provider warning", runtime.health()["stderr_tail"])
        finally:
            await runtime.stop()

    async def test_bad_readiness_fails(self):
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", "print('bad',flush=True)"),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
        )
        with self.assertRaises(RuntimeError):
            await runtime.start()
        await runtime.stop()

    async def test_tampered_mapping_rejected_before_dispatch_and_counted(self):
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", SCRIPT),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
        )
        await runtime.start()
        try:
            tampered = json.loads(packet().to_bytes())
            tampered["value"]["x"] = 2
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                await runtime.submit(tampered)
            self.assertEqual(runtime.health()["integrity_rejected"], 1)
        finally:
            await runtime.stop()

    async def test_oversized_response_is_rejected_before_buffer_growth(self):
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", OVERSIZED_RESPONSE_SCRIPT),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
            max_line=1024,
        )
        await runtime.start()
        try:
            with self.assertRaisesRegex(ValueError, "line exceeds line budget"):
                await runtime.submit(packet())
        finally:
            await runtime.stop()

    async def test_manager_register_process_roundtrip(self):
        manager = RuntimeManager(capacity=2)
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", SCRIPT),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
        )
        await manager.register_process("x", runtime)
        try:
            await manager.submit("x", packet())
            await asyncio.sleep(0.02)
            self.assertTrue(manager.health("x").descriptor.ready)
            runtime.process.terminate()
            await runtime.process.wait()
            self.assertEqual(manager.health("x").state, "crashed")
        finally:
            await manager.close("x")

    async def test_manager_health_exposes_integrity_rejections(self):
        manager = RuntimeManager(capacity=2)
        runtime = ProcessRuntime(
            (sys.executable, "-u", "-c", SCRIPT),
            RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True),
        )
        await manager.register_process("x", runtime)
        try:
            tampered = json.loads(packet().to_bytes())
            tampered["value"]["x"] = 2
            await manager.submit("x", tampered)
            await asyncio.sleep(0.02)
            self.assertEqual(manager.health("x").integrity_rejected, 1)
        finally:
            await manager.close("x")


if __name__ == "__main__":
    unittest.main()
