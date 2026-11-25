import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from gamst.crew.launcher import launch_profile, preflight_profile
from gamst.crew.manager import RuntimeManager
from gamst.crew.resources import ResourceAllocator, ResourceCapacity, ResourceRequest

SCRIPT = "import json;print(json.dumps({'type':'ready'}),flush=True)"
RUN_SCRIPT = "import json,sys;print(json.dumps({'type':'ready'}),flush=True);sys.stdin.read()"


class Launcher(unittest.IsolatedAsyncioTestCase):
    def test_preflight_reports_host_and_artifact_evidence(self):
        profile = {
            "name": "locked", "runtime": "darwin", "python": "3.11",
            "accelerator": "mps", "provider": "x", "model_revision": "r",
            "license": "MIT", "lock_sha256": "a" * 64, "lockfile": "lock.txt",
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profiles.json"
            path.write_text(json.dumps([profile]))
            result = preflight_profile(
                path,
                "locked",
                host={"os": "darwin", "arch": "arm64", "accelerator": "mps"},
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["artifact_checks"]["lockfile"], "missing")
            self.assertEqual(result["host"]["os"], "darwin")
            self.assertIn("lockfile verification failed: missing", result["reason"])

    def test_preflight_reports_ready_when_declared_artifact_matches(self):
        import hashlib

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lock = root / "lock.txt"
            lock.write_text("locked")
            profile = {
                "name": "locked", "runtime": "darwin", "python": "3.11",
                "accelerator": "mps", "provider": "x", "model_revision": "r",
                "license": "MIT", "lock_sha256": hashlib.sha256(b"locked").hexdigest(),
                "lockfile": "lock.txt",
            }
            path = root / "profiles.json"
            path.write_text(json.dumps([profile]))
            result = preflight_profile(
                path,
                "locked",
                host={"os": "darwin", "arch": "arm64", "accelerator": "mps"},
            )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["artifact_checks"], {"lockfile": "verified", "artifact": "unmeasured"})
            self.assertIsNone(result["reason"])

    def test_preflight_captures_incompatible_host_without_starting(self):
        profile = {
            "name": "linux", "runtime": "linux", "python": "3.11",
            "accelerator": "cuda", "provider": "x", "model_revision": "r",
            "license": "MIT", "lock_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profiles.json"
            path.write_text(json.dumps([profile]))
            result = preflight_profile(
                path,
                "linux",
                host={"os": "darwin", "arch": "arm64", "accelerator": "mps"},
            )
            self.assertEqual(result["status"], "blocked")
            self.assertIn("not compatible", result["reason"])

    async def test_incompatible_profile_rejected_before_start(self):
        profile = {
            "name": "linux",
            "runtime": "linux",
            "python": "3.11",
            "accelerator": "cuda",
            "provider": "x",
            "model_revision": "r",
            "license": "MIT",
            "lock_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profiles.json"
            path.write_text(json.dumps([profile]))
            with patch("gamst.crew.launcher.profile") as host:
                host.return_value.descriptor.return_value = {"os": "darwin", "accelerator": "mps"}
                with self.assertRaises(RuntimeError):
                    await launch_profile(
                        RuntimeManager(), path, "linux", [sys.executable, "-c", SCRIPT]
                    )

    async def test_declared_lock_must_verify_before_start(self):
        profile = {
            "name": "locked", "runtime": "darwin", "python": "3.11",
            "accelerator": "mps", "provider": "x", "model_revision": "r",
            "license": "MIT", "lock_sha256": "a" * 64, "lockfile": "lock.txt",
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profiles.json"
            path.write_text(json.dumps([profile]))
            with patch("gamst.crew.launcher.profile") as host:
                host.return_value.descriptor.return_value = {"os": "darwin", "arch": "arm64", "accelerator": "mps"}
                with self.assertRaises(RuntimeError):
                    await launch_profile(RuntimeManager(), path, "locked", [sys.executable, "-c", SCRIPT])

    async def test_launch_forwards_resource_reservation_and_releases_on_close(self):
        profile = {
            "name": "cpu", "runtime": "darwin", "python": "3.11",
            "accelerator": "mps", "provider": "x", "model_revision": "r",
            "license": "MIT", "lock_sha256": "a" * 64,
        }
        allocator = ResourceAllocator(ResourceCapacity(cpu_cores=2))
        manager = RuntimeManager(resource_allocator=allocator)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profiles.json"
            path.write_text(json.dumps([profile]))
            with patch("gamst.crew.launcher.profile") as host:
                host.return_value.descriptor.return_value = {
                    "os": "darwin", "arch": "arm64", "accelerator": "mps",
                }
                await launch_profile(
                    manager,
                    path,
                    "cpu",
                    [sys.executable, "-u", "-c", RUN_SCRIPT],
                    resources=ResourceRequest(cpu_cores=1),
                )
            try:
                self.assertEqual(manager.resource_telemetry()["active_reservations"], 1)
            finally:
                await manager.close("cpu")
        self.assertEqual(allocator.telemetry()["active_reservations"], 0)


if __name__ == "__main__":
    unittest.main()
