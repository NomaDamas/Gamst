import sys
import tempfile
import unittest
from pathlib import Path

from gamst.crew import (
    HostLimitController,
    HostLimitUnsupported,
    ProcessRuntime,
    ResourceRequest,
    RuntimeDescriptor,
)


READY_SCRIPT = "import json,sys; print(json.dumps({'type':'ready'}),flush=True); sys.stdin.readline()"


class HostLimits(unittest.TestCase):
    def test_accounting_mode_is_explicit_and_non_mutating(self):
        controller = HostLimitController(mode="accounting", system="Darwin")
        report = controller.attach(1, ResourceRequest(ram_bytes=1024))
        self.assertEqual(report.status, "accounting")
        self.assertEqual(report.mode, "accounting")

    def test_macos_strict_mode_fails_closed(self):
        controller = HostLimitController(mode="strict", system="Darwin")
        with self.assertRaises(HostLimitUnsupported):
            controller.attach(1, ResourceRequest(ram_bytes=1024))

    def test_linux_fake_cgroup_attaches_and_detaches(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "cgroup.controllers").write_text("cpu memory", encoding="ascii")
            controller = HostLimitController(
                mode="strict", system="Linux", cgroup_root=root, cgroup_name="test"
            )
            scope = root / "test" / "runtime-123"
            scope.mkdir(parents=True)
            for name in ("cpu.max", "memory.max", "cgroup.procs"):
                (scope / name).touch()
            report = controller.attach(123, ResourceRequest(cpu_cores=1, ram_bytes=4096))
            self.assertEqual(report.status, "enforced")
            self.assertEqual((scope / "cpu.max").read_text(), "100000 100000")
            self.assertEqual((scope / "memory.max").read_text(), "4096")
            self.assertEqual((scope / "cgroup.procs").read_text(), "123")
            for path in scope.iterdir():
                path.unlink()
            self.assertEqual(controller.detach(123).status, "detached")
            self.assertFalse(scope.exists())


class ProcessHostLimits(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_reports_attached_host_limit(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "cgroup.controllers").write_text("cpu memory", encoding="ascii")

            class FixtureController(HostLimitController):
                def attach(self, pid, request):
                    scope = root / self.cgroup_name / f"runtime-{pid}"
                    scope.mkdir(parents=True, exist_ok=True)
                    for name in ("cpu.max", "memory.max", "cgroup.procs"):
                        (scope / name).touch()
                    return super().attach(pid, request)

                def detach(self, pid):
                    scope = root / self.cgroup_name / f"runtime-{pid}"
                    for path in scope.iterdir():
                        path.unlink()
                    return super().detach(pid)

            runtime = ProcessRuntime(
                (sys.executable, "-u", "-c", READY_SCRIPT),
                RuntimeDescriptor("x", "linux", "fixture", "1", "MIT", True),
            )
            runtime.configure_host_limits(
                FixtureController(mode="strict", system="Linux", cgroup_root=root),
                ResourceRequest(cpu_cores=0.5, ram_bytes=4096),
            )
            await runtime.start()
            try:
                self.assertEqual(runtime.health()["host_limits"]["status"], "enforced")
            finally:
                await runtime.stop()
            self.assertEqual(runtime.health()["host_limits"]["status"], "detached")


if __name__ == "__main__":
    unittest.main()
