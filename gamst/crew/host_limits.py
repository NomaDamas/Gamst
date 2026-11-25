"""Explicit host-boundary resource enforcement for provider processes.

ResourceAllocator remains an accounting ledger.  This module is an opt-in
attachment seam for a real host controller.  Linux cgroup v2 is supported when
the caller supplies a writable cgroup root.  macOS and Windows are reported as
unsupported until a native controller is added; they are never reported as
enforced by accident.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from .resources import ResourceRequest


HOST_LIMIT_MODES = frozenset(("accounting", "best_effort", "strict"))


class HostLimitError(RuntimeError):
    """Base class for an explicit host-enforcement failure."""


class HostLimitUnsupported(HostLimitError):
    """The requested host cannot enforce the requested dimensions."""


class HostLimitUnavailable(HostLimitError):
    """A controller exists in principle but is not available or writable."""


@dataclass(frozen=True)
class HostLimitReport:
    mode: str
    status: str
    platform: str
    reason: str
    cgroup_path: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "mode": self.mode,
            "status": self.status,
            "platform": self.platform,
            "reason": self.reason,
            "cgroup_path": self.cgroup_path,
        }


@dataclass
class HostLimitController:
    """Attach declared limits to a process without hiding host limitations.

    ``accounting`` is the default and performs no host operation.  ``best_effort``
    reports ``unavailable``/``unsupported`` and lets the process run when the
    host cannot enforce a request.  ``strict`` raises in those cases so callers
    can fail closed before accepting a runtime as ready.
    """

    mode: str = "accounting"
    system: str | None = None
    cgroup_root: Path | None = None
    cgroup_name: str = "gamst-crew"

    def __post_init__(self):
        if self.mode not in HOST_LIMIT_MODES:
            raise ValueError(f"mode must be one of {sorted(HOST_LIMIT_MODES)}")
        if not isinstance(self.cgroup_name, str) or not self.cgroup_name or len(self.cgroup_name) > 64:
            raise ValueError("cgroup_name must be a bounded non-empty string")
        if self.cgroup_root is not None:
            self.cgroup_root = Path(self.cgroup_root)

    def _platform(self) -> str:
        return (self.system or platform.system()).lower()

    def _report(self, status: str, reason: str, path: Path | None = None) -> HostLimitReport:
        return HostLimitReport(
            self.mode, status, self._platform(), reason,
            None if path is None else str(path),
        )

    def _failure(self, report: HostLimitReport):
        if self.mode == "strict":
            if report.status == "unsupported":
                raise HostLimitUnsupported(report.reason)
            raise HostLimitUnavailable(report.reason)
        return report

    @staticmethod
    def _request_dimensions(request: ResourceRequest) -> tuple[bool, bool, bool]:
        return request.cpu_cores > 0, request.ram_bytes > 0, request.gpu_count > 0 or request.vram_bytes > 0

    def attach(self, pid: int, request: ResourceRequest) -> HostLimitReport:
        if type(pid) is not int or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not isinstance(request, ResourceRequest):
            raise TypeError("ResourceRequest required")
        cpu, ram, gpu = self._request_dimensions(request)
        if not (cpu or ram or gpu):
            return self._report("no_limits", "request declares no host dimensions")
        if self.mode == "accounting":
            return self._report("accounting", "host enforcement is disabled by default")
        system = self._platform()
        if system != "linux":
            return self._failure(self._report("unsupported", "native host controller is not implemented"))
        if gpu:
            return self._failure(self._report("unsupported", "cgroup v2 seam does not enforce GPU or VRAM"))
        root = Path(self.cgroup_root or "/sys/fs/cgroup")
        controllers = root / "cgroup.controllers"
        if not controllers.is_file():
            return self._failure(self._report("unavailable", "cgroup v2 controller file is missing", root))
        try:
            available = set(controllers.read_text(encoding="utf-8").split())
        except OSError as exc:
            return self._failure(self._report("unavailable", f"cannot read cgroup controllers: {exc}", root))
        required = {name for name, enabled in (("cpu", cpu), ("memory", ram)) if enabled}
        if not required.issubset(available):
            missing = ",".join(sorted(required - available))
            return self._failure(self._report("unsupported", f"cgroup controllers unavailable: {missing}", root))
        scope = root / self.cgroup_name / f"runtime-{pid}"
        try:
            scope.mkdir(parents=True, exist_ok=True)
            cpu_max = scope / "cpu.max"
            memory_max = scope / "memory.max"
            procs = scope / "cgroup.procs"
            if cpu and not cpu_max.is_file():
                raise OSError("cpu.max is missing from cgroup scope")
            if ram and not memory_max.is_file():
                raise OSError("memory.max is missing from cgroup scope")
            if not procs.is_file():
                raise OSError("cgroup.procs is missing from cgroup scope")
            if cpu:
                period = 100_000
                quota = max(1, round(request.cpu_cores * period))
                cpu_max.write_text(f"{quota} {period}", encoding="ascii")
            if ram:
                memory_max.write_text(str(request.ram_bytes), encoding="ascii")
            procs.write_text(str(pid), encoding="ascii")
        except OSError as exc:
            return self._failure(self._report("unavailable", f"cannot configure cgroup scope: {exc}", scope))
        return self._report("enforced", "cgroup v2 limits attached", scope)

    def detach(self, pid: int) -> HostLimitReport:
        if type(pid) is not int or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if self.mode == "accounting":
            return self._report("accounting", "host enforcement is disabled by default")
        if self._platform() != "linux":
            return self._report("unsupported", "native host controller is not implemented")
        root = Path(self.cgroup_root or "/sys/fs/cgroup")
        scope = root / self.cgroup_name / f"runtime-{pid}"
        try:
            scope.rmdir()
        except FileNotFoundError:
            return self._report("unavailable", "cgroup scope was already absent", scope)
        except OSError as exc:
            return self._failure(self._report("unavailable", f"cannot remove cgroup scope: {exc}", scope))
        return self._report("detached", "cgroup v2 scope removed", scope)

    def diagnostics(self) -> HostLimitReport:
        """Return capability diagnostics without mutating the host."""
        if self.mode == "accounting":
            return self._report("accounting", "host enforcement is disabled by default")
        if self._platform() != "linux":
            return self._report("unsupported", "native host controller is not implemented")
        root = Path(self.cgroup_root or "/sys/fs/cgroup")
        return self._report(
            "available" if (root / "cgroup.controllers").is_file() else "unavailable",
            "cgroup v2 controller file present" if (root / "cgroup.controllers").is_file() else "cgroup v2 controller file is missing",
            root,
        )
