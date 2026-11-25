from __future__ import annotations
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    python_major: int = 3
    python_minor: int = 11
    arch: str | None = None
    opencv_max_major: int | None = 4
    accelerator: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    status: str
    checks: dict[str, str]


def check(
    requirement: Requirement,
    *,
    python_version=None,
    arch=None,
    os_name=None,
    opencv_version=None,
    accelerator=None,
):
    version = python_version or sys.version_info[:2]
    arch = arch or platform.machine().lower()
    os_name = os_name or platform.system().lower()
    accelerator = accelerator or (
        "mps" if os_name == "darwin" and arch in ("arm64", "aarch64") else "unknown"
    )
    checks = {}
    checks["python"] = (
        "ok"
        if tuple(version) >= (requirement.python_major, requirement.python_minor)
        else "incompatible"
    )
    checks["arch"] = (
        "ok" if requirement.arch is None or arch == requirement.arch else "incompatible"
    )
    if requirement.opencv_max_major is None or opencv_version is None:
        checks["opencv"] = "unknown"
    else:
        try:
            major = int(str(opencv_version).split(".")[0])
        except (TypeError, ValueError):
            major = None
        checks["opencv"] = (
            "ok"
            if major is not None and major <= requirement.opencv_max_major
            else "incompatible"
        )
    checks["accelerator"] = (
        "ok"
        if requirement.accelerator is None or accelerator == requirement.accelerator
        else "unknown"
    )
    status = (
        "incompatible"
        if "incompatible" in checks.values()
        else "ready"
        if all(v == "ok" for v in checks.values())
        else "unknown"
    )
    return Diagnostic(status, checks)
