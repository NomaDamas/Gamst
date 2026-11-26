from __future__ import annotations
import platform
import importlib.util
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    os: str
    arch: str
    accelerator: str
    python: str
    supports_cuda: bool
    supports_mps: bool

    def descriptor(self):
        return {
            "name": self.name,
            "os": self.os,
            "arch": self.arch,
            "accelerator": self.accelerator,
            "python": self.python,
            "supports_cuda": self.supports_cuda,
            "supports_mps": self.supports_mps,
        }


PROFILE_CATALOG = {
    "macos-arm64": {"os": "darwin", "arch": "arm64", "accelerators": ("mps", "cpu"), "runtime": "uv+onnx+mlx+llama.cpp"},
    "linux-cuda": {"os": "linux", "arch": "any", "accelerators": ("cuda", "cpu"), "runtime": "docker+nvidia-toolkit+triton"},
    "linux-cpu": {"os": "linux", "arch": "any", "accelerators": ("cpu",), "runtime": "uv+onnx+llama.cpp"},
    "windows-cpu-directml": {"os": "windows", "arch": "any", "accelerators": ("cpu", "directml"), "runtime": "uv+onnx+llama.cpp"},
    "jetson-arm64": {"os": "linux", "arch": ("aarch64", "arm64"), "accelerators": ("cuda", "tensorrt"), "runtime": "jetpack+tensorrt+gstreamer"},
}


def profile_catalog():
    """Return the declarative OS/runtime matrix without claiming host acceptance."""
    return {name: dict(value) for name, value in PROFILE_CATALOG.items()}


def provider_compatibility(*, provider: str, profile_name: str, installed: bool = False,
                           runtime_compatible: bool = False, weight_ready: bool = False,
                           benchmark_accepted: bool = False, available: bool | None = None):
    """Report the five compatibility states used by the cross-runtime contract."""
    if profile_name not in PROFILE_CATALOG:
        raise ValueError("unknown profile")
    if not isinstance(provider, str) or not provider:
        raise ValueError("provider must be non-empty")
    if available is None:
        available = provider in {"onnxruntime", "llama.cpp", "python"}
    return {"provider": provider, "profile": profile_name,
            "available": bool(available), "installed": bool(installed),
            "runtime-compatible": bool(runtime_compatible),
            "weight-ready": bool(weight_ready),
            "benchmark-accepted": bool(benchmark_accepted)}


def install_diagnostics(*, profile_name: str, required_commands=(), required_modules=()):
    """Collect deterministic preflight diagnostics; never installs or downloads."""
    if profile_name not in PROFILE_CATALOG:
        raise ValueError("unknown profile")
    commands = {str(command): shutil.which(str(command)) is not None for command in required_commands}
    modules = {str(module): importlib.util.find_spec(str(module)) is not None for module in required_modules}
    return {"profile": profile_name, "commands": commands, "modules": modules,
            "host_acceptance": "unmeasured", "weight_download": "not attempted"}


def profile():
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_mac_arm = system == "darwin" and machine in ("arm64", "aarch64")
    is_linux = system == "linux"
    return RuntimeProfile(
        name="mac-lightweight"
        if is_mac_arm
        else "linux-cuda-capable"
        if is_linux
        else system + "-" + machine,
        os=system,
        arch=machine,
        accelerator="mps" if is_mac_arm else "cuda-possible" if is_linux else "cpu",
        python=platform.python_version(),
        supports_cuda=is_linux,
        supports_mps=is_mac_arm,
    )


def require_profile(descriptor, *, required_os=None, required_accelerator=None):
    if not isinstance(descriptor, dict):
        raise ValueError("runtime descriptor must be an object")
    if required_os and descriptor.get("os") != required_os:
        raise RuntimeError(
            f"runtime requires {required_os}, got {descriptor.get('os')}"
        )
    if required_accelerator and descriptor.get("accelerator") != required_accelerator:
        raise RuntimeError(
            f"runtime requires {required_accelerator}, got {descriptor.get('accelerator')}"
        )
    return descriptor
