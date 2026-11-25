from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from .manager import RuntimeManager
from .process_runtime import ProcessRuntime
from .registry import read_profiles
from .os_profiles import profile


def _preflight_profile(profiles_path, name, *, host=None):
    """Return a bounded launch report without starting a provider process.

    The report deliberately separates metadata validation, host compatibility
    and local artifact checks so callers can present an actionable diagnostic.
    A ``blocked`` report is ordinary data; :func:`launch_profile` turns it into
    a ``RuntimeError`` immediately before manager registration.
    """
    host_descriptor = profile().descriptor() if host is None else host
    report = {
        "status": "blocked",
        "profile": name,
        "profile_metadata": None,
        "host": host_descriptor,
        "artifact_checks": {},
        "reason": None,
    }
    if not isinstance(host_descriptor, dict):
        report["reason"] = "host descriptor must be an object"
        return report, None
    try:
        descriptors = {item.name: item for item in read_profiles(profiles_path)}
    except (OSError, TypeError, ValueError) as exc:
        report["reason"] = f"profile manifest invalid: {exc}"
        return report, None
    if name not in descriptors:
        raise KeyError(f"unknown runtime profile: {name}")
    descriptor = descriptors[name]
    report["profile_metadata"] = asdict(descriptor)
    try:
        descriptor.validate_for(host_descriptor)
    except (RuntimeError, ValueError, TypeError) as exc:
        report["reason"] = f"host compatibility failed: {exc}"
        return report, descriptor
    # A declared lock/artifact is part of launch identity. Metadata-only
    # profiles remain explicitly unmeasured, but a declared file must exist
    # and match before a child process is started.
    try:
        checks = descriptor.verify_artifacts(Path(profiles_path).parent)
    except (OSError, ValueError) as exc:
        # ``verify_artifacts`` raises after detecting a digest mismatch. Keep
        # that distinction visible even though the legacy helper has no report
        # return path for an exception.
        message = str(exc)
        field = next((item for item in ("lockfile", "artifact") if item in message), None)
        if field is not None and "sha256 mismatch" in message:
            report["artifact_checks"][field] = "mismatch"
        report["reason"] = f"artifact verification failed: {message}"
        return report, descriptor
    report["artifact_checks"] = checks
    for field, status in checks.items():
        if status in ("missing", "mismatch"):
            report["reason"] = f"{field} verification failed: {status}"
            return report, descriptor
    report["status"] = "ready"
    return report, descriptor


def preflight_profile(profiles_path, name, *, host=None):
    """Return a bounded launch report without starting a provider process."""
    report, _descriptor = _preflight_profile(profiles_path, name, host=host)
    return report


async def launch_profile(manager: RuntimeManager, profiles_path, name, argv, *,
                         resources=None, host_controller=None, host=None):
    """Launch a provider only after the shared preflight report is ready."""
    report, descriptor = _preflight_profile(profiles_path, name, host=host)
    if report["status"] != "ready":
        raise RuntimeError(f"{name} launch preflight blocked: {report['reason']}")
    runtime = ProcessRuntime(tuple(argv), descriptor)
    await manager.register_process(
        name,
        runtime,
        resources=resources,
        host_controller=host_controller,
    )
    return runtime
