"""Crew: GAMST's setup and runtime orchestration (formerly the separate GamstCrew package).

* :mod:`~gamst.crew.registry` - roles, candidate models, packages, pinned weights, resources;
* :mod:`~gamst.crew.setup` - hardware detection, ``make_plan`` / ``apply_plan``, the profile;
* :mod:`~gamst.crew.modules` - a profile role as a :class:`gamst.pipeline.Pipeline` module;
* :mod:`~gamst.crew.manager`, :mod:`~gamst.crew.process_runtime`, :mod:`~gamst.crew.launcher` -
  supervised packet runtimes (any argv, e.g. a container command) with bounded queues,
  restarts, resource reservations and optional host limits;
* :mod:`~gamst.crew.media_bus`, :mod:`~gamst.crew.media_source`, :mod:`~gamst.crew.parallel` -
  one decoded media clock shared by many model processes;
* :mod:`~gamst.crew.os_profiles`, :mod:`~gamst.crew.diagnostics`, :mod:`~gamst.crew.resources`,
  :mod:`~gamst.crew.host_limits`, :mod:`~gamst.crew.shared_memory_transport`,
  :mod:`~gamst.crew.transport`.

Packet checksums, canonical JSON and clock alignment are GAMST's (:mod:`gamst.temporal`);
provider workers are GAMST's :class:`gamst.ProcessProvider`. docs/crew.md records what was
kept, replaced and dropped. Everything here is stdlib (numpy only for the media rings).
"""
from .packet import (
    EventPacket,
    RuntimeDescriptor,
    HealthSnapshot,
    PacketIntegrityCounter,
    canonical_packet_bytes,
    packet_checksum,
    packet_sha256,
    verify_packet_checksum,
)
from .manager import RuntimeAdapter, RuntimeManager
from .process_runtime import ProcessRuntime
from .launcher import launch_profile, preflight_profile
from .resources import (
    ResourceAllocator,
    ResourceCapacity,
    ResourceRequest,
    ResourceReservation,
    ResourceUnavailable,
)
from .host_limits import (
    HOST_LIMIT_MODES,
    HostLimitController,
    HostLimitError,
    HostLimitReport,
    HostLimitUnsupported,
    HostLimitUnavailable,
)
from .shared_memory_transport import (
    SharedPacket, make_packet_transfer, read_packet_transfer, read_shared_packet,
)
from .setup import SetupError, apply_plan, detect_hardware, download_weight, make_plan
from .modules import RoleUnavailable, WorkerModule, role_module

__all__ = [
    "EventPacket",
    "RuntimeDescriptor",
    "HealthSnapshot",
    "PacketIntegrityCounter",
    "canonical_packet_bytes",
    "packet_checksum",
    "packet_sha256",
    "verify_packet_checksum",
    "RuntimeAdapter",
    "RuntimeManager",
    "ProcessRuntime",
    "launch_profile",
    "preflight_profile",
    "ResourceAllocator",
    "ResourceCapacity",
    "ResourceRequest",
    "ResourceReservation",
    "ResourceUnavailable",
    "HOST_LIMIT_MODES",
    "HostLimitController",
    "HostLimitError",
    "HostLimitReport",
    "HostLimitUnsupported",
    "HostLimitUnavailable",
    "SharedPacket",
    "make_packet_transfer",
    "read_packet_transfer",
    "read_shared_packet",
    "SetupError",
    "apply_plan",
    "detect_hardware",
    "download_weight",
    "make_plan",
    "RoleUnavailable",
    "WorkerModule",
    "role_module",
]
