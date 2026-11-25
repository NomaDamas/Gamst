"""Bounded runtime resource reservations.

The allocator accounts for resources declared by a runtime profile.  It does
not probe or claim ownership of host resources; deployment code supplies the
capacity after its own host check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import threading
import time


def _cpu(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _gpu(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ResourceVector:
    """A bounded amount of CPU, memory, GPU and GPU memory."""

    cpu_cores: float = 0.0
    ram_bytes: int = 0
    gpu_count: int = 0
    vram_bytes: int = 0

    def __post_init__(self):
        object.__setattr__(self, "cpu_cores", _cpu(self.cpu_cores, "cpu_cores"))
        object.__setattr__(self, "ram_bytes", _bytes(self.ram_bytes, "ram_bytes"))
        object.__setattr__(self, "gpu_count", _gpu(self.gpu_count, "gpu_count"))
        object.__setattr__(self, "vram_bytes", _bytes(self.vram_bytes, "vram_bytes"))

    def add(self, other: "ResourceVector") -> "ResourceVector":
        if not isinstance(other, ResourceVector):
            raise TypeError("resource vector required")
        return ResourceVector(
            self.cpu_cores + other.cpu_cores,
            self.ram_bytes + other.ram_bytes,
            self.gpu_count + other.gpu_count,
            self.vram_bytes + other.vram_bytes,
        )

    def subtract(self, other: "ResourceVector") -> "ResourceVector":
        if not isinstance(other, ResourceVector):
            raise TypeError("resource vector required")
        values = (
            self.cpu_cores - other.cpu_cores,
            self.ram_bytes - other.ram_bytes,
            self.gpu_count - other.gpu_count,
            self.vram_bytes - other.vram_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("resource vector cannot become negative")
        return ResourceVector(*values)


class ResourceCapacity(ResourceVector):
    """Host capacity supplied by deployment discovery."""


class ResourceRequest(ResourceVector):
    """Resources reserved by one runtime."""


@dataclass(frozen=True)
class ResourceReservation:
    reservation_id: str
    owner: str
    request: ResourceRequest
    created_monotonic_ns: int
    _nonce: object = field(repr=False, compare=False)


class ResourceUnavailable(RuntimeError):
    """Raised when a request would overcommit the declared capacity."""

    def __init__(self, request, available):
        self.request = request
        self.available = available
        super().__init__("runtime resource request exceeds available capacity")


class ResourceAllocator:
    """Thread-safe, bounded reservation ledger.

    Reservations are all-or-nothing.  A reservation object is an opaque release
    token: releasing a clone or releasing the same token twice is rejected.
    """

    def __init__(self, capacity: ResourceCapacity, *, max_reservations=1024):
        if not isinstance(capacity, ResourceCapacity):
            raise TypeError("ResourceCapacity required")
        if type(max_reservations) is not int or not 1 <= max_reservations <= 65536:
            raise ValueError("max_reservations must be an integer from 1 to 65536")
        self.capacity = capacity
        self.max_reservations = max_reservations
        self._reserved = ResourceVector()
        self._active = {}
        self._sequence = 0
        self._attempts = 0
        self._accepted = 0
        self._overcommit_rejections = 0
        self._capacity_rejections = 0
        self._releases = 0
        self._lock = threading.RLock()

    @staticmethod
    def _owner(owner):
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 256:
            raise ValueError("reservation owner must be a bounded non-empty string")
        return owner

    def reserve(self, request: ResourceRequest, *, owner: str) -> ResourceReservation:
        if not isinstance(request, ResourceRequest):
            raise TypeError("ResourceRequest required")
        owner = self._owner(owner)
        with self._lock:
            self._attempts += 1
            available = self.capacity.subtract(self._reserved)
            if len(self._active) >= self.max_reservations:
                self._capacity_rejections += 1
                raise ResourceUnavailable(request, available)
            if any(
                requested > free
                for requested, free in zip(
                    (request.cpu_cores, request.ram_bytes, request.gpu_count, request.vram_bytes),
                    (available.cpu_cores, available.ram_bytes, available.gpu_count, available.vram_bytes),
                )
            ):
                self._overcommit_rejections += 1
                raise ResourceUnavailable(request, available)
            self._sequence += 1
            reservation = ResourceReservation(
                f"reservation-{self._sequence}", owner, request,
                time.monotonic_ns(), object(),
            )
            self._active[reservation.reservation_id] = reservation
            self._reserved = self._reserved.add(request)
            self._accepted += 1
            return reservation

    def release(self, reservation: ResourceReservation):
        if not isinstance(reservation, ResourceReservation):
            raise TypeError("ResourceReservation required")
        with self._lock:
            active = self._active.get(reservation.reservation_id)
            if active is None:
                raise ValueError("reservation is already released or unknown")
            if active is not reservation:
                raise ValueError("exact reservation token required")
            del self._active[reservation.reservation_id]
            self._reserved = self._reserved.subtract(reservation.request)
            self._releases += 1

    def telemetry(self):
        """Return bounded JSON-safe capacity, usage and reservation counters."""
        with self._lock:
            available = self.capacity.subtract(self._reserved)
            return {
                "capacity": asdict(self.capacity),
                "reserved": asdict(self._reserved),
                "available": asdict(available),
                "active_reservations": len(self._active),
                "max_reservations": self.max_reservations,
                "counters": {
                    "attempts": self._attempts,
                    "accepted": self._accepted,
                    "overcommit_rejections": self._overcommit_rejections,
                    "capacity_rejections": self._capacity_rejections,
                    "releases": self._releases,
                },
                "reservations": [
                    {
                        "reservation_id": item.reservation_id,
                        "owner": item.owner,
                        "request": asdict(item.request),
                    }
                    for item in self._active.values()
                ],
            }
