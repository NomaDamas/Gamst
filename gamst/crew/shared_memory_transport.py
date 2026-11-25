"""Optional bounded local shared-memory transfer of canonical EventPackets.

The caller owns the segment and must close/unlink it after acknowledgement.
This is a portable CPU contract; GPU shared memory requires a separate host
backend and acceptance run.  The socket fallback preserves identical bytes.
"""
from __future__ import annotations

import hashlib
import hmac
import base64
from multiprocessing import shared_memory

from .packet import EventPacket, MAX_PACKET_BYTES


SHARED_MEMORY_PREFIX = "gamst_crew_"


def _canonical_packet(packet: EventPacket | bytes) -> bytes:
    raw = packet.to_bytes() if isinstance(packet, EventPacket) else bytes(packet)
    if not 1 <= len(raw) <= MAX_PACKET_BYTES:
        raise ValueError("shared packet exceeds byte budget")
    # Strict canonical/checksum verification happens before allocating a
    # segment. The receiving process repeats it after reading the bytes.
    EventPacket.from_bytes(raw)
    return raw


class SharedPacket:
    """Owned single-packet segment with explicit descriptor and cleanup."""

    def __init__(self, segment: shared_memory.SharedMemory, length: int, digest: str):
        self._segment = segment
        self.length = length
        self.digest = digest
        self._closed = False

    @classmethod
    def create(cls, packet: EventPacket | bytes) -> "SharedPacket":
        raw = _canonical_packet(packet)
        segment = shared_memory.SharedMemory(create=True, size=len(raw), name=None)
        try:
            segment.buf[:len(raw)] = raw
            return cls(segment, len(raw), hashlib.sha256(raw).hexdigest())
        except BaseException:
            segment.close()
            segment.unlink()
            raise

    def descriptor(self) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("shared packet is closed")
        return {"transport": "shared_memory", "version": 1,
                "name": self._segment.name, "length": self.length,
                "sha256": self.digest}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._segment.close()
        finally:
            self._segment.unlink()

    def __enter__(self) -> "SharedPacket":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_shared_packet(descriptor: dict[str, object]) -> EventPacket:
    """Read exactly one bounded frame and re-verify wire and packet digests."""
    if not isinstance(descriptor, dict) or set(descriptor) != {"transport", "version", "name", "length", "sha256"}:
        raise ValueError("invalid shared packet descriptor")
    name, length, digest = descriptor["name"], descriptor["length"], descriptor["sha256"]
    if descriptor["transport"] != "shared_memory" or descriptor["version"] != 1:
        raise ValueError("unsupported shared packet transport")
    if not isinstance(name, str) or not name or len(name) > 256:
        raise ValueError("invalid shared packet name")
    if type(length) is not int or not 1 <= length <= MAX_PACKET_BYTES:
        raise ValueError("shared packet exceeds byte budget")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid shared packet digest")
    segment = shared_memory.SharedMemory(name=name, create=False)
    try:
        if segment.size < length:
            raise ValueError("shared packet length mismatch")
        raw = bytes(segment.buf[:length])
    finally:
        segment.close()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), digest):
        raise ValueError("shared packet digest mismatch")
    return EventPacket.from_bytes(raw)


def make_packet_transfer(packet: EventPacket | bytes, *, prefer_shared_memory: bool = True):
    """Return ``(descriptor, owner)``; fall back to identical socket bytes.

    The caller sends the descriptor through its control plane, then releases
    ``owner`` only after the receiver acknowledges or a timeout occurs.
    ``owner`` is ``None`` for the socket fallback.
    """
    raw = _canonical_packet(packet)
    if prefer_shared_memory:
        try:
            owner = SharedPacket.create(raw)
            return owner.descriptor(), owner
        except (OSError, BufferError):
            pass
    return {"transport": "socket", "version": 1,
            "packet_b64": base64.b64encode(raw).decode("ascii")}, None


def read_packet_transfer(descriptor: dict[str, object]) -> EventPacket:
    if isinstance(descriptor, dict) and descriptor.get("transport") == "shared_memory":
        return read_shared_packet(descriptor)
    if not isinstance(descriptor, dict) or set(descriptor) != {"transport", "version", "packet_b64"}:
        raise ValueError("invalid packet transfer descriptor")
    if descriptor["transport"] != "socket" or descriptor["version"] != 1:
        raise ValueError("unsupported packet transfer")
    encoded = descriptor["packet_b64"]
    if not isinstance(encoded, str) or len(encoded) > MAX_PACKET_BYTES * 2:
        raise ValueError("invalid socket packet encoding")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeError):
        raise ValueError("invalid socket packet encoding")
    return EventPacket.from_bytes(_canonical_packet(raw))
