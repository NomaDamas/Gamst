"""Single-event runtime packets for Crew-supervised runtimes.

The checksum, canonical JSON form and integrity counter are GAMST's own
(:mod:`gamst.temporal`); this module only adds the one-event packet shape that
:class:`~gamst.crew.manager.RuntimeManager` queues and
:class:`~gamst.crew.process_runtime.ProcessRuntime` writes to a worker.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

from ..temporal import (
    CHECKSUM_ALGORITHM,
    CHECKSUM_VERSION,
    INTEGRITY_COUNTER_LIMIT,
    MAX_PACKET_BYTES,
    PacketIntegrityCounter,
    _integrity_rejection as _record_rejection,
    packet_checksum,
    packet_sha256,
    verify_packet_checksum,
)
from ..temporal import canonical_packet_bytes as _canonical


def canonical_packet_bytes(packet: Mapping[str, Any]) -> bytes:
    """GAMST temporal canonical JSON (``gamst.temporal``), bounded to one packet."""
    raw = _canonical(packet)
    if len(raw) > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds byte budget")
    return raw


verify_packet_integrity = verify_packet_checksum


def _text(v, name, limit=256):
    if not isinstance(v, str) or not v.strip() or len(v) > limit:
        raise ValueError(f"{name} invalid")
    return v


def _safe(v):
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in list(v.items())[:64]}
    if isinstance(v, list):
        return [_safe(x) for x in v[:128]]
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float) and math.isfinite(v):
        return v
    raise ValueError("value must be finite JSON")


@dataclass(frozen=True)
class EventPacket:
    event_id: str
    stream_id: str
    subject_id: str
    modality: str
    event_type: str
    source_clock: str
    capture_timestamp_ms: int
    receive_timestamp_ms: int
    processing_start_ms: int | None
    processing_end_ms: int | None
    start_ms: int
    end_ms: int
    sequence: int
    gap: bool
    revision: int
    provider: str
    model_revision: str
    value: Any
    uncertainty: Any = None
    freshness_ms: int = 5000
    evidence_ids: tuple[str, ...] = ()
    source_spans: tuple[dict[str, Any], ...] = ()
    query_cutoff_ms: int | None = None
    monotonic_capture_ms: int | None = None
    clock_id: str = "legacy"
    clock_epoch: str = "legacy"
    supersedes: str | None = None

    def __post_init__(self):
        for n in (
            "event_id",
            "stream_id",
            "subject_id",
            "modality",
            "event_type",
            "source_clock",
            "clock_id",
            "clock_epoch",
            "provider",
            "model_revision",
        ):
            _text(getattr(self, n), n)
        ints = (
            "capture_timestamp_ms",
            "receive_timestamp_ms",
            "start_ms",
            "end_ms",
            "sequence",
            "revision",
            "freshness_ms",
        )
        if any(type(getattr(self, n)) is not int for n in ints):
            raise ValueError("integer packet fields required")
        if (
            self.start_ms > self.end_ms
            or self.sequence < 0
            or self.revision < 0
            or not 0 <= self.freshness_ms <= 86400000
        ):
            raise ValueError("invalid packet bounds")
        if (
            self.query_cutoff_ms is not None
            and self.capture_timestamp_ms > self.query_cutoff_ms
        ):
            raise ValueError("post-query evidence")
        if len(self.evidence_ids) > 64 or len(self.source_spans) > 64:
            raise ValueError("evidence bound exceeded")
        _safe(self.value)
        _safe(self.uncertainty)
        _safe(list(self.source_spans))

    def to_dict(self, *, include_checksum=True):
        d = asdict(self)
        d["evidence_ids"] = list(self.evidence_ids)
        d["source_spans"] = list(self.source_spans)
        if include_checksum:
            d["checksum"] = packet_checksum(d)
        return d

    def json(self, *, include_checksum=True):
        d = self.to_dict(include_checksum=include_checksum)
        raw = json.dumps(
            d, ensure_ascii=False, sort_keys=True, allow_nan=False,
            separators=(",", ":")
        ).encode("utf-8")
        if len(raw) > MAX_PACKET_BYTES:
            raise ValueError("packet exceeds byte budget")
        return raw

    def to_bytes(self) -> bytes:
        """Serialize a complete packet in canonical UTF-8 JSON form."""
        return self.json()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, counter=None,
                  allow_legacy: bool = False) -> "EventPacket":
        """Verify and parse a packet before it reaches a runtime adapter."""
        verify_packet_checksum(value, counter=counter, allow_legacy=allow_legacy)
        if not isinstance(value, Mapping):
            raise ValueError("packet must be an object")
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        expected = fields | {"checksum"}
        if "checksum" not in value:
            expected.remove("checksum")
        if set(value) != expected:
            raise ValueError("packet has unknown or missing fields")
        kwargs = {name: value[name] for name in fields}
        kwargs["evidence_ids"] = tuple(kwargs["evidence_ids"])
        kwargs["source_spans"] = tuple(kwargs["source_spans"])
        return cls(**kwargs)

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview, *, counter=None,
                   allow_legacy: bool = False) -> "EventPacket":
        """Decode canonical packet bytes and verify integrity before parsing."""
        try:
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            if type(raw) is not bytes or not 1 <= len(raw) <= MAX_PACKET_BYTES:
                raise ValueError("packet bytes exceed bound")

            def reject_duplicates(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError("duplicate packet field")
                    result[key] = item
                return result

            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("nonfinite packet number")
                ),
            )
            packet = cls.from_dict(value, counter=None, allow_legacy=allow_legacy)
            # Current packets bind exact canonical bytes. Legacy packets are
            # accepted under the explicit compatibility switch and upgraded
            # when re-serialized.
            if "checksum" in value:
                canonical = json.dumps(
                    value, ensure_ascii=False, sort_keys=True,
                    allow_nan=False, separators=(",", ":")
                ).encode("utf-8")
                if raw != canonical:
                    raise ValueError("packet bytes are not canonical")
            return packet
        except (UnicodeDecodeError, json.JSONDecodeError):
            _record_rejection(counter)
            raise ValueError("invalid packet bytes")
        except (TypeError, ValueError):
            _record_rejection(counter)
            raise


decode_packet = EventPacket.from_bytes


@dataclass(frozen=True)
class RuntimeDescriptor:
    name: str
    profile: str
    provider: str
    model_revision: str
    license: str
    ready: bool = False

    def __post_init__(self):
        for value, name in (
            (self.name, "runtime name"),
            (self.profile, "runtime profile"),
            (self.provider, "provider"),
            (self.model_revision, "model revision"),
            (self.license, "license"),
        ):
            _text(value, name)
        if type(self.ready) is not bool:
            raise ValueError("ready must be boolean")


@dataclass
class HealthSnapshot:
    runtime: str
    state: str
    queue_depth: int
    capacity: int
    dropped: int
    packet_age_ms: int | None = None
    errors: list[str] = field(default_factory=list)
    descriptor: RuntimeDescriptor | None = None
    integrity_rejected: int = 0
    crash_count: int = 0
    restart_count: int = 0
