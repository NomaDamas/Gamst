"""Bounded temporal event packets and cross-modal alignment utilities.

This module is dependency-free. It keeps capture provenance separate from
processing latency and never upgrades a relation candidate into a fact.
"""
from __future__ import annotations

import json
import hashlib
import hmac
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

MAX_PACKET_BYTES = 131072
MAX_EVIDENCE = 64
MAX_SAMPLES = 128
MAX_RELATION_TYPE = 64
CHECKSUM_ALGORITHM = "sha256"
CHECKSUM_VERSION = 1
INTEGRITY_COUNTER_LIMIT = 2**31 - 1


class PacketIntegrityCounter:
    """Bounded receiver-side count of rejected packet integrity checks.

    The counter is operational metadata only. It never enters a temporal event,
    relation, or evidence list.
    """

    __slots__ = ("limit", "rejected")

    def __init__(self, *, limit: int = INTEGRITY_COUNTER_LIMIT):
        if type(limit) is not int or not 1 <= limit <= INTEGRITY_COUNTER_LIMIT:
            raise ValueError("integrity counter limit must be a positive bounded integer")
        self.limit = limit
        self.rejected = 0

    def record_rejection(self) -> int:
        self.rejected = min(self.limit, self.rejected + 1)
        return self.rejected

    @property
    def integrity_rejected(self) -> int:
        return self.rejected

    def snapshot(self) -> dict[str, int]:
        return {"integrity_rejected": self.rejected}


def _integrity_rejection(counter: Any) -> None:
    """Increment an optional bounded counter without making it packet data."""
    if counter is None:
        return
    record = getattr(counter, "record_rejection", None)
    if callable(record):
        record()
        return
    if isinstance(counter, dict):
        current = counter.get("integrity_rejected", 0)
        if type(current) is not int or current < 0:
            raise ValueError("integrity counter must contain a nonnegative integer")
        counter["integrity_rejected"] = min(INTEGRITY_COUNTER_LIMIT, current + 1)
        return
    raise TypeError("counter must be PacketIntegrityCounter or a mutable counter mapping")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode a packet body using the cross-runtime canonical JSON form."""
    if not isinstance(value, Mapping):
        raise ValueError("temporal packet must be an object")
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("temporal packet is not canonical JSON") from exc


def canonical_packet_bytes(packet: Mapping[str, Any]) -> bytes:
    """Return canonical bytes covered by a temporal packet checksum.

    The checksum envelope is deliberately excluded from the covered body so a
    receiver can recompute it. The caller's mapping is never modified.
    """
    if not isinstance(packet, Mapping):
        raise ValueError("temporal packet must be an object")
    body = dict(packet)
    body.pop("checksum", None)
    return _canonical_json(body)


def packet_sha256(packet: Mapping[str, Any]) -> str:
    """Compute the lowercase SHA-256 digest of canonical packet bytes."""
    return hashlib.sha256(canonical_packet_bytes(packet)).hexdigest()


def packet_checksum(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Build the versioned checksum envelope for a packet body."""
    return {"algorithm": CHECKSUM_ALGORITHM, "version": CHECKSUM_VERSION,
            "sha256": packet_sha256(packet)}


def verify_packet_checksum(packet: Mapping[str, Any], *, counter: Any = None,
                           allow_legacy: bool = False) -> bool:
    """Verify a packet before dispatch.

    Current packets require the exact SHA-256/version envelope. Packets from
    the pre-checksum contract may be accepted only with ``allow_legacy=True``;
    that path returns ``False`` to make its unverified status explicit.
    Any malformed, unsupported, or mismatched checksum increments ``counter``
    (when supplied) and raises ``ValueError``.
    """
    try:
        if not isinstance(packet, Mapping):
            raise ValueError("temporal packet must be an object")
        checksum = packet.get("checksum")
        if checksum is None:
            if allow_legacy:
                return False
            raise ValueError("temporal packet checksum is required")
        if (not isinstance(checksum, dict)
                or set(checksum) != {"algorithm", "version", "sha256"}
                or checksum.get("algorithm") != CHECKSUM_ALGORITHM
                or checksum.get("version") != CHECKSUM_VERSION
                or not isinstance(checksum.get("sha256"), str)
                or len(checksum["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in checksum["sha256"])):
            raise ValueError("unsupported temporal packet checksum")
        actual = packet_sha256(packet)
        if not hmac.compare_digest(actual, checksum["sha256"]):
            raise ValueError("temporal packet checksum mismatch")
        return True
    except (TypeError, ValueError):
        _integrity_rejection(counter)
        raise


# Explicit integrity spelling for receivers that do not use "checksum" in
# their local terminology.
verify_packet_integrity = verify_packet_checksum


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _bounded_text(value: Any, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string of at most {limit} characters")
    return value


def _relation_shape(value: Any) -> None:
    """Validate the bounded, provenance-carrying relation wire shape."""
    if not isinstance(value, dict):
        raise ValueError("temporal relation must be an object")
    if value.get("schema_version") != "gamst.relation.v1":
        raise ValueError("unsupported temporal relation schema")
    _bounded_text(value.get("relation"), "relation", MAX_RELATION_TYPE)
    if value.get("status") not in ("candidate", "accepted", "rejected"):
        raise ValueError("invalid temporal relation status")
    ids = value.get("event_ids")
    if (not isinstance(ids, list) or len(ids) != 2 or ids[0] == ids[1]
            or any(not isinstance(x, str) or not x for x in ids)):
        raise ValueError("temporal relations require two event IDs")
    overlap = value.get("overlap")
    if type(overlap) not in (int, float) or not math.isfinite(overlap) or not 0 <= overlap <= 1:
        raise ValueError("temporal relation overlap must be in 0..1")
    evidence = value.get("evidence_ids", [])
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE or any(not isinstance(x, str) for x in evidence):
        raise ValueError("temporal relation evidence is bounded")
    provenance = value.get("provenance")
    if not isinstance(provenance, list) or len(provenance) != 2:
        raise ValueError("temporal relation provenance must contain both events")
    for item in provenance:
        if not isinstance(item, dict) or item.get("event_id") not in ids:
            raise ValueError("temporal relation provenance does not match event IDs")
        for key in ("source_clock", "clock_id", "clock_epoch"):
            _bounded_text(item.get(key), key)
        if not isinstance(item.get("evidence_ids"), list) or len(item["evidence_ids"]) > MAX_EVIDENCE:
            raise ValueError("temporal relation provenance evidence is bounded")
    _json_safe(value.get("clock_calibration", {}))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in list(value.items())[:32]}
    if isinstance(value, list):
        return [_json_safe(v) for v in value[:64]]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("value contains non-JSON or non-finite data")


@dataclass(frozen=True)
class TemporalEvent:
    event_id: str
    stream_id: str
    subject_id: str
    modality: str
    event_type: str
    source_clock: str
    clock_id: str
    clock_epoch: str
    capture_timestamp_ms: int
    start_ms: int
    end_ms: int
    sequence: int
    provider: str
    model_revision: str
    value: Any
    monotonic_timestamp_ms: int | None = None
    receive_timestamp_ms: int | None = None
    processing_start_ms: int | None = None
    processing_end_ms: int | None = None
    gap: bool = False
    revision: int = 0
    supersedes: str | None = None
    uncertainty: Any = None
    freshness_ms: int = 5000
    evidence_ids: tuple[str, ...] = ()
    source_spans: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        for name in ("event_id", "stream_id", "subject_id", "modality", "event_type", "source_clock", "clock_id", "clock_epoch", "provider", "model_revision"):
            _bounded_text(getattr(self, name), name)
        if not isinstance(self.capture_timestamp_ms, int) or not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int):
            raise ValueError("temporal timestamps must be integers")
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must not exceed end_ms")
        if type(self.sequence) is not int or self.sequence < 0 or type(self.revision) is not int or self.revision < 0:
            raise ValueError("sequence and revision must be nonnegative integers")
        if type(self.freshness_ms) is not int or not 0 <= self.freshness_ms <= 86400000:
            raise ValueError("freshness_ms must be bounded")
        if len(self.evidence_ids) > MAX_EVIDENCE or any(not isinstance(x, str) for x in self.evidence_ids):
            raise ValueError("evidence_ids are bounded strings")
        _json_safe(self.value)
        _json_safe(self.uncertainty)
        _json_safe(list(self.source_spans))
        for name in ("monotonic_timestamp_ms", "receive_timestamp_ms", "processing_start_ms", "processing_end_ms"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be an integer or null")

    def as_dict(self, *, now_ms: int | None = None) -> dict[str, Any]:
        now = int(time.time() * 1000) if now_ms is None else now_ms
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        result["source_spans"] = list(self.source_spans)
        result["age_ms"] = max(0, now - self.capture_timestamp_ms)
        result["fresh"] = result["age_ms"] <= self.freshness_ms
        result["evidence_state"] = "current" if result["fresh"] else "last_seen"
        return result


@dataclass(frozen=True)
class TemporalPacket:
    schema_version: str
    query_cutoff_ms: int
    events: tuple[TemporalEvent, ...]
    relations: tuple[dict[str, Any], ...] = ()
    clock: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != "gamst.temporal.v1":
            raise ValueError("unsupported temporal schema")
        if type(self.query_cutoff_ms) is not int:
            raise ValueError("query_cutoff_ms must be an integer")
        if len(self.events) > MAX_EVIDENCE or len(self.relations) > MAX_EVIDENCE:
            raise ValueError("temporal packet is too large")
        _json_safe(self.clock)
        _json_safe(list(self.relations))
        for relation in self.relations:
            _relation_shape(relation)
        event_ids = {event.event_id for event in self.events}
        if any(set(relation["event_ids"]) - event_ids for relation in self.relations):
            raise ValueError("relation references an event outside the packet")
        if any(event.capture_timestamp_ms > self.query_cutoff_ms for event in self.events):
            raise ValueError("events after query cutoff are not eligible")

    def _body_dict(self) -> dict[str, Any]:
        """Return the packet body covered by the checksum."""
        return {"schema_version": self.schema_version, "query_cutoff_ms": self.query_cutoff_ms,
                "clock": self.clock, "events": [event.as_dict(now_ms=self.query_cutoff_ms) for event in self.events],
                "relations": list(self.relations)}

    def canonical_bytes(self) -> bytes:
        """Return deterministic UTF-8 bytes covered by the packet checksum."""
        encoded = canonical_packet_bytes(self._body_dict())
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("temporal packet exceeds byte budget")
        return encoded

    def checksum(self) -> dict[str, Any]:
        """Return the versioned checksum for this packet body."""
        return packet_checksum(self._body_dict())

    def as_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        """Render the bounded packet, including integrity by default."""
        result = self._body_dict()
        if include_checksum:
            result["checksum"] = self.checksum()
        encoded = _canonical_json(result)
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("temporal packet exceeds byte budget")
        return result

    def to_bytes(self) -> bytes:
        """Serialize a complete packet in canonical JSON form."""
        encoded = _canonical_json(self.as_dict())
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("temporal packet exceeds byte budget")
        return encoded

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, counter: Any = None,
                  allow_legacy: bool = False) -> "TemporalPacket":
        """Verify and parse a wire packet before it reaches a consumer.

        ``allow_legacy`` is an explicit compatibility switch for packets from
        the pre-B4 contract that have no checksum. Such packets are parsed but
        remain unverified until re-serialized with the current contract.
        """
        verify_packet_checksum(value, counter=counter, allow_legacy=allow_legacy)
        if not isinstance(value, Mapping):
            raise ValueError("temporal packet must be an object")
        keys = set(value)
        expected = {"schema_version", "query_cutoff_ms", "clock", "events", "relations", "checksum"}
        if "checksum" not in value:
            expected.remove("checksum")
        if keys != expected:
            raise ValueError("temporal packet has unknown or missing fields")
        if value.get("schema_version") != "gamst.temporal.v1":
            raise ValueError("unsupported temporal schema")
        events = []
        fields = {field.name for field in TemporalEvent.__dataclass_fields__.values()}
        for item in value.get("events", ()):
            if not isinstance(item, Mapping):
                raise ValueError("temporal packet events must be objects")
            if set(item) - fields - {"age_ms", "fresh", "evidence_state"}:
                raise ValueError("temporal event has unknown fields")
            missing = fields - set(item)
            # Derived fields are intentionally absent from the constructor,
            # while every declared event field remains explicit on the wire.
            missing -= {"monotonic_timestamp_ms", "receive_timestamp_ms", "processing_start_ms",
                        "processing_end_ms", "gap", "revision", "supersedes", "uncertainty",
                        "freshness_ms", "evidence_ids", "source_spans"}
            if missing:
                raise ValueError("temporal event is missing fields")
            kwargs = {name: item[name] for name in fields if name in item}
            kwargs["evidence_ids"] = tuple(kwargs.get("evidence_ids", ()))
            kwargs["source_spans"] = tuple(kwargs.get("source_spans", ()))
            events.append(TemporalEvent(**kwargs))
        return cls(value["schema_version"], value["query_cutoff_ms"], tuple(events),
                   tuple(value.get("relations", ())), value.get("clock", {}))

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview, *, counter: Any = None,
                   allow_legacy: bool = False) -> "TemporalPacket":
        """Decode canonical packet bytes and verify integrity before parsing."""
        try:
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            if type(raw) is not bytes or not 1 <= len(raw) <= MAX_PACKET_BYTES:
                raise ValueError("temporal packet bytes exceed bound")
            def reject_duplicates(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError("duplicate temporal packet field")
                    result[key] = item
                return result
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite packet number")))
            # ``from_dict`` owns the detailed schema checks. Suppress its
            # counter here so the byte-level receiver records one rejection per
            # packet, even when a nested schema/checksum check fails.
            packet = cls.from_dict(value, counter=None, allow_legacy=allow_legacy)
            # The wire representation itself is canonical. A semantically
            # unchanged but noncanonical spelling is rejected at this boundary.
            if raw != _canonical_json(value):
                raise ValueError("temporal packet bytes are not canonical")
            return packet
        except (UnicodeDecodeError, json.JSONDecodeError):
            _integrity_rejection(counter)
            raise ValueError("invalid temporal packet bytes")
        except (TypeError, ValueError):
            _integrity_rejection(counter)
            raise


def decode_packet(raw: bytes | bytearray | memoryview, *, counter: Any = None,
                  allow_legacy: bool = False) -> TemporalPacket:
    """Compatibility function for receivers decoding packet bytes."""
    return TemporalPacket.from_bytes(raw, counter=counter, allow_legacy=allow_legacy)


verify_packet = verify_packet_checksum


class TemporalAligner:
    def __init__(self, *, min_samples: int = 8, max_samples: int = MAX_SAMPLES, freshness_ms: int = 5000):
        if type(min_samples) is not int or min_samples < 2 or type(max_samples) is not int or max_samples < min_samples:
            raise ValueError("invalid temporal sample bounds")
        self.min_samples, self.max_samples, self.freshness_ms = min_samples, max_samples, freshness_ms
        self._clock_samples: dict[str, list[tuple[int, int]]] = {}
        self._revisions: dict[str, TemporalEvent] = {}

    def add_clock_sample(self, source_clock: str, source_ms: int, monotonic_ms: int) -> None:
        _bounded_text(source_clock, "source_clock")
        if type(source_ms) is not int or type(monotonic_ms) is not int:
            raise ValueError("clock samples must be integer milliseconds")
        samples = self._clock_samples.setdefault(source_clock, [])
        samples.append((source_ms, monotonic_ms))
        del samples[:-self.max_samples]

    def clock_status(self, source_clock: str) -> dict[str, Any]:
        samples = self._clock_samples.get(source_clock, [])
        if len(samples) < self.min_samples:
            return {"source_clock": source_clock, "calibrated": False, "sample_count": len(samples), "offset_ms": None, "drift": None, "residual_ms": None}
        x0, y0 = samples[0]
        dx = [x - x0 for x, _ in samples]
        dy = [y - y0 for _, y in samples]
        denom = sum(v * v for v in dx)
        slope = sum(a * b for a, b in zip(dx, dy)) / denom if denom else 1.0
        offset = statistics.mean(y - slope * x for x, y in samples)
        residual = statistics.mean(abs(y - (offset + slope * x)) for x, y in samples)
        return {"source_clock": source_clock, "calibrated": True, "sample_count": len(samples), "offset_ms": offset, "drift": slope - 1.0, "residual_ms": residual}

    @staticmethod
    def overlap(left: TemporalEvent, right: TemporalEvent, *, max_gap_ms: int = 0) -> float:
        intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
        union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
        if intersection == 0 and abs(left.end_ms - right.start_ms) > max_gap_ms and abs(right.end_ms - left.start_ms) > max_gap_ms:
            return 0.0
        return intersection / union if union else 1.0

    def accept(self, event: TemporalEvent) -> dict[str, Any] | None:
        previous = self._revisions.get(event.event_id)
        if previous and (previous.clock_id, previous.clock_epoch) != (event.clock_id, event.clock_epoch):
            raise ValueError("event identity cannot cross clock epochs")
        if previous and event.revision <= previous.revision:
            return None
        if event.revision > 0 and (not previous or event.supersedes != previous.event_id):
            raise ValueError("newer revision must supersede the previous event")
        self._revisions[event.event_id] = event
        return event.as_dict()

    def eligible(self, query_cutoff_ms: int, *, now_ms: int | None = None) -> list[TemporalEvent]:
        now = int(time.time() * 1000) if now_ms is None else now_ms
        return [event for event in self._revisions.values() if event.capture_timestamp_ms <= query_cutoff_ms]

    def materialize_relation(self, left: TemporalEvent, right: TemporalEvent, relation: str, *,
                             status: str = "candidate", max_gap_ms: int = 250,
                             evidence_ids: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
        """Create a relation without dropping either event's temporal provenance.

        Relations remain explicit candidates unless a caller supplies a reviewed
        status. Clock calibration is diagnostic metadata; it never upgrades a
        relation or confidence into a fact.
        """
        if not isinstance(left, TemporalEvent) or not isinstance(right, TemporalEvent):
            raise TypeError("relations require TemporalEvent operands")
        _bounded_text(relation, "relation", MAX_RELATION_TYPE)
        if status not in ("candidate", "accepted", "rejected"):
            raise ValueError("invalid temporal relation status")
        if type(max_gap_ms) is not int or not 0 <= max_gap_ms <= 60000:
            raise ValueError("max_gap_ms must be bounded")
        combined = list(left.evidence_ids) + list(right.evidence_ids) + list(evidence_ids or ())
        if len(combined) > MAX_EVIDENCE or any(not isinstance(item, str) for item in combined):
            raise ValueError("temporal relation evidence is bounded")
        # Keep first-seen order while preventing duplicate citations.
        combined = list(dict.fromkeys(combined))
        clocks = {}
        for event in (left, right):
            clocks.setdefault(event.source_clock, self.clock_status(event.source_clock))
        provenance = [
            {"event_id": event.event_id, "stream_id": event.stream_id,
             "subject_id": event.subject_id, "modality": event.modality,
             "source_clock": event.source_clock, "clock_id": event.clock_id,
             "clock_epoch": event.clock_epoch, "capture_timestamp_ms": event.capture_timestamp_ms,
             "start_ms": event.start_ms, "end_ms": event.end_ms, "sequence": event.sequence,
             "evidence_ids": list(event.evidence_ids)}
            for event in (left, right)
        ]
        result = {"schema_version": "gamst.relation.v1", "status": status,
                  "relation": relation, "event_ids": [left.event_id, right.event_id],
                  "overlap": self.overlap(left, right, max_gap_ms=max_gap_ms),
                  "uncertainty": "unverified", "evidence_ids": combined,
                  "provenance": provenance, "clock_calibration": clocks,
                  "max_gap_ms": max_gap_ms}
        _relation_shape(result)
        return result

    def relation_candidate(self, left: TemporalEvent, right: TemporalEvent, relation: str) -> dict[str, Any]:
        return self.materialize_relation(left, right, relation)

    def packet(self, query_cutoff_ms: int, *, now_ms: int | None = None, relations: list[dict[str, Any]] | None = None) -> TemporalPacket:
        events = tuple(self.eligible(query_cutoff_ms, now_ms=now_ms))
        clocks = {name: self.clock_status(name) for name in self._clock_samples}
        packet = TemporalPacket("gamst.temporal.v1", query_cutoff_ms, events, tuple(relations or ()), clocks)
        packet.as_dict()
        return packet
