"""Bounded temporal event packets and cross-modal alignment utilities.

This module is dependency-free. It keeps capture provenance separate from
processing latency and never upgrades a relation candidate into a fact.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_PACKET_BYTES = 131072
MAX_EVIDENCE = 64
MAX_SAMPLES = 128
MAX_RELATION_TYPE = 64


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

    def as_dict(self) -> dict[str, Any]:
        result = {"schema_version": self.schema_version, "query_cutoff_ms": self.query_cutoff_ms,
                  "clock": self.clock, "events": [event.as_dict(now_ms=self.query_cutoff_ms) for event in self.events],
                  "relations": list(self.relations)}
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("temporal packet exceeds byte budget")
        return result


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
