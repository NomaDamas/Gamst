"""Bounded temporal context orchestration.

The low-level :mod:`gamst.temporal` packet contract remains deliberately
small.  This module provides the stateful application layer that accepts
events, calibrates source clocks, materializes cross-modal relation
candidates, and exposes revision-aware current/last-seen state.  A relation is
always a candidate unless an independent reviewer explicitly accepts it.
"""
from __future__ import annotations

from typing import Any, Iterable

from .temporal import TemporalAligner, TemporalEvent, TemporalPacket


class TemporalContextEngine:
    """Bounded event/relation engine for one stream session."""

    def __init__(self, *, min_clock_samples: int = 8, max_events: int = 512,
                 max_relations: int = 512, freshness_ms: int = 5000):
        if type(max_events) is not int or not 1 <= max_events <= 4096:
            raise ValueError("max_events must be 1..4096")
        if type(max_relations) is not int or not 1 <= max_relations <= 4096:
            raise ValueError("max_relations must be 1..4096")
        self.aligner = TemporalAligner(min_samples=min_clock_samples,
                                       max_samples=128,
                                       freshness_ms=freshness_ms)
        self.max_events, self.max_relations = max_events, max_relations
        self._events: dict[str, TemporalEvent] = {}
        self._relations: list[dict[str, Any]] = []

    def add_clock_sample(self, source_clock: str, source_ms: int,
                         monotonic_ms: int) -> None:
        self.aligner.add_clock_sample(source_clock, source_ms, monotonic_ms)

    def clock_status(self, source_clock: str) -> dict[str, Any]:
        return self.aligner.clock_status(source_clock)

    def _aligned_ms(self, event: TemporalEvent, value: int) -> float:
        status = self.clock_status(event.source_clock)
        if not status["calibrated"]:
            return float(value)
        return float(status["offset_ms"] + (1.0 + status["drift"]) * value)

    def _interval(self, event: TemporalEvent) -> tuple[float, float]:
        return self._aligned_ms(event, event.start_ms), self._aligned_ms(event, event.end_ms)

    def ingest(self, event: TemporalEvent) -> dict[str, Any] | None:
        """Accept an event revision and return its wire representation.

        Older revisions are ignored.  The bounded event map is evicted in
        insertion order, while the current revision for an event ID remains
        the only state visible to packet materialization.
        """
        result = self.aligner.accept(event)
        if result is None:
            return None
        self._events[event.event_id] = event
        while len(self._events) > self.max_events:
            self._events.pop(next(iter(self._events)))
        return result

    accept = ingest

    def _overlap(self, left: TemporalEvent, right: TemporalEvent,
                 max_gap_ms: int) -> float:
        ls, le = self._interval(left)
        rs, re = self._interval(right)
        intersection = max(0.0, min(le, re) - max(ls, rs))
        union = max(le, re) - min(ls, rs)
        if intersection == 0 and abs(le - rs) > max_gap_ms and abs(re - ls) > max_gap_ms:
            return 0.0
        return intersection / union if union else 1.0

    def relation(self, left: TemporalEvent, right: TemporalEvent, relation: str,
                 *, status: str = "candidate", max_gap_ms: int = 250,
                 evidence_ids: Iterable[str] = ()) -> dict[str, Any]:
        """Materialize a calibrated, provenance-preserving relation candidate."""
        value = self.aligner.materialize_relation(left, right, relation,
                                                   status=status,
                                                   max_gap_ms=max_gap_ms,
                                                   evidence_ids=tuple(evidence_ids))
        value["overlap"] = self._overlap(left, right, max_gap_ms)
        # Preserve the base schema validation after replacing raw overlap.
        from .temporal import _relation_shape
        _relation_shape(value)
        if value not in self._relations:
            self._relations.append(value)
            if len(self._relations) > self.max_relations:
                del self._relations[:-self.max_relations]
        return value

    materialize_relation = relation

    def association_candidate(self, left: TemporalEvent, right: TemporalEvent,
                              *, max_gap_ms: int = 250) -> dict[str, Any]:
        """Create an object/sound or subject association candidate."""
        return self.relation(left, right, "association", max_gap_ms=max_gap_ms)

    def causal_candidate(self, left: TemporalEvent, right: TemporalEvent,
                         *, max_gap_ms: int = 250) -> dict[str, Any]:
        """Create an ordered temporal causal candidate; it is never a fact."""
        if self._interval(left)[0] > self._interval(right)[0]:
            left, right = right, left
        return self.relation(left, right, "causal_candidate", max_gap_ms=max_gap_ms)

    def relation_candidates(self, *, query_cutoff_ms: int,
                            modalities: tuple[str, ...] | None = None,
                            max_gap_ms: int = 250) -> list[dict[str, Any]]:
        events = [event for event in self._events.values()
                  if event.capture_timestamp_ms <= query_cutoff_ms]
        if modalities is not None:
            events = [event for event in events if event.modality in modalities]
        result: list[dict[str, Any]] = []
        for index, left in enumerate(events):
            for right in events[index + 1:]:
                if left.modality == right.modality:
                    continue
                overlap = self._overlap(left, right, max_gap_ms)
                ls, le = self._interval(left)
                rs, re = self._interval(right)
                near = min(abs(le - rs), abs(re - ls)) <= max_gap_ms
                if overlap <= 0 and not near:
                    continue
                relation = "temporal_overlap" if overlap > 0 else "temporal_proximity"
                result.append(self.relation(left, right, relation,
                                             max_gap_ms=max_gap_ms))
                if len(result) >= self.max_relations:
                    return result
        return result

    def current_state(self, *, query_cutoff_ms: int, now_ms: int | None = None) -> list[dict[str, Any]]:
        """Return current and last-seen event state without losing revisions."""
        now = query_cutoff_ms if now_ms is None else now_ms
        result = []
        for event in self._events.values():
            if event.capture_timestamp_ms > query_cutoff_ms:
                continue
            item = event.as_dict(now_ms=now)
            item["state"] = "current" if item["fresh"] else "last_seen"
            result.append(item)
        return result

    def packet(self, query_cutoff_ms: int) -> TemporalPacket:
        events = tuple(event for event in self._events.values()
                       if event.capture_timestamp_ms <= query_cutoff_ms)
        ids = {event.event_id for event in events}
        relations = tuple(relation for relation in self._relations
                           if set(relation["event_ids"]) <= ids)
        return TemporalPacket("gamst.temporal.v1", query_cutoff_ms, events,
                              relations, {name: self.clock_status(name)
                                          for name in self.aligner._clock_samples})

