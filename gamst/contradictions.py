"""Temporal, cross-modal contradiction candidates with explicit abstention.

The detector compares already-normalized :class:`~gamst.temporal.TemporalEvent`
objects.  It deliberately reports *candidates*: disagreement between two
measurements is not evidence of a true world contradiction, and no confidence
or deception judgment is inferred here.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from .temporal import MAX_EVIDENCE, TemporalAligner, TemporalEvent, TemporalPacket

MAX_CONTRADICTIONS = 32
MAX_ABSTENTIONS = 64


def _canonical(value: Any) -> str:
    """Use type-preserving JSON for deterministic value comparison."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _same_value(left: Any, right: Any) -> bool:
    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError):
        # TemporalEvent already validates JSON-safe values.  Keep a defensive
        # fallback for custom event-like objects supplied to the public helper.
        return type(left) is type(right) and left == right


def _quality_uncertainty(event: TemporalEvent) -> list[str]:
    value = event.uncertainty
    if value is None:
        return []
    if isinstance(value, dict):
        status = value.get("status")
        if status in ("insufficient", "unavailable"):
            return ["insufficient_quality"]
        flags = value.get("flags", [])
        if isinstance(flags, list) and flags:
            return ["quality_flags"]
    if isinstance(value, str) and value.lower() in ("insufficient", "unavailable", "uncertain"):
        return ["insufficient_quality"]
    return []


def _freshness_reason(event: TemporalEvent, now_ms: int) -> str | None:
    if now_ms < event.capture_timestamp_ms:
        return "future_evidence"
    if now_ms - event.capture_timestamp_ms > event.freshness_ms:
        return "stale_evidence"
    return None


def _event_key(event: TemporalEvent) -> tuple[str, str]:
    # A stream can publish the same semantic observation under multiple source
    # IDs; event_type is the declared semantic key and subject scopes it.
    return event.subject_id, event.event_type


def _dedupe_revisions(events: Iterable[TemporalEvent]) -> tuple[list[TemporalEvent], list[dict[str, Any]]]:
    """Keep the newest revision and record malformed duplicate identity input.

    TemporalAligner.accept normally enforces this invariant.  The helper also
    accepts a packet/event iterable directly, so duplicate IDs are handled
    deterministically without silently replacing a newer revision.
    """
    current: dict[str, TemporalEvent] = {}
    abstentions: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, TemporalEvent):
            abstentions.append({"status": "abstained", "reason": "invalid_event"})
            continue
        previous = current.get(event.event_id)
        if previous is None:
            current[event.event_id] = event
            continue
        if (previous.clock_id, previous.clock_epoch) != (event.clock_id, event.clock_epoch):
            abstentions.append({"status": "abstained", "reason": "clock_epoch_mismatch",
                                "event_ids": [previous.event_id, event.event_id],
                                "evidence_ids": list(dict.fromkeys(previous.evidence_ids + event.evidence_ids))})
            continue
        if event.revision > previous.revision:
            if event.supersedes != previous.event_id:
                abstentions.append({"status": "abstained", "reason": "revision_lineage_missing",
                                    "event_ids": [previous.event_id, event.event_id],
                                    "evidence_ids": list(dict.fromkeys(previous.evidence_ids + event.evidence_ids))})
                continue
            current[event.event_id] = event
        elif event.revision == previous.revision and not _same_value(event.value, previous.value):
            abstentions.append({"status": "abstained", "reason": "same_revision_disagreement",
                                "event_ids": [previous.event_id, event.event_id],
                                "evidence_ids": list(dict.fromkeys(previous.evidence_ids + event.evidence_ids))})
    return list(current.values()), abstentions


def _abstention(left: TemporalEvent, right: TemporalEvent, reason: str,
                *, aligner: TemporalAligner, overlap: float | None = None) -> dict[str, Any]:
    evidence = list(dict.fromkeys(left.evidence_ids + right.evidence_ids))
    # Both event IDs are included even for abstentions, making the reason
    # inspectable without fabricating a contradiction candidate.
    return {
        "status": "abstained", "reason": reason,
        "event_ids": [left.event_id, right.event_id],
        "evidence_ids": evidence[:MAX_EVIDENCE],
        "intervals": [{"event_id": event.event_id, "start_ms": event.start_ms,
                        "end_ms": event.end_ms} for event in (left, right)],
        "source_clocks": [{"event_id": event.event_id, "source_clock": event.source_clock,
                            "clock_id": event.clock_id, "clock_epoch": event.clock_epoch}
                           for event in (left, right)],
        "overlap": aligner.overlap(left, right) if overlap is None else overlap,
        "clock_calibration": {name: aligner.clock_status(name)
                              for name in dict.fromkeys((left.source_clock, right.source_clock))},
        "uncertainty": {"status": "unresolved", "reasons": [reason]},
        "verification_status": "not_checked",
    }


def contradiction_candidate(left: TemporalEvent, right: TemporalEvent, *,
                             aligner: TemporalAligner | None = None,
                             now_ms: int | None = None, max_gap_ms: int = 250,
                             require_calibrated: bool = False) -> dict[str, Any] | None:
    """Compare one cross-modal pair and return a candidate or ``None``.

    ``None`` means the pair is compatible or not comparable by semantic key.
    A returned ``status=abstained`` object means the pair *could* disagree but
    temporal/quality/provenance evidence is unresolved.  A ``status=candidate``
    object is still unverified and must not be promoted to fact.
    """
    if not isinstance(left, TemporalEvent) or not isinstance(right, TemporalEvent):
        raise TypeError("contradiction operands must be TemporalEvent instances")
    if type(max_gap_ms) is not int or not 0 <= max_gap_ms <= 60000:
        raise ValueError("max_gap_ms must be bounded")
    aligner = aligner or TemporalAligner()
    if left.subject_id != right.subject_id or left.event_type != right.event_type:
        return None
    if left.modality == right.modality:
        return None
    if left.modality == "unknown" or right.modality == "unknown":
        return _abstention(left, right, "modality_unknown", aligner=aligner)
    overlap = aligner.overlap(left, right, max_gap_ms=max_gap_ms)
    if overlap <= 0:
        return _abstention(left, right, "temporal_misalignment", aligner=aligner, overlap=overlap)
    if now_ms is None:
        now_ms = max(left.capture_timestamp_ms, right.capture_timestamp_ms)
    stale = _freshness_reason(left, now_ms) or _freshness_reason(right, now_ms)
    if stale:
        return _abstention(left, right, stale, aligner=aligner, overlap=overlap)
    if left.gap or right.gap:
        return _abstention(left, right, "capture_gap", aligner=aligner, overlap=overlap)
    quality_reasons = _quality_uncertainty(left) + _quality_uncertainty(right)
    if quality_reasons:
        return _abstention(left, right, quality_reasons[0], aligner=aligner, overlap=overlap)
    if (left.clock_epoch != right.clock_epoch and
            left.source_clock == right.source_clock):
        return _abstention(left, right, "clock_epoch_mismatch", aligner=aligner, overlap=overlap)
    clocks = {name: aligner.clock_status(name)
              for name in dict.fromkeys((left.source_clock, right.source_clock))}
    if require_calibrated and any(not status["calibrated"] for status in clocks.values()):
        return _abstention(left, right, "clock_uncalibrated", aligner=aligner, overlap=overlap)
    if _same_value(left.value, right.value):
        return None
    result = aligner.materialize_relation(left, right, "contradiction", max_gap_ms=max_gap_ms)
    result.update({
        "contradiction_status": "candidate",
        "verification_status": "not_checked",
        "deception_assessed": False,
        "left_value": left.value,
        "right_value": right.value,
        "intervals": [{"event_id": event.event_id, "start_ms": event.start_ms,
                        "end_ms": event.end_ms} for event in (left, right)],
        "uncertainty": {"status": "unverified",
                        "reasons": (["clock_uncalibrated"]
                                    if any(not status["calibrated"] for status in clocks.values())
                                    else []),
                        "clock_calibration": clocks,
                        "event_uncertainty": {"left": left.uncertainty,
                                              "right": right.uncertainty}},
    })
    return result


def detect_contradictions(events: Iterable[TemporalEvent], *,
                          query_cutoff_ms: int | None = None,
                          now_ms: int | None = None,
                          aligner: TemporalAligner | None = None,
                          max_gap_ms: int = 250,
                          require_calibrated: bool = False,
                          max_results: int = MAX_CONTRADICTIONS) -> dict[str, Any]:
    """Detect bounded cross-modal disagreement with explicit abstentions.

    Events after ``query_cutoff_ms`` are excluded before pairing.  The returned
    packet has no inferred truth: candidates carry ``verification_status`` and
    unresolved pairs are listed under ``abstentions`` for inspection/routing.
    """
    if isinstance(events, TemporalPacket):
        if query_cutoff_ms is None:
            query_cutoff_ms = events.query_cutoff_ms
        events = events.events
    if query_cutoff_ms is not None and type(query_cutoff_ms) is not int:
        raise ValueError("query_cutoff_ms must be an integer")
    if type(max_results) is not int or not 1 <= max_results <= MAX_CONTRADICTIONS:
        raise ValueError("max_results must be bounded")
    aligner = aligner or TemporalAligner()
    accepted, revision_abstentions = _dedupe_revisions(events)
    eligible = [event for event in accepted
                if query_cutoff_ms is None or event.capture_timestamp_ms <= query_cutoff_ms]
    abstentions = list(revision_abstentions)
    by_key: dict[tuple[str, str], list[TemporalEvent]] = defaultdict(list)
    for event in eligible:
        by_key[_event_key(event)].append(event)
    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for group in by_key.values():
        group.sort(key=lambda event: (event.capture_timestamp_ms, event.sequence, event.event_id))
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if left.modality == right.modality:
                    continue
                pair = tuple(sorted((left.event_id, right.event_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                item = contradiction_candidate(left, right, aligner=aligner,
                                               now_ms=now_ms if now_ms is not None else query_cutoff_ms,
                                               max_gap_ms=max_gap_ms,
                                               require_calibrated=require_calibrated)
                if item is None:
                    continue
                if item.get("status") == "abstained":
                    if len(abstentions) < MAX_ABSTENTIONS:
                        abstentions.append(item)
                elif len(candidates) < max_results:
                    candidates.append(item)
    total_pairs = len(candidates) + len(abstentions)
    status = "available" if candidates else "abstained" if abstentions else "insufficient"
    return {
        "status": status,
        "candidates": candidates,
        "abstentions": abstentions[:MAX_ABSTENTIONS],
        "contradiction_detected": bool(candidates),
        "verification_status": "not_checked",
        "deception_assessed": False,
        "pair_count": total_pairs,
        "candidate_count": len(candidates),
        "abstention_count": len(abstentions),
        "query_cutoff_ms": query_cutoff_ms,
        "uncertainty": "unverified",
    }


# Explicit names for callers that prefer detector/packet terminology.
detect_contradiction_candidates = detect_contradictions
