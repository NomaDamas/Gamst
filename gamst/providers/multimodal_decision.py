"""Deterministic multimodal decision provider.

This provider is an internal perception module: it consumes bounded, retained
registered observations from several modalities and emits a structured decision
observation. It never receives raw media and does not claim calibrated intent,
emotion, identity, or probability.
"""
from __future__ import annotations

import hashlib
import json

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..schema import timestamp


CAPABILITY_ID = "perception.multimodal_decision"
OUTPUT_NAME = "decision.multimodal"
PROVIDER_ID = "gamst.multimodal_decision.baseline"
MODEL_VERSION = "0.1.0"


def select_decision_evidence(records: list[dict], *, now_ms: int | None = None,
                             requested_capabilities: list[str] | None = None,
                             input_window_ms: int = 0, max_evidence: int = 16) -> dict:
    """Select fresh, newest-per-capability evidence for any decision provider.

    Selected tuples contain ``(timestamp, evidence_id, capability, modality,
    record)``. The original record is retained for adapters, but evidence IDs
    and all provenance remain host-owned. No observation value is copied here.
    """
    now = timestamp(now_ms)
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    if type(max_evidence) is not int or not 1 <= max_evidence <= 32:
        raise ValueError("max_evidence must be an integer from 1 to 32")
    requested = [] if requested_capabilities is None else list(requested_capabilities)
    if type(input_window_ms) is not int or not 0 <= input_window_ms <= 86400000:
        raise ValueError("input_window_ms must be a bounded integer")
    if len(requested) > 16 or any(not isinstance(item, str) for item in requested):
        raise ValueError("requested_capabilities must contain bounded strings")
    for item in requested:
        identifier(item, "requested_capability")

    candidates = []
    stale = False
    requested_set = set(requested)
    for record in records:
        if not isinstance(record, dict) or record.get("type") != "observation.signal":
            continue
        observation = record.get("observation", {})
        if not isinstance(observation, dict):
            continue
        capability = observation.get("capability", {})
        capability_id = capability.get("id") if isinstance(capability, dict) else None
        if not capability_id or capability_id == CAPABILITY_ID:
            continue
        if requested_set and capability_id not in requested_set:
            continue
        evidence_id = record.get("id")
        ts = record.get("timestamp_ms")
        modality = observation.get("modality")
        if not isinstance(modality, str) and isinstance(capability, dict):
            modality = capability.get("modality")
        if not isinstance(evidence_id, str) or not isinstance(ts, int) or not isinstance(modality, str):
            continue
        age = max(0, now - ts)
        freshness = capability.get("freshness_ms", 5000) if isinstance(capability, dict) else 5000
        if not isinstance(freshness, int) or isinstance(freshness, bool) or freshness < 0:
            continue
        if age > freshness:
            stale = True
            continue
        candidates.append((ts, evidence_id, capability_id, modality, record))

    # One newest item per capability keeps decisions stable and bounded.
    newest = {}
    for item in sorted(candidates, key=lambda value: value[:4], reverse=True):
        newest.setdefault(item[2], item)
    selected = sorted(newest.values(), key=lambda value: value[:4], reverse=True)[:max_evidence]
    watermark = hashlib.sha256(json.dumps(
        [{"id": item[1], "timestamp_ms": item[0], "capability": item[2]} for item in selected],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "now_ms": now,
        "selected": selected,
        "evidence_ids": [item[1] for item in selected],
        "selected_capabilities": [item[2] for item in selected],
        "modalities": sorted({item[3] for item in selected}),
        "input_watermark": watermark,
        "input_window_ms": input_window_ms,
        "evidence_ages_ms": [max(0, now - item[0]) for item in selected],
        "requested_capabilities": requested,
        "stale_evidence": stale,
    }


class MultimodalDecisionProvider:
    """A bounded, deterministic baseline replaceable by a real multimodal model."""

    @staticmethod
    def capability() -> CapabilitySpec:
        return CapabilitySpec(
            CAPABILITY_ID,
            OUTPUT_NAME,
            "events",
            PROVIDER_ID,
            MODEL_VERSION,
            (
                OutputField("decision", "string"),
                OutputField("selected_capabilities", "array"),
                OutputField("evidence_ids", "array"),
                OutputField("reason", "string"),
                OutputField("confidence", "number"),
                OutputField("abstained", "boolean"),
                OutputField("modalities", "array"),
                OutputField("input_watermark", "string"),
                OutputField("input_window_ms", "integer"),
                OutputField("evidence_ages_ms", "array"),
                OutputField("requested_capabilities", "array"),
            ),
            "frame",
            freshness_ms=5000,
            stateful=False,
            interpretation="estimate",
            devices=("cpu",),
            sensor_requirements=("retained-registered-observations",),
            license="Apache-2.0",
        )

    def __init__(self, *, min_modalities: int = 2, max_evidence: int = 16):
        if type(min_modalities) is not int or not 1 <= min_modalities <= 8:
            raise ValueError("min_modalities must be an integer from 1 to 8")
        if type(max_evidence) is not int or not 1 <= max_evidence <= 32:
            raise ValueError("max_evidence must be an integer from 1 to 32")
        self.min_modalities = min_modalities
        self.max_evidence = max_evidence
        self.spec = self.capability()
        self._sequence = 0

    @property
    def descriptor(self) -> dict:
        return {
            "capability": self.spec.describe(),
            "provider_kind": "internal_perception_module",
            "model_status": "deterministic_baseline",
            "raw_media": False,
            "calibrated": False,
            "min_modalities": self.min_modalities,
            "max_evidence": self.max_evidence,
        }

    def decide(self, records: list[dict], *, now_ms: int | None = None,
               requested_capabilities: list[str] | None = None,
               input_window_ms: int = 0) -> dict:
        """Return a bounded decision value from retained records.

        Records are evidence only. The returned confidence is an uncalibrated
        heuristic score and is deliberately not named ``probability``.
        """
        selection = select_decision_evidence(
            records, now_ms=now_ms, requested_capabilities=requested_capabilities,
            input_window_ms=input_window_ms, max_evidence=self.max_evidence)
        selected = selection["selected"]
        evidence_ids = selection["evidence_ids"]
        capabilities = selection["selected_capabilities"]
        modalities = selection["modalities"]
        watermark = selection["input_watermark"]
        now = selection["now_ms"]
        requested = selection["requested_capabilities"]

        if len(modalities) < self.min_modalities:
            if modalities == ["audio"]:
                decision, reason = "request_more_video", "insufficient_multimodal_evidence"
            elif modalities == ["video"]:
                decision, reason = "request_more_audio", "insufficient_multimodal_evidence"
            else:
                decision, reason = "request_additional_modality", "insufficient_multimodal_evidence"
            confidence = 0.25 if selected else 0.0
            abstained = True
        else:
            decision, reason = "fuse_multimodal_context", "cross_modal_evidence_available"
            confidence = min(0.95, 0.5 + 0.15 * len(modalities) + 0.03 * min(len(selected), 8))
            abstained = False

        value = {
            "decision": decision,
            "selected_capabilities": capabilities,
            "evidence_ids": evidence_ids,
            "reason": reason,
            "confidence": round(confidence, 4),
            "abstained": abstained,
            "modalities": modalities,
            "input_watermark": watermark,
            "input_window_ms": input_window_ms,
            "evidence_ages_ms": [max(0, now - item[0]) for item in selected],
            "requested_capabilities": requested,
        }
        return self.spec.validate(bounded_json(value, forbid_raw=True))

    def next_sensor(self, *, source_id: str = "multimodal", sensor_id: str = "decision") -> SensorMetadata:
        identifier(source_id)
        identifier(sensor_id)
        sensor = SensorMetadata(source_id, sensor_id, self._sequence)
        self._sequence += 1
        return sensor
