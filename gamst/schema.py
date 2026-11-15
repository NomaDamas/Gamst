"""Versioned, JSON-safe output contract; scores never imply inner mental state."""
from __future__ import annotations

import math
import time
import uuid

SCHEMA_VERSION = "0.1"
MODEL_ID = "humansense.observation-rules"
MODEL_VERSION = "0.1.0"


def timestamp(value=None):
    if value is None:
        return time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("timestamp_ms must be a finite non-negative number")
    return int(value)


def envelope(kind, subject_id, timestamp_ms, *, window_ms=0, observation=None,
             evidence=None, quality=None, models=None):
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id must be a non-empty string")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        "type": kind,
        "subject_id": subject_id,
        "timestamp_ms": timestamp(timestamp_ms),
        "window_ms": window_ms,
        "window": {"start_ms": max(0, timestamp_ms - window_ms), "end_ms": timestamp_ms},
        "models": models or [{"id": MODEL_ID, "version": MODEL_VERSION, "backend": "local"}],
        "quality": quality or {"status": "sufficient", "flags": []},
        "observation": observation or {},
        "evidence": evidence or [],
        "inference": {"hypotheses": [], "verified": False},
    }


def hypothesis(target, evidence_ids, *, description, status="available"):
    return {"status": status, "target": target, "hypothesis": description,
            "model_score": None, "probability": None, "calibrated": False,
            "calibration": None, "evidence_ids": list(evidence_ids)}


def validate_output(value):
    """Reject fabricated probabilities, including in third-party plugin output."""
    if isinstance(value, dict):
        if "probability" in value:
            probability = value["probability"]
            if probability is not None:
                calibration = value.get("calibration")
                if not value.get("calibrated") or not isinstance(calibration, dict):
                    raise ValueError("probability requires calibration metadata")
                if not all(calibration.get(key) for key in ("dataset", "model_version", "target", "evaluation")):
                    raise ValueError("incomplete calibration metadata")
                if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                    raise ValueError("probability must be between 0 and 1")
            elif value.get("calibrated"):
                raise ValueError("calibrated=true requires probability")
        for child in value.values():
            validate_output(child)
    elif isinstance(value, list):
        for child in value:
            validate_output(child)
    return value
