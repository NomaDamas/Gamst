"""Deterministic fusion of streaming video state and typed perception signals."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class SemanticStateFusion:
    """Combine bounded plugin observations with a streaming-video state.

    This class performs no model inference and never decides to call a strong
    model. It is the host-owned state merge used before MCP context delivery.
    """

    def __init__(self, *, max_signals: int = 64, max_evidence: int = 32):
        if type(max_signals) is not int or not 1 <= max_signals <= 256:
            raise ValueError("max_signals must be 1..256")
        if type(max_evidence) is not int or not 1 <= max_evidence <= 128:
            raise ValueError("max_evidence must be 1..128")
        self.max_signals = max_signals
        self.max_evidence = max_evidence
        self._signals: dict[str, dict[str, Any]] = {}
        self._video_state: dict[str, Any] | None = None

    def observe_signal(self, signal: dict[str, Any]) -> None:
        name = signal.get("name") or signal.get("observation", {}).get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("signal requires a name")
        self._signals[name] = deepcopy(signal)
        while len(self._signals) > self.max_signals:
            self._signals.pop(next(iter(self._signals)))

    def observe_video_state(self, state: dict[str, Any], *, timestamp_ms: int,
                            evidence_ids: list[str], model_id: str,
                            model_version: str) -> None:
        if type(state) is not dict or type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError("video state and timestamp_ms are required")
        self._video_state = {
            "state": deepcopy(state),
            "timestamp_ms": timestamp_ms,
            "evidence_ids": list(evidence_ids)[-self.max_evidence:],
            "model_id": model_id,
            "model_version": model_version,
        }

    def snapshot(self, *, timestamp_ms: int) -> dict[str, Any]:
        evidence: list[str] = []
        for signal in self._signals.values():
            evidence.extend(signal.get("evidence_ids", []))
            if signal.get("id"):
                evidence.append(signal["id"])
        if self._video_state:
            evidence.extend(self._video_state["evidence_ids"])
        return {
            "timestamp_ms": timestamp_ms,
            "video_state": deepcopy(self._video_state),
            "signals": deepcopy(list(self._signals.values())),
            "evidence_ids": list(dict.fromkeys(evidence))[-self.max_evidence:],
            "modalities": sorted({str(s.get("modality", "unknown")) for s in self._signals.values()}),
            "fusion": "host_state_merge_no_model_inference",
        }
