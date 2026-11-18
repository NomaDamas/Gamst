"""Bounded short-term semantic state chains for streaming perception."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import time
from typing import Any


class SemanticStateChain:
    """Keep recent state updates; no model calls and no raw media storage."""

    def __init__(self, *, max_updates: int = 128, max_bytes: int = 262144,
                 retention_ms: int = 300000):
        if type(max_updates) is not int or not 1 <= max_updates <= 4096:
            raise ValueError("max_updates must be 1..4096")
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 4 * 1024 * 1024:
            raise ValueError("max_bytes must be 1024..4194304")
        if type(retention_ms) is not int or retention_ms <= 0:
            raise ValueError("retention_ms must be positive")
        self.max_updates = max_updates
        self.max_bytes = max_bytes
        self.retention_ms = retention_ms
        self._items: dict[str, deque[dict[str, Any]]] = {}
        self.dropped = 0

    def append(self, subject_id: str, *, timestamp_ms: int | None = None,
               state: dict[str, Any], events: list[dict[str, Any]] | None = None,
               evidence_ids: list[str] | None = None, sources: list[str] | None = None,
               model_id: str = "semantic_state.rules", model_version: str = "1") -> dict[str, Any]:
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("subject_id must be nonempty")
        if type(state) is not dict:
            raise ValueError("state must be an object")
        if events is not None and type(events) is not list:
            raise ValueError("events must be an array")
        ts = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
        if type(ts) is not int or ts < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        item = {
            "timestamp_ms": ts,
            "state": deepcopy(state),
            "events": deepcopy(events or []),
            "evidence_ids": list(evidence_ids or []),
            "sources": list(sources or []),
            "model_id": model_id,
            "model_version": model_version,
        }
        encoded = json.dumps(item, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > self.max_bytes:
            raise ValueError("semantic state update exceeds byte budget")
        chain = self._items.setdefault(subject_id, deque())
        chain.append(item)
        self._prune(subject_id, ts)
        while self._size(subject_id) > self.max_bytes and chain:
            chain.popleft()
            self.dropped += 1
        return deepcopy(item)

    def _prune(self, subject_id: str, now_ms: int) -> None:
        chain = self._items.setdefault(subject_id, deque())
        cutoff = max(0, now_ms - self.retention_ms)
        while chain and chain[0]["timestamp_ms"] < cutoff:
            chain.popleft()
            self.dropped += 1
        while len(chain) > self.max_updates:
            chain.popleft()
            self.dropped += 1

    def _size(self, subject_id: str) -> int:
        return len(json.dumps(list(self._items.get(subject_id, ())), ensure_ascii=False).encode())

    def get(self, subject_id: str, *, start_ms: int | None = None,
            end_ms: int | None = None, limit: int = 32) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be 1..256")
        chain = list(self._items.get(subject_id, ()))
        items = [item for item in chain
                 if (start_ms is None or item["timestamp_ms"] >= start_ms)
                 and (end_ms is None or item["timestamp_ms"] <= end_ms)]
        items = items[-limit:]
        return {
            "subject_id": subject_id,
            "updates": deepcopy(items),
            "count": len(items),
            "dropped": self.dropped,
            "retention_ms": self.retention_ms,
            "scope": "semantic_state_chain",
        }

    def clear(self, subject_id: str | None = None) -> None:
        if subject_id is None:
            self._items.clear()
        else:
            self._items.pop(subject_id, None)
