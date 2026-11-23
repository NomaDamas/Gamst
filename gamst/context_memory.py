"""Bounded current/recent/episodic context memory with replayable deltas."""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable


def _json(value: Any) -> Any:
    json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return deepcopy(value)


class ContextMemory:
    """A dependency-free, bounded memory engine for normalized evidence.

    ``query_cutoff_ms`` is an internal evidence-causality watermark.  It never
    truncates a user's query or removes user-visible context.
    """

    def __init__(self, *, session_id: str = "session", max_recent: int = 256,
                 max_episodic: int = 1024, max_current: int = 256,
                 max_bytes: int = 4 * 1024 * 1024):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("session_id must be a bounded string")
        if any(type(v) is not int or v < 1 for v in (max_recent, max_episodic, max_bytes)):
            raise ValueError("memory bounds must be positive integers")
        self.session_id = session_id
        if type(max_current) is not int or max_current < 1:
            raise ValueError("max_current must be a positive integer")
        self.max_recent, self.max_episodic = max_recent, max_episodic
        self.max_current, self.max_bytes = max_current, max_bytes
        self.current_state: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.recent_timeline: deque[dict[str, Any]] = deque(maxlen=max_recent)
        self.episodic_memory: deque[dict[str, Any]] = deque(maxlen=max_episodic)
        self._evidence: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._revision = 0
        self._sizes: dict[int, tuple[dict[str, Any], int]] = {}

    def _encoded(self, value) -> int:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())

    def _item_size(self, item: dict[str, Any]) -> int:
        cached = self._sizes.get(id(item))
        if cached is None or cached[0] is not item:
            cached = self._sizes[id(item)] = (item, self._encoded(item))
        return cached[1]

    def _snapshot_bytes(self) -> int:
        """Exact byte length of the compact snapshot JSON without copying or re-encoding retained items."""
        empty = {"session_id": self.session_id, "revision": self._revision,
                 "current_state": {}, "recent_timeline": [], "episodic_memory": []}
        total = self._encoded(empty)
        for items in (self.recent_timeline, self.episodic_memory):
            total += sum(self._item_size(item) for item in items) + max(0, len(items) - 1)
        total += sum(self._encoded(key) + 1 + self._item_size(item) for key, item in self.current_state.items())
        return total + max(0, len(self.current_state) - 1)

    def _check_size(self) -> None:
        # Retention is bounded across every index.  Previously only episodic
        # entries were evicted, allowing non-episodic evidence and current
        # state to grow without limit.
        while True:
            if self._snapshot_bytes() <= self.max_bytes:
                break
            if self.episodic_memory:
                self.episodic_memory.popleft()
            elif self.recent_timeline:
                self.recent_timeline.popleft()
            elif self.current_state:
                self.current_state.popitem(last=False)
            else:
                break
        retained = list(self.recent_timeline) + list(self.episodic_memory) + list(self.current_state.values())
        self._evidence = OrderedDict(
            (item["evidence_id"], item) for item in retained if item.get("evidence_id")
        )
        live = {id(item) for item in retained}
        self._sizes = {key: value for key, value in self._sizes.items() if key in live}

    def append(self, event: dict[str, Any], *, entity_id: str | None = None,
               episodic: bool = False) -> dict[str, Any]:
        if not isinstance(event, dict) or not event.get("event_id"):
            raise ValueError("event must contain event_id")
        item = _json(event)
        item.setdefault("session_id", self.session_id)
        item.setdefault("stored_at_ms", int(time.time() * 1000))
        if entity_id is not None:
            item.setdefault("entity_id", str(entity_id))
        evidence_id = item.get("evidence_id") or item.get("event_id")
        # Delta replay and provider retries are idempotent by evidence ID.
        prior = self._evidence.get(evidence_id)
        if prior is not None:
            return deepcopy(prior)
        self.recent_timeline.append(item)
        item["evidence_id"] = evidence_id
        self._evidence[evidence_id] = item
        if entity_id is not None:
            self.current_state[str(entity_id)] = item
            self.current_state.move_to_end(str(entity_id))
            while len(self.current_state) > self.max_current:
                self.current_state.popitem(last=False)
        if episodic:
            self.episodic_memory.append(item)
        self._revision += 1
        self._check_size()
        return deepcopy(item)

    update = append

    def snapshot(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "revision": self._revision,
                "current_state": deepcopy(dict(self.current_state)),
                "recent_timeline": deepcopy(list(self.recent_timeline)),
                "episodic_memory": deepcopy(list(self.episodic_memory))}

    def delta(self, previous: dict[str, Any] | None) -> dict[str, Any]:
        before = previous or {"revision": 0, "current_state": {},
                              "recent_timeline": [], "episodic_memory": []}
        now = self.snapshot()
        old_ids = {item.get("event_id") for item in before.get("recent_timeline", [])}
        return {"session_id": self.session_id, "from_revision": before.get("revision", 0),
                "to_revision": self._revision,
                "current_state": now["current_state"],
                "events": [item for item in now["recent_timeline"]
                           if item.get("event_id") not in old_ids]}

    def replay(self, delta: dict[str, Any]) -> None:
        if not isinstance(delta, dict) or delta.get("session_id") != self.session_id:
            raise ValueError("delta belongs to another session")
        for item in delta.get("events", []):
            self.append(item)
        for entity_id, item in delta.get("current_state", {}).items():
            self.current_state[str(entity_id)] = _json(item)
            self.current_state.move_to_end(str(entity_id))
        while len(self.current_state) > self.max_current:
            self.current_state.popitem(last=False)
        self._check_size()

    apply_delta = replay

    def retrieve(self, *, evidence_id: str | None = None, entity_id: str | None = None,
                 start_ms: int | None = None, end_ms: int | None = None,
                 query_cutoff_ms: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be 1..256")
        result = []
        for item in reversed(list(self._evidence.values())):
            if evidence_id is not None and item.get("evidence_id") != evidence_id and item.get("event_id") != evidence_id:
                continue
            if entity_id is not None and item.get("entity_id") != entity_id:
                continue
            timestamp = item.get("timestamp_ms", item.get("capture_timestamp_ms"))
            if start_ms is not None and (not isinstance(timestamp, int) or timestamp < start_ms):
                continue
            if end_ms is not None and (not isinstance(timestamp, int) or timestamp > end_ms):
                continue
            if query_cutoff_ms is not None:
                # A causality watermark cannot prove an untimestamped record
                # was known in time; fail closed instead of returning it.
                if type(timestamp) is not int or timestamp > query_cutoff_ms:
                    continue
            result.append(deepcopy(item))
            if len(result) >= limit:
                break
        return result

    def checkpoint(self) -> dict[str, Any]:
        state = self.snapshot()
        body = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return {"schema_version": "gamst.memory.v1", "session_id": self.session_id,
                "lineage": hashlib.sha256(body).hexdigest(), "state": state}

    @classmethod
    def resume(cls, checkpoint: dict[str, Any], **kwargs) -> "ContextMemory":
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "gamst.memory.v1":
            raise ValueError("unsupported memory checkpoint")
        state = checkpoint.get("state")
        body = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(body).hexdigest() != checkpoint.get("lineage"):
            raise ValueError("memory checkpoint lineage mismatch")
        memory = cls(session_id=checkpoint["session_id"], **kwargs)
        memory.current_state.update(_json(state.get("current_state", {})))
        while len(memory.current_state) > memory.max_current:
            memory.current_state.popitem(last=False)
        memory.recent_timeline.extend(_json(state.get("recent_timeline", [])))
        memory.episodic_memory.extend(_json(state.get("episodic_memory", [])))
        for item in list(memory.recent_timeline) + list(memory.episodic_memory):
            if item.get("evidence_id"):
                memory._evidence[item["evidence_id"]] = item
        memory._revision = int(state.get("revision", 0))
        memory._check_size()
        return memory


ContextMemoryEngine = ContextMemory
MemoryContext = ContextMemory


def project_record(record: dict[str, Any], *, session_id: str | None = None) -> dict[str, Any]:
    """Project a retained GAMST envelope into the portable memory contract.

    The projection deliberately excludes raw media and keeps only bounded
    normalized observation metadata plus evidence lineage.  It is the bridge
    used by the Core runtime; provider-specific payloads remain in the Store.
    """
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        raise ValueError("record must be a retained GAMST envelope")
    observation = record.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("record observation must be an object")
    event: dict[str, Any] = {
        "event_id": record["id"],
        "evidence_id": record["id"],
        "timestamp_ms": record.get("timestamp_ms"),
        "type": record.get("type"),
        "name": observation.get("name"),
        "modality": observation.get("modality"),
        "value": observation.get("value"),
        "session_id": session_id or record.get("session_id"),
        "source_record_type": record.get("type"),
    }
    entity_id = observation.get("entity_id") or observation.get("track_id")
    if entity_id is not None:
        event["entity_id"] = str(entity_id)
    if record.get("evidence"):
        event["lineage"] = [{"record_id": item.get("record_id")}
                             for item in record["evidence"] if isinstance(item, dict)]
    # Fail before mutating memory if a provider returned non-JSON or raw data.
    return _json(event)


def sync_records(memory: ContextMemory, records: Iterable[dict[str, Any]], *,
                 episodic_types: tuple[str, ...] = ("observation.event", "human.inference")) -> int:
    """Idempotently import retained Core envelopes into ContextMemory."""
    if not isinstance(memory, ContextMemory):
        raise TypeError("memory must be a ContextMemory")
    count = 0
    for record in records:
        projected = project_record(record, session_id=memory.session_id)
        memory.append(projected, entity_id=projected.get("entity_id"),
                      episodic=projected.get("type") in episodic_types)
        count += 1
    return count


class CheckpointStore:
    """Atomic, bounded JSON checkpoint storage for :class:`ContextMemory`.

    The store is deliberately a small adapter around the existing checkpoint
    contract. It does not make a provider's model state durable and it never
    treats a missing file as an empty session; callers must choose whether to
    create a new session. Session IDs are hashed for filenames so arbitrary
    user IDs cannot escape the configured root.
    """

    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int = 4 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("session_id must be a bounded string")
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def save(self, memory: ContextMemory) -> dict[str, Any]:
        if not isinstance(memory, ContextMemory):
            raise TypeError("memory must be a ContextMemory")
        checkpoint = memory.checkpoint()
        encoded = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.max_bytes:
            raise ValueError("checkpoint exceeds store byte budget")
        target = self._path(memory.session_id)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return checkpoint

    def load(self, session_id: str, **kwargs) -> ContextMemory | None:
        path = self._path(session_id)
        try:
            encoded = path.read_bytes()
        except FileNotFoundError:
            return None
        if len(encoded) > self.max_bytes:
            raise ValueError("checkpoint exceeds store byte budget")
        try:
            checkpoint = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint is not valid JSON") from exc
        if checkpoint.get("session_id") != session_id:
            raise ValueError("checkpoint session mismatch")
        return ContextMemory.resume(checkpoint, **kwargs)

    def clear(self, session_id: str) -> bool:
        path = self._path(session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

