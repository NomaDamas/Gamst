"""Owner-declared Watch rules evaluated deterministically over emitted records.

A Watch is the only way Gamst raises an alarm: there is no hidden trigger.
Owners (the user, or the application-side decision model ``jev``) declare a
structured rule; Gamst evaluates it on every emitted record, rate-limits it
with the owner's cooldown and per-minute cap, and delivers the alarm with a
bounded time-window context to the owner's recipient.

Rules are versioned by the sha256 of their canonical JSON. Updates must name
the version they replace, every mutation and proposal is appended to an audit
log, and every alarm records how each condition's evidence contributed.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import deque
from typing import Any, Callable

OWNERS = ("user", "jev")
OPERATORS = ("gt", "gte", "lt", "lte", "eq", "ne", "exists", "contains")
RULE_FIELDS = ("subject_id", "conditions", "threshold", "cooldown_ms", "max_per_minute",
               "context_window_ms", "recipient", "enabled")
MAX_WATCHES = 64
MAX_CONDITIONS = 16
MAX_ALARMS = 1000
MAX_AUDIT = 2000
MAX_CONTEXT_RECORDS = 200


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def rule_version(rule: dict) -> str:
    return hashlib.sha256(_canonical({k: rule[k] for k in RULE_FIELDS}).encode()).hexdigest()


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_condition(index: int, condition: Any) -> dict:
    where = f"conditions[{index}]"
    if not isinstance(condition, dict):
        raise ValueError(f"{where}: must be an object")
    allowed = {"capability_id", "signal", "path", "op", "value", "weight", "within_ms"}
    unknown = set(condition) - allowed
    if unknown:
        raise ValueError(f"{where}: unexpected keys {sorted(unknown)}")
    if ("capability_id" in condition) == ("signal" in condition):
        raise ValueError(f"{where}: exactly one of capability_id or signal is required")
    source = condition.get("capability_id", condition.get("signal"))
    if not isinstance(source, str) or not 1 <= len(source) <= 128:
        raise ValueError(f"{where}: source must be a 1..128 character string")
    op = condition.get("op")
    if op not in OPERATORS:
        raise ValueError(f"{where}.op: one of {OPERATORS}")
    if op != "exists" and "value" not in condition:
        raise ValueError(f"{where}.value: required for op {op}")
    if op in ("gt", "gte", "lt", "lte") and not _number(condition["value"]):
        raise ValueError(f"{where}.value: finite number required for op {op}")
    path = condition.get("path", "")
    if not isinstance(path, str) or len(path) > 256:
        raise ValueError(f"{where}.path: string up to 256 characters")
    weight = condition.get("weight", 1.0)
    if not _number(weight) or weight <= 0:
        raise ValueError(f"{where}.weight: positive finite number")
    within = condition.get("within_ms", 5000)
    if type(within) is not int or not 0 < within <= 3_600_000:
        raise ValueError(f"{where}.within_ms: integer 1..3600000")
    _canonical(condition.get("value"))
    result = {"path": path, "op": op, "weight": float(weight), "within_ms": within}
    result["capability_id" if "capability_id" in condition else "signal"] = source
    if op != "exists":
        result["value"] = condition["value"]
    return result


def validate_rule(rule: dict) -> dict:
    conditions = rule.get("conditions")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= MAX_CONDITIONS:
        raise ValueError(f"conditions: 1..{MAX_CONDITIONS} conditions required")
    checked = {
        "subject_id": rule.get("subject_id", "local-user"),
        "conditions": [_validate_condition(i, c) for i, c in enumerate(conditions)],
        "threshold": rule.get("threshold", 1.0),
        "cooldown_ms": rule.get("cooldown_ms", 10_000),
        "max_per_minute": rule.get("max_per_minute", 6),
        "context_window_ms": rule.get("context_window_ms", 10_000),
        "recipient": rule.get("recipient", "owner"),
        "enabled": rule.get("enabled", True),
    }
    if not isinstance(checked["subject_id"], str) or not checked["subject_id"]:
        raise ValueError("subject_id: non-empty string")
    if not _number(checked["threshold"]) or not 0 < checked["threshold"] <= 1:
        raise ValueError("threshold: weighted fraction in (0, 1]")
    for key, upper in (("cooldown_ms", 86_400_000), ("context_window_ms", 600_000)):
        if type(checked[key]) is not int or not 0 <= checked[key] <= upper:
            raise ValueError(f"{key}: integer 0..{upper}")
    if type(checked["max_per_minute"]) is not int or not 1 <= checked["max_per_minute"] <= 600:
        raise ValueError("max_per_minute: integer 1..600")
    if not isinstance(checked["recipient"], str) or not 1 <= len(checked["recipient"]) <= 128:
        raise ValueError("recipient: 1..128 character string")
    if type(checked["enabled"]) is not bool:
        raise ValueError("enabled: boolean")
    checked["threshold"] = float(checked["threshold"])
    return checked


def _lookup(value: Any, path: str) -> tuple[bool, Any]:
    if not path:
        return True, value
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return False, None
    return True, value


def _compare(op: str, observed: Any, target: Any) -> bool:
    if op == "exists":
        return True
    if op == "eq":
        return observed == target
    if op == "ne":
        return observed != target
    if op == "contains":
        if isinstance(observed, str):
            return isinstance(target, str) and target in observed
        if isinstance(observed, list):
            return any(item == target or (isinstance(item, dict) and target in (item.get("type"), item.get("name"), item.get("label")))
                       for item in observed)
        return False
    if not _number(observed) or isinstance(observed, bool):
        return False
    return {"gt": observed > target, "gte": observed >= target,
            "lt": observed < target, "lte": observed <= target}[op]


def _source_of(record: dict) -> tuple[str | None, str | None]:
    observation = record.get("observation") or {}
    capability = observation.get("capability") or {}
    return capability.get("id"), observation.get("name")


class WatchManager:
    """Bounded watch registry, evaluator, alarm log and audit log."""

    def __init__(self, query: Callable[..., list[dict]], clock: Callable[[], int]):
        self._query = query
        self._clock = clock
        self.watches: dict[str, dict] = {}
        self.alarms: deque[dict] = deque(maxlen=MAX_ALARMS)
        self.audit: deque[dict] = deque(maxlen=MAX_AUDIT)
        self.proposals: dict[str, dict] = {}
        self._latest: dict[tuple[str, str, str], dict] = {}
        self._fired: dict[str, deque[int]] = {}
        self._sinks: list[Callable[[dict], None]] = []
        self.sink_errors = 0

    # -- delivery -----------------------------------------------------------
    def add_sink(self, sink: Callable[[dict], None]) -> None:
        if sink not in self._sinks:
            self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[dict], None]) -> None:
        if sink in self._sinks:
            self._sinks.remove(sink)

    # -- audit ----------------------------------------------------------------
    def _log(self, action: str, actor: str, watch_id: str, **details) -> dict:
        entry = {"id": uuid.uuid4().hex, "timestamp_ms": self._clock(), "action": action,
                 "actor": actor, "watch_id": watch_id, **details}
        self.audit.append(entry)
        return entry

    @staticmethod
    def _actor(actor: Any) -> str:
        if actor not in OWNERS:
            raise ValueError(f"actor/owner must be one of {OWNERS}")
        return actor

    def _public(self, watch: dict) -> dict:
        state = {k: watch[k] for k in ("id", "owner", "description", "version", "created_ms", "updated_ms")}
        state["rule"] = {k: watch["rule"][k] for k in RULE_FIELDS}
        state["stats"] = dict(watch["stats"])
        return json.loads(_canonical(state))

    # -- CRUD -------------------------------------------------------------------
    def create(self, *, owner: str, description: str, rule: dict, reason: str = "") -> dict:
        owner = self._actor(owner)
        if len(self.watches) >= MAX_WATCHES:
            raise ValueError(f"at most {MAX_WATCHES} watches")
        if not isinstance(description, str) or not 1 <= len(description) <= 2000:
            raise ValueError("description: 1..2000 characters (owner's natural-language intent)")
        checked = validate_rule(rule)
        if checked["recipient"] == "owner":
            checked["recipient"] = owner
        now = self._clock()
        watch = {"id": uuid.uuid4().hex, "owner": owner, "description": description,
                 "rule": checked, "version": rule_version(checked), "created_ms": now, "updated_ms": now,
                 "stats": {"evaluations": 0, "fired": 0, "suppressed_cooldown": 0, "suppressed_rate": 0},
                 "last_fired_ms": None}
        self.watches[watch["id"]] = watch
        self._log("create", owner, watch["id"], version=watch["version"], reason=reason)
        return self._public(watch)

    def _owned(self, watch_id: str, actor: str) -> dict:
        watch = self.watches.get(watch_id)
        if watch is None:
            raise ValueError("unknown watch_id")
        if watch["owner"] != actor:
            raise ValueError(f"watch is owned by {watch['owner']}; {actor} may only propose changes")
        return watch

    def update(self, *, watch_id: str, actor: str, expected_version: str, changes: dict,
               reason: str = "", proposal_id: str | None = None) -> dict:
        actor = self._actor(actor)
        watch = self._owned(watch_id, actor)
        if expected_version != watch["version"]:
            raise ValueError("expected_version does not match the current rule version")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("changes: non-empty object")
        unknown = set(changes) - set(RULE_FIELDS) - {"description"}
        if unknown:
            raise ValueError(f"changes: unexpected keys {sorted(unknown)}")
        if proposal_id is not None:
            proposal = self.proposals.get(proposal_id)
            if proposal is None or proposal["watch_id"] != watch_id or proposal["status"] != "open":
                raise ValueError("proposal_id is not an open proposal for this watch")
        if "description" in changes and (not isinstance(changes["description"], str)
                                         or not 1 <= len(changes["description"]) <= 2000):
            raise ValueError("description: 1..2000 characters")
        rule = {**watch["rule"], **{k: v for k, v in changes.items() if k != "description"}}
        checked = validate_rule(rule)
        if checked["recipient"] == "owner":
            checked["recipient"] = watch["owner"]
        previous = watch["version"]
        watch["rule"] = checked
        watch["version"] = rule_version(checked)
        if "description" in changes:
            watch["description"] = changes["description"]
        watch["updated_ms"] = self._clock()
        if proposal_id is not None:
            self.proposals[proposal_id]["status"] = "applied"
        self._log("update", actor, watch_id, previous_version=previous, version=watch["version"],
                  changes=json.loads(_canonical(changes)), reason=reason, proposal_id=proposal_id)
        return self._public(watch)

    def delete(self, *, watch_id: str, actor: str, reason: str = "") -> dict:
        actor = self._actor(actor)
        watch = self._owned(watch_id, actor)
        del self.watches[watch_id]
        self._fired.pop(watch_id, None)
        self._log("delete", actor, watch_id, version=watch["version"], reason=reason)
        return {"deleted": watch_id, "version": watch["version"]}

    def propose(self, *, watch_id: str, proposer: str, changes: dict, reason: str) -> dict:
        proposer = self._actor(proposer)
        watch = self.watches.get(watch_id)
        if watch is None:
            raise ValueError("unknown watch_id")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason: required")
        validate_rule({**watch["rule"], **{k: v for k, v in changes.items() if k != "description"}})
        proposal = {"id": uuid.uuid4().hex, "watch_id": watch_id, "proposer": proposer,
                    "base_version": watch["version"], "changes": json.loads(_canonical(changes)),
                    "reason": reason, "status": "open", "created_ms": self._clock()}
        self.proposals[proposal["id"]] = proposal
        self._log("propose", proposer, watch_id, proposal_id=proposal["id"], base_version=watch["version"],
                  changes=proposal["changes"], reason=reason)
        return dict(proposal, applied=False,
                    note="Not applied. The watch owner must call update_watch with expected_version and proposal_id.")

    def list(self, *, owner: str | None = None, include_audit: bool = False, limit: int = 100) -> dict:
        watches = [self._public(w) for w in self.watches.values() if owner in (None, w["owner"])]
        result = {"watches": watches,
                  "open_proposals": [p for p in self.proposals.values() if p["status"] == "open"]}
        if include_audit:
            result["audit"] = list(self.audit)[-limit:]
        return result

    def get_alarms(self, *, watch_id: str | None = None, since_ms: int | None = None, limit: int = 20) -> dict:
        items = [a for a in self.alarms if watch_id in (None, a["watch_id"])
                 and (since_ms is None or a["fired_ms"] >= since_ms)]
        return {"items": items[-limit:][::-1], "order": "newest_first", "returned_count": min(limit, len(items))}

    def clear_subject(self, subject_id: str | None) -> None:
        self._latest = {k: v for k, v in self._latest.items() if subject_id is not None and k[0] != subject_id}
        kept = [a for a in self.alarms if subject_id is not None and a["subject_id"] != subject_id]
        self.alarms.clear()
        self.alarms.extend(kept)

    # -- evaluation -------------------------------------------------------------
    def observe(self, record: dict) -> list[dict]:
        if record.get("type") not in ("observation.signal", "observation.event") or not self.watches:
            return []
        subject = record["subject_id"]
        capability_id, signal = _source_of(record)
        if capability_id:
            self._latest[(subject, "capability_id", capability_id)] = record
        if signal:
            self._latest[(subject, "signal", signal)] = record
        fired = []
        for watch in list(self.watches.values()):
            rule = watch["rule"]
            if not rule["enabled"] or rule["subject_id"] != subject:
                continue
            if not any((c.get("capability_id") and c["capability_id"] == capability_id)
                       or (c.get("signal") and c["signal"] == signal) for c in rule["conditions"]):
                continue
            alarm = self._evaluate(watch, record)
            if alarm is not None:
                fired.append(alarm)
        return fired

    def _evaluate(self, watch: dict, trigger: dict) -> dict | None:
        rule, now = watch["rule"], trigger["timestamp_ms"]
        watch["stats"]["evaluations"] += 1
        total = sum(c["weight"] for c in rule["conditions"])
        contributions, score = [], 0.0
        for index, condition in enumerate(rule["conditions"]):
            kind = "capability_id" if "capability_id" in condition else "signal"
            evidence = self._latest.get((rule["subject_id"], kind, condition[kind]))
            fresh = evidence is not None and 0 <= now - evidence["timestamp_ms"] <= condition["within_ms"]
            sufficient = fresh and (evidence.get("quality") or {}).get("status", "sufficient") == "sufficient"
            found, observed = (_lookup((evidence.get("observation") or {}).get("value"), condition["path"])
                               if sufficient else (False, None))
            satisfied = bool(found and _compare(condition["op"], observed, condition.get("value")))
            contribution = condition["weight"] / total if satisfied else 0.0
            score += contribution
            contributions.append({
                "condition_index": index, kind: condition[kind], "path": condition["path"],
                "op": condition["op"], "target": condition.get("value"),
                "evidence_id": evidence["id"] if evidence else None,
                "evidence_timestamp_ms": evidence["timestamp_ms"] if evidence else None,
                "observed": observed if found else None,
                "status": ("missing" if evidence is None else "stale" if not fresh
                           else "insufficient_quality" if not sufficient
                           else "path_missing" if not found else "evaluated"),
                "satisfied": satisfied, "weight": condition["weight"], "contribution": round(contribution, 6)})
        if score + 1e-9 < rule["threshold"]:
            return None
        last = watch["last_fired_ms"]
        if last is not None and now - last < rule["cooldown_ms"]:
            watch["stats"]["suppressed_cooldown"] += 1
            return None
        recent = self._fired.setdefault(watch["id"], deque())
        while recent and now - recent[0] >= 60_000:
            recent.popleft()
        if len(recent) >= rule["max_per_minute"]:
            watch["stats"]["suppressed_rate"] += 1
            return None
        recent.append(now)
        watch["last_fired_ms"] = now
        watch["stats"]["fired"] += 1
        start = now - rule["context_window_ms"]
        context = [r for kind in ("observation.signal", "observation.event")
                   for r in self._query(rule["subject_id"], start, now, kind, MAX_CONTEXT_RECORDS)]
        context.sort(key=lambda r: (r["timestamp_ms"], r["id"]))
        alarm = {"id": uuid.uuid4().hex, "watch_id": watch["id"], "owner": watch["owner"],
                 "recipient": rule["recipient"], "description": watch["description"],
                 "version": watch["version"], "subject_id": rule["subject_id"], "fired_ms": now,
                 "trigger_record_id": trigger["id"], "score": round(score, 6), "threshold": rule["threshold"],
                 "contributions": contributions,
                 "context": {"start_ms": start, "end_ms": now,
                             "records": json.loads(_canonical(context[-MAX_CONTEXT_RECORDS:])),
                             "truncated": len(context) > MAX_CONTEXT_RECORDS}}
        self.alarms.append(alarm)
        for sink in list(self._sinks):
            try:
                sink(alarm)
            except Exception:
                self.sink_errors += 1
        return alarm
