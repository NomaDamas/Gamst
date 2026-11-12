"""Thread-safe bounded processing-latency telemetry with a monotonic clock.

Percentiles describe each stage's most recent ``capacity`` measurements. Count,
total, mean and maximum describe all observations since its last reset. These
processing durations do not estimate sensor clock alignment or capture lag.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import math
import threading
import time


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("stage name must contain 1 to 128 characters")
    return value


class LatencyMonitor:
    """Store finite non-negative milliseconds without retaining input payloads."""

    def __init__(self, capacity=10_000, max_stages=128):
        self.capacity = _positive_integer(capacity, "capacity")
        self.max_stages = _positive_integer(max_stages, "max_stages")
        self._stages = {}
        self._lock = threading.RLock()

    def record(self, name, elapsed_ms):
        """Record one duration; fixed stage names keep cardinality bounded.

        A new stage exceeding ``max_stages`` raises ValueError rather than
        discarding existing stages or retaining unbounded label cardinality.
        """
        name = _name(name)
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        elapsed_ms = float(elapsed_ms)
        with self._lock:
            stage = self._stages.get(name)
            if stage is None:
                if len(self._stages) >= self.max_stages:
                    raise ValueError("latency stage capacity exceeded")
                stage = {"values": deque(maxlen=self.capacity), "count": 0,
                         "total_ms": 0.0, "max_ms": 0.0}
                self._stages[name] = stage
            total = stage["total_ms"] + elapsed_ms
            if not math.isfinite(total):
                raise ValueError("latency total would exceed finite numeric range")
            stage["values"].append(elapsed_ms)
            stage["count"] += 1
            stage["total_ms"] = total
            stage["max_ms"] = max(stage["max_ms"], elapsed_ms)

    @contextmanager
    def measure(self, name):
        """Measure a processing block, including failed blocks, in milliseconds.

        Uses ``perf_counter_ns`` for durations; do not subtract wall-clock input
        timestamps from this clock. Exceptions from the measured body propagate.
        """
        name = _name(name)
        with self._lock:
            if name not in self._stages and len(self._stages) >= self.max_stages:
                raise ValueError("latency stage capacity exceeded")
            # Reserve the stage before executing the body so another worker
            # cannot consume its cardinality slot while this timer is running.
            stage = self._stages.setdefault(name, {"values": deque(maxlen=self.capacity),
                "count": 0, "total_ms": 0.0, "max_ms": 0.0})
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter_ns()-start)/1_000_000
            with self._lock:
                # Reset discards timers that started in the previous stage
                # generation, rather than resurrecting deleted telemetry.
                if self._stages.get(name) is stage:
                    self.record(name, elapsed_ms)

    def snapshot(self):
        """Return an independent JSON-safe point-in-time measurement report."""
        with self._lock:
            copied = {name: (stage["count"], stage["total_ms"], stage["max_ms"], list(stage["values"]))
                      for name, stage in self._stages.items()}
        stages = {}
        for name, (count, total, maximum, values) in copied.items():
            if not count:
                continue
            ordered = sorted(values)
            def percentile(fraction):
                return ordered[max(0, math.ceil(fraction*len(ordered))-1)]
            stages[name] = {"count": count, "total_ms": total, "mean_ms": total/count,
                "max_ms": maximum, "window_count": len(ordered),
                "p50_ms": percentile(.5), "p95_ms": percentile(.95), "p99_ms": percentile(.99),
                "percentile_scope": "last_capacity_observations", "aggregate_scope": "since_last_reset",
                "percentile_method": "nearest_rank"}
        return {"schema_version": "0.1", "unit": "milliseconds", "clock": "perf_counter_ns",
                "capacity_per_stage": self.capacity, "max_stages": self.max_stages,
                "stage_count": len(stages), "stages": stages}

    def reset(self, name=None):
        """Reset one stage or all stages, discarding timers already in flight."""
        if name is not None:
            name = _name(name)
        with self._lock:
            if name is None:
                self._stages.clear()
            else:
                self._stages.pop(name, None)
