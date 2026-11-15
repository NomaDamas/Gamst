"""Passive connection-local context delivery profiling, never LLM calibration."""
from collections import deque
import math


class ContextCadenceProfile:
    def __init__(self, interval_ms=None):
        if interval_ms is not None and (type(interval_ms) is not int or not 50 <= interval_ms <= 60000):
            raise ValueError('interval_ms must be 50..60000')
        self.interval_ms = interval_ms
        self.running = False
        self.samples = deque(maxlen=64)
        self.gaps = deque(maxlen=64)
        self.completed = 0
        self.last_completion = None
        self.failures = 0

    def record(self, work_ms, completion_ms):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (work_ms, completion_ms)):
            raise ValueError('profile samples must be finite nonnegative milliseconds')
        if self.last_completion is not None and completion_ms < self.last_completion:
            raise ValueError('completion clock moved backwards')
        if self.last_completion is not None:
            self.gaps.append(completion_ms-self.last_completion)
        self.last_completion = completion_ms
        self.samples.append(work_ms)
        self.completed += 1

    @staticmethod
    def _summary(values):
        if not values:
            return None
        ordered = sorted(values)
        return dict(p50_ms=ordered[math.ceil(len(ordered)*.5)-1],
                    p95_ms=ordered[math.ceil(len(ordered)*.95)-1])

    def snapshot(self):
        work = self._summary(self.samples)
        enough = len(self.samples) >= 8
        desired = max(50, math.ceil(work['p95_ms'])) if enough else None
        return dict(requested_interval_ms=self.interval_ms,
            applied_interval_ms=self.interval_ms,
            recommended_interval_ms=min(60000, desired) if enough else None,
            recommendation_exceeds_supported_range=desired > 60000 if enough else False,
            recommendation_basis='heuristic: idle at least rolling p95 generation+send time; minimum 8 completions',
            interval_semantics='idle delay after completed send; not a guaranteed delivery period',
            configured=self.interval_ms is not None, running=self.running,
            completed=self.completed, failures=self.failures, retained_samples=len(self.samples),
            work=work, observed_completion_intervals=self._summary(self.gaps),
            consumer_comprehension='unknown', consumer_input_capacity='unknown',
            automatic_application=False, probes_sent=0)
