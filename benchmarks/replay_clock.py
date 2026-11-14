"""Explicit 1x replay mapping; original capture clocks are never overwritten."""
import time

from .recorded_source import _integer, _keys, digest, encoded
from humansense.capabilities import identifier
from humansense.engine import _window_ms


class ReplayClock:
    def __init__(self, manifest, *, max_drift_ns=50_000_000):
        self.max_drift_ns = _integer(max_drift_ns, 1, 1_000_000_000)
        self.wall_ns = time.time_ns()
        self.monotonic_ns = time.monotonic_ns()
        self.mapping = dict(schema_version=1, source_manifest_sha256=digest(manifest),
            recording_id=manifest['recording_id'], origin_capture_ns=manifest['origin_capture_ns'],
            duration_ns=manifest['duration_ns'], sdk_anchor_ms=self.wall_ns//1_000_000,
            wall_anchor_ns=self.wall_ns, monotonic_anchor_ns=self.monotonic_ns,
            scale_numerator=1, scale_denominator=1, rounding='floor_to_sdk_milliseconds',
            max_wall_monotonic_drift_ns=self.max_drift_ns)
        self.sha256 = digest(self.mapping)

    def check(self):
        wall = time.time_ns(); mono = time.monotonic_ns()
        drift = (wall-self.wall_ns)-(mono-self.monotonic_ns)
        if abs(drift) > self.max_drift_ns or mono < self.monotonic_ns:
            raise RuntimeError('replay clock discontinuity; original mapping cannot be refreshed')
        return mono

    def capture(self, offset_ns):
        offset = _integer(offset_ns, 0, self.mapping['duration_ns']-1)
        return dict(mapping_sha256=self.sha256,
            original_capture_ns=self.mapping['origin_capture_ns']+offset,
            scheduled_monotonic_ns=self.monotonic_ns+offset,
            sdk_timestamp_ms=self.mapping['sdk_anchor_ms']+offset//1_000_000,
            submillisecond_remainder_ns=offset%1_000_000)


def validate_tasks(tasks, duration_ns):
    if type(tasks) is not list or not 1 <= len(tasks) <= 64:
        raise ValueError('one to64 explicitly frozen snapshot tasks required')
    seen = set()
    for task in tasks:
        _keys(task, ('task_id','input','input_sha256','opens_ns','snapshot_ns','deadline_ns','window'))
        identifier(task['task_id'])
        if task['task_id'] in seen: raise ValueError('duplicate snapshot task')
        seen.add(task['task_id'])
        begin = _integer(task['opens_ns'], 0, duration_ns)
        snapshot = _integer(task['snapshot_ns'], 0, duration_ns)
        deadline = _integer(task['deadline_ns'], 1, duration_ns)
        if not begin <= snapshot < deadline: raise ValueError('ordered task window required')
        # Reuse the SDK's bounded JSON contract for original question/context input.
        from humansense.capabilities import bounded_json
        bounded_json(task['input'], max_bytes=65536)
        if type(task['input']) is not dict or digest(task['input']) != task['input_sha256']:
            raise ValueError('original task input digest mismatch')
        if _window_ms(task['window']) > 3600000:
            raise ValueError('snapshot Context window exceeds one hour')
    data = encoded(tasks)
    if len(data) > 4*1024**2: raise ValueError('snapshot task inventory exceeds4MiB')
    import json
    return json.loads(data)
