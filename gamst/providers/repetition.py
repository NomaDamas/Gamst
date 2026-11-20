"""Online repetition counting over structured per-frame channels (e.g. pose landmark coordinates).

The provider never sees pixels. A caller-owned light model (pose, optical flow, IMU) supplies a fixed-length
vector per capture; the provider picks the most energetic detrended channel, estimates the period by
autocorrelation and confirms one repetition per period-spaced peak. Each confirmed repetition is a
timestamped event so the count and the last repetition are answerable later from retained observations.
Measured on RepCountA test (pose input, native rate): MAE 0.429, OBO 0.437; at 1 Hz OBO drops to 0.172.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from .timeseries import autocorrelation_period, detrend, mean_interval_s, peak_indices, std


class RepetitionCounter:
    def __init__(self, *, channels: int, window_s: float = 20.0, min_period_s: float = 0.33,
                 max_period_s: float = 6.7, min_periodicity: float = 0.3, update_ms: int = 500,
                 max_gap_ms: int = 1000, modality: str = 'video'):
        if type(channels) is not int or not 1 <= channels <= 128:
            raise ValueError('channels must be 1..128')
        if not 0 < min_period_s < max_period_s <= 30 or not 2 * max_period_s <= window_s <= 120:
            raise ValueError('require 0 < min_period < max_period <= 30 s and window >= 2 x max_period (<= 120 s)')
        if not 0 <= min_periodicity < 1 or type(update_ms) is not int or not 50 <= update_ms <= 10000:
            raise ValueError('min_periodicity must be [0, 1) and update_ms 50..10000')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000 or modality not in ('video', 'imu'):
            raise ValueError('max_gap_ms must be 1..10000 and modality video or imu')
        self.channels, self.window_ms = channels, int(window_s * 1000)
        self.min_period_s, self.max_period_s = min_period_s, max_period_s
        self.min_periodicity, self.update_ms, self.max_gap_ms = min_periodicity, update_ms, max_gap_ms
        options = [channels, window_s, min_period_s, max_period_s, min_periodicity, update_ms, max_gap_ms, modality]
        version = '1.' + hashlib.sha256(json.dumps(options).encode()).hexdigest()[:16]
        self.spec = CapabilitySpec(
            'temporal.repetition_count', 'temporal.repetitions', modality, 'gamst.repetition_counter', version,
            (OutputField('count', 'integer'), OutputField('status', 'string'),
             OutputField('period_s', 'number', 's', required=False),
             OutputField('periodicity', 'number', required=False), OutputField('events', 'array')),
            'native', stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms, interpretation='estimate',
            devices=('cpu',), sensor_requirements=('fixed-length-channels', 'native-capture-rate'),
            license='Apache-2.0')
        self._lock = threading.RLock()
        self._closed = False
        self._clear()

    def _clear(self):
        self._owner = self._last = None
        self._buffer: deque = deque()
        self._count, self._last_peak_ms, self._next_update_ms = 0, None, None
        self._period_s = self._periodicity = None

    def reset(self):
        with self._lock:
            self._clear()

    def close(self):
        with self._lock:
            self._clear()
            self._closed = True

    def process(self, frame):
        with self._lock:
            if self._closed:
                raise RuntimeError('repetition provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _process(self, frame):
        if not isinstance(frame, InputFrame) or frame.modality != self.spec.modality or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError('structured InputFrame with sensor metadata required')
        identifier(frame.subject_id)
        ts = frame.timestamp_ms
        if type(ts) is not int or not 0 <= ts <= 2**53 - 1:
            raise ValueError('integer capture timestamp required')
        data = bounded_json(frame.data, max_bytes=16384)
        if type(data) is not dict or set(data) != {'channels'} or type(data['channels']) is not list \
                or len(data['channels']) != self.channels \
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in data['channels']):
            raise ValueError(f'data must be {{"channels": [{self.channels} finite numbers]}}')
        sensor = frame.sensor
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch)
        sequence = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        if owner == self._owner and self._last and (ts <= self._last[0] or sequence <= self._last[1]):
            raise ValueError('capture timestamps and sequence must strictly increase')
        if owner != self._owner or not self._last or ts - self._last[0] > self.max_gap_ms:
            self._clear()  # a gap breaks the repetition rhythm; never bridge it
        self._owner, self._last = owner, (ts, sequence)
        self._buffer.append((ts, [float(v) for v in data['channels']]))
        while self._buffer and ts - self._buffer[0][0] > self.window_ms:
            self._buffer.popleft()
        events = []
        span_ms = ts - self._buffer[0][0]
        status = 'warmup'
        if span_ms >= max(3000, 2 * self.min_period_s * 1000) and len(self._buffer) >= 8:
            status = 'idle'
            if self._next_update_ms is None or ts >= self._next_update_ms:
                self._next_update_ms = ts + self.update_ms
                events = self._update()
            if self._periodicity is not None and self._periodicity >= self.min_periodicity:
                status = 'counting'
        value = {'count': self._count, 'status': status, 'events': events}
        if self._period_s is not None:
            value['period_s'] = round(self._period_s, 4)
            value['periodicity'] = round(self._periodicity, 4)
        return self.spec.validate(value)

    def finish(self):
        """End of stream: confirm peaks in the final half period (no future samples will arrive)."""
        with self._lock:
            if len(self._buffer) < 8:
                return self.spec.validate({'count': self._count, 'status': 'finished', 'events': []})
            events = self._update(final=True)
            value = {'count': self._count, 'status': 'finished', 'events': events}
            if self._period_s is not None:
                value['period_s'] = round(self._period_s, 4)
                value['periodicity'] = round(self._periodicity, 4)
            return self.spec.validate(value)

    def _update(self, final=False):
        times = [t for t, _ in self._buffer]
        dt = mean_interval_s(times)
        win = max(3, int(self.max_period_s / dt))
        series = [detrend([vec[c] for _, vec in self._buffer], win) for c in range(self.channels)]
        signal = max(series, key=std)
        found = autocorrelation_period(signal, max(1, int(self.min_period_s / dt)), int(self.max_period_s / dt))
        if found is None:
            self._period_s = self._periodicity = None
            return []
        lag, score = found
        self._period_s, self._periodicity = lag * dt, score
        if score < self.min_periodicity:
            return []
        min_distance = max(1, int(0.6 * lag))
        confirmed_before = times[-1] if final else times[-1] - int(0.5 * lag * dt * 1000)
        events = []
        for i in peak_indices(signal, min_distance, 0.3 * std(signal)):
            t_peak = times[i]
            if t_peak > confirmed_before:
                continue
            if self._last_peak_ms is not None and t_peak <= self._last_peak_ms + 0.6 * self._period_s * 1000:
                continue  # already counted (windows overlap) or too close to the previous repetition
            self._count += 1
            self._last_peak_ms = t_peak
            events.append({'event': 'repetition', 'index': self._count, 'timestamp_ms': t_peak,
                           'period_s': round(self._period_s, 4)})
        return events
