"""Rhythm: dominant frequency of a structured per-frame signal (heart, breath, tremor, machine bands).

Input per capture is either one scalar (`value`: e.g. chest motion, IMU magnitude, wheel brightness) or a
face-skin mean colour (`rgb`), which is converted to a pulse signal with the POS projection (Wang et al.
2017). Every `update_ms` the provider scans its band over the last `window_s` seconds with a Goertzel power
scan and reports the peak frequency, rate per minute and spectral purity. When the stream's sample rate
cannot represent the band (rate < 2.5 x band top) it reports `undersampled` instead of a number: a 1 FPS
observer cannot measure a pulse at all. Optional per-minute thresholds emit hysteretic crossing events.
Measured on MCD-rPPG phone video (40 videos, 60 s): 50 % within +-5 bpm of synchronous contact PPG.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame
from .timeseries import band_peak, detrend, mean_interval_s

BANDS = {'heart': (0.7, 3.0), 'breath': (0.1, 0.5), 'tremor': (3.0, 12.0)}


def pos_pulse(rgb: list[list[float]], rate_hz: float) -> list[float]:
    """Plane-orthogonal-to-skin projection over 1.6 s sliding windows (overlap-added)."""
    n, w = len(rgb), max(2, int(1.6 * rate_hz))
    h = [0.0] * n
    for t in range(0, max(0, n - w) + 1):
        seg = rgb[t:t + w]
        mean = [sum(p[c] for p in seg) / len(seg) or 1e-9 for c in range(3)]
        cn = [[p[c] / mean[c] for c in range(3)] for p in seg]
        s1 = [c[1] - c[2] for c in cn]
        s2 = [-2 * c[0] + c[1] + c[2] for c in cn]
        m1, m2 = sum(s1) / len(s1), sum(s2) / len(s2)
        d1 = math.sqrt(sum((x - m1) ** 2 for x in s1) / len(s1))
        d2 = math.sqrt(sum((x - m2) ** 2 for x in s2) / len(s2)) or 1e-9
        q = [a + (d1 / d2) * b for a, b in zip(s1, s2)]
        mq = sum(q) / len(q)
        for k, v in enumerate(q):
            h[t + k] += v - mq
    return h


class RhythmEstimator:
    def __init__(self, *, band: str | tuple[float, float] = 'heart', input_kind: str = 'value',
                 window_s: float = 16.0, update_ms: int = 1000, min_purity: float = 0.15,
                 above_per_minute: float | None = None, below_per_minute: float | None = None,
                 hysteresis_per_minute: float = 3.0, max_gap_ms: int = 1000, modality: str = 'video'):
        low, high = BANDS[band] if isinstance(band, str) else band
        if not 0 < low < high <= 100:
            raise ValueError('band must satisfy 0 < low < high <= 100 Hz')
        if input_kind not in ('value', 'rgb') or modality not in ('video', 'imu', 'audio', 'environment'):
            raise ValueError('input_kind must be value or rgb; unsupported modality')
        if not 2 <= window_s <= 120 or type(update_ms) is not int or not 100 <= update_ms <= 60000:
            raise ValueError('window_s must be 2..120 s and update_ms 100..60000')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000 or not 0 <= min_purity < 1:
            raise ValueError('max_gap_ms must be 1..10000 and min_purity [0, 1)')
        self.low, self.high, self.input_kind = float(low), float(high), input_kind
        self.window_ms, self.update_ms, self.min_purity, self.max_gap_ms = int(window_s * 1000), update_ms, min_purity, max_gap_ms
        self.above, self.below, self.hyst = above_per_minute, below_per_minute, hysteresis_per_minute
        name = band if isinstance(band, str) else 'custom'
        options = [self.low, self.high, input_kind, window_s, update_ms, min_purity, above_per_minute,
                   below_per_minute, hysteresis_per_minute, max_gap_ms, modality]
        version = '1.' + hashlib.sha256(json.dumps(options).encode()).hexdigest()[:16]
        self.spec = CapabilitySpec(
            'temporal.rhythm.' + name, 'temporal.rhythm', modality, 'gamst.rhythm_estimator', version,
            (OutputField('status', 'string'), OutputField('band_hz', 'array'),
             OutputField('frequency_hz', 'number', 'Hz', required=False),
             OutputField('per_minute', 'number', '1/min', required=False),
             OutputField('purity', 'number', required=False),
             OutputField('sample_rate_hz', 'number', 'Hz', required=False), OutputField('events', 'array')),
            'window', window_ms=self.window_ms, stateful=True, reset_on_gap=True, freshness_ms=max(max_gap_ms, update_ms * 2),
            interpretation='estimate', devices=('cpu',),
            sensor_requirements=('sample-rate-above-2.5x-band-top',) + (('face-skin-mean-rgb',) if input_kind == 'rgb' else ()),
            license='Apache-2.0')
        self._lock = threading.RLock()
        self._closed = False
        self._clear()

    def _clear(self):
        self._owner = self._last = None
        self._buffer: deque = deque()
        self._next_update_ms = None
        self._result: dict | None = None
        self._above_active = self._below_active = False

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
                raise RuntimeError('rhythm provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _sample(self, data):
        if self.input_kind == 'value':
            if type(data) is not dict or set(data) != {'value'} or type(data['value']) not in (int, float) \
                    or not math.isfinite(data['value']):
                raise ValueError('data must be {"value": finite number}')
            return float(data['value'])
        rgb = data.get('rgb') if type(data) is dict and set(data) == {'rgb'} else None
        if type(rgb) is not list or len(rgb) != 3 or any(type(v) not in (int, float) or not 0 < v <= 65535 for v in rgb):
            raise ValueError('data must be {"rgb": [r, g, b]} positive skin means')
        return [float(v) for v in rgb]

    def _process(self, frame):
        if not isinstance(frame, InputFrame) or frame.modality != self.spec.modality or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError('structured InputFrame with sensor metadata required')
        identifier(frame.subject_id)
        ts = frame.timestamp_ms
        if type(ts) is not int or not 0 <= ts <= 2**53 - 1:
            raise ValueError('integer capture timestamp required')
        sample = self._sample(bounded_json(frame.data, max_bytes=1024))
        sensor = frame.sensor
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch)
        sequence = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        if owner == self._owner and self._last and (ts <= self._last[0] or sequence <= self._last[1]):
            raise ValueError('capture timestamps and sequence must strictly increase')
        if owner != self._owner or not self._last or ts - self._last[0] > self.max_gap_ms:
            self._clear()
        self._owner, self._last = owner, (ts, sequence)
        self._buffer.append((ts, sample))
        while self._buffer and ts - self._buffer[0][0] > self.window_ms:
            self._buffer.popleft()
        events = []
        if self._next_update_ms is None or ts >= self._next_update_ms:
            self._next_update_ms = ts + self.update_ms
            events = self._update(ts)
        value = {'status': 'warmup', 'band_hz': [self.low, self.high], 'events': events}
        if self._result:
            value.update(self._result)
        return self.spec.validate(value)

    def _update(self, ts):
        if len(self._buffer) < 8 or ts - self._buffer[0][0] < 0.75 * self.window_ms:
            return []
        times = [t for t, _ in self._buffer]
        rate = 1.0 / mean_interval_s(times)
        result = {'sample_rate_hz': round(rate, 3)}
        if rate < 2.5 * self.high:
            result['status'] = 'undersampled'
            self._result = result
            return []
        if self.input_kind == 'rgb':
            signal = pos_pulse([s for _, s in self._buffer], rate)
        else:
            signal = [s for _, s in self._buffer]
        signal = detrend(signal, max(3, int(2 * rate / self.low)))  # 2 band-bottom periods: keeps the band intact
        peak = band_peak(signal, rate, self.low, self.high, step_hz=max(0.005, (self.high - self.low) / 400))
        if peak is None or peak[1] < self.min_purity:
            result['status'] = 'no_rhythm'
            self._result = result
            return []
        f, purity = peak
        per_minute = f * 60.0
        result.update(status='measuring', frequency_hz=round(f, 4), per_minute=round(per_minute, 2), purity=round(purity, 4))
        self._result = result
        events = []
        if self.above is not None:
            if not self._above_active and per_minute > self.above:
                self._above_active = True
                events.append({'event': 'rate_above', 'timestamp_ms': ts, 'per_minute': round(per_minute, 2), 'threshold': self.above})
            elif self._above_active and per_minute < self.above - self.hyst:
                self._above_active = False
                events.append({'event': 'rate_back_below', 'timestamp_ms': ts, 'per_minute': round(per_minute, 2), 'threshold': self.above})
        if self.below is not None:
            if not self._below_active and per_minute < self.below:
                self._below_active = True
                events.append({'event': 'rate_below', 'timestamp_ms': ts, 'per_minute': round(per_minute, 2), 'threshold': self.below})
            elif self._below_active and per_minute > self.below + self.hyst:
                self._below_active = False
                events.append({'event': 'rate_back_above', 'timestamp_ms': ts, 'per_minute': round(per_minute, 2), 'threshold': self.below})
        return events
