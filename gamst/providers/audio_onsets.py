"""Sound onsets: when something starts to sound (a knock, a clap, a slam), from spectral flux; no model.

One band of the spectrum (default 60-250 Hz: the body of a knock, below most of the voice's energy) is
watched every 10 ms. An onset is a local peak of how fast its log magnitude rises, above what the last
second looked like. It says that a sound began and when, not what made it.
"""
from __future__ import annotations

from collections import deque
import statistics
import threading

from ..capabilities import CapabilitySpec, OutputField
from ..plugins import _pcm_floats

HOP_S, SIZE_S = .01, .032
HEARTBEAT_MS = 500


class AudioOnsets:
    def __init__(self, *, low_hz=60, high_hz=250, sensitivity=3., min_gap_ms=90, history_s=1., max_gap_ms=100,
                 min_dbfs=-45.):
        if not 20 <= low_hz < high_hz <= 20000:
            raise ValueError('require 20 <= low_hz < high_hz <= 20000')
        if not 1 <= sensitivity <= 20 or not 20 <= min_gap_ms <= 2000:
            raise ValueError('sensitivity must be 1..20 and min_gap_ms 20..2000')
        import numpy as np
        self.np = np
        self.low_hz, self.high_hz, self.sensitivity = float(low_hz), float(high_hz), float(sensitivity)
        self.min_dbfs = float(min_dbfs)      # what begins more quietly than this is the room, not a knock
        self.min_gap_ms, self.max_gap_ms = int(min_gap_ms), int(max_gap_ms)
        self.history = max(20, round(history_s / HOP_S))
        self.spec = CapabilitySpec(
            'audio.onsets', 'audio.onsets', 'audio', 'gamst.audio_onsets', f'1.{low_hz:g}-{high_hz:g}.s{sensitivity:g}',
            (OutputField('count', 'integer'), OutputField('events', 'array')), 'native', stateful=True,
            reset_on_gap=True, freshness_ms=1000, interpretation='measurement', devices=('cpu',), license='Apache-2.0')
        self._lock = threading.RLock()
        self._count = 0
        self._clear()

    def _clear(self):
        self._rate = self._end_ms = None
        self._samples = self.np.zeros(0)
        self._bands, self._flux, self._recent = deque(maxlen=3), deque(maxlen=self.history), deque(maxlen=7)
        self._last_ms, self._silent_ms = None, 0

    def reset(self):
        with self._lock:
            self._clear()

    def process(self, frame):
        with self._lock:
            values = self.np.asarray(_pcm_floats(frame.data), dtype=float)
            rate = frame.sample_rate
            start_ms = frame.timestamp_ms - 1000 * len(values) / rate
            if rate != self._rate or self._end_ms is None or abs(start_ms - self._end_ms) > self.max_gap_ms:
                self._clear()      # never a jump in level across missing sound
                self._rate = rate
            self._end_ms = frame.timestamp_ms
            self._samples = self.np.concatenate([self._samples, values])
            size, hop = max(64, round(SIZE_S * rate)), max(1, round(HOP_S * rate))
            window = self.np.hanning(size)
            low, high = (max(1, round(hz * size / rate)) for hz in (self.low_hz, self.high_hz))
            events, position = [], 0
            while position + size <= len(self._samples):
                piece = self._samples[position:position + size]
                level = 10 * self.np.log10(float((piece ** 2).mean()) + 1e-12)
                spectrum = self.np.abs(self.np.fft.rfft(piece * window))
                at_ms = self._end_ms - 1000 * (len(self._samples) - position - size / 2) / rate
                event = self._step(self.np.log(1e-3 + spectrum[low:max(high, low + 1) + 1]), at_ms, level)
                if event is not None:
                    events.append(event)
                position += hop
            self._samples = self._samples[position:]
            self._silent_ms = 0 if events else self._silent_ms + 1000 * len(values) / rate
            if not events and self._silent_ms < HEARTBEAT_MS:
                return None
            self._silent_ms = 0     # 'nothing began' is an answer too: what waits for this module need not wait on
            return self.spec.validate(dict(count=self._count, events=events))

    def _step(self, band, at_ms, level):
        rise = float(self.np.maximum(0, band - self._bands[0]).mean()) if len(self._bands) == 3 else 0.
        self._bands.append(band)
        self._recent.append((at_ms, rise, level))
        self._flux.append(rise)
        if len(self._recent) < 7 or len(self._flux) < 20:
            return None
        peak_ms, peak, _ = self._recent[3]    # decided 30 ms later, when it is known to be the local peak
        if peak < max(value for _, value, _ in self._recent) or max(item[2] for item in list(self._recent)[3:]) < self.min_dbfs:
            return None
        usual = statistics.median(self._flux)
        spread = statistics.median(abs(value - usual) for value in self._flux)
        if peak < usual + self.sensitivity * max(spread, .05) * 1.4826:
            return None
        if self._last_ms is not None and peak_ms - self._last_ms < self.min_gap_ms:
            return None
        self._last_ms = peak_ms
        self._count += 1
        return dict(event='audio.onset', index=self._count, timestamp_ms=round(peak_ms), strength=round(peak, 4))
