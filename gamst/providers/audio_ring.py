"""Audio helpers shared by the speech modules: a capture-time ring buffer, a sample clock and a resampler.

No model. Pipeline audio frames carry the capture time of their *last* sample (``timestamp_ms``), like
``Gamst.push_audio`` and :class:`gamst.inputs.AudioFileInput`; the helpers here keep that convention.

- :class:`AudioRing` keeps the last ``seconds`` of mono float32 audio and cuts a span by capture time
  (an utterance to align, a segment to embed, a block to separate).
- :class:`HopClock` re-cuts chunks of any size into fixed hops with a gap-free sample clock, which is
  what streaming ASR engines that check hop continuity (``QwenStreamingASR``) need; it reports a
  discontinuity when the capture clock and the sample count disagree by more than ``max_drift_ms``.
- :func:`resample` is a windowed-sinc polyphase rational resampler (NumPy only), so a module that needs
  another rate (SAM-Audio needs 48 kHz) converts explicitly instead of a model resampling silently.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import math


def _np():
    import numpy as np
    return np


def mono_float(data):
    """Mono float32 samples from a frame's data (float array, int16 array or PCM16 bytes)."""
    np = _np()
    if isinstance(data, (bytes, bytearray, memoryview)):
        values = np.frombuffer(bytes(data), dtype='<i2').astype(np.float32) / 32768.
    else:
        values = np.asarray(data)
        if values.dtype == np.int16:
            values = values.astype(np.float32) / 32768.
        values = values.astype(np.float32, copy=False)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError('mono audio required; mix channels explicitly')
    if not np.isfinite(values).all():
        raise ValueError('audio samples must be finite')
    return values


class AudioRing:
    """The last ``seconds`` of one mono stream, addressable by capture time in milliseconds."""

    def __init__(self, sample_rate=16000, seconds=30., max_drift_ms=250):
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError('sample_rate must be a positive integer')
        self.sample_rate, self.seconds, self.max_drift_ms = sample_rate, float(seconds), float(max_drift_ms)
        self.reset()

    def reset(self):
        self._chunks = deque()   # (start_ms, end_ms, samples)
        self._samples = 0

    @property
    def end_ms(self):
        return self._chunks[-1][1] if self._chunks else None

    @property
    def start_ms(self):
        return self._chunks[0][0] if self._chunks else None

    def append(self, samples, end_ms, sample_rate=None):
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f'ring holds {self.sample_rate} Hz audio, got {sample_rate} Hz')
        """Add a chunk; returns True when it does not continue the previous one (history restarted).

        A live callback's arrival time jitters, so a chunk within ``max_drift_ms`` of the sample clock
        continues it; a larger disagreement is a real gap or another clock.
        """
        samples = mono_float(samples)
        if not len(samples):
            return False
        duration = 1000. * len(samples) / self.sample_rate
        start_ms, gap = end_ms - duration, False
        if self._chunks:
            if abs(start_ms - self._chunks[-1][1]) <= self.max_drift_ms:
                start_ms = self._chunks[-1][1]
            else:
                gap = True
                self.reset()
        self._chunks.append((start_ms, start_ms + duration, samples))
        self._samples += len(samples)
        limit = int(self.seconds * self.sample_rate)
        while self._samples - len(self._chunks[0][2]) >= limit:
            self._samples -= len(self._chunks.popleft()[2])
        return gap

    def span(self, start_ms, end_ms):
        """Samples captured in [start_ms, end_ms) that are still held."""
        np = _np()
        parts = []
        for chunk_start, chunk_end, samples in self._chunks:
            if chunk_end <= start_ms or chunk_start >= end_ms:
                continue
            first = max(0, int(round((start_ms - chunk_start) * self.sample_rate / 1000)))
            last = min(len(samples), int(round((end_ms - chunk_start) * self.sample_rate / 1000)))
            if last > first:
                parts.append(samples[first:last])
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def covers(self, start_ms, end_ms):
        return bool(self._chunks) and self.start_ms <= start_ms + 1 and self.end_ms >= end_ms - 1


class HopClock:
    """Chunks of any length -> fixed hops on a gap-free sample clock anchored at capture time."""

    def __init__(self, sample_rate=16000, hop_ms=20, max_drift_ms=250):
        if (sample_rate * hop_ms) % 1000:
            raise ValueError('hop_ms must be a whole number of samples')
        self.sample_rate, self.hop_ms, self.max_drift_ms = sample_rate, hop_ms, max_drift_ms
        self.hop = sample_rate * hop_ms // 1000
        self.reset()

    def reset(self):
        self._pending = None
        self._next_end_ms = None   # capture time of the next hop's last sample on the sample clock

    def push(self, samples, end_ms):
        """Returns (hops, discontinuity): hops are (end_ms, samples[hop]) on the sample clock."""
        np = _np()
        samples = mono_float(samples)
        discontinuity = False
        pending_count = 0 if self._pending is None else len(self._pending)
        if self._next_end_ms is not None:
            # Capture time of the first new sample per the sample clock vs per this frame.
            expected_end = self._next_end_ms - self.hop_ms + 1000. * (pending_count + len(samples)) / self.sample_rate
            if abs(expected_end - end_ms) > self.max_drift_ms:
                discontinuity = True
                self.reset()
                pending_count = 0
        buffer = samples if not pending_count else np.concatenate([self._pending, samples])
        if self._next_end_ms is None:
            whole = len(buffer) // self.hop
            if not whole:
                self._pending = buffer
                return [], discontinuity
            # The last whole hop ends where the buffer's last whole-hop sample was captured.
            tail = len(buffer) - whole * self.hop
            first_end = end_ms - 1000. * tail / self.sample_rate - (whole - 1) * self.hop_ms
            self._next_end_ms = int(round(first_end))
        hops = []
        offset = 0
        while len(buffer) - offset >= self.hop:
            hops.append((self._next_end_ms, buffer[offset:offset + self.hop].copy()))
            offset += self.hop
            self._next_end_ms += self.hop_ms
        self._pending = buffer[offset:] if offset < len(buffer) else None
        return hops, discontinuity


def _kernel(up, down, half_width=16, beta=8.6):
    np = _np()
    cutoff = 1. / max(up, down)
    length = 2 * half_width * max(up, down) + 1
    n = np.arange(length) - (length - 1) / 2
    taps = cutoff * np.sinc(cutoff * n) * np.kaiser(length, beta)
    return taps * up / taps.sum()  # unity DC gain after zero stuffing


def resample(samples, source_rate, target_rate):
    """Rational-ratio resampling with a Kaiser-windowed sinc low-pass; output length = round(n * ratio)."""
    np = _np()
    samples = mono_float(samples)
    if source_rate == target_rate:
        return samples.copy()
    ratio = Fraction(int(target_rate), int(source_rate))
    up, down = ratio.numerator, ratio.denominator
    if up > 64 or down > 64:
        raise ValueError('resample supports ratios with numerator and denominator up to 64')
    taps = _kernel(up, down)
    stuffed = np.zeros(len(samples) * up, np.float64)
    stuffed[::up] = samples
    filtered = np.convolve(stuffed, taps, mode='full')
    delay = (len(taps) - 1) // 2
    filtered = filtered[delay:delay + len(stuffed)]
    out = filtered[::down]
    expected = int(round(len(samples) * up / down))
    return np.asarray(out[:expected], dtype=np.float32)


def rms_dbfs(samples):
    np = _np()
    samples = mono_float(samples)
    if not len(samples):
        return -math.inf
    power = float(np.mean(np.square(samples, dtype=np.float64)))
    return 10 * math.log10(power) if power > 0 else -math.inf
