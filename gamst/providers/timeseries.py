"""Dependency-free time-series primitives shared by the temporal capability providers.

Inputs are short, bounded, near-uniformly sampled windows (at most a few thousand samples). Nothing here
allocates beyond O(n) or runs a model.
"""
from __future__ import annotations

import math


def mean_interval_s(times_ms: list[int]) -> float:
    if len(times_ms) < 2:
        raise ValueError('at least two samples are required')
    return (times_ms[-1] - times_ms[0]) / 1000.0 / (len(times_ms) - 1)


def detrend(values: list[float], window: int) -> list[float]:
    """Subtract a centred moving average (removes posture drift, lighting drift, DC)."""
    n = len(values)
    if n == 0:
        return []
    half = max(1, window // 2)
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    out = []
    for i, v in enumerate(values):
        a, b = max(0, i - half), min(n, i + half + 1)
        out.append(v - (prefix[b] - prefix[a]) / (b - a))
    return out


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def autocorrelation_period(values: list[float], min_lag: int, max_lag: int) -> tuple[int, float] | None:
    """First autocorrelation peak after the first zero crossing, within [min_lag, max_lag] samples."""
    n = len(values)
    max_lag = min(max_lag, n - 2)
    if n < 4 or min_lag < 1 or max_lag <= min_lag:
        return None
    energy = sum(v * v for v in values)
    if energy <= 0:
        return None
    ac = [sum(values[i] * values[i + lag] for i in range(n - lag)) / energy for lag in range(max_lag + 1)]
    crossing = next((lag for lag in range(1, max_lag + 1) if ac[lag] < 0), None)
    if crossing is None:
        return None
    start = max(min_lag, crossing)
    if start > max_lag:
        return None
    lag = max(range(start, max_lag + 1), key=lambda k: ac[k])
    return lag, ac[lag]


def band_peak(values: list[float], rate_hz: float, low_hz: float, high_hz: float,
              step_hz: float = 0.01) -> tuple[float, float] | None:
    """Goertzel power scan of [low, high]; returns (peak frequency, purity).

    Purity is the share of band power within +-5 % of the band width around the peak, so it does not depend
    on the scan step (a pure tone -> ~1, white noise -> ~0.1)."""
    n = len(values)
    if n < 8 or rate_hz <= 0 or high_hz >= rate_hz / 2 or low_hz <= 0 or high_hz <= low_hz:
        return None
    mean = sum(values) / n
    xs = [v - mean for v in values]
    powers = []
    f = low_hz
    while f <= high_hz + 1e-9:
        w = 2 * math.pi * f / rate_hz
        coeff = 2 * math.cos(w)
        s1 = s2 = 0.0
        for x in xs:
            s1, s2 = x + coeff * s1 - s2, s1
        powers.append((f, s1 * s1 + s2 * s2 - coeff * s1 * s2))
        f += step_hz
    total = sum(p for _, p in powers)
    if total <= 0:
        return None
    best = max(powers, key=lambda item: item[1])
    half = 0.05 * (high_hz - low_hz)
    near = sum(p for f, p in powers if abs(f - best[0]) <= half)
    return best[0], near / total


def peak_indices(values: list[float], min_distance: int, prominence: float) -> list[int]:
    """Local maxima at least `min_distance` apart whose height above the neighbouring minima exceeds
    `prominence`; the larger peak wins when two candidates are too close."""
    n = len(values)
    candidates = [i for i in range(1, n - 1) if values[i] > values[i - 1] and values[i] >= values[i + 1]]
    kept: list[int] = []
    for i in sorted(candidates, key=lambda k: values[k], reverse=True):
        if any(abs(i - j) < min_distance for j in kept):
            continue
        left = min(values[max(0, i - min_distance):i + 1])
        right = min(values[i:min(n, i + min_distance + 1)])
        if values[i] - max(left, right) >= prominence:
            kept.append(i)
    return sorted(kept)
