"""Out-of-plane spin counting from per-frame apparent ellipse shape.

A coin, disc or card turning about an axis that lies in its own plane shows the
camera an ellipse whose minor/major ratio r follows |cos(angle)|. Its square,
r^2 = (1 + cos(2 * angle)) / 2, is a sinusoid at the half-turn rate. The
provider never sees pixels: a caller-owned tracker supplies each object's
fitted ellipse axes per capture. During a spin episode the provider fits that
sinusoid in short overlapping windows (least squares over a frequency grid,
parabolic refinement) and integrates the half-turn rate over the observed span:

    turns = (1/2) * sum(half_turn_rate(window) * window_duration)

so a coin that slows down is still followed. Direction is not observable.

Sampling decides whether a count exists. Uniform samples cannot tell a spin
from its alias, so the guard is physical, not statistical: the half-turn rate
must stay below the Nyquist limit (capture_fps / 2), i.e. turns per second
below capture_fps / 4. ``expected_max_rev_s`` declares the fastest spin the
application must handle. A hand-tossed coin turns about 36-40 times per second
(Diaconis, Holmes and Montgomery 2007), the default 40 rev/s therefore needs
at least 160 fps, and 240 fps leaves a margin. Below that the count is
withheld with status ``undersampled``; GAMST never reports an aliased number.

Episodes are spans where the ratio swings by at least ``min_swing``; each ends
with an ``object.spin.ended`` event carrying turns (or null with the reason),
measured rate, duration and fit quality.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, bounded_json, identifier
from ..plugins import InputFrame


def _fit(samples):
    """Best single sinusoid (frequency Hz, explained variance) for [(t_s, value)]."""
    n = len(samples)
    if n < 6:
        return None, 0.
    t0 = samples[0][0]
    times = [t - t0 for t, _ in samples]
    mean = sum(v for _, v in samples) / n
    values = [v - mean for _, v in samples]
    energy = sum(v * v for v in values)
    span = times[-1]
    if energy <= 1e-9 or span <= 0:
        return None, 0.
    dt = span / (n - 1)
    nyquist = 0.5 / dt
    low, step = max(1.0 / span, 0.5), min(0.5, 0.25 / span)

    def power(freq):
        c = s = 0.
        for t, v in zip(times, values):
            angle = 2 * math.pi * freq * t
            c += v * math.cos(angle)
            s += v * math.sin(angle)
        return c * c + s * s

    grid = []
    freq = low
    while freq < nyquist * .98:
        grid.append(freq)
        freq += step
    if len(grid) < 3:
        return None, 0.
    powers = [power(f) for f in grid]
    best = max(range(len(grid)), key=powers.__getitem__)
    freq = grid[best]
    if 0 < best < len(grid) - 1:
        a, b, c = powers[best - 1], powers[best], powers[best + 1]
        denom = a - 2 * b + c
        if denom < 0:
            freq += step * .5 * (a - c) / denom
    # Fraction of variance explained by the fitted sinusoid (2/n * power ~ amplitude^2 * n/2).
    return freq, min(1., 2 * power(freq) / n / energy)


class ObjectSpinCounter:
    def __init__(self, *, max_tracks: int = 16, max_gap_ms: int = 300, expected_max_rev_s: float = 40.0,
                 min_swing: float = 0.35, activity_ms: int = 60, idle_ms: int = 200, window_ms: int = 250,
                 min_fit: float = 0.3):
        if type(max_tracks) is not int or not 1 <= max_tracks <= 64:
            raise ValueError('max_tracks must be 1..64')
        for name, value, upper in (('max_gap_ms', max_gap_ms, 5000), ('activity_ms', activity_ms, 2000),
                                   ('idle_ms', idle_ms, 5000), ('window_ms', window_ms, 5000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{name} must be 1..{upper}')
        if not 0 < expected_max_rev_s <= 1000 or not 0 < min_swing <= 1 or not 0 <= min_fit < 1:
            raise ValueError('expected_max_rev_s 0..1000, min_swing 0..1, min_fit 0..1')
        self.max_tracks, self.max_gap_ms = max_tracks, max_gap_ms
        self.expected_max_rev_s, self.min_swing, self.min_fit = float(expected_max_rev_s), float(min_swing), float(min_fit)
        self.activity_ms, self.idle_ms, self.window_ms = activity_ms, idle_ms, window_ms
        options = [max_tracks, max_gap_ms, expected_max_rev_s, min_swing, activity_ms, idle_ms, window_ms, min_fit]
        self.spec = self.capability(version='2.' + hashlib.sha256(json.dumps(options).encode()).hexdigest()[:16],
                                    max_gap_ms=max_gap_ms)
        self._lock = threading.RLock()
        self.closed = False
        self._tracks: dict[str, dict] = {}
        self._owner = None

    @staticmethod
    def capability(*, version: str = 'installed', max_gap_ms: int = 300):
        identifier(version, 'model version')
        return CapabilitySpec(
            'vision.object_spin', 'vision.object_spin', 'video', 'gamst.object_spin', version,
            (OutputField('objects', 'array'), OutputField('events', 'array')), 'frame',
            stateful=True, reset_on_gap=True, freshness_ms=max_gap_ms, interpretation='measurement',
            devices=('cpu',), sensor_requirements=('tracked-object-ellipse-axes', 'capture-timestamps'),
            license='Apache-2.0')

    def reset(self):
        with self._lock:
            self._tracks.clear()
            self._owner = None

    def close(self):
        with self._lock:
            self.reset()
            self.closed = True

    def process(self, frame: InputFrame):
        with self._lock:
            if self.closed:
                raise RuntimeError('object spin counter is closed')
            if not isinstance(frame, InputFrame) or frame.modality != 'video' or not isinstance(frame.sensor, SensorMetadata):
                raise ValueError('structured video InputFrame with sensor metadata required')
            stamp = frame.timestamp_ms
            if type(stamp) is not int or stamp < 0:
                raise ValueError('integer capture timestamp required')
            data = bounded_json(frame.data, max_bytes=64 * 1024)
            if not isinstance(data, dict) or set(data) != {'objects'} or type(data['objects']) is not list:
                raise ValueError('data must contain only an objects list')
            if len(data['objects']) > self.max_tracks:
                raise ValueError('too many tracked objects')
            owner = (frame.subject_id, frame.sensor.source_id, frame.sensor.sensor_id, frame.sensor.stream_epoch)
            if owner != self._owner:
                self._tracks.clear()
                self._owner = owner
            events, outputs, seen = [], [], set()
            for item in data['objects']:
                track_id, ratio = self._read(item)
                if track_id in seen:
                    raise ValueError('duplicate track_id')
                seen.add(track_id)
                track = self._tracks.get(track_id)
                if track is not None and stamp <= track['last_ts']:
                    raise ValueError('capture timestamps must advance per track')
                if track is not None and stamp - track['last_seen'] > self.max_gap_ms:
                    events += self._end(track_id, track, 'gap')
                    track = None
                if track is None:
                    track = self._tracks[track_id] = dict(last_ts=stamp, last_seen=stamp, intervals=deque(maxlen=120),
                                                          recent=deque(), episode=None)
                else:
                    track['intervals'].append(stamp - track['last_ts'])
                track['last_ts'] = stamp
                if ratio is not None:
                    track['last_seen'] = stamp
                    events += self._step(track_id, track, stamp, ratio)
                outputs.append(self._output(track_id, track, ratio))
            for track_id in list(self._tracks):
                track = self._tracks[track_id]
                if track_id not in seen and stamp - track['last_seen'] > self.max_gap_ms:
                    events += self._end(track_id, track, 'lost')
                    del self._tracks[track_id]
            return self.spec.validate({'objects': outputs, 'events': events})

    @staticmethod
    def _read(item):
        if not isinstance(item, dict) or set(item) - {'track_id', 'axes', 'observed'}:
            raise ValueError('spin object contains unsupported fields')
        track_id = item.get('track_id')
        if not isinstance(track_id, str) or not 1 <= len(track_id) <= 128:
            raise ValueError('track_id is required')
        if item.get('observed', True) is False or item.get('axes') is None:
            return track_id, None
        axes = item['axes']
        if (type(axes) is not list or len(axes) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in axes)):
            raise ValueError('axes must be [major, minor] finite non-negative pixels')
        major, minor = max(axes), min(axes)
        return track_id, (minor / major if major > 0 else None)

    @staticmethod
    def _fps(track):
        if not track['intervals']:
            return None
        return 1000 / (sum(track['intervals']) / len(track['intervals']))

    def _step(self, track_id, track, stamp, ratio):
        recent = track['recent']
        recent.append((stamp, ratio))
        interval = 1000 / self._fps(track) if self._fps(track) else 0
        # At least four samples per activity window and idle period, whatever the camera rate.
        span, idle = max(self.activity_ms, 4 * interval), max(self.idle_ms, 6 * interval)
        while recent and stamp - recent[0][0] > span:
            recent.popleft()
        values = [r for _, r in recent]
        active = len(values) >= 3 and max(values) - min(values) >= self.min_swing
        episode = track['episode']
        events = []
        if episode is None and active:
            episode = track['episode'] = dict(samples=[(t, r) for t, r in recent], last_active=stamp,
                                              done_ms=0., half_turns=0., fits=[])
            events.append(dict(type='object.spin.started', track_id=track_id, timestamp_ms=recent[0][0]))
        elif episode is not None:
            episode['samples'].append((stamp, ratio))
            if active:
                episode['last_active'] = stamp
            elif stamp - episode['last_active'] > idle:
                return events + self._end(track_id, track, 'stopped')
            self._integrate(episode, final=False)
        return events

    def _integrate(self, episode, *, final):
        """Fit consecutive windows; keep the open tail for the next call unless final."""
        samples = episode['samples']
        while samples:
            start = samples[0][0]
            window = [(t, r) for t, r in samples if t - start <= self.window_ms]
            if not final and samples[-1][0] - start < self.window_ms:
                break
            freq, quality = _fit([(t / 1000, r * r) for t, r in window])
            span = window[-1][0] - start
            if freq is not None and span > 0:
                episode['half_turns'] += freq * span / 1000
                episode['fits'].append((quality, span))
            episode['done_ms'] += span
            if final or len(window) == len(samples):
                episode['samples'] = []
                break
            # Windows share their boundary sample so no interval is lost.
            episode['samples'] = samples = samples[len(window) - 1:]

    def _quality(self, episode):
        total = sum(span for _, span in episode['fits'])
        return sum(q * span for q, span in episode['fits']) / total if total else 0.

    def _verdict(self, track, episode):
        fps = self._fps(track)
        limit = None if fps is None else fps / 4
        if limit is None or limit < self.expected_max_rev_s:
            return 'undersampled', limit
        if self._quality(episode) < self.min_fit:
            return 'insufficient', limit
        return 'counting', limit

    def _end(self, track_id, track, reason):
        episode = track['episode']
        if episode is None:
            return []
        self._integrate(episode, final=True)
        track['episode'] = None
        status, limit = self._verdict(track, episode)
        duration = episode['done_ms']
        counting = status == 'counting' and duration > 0
        return [dict(type='object.spin.ended', track_id=track_id, timestamp_ms=track['last_seen'], reason=reason,
                     status=status, turns=round(episode['half_turns'] / 2, 1) if counting else None,
                     rev_s=round(episode['half_turns'] / 2 / (duration / 1000), 2) if counting else None,
                     duration_ms=round(duration), fit_quality=round(self._quality(episode), 3),
                     capture_fps=None if self._fps(track) is None else round(self._fps(track), 1),
                     max_countable_rev_s=None if limit is None else round(limit, 1),
                     expected_max_rev_s=self.expected_max_rev_s)]

    def _output(self, track_id, track, ratio):
        episode = track['episode']
        fps = self._fps(track)
        base = dict(track_id=track_id, ratio=None if ratio is None else round(ratio, 4),
                    capture_fps=None if fps is None else round(fps, 1),
                    max_countable_rev_s=None if fps is None else round(fps / 4, 1))
        if episode is None:
            return dict(base, status='idle', turns=None, rev_s=None, duration_ms=0)
        status, _ = self._verdict(track, episode)
        if status == 'insufficient' and not episode['fits']:
            status = 'spinning'  # first window still filling
        counting = status in ('counting', 'spinning') and episode['done_ms'] > 0
        return dict(base, status=status, duration_ms=round(episode['done_ms']),
                    turns=round(episode['half_turns'] / 2, 1) if counting else None,
                    rev_s=round(episode['half_turns'] / 2 / (episode['done_ms'] / 1000), 2) if counting else None)
