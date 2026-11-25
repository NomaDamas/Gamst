"""Voice-style channels: how something is said, as numbers an exemplar can be matched against.

Consumes ``audio.prosody`` feature windows (``gamst.providers.prosody.AudioProsody``: 80 ms windows
every 20 ms, DSP only) and emits one channel vector per ``step_ms`` (default 100 ms):

- ``pitch_st``: voiced pitch in semitones relative to the speaker's running median (intonation
  contour, independent of whether the voice is high or low);
- ``level_db``: level in dB relative to the running median level (shouting, whispering);
- ``voiced``: share of windows with a periodic (voiced) signal;
- ``periodicity``: mean periodicity score;
- ``brightness``: log2 of the spectral centroid relative to its running median.

Channels are scaled to roughly [-1, 1] so :class:`gamst.providers.pose_exemplar.ExemplarEvents`
can compare them with the same distance as pose channels. No emotion, identity or meaning is
inferred; an exemplar is the owner's recording of the way of speaking to count.
"""
from __future__ import annotations

from collections import deque
import math
import statistics

from ..capabilities import CapabilitySpec, OutputField

CHANNELS = ('pitch_st', 'level_db', 'voiced', 'periodicity', 'brightness')


class VoiceChannels:
    def __init__(self, *, step_ms=100, history_s=30, max_gap_ms=1000):
        self.step_ms, self.max_gap_ms = int(step_ms), int(max_gap_ms)
        self.history = max(10, int(history_s * 1000 / self.step_ms))
        self.spec = CapabilitySpec(
            'audio.voice_channels', 'audio.voice_channels', 'audio', 'gamst.voice_channels', '1',
            (OutputField('channels', 'array'), OutputField('mask', 'array'), OutputField('voiced_share', 'number')),
            'window', window_ms=self.step_ms, freshness_ms=1000, interpretation='measurement', devices=('cpu',),
            license='Apache-2.0')
        self.reset()

    def reset(self):
        self._bucket, self._bucket_end, self._last_ms = [], None, None
        self._pitch, self._level, self._centroid = (deque(maxlen=self.history) for _ in range(3))

    def process(self, frame):
        features = (frame.data or {}).get('features') if isinstance(frame.data, dict) else None
        if not isinstance(features, dict):
            return None
        ts = frame.timestamp_ms
        if self._last_ms is not None and ts - self._last_ms > self.max_gap_ms:
            self.reset()
        self._last_ms = ts
        if self._bucket_end is None:
            self._bucket_end = ts + self.step_ms
        self._bucket.append(features)
        if ts < self._bucket_end:
            return None
        bucket, self._bucket, self._bucket_end = self._bucket, [], ts + self.step_ms
        return self._channels(bucket)

    def _channels(self, bucket):
        pitches = [f['pitch_hz'] for f in bucket if f.get('pitch_hz')]
        levels = [f['dbfs'] for f in bucket if f.get('dbfs') is not None]
        centroids = [f['spectral_centroid_hz'] for f in bucket if f.get('spectral_centroid_hz')]
        voiced = sum(1 for f in bucket if f.get('periodic_signal_candidate')) / len(bucket)
        periodicity = statistics.fmean(f.get('periodicity_score') or 0. for f in bucket)
        values, mask = [0.] * len(CHANNELS), [False] * len(CHANNELS)
        if pitches:
            pitch = 12 * math.log2(statistics.median(pitches))
            self._pitch.append(pitch)
            values[0], mask[0] = max(-1., min(1., (pitch - statistics.median(self._pitch)) / 12)), True
        if levels:
            level = statistics.fmean(levels)
            self._level.append(level)
            values[1], mask[1] = max(-1., min(1., (level - statistics.median(self._level)) / 20)), True
        values[2], mask[2] = voiced, True
        values[3], mask[3] = periodicity, True
        if centroids:
            brightness = math.log2(statistics.median(centroids))
            self._centroid.append(brightness)
            values[4], mask[4] = max(-1., min(1., brightness - statistics.median(self._centroid))), True
        return self.spec.validate(dict(channels=[round(v, 5) for v in values], mask=mask,
                                       voiced_share=round(voiced, 4)))
