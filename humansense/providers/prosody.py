"""Bounded streaming audio features; no emotion or calibrated SPL inference."""
import math

from ..capabilities import CapabilitySpec, OutputField
from ..plugins import _pcm_floats


class AudioProsody:
    def __init__(self, *, window_ms=80, hop_ms=20, min_pitch_hz=60,
                 max_pitch_hz=500, channel=None, energy_floor=.005):
        if type(window_ms) is not int or not 40 <= window_ms <= 250:
            raise ValueError('window_ms must be 40..250')
        if type(hop_ms) is not int or not 1 <= hop_ms <= window_ms:
            raise ValueError('hop_ms must be 1..window_ms')
        for value in (min_pitch_hz, max_pitch_hz, energy_floor):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('feature thresholds must be finite numbers')
        if not 40 <= min_pitch_hz < max_pitch_hz <= 1000 or window_ms*min_pitch_hz < 2000:
            raise ValueError('pitch range requires at least two periods per window')
        if not 0 < energy_floor <= 1:
            raise ValueError('energy_floor must be in (0, 1]')
        if channel is not None and (type(channel) is not int or not 0 <= channel < 64):
            raise ValueError('channel must be None or 0..63')
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy or humansense-sdk[audio] for audio features') from exc
        self.np = np
        self.window_ms, self.hop_ms = window_ms, hop_ms
        self.min_pitch_hz, self.max_pitch_hz = min_pitch_hz, max_pitch_hz
        self.channel, self.energy_floor = channel, energy_floor
        self.closed = False
        self.reset()

    @classmethod
    def capability(cls, *, window_ms=80):
        if type(window_ms) is not int or not 40 <= window_ms <= 250:
            raise ValueError('invalid feature window')
        return CapabilitySpec('audio.prosody', 'audio.features', 'audio', 'humansense.audio_dsp', '1',
            (OutputField('features', 'object'),), 'window', window_ms=window_ms,
            stateful=True, reset_on_gap=True, freshness_ms=1000, interpretation='estimate',
            devices=('cpu',), sensor_requirements=('normalized-pcm',), license='Apache-2.0')

    def reset(self):
        self.buffer = []
        self.rate = self.last_ts = None
        self.since_output = 0

    def close(self):
        self.reset()
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('audio feature provider is closed')
        rate = frame.sample_rate
        if frame.modality != 'audio' or type(rate) is not int or not 8000 <= rate <= 96000:
            raise ValueError('audio sample rate must be 8000..96000')
        ts = frame.timestamp_ms
        if type(ts) not in (int, float) or not math.isfinite(ts) or ts < 0:
            raise ValueError('invalid capture end timestamp')
        data = frame.data
        if isinstance(data, (bytes, bytearray, memoryview)):
            if self.channel not in (None, 0) or frame.sensor and len(frame.sensor.channels) > 1:
                raise ValueError('byte PCM input must be explicitly mono')
            values = _pcm_floats(data)
        else:
            if len(data) > 96000:
                raise ValueError('audio input block too long')
            array = self.np.asarray(data)
            if array.ndim == 2:
                if self.channel is None or not 1 <= array.shape[1] <= 64 or self.channel >= array.shape[1]:
                    raise ValueError('multichannel audio requires an explicit valid channel')
                if frame.sensor and frame.sensor.channels and len(frame.sensor.channels) != array.shape[1]:
                    raise ValueError('channel metadata does not match sample layout')
                values = _pcm_floats(array[:, self.channel])
            elif array.ndim == 1:
                if self.channel not in (None, 0) or frame.sensor and len(frame.sensor.channels) > 1:
                    raise ValueError('mono audio cannot satisfy multichannel configuration')
                values = _pcm_floats(array)
            else:
                raise ValueError('audio must be a sample vector or samples-by-channels matrix')
        if not values:
            return None
        if self.last_ts is not None and ts <= self.last_ts:
            raise ValueError('capture end timestamps must increase')
        duration = len(values)*1000/rate
        # End timestamps describe sample capture, not callback arrival time.
        # Allow millisecond rounding, but never merge overlapping short blocks.
        tolerance_ms = max(1., 2000/rate)
        if self.rate is not None and (rate != self.rate or abs(ts-self.last_ts-duration) > tolerance_ms):
            self.reset()
        self.rate, self.last_ts = rate, ts
        count = round(rate*self.window_ms/1000)
        self.buffer = (self.buffer+values)[-count:]
        self.since_output += len(values)
        if len(self.buffer) < count or self.since_output < rate*self.hop_ms/1000:
            return None
        self.since_output = 0
        np = self.np
        samples = np.asarray(self.buffer, dtype=np.float64)
        rms = float(np.sqrt(np.mean(samples*samples)))
        centered = samples-float(np.mean(samples))
        ac_rms = float(np.sqrt(np.mean(centered*centered)))
        spectrum = np.abs(np.fft.rfft(centered*np.hanning(count)))
        frequencies = np.fft.rfftfreq(count, 1/rate)
        centroid = float(np.sum(spectrum*frequencies)/spectrum.sum()) if spectrum.sum() > 1e-12 else None
        pitch, periodicity = None, 0.
        if ac_rms >= self.energy_floor:
            fft_size = 1 << (2*count-1).bit_length()
            transformed = np.fft.rfft(centered, n=fft_size)
            correlation = np.fft.irfft(transformed*np.conj(transformed), n=fft_size)[:count]
            lo, hi = max(2, math.floor(rate/self.max_pitch_hz)), min(count//2, math.ceil(rate/self.min_pitch_hz))
            lags = np.arange(lo-1, hi+2)
            energy = np.concatenate(([0.], np.cumsum(centered*centered)))
            denominator = np.sqrt(energy[count-lags]*(energy[count]-energy[lags]))
            scores = correlation[lags]/np.maximum(denominator, 1e-12)
            peaks = [i for i in range(1, len(scores)-1) if scores[i] >= scores[i-1] and scores[i] > scores[i+1]]
            if peaks:
                best = max(float(scores[i]) for i in peaks)
                periodicity = max(0., min(1., best))
                if best >= .7:
                    index = next(i for i in peaks if scores[i] >= max(.7, best*.9))
                    lag = float(lags[index])
                    left, middle, right = scores[index-1:index+2]
                    divisor = left-2*middle+right
                    if abs(divisor) > 1e-12:
                        lag += float(.5*(left-right)/divisor)
                    pitch = float(min(self.max_pitch_hz, max(self.min_pitch_hz, rate/lag)))
        return {'features': dict(rms=rms, dbfs=20*math.log10(rms) if rms > 0 else None,
            ac_rms=ac_rms, pitch_hz=pitch, periodicity_score=periodicity,
            periodic_signal_candidate=pitch is not None, spectral_centroid_hz=centroid,
            clipping_fraction=float(np.mean(np.abs(samples) >= .999)),
            sample_rate=rate, window_samples=count, window_ms=count*1000/rate,
            channel=self.channel if self.channel is not None else 0,
            interpretation='signal_features_not_emotion_or_spl')}
