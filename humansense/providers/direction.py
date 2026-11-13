"""Two synchronized microphones: GCC-PHAT delay and far-field cone angle.

This is a window DSP provider, not a 3D locator, speaker classifier or beamformer.
"""
import math
from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier


class GCCPHATDirection:
    def __init__(self, *, spacing_m, sound_speed_m_s, calibration_id,
                 channels=('mic0', 'mic1'), coordinate_frame='array', window_ms=80):
        for value, low, high in ((spacing_m, .02, 1), (sound_speed_m_s, 300, 380)):
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('spacing must be .02..1 m and sound speed 300..380 m/s')
        identifier(calibration_id, 'calibration_id')
        identifier(coordinate_frame, 'coordinate_frame')
        if type(channels) is not tuple or len(channels) != 2 or channels[0] == channels[1]:
            raise ValueError('declare two distinct synchronized microphone channels')
        for channel in channels:
            identifier(channel, 'channel')
        self.spec = self.capability(window_ms=window_ms)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy or humansense-sdk[audio] for GCC-PHAT') from exc
        self.np = np
        self.spacing, self.speed = float(spacing_m), float(sound_speed_m_s)
        self.calibration_id, self.channels = calibration_id, channels
        self.coordinate_frame, self.window_ms = coordinate_frame, window_ms
        self.closed = False

    @staticmethod
    def capability(*, window_ms=80):
        if type(window_ms) is not int or not 20 <= window_ms <= 200:
            raise ValueError('window_ms must be 20..200')
        return CapabilitySpec('audio.gcc_phat.direction', 'audio.direction', 'audio',
            'gamst.gcc_phat', '1', (OutputField('direction', 'object'),),
            'window', window_ms=window_ms, freshness_ms=1000, interpretation='estimate',
            devices=('cpu',), sensor_requirements=('synchronized-two-channel-pcm', 'array-calibration'),
            license='Apache-2.0')

    def reset(self):
        pass  # Windows are independent; never stitch audio across a gap.

    def close(self):
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('direction provider is closed')
        np, rate, data = self.np, frame.sample_rate, frame.data
        if frame.modality != 'audio' or type(rate) is not int or not 16000 <= rate <= 96000:
            raise ValueError('synchronized audio sample rate must be 16000..96000')
        if type(frame.timestamp_ms) not in (int, float) or not math.isfinite(frame.timestamp_ms) or frame.timestamp_ms < self.window_ms:
            raise ValueError('capture timestamp is window end and must cover the entire window')
        sensor = frame.sensor
        if not isinstance(sensor, SensorMetadata) or (sensor.calibration_id, sensor.channels, sensor.coordinate_frame) != (self.calibration_id, self.channels, self.coordinate_frame):
            raise ValueError('sensor calibration, channel order and coordinate frame must match array profile')
        if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2 or data.dtype not in (np.dtype('float32'), np.dtype('float64'), np.dtype('int16')):
            raise ValueError('input must be samples-by-two float PCM or int16 array')
        if data.shape[0]*1000 != rate*self.window_ms:
            raise ValueError('supply exactly one configured window; assemble blocks upstream')
        values = data.astype(np.float64)
        if data.dtype == np.int16:
            values /= 32768.
        if not np.isfinite(values).all() or np.abs(values).max() > 1:
            raise ValueError('PCM must be finite and normalized to [-1, 1]')
        values -= values.mean(axis=0)
        direction = dict(available=False, reason=None, delay_seconds=None,
            source_projection_on_baseline=None, cone_angle_degrees=None,
            baseline_from=self.channels[0], baseline_to=self.channels[1],
            coordinate_frame=self.coordinate_frame, calibration_id=self.calibration_id,
            spacing_m=self.spacing, sound_speed_m_s=self.speed,
            start_ms=frame.timestamp_ms-self.window_ms, end_ms=frame.timestamp_ms,
            window_ms=self.window_ms, sample_rate=rate, delay_resolution_seconds=1/rate,
            far_field_assumption=True, unique_3d_direction=False, position_available=False,
            confidence_calibrated=False, peak_strength=None, peak_ratio=None)
        def unavailable(reason):
            direction['reason'] = reason
            return {'direction': direction}
        if np.min(np.sqrt(np.mean(values*values, axis=0))) < 1e-4:
            return unavailable('insufficient_energy')
        max_lag = math.floor(self.spacing*rate/self.speed)
        if max_lag < 2:
            return unavailable('insufficient_delay_resolution')
        length = len(values)
        nfft = 1 << (2*length-1).bit_length()
        spectra = np.fft.rfft(values*np.hanning(length)[:, None], n=nfft, axis=0)
        cross = spectra[:, 0]*spectra[:, 1].conj()
        magnitude = np.abs(cross)
        usable = magnitude > max(float(magnitude.max())*1e-6, 1e-20)
        if np.count_nonzero(usable) < max(8, len(usable)*.02):
            return unavailable('insufficient_spectral_support')
        weighted = np.zeros_like(cross)
        weighted[usable] = cross[usable]/magnitude[usable]
        correlation = np.abs(np.fft.irfft(weighted, n=nfft))
        global_peak = int(np.argmax(correlation))
        global_lag = global_peak if global_peak <= nfft//2 else global_peak-nfft
        if abs(global_lag) > max_lag:
            return unavailable('dominant_delay_outside_array_geometry')
        lags = np.arange(-max_lag, max_lag+1)
        candidates = correlation[lags % nfft]
        peak_index = int(np.argmax(candidates))
        lag = int(lags[peak_index])
        strength = float(candidates[peak_index])
        others = candidates[np.abs(lags-lag) > 1]
        second = float(others.max()) if len(others) else 0.
        ratio = strength/max(second, 1e-12)
        direction.update(peak_strength=strength, peak_ratio=ratio)
        if strength < .05 or ratio < 1.2:
            return unavailable('weak_or_ambiguous_peak')
        delay = lag/rate  # t(mic0) - t(mic1); positive means toward mic1.
        projection = delay*self.speed/self.spacing
        direction.update(available=True, delay_seconds=delay,
            source_projection_on_baseline=projection,
            cone_angle_degrees=math.degrees(math.acos(max(-1., min(1., projection)))))
        return {'direction': direction}
