"""Bounded planar far-field SRP-PHAT azimuth scan; no position estimation."""
import math
from itertools import combinations
from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier


class SRPPHATDirection:
    def __init__(self, *, positions_m, channels, calibration_id, sound_speed_m_s=343,
                 coordinate_frame='array', window_ms=80, grid_degrees=5):
        if type(channels) is not tuple or not 3 <= len(channels) <= 8 or len(set(channels)) != len(channels):
            raise ValueError('declare 3..8 distinct ordered microphone channels')
        for value in (*channels, calibration_id, coordinate_frame): identifier(value)
        if type(positions_m) not in (list, tuple) or len(positions_m) != len(channels):
            raise ValueError('one XY position per channel is required')
        if any(type(p) not in (list, tuple) or len(p) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in p) for p in positions_m):
            raise ValueError('XY coordinates must be finite metres within [-1, 1]')
        if type(sound_speed_m_s) not in (int, float) or not math.isfinite(sound_speed_m_s) or not 300 <= sound_speed_m_s <= 380:
            raise ValueError('sound speed must be 300..380 m/s')
        if type(grid_degrees) is not int or grid_degrees not in (1, 2, 5, 10):
            raise ValueError('grid_degrees must be 1, 2, 5 or 10')
        self.spec = self.capability(window_ms=window_ms)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install numpy or humansense-sdk[audio] for SRP-PHAT') from exc
        positions = np.array(positions_m, dtype=np.float64)
        singular = np.linalg.svd(positions-positions.mean(axis=0), compute_uv=False)
        if singular[1] < .01:
            raise ValueError('array must span two planar axes by at least 0.01 m singular value')
        self.pairs = tuple(combinations(range(len(channels)), 2))
        if any(np.linalg.norm(positions[i]-positions[j]) < .02 for i, j in self.pairs):
            raise ValueError('microphones must be separated by at least 0.02 m')
        self.np, self.positions = np, positions
        self.channels, self.calibration_id, self.coordinate_frame = channels, calibration_id, coordinate_frame
        self.speed, self.window_ms, self.grid_degrees = float(sound_speed_m_s), window_ms, grid_degrees
        self.angles = np.arange(0, 360, grid_degrees)
        radians = np.deg2rad(self.angles)
        self.directions = np.column_stack((np.cos(radians), np.sin(radians)))
        self.closed = False

    @staticmethod
    def capability(*, window_ms=80):
        if type(window_ms) is not int or not 20 <= window_ms <= 200:
            raise ValueError('window_ms must be 20..200')
        return CapabilitySpec('audio.srp_phat.direction', 'audio.array_direction', 'audio',
            'gamst.srp_phat', '1', (OutputField('direction', 'object'),), 'window',
            window_ms=window_ms, freshness_ms=1000, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('synchronized-planar-array-pcm', 'array-calibration'), license='Apache-2.0')

    def reset(self):
        pass

    def close(self):
        self.closed = True

    def process(self, frame):
        if self.closed: raise RuntimeError('SRP provider is closed')
        np, rate, sensor, data = self.np, frame.sample_rate, frame.sensor, frame.data
        if frame.modality != 'audio' or type(rate) is not int or not 16000 <= rate <= 96000:
            raise ValueError('audio rate must be 16000..96000')
        if type(frame.timestamp_ms) not in (int, float) or not math.isfinite(frame.timestamp_ms) or frame.timestamp_ms < self.window_ms:
            raise ValueError('timestamp must be the complete window capture end')
        if not isinstance(sensor, SensorMetadata) or (sensor.channels, sensor.calibration_id, sensor.coordinate_frame) != (self.channels, self.calibration_id, self.coordinate_frame):
            raise ValueError('array channel order, calibration and coordinate frame must match')
        if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != len(self.channels) or data.shape[0]*1000 != rate*self.window_ms or data.dtype not in (np.dtype('float32'), np.dtype('float64'), np.dtype('int16')):
            raise ValueError('supply exact-window samples-by-channels normalized float or int16 PCM')
        values = data.astype(np.float64)
        if data.dtype == np.int16: values /= 32768.
        if not np.isfinite(values).all() or np.abs(values).max() > 1:
            raise ValueError('PCM must be finite and normalized')
        result = dict(available=False, reason=None, azimuth_degrees=None,
            coordinate_frame=self.coordinate_frame, calibration_id=self.calibration_id,
            channels=list(self.channels), positions_m=self.positions.tolist(),
            sound_speed_m_s=self.speed, grid_degrees=self.grid_degrees,
            sample_rate=rate, delay_resolution_seconds=1/rate, window_ms=self.window_ms,
            start_ms=frame.timestamp_ms-self.window_ms, end_ms=frame.timestamp_ms,
            pair_count=len(self.pairs), peak_score=None, competing_score=None,
            far_field_assumption=True, in_plane_source_assumption=True,
            elevation_available=False, position_available=False, confidence_calibrated=False,
            physical_calibration_verified=False)
        def unavailable(reason):
            result['reason'] = reason
            return {'direction': result}
        values -= values.mean(axis=0)
        if np.min(np.sqrt(np.mean(values*values, axis=0))) < 1e-4:
            return unavailable('insufficient_channel_energy')
        nfft = 1 << (2*len(values)-1).bit_length()
        spectra = np.fft.rfft(values*np.hanning(len(values))[:, None], n=nfft, axis=0)
        scores = np.zeros(len(self.angles))
        for i, j in self.pairs:
            cross = spectra[:, i]*spectra[:, j].conj()
            magnitude = np.abs(cross)
            usable = magnitude > max(float(magnitude.max())*1e-6, 1e-20)
            if np.count_nonzero(usable) < max(8, len(usable)*.02):
                return unavailable('insufficient_pair_spectral_support')
            weighted = np.zeros_like(cross)
            weighted[usable] = cross[usable]/magnitude[usable]
            correlation = np.fft.irfft(weighted, n=nfft)
            # t_i-t_j = dot(position_j-position_i, direction_toward_source)/c.
            lags = np.rint(self.directions @ (self.positions[j]-self.positions[i])*rate/self.speed).astype(int)
            scores += correlation[lags % nfft]
        scores /= len(self.pairs)
        best = int(np.argmax(scores))
        distance = np.abs((self.angles-self.angles[best]+180) % 360-180)
        competitor = float(scores[distance > 10].max())
        peak = float(scores[best])
        result.update(peak_score=peak, competing_score=competitor)
        if peak < .05 or peak-competitor < .02:
            return unavailable('weak_or_ambiguous_spatial_peak')
        result.update(available=True, azimuth_degrees=int(self.angles[best]))
        return {'direction': result}
