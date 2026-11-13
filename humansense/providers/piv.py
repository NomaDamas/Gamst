"""Bounded OpenPIV pair processing with explicit spatial/time calibration."""
import hashlib
import importlib.metadata
import json
import math
import threading

from ..capabilities import CapabilitySpec, OutputField, SensorMetadata, identifier
from ..plugins import InputFrame


class OpenPIVFlow:
    def __init__(self, backend, *, meters_per_pixel, calibration_id, coordinate_frame,
                 max_gap_ms=1000, snr_threshold=1.3, max_speed_m_s=100):
        self.spec = self.capability(meters_per_pixel=meters_per_pixel, calibration_id=calibration_id,
            coordinate_frame=coordinate_frame, max_gap_ms=max_gap_ms,
            snr_threshold=snr_threshold, max_speed_m_s=max_speed_m_s)
        if not callable(backend):
            raise ValueError('OpenPIV backend must be callable')
        import numpy as np
        self.np, self.backend = np, backend
        self.scale, self.calibration_id, self.coordinate_frame = float(meters_per_pixel), calibration_id, coordinate_frame
        self.max_gap_ms, self.snr_threshold, self.max_speed = max_gap_ms, float(snr_threshold), float(max_speed_m_s)
        self._lock = threading.RLock()
        self.closed = False
        self._clear()

    @staticmethod
    def capability(*, meters_per_pixel, calibration_id, coordinate_frame,
                   max_gap_ms=1000, snr_threshold=1.3, max_speed_m_s=100):
        identifier(calibration_id); identifier(coordinate_frame)
        for value, lo, hi in ((meters_per_pixel, 1e-9, 1), (snr_threshold, 1, 100), (max_speed_m_s, 1e-6, 10000)):
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('invalid PIV scale, SNR threshold or speed bound')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError('max_gap_ms must be 1..10000')
        config = [float(meters_per_pixel), calibration_id, coordinate_frame, max_gap_ms,
                  float(snr_threshold), float(max_speed_m_s)]
        version = '0.26.1.' + hashlib.sha256(json.dumps(config).encode()).hexdigest()
        return CapabilitySpec('vision.openpiv.flow', 'vision.planar_flow', 'video', 'openpiv.pyprocess',
            version, (OutputField('flow', 'object'),), 'native', stateful=True, reset_on_gap=True,
            freshness_ms=max_gap_ms, interpretation='estimate', devices=('cpu',),
            sensor_requirements=('uint8-bgr-particle-images', 'uniform-metric-calibration'),
            license='GPL-3.0-or-later (OpenPIV); Apache-2.0 (adapter)')

    @classmethod
    def from_installed(cls, **config):
        cls.capability(**config)
        try:
            installed = importlib.metadata.version('openpiv')
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Install optional humansense-sdk[piv] in the selected runtime; OpenPIV is absent') from exc
        if installed != '0.26.1':
            raise RuntimeError('OpenPIV provider requires exactly openpiv==0.26.1')
        from openpiv.pyprocess import extended_search_area_piv
        return cls(extended_search_area_piv, **config)

    def _clear(self):
        self._previous = self._owner = self._last = None

    def reset(self):
        with self._lock:
            self._clear()

    def close(self):
        with self._lock:
            self._clear()
            self.backend = None
            self.closed = True

    def process(self, frame):
        with self._lock:
            if self.closed:
                raise RuntimeError('PIV provider is closed')
            try:
                return self._process(frame)
            except Exception:
                self._clear()
                raise

    def _process(self, frame):
        np = self.np
        if not isinstance(frame, InputFrame) or frame.modality != 'video' or not isinstance(frame.sensor, SensorMetadata):
            raise ValueError('video InputFrame with sensor metadata required')
        identifier(frame.subject_id)
        data, sensor, ts = frame.data, frame.sensor, frame.timestamp_ms
        if (sensor.calibration_id, sensor.coordinate_frame) != (self.calibration_id, self.coordinate_frame):
            raise ValueError('PIV metric calibration and coordinate frame must match')
        if type(ts) is not int or not 0 <= ts <= 2**53-1:
            raise ValueError('capture timestamp must be nonnegative integer milliseconds')
        if type(data) is not np.ndarray or data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3:
            raise ValueError('PIV requires uint8 BGR images')
        h, w = data.shape[:2]
        if not 64 <= h <= 256 or not 64 <= w <= 256:
            raise ValueError('PIV images must be 64..256 pixels per side; no implicit resizing')
        # Explicit luminance conversion; owned int32 arrays satisfy the pinned backend.
        gray = np.rint(data[:,:,0]*.114 + data[:,:,1]*.587 + data[:,:,2]*.299).astype(np.int32)
        sequence = sensor.capture_sequence if sensor.capture_sequence is not None else sensor.sequence
        owner = (frame.subject_id, sensor.source_id, sensor.sensor_id, sensor.stream_epoch,
                 sensor.calibration_id, sensor.coordinate_frame, sensor.capture_sequence is not None, (h,w))
        same = owner == self._owner
        if same and self._last and (ts <= self._last[0] or sequence <= self._last[1]):
            raise ValueError('PIV capture timestamps and sequences must increase')
        continuous = bool(same and self._last and sequence == self._last[1]+1 and ts-self._last[0] <= self.max_gap_ms)
        flow = dict(status='warmup', vectors=[], valid_vectors=0, rejected_vectors=0,
            interval_ms=None, dt_s=None, image_size_px=[w,h], coordinate_frame=self.coordinate_frame,
            axes=['image_right','image_down'], origin='top_left_pixel', meters_per_pixel=self.scale,
            calibration_id=self.calibration_id, calibration_verified=False, fluid_identity_verified=False,
            interrogation_size_px=32, search_size_px=64, overlap_px=32,
            snr_threshold=self.snr_threshold, max_speed_m_s=self.max_speed, outliers_filled=False)
        if continuous:
            dt = (ts-self._last[0])/1000
            flow.update(interval_ms=[self._last[0],ts], dt_s=dt)
            if float(gray.std()) < 1 or float(self._previous.std()) < 1:
                flow['status'] = 'insufficient_texture'
            else:
                # Never allow backend mutation to corrupt retained or caller-owned images.
                output = self.backend(self._previous.copy(), gray.copy(), window_size=32, overlap=32,
                    search_area_size=64, dt=dt, correlation_method='linear', subpixel_method='gaussian',
                    sig2noise_method='peak2peak', width=2, normalized_correlation=True,
                    use_vectorized=False, backend='scipy')
                shape = ((h-64)//32+1, (w-64)//32+1)
                if not isinstance(output, (tuple,list)) or len(output) != 3 or any(
                    type(a) is not np.ndarray or a.shape != shape or a.dtype.kind not in 'fi' for a in output):
                    raise ValueError('OpenPIV returned invalid vector grid')
                u,v,snr = output
                for row in range(shape[0]):
                    for col in range(shape[1]):
                        ux, vy, score = float(u[row,col])*self.scale, float(v[row,col])*self.scale, float(snr[row,col])
                        valid = (all(math.isfinite(x) for x in (ux,vy,score)) and
                                 self.snr_threshold <= score <= 1e6 and math.hypot(ux,vy) <= self.max_speed)
                        flow['vectors'].append(dict(center_m=[(32+32*col)*self.scale,(32+32*row)*self.scale],
                            velocity_m_s=[ux,vy] if valid else None,
                            snr=score if math.isfinite(score) and 0 <= score <= 1e6 else None, valid=valid))
                flow['valid_vectors'] = sum(v['valid'] for v in flow['vectors'])
                flow['rejected_vectors'] = len(flow['vectors'])-flow['valid_vectors']
                flow['status'] = 'estimated' if flow['valid_vectors'] else 'unavailable_vectors'
        result = self.spec.validate({'flow':flow})
        self._previous, self._owner, self._last = gray, owner, (ts,sequence)
        return result
