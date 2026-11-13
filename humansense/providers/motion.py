"""Actual OpenCV optical flow with bounded image state and explicit image units."""
import math

from ..capabilities import CapabilitySpec, OutputField


class OpenCVFlow:
    """Sparse Lucas-Kanade or dense Farneback motion, not physical world velocity."""
    def __init__(self, *, method='lucas_kanade', max_side=320, max_gap_ms=1000,
                 max_points=128, motion_threshold_px=.5):
        if method not in ('lucas_kanade', 'farneback'):
            raise ValueError('method must be lucas_kanade or farneback')
        if type(max_side) is not int or not 32 <= max_side <= 640:
            raise ValueError('max_side must be 32..640')
        if type(max_points) is not int or not 8 <= max_points <= 512:
            raise ValueError('max_points must be 8..512')
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 10000:
            raise ValueError('max_gap_ms must be 1..10000')
        if type(motion_threshold_px) not in (int, float) or not math.isfinite(motion_threshold_px) or not 0 < motion_threshold_px <= 100:
            raise ValueError('invalid motion threshold')
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[vision] for optical flow') from exc
        self.cv2, self.np = cv2, np
        self.method, self.max_side, self.max_gap_ms = method, max_side, max_gap_ms
        self.max_points, self.threshold = max_points, float(motion_threshold_px)
        self.closed = False
        self.reset()

    @classmethod
    def capability(cls, method='lucas_kanade', *, version='installed'):
        if method not in ('lucas_kanade', 'farneback'):
            raise ValueError('unsupported flow method')
        return CapabilitySpec(f'vision.flow.{method}', 'vision.image_motion', 'video',
            f'opencv.{method}', version, (OutputField('motion', 'object'),),
            'native', stateful=True, reset_on_gap=True, interpretation='estimate',
            devices=('cpu',), sensor_requirements=('uint8-gray-or-bgr-camera',), license='Apache-2.0')

    def reset(self):
        self.previous = self.previous_ts = self.original_shape = None

    def close(self):
        self.reset()
        self.closed = True

    def process(self, frame):
        if self.closed:
            raise RuntimeError('flow provider is closed')
        np, cv = self.np, self.cv2
        data = frame.data
        if frame.modality != 'video' or not isinstance(data, np.ndarray) or data.dtype != np.uint8:
            raise ValueError('flow requires a uint8 video array')
        if data.ndim not in (2, 3) or data.ndim == 3 and data.shape[2] != 3:
            raise ValueError('flow accepts grayscale or BGR images')
        h, w = data.shape[:2]
        if min(h, w) < 16 or max(h, w) > 4096 or data.nbytes > 16*1024*1024:
            raise ValueError('video dimensions exceed flow limits')
        ts = frame.timestamp_ms
        if type(ts) not in (int, float) or not math.isfinite(ts) or ts < 0:
            raise ValueError('invalid capture timestamp')
        if self.previous_ts is not None and ts <= self.previous_ts:
            raise ValueError('flow capture timestamps must strictly increase')
        gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY) if data.ndim == 3 else data
        ratio = min(1., self.max_side/max(h, w))
        width, height = max(1, round(w*ratio)), max(1, round(h*ratio))
        if (width, height) != (w, h):
            gray = cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)
        else:
            gray = gray.copy()
        previous, previous_ts, original_shape = self.previous, self.previous_ts, self.original_shape
        self.previous, self.previous_ts, self.original_shape = gray, ts, (h, w)
        if previous is None or original_shape != (h, w) or ts-previous_ts > self.max_gap_ms:
            return None
        scale = np.array([w/width, h/height], dtype=np.float32)
        if self.method == 'lucas_kanade':
            points = cv.goodFeaturesToTrack(previous, maxCorners=self.max_points,
                qualityLevel=.01, minDistance=5, blockSize=7)
            if points is None:
                return None
            next_points, status, _ = cv.calcOpticalFlowPyrLK(previous, gray, points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, .01))
            if next_points is None:
                return None
            valid = status.reshape(-1).astype(bool)
            starts, ends = points.reshape(-1, 2)[valid], next_points.reshape(-1, 2)[valid]
            # Forward/backward agreement rejects points lost through occlusion.
            if not len(ends):
                return None
            back, backward_status, _ = cv.calcOpticalFlowPyrLK(gray, previous, ends.reshape(-1, 1, 2), None)
            if back is None:
                return None
            valid_back = backward_status.reshape(-1).astype(bool) & (np.linalg.norm(back.reshape(-1, 2)-starts, axis=1) < 1.5)
            vectors = (ends-starts)[valid_back]*scale
        else:
            flow = cv.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            # Sample textured pixels; a blank image cannot establish zero motion.
            texture = cv.cornerMinEigenVal(previous, blockSize=3)
            mask = texture > max(.0001, float(texture.max())*.01)
            vectors = flow[mask]*scale
            if len(vectors) > self.max_points:
                vectors = vectors[np.linspace(0, len(vectors)-1, self.max_points).astype(int)]
        vectors = vectors[np.isfinite(vectors).all(axis=1)]
        if len(vectors) < 4:
            return None
        dx, dy = np.median(vectors, axis=0).tolist()
        dt = ts-previous_ts
        magnitudes = np.linalg.norm(vectors, axis=1)
        return {'motion': dict(dx_pixels=float(dx), dy_pixels=float(dy),
            median_speed_pixels_per_second=float(np.median(magnitudes)*1000/dt),
            moving_fraction=float(np.mean(magnitudes >= self.threshold)),
            sample_count=len(vectors), interval_ms=dt, width=w, height=h,
            coordinate_frame='image_pixels', camera_motion_separated=False,
            method=self.method, opencv_version=cv.__version__)}
