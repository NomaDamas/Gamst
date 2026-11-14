"""Selected OpenCV stereo geometry; declared calibration is not measured accuracy.

This module does not open cameras or infer object identities. Dense arrays stay
local to the caller; the Context adapter must publish only bounded summaries.
"""
import hashlib
import json
import threading


class StereoGeometry:
    """Bounded horizontal stereo matching in the rectified left camera frame."""
    def __init__(self, calibration, *, disparities=64, block_size=5):
        import cv2
        import numpy as np
        from ..capabilities import bounded_json, identifier
        c = bounded_json(calibration, max_bytes=8192)
        fields = {'calibration_id', 'width', 'height', 'left_intrinsics',
                  'right_intrinsics', 'left_distortion', 'right_distortion',
                  'right_from_left_rotation', 'right_from_left_translation_m'}
        if type(c) is not dict or set(c) != fields:
            raise ValueError('exact stereo calibration fields required')
        identifier(c['calibration_id'])
        w, h = c['width'], c['height']
        if type(w) is not int or type(h) is not int or not (96 <= w <= 1280 and 64 <= h <= 720):
            raise ValueError('stereo image dimensions outside SDK bounds')
        if (type(disparities) is not int or disparities % 16 or not 16 <= disparities <= 128
                or disparities >= w // 2 or w*h*disparities > 64*1024*1024):
            raise ValueError('disparity search outside bounded working set')
        if type(block_size) is not int or block_size not in (3, 5, 7, 9):
            raise ValueError('supported odd matching block required')
        def array(name, shape, bound):
            value = np.asarray(c[name])
            if value.shape != shape or value.dtype.kind not in 'if':
                raise ValueError('invalid calibration matrix')
            value = value.astype(np.float64, copy=True)
            if not np.isfinite(value).all() or np.max(np.abs(value)) > bound:
                raise ValueError('calibration matrix outside bounds')
            return value
        left = array('left_intrinsics', (3, 3), 100000)
        right = array('right_intrinsics', (3, 3), 100000)
        for k in (left, right):
            if (not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0
                    or not 1 <= k[0, 0] <= 100000 or not 1 <= k[1, 1] <= 100000
                    or not 0 <= k[0, 2] < w or not 0 <= k[1, 2] < h):
                raise ValueError('supported pinhole intrinsics required')
        dl = array('left_distortion', (5,), 10)
        dr = array('right_distortion', (5,), 10)
        rotation = array('right_from_left_rotation', (3, 3), 1)
        translation = array('right_from_left_translation_m', (3,), 10)
        if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(rotation)-1) > 1e-7
                or not .001 <= np.linalg.norm(translation) <= 10):
            raise ValueError('proper rigid stereo transform with meter baseline required')
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            left, dl, right, dr, (w, h), rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if (not all(np.isfinite(v).all() for v in (r1, r2, p1, p2, q))
                or p2[0, 3] >= 0 or abs(p2[1, 3]) > 1e-7
                or p1[0, 0] <= 0 or p1[1, 1] <= 0):
            raise ValueError('only positive-disparity horizontal rectification supported')
        self._maps = (cv2.initUndistortRectifyMap(left, dl, r1, p1, (w,h), cv2.CV_32FC1),
                      cv2.initUndistortRectifyMap(right, dr, r2, p2, (w,h), cv2.CV_32FC1))
        self._matchers = tuple(cv2.StereoSGBM_create(
            minDisparity=minimum, numDisparities=disparities, blockSize=block_size,
            P1=8*block_size**2, P2=32*block_size**2, disp12MaxDiff=1,
            preFilterCap=31, uniquenessRatio=10, speckleWindowSize=50,
            speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM)
            for minimum in (0, -disparities))
        self._roi = []
        for x, y, rw, rh in (roi1, roi2):
            mask = np.zeros((h, w), dtype=bool)
            mask[y:y+rh, x:x+rw] = True
            self._roi.append(mask)
        self.width, self.height = w, h
        self.disparities, self.block_size = disparities, block_size
        self.calibration_id = c['calibration_id']
        self.calibration_sha256 = hashlib.sha256(json.dumps(
            c, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.runtime_version = cv2.__version__
        self._q, self._r1 = q, r1
        self._p1, self._p2 = p1, p2
        self._lock = threading.RLock()
        self._closed = False

    def _image(self, image):
        import numpy as np
        if (type(image) is not np.ndarray or image.dtype != np.uint8
                or image.shape != (self.height, self.width)):
            raise ValueError('exact calibrated uint8 grayscale image required')
        owned = image.copy(order='C')
        if owned.dtype != np.uint8 or owned.shape != (self.height,self.width):
            raise ValueError('stereo image changed shape during copy')
        return owned

    def compute(self, left, right):
        """Return owned point/validity arrays; no measured calibration verdict.

        Inputs must already be capture-aligned by the consuming adapter. Right
        consistency, positive depth and valid rectification ROIs are mandatory.
        """
        import cv2
        import numpy as np
        with self._lock:
            if self._closed:
                raise RuntimeError('stereo geometry closed')
            left, right = self._image(left), self._image(right)
            pair = [cv2.remap(im, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    for im, maps in zip((left, right), self._maps)]
            dl = self._matchers[0].compute(pair[0], pair[1]).astype(np.float32)/16
            dr = self._matchers[1].compute(pair[1], pair[0]).astype(np.float32)/16
            yy, xx = np.indices(dl.shape)
            rx = np.rint(xx-dl).astype(np.int32)
            bounded = (rx >= 0) & (rx < self.width)
            safe_rx = np.clip(rx, 0, self.width-1)
            corresponding = dr[yy, safe_rx]
            valid = (bounded & self._roi[0] & self._roi[1][yy, safe_rx]
                     & (dl > 0) & (dl < self.disparities)
                     & (corresponding >= -self.disparities) & (corresponding < 0)
                     & (np.abs(dl+corresponding) <= 1))
            points = cv2.reprojectImageTo3D(dl, self._q)
            valid &= np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0) & (points[:, :, 2] <= 1000)
            points[~valid] = np.nan
            return points, valid

    def rectified_mask(self, mask):
        """Transform an original-left image mask into rectified coordinates."""
        import cv2
        with self._lock:
            if self._closed:
                raise RuntimeError('stereo geometry closed')
            return cv2.remap(self._image(mask), *self._maps[0], cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT) != 0

    def provenance(self):
        """Detached actual transforms; R1 maps original-left into rectified-left."""
        with self._lock:
            if self._closed:
                raise RuntimeError('stereo geometry closed')
            return dict(rectified_from_original_left_rotation=self._r1.tolist(),
                        rectified_left_projection=self._p1.tolist(),
                        rectified_right_projection=self._p2.tolist(),
                        rectified_baseline_m=float(-self._p2[0,3]/self._p2[0,0]))

    def close(self):
        with self._lock:
            self._closed = True
            self._maps = self._matchers = self._roi = ()
