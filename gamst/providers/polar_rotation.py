"""In-plane rotation by polar registration, for any object turning about its axis.

Each frame is resampled around the axis into angle x radius. The difference of two consecutive
frames keeps only what moves; a static grill, frame or background cancels. For a steady rotation
the next difference is the previous one shifted along the angle axis. The circular
cross-correlation along that axis, summed over radii, peaks at the shift. Plain correlation, not
phase correlation: whitening would turn the static occluders' pattern into a false peak at zero.
No feature has to survive from frame to frame, so textured, blurred or repetitive objects (fan
blades, spokes) are measured as well as plain ones.

An object that looks the same ``symmetry`` times per turn (a 3-blade fan: 3) gives the same
picture every ``360 / symmetry`` degrees. The shift is therefore taken modulo that period: a turn
of less than half a period between two frames is measured exactly, a faster one is an alias. That
is the ``capture_fps / (2 * symmetry)`` limit that ``vision.rotation`` reports.
"""
from __future__ import annotations

import collections
import math
import statistics


class PolarRotationCounter:
    """One track's unwrapped in-plane phase; same output fields as ``AngularMotionCounter``."""

    def __init__(self, *, symmetry=1, max_gap_ms=500, bins=720, min_correlation=.3):
        if type(symmetry) is not int or not 1 <= symmetry <= 64:
            raise ValueError('symmetry must be an integer in [1, 64]')
        self.symmetry, self.max_gap_ms, self.bins, self.min_correlation = symmetry, max_gap_ms, bins, min_correlation
        self.reset()

    def reset(self):
        self.phase = 0.
        self._polar = self._difference = self._last_ms = self._size = self._center = None
        self._centers = collections.deque(maxlen=48)

    def _shift(self, before, after):
        """(degrees, correlation) of the angular shift from ``before`` to ``after``."""
        import numpy as np
        spectrum = (np.conj(np.fft.rfft(before, axis=0)) * np.fft.rfft(after, axis=0)).sum(axis=1)
        profile = np.fft.irfft(spectrum, n=self.bins)
        norm = math.sqrt(float((before * before).sum()) * float((after * after).sum()))
        peak = int(np.argmax(profile))
        if not norm:
            return 0., 0.
        left, middle, right = profile[peak - 1], profile[peak], profile[(peak + 1) % self.bins]
        curvature = left - 2 * middle + right
        rows = peak + float(.5 * (left - right) / curvature if curvature else 0.)
        period = 360 / self.symmetry
        degrees = (rows * 360 / self.bins + period / 2) % period - period / 2
        return degrees, float(middle) / norm

    def process(self, gray, center, radius, timestamp_ms):
        """``gray``: float32 frame; ``center``: (x, y) pixels near the rotation axis; ``radius``: pixels."""
        import cv2
        radius = max(8, int(radius))
        dt = None if self._last_ms is None else timestamp_ms - self._last_ms
        # A shape's fitted center wobbles as the shape turns (blades are not an ellipse); the axis
        # does not. Sample around the median of recent centers and move the grid only when the
        # median really moved, or the region grew or shrank.
        self._centers.append((float(center[0]), float(center[1])))
        axis = tuple(statistics.median(values) for values in zip(*self._centers))
        moved = self._center is None or math.dist(axis, self._center) > max(2., .03 * radius)
        rescaled = self._size is None or abs(radius - self._size) > .15 * self._size
        if dt is None or not 0 < dt <= self.max_gap_ms or moved or rescaled:
            # a gap or a new grid breaks the correspondence; the accumulated phase stays
            self._polar = self._difference = None
        if moved:
            self._center = axis
        if rescaled:
            self._size = radius
        self._last_ms, radius = timestamp_ms, self._size
        # WARP_FILL_OUTLIERS: samples beyond the image are 0, not uninitialized memory
        polar = cv2.warpPolar(gray, (radius, self.bins), self._center, radius,
                              cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR)
        difference = None if self._polar is None else polar - self._polar
        degrees = correlation = 0.
        if difference is not None and self._difference is not None:
            degrees, correlation = self._shift(self._difference, difference)
            if not math.isfinite(degrees) or correlation < self.min_correlation:
                degrees = correlation = 0.
        self._polar, self._difference = polar, difference
        self.phase += math.radians(degrees)
        velocity = degrees * 1000 / dt if degrees else 0.
        direction = 'unknown' if not degrees else 'counterclockwise' if degrees > 0 else 'clockwise'
        return dict(rotation_count=round(abs(self.phase) / (2 * math.pi), 4),
                    angular_velocity_deg_s=round(velocity, 3), direction=direction,
                    confidence=round(max(0., correlation), 4))
