"""Scene cuts of a video, frame by frame (math only): hard cuts and dissolves.

Each frame is reduced to a 32 x 18 grayscale thumbnail. A cell "changed" when its gray level moved by more than
``CELL`` (25 of 255).

- **Hard cut**: at least ``HARD`` (half) of the cells changed from the previous frame. Decided at that frame.
- **Dissolve** (a cross-fade of about a second, as in compilations): at least ``SPAN`` (40 %) of the cells differ between
  the frames ``SPAN_MS`` (0.5 s) before and after a moment. The cut is the middle of the moments where that is highest,
  decided when it falls below again (about half a second to a second later).

A person moving, even close to the camera, changes fewer cells than a change of place. Two cuts closer than
``REFRACTORY_MS`` are one. Measured on the owner's two clips (docs/measurements/person-identity-2026-11-27.md, section
12); the thresholds are not set on labelled cuts.
"""
from __future__ import annotations

from collections import deque

CELL = 25.
HARD = .5
SPAN = .4
SPAN_MS = 500
REFRACTORY_MS = 1500
SIZE = (32, 18)


def thumbnail(image):
    """A 32 x 18 grayscale float thumbnail of a BGR (or gray) frame."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    step = max(1, min(height // (4 * SIZE[1]), width // (4 * SIZE[0])))     # cheap: subsample before resizing
    small = image[::step, ::step]
    if small.ndim == 3:
        small = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_BGR2GRAY)
    return cv2.resize(small, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


class SceneCuts:
    """Feed every frame (``feed(ts, image)``); it answers the time of a cut when one is decided, else None."""

    def __init__(self, *, cell=CELL, hard=HARD, span=SPAN, span_ms=SPAN_MS, refractory_ms=REFRACTORY_MS):
        self.cell, self.hard, self.span, self.span_ms, self.refractory_ms = cell, hard, span, span_ms, refractory_ms
        self.frames = deque()        # (ts, thumbnail) of the last 2 x span_ms
        self.rising = None           # (score, ts) of the highest span score while above ``span``
        self.last_cut = None
        self.cuts = []

    def _changed(self, a, b):
        import numpy as np
        return float((np.abs(a - b) > self.cell).mean())

    def _cut(self, ts):
        if self.last_cut is not None and ts - self.last_cut < self.refractory_ms:
            return None
        self.last_cut = ts
        self.cuts.append(ts)
        return ts

    def feed(self, ts, image=None, *, small=None):
        """``small``: the frame's :func:`thumbnail` when it is made already."""
        current = thumbnail(image) if small is None else small
        found = None
        if self.frames and self._changed(current, self.frames[-1][1]) >= self.hard:
            found = self._cut(ts)
        self.frames.append((ts, current))
        while self.frames and ts - self.frames[0][0] > 2 * self.span_ms:
            self.frames.popleft()
        # the moment span_ms ago, compared span_ms before and after it
        if ts - self.frames[0][0] >= 2 * self.span_ms - 50:
            middle = min(self.frames, key=lambda item: abs(item[0] - (ts - self.span_ms)))[0]
            score = self._changed(current, self.frames[0][1])
            if score >= self.span:      # (best score, first and last moment at it): the cut is the middle of the best
                if self.rising is None or score > self.rising[0]:
                    self.rising = (score, middle, middle)
                elif score == self.rising[0]:
                    self.rising = (score, self.rising[1], middle)
            elif self.rising is not None:
                at, self.rising = (self.rising[1] + self.rising[2]) // 2, None
                found = self._cut(at) if found is None else found
        return found
