"""Per-frame object tracking and apparent-shape measurement for detected objects.

A semantic detector (YOLOE through ProcessProvider, or another caller-owned
model) names objects and supplies boxes at its own, slower cadence. Between
detections every capture re-finds each object by local segmentation inside a
velocity-predicted search window and fits an ellipse, so the high-rate signal
does not wait for the detector. Output per track:

* ``axes``: fitted ellipse [major, minor] in pixels (for ObjectSpinCounter)
* ``points`` with ``points_epoch``: corresponding features inside the object
  (for AngularMotionCounter); the epoch changes whenever points are reseeded so
  in-plane phase never jumps across unrelated points
* ``observed``: False when the object was not re-found in this capture

Segmentation separates the object from the median of a ring around the search
window. It assumes a locally plainer background than the object and is not a
general instance segmenter; ``observed`` is False rather than guessed when the
foreground is ambiguous. No pixels leave this class.
"""
from __future__ import annotations

import math


def round_candidates(image, max_count: int = 8) -> list[dict]:
    """Detector-free fallback: closed round contours, labelled only 'round candidate'.

    Geometry alone never names an object; use a semantic detector for labels.
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(1., 640 / max(width, height))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 50, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < 90:
            break
        if area > small.size * .4 or len(contour) < 8:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) / max(w, h) < .65 or 4 * math.pi * area / max(cv2.arcLength(contour, True) ** 2, 1) < .72:
            continue
        box = dict(x=x / small.shape[1], y=y / small.shape[0], width=w / small.shape[1], height=h / small.shape[0])
        if any(b['x'] <= box['x'] + box['width'] / 2 <= b['x'] + b['width'] and
               b['y'] <= box['y'] + box['height'] / 2 <= b['y'] + b['height'] for b in (f['bbox'] for f in found)):
            continue
        found.append(dict(label='round candidate', score=0.0, bbox=box))
        if len(found) >= max_count:
            break
    return found


class ShapeTracks:
    def __init__(self, *, max_tracks: int = 8, max_age_ms: int = 500, search_scale: float = 2.2,
                 max_points: int = 24):
        import cv2
        import numpy as np
        if not 1 <= max_tracks <= 32 or not 1.2 <= search_scale <= 6 or not 4 <= max_points <= 128:
            raise ValueError('max_tracks 1..32, search_scale 1.2..6, max_points 4..128')
        self.cv, self.np = cv2, np
        self.max_tracks, self.max_age_ms = max_tracks, max_age_ms
        self.search_scale, self.max_points = search_scale, max_points
        self.sequence = 0
        self.reset()

    def reset(self):
        self.tracks: dict[str, dict] = {}
        self.prev = self.owner = self.last_ts = None

    @staticmethod
    def _gray(image, cv, np):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('expected uint8 BGR image')
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def detections(self, items, image_shape, stamp):
        """Apply labelled detections captured at ``stamp``; boxes are normalized x/y/width/height."""
        h, w = image_shape[:2]
        for item in items:
            box = item['bbox']
            bw, bh = box['width'] * w, box['height'] * h
            cx, cy = box['x'] * w + bw / 2, box['y'] * h + bh / 2
            size = max(bw, bh)
            match = None
            for track_id, track in self.tracks.items():
                if math.dist(track['center'], (cx, cy)) < max(track['size'], size):
                    if match is None or math.dist(track['center'], (cx, cy)) < math.dist(self.tracks[match]['center'], (cx, cy)):
                        match = track_id
            if match is not None:
                track = self.tracks[match]
                track.update(label=item['label'], score=float(item['score']))
                if not track['observed']:
                    track.update(center=(cx, cy), velocity=(0., 0.))
                track['anchor'] = .7 * track['anchor'] + .3 * size
                track['size'] = min(track['size'], 1.3 * track['anchor'])
                continue
            if len(self.tracks) >= self.max_tracks:
                continue
            self.sequence += 1
            self.tracks[f'object-{self.sequence}'] = dict(
                label=item['label'], score=float(item['score']), center=(cx, cy), size=size, anchor=size,
                velocity=(0., 0.), last_seen=stamp, observed=False, axes=None, angle=None,
                points=None, points_epoch=0)

    def _segment(self, gray, track, dt):
        cv, np = self.cv, self.np
        h, w = gray.shape
        px = track['center'][0] + track['velocity'][0] * dt
        py = track['center'][1] + track['velocity'][1] * dt
        half = track['size'] * self.search_scale / 2
        x0, y0 = int(max(0, px - half)), int(max(0, py - half))
        x1, y1 = int(min(w, px + half)), int(min(h, py + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        roi = cv.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 0)
        ring = np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]])
        diff = cv.absdiff(roi, np.full_like(roi, int(np.median(ring))))
        level, fg = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if level < 12:  # no contrast against the local background
            return None
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv.connectedComponentsWithStats(fg)
        face = math.pi / 4 * track['anchor'] ** 2
        best = None
        for index in range(1, count):
            area = stats[index, cv.CC_STAT_AREA]
            extent = max(stats[index, cv.CC_STAT_WIDTH], stats[index, cv.CC_STAT_HEIGHT])
            # The detector fixes the object's scale; larger blobs are background or blur envelope.
            if not .02 * face <= area <= 1.5 * face or extent > 1.5 * track['anchor']:
                continue
            distance = math.dist((centers[index][0] + x0, centers[index][1] + y0), (px, py))
            if distance > track['size'] * 1.2:
                continue
            if best is None or distance < best[1]:
                best = (index, distance)
        if best is None:
            return None
        mask = (labels == best[0]).astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv.contourArea)
        if len(contour) < 5:
            return None
        (ex, ey), (a, b), angle = cv.fitEllipse(contour)
        if max(a, b) > 1.6 * track['anchor']:
            return None
        return (ex + x0, ey + y0), (max(a, b), min(a, b)), angle, (x0, y0, mask)

    def _points(self, gray, track, region):
        """Corresponding features inside the object's mask; LK forward-backward checked."""
        cv, np = self.cv, self.np
        x0, y0, mask = region
        mh, mw = mask.shape
        old = track['points']
        if old is not None and self.prev is not None:
            nxt, status, _ = cv.calcOpticalFlowPyrLK(self.prev, gray, old, None, winSize=(21, 21), maxLevel=3)
            if nxt is not None and status.all():
                back, rev, _ = cv.calcOpticalFlowPyrLK(gray, self.prev, nxt, None, winSize=(21, 21), maxLevel=3)
                local = np.rint(nxt.reshape(-1, 2) - (x0, y0)).astype(int)
                ok = ((local >= 0).all() and (local < (mw, mh)).all() and back is not None and rev.all()
                      and float(np.max(np.linalg.norm(back - old, axis=2))) < 1.5
                      and mask[local[:, 1], local[:, 0]].all())
                if ok:
                    return nxt
        seeded = cv.goodFeaturesToTrack(gray[y0:y0 + mh, x0:x0 + mw], self.max_points, .02, 3, mask=mask)
        track['points_epoch'] += 1
        if seeded is None or len(seeded) < 4:
            return None
        return (seeded + np.float32([x0, y0])).astype(np.float32)

    def observe(self, image, stamp, owner):
        cv, np = self.cv, self.np
        gray = self._gray(image, cv, np)
        if owner != self.owner or (self.prev is not None and gray.shape != self.prev.shape) or (
                self.last_ts is not None and not 0 < stamp - self.last_ts <= self.max_age_ms):
            self.reset()
            self.owner = owner
        dt = 0 if self.last_ts is None else (stamp - self.last_ts) / 1000
        output = []
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            found = self._segment(gray, track, dt)
            if found is None:
                track.update(observed=False, axes=None, points=None)
                if track['last_seen'] is not None and stamp - track['last_seen'] > self.max_age_ms:
                    del self.tracks[track_id]
                    continue
            else:
                center, axes, angle, mask = found
                if track['observed'] and dt > 0:
                    vx, vy = (center[0] - track['center'][0]) / dt, (center[1] - track['center'][1]) / dt
                    track['velocity'] = (.5 * track['velocity'][0] + .5 * vx, .5 * track['velocity'][1] + .5 * vy)
                track.update(center=center, axes=axes, angle=angle, observed=True, last_seen=stamp)
                track['size'] = min(max(.97 * track['size'], axes[0]), 1.3 * track['anchor'])
                track['points'] = self._points(gray, track, mask)
        # Two tracks that converged on the same object (e.g. a detector box around the
        # motion-blur envelope and another around the coin) keep only the older track.
        order = sorted(self.tracks, key=lambda key: int(key.rsplit('-', 1)[1]))
        for index, track_id in enumerate(order):
            track = self.tracks[track_id]
            if track['observed'] and any(
                    key in self.tracks and self.tracks[key]['observed'] and
                    math.dist(self.tracks[key]['center'], track['center']) < .5 * min(self.tracks[key]['size'], track['size'])
                    for key in order[:index]):
                del self.tracks[track_id]
        for track_id in order:
            if track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            x, y = track['center']
            s = track['size']
            output.append(dict(track_id=track_id, label=track['label'], score=round(track['score'], 3),
                               box=[int(x - s / 2), int(y - s / 2), int(s), int(s)], center=[float(x), float(y)],
                               observed=track['observed'],
                               axes=None if track['axes'] is None else [float(track['axes'][0]), float(track['axes'][1])],
                               angle=None if track['angle'] is None else float(track['angle']),
                               points=[] if track['points'] is None else track['points'].reshape(-1, 2).tolist(),
                               points_epoch=track['points_epoch']))
        self.prev, self.last_ts = gray, stamp
        return output
