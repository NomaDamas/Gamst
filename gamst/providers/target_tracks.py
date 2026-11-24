"""Frame-to-frame tracking of a small object of known size (a target exemplar).

Generic motion tracking follows whatever moves, and a small thrown object (a coin,
a die, a ring) loses every contest against the hand that threw it. When the user
has shown GAMST the object's size, this tracker instead:

* acquires candidates only from exemplar-sized motion proposals, and keeps one
  only after it is re-found in consecutive frames while actually travelling;
* predicts each next position from a quadratic fit of the recent path (a thrown
  object accelerates steadily under gravity), so fast flight stays in a small
  search window and hands nearby are never considered;
* segments the object in that window by contrast against the window's own
  border, accepting blobs from edge-on (a thin line) to face-on (a full disc) of
  the exemplar's size, and reports the fitted ellipse axes every frame.

The output matches :class:`ShapeTracks` objects, so the spin counter is unchanged.
"""
from __future__ import annotations

import math


class TargetTracks:
    def __init__(self, size_px: float, *, max_tracks: int = 4, max_misses: int = 12,
                 confirm_frames: int = 4):
        import cv2
        import numpy as np
        self.cv, self.np = cv2, np
        self.size = float(size_px)
        self.max_tracks, self.max_misses, self.confirm_frames = max_tracks, max_misses, confirm_frames
        self.sequence = 0
        self.reset()

    def reset(self):
        self.tracks: dict[str, dict] = {}
        self.shape = None
        self.history = []  # recent grays; their per-pixel median is the static background

        """Best exemplar-sized blob near ``center``: ((x, y), (major, minor), angle) or None.

        The object is separated from what stays still (a face, a wall, a shirt) by comparing the
        window with the per-pixel median of the last ~120 ms: a flying object covers each pixel only
        briefly, so the median shows the background behind it. Until that history exists, the
        window's border median stands in for the background.
        """
        cv, np = self.cv, self.np
        h, w = gray.shape
        half = radius + .75 * self.size
        x0, y0 = int(max(0, center[0] - half)), int(max(0, center[1] - half))
        x1, y1 = int(min(w, center[0] + half)), int(min(h, center[1] + half))
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        roi = cv.GaussianBlur(gray[y0:y1, x0:x1], (3, 3), 0)
        if len(self.history) >= 6:
            background = np.median(np.stack([frame[y0:y1, x0:x1] for frame in self.history]), axis=0)
            signed = roi.astype(np.float32) - cv.GaussianBlur(background.astype(np.uint8), (3, 3), 0)
            # Flickering light and auto exposure shift the whole window; only the residual is motion.
            diff = cv.convertScaleAbs(np.abs(signed - float(np.median(signed))))
        else:
            border = np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]])
            diff = cv.absdiff(roi, np.full_like(roi, int(np.median(border))))
        level, fg = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            return None
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv.connectedComponentsWithStats(fg)
        face = math.pi / 4 * self.size ** 2
        best = None
        for index in range(1, count):
            area = stats[index, cv.CC_STAT_AREA]
            extent = max(stats[index, cv.CC_STAT_WIDTH], stats[index, cv.CC_STAT_HEIGHT])
            # Edge-on the object is a thin line of its full length; face-on, a full disc.
            if not .06 * face <= area <= 2 * face or not .45 * self.size <= extent <= 1.6 * self.size:
                continue
            cx, cy = centers[index][0] + x0, centers[index][1] + y0
            distance = math.dist((cx, cy), center)
            if distance > radius + .25 * self.size:
                continue
            score = distance / max(1., radius) + abs(math.log(extent / self.size))
            if best is None or score < best[1]:
                best = (index, score)
        if best is None:
            return None
        contours, _ = cv.findContours((labels == best[0]).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv.contourArea)
        if len(contour) < 5:
            return None
        (ex, ey), (a, b), angle = cv.fitEllipse(contour)
        if max(a, b) > 1.8 * self.size:
            return None
        return (ex + x0, ey + y0), (max(a, b), min(a, b)), angle

    def _predict(self, track, stamp):
        path = track['path'][-6:]
        if len(path) == 1:
            return path[-1][1:3], .6 * self.size
        t = self.np.array([(item[0] - stamp) / 1000 for item in path])
        x, y = self.np.array([item[1] for item in path]), self.np.array([item[2] for item in path])
        degree = 2 if len(path) >= 4 else 1
        px = float(self.np.polyval(self.np.polyfit(t, x, 1), 0))
        py = float(self.np.polyval(self.np.polyfit(t, y, degree), 0))
        step = math.dist(path[-1][1:3], path[-2][1:3])
        # Search wider after misses and for fast flight; never wider than a few object sizes.
        radius = min(3 * self.size, max(.5 * self.size, 1.5 * step) * (1 + .5 * track['misses']))
        return (px, py), radius

    def observe(self, gray, stamp, proposals):
        """``proposals``: candidate centres (pixels) from motion; returns ShapeTracks-style objects."""
        if self.shape != gray.shape:
            self.reset()
            self.shape = gray.shape
        output = self._observe(gray, stamp, proposals)
        # Background history: every other frame, ~120 ms at 240 fps; the current frame joins after use.
        self.frames = getattr(self, 'frames', 0) + 1
        if self.frames % 2 == 0:
            self.history.append(gray.copy())
            del self.history[:-15]
        return output

    def _observe(self, gray, stamp, proposals):
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            center, radius = self._predict(track, stamp)
            found = self._find(gray, center, radius)
            if found is None:
                track['misses'] += 1
                track['observed'] = False
                if track['misses'] > self.max_misses or not track['confirmed']:
                    del self.tracks[track_id]
                continue
            (cx, cy), axes, angle = found
            track['path'].append((stamp, cx, cy))
            track['path'] = track['path'][-32:]
            track['majors'] = (track['majors'] + [axes[0]])[-32:]
            track.update(misses=0, observed=True, axes=axes, angle=angle, last_seen=stamp)
            majors = self.np.array(track['majors'])
            # A flipping rigid disc keeps its major axis; hands and sleeves change shape.
            rigid = len(majors) < 4 or float(majors.std() / max(1., majors.mean())) <= .25
            if not rigid:
                del self.tracks[track_id]
                continue
            if not track['confirmed'] and len(track['path']) >= self.confirm_frames:
                travel = math.dist(track['path'][0][1:3], track['path'][-1][1:3])
                if travel >= .5 * self.size:
                    track['confirmed'] = True
                elif len(track['path']) >= 3 * self.confirm_frames:
                    del self.tracks[track_id]  # an exemplar-sized thing that does not travel
        # Two tracks on the same object: keep the older one.
        for track_id in sorted(self.tracks, key=lambda key: int(key.rsplit('-', 1)[1]), reverse=True):
            track = self.tracks[track_id]
            if any(other is not track and other['observed'] and track['observed'] and
                   math.dist(other['path'][-1][1:3], track['path'][-1][1:3]) < .5 * self.size and
                   int(key.rsplit('-', 1)[1]) < int(track_id.rsplit('-', 1)[1])
                   for key, other in self.tracks.items()):
                del self.tracks[track_id]
        for center in proposals:
            if len(self.tracks) >= self.max_tracks:
                break
            if any(math.dist(center, track['path'][-1][1:3]) < 1.5 * self.size for track in self.tracks.values()):
                continue
            found = self._find(gray, center, 1.2 * self.size)
            if found is None:
                continue
            (cx, cy), axes, angle = found
            self.sequence += 1
            self.tracks[f'target-{self.sequence}'] = dict(
                path=[(stamp, cx, cy)], majors=[axes[0]], misses=0, observed=True, axes=axes, angle=angle,
                confirmed=False, last_seen=stamp)
        output = []
        for track_id, track in self.tracks.items():
            if not track['confirmed']:
                continue
            x, y = track['path'][-1][1:3]
            s = self.size
            output.append(dict(track_id=track_id, label='unknown', score=1., observed=track['observed'],
                               box=[int(x - s / 2), int(y - s / 2), int(s), int(s)], center=[float(x), float(y)],
                               axes=[float(track['axes'][0]), float(track['axes'][1])] if track['observed'] else None,
                               angle=float(track['angle']) if track['observed'] else None,
                               points=[], points_epoch=0))
        return output


def measure_exemplar(image, box):
    """Diameter (pixels) of the object a user boxed: the most compact contrasting blob inside ``box``.

    ``box`` is (x0, y0, x1, y1) in ``image`` pixels. Boxes are drawn loosely around a coin held in
    fingers, so the box size overstates the object; returns None when no blob stands out.
    """
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
    x1, y1 = int(min(w, box[2])), int(min(h, box[3]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image[y0:y1, x0:x1]
    roi = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]])
    diff = cv2.absdiff(roi, np.full_like(roi, int(np.median(border))))
    level, fg = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if level < 10:
        return None
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    bw, bh = x1 - x0, y1 - y0
    for contour in contours:
        area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
        if area < 12 or perimeter <= 0 or len(contour) < 5:
            continue
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rx <= 1 or ry <= 1 or rx + rw >= bw - 1 or ry + rh >= bh - 1:
            continue  # background or fingers running out of the box, not the boxed object
        compactness = 4 * math.pi * area / perimeter ** 2  # 1 for a disc
        offset = math.dist((rx + rw / 2, ry + rh / 2), (bw / 2, bh / 2)) / max(bw, bh)
        score = area * compactness ** 2 * (1 - offset)
        if best is None or score > best[0]:
            best = (score, contour)
    if best is None:
        return None
    (_, _), (a, b), _ = cv2.fitEllipse(best[1])
    size = float(max(a, b))
    return size if max(4., .2 * max(bw, bh)) <= size <= max(bw, bh) * 1.2 else None
