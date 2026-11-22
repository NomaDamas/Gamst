"""Local image-plane rotation perception. No camera, network or model downloads.

Tightly cropped, upright reference photographs give named objects repeatable
landmarks. Without references, closed round contours are *candidates*, never
semantic coin/wheel labels. Both modes produce corresponding pixel points for
the existing ObjectRotationBank, not independent turn counts.
"""
from __future__ import annotations

import math

from ..tracking import PersonTracker


class RotationTracks:
    def __init__(self, references=None, max_tracks=12):
        import cv2
        import numpy as np
        if not 1 <= max_tracks <= 32:
            raise ValueError('max_tracks must be 1..32')
        self.cv, self.np = cv2, np
        self.max_tracks = max_tracks
        self.extractor = cv2.SIFT_create(nfeatures=800) if references else None
        self.references = []
        for label, photo in (references or {}).items():
            if len(self.references) >= max_tracks or not isinstance(label, str) or not 1 <= len(label) <= 48:
                raise ValueError('invalid reference label or too many reference objects')
            gray = self._gray(photo)
            h, w = gray.shape
            # Keep the object's interior, not the backdrop in the crop corners.
            keypoints, desc = self.extractor.detectAndCompute(gray, self._mask(gray.shape, (0, 0, w, h)))
            if len(keypoints) < 8:
                raise ValueError(f'{label}: need a closer reference with visible texture')
            self.references.append((label, np.float32([p.pt for p in keypoints]), desc, w, h))
        self.tracker = PersonTracker(max_age_ms=500)
        self.sequence = 0
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.prev = self.last_ts = self.owner = None
        self.features = {}
        self.named_ids = {}

    def _gray(self, image):
        cv, np = self.cv, self.np
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('expected uint8 BGR image')
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _mask(self, shape, box):
        x, y, w, h = box
        mask = self.np.zeros(shape, self.np.uint8)
        self.cv.ellipse(mask, (x+w//2, y+h//2), (max(1,int(.43*w)), max(1,int(.43*h))), 0, 0, 360, 255, -1)
        return mask

    def _named(self, gray, stamp):
        cv, np = self.cv, self.np
        kps, descriptors = self.extractor.detectAndCompute(gray, None)
        if descriptors is None or len(kps) < 8:
            return []
        live, found = {}, []
        for label, ref, desc, w, h in self.references:
            matches = [a for a, b in cv.BFMatcher().knnMatch(desc, descriptors, k=2) if a.distance < .72*b.distance]
            if len(matches) < 8:
                continue
            src = np.float32([ref[m.queryIdx] for m in matches])
            dst = np.float32([kps[m.trainIdx].pt for m in matches])
            affine, inliers = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=2.0)
            if affine is None or inliers is None or inliers.sum() < max(8, .65*len(matches)):
                continue
            support = src[inliers.ravel().astype(bool)]
            if cv.contourArea(cv.convexHull(support)) < .045*w*h:
                continue
            # Same virtual landmarks, in template order, on every verified fit.
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]) @ affine[:, :2].T + affine[:, 2]
            x,y,bw,bh = cv.boundingRect(corners.astype(np.float32))
            if x < 0 or y < 0 or x+bw > gray.shape[1] or y+bh > gray.shape[0] or min(bw,bh) < 15:
                continue
            group, sep, side = label.rpartition(':')
            paired = bool(group and sep and side in ('heads', 'tails'))
            row = dict(label=group if paired else label, method='reference-match', box=[x,y,bw,bh],
                       points=corners.tolist(), center=corners.mean(axis=0).tolist())
            if paired:
                row.update(face=side, face_confidence=float(inliers.mean()))
            found.append(row)
        output = []
        for row in found:
            name, face = row['label'], row.get('face')
            if sum(r['label'] == name for r in found) != 1:
                continue
            last = self.named_ids.get(name)
            if last is None or stamp - last[1] > 150 or math.dist(last[3], row['center']) > max(row['box'][2:]):
                self.sequence += 1
                ident = f'ref-{self.sequence}'
            else:
                ident = last[0]
                if last[2] != face or last[1] != self.last_ts:
                    row['points'] = []  # never measure planar phase across different faces or gaps
            row['track_id'] = ident
            live[name] = (ident, stamp, face, row['center'])
            output.append(row)
        self.named_ids = {**{k:v for k,v in self.named_ids.items() if stamp-v[1] <= 150}, **live}
        return output

    def _round_boxes(self, gray):
        cv = self.cv
        height, width = gray.shape
        scale = min(1., 640 / max(width, height))
        small = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale < 1 else gray
        edges = cv.Canny(cv.GaussianBlur(small, (5,5), 0), 50, 120)
        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in sorted(contours, key=cv.contourArea, reverse=True):
            area = cv.contourArea(c)
            if area < 90:
                break
            if area > small.size*.4 or len(c) < 8:
                continue
            x,y,w,h = cv.boundingRect(c)
            perimeter = cv.arcLength(c, True)
            if min(w,h)/max(w,h) < .65 or 4*math.pi*area/max(perimeter**2,1) < .72:
                continue
            box = (x/small.shape[1], y/small.shape[0], w/small.shape[1], h/small.shape[0])
            if any(b[0] <= box[0]+box[2]/2 <= b[0]+b[2] and b[1] <= box[1]+box[3]/2 <= b[1]+b[3] for b in boxes):
                continue
            boxes.append(box)
            if len(boxes) >= self.max_tracks:
                break
        return boxes

    def _unnamed(self, gray, stamp):
        cv, np = self.cv, self.np
        h,w = gray.shape
        fresh, found = {}, []
        for track in self.tracker.update(self._round_boxes(gray), stamp):
            if not track['visible'] or track['association_quality'] == 'uncertain':
                continue
            box = track['bbox']
            x,y,bw,bh = [round(box[k]*dim) for k,dim in zip(('x','y','width','height'),(w,h,w,h))]
            key = track['track_id']
            mask = self._mask(gray.shape, (x,y,bw,bh))
            old = self.features.get(key)
            points = None
            if old is not None and self.prev is not None:
                ident, before = old
                nxt, status, _ = cv.calcOpticalFlowPyrLK(self.prev, gray, before, None)
                if nxt is not None and status.all():
                    back, rev, _ = cv.calcOpticalFlowPyrLK(gray, self.prev, nxt, None)
                    xy = nxt.reshape(-1,2)
                    px = np.rint(xy).astype(int)
                    valid = (back is not None and rev.all() and np.max(np.linalg.norm(back-before, axis=2)) < 1.5
                             and (px >= 0).all() and (px < (w,h)).all() and mask[px[:,1],px[:,0]].all())
                    if valid:
                        points = nxt
            if points is None:
                points = cv.goodFeaturesToTrack(gray, 12, .03, 4, mask=mask)
                if points is None or len(points) < 4:
                    continue
                self.sequence += 1
                ident = f'candidate-{self.sequence}'  # never join unrelated or reseeded points
            fresh[key] = ident, points
            found.append(dict(track_id=ident, label='round candidate', method='geometry-only', box=[x,y,bw,bh],
                              center=[x+bw/2,y+bh/2], points=points.reshape(-1,2).tolist()))
        self.features = fresh
        return found

    def observe(self, image, stamp, owner):
        gray = self._gray(image)
        if (owner != self.owner or self.prev is None or gray.shape != self.prev.shape or
                self.last_ts is not None and not 0 < stamp-self.last_ts <= 500):
            self.reset()
        objects = self._named(gray, stamp) if self.references else self._unnamed(gray, stamp)
        self.prev, self.last_ts, self.owner = gray, stamp, owner
        return objects
