"""Pure-NumPy OC-SORT person tracking with stream-local, non-biometric IDs.

Reimplements the observation-centric components of Cao et al., "Observation-
Centric SORT: Rethinking SORT for Robust Multi-Object Tracking", CVPR 2023
(https://arxiv.org/abs/2203.14360), against upstream MIT revision
8462e7e729a93ccd3bd995c0a79a890336cb3a0b. No upstream code is imported.
"""
from __future__ import annotations

from copy import deepcopy
import math

from ..tracking import normalize_bbox

UPSTREAM_REVISION = '8462e7e729a93ccd3bd995c0a79a890336cb3a0b'
PAPER = 'Cao et al., Observation-Centric SORT, CVPR 2023, arXiv:2203.14360'


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('OC-SORT requires NumPy; install gamst-sdk[vision]') from exc
    return np


def _linear_assignment(cost):
    """Deterministic O(n^3) Hungarian assignment, without SciPy/lap."""
    np = _numpy()
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError('assignment cost must be a finite matrix')
    if not cost.size:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    u = np.zeros(n+1)
    v = np.zeros(m+1)
    p = np.zeros(m+1, dtype=np.int64)
    way = np.zeros(m+1, dtype=np.int64)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minv = np.full(m+1, np.inf)
        used = np.zeros(m+1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m+1):
                if used[j]:
                    continue
                current = cost[i0-1, j-1]-u[i0]-v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]-1), j-1) for j in range(1, m+1) if p[j]]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return sorted(pairs)


def _iou_matrix(left, right):
    np = _numpy()
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)))
    a, b = np.asarray(left), np.asarray(right)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    extent = np.maximum(0., bottom_right-top_left)
    intersection = extent[..., 0]*extent[..., 1]
    areas_a = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1])
    areas_b = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    union = areas_a[:, None]+areas_b[None, :]-intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _bbox_to_z(box):
    np = _numpy()
    width, height = box[2]-box[0], box[3]-box[1]
    return np.array([box[0]+width/2, box[1]+height/2,
                     width*height, width/(height+1e-6)], dtype=np.float64)


def _z_to_bbox(state):
    np = _numpy()
    area, ratio = state[2], state[3]
    if area <= 0 or ratio <= 0:
        return np.array([np.nan]*4)
    width = math.sqrt(area*ratio)
    height = area/width
    return np.array([state[0]-width/2, state[1]-height/2,
                     state[0]+width/2, state[1]+height/2])


def _direction(first, second):
    np = _numpy()
    delta = np.array([(second[1]+second[3]-first[1]-first[3])/2,
                      (second[0]+second[2]-first[0]-first[2])/2])
    return delta/(np.linalg.norm(delta)+1e-6)


class _KalmanBox:
    """Seven-state SORT filter plus OC-SORT observation-centric re-update."""
    def __init__(self, box):
        np = _numpy()
        self.x = np.zeros(7)
        self.x[:4] = _bbox_to_z(box)
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.
        self.P *= 10.
        self.Q = np.eye(7)
        self.Q[-1, -1] *= .01
        self.Q[4:, 4:] *= .01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.
        self.H = np.zeros((4, 7))
        self.H[np.arange(4), np.arange(4)] = 1.
        self.frozen = None

    def predict(self):
        if self.x[6]+self.x[2] <= 0:
            self.x[6] = 0.
        self.x = self.F@self.x
        self.P = self.F@self.P@self.F.T+self.Q

    def correct(self, z):
        np = _numpy()
        residual = z-self.H@self.x
        projected = self.P@self.H.T
        innovation = self.H@projected+self.R
        gain = np.linalg.solve(innovation.T, projected.T).T
        self.x = self.x+gain@residual
        remainder = np.eye(7)-gain@self.H
        self.P = remainder@self.P@remainder.T+gain@self.R@gain.T

    def miss(self):
        if self.frozen is None:
            self.frozen = (self.x.copy(), self.P.copy())

    def observe(self, box, previous, gap):
        """Replay linear virtual observations before the real correction.

        ``gap`` includes the current frame. This is the paper's OCR re-update,
        equivalent to upstream freeze/unfreeze rather than smoothing future data.
        """
        np = _numpy()
        if self.frozen is not None:
            self.x, self.P = (item.copy() for item in self.frozen)
            self.frozen = None
            if previous is not None and gap > 1:
                first, last = _bbox_to_z(previous), _bbox_to_z(box)
                first_width = math.sqrt(first[2]*first[3])
                first_height = math.sqrt(first[2]/first[3])
                last_width = math.sqrt(last[2]*last[3])
                last_height = math.sqrt(last[2]/last[3])
                for index in range(gap):
                    fraction = (index+1)/gap
                    center = first[:2]+fraction*(last[:2]-first[:2])
                    width = first_width+fraction*(last_width-first_width)
                    height = first_height+fraction*(last_height-first_height)
                    width, height = max(width, 1e-9), max(height, 1e-9)
                    z = np.array([center[0], center[1], width*height, width/height])
                    self.correct(z)
                    if index != gap-1:
                        self.predict()
        self.correct(_bbox_to_z(box))


class _Track:
    def __init__(self, track_id, box, timestamp_ms, delta_t):
        self.track_id = track_id
        self.kalman = _KalmanBox(box)
        self.last_observation = box.copy()
        self.observations = {0: box.copy()}
        self.velocity = None
        self.delta_t = delta_t
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.first_seen_ms = self.last_seen_ms = timestamp_ms
        self.association_history_uncertain = False

    def predict(self):
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return _z_to_bbox(self.kalman.x)

    def update(self, box, timestamp_ms, delta_t):
        previous = self.last_observation.copy()
        for offset in range(delta_t, 0, -1):
            candidate = self.observations.get(self.age-offset)
            if candidate is not None:
                previous = candidate
                break
        self.velocity = _direction(previous, box)
        self.kalman.observe(box, previous, self.time_since_update)
        self.last_observation = box.copy()
        self.observations[self.age] = box.copy()
        cutoff = self.age-delta_t-2
        for key in tuple(self.observations):
            if key < cutoff:
                del self.observations[key]
        self.last_seen_ms = timestamp_ms
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1


def _ambiguities(matrix, threshold, margin):
    np = _numpy()
    rows, columns = set(), set()
    for axis, target in ((0, rows), (1, columns)):
        count = matrix.shape[axis]
        for index in range(count):
            values = matrix[index, :] if axis == 0 else matrix[:, index]
            eligible = np.flatnonzero(values >= threshold)
            if len(eligible) < 2:
                continue
            ordered = sorted(eligible, key=lambda other: (-values[other], int(other)))
            best = values[ordered[0]]
            if best-values[ordered[1]] <= margin:
                target.add(index)
    return rows, columns


def _associate(detections, predicted, previous, velocities, threshold, inertia, margin):
    np = _numpy()
    iou = _iou_matrix(detections, predicted)
    rows, columns = iou.shape
    if not rows or not columns:
        return [], list(range(rows)), list(range(columns)), set(), set(), iou
    centers_d = np.column_stack(((detections[:, 1]+detections[:, 3])/2,
                                 (detections[:, 0]+detections[:, 2])/2))
    centers_t = np.column_stack(((previous[:, 1]+previous[:, 3])/2,
                                 (previous[:, 0]+previous[:, 2])/2))
    delta = centers_d[None, :, :]-centers_t[:, None, :]
    directions = delta/(np.linalg.norm(delta, axis=2, keepdims=True)+1e-6)
    cosine = np.clip(np.sum(directions*velocities[:, None, :], axis=2), -1, 1)
    angle = (math.pi/2-np.abs(np.arccos(cosine)))/math.pi
    momentum = inertia*angle.T
    eligible_d, eligible_t = _ambiguities(iou, threshold, margin)
    pairs = _linear_assignment(-(iou+momentum))
    matched, matched_d, matched_t = [], set(), set()
    uncertain_d, uncertain_t = set(), set()
    for det, track in pairs:
        if iou[det, track] < threshold:
            continue
        matched.append((det, track, float(iou[det, track])))
        matched_d.add(det)
        matched_t.add(track)
        if det in eligible_d or track in eligible_t:
            uncertain_d.add(det)
            uncertain_t.add(track)
    return matched, [i for i in range(rows) if i not in matched_d], [i for i in range(columns) if i not in matched_t], uncertain_d, uncertain_t, iou


class OCSORTPersonTracker:
    """Drop-in geometric tracker; defaults differ only when explicitly selected.

    Every call is one temporal step, including an empty-detection call. Tracks
    expire after either ``max_age`` steps or ``max_age_ms``. Visible boxes are
    observations; missing boxes are clipped Kalman predictions and marked as
    such. OC-SORT continuity is not verified human re-identification.
    """
    def __init__(self, *, iou_threshold=.3, max_age_ms=1500, ambiguity_margin=.1,
                 max_age=30, delta_t=3, inertia=.2, min_hits=1):
        if not isinstance(iou_threshold, (int, float)) or isinstance(iou_threshold, bool) or not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError('iou_threshold must be in (0, 1]')
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or max_age_ms < 0:
            raise ValueError('max_age_ms must be a nonnegative integer')
        if not isinstance(ambiguity_margin, (int, float)) or isinstance(ambiguity_margin, bool) or not math.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
            raise ValueError('ambiguity_margin must be in [0, 1]')
        if isinstance(max_age, bool) or not isinstance(max_age, int) or not 0 <= max_age <= 10000:
            raise ValueError('max_age must be in 0..10000')
        if isinstance(delta_t, bool) or not isinstance(delta_t, int) or not 1 <= delta_t <= 30:
            raise ValueError('delta_t must be in 1..30')
        if not isinstance(inertia, (int, float)) or isinstance(inertia, bool) or not math.isfinite(inertia) or not 0 <= inertia <= 1:
            raise ValueError('inertia must be in [0, 1]')
        if isinstance(min_hits, bool) or not isinstance(min_hits, int) or not 1 <= min_hits <= 30:
            raise ValueError('min_hits must be in 1..30')
        self.iou_threshold, self.max_age_ms = float(iou_threshold), max_age_ms
        self.ambiguity_margin, self.max_age = float(ambiguity_margin), max_age
        self.delta_t, self.inertia, self.min_hits = delta_t, float(inertia), min_hits
        self._streams = {}
        self._counters = {}
        _numpy()

    @staticmethod
    def _xyxy(box):
        return [box['x'], box['y'], box['x']+box['width'], box['y']+box['height']]

    @staticmethod
    def _xywh(box):
        x0, y0, x1, y1 = (max(0., min(1., float(value))) for value in box)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('degenerate prediction')
        return {'x': x0, 'y': y0, 'width': x1-x0, 'height': y1-y0}

    def update(self, boxes, timestamp_ms: int, stream_id: str = 'camera-0'):
        np = _numpy()
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
            raise ValueError('timestamp_ms must be a nonnegative integer')
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError('stream_id must be non-empty')
        if isinstance(boxes, (str, bytes)) or hasattr(boxes, 'keys'):
            raise ValueError('boxes must be an iterable of detections')
        normalized = [normalize_bbox(box) for box in boxes]
        if len(normalized) > 256:
            raise ValueError('at most 256 detections per frame')
        previous = self._streams.get(stream_id)
        if previous is not None and timestamp_ms < previous['timestamp_ms']:
            raise ValueError('out-of-order timestamp; tracker state is unchanged')
        state = deepcopy(previous) if previous is not None else {
            'timestamp_ms': timestamp_ms, 'frame_count': 0,
            'counter': self._counters.get(stream_id, 0), 'tracks': {}}
        state['frame_count'] += 1
        tracks = state['tracks']
        detections = np.asarray([self._xyxy(box) for box in normalized], dtype=np.float64).reshape(-1, 4)
        ordered = sorted(tracks.values(), key=lambda track: track.track_id)
        predicted, invalid = [], set()
        for index, track in enumerate(ordered):
            box = track.predict()
            predicted.append(box)
            if not np.isfinite(box).all():
                invalid.add(index)
        for index in sorted(invalid, reverse=True):
            del ordered[index]
            del predicted[index]
        matched_by_track = {}
        uncertain_d, uncertain_t = set(), set()
        match_scores = {}
        if ordered and len(detections):
            previous_boxes = np.asarray([track.last_observation for track in ordered])
            velocities = np.asarray([track.velocity if track.velocity is not None else (0., 0.) for track in ordered])
            matches, unmatched_d, unmatched_t, uncertain_d, uncertain_t, iou = _associate(
                detections, np.asarray(predicted), previous_boxes, velocities,
                self.iou_threshold, self.inertia, self.ambiguity_margin)
            for det, track, score in matches:
                matched_by_track[track] = det
                match_scores[ordered[track].track_id] = score
            # Observation-centric recovery: retry against the last observation,
            # not only the potentially drifted Kalman prediction.
            if unmatched_d and unmatched_t:
                recovery = _iou_matrix(detections[unmatched_d], previous_boxes[unmatched_t])
                rd, rt = _ambiguities(recovery, self.iou_threshold, self.ambiguity_margin)
                for row, column in _linear_assignment(-recovery):
                    det, track = unmatched_d[row], unmatched_t[column]
                    score = float(recovery[row, column])
                    if score < self.iou_threshold or track in matched_by_track:
                        continue
                    matched_by_track[track] = det
                    match_scores[ordered[track].track_id] = score
                    if row in rd or column in rt:
                        uncertain_d.add(det)
                        uncertain_t.add(track)
        uncertain_tracks = {ordered[index].track_id for index in uncertain_t if index < len(ordered)}
        visible_by_track = {}
        for index, track in enumerate(ordered):
            det = matched_by_track.get(index)
            if det is None:
                track.kalman.miss()
                continue
            ambiguous = index in uncertain_t or det in uncertain_d
            track.update(detections[det], timestamp_ms, self.delta_t)
            track.association_history_uncertain |= ambiguous
            visible_by_track[track.track_id] = (det, ambiguous)
        for det, box in enumerate(normalized):
            if det in matched_by_track.values():
                continue
            state['counter'] += 1
            track_id = f'{stream_id}:person-{state["counter"]}'
            track = _Track(track_id, detections[det], timestamp_ms, self.delta_t)
            ambiguous = det in uncertain_d
            track.association_history_uncertain = ambiguous
            tracks[track_id] = track
            visible_by_track[track_id] = (det, ambiguous)
        result = []
        for track_id, track in list(tracks.items()):
            if track.time_since_update > self.max_age or timestamp_ms-track.last_seen_ms > self.max_age_ms:
                del tracks[track_id]
                continue
            visible = track_id in visible_by_track
            if visible:
                det, ambiguous = visible_by_track[track_id]
                box = normalized[det]
                quality = 'uncertain' if ambiguous else ('new' if track.hits == 1 else 'matched')
                reason = 'ambiguous_geometry' if ambiguous else ('new_detection' if track.hits == 1 else 'observation_centric_match')
                score = match_scores.get(track_id)
            else:
                try:
                    box = self._xywh(_z_to_bbox(track.kalman.x))
                except (ValueError, OverflowError):
                    box = self._xywh(track.last_observation)
                quality, reason, score = 'predicted', 'kalman_prediction', None
            confirmed = track.hit_streak >= self.min_hits or state['frame_count'] <= self.min_hits
            result.append({'track_id': track_id, 'stream_id': stream_id, 'bbox': box,
                'timestamp_ms': timestamp_ms, 'first_seen_ms': track.first_seen_ms,
                'last_seen_ms': track.last_seen_ms, 'visible': bool(visible and confirmed),
                'identity_verified': False, 'speaker_association': 'unavailable',
                'association_quality': quality if confirmed else 'unconfirmed',
                'association_reason': reason if confirmed else 'minimum_hits_not_reached',
                'association_iou': score,
                'association_history_uncertain': track.association_history_uncertain})
        state['timestamp_ms'] = timestamp_ms
        state['tracks'] = {track.track_id: track for track in ordered if track.track_id in tracks}
        # New tracks were appended to ``tracks`` but are not in ``ordered``.
        state['tracks'].update({key: value for key, value in tracks.items() if key not in state['tracks']})
        self._streams[stream_id] = state
        self._counters[stream_id] = state['counter']
        return deepcopy(sorted(result, key=lambda track: track['track_id']))

    def reset(self, stream_id: str | None = None):
        if stream_id is None:
            self._streams.clear()
        else:
            self._streams.pop(stream_id, None)
