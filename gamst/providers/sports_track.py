"""Track IDs for a detector's objects: the same player keeps the same ``track_id`` across frames.

Reads an ``objects`` module (``objects``; ``labels``: the classes to track, others pass through unchanged) and
re-commits the objects with this tracker's IDs. ``tracker`` picks the method; the stack of a recipe picks it:

- ``none``: the objects pass unchanged (a detector that tracks by itself: YOLO26 with Ultralytics' BoT-SORT);
- ``ocsort``: GAMST's own NumPy OC-SORT (docs/ocsort.md), no extra package;
- ``bytetrack``, ``botsort``, ``ocsort.trackers``, ``sort``, ``cbiou``: Roboflow ``trackers`` (Apache-2.0, clean-room
  re-implementations; SportsMOT HOTA in its README: ByteTrack 73.0, BoT-SORT 73.8). ``botsort`` also reads the video
  input (``video``) for its camera motion compensation; appearance ReID is not wired (docs/sports.md).

A detector that tracks by itself (YOLO26 with its built-in BoT-SORT, the Ultralytics stack) needs no tracker module.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from ..capabilities import CapabilitySpec, OutputField, identifier
from ..sports.objects import Box, boxes, object_value
from ..tracking import intersection_over_union

TRACKERS = {'bytetrack': 'ByteTrackTracker', 'botsort': 'BoTSORTTracker', 'ocsort.trackers': 'OCSORTTracker',
            'sort': 'SORTTracker', 'cbiou': 'CBIoUTracker'}
SCALE = 1000        # trackers' IoU and Kalman work in pixels; normalised boxes are scaled when the size is unknown


class ObjectTracker:
    def __init__(self, *, objects, tracker='ocsort', labels=('person',), video=None, frame_rate=30.,
                 options=None, capability_id='sports.tracks'):
        if tracker not in ('none', 'ocsort') and tracker not in TRACKERS:
            raise ValueError(f"tracker is none, ocsort or one of {', '.join(TRACKERS)}")
        self.objects, self.tracker_name, self.labels = identifier(objects, 'module name'), tracker, tuple(labels)
        self.video, self.frame_rate, self.options = video, float(frame_rate), dict(options or {})
        self._lock = threading.RLock()
        self.spec = CapabilitySpec(
            capability_id, capability_id, 'video', 'gamst.sports_track', f'1.{tracker}',
            (OutputField('objects', 'array'), OutputField('summary', 'object')),
            'frame', stateful=True, freshness_ms=1000, interpretation='estimate', devices=('cpu',), license='Apache-2.0')
        self.reset()

    def reset(self):
        with self._lock:
            self._frames = OrderedDict()
            if self.tracker_name == 'none':
                self._tracker = None
            elif self.tracker_name == 'ocsort':
                from ..tracking import create_tracker
                self._tracker = create_tracker('ocsort', **self.options)
            else:
                try:
                    import trackers
                except ImportError as error:
                    raise RuntimeError('install the tracker package: pip install "gamst-sdk[sports-trackers]"') from error
                kind = getattr(trackers, TRACKERS[self.tracker_name])
                options = dict(self.options)
                if self.tracker_name in ('bytetrack', 'botsort'):
                    options.setdefault('frame_rate', self.frame_rate)
                self._tracker = kind(**options)

    def _ocsort(self, items, at_ms):
        tracks = self._tracker.update([dict(x=b.x, y=b.y, width=b.width, height=b.height) for b in items], at_ms)
        found, used = [], set()
        for box in items:
            area = dict(x=box.x, y=box.y, width=box.width, height=box.height)
            scored = [(intersection_over_union(track['bbox'], area), track['track_id']) for track in tracks
                      if track['visible'] and track['track_id'] not in used]
            best = max(scored, default=(0., None))
            track_id = best[1] if best[0] >= .3 else None
            used.add(track_id)
            found.append(Box(None if track_id is None else f'track-{track_id}', box.label, box.score,
                             box.x, box.y, box.width, box.height, box.team))
        return [box for box in found if box.track_id is not None]

    def _trackers(self, items, at_ms, size):
        import numpy as np
        import supervision as sv
        width, height = size or (SCALE, SCALE)
        labels = sorted({box.label for box in items})
        xyxy = np.asarray([[b.x * width, b.y * height, (b.x + b.width) * width, (b.y + b.height) * height]
                           for b in items], dtype=np.float32).reshape(-1, 4)
        detections = sv.Detections(xyxy=xyxy, confidence=np.asarray([b.score for b in items], dtype=np.float32),
                                   class_id=np.asarray([labels.index(b.label) for b in items], dtype=int),
                                   data={'index': np.arange(len(items))})
        image = self._frames.get(at_ms) if self.tracker_name == 'botsort' else None
        tracked = self._tracker.update(detections, frame=image) if self.tracker_name == 'botsort' else \
            self._tracker.update(detections)
        found = []
        for row, track_id in zip(range(len(tracked)), tracked.tracker_id if tracked.tracker_id is not None else []):
            if track_id is None or int(track_id) < 0:
                continue
            source = items[int(tracked.data['index'][row])]
            x0, y0, x1, y1 = (float(v) for v in tracked.xyxy[row])
            found.append(Box(f'track-{int(track_id)}', source.label, source.score, x0 / width, y0 / height,
                             (x1 - x0) / width, (y1 - y0) / height, source.team))
        return found

    def process(self, frame):
        import numpy as np
        with self._lock:
            if isinstance(frame.data, np.ndarray):
                if self.tracker_name == 'botsort' and frame.modality == 'video':
                    self._frames[frame.timestamp_ms] = frame.data
                    while len(self._frames) > 8:
                        self._frames.popitem(last=False)
                return None
            source = None if frame.sensor is None else frame.sensor.sensor_id
            if source != self.objects or not isinstance(frame.data, dict):
                return None
            if self._tracker is None:
                return self.spec.validate({key: frame.data[key] for key in ('objects', 'summary') if key in frame.data})
            summary = frame.data.get('summary') or {}
            size = (summary['width'], summary['height']) if summary.get('width') and summary.get('height') else None
            every = boxes(frame.data, None)
            tracked = [box for box in every if box.label in self.labels]
            rest = [box for box in every if box.label not in self.labels]
            items = self._ocsort(tracked, frame.timestamp_ms) if self.tracker_name == 'ocsort' else \
                self._trackers(tracked, frame.timestamp_ms, size)
            for older in [ms for ms in self._frames if ms <= frame.timestamp_ms]:
                del self._frames[older]
            value = object_value(items + rest, tracking=self.tracker_name, width=size and size[0],
                                 height=size and size[1])
            return self.spec.validate(value)
