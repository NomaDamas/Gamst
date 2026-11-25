"""What the local screen draws: geometry of the newest frames, separate from records and from agent context.

Points of a face or a body (:class:`Drawing`) stay with the module that found them, in this process: at most the
newest two frames of it, a few seconds old. They are no record, cannot be serialized, and only
:func:`guide_packet` (the dashboard of the same session, behind its token) reads them.
"""
from collections import deque
from copy import deepcopy
import time

MAX_ITEMS, MAX_POINTS, FRESH_MS = 4, 1024, 5000


class Drawing:
    """Newest drawing geometry of one module: ``put`` per frame, ``points`` for one committed frame."""
    __slots__ = ('_kept',)

    def __init__(self):
        self._kept = {}

    def put(self, name, frame, items):
        kept = self._kept.setdefault(name, deque(maxlen=2))     # a frame is drawn while the next is being committed
        kept.append((frame.subject_id, frame.timestamp_ms, time.monotonic(),
                     [list(points)[:MAX_POINTS] for points in items[:MAX_ITEMS]]))

    def points(self, name, subject_id, timestamp_ms):
        """One list of points per item of that frame's record, or None."""
        return next((items for subject, stamp, put, items in self._kept.get(name, ())
                     if (subject, stamp) == (subject_id, timestamp_ms) and 1000 * (time.monotonic() - put) <= FRESH_MS), None)

    def clear(self):
        self._kept.clear()

    def __repr__(self):
        return '<drawing geometry>'

    def __reduce__(self):
        raise TypeError('drawing geometry is not serialized or copied')


def drawing_of(module):
    """The module's :class:`Drawing` (made when it first finds something)."""
    found = getattr(module, 'drawing', None)
    if found is None:
        found = module.drawing = Drawing()
    return found


def drawn(human, name, subject_id, timestamp_ms):
    """Points of one committed frame from whichever plugin or Pipeline module of this session found them."""
    pipeline = getattr(human, 'pipeline', None)
    modules = [getattr(plugin, 'plugin', plugin) for plugin in human.plugins] + (
        [entry['module'] for entry in pipeline.modules.values()] if pipeline is not None else [])
    for module in modules:
        drawing = getattr(module, 'drawing', None)
        points = drawing.points(name, subject_id, timestamp_ms) if isinstance(drawing, Drawing) else None
        if points is not None:
            return points
    return None


def with_points(items, points):
    return [dict(item, landmarks=deepcopy(points[index])) if index < len(points) else item
            for index, item in enumerate(items)] if points else items


def guide_packet(state, human=None):
    result = {}
    now = state['timestamp_ms']
    for name in ('scene.faces', 'scene.poses', 'scene.person_tracks', 'scene.body_tracks', 'scene.segments'):
        record = state.get('signals', {}).get(name)
        if not record:
            continue
        value = record['observation']['value']
        if not isinstance(value, list):
            continue
        age = max(0, now - record['timestamp_ms'])
        # Raw pixels/audio are never included. Preserve geometry needed to draw.
        allowed = {'scene.faces': ('frame_face_index', 'blendshapes'),
                   'scene.poses': ('frame_pose_index', 'bbox', 'geometry', 'track_id', 'association_quality'),
                   'scene.segments': ('bbox', 'score', 'polygons', 'prompt', 'track_id'),
                   'scene.person_tracks': ('bbox', 'track_id', 'visible', 'association_quality'),
                   'scene.body_tracks': ('bbox', 'track_id', 'visible', 'association_quality')}[name]
        geometry = [{k: deepcopy(item[k]) for k in allowed if k in item} for item in value[:16 if name == 'scene.segments' else MAX_ITEMS] if isinstance(item, dict)]
        if human is not None and name in ('scene.faces', 'scene.poses'):
            geometry = with_points(geometry, drawn(human, name, record['subject_id'], record['timestamp_ms']))
        result[name] = {'value': geometry,
                        'timestamp_ms': record['timestamp_ms'], 'age_ms': age,
                        'fresh': age <= FRESH_MS, 'evidence_id': record['id']}
    return result
