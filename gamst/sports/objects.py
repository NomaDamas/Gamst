"""The object and ball contracts that sports modules read.

Every detector of GAMST commits ``objects``: a list of ``{track_id, label, attributes: {score, ...},
location: {bbox: {x, y, width, height}, unit: "normalized_image"}}`` (RF-DETR, YOLOE, YOLO26; a tracker module
re-commits the same list with its own ``track_id``). A ball model (TrackNetV3) commits ``balls``: a list of
``{x, y, score}`` in normalised image coordinates, with ``frame_ms`` when the answer is for an earlier frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    track_id: str | None
    label: str
    score: float
    x: float
    y: float
    width: float
    height: float
    team: str | None = None

    @property
    def cx(self):
        return self.x + self.width / 2

    @property
    def cy(self):
        return self.y + self.height / 2

    @property
    def foot(self):
        """Where a standing player touches the ground: the bottom centre of the box."""
        return self.cx, self.y + self.height

    def contains(self, x, y, margin=0.):
        grow_x, grow_y = self.width * margin, self.height * margin
        return self.x - grow_x <= x <= self.x + self.width + grow_x and self.y - grow_y <= y <= self.y + self.height + grow_y

    def distance(self, x, y):
        """0 inside the box, else the distance to its edge (normalised image units)."""
        dx = max(self.x - x, 0., x - self.x - self.width)
        dy = max(self.y - y, 0., y - self.y - self.height)
        return math.hypot(dx, dy)


def boxes(value, labels=None):
    """The ``objects`` of a record value as :class:`Box` es, optionally only these labels."""
    found = []
    for item in (value or {}).get('objects') or ():
        if not isinstance(item, dict):
            continue
        label = item.get('label')
        if labels is not None and label not in labels:
            continue
        box = ((item.get('location') or {}).get('bbox') or item.get('bbox') or {})
        try:
            x, y, width, height = (float(box[key]) for key in ('x', 'y', 'width', 'height'))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
            continue
        attributes = item.get('attributes') or {}
        score = attributes.get('score', item.get('score', 1.))
        track = item.get('track_id')
        found.append(Box(None if track is None else str(track), str(label), float(score), x, y, width, height,
                         attributes.get('team') or item.get('team')))
    return found


def ball_points(value, labels=('sports ball',)):
    """Ball candidates of a record: ``balls`` of a ball model, else the centres of detector boxes with a ball label.
    Each is ``(x, y, score, frame_ms or None)``."""
    value = value or {}
    if 'balls' in value:
        found = []
        for item in value.get('balls') or ():
            try:
                x, y = float(item['x']), float(item['y'])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                found.append((x, y, float(item.get('score', 1.)), item.get('frame_ms', value.get('frame_ms'))))
        return found
    return [(box.cx, box.cy, box.score, None) for box in boxes(value, labels)]


def object_value(items, *, tracking, width=None, height=None):
    """An ``objects`` record value (the RF-DETR layout) from :class:`Box` es."""
    objects = [dict(track_id=item.track_id or f'object-{index}', label=item.label,
                    attributes=dict(score=round(item.score, 4), **({'team': item.team} if item.team else {})),
                    location=dict(bbox=dict(x=item.x, y=item.y, width=item.width, height=item.height),
                                  unit='normalized_image'))
               for index, item in enumerate(items)]
    summary = dict(detected_count=len(objects), returned_count=len(objects), tracking=tracking)
    if width and height:
        summary.update(width=int(width), height=int(height))
    return dict(objects=objects, summary=summary)
