"""Image-to-court mapping: a plane homography from four or more marked points, and court sizes.

A fixed camera sees the playing surface as a plane. Four image points whose court positions are known (the corners
of a singles court, the corners of a pitch, the pockets of a table) fix the homography H that maps an image point
on that plane to metres on the court. The user marks them once (a reference, like a voice sample); GAMST runs no
model for it. Points off the plane (a ball in the air, a player's head) do not map to their position: only what
touches the surface does (a bounce, a player's feet).

The direct linear transform (Hartley and Zisserman, *Multiple View Geometry*, 2nd ed., algorithm 4.2, with point
normalisation) is solved with NumPy's SVD; with more than four points it is the least-squares fit.
"""
from __future__ import annotations

import math

# Playing-surface sizes in metres (length along the long axis, width), from the rules of each sport. "corners" names
# the four points in the order the user marks them: far left, far right, near right, near left, as the camera sees
# the court from behind the near baseline.
COURTS = {
    'tennis.singles': dict(length=23.77, width=8.23, source='ITF Rules of Tennis, rule 1'),
    'tennis.doubles': dict(length=23.77, width=10.97, source='ITF Rules of Tennis, rule 1'),
    'badminton.singles': dict(length=13.40, width=5.18, source='BWF Laws of Badminton, law 1'),
    'badminton.doubles': dict(length=13.40, width=6.10, source='BWF Laws of Badminton, law 1'),
    'table-tennis': dict(length=2.74, width=1.525, source='ITTF Handbook, law 2.1'),
    'padel': dict(length=20.0, width=10.0, source='FIP Rules of Padel, annex'),
    'squash': dict(length=9.75, width=6.40, source='WSF Rules of Squash, court specification'),
    'volleyball': dict(length=18.0, width=9.0, source='FIVB Official Volleyball Rules, rule 1.1'),
    'basketball': dict(length=28.0, width=15.0, source='FIBA Official Basketball Rules, art. 2'),
    'football': dict(length=105.0, width=68.0, source='IFAB Laws of the Game, law 1 (recommended size)'),
    'futsal': dict(length=40.0, width=20.0, source='FIFA Futsal Laws of the Game, law 1'),
    'handball': dict(length=40.0, width=20.0, source='IHF Rules of the Game, rule 1'),
    'ice-hockey': dict(length=60.0, width=26.0, source='IIHF Official Rule Book, rule 17'),
    'billiards.pool9ft': dict(length=2.54, width=1.27, source='WPA playing surface, 9 ft table'),
    'billiards.carom': dict(length=2.84, width=1.42, source='UMB match table'),
}
CORNERS = ('far_left', 'far_right', 'near_right', 'near_left')


def court_corners(court):
    """Court coordinates (metres) of the four corners in ``CORNERS`` order; x across, y along, origin far left."""
    size = COURTS[court] if isinstance(court, str) else court
    length, width = float(size['length']), float(size['width'])
    return [(0., 0.), (width, 0.), (width, length), (0., length)]


def _normalise(points):
    import numpy as np
    points = np.asarray(points, dtype=float)
    centre = points.mean(0)
    scale = math.sqrt(2) / max(1e-12, float(np.mean(np.linalg.norm(points - centre, axis=1))))
    transform = np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])
    return np.c_[points, np.ones(len(points))] @ transform.T, transform


def homography(source, target):
    """3x3 H with ``target ~ H @ source`` for four or more point pairs (normalised DLT)."""
    import numpy as np
    if len(source) != len(target) or len(source) < 4:
        raise ValueError('a homography needs four or more point pairs')
    a, ta = _normalise(source)
    b, tb = _normalise(target)
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    if singular[-2] < 1e-9:
        raise ValueError('the points do not fix a plane (three of them on one line?)')
    h = np.linalg.inv(tb) @ vt[-1].reshape(3, 3) @ ta
    return h / h[2, 2]


class CourtMap:
    """Image points (normalised 0..1, or pixels with ``image_size``) on the court plane -> metres on the court.

    ``points``: the user's marks, a list of [x, y] in ``CORNERS`` order (or ``court_points`` gives their court
    positions in metres for other marks, e.g. the service line). ``margin_m``: how far outside the lines still counts
    as on the court (a ball that bounces just out is out, a player's feet behind the baseline are in play).
    """

    def __init__(self, court, points, *, court_points=None, image_size=None, margin_m=None):
        import numpy as np
        size = COURTS[court] if isinstance(court, str) else dict(court)
        if not {'length', 'width'} <= set(size):
            raise ValueError(f'unknown court {court!r}; known: {", ".join(sorted(COURTS))}')
        self.court, self.length, self.width = court if isinstance(court, str) else 'custom', size['length'], size['width']
        points = [[float(value) for value in point] for point in points]
        if image_size is not None:
            width, height = image_size
            points = [[x / width, y / height] for x, y in points]
        if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in points):
            raise ValueError('court points are finite [x, y] pairs')
        target = court_points if court_points is not None else court_corners(size)
        if len(target) != len(points):
            raise ValueError(f'{len(points)} marked points for {len(target)} court positions')
        self.points = points
        self.h = homography(points, target)
        self.inverse = np.linalg.inv(self.h)
        self.margin_m = float(max(self.length, self.width) * .1 if margin_m is None else margin_m)

    def to_court(self, x, y):
        """Normalised image point -> (x_m, y_m) on the court plane, or None behind the camera's horizon."""
        u, v, w = self.h @ [x, y, 1.]
        if w <= 1e-12:
            return None
        return float(u / w), float(v / w)

    def to_image(self, x_m, y_m):
        u, v, w = self.inverse @ [x_m, y_m, 1.]
        return float(u / w), float(v / w)

    def inside(self, x_m, y_m, margin_m=None):
        margin = self.margin_m if margin_m is None else margin_m
        return -margin <= x_m <= self.width + margin and -margin <= y_m <= self.length + margin

    def half(self, y_m):
        """Which half a court point is in: ``far`` or ``near`` (the net or halfway line is at length / 2)."""
        return 'far' if y_m < self.length / 2 else 'near'

    def describe(self):
        return dict(court=self.court, length_m=self.length, width_m=self.width, points=self.points)


def court_map(config):
    """A :class:`CourtMap` from a module item's ``court`` setting (or None when there is none)."""
    if not config:
        return None
    if isinstance(config, CourtMap):
        return config
    config = dict(config)
    return CourtMap(config.pop('court'), config.pop('points'), **config)
