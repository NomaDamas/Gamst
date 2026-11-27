"""Per-sport physical presets and the frame-rate check of a ball recipe (physics, no tuning).

Each preset says what the camera must resolve: the ball, its fastest speed, the shortest time between two contacts,
the court and which detector labels name the players, the ball and the tools. ``gamst recipe check RECIPE --sport S
--fps F`` uses them; a recipe's ``"sports"`` section takes them as defaults of its modules.

Speeds are reported records (``speed_source``) and contact intervals are estimates of the fastest exchanges; both
are marked as such and were not measured by GAMST. They set a sampling bound, not an accuracy claim.
"""
from __future__ import annotations

RECORD = 'widely reported record; not checked by GAMST'
ESTIMATE = 'estimate of the fastest exchange; not measured by GAMST'

# max_ball_kmh: fastest ball. min_contact_s: shortest time between two contacts of the ball (hit, bounce, pass).
# labels: COCO class names of RF-DETR / YOLO26 trained on COCO (players, ball, tools).
SPORTS = {
    'tennis': dict(court='tennis.singles', ball='tennis ball', ball_diameter_m=.067, max_ball_kmh=263,
                   min_contact_s=.30, labels=dict(player='person', ball='sports ball', tool='tennis racket'),
                   ball_model='tracknetv3.tennis', recipe='racket-rally'),
    'badminton': dict(court='badminton.singles', ball='shuttlecock', ball_diameter_m=.066, max_ball_kmh=565,
                      min_contact_s=.25, labels=dict(player='person', ball='sports ball', tool='tennis racket'),
                      ball_model='tracknetv3.badminton', recipe='racket-rally'),
    'table-tennis': dict(court='table-tennis', ball='table tennis ball', ball_diameter_m=.040, max_ball_kmh=116,
                         min_contact_s=.12, labels=dict(player='person', ball='sports ball', tool='tennis racket'),
                         ball_model='tracknetv3.tennis', recipe='racket-rally'),
    'padel': dict(court='padel', ball='padel ball', ball_diameter_m=.065, max_ball_kmh=200, min_contact_s=.25,
                  labels=dict(player='person', ball='sports ball', tool='tennis racket'),
                  ball_model='tracknetv3.tennis', recipe='racket-rally'),
    'squash': dict(court='squash', ball='squash ball', ball_diameter_m=.040, max_ball_kmh=281, min_contact_s=.20,
                   labels=dict(player='person', ball='sports ball', tool='tennis racket'),
                   ball_model='tracknetv3.tennis', recipe='racket-rally'),
    'volleyball': dict(court='volleyball', ball='volleyball', ball_diameter_m=.21, max_ball_kmh=130, min_contact_s=.30,
                       labels=dict(player='person', ball='sports ball'), recipe='team-field'),
    'basketball': dict(court='basketball', ball='basketball', ball_diameter_m=.24, max_ball_kmh=60, min_contact_s=.25,
                       labels=dict(player='person', ball='sports ball'), recipe='team-field'),
    'football': dict(court='football', ball='football', ball_diameter_m=.22, max_ball_kmh=211, min_contact_s=.30,
                     labels=dict(player='person', ball='sports ball'), recipe='team-field'),
    'futsal': dict(court='futsal', ball='futsal ball', ball_diameter_m=.20, max_ball_kmh=130, min_contact_s=.25,
                   labels=dict(player='person', ball='sports ball'), recipe='team-field'),
    'handball': dict(court='handball', ball='handball', ball_diameter_m=.19, max_ball_kmh=130, min_contact_s=.25,
                     labels=dict(player='person', ball='sports ball'), recipe='team-field'),
    'ice-hockey': dict(court='ice-hockey', ball='puck', ball_diameter_m=.076, max_ball_kmh=184, min_contact_s=.25,
                       labels=dict(player='person', ball='sports ball'), recipe='team-field'),
    'billiards': dict(court='billiards.pool9ft', ball='billiard ball', ball_diameter_m=.057, max_ball_kmh=55,
                      min_contact_s=.05, labels=dict(player='person', ball='sports ball'), recipe='billiards'),
}
for _preset in SPORTS.values():
    _preset.update(speed_source=RECORD, contact_source=ESTIMATE)

FRAMES_PER_CONTACT = 4     # samples between two contacts: two to fix the direction before, two after
MAX_TRAVEL_SHARE = .25     # the ball moves at most this share of the court length between two frames (tracker gate)


def preset(sport):
    if sport not in SPORTS:
        raise SystemExit(f'unknown sport {sport!r}; presets: {", ".join(sorted(SPORTS))}')
    return dict(SPORTS[sport], sport=sport)


def ball_sampling(options, fps, *, sport=None, **_):
    """The frame rate a ball recipe needs (``required_fps``, a hard bound): enough samples between the two fastest
    contacts to see the ball's path break. The travel per frame at the record speed is reported beside it
    (``recommended_fps``, a warning only): a record is a launch peak (a smash leaves at 565 km/h and slows within
    metres), and TrackNetV3's own badminton set is 30 fps. Returns the verdict for ``fps`` (``ok``, ``problem``,
    ``warning``)."""
    from .court import COURTS
    chosen = preset(sport or options.get('sport') or 'tennis')
    speed = float(options.get('max_ball_kmh') or chosen['max_ball_kmh']) / 3.6
    contact = float(options.get('min_contact_s') or chosen['min_contact_s'])
    length = COURTS[chosen['court']]['length']
    frames = int(options.get('frames_per_contact', FRAMES_PER_CONTACT))
    share = float(options.get('max_travel_share', MAX_TRAVEL_SHARE))
    by_contact, by_travel = frames / contact, speed / (share * length)
    result = dict(sport=chosen['sport'], max_ball_kmh=round(speed * 3.6), min_contact_s=contact,
                  required_fps=round(by_contact, 1), recommended_fps=round(max(by_contact, by_travel), 1),
                  speed_source=chosen['speed_source'], contact_source=chosen['contact_source'])
    if fps:
        travel = speed / fps
        result.update(travel_m_per_frame=round(travel, 2), contact_time_error_ms=round(500 / fps, 1),
                      ok=fps >= by_contact,
                      problem=None if fps >= by_contact else
                      f"{fps:g} fps: only {contact * fps:.1f} frames lie between the {chosen['ball']}'s fastest two "
                      f"contacts ({frames} are needed to see its path break); {chosen['sport']} needs {by_contact:.0f} fps")
        if fps < by_travel:
            result['warning'] = (f"at its record speed the {chosen['ball']} moves {travel:.2f} m between frames "
                                 f"({travel / length:.0%} of the court): the fastest shots may be lost; "
                                 f"{by_travel:.0f} fps or more is better")
    return result
