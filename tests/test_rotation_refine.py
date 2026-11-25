import math
from unittest.mock import patch

from gamst.providers.rotation_refine import count_spins, episodes, flight_span, refine

FPS = 240.0


def _throw(rev_s=10.0, flight_s=.35, hold_s=.1, size=50.0, drop=()):
    """Ellipses of a disc flipped about a horizontal axis: held, thrown on a parabola, caught.

    In the hand it is jostled (not a parabola) and does not spin. Frames where the disc is within
    a few degrees of edge-on return None, as a segmenter loses a one-pixel line; ``drop`` frames
    are lost for other reasons.
    """
    stamps, shapes = [], {}
    hold, flight = round(hold_s * FPS), round(flight_s * FPS)
    for index in range(hold + flight + hold):
        t = index / FPS
        stamps.append(1000 * t)
        if index < hold or index >= hold + flight:
            jitter = 8 * math.sin(index * 1.7)
            y = 600 + jitter if index < hold else 600 + 3 * jitter
            shapes[index] = (300.0, y, size, .7 * size)
            continue
        ft = (index - hold) / FPS
        y = 600 - 1.6 * size * 120 * ft * flight_s + .5 * 400 * size * ft * ft  # up, then gravity
        ratio = abs(math.cos(2 * math.pi * rev_s * ft))
        shapes[index] = None if ratio < .08 or index in drop else (300.0 + 20 * ft, y, size, max(1., ratio * size))
    return stamps, shapes


def test_counts_only_the_flight_of_a_thrown_disc():
    stamps, shapes = _throw(rev_s=10.0, flight_s=.35)
    span = flight_span(stamps, shapes)
    assert span is not None
    assert abs(span[0] - 24) <= 3 and abs(span[1] - (24 + 84)) <= 3
    [event] = count_spins(stamps, shapes)
    assert event['status'] == 'counting'
    assert abs(event['turns'] - 3.5) <= .3
    assert abs(event['rev_s'] - 10.0) <= .8
    assert event['trajectory']['kind'] == 'free_fall'


def test_long_loss_makes_the_count_partial():
    hold = 24
    stamps, shapes = _throw(rev_s=10.0, flight_s=.35, drop=set(range(hold + 20, hold + 60)))
    events = count_spins(stamps, shapes)
    assert events and all(event['status'] == 'partial' for event in events)
    assert all(event['coverage'] < .8 for event in events)


def test_episodes_group_by_time_with_one_candidate_per_frame():
    detections = [(0, 0, .3, (0, 0, 40, 40)), (40, 10, .8, (5, 5, 45, 45)), (40, 10, .5, (90, 90, 130, 130)),
                  (2000, 480, .25, (0, 0, 40, 40)), (2010, 482, .1, (0, 0, 40, 40))]
    first, second = episodes(detections)
    assert first['seed'][1] == 10
    assert [(item[1], item[2]) for item in first['prompts']] == [(0, .3), (10, .8)]
    assert second['hits'] == 1 and second['seed'][1] == 480


def test_max_throws_prioritizes_scores_stops_on_complete_and_restores_time_order():
    frames = [(stamp, lambda: None) for stamp in range(0, 3001, 100)]
    detections = [(0, 0, .2, (0, 0, 40, 40)), (1000, 6, .9, (10, 0, 50, 40)),
                  (2000, 12, .8, (20, 0, 60, 40))]

    class Segmenter:
        def __init__(self):
            self.seeds = []

        def track(self, images, seeds):
            self.seeds.append(seeds[0][1][0])
            return {}

    def fake_count(stamps, shapes, *, track_id, **kwargs):
        number = int(track_id.rsplit('-', 1)[1])
        counting = number != 2  # The highest-scoring episode is not a complete throw.
        return [dict(track_id=track_id, status='counting' if counting else 'partial',
                     turns=1. if counting else None,
                     rev_s=10. if counting else None, coverage=.95 if counting else .5,
                     duration_ms=100, timestamp_ms=(number - 1) * 1000,
                     trajectory=dict(kind='free_fall'))]

    segmenter = Segmenter()
    progress = []
    with (patch('gamst.providers.rotation_refine.ellipses', lambda masks: masks),
          patch('gamst.providers.rotation_refine.count_spins', side_effect=fake_count)):
        results = refine(frames, detections, segmenter, label='coin', max_throws=1,
                         progress=lambda *args: progress.append(args))
    assert segmenter.seeds == [10, 20]
    assert [event['track_id'] for event in results] == ['coin-2', 'coin-3']
    assert [event['timestamp_ms'] for event in results] == [1000, 2000]
    assert progress == [('tracking', 1, 3), ('tracking', 2, 3)]
