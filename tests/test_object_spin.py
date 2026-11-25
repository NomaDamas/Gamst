import math

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.object_spin import ObjectSpinCounter

SENSOR = SensorMetadata(source_id='camera', sensor_id='spin', sequence=0, stream_epoch='test')


def run(phase_turns, fps, seconds, **options):
    """phase_turns(t) -> turns completed at time t; the coin then rests face-on."""
    counter, events, last = ObjectSpinCounter(**options), [], None
    frames = int(fps * seconds) + int(fps * .5)
    for i in range(frames):
        t = i / fps
        turned = phase_turns(min(t, seconds))
        ratio = abs(math.cos(2 * math.pi * turned)) * .95 + .05 if t <= seconds else .95
        last = counter.process(InputFrame({'objects': [{'track_id': 'coin', 'axes': [100.0, 100.0 * ratio]}]},
                                          'video', round(1000 * t), sensor=SENSOR))
        events += last['events']
    return [e for e in events if e['type'] == 'object.spin.ended'], last


def test_counts_a_hand_toss_at_240fps_within_one_turn():
    ended, _ = run(lambda t: 38 * t, fps=240, seconds=.5)
    assert len(ended) == 1 and ended[0]['status'] == 'counting'
    assert abs(ended[0]['turns'] - 19) <= 1
    assert ended[0]['max_countable_rev_s'] == 60.0


def test_withholds_the_aliased_count_of_a_toss_filmed_at_30fps():
    ended, last = run(lambda t: 38 * t, fps=30, seconds=.5)
    assert ended and all(e['status'] == 'undersampled' and e['turns'] is None for e in ended)
    assert last['objects'][0]['turns'] is None


def test_follows_a_slowing_spin_when_the_app_declares_its_range():
    # 6 -> 2 rev/s over 2 s: 8 turns; a 60 fps camera resolves up to 15 rev/s
    ended, _ = run(lambda t: 6 * t - t * t, fps=60, seconds=2.0, expected_max_rev_s=8)
    assert len(ended) == 1 and ended[0]['status'] == 'counting'
    assert abs(ended[0]['turns'] - 8) <= 1


def test_still_object_never_starts_an_episode():
    counter = ObjectSpinCounter()
    for i in range(100):
        value = counter.process(InputFrame({'objects': [{'track_id': 'coin', 'axes': [100.0, 90.0 + (i % 3)]}]},
                                           'video', i * 10, sensor=SENSOR))
        assert value['events'] == [] and value['objects'][0]['status'] == 'idle'
