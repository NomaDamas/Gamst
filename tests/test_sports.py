"""Sports parts on synthetic data: court mapping, the frame-rate check, one ball's flight, contacts (hit, bounce),
rallies, possession, occupancy, team split, form against a reference, body turns, shots and contacts of balls, the
tracker module, TrackNetV3's streaming ensemble, the Ultralytics adapter, and the sports recipes with their stacks.

Nothing here runs a real model or reads real media: accuracy on sports footage is not measured by these tests."""
import asyncio
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame

try:
    import numpy as np
except ImportError:      # the core CI job has no NumPy
    np = None


def record(module, value, ms, sequence=0):
    """A record of ``module`` as a consumer receives it."""
    return InputFrame(value, 'video', int(ms), 'local-user', None,
                      SensorMetadata(source_id='test', sensor_id=module, sequence=sequence))


def objects(*items):
    """``objects`` value of (track_id, label, x, y, width, height[, team])."""
    found = []
    for item in items:
        track, label, x, y, width, height = item[:6]
        attributes = dict(score=.9, **({'team': item[6]} if len(item) > 6 else {}))
        found.append(dict(track_id=track, label=label, attributes=attributes,
                          location=dict(bbox=dict(x=x, y=y, width=width, height=height), unit='normalized_image')))
    return dict(objects=found, summary=dict(detected_count=len(found)))


def ball_box(x, y, size=.01):
    return ('ball', 'sports ball', x - size / 2, y - size / 2, size, size)


def run(coroutine):
    return asyncio.run(coroutine)


def events(values, name=None):
    found = []
    for value in values:
        if value is None:
            continue
        for item in value if isinstance(value, list) else [value]:
            found += [event for event in item.get('events', ()) if name is None or event['event'] == name]
    return found


@unittest.skipIf(np is None, 'numpy')
class CourtTest(unittest.TestCase):
    def test_marked_corners_map_to_the_court_and_back(self):
        from gamst.sports.court import COURTS, CourtMap
        corners = [[.3, .2], [.7, .2], [.9, .9], [.1, .9]]      # a trapezoid: the far baseline is shorter
        court = CourtMap('tennis.singles', corners)
        width, length = COURTS['tennis.singles']['width'], COURTS['tennis.singles']['length']
        for (x, y), expected in zip(corners, [(0, 0), (width, 0), (width, length), (0, length)]):
            mapped = court.to_court(x, y)
            self.assertAlmostEqual(mapped[0], expected[0], places=6)
            self.assertAlmostEqual(mapped[1], expected[1], places=6)
        back = court.to_image(width / 2, length / 2)
        self.assertAlmostEqual(court.to_court(*back)[1], length / 2, places=6)
        self.assertTrue(court.inside(width / 2, length / 2))
        self.assertFalse(court.inside(width / 2, length + 10, margin_m=0))
        self.assertEqual(court.half(1.), 'far')

    def test_points_on_one_line_are_refused(self):
        from gamst.sports.court import CourtMap
        with self.assertRaises(ValueError):
            CourtMap('tennis.singles', [[.1, .1], [.2, .2], [.3, .3], [.4, .4]])

    def test_pixels_with_image_size(self):
        from gamst.sports.court import CourtMap
        court = CourtMap('table-tennis', [[300, 200], [700, 200], [900, 900], [100, 900]], image_size=(1000, 1000))
        self.assertAlmostEqual(court.to_court(.3, .2)[0], 0., places=6)


class PresetTest(unittest.TestCase):
    def test_contacts_bound_is_hard_and_record_speed_only_warns(self):
        from gamst.sports.presets import ball_sampling
        badminton = ball_sampling({}, 30, sport='badminton')      # TrackNetV3's own set is 30 fps: allowed
        self.assertTrue(badminton['ok'])
        self.assertIn('fastest shots may be lost', badminton['warning'])
        self.assertAlmostEqual(badminton['travel_m_per_frame'], 565 / 3.6 / 30, places=2)
        table = ball_sampling({}, 25, sport='table-tennis')        # 3 frames between the fastest two contacts
        self.assertFalse(table['ok'])
        self.assertIn('table-tennis needs 33 fps', table['problem'])
        self.assertTrue(ball_sampling({}, 60, sport='tennis')['ok'])
        self.assertEqual(ball_sampling({}, None, sport='tennis').get('ok'), None)

    def test_unknown_sport(self):
        from gamst.sports.presets import preset
        with self.assertRaises(SystemExit):
            preset('quidditch')


def flight(points, start_ms=0, step_ms=10):
    """Detector records of a ball along ``points``, one per step, with a distractor that never moves."""
    return [record('detect', objects(ball_box(x, y), ('lamp', 'sports ball', .95, .02, .01, .01)), start_ms + i * step_ms, i)
            for i, (x, y) in enumerate(points)]


class BallTrackTest(unittest.TestCase):
    def test_follows_the_moving_ball_and_ignores_the_static_one(self):
        from gamst.providers.sports_ball import BallTrack
        track = BallTrack(ball='detect', min_score=.5)
        points = [(.2 + .01 * i, .5) for i in range(10)]
        values = [track.process(frame) for frame in flight(points)]
        self.assertEqual(values[0]['state'], 'tracking')
        self.assertAlmostEqual(values[-1]['x'], .29, places=4)
        self.assertAlmostEqual(values[-1]['vx'], 1.0, places=2)      # .01 per 10 ms
        self.assertEqual(events(values, 'ball.found')[0]['flight'], 1)

    def test_a_flight_ends_when_the_ball_is_gone(self):
        from gamst.providers.sports_ball import BallTrack
        track = BallTrack(ball='detect', lost_ms=50)
        for frame in flight([(.5, .5), (.51, .5)]):
            track.process(frame)
        empty = [track.process(record('detect', objects(), ms)) for ms in (20, 40, 60, 80)]
        self.assertEqual(empty[0]['state'], 'missing')
        self.assertEqual(events(empty, 'ball.lost')[0]['timestamp_ms'], 10)

    def test_ball_model_frames_each_become_a_state(self):
        from gamst.providers.sports_ball import BallTrack
        track = BallTrack(ball='ballmodel')
        values = track.process(record('ballmodel', dict(frames=[dict(frame_ms=0, balls=[dict(x=.1, y=.1, score=.9)]),
                                                                dict(frame_ms=10, balls=[dict(x=.11, y=.1, score=.9)])],
                                                        model='fake'), 70))
        self.assertEqual([value['at_ms'] for value in values], [0, 10])


def rally_points():
    """A ball from the near player (bottom) down the court, a bounce, up to the far player, and back."""
    points = []
    x, y = .5, .8
    for _ in range(20):         # rising towards the far side: y decreases
        y -= .02
        points.append((x, y))
    for _ in range(20):         # falls for 200 ms: y increases (image down)
        y += .01
        points.append((x, y))
    for _ in range(12):         # bounce: up again
        y -= .02
        points.append((x, y))
    return points


class ContactsTest(unittest.TestCase):
    def setUp(self):
        from gamst.providers.sports_ball import BallTrack
        from gamst.providers.sports_contacts import BallContacts
        self.track = BallTrack(ball='detect', max_speed=10.)
        self.contacts = BallContacts(ball='ball', players='players', span=2, min_turn_deg=60)

    def feed(self, points, players):
        values = []
        for i, frame in enumerate(flight(points)):
            values.append(self.contacts.process(record('players', players, frame.timestamp_ms, i)))
            state = self.track.process(frame)
            values.append(self.contacts.process(record('ball', state, frame.timestamp_ms, i)))
        values.append(self.contacts.finish())
        return values

    def test_bounce_away_from_players_and_hit_near_one(self):
        # the ball turns down at sample 20 near the far player's box (a hit), and up at sample 40 (a bounce)
        points = rally_points()
        hit_point, bounce_point = points[19], points[39]
        players = objects(('far', 'person', hit_point[0] - .03, hit_point[1] - .05, .06, .1),
                          ('near', 'person', .4, .85, .1, .12))
        found = events(self.feed(points, players))
        kinds = [(event['kind'], event.get('who')) for event in found]
        self.assertIn(('hit', 'far'), kinds)
        self.assertIn(('bounce', None), kinds)
        bounce = next(event for event in found if event['kind'] == 'bounce')
        self.assertAlmostEqual(bounce['y'], bounce_point[1], delta=.03)

    def test_a_straight_flight_has_no_contact(self):
        found = events(self.feed([(.1 + .01 * i, .5) for i in range(30)], objects()))
        self.assertEqual(found, [])

    def test_waits_for_the_sound_module_and_marks_the_impact(self):
        from gamst.providers.sports_ball import BallTrack
        from gamst.providers.sports_contacts import BallContacts
        track, contacts = BallTrack(ball='detect', max_speed=10.), BallContacts(ball='ball', sound='onsets', span=2)
        points = [(.5, .5 - .02 * i) for i in range(10)] + [(.5, .32 + .01 * i) for i in range(25)]
        answers = []
        for i, frame in enumerate(flight(points)):
            answers.append(contacts.process(record('ball', track.process(frame), frame.timestamp_ms, i)))
        self.assertEqual(events(answers), [])          # nothing decided before the sound module answered
        onset = dict(events=[dict(event='audio.onset', timestamp_ms=95)])
        answer = contacts.process(record('onsets', onset, 400))
        event = events([answer])[0]
        self.assertTrue(event['sound'])
        self.assertEqual(event['kind'], 'deflection')     # no players module: not a hit, and up->down is no bounce


class RallyTest(unittest.TestCase):
    def contact(self, kind, ms, **extra):
        return record('contacts', dict(events=[dict(event=f'contact.{kind}', kind=kind, timestamp_ms=ms, **extra)]), ms)

    def test_hits_and_bounces_until_no_contact(self):
        from gamst.providers.sports_rally import Rally
        rally = Rally(contacts='contacts', ball='ball', max_gap_ms=1000, min_hits=2)
        out = [rally.process(self.contact('hit', 0, who='a')), rally.process(self.contact('bounce', 400, half='far')),
               rally.process(self.contact('hit', 600, who='b')), rally.process(self.contact('bounce', 900, half='near'))]
        out.append(rally.process(record('ball', dict(state='tracking', at_ms=2000), 2000)))
        ended = events(out, 'rally.ended')[0]
        self.assertEqual((ended['hits'], ended['bounces'], ended['who'], ended['ended_by']), (2, 2, ['a', 'b'], 'no_contact'))
        self.assertFalse(ended['short'])

    def test_two_bounces_on_one_side_end_the_point(self):
        from gamst.providers.sports_rally import Rally
        rally = Rally(contacts='contacts', ball='ball')
        out = [rally.process(self.contact('hit', 0)), rally.process(self.contact('bounce', 300, half='far')),
               rally.process(self.contact('bounce', 700, half='far'))]
        self.assertEqual(events(out, 'rally.ended')[0]['ended_by'], 'double_bounce')

    def test_a_bounce_before_any_hit_starts_nothing(self):
        from gamst.providers.sports_rally import Rally
        rally = Rally(contacts='contacts', ball='ball')
        self.assertIsNone(rally.process(self.contact('bounce', 0)))
        self.assertIsNone(rally.finish())


class LongPassTest(unittest.TestCase):
    def test_a_pass_that_flies_loose_is_still_a_pass(self):
        from gamst.providers.sports_possession import Possession
        holder = Possession(ball='ball', players='teams', hold_ms=100, loose_ms=300, hidden_ms=200)
        players = objects(('a1', 'person', .10, .40, .05, .1, 'A'), ('a2', 'person', .80, .40, .05, .1, 'A'))
        out, ms = [], 0
        for x in [.12] * 5 + [.3 + .01 * i for i in range(30)] + [.82] * 5:     # 1.2 s in the air, far from both
            out.append(holder.process(record('teams', players, ms)))
            out.append(holder.process(record('ball', dict(state='tracking', at_ms=ms, x=x, y=.45, flight=1, events=[]), ms)))
            ms += 40
        kinds = [event['event'] for event in events(out)]
        self.assertEqual(kinds, ['possession.started', 'possession.loose', 'possession.pass'])
        self.assertTrue(events(out, 'possession.pass')[0]['via_loose'])


class PossessionTest(unittest.TestCase):
    def ball(self, x, y, ms):
        return record('ball', dict(state='tracking', at_ms=ms, x=x, y=y, flight=1, events=[]), ms)

    def test_pass_within_a_team_and_turnover_to_the_other(self):
        from gamst.providers.sports_possession import Possession
        holder = Possession(ball='ball', players='teams', hold_ms=100)
        players = objects(('a1', 'person', .10, .40, .05, .1, 'A'), ('a2', 'person', .50, .40, .05, .1, 'A'),
                          ('b1', 'person', .80, .40, .05, .1, 'B'))
        out = []
        ms = 0
        for x in [.12] * 5 + [.3] * 3 + [.52] * 5 + [.65] * 3 + [.82] * 5:
            out.append(holder.process(record('teams', players, ms)))
            out.append(holder.process(self.ball(x, .45, ms)))
            ms += 40
        kinds = [event['event'] for event in events(out)]
        self.assertEqual(kinds, ['possession.started', 'possession.pass', 'possession.turnover'])
        last = [value for value in out if value][-1]
        self.assertEqual((last['passes'], last['turnovers'], last['holder'], last['team']), (1, 1, 'b1', 'B'))
        self.assertTrue(all(event['estimated'] for event in events(out)))


@unittest.skipIf(np is None, 'numpy')
class OccupancyTest(unittest.TestCase):
    def test_distance_in_metres_with_a_court(self):
        from gamst.providers.sports_occupancy import Occupancy
        court = dict(court='football', points=[[0, 0], [1, 0], [1, 1], [0, 1]])
        occupancy = Occupancy(players='players', court=court, every_ms=1000, cols=4, rows=2)
        out = []
        for i in range(11):             # the feet move 1/68 of the width (1 m) every 100 ms
            x = .1 + i / 68
            out.append(occupancy.process(record('players', objects(('p', 'person', x - .01, .3, .02, .2, 'A')), i * 100)))
        summary = [value for value in out if value][-1]
        self.assertEqual(summary['unit'], 'm')
        self.assertAlmostEqual(summary['tracks'][0]['distance'], 10., delta=.05)
        self.assertAlmostEqual(summary['tracks'][0]['speed'], 10., delta=1.)
        self.assertEqual(sum(summary['grid']['cells']), 11)
        self.assertEqual(summary['teams']['A']['players'], 1)


@unittest.skipIf(np is None, 'numpy')
class TeamSplitTest(unittest.TestCase):
    def test_red_and_blue_shirts_become_two_teams(self):
        from gamst.providers.sports_team import TeamSplit
        split = TeamSplit(players='players', video='camera', fit_ms=0, min_samples=4, downsample=1)
        image = np.zeros((100, 200, 3), np.uint8)
        image[:, :100] = (0, 0, 220)        # BGR red on the left half
        image[:, 100:] = (220, 0, 0)        # blue on the right
        players = objects(('r1', 'person', .05, .1, .2, .8), ('r2', 'person', .25, .1, .2, .8),
                          ('b1', 'person', .55, .1, .2, .8), ('b2', 'person', .75, .1, .2, .8))
        value = None
        for ms in range(0, 60, 10):
            run(split.process(InputFrame(image, 'video', ms, 'local-user', None,
                                         SensorMetadata(source_id='test', sensor_id='camera', sequence=ms))))
            value = run(split.process(record('players', players, ms)))
        teams = {item['track_id']: item['attributes'].get('team') for item in value['objects']}
        self.assertEqual(teams['r1'], teams['r2'])
        self.assertEqual(teams['b1'], teams['b2'])
        self.assertNotEqual(teams['r1'], teams['b1'])
        self.assertTrue(value['summary']['fitted'])

    def test_kmeans_is_deterministic(self):
        from gamst.providers.sports_team import kmeans
        points = [[0, 0], [0, .1], [5, 5], [5, 5.1]]
        self.assertEqual(list(kmeans(points, 2)[1]), list(kmeans(points, 2)[1]))


class FormTest(unittest.TestCase):
    def test_reached_range_against_the_examples(self):
        from gamst.providers.pose_exemplar import CHANNELS
        from gamst.providers.sports_form import FormCompare
        elbow = CHANNELS.index('angle_left_elbow')

        def vector(angle):
            values = [0.] * len(CHANNELS)
            values[elbow] = angle
            return values
        form = FormCompare(channels='pose_channels', exemplar='exemplar', exit=.5)
        reference = dict(examples=[dict(items=[dict(channels=vector(.2)), dict(channels=vector(.9))])])
        form.bind(SimpleNamespace(references=dict(exemplar=dict(kind='exemplar', value=reference))))
        for ms, angle in ((100, .3), (200, .6), (300, .7)):
            form.process(record('pose_channels', dict(channels=vector(angle), mask=[True] * len(CHANNELS)), ms))
        out = form.process(record('exemplar', dict(events=[dict(event='exemplar.matched', index=1, start_ms=100,
                                                               end_ms=300, distance=.1)]), 300))
        event = out['events'][0]
        angles = event['angles']['angle_left_elbow']
        self.assertEqual((angles['min'], angles['max'], angles['ref_min'], angles['ref_max']), (54.0, 126.0, 36.0, 162.0))
        self.assertEqual(angles['range_diff'], -54.0)
        self.assertEqual(event['largest'], 'angle_left_elbow')
        self.assertEqual(event['score'], .8)
        self.assertEqual(out['compared'], 1)


class SpinTest(unittest.TestCase):
    def pose(self, width, ankle_y=.9):
        """COCO-WholeBody landmarks: shoulders at y .3, hips at .5, signed half width ``width``."""
        points = [dict(x=.5, y=.5, visibility=1.) for _ in range(133)]
        for left, right, y in ((5, 6, .3), (11, 12, .5)):
            points[left] = dict(x=.5 - width / 2, y=y, visibility=1.)
            points[right] = dict(x=.5 + width / 2, y=y, visibility=1.)
        points[15] = points[16] = dict(x=.5, y=ankle_y, visibility=1.)
        return [dict(keypoint_set='coco_wholebody133', bbox=dict(x=.3, y=.2, width=.4, height=.7), landmarks=points)]

    def test_two_turns_in_the_air(self):
        from gamst.providers.sports_spin import BodySpin
        spin = BodySpin()
        out = []
        for i in range(10):
            out.append(spin.process(record('pose_joints', self.pose(.1), i * 10)))
        for i in range(40):     # airborne (ankles .2 up), shoulders rotating: cos goes through 4 sign changes
            width = .1 * math.cos(math.pi * (i + 1) / 10)
            out.append(spin.process(record('pose_joints', self.pose(width, ankle_y=.7), 100 + i * 10)))
        for i in range(5):
            out.append(spin.process(record('pose_joints', self.pose(.1), 500 + i * 10)))
        jump = events(out, 'spin.jump')[0]
        self.assertEqual(jump['turns'], 2.)
        self.assertEqual(jump['airtime_ms'], 400)
        self.assertEqual(out[-1]['turns'], 2.)


class MotionTest(unittest.TestCase):
    def test_a_shot_with_one_contact(self):
        from gamst.providers.sports_motion import ObjectMotion
        motion = ObjectMotion(objects='balls', still_ms=100, moving=.05, touch=1.5, min_dv=.05)
        out, ms = [], 0
        cue, red = [.2, .5], [.6, .5]
        for _ in range(10):         # all still
            out.append(motion.process(record('balls', objects(('cue', 'sports ball', *cue, .04, .04),
                                                             ('red', 'sports ball', *red, .04, .04)), ms)))
            ms += 20
        for step in range(40):
            if cue[0] < red[0] - .045:
                cue[0] += .01                      # 0.5 widths per second towards the red
            else:
                red[0] += .01                      # the red takes over, the cue stops
            seen = [('red', 'sports ball', *red, .04, .04)] if step in (5, 6) else \
                [('cue', 'sports ball', *cue, .04, .04), ('red', 'sports ball', *red, .04, .04)]   # the cue missed twice
            out.append(motion.process(record('balls', objects(*seen), ms)))
            ms += 20
        for _ in range(15):
            out.append(motion.process(record('balls', objects(('cue', 'sports ball', *cue, .04, .04),
                                                             ('red', 'sports ball', *red, .04, .04)), ms)))
            ms += 20
        started = events(out, 'shot.started')
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]['first'], 'cue')
        self.assertGreaterEqual(len(events(out, 'contact.objects')), 1)
        self.assertEqual(events(out, 'shot.ended')[0]['moved'], 2)


@unittest.skipIf(np is None, 'numpy')
class TrackerTest(unittest.TestCase):
    def test_ocsort_keeps_one_id_and_passes_the_ball_through(self):
        from gamst.providers.sports_track import ObjectTracker
        tracker = ObjectTracker(objects='detect', tracker='ocsort', labels=('person',))
        ids = set()
        for i in range(8):
            value = tracker.process(record('detect', objects(('d', 'person', .1 + .005 * i, .2, .1, .3), ball_box(.5, .5)),
                                           i * 33))
            ids.update(item['track_id'] for item in value['objects'] if item['label'] == 'person')
            self.assertIn('ball', [item['track_id'] for item in value['objects']])
        self.assertEqual(len(ids), 1)

    def test_roboflow_trackers_path_with_stand_ins(self):
        """The trackers/supervision calls as their docs show them (stand-ins: the packages are not installed here)."""
        import sys
        from unittest import mock

        class Detections:
            def __init__(self, xyxy, confidence=None, class_id=None, data=None):
                self.xyxy, self.confidence, self.class_id, self.data = xyxy, confidence, class_id, data or {}
                self.tracker_id = None

            def __len__(self):
                return len(self.xyxy)

        class ByteTrackTracker:
            def __init__(self, frame_rate=30.):
                self.frame_rate = frame_rate

            def update(self, detections, frame=None, timestamp=None):
                self.timestamp = timestamp
                detections.tracker_id = np.arange(len(detections)) + 41
                return detections
        modules = {'supervision': SimpleNamespace(Detections=Detections),
                   'trackers': SimpleNamespace(ByteTrackTracker=ByteTrackTracker)}
        with mock.patch.dict(sys.modules, modules):
            from gamst.providers.sports_track import ObjectTracker
            tracker = ObjectTracker(objects='detect', tracker='bytetrack', labels=('person',), frame_rate=60.)
            self.assertEqual(tracker._tracker.frame_rate, 60.)
            value = tracker.process(record('detect', dict(objects(('d', 'person', .1, .2, .1, .3), ball_box(.5, .5)),
                                                          summary=dict(width=1280, height=720)), 0))
        person = next(item for item in value['objects'] if item['label'] == 'person')
        self.assertEqual(person['track_id'], 'track-41')
        self.assertEqual(tracker._tracker.timestamp, 0.)
        self.assertAlmostEqual(person['location']['bbox']['x'], .1, places=5)
        self.assertEqual(value['summary']['width'], 1280)

    def test_none_passes_a_tracking_detector_through(self):
        from gamst.providers.sports_track import ObjectTracker
        tracker = ObjectTracker(objects='detect', tracker='none')
        value = tracker.process(record('detect', objects(('track-7', 'person', .1, .2, .1, .3)), 0))
        self.assertEqual(value['objects'][0]['track_id'], 'track-7')


@unittest.skipIf(np is None, 'numpy')
class TrackNetStreamTest(unittest.TestCase):
    def heatmaps(self, positions):
        """A fake network: one heatmap per window position with a blob where ``positions`` says."""
        def infer(stacked):
            maps = np.zeros((len(positions), 288, 512), np.float32)
            for index, (x, y) in enumerate(positions):
                maps[index, y - 2:y + 3, x - 2:x + 3] = .9
            return maps
        return infer

    def ball(self, ensemble, seq_len=3):
        from gamst.providers.tracknet_ball import TrackNetBall
        return TrackNetBall(None, seq_len=seq_len, bg_mode='concat', ensemble=ensemble,
                            infer=self.heatmaps([(100, 50), (200, 100), (300, 150)][:seq_len]))

    def frame(self, ms):
        return InputFrame(np.zeros((72, 128, 3), np.uint8), 'video', ms, 'local-user', None,
                          SensorMetadata(source_id='test', sensor_id='camera', sequence=ms))

    def test_causal_answers_each_frame_at_once(self):
        ball = self.ball('causal')
        value = ball.process(self.frame(0))
        self.assertEqual(value['frames'][0]['frame_ms'], 0)
        self.assertAlmostEqual(value['frames'][0]['balls'][0]['x'], 300 / 512, places=3)

    def test_weighted_ensemble_answers_seq_len_minus_one_frames_later_and_finish_flushes(self):
        ball = self.ball('weight')
        answers = [ball.process(self.frame(ms))['frames'] for ms in (0, 10, 20, 30, 40)]
        self.assertEqual([[item['frame_ms'] for item in found] for found in answers], [[], [], [0], [10], [20]])
        rest = ball.finish()['frames']
        self.assertEqual([item['frame_ms'] for item in rest], [30, 40])

    def test_edge_frames_get_the_plain_mean_and_full_ones_the_weights(self):
        from gamst.providers.tracknet_ball import TrackNetBall
        calls = []

        def infer(stacked):
            calls.append(1)
            maps = np.zeros((3, 288, 512), np.float32)
            maps[:, 10:15, 10:15] = [[[.4]], [[.9]], [[.4]]]     # the middle of each window sees the ball best
            return maps
        ball = TrackNetBall(None, seq_len=3, ensemble='weight', infer=infer)
        out = [ball.process(self.frame(ms))['frames'] for ms in (0, 10, 20, 30, 40)]
        first, full = out[2][0], out[4][0]          # frame 0: one window, plain .4; frame 20: three windows
        self.assertEqual(first['balls'], [])         # .4 is below the authors' 0.5 threshold
        self.assertEqual(full['balls'][0]['score'], round((.4 * 1 + .9 * 2 + .4 * 1) / 4, 4))
        self.assertEqual(len(calls), 3)              # padded start windows are not run in post

    def test_ensemble_weights_are_the_authors(self):
        from gamst.providers.tracknet_ball import ensemble_weight, input_channels
        self.assertEqual([round(v * 20, 6) for v in ensemble_weight(8)], [1., 2., 3., 4., 4., 3., 2., 1.])
        self.assertEqual(input_channels(8, 'concat'), 27)

    def test_network_shape_and_checkpoint_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch')
        from gamst.providers.tracknet_ball import TrackNetBall, network
        model = network(12, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'TrackNet_best.pt')
            torch.save(dict(param_dict=dict(seq_len=3, bg_mode='concat'), model=model.state_dict()), path)
            import hashlib
            digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            ball = TrackNetBall.from_local(path=path, sha256=digest)
        with torch.no_grad():
            out = ball.model(torch.zeros(1, 12, 32, 64))
        self.assertEqual(tuple(out.shape), (1, 3, 32, 64))


@unittest.skipIf(np is None, 'numpy')
class UltralyticsAdapterTest(unittest.TestCase):
    def test_boxes_and_track_ids_become_objects(self):
        from gamst.providers.ultralytics_sports import UltralyticsObjects

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Boxes:
            xyxy, conf = Tensor([[10, 20, 30, 60], [50, 50, 54, 54]]), Tensor([.9, .6])
            cls, id = Tensor([0, 32]), None

            def __len__(self):
                return 2

            def cpu(self):
                return self

            def numpy(self):
                return self

        class Model:
            names = {0: 'person', 32: 'sports ball'}

            def predict(self, image, **options):
                self.options = options
                return [SimpleNamespace(boxes=Boxes(), names=self.names)]

        class Tracker:      # Ultralytics' tracker: only the person is a confirmed track; the fast ball is not
            def update(self, det, image):
                return np.asarray([[10, 20, 30, 60, 3, .9, 0, 0]])

            def reset(self):
                self.was_reset = True
        config = dict(path='-', sha256='0' * 64, tracker='botsort')
        provider = UltralyticsObjects(Model(), config, tracker=Tracker())
        value = provider.process(InputFrame(np.zeros((100, 100, 3), np.uint8), 'video', 0))
        self.assertEqual([item['track_id'] for item in value['objects']], ['track-3', 'object-1'])
        self.assertEqual(value['objects'][1]['label'], 'sports ball')      # kept although its track is unconfirmed
        self.assertAlmostEqual(value['objects'][0]['location']['bbox']['height'], .4)
        provider.reset()
        self.assertTrue(provider._tracker.was_reset)

    def test_config_is_checked(self):
        from gamst.providers.ultralytics_sports import check_config
        with self.assertRaises(ValueError):
            check_config(dict(path='x', sha256='0' * 64, tracker='deepsort'))
        with self.assertRaises(ValueError):
            check_config(dict(path='x', sha256='0' * 64, imgsz=641))
        with self.assertRaises(ValueError):
            check_config(dict(path='x', sha256='0' * 64, max_poses=8), pose=True)


RECIPES = ('racket-rally', 'team-field', 'body-reps', 'swing', 'body-rotation', 'billiards')


class SportsRecipesTest(unittest.TestCase):
    def test_every_stack_and_timing_loads_and_names_its_models(self):
        from gamst.recipes import TIMINGS, load, read_recipe, roles, stack_names
        for name in RECIPES:
            raw = read_recipe(name)[1]
            self.assertEqual(raw.get('category'), 'sports')
            for stack in stack_names(raw):
                for timing in TIMINGS:
                    recipe = load(name, stack=stack, timing=timing)
                    self.assertEqual(recipe['chosen']['stack'], stack)
                    self.assertTrue(roles(recipe))

    def test_the_ultralytics_stack_uses_yolo26_and_its_own_tracker(self):
        from gamst.recipes import load
        post = {item['name']: item for item in load('racket-rally', stack='ultralytics', timing='post')['modules']}
        live = {item['name']: item for item in load('racket-rally', stack='ultralytics', timing='realtime')['modules']}
        self.assertEqual(post['detect']['candidate'], 'yolo26.x')
        self.assertEqual(live['detect']['candidate'], 'yolo26.m')
        self.assertEqual(live['detect']['choose']['fallback'], 'yolo26.s')
        self.assertEqual(post['players']['tracker'], 'none')
        self.assertNotIn('ballmodel', post)
        self.assertEqual(post['ball']['consumes'], ['detect'])

    def test_sport_presets_reach_the_modules(self):
        from gamst.recipes import load
        modules = {item['name']: item for item in load('racket-rally', sport='table-tennis')['modules']}
        self.assertEqual(modules['ball']['sport'], 'table-tennis')
        self.assertEqual(modules['contacts']['min_contact_ms'], 80)
        with self.assertRaises(SystemExit):
            load('racket-rally', sport='football')

    def test_sound_is_optional(self):
        from gamst.recipes import load
        recipe = load('racket-rally', exclude=('onsets',))
        contacts = next(item for item in recipe['modules'] if item['name'] == 'contacts')
        self.assertEqual(contacts['consumes'], ['ball', 'players'])
        self.assertEqual([item['id'] for item in recipe['inputs']], ['camera'])

    def test_frame_rate_check_uses_the_sport(self):
        from gamst.recipes import check, load
        self.assertFalse(check(load('racket-rally', sport='table-tennis'), fps=25)['ok'])
        verdict = check(load('racket-rally', sport='badminton'), fps=30)
        self.assertTrue(verdict['ok'])
        self.assertEqual(len(verdict['warnings']), 1)
        self.assertTrue(check(load('racket-rally', sport='tennis'), fps=60)['ok'])
        with self.assertRaises(SystemExit):
            load('racket-rally', stack='default')

    def test_a_part_prepared_by_hand_is_left_to_its_config(self):
        from gamst.recipes import load, roles, with_settings
        recipe = with_settings(load('racket-rally', stack='sota', timing='post'),
                               ['ballmodel.config={"path": "/m/TrackNet_best.pt", "sha256": "' + '0' * 64 + '"}'])
        self.assertNotIn('vision.ball', roles(recipe))
        self.assertIn('vision.ball', roles(load('racket-rally', stack='sota', timing='post')))

    def test_stacks_table(self):
        from gamst.recipes import stacks_table
        table = stacks_table('racket-rally')
        self.assertEqual(table['default'], 'sota')
        self.assertEqual(set(table['stacks']), {'sota', 'roboflow', 'ultralytics'})
        self.assertEqual(table['stacks']['roboflow']['post']['players']['tracker'], 'bytetrack')

    def test_override_rules(self):
        from gamst.recipes import _override
        modules = [dict(name='a', use='x', k=1), dict(name='b', use='y')]
        changed = _override(modules, dict(a=dict(k=None, j=2), b=dict(remove=True)), 'test')
        self.assertEqual(changed, [dict(name='a', use='x', j=2)])
        with self.assertRaises(ValueError):
            _override(modules, dict(c=dict(k=1)), 'test')


class ChooseModelsTest(unittest.TestCase):
    def test_a_slow_detector_falls_back_in_realtime_and_nothing_is_chosen_in_post(self):
        if np is None:
            self.skipTest('numpy')
        from gamst import recipes

        class Slow:
            resets = 0

            async def process(self, frame):
                await asyncio.sleep(.05)        # 20 fps at most

            async def reset(self):
                Slow.resets += 1
        recipe = dict(modules=[dict(name='detect', use='detect', candidate='rfdetr.large',
                                    choose=dict(fallback='rfdetr.nano', seconds=.2, size=[16, 16]))])
        pipeline = SimpleNamespace(modules=dict(detect=dict(module=Slow())))
        live = SimpleNamespace(camera=0, video=None, take=None, iphone=None, audio=None, microphone=None, sensor=None,
                               timing=None, as_live=False, realtime=None, capture_fps=30.)
        original = recipes._source_fps
        recipes._source_fps = lambda args: 30.
        try:
            chosen = run(recipes.choose_models(recipe, live, pipeline))
        finally:
            recipes._source_fps = original
        self.assertEqual(chosen[0]['chosen'], 'rfdetr.nano')
        self.assertEqual(Slow.resets, 1)
        post = SimpleNamespace(**dict(vars(live), camera=None, video='a.mp4', timing='post'))
        self.assertEqual(run(recipes.choose_models(recipe, post, pipeline)), [])


if __name__ == '__main__':
    unittest.main()


class FakeDetector:
    """A detector module: the n-th frame shows the ball at the n-th point and two players."""

    def __init__(self, points, players):
        from gamst.capabilities import CapabilitySpec, OutputField
        self.points, self.players = points, players
        self.spec = CapabilitySpec('vision.test.objects', 'vision.objects', 'video', 'test.detector', '1',
                                   (OutputField('objects', 'array'), OutputField('summary', 'object')), 'frame')

    def process(self, frame):
        x, y = self.points[frame.data]
        value = objects(*self.players, ball_box(x, y))
        return self.spec.validate(value)


class FrameSource:
    modality = 'video'

    def __init__(self, count, start, step_ms=10):
        self.count, self.start, self.step = count, start, step_ms

    async def frames(self):
        for index in range(self.count):
            yield InputFrame(index, 'video', self.start + self.step * index)


class SportsPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_records_flow_from_detector_to_rally_and_are_counted(self):
        """The record links of racket-rally (detect -> ball -> contacts -> rally) in a real Pipeline and Store."""
        import time
        from gamst import Gamst
        from gamst.pipeline import Pipeline
        from gamst.providers.sports_ball import BallTrack
        from gamst.providers.sports_contacts import BallContacts
        from gamst.providers.sports_rally import Rally
        points = rally_points()
        players = (('far', 'person', points[19][0] - .03, points[19][1] - .05, .06, .1), ('near', 'person', .4, .85, .1, .12))
        start = time.time_ns() // 1_000_000 - 5000
        async with Gamst(consent=True) as human:
            pipeline = Pipeline(human, 'sports-test')
            pipeline.add_input('camera', FrameSource(len(points), start))
            detector = FakeDetector(points, players)
            pipeline.add_module('detect', detector, consumes='camera', spec=detector.spec)
            ball = BallTrack(ball='detect', max_speed=10.)
            pipeline.add_module('ball', ball, consumes='detect', spec=ball.spec)
            contacts = BallContacts(ball='ball', players='detect', span=2, min_turn_deg=60)
            pipeline.add_module('contacts', contacts, consumes=['ball', 'detect'], spec=contacts.spec)
            rally = Rally(contacts='contacts', ball='ball')
            pipeline.add_module('rally', rally, consumes=['contacts', 'ball'], spec=rally.spec)
            await pipeline.run()
            hits = await human.count_events('sports.contacts', event='contact.hit')
            bounces = await human.count_events('sports.contacts', event='contact.bounce')
            rallies = await human.count_events('sports.rally', event='rally.ended')
        self.assertEqual(hits['count'], 1)
        self.assertEqual(bounces['count'], 1)
        self.assertEqual(rallies['count'], 1)
