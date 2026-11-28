"""Identity recall: SCRFD decoding and face choice, the head region, which crop a face model is given, the face worker's
input kind, and the binding replay over a recorded dump. Test doubles only (no model weights)."""
import unittest

try:
    import numpy as np
    import cv2  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False


def outputs_with(faces, size=640):
    """Nine SCRFD outputs (scores, boxes, keypoints of strides 8, 16, 32) with ``faces``: (stride, row, col, anchor,
    score, (l, t, r, b) in strides, five (dx, dy) in strides)."""
    scores, boxes, points = [], [], []
    for stride in (8, 16, 32):
        n = (size // stride) ** 2 * 2
        scores.append(np.zeros((n, 1), np.float32))
        boxes.append(np.zeros((n, 4), np.float32))
        points.append(np.zeros((n, 10), np.float32))
    for stride, row, col, anchor, score, ltrb, five in faces:
        level = (8, 16, 32).index(stride)
        index = (row * (size // stride) + col) * 2 + anchor
        scores[level][index] = score
        boxes[level][index] = ltrb
        points[level][index] = [v for point in five for v in point]
    return scores + boxes + points


FIVE = ((-1., -1.), (1., -1.), (0., 0.), (-.8, 1.), (.8, 1.))


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class SCRFDTest(unittest.TestCase):
    def test_decode_places_boxes_and_points_at_their_anchor_in_image_pixels(self):
        from gamst.providers.face_detect import decode
        found = decode(outputs_with([(16, 10, 20, 1, .9, (2., 3., 2., 3.), FIVE)]), scale=2.)
        self.assertEqual(len(found), 1)
        score, box, points = found[0]
        self.assertAlmostEqual(score, .9, places=5)
        # centre (20 * 16, 10 * 16) = (320, 160); distances times the stride; then / scale
        self.assertEqual(tuple(round(v, 3) for v in box), ((320 - 32) / 2, (160 - 48) / 2, (320 + 32) / 2, (160 + 48) / 2))
        self.assertEqual(tuple(round(v, 3) for v in points[0]), ((320 - 16) / 2, (160 - 16) / 2))
        self.assertEqual(decode(outputs_with([(16, 10, 20, 1, .2, (2., 3., 2., 3.), FIVE)]), scale=1.), [])

    def test_overlapping_detections_keep_the_best_and_the_face_covering_the_centre_is_picked(self):
        from gamst.providers.face_detect import decode, pick
        found = decode(outputs_with([(8, 40, 40, 0, .8, (4., 4., 4., 4.), FIVE), (8, 40, 40, 1, .95, (4., 4., 4., 4.), FIVE),
                                     (8, 10, 10, 0, .99, (3., 3., 3., 3.), FIVE)]), scale=1.)
        self.assertEqual([round(item[0], 2) for item in found], [.99, .95])
        chosen = pick(found, 640, 640)          # the centre (320, 320) is the second face's
        self.assertAlmostEqual(chosen[0], .95, places=5)
        self.assertIsNone(pick(found[:1], 640, 640))      # only a face beside the head: none

    def test_the_detected_face_is_aligned_from_its_own_points_and_a_missing_face_is_no_evidence(self):
        from gamst.providers.face_detect import DetectedFaceEmbedder

        class Detector:
            def __init__(self, faces):
                self.faces = faces

            def detect(self, image):
                return self.faces

        class Embedder:
            model_id = 'lvface:test'

            def embed(self, face):
                return [float(face.shape[0]), float(face.shape[1])]
        region = np.zeros((100, 100, 3), np.uint8)
        five = [(40., 45.), (60., 45.), (50., 55.), (42., 65.), (58., 65.)]
        model = DetectedFaceEmbedder(Detector([(.9, (35., 35., 65., 75.), five)]), Embedder())
        self.assertEqual((model.model_id, model.face_input), ('lvface:test', 'region'))
        self.assertEqual(model.embed(region), [112., 112.])
        tiny = [(49., 50.), (51., 50.), (50., 51.), (49., 52.), (51., 52.)]
        for faces in ([], [(.9, (45., 45., 55., 55.), tiny)], [(.4, (35., 35., 65., 75.), five)]):
            with self.assertRaises(ValueError):
                DetectedFaceEmbedder(Detector(faces), Embedder(), min_eye_px=6.).embed(region)
        with self.assertRaises(ValueError):
            DetectedFaceEmbedder(Detector([]), Embedder(), interpolation='nearest')


def rtmw_pose(head=(.5, .3), shoulders=((.4, .5), (.6, .5)), visible=1., box=(.3, .2, .4, .7)):
    marks = [dict(x=0., y=0., visibility=0.) for _ in range(133)]
    for index, (dx, dy) in zip(range(5), ((0., 0.), (.02, -.02), (-.02, -.02), (.04, 0.), (-.04, 0.))):
        marks[index] = dict(x=head[0] + dx, y=head[1] + dy, visibility=visible)
    for index, point in zip((5, 6), shoulders):
        marks[index] = dict(x=point[0], y=point[1], visibility=1.)
    return dict(keypoint_set='coco_wholebody133', landmarks=marks,
                bbox=dict(x=box[0], y=box[1], width=box[2], height=box[3]))


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class HeadRegionTest(unittest.TestCase):
    def test_a_square_around_the_head_sized_by_the_shoulders(self):
        from gamst.providers.person_match import HEAD_REGION, head_region
        image = np.zeros((400, 400, 3), np.uint8)
        image[110:130, 190:210] = 255         # a mark at the head (200, 120)
        region = head_region(image, rtmw_pose())
        side = round(HEAD_REGION * 80)       # shoulders 80 px apart
        self.assertEqual(region.shape[:2], (side, side))
        ys, xs = np.nonzero(region[:, :, 0])
        self.assertAlmostEqual((ys.mean() + .5) / side, .5, delta=.05)
        self.assertAlmostEqual((xs.mean() + .5) / side, .5, delta=.05)

    def test_no_head_point_or_a_tiny_or_mostly_outside_square_is_no_region(self):
        from gamst.providers.person_match import head_region
        image = np.zeros((400, 400, 3), np.uint8)
        self.assertIsNone(head_region(image, rtmw_pose(visible=0.)))
        self.assertIsNone(head_region(image, rtmw_pose(shoulders=((.49, .5), (.51, .5)), box=(.45, .2, .05, .7))))
        self.assertIsNone(head_region(image, rtmw_pose(head=(1.1, -.1))))
        edge = head_region(image, rtmw_pose(head=(.05, .3)))      # partly outside: kept centred, black outside
        self.assertEqual(edge.shape[0], edge.shape[1])

    def test_large_regions_are_shrunk_and_a_face_model_gets_the_crop_it_asks_for(self):
        from gamst.providers.person_match import face_crop, head_region
        image = np.zeros((2000, 2000, 3), np.uint8)
        self.assertEqual(head_region(image, rtmw_pose()).shape[:2], (448, 448))

        class Region:
            face_input = 'region'

        class Aligned:
            face_input = 'aligned'
        small = np.zeros((400, 400, 3), np.uint8)
        self.assertEqual(face_crop(Region(), small, rtmw_pose()).shape[:2], (104, 104))
        aligned = face_crop(Aligned(), small, dict(rtmw_pose(), landmarks=[
            dict(m, visibility=1.) if i in (0, 1, 2, 71, 77) else m for i, m in enumerate(rtmw_pose()['landmarks'])]))
        self.assertEqual(aligned.shape, (112, 112, 3))      # all five points visible: the pose's own alignment
        self.assertIsNone(face_crop(Aligned(), small, rtmw_pose()))      # mouth corners not visible: no aligned face

    def test_the_face_worker_with_a_detector_asks_for_regions(self):
        from gamst.providers.model_workers import _Client
        sha = '0' * 64
        self.assertEqual(_Client(None, 'face.embedding', dict(sha256=sha)).face_input, 'aligned')
        self.assertEqual(_Client(None, 'face.embedding', dict(sha256=sha, detector_path='d.onnx')).face_input, 'region')
        self.assertEqual(_Client(None, 'face.embedding', dict(sha256=sha, detector_path='d.onnx')).model_id,
                         'lvface:' + sha[:16])


if __name__ == '__main__':
    unittest.main()


class SetupTest(unittest.TestCase):
    def test_setup_gives_the_face_worker_its_detector_and_the_registry_pins_it(self):
        from gamst.crew.registry import ROLES
        from gamst.crew.setup import _role_config
        weights = [dict(name='LVFace-B_Glint360K.onnx', path='/m/l.onnx', sha256='a' * 64),
                   dict(name='det_10g.onnx', path='/m/d.onnx', sha256='b' * 64)]
        self.assertEqual(_role_config(dict(model='lvface.b_scrfd', config={}), weights, None),
                         dict(path='/m/l.onnx', sha256='a' * 64, detector_path='/m/d.onnx', detector_sha256='b' * 64))
        self.assertEqual(_role_config(dict(model='lvface.b', config={}), weights[:1], None),
                         dict(path='/m/l.onnx', sha256='a' * 64))
        candidate = ROLES['vision.face_identity'].candidates['lvface.b_scrfd']
        self.assertEqual([item.name for item in candidate.weights], ['LVFace-B_Glint360K.onnx', 'det_10g.onnx'])
        self.assertEqual(candidate.weights[1].sha256, '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91')


def synthetic_dump(tracks, cuts=(), step=100, until=4000):
    """A pose dump with boxes only: ``tracks`` {track: (first, last, x)}; ``cuts``: scene cuts decided at once."""
    from gamst.eval.pose_dump import kept_indices
    indices = sorted(set(kept_indices().values()))
    frames = []
    for ts in range(0, until + 1, step):
        rows = [[track, [x, .2, .2, .6], 640, 360, 'coco_wholebody133', [None] * len(indices)]
                for track, (first, last, x) in tracks.items() if first <= ts <= last]
        frames.append(('poses', ts, rows))
        if ts in cuts:
            frames.append(('binding', ts, [dict(event='scene.cut', start_ms=ts, end_ms=ts)]))
    return dict(header=dict(indices=indices), frames=frames, events=[])


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class ScenePropagationTest(unittest.TestCase):
    """Track 'a' (no face, appearance A) ends; then track 'b' shows the person's face with appearance A."""
    FACE, OTHER, LOOK = [0., 0., 1.], [1., 0., 0.], [1., 0., .05]

    def replay(self, *, cuts=(), a_face=None, a_look=None, **options):
        from gamst.eval.binding_replay import replay
        tracks = dict(a=(0, 1000, .1), b=(1200, 4000, .5))
        evidence = {}
        for ts in range(0, 4001, 100):
            if ts <= 1000:
                evidence[(ts, 'a')] = (a_face, a_look or self.LOOK)
            if ts >= 1200:
                evidence[(ts, 'b')] = (self.FACE, self.LOOK)
        reference = dict(kind='person', name='P', face_model='face', face=[self.FACE] * 3, consent=dict(statement='t'))
        options = dict(dict(face_threshold=.5, face_reject=.2, body_threshold=.99, body_reject=.5, anonymous=False,
                            delay_ms=100, slack_ms=0), **options)
        events, module = replay(synthetic_dump(tracks, cuts), evidence, {'person': reference}, face_model='face',
                                body_model='body', **options)
        return [(item['event'], item['track_id'], item.get('by') or item.get('reason'), item['start_ms'], item['end_ms'])
                for item in events if item.get('track_id') == 'a'], module

    def test_off_the_ended_track_stays_unknown(self):
        found, _ = self.replay()
        self.assertEqual(found, [])

    def test_the_face_confirmed_later_in_the_scene_reaches_back_to_the_ended_track_over_its_own_span(self):
        found, module = self.replay(scene_threshold=.9)
        self.assertEqual([item[:3] for item in found],
                         [('person.bound', 'a', 'scene'), ('person.unbound', 'a', 'scene_ended')])
        self.assertEqual(found[1][3:], (0, 1000))
        self.assertEqual(module.counts['scene_propagated'], 1)

    def test_not_over_a_cut_and_not_onto_a_face_that_contradicts(self):
        self.assertEqual(self.replay(scene_threshold=.9, cuts=(1100,))[0], [])
        self.assertEqual(self.replay(scene_threshold=.9, a_face=self.OTHER)[0], [])
        self.assertEqual(self.replay(scene_threshold=.99, a_look=[1., .3, 0.])[0], [])     # a look not close enough
        self.assertEqual(len(self.replay(scene_threshold=.9, a_look=[1., .3, 0.])[0]), 2)

    def test_not_over_a_time_the_person_was_on_another_track(self):
        from gamst.eval.binding_replay import replay
        tracks = dict(a=(0, 2000, .1), b=(1200, 4000, .5))       # 'a' still there while 'b' is the person
        evidence = {(ts, 'a'): (None, self.LOOK) for ts in range(0, 2001, 100)}
        evidence.update({(ts, 'b'): (self.FACE, self.LOOK) for ts in range(1200, 4001, 100)})
        reference = dict(kind='person', name='P', face_model='face', face=[self.FACE] * 3, consent=dict(statement='t'))
        events, _ = replay(synthetic_dump(tracks), evidence, {'person': reference}, face_model='face', body_model='body',
                           face_threshold=.5, face_reject=.2, body_threshold=.99, anonymous=False, scene_threshold=.9,
                           slack_ms=0)
        self.assertEqual([item for item in events if item.get('track_id') == 'a' and item['event'] == 'person.bound'], [])


class ReachBackTest(unittest.TestCase):
    def bound_again(self, reach_back, other=False):
        from gamst.providers.person_binding import TrackBindings
        b = TrackBindings(reach_back=reach_back)
        for ts in (0, 100, 300):
            b.seen('a', ts)
        b.bind('a', 'P', 100, 'face')
        b.unbind('a', 300, 'seen_on_another_track')
        if other:       # the person was on another track in between
            b.seen('b', 400)
            b.bind('b', 'P', 400, 'face')
            b.seen('b', 500)
            b.unbind('b', 500, 'track_lost')
        b.seen('a', 600)
        return b.bind('a', 'P', 600, 'face')

    def test_a_track_bound_again_reaches_back_to_its_last_binding_but_not_over_the_person_elsewhere(self):
        self.assertEqual((self.bound_again(False)['start_ms'], self.bound_again(False)['retroactive']), (600, False))
        self.assertEqual((self.bound_again(True)['start_ms'], self.bound_again(True)['retroactive']), (301, True))
        self.assertEqual(self.bound_again(True, other=True)['start_ms'], 501)

    def test_a_face_of_the_track_that_said_someone_else_still_bounds_it(self):
        from gamst.providers.person_binding import TrackBindings
        b = TrackBindings(reach_back=True)
        for ts in (0, 100, 300, 600):
            b.seen('a', ts)
        b.bind('a', 'P', 100, 'face')
        b.unbind('a', 300, 'face_contradicts')
        self.assertEqual(b.bind('a', 'P', 600, 'face', floor=450)['start_ms'], 451)


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class RejectAfterTest(unittest.TestCase):
    """Track 'a' shows the person's face, then odd faces that contradict it."""
    FACE, OTHER = [0., 0., 1.], [1., 0., 0.]

    def replay(self, faces, **options):
        from gamst.eval.binding_replay import replay
        evidence = {(ts, 'a'): (face, None) for ts, face in faces.items()}
        reference = dict(kind='person', name='P', face_model='face', face=[self.FACE] * 3, consent=dict(statement='t'))
        events, _ = replay(synthetic_dump(dict(a=(0, 4000, .3))), evidence, {'person': reference}, face_model='face',
                           face_threshold=.5, face_reject=.2, anonymous=False, face_every_ms=200, confirm_every_ms=1000,
                           slack_ms=0, **options)
        return [(item['event'], item.get('reason') or item.get('by'), item['start_ms'], item['end_ms']) for item in events
                if item['event'] in ('person.bound', 'person.unbound')]

    def test_one_odd_face_releases_by_default_but_not_with_two_needed(self):
        faces = {ts: self.FACE for ts in range(0, 4001, 100)}
        faces[1000] = self.OTHER           # one blurred face that scores as someone else, then the person again
        once = self.replay(faces)
        self.assertIn('face_contradicts', [item[1] for item in once])
        twice = self.replay(faces, reject_after=2)
        self.assertEqual([item[:2] for item in twice], [('person.bound', 'face'), ('person.unbound', 'end_of_input')])

    def test_two_contradicting_faces_in_a_row_release_back_to_the_last_confirmation(self):
        faces = {ts: self.FACE for ts in range(0, 1001, 100)}
        faces.update({ts: self.OTHER for ts in range(1100, 4001, 100)})
        found = self.replay(faces, reject_after=2)
        released = [item for item in found if item[1] == 'face_contradicts']
        self.assertEqual(len(released), 1)
        self.assertLessEqual(released[0][3], 1000)       # withdrawn back to the last face that said the person


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class WaitForScoresTest(unittest.IsolatedAsyncioTestCase):
    async def test_post_mode_frames_wait_for_their_scores_so_none_is_replaced(self):
        import asyncio
        from tests.test_person_identity import BLUE, RED, frame_of, person, reference
        from gamst.plugins import InputFrame
        from gamst.private_payload import PrivatePoses
        from gamst.providers.person_binding import PersonBinding

        class Slow:
            def __init__(self, model_id):
                self.model_id = model_id

            async def embed(self, image):
                await asyncio.sleep(.02)
                return [float(v) + 1e-6 for v in image.reshape(-1, 3).mean(0)]

        async def run(wait):
            module = PersonBinding(Slow('face'), Slow('body'), face_threshold=.9, face_reject=.5, body_threshold=.95,
                                   face_every_ms=0, body_every_ms=0, session='file:s', anonymous=False,
                                   scene_cuts=False, wait_for_scores=wait)
            module.set_references(dict(person=reference(face=3, name='P') | dict(face=[[0., 0., 1.]] * 3)))
            bound = []
            for ts in range(0, 200, 33):        # frames come faster than the scorer answers
                await module.process(InputFrame(PrivatePoses([person(.75, 'b')]), 'video', ts))
                out = await module.process(InputFrame(frame_of(BLUE, RED), 'video', ts))
                bound += [item for item in (out or {}).get('events') or () if item['event'] == 'person.bound']
            await module.settled()
            await module.close()
            return module.counts.get('replaced_jobs', 0), bound
        replaced, _ = await run(False)
        self.assertGreater(replaced, 0)
        replaced, bound = await run(True)
        self.assertEqual(replaced, 0)
        self.assertEqual([(item['track_id'], item['start_ms']) for item in bound], [('b', 0)])


class WithBindingsTest(unittest.TestCase):
    def test_a_replays_bindings_replace_the_recorded_ones_after_the_poses_of_their_frame(self):
        from gamst.eval.binding_replay import with_bindings
        dump = dict(header={}, events=[], frames=[('poses', 0, []), ('binding', 0, ['old']), ('poses', 33, []),
                                                   ('binding', 33, ['old'])])
        out = with_bindings(dump, [(33, ['new']), (99, ['end'])])
        self.assertEqual(out['frames'], [('poses', 0, []), ('poses', 33, []), ('binding', 33, ['new']),
                                         ('binding', 99, ['end'])])
