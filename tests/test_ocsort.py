import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gamst.tracking import PersonTracker, create_tracker


@unittest.skipIf(np is None, 'optional numpy unavailable')
class OCSORTTests(unittest.TestCase):
    def setUp(self):
        from gamst.providers.ocsort import OCSORTPersonTracker
        self.tracker = OCSORTPersonTracker(max_age=5, max_age_ms=1000)

    @staticmethod
    def box(x, y=.2, width=.12, height=.25):
        return dict(x=x, y=y, width=width, height=height)

    def test_selection_keeps_iou_default(self):
        from gamst.providers.ocsort import OCSORTPersonTracker
        self.assertIsInstance(create_tracker(), PersonTracker)
        self.assertIsInstance(create_tracker('iou'), PersonTracker)
        self.assertIsInstance(create_tracker('ocsort'), OCSORTPersonTracker)
        with self.assertRaises(ValueError):
            create_tracker('bytetrack')

    def test_constant_motion_survives_occlusion_and_reappearance(self):
        first = self.tracker.update([self.box(.05)], 0)[0]
        self.tracker.update([self.box(.08)], 40)
        self.tracker.update([self.box(.11)], 80)
        missing = self.tracker.update([], 120)[0]
        self.assertFalse(missing['visible'])
        self.assertEqual(missing['association_quality'], 'predicted')
        self.assertGreater(missing['bbox']['x'], .11)
        self.tracker.update([], 160)
        recovered = self.tracker.update([self.box(.17)], 200)[0]
        self.assertEqual(recovered['track_id'], first['track_id'])
        self.assertTrue(recovered['visible'])
        self.assertEqual(recovered['last_seen_ms'], 200)
        self.assertFalse(recovered['identity_verified'])

    def test_two_people_keep_ids_through_reordered_detections(self):
        initial = self.tracker.update([self.box(.1), self.box(.65)], 0)
        second = self.tracker.update([self.box(.66), self.box(.11)], 40)
        third = self.tracker.update([self.box(.12), self.box(.67)], 80)
        self.assertEqual([row['track_id'] for row in initial], [row['track_id'] for row in third])
        self.assertEqual([row['track_id'] for row in second], [row['track_id'] for row in third])
        self.assertTrue(all(row['association_quality'] == 'matched' for row in third))

    def test_expiry_reset_and_stream_isolation_never_recycle_ids(self):
        original = self.tracker.update([self.box(.1)], 0, 'a')[0]
        other = self.tracker.update([self.box(.1)], 0, 'b')[0]
        self.assertNotEqual(original['track_id'], other['track_id'])
        self.assertEqual(self.tracker.update([], 1100, 'a'), [])
        new = self.tracker.update([self.box(.1)], 1101, 'a')[0]
        self.assertNotEqual(new['track_id'], original['track_id'])
        self.tracker.reset('a')
        after_reset = self.tracker.update([self.box(.1)], 1200, 'a')[0]
        self.assertNotEqual(after_reset['track_id'], new['track_id'])
        self.assertEqual(self.tracker.update([], 40, 'b')[0]['track_id'], other['track_id'])

    def test_stale_and_invalid_input_do_not_mutate_state(self):
        first = self.tracker.update([self.box(.1)], 100)
        for boxes, timestamp in (([self.box(.2)], 99), ([self.box(float('nan'))], 101)):
            with self.assertRaises(ValueError):
                self.tracker.update(boxes, timestamp)
        current = self.tracker.update([], 101)
        self.assertEqual(current[0]['track_id'], first[0]['track_id'])
        self.assertEqual(current[0]['last_seen_ms'], 100)

    def test_hungarian_assignment_is_globally_optimal(self):
        from gamst.providers.ocsort import _linear_assignment
        cost = np.array([[4., 1., 3.], [2., 0., 5.], [3., 2., 2.]])
        self.assertEqual(_linear_assignment(cost), [(0, 1), (1, 0), (2, 2)])
        self.assertEqual(_linear_assignment(cost.T), [(0, 1), (1, 0), (2, 2)])

    def test_observation_centric_reupdate_restores_frozen_filter(self):
        from gamst.providers.ocsort import _KalmanBox, _bbox_to_z
        box = np.array([.1, .2, .2, .4])
        later = np.array([.16, .2, .26, .4])
        kalman = _KalmanBox(box)
        kalman.predict()
        kalman.miss()
        frozen = kalman.frozen[1].copy()
        kalman.predict()
        self.assertFalse(np.array_equal(kalman.P, frozen))
        kalman.observe(later, box, gap=2)
        self.assertIsNone(kalman.frozen)
        self.assertTrue(np.isfinite(kalman.x).all())
        self.assertTrue(np.isfinite(kalman.P).all())
        self.assertGreater(kalman.x[0], _bbox_to_z(box)[0])


if __name__ == '__main__':
    unittest.main()
