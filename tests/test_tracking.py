import unittest

from humansense.tracking import PersonTracker, normalize_bbox, intersection_over_union


class BoxTests(unittest.TestCase):
    def test_formats_and_overlap(self):
        self.assertEqual(normalize_bbox({"x1": .1, "y1": .2, "x2": .3, "y2": .6})["x"], .1)
        self.assertEqual(normalize_bbox((.1, .2, .3, .4)), {"x": .1, "y": .2, "width": .3, "height": .4})
        self.assertEqual(intersection_over_union((0, 0, .5, .5), (0, 0, .5, .5)), 1)
        self.assertEqual(intersection_over_union((0, 0, .2, .2), (.5, .5, .2, .2)), 0)

    def test_invalid_boxes_rejected(self):
        for box in [(0, 0, 0, .2), (-.1, 0, .2, .2), (0, 0, 2, 1), (float("nan"), 0, .2, .2), (False, 0, .2, .2), "box",
                    {"x1": 0, "y1": 0, "x2": None, "y2": 1}, {"x1": 0, "y1": 0, "x2": True, "y2": 1}]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                normalize_bbox(box)


class TrackerTests(unittest.TestCase):
    def test_motion_keeps_track_and_returns_independent_snapshots(self):
        tracker = PersonTracker()
        first = tracker.update([(.1, .1, .3, .5)], 1000)
        second = tracker.update([(.12, .1, .3, .5)], 1100)
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertEqual(second[0]["first_seen_ms"], 1000)
        self.assertEqual(second[0]["last_seen_ms"], 1100)
        self.assertEqual(first[0]["bbox"]["x"], .1)
        second[0]["bbox"]["x"] = .9
        self.assertEqual(tracker.update([], 1150)[0]["bbox"]["x"], .12)

    def test_multiple_people_survive_reordered_detections(self):
        tracker = PersonTracker()
        first = tracker.update([(.1, .1, .2, .5), (.65, .1, .2, .5)], 1000)
        second = tracker.update([(.66, .1, .2, .5), (.11, .1, .2, .5)], 1100)
        self.assertEqual([t["track_id"] for t in first], [t["track_id"] for t in second])
        self.assertEqual([t["bbox"]["x"] for t in second], [.11, .66])
        self.assertTrue(all(t["association_quality"] == "matched" for t in second))

    def test_missing_expiry_and_reappearance_never_reuse_identity(self):
        tracker = PersonTracker(max_age_ms=100)
        first = tracker.update([(.1, .1, .2, .5)], 1000)
        missing = tracker.update([], 1050)
        self.assertFalse(missing[0]["visible"])
        self.assertEqual(missing[0]["last_seen_ms"], 1000)
        self.assertEqual(tracker.update([], 1101), [])
        reappeared = tracker.update([(.1, .1, .2, .5)], 1200)
        self.assertNotEqual(first[0]["track_id"], reappeared[0]["track_id"])

    def test_stale_or_invalid_frame_does_not_mutate_state(self):
        tracker = PersonTracker()
        first = tracker.update([(.1, .1, .2, .5)], 1000)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            tracker.update([(.5, .1, .2, .5)], 900)
        with self.assertRaises(ValueError):
            tracker.update([(.1, .1, .2, .5), (0, 0, 2, 1)], 1100)
        current = tracker.update([(.1, .1, .2, .5)], 1000)
        self.assertEqual(current[0]["track_id"], first[0]["track_id"])
        self.assertEqual(len(current), 1)

    def test_streams_are_isolated_and_never_claim_identity_or_speaker(self):
        tracker = PersonTracker()
        a = tracker.update([(.1, .1, .2, .5)], 1000, "cam-a")
        b = tracker.update([(.1, .1, .2, .5)], 100, "cam-b")
        self.assertNotEqual(a[0]["track_id"], b[0]["track_id"])
        for track in a + b:
            self.assertFalse(track["identity_verified"])
            self.assertEqual(track["speaker_association"], "unavailable")

    def test_crossing_boxes_report_uncertain_association(self):
        tracker = PersonTracker(iou_threshold=.2)
        tracker.update([(.2, .1, .3, .5), (.3, .1, .3, .5)], 1000)
        crossing = tracker.update([(.25, .1, .3, .5), (.25, .1, .3, .5)], 1100)
        self.assertEqual(len(crossing), 2)
        self.assertTrue(all(t["visible"] and t["association_quality"] == "uncertain" for t in crossing))
        self.assertTrue(all(t["association_reason"] == "ambiguous_geometry" for t in crossing))
        self.assertTrue(all(t["association_history_uncertain"] for t in crossing))
        after = tracker.update([(.2, .1, .3, .5), (.3, .1, .3, .5)], 1200)
        self.assertTrue(all(t["association_history_uncertain"] for t in after))

    def test_one_detection_cannot_overwrite_two_tracks(self):
        tracker = PersonTracker(iou_threshold=.2)
        tracker.update([(.2, .1, .3, .5), (.3, .1, .3, .5)], 1000)
        result = tracker.update([(.25, .1, .3, .5)], 1100)
        self.assertEqual(sum(t["visible"] for t in result), 1)
        self.assertTrue(all(t["association_quality"] == "uncertain" for t in result))

    def test_reset_only_specified_stream(self):
        tracker = PersonTracker()
        a = tracker.update([(.1, .1, .2, .5)], 1000, "a")
        b = tracker.update([(.1, .1, .2, .5)], 1000, "b")
        tracker.reset("a")
        self.assertEqual(tracker.update([], 1100, "a"), [])
        self.assertEqual(tracker.update([], 1100, "b")[0]["track_id"], b[0]["track_id"])
        new_a = tracker.update([(.1, .1, .2, .5)], 1200, "a")
        self.assertNotEqual(a[0]["track_id"], new_a[0]["track_id"])


if __name__ == "__main__":
    unittest.main()
