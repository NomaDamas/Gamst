import unittest
from dataclasses import replace
from gamst.temporal import TemporalAligner, TemporalEvent, TemporalPacket


def event(event_id, ts, *, modality='vision', revision=0, value=None):
    return TemporalEvent(event_id, 'stream', 'subject', modality, 'observation', 'clock', 'clock-1', 'epoch-1', ts, ts, ts + 100, ts // 100, 'fixture', '1', value or {'x': 1}, revision=revision, evidence_ids=(event_id,))


class TemporalTests(unittest.TestCase):
    def test_event_bounds_and_packet_json(self):
        packet = TemporalPacket('gamst.temporal.v1', 1000, (event('e', 900),))
        self.assertEqual(packet.as_dict()['events'][0]['event_id'], 'e')
        with self.assertRaises(ValueError):
            TemporalPacket('gamst.temporal.v1', 1000, (event('late', 1001),))

    def test_post_query_events_are_excluded(self):
        aligner = TemporalAligner()
        aligner.accept(event('old', 90)); aligner.accept(event('new', 110))
        self.assertEqual([e.event_id for e in aligner.eligible(100)], ['old'])

    def test_drift_is_uncalibrated_until_minimum_samples(self):
        aligner = TemporalAligner(min_samples=3)
        aligner.add_clock_sample('clock', 0, 10)
        aligner.add_clock_sample('clock', 100, 110)
        self.assertFalse(aligner.clock_status('clock')['calibrated'])
        aligner.add_clock_sample('clock', 200, 210)
        self.assertTrue(aligner.clock_status('clock')['calibrated'])
        self.assertAlmostEqual(aligner.clock_status('clock')['offset_ms'], 10)

    def test_overlap_and_relation_remain_candidates(self):
        aligner = TemporalAligner()
        left, right = event('a', 100), event('b', 150, modality='audio')
        self.assertGreater(aligner.overlap(left, right), 0)
        relation = aligner.relation_candidate(left, right, 'temporally_related')
        self.assertEqual(relation['status'], 'candidate')
        self.assertEqual(relation['uncertainty'], 'unverified')

    def test_revision_replaces_only_newer_revision(self):
        aligner = TemporalAligner()
        self.assertIsNotNone(aligner.accept(event('e', 100, revision=0)))
        self.assertIsNone(aligner.accept(event('e', 100, revision=0)))
        revised = replace(event('e', 100, revision=1, value={'x': 2}), supersedes='e')
        self.assertIsNotNone(aligner.accept(revised))
        self.assertEqual(aligner.eligible(200)[0].value, {'x': 2})

    def test_revision_requires_lineage_and_stable_clock_epoch(self):
        aligner = TemporalAligner()
        aligner.accept(event('e', 100))
        with self.assertRaises(ValueError):
            aligner.accept(event('e', 100, revision=1))
        with self.assertRaises(ValueError):
            aligner.accept(replace(event('e', 100), clock_epoch='restarted'))

    def test_current_and_last_seen_are_explicit(self):
        packet = TemporalPacket('gamst.temporal.v1', 10000, (event('e', 100),))
        rendered = packet.as_dict()['events'][0]
        self.assertFalse(rendered['fresh'])
        self.assertEqual(rendered['evidence_state'], 'last_seen')
        self.assertNotEqual(rendered['evidence_state'], 'current')

    def test_packet_byte_budget(self):
        huge = event('huge', 100, value={'text': 'x' * 200000})
        with self.assertRaises(ValueError):
            TemporalPacket('gamst.temporal.v1', 200, (huge,)).as_dict()


if __name__ == '__main__':
    unittest.main()
