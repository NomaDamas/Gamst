import unittest
from dataclasses import replace
from gamst.temporal import (TemporalAligner, TemporalEvent, TemporalPacket,
                             PacketIntegrityCounter, canonical_packet_bytes,
                             packet_sha256, verify_packet_checksum)


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
        self.assertEqual(relation['schema_version'], 'gamst.relation.v1')
        self.assertEqual([item['clock_epoch'] for item in relation['provenance']], ['epoch-1', 'epoch-1'])
        self.assertEqual(relation['provenance'][0]['capture_timestamp_ms'], 100)
        self.assertEqual(relation['evidence_ids'], ['a', 'b'])

    def test_relation_preserves_calibration_and_packet_cutoff(self):
        aligner = TemporalAligner(min_samples=2)
        for source in ('clock', 'audio-clock'):
            aligner.add_clock_sample(source, 0, 10)
            aligner.add_clock_sample(source, 100, 110)
        left = event('vision', 100)
        right = replace(event('audio', 120, modality='audio'), source_clock='audio-clock',
                        clock_id='audio-clock-id', evidence_ids=('audio-evidence',))
        relation = aligner.materialize_relation(left, right, 'cross_modal', evidence_ids=('join',))
        self.assertTrue(relation['clock_calibration']['clock']['calibrated'])
        self.assertTrue(relation['clock_calibration']['audio-clock']['calibrated'])
        packet = TemporalPacket('gamst.temporal.v1', 200, (left, right), (relation,))
        self.assertEqual(packet.as_dict()['relations'][0]['event_ids'], ['vision', 'audio'])
        with self.assertRaises(ValueError):
            TemporalPacket('gamst.temporal.v1', 110, (left, right), (relation,))

    def test_relation_rejects_unbounded_or_malformed_shape(self):
        aligner = TemporalAligner()
        left, right = event('a', 100), event('b', 100)
        with self.assertRaises(ValueError):
            aligner.materialize_relation(left, right, 'x' * 65)
        relation = aligner.relation_candidate(left, right, 'related')
        relation['provenance'][1]['clock_epoch'] = ''
        with self.assertRaises(ValueError):
            TemporalPacket('gamst.temporal.v1', 200, (left, right), (relation,))

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

    def test_checksum_is_versioned_and_outside_evidence_content(self):
        packet = TemporalPacket('gamst.temporal.v1', 1000, (event('e', 900),))
        rendered = packet.as_dict()
        self.assertEqual(rendered['checksum']['algorithm'], 'sha256')
        self.assertEqual(rendered['checksum']['version'], 1)
        self.assertEqual(rendered['checksum']['sha256'], packet_sha256(rendered))
        self.assertNotIn('checksum', rendered['events'][0])
        self.assertEqual(canonical_packet_bytes(rendered), packet.canonical_bytes())
        self.assertEqual(verify_packet_checksum(rendered), True)

    def test_tampered_packet_is_rejected_and_count_is_bounded(self):
        packet = TemporalPacket('gamst.temporal.v1', 1000, (event('e', 900),))
        rendered = packet.as_dict()
        counter = PacketIntegrityCounter(limit=1)
        rendered['events'][0]['value']['x'] = 2
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_packet_checksum(rendered, counter=counter)
        with self.assertRaises(ValueError):
            verify_packet_checksum(rendered, counter=counter)
        self.assertEqual(counter.integrity_rejected, 1)

    def test_checksum_version_and_legacy_behavior_are_explicit(self):
        packet = TemporalPacket('gamst.temporal.v1', 1000, (event('e', 900),))
        versioned = packet.as_dict()
        versioned['checksum']['version'] = 2
        counter = PacketIntegrityCounter()
        with self.assertRaisesRegex(ValueError, 'unsupported temporal packet checksum'):
            TemporalPacket.from_dict(versioned, counter=counter)
        self.assertEqual(counter.integrity_rejected, 1)
        legacy = packet.as_dict(include_checksum=False)
        with self.assertRaisesRegex(ValueError, 'checksum is required'):
            TemporalPacket.from_dict(legacy, counter=counter)
        self.assertEqual(counter.integrity_rejected, 2)
        self.assertFalse(verify_packet_checksum(legacy, allow_legacy=True))
        restored = TemporalPacket.from_dict(legacy, allow_legacy=True)
        self.assertEqual(restored.as_dict()['checksum']['algorithm'], 'sha256')

    def test_canonical_bytes_roundtrip_rejects_noncanonical_wire(self):
        packet = TemporalPacket('gamst.temporal.v1', 1000, (event('e', 900),))
        raw = packet.to_bytes()
        self.assertEqual(TemporalPacket.from_bytes(raw), packet)
        # Parsing equivalent JSON with a different key order is deliberately
        # rejected so every repository hashes exactly the same bytes.
        noncanonical = b'{"schema_version":"gamst.temporal.v1","query_cutoff_ms":1000,"clock":{},"events":[],"relations":[],"checksum":{"algorithm":"sha256","version":1,"sha256":"' + packet.checksum()['sha256'].encode() + b'"}}'
        with self.assertRaises(ValueError):
            TemporalPacket.from_bytes(noncanonical)


if __name__ == '__main__':
    unittest.main()
