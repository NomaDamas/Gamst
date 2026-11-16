import unittest
from dataclasses import replace

from gamst.contradictions import (contradiction_candidate,
                                  detect_contradictions)
from gamst.temporal import TemporalAligner, TemporalEvent


def event(event_id, ts=1000, *, modality="video", value=None,
          event_type="scene.state", revision=0, supersedes=None,
          uncertainty=None, freshness_ms=5000, gap=False,
          source_clock="clock", clock_epoch="epoch-1"):
    return TemporalEvent(event_id, "stream", "subject", modality, event_type,
                         source_clock, source_clock + "-id", clock_epoch,
                         ts, ts, ts + 100, ts // 10, "fixture", "1",
                         {"state": value} if value is not None else {"state": "same"},
                         revision=revision, supersedes=supersedes,
                         uncertainty=uncertainty, freshness_ms=freshness_ms,
                         gap=gap, evidence_ids=("evidence-" + event_id,))


class ContradictionTests(unittest.TestCase):
    def test_candidate_contains_both_evidence_intervals_clocks_and_uncertainty(self):
        aligner = TemporalAligner()
        left = event("video-1", value="open", modality="video")
        right = event("audio-1", value="closed", modality="audio")
        result = contradiction_candidate(left, right, aligner=aligner, now_ms=1000)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["relation"], "contradiction")
        self.assertEqual(result["event_ids"], ["video-1", "audio-1"])
        self.assertEqual(result["evidence_ids"], ["evidence-video-1", "evidence-audio-1"])
        self.assertEqual(result["intervals"][0]["start_ms"], 1000)
        self.assertEqual(result["provenance"][1]["source_clock"], "clock")
        self.assertEqual(result["uncertainty"]["status"], "unverified")
        self.assertFalse(result["deception_assessed"])

    def test_stale_and_misaligned_pairs_abstain(self):
        stale = contradiction_candidate(
            event("old-video", ts=900, value=1, freshness_ms=10),
            event("old-audio", ts=900, value=2, freshness_ms=10, modality="audio"),
            now_ms=1000)
        self.assertEqual(stale["status"], "abstained")
        self.assertEqual(stale["reason"], "stale_evidence")
        misaligned = contradiction_candidate(
            event("far-video", ts=1000, value=1),
            event("far-audio", ts=2000, value=2, modality="audio"),
            now_ms=2000, max_gap_ms=0)
        self.assertEqual(misaligned["status"], "abstained")
        self.assertEqual(misaligned["reason"], "temporal_misalignment")

    def test_cutoff_excludes_late_event_and_same_values_are_not_conflicts(self):
        left = event("v", value=1)
        late = event("a-late", ts=1101, value=2, modality="audio")
        same = event("a-same", value=1, modality="audio")
        result = detect_contradictions([left, late, same], query_cutoff_ms=1000, now_ms=1000)
        self.assertFalse(result["contradiction_detected"])
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["abstention_count"], 0)

    def test_revision_lineage_keeps_newer_state_and_abstains_unresolved_duplicate(self):
        old = event("v", value="open", revision=0)
        newer = replace(old, value={"state": "closed"}, revision=1, supersedes="v")
        audio = event("a", value="closed", modality="audio")
        result = detect_contradictions([old, newer, audio], query_cutoff_ms=1000, now_ms=1000)
        self.assertFalse(result["contradiction_detected"])
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["abstention_count"], 0)

    def test_temporal_context_emits_candidate_and_excludes_post_cutoff_signal(self):
        import asyncio
        from gamst import Gamst

        async def exercise():
            async with Gamst(consent=True) as human:
                from gamst.schema import timestamp
                cutoff = timestamp()
                await human.push_signal("scene.state", "open", "video", timestamp_ms=cutoff)
                await human.push_signal("scene.state", "closed", "audio", timestamp_ms=cutoff)
                await human.push_signal("scene.state", "future", "audio", timestamp_ms=cutoff + 1)
                packet = await human.build_temporal_context(query_cutoff_ms=cutoff)
                return packet

        packet = asyncio.run(exercise())
        self.assertEqual(len(packet["events"]), 2)
        candidates = [item for item in packet["relations"]
                      if item.get("relation") == "contradiction"
                      and item.get("status") == "candidate"]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(len(candidate["evidence_ids"]), 2)
        self.assertEqual(len(candidate["provenance"]), 2)
        self.assertEqual(candidate["uncertainty"]["status"], "unverified")

    def test_calibration_requirement_abstains_without_upgrading_uncertainty(self):
        left = event("v", value=1, source_clock="video-clock")
        right = event("a", value=2, modality="audio", source_clock="audio-clock")
        result = detect_contradictions([left, right], now_ms=1000,
                                       require_calibrated=True)
        self.assertEqual(result["status"], "abstained")
        self.assertFalse(result["contradiction_detected"])
        self.assertEqual(result["abstentions"][0]["reason"], "clock_uncalibrated")


if __name__ == "__main__":
    unittest.main()
