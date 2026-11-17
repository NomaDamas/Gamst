import unittest

from gamst.context_memory import ContextMemory
from gamst.temporal import TemporalEvent
from gamst.temporal_context import TemporalContextEngine


def event(event_id, modality, capture, *, revision=0, subject="x", clock="mono"):
    return TemporalEvent(event_id, "stream", subject, modality, "state", clock,
                         "clock-id", "epoch-1", capture, capture, capture + 100,
                         capture, "provider", "model", {"value": event_id},
                         evidence_ids=("ev-" + event_id,), revision=revision,
                         supersedes=event_id if revision else None)


class TemporalContextTests(unittest.TestCase):
    def test_calibrated_cross_modal_relation_and_cutoff(self):
        engine = TemporalContextEngine(min_clock_samples=2)
        engine.add_clock_sample("mono", 0, 10)
        engine.add_clock_sample("mono", 1000, 1010)
        left, right = event("a", "vision", 100), event("b", "audio", 150)
        engine.ingest(left)
        engine.ingest(right)
        relation = engine.relation(left, right, "temporal_overlap")
        self.assertEqual(relation["status"], "candidate")
        self.assertGreater(relation["overlap"], 0)
        packet = engine.packet(100)
        self.assertEqual([item.event_id for item in packet.events], ["a"])

    def test_newer_revision_wins_and_state_marks_stale(self):
        engine = TemporalContextEngine()
        engine.ingest(event("a", "vision", 100))
        self.assertIsNone(engine.ingest(event("a", "vision", 100, revision=0)))
        engine.ingest(event("a", "vision", 100, revision=1))
        state = engine.current_state(query_cutoff_ms=100, now_ms=10000)
        self.assertEqual(state[0]["state"], "last_seen")


class ContextMemoryTests(unittest.TestCase):
    def test_layers_delta_replay_and_cutoff_retrieval(self):
        memory = ContextMemory(session_id="s", max_recent=8)
        first = memory.append({"event_id": "a", "timestamp_ms": 10, "value": 1},
                              entity_id="thing", episodic=True)
        before = memory.snapshot()
        memory.append({"event_id": "b", "timestamp_ms": 20, "value": 2}, episodic=True)
        delta = memory.delta(before)
        replay = ContextMemory(session_id="s", max_recent=8)
        replay.replay({"session_id": "s", "events": [first], "current_state": {},
                       "from_revision": 0, "to_revision": 1})
        replay.replay(delta)
        self.assertEqual([x["event_id"] for x in replay.recent_timeline], ["a", "b"])
        self.assertEqual([x["event_id"] for x in memory.retrieve(query_cutoff_ms=10)], ["a"])

    def test_checkpoint_lineage_and_resume(self):
        memory = ContextMemory(session_id="s")
        memory.append({"event_id": "a", "timestamp_ms": 1}, episodic=True)
        resumed = ContextMemory.resume(memory.checkpoint())
        self.assertEqual(resumed.snapshot(), memory.snapshot())
        checkpoint = memory.checkpoint()
        checkpoint["lineage"] = "bad"
        with self.assertRaises(ValueError):
            ContextMemory.resume(checkpoint)


if __name__ == "__main__":
    unittest.main()
