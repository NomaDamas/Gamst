import unittest
import asyncio
from gamst.crew import EventPacket, RuntimeDescriptor, RuntimeManager
# Crew's own TemporalAligner was a copy of GAMST's; the behaviour is tested on GAMST's aligner.
from gamst.temporal import TemporalAligner, TemporalEvent


class Adapter:
    def __init__(self):
        self.items = []
        self.stopped = False

    async def start(self):
        pass

    async def stop(self):
        self.stopped = True

    async def submit(self, p):
        self.items.append(p)

    def health(self):
        return {}


class FlakyAdapter(Adapter):
    def __init__(self):
        super().__init__()
        self.starts = 0

    async def start(self):
        self.starts += 1

    async def submit(self, p):
        if not self.items:
            raise RuntimeError("transient provider failure")
        self.items.append(p)


def packet(i, cut=None):
    return EventPacket(
        str(i),
        "s",
        "u",
        "audio",
        "event",
        "mono",
        i,
        i,
        None,
        None,
        i,
        i,
        i,
        False,
        0,
        "fixture",
        "1",
        {"i": i},
        query_cutoff_ms=cut,
    )


def event(i):
    return TemporalEvent(str(i), "s", "u", "audio", "event", "mono", "legacy", "legacy", i, i, i, i,
                         "fixture", "1", {"i": i})


class Foundation(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_registration_is_serialized(self):
        manager = RuntimeManager()
        adapters = [Adapter(), Adapter()]
        descriptors = [RuntimeDescriptor("same", "mac", "fixture", "1", "MIT", True)] * 2
        results = await asyncio.gather(
            manager.register("same", adapters[0], descriptors[0]),
            manager.register("same", adapters[1], descriptors[1]),
            return_exceptions=True,
        )
        self.assertEqual(sum(isinstance(result, ValueError) for result in results), 1)
        await manager.close("same")

    async def test_manager_bounded_health(self):
        m = RuntimeManager(capacity=2)
        a = Adapter()
        await m.register(
            "x", a, RuntimeDescriptor("x", "mac", "fixture", "1", "MIT", True)
        )
        for i in range(5):
            await m.submit("x", packet(i))
        self.assertEqual(m.health("x").dropped, 3)
        await m.close("x")
        self.assertTrue(a.stopped)

    async def test_failed_start_is_stopped(self):
        adapter = Adapter()
        async def fail():
            raise RuntimeError('readiness failed')
        adapter.start = fail
        manager = RuntimeManager()
        with self.assertRaises(RuntimeError):
            await manager.register('x', adapter, RuntimeDescriptor('x', 'mac', 'fixture', '1', 'MIT', True))
        self.assertTrue(adapter.stopped)
        self.assertNotIn('x', manager.runtimes)

    async def test_close_all_stops_every_runtime_and_releases_registry(self):
        manager = RuntimeManager()
        adapters = [Adapter(), Adapter()]
        descriptor = RuntimeDescriptor("runtime", "mac", "fixture", "1", "MIT", True)
        await manager.register("one", adapters[0], descriptor)
        await manager.register("two", adapters[1], descriptor)
        await manager.close_all()
        self.assertEqual(manager.runtimes, {})
        self.assertTrue(adapters[0].stopped)
        self.assertTrue(adapters[1].stopped)

    async def test_manager_async_context_closes_registered_runtimes(self):
        adapter = Adapter()
        async with RuntimeManager() as manager:
            await manager.register(
                "ctx", adapter,
                RuntimeDescriptor("ctx", "mac", "fixture", "1", "MIT", True),
            )
            self.assertIn("ctx", manager.runtimes)
        self.assertTrue(adapter.stopped)

    async def test_runtime_error_is_visible_and_restart_is_bounded(self):
        manager = RuntimeManager(restart_limit=1)
        adapter = FlakyAdapter()
        await manager.register('x', adapter, RuntimeDescriptor('x', 'mac', 'fixture', '1', 'MIT', True))
        try:
            await manager.submit('x', packet(1))
            await asyncio.sleep(0.03)
            health = manager.health('x')
            self.assertEqual(health.crash_count, 1)
            self.assertEqual(health.restart_count, 1)
            self.assertEqual(adapter.starts, 2)
        finally:
            await manager.close('x')

    async def test_temporal_clock_cutoff_revision(self):
        t = TemporalAligner(min_samples=3)
        for i in range(3):
            t.add_clock_sample("mono", i * 100, i * 100 + 5)
        self.assertTrue(t.clock_status("mono")["calibrated"])
        self.assertIsNotNone(t.accept(event(1)))
        self.assertIsNone(t.accept(event(1)))
        self.assertEqual(len(t.eligible(1)), 1)
        self.assertEqual(len(t.eligible(0)), 0)

    async def test_temporal_normalization_is_explicit_before_and_after_calibration(self):
        t = TemporalAligner(min_samples=3)
        self.assertIsNone(t.normalize_timestamp("camera", 100)["normalized_timestamp_ms"])
        for i in range(3):
            t.add_clock_sample("camera", i * 100, i * 100 + 5)
        normalized = t.normalize_timestamp("camera", 300)
        self.assertTrue(normalized["calibrated"])
        self.assertEqual(normalized["normalized_timestamp_ms"], 305)

    async def test_sequence_gap_preserves_provenance(self):
        from dataclasses import replace
        t = TemporalAligner()
        first = event(10)
        second = replace(event(12), event_id="12", value={"state": "gone"}, evidence_ids=("frame-12",))
        self.assertEqual(t.sequence_gap(first)["state"], "initial")
        self.assertIsNotNone(t.accept(first))
        gap = t.sequence_gap(second)
        self.assertEqual(gap["state"], "gap")
        self.assertTrue(gap["detected_gap"])
        self.assertEqual(gap["expected_sequence"], 11)
        t.accept(second)
        self.assertEqual(t.sequence_gap(second)["state"], "gap")  # stable once accepted
        self.assertEqual(t.sequence_gap(replace(event(11), event_id="11"))["state"], "reordered")
        candidate = t.relation_candidate(first, second, "follows")
        self.assertEqual(candidate["event_ids"], ["10", "12"])
        self.assertEqual(candidate["evidence_ids"], ["frame-12"])

    async def test_revision_lineage_and_clock_epoch(self):
        from dataclasses import replace
        aligner = TemporalAligner()
        self.assertIsNotNone(aligner.accept(event(1)))
        with self.assertRaises(ValueError):
            aligner.accept(replace(event(1), revision=1))
        self.assertIsNotNone(aligner.accept(replace(event(1), revision=1, supersedes='1')))
        with self.assertRaises(ValueError):
            aligner.accept(replace(event(1), revision=2, supersedes='1', clock_epoch='restarted'))

    def test_clock_identity_survives_packet_serialization(self):
        from dataclasses import replace
        import json
        value = replace(packet(1), clock_id='camera-clock', clock_epoch='boot-7')
        decoded = json.loads(value.json())
        self.assertEqual(decoded['clock_id'], 'camera-clock')
        self.assertEqual(decoded['clock_epoch'], 'boot-7')
        with self.assertRaises(ValueError):
            replace(value, clock_epoch='')

    def test_descriptor_requires_revision_and_license(self):
        with self.assertRaises(ValueError):
            RuntimeDescriptor("x", "mac", "fixture", "", "MIT", True)


if __name__ == "__main__":
    unittest.main()
