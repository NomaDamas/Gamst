from dataclasses import replace
import unittest

from gamst.crew import (
    ResourceAllocator,
    ResourceCapacity,
    ResourceRequest,
    ResourceUnavailable,
    RuntimeDescriptor,
    RuntimeManager,
)


class Resources(unittest.TestCase):
    def setUp(self):
        self.allocator = ResourceAllocator(
            ResourceCapacity(cpu_cores=4, ram_bytes=1024, gpu_count=1, vram_bytes=512)
        )

    def test_atomic_overcommit_release_and_telemetry(self):
        request = ResourceRequest(cpu_cores=2, ram_bytes=600, gpu_count=1, vram_bytes=256)
        reservation = self.allocator.reserve(request, owner="vision")
        value = self.allocator.telemetry()
        self.assertEqual(value["reserved"], {
            "cpu_cores": 2.0, "ram_bytes": 600, "gpu_count": 1, "vram_bytes": 256,
        })
        self.assertEqual(value["active_reservations"], 1)
        with self.assertRaises(ResourceUnavailable):
            self.allocator.reserve(ResourceRequest(cpu_cores=2.1, ram_bytes=1), owner="asr")
        after = self.allocator.telemetry()
        self.assertEqual(after["reserved"], value["reserved"])
        self.assertEqual(after["counters"]["overcommit_rejections"], 1)
        self.allocator.release(reservation)
        self.assertEqual(self.allocator.telemetry()["active_reservations"], 0)
        self.assertEqual(self.allocator.telemetry()["counters"]["releases"], 1)
        with self.assertRaises(ValueError):
            self.allocator.release(reservation)

    def test_exact_token_and_bounded_inputs(self):
        with self.assertRaises(ValueError):
            ResourceRequest(cpu_cores=True)
        with self.assertRaises(ValueError):
            ResourceRequest(ram_bytes=-1)
        reservation = self.allocator.reserve(ResourceRequest(cpu_cores=1), owner="worker")
        clone = replace(reservation)
        with self.assertRaises(ValueError):
            self.allocator.release(clone)
        self.allocator.release(reservation)

    def test_reservation_count_is_bounded(self):
        allocator = ResourceAllocator(ResourceCapacity(cpu_cores=4), max_reservations=1)
        first = allocator.reserve(ResourceRequest(cpu_cores=1), owner="one")
        with self.assertRaises(ResourceUnavailable):
            allocator.reserve(ResourceRequest(cpu_cores=1), owner="two")
        self.assertEqual(allocator.telemetry()["counters"]["capacity_rejections"], 1)
        allocator.release(first)


class Adapter:
    def __init__(self, fail=False):
        self.fail = fail
        self.started = self.stopped = False

    async def start(self):
        self.started = True
        if self.fail:
            raise RuntimeError("startup failed")

    async def stop(self):
        self.stopped = True

    async def submit(self, packet):
        pass

    def health(self):
        return {"running": self.started and not self.stopped}


class ManagerResources(unittest.IsolatedAsyncioTestCase):
    async def test_register_and_close_release_exact_resources(self):
        allocator = ResourceAllocator(ResourceCapacity(cpu_cores=2, ram_bytes=100))
        manager = RuntimeManager(resource_allocator=allocator)
        adapter = Adapter()
        descriptor = RuntimeDescriptor("worker", "cpu", "fixture", "1", "MIT", True)
        await manager.register(
            "worker", adapter, descriptor,
            resources=ResourceRequest(cpu_cores=1, ram_bytes=50),
        )
        self.assertTrue(manager.resource_telemetry()["enabled"])
        self.assertEqual(manager.resource_telemetry()["active_reservations"], 1)
        await manager.close("worker")
        self.assertEqual(manager.resource_telemetry()["active_reservations"], 0)
        self.assertEqual(manager.resource_telemetry()["counters"]["releases"], 1)

    async def test_overcommit_rejects_before_start_and_start_failure_releases(self):
        allocator = ResourceAllocator(ResourceCapacity(cpu_cores=1))
        manager = RuntimeManager(resource_allocator=allocator)
        descriptor = RuntimeDescriptor("worker", "cpu", "fixture", "1", "MIT", True)
        blocked = Adapter()
        with self.assertRaises(ResourceUnavailable):
            await manager.register(
                "blocked", blocked, descriptor,
                resources=ResourceRequest(cpu_cores=2),
            )
        self.assertFalse(blocked.started)
        failing = Adapter(fail=True)
        with self.assertRaises(RuntimeError):
            await manager.register(
                "failing", failing, descriptor,
                resources=ResourceRequest(cpu_cores=1),
            )
        self.assertTrue(failing.stopped)
        self.assertEqual(manager.resource_telemetry()["active_reservations"], 0)


if __name__ == "__main__":
    unittest.main()
