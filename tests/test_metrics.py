import json
import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from humansense.metrics import LatencyMonitor


class MetricsTests(unittest.TestCase):
    def test_bounded_window_percentiles_and_lifetime_aggregates_are_distinct(self):
        monitor = LatencyMonitor(capacity=3)
        for value in (100, 1, 2, 3):
            monitor.record("state", value)
        snapshot = monitor.snapshot()
        stage = snapshot["stages"]["state"]
        self.assertEqual(stage["count"], 4)
        self.assertEqual(stage["window_count"], 3)
        self.assertEqual(stage["p50_ms"], 2)
        self.assertEqual(stage["p95_ms"], 3)
        self.assertEqual(stage["p99_ms"], 3)
        self.assertEqual(stage["total_ms"], 106)
        self.assertEqual(stage["mean_ms"], 26.5)
        self.assertEqual(stage["max_ms"], 100)
        self.assertEqual(stage["percentile_scope"], "last_capacity_observations")
        json.dumps(snapshot, allow_nan=False)

    def test_measure_uses_monotonic_duration_and_records_failure(self):
        monitor = LatencyMonitor()
        with patch("humansense.metrics.time.perf_counter_ns", side_effect=[1_000_000, 3_500_000]):
            with monitor.measure("model"):
                pass
        self.assertEqual(monitor.snapshot()["stages"]["model"]["mean_ms"], 2.5)
        with patch("humansense.metrics.time.perf_counter_ns", side_effect=[0, 500_000]):
            with self.assertRaisesRegex(RuntimeError, "body failure"):
                with monitor.measure("model"):
                    raise RuntimeError("body failure")
        self.assertEqual(monitor.snapshot()["stages"]["model"]["count"], 2)

    def test_worker_threads_do_not_lose_observations(self):
        monitor = LatencyMonitor(capacity=7)
        def worker(_):
            for i in range(500):
                monitor.record("model", 1)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(worker, range(4)))
        stage = monitor.snapshot()["stages"]["model"]
        self.assertEqual(stage["count"], 2000)
        self.assertEqual(stage["total_ms"], 2000)
        self.assertEqual(stage["window_count"], 7)

    def test_reset_discards_in_flight_timers_without_resurrecting_stages(self):
        monitor = LatencyMonitor(max_stages=1)
        with monitor.measure("before-reset"):
            self.assertEqual(monitor.snapshot()["stages"], {})
            monitor.reset()
            monitor.record("after-reset", 2)
        self.assertEqual(set(monitor.snapshot()["stages"]), {"after-reset"})

    def test_snapshots_do_not_expose_mutable_internal_state_and_reset_is_scoped(self):
        monitor = LatencyMonitor()
        monitor.record("a", 1)
        monitor.record("b", 2)
        snapshot = monitor.snapshot()
        snapshot["stages"]["a"]["count"] = 99
        snapshot["stages"].clear()
        self.assertEqual(monitor.snapshot()["stages"]["a"]["count"], 1)
        monitor.reset("a")
        self.assertEqual(set(monitor.snapshot()["stages"]), {"b"})
        monitor.reset()
        self.assertEqual(monitor.snapshot()["stages"], {})

    def test_invalid_measurements_and_bounded_cardinality(self):
        monitor = LatencyMonitor(max_stages=1)
        for value in (-1, math.nan, math.inf, True, "1"):
            with self.assertRaises(ValueError):
                monitor.record("a", value)
        for name in ("", " ", None, "x"*129):
            with self.assertRaises(ValueError):
                monitor.record(name, 1)
        monitor.record("a", 0)
        with self.assertRaises(ValueError):
            monitor.record("b", 1)
        with self.assertRaises(ValueError):
            with monitor.measure("b"):
                self.fail("body must not run without measurement capacity")
        for capacity in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                LatencyMonitor(capacity=capacity)


if __name__ == "__main__":
    unittest.main()
