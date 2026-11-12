import math
import unittest

from benchmarks.runtime import percentile, run_benchmark


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_on_unsorted_data_and_boundaries(self):
        self.assertEqual(percentile([5, 1, 3, 4, 2], .5), 3)
        self.assertEqual(percentile(list(range(1, 101)), .95), 95)
        self.assertEqual(percentile([5, 1], 0), 1)
        self.assertEqual(percentile([5, 1], 1), 5)

    def test_empty_and_invalid_percentiles(self):
        for values, fraction in (([], .5), ([1], -1), ([1], 1.1), ([1], math.nan)):
            with self.assertRaises(ValueError):
                percentile(values, fraction)


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_real_run_reports_timing_resources_and_bounded_drop(self):
        result = await run_benchmark(samples=5, warmup=1, queue_size=2)
        self.assertTrue(result["configuration"]["synthetic"])
        self.assertFalse(result["configuration"]["capture_devices_opened"])
        self.assertEqual(result["configuration"]["samples_completed"], 5)
        self.assertTrue(result["environment"]["python"])
        for metric in result["latency"].values():
            self.assertEqual(metric["count"], 5)
            self.assertGreater(metric["p50_ms"], 0)
            self.assertGreaterEqual(metric["p95_ms"], metric["p50_ms"])
        self.assertGreater(result["resources"]["wall_seconds"], 0)
        self.assertGreater(result["resources"]["sqlite_allocated_bytes"], 0)
        self.assertEqual(result["backpressure"]["max_observed_queue_depth"], 2)
        self.assertGreater(result["backpressure"]["dropped_deliveries"], 0)

    async def test_duration_mode_records_actual_work(self):
        result = await run_benchmark(samples=1, warmup=0, duration_seconds=.01)
        self.assertGreater(result["configuration"]["samples_completed"], 0)
        self.assertGreaterEqual(result["resources"]["wall_seconds"], .01)


if __name__ == "__main__":
    unittest.main()
