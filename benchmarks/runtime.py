"""Reproducible synthetic SDK workload; does not exercise perception models.

Run from the checkout: python -m benchmarks.runtime --samples 200
Latency percentiles use a bounded deterministic reservoir for long soak runs.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humansense import HumanSense, __version__


def percentile(values, fraction):
    """Nearest-rank percentile: the ceil(p*N) ordered observation (p in [0,1])."""
    if not values:
        raise ValueError("percentile requires observations")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction*len(ordered))-1)]


class _Timings:
    """Reservoir sampling retains a fixed-size representative timing sample."""
    def __init__(self, capacity=10_000):
        self.capacity = capacity
        self.samples = []
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.random = random.Random(0)

    def add(self, elapsed_ms):
        self.count += 1
        self.total += elapsed_ms
        self.maximum = max(self.maximum, elapsed_ms)
        if len(self.samples) < self.capacity:
            self.samples.append(elapsed_ms)
        else:
            index = self.random.randrange(self.count)
            if index < self.capacity:
                self.samples[index] = elapsed_ms

    def report(self):
        return {"count": self.count, "retained_timing_samples": len(self.samples),
                "percentile_method": "nearest_rank_on_deterministic_reservoir",
                "p50_ms": percentile(self.samples, .5), "p95_ms": percentile(self.samples, .95),
                "max_ms": self.maximum, "mean_ms": self.total/self.count}


def _peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        return None
    # macOS exposes bytes; Linux and BSD expose KiB.
    return int(value if sys.platform == "darwin" else value*1024)


def _environment():
    cpu_model = platform.processor()
    if not cpu_model and sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "cpu_model": cpu_model or "unavailable",
            "humansense_version": __version__}


async def run_benchmark(*, samples=200, warmup=20, duration_seconds=None, store=":memory:",
                        retention_hours=24, queue_size=8):
    """Run sequential text→signal→state cycles as fast as the SDK permits.

    Duration mode supersedes sample count and is unpaced. Latency includes
    ingestion, storage, inference and queue publication; no camera, ASR, VAD,
    network, transport or external reasoning backend is enabled.
    """
    for name, value in (("samples", samples), ("queue_size", queue_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if duration_seconds is not None and (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)) or not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be positive and finite")
    store_path = str(store).removeprefix("sqlite:///")
    if store_path != ":memory:" and Path(store_path).exists():
        raise ValueError("benchmark requires a new SQLite path so existing observations are not modified")
    metrics = {name: _Timings() for name in ("text_input", "signal_input", "state_query")}
    async with HumanSense(consent=True, store=store, retention_hours=retention_hours, queue_size=queue_size) as human:
        for i in range(warmup):
            await human.push_text(f"synthetic warmup item {i}")
            await human.push_signal("audio.rms", .1+(i % 10)/100, "audio",
                                    model_id="synthetic-benchmark", model_version="1")
            await human.get_state()
        stream = human.events()
        waiting_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        queue_depth_max = 0
        observed_first_event = False
        initial_rss = _peak_rss_bytes()
        started = time.perf_counter()
        cpu_started = time.process_time()
        completed = 0
        try:
            while (completed < samples if duration_seconds is None else completed == 0 or time.perf_counter()-started < duration_seconds):
                begin = time.perf_counter_ns()
                await human.push_text(f"Please clarify synthetic question number {completed}")
                metrics["text_input"].add((time.perf_counter_ns()-begin)/1_000_000)
                begin = time.perf_counter_ns()
                await human.push_signal("audio.rms", .1+(completed % 10)/100, "audio",
                                        model_id="synthetic-benchmark", model_version="1")
                metrics["signal_input"].add((time.perf_counter_ns()-begin)/1_000_000)
                begin = time.perf_counter_ns()
                await human.get_state()
                metrics["state_query"].add((time.perf_counter_ns()-begin)/1_000_000)
                completed += 1
                if not observed_first_event:
                    # Let the first subscription receive one event, then leave it
                    # stalled so subsequent drops exercise bounded backpressure.
                    await asyncio.sleep(0)
                    await waiting_event
                    observed_first_event = True
                queue_depth_max = max(queue_depth_max, *(queue.qsize() for queue, _ in human._subscribers))
                # Soak runs remain cancellable even when in-memory SDK calls do
                # not themselves suspend the event loop.
                await asyncio.sleep(0)
            elapsed = time.perf_counter()-started
            cpu_seconds = time.process_time()-cpu_started
            page_count = human.store.connection.execute("PRAGMA page_count").fetchone()[0]
            page_size = human.store.connection.execute("PRAGMA page_size").fetchone()[0]
            record_count = human.store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            dropped = human.dropped_records
            final_rss = _peak_rss_bytes()
        finally:
            if not waiting_event.done():
                waiting_event.cancel()
                try:
                    await waiting_event
                except asyncio.CancelledError:
                    pass
            await stream.aclose()
    return {"schema_version": "0.1", "benchmark": "synthetic-sdk-runtime",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "configuration": {"samples_requested": samples, "samples_completed": completed, "warmup": warmup,
            "duration_seconds_requested": duration_seconds, "store": str(store), "retention_hours": retention_hours,
            "queue_capacity": queue_size, "synthetic": True, "capture_devices_opened": False,
            "external_models_enabled": False, "pacing": "unpaced_sequential_cycles",
            "payload": "unique explicit clarification text plus numeric audio.rms measurement",
            "subject_count": 1, "window_ms": 30_000},
        "latency": {name: value.report() for name, value in metrics.items()},
        "resources": {"wall_seconds": elapsed, "process_cpu_seconds": cpu_seconds,
            "process_cpu_percent_of_one_core": cpu_seconds/elapsed*100 if elapsed else None,
            "peak_rss_bytes_before_measurement": initial_rss, "peak_rss_bytes_after_measurement": final_rss,
            "rss_semantics": "process_lifetime_high_water_mark_including_interpreter_and_warmup",
            "sqlite_allocated_bytes": page_count*page_size, "stored_record_count": record_count},
        "backpressure": {"stalled_event_subscribers": 1, "first_event_received": observed_first_event,
            "max_observed_queue_depth": queue_depth_max, "queue_capacity": queue_size,
            "dropped_deliveries": dropped, "scope": "subscriber_deliveries_not_sensor_frames"},
        "limitations": ["Synthetic inputs; no sensor or model accuracy measurement",
            "Sequential unpaced workload; not a concurrent application load test",
            "No ASR, camera, VAD, transport, GPU, or external reasoning latency included",
            "RSS is a process lifetime peak, not a precise SDK allocation measurement",
            "Duration mode does not certify an 8-hour soak unless actually run for that duration",
            "Timing reservoirs hold at most 10000 observations per operation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--store", default=":memory:", help="SQLite file path/URI, or :memory:")
    parser.add_argument("--retention-hours", type=float, default=24)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--json-output", type=Path, help="New output file; existing files are never overwritten")
    args = parser.parse_args()
    if args.json_output is not None and args.json_output.exists():
        parser.error("--json-output already exists; choose a new filename")
    try:
        report = asyncio.run(run_benchmark(samples=args.samples, warmup=args.warmup,
            duration_seconds=args.duration_seconds, store=args.store,
            retention_hours=args.retention_hours, queue_size=args.queue_size))
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.json_output is not None:
        with args.json_output.open("x", encoding="utf-8") as handle:
            handle.write(rendered+"\n")
    print(rendered)


if __name__ == "__main__":
    main()
