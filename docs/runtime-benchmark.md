# Synthetic runtime benchmark

Run from the repository root with Python 3.10 or newer:

```bash
python -m benchmarks.runtime --samples 200 --warmup 20 --json-output /tmp/humansense-run-200.json
python -m benchmarks.runtime --samples 500 --warmup 20 --json-output /tmp/humansense-run-500.json
```

Each measured cycle submits a unique synthetic clarification request, submits a
synthetic numeric `audio.rms` signal, and queries the current state. The runtime
uses one subject, its default 30-second window, and no perception models or
capture devices. Text and signal timings include SDK ingestion, SQLite storage,
rule-based inference and event publication. This measures processing time after
input arrival, not camera-to-event or microphone-to-transcript latency.

The harness creates one event subscriber, consumes its first event, then leaves
it stalled. Its queue capacity is eight by default. Reported drops count
subscriber deliveries, not missing sensor frames. Each latency metric reports
nearest-rank p50/p95, arithmetic mean and maximum. Runs above 10,000 operations
use a deterministic bounded reservoir for percentiles, while count, mean and
maximum still cover the complete run.

The JSON report includes Python/platform information, wall and process CPU time,
SQLite allocated pages, and process peak RSS. RSS is the process lifetime high
water mark, including Python, imports and warmup; it is not a measurement of
precise SDK-only allocations. CPU usage is relative to one core and can differ
slightly from 100% due to clock measurement resolution. Background host activity
is not controlled.

## Measured checkout result

Measured on 2026-11-12 UTC, CPython 3.12.3,
Linux 7.0.0-31-generic x86_64, eight logical CPUs reported by the environment.
These are actual synthetic runtime measurements, with twenty warmup cycles and
an in-memory SQLite store. The first two rows measured the initial implementation;
the final row measured the same workload after bounding copied evidence.
They are not performance guarantees.

| Measured cycles / implementation | Text p50 / p95 | Signal p50 / p95 | State p50 / p95 | Wall time | SQLite allocation | Peak process RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 200 / initial | 4.50 / 7.99 ms | 2.92 / 5.11 ms | 0.80 / 1.34 ms | 1.67 s | 8.82 MiB | 39.08 MiB |
| 500 / initial | 11.24 / 24.05 ms | 7.16 / 15.19 ms | 1.93 / 4.18 ms | 10.78 s | 47.85 MiB | 84.76 MiB |
| 500 / bounded evidence | 12.06 / 23.66 ms | 5.77 / 11.53 ms | 1.77 / 3.48 ms | 10.07 s | 8.32 MiB | 40.15 MiB |

The 200-cycle run retained 1,080 records and dropped 191 stalled subscriber
deliveries. Each 500-cycle run retained 2,580 records and dropped 491 deliveries.
All queues reached their configured capacity of eight. These unpaced runs used
approximately 100% of one CPU core. The JSON artifacts from this execution are
`/tmp/humansense-runtime-200.json`, `/tmp/humansense-runtime-500.json`, and
`/tmp/humansense-runtime-500-bounded-evidence.json` on the development host;
these temporary files are not part of the repository.

The initial increase exposed retained-history costs. Each inference copied all
current-window clarification references. The corrected runtime keeps exact
event counts but includes only the latest eight evidence references per event
type. Repeated-utterance events retain at most sixteen cited utterances while
reporting their full measured count. Context summaries follow the same bounded
evidence approach and explicitly flag truncation. Detailed records remain
available through history and interval inspection until retention removes them.
This reduced the 500-cycle SQLite allocation by approximately 82.6% and peak
process RSS by approximately 52.6% in the measured rerun.

Retained numerical-history and window queries still grow with observation
volume, and text latency changed little in this comparison. The short-run
numbers do not establish bounded eight-hour memory or stable long-run latency.
The runtime still requires a genuine long-run soak and optimization before
claiming those engineering targets. Perception model and sensor accuracy,
multi-user fairness, calibration validity, GPU usage and ASR latency are outside
this benchmark.

## Disk storage and longer observation

Use a new database path for a disk-backed run:

```bash
python -m benchmarks.runtime --samples 200 --store /tmp/humansense-benchmark-new.db
```

The harness refuses an existing database path so it cannot mix benchmark data
with existing observations. JSON output also refuses to overwrite a file.
SQLite records remain in the selected file until explicitly removed; the
workload itself is synthetic. The retention duration is configurable using
`--retention-hours`.

Duration mode supersedes the sample count and keeps running unpaced until the
specified elapsed time. It yields to the event loop between cycles and limits
its own timing-reservoir memory:

```bash
python -m benchmarks.runtime --duration-seconds 60 --store /tmp/humansense-soak-new.db
```

An eight-hour test would use `--duration-seconds 28800`, but **that test has not
been run**. Duration mode is an unpaced stress workload, not a realistic camera
frame schedule. It can create a large derived SQLite database. Select an
appropriate dedicated path and monitor the host when running a long test.
