# Owned recorded-source Context replay

This benchmark runtime opens one real `HumanSense` and `SensorStreams` instance,
replays explicitly supplied BGR/PCM captures at 1x, and takes scheduled Context
snapshots. It is the source/clock part of #256 (#268). It does not yet execute
the persistent A/B/C experiment, reconcile recording scopes, score answers or
establish cheaper inference. Those remain #269/#270 and actual #57/#135 gates.

```sh
python3 -m examples.recorded_source_context /absolute/path/to/new-example
```

The example creates a constant PCM fixture, runs the actual dependency-free
audio-level calculation and records two overlapping task windows. The signal is
synthetic; PCM energy is neither speech recognition nor emotion classification.
No live source, network endpoint, trained model or package install is activated.
The output directory must not already exist.

For original recordings, supply strict JSON files and an explicit asset root:

```sh
python3 -m benchmarks.recorded_replay \
  --manifest source.json --tasks tasks.json --profiles profiles.json \
  --assets /absolute/recording/assets --output /absolute/new-replay \
  --retain-evaluation-artifacts
```

Retention consent is separate from SDK consent. These files contain original
task inputs, source paths and sensory Context. SDK deletion does not erase the
external evaluation artifact. Read-only CLI help does not start a source.

## Source contract and ownership

`benchmarks.recorded_source.validate_manifest` defines schema1. Every source has
`source_id`, `sensor_id`, original `stream_epoch`, `format`, `modality` and an
explicit `calibration_id` (nullable). Records bind these identities, original
sequence/time, playback offset, relative payload path, exact byte count and SHA256.
Availability partitions are declared per source and are not observed runtime
availability. Records in declared unavailable intervals are rejected.

Supported byte layouts are `bgr8_hwc` (plain contiguous uint8 H×W×3, at most
1920×1080) and `pcm_s16le_mono` (8–48 kHz, at most 100 ms per block). No implicit
color conversion, resampling, compressed decoder or extension-based guessing is
performed. Source paths stay beneath the selected asset root; every directory
and file is opened without following symlinks. The runtime checks regular-file
size and hashes the actual bounded, owned bytes, both before activation and again
for acquisition. A changed file is not delivered merely because preflight passed.

The joint ceilings are 262,144 records, 64 MiB of manifest, 64 GiB of unique
payloads, 16 sources, 64 snapshot tasks and one hour of playback. An initial
implementation selects exactly one provider lane per source. These independent
ceilings are not a promise that every combination fits: source copies, provider
queues, journal events and result files must also pass joint admission. Input is
never silently downsampled to fit. One hour of 30 fps uncompressed 1080p video
exceeds the payload limit.

Each source has its own asynchronous read/schedule loop. File reads run in owned
threads and cancellation joins them. Source `attempted`, `captured` and `dropped`
describe due acquisition opportunities. Submitted packets, lane evictions,
processing and errors are separate downstream counts. Prefetch reads do not
count as captured before their opportunity is reached. Physical-device internal
losses remain unknown. Late opportunities are dropped under the frozen tolerance,
with no burst catch-up. Vision inference is never awaited by the PCM loop.

Profiles are a closed enum in `benchmarks.recorded_profiles`: `pcm_level` uses
the actual existing audio math; `opencv_lk` uses actual installed OpenCV;
`fixture_bgr` is explicitly synthetic and has a declared test delay of 0–500 ms.
The profile declaration has exactly `profile` and `delay_ms`. Dependencies must
already exist; discovery does not install them. No arbitrary factory, import
path or command is accepted from the manifest. Captured copies are conservatively
admitted separately from unmeasured provider memory and total RSS.

## Time and Context meaning

After provider startup, a fixed mapping associates original capture offsets with
one SDK Unix-millisecond anchor and one host monotonic-nanosecond anchor. SDK time
uses floor rounding; original nanoseconds and the submillisecond remainder are
retained. Scheduling uses monotonic time. Wall/monotonic disagreement beyond the
frozen tolerance aborts the run; clocks are not monkeypatched or re-anchored.

A registered `replay.mapping.v1` observation identifies replay metadata, not a
specialist inference. It binds the original manifest/epoch, mapping hash and SDK
provider lane epochs. SDK publication replaces the original source epoch with its
owned lane epoch while preserving capture sequence. Journal packet identities
and mapping records provide that explicit bridge.

Tasks freeze original input/digest, opening, snapshot, deadline and Context window.
At the scheduled snapshot the runtime calls real `get_context` without waiting
for future provider output. Each detached result includes actual snapshot begin
and end, input/task/mapping hashes, submitted source watermarks and exact included
evidence IDs and capture sequences. This is not a complete global publication log.
Unknown, stale, missing and truncated Context retain their existing SDK meaning.

Normal termination drains the selected lanes before closing the SDK. Cancellation
stops admission and joins owned jobs, reads and cleanup; it does not manufacture
successful future snapshots. Startup, acquisition, task timing and teardown are
recorded, while financial costs and model quality remain unknown/unverified.

## Artifact limits and verification

The new source journal is independent of #255's per-task transport. It admits at
most 2 KiB per event, 64 queued events and 1 GiB total, with conservative admission
before activation. Output creation is exclusive. Task Context bodies are at most
64 KiB each and 4 MiB in aggregate; envelopes include separate bounded metadata.
The writer preserves valid prefixes after failures and writes an immutable terminal
only after owned cleanup. A completed fixture run is still unscored and actual
acceptance remains incomplete.

Tests use real files, the actual SDK, PCM math and explicitly labeled video doubles
to check original-time replay, overlapping snapshots, slow video with progressing
audio, source-versus-lane losses, changed source bytes and owned cancellation.
They do not prove trained perception, physical sensing, billing authenticity,
whole-pipeline savings or matched-recording isolation.

PCM availability is checked over the entire half-open sample interval using exact rational endpoints; a chunk cannot overlap a declared unavailable interval. Asset access uses POSIX directory descriptors and no-follow opens; Windows asset replay is not supported by this implementation.

The independent `source-mappings.json` inventory preserves original source identities and initial provider epochs even if no Context snapshot succeeds. Terminal metadata hashes that inventory separately from the immutable clock mapping. Cleanup records pre-close queued/unfinished/active work and final provider counters. Residual accepted work is labeled `unresolved_delivery` (unknown discarded or cancelled), never acquisition loss or proof of exact backend cancellation. A final clock check after SDK cleanup prevents cleanup-time wall-clock changes from producing an unfailed replay.
