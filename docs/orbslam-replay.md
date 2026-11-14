# Recorded-input ORB replay

This checkout command makes a pinned input run repeatable; it is NOT a quality,
physical trajectory or real-device latency benchmark. Preprovision the native
runtime and assets from [the build recipe](orbslam-native.md) in the selected
Python environment. No installation or download happens in this command.

```sh
/path/to/runtime/python -m benchmarks.orbslam_replay \
  --manifest /absolute/path/replay/manifest.json \
  --output /absolute/path/replay/run-001.jsonl
```

Run from this repository checkout. Use `--rotate --max-starts-per-minute 6` to opt
into [bounded session rotation](orbslam-rotation.md); the finite default rejects a
manifest longer than the configured native frame budget, including explicit resets.
Rate exhaustion in an unpaced replay can stop rotation intentionally. No sleeping,
hidden retry or bypass of the startup limit occurs. Use an explicitly chosen limit
up to60 or split recordings into separately identified runs. Do not combine those
runs as one continuous SLAM map.

## Manifest

The manifest is strict JSON, schema1, at most1MiB and3000captures. Example structure
(replace paths/hashes with actual local assets; placeholders are not runnable):

```json
{
  "schema_version": 1,
  "config": {
    "library_path": "/absolute/libgamst_orb_bridge.so",
    "vocabulary_path": "/absolute/ORBvoc.txt",
    "settings_path": "/absolute/camera.yaml",
    "library_sha256": "<64 lowercase hex characters>",
    "vocabulary_sha256": "<64 lowercase hex characters>",
    "settings_sha256": "<64 lowercase hex characters>",
    "calibration_id": "cal1",
    "max_frames": 300,
    "max_gap_ms": 1000,
    "sequence_step": 1
  },
  "sensor": {"source_id":"cam", "sensor_id":"video", "coordinate_frame":"camera", "stream_epoch":"recording1"},
  "shape": [480,640,3],
  "captures": [
    {"path":"captures/0000.bgr", "sha256":"<64 lowercase hex characters>", "capture_ms":1000, "capture_sequence":0, "reset_before":false}
  ]
}
```

Capture files are headerless, row-major uint8 BGR bytes matching `shape` exactly;
no image decoding, resizing, color inference or calibration estimation happens.
Existing recorded arrays can be written with `array.tobytes(order='C')` after their
BGR convention and calibration are independently checked. File dimensions must
match native settings, checked by the worker. Each capture path must resolve inside
the manifest directory and be a regular file. Use trusted immutable local files;
path checks/hashes are not a filesystem sandbox against concurrent replacement.

Capture milliseconds and original indices strictly increase between explicit
reset markers. Delivery sequence is the manifest row index; original capture index
is preserved separately. Source and calibration identity remain fixed in one
manifest. A reset permits a new time origin but never replenishes the frame budget.
Missing/altered/wrong-size files stop the run and do not produce synthetic poses.

## Report and timing boundaries

The output uses exclusive file creation; existing files/symlinks are never replaced.
Maximum report size32MiB, line64KiB. Lines are flushed individually:

- `run`: status `incomplete`, manifest hash, full explicit native config (including
  file hashes), sensor/shape, local Python/package/platform and replay/provider
  source hashes. These identify declared files, not all dynamically linked binary
  dependencies or compiler flags. Native library/source attestation is not proven.
- `observation`: input hash, original returned pose/provenance and local elapsed
  time around this provider request. This can include rotation and transport, but
  excludes file read/hash and the initial worker startup. Reset time is outside
  the per-request measurement. No pixel bytes enter the report.
- `summary`: final `complete` only after all requested captures and successful
  worker cleanup; otherwise `failed` and an error type. No summary means interrupted
  or incomplete, never success. Process failures/cancellation reap the owned child.
  Abrupt OS termination or disk write failure may leave only a partial artifact.

Timing is **unpaced recorded input**. p50/p95 are nearest-rank statistics over
completed requests only; failed attempts are not included. Total replay time is a
separate local duration. Never subtract recorded capture timestamps from the replay
wall clock or call this real capture-to-context latency. Rotation diagnostics may
have overlapping durations; do not sum them as independent costs.

`tracked`/`abstained` counts describe reported states, not pose correctness. Reference
trajectory error, actual cost, native peak memory and capture-to-context latency are
null; quality/actual-SLAM/continuous-coverage verification remains false. The CLI does
not validate a ground-truth trajectory or certify that a supplied native binary is
real upstream inference. Those gates remain in [#186](https://github.com/NomaDamas/Gamst/issues/186).
Software slice: [#197](https://github.com/NomaDamas/Gamst/issues/197).

```sh
python3 -m unittest discover -s tests -p test_orbslam_replay.py -v
```

Tests execute the real CLI/owned worker/ctypes/compiled bridge with an explicitly
**test-double System.h**. They verify artifacts and lifecycle only, not SLAM quality.
