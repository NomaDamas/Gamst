# Capture-bound Sparsh slip Context

`SparshSlipContext` connects the #224 tactile packets to the #235 supplied local
graph, registered observations, SensorStreams, an owned ProcessProvider worker,
API/MCP and recorded replay. It supplies sensory context only; it never calls an
LLM or changes user delivery cadence. This is overlapping finite-window inference,
not a native stateful streaming model or an achieved realtime throughput claim.

The first five captures report `warming` with no slip decision. Six consecutive
captures form **current then t-5**. Every subsequent admitted capture can produce
an overlapping window. A skipped capture invalidates history; “t-5” never means
five sparse inference calls. Slow CPU inference can cause the bounded manager's
queue to drop captures and continually warm up; report the queue/drop/reset
metrics instead of silently resampling or claiming sustained sensor-rate inference.

## Local configuration and reference consent

Use `ProcessProvider.capability('sparsh.slip.local', config)` for dependency-light
discovery and `await ProcessProvider.open(existing_python, 'sparsh.slip.local',
config)` for explicit owned runtime activation. Select an already configured
runtime; no installation, weight download or sensor opening happens implicitly.
The `sparsh` optional group includes the native runtime plus NumPy/Pillow.

Configuration has exactly these fields:

```json
{
  "source_root": "/absolute/sparsh",
  "encoder_path": "/absolute/encoder.safetensors",
  "head_path": "/absolute/slipforce.safetensors",
  "encoder_sha256": "<encoder SHA256>",
  "head_sha256": "<head SHA256>",
  "reference_path": "/absolute/reference.uint8",
  "profile": {
    "profile_id": "declared-task-v1",
    "source_id": "touch",
    "sensor_id": "finger",
    "sensor_model": "digit",
    "serial": "declared-serial",
    "width": 320,
    "height": 240,
    "color_order": "BGR",
    "calibration_id": "reference-v1",
    "illumination_id": "lighting-v1",
    "coordinate_frame": "native-bgr",
    "clock_id": "host-unix",
    "reference_sha256": "<raw reference SHA256>",
    "orientation": "landscape_cw",
    "training_config_sha256": "<verified effective training config SHA256>",
    "training_domain_id": "declared-domain",
    "min_interval_ms": 15,
    "max_interval_ms": 19,
    "min_pair_span_ms": 75,
    "max_pair_span_ms": 95,
    "reference_consent": true
  }
}
```

These cadence values are **illustrative caller declarations**, not verified head
tolerances. Replace them and the placeholders with actual task provenance.
The profile binds sensor/serial/color/geometry/clock/calibration/illumination and
reference hash. The reference file must be exactly width×height×3 immutable raw
uint8 bytes in the declared layout. It loads and verifies before the native model.
Explicit `reference_consent: true` authorizes local reference use; it does not
prove that the reference is contact-free or physically calibrated. Source epoch
and manager runtime epoch are independently tracked. A profile mismatch fails and
clears history; reconnect/gap/epoch changes reset warmup and the causal decision
history. Changing a profile/reference requires constructing a new provider.

## Pinned preprocessing and decision policy

Reference source is
[facebookresearch/sparsh fee6a05a](https://github.com/facebookresearch/sparsh/tree/fee6a05a97330eca014b59c71ac41596fe3974a7):
`data/vision_based_forces_slip_probes.py`, `data/digit/utils.py`,
`config/task/t2_slip_detection.yaml` and `test/test_t2_slip.py`.
The executable preprocessing helper SHA256 is
`8bfc5e3c255ec7617236e41ad6112e001d59badf7da94b2fe81dbf20d6b5103c`.

1. Convert both declared BGR images and reference to RGB once.
2. Subtract in signed int32, divide by255, add0.5, clip to[0,1], multiply255 and
   truncate to uint8. Identical images yield127, not128.
3. Rotate landscape clockwise. Only exact4:3 landscape and3:4 portrait layouts
   that fit the capture bounds are supported. The upstream unusual other-aspect
   crop branch is not approximated.
4. PIL RGB bilinear resize to width240/height320, convert float32 CHW and divide255.
   No enhancement or ImageNet normalization. Concatenate current RGB then previous
   RGB into one6channel input.

Outputs preserve both logits, a stable float64 softmax slip probability (uncalibrated),
and `candidate_slip = probability > 0.4`. A separate causal debounce requires three
consecutive positive decisions, initialized/reset with false entries as in the
upstream tester. This reproduces the declared decision rule, not a claim of exact
floating-point parity to upstream Torch softmax at a threshold boundary. Evidence
includes all three decisions' input-pair hashes/sequences/timestamps, not just the
latest pair. Warmup is not a no-slip observation.

`no_slip` never means `no_contact`. Contact is `not_assessed`. Optional three-axis
normalized delta-force stays unitless; config-dependent scale factors are never
guessed into Newtons. No pose, safe-grasp or validated contact interpretation is
provided. Original capture times and source/clock identity are separate from
preprocessing and inference duration. Native artifact validity, trained quality,
reference no-contact truth and physical calibration are separate flags.

## Ownership, API and replay

Each provider owns one bounded six-frame history, one immutable reference and at
most three decision evidence records. Prediction/reset/close share a lock.
`SensorStreams` additionally owns queue limits, generation invalidation and
cancellation; use the registered ProcessProvider path for an isolated native
worker. The synchronous direct API cannot interrupt native work. A caller doing
its own `to_thread` must retain and drain that task, as documented for the native
bridge. Do not equate direct synchronous calls with an async publication guarantee.

```sh
python -m humansense.sparsh_replay \
  --manifest /absolute/tactile-recording.json \
  --config /absolute/sparsh-context.json
```

This implemented command emits one bounded observation per validated packet and
closes its owned provider on completion/failure. It requires all source/native/
reference assets; missing assets fail, never substitute a mock. The existing
`humansense.tactile_replay` remains the model-free raw-input inspection command.
A partial replay followed by a bad packet is incomplete evidence, not acceptance.

## Verification boundary

Focused tests cover exact color/quantization/order, strict profiles, cadence and
pair-span bounds, source and runtime epoch resets, inference failure, blocking
lifecycle ownership, actual worker dispatch/stream/MCP and the real replay command
with a clearly labeled fixture native predictor. Optional source verification runs
the actual pinned helper for random landscape/portrait/downsample fixtures:

```sh
GAMST_TEST_SPARSH_PREPROCESS_SOURCE=/absolute/sparsh/tactile_ssl/data/digit/utils.py \
  python -m unittest discover -s tests -p test_sparsh_context.py -v
```

The helper source hash is checked before execution; tests never download it.
Existing local dependencies used: Pillow12.2.0, NumPy2.2.6 and torchvision0.23.0
for the optional upstream oracle. Equality is exact for tested tensors, not proof
of a sensor's training-domain fit. No packages, trained assets or hardware were
installed. #226 remains open for actual trained slip/contact heads, reference
truth/domain/orientation/cadence validity, held-out replay and physical force
calibration. Native graph tests and fixture Context tests cannot close that gate.
