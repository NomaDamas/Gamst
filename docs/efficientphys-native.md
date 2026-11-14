# EfficientPhys local graph bridge

This internal bridge executes a verified local EfficientPhys graph on an already
preprocessed bounded RGB window. The [face-window context adapter](efficientphys-context.md) now adds capture binding,
temporal assembly, explicit online normalization and SensorStreams/MCP integration.
Trained physiological quality and real reference acceptance remain required in [#137](https://github.com/NomaDamas/Gamst/issues/137), under [#35](https://github.com/NomaDamas/Gamst/issues/35).
No existing family is marked supported or complete by this bridge.

## Source and runtime

Pinned [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox/tree/b7500b848f84ad7f86e277b4612563b69f4f88f9),
`neural_methods/model/EfficientPhys.py`, SHA256
`384b01b5c99d6d280b5ddb9e2358631f89747e40e3181b6a174e26d6e74798c9`.
The loader executes only these exact user-supplied source bytes, without modifying
or redistributing the upstream model. No source/checkpoint download occurs.
The upstream repository's license is Responsible Artificial Intelligence Source
Code License1.1; do not label it MIT/Apache. Weights require separate provenance
and license validation. Adapter code retains the repository license.

Native runtime requires existing Torch2.8.0 and NumPy. Default CPU; explicit CUDA
requires an available device and is not validated here. Absolute local source and
checkpoint paths plus checkpoint SHA256 are mandatory. Source reads are bounded
at16KiB, checkpoint reads at32MiB. `weights_only=True`, exact tensor names, shapes,
dtypes and finite values are required. Both bare state dictionaries and uniformly
`module.`-prefixed DataParallel state dictionaries are accepted; mixed or wrapped
formats fail. No runtime dependency is imported merely by importing this module.
Untrusted artifacts should execute through the existing isolated worker when the
context profile is connected; this low-level API itself does not spawn a process.

## Input and temporal meaning

`EfficientPhysNative.from_local(source_path, checkpoint_path,
checkpoint_sha256=..., image_size=72).predict(rgb)` consumes float32 finite NHWC RGB,
image size36/72/96,10..300frames in multiples of10. Inputs are copied, channels are
transposed, the last frame is duplicated once, and the model returns N scalar
samples. Values are arbitrary model units, **not BPM**, blood pressure, emotion or
a diagnosis. There is no automatic resizing, color conversion or standardization.

The pinned official PURE-to-UBFC-rPPG configuration uses72px,30fps,180frame chunks,
frame_depth10 and Standardized data. Its BaseLoader globally standardizes the
complete source video before chunking. A live trailing-window normalization is
NOT numerically equivalent. The forthcoming capture adapter must declare and
validate this difference rather than use future frames while claiming causality.

The graph first differences frames and uses four bidirectional temporal-shift
layers inside ten-frame groups. Full input must be available before predict;
there is no persistent temporal cache or early per-frame emission. Terminal frame
duplication follows the official single-device trainer, creating a final zero
input difference; it does not imply a zero final prediction. No capture clock is
inferred here. A180frame input sampled at30fps spans179/30seconds between first and
last timestamps, before processing/transport. Report availability from actual
capture times, not merely graph runtime. No performance or quality claim follows
from supporting small inputs.

Each call is independent; close is idempotent, releases the graph reference, and
prevents further prediction. It does not promise immediate allocator/GPU-memory
release. Calls and close are serialized. Invalid input/output raises explicitly.

## Validation

```sh
python3 -m unittest discover -s tests -p test_efficientphys_native.py -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. python3 examples/validate_efficientphys_fixture.py \
  --source /absolute/EfficientPhys.py \
  --runtime-root /absolute/project/runtime \
  --output /absolute/project/efficientphys-fixture.json
```

The second command uses only a previously supplied exact source file and existing
libraries. It generates a temporary **untrained** DataParallel checkpoint in its
own temporary directory, reloads through the strict bridge, compares180 outputs
against a direct graph invocation, then removes only its own generated fixture.
Output is created exclusively; existing reports are never overwritten. This
checks actual native graph wiring, not physiological accuracy or trained-model
parity. It does not install or fetch weights.

Local macOS/Python3.10/Torch2.8.0 check:2,163,081 parameters,180 outputs,
exact direct-graph parity. Fixture SHA256
`2acc0f72bd5d83290df2d830fd166a9b8d7b72dea3842722ab9460698947dab0`.
The1.785second local invocation is a single fixture measurement, not a live
latency, hardware throughput or cost benchmark. Four focused tests verify input
bounds, no hidden normalization, tensor/layout ownership, terminal duplication,
strict checkpoint formats, output validation, import isolation and closure.
CI runs them with CPU Torch; the upstream source is not fetched by CI.

Required next: real checkpoint and reference pulse data, preprocessing quality,
validated motion/lighting rejection, measured capture-to-output latency and
licensing evidence. Capture/ROI assembly and profile integration are now provided
by the linked context adapter. OpenEB and
visual vibration in #35 remain separate outstanding deliverables.
