# Calibrated planar PIV context

```mermaid
flowchart LR
  A[Paired calibrated tracer images] --> B[OpenPIVFlow: previous and current capture]
  B --> C[OpenPIV 0.26.1 SciPy correlation]
  C --> D[Bounded vectors with invalid entries marked]
  D --> E[SDK / MCP / LLM context]
```

The capture owner supplies uniformly calibrated, stationary-camera particle images
and capture timestamps. The provider owns one previous grayscale frame, pair
continuity, OpenPIV invocation, unit conversion and bounded context. It reports
planar image flow estimates; it does not identify fluid, reconstruct 3D velocity,
infer volume/fill or verify physical calibration. Family #37 remains open, including
actual-runtime/fluid acceptance [#107](https://github.com/NomaDamas/Gamst/issues/107).

## Selected implementation

Optional environment: `humansense-sdk[piv]`, selecting **openpiv==0.26.1**.
No automatic install/download/device startup. Core import and profile discovery
need only the standard library. `OpenPIVFlow.from_installed(**config)` checks the
installed version and imports `openpiv.pyprocess.extended_search_area_piv` only
when explicitly selected. The low-level backend argument permits focused tests;
that argument alone is not installed-package compatibility evidence.

Pinned upstream source: [OpenPIV 0.26.1 pyprocess](https://github.com/OpenPIV/openpiv-python/blob/048474c4db04c66156bca2b89632a76d778cddcb/openpiv/pyprocess.py).
OpenPIV is GPL-3.0-or-later; the gamst adapter is Apache-2.0. OpenPIV's package
metadata requires Python>=3.10, NumPy>=2 and SciPy>=1.11 plus its other declared
dependencies. Keep this optional runtime/license distinct from the core SDK.
No upstream implementation is vendored into gamst.

The fixed algorithm uses 32px interrogation windows, 64px search windows, 32px
overlap, linear normalized correlation, Gaussian subpixel peaks, peak-to-peak SNR
with exclusion width2, and `backend='scipy'`. It never selects optional Rust
implicitly, fills invalid vectors, smooths flow, or runs multigrid deformation.
The pinned scalar subpixel routine omits backend forwarding internally. Therefore
the installed loader and source smoke explicitly reject `pyprocess.HAS_RUST=True`:
use a dedicated runtime without optional `openpiv_rust`. Setting `backend=scipy`
alone is insufficient in this release. The adapter does not uninstall anything
or mutate shared upstream globals. Test backends bypass installed-runtime loading.

## Input and context

```python
config = dict(meters_per_pixel=0.001, calibration_id='lab-plane-v1',
              coordinate_frame='camera-plane', max_gap_ms=1000,
              snr_threshold=1.3, max_speed_m_s=100)
```

Use `InputFrame(uint8_bgr, 'video', capture_timestamp_ms, sensor=metadata)`.
Inputs and backend grids must be plain NumPy arrays; masked arrays/subclasses
are rejected so missing pixel/vector masks cannot be silently discarded.
Both axes must be64..256px; no implicit crop/resize changes calibration. Pixel
values become owned int32 grayscale using rounded BGR luminance weights
0.114/0.587/0.299. Metadata frame/calibration must match the configuration.
Scale is1e-9..1meter/pixel; gap1..10000ms, SNR threshold1..100,
maximum speed1e-6..10000m/s. These bounds constrain configuration, not accuracy.
Configuration is hashed into the provider version; do not mutate it after creation.

For a continuous pair, dt is the integer capture-time difference in seconds.
OpenPIV returns pixels/second; multiply by meters/pixel exactly once. Vector
centers are search-window centers on the actual sliding grid (starting32px with
32px stride). x increases right and y down from the top-left image pixel origin;
no Cartesian-axis inversion or camera-motion compensation is performed.
Uniform scale assumes an appropriate planar calibrated imaging setup.

Context `vision.planar_flow` contains up to49vectors, each with `center_m`,
`velocity_m_s`, SNR and validity. Nonfinite velocity/SNR, SNR below threshold,
SNR above1e6, or speed above the configured bound makes velocity `None`.
Invalid SNR is also `None`; rejection count is explicit. A finite low SNR is
retained for diagnosis. `calibration_verified` and `fluid_identity_verified`
remain false. No validity score establishes that an image is physically a fluid.

First/discontinuous frames emit `warmup` with no pair interval/vectors. Global
grayscale standard deviation below1 in either frame emits `insufficient_texture`
without calling correlation; textured frames can still return `unavailable_vectors`.
The texture check is a conservative shortcut, not proof of adequate tracer density.
Partial valid grids report `estimated` and explicit valid/rejected counts.

Owner/subject/epoch, shape, or capture-sequence mode changes warm up. Sequence
skips and gaps above max_gap_ms also warm up. Capture sequence is authoritative
when provided, otherwise delivery sequence. Repeated/backward time or sequence,
invalid input and malformed backend grids reject and clear state. Calibration
mismatch rejects; use a new instance for new calibration. Only previous-frame
state is retained, at most256KiB int32, until next process/reset/close. Copies and
FFT workspace are additional bounded per-call allocations. Processing/reset/close
are serialized; close clears state/backend. Direct calls have no hard timeout.

## Streaming and process isolation

```python
from humansense import ProcessProvider, SensorStreams

spec = ProcessProvider.capability('openpiv.flow', config)
async def create():
    return await ProcessProvider.open('/explicit/piv-env/bin/python', 'openpiv.flow',
                                      config, inference_timeout=5)
streams = SensorStreams(human)
streams.register(spec, create)
await streams.open('camera', 'piv-device', [spec.id])
# submit matching InputFrame/SensorMetadata; drain/close through normal SDK lifecycle
```

Native mode means a causal, stateful two-capture algorithm, not proof of a target
FPS. Time window is the observed pair interval with zero lookahead. The managed
lane resets on gaps; worker ownership/timeouts use existing ProcessProvider rules.
MCP/API receive vectors/provenance, not image arrays. Capture and context delivery
cadences stay separately configured by the host.

## Reproduction and evidence

`python3 -m unittest discover -s tests -p test_piv.py -v` checks arithmetic scaling,
invalid/native-output rejection, copies, gaps, lifecycle, metadata-only discovery,
pinned loader and actual managed SDK/MCP transport with a backend double.

Installed-runtime synthetic smoke:

```sh
PYTHONPATH=. python3 examples/validate_piv.py --output new-piv-report.json
```

For source-level validation without installing OpenPIV, provide an already local
copy of the pinned `pyprocess.py` with `--reference-source /path/pyprocess.py`.
The script accepts only SHA256
`09019b6486fbd28628d977eb86112bde97310c2381289ab145340e0811030601` and executes those
verified bytes; it never downloads code. NumPy/SciPy must already be installed.
The report labels source mode separately from installed-package execution.

Development evidence: Python3.10.11, NumPy2.2.6, SciPy1.15.3, macOS15.6.1arm64;
pinned core source executed four seeded128x128 translated-particle cases (positive,
negative, stationary, doubled dt). Each produced9valid vectors; maximum component
error across cases0.000683m/s against a predeclared0.002m/s bound. Pair-processing
observations were3.28..5.71ms in one short run; these are neither latency percentiles
nor real-time acceptance. No packages/weights were installed. This proves core
numerical invocation on synthetic images, not the installed distribution, real
fluid accuracy, live capture timing or sustained memory/latency. Those remain#107.
