# EfficientPhys face-window context

`efficientphys.local` connects a caller-selected face ROI to an owned local
EfficientPhys worker and structured `vision.rppg_context` observations. Capture
and face selection belong to the caller; window buffering and inference belong
to the provider; delivery cadence belongs to the existing SensorStreams/MCP host.
The provider never opens a camera or downloads a checkpoint.

This is window inference, not native causal per-frame inference.180consecutive
frames at nominal30fps are required. The180frame interval spans about5.967seconds
from first to last capture; inference and transport add time. `window_ms=6000`
reserves the nominal duration and `lookahead_ms=0` means no frames beyond that
window, NOT immediate output for its earlier frames. Bidirectional model shifts
and window normalization use later frames within that window.

## Explicit runtime

Use an existing environment with Torch2.8.0, NumPy and OpenCV (the optional
`efficientphys` extra declares them). CPU is the locally exercised device;
CUDA is selectable but unverified. Local absolute source/checkpoint paths and
checkpoint SHA256 are required:

```python
from humansense import ProcessProvider

config = {
    "source_path": "/absolute/EfficientPhys.py",
    "checkpoint_path": "/absolute/PURE_EfficientPhys.pth",
    "checkpoint_sha256": "<actual lowercase SHA256>",
}
spec = ProcessProvider.capability("efficientphys.local", config)
# Discovery does not import Torch, OpenCV or NumPy or launch a worker.
worker = await ProcessProvider.open("/absolute/runtime/bin/python", "efficientphys.local", config)
```

The [native bridge](efficientphys-native.md) pins upstream source revision/hash,
strict checkpoint formats and license provenance. This profile selects72px,
frame_depth10,180frames and30fps. Source and checkpoint must already be supplied;
trained assets have not been installed or validated by this implementation.

## Capture and stream integration

```python
from humansense import SensorStreams
from humansense.providers.efficientphys_context import face_frame

streams = SensorStreams(human)  # existing consent-enabled HumanSense instance
async def factory():
    return await ProcessProvider.open("/absolute/runtime/bin/python", "efficientphys.local", config)
streams.register(spec, factory)
await streams.open("camera", "rgb", [spec.id])

# capture is a video InputFrame: uint8 BGR, source=camera, sensor=rgb,
# actual timestamp_ms, monotonically increasing capture sequence and declared
# image coordinate frame. It is not an already cropped face or encoded JPEG.
packet = face_frame(capture, track_id="face-track-7", box=[120, 80, 220, 200],
                    detector_id="selected-face-detector", detector_version="pinned-version",
                    clock_id="camera-capture-clock")
await streams.submit(packet)
# Submit once per arriving capture, with the selected face's current pixel box.
# On lost face, submit track_id=None, box=None; do not reuse its last box.
```

`face_frame` validates a32px-minimum, at most1920×1080pixel BGR input (axis limits
2160/3840), copies it, hashes the owned original, crops an integer in-image xyxy
box of at least16×16px, converts to RGB and resizes to72×72 with OpenCV INTER_AREA.
No face detector is silently chosen, no bounding-box smoothing or interpolation
is applied. Caller may use an existing landmark/detection provider; coordinate
and capture matching remain the caller's responsibility. A caller-declared track
is not verified identity. A stable ID cannot detect an unreported ID switch.

The private gamst packet uses8byte GRPP001-newline magic, uint32 big-endian JSON
header length, at most8192header bytes, and either15552RGB bytes or no pixels for
explicit face loss. Strict duplicate-key rejection, payload SHA256, bounded
geometry and exact outer capture-provenance matching precede buffering. The
original-frame hash binds evidence by declaration; the worker only receives the
crop and cannot authenticate the original or prove face detection. Hashes are not
authentication. SensorStreams owns lifecycle epochs, so they are kept in the outer
sensor metadata, not frozen into the packet. Header sequence matches original
capture_sequence when present, otherwise sequence. Source/subject/calibration/
coordinate/channel provenance and timestamps are bound to the outer InputFrame.

## Continuity and output

A source, epoch, calibration, coordinate system, original image geometry, track,
clock ID or detector revision change starts a new window. Within one owner,
sequence and capture timestamps must advance. Missing sequences, per-frame
cadence error exceeding2ms, or cumulative error exceeding2ms relative to30fps
restart the window with the arriving frame. Duplicate/backward capture or malformed
input clears the unfinished window and raises. There is no invented frame,
interpolation, cross-person history or carryover of prior spectral candidates.
The unverified clock declaration does not establish actual physical synchronization.

Completed windows are nonoverlapping. Buffer retention is bounded to180RGB crops
(about2.8MB) and their evidence hashes; normalization and native tensors require
additional bounded temporary memory. While collecting, only current progress is
returned. Explicit face loss clears the window. Static RGB crops, including static
spatial texture, skip inference and report `insufficient_variation`. Dynamic
lighting/motion can still mimic physiology: this gate is not a quality classifier.
Reset clears all retained media; close also closes the backend and forbids reuse.
Existing ProcessProvider cancellation/reaping applies to isolated execution.

Every completed window globally standardizes its own RGB pixels with float64
mean/std then casts to float32. Upstream globally standardizes the **entire video
before chunking**; our online scope is intentionally different and
`offline_preprocessing_parity=False`. Output quality must be revalidated under
this policy. Constant/degenerate windows do not generate plausible-looking rates.

`signal.pulse_bpm` is always null. `spectral_candidate`, when available, is an
**unvalidated feature** of the network output: cumulative sum, linear detrend,
Hann taper, unpadded FFT and maximum bin in0.75..2.5Hz. It reports frequency,
frequency×60 (`equivalent_bpm`), approximately10BPM bin width and band-power
fraction. That fraction is not confidence; `confidence` is null. This DSP differs
from upstream full-clip smoothness-prior/filter evaluation and is not claimed
numerically equivalent. Sampling frequency is calculated from actual first/last
capture times after cadence checks. Bin selection does not establish that a pulse
was observed; aliases, motion, lighting and model errors remain unresolved.
No blood pressure, emotion, temperature, diagnosis or measured vital sign follows.

Summary includes window start/end, current capture hash, ordered window evidence
hash, frame count, segment ID, completeness, normalization, artifact verification
and explicit false quality/identity/clock verification flags. Model/checkpoint
identity participates in capability version. Observations use existing store,
MCP `get_observations` and user cadence APIs. Historical reads can return earlier
windows; consumers must respect capture interval and freshness, not call history
current physiology.

## Evidence and remaining acceptance

Six focused tests cover strict packet provenance/color/loss, real cadence/jitter/
drift/gaps, identity reset, static-window rejection,180frame buffering and exact
normalization, synthetic known-frequency DSP, typed SensorStreams→MCP context and
closure. Tests use synthetic predictors for contract coverage, not accuracy.
The extended `examples/validate_efficientphys_fixture.py` additionally executes the
actual pinned native graph through a real owned worker using temporary generated
**untrained** parameters, checks a complete window and resets it. No downloaded
weights or physiological validation are involved. See the native guide command.

Required [#137](https://github.com/NomaDamas/Gamst/issues/137) remains open: real
trained checkpoint/license, synchronized reference pulse data, camera/motion/
lighting/skin-tone coverage, online-vs-offline preprocessing quality, calibrated
quality rejection, sustained capture-to-output p50/p95/resource measurements and
real device acceptance. OpenEB/visual vibration in #35 are separate unfinished
requirements. Neither generated fixtures nor green tests close these gates.
