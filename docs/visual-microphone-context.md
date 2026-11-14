# Timed visual vibration context

`visual_microphone.phase` runs the [offline numerical engine](visual-microphone.md)
in an explicitly selected, owned `ProcessProvider` child. It publishes a compact
vibration estimate through SensorStreams/MCP, without raw pixels or waveform.
It is an offline window processor with full-window lookahead, not causal audio
streaming, ASR, sound-pressure measurement or calibrated acoustic sensing.

```mermaid
flowchart LR
  A[Caller: fixed high-speed ROI and native times] --> B[Bounded window packet]
  B --> C[Owned phase reconstruction worker]
  C --> D[Numeric observation with original capture end]
  D --> E[SensorStreams retained evidence]
  E --> F[Independent user cadence: MCP context]
```

## Explicit input

```python
from humansense.providers.visual_microphone_context import window_frame

config = dict(fps_numerator=2200, fps_denominator=1, jitter_ns=100,
              native_origin_ns="0", capture_origin_ms=known_capture_origin_ms,
              calibration_id="capture-clock")
frame = window_frame(
    pixels, native_times_ns, capture_indices, config=config,
    source_id="highspeed-camera", sensor_id="video", coordinate_frame="sensor-pixels",
    window_sequence=0, roi_id="foil1", roi_box=[10,20,42,52], source_shape=[480,640],
    source_sha256=declared_source_evidence_sha256, shutter="global", exposure_ns=1000)
```

`pixels`: plain uint8 NumPy `(N,H,W)` with128..4096 samples and32..128 pixel axes.
`native_times_ns` and `capture_indices`: plain signed/unsigned64-bit integer
vectors of lengthN. All arrays are validated before copying, then the owned
snapshot is checked. Caller must synchronize mutations during snapshotting.
Per-frame acquisition times are nonnegative int64; original frame indices are
consecutive integers in0..2^53-1. No dropped frames are interpolated.

True acquisition rate is the rational `fps_numerator/fps_denominator`,100..100000Hz,
with bounded positive integers. Both adjacent intervals and cumulative timing
from the first sample must agree with that rational rate within `jitter_ns`, at
most1% of a sample period. Integer arithmetic avoids large native-time float
precision loss. This is declared timing consistency, not calibration proof.
Playback/container FPS is never substituted. Both clock origins are explicit:

`capture_end_ms = capture_origin_ms + (native_end_ns - native_origin_ns) // 1000000`.

The returned outer timestamp must exactly match this value. Source/sensor/subject,
coordinate frame, clock calibration and original window sequence are bound to
the packet. SensorStreams may assign a delivery index/epoch, while
`capture_sequence` retains the original **window index**. The summary separately
preserves first/last original **video frame indices**. They are not interchangeable.

The fixed integer xyxy ROI must fit the declared source height/width and match
unresized pixel dimensions exactly. The caller identifies the fixed ROI and
supplies an original evidence SHA256. That hash is declared provenance, not
verified by this module against the original file or camera. Cuts within a
window cannot be detected solely from valid timestamps: the producer must split
windows at every cut, source/ROI/shutter/clock change. `global` and `unknown`
shutter declarations are accepted; rolling-shutter reconstruction is unsupported.
Optional exposure is positive nanoseconds no longer than one sample period.
Declarations do not prove the physical camera exposure/shutter.

## Wire and lifecycle

The gamst-specific format is little-endian prefix`<8sI>` with magic`GVMC001\n`
and JSON-header length. Strict bounded JSON metadata (at most8192bytes) is followed
by N int64 native nanoseconds, N int64 original frame indices, N×H×W uint8 pixels
and CRC32 of preceding bytes. Total at most4MiB, compatible with default stream
packet bounds. Exact lengths, CRC, duplicate JSON keys, dimensions, fields and
numeric limits are checked before inference. CRC is not authentication.

Within the same owner, original windows and frame indices must advance without
overlap. Adjacent indices, window sequence and sample timing retain a state
segment; gaps or a changed owner/ROI/evidence/shutter/epoch start a new segment.
No history is stitched across boundaries. Duplicate/overlapping windows reject;
malformed processing clears continuity before raising. Reset clears state;
close is permanent and idempotent. An error does not refresh old evidence; an
older successful observation remains historical until normal retention/freshness
rules apply. The next complete window starts a fresh segment after reset/error.

```python
from humansense import ProcessProvider
spec = ProcessProvider.capability("visual_microphone.phase", config)
worker = await ProcessProvider.open("/absolute/python", "visual_microphone.phase", config)
try:
    numeric = await worker.process(frame)
finally:
    await worker.close()
```

Capability discovery imports no NumPy/SciPy and starts no process. The explicitly
selected runtime must already contain gamst and the `visual-microphone` optional
dependencies; nothing installs automatically. Worker deadlines/cancellation/
reaping use the existing owned-process lifecycle. Direct `VisualMicrophoneContext`
execution is synchronous; use the child for full-duplex host isolation.

## Output and cadence

Output includes status, unvalidated peak frequency/power-bin fraction, frequency
resolution, Nyquist/high-pass limits, window native start/end, original indices,
ROI/source/clock evidence, hashes and continuity. Waveforms are discarded by this
provider, never serialized to an LLM or stored as audio. The local numerical API
is separate if an application explicitly needs a reconstructed waveform.

Flat/lost-texture or numerically static windows publish an abstention with
`peak_hz=None`, replacing the latest estimate. `quality='not_assessed'`, hardware
and clock validation false, source hash verification false and capture
completeness unverified remain explicit. A spectrum peak/power fraction is not
proof of sound, confidence, intelligibility or physical displacement. Numerical
samples can alias and lighting/camera motion can dominate the result.

The caller submits complete windows; no accumulator publishes partial claims or
refreshes the timestamp of an old completed result. Observation freshness is
1000ms from the original capture end, not processing completion. Inference and
Context delivery are independent schedules. Old offline evidence may be outside
current-query/retention windows; never rewrite capture time to make it fresh.

## Runnable example and tests

Save `frame.data` from the helper into an explicitly chosen new local `.gvm` file,
then run from the checkout:

```sh
python -m examples.visual_microphone_context \
  --packet /absolute/window.gvm --python /absolute/numpy-scipy-python \
  --context-interval-ms 100
python -m unittest discover -s tests -p test_visual_microphone_context.py -v
```

The example bounds reads to4MiB, validates the packet, runs a real worker and
polls numeric MCP observations in a separately owned task. It emits a final
query and joins both tasks before closing streams. A failed context lane is an
explicit error. Missed host cadence ticks are skipped without bursts; OS stalls
or blocked stdout can delay delivery. The example uses the normal1h retained
observation query and does not bypass retention for old recordings. It starts no
camera, network MCP server or LLM request. Running it explicitly authorizes local
processing; source consent is still the caller's responsibility.

Tests execute actual reconstruction through packet→owned worker→SensorStreams→MCP,
clock/index/CRC/type/geometry checks, rational2200Hz timing, gap/reset/overlap,
static abstention, original-timestamp preservation and independently polled
Context during delayed worker processing. Existing ProcessProvider cancellation/
timeout/cleanup tests remain applicable. These generated numerical fixtures do
not close [#144](https://github.com/NomaDamas/Gamst/issues/144)'s actual high-speed
camera/reference-microphone, MATLAB parity, quality or performance acceptance.
Software integration is tracked in [#146](https://github.com/NomaDamas/Gamst/issues/146).
