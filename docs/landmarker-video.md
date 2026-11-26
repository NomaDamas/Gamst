# Stateful body, hand and face video context

`MediaPipeVideoContext` adds native VIDEO-mode processing to the existing bounded
body, hand and face context formats. Each lane retains one native detector between
contiguous frames and calls `detect_for_video`, allowing the runtime to reuse its
tracking state. It does not assign persistent person/hand/face identities. Capture
and context delivery are still owned by SensorStreams and the consuming application.

```python
from gamst import ProcessProvider
config = {'path': '/absolute/face_landmarker.task', 'sha256': checkpoint_sha256,
          'max_entities': 1, 'max_gap_ms': 1000}
spec = ProcessProvider.capability('mediapipe.face.video', config)
provider = await ProcessProvider.open('/absolute/runtime/bin/python', 'mediapipe.face.video', config)
# Register spec/factory with SensorStreams, then explicitly open and submit capture.
```

Profiles: `mediapipe.body.video`, `mediapipe.hand.video`, `mediapipe.face.video`.
The separate [gesture profile](gesture-context.md), `mediapipe.gesture.video`,
reuses this lifecycle with `GestureRecognizer.recognize_for_video` and a distinct
`vision.hand_gestures` output; it does not reinterpret hand geometry as gestures.
The separate [head-orientation profile](head-orientation.md),
`mediapipe.head_orientation.video`, adds explicit camera/3D-model PnP geometry to
the face VIDEO lifecycle without assuming default camera or anatomical geometry.
`max_entities` is 1..4 (default 4); `max_gap_ms` is 1..60000 (default 1000).
Discovery is dependency-light and starts no model. The descriptor is `native`,
stateful and reset-on-gap; its ID differs from the IMAGE-mode provider. It retains
the same output name, so select the desired IMAGE or VIDEO provider per stream
rather than treating their outputs as different sensed quantities.

## Native state and continuity

Input must carry SensorMetadata, a subject identifier, and an integer capture
millisecond timestamp in 0..2**53-1. Within the same owner, both timestamp and capture
sequence must strictly increase. The native clock starts at zero for each state
segment and advances by actual capture time; publication timestamps remain original
capture timestamps. No timestamp is synthesized to hide repeated/out-of-order input.

A sequence gap (unless `allow_skips` is set, below), gap longer than `max_gap_ms`, changed image dimensions, subject,
source/sensor, epoch, coordinate frame, calibration ID or channel layout closes the
old native detector. The next valid image creates a fresh detector. Explicit reset,
invalid metadata/image, malformed native output and inference exceptions also clear
state. Backward/repeated capture sequence or timestamp raises and resets; it is not
published. Managed lanes additionally enforce their own input sequence contract.

`allow_skips` (default false: the contract above, unchanged) is for live capture where
frames can be lost before this lane. MediaPipe VIDEO mode requires strictly increasing
timestamps, not every frame, and the caller passes real capture times, so a skipped frame is
the same to it as a lower frame rate. Rebuilding the native detector for each skip is
what is expensive: with 30 fps capture and a one-frame queue it left about 2 pose results
a second (measured, docs/handoff-feat-realtime-pipeline.md). With `allow_skips: true` a
sequence jump within `max_gap_ms` keeps the detector; every other boundary is the same: a
longer gap, or a change of image dimensions, subject, source/sensor, epoch, coordinate frame,
calibration ID or channel layout, still closes it, and backward or repeated sequence or
timestamp still raises. The capability version ends in `.skips`, so a record says which
contract produced it. A `state_segment` may then span skipped frames: it means one native
tracking state, not gap-free capture. The recipe part `mediapipe-pose` sets it; what was lost
is counted and reported by the Pipeline, not by this provider.

Output summary adds `running_mode=VIDEO`, `temporal_tracking=true`, an opaque
`state_segment` token, `native_timestamp_ms`, `persistent_track_ids=unavailable` and
`smoothing=runtime_default`. Tracking here means the native runtime may reuse regions
between frames, not stable entity IDs. Existing frame-local indices cannot be
joined across frames as identities. Runtime default smoothing is not a configurable
SDK filter or a measured quality improvement; face smoothing is enabled by the
runtime only for its supported single-face configuration. Gaze, emotion, metric 3D
and contact limitations of the IMAGE adapters remain unchanged.

## Resource ownership

The loader verifies an absolute, nonempty local task file of at most 64 MiB against
the supplied hash, pins mediapipe==0.10.35, and keeps the verified bytes for resets.
Every native construction uses those same bytes with CPU and VIDEO options; replacing
the file afterward cannot change the loaded model. There is no model download.
The first valid image initializes native resources; startup validates the file and
runtime but cannot prove task compatibility before that first inference.

One native instance and one verified byte buffer are retained per provider. No frame
history is accumulated by the SDK. Input/image/result limits and geometry conversion
are shared with [body](body-context.md), [hand](hand-context.md) and
[face](face-context.md) adapters. Native internal memory is not measured or bounded
by these Python tests; use an explicit worker and evaluate its actual resources.
Process, reset and close serialize behind a lock. Close waits for in-process native
work; only worker isolation supplies a timeout/termination boundary. A native close
failure disables the provider instead of reusing uncertain resources. Do not share
a provider instance between managed lanes; each lane owns its factory result.

## Evidence and acceptance

Source-audited API: official pinned
[Pose Landmarker](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/pose_landmarker.py),
[Hand Landmarker](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/hand_landmarker.py), and
[Face Landmarker](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/face_landmarker.py).
VIDEO calls are synchronous per frame, with retained native state. This is not the
asynchronous LIVE_STREAM callback API or a demonstrated latency/FPS improvement.

`python3 -m unittest discover -s tests -p test_landmarker_video.py -v` uses explicit
native doubles to verify all three VIDEO option sets, verified-buffer reuse,
continuity boundaries, state disposal on errors, dependency-light discovery and
managed/MCP capture provenance. No weights are installed. Actual native quality,
smoothing lag, resource lifetime, sustained throughput and reset behavior must be
validated on chosen checkpoints/devices in [#87](https://github.com/NomaDamas/Gamst/issues/87)
before family #20 is complete.
