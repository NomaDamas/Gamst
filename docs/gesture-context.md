# Local MediaPipe gesture candidates

`mediapipe.gesture.video` selects the concrete MediaPipe 0.10.35
`GestureRecognizer` task in VIDEO mode. It emits bounded `vision.hand_gestures`
Context through the existing ProcessProvider, SensorStreams and MCP interfaces.
The hand-landmark geometry provider remains separate: neither its angles nor
pinch ratios are relabeled as trained gesture classification.

The adapter accepts only the eight reference task labels: `None`, `Closed_Fist`,
`Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.
These strings are unverified model categories, not intent, identity, a command,
sign-language interpretation or permission to act. `None` and an empty/below-.5
score list produce `gesture_status=unknown`; they do not prove no gesture occurred.
Scores at or above .5 are retained in descending order. The threshold is a fixed
uncalibrated filtering rule, not an accuracy guarantee. Unknown/duplicate labels,
nonfinite/out-of-range scores and inconsistent hand/result counts reject the frame.

## Explicit local runtime

Use an already provisioned Python environment with `mediapipe==0.10.35` and NumPy
(the existing `mediapipe`/`vision` extras describe the optional dependency family).
Supply the official compatible Gesture Recognizer `.task` bundle as an absolute
local path plus its independently checked SHA256. The loader reads at most64MiB,
verifies the digest before importing MediaPipe and retains those exact bytes for
reset. No task download, URL, model-ID lookup, package installation or device
capture is performed. A hash identifies supplied bytes; it does not authenticate
training or grant a license. MediaPipe code is Apache-2.0; selected task weights
and training data terms require separate verification.

```python
from gamst import ProcessProvider

config = dict(path='/absolute/gesture_recognizer.task', sha256=task_sha256,
              max_entities=2, max_gap_ms=1000)
spec = ProcessProvider.capability('mediapipe.gesture.video', config)
provider = await ProcessProvider.open('/absolute/runtime/bin/python',
    'mediapipe.gesture.video', config)
# Register a fresh provider factory with SensorStreams, open the selected lane,
# submit explicit capture InputFrame objects, and close the owner in finally.
```

`max_entities` is1..4; `max_gap_ms` is1..60000. Configuration, model digest and
fixed label/threshold policy enter the capability version. Native options use CPU,
VIDEO, explicit .5 hand-detection/presence/tracking thresholds and at most8
classifier results per hand. Only the supported reference labels enter Context;
custom classes are unsupported. The first valid frame constructs the recognizer
and calls `recognize_for_video`, so loader success alone is not successful inference.

## Capture, continuity and output

The [existing VIDEO lifecycle](landmarker-video.md) owns timestamps, source/subject
binding, state segmentation and serialization. Native timestamps are relative to
each new segment; observation timestamps preserve actual capture time. Changed
source, epoch, calibration, channel layout or image dimensions, sequence/time
gaps and explicit reset close the old recognizer. Repeated/backward timestamps,
malformed frames/results and native errors reject and reset. State is recreated
only for a subsequent valid frame. A close failure disables the provider.

Input is uint8 BGR, at most1920×1080 pixels, bounded by3840×2160 dimensions,
with SensorMetadata and monotonic capture sequence/timestamps. BGR is copied to
RGB. Each hand has a frame-local index, handedness scores, compact projected
geometry and `gesture_candidates`; indices are not persistent identities.
No images or landmark arrays enter Context. Raw native results are discarded
after formatting. There is no gesture smoothing, dwell/action threshold or
automatic host interruption. Native tracking may reuse state, but no fixed FPS
or classification quality is promised. User delivery cadence remains independent
of capture and inference; the bounded lane reports overload/reset instead of
inventing observations. Only process isolation supplies a hard timeout boundary
for blocked native work.

## Software evidence and deferred acceptance

API source: official pinned
[GestureRecognizer](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/gesture_recognizer.py).
The integration uses its synchronous VIDEO API and explicit verified model buffer,
not LIVE_STREAM callbacks or a generic classification callback.

```sh
PYTHONPATH=tests:. python3 -m unittest test_gesture_context test_landmarker_video test_hand_context -v
```

Tests replace only the native model boundary, including inside a real owned
worker running the production constructor/wire protocol. They verify exact local
bytes/options, label filtering and abstention, bounded results, capture continuity,
malformed output rejection, reset/close and SDK/MCP provenance. No trained task,
sensor or package was installed for this work. Actual official-task compatibility,
handedness/mirroring, held-out gesture accuracy, confusions and sustained resource
and timing evidence remain [#20](https://github.com/NomaDamas/Gamst/issues/20) and
[#83](https://github.com/NomaDamas/Gamst/issues/83), deferred by the user's
2026-11-15 instruction. These tests do not close full model/device acceptance.
