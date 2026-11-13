# Hand geometry context

`MediaPipeHandContext` supplies bounded `vision.hand_geometry` observations to
SensorStreams and Python/MCP context consumers. The `mediapipe.hand` ProcessProvider
profile runs an explicitly chosen local Hand Landmarker task in CPU IMAGE mode.

```python
from humansense import ProcessProvider
config = {'path': '/absolute/hand_landmarker.task',
          'sha256': checkpoint_sha256, 'max_hands': 4}
spec = ProcessProvider.capability('mediapipe.hand', config)
provider = await ProcessProvider.open('/absolute/runtime/bin/python', 'mediapipe.hand', config)
# Register spec and an isolated factory with SensorStreams; open capture explicitly.
```

The loader pins mediapipe==0.10.35, verifies a nonempty absolute local task file
of at most 64 MiB against the supplied SHA256 before optional imports, and passes
those exact bytes to BaseOptions. It downloads nothing. A hash proves integrity,
not authenticity: select a trusted, licensed checkpoint. The capability version
includes the runtime pin, task hash and maximum hand count (1..4). Weight license
remains unspecified until verified for the selected artifact.

Input is uint8 BGR H×W×3, at most 2,073,600 pixels and 3840×2160 dimensions.
An owned RGB copy goes to the detector. Native output must contain exactly 21 finite
bounded coordinates per hand and equally many handedness lists. Invalid output
fails the lane; an empty detection list is valid. Handedness candidates contain
Left/Right and classification scores. These are not joint location confidence;
`joint_confidence` is explicitly unavailable. Mirror status is unspecified.

Each hand has a frame-local index, a normalized image bounding box over at least
three in-image points with positive extent, normalized index fingertip coordinates,
five projected finger-joint angles (degrees), and thumb-to-index fingertip distance
divided by index-to-pinky MCP distance. Distances and angles use the pixel plane
(x×width, y×height), accounting for non-square images. Off-image endpoints,
degenerate angle vectors, or projected palm width below one pixel produce null.
The ratio describes projected aperture, not physical contact or a pinch decision.
A box over the in-image subset need not cover the entire hand.

No raw landmarks/images, metric 3D measurements, temporal tracking, persistent
identity, gesture classification or inferred intent are emitted. IMAGE mode reruns
inference for each submitted frame; it is not stateful input-stream inference.
SensorStreams provides provenance and bounded queues. ProcessProvider supplies
worker isolation/timeouts; in-process native calls cannot be forcibly interrupted.
Actual FPS and capture-to-context latency remain unverified.

## Evidence and remaining acceptance

The official [MediaPipe v0.10.35 Hand Landmarker source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/hand_landmarker.py)
defines the result association, 21-point topology and IMAGE-mode options. The
[BaseOptions source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/core/base_options.py)
defines the verified buffer and CPU delegate interface.

`python3 -m unittest discover -s tests -p test_hand_context.py -v` checks projected
geometry, scale invariance, RGB ownership, abstention, malformed associations,
bounded native output, loader integrity/options, managed publication and MCP reads.
These use explicit detector/loader doubles; no weights were installed and no actual
checkpoint accuracy or throughput is claimed. Family #20 remains open for actual
hand replay/device evidence ([#83](https://github.com/NomaDamas/Gamst/issues/83)),
face/gaze and temporal capabilities.
