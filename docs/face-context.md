# Face movement context

`MediaPipeFaceContext` publishes bounded `vision.face_movements` observations through
SensorStreams and Python/MCP. The `mediapipe.face` ProcessProvider profile loads an
explicit local Face Landmarker task in CPU IMAGE mode. Existing legacy face plugin
behavior is unchanged; this adapter exposes derived context instead of raw points.

```python
from humansense import ProcessProvider
config = {'path': '/absolute/face_landmarker.task',
          'sha256': checkpoint_sha256, 'max_faces': 4}
spec = ProcessProvider.capability('mediapipe.face', config)
provider = await ProcessProvider.open('/absolute/runtime/bin/python', 'mediapipe.face', config)
# Register spec and an isolated factory with SensorStreams; open capture explicitly.
```

Only mediapipe==0.10.35 is accepted. The loader verifies a nonempty absolute local
file no larger than 64 MiB before optional imports, then passes those exact bytes
to BaseOptions. There is no implicit model download. Select a trusted licensed
checkpoint; a digest proves integrity, not authenticity. The descriptor includes
runtime pin, hash and maximum faces (1..4); weight license remains unspecified.

Input is uint8 BGR H×W×3, at most 2,073,600 pixels and 3840×2160 dimensions. An owned
RGB copy goes to the detector. Each detected face must have 478 finite bounded
coordinates and an associated coefficient list. Invalid counts or associations fail
the lane rather than becoming an empty scene. Zero detections is a valid empty list.

Output per face contains a frame-local index, normalized bounding box over at least
three in-image points with positive extent, and three pixel-plane ratios:
- Left eye vertical gap (386,374) divided by corner distance (263,362).
- Right eye vertical gap (159,145) divided by corner distance (33,133).
- Inner mouth gap (13,14) divided by mouth corner distance (61,291).

Distances use x×width and y×height, preserving image aspect ratio. Off-image endpoints
or denominators below one pixel yield null. These are simple projected ratios, not
calibrated eyelid openness, a standardized multi-point EAR, fatigue or blink events.
A box over the in-image subset need not cover the entire face. Left/right follows
the model's anatomical topology; input mirror status remains unspecified.

Up to 52 unique named blendshape coefficients are retained without aggregation or
thresholding, each finite in [0,1]. They are expression-model coefficients, not
emotion probabilities, measured muscle activation or joint confidence. The empty
list is explicitly `blendshapes_available=false`; no absent coefficient is invented
as zero. `eyeLook*` values are not calibrated gaze direction or attention. Consumers
must not interpret `mouthSmile*` as happiness or infer identity/intent from scores.
No raw landmarks, facial matrices, metric 3D, gaze, emotion or temporal tracking are
claimed. Per-joint confidence is unavailable.

This is repeated IMAGE inference, not stateful streaming model execution.
SensorStreams supplies capture provenance and bounded queues. ProcessProvider adds
worker isolation/timeouts; in-process native calls cannot be forcibly interrupted.
No actual FPS, latency, checkpoint accuracy or quality-preserving VLM savings have
been established by these adapter tests.

## Sources and validation

The official [MediaPipe v0.10.35 Face Landmarker source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/face_landmarker.py)
defines coefficient categories, topology connections, result association and IMAGE
options. The [Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python)
describes 478 points and 52 blendshape scores. Those interfaces do not establish
performance or prediction quality in gamst.

`python3 -m unittest discover -s tests -p test_face_context.py -v` tests aspect-ratio
math, scale invariance, RGB ownership, off-image/subpixel abstention, coefficient
bounds and missing data, exact loader bytes/options and managed publication/MCP
retrieval. Tests use explicit detector/loader doubles and install no model weights.
Family #20 remains open. Actual checkpoint replay/device evidence is tracked in
[#85](https://github.com/NomaDamas/Gamst/issues/85); gaze and stateful temporal
capabilities also remain required.
