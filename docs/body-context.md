# Typed body geometry context

`MediaPipeBodyContext` runs an explicitly selected local MediaPipe Pose Landmarker
Tasks model in IMAGE mode. It publishes registered `vision.body_geometry` observations
for SensorStreams, Python/MCP queries and context consumers. `mediapipe.body` is the
matching ProcessProvider profile. It reuses the existing pose plugin's bounding-box
and wrist comparison helpers while leaving that legacy plugin compatible.

```python
from humansense import ProcessProvider
config = {'path': '/absolute/pose_landmarker_lite.task',
          'sha256': checkpoint_sha256, 'max_poses': 4}
spec = ProcessProvider.capability('mediapipe.body', config)
provider = await ProcessProvider.open('/absolute/runtime/bin/python', 'mediapipe.body', config)
# Register spec and an isolated factory with SensorStreams; open capture explicitly.
```

The source-audited runtime is mediapipe==0.10.35. The loader rejects relative paths,
checks a nonempty model at most 64 MiB against the caller's SHA256, and passes those
exact verified bytes to BaseOptions.model_asset_buffer with a CPU delegate. It
never accepts a model identifier as a download instruction. The digest is integrity,
not model authenticity; choose a trusted licensed task file. SDK code is Apache-2.0;
weight license is explicitly unspecified in the contract until verified for the
selected artifact. The version includes runtime pin, task hash and maximum poses.

Input is an uint8 BGR H×W×3 array, at most 2,073,600 pixels and dimensions no greater
than 3840×2160. A copied RGB image is passed to MediaPipe. Output is capped at 1..4
poses, each requiring exactly 33 finite bounded points and valid optional visibility/
presence scores. Missing scores count as zero. Invalid native output fails the lane
rather than silently claiming an empty scene. Empty detections are a valid empty list.

Per-frame output includes a frame-local pose index, visibility-gated normalized
image bounding box, model-score-gated joint count, wrist-above-shoulder comparisons,
and left/right elbow/knee projected angles. Angles use pixel-plane geometry (x×width,
y×height), not angles in a distorted normalized square. Low-score, off-image or
degenerate joint triples produce None. No raw image, 33-point array or body world
coordinates are emitted. Bounding boxes use the existing helper: at least three
score-qualified points, clamped to the image boundary. Mirror state is unspecified;
left/right labels are the model's anatomical labels, not a viewer-side claim.

This is frame-mode inference, with no temporal state/smoothing/tracking, no person
identity, no hand/facial-detail/gaze model, no calibrated 3D angle, physical force,
medical assessment or inferred intention. SensorStreams supplies capture provenance,
bounded queues and cancellation/clear suppression. In-process native calls cannot be
forcibly interrupted; the owned worker supplies an inference timeout. Actual FPS and
capture-to-context latency have not been established for this adapter.

## Sources and validation

API behavior, model options and 33-point topology were checked against the official
[MediaPipe v0.10.35 Pose Landmarker source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/vision/pose_landmarker.py)
and [BaseOptions source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/python/core/base_options.py).
These establish interface expectations, not gamst runtime performance.

`python3 -m unittest discover -s tests -p test_body_context.py -v` verifies non-square
image angle math, RGB copying, empty/low-score/off-image/degenerate cases, malformed
inputs/results, bounded outputs, pre-import integrity checks, exact loader buffer
and options, and managed publication/MCP retrieval. Detector and loader doubles are
explicit fixtures, not actual pose inference. No weights were installed for these tests.

Family #20 still requires actual multi-person/occlusion/no-person/moving-camera
validation with the selected checkpoint, source fixture provenance, pinned runtime,
device, accuracy and p50/p95/resource evidence, plus hands, face, gaze and temporal
capabilities in the approved plan. This provider does not close that family or the
full expansion goal.
