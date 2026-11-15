# Explicit camera/model head orientation

`mediapipe.head_orientation.video` combines bounded native Face Landmarker VIDEO
landmarks with OpenCV `solvePnP(SOLVEPNP_ITERATIVE)`. It publishes
`vision.head_orientation` through ProcessProvider, SensorStreams and MCP. This is
a rotation estimate relative to caller-supplied 3D face geometry and camera
calibration. No focal length, standard face size, anthropometry or anatomical
axes are guessed. Supplying unrelated 3D points makes the estimate unrelated to
head orientation; the adapter cannot authenticate the caller's model geometry.

## Required configuration

```python
from gamst import ProcessProvider

config = dict(
    path='/absolute/face_landmarker.task', sha256=task_sha256,
    camera=dict(width=capture_width, height=capture_height,
                fx=calibrated_fx_px, fy=calibrated_fy_px,
                cx=calibrated_cx_px, cy=calibrated_cy_px,
                distortion=[k1, k2, p1, p2, k3], mirrored=False),
    sensor=dict(source_id='camera', sensor_id='sensor',
                calibration_id='camera.calibration.v1',
                coordinate_frame='camera.optical'),
    model_frame='declared.face.model',
    model_points=explicit_correspondences,
    max_reprojection_error_px=1.0, max_faces=1, max_gap_ms=1000,
)
# explicit_correspondences contains 6..32 entries of exactly:
# {'landmark_index': integer_0_to_477, 'xyz_m': [x_m, y_m, z_m]}
spec = ProcessProvider.capability('mediapipe.head_orientation.video', config)
provider = await ProcessProvider.open('/absolute/runtime/bin/python',
    'mediapipe.head_orientation.video', config)
```

The exact configuration is limited to 16 KiB and enters the capability version
hash. Every correspondence must be unique, finite, noncoplanar and nondegenerate;
coordinates are bounded to ±1 metre. The input must use the declared optical frame
and unmirrored pixel geometry. Mirroring, cropping, rotating or resizing after
calibration requires appropriately recomputed camera parameters and a new
calibration binding; the provider performs none of those transformations. A
`mirrored=False` declaration is mandatory but is not visually verified.

Image dimensions are 32..3840 by 32..2160 with at most 1920×1080 pixels. Focal
lengths are 1..100000 pixels, principal points lie inside the image, and exactly
five OpenCV distortion coefficients are bounded to ±2. The caller chooses a
reprojection threshold in .01..20 pixels; it is not an accuracy guarantee.
`max_faces` is 1..4 and `max_gap_ms` is 1..60000. The model and camera frame names
must differ. Each InputFrame's source, sensor, calibration and coordinate frame
must exactly match configuration, as must its uint8 BGR image dimensions.

## Direction, units and abstention

For each frame-local face, `orientation` contains `status`, `reason`,
`rotation_model_to_camera` (3×3), `rodrigues_model_to_camera_rad` (3), and
`reprojection_rmse_px`. The transform is
`X_camera = R_model_to_camera * X_model + t`, with camera x right, y down and z
forward. Rodrigues values describe an axis-angle rotation vector in radians,
not Euler yaw/pitch/roll. Model axes remain caller-defined; no anatomical axis or
left/right convention is inferred. Translation is used internally for perspective
projection and positive-depth checks and is never published. No gaze, attention,
intent, identity, expression meaning or calibrated metric location is inferred.

Off-image selected landmarks, image correspondences below the spatial/rank
threshold, failed/nonfinite solves, improper rotation, nonpositive camera depth
and excessive reprojection error return `status=unavailable` and null rotation.
Malformed native results reject and reset the frame. Empty detections remain an
empty face list. Multiple faces share the same configured reference geometry;
indices do not persist as identities. Reprojection consistency alone cannot
establish correct anatomy, camera calibration, true pose or ambiguity resolution.
Output explicitly marks calibration, model geometry and pose accuracy unverified.

## Runtime, ownership and evidence

Use an explicitly provisioned runtime with `mediapipe==0.10.35`, NumPy and OpenCV
(the optional `mediapipe` and `vision` dependency families). The loader requires
an absolute local Face Landmarker task of at most 64 MiB and verifies SHA256
before importing MediaPipe. It retains the verified bytes for CPU VIDEO resets.
No dependency, task or model is downloaded. MediaPipe code is Apache-2.0; OpenCV
license depends on the selected release (supported releases declare Apache-2.0).
Selected face weights and caller-supplied geometry/data require separate terms.

The [VIDEO lifecycle](landmarker-video.md) owns monotonic native timestamps,
bounded frame processing, segment resets and native resources. Processing, reset
and close are serialized. Native results are retained only during formatting and
then cleared. Worker isolation supplies the timeout boundary. Native memory,
sustained FPS and runtime stability are not established by these tests.

```sh
PYTHONPATH=tests:. python3 -m unittest test_head_orientation test_landmarker_video test_face_context -v
```

Tests execute real OpenCV projection/PnP on explicitly synthetic cuboid geometry
and recover a known model-to-camera rotation. They also test abstention,
calibration mismatch, continuity and a real owned worker/MCP path with only the
native MediaPipe boundary replaced. No human model, device, trained task or
accuracy experiment ran. Real camera calibration, topology conventions, known
orientation replay, ambiguity/error analysis and device resources remain deferred
under [#85](https://github.com/NomaDamas/Gamst/issues/85) and
[#20](https://github.com/NomaDamas/Gamst/issues/20). The separate L2CS gaze gate
[#93](https://github.com/NomaDamas/Gamst/issues/93) cannot establish head-pose
acceptance. This software path does not close those real-evidence gates.
