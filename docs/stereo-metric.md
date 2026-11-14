# Stereo tracks to metric Context

This selected CPU provider combines existing RF-DETR detection/tracking with
OpenCV StereoSGBM and MetricKinematics. It implements the software path in
[#241](https://github.com/NomaDamas/Gamst/issues/241), under #105. The image
pair must come from a declared fixed stereo rig with a shared capture clock and
explicit calibration. No camera is opened and no model is downloaded by discovery.

`stereo_frame(left, right, left_interval_ms=[start,end],
right_interval_ms=[start,end], clock_id=..., alignment_id=...)` in
`humansense.stereo_packet` packages two BGR uint8 InputFrames into immutable
bytes. Both metadata records need source/sensor IDs, capture sequence, source
epoch, original optical coordinate frame and matching calibration ID. Frame
timestamps equal capture interval ends. An interval can be a single timestamp;
this is a caller declaration, not verified exposure timing. The consumer checks
both interval endpoints against its configured maximum skew. Arrival order does
not establish alignment.

Explicitly open ProcessProvider profile `opencv.rfdetr.stereo` using config:

```python
config = {
    "path": "/absolute/local/rfdetr-checkpoint.pth", "sha256": "<64 hex digits>",
    "calibration": {
        "calibration_id": "rig-calibration-1", "width": 640, "height": 480,
        "left_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "right_intrinsics": [[rfx, 0, rcx], [0, rfy, rcy], [0, 0, 1]],
        "left_distortion": [k1, k2, p1, p2, k3],
        "right_distortion": [rk1, rk2, rp1, rp2, rk3],
        "right_from_left_rotation": rotation3x3,
        "right_from_left_translation_m": translation3,
    },
    "left_sensor": {"source_id": "rig", "sensor_id": "left", "coordinate_frame": "left.optical"},
    "right_sensor": {"source_id": "rig", "sensor_id": "right", "coordinate_frame": "right.optical"},
    "clock_id": "capture-clock", "alignment_id": "alignment-1",
    "rectified_frame": "left.rectified", "max_skew_ms": 2,
    "camera_motion": "stationary", "disparities": 64, "block_size": 5,
    "max_gap_ms": 1000,
}
```

All calibration numbers above are placeholders for actual calibrated values.
No default matrix or baseline is inferred. The rotation maps original left-camera
coordinates to original right-camera coordinates; translation is meters in that
transform. Supported geometry must rectify to positive horizontal disparities.
Intrinsics require zero skew; the five distortion coefficients follow OpenCV's
pinhole convention. Moving rigs are unsupported: this profile has no ego-motion
compensation. Updating calibration, image geometry or model requires a new
provider/configuration identity.

The original-left, original-right and rectified coordinate frame names must all
be distinct. Results carry the actual `rectified_from_original_left_rotation`
(OpenCV R1), left/right rectified projection matrices and rectified baseline in
meters. R1 maps original-left 3D coordinates into the reported rectified-left
frame; its transpose reverses that rotation. Projection matrices act on metric
points in the common rectified coordinate system. MetricKinematics' identity
transform is separately a rectified-left to rectified-left transform, not a
claim that the original optical frame was unchanged. Detector labels/class
scores remain explicitly separate from the depth/geometry estimates.

The selected worker calls stereoRectify, initUndistortRectifyMap, remap,
StereoSGBM.compute and reprojectImageTo3D. Fixed-point disparities are divided by
16; invalid/zero disparity, invalid rectification ROIs, failed left/right
consistency and invalid/out-of-bound depth are excluded. Dimensions are bounded
to 96..1280 by 64..720; disparity search is 16..128 in multiples of 16, below half
the image width, with width*height*disparities <=64Mi. These are engineering
allocation bounds, not measured peak-memory or speed guarantees. Explicit
installed OpenCV/runtime provenance accompanies observations.

RF-DETR runs on the same owned original left pixels. Its conservative class/IoU
tracking supports at most eight observed objects. For each detected box the
central half is remapped from original-left to rectified coordinates. At least
16 valid pixels and 50% ROI support are required; depth IQR/median must be <=0.25.
The reported position is the median visible surface, **not center-of-mass**.
These uncalibrated support thresholds cannot guarantee object foreground: boxes
may contain background or multiple depths. Reflective/transparent/textureless
and occluded scenes remain expected failure cases. Low support/ambiguous depth
abstains rather than publishing a fabricated position.

Metric derivatives use actual left capture timestamps. Gaps, source/runtime
epochs, invalid packets and incomplete detections reset histories. Absent or
invalid-depth tracks are omitted from the complete set of valid positions,
preventing velocity across a missing interval. Rectified-left coordinates are
explicitly distinct from original optical coordinates. Positive x/y follow
rectified image right/down and positive z is forward. No force/contact/mass or
world-camera compensation claim is made.

Use SensorStreams with the returned packet InputFrame and the selected provider
spec. Registered `vision.kinematics` values appear in Python Context, API and MCP.
Only scalar summaries, hashes, capture/source metadata and bounded track rows
are published. Original image packets and dense point clouds remain caller-owned;
a hash does not promise retained media. User delivery cadence is independent.
ProcessProvider owns worker timeout/cancel/close; native synchronous inference is
serialized in that worker.

## Recorded replay

Write packet bytes to `.gstereo` files with simple local basenames. A recording
manifest has exact fields `schema_version: 1`, `config` as above and `packets`,
a list of `{file, sha256}` entries in capture order. Freeze the manifest's raw
file SHA256 separately. At most 1024 packets/128MiB are accepted; files must be
bounded local regular files without symlink traversal. All files are preflighted
before worker startup and rechecked when consumed. Replay never changes capture
time to the wall clock.

```sh
python -m humansense.stereo_replay --manifest /absolute/recording/manifest.json \
  --sha256 FROZEN_MANIFEST_SHA256 --python /absolute/selected/python --consent
```

JSONL contains the registered worker result, source packet hash, capture time and
explicit unverified physical accuracy. An interrupted replay may emit a prefix;
it is not a completion/quality certificate. Actual detector checkpoint execution,
calibration reference/error, live capture timing, camera stationarity and held-out
depth/identity accuracy remain required in #105. Tests use generated stereo
images and a labeled fixture detector through the real process protocol; they
prove the OpenCV/stream/Context path, not trained detector or device quality.
