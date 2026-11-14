# Conditional moving-camera trajectories

The opt-in `rfdetr.compensated_trajectories` profile connects actual local RF-DETR
inference, background pixel tracking, conservative object association and bounded
anchor-image trajectories. The owned worker performs inference and retains two
small grayscale images; SensorStreams owns scheduling, cancellation and delivery.
The consumer LLM receives numerical context, not image arrays. This is a
conditional 2D image estimate, not metric camera pose or verified object identity.

```python
from humansense import ProcessProvider

config = {
    'path': '/absolute/local/rfdetr-checkpoint.pth',
    'sha256': checkpoint_sha256,
    'background_roi': [0, 0, 1, 1],
    'operating_assumption': 'approximately-similarity-static-background',
    'device': 'cpu',
    'max_gap_ms': 1000,
}
worker = await ProcessProvider.open(
    existing_worker_python, 'rfdetr.compensated_trajectories', config)
```

Use the worker registration/open/submit lifecycle from
[RF-DETR trajectories](rfdetr-trajectories.md), with this profile/spec. Explicit
standalone `CompensatedTrajectories.from_local(config)` is synchronous and
caller-owned. Default SDK operation starts neither model nor camera. The worker
requires already installed OpenCV/NumPy and the existing RF-DETR1.12.1 checkpoint
runtime; it downloads/installs nothing. Discovery imports no optional runtime.

Input is copied plain uint8 HWC BGR, dimensions16..4096, at most16MiB, with explicit
source, sensor, image coordinate frame, epoch and original capture timestamp.
When sending larger frames, explicitly configure SensorStreams packet and queue
budgets as described in the existing trajectory guide. The unchanged detector
threshold/association policy returns at most8 objects. Partial detection snapshots
still expose raw detections but do not produce compensated paths. Source/epoch,
subject, calibration, channels, dimensions or sequence-kind changes reset state.
Skipped sequences, time gaps, invalid input, reset, clear, pause, delete and close
also invalidate retained state. Duplicate/regressing time or sequence rejects.

## What is estimated

The fixed caller-selected background rectangle declares an approximately static
background and similarity-compatible view. It must span at least half of each
image dimension and30% of image area. This declaration is not verified. Current
and previous foreground detection rectangles are excluded, padded by5% of each
box extent plus3 analysis pixels. Undetected moving objects can contaminate the
fit; RANSAC does not prove that matched features belong to a static background.

Grayscale analysis uses long side<=640 and at most256 sparse corners. Actual
Lucas–Kanade tracking uses21×21 windows, pyramid maxLevel3 and30 iterations;
forward/backward disagreement above1 analysis pixel rejects a correspondence.
Actual `estimateAffinePartial2D` uses bounded RANSAC1000 iterations, confidence.99,
2px reprojection and10 refinement iterations. At least24 inliers,75% inlier ratio,
6 occupied4×4 tiles and25% image hull coverage are required in both images.
Median/p95 residuals must be<=1/2 analysis pixels. Pairwise scale must be.95..1.05,
rotation<=5degrees, translation<=10% diagonal and overlap>=70%. These are fixed
engineering admission policies, not calibrated accuracy or probability bounds.
[OpenCV tracking](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html) and
[transform estimation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
describe the original algorithms used here.

The fit is in analysis pixels, never unequal normalized x/y coordinates. Exact
resize ratios conjugate it back to original pixels. `G_t` maps previous pixels
to current pixels. Temporary projected prior boxes use all four corners before
class-constrained IoU association. Boxes outside the image or otherwise invalid
are excluded from association, not clipped into invented measurements. Original
measured boxes remain in `detections`; missing tracks retire on each complete
frame, and ambiguous associations start new track/trajectory segments.

`C_t = C_(t-1) @ inverse(G_t)` maps current pixels to a named anchor image. Each
history sample retains its raw centroid, capture time/sequence, evidence hash,
separate estimated anchor projection. Up to8 `geometry.projection_records` store
each frame's actual mapping once. Resolve a sample within the same output and
anchor by exact capture sequence, timestamp and evidence SHA256; the match must
be unique. Keep the complete snapshot when forwarding/persisting histories.
Original anchor capture provenance remains even after its sample leaves the
8-frame history.
Anchor coordinates are normalized by the anchor image width/height and may lie
outside[0,1]. Values outside[-4,5] invalidate the reference rather than being
clipped. These units are not meters. Velocity uses actual capture intervals and
only samples in the same reference and trajectory segment. No interpolation,
future position, physical identity or 3D speed is produced.

Reference composition is bounded to60 links and2seconds, cumulative scale.8..1.25,
rotation<=15degrees and overlap>=50%. Crossing a bound, invalid fit, insufficient
background, texture/support loss or detected discontinuity starts a new anchor
and segments. Warmup/unavailable output has raw detections and no compensated
objects; the internal anchor sample may appear as history after a subsequent
valid fit. Unsupported parallax, rolling shutter, repetitive texture or unnoticed
cuts can still produce a wrong accepted estimate. `camera_motion_verified=false`
and the explicit operating assumption always remain in the output.

## Context and memory

Output contains `detections`, `objects` and `geometry`. Each compensated object
has both `track_id` and `segment_id`, plus `anchor_id`; retain all three when
connecting memory. Do not join separate segments merely because a detector track
ID appears similar. Current versus historical capture times remain explicit.
Source evidence SHA256 covers the domain
`gamst.compensated.trajectory.uint8.HWC.BGR.v1`, a NUL byte, three big-endian32-bit
shape dimensions and original copied BGR bytes. Hash integrity is not accuracy.
Evidence storage and later image retrieval belong to the caller.

`geometry` exposes transform direction, current/previous capture provenance,
anchor identity, fit diagnostics, OpenCV/detector/config identity and separate
complete-snapshot status. Retention is at most8tracks×8observed samples. Context
contains no raw pixels/features. User delivery cadence is unchanged. Existing
fixed-image ROI/line rules are deliberately unavailable in this new profile:
after re-anchoring those same numbers would no longer describe the same court
region. Future rules must explicitly bind/rebind their reference; the original
stationary image-event provider remains available.

Tests run real OpenCV on rendered images and actual owned worker/Context/MCP
transport with an explicitly synthetic detector. They cover transform direction,
resize, independent object motion, cut/support failures, original preservation,
segment boundaries, bounded output, cancellation and clear. They do not establish
trained sports tracking accuracy, realtime device performance or VLM cost savings.
#260 is this software path; #134 retains calibrated court mapping, reference-aware
task packs, fast small-object/occlusion work and real held-out evaluation.
