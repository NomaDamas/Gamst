# Optional OC-SORT person tracking

**Status: implemented adapter; real-weight acceptance pending.** OC-SORT itself uses no learned weights; this status means detector-in-the-loop quality and upstream parity have not been accepted. The existing greedy IoU `PersonTracker` remains the default everywhere.

`OCSORTPersonTracker` is a pure-NumPy reimplementation of Cao et al., [*Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*](https://arxiv.org/abs/2203.14360), CVPR 2023. The reference is [OC_SORT revision 8462e7e](https://github.com/noahcao/OC_SORT/tree/8462e7e729a93ccd3bd995c0a79a890336cb3a0b), MIT licensed. No upstream package, SciPy, lap, FilterPy, detector or checkpoint is imported by the adapter.

The implementation includes the seven-state SORT Kalman filter, observation-centric re-update with virtual observations after missed frames, observation-centric momentum in the first assignment, and observation-centric recovery against the last observed box. Assignment uses a deterministic NumPy Hungarian solver. Each `update` is one temporal step, including an empty-detection update.

```python
from gamst.tracking import create_tracker

tracker = create_tracker('ocsort', max_age=30, max_age_ms=1500,
                         delta_t=3, inertia=.2, min_hits=1)
tracks = tracker.update(boxes, timestamp_ms, stream_id='camera-0')
```

The two existing construction sites accept the same explicit selector:

```python
from gamst.plugins import OpenCVPersonPlugin
from gamst.providers.pose import MediaPipePosePlugin

hog = OpenCVPersonPlugin(tracker='ocsort')
pose = MediaPipePosePlugin('/absolute/pose_landmarker.task', tracker='ocsort')
# tracker='iou' is the unchanged default for both.
```

Visible boxes are current normalized detections. Missing boxes are bounded Kalman predictions with `visible=false` and `association_quality='predicted'`; an out-of-image prediction falls back to the last observation rather than emitting invalid geometry. Tracks expire after either `max_age` steps or `max_age_ms`. `min_hits=1` preserves the existing plugin's immediate-output behavior; larger values mark tracks `unconfirmed` until the published hit requirement is met.

IDs remain stream-local and are not recycled after expiry or reset. Ambiguous candidates expose uncertainty, and every track retains `identity_verified=false` and `speaker_association='unavailable'`. OC-SORT recovery after a geometric gap is not biometric re-identification, cross-camera association, or proof that a detector found the same person.

Run the dependency-light synthetic checks with:

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest test_ocsort -v
```

They cover constant motion, missed-frame prediction and reappearance, multiple people, reordered detections, expiry/reset/stream isolation, stale-input atomicity, optimal assignment, and frozen-state re-update. They do not establish MOT benchmark parity or real detector/camera accuracy. Real-sequence parity, crossing/occlusion quality, cadence sensitivity and resource acceptance remain required.
