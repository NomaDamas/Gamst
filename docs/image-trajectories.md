# Observed image trajectories

```mermaid
flowchart LR
  A[Caller: detection and association] --> B[ImageTrajectories: bounded observed paths]
  B --> C[gamst Context store]
  C --> D[MCP or consumer LLM]
```

`humansense.providers.trajectories.ImageTrajectories` is a standard-library local
structured-input processor. It consumes complete caller-associated snapshots,
retains up to eight observed samples for each of up to eight currently observed
tracks, and computes normalized image-plane interval velocity. It does not load
models or perform detection, association, prediction or camera-motion compensation.
It is not a pixel ProcessProvider/SensorStreams transport adapter.

```python
from humansense.providers.trajectories import ImageTrajectories
from humansense.plugins import InputFrame
from humansense import SensorMetadata

provider = ImageTrajectories(max_gap_ms=1000, max_samples=8)
sensor = SensorMetadata('camera', 'rgb', 0, 'image', stream_epoch='capture-1')
frame = InputFrame({
    'units': 'normalized_image', 'image_size': [1920, 1080],
    'tracker_id': 'my-tracker', 'tracker_version': 'v1',
    'camera_motion': 'stationary',
    'objects': [{'track_id': 'ball', 'centroid': [0.25, 0.5],
                 'association_confidence': None, 'evidence_sha256': evidence_sha256}],
}, 'video', capture_timestamp_ms, sensor=sensor)
human.register_capability(provider.spec)
await human.push_observation(provider.spec.id, provider.process(frame),
    sensor=sensor, timestamp_ms=frame.timestamp_ms, subject_id=frame.subject_id)
```

The application supplies real capture timestamps and a lowercase SHA256 reference
to original retained evidence. The provider validates hash syntax, not content or
existence; evidence storage, expiry and retrieval belong to the application.
Coordinates are x right/y down, normalized independently by image width/height,
within [0,1]. Input is strict JSON <=16KiB, with exactly the illustrated fields.
Image size is [width,height], each 1..16384. Confidence is caller-declared null or
0..1; it is not calibrated by this processor. All samples are tagged `observed`;
association is not verified person identity. No interpolated or future locations
are emitted. Velocity is `(last centroid - previous centroid) / capture seconds`,
with exact interval timestamps, in image units/second, never meters/second.

Each track segment has a unique ID. Source, subject, sensor, epoch, image size,
coordinate frame, calibration, channels or tracker changes start new segments.
Increasing capture sequence (when supplied, otherwise sensor sequence) and capture
time are required within a source configuration. Skipped sequence or intervals
above max_gap_ms start new segments. `camera_motion` must be stationary, moving,
cut or unknown. Only consecutive stationary frames continue history; all other
states reset it, including transition back to stationary. This is conservative:
undetected camera movement or tracker ID switches remain upstream errors.

Missing tracks in a complete snapshot are dropped immediately; reappearance
starts a new segment. Stored samples are historical observations, not claims of
current visibility after their capture time. Longer off-screen memory is a
separate store capability. Invalid input clears history before raising. `reset()`
clears it; idempotent `close()` clears and permanently stops processing. The
application owns this local processor's lifecycle and must reset/close it alongside
host consent/reset/delete/close. Configuration is fixed at construction; max_gap_ms
is 1..10000 and max_samples 2..8. Output and internal histories are bounded and
copied; returned objects cannot mutate subsequent state.

Software tests cover unequal intervals, provenance boundaries, missing tracks,
camera motion, bounds, lifecycle and actual Context/MCP publication. #170 is this
software slice; #134 remains open for tracker integration, sports event packs,
calibrated mapping, occlusion handling and real annotated clips. Synthetic geometry
does not establish sports accuracy, realtime hardware performance, or VLM savings.
