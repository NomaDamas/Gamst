# Monocular camera-pose Context

`orbslam3.monocular` is an explicit ProcessProvider profile. It loads the
[local native bridge](orbslam-native.md) only in its owned child process, converts
calibrated plain numpy uint8 BGR frames using OpenCV BGR2GRAY without resizing, and publishes
`vision.camera_pose`. Discovery does not open assets or import numpy/OpenCV. The shared worker wire
rejects masked arrays and ndarray subclasses before serialization, so mask or
subclass semantics cannot silently turn into ordinary pixel observations.
The selected runtime needs numpy/OpenCV and a prebuilt native library with all
its dependencies; no model or dependency downloads happen during provider startup.

```python
import sys
from humansense import ProcessProvider, SensorStreams

config = dict(
    library_path='/absolute/path/libgamst_orb_bridge.so',
    vocabulary_path='/absolute/path/ORBvoc.txt',
    settings_path='/absolute/path/camera.yaml',
    library_sha256=library_digest,
    vocabulary_sha256=vocabulary_digest,
    settings_sha256=settings_digest,
    calibration_id='camera-calibration-1',
    max_frames=300, max_gap_ms=1000, sequence_step=1,
)
worker = await ProcessProvider.open(sys.executable, 'orbslam3.monocular', config)
# Existing consent-enabled HumanSense; enough packet space for 1920x1080 BGR.
streams = SensorStreams(human, max_packet_bytes=8*1024*1024)
streams.register(worker.spec, lambda: worker)
await streams.open('camera', 'video', [worker.spec.id])
# Submit InputFrames with matching source/sensor IDs, calibration_id,
# declared camera coordinate_frame and the calibrated image dimensions.
# Leave stream_epoch unset for SensorStreams to assign its owned epoch.
# Always await streams.clear() and worker.close() in a finally block.
```

Asset hashes and calibration are explicit caller inputs, not automatically
estimated or independently certified. `calibration_id` on every frame must match
the selected configuration; changing calibration requires a new configured worker.
Original capture milliseconds and capture sequence are preserved, including through
SensorStreams observations and MCP `get_observations`. Native time is relative to
the first capture after reset, so epoch-sized timestamps do not lose precision.

The provider owns gap detection (`reset_on_gap=False`), preserving an explicitly
selected capture stride through SensorStreams instead of resetting on every
skipped source index. Actual omissions beyond that stride still reset state.
Ownership changes, sequence gaps (relative to declared sequence_step), time gaps
over max_gap_ms, explicit reset and recovery after rejected input reset native
state before another frame is tracked. Same-owner reordered/duplicate captures
are rejected. Lifetime frame limits are never replenished by resets. Reopen an
explicit new worker when exhausted; no automatic unlimited session renewal occurs.
For planned bounded rotation, opt into [ORBSLAMProcess](orbslam-rotation.md).
A native failure can require reopening rather than successful reset. Deadlines
and cancellation belong to ProcessProvider, which reaps only its owned child.

The output has `pose` and `provenance` objects. A tracked `world_to_camera` is T_cw,
an arbitrary-scale monocular rigid transform. Other tracking states emit null.
`reset_segment` identifies SDK resets; it is NOT a verified native map identity.
`map_changed_hint` and a cumulative count preserve the upstream hint without
asserting complete change detection. `coordinate_continuity_verified` and
`continuous_world_motion_available` remain false even between tracked frames.
Loop closure/relocalization can change coordinates. Do not derive metric object
speed, physical forces or cross-frame world motion from this output.

## Evidence and remaining acceptance

```sh
python3 -m unittest discover -s tests -p 'test_orbslam*.py' -v
```

Tests use the real profile, worker protocol, ctypes wrapper and compiled C++ bridge
with an explicitly **test-double System.h**, covering capture/MCP publication,
reset/gap/budget behavior and native-block timeout/reaping. They are software
integration evidence, not actual ORB-SLAM3 inference. Real upstream build,
vocabulary, calibrated reference replay, trajectory quality, native memory,
latency and deployment license review remain required in
[#186](https://github.com/NomaDamas/Gamst/issues/186). This does not close family6.
The provider implementation slice is [#193](https://github.com/NomaDamas/Gamst/issues/193).
