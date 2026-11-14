# Bounded ORB session rotation

`ORBSLAMProcess` opts into rotation of the concrete `orbslam3.monocular` child
when its configured frame budget has been used. The ordinary ProcessProvider
profile stays finite and never rotates implicitly. Both require the same explicit
local library/vocabulary/calibration hashes; no assets are installed automatically.

```python
import sys
from humansense import ORBSLAMProcess, SensorStreams

# config is the explicit local asset configuration in orbslam-context.md.
provider = await ORBSLAMProcess.open(
    sys.executable, config, max_starts_per_minute=6,
    startup_timeout=120, inference_timeout=30,
)
streams = SensorStreams(human, max_packet_bytes=8*1024*1024)
streams.register(provider.spec, lambda: provider)
await streams.open('camera', 'video', [provider.spec.id])
# Submit matching calibrated InputFrames, as in orbslam-context.md.
# In finally: await streams.clear(); await provider.close()
```

Only one operation and one owned child can exist at a time. Concurrent calls are
rejected, not queued. Process inputs are copied before rotation awaits, at the
existing16MiB wire bound. SensorStreams owns its separately bounded input queue;
its `status()` reports accepted/processed/dropped frames and latency. Frames dropped
there are never invented by this provider. A new native child is started only
AFTER the previous child has exited and been reaped, including stalled shutdown
escalation by ProcessProvider. Thus rotation can create observation gaps; this is
not uninterrupted scene coverage or a hard real-time guarantee.

The default300-frame native budget spans resets; a reset never buys more frames.
After budget exhaustion, the next admitted frame starts a new session, rather
than retrying a rejected native call. Original capture ordering is checked across
rotations (capture_sequence when present, otherwise sequence). Duplicate/reordered
captures stop the session. Explicit reset permits a new capture origin. Worker
errors, cancellation, failed shutdown/startup or rate exhaustion close the wrapper without
retry/restart. Successful planned rotations alone continue automatically. Startup
attempts are limited to1..60 per rolling minute (default6), including initial open;
exhaustion stops explicitly, without sleeping while retaining frames or crash loops.

Each observation keeps the native session ID, reset segment and original capture
provenance. `provenance.rotation` adds session number, wrapper request-attempt
number, session request count, whether rotation preceded this capture, and local
startup/rotation durations. Coordinates are NOT continuous across sessions, or
independently verified within them. Monocular scale remains arbitrary.

`diagnostics()` keeps bounded scalar counters: completed observations, admitted
worker process requests, successful starts, startup attempts, and elapsed request /
successful startup / successful rotation time. Startup time is part of rotation
time; these are overlapping measurements and must not be summed as independent
costs. Failed startup/rotation durations are not included in the successful totals.
A request counter is not proof of actual native execution after ambiguous failure.
Actual cost is unknown, native-attempt verification is false, and complete cost/
quality/latency acceptance remains separate. No raw frame history or unbounded
attempt journal is retained; application observability can snapshot these counters.

Direct `close()` cancels and joins the active operation before returning. Managed
HumanSense pause/delete/stream clear cancels rotation, prevents late publication
and reaps the child. Reopening is explicit after stop. Calling close repeatedly or
cancelling its waiter does not abandon owned cleanup.

```sh
python3 -m unittest discover -s tests -p test_orbslam_rotation.py -v
```

Tests use the real owned process protocol, ctypes and compiled bridge against a
**test-double System.h**, not actual SLAM. They cross several small session budgets,
check single-child ownership, MCP capture/session provenance, ordering, rate limits,
startup failure/stall, shutdown stall, close/cancel/pause/delete and no replay.
Real runtime build, reference trajectory quality, native memory and capture latency
remain required in [#186](https://github.com/NomaDamas/Gamst/issues/186).
