# Owned OpenEB capture process

`OpenEBProcess` supervises one explicitly selected camera or RAW recording in a
caller-selected Python runtime. It sends EventCD packets to the application;
`ProcessProvider('openeb.eventcd')` remains a separate numeric context worker.
Neither worker invokes an LLM. SensorStreams/MCP expose observations at the
application's chosen cadence. Raw pixels are not included in that context.

```mermaid
flowchart LR
  A[Explicit camera or RAW] --> B[Owned native capture child]
  B --> C[Async parent: bounds, clock, identity]
  C --> D[Owned numeric context child]
  D --> E[SensorStreams state]
  E --> F[MCP context at user cadence]
```

## API and lifecycle

```python
from humansense import OpenEBProcess

selection = dict(kind="camera", source="EXACT_SERIAL", source_id="event-camera",
    sensor_id="events", sensor_origin_us="0", capture_origin_ms=known_capture_origin_ms,
    calibration_id="camera-clock-calibration", coordinate_frame="sensor-pixels")
async with await OpenEBProcess.open("/absolute/python", selection,
        startup_timeout=30, read_timeout=5, close_timeout=2) as source:
    frame = await source.next_frame()  # Own epoch and original capture sequence.
    context_config = source.config    # Actual reader dimensions plus declared clock.
```

The selected runtime must already contain gamst and the pinned OpenEB wrappers/
native SDK/plugins described in [the context contract](openeb-context.md).
No installation, weight download, implicit device selection or restart occurs.
A recording must be an existing absolute `.raw` path. Clock mapping, source IDs,
coordinate frame and a native origin aligned to 10000 microseconds are explicit.
Mapping configuration does not prove calibration. Native time remains exact.

Only one read can be pending; concurrent reads reject instead of accumulating.
Each command has a deadline. Read cancellation, timeout, malformed response or
EOF permanently closes the source. `close()` invalidates publication immediately,
interrupts a pending read and awaits the shared cleanup task even when the caller
is repeatedly cancelled. Cleanup closes pipes, waits briefly for normal exit,
then terminates/kills only the owned child handle and waits for reaping. Other
streams can progress while native capture is blocked. This is not a hard-real-time
OS guarantee: spawn/reaping can exceed deadlines under OS/kernel stalls, and
cleanup adds the existing 0.1-second exit and 1-second termination grace periods.

Explicit reopening creates a new UUID epoch. With `next_frame(managed=True)`, the
source epoch is omitted from the submitted frame because SensorStreams owns that
field; original capture sequence is retained. **Close/reopen the managed stream
alongside the source**, creating a fresh worker/state epoch. Never feed a newly
opened source's sequence zero into an old stream. A native clock reset requires
renewed mapping; no continuity or missing events are fabricated.

Diagnostics contain stop reason, close acknowledgement, child return code,
reaping state and number of received frames. A close acknowledgement reports the
child's completed cleanup call; a negative POSIX return code records signal exit.
An acknowledgement alone does not prove a successful process exit. Native/device
cleanup is always `native_cleanup_verified=False`; forced exit may leave external
driver/device resources in an unknown state. SDK stdout/stderr are discarded,
so retained stderr is zero; errors expose a sanitized exception type only.

## Bounded wire

Requests/responses use an 8-byte network-order header/payload length prefix.
Headers are at most 16384 bytes, packets at most 106532 bytes, request IDs at most
2^53-1. Duplicate JSON keys, nonfinite values, mismatched IDs/fields, CRC, geometry,
source identity, packet continuity and mapped timestamps reject before delivery.
The asyncio reader has a 32768-byte flow-control limit (not an exact total RSS
cap); native SDK/driver buffers are outside this protocol's memory bound. There
is no unbounded Python read queue. Parent NumPy validates each event packet.

## Explicit replay/camera example

Run from the checkout with the host's existing gamst/NumPy environment:

```sh
python -m examples.openeb_stream \
  --kind recording --source /absolute/input.raw \
  --capture-python /absolute/openeb-python --context-python /absolute/numpy-python \
  --source-id event-camera --sensor-id events --sensor-origin-us 0 \
  --capture-origin-ms 1700000000000 --calibration-id declared-recording-clock \
  --coordinate-frame sensor-pixels --max-frames 1000 \
  --pacing capture-relative --context-interval-ms 1000
```

Replace the example clock with the recording's known capture origin. The example
prints explicit selection/configuration, context and final diagnostics. It uses
bounded iteration, owns cleanup and starts no MCP network server or LLM call.
For camera capture choose `--kind camera --source EXACT_SERIAL --pacing none`
and the measured clock origin. User consent is explicit in running this example.

`capture-relative` schedules replay using capture-time differences from the
origin and monotonic host time. `none` processes as quickly as downstream allows.
Neither rewrites timestamps to current time or proves live latency. Old recordings
may therefore be stale/absent from MCP's current-state query; use historical
queries for their original capture interval. Processing backpressure can make
replay late; frames are not silently dropped by the example. Slow camera consumers
can still overflow native buffers: actual drop/queue behavior requires #140.
The context output runs in a separately owned async task, so polling continues
while capture waits or replay is paced. Both tasks are cancelled/joined before
stream cleanup on completion, timeout, failure or cancellation. Polling is
best-effort on the host event loop: missed ticks are skipped without bursts;
blocked stdout or a stalled event loop can still delay delivery. The context
output interval is independent of packet inference frequency.

## Verification boundary

`python -m unittest discover -s tests -p test_openeb_process.py -v` exercises real
owned subprocesses with a generated source fixture: blocked startup/read/close,
repeated cancellation, one-read concurrency, EOF/reopen, bad CRC/metadata/ID/size,
partial writes, abrupt exit, native output flood, unrelated async heartbeat and
source → context child → SensorStreams → MCP. These are supervision and integration
tests, not installed SDK decoding, real camera throughput or hardware cleanup.
Software issue [#142](https://github.com/NomaDamas/Gamst/issues/142) remains separate
from mandatory [native acceptance #140](https://github.com/NomaDamas/Gamst/issues/140).
