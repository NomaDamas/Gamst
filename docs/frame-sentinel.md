# Fresh-frame sentinel

`FreshFrameSentinel` supplies the no-detection sampling path from an explicitly
submitted capture to the existing controller and local `SmolVLMCalls` executor.
The application owns capture, consent, a registered controller, a verified quote,
and pre-existing local model assets. Opening a sentinel starts its one selected
worker; `start()` separately activates its frozen timer. It does not open a camera,
download assets, create a consumer LLM, or change capture/consumer cadence.

```python
from humansense import FreshFrameSentinel

# human has consent and a registered controller; sensor is the original camera
# SensorMetadata, including an explicit stream_epoch. config is the existing
# strictly local SmolVLM2-500M configuration, not a model identifier to download.
async with await FreshFrameSentinel.open(
    human, "blindspot", "controller-id", "/existing/environment/bin/python", config,
    subject_id="local-user", sensor=sensor, frame_shape=(height, width, 3),
    maximum_charge=verified_upper_bound, interval_ms=5000, phase_ms=0,
    deadline_tolerance_ms=250, max_age_ms=1000,
    evaluation_observer=existing_opt_in_evaluation_session,
) as sentinel:
    sentinel.start()
    # Called on HumanSense's event loop when the application returns a capture.
    admission = sentinel.offer(original_input_frame)
    # Consumer reads/polls remain independent; polling never triggers inference.
    status = await human.get_sentinel_status(sentinel_id="blindspot")
```

The runnable `python -m examples.fresh_frame_sentinel --help` example consumes a
bounded application-owned raw-frame manifest. It uses this actual worker path,
preserves original hashes/timestamps, polls Context independently, and leaves
unverified costs unknown. It does not contain a substitute model backend. Its
recorded timestamps must still be fresh when offered: old recordings are reported
stale, never retimestamped to look live. Trained assets/dependencies are supplied
by the caller and are never installed by the example.

The manifest schema is closed and bounded (128 captures, one-minute span). Files
contain exactly `height * width * 3` uint8 HWC BGR bytes, with no image decoder:

```json
{"schema_version":1,"subject_id":"local-user","shape":[32,48,3],
 "sensor":{"source_id":"camera","sensor_id":"video","sequence":0,
           "coordinate_frame":"pixels","channels":[],"calibration_id":null,
           "stream_epoch":"original-session","capture_sequence":0},
 "captures":[{"path":"capture.bgr","timestamp_ms":1790000000000,
              "capture_index":0,"frame_sha256":"<smolvlm_frame_sha256 of original pixels>"}]}
```

Replace the illustrative timestamp/hash with the original producer values. Raw
paths stay inside the manifest's owner directory. The example rejects malformed
hashes, oversized inputs, path escapes or changed pixels; it erases its mutable
producer copy after each offer and joins its own reader/sentinel on failure.

## Clocks, ownership and admission

This API accepts **Unix milliseconds** for `InputFrame.timestamp_ms`, from the
same host wall-clock domain used by HumanSense. Neither `InputFrame` nor
`SensorMetadata` has a clock-domain field. A producer using a device/stream clock
must first apply its own explicit, bounded mapping and retain its mapping evidence;
this API cannot authenticate that mapping. Arbitrary stream milliseconds will
usually be stale and cannot silently become fresh. Future, regressing or replayed
captures are rejected. A regressing host Unix clock invalidates the binding.
Equal timestamps with strictly increasing capture indices are valid distinct
captures; the timestamp alone is not a unique ID.

Scheduling uses this host's **`time.monotonic_ns()`** exclusively. The origin is
the explicit `start()` reading; phase is relative to it. An elapsed tick is skipped
without catch-up bursts. A due point outside its frozen tolerance is uninspected.
The executor rechecks an optional absolute monotonic deadline after async worker
reset and immediately before controller dispatch. It never compares a capture
timestamp to that deadline. Expired unadmitted reservations are cancelled without
a backend attempt; an earlier dispatched liability remains untouched. Existing
`SmolVLMCalls.execute(..., dispatch_deadline_ns=None)` behavior is unchanged.

`offer()` has no await or model work. It validates a plain NumPy uint8 HWC BGR
array, original shape (32-pixel minimum side; native 1080p area bound), subject,
source, sensor, channels, epoch, calibration, coordinates, capture-index kind,
timestamp and maximum age before copying. The copied snapshot determines its
versioned shape/dtype/layout/pixel hash. Caller mutation after return cannot change
the retained evidence. The next accepted capture has a strictly larger original
capture index (or sensor sequence when no capture sequence is supplied). Sequence
gaps are diagnostics, not proof that physical frames were captured or lost.

There is one active capture and one replaceable latest candidate. Their combined
owned original pixel bytes never exceed `max_owned_bytes` (at most 16 MiB).
Replacement erases the old candidate before copying its successor. When the active
copy leaves insufficient capacity, a new offer is rejected before copying. An
overwritten candidate is explicitly unobserved. A slow backend cannot suspend
the capture, audio, or Context paths. Adapter copies, IPC serialization, native
image tensors, model state and caller buffers are additional allocations and must
be included in whole-process resource measurements; 16 MiB is not an RSS claim.

Shape, calibration, source, epoch or capture-index-kind changes invalidate the
binding. Explicitly open a new instance after cleanup, using the same controller
and financial ledger. Configuration views are detached; changing cadence requires
a new explicit open. This path does not replay cached successful pixels.

## Sampled points, gaps and Context

At a valid tick the selected capture produces a registered `capture.available`
measurement even when there is no detector observation. The measurement asserts
only that validated capture bytes were supplied. It includes original capture
time, source/sensor/epoch, index, shape and the shape-bound pixel hash; it does not
claim scene content, detector quality or physical camera authenticity. The normal
controller policy still decides whether the source needs a small model. Budget,
cooldown, unknown quote, liability, freshness and lifecycle checks remain active.

The sentinel submits and executes only its own exact request/evidence join. Other
callers sharing the controller cannot have their requests dispatched with these
pixels. The existing executor resets its frame window, verifies model/prompt and
capture identity, and publishes accepted descriptions at the **original capture
timestamp** through normal Context/history/MCP. Descriptions remain untrusted.
Empty output, errors, rejected admissions, missed deadlines, busy ticks and missing
captures never advance the last accepted inspected point.

Status distinguishes due, selected, controller-dispatched, completed and accepted
points. Controller dispatch is not proof of backend entry; use the optional #255
observer's existing native callback/worker hooks for that boundary. A successful
inspection covers a sampled point/window only. Intervals between points, dropped
candidates and unavailable capture intervals remain unobserved; a brief event can
fall completely between ticks. Status makes no continuous-coverage or recognition
accuracy claim. First absence and capture return are recorded separately from an
accepted point. Bounded metadata history omits older rows with an explicit count.

`await human.get_sentinel_status(...)` and MCP `get_sentinel_status` are read-only,
bounded metadata queries. They do not expose pixels/model answer bodies or activate
workers/capture. Once cleanup retires a sentinel, HumanSense no longer retains it;
an application holding the instance may still read its scrubbed terminal counters.
Normal registered observations have the existing store retention rules; delete
uses the existing sensory deletion path. External copies have external owners.

## Lifecycle and costs

Pause, reset, subject deletion, consent revocation, source mismatch and host close
synchronously erase pending/active original copies and retained private bindings
before cancellation/reaping awaits. Late results cannot publish into a new epoch.
Startup cancellation/revocation and registration overflow cannot leave an unowned
child. `close()` joins only owned tasks and worker handles and tolerates repeated
cancellation. No external client is signalled or awaited by this driver.

The shared internal HumanSense driver registry accepts `frame_sentinel` and
`context_fill`: at most four of each and eight total, including invalidated owners
whose cleanup is not terminal. The driver protocol is synchronous
`invalidate(reason)` plus `owns_task(task)`. Cleanup is registered through
`_track_call_driver_cleanup(kind, id, identity, task)` and retires that exact
identity only after task termination. Consent's synchronous setter schedules and
retains cleanup; the next async lifecycle joins it. Calling a joining host lifecycle
from an owned operation fails after the synchronous invalidation barrier, preventing
self/ancestor deadlock while leaving cleanup host-owned. These are internal
ownership seams, not an arbitrary plugin transport.

Unknown quotes block dispatch. Dispatched calls without verified receipts retain
unknown financial liability after cancellation/deletion, including empty/error
results. Early verified receipts are preserved. The driver neither invents prices
nor settles zero from elapsed time/token counts. Settle verified complete charges
through the existing controller. Observer worker startup/residency/reset/reap
measurements include explicit worker lifetime, and do not establish total machine
energy, full-pipeline savings or billing authenticity.

## Validation and open gates

Owned-pipe fixtures exercise the actual ProcessProvider/controller/Smol executor,
registered availability and Context/MCP path without detections. They cover original
pixel mutation, slow workers alongside audio/Context, overwrite/byte limits, missed
and skipped ticks, distinct same-time indices, rare short-event phase aliasing,
unknown quotes/budget, empty/error output, startup cancellation, host lifecycle,
late bills, repeated close cancellation and mixed bounded driver cleanup. Controlled
backend descriptions establish wiring only, not trained-model quality.

This is the #265 sampling child of #133. Semantic contradiction resolution and
strong/text escalation remain separate required #133 work. Trained SmolVLM
acceptance #40/#150, A/B/C cost-quality-latency evaluation #57/#135 and semantic
reliability remain open. No quality-preserving savings or all-provider completion
is claimed.
