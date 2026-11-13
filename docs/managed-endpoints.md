# Managed nonterminal utterance control

`SensorStreams.endpoint` finalizes one existing provider lane without closing its
capture stream. Use `endpoint_sensor_stream` through HumanSense, MCP or the live
WebSocket `tools/call` API. No sensor/provider is opened, installed or inferred by
discovery. The provider must explicitly implement `endpoint()`.

Get manager/source/sensor/capability IDs and the provider's `stream_epoch` using
`get_sensor_status`. Epoch is required so a command cannot silently target a
reopened or gap-reset lane. IDs and epochs are selectors, not authentication.
These controls use the existing shared-runtime transport trust boundary.

```python
result = await human.endpoint_sensor_stream(
    manager_id=manager_id, source_id='mic-source', sensor_id='mic-device',
    capability_id='audio.qwen3_asr.silero_endpoint', stream_epoch=provider_epoch)
# {'published': True/False, 'closed': False, 'capability_id': ..., 'after_sequence': ...}
```

The equivalent MCP tool name is `endpoint_sensor_stream`, with those same five
required fields. Optional `after_sequence` must exactly match the current last
accepted input when the request is admitted. Omit it for a running capture source:
the manager atomically selects its current accepted-input barrier and returns it.
Otherwise a remote status/read roundtrip can naturally become stale as audio
continues arriving. Empty streams reject endpoints. An old epoch or explicit
sequence mismatch is rejected before invoking the provider. A successful endpoint
means the current buffered utterance was processed, not that an older utterance
seen by the caller was still active or a person naturally completed their turn.

## Ordering and resource bounds

One pending/in-flight control is allowed per lane, outside the bounded audio queue.
It contains metadata and one completion future, no PCM. Already accepted input up
to the selected barrier runs first; the endpoint then runs before later input.
Later captures may continue arriving and other lanes continue processing. Repeated
pending controls fail rather than building a backlog. The existing data queue can
still drop its oldest input. A missing barrier, gap, changed epoch/generation or
stream cleanup invalidates the command instead of finalizing another interval.
A dropped future input while finalization is in flight conservatively invalidates
its output too. The next input resets affected state.

Capture sequences delivered to providers remain the actual input sequence. Context
publication maintains a separate per-lane monotonically increasing `sensor.sequence`.
It is at least the input sequence and advances for no-input finals as well. Every
managed publication includes `sensor.capture_sequence`, the last actual input used
for that result; transcript queries expose it as `capture_sequence`. Timestamp,
source, subject and provider epoch remain capture evidence. Neither output sequence
advancement nor a manual final means another frame was captured. Old direct
publishers may omit capture_sequence (None). Parent/worker SDK versions must match;
older strict metadata consumers need an upgrade for the additional optional field.
64-bit sequence overflow fails instead of wrapping.

## Cancellation, clear and terminal finish

Cancelling before provider invocation suppresses the command without touching the
partial. Cancelling during invocation waits for owned execution, suppresses a late
final and interrupts the affected partial; subsequent input resets its lane. An
already published final cannot be retracted by a later cancellation. Provider
errors likewise invalidate that lane. The stream remains open; a failed isolated
worker may require explicit stream reopen because ProcessProvider fails closed.

Pause/delete/clear and generation changes invalidate pending results. Clear waits
for active owned work before disposing providers and resolves pending control
waiters. Terminal `finish` waits for pending endpoints before its own finalizer and
then closes the stream; the two never run concurrently on one provider. `drain`
also waits for currently owned lane execution, including an endpoint with no PCM
queued. A hung in-process native call can delay completion/cleanup; use an isolated
profile watchdog for enforceable inference limits.

The MCP control does not hold the global tool lock while waiting. Another
connection can delete data or query status. One WebSocket processes its requests
in order, so use another connection for concurrent controls.

## Validation and remaining acceptance

`test_managed_endpoints.py` verifies barrier order, independent output/capture
sequences, post-endpoint input, repeat/empty behavior, stale epochs and sequences,
one-control bounds, pending/active cancellation, another sensor progressing,
dropped barrier rejection, clear/delete races, terminal finish ordering and integer
overflow. `test_perception_api.py` exercises the new control and transcript query
over an actual loopback WebSocket. Providers are deterministic fixtures; they do
not establish speech recognition or VAD quality.

Concrete Qwen/Silero adapters and owned-child control tests are documented in
[ASR endpoints](asr-endpoints.md). Actual Korean/English model/device/runtime,
endpoint errors, sustained overload and capture-to-context latency evidence remain
required in #68/#76. The overall expansion plan is not complete.
