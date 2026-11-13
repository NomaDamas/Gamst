# Managed end-of-input finalization

`await streams.finish(source_id, sensor_id)` is a terminal end-of-input operation for one existing SensorStreams stream. It rejects new submissions to that stream, drains already accepted packets, calls each eligible provider's optional zero-argument `finish()` once, publishes a returned observation and disposes all providers. The result lists published and skipped capability IDs. Reopen explicitly to capture again; this creates new stream/provider epochs.

For Qwen ASR this flushes the residual audio tail without inventing a new capture frame. The final observation uses the last successfully processed capture timestamp and metadata, plus a sequence strictly above the stream's last accepted input. Existing TranscriptMemory can therefore accept a final replacement at the same capture time as the prior partial. The output sequence is a publication control step, not evidence of another sampled packet. Sequence overflow is an error, not a wraparound.

Only metadata from the last successful processing operation is retained for finalization; raw input is not retained by this control path. A provider returning None has no additional final observation. Providers without finish, empty streams, lanes with a pending gap/error and invalidated generations are skipped. Pending partials are marked interrupted when disposed; already final transcripts remain final. Finalization does not fabricate a success from incomplete audio. Exceptions in finalization/publication/disposal are reported after cleanup, and previously published finals from other lanes are not rolled back.

```python
# All desired capture hops have been submitted; no further capture is intended.
result = await streams.finish('microphone-source', 'microphone-sensor')
# result: {'closed': True, 'published': [...], 'skipped': [...]}
```

The shared Python runtime exposes `await human.finish_sensor_stream(manager_id, source_id, sensor_id)`. Discover the exact manager and stream IDs using `get_sensor_status`. The MCP tool `finish_sensor_stream` accepts those same three required string arguments; the live WebSocket API exposes it through `tools/call`:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"finish_sensor_stream","arguments":{"manager_id":"ID_FROM_STATUS","source_id":"microphone-source","sensor_id":"microphone-sensor"}}}
```

The caller must stop its application-owned producer first; this command does not stop a physical microphone or legacy MediaRouter capture. It never opens a stream, installs a model or loads a provider. Unknown manager/stream IDs and repeated completion fail explicitly. Manager IDs distinguish identical source/sensor names in different managers and are not authentication credentials. Access uses the existing shared-runtime transport trust boundary.

Finalization does not hold the MCP server's global tool lock: another connection can inspect status or delete data while inference drains. Requests on one WebSocket are processed in order, so use a separate connection for concurrent control. Deletion invalidates pending output before waiting for provider cleanup. Existing worker timeouts still apply; an in-process provider can block cleanup indefinitely.

ProcessProvider forwards finish over its owned pipe protocol; the selected concrete profile must still implement a finalizer to produce output. See [provider processes](provider-processes.md) and [Qwen local runtime](qwen-local-runtime.md).

## Ordering, cancellation and isolation

The selected stream stops accepting packets before draining. Other already-open streams can still submit and process; lifecycle operations using this manager's lock wait until cleanup finishes. Finalizers run serially and synchronous provider work runs outside the event loop. No new queue or unbounded task backlog is introduced by the selected stream.

Once finalization has acquired ownership, cancellation marks that stream closing and suppresses late final output, waits for owned worker execution and disposes state before returning. It does not kill an inference process. Cancellation before ownership leaves the stream untouched. Pause, clear and runtime generation changes invalidate pending finals. Caller cancellation cannot retract a final already published before cancellation arrived. A hung in-process provider can delay cleanup; an isolated worker watchdog is required for enforceable inference timeouts.

`clear`, normal `close` and runtime pause remain abort/discard operations. They do not imply successful end-of-input or call finish. Choose finish only for audio that should be finalized; missing or dropped audio must not silently become a complete utterance.

## Evidence and remaining gates

`python3 -m unittest discover -s tests -p test_stream_finalization.py -v` verifies drain-before-final, timestamp/sequence semantics, terminal close/reopen, cancellation during a blocked worker, another sensor's progress, generation invalidation, finalizer failure and empty-stream cleanup. The Qwen adapter tests also run its concrete lifecycle methods through this runtime using a labeled engine double. These establish control flow and publication, not speech-recognition accuracy. Actual Qwen inference remains #68; #67 tracks remaining integration requirements. Tests also verify exact manager selection, MCP deletion during a blocked finalizer, and final transcript retrieval through a real loopback WebSocket. These use provider fixtures, not microphone or model inference.
