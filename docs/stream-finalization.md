# Managed end-of-input finalization

`await streams.finish(source_id, sensor_id)` is a terminal end-of-input operation for one existing SensorStreams stream. It rejects new submissions to that stream, drains already accepted packets, calls each eligible provider's optional zero-argument `finish()` once, publishes a returned observation and disposes all providers. The result lists published and skipped capability IDs. Reopen explicitly to capture again; this creates new stream/provider epochs.

For Qwen ASR this flushes the residual audio tail without inventing a new capture frame. The final observation uses the last successfully processed capture timestamp and metadata, plus a sequence strictly above the stream's last accepted input. Existing TranscriptMemory can therefore accept a final replacement at the same capture time as the prior partial. The output sequence is a publication control step, not evidence of another sampled packet. Sequence overflow is an error, not a wraparound.

Only metadata from the last successful processing operation is retained for finalization; raw input is not retained by this control path. A provider returning None has no additional final observation. Providers without finish, empty streams, lanes with a pending gap/error and invalidated generations are skipped. Pending partials are marked interrupted when disposed; already final transcripts remain final. Finalization does not fabricate a success from incomplete audio. Exceptions in finalization/publication/disposal are reported after cleanup, and previously published finals from other lanes are not rolled back.

```python
# All desired capture hops have been submitted; no further capture is intended.
result = await streams.finish('microphone-source', 'microphone-sensor')
# result: {'closed': True, 'published': [...], 'skipped': [...]}
```

This is a Python runtime control. Remote MCP/live-API exposure and the isolated Qwen worker's finalization protocol remain separate integration work in #67. Generic ProcessProvider profiles do not gain a remote finish method automatically. No model is loaded or installed by this operation.

## Ordering, cancellation and isolation

The selected stream stops accepting packets before draining. Other already-open streams can still submit and process; lifecycle operations using this manager's lock wait until cleanup finishes. Finalizers run serially and synchronous provider work runs outside the event loop. No new queue or unbounded task backlog is introduced by the selected stream.

Once finalization has acquired ownership, cancellation marks that stream closing and suppresses late final output, waits for owned worker execution and disposes state before returning. It does not kill an inference process. Cancellation before ownership leaves the stream untouched. Pause, clear and runtime generation changes invalidate pending finals. Caller cancellation cannot retract a final already published before cancellation arrived. A hung in-process provider can delay cleanup; an isolated worker watchdog is required for enforceable inference timeouts.

`clear`, normal `close` and runtime pause remain abort/discard operations. They do not imply successful end-of-input or call finish. Choose finish only for audio that should be finalized; missing or dropped audio must not silently become a complete utterance.

## Evidence and remaining gates

`python3 -m unittest discover -s tests -p test_stream_finalization.py -v` verifies drain-before-final, timestamp/sequence semantics, terminal close/reopen, cancellation during a blocked worker, another sensor's progress, generation invalidation, finalizer failure and empty-stream cleanup. The Qwen adapter tests also run its concrete lifecycle methods through this runtime using a labeled engine double. These establish control flow and publication, not speech-recognition accuracy. Actual Qwen inference remains #68; #67 retains loader, isolated runtime and remote control requirements.
