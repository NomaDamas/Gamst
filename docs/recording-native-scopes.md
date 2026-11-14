# Recording-owned operation evidence

This opt-in runtime records actual SDK callback entry and owned source/process lifecycle work under immutable recording, stream, task and provider bindings. It does not establish model-backend coverage, authentic invoices, real quality, independence, cache isolation or savings. `actual_acceptance` remains `incomplete`; receipt authenticity and worker-backend coverage remain false. Native4 is not converted into legacy native3 or paired evidence.

## Runtime ownership

`RecordingEvaluationSession.from_frozen(...)` binds collection2, scope1 and instrumentation2 declarations to an exclusive `RecordingNativeWriter`. Separate `retain_evaluation_hashes=True` consent is mandatory. Financial document retention additionally defaults false and must match the frozen scope policy. Hashes remain correlation data in externally owned evaluation artifacts; SDK deletion scrubs live private mappings, not retained immutable evaluation files.

Before activation, bind each real controller with `bind_semantic_lane(controller, binding_index)` and actual callback/provider with `bind_backend(binding_index, backend, kind)`. `stream_scope(binding_index)` and `task_scope(original_task_id, binding_index)` return owner-issued immutable handles. Text, Visual and Smol adapters capture both the handle and actual backend before their first await. HumanSense registers the one root owner; handles do not create extra controllers or privacy observers.

Smol startup additionally requires `evaluation_worker_scope=owner.stream_scope(binding_index)`. The normalized provider, revision and configuration are checked before spawn. Actual ProcessProvider lifecycle hooks register acquired worker identity and verified reap; adapter-entry does not certify model-worker entry. Callback detachment after `await adapter.close()` uses `backend_closed(...)`, and never claims closure of an external client or process.

SensorStreams accepts explicit `evaluation_bindings` and `evaluation_configurations` at open, plus `evaluation_capture_ref` at submit. It verifies the owned payload bytes, dimensions/sample rate, source sequence and mapped capture timestamp before enqueue. Actual entry occurs inside the owned thread/coroutine. Factory/startup, reset, idle, residency, close and thread join are separate work intervals. A queued thread revoked before entry cannot invoke the provider. Explicit configuration declarations are checked; they do not prove an arbitrary closure honored its configuration.

## Capacity and cleanup

Run admission is bounded independently from native entry. Refused, cancelled and revoked pre-entry reservations release native credits only after callback entry is impossible. Original controller limits remain unchanged: default128/max256 retained requests and the existing1024 budget slots can stop unknown-bill SDK calls long before a one-hour3600-call workload. The3601 callback fixture proves journal capacity only, not financial admission or model throughput.

The collection supports at most32 recording groups/96 sequential ABC invocations,1000 original tasks,64 tasks/recording and3096 derived scopes. Original statistical minimums remain unchanged. Smaller structural fixtures are permanently numeric-ineligible. Per-run caps are16384 entries,131072 reservations,262144 events,1024 chunks and256 active tickets; joint caps include131072 entries,8192 chunks,262144 claims,1048576 evidence references and4GiB artifacts. These maxima are not simultaneously admissible: conservative fragmentation and aggregate byte preflight reject incompatible combinations.

`preflight_recording_collection(all_original_bundles)` validates every source and the shared C/Q/I identity before parent spawn. `ParentNativeChannel(..., collection_declarations=all_original_bundles, budget=shared_budget)` performs this preflight and automatically precharges source/metadata and retained child staging once through `ArtifactBudget.bind_collection`. Reuse that budget across all runs; do not manually double-precharge it. Parent copies debit the remaining budget. Child sessions validate all native maxima derivable from C and their selected original source; they do not claim to have read unavailable other recordings.

Events are at most2048 bytes. The owned queue is at most64 frames/1MiB including in-flight writes. Chunks are at most1MiB/256 events/128 entries; index pages contain1..128 references, with at most8 pages. Hash chains and original fixed filenames cover each sealed chunk. Failed partial chunks retain a separately hashed valid prefix. Cancellation joins owned writer/channel work; terminal publication follows cleanup. No late attachment mutates a frozen terminal.

## Transport and public readers

The parent creates the socket pair, calls `prepare(...)`, passes only the explicit FD delta, and calls `spawned(actual_process)`. Drain `collect()` concurrently with stdout/stderr and owned process work. Protocol errors fail immediately; the caller must cancel/reap its actual child. `record_reaped(actual_process)` accepts only that registered completed subprocess after `wait()`. `snapshot()` returns immutable `NativeRunInventory`; child terminal/EOF alone never sets parent reap.

The child uses `await ChildNativeChannel.from_environment()`, opens its exclusive staging writer, starts `channel.follow_writer(writer)`, then uses the actual session/SDK. After source/adapter cleanup and session close, send `await channel.close(terminal)`. Startup, cancellation and serializer failures use `abort()`; they do not manufacture successful terminal state. Parent-first bootstrap and hello share the original deadline. Nested worker FDs are non-inherited.

`validate_recording_native_inventory(declarations, directory, terminal_sha256, source_inventory=original_manifest)` returns a reader-issued immutable `ValidatedRecordingIndex`. Bounded regular files are opened nonblocking through a no-symlink directory walk and rehashed on subsequent reads. Callers cannot construct or mutate an ordinary validated index. The reader uses compact packed entry/digest/settlement tables and bounded active state; it does not retain all event dictionaries.

- `canonical_lane_inventory(index, scope_index, binding_index)` exposes complete represented reservation/release/entry/finish/publication counts before matching receipts or transports. An incomplete prefix explicitly leaves unobserved entry inventory possible.
- `iter_canonical_operations(index, scope_index=None, binding_index=None)` streams every actual entry, including failed/cancelled entries, with exact immutable owner, provider/model/role, original decision/prompt/evidence digests, original chunk/source locators, terminal outcome and unknown liability.
- `iter_canonical_inventory(...)` is the combined second pass for operations and worker intervals. Use it when both are needed rather than rescanning once per operation. Intervals retain actual worker ordinal, parent entry, host-monotonic endpoints and start/end event/chunk lineage. Residency overlaps are not CPU time or critical-path latency.
- `lookup_canonical_operation(index, entry_ordinal)` reads one bounded original chunk and uses reservation digests captured during validation. It does not trust a caller identity cache or select the latest attempt.
- `validate_recording_native_collection(runs)` requires every original invocation once, distinct nonces, terminal hashes, zero-inclusive inventories and global claim identity conservation. It produces no numeric acceptance.

Portable CLI:

```sh
python3 -m benchmarks.recording_native_cli \
  --declarations declarations.json --artifacts native-run \
  --terminal-sha256 ORIGINAL_SHA256 --output structural-audit.json
```

The CLI validates before exclusive output creation. It never modifies source artifacts or executes a source command.

## Money and remaining work

A declared receipt attachment retains bounded original bytes and line quantities under a separate consent policy. Its owner-independent claim identity uses provider, account scope, invoice ID and line ID. Changing task, arm, document revision or execution owner cannot hide duplicate original lines. All such attachments remain money-unknown and never settle controller debt. Aggregate provider exports without original invoice-line identity are not fabricated into this format.

Persistent ABC recording execution and versioned matched-source reconciliation belong to#270/#274. Verified provider billing/transport settlement belongs to#257/#275/#276. Actual device/model inference, independent assessment, source opportunity, isolation and original quality/cost/latency gates remain open.
