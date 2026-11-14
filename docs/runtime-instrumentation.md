# Opt-in runtime execution instrumentation

The runtime observer records actual SDK invocation boundaries and owned worker
lifecycle events. It does not turn response usage, reservation ceilings or elapsed
time into verified money. Unavailable source counters and billing remain unknown.
The persistent recording runner (#256) and concrete billing adapters (#257) remain
required follow-ups; per-task subprocesses cannot establish continuous recording
residency or whole-stream savings.

## Ownership and privacy

Evaluation artifacts have an explicit owner separate from the sensory store.
Constructing `EvaluationSession` requires `retain_evaluation_hashes=True`. Task,
input, prompt and evidence hashes remain correlation artifacts retained by that
owner. `HumanSense.delete_data()` does not erase those immutable external artifacts.
This consent does not waive normal SDK deletion, freshness or publication rules.

The observer never retains prompts, answers, pixels, raw subject IDs, private
source mappings or exception messages in its journal. During an active operation,
a private subject mapping allows targeted invalidation; it is scrubbed at operation
completion or subject deletion. Pause, reset, consent revocation and close revoke
future observer admission and scrub private mappings synchronously. Outstanding
financial linkage uses opaque random operation tokens. Late billing cannot restore
private mappings. At most 32 distinct observers may register with one HumanSense
runtime; repeated registration of the same observer does not consume another slot.

## Actual SDK boundaries

`TextContextCalls` (and inherited `VisualContextCalls`) accept the optional
`evaluation_observer` argument. `SmolVLMCalls.open` accepts the same argument and
propagates it to `ProcessProvider.open`. Omitting it preserves ordinary operation.

Before reserving or dispatching a controller request, the adapter synchronously
reserves observer capacity. Controller reservation/dispatch and backend invocation
are distinct events. A native3 attempt is created only when the owned invocation
coroutine enters, immediately before the selected backend callback. Cancellation or
sink failure before entry creates no native attempt/provider-call count; already
dispatched financial liability remains unknown. Observer failures never bypass the
adapter's backend draining or worker cleanup.

The session derives original provider/revision/route/prompt/evidence bindings from
the actual controller. The native `execution_sha256` is the digest of its actual
dispatched execution. Instance and operation identities are generated automatically.
Explicit `session.link_call(request_id, predecessor_operation_id=..., relationship=
'retry'|'escalation')` binds a later actual invocation to a completed predecessor;
it does not start a retry or change routing. The original native recorder enforces
same-evidence linkage and retry provider/revision/role consistency.

Callback success and publication acceptance are separate. Errors, cancellation,
timeouts and discarded results remain in the journal. Response token counts are
retained only when the exact bounded usage shape is recognized, labeled unverified.
They are never implicitly assigned prices or treated as invoice lines. The frozen
`phase` value (`cold`, `warm`, or `unknown`) is an explicit experiment declaration;
these hooks do not independently prove cache state or cold-start isolation.

## Worker measurement boundaries

ProcessProvider receives explicit observer and parent-operation identities;
attribution does not depend on global state or context variables. It records
spawn-begin, handle-acquired, handshake-ready/failure, each owned transport
operation and verified reap. Startup before semantic dispatch belongs to stream
worker work, not a fabricated VLM call. Worker operations are children of semantic
calls where explicitly bound and are not counted as second provider charges.

The clock is host `time.monotonic_ns`, named by the session's unique clock domain.
Startup elapsed time ends at handshake success/failure. Residency begins at actual
process-handle acquisition and ends only after owned `wait()` confirms reap. The
union of observed transport-operation intervals and its residency gaps are reported
separately; overlapping intervals are not summed. These are host elapsed proxies,
not CPU/GPU utilization, power, billed money or the task's critical-path latency.
Missing endpoints remain null. EOF and close acknowledgements do not prove death.
Native3 callback clocks retain their existing relative `perf_counter_ns` origin;
the journal uses absolute host monotonic readings. The parent compares callback
duration against its enclosing observed entry/finish duration, not timestamps from
these distinct clock representations, and does not invent an origin mapping.

The measurement bundle enumerates every frozen required resource lane with unknown
quantity/billing until a supported measurement adapter supplies it. Physical capture
availability and attempted/captured/dropped counters are not inferred from SDK
accepted/processed counts. No idle/startup/resource lane silently becomes free.

## Native reports and terminal lifetime

The session emits native3 task and stream reports automatically. Unknown quantities
and line claims stay null. `unresolved_attempt_billing` describes the native
attempts only; `unresolved_billing` also includes dispatched pre-invocation
liabilities and unmeasured required monetary resource lanes. Settling an attempt never settles startup/idle costs implicitly. Optional `attach_declared_lines(operation_id, quantities=
..., line_claims=..., evidence_sha256=...)` attaches explicit external line evidence;
it does not fetch invoices or authenticate them. Synthetic tests exercise this path
without creating a production fixture fallback. Real automatic receipt extraction
belongs to #257.

Close stops admission, waits a bounded time for observed invocation owners and
freezes a detached terminal bundle. Unfinished work/reap or journal failures remain
explicit incomplete reasons. SDK adapters retain cancellation and worker ownership;
close them before closing the evaluation session. Session close drains its owned
writer and resists repeated caller cancellation. The channel writer drains for at
most five seconds, bounded further by the original task deadline, before cancelling
and joining its owned writer and closing its socket. It never waits indefinitely for
future invoices. Billing after terminal cannot mutate the original report, journal
or source; a future amendment requires a separate versioned exclusive artifact.

`actual_acceptance` is always `incomplete`. Instrumentation coverage, receipt
authenticity, measurement truth, isolation and whole-pipeline cost verification
remain false. Unsupported adapters, hidden third-party retries, unobserved local
resources and physical capture are named coverage gaps.

## Instrumented collector transport

The separately versioned collector preserves original requests and source trace
format. It freezes an external instrumentation protocol after P/C/S/Q/R, avoiding a
collection/reconciliation hash cycle. Executable/artifact/argv/environment preflight
remains mandatory. The sole explicitly declared runtime environment addition is
`GAMST_EVALUATION_FD`, identifying an owned inherited socket descriptor.

The parent sends the bounded bootstrap first; the child validates its nonce,
original request/task/arm/stream and declaration digests and returns one hello before
instrumented work. The shared hello deadline includes bootstrap transmission. The
child marks the descriptor non-inheritable; nested provider workers use
`close_fds=True`. Each frame has a four-byte big-endian length and strict JSON, with
sequence/hash-chain checks, bounded report inventory, exactly one terminal and EOF.
Trailing data, changed identities, missing terminal and partial writes remain
incomplete rather than being accepted as measured success.

Parent stdout/stderr and journal readers run concurrently. Timeout, protocol failure
and repeated cancellation close owned descriptors and join readers while preserving
owned process-group cleanup. The parent retains a valid journal prefix and binds its
final sidecar to the original source trace and response digest. No child terminal
self-certifies that parent binding or process death.

Limits are 512 KiB per frame, 16 MiB aggregate collection journal/report bytes, 8192 events
per child, 128 native attempts per task/stream scope, 32 workers, 1024 worker operations,
and 64 queued frames/1 MiB queued bytes. Admission reserves terminal observation
capacity before work; overflow rejects future admission without evicting financial
history. Serialized bundles are also checked before publication. See #255 for the
frozen design and failure semantics.


The instrumented command initializes the channel before observed work, checks the
unchanged stdin request against `channel.bootstrap["request_sha256"]`, constructs
`EvaluationSession.from_channel(channel)`, and supplies that session to its SDK
adapters. It closes its SDK adapters before awaiting `session.close()`. Declaration
payloads come from the channel's detached, digest-verified bounded snapshot; session
construction does not re-read potentially changed declaration files.

Run `python3 -m benchmarks.instrumented_replay --help` for the collector interface.
It requires the original `--plan`, `--collection-spec`, `--inputs`, frozen
`--stream-protocol`, `--allocation-protocol`, `--reconciliation-protocol`, and the
new `--instrumentation-protocol`, together with each corresponding SHA-256 flag
and an exclusive `--output` directory. Hash flags identify pre-frozen artifacts;
they are not permission to alter original gates or replace missing measurements.

`source-traces.jsonl` and `complete.json` retain the original collector format.
`runtime-sidecars.jsonl` carries separate native reports and the valid journal
prefix, bound to the original request, response and source trace. Its completion
artifact is `runtime-complete.json`. `instrumentation_complete` means the bounded
transport inventory closed consistently; it never means complete bills, complete
physical coverage, authentic receipts or actual acceptance. Failed transport can
therefore coexist with an otherwise successful original source response.
