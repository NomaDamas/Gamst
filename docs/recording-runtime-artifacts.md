# Runtime artifacts and coordinator1

These are storage and transport components for the approved persistent recording
runtime. They do not activate a provider, score a result, settle a bill, attest
backend/GPU entry, or prove child process reap. Actual evaluation and dataset
downloads remain subject to the repository's GCP-only policy. Local tests use
synthetic callbacks and subprocess fixtures, not real measurement claims.

`RecordingRuntimeArtifacts.open(output, execution, budget=shared_budget,
retain_evaluation_artifacts=True)` creates one exclusive0700 directory with0600
files. Consent covers separately retained evaluation records; SDK deletion still
scrubs its own private state and does not erase these immutable external files.
The supplied execution declaration is copied before use. No caller-selected
artifact path is accepted.

`record_tick`, `record_adapter`, `record_delivery`, `record_delivery_summary`,
`record_channel`, `record_transport`, and `record_mapping_update` synchronously
admit detached bounded rows. `record_publication` takes the actual accepted SDK
record and, for backend-produced observations, an observer-issued
`NativeOperationRef`. Its immutable handle exposes the record's SDK object hash;
`await publication_ready(handle)` returns the sealed **publication row** Ref.
The original accepted record is a separate immutable file. Deterministic raw
capture/task snapshot publications use an explicit pinned builder and no invented
native operation. The runtime join must independently verify captures, actual
operation ownership and accepted-publication provenance.

`await write_task(ordinal, kind, value)` accepts the fixed context/envelope/record/
evidence/result/receipt kinds. `write_fill_case` uses `fill/case-NNNNNN.json`.
`write_request` retains a separately bounded original request only under the
explicit external retention consent. Ref hashes cover canonical ASCII JSON plus
LF; SDK record hashes use the SDK's existing `_digest`; answer hashes cover raw
UTF-8 answer text, without LF.

`seal_inventory()` drains owned writes and returns fixed outcome index refs.
Only final child/run summaries may be written afterward; the child summary must
retain the exact sealed indices and all written task results. Then call
`close(cleanup_finished=True)`. Abort and cancellation join owned write work;
previous immutable files remain available after failure. An incomplete prefix
cannot become complete merely because cleanup succeeds.

The outcome grammar is the approved common OutcomePage/OutcomeIndex grammar,
including publication rows stored through Ref-only pages. The omitted filename
for the already approved delivery-summary kind is explicitly
`delivery-summaries-index.json` and
`delivery-summaries/{page,index}-NNNNNN.json`, matching the other finite kinds.
Readers enforce contiguous chains, nonempty data pages, full nonfinal128-ref
index pages, exact byte hashes and regular nofollow files. Original source and
native refs use their fixed `source/` and `native/` paths and must be read with
their unchanged format readers; runtime JSON readers never relabel those bytes.

Admission bounds are64 queued/inflight jobs and1MiB, <=2048B outcome rows,
<=128rows/64KiB pages, <=2048 combined data/index pages and16MiB retained runtime
metadata per run. Small per-kind page builders are separately bounded by64KiB
each. Context artifacts are <=32768B, records/envelopes <=128KiB, requests
<=256KiB, and task snapshots <=4MiB aggregate. All retained bytes are charged to
the supplied shared4GiB ArtifactBudget; failures do not refund retained files.
`preflight_runtime_artifacts` provides conservative per-run page/byte bounds;
the parent still combines these with source/native/memory reservations before
activation and independently checks the collection-wide inventory.

`ParentCoordinatorChannel(bootstrap).spawn_options()` returns only the owned
socket FD and `GAMST_RECORDING_RUNTIME_FD_V1` environment delta. The fixed runtime
launcher combines this with native2's FD, uses `close_fds=True`, and calls
`await spawned()` immediately after successful spawn. Child construction takes
the original trusted startup deadline, reads the parent-first bootstrap, and
marks the inherited FD noninheritable before any nested model process starts.

`send` admits one exact fixed frame and returns an owned send-completion future;
`send_wait` waits for it. `receive` returns an immutable parsed frame plus its
complete-frame diagnostic timestamp. The parent must validate original result
and evidence artifacts before taking the actual acceptance timestamp. Neither
the raw receive timestamp nor a child's finish time substitutes for acceptance.
START-begin, START-complete and playback-receive are independently preserved;
the anchor must lie between START-begin and playback-receive. The clock
implementation string is compared with the actual local monotonic clock.

The coordinator has its own64-frame/1MiB queues, separate from native2 and source
writing. Both channels must be drained concurrently by the fixed parent. A
runtime-owned `wait_stop()` watcher stops admission on cancel/disconnect/failure.
Malformed/duplicate/out-of-order/trailing frames fail closed while parsed prefix
frames remain available. `close` drains and closes only owned channel resources;
`abort` closes before cancelling/joining tasks to handle Python3.10 cancellation
races. Process-group cleanup and actual reap remain the fixed launcher's job.

Validation includes actual Gamst accepted records, an actual native4
callback joined to an accepted record, immutable artifact aliases and mutations,
queue pressure, exclusive paths/FIFO/symlink rejection, write failure and repeated
cancellation, and a real fixture subprocess using the fixed channel handshake.

The fixed parent is `python -m benchmarks.recording_coordinator --runtime-inputs
ABSROOT --output NEWROOT --expected-execution-sha256 ORIGINAL_INVOCATION_ZERO_SHA
--retain-evaluation-artifacts`. Financial retention additionally requires the
original U2/Q2 consent and `--retain-financial-documents`. Actual evaluations and
dataset acquisition remain GCP-only; synthetic local fixtures establish software
behavior only. No commands, modules, models, or downloads are selected by input.

The parent preflights every original recording and all joint source/native/runtime
copy bounds before creating output. Each invocation owns one fixed child, two
bounded channels, stdout/stderr draining, and cancellation-resistant group reap.
A task receipt is stamped only after its original result, evidence, snapshot,
envelope and record bytes are validated. Final native/source/publication joins
run before the collection can be structurally complete. Failure latency keeps
the original task floor and later observed failure. Child exit zero alone is
insufficient evidence.

Child files remain under `.staging/NN`; byte-identical imports are published under
`runs/NN`. The parent alone writes `native-inventory.json`, `handshake.json`, task
receipts, channel rows and `run-inventory.json`. Native inventory is the actual
parent-issued channel/reap snapshot, including null terminal on failed prefixes.
Source and native schemas are unchanged. Both retained copies consume the
collection budget. The exclusive final execution inventory includes actual
retained byte counts, remains immutable with `assessments: []`, and does not
claim numeric or actual acceptance. Subsequent assessments belong to separate
artifacts.

The Python collection API also requires explicit
`retain_evaluation_artifacts=True`; omission fails before opening inputs. Frozen
credential environment names must be `API_KEY`, `TOKEN`, or uppercase names
ending `_API_KEY`/`_TOKEN` (bounded to 121 characters), and may never use Python,
loader, transport, or process-activation prefixes. Other inherited search-path
and loader variables are not forwarded. Selected Smol startup uses the original
frozen timeout in addition to the parent's 30-second generic startup ceiling;
this deadline is never restarted and does not shift original task clocks.

The nonwire result preserves `validated_runs` as a tuple of original
validator-issued objects and `artifact_budget` as the same shared budget owner.
These are not serialized or reconstructed from JSON. Finance must reuse that
budget for later separate artifacts. Parent preflight accounts for original
input workspace, each run's peak join, and all previously retained typed runs;
retained runs are bounded jointly to 128 MiB and the combined parent workspace
to 256 MiB. Structural completeness includes the typed join's completeness,
while assessments and numeric eligibility remain unchanged in the sealed
execution inventory.
