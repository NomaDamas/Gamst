# Attempt recording for evaluation

`benchmarks.call_attempts.AttemptRecorder` is a repository evaluation utility for
explicit application instrumentation. It wraps an async callback; it is not a
model adapter, scheduler, budget gate or automatic provider discovery mechanism.
The [actual SDK client bridge](controller-attempts.md) binds text/visual client
attempts to dispatched controller execution. Complete production/MCP telemetry,
whole-pipeline coverage and actual provider acceptance remain separate.
It has no third-party dependencies and never installs models or makes requests
unless the caller's explicit callback does so.

```sh
python -m examples.attempt_cost_demo
```

This runnable example records one failing synthetic small call and one synthetic
strong fallback, with declared fixture receipts. It does not execute a model or
prove quality/cost savings.

## Frozen scope and invocation

Create one recorder per frozen task/system with an `itemized` evaluation plan,
its independently frozen digest, task ID and baseline/candidate identity. A scope
allows 1–1024 attempts (default 128). Provider and revision must match at least
one frozen rate-card entry. Configurations are copied on construction.

Call `await recorder.invoke(callback, ...)`; callback receives a unique attempt
ID and may return an arbitrary application result. The recorder returns that
result without retaining it. Required metadata includes provider, revision, role
(perception/small_vlm/strong_vlm/text_llm), declared phase (cold/warm/unknown), and
evidence SHA256. Optional prompt and controller execution SHA256 values bind the
application's real artifacts when available; null explicitly means unbound.
The recorder cannot authenticate those artifacts or whether the callback invoked
a model. Native code, remote retries and cache work inside a callback remain
opaque unless separately instrumented and included in receipts.

Every separately wrapped retry or escalation is another attempt. Pass the prior
attempt ID and `relationship: retry` or `escalation`. The predecessor must already
be terminal, in this recorder, and bound to the same evidence. Retry also retains
provider/revision/role. A root must have no predecessor. There is no automatic
retry, timeout, cancellation of provider work or escalation policy.

Errors and cancellation propagate after a terminal record is written; no result
body, raw evidence, prompt or exception message is retained. All records remain
until the recorder is discarded. On capacity exhaustion the callback is not
invoked, a rejection counter is incremented, and complete cost reduction is
refused. Records are never evicted to make an apparently complete cheaper run.
`close()` stops new admission but does not cancel or wait for active callbacks.
Their completion and late billing remain recordable. Attempting a new invocation
after close also marks the scope rejected/incomplete.

Start/end values are monotonic nanoseconds relative to recorder creation. They
measure callback scope only, not capture-to-answer-ready latency. Concurrent
attempts may overlap and complete out of order; do not sum durations as wall time.
Declared cold/warm labels are assertions, not detected cache state. Blocking
callbacks still block their event loop; use the existing bounded provider workers.

## Complete receipts and reduction

`report(attempt_id, quantities=..., receipt_sha256=...)` attaches one complete,
externally verified receipt. Quantities must contain exactly all frozen rate IDs
for that attempt's provider/revision. Every quantity is an exact nonnegative
integer <=1e12. Reports are copied and cannot be overwritten. An error/cancelled
attempt may still have charges and may receive a late receipt after close.
Until then its quantities remain null. Never fabricate a zero receipt for an
unknown bill. Final reporting should await complete provider billing rather than
using an estimate that might need amendment.

After all work/receipts finish, close and snapshot the recorder. Save the report
as a separate bounded-count artifact, retain the frozen plan, and independently
review the report digest. The report can be larger than the paired runner's
response budget; a complete report belongs in an external artifact bound by its
digest, not a truncated inline response. For small fixtures inline reports work.

Use `cost_quantities(plan, report, expected_plan_sha256=..., expected_report_sha256=...,
task_id=..., system=...)` with the specific source trace's task/system. It verifies
the frozen task/input/configuration/cost bindings, report digest, terminal ordered
attempts, lineage, quantities and timing structure. It refuses inflight/unknown
accounting, rejected admissions, changed reports and aggregate quantities >1e12.
It returns all rate IDs, including explicit zero for providers with no recorded
attempts. An empty closed scope returns zero: an external coverage audit must
establish that no model calls occurred outside the wrapper.

Place these quantities into the matching externally reviewed evidence-join
assessment entry. Bind the report artifact(s) through the assessment's source
evidence digest and retain per-attempt receipts. Token diagnostics, memory,
drops, actual quality and task-level latency require their existing independent
measurements; this helper does not invent them. The collector integration test
executes the wrapper inside real subprocesses, then reduces receipts and joins
both source traces; fixture quality remains zero and measured gates fail.

## Limits and remaining acceptance

Hash binding detects changes relative to a trusted reviewed digest; a digest
supplied by the same mutable report is not independent provenance. Scope, unit,
cold/warm, coverage, receipt authenticity and actual model execution require audit.
The utility cannot detect bypass calls, opaque provider retries, falsely declared
zero usage, duplicated billing allocation or fabricated receipts. Unknown local
cost still blocks full-pipeline claims. Rate IDs must reflect mutually exclusive
billing semantics from the frozen protocol.

This is in-memory, with no crash durability, incremental file journal or automatic
export/redaction. Use pseudonymous evaluation task IDs and hashes; hashes are not
anonymization. Explicitly manage artifact retention. HumanSense consent/deletion
controls do not manage these application-owned evaluation artifacts. Never put
personal names, secrets or raw payloads into provider/revision/task labels.

Issues #161/#57/#135 remain open for actual provider integration, systematic replay,
all-call coverage, real cold/warm/resource measurements, quality and full A/B/C
acceptance. Synthetic callbacks and verified report structure cannot complete them.
