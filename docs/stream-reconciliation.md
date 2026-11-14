# Declared stream reconciliation (schema 1 protocol, schema 2 manifest)

`python -m benchmarks.stream_reconciliation` joins original A/B/C source traces,
external assessment, native attempt reports, whole-stream ledger and frozen
allocation. It produces one exclusive `report.json`, only after every required
numeric reconciliation succeeds. Existing paired v1, native v2, collector,
assessment, ledger and allocation schemas remain unchanged.

This is software verification of declared accounting. Receipt authenticity,
real inference, independent scoring, instrumentation coverage, isolation,
recording independence and actual savings remain unverified. `actual_acceptance`
is always `incomplete`. A hash binding does not prove a declaration existed
before execution or that a callback produced the collector's response.

## Freeze order and scope

Freeze comparison plan P, collection C, stream protocol S, allocation Q, then
reconciliation R. R binds P/C/S/Q; native v3 reports bind R/S/P/C plus original
configuration, recording, arm and task/input (or stream scope with null task/input).
R is never inserted into C: that would create a hash cycle. Freeze the completed
ledger, reports, assessment and integration manifest independently. The manifest
associates original collector traces and reports; it does not rewrite them.

Every original task has exactly one A/B/C target and report. Its three targets
must belong to the same matched recording. Every stream also has exactly one
stream-scope report, including an explicitly closed empty report. Canonical A is
counted once before either pair projection.

## Reconciliation protocol

The exact top-level fields are:

```
schema_version: 1
comparison_plan_sha256, collection_sha256, stream_protocol_sha256,
allocation_protocol_sha256
currency_atomic_per_comparison_unit: {numerator, denominator}
rate_map: [{component_id, comparison_rate_id,
            atomic_quantity_per_comparison_unit: {numerator, denominator}}]
lane_owners: [{stream_id, component_id, resource_id, lane_id, owner_kind}]
provider_counter_groups: [{id, stream_id, provider, revision, role}]
vlm_counter_bindings: [{stream_id, source_id, counter_ids}]
recording_groups: [{recording_sha256, streams: {A, B, C}}]
whole_stream_cost_policy: {
  version: "matched_recording_elapsed_hour_v1",
  threshold: {numerator: 4, denominator: 5},
  interval: "nearest_rank_95",
  bootstrap_seed, bootstrap_repetitions, min_recordings
}
```

Every monetary component maps once, and every comparison rate is covered.
Currency names, provider and revision must match. Positive reduced quantity and
currency conversion ratios relate the frozen unit names; identical unit names
require identity conversion. Source atomic rate × quantity scale must equal
comparison rate × currency scale exactly. Required proxy components refuse
numeric reconciliation; a proxy cannot become a zero monetary cost.

Lane owner kinds are `task_attempt`, `stream_attempt`, or `non_attempt`.
VLM/text-fallback lanes cannot be non-attempt lanes. Each required resource lane
has exactly one frozen owner kind. Counter groups are unique by both ID and the
full stream/provider/revision/role tuple. Every attempt belongs to one group,
including retries, errors and cancellations. Every stream binds a required
`vlm_calls` source counter to all of its small/strong VLM groups, including an
explicit empty group list for known zero calls.

## Native reports and ownership

`StreamAttemptRecorder` subclasses the existing recorder for SDK client use.
It takes P/C/inputs/S/R, their expected hashes and `scope_kind`, `task_id`, `arm`,
`stream_id`. Its `report_lines(attempt_id, quantities=..., line_claims=...,
execution_instance_sha256=...)` accepts complete provider quantities and unique
original ledger-entry digests. Billing may arrive after callback termination or
scope closure. Unknown snapshots can be retained, but cannot pass the strict
join. Native v3 forbids the old whole-receipt `report()` API.

The schema 2 integration manifest has exactly these fields:

```
schema_version: 2
comparison_plan_sha256, collection_sha256, stream_protocol_sha256,
allocation_protocol_sha256, reconciliation_protocol_sha256,
last_trace_sha256, ledger_sha256, assessment_sha256,
reviewer, coverage_evidence_sha256
trace_reports: [{trace_sha256, report_sha256}]
stream_reports: [{stream_id, report_sha256}]
attempt_instances: [{report_sha256, attempt_id,
                     execution_instance_sha256, evidence_sha256}]
line_owners: [attempt owner OR non-attempt owner]
provider_counters: [{counter_id, count, evidence_sha256}]
```

An attempt owner has `entry_sha256, owner_kind, report_sha256, attempt_id`.
A non-attempt owner has `entry_sha256, owner_kind, reason, evidence_sha256`.
Inventories are exact: no omitted/duplicate reports, attempts, instances, groups
or ledger lines. Each original line has one owner; native v3 stored claims must
match that owner set. Instances are globally unique external declarations, not
proof of actual execution uniqueness.

Native v2 task reports may participate without alteration. Their receipt digest
must match the complete invoice document represented in the ledger, and all
represented lines in that document (including zero lines) belong exclusively to
that one v2 attempt. Another v2, v3 or non-attempt owner cannot share that document.
Native v3 can own distinct lines of one document. An indivisible shared line
cannot be assigned to several actual attempts without original line-level
ownership evidence. Allocation of a genuinely stream-owned charge is separate
from attempt ownership.

Task-owned lines require a direct rule to their original task/arm. Failure never
turns a task charge into a stream or overhead charge. Stream/non-attempt lines may
use the previously frozen exact allocation. Final task quantities come from
allocation once, never attempt totals plus allocation totals. Assessment quantities
must match them exactly. Original source failure policy still gives failed traces
zero quality/no critical facts and the maximum of lifetime and timeout latency.

## Exact money and whole-stream decision

Every source line and every allocated line must independently equal its exact
comparison-unit charge after currency conversion. Fractional comparison
quantities, quantities above 1e12, and billed-versus-rate discrepancies refuse the
numeric artifact. Signed discrepancies cannot cancel across lines. Original
invoice rounding is checked only at the original line by the ledger validator;
allocated charges are never treated as new invoices.

Task quality/latency/cost results remain the unchanged numeric comparator's
attributed-task diagnostic. The combined decision additionally requires a separate
whole-stream 20% cost gate. It includes every original monetary charge, including
explicit startup/idle/overhead targets outside the task set. Cheap tasks cannot
hide expensive idle time.

Whole-stream groups contain every stream once, matched by original recording
and A/B/C. Capture source IDs, semantics and required-counter contracts must match;
all captured/attempted/dropped deltas must be known and equal even when optional
in the underlying ledger schema. Relative availability partitions and requested/
observed elapsed durations must match. Sampling and VLM counts may differ.

Costs use pooled observed elapsed-hour denominators. Bootstrap draws resample
matched recordings together, ordered by recording digest, using the original
seed and repetitions. Minimum recordings equals original `min_pairs`; multiple
tasks from one stream are not independent cost replicates. Both the exact point
ratio and upper 95% bound must be at most 4/5. Quantile indices are
`ceil(.025*N)-1` and `ceil(.975*N)-1` (zero-based). Every baseline must be positive;
any zero-baseline draw yields an undefined-draw diagnostic, null interval and
failed gate, never a selectively narrowed interval. Stream correlation with task
quality or latency is not certified.

## CLI and bounds

Supply `--run` (original collector directory), `--assessment`, `--stream-protocol`,
`--ledger`, `--allocation-protocol`, `--reconciliation-protocol`, `--manifest`,
`--reports` (JSON array), and all nine external `--*-sha256` flags: plan,
collection, trace, assessment, stream-protocol, ledger, allocation-protocol,
reconciliation-protocol, manifest. `--output` must not exist.

Inputs remain immutable. Strict JSON rejects duplicate keys and nonfinite values.
Declarations/reports and output are bounded at 16 MiB; original source lines stay
bounded at 512 KiB and 3000 rows. Original ledger/allocation bounds apply, plus
10000 canonical attempts/claims, 100000 combined reconciliation references,
256-bit exact rational intermediates and one million sampled recording draws.
The CLI validates and serializes before creating output; numeric refusal creates
no output directory. Compact report/attempt/line index tables preserve lineage
without repeating long identities per allocation edge.
