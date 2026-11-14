# Frozen observed stream cost ledger

`benchmarks.stream_cost_ledger` validates a frozen protocol and externally reviewed observation ledger, then reports exact per-stream arithmetic. It does not collect measurements, allocate shared costs, run models or certify that a declared boundary covers the real pipeline. This implements the software ledger slice #229 of #206; allocation and integration remain #230/#231. Existing three-arm task numerical gates and native receipt-reuse rejection are unchanged.

```sh
python -m benchmarks.stream_cost_ledger \
  --protocol /path/to/protocol.json --ledger /path/to/ledger.json \
  --protocol-sha256 EXTERNALLY_FROZEN_PROTOCOL \
  --ledger-sha256 EXTERNALLY_REVIEWED_LEDGER --output /path/to/new-report
```

The new output directory contains `report.json`. Existing outputs are refused. Protocol, ledger, invoice and measurement artifacts are never modified. Hashes must come from independent freeze/review, not merely be recalculated from an untrusted package. The report binds plan/spec hashes as declared identities; it does not load or authenticate the corresponding three-arm source collection.

## Protocol schema 1

The exact top-level fields are `schema_version` (integer 1), `comparison_plan_sha256`, `collection_sha256`, `measurement_protocol_sha256`, `boundary`, `currency`, `currency_unit`, `rate_revision`, `invoice_rounding`, `isolation`, `components`, `rates`, and `streams`.

`boundary` is a nonempty frozen description of included operation. Currency, atomic currency unit and rate revision are explicit strings; no FX or implicit currency conversion exists. `invoice_rounding` must be `line_atomic_half_even`; `isolation` must be `isolated_arms`. Rates are an ID-keyed object, each value exactly `{provider, revision, unit, per_unit}`. `per_unit` is a nonnegative decimal string (up to 64 characters, no exponent) in atomic currency units per atomic quantity. Arithmetic converts it to an exact rational.

Each component is exactly `{id, category, kind, unit, source_unit, atomic_per_source, coverage, rate_id}`. `kind` is `monetary` or `proxy`; `coverage` is `interval` or `event`. `atomic_per_source` is a positive `{numerator, denominator}` ratio. A monetary component names a rate with the same atomic `unit`; a proxy has null `rate_id` and is never added to money. An explicitly priced local measurement can be monetary, but that declaration is not independently verified pricing or cost.

All ten categories must be explicitly represented by components: `perception`, `resident_idle`, `startup`, `capture_sensor`, `media_preparation`, `storage`, `transport`, `vlm`, `text_fallback`, and `local_workers`. Components may subdivide categories. Per-stream requirements declare whether each applies; merely labeling every component not applicable cannot make `whole_pipeline_cost_verified` true.

Each frozen stream is exactly `{id, arm, recording_sha256, clock_domain, requested_duration_ns, execution_namespace, cache_namespace, isolation_evidence_sha256, sources, requirements}`. There must be streams for A, B and C, with at most 1,000 total. Stream IDs and execution/cache namespaces are unique. No execution/cache namespace reuse is supported, including a shared result between arms. Isolation evidence hashes are assertions, not proof of cache reset.

Each source is `{id, semantics, required_counters}`. Supported semantics are `captured_only` and `attempted_equals_captured_plus_dropped`. Counter names are `captured`, `sampled`, `vlm_calls`, `attempted`, `dropped`; `captured` is always required, and attempted/drop counters are required by the second semantics. `sampled` counts unique captured frames, not repeated processing or calls. A call counter can exceed sampled frames; no conversion from frames to model calls is inferred.

`requirements` contains every component ID exactly once. Each value is `{state, reason, lanes}`: `state` is `required` or `not_applicable`, with a nonempty frozen reason. Required components have one or more `{resource_id, lane_id}` entries; not-applicable components have none. Each measurement lane is owned once in the protocol. Different components may meter distinct lanes of one resource, but the same lane cannot be counted twice. A resource ID cannot belong to different arms. The software cannot detect aliases for the same real resource or semantically duplicated measurements; external audit remains required.

## Observed ledger schema 1

Exact top-level fields are `schema_version` (integer 1), `protocol_sha256`, `comparison_plan_sha256`, `collection_sha256`, `reviewer`, `source_evidence_sha256`, `streams`, and `entries`. The reviewer and source evidence are frozen by the reviewed ledger digest.

Each observed stream is exactly `{id, clock_domain, start_ns, end_ns, utc_anchor, availability, counters}` and matches one frozen stream. Actual monotonic `end_ns` must exceed `start_ns`; elapsed time is exactly their difference. It is never the requested duration or the sum of overlapping callback times. A difference from frozen requested duration is explicitly incomplete. `clock_domain` must match the frozen declaration. `utc_anchor` is `{monotonic_ns, unix_ns, evidence_sha256}`; it binds a declared wall-clock anchor without pretending to authenticate clock synchronization.

Availability is a sorted, gap-free, non-overlapping partition of the complete observed interval. Each segment is `{start_ns, end_ns, state, reason, evidence_sha256}` with state `available`, `unavailable` or `unknown`. Known states require evidence; unknown permits null evidence and makes accounting incomplete. Downtime stays in the elapsed denominator. Missing or overlapping availability segments are rejected rather than inferred.

Counters contain exactly one `{source_id, start, end}` record per frozen source. Both endpoint objects contain all five counter fields; each is an exact nonnegative integer or null. Null is explicit unknown. Missing required counters make accounting incomplete; resets/wraps are rejected and require a separate stream. Known deltas must reconcile according to frozen semantics. Unique sampled frames cannot exceed captured frames. Captured frames with entirely unavailable time and no unknown availability are inconsistent and rejected.

Each cost entry has exactly `stream_id`, `component_id`, `resource_id`, `lane_id`, `start_ns`, `end_ns`, `event_ns`, `identity`, `source_quantity`, `quantity`, `status`, `billed_amount`, and `evidence_sha256`.

- Interval components have bounded start/end within observed time and null event time. Event components have a timestamp in `[start,end)` and null start/end. Every required interval lane must be fully covered: unexplained gaps make accounting incomplete; overlaps are rejected as double counting. Required event lanes need at least one event. Distinct physical resource lanes can overlap.
- Source quantity is null or `{numerator, denominator}`. Its exact product with `atomic_per_source` must equal the integer atomic `quantity`, without rounding. A known atomic quantity requires its original source quantity. Nonintegral conversion is rejected; choose an adequate frozen atomic unit rather than approximating.
- Monetary status is `settled`, `pending` or `unknown`; proxy status is `measured` or `unknown`. Known entries require quantity and evidence. Settled money also requires integer `billed_amount`; unsettled bills require null billed amount. Pending/unknown entries remain incomplete even if a partial quantity is known. A proxy always has null billed amount.
- Invoice identity is `{kind: "invoice", provider, revision, account_scope, invoice_id, document_sha256, line_id}`. The same logical document cannot change revision/digest; each logical line is counted once globally. Distinct lines in one invoice may represent distinct charges. Provider/revision must match the frozen monetary rate. Local identity is `{kind: "local", meter_id, measurement_id}` and is also globally unique. Invented alternate IDs cannot be detected as duplicates without external evidence.

For monetary entries, rate-derived charge is exact quantity × frozen price. The invoice contract rounds **each line** to the nearest atomic currency integer with ties to even. If it does not equal declared billed amount, the report remains incomplete; no arbitrary tolerance or hidden fee reconciles the discrepancy. Discounts, fees or a different aggregation boundary need an appropriately frozen component/rate/protocol, not a post-result rule change. All billed amounts remain separately visible from exact rate-derived totals.

## Report semantics and bounds

Per stream, the report includes actual elapsed and available/unavailable/unknown nanoseconds, capture/call counter deltas, captured-per-elapsed-second and captured-per-available-second with explicit denominators, known monetary subtotal, exact rate-derived subtotal, separate proxy totals, component totals and incompleteness reasons. The available-time rate is null when availability is unknown or available time is zero.

`known_subtotal_per_elapsed_hour` is exactly `known_monetary_atomic × 3,600,000,000,000 / elapsed_ns`. It remains a **known subtotal**, including when some charges are missing. `declared_complete_cost_per_elapsed_hour` is null unless all declared requirements reconcile. Fractions are returned as `{numerator, denominator}`, without float/display rounding. No daily extrapolation is emitted in v1. Proxy/hour values retain their own units and are never added to currency. Canonical arm totals include each stream once; they are not sums of B_vs_A and C_vs_A reports. Startup, idle and non-task overhead remain explicit components, with no task allocation.

`declared_accounting_complete` and each stream's `complete_accounting` mean only completeness against this frozen declaration. They do not establish that it is an adequate real-world boundary. `structural_hash_integrity_verified` means the supplied frozen artifacts and structural checks match. `receipt_authenticity_verified`, `measurement_truth_verified`, `isolation_verified`, `instrumentation_coverage_verified`, and `whole_pipeline_cost_verified` remain false. `actual_acceptance` always remains `incomplete`; `allocation_implemented` is false.

Each input is bounded to 16 MiB, with at most 1,000 streams, 256 components/rates/sources/lanes per relevant declaration, and 10,000 combined availability, counter and cost entries. Integer inputs are in `[0, 2^63−1]`, with positive denominators; exact intermediate rationals are bounded to 256-bit numerators/denominators. JSON duplicate keys, nonfinite values, booleans in integer fields and extra fields are rejected. APIs detach inputs before validation. Unknown required components, missing cost coverage, pending bills or unreconciled rounding produce explicit incomplete reports; malformed identities, partitions, duplicate charge lines and invalid quantities fail closed.

Resource declarations also have a global 10,000-lane limit. IDs and unit labels
are limited to 256 characters. Repeated incompleteness findings for the same
stream and reason/component are aggregated with an exact `occurrences` count;
for coverage gaps this counts deficient interval segments, not inferred lost
frames. Each affected component remains visible, and the frozen protocol retains
its resource-lane identities for audit. Diagnostics are never silently dropped.
The complete compact JSON report, including its final newline, is limited to
16 MiB before either API return or CLI output creation. Oversized reports fail
closed. Both `report.json` and CLI stdout use that same bounded compact encoding.

[Within-stream allocation](stream-cost-allocation.md) is an additive protocol for
exact attribution of this ledger's known quantities and billed amounts. It leaves
original invoices and this ledger validator unchanged; source-attempt and numeric
integration remain separate work.
