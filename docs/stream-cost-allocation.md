# Frozen within-stream cost allocation

`benchmarks.stream_cost_allocation` attributes the original observed ledger's known integer quantities and billed amounts to frozen targets. It preserves original invoice-line identity and counts canonical A once. This is accounting attribution: it does not show that removing a task would save its attributed cost or prove same-quality cost/latency gains. Native receipt reconciliation and numeric integration remain #231; the existing direct-receipt join still rejects shared receipt reuse.

```sh
python -m benchmarks.stream_cost_allocation \
  --plan /path/to/three-arm-plan.json --stream-protocol /path/to/stream-protocol.json \
  --ledger /path/to/ledger.json --allocation-protocol /path/to/allocation-protocol.json \
  --plan-sha256 FROZEN_PLAN --stream-protocol-sha256 FROZEN_STREAM_PROTOCOL \
  --ledger-sha256 REVIEWED_LEDGER --allocation-protocol-sha256 FROZEN_ALLOCATION_PROTOCOL \
  --output /path/to/new-allocation
```

The original three-arm plan and [stream ledger](stream-cost-ledger.md) are validated through their unchanged validators. The frozen allocation protocol binds the comparison plan and predeclared stream protocol. It does not bind the later observed ledger before collection. The reviewed ledger digest is supplied separately and bound in the result. Rules and task windows must be frozen before observing outcomes; hashes supplied from the same mutable inputs do not prove that chronology.

## Exact schema 1

Allocation protocol fields are exactly `schema_version` (integer 1), `comparison_plan_sha256`, `stream_protocol_sha256`, `rounding`, `tie_break`, `duration_model`, `targets`, and `rules`. Required values are `largest_remainder_v1`, `unicode_codepoint_target_id`, and `uniform_line_quantity_over_interval` respectively.

Each target is `{id, stream_id, kind, task_id, windows}`. Each original comparison task must have exactly one target in each original arm A/B/C. Its declared stream must belong to that arm. Each stream additionally has exactly one `startup`, `idle` and `overhead` target with null task ID and no windows, even if its total is zero. Task targets have kind `task` and the original task ID. Target IDs are unique, unchanged strings; no Unicode normalization or locale collation is applied.

Task windows are sorted non-overlapping `{start_offset_ns, end_offset_ns}` intervals relative to the actual observed stream start, bounded by its frozen requested duration. They are frozen task/capture observation windows, not callback durations or answer-success intervals. Direct/fixed targets may have no windows. Duration-roster tasks require nonempty windows. The task-to-stream association is a declaration, not source execution proof; #231 supplies separate reconciliation.

Every required original resource lane has exactly one rule. All rules include `stream_id`, `component_id`, `resource_id`, `lane_id`, and `method`, plus only their method's fields:

- `direct` has `target_id`.
- `fixed` has `shares`, each `{target_id, numerator, denominator}`. Shares are unique reduced nonnegative fractions summing exactly to one.
- `duration` has `task_target_ids` and `idle_target_id`. Its task roster is unique, nonempty and frozen; the fallback is an explicit idle/overhead target. It applies only to interval components. Startup/event sources use direct or fixed rules.

Every referenced target belongs to the original source stream. Cross-stream and cross-arm allocation is rejected. The underlying ledger rejects reused execution/cache namespaces, cross-arm resources and duplicate invoice-line/local-measurement identities. No whole receipt is duplicated or reinterpreted as multiple actual charges.

## Conservation and rounding

Direct assigns weight one. Fixed uses the frozen rational shares. Duration clips frozen task windows to the source interval and splits at every boundary. Each segment is divided equally among active roster tasks; a segment with none goes entirely to the fallback. Segment fractions are then summed into one exact weight per target. This explicitly assumes a uniform quantity over the source interval. Nonuniform measurements must first be represented as finer original ledger entries; allocation does not invent that information.

For each source line, integer atomic quantity and settled billed money are allocated **independently**. Each exact weight×total is floored, then remaining atomic units go to descending fractional remainders. Ties use lexicographic Unicode code-point order of unchanged target IDs, including non-BMP characters. The remainder loop is bounded by the number of recipients. No float, epsilon, discarded residual or result-based tie choice exists.

Original invoice half-even rounding remains solely in the source ledger validator. Allocated target charges are not new invoices and are not rounded again. Each target's allocated quantity times the frozen rate yields an exact rational charge; its signed `billed_minus_rate_derived_atomic` difference is retained. A nonzero difference is ordinary attribution-rounding information, not an invented fee or silently reconciled cost. #231 must separately determine whether quantities and monetary attribution can legitimately enter the unchanged task numeric wrapper.

For example, over `[0,8)`, task A active `[0,6)` and B active `[2,4)` produce weights A=`5/8`, B=`1/8`, idle=`1/4`. Quantity 7 allocates `4,1,2`; billed amount 4 allocates `3,0,1` when IDs sort A, B, idle. At rate 0.5, derived charges are `2,0.5,1`, with signed differences `1,-0.5,0`. Both original totals are conserved without claiming every target's billed amount equals its rate-derived charge.

Proxy quantities remain attached to their original component/unit and have null monetary fields. Every known source quantity/billed amount and each stream's known component/monetary total is reconciled against the sum of target allocations. Startup, idle and overhead remain explicit targets. Canonical A is processed once; no pair comparisons or totals are added together.

## Unknowns, output and limits

Pending/unknown original lines retain their deterministic exposure shares and null `allocations`. Each affected target lists the original source index. Moreover, any incomplete source stream conservatively makes all of its targets' `declared_complete_cost_atomic` null, including when a required measurement interval is absent and therefore has no source line. Known subtotals are still reported as known subtotals. Unknown quantities are never silently allocated as zero.

The exclusively created `report.json` binds all four input digests plus the source-ledger-report digest. Compact indexed tables hold targets, components and rule identities; source rows retain original ordinal, entry and identity digests, rule/component references, exact shares and allocations. Edges use target indexes instead of repeating long IDs. Target totals are indexed by target and component. Original inputs and receipts are unchanged. Both report and stdout use the same bounded compact JSON with a final newline.

All input/output artifacts are at most 16 MiB. Limits include 10,000 targets, total frozen windows and rules, 100,000 frozen rule references, and 100,000 combined processing steps (line visits, fixed-share visits, window inspections, sweep distribution and allocation edges); allocation edges also have a 100,000 cap. Budgets are checked before expanding each unit of work, including pending/unknown lines. Output byte accounting is incremental and checked again on the complete serialized report before output creation. IDs retain the ledger's 256-character bound, integer inputs are bounded by `2^63−1`, and exact rational intermediates by 256-bit numerators/denominators. Strict JSON, extra-field, duplicate-key and nonfinite rejection remain in effect. Oversized reports are rejected, never truncated.

`known_line_conservation_verified` covers only the declared known lines. `declared_allocation_complete` cannot exceed the original ledger's declared completeness. Receipt authenticity, measurement truth, isolation, instrumentation coverage and whole-pipeline verification remain false. `numeric_integration_implemented` is false and `actual_acceptance` is always `incomplete`. Allocations can make task costs statistically dependent; this software does not validate recording-cluster uncertainty or prove real 1fps A / specialists-only B / hybrid C same-quality savings. #206/#207/#57 and the Goal remain open.
