# Native three-arm attempt and receipt binding

`benchmarks.three_arm_attempts` records native schema-2 A/B/C attempts and joins their itemized receipt quantities to the original three-arm collector and frozen assessment. Original reports, traces and assessments remain immutable. The existing paired schema-1 recorder/reducer API and numerical gates are unchanged.

Create `ThreeArmAttemptRecorder(plan, spec, inputs, expected_plan_sha256=..., expected_collection_sha256=..., task_id=..., arm=...)` for each original task/arm. The plan must use itemized costs. The recorder validates portable frozen declarations without opening historical assets; the collector continues to perform live executable/artifact preflight. `invoke`, `report`, `close`, `snapshot`, retry/escalation and late-receipt behavior match [the paired recorder](call-attempts.md). The native recorder is compatible with `RecordedContextClient`, preserving dispatched prompt/execution hashes. It does not automatically discover, execute or retry providers.

A native report has `schema_version: 2`, `comparison_plan_sha256`, `collection_sha256`, `cost_sha256`, `task_id`, `arm`, `input_sha256`, `configuration_sha256`, `evidence`, `scope`, `max_attempts`, `closed`, `rejected`, `attempts`, `clock`, and `sha256`. Its digest covers every field except `sha256` itself, using `plan_digest`. Do not relabel a paired report: native scopes must originate with A/B/C binding. Shared code validates attempt bodies while each format validates its own original header.

`cost_quantities(plan, spec, inputs, report, expected_plan_sha256=..., expected_collection_sha256=..., expected_report_sha256=..., task_id=..., arm=...)` checks the native header and reduces every terminal attempt. Errors and cancellations may receive receipts after close and remain charged. Unknown, inflight, missing receipts, rejected admissions, invalid retry lineage and duplicate receipt identities fail closed. An empty closed report explicitly returns zero recorded quantities; it cannot prove there were no hidden or uninstrumented calls.

## Frozen external manifest

After collection, retain every report and receipt artifact. Independently review and freeze a manifest with exact fields:

- `schema_version`: integer 1.
- `comparison_plan_sha256`, `collection_sha256`, `last_trace_sha256`, `assessment_sha256`: the original frozen source and assessment bindings.
- `reviewer`: nonempty reviewer identifier; `coverage_evidence_sha256`: retained external coverage-audit evidence digest.
- `entries`: exactly one `{trace_sha256, report_sha256, coverage_statement}` per original source trace. Each statement is nonempty and must describe the declared instrumentation scope, including an explicit explanation for empty scopes.

The software requires the statement and freezes its text; it cannot judge whether the explanation is true or adequate. The manifest digest must come from external review, not be regenerated from an untrusted artifact just to pass validation. The assessment remains separately frozen: the manifest links its digest and the report digest without rewriting its existing source-evidence digest or creating a circular hash dependency.

The reports file is a JSON list containing exactly one native report per source trace. The join revalidates the original source/assessment package, then checks the manifest and reports and requires report totals to equal each assessment's `cost_quantities`. The manifest is an externally reviewed association, not a cryptographic proof that the reported callbacks produced the source answer. It does not turn a collector subprocess's opaque measurements into verified accounting.

## CLI and outputs

```sh
python -m benchmarks.three_arm_attempts \
  --run /path/to/original-run --assessment /path/to/assessment.json \
  --manifest /path/to/manifest.json --reports /path/to/reports.json \
  --plan-sha256 FROZEN_PLAN --collection-sha256 FROZEN_SPEC \
  --trace-sha256 FROZEN_FINAL_TRACE --assessment-sha256 FROZEN_ASSESSMENT \
  --manifest-sha256 EXTERNALLY_REVIEWED_MANIFEST --output /path/to/new-output
```

Only a new output directory is accepted. It contains `rows.json`, source `lineage.json`, `attempt-lineage.json`, and `join.json`. Attempt lineage binds original ordinal/task/arm, trace digest, report digest and numeric row digest. The manifest and lineage digests, per-arm `canonical_cost_quantities`, `recorded_attempts`, and `unique_recorded_receipts` appear in the report. Original A is processed once alongside B and C; both downstream pair projections reference its single source row. Never add the two pair reports to estimate total actual cost.

Recorder scopes, attempt IDs and `(provider, revision, receipt_sha256)` identities must be unique across the entire original 3N report set. Reusing a receipt across arms or tasks, including a shared invoice, is refused. No shared-resource allocation is inferred; that protocol remains #206. Zero quantities are accepted only as explicit receipt quantities or an explicit empty recorded scope, never as a replacement for missing bills. Failure quality and latency floors remain those of the original evidence join.

Declarations, assessments, manifest and the reports list each have a 16 MiB limit, with 1–1,024 attempts per recorder and the existing 512 KiB/3,000 source bounds. CLI readers reject duplicate JSON keys and nonfinite values; APIs detach and bound their JSON inputs. Failed output creation can leave an incomplete newly created directory; existing files are never overwritten.

## Meaning of verification flags

`attempt_binding_verified` and `recorded_receipt_totals_verified` cover only the declared original association, structural report checks and exact reduction. `source_hash_integrity_verified` retains the source join's limited hash meaning. `receipts_verified`, `instrumentation_coverage_verified`, `source_verified`, `live_assets_verified`, `independent_scoring_verified`, and `measurement_truth_verified` remain false. `actual_acceptance` always remains `incomplete`.

A successful callback does not prove model execution or answer correctness. Receipt hashes do not authenticate a provider or bill. Native/provider retries, discarded work, hidden workers and calls outside instrumentation can be absent despite a complete declared scope; external coverage evidence is not automatically certified. Whole-stream resource accounting, real held-out inference, independent scoring, sampling execution and recording-cluster uncertainty remain outstanding. This software slice (#218) does not by itself close #205, #206, #207, #57 or the overall Goal.
