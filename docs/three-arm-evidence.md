# Original three-arm source/assessment join

`benchmarks.three_arm_evidence` joins the original A/B/C collector artifacts to an externally frozen assessment, then calls the unchanged `compare_three_arm` numerical wrapper. It never relabels paired execution traces. This is the evidence-join software slice of #205 (#216); receipt/attempt binding and shared allocation remain separate work.

```sh
python -m benchmarks.three_arm_evidence \
  --run /path/to/collected-run --assessment /path/to/assessment.json \
  --plan-sha256 FROZEN_COMPARISON_PLAN_DIGEST \
  --collection-sha256 FROZEN_COLLECTION_SPEC_DIGEST \
  --trace-sha256 EXTERNALLY_RECORDED_FINAL_TRACE_DIGEST \
  --assessment-sha256 EXTERNALLY_REVIEWED_ASSESSMENT_DIGEST \
  --output /path/to/new-joined-output
```

All digests must be frozen outside this join, not computed from an untrusted package solely to make it pass. The run directory must contain `comparison-plan.json`, `collection-spec.json`, `inputs.json`, `source-traces.jsonl`, and `complete.json` from the collector. The join validates portable command declarations and their frozen hashes without requiring historical executables, checkpoints or working directories to remain available. Collector preflight still verifies the existence and bytes of all declared artifacts before executing commands.

The assessment has exact fields `schema_version` (integer 1), `comparison_plan_sha256`, `collection_sha256`, `last_trace_sha256`, `reviewer` (nonempty identifier), `rubric_sha256`, `measurement_protocol_sha256`, `source_evidence_sha256`, and `entries`. Its external digest binds the reviewer identity and evidence digest as well as all scores. Rubric/protocol digests must equal the collection specification. Each entry has exact fields `trace_sha256`, `quality_score`, `correct_critical_facts`, `usage`, `peak_memory_bytes`, `dropped_frames`, and `resource_cost`; itemized mode additionally requires `cost_quantities`. Resource cost is null outside resource mode. Numeric validation and unknown-cost rejection are delegated to the existing numerical gates. No pipeline `measurements` dictionary is automatically trusted or converted into a score/cost.

There must be exactly one assessment per source trace. Original task order, all-six-permutations arm order, ordinals, chain, input/configuration/request hashes, count/position completion metadata and final digest are checked. Failure rows retain zero quality, no credited critical facts, and latency at least `max(process_lifetime_ms, frozen timeout_ms)`; failed work does not become free. Source response hashes refer to original raw response bytes, which the stored parsed response cannot independently reconstruct; source hash binding does not authenticate those bytes.

The exclusively created output directory contains `rows.json`, `lineage.json`, and `join.json`. Each lineage entry binds the original ordinal/task/arm and trace digest to the assessment-entry digest and numeric-row digest. The report includes the lineage digest, assessment digest, numerical wrapper result and both pair projections. Source files remain untouched. Existing outputs are refused, including reruns; a write failure may leave a partial new output directory, which is not a completed join.

Declarations/assessments are limited to 16 MiB; source lines to 512 KiB and 3,000 records. Duplicate JSON keys, nonfinite numbers and incomplete source lines are rejected. Public in-memory arguments are detached and bounded as well.

`source_hash_integrity_verified=true` means only that the supplied frozen declarations and source chain match. `source_verified`, `live_assets_verified`, `receipts_verified`, `independent_scoring_verified`, and `measurement_truth_verified` remain false. The unchanged numerical report also retains its false verification flags. `actual_acceptance` is always `incomplete`, even if numeric gates pass. This join does not certify real inference, sampling execution, assessment independence, measurement truth, receipt authenticity, attempt coverage, shared-resource allocation or recording-cluster uncertainty. #205, #206, #207, #57 and overall acceptance remain open.
