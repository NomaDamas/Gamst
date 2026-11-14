# Frozen A/B/C numerical comparison

`python -m benchmarks.three_arm_comparison` evaluates declared rows for:

- A: periodic VLM baseline, one frame opportunity every1000ms.
- B: specialist-only diagnostic ablation.
- C: specialist plus configured VLM policy, the primary candidate.

This is a numerical wrapper, not a three-arm inference runner. It never rewrites
collected source traces or receipts. Numeric gates cannot establish real acceptance:
`actual_acceptance` is always `incomplete`, and source/receipt/sampling-execution/
recording-cluster-uncertainty verification is false. Models are not installed or run.

## Frozen inputs

The plan has exactly these fields:

- `schema_version`: integer1.
- `primary_comparison`: `C_vs_A`.
- `comparison`: the existing paired evaluator's `evidence`, `held_out`,
  `evaluation_id`, `scoring_protocol`, `latency_protocol`, `trace_provenance`,
  `min_pairs`, `bootstrap_repetitions`, `bootstrap_seed`, `tasks`, and `cost`.
  See [paired evaluation](perception-evaluation.md);
  the executable schema authority is `benchmarks/perception_comparison.py`.
- `systems`: exactly `A`, `B`, `C`. Each configuration has `model`, `revision`,
  `precision`, `hardware`, `policy`, `warm_state`, `sampling`, `window_lookahead`.
  A's sampling string must be `periodic_1fps`.
- `baseline_sampling`: exactly `period_ms:1000`, integer `phase_ms` in0..999,
  mandatory `time_origin_policy` (`recording_start` or `capture_epoch`),
  and nonempty `audio_policy`/`task_policy` strings. `recording_start` measures phase
  from the first capture timestamp of each immutable recording; `capture_epoch`
  measures phase from zero in that recording’s capture timestamp domain. These
  produce different sampling schedules and are frozen into both derived plans.
  Frame opportunities do not
  imply one billed request per frame; batching/cache accounting stays separate.

Rows are exactly3 per declared task, one for each arm. Each row contains:
`task_id`, `arm`, `plan_sha256`, `configuration_sha256`, `input_sha256`, `latency_ms`,
`quality_score`, `correct_critical_facts`, `peak_memory_bytes`, `dropped_frames`,
`usage`; additionally `resource_cost` only in resource mode or `cost_quantities`
only in itemized mode. Configuration digest covers the selected `systems[arm]`.
Input digest must match the task declaration. Unknown/extra/duplicate rows and
fields, changed hashes, invalid existing numerical contracts fail closed.

Use `benchmarks.perception_comparison.plan_digest` to compute canonical hashes;
these are hashes of parsed values, not raw file bytes. Freeze the plan before
collection, then freeze the complete reviewed row package. Neither digest proves
when it was frozen, independent scoring, actual inference or measurement truth.
File/API inputs are bounded to16MiB each. CLI rejects duplicate JSON keys and
nonfinite constants. Existing input files are read only; output goes to stdout.

```sh
python -m benchmarks.three_arm_comparison \
  --plan /absolute/abc-plan.json --rows /absolute/abc-rows.json \
  --plan-sha256 <frozen-canonical-plan-sha256> \
  --rows-sha256 <frozen-canonical-rows-sha256>
```

## Preserved gates and evidence boundaries

Each deterministic A/B and A/C projection uses the unchanged paired evaluator.
The report preserves source-row hashes, derived-row hashes, derived plan/row-package
hashes and the structured baseline declaration. A appears once in the input and
once in the three-arm summaries, reused identically in both pair computations.
Do not sum baseline costs from both pair reports. The wrapper does not aggregate
an experiment total, allocate shared sensing costs, reconcile receipts or prevent
all forms of double billing; those remain unverified.

`primary_meets_numeric_gates` is solely C versus A: existing20% cost and10% p95
latency reduction gates with paired intervals, critical facts and quality gates.
B versus A is explicitly diagnostic only. Unknown cost and synthetic evidence
cannot pass the primary gates. Existing bootstrap resamples task pairs; it does
not provide recording-cluster uncertainty for correlated tasks. Shared recordings
need the separately frozen cluster protocol before actual acceptance.

The next stages are native three-arm collection/source joining/attempt binding
[#205](https://github.com/NomaDamas/Gamst/issues/205), stream-hour/shared-cost allocation
[#206](https://github.com/NomaDamas/Gamst/issues/206), and held-out real-model evidence
[#207](https://github.com/NomaDamas/Gamst/issues/207). This software slice
[#204](https://github.com/NomaDamas/Gamst/issues/204) does not close
[#135](https://github.com/NomaDamas/Gamst/issues/135) or the existing real acceptance
[#57](https://github.com/NomaDamas/Gamst/issues/57).
