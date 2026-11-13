# Matched perception/VLM evaluation

`benchmarks.perception_comparison` evaluates paired traces against a frozen declaration. It implements numerical checks for P08/#10; it does **not** run inference, attest hardware measurements, verify held-out provenance or replace independent scoring review. Actual model integration and matched recorded-input runs remain required. Synthetic test traces cannot pass the combined gates.

Before collecting results, commit a plan containing:

- `schema_version: 1`, a unique `evaluation_id`, `evidence` (`measured` or `synthetic`), `held_out`, `trace_provenance`, `scoring_protocol` and `latency_protocol`.
- `systems.baseline` and `systems.candidate`, each with nonempty strings for `model`, `revision`, `precision`, `hardware`, `policy`, `warm_state`, `sampling`, `window_lookahead`. Identify all model/provider versions and configuration in these descriptions or immutable referenced artifacts. Baseline must actually be the declared VLM-only pipeline.
- `tasks`: 1–1000 unique `id` entries, each with `input_sha256` and `critical_facts` (up to 128 unique fact IDs). Input hashes must cover the complete task prompt and recorded sensor inputs, not just a filename. Include rare events, OCR, identity and ASR revision cases. Critical fact IDs refer to a frozen scoring rubric; merely echoing an ID from a model is not correct recognition.
- `min_pairs` (20–1000), `bootstrap_seed` (uint32) and `bootstrap_repetitions` (500–5000). Twenty is a software minimum, not a power analysis. Select enough independent tasks for the actual task distribution before collection.
- `cost.mode`: `priced`, `resource` or `unknown`. Priced mode requires `currency` and `per_million` rates for **all** of `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, including explicit zero prices where appropriate. These are mutually exclusive buckets: uncached input must exclude cached tokens. Choose this mode only when these four charges describe the complete declared VLM tariff. Non-token fees or variable tariffs need an extended accounting implementation; do not omit charges to claim savings. Resource mode requires a predeclared `unit` and `definition` for measured whole-pipeline cost (for example energy), including lightweight providers. Unknown mode leaves cost acceptance incomplete.

Compute `plan_digest(plan)` and retain its returned SHA256 in an immutable, independently timestamped record **before execution**. The CLI's expected hash and each trace row must match it. Hash matching detects later edits relative to that supplied hash; it cannot prove when a caller created a hash. Independent audit must establish predeclaration, prevent selecting only favourable tasks, and confirm scoring and trace authenticity.

## Trace contract and invocation

Provide a JSON array containing exactly one baseline and one candidate result per declared task. Each row has:

```json
{
  "task_id": "example-1",
  "system": "candidate",
  "plan_sha256": "<frozen SHA256>",
  "input_sha256": "<declared task/input SHA256>",
  "latency_ms": 120,
  "quality_score": 0.9,
  "correct_critical_facts": ["cup-left-on-table"],
  "usage": {"calls": 1, "input_tokens": 200, "output_tokens": 30,
            "cache_read_tokens": 0, "cache_write_tokens": 0},
  "peak_memory_bytes": 1000000,
  "dropped_frames": 0
}
```

Resource mode additionally requires `resource_cost` in the declared unit. Scores are externally assessed values in [0,1] from the frozen rubric, not self-reported model confidence. Latency must run from the agreed task/capture trigger through task-answer-ready, including input collection, window/lookahead, providers, queueing and LLM completion. VLM-call latency alone is insufficient. Memory and drops must use consistent, declared measurement scopes. Calls/tokens are usage evidence, not accuracy or total compute cost.

```
python3 -m benchmarks.perception_comparison --plan plan.json --rows traces.json --plan-sha256 FROZEN_HASH
```

Each input file is capped at 16 MiB. Missing/duplicate task pairs and mismatched inputs/plans are errors, so failed/slow tasks cannot be silently removed. A failed task must still be represented and scored under the predeclared failure protocol; do not replace failure latency with a fast successful-subset result. This first evaluator consumes trace files; automatic provider instrumentation and a real paired inference runner remain open in #10.

## Gates and uncertainty

Priced totals are emitted as decimal strings to preserve charge precision. Decisions use an isolated 400-digit decimal context, covering the accepted numeric range and bounded task totals; row charges are precomputed once. Numeric ratios/intervals are display conveniences. `exact_point` and `exact_paired_bootstrap_95` retain decimal strings for boundary audits: a displayed 0.8 may still fail when the unrounded value exceeds 0.8. The report contains count, per-system p50/p95 latency, mean quality, summed charges/resource costs, all usage buckets, peak memory and total drops. Paired bootstrap resamples whole task pairs with replacement using the frozen seed. It reports empirical 2.5/97.5 percentile intervals for cost ratio, p95 latency ratio and mean quality difference. Pairing preserves the shared task difficulty; it does not correct for correlated tasks or a biased held-out set. When frames from one recording are correlated, construct independent recording-level tasks or use a reviewed cluster-aware design instead.

The evaluator conservatively requires both point estimates and interval bounds to meet the fixed plan targets: candidate/baseline total cost ≤0.8, candidate/baseline p95 latency ≤0.9, and quality difference ≥0. Candidate recognition of every declared critical fact is mandatory, with at least one declared critical fact overall. Missing prices, zero baseline cost or undefined resample ratios fail cost acceptance. Insufficient pairs, synthetic evidence or non-held-out data prevent the combined numerical result from passing. No tolerance is relaxed after seeing results.

`meets_numeric_gates` is a calculation over supplied traces, **not a certificate that the approved goal is complete**. The independent audit must still verify real inference, unchanged configuration, source traces, scoring, correct pricing and complete pipeline timing. The original VLM-only comparative acceptance and all other capability requirements remain open until that evidence exists.
