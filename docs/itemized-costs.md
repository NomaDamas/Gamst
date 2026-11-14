# Itemized full-pipeline cost evaluation

`benchmarks.perception_comparison` accepts `cost.mode: itemized` for mixed
specialist/small-VLM/strong-VLM/text providers, per-request and media billing,
cache reads/writes and declared local compute cost. The existing `priced`
(single set of token prices), `resource` (whole-pipeline measured proxy), and
`unknown` modes remain compatible. Itemized mode sums only itemized quantities;
token `usage` remains required diagnostic data and is not charged twice.

Freeze the rate card in the plan **before** collecting inputs and running tasks:

```json
{
  "mode": "itemized",
  "currency": "USD",
  "scope": "full_pipeline",
  "accounting_protocol": "Reviewed all-call receipt protocol and local-cost allocation revision",
  "rate_card": {
    "small.uncached": {
      "provider": "small-provider", "revision": "model-and-price-revision",
      "unit": "uncached-token", "category": "input_tokens", "per_unit": 0.000001
    },
    "strong.image": {
      "provider": "strong-provider", "revision": "model-and-price-revision",
      "unit": "image", "category": "media", "per_unit": 0.01
    },
    "local.compute": {
      "provider": "local", "revision": "audited-allocation-v1",
      "unit": "millisecond", "category": "local_compute", "per_unit": 0.000001
    }
  }
}
```

These rates are illustrations, not current provider prices. There must be 1–64
rates, each with exactly provider, revision, unit, category and per_unit. Category
is input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, media,
request, local_compute or other. Unit price is finite, nonnegative and at most
1e9. Every quantity is an exact nonnegative integer at most 1e12; choose atomic
units before collection (e.g. milliseconds instead of fractional seconds).

Every task/system row supplies `cost_quantities` with exactly every rate ID:

```json
{"cost_quantities": {"small.uncached": 400, "strong.image": 0, "local.compute": 250}}
```

Zero asserts verified zero usage, not unknown usage. Missing, null, extra,
fractional, boolean, negative or oversized quantities are rejected. Missing
prices must remain unknown and block acceptance, not become free rates. Include
retry/cancelled/failed attempts, both tiers in an escalation, media preprocessing,
cache initialization and text-consumer work. Never omit a provider simply because
one arm did not use it: use the same frozen union of rate IDs in both arms.

Sum quantities across the complete task, including cold/warm work under the
frozen run protocol. Token buckets must be exclusive according to each provider's
billing semantics. Use distinct rates for different prices/units/revisions. Do not
count both image tokens and a separate image charge for the same billed work.
Local resource quantities can be converted to money only through an audited
allocation rate; otherwise use the existing whole-pipeline resource proxy mode,
not a mixture of dollars and joules. Aggregate quantities do not establish the
provenance of individual attempts; preserve the underlying attempt receipts and
coverage evidence for independent review.

The evaluator uses Decimal arithmetic and existing paired bootstrap gates. Tiny
additional charges cannot round an actual >0.8 cost ratio into the <=0.8 gate.
Summary cost remains an exact decimal string, and per-rate aggregate quantities
are exposed for inspection. Diagnostic float ratios may round; the exact ratios
and boolean gates govern acceptance.

## Evidence join

In itemized mode each externally reviewed `benchmarks.replay_evidence` assessment
entry adds `cost_quantities` to its existing fields. `resource_cost` stays null.
The join requires the complete dictionary, copies it into the comparison row,
and binds it through the reviewed assessment digest and original source trace.
Failed task responses retain their charges and zero quality under the existing
failure policy. Other modes reject the extra assessment field.

`full_pipeline` is a caller declaration, not a completeness detector. The evaluator
cannot discover an omitted call, verify prices or independent scoring, infer
actual inference from a successful process, or prove held-out provenance. Audit
receipts and instrumentation against the frozen protocol. Synthetic fixtures
remain synthetic and cannot satisfy measured acceptance. Issues #57/#135/#161
continue to track actual all-call instrumentation, quality, cold/warm latency,
resource measurements and full A/B/C evaluation.
