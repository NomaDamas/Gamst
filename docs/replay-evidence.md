# Joining source traces with reviewed evidence

`benchmarks.replay_evidence` verifies the connection between a complete paired
collection and externally reviewed answer scores/accounting. It emits comparison
rows and the existing numerical-gate report. It verifies hash binding and structure,
not reviewer identity, scoring truth, measurement accuracy or actual inference.
A numerical pass is still insufficient for the approved project's acceptance.

## Predeclare the rubric and failure policy

Before collecting any results, add this field to the complete comparison plan:

```json
"evidence_join": {
  "schema_version": 1,
  "rubric_sha256": "<hash of independently audited scoring rubric>",
  "measurement_protocol_sha256": "<hash of complete timing/accounting/resource protocol>",
  "failure_policy": "zero_quality_timeout_floor"
}
```

These additions change the frozen plan digest. Existing collections that lack this
predeclaration cannot be retrofitted by editing their plan. Choose independent tasks,
review the rubric and fix all measurement scopes before rerunning collection.

The fixed failure policy gives every error/timeout zero quality and no correct
critical facts. Its evaluator latency is the larger of the actual process lifetime
and the frozen runner timeout. This prevents fast failures from improving latency
relative to successful answers. Successful latency is the parent-observed answer
ready time, including command startup; no assessor-supplied latency overrides it.
This protocol evaluates the existing fresh-command collector boundary, not arbitrary
persistent-session latency. The original cost, quality and critical-fact gates remain.

## Review and freeze the assessment

Retain the collector's final trace hash in an independent trusted record. Review
answers and source accounting against the frozen rubric/protocol, including every
failure. Prepare one assessment entry per source trace:

```json
{
  "schema_version": 1,
  "plan_sha256": "<frozen plan digest>",
  "last_trace_sha256": "<independently retained collection tail>",
  "reviewer": "<responsible assessor identifier>",
  "rubric_sha256": "<same predeclared rubric hash>",
  "measurement_protocol_sha256": "<same predeclared protocol hash>",
  "source_evidence_sha256": "<reviewed evidence bundle hash>",
  "entries": [{
    "trace_sha256": "<specific source trace hash>",
    "quality_score": 1,
    "correct_critical_facts": ["<fact ID adjudicated from the answer>"],
    "usage": {"calls": 1, "input_tokens": 100, "output_tokens": 20,
              "cache_read_tokens": 0, "cache_write_tokens": 0},
    "peak_memory_bytes": 1000000,
    "dropped_frames": 0,
    "resource_cost": null
  }]
}
```

Values above are illustrative, not defaults. Usage buckets are exclusive and must
all be known; missing values cannot become zero. Known zero is valid only with
supporting evidence. Resource mode requires measured total pipeline `resource_cost`
in the predeclared unit; other modes require null. All memory/drop/usage/scoring
fields pass the existing strict comparison validation. Unknown data leaves the
join incomplete. Failed calls may still consume billable tokens and resources:
retain measured costs even when their answer receives no credit.

`source_evidence_sha256` identifies the evidence bundle reviewed by the assessor.
The join does not fetch that bundle or prove that a named reviewer is independent.
Independent audit must inspect its bytes/hash, usage semantics and all required
local/remote compute coverage. Likewise, rubric/protocol hashes bind declarations
but do not prove their content is correct. A model's confidence or echo of a fact ID
is not a correct-answer label. Freeze `plan_digest(assessment)` after review and
retain it independently before generating rows.

## Invocation and integrity checks

```sh
python3 -m benchmarks.replay_evidence --run collected-run \
  --assessment assessment.json --plan-sha256 FROZEN_PLAN_HASH \
  --trace-sha256 TRUSTED_TAIL_HASH --assessment-sha256 REVIEWED_ASSESSMENT_HASH \
  --output new-evaluation-directory
```

The verifier reads the frozen plan, full input objects, completion manifest and
streamed source JSONL. It reconstructs every request hash, checks all canonical
trace hashes and previous links, exact alternating pair order and ordinal, complete
task coverage and trusted final hash. Truncated, extra, reordered, mismatched or
unassessed traces fail. A completion file alone does not prove completeness.

Source lines are capped at 512 KiB and 2000 records, processed incrementally. Other
JSON inputs retain the 16 MiB file bound. Source hashes describe canonical JSON
content; they are not byte-for-byte signatures of JSON whitespace. No source trace
is modified. A new private directory contains `rows.json` and `join.json`, with the
row digest, source/assessment bindings and numerical report. Existing destinations
are never overwritten. The rows can also be passed to the comparison CLI directly.

Tests use real collector subprocess fixtures and synthetic reviewed assessments.
They verify tampering/completeness failures, rubric and assessment binding, strict
unknown-accounting rejection and failure credit/latency. Synthetic evidence cannot
pass the combined gate. Actual VLM-only/SDK+VLM pipelines, model/resource measurement,
independent scoring and held-out acceptance remain required by #57/#10; no weights
are installed and no performance improvement is established here.

For mixed provider, media and local costs, use the frozen
[itemized cost mode](itemized-costs.md). Each reviewed assessment entry adds the
complete `cost_quantities` dictionary; unknown quantities cannot default to zero.
