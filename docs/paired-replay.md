# Paired pipeline trace collection

`python3 -m benchmarks.paired_replay` executes explicitly selected local pipeline
commands against the same frozen recorded inputs. It fills the collection gap
between an application pipeline and [numerical comparison](perception-evaluation.md).
It does not score answers, normalize tariffs, certify hardware measurements or
establish that a command actually runs a VLM. Concrete baseline/candidate SDK
pipeline integration and independently audited real runs remain required by #57.

## Freeze before execution

Start with the existing comparison plan and add `runner`:

```json
{
  "schema_version": 1,
  "timeout_ms": 60000,
  "order": "alternating",
  "commands": {
    "baseline": {
      "argv": ["/absolute/python", "/absolute/baseline.py"],
      "cwd": "/absolute/project",
      "artifacts": {"/absolute/python": "<SHA256>", "/absolute/baseline.py": "<SHA256>"},
      "environment": {}
    },
    "candidate": {
      "argv": ["/absolute/python", "/absolute/candidate.py"],
      "cwd": "/absolute/project",
      "artifacts": {"/absolute/python": "<SHA256>", "/absolute/candidate.py": "<SHA256>"},
      "environment": {}
    }
  }
}
```

This object is the `runner` field, not a replacement for the complete evaluation
plan. Freeze all model/runtime/policy/sampling/window and failure/scoring settings
in that plan. Include all relevant local code/config/checkpoint hashes in the
explicit artifact map, not just the command executable. Executable hashing is
mandatory; the collector cannot infer transitive imports or external service
revisions. Verification happens at preflight, so independent audit must ensure
artifacts remain unchanged during collection. Local paths are not a filesystem
snapshot. Hashing a plan does not independently prove its predeclaration time.

`inputs.json` is an array of `{ "task_id": "...", "input": {...} }`, exactly one per
planned task. The input object must contain the complete prompt and recorded media,
including required timestamps/encodings, not merely mutable filenames. Set each
plan task's `input_sha256` to `plan_digest(input)`. The collector hashes the full
object, never interprets media encodings, and sends identical objects to both
pipelines. The whole canonical input package is capped at 16 MiB. Larger recordings
need a separately audited content-addressed streaming input transport; this first
collector does not replace that requirement. Do not smuggle ground-truth labels
into model-visible inputs. Critical fact IDs and scoring rubric are not included
in the subprocess request.

Compute `plan_digest(plan)` after freezing runner/input hashes and independently
record it before collection. The destination must not exist:

```sh
python3 -m benchmarks.paired_replay --plan plan.json --inputs inputs.json \
  --plan-sha256 FROZEN_HASH --output new-run-directory
```

## Command protocol and timing

The runner uses POSIX process groups, explicit argv (no shell), explicit cwd and
only the declared environment (nothing inherited). Plan/environment files are
plaintext: use nonsecret local runtime configuration and no credentials in argv or
plan. Commands are trusted local programs, not sandboxed; this collector does not
prevent network access or model downloads inside a command. During the current
user-requested deferred installation phase, do not configure commands that download
weights. No commands or models start during import or plan validation.

Each command invocation receives one JSON line on stdin with `schema_version:1`,
`task_id`, `system`, `plan_sha256`, `configuration` (the system's frozen declaration)
and `input`. It must perform the entire declared pipeline, including replay timing,
provider work, queues and final LLM completion. It emits one JSON line:

```json
{"schema_version":1,"status":"success","answer":"actual answer","measurements":{}}
```

`status` may be `error`. Measurements are opaque, explicitly unverified source
reports: preserve raw call usage/cache buckets, queue drops and resource readings
with their scopes and unknowns. Missing usage/resource data is not replaced with
zero. Stdout must contain only that response; logs use stderr. Stdout is capped at
64 KiB and stderr at 4 KiB. Extra protocol output, malformed/nonfinite JSON,
nonzero exit, overflow and timeout yield retained error records.

The parent times from before process creation through receipt and validation of
the complete response line as `answer_ready_ms`, and separately records process
lifetime through cleanup. Each task/system starts a fresh command process. Therefore
startup is included and external model-service cache state may still be warm;
declare this exact boundary/warm-state policy. The collector cannot prove a command
has waited for real task completion or included historical input collection time.
For capture-trigger evaluation, commands must pace recorded inputs from that trigger
and not precompute candidate context outside the timed process. Do not reinterpret
this boundary as persistent-session latency without an appropriate integration.

Pair order alternates baseline/candidate then candidate/baseline, in frozen task
order. This reduces fixed ordering bias but is not a randomized experimental design.
Every pair is retained, including failed/timeout runs. The timeout covers invocation
and response/exit; owned process groups are stopped after every invocation, including successful
commands that left descendants with detached stdio. Spawn is shielded: cancellation
or timeout waits to acquire the creation handle, then cleans the owned group.
Cleanup may exceed the configured response deadline; an OS-level spawn that never
returns can delay cleanup. Repeated cancellation cannot abandon the owned cleanup.
On macOS, the [XNU group signal implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_sig.c)
can return EPERM for a group containing only zombie members. Group signalling is
retried briefly; persistent permission failure aborts collection and is never
silently treated as successful cleanup. Commands must not escape the owned process
group or elevate privileges; this runner is not a sandbox or job-control service.
Failures require the frozen failure-scoring/latency policy, never a favourable fast
error or success-only subset. An externally cancelled collector leaves an incomplete
run without `complete.json`; incomplete runs cannot be compared as complete evidence.

## Source evidence and remaining gates

The new destination contains `plan.json`, `inputs.json`, `collection.json`,
`source-traces.jsonl` and, on completion, `complete.json`. Files are exclusively
created with private permissions and flushed to disk; existing files are never
resumed or overwritten. Answers and recorded media may be sensitive. Each trace
hash includes the prior trace hash, ordinal, plan/input/request hashes, status,
response, timings and explicit `quality_status=unscored`. Retain the final hash in
an independent record. The hash chain detects changes relative to a trusted final
hash; it is not tamper-proof storage or proof of held-out provenance.

There is deliberately no automatic conversion to passing evaluator rows. Independently
score answers using the frozen rubric, retain failure cases, verify actual inference
and all accounting/resource scopes, and bind those audited records to trace hashes
before generating comparison rows. Unknown accounting or missing whole-pipeline
resource measurements leave acceptance incomplete. Concrete VLM-only versus SDK+VLM
commands, larger recorded-input transport, scoring join and the real paired run are
still required under [#57](https://github.com/NomaDamas/Gamst/issues/57).

Tests launch real isolated Python subprocesses containing explicit synthetic fixture
pipelines. They check pair order/hash binding, private exclusive output, malformed
responses, missing accounting, timeouts, cancellation/reaping and preflight failures.
They install no weights and are not evidence of model quality or VLM savings.
