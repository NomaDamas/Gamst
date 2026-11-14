# Original A/B/C command trace collection

`python -m benchmarks.three_arm_replay` executes explicitly selected trusted local
commands once per task and arm. It produces original A/B/C source traces for the
[frozen numerical plan](three-arm-comparison.md). It does not score answers,
verify model execution, enforce frame sampling inside commands, reconcile receipts
or claim quality/cost savings. `actual_acceptance` remains `incomplete`.

## Two frozen declarations

Keep the original numerical comparison plan unchanged. Freeze its canonical
`plan_digest` plus a separate collection spec. Exact spec shape (replace all
placeholders with actual local artifacts and independently frozen policy hashes):

```json
{
  "schema_version": 1,
  "comparison_plan_sha256": "<ABC plan SHA256>",
  "runner": {
    "timeout_ms": 60000,
    "order": "cycle_all_six_permutations_v1",
    "commands": {
      "A": {"argv":["/absolute/python","/absolute/a.py"],"cwd":"/absolute/project","artifacts":{"/absolute/python":"<SHA256>","/absolute/a.py":"<SHA256>"},"environment":{}},
      "B": {"argv":["/absolute/python","/absolute/b.py"],"cwd":"/absolute/project","artifacts":{"/absolute/python":"<SHA256>","/absolute/b.py":"<SHA256>"},"environment":{}},
      "C": {"argv":["/absolute/python","/absolute/c.py"],"cwd":"/absolute/project","artifacts":{"/absolute/python":"<SHA256>","/absolute/c.py":"<SHA256>"},"environment":{}}
    }
  },
  "evidence_join": {
    "schema_version": 1,
    "rubric_sha256": "<frozen rubric SHA256>",
    "measurement_protocol_sha256": "<frozen measurement protocol SHA256>",
    "failure_policy": "zero_quality_timeout_floor"
  }
}
```

The future evidence-join policy is frozen now; this collector does not perform that
join. Inputs are an array of `{task_id,input}` exactly matching declared tasks.
Each input is a complete immutable JSON object whose canonical digest equals the
numerical task's `input_sha256`. Include prompts, recorded content and timestamps;
mutable external file paths alone are not content provenance. Keep hidden labels
out of model-visible inputs. Critical fact IDs/rubrics are not sent to commands.

The plan, spec and input package are each limited to16MiB. Task count1..1000 means
at most3000 source traces. Large recordings need a separately audited immutable
content transport; this command does not provide it. Artifacts are checked before
launch, not continually snapshotted; concurrent replacement, transitive imports and
external endpoints still require independent provenance audit.

```sh
python -m benchmarks.three_arm_replay \
  --plan /absolute/abc-plan.json --collection-spec /absolute/collection-spec.json \
  --inputs /absolute/inputs.json --plan-sha256 <frozen-plan-digest> \
  --collection-sha256 <frozen-collection-digest> --output /absolute/new-run
```

Import and validation never launch commands. Commands are explicit argv/cwd/env,
not shell strings and not sandboxed. POSIX owned process groups are required.
No environment is inherited; declarations/artifacts are plaintext, not credential
storage. Do not configure commands that install/download weights during the current
model-installation deferral. See [shared execution limits](paired-replay.md).

## Order, protocol and lifecycle

Task order follows the original plan. Successive tasks use `ABC, ACB, BAC, BCA,
CAB, CBA`, then repeat. Each task has one actual A invocation, reused only in later
numeric comparisons. Six-task blocks balance arm positions; a partial block is
reported using actual `arm_position_counts`, not called perfectly balanced.

Each command receives a JSON line with exactly `schema_version:1`, `task_id`,
`arm`, `comparison_plan_sha256`, `collection_sha256`, `configuration_sha256`,
`configuration` and `input`. These are original arm identities, never renamed
baseline/candidate traces. The response uses the existing command protocol:

```json
{"schema_version":1,"status":"success","answer":"answer","measurements":{}}
```

The existing owned executor bounds stdout to64KiB, stderr to4096B, and uses a shared
spawn/collection deadline of1..3,600,000ms. Measurements remain unverified source
reports. Errors, malformed replies and timeouts remain rows, never replaced by a
rerun. Caller cancellation reaps owned work, stops before the next invocation and
leaves no complete marker. Persistent cleanup errors remain visible.

The output directory must not exist. Files are exclusively created with private
permissions: `comparison-plan.json`, `collection-spec.json`, `inputs.json`,
`collection.json`, `source-traces.jsonl`, and only after all3N records, `complete.json`.
Each trace records original plan/spec/config/input/request hashes, a previous-trace
hash and the executor's original result. Lines are bounded to512KiB and individually
flushed/fsynced. A cancelled/interrupted prefix remains incomplete. A completion
marker means all command outcomes were collected **unscored**, including failed
outcomes; it does not mean successful inference or verified evidence.

The full join/attempt/receipt requirements remain in
[#205](https://github.com/NomaDamas/Gamst/issues/205), shared stream-hour allocation
in[#206](https://github.com/NomaDamas/Gamst/issues/206), real held-out evidence in
[#207](https://github.com/NomaDamas/Gamst/issues/207). This collector slice is
[#211](https://github.com/NomaDamas/Gamst/issues/211).
