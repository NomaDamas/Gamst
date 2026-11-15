# Owned recording runtime metadata loader

`OwnedRuntimeInputs.open(root, expected_execution_sha256=...,
invocation_ordinal=...)` reads only the fixed input inventory from the approved
runtime input-root contract. It activates no providers, imports no caller code,
loads no model/checkpoint, and supplies no financial finality.

The loader uses one owned read/parse thread. It walks every absolute directory
component without following symlinks, requires a private mode0700 root, bounds
all fixed metadata sizes before JSON conversion, and retains a128-times-byte
parse-workspace quota until close (maximum256MiB). This quota is implementation
ownership, not global joint-memory certification: parent preflight must charge
the maximum before opening and combine the held workspace with other owned
memory. Actual payload reads/hashes, dependency/asset byte certification,
native/source/artifact admission, and provider activation remain separate.

The immutable issuer-guarded `.receipt` exposes detached documents, all original
source bundles, and selected native declarations; `.identities` names exact
execution/U/C/Q/I/L/assets/input-index object hashes. The receipt becomes unusable
on close. Copies returned by its methods belong to the trusted caller and must
remain in the caller's own admitted ownership. `duplicate_payload_root(id)`
returns a caller-owned non-inheritable duplicate FD, never a borrowed FD to close.
The complete original payload path inventory and regular-file sizes are checked;
OwnedCaptureSource independently checks actual original bytes before replay.

Before activation, call `.validate_native_paths(paths)` with the exact original
native bootstrap paths. It checks both fixed absolute names and retained file
identity/original bytes, rejecting copied-file and hard-link aliases. This
synchronous bounded check belongs to startup; a caller performing it in an owned
thread must join that thread before closing the input owner. The loader's own
open/close path joins its read thread despite repeated cancellation.

U1 rejects every prospective numeric input. U2 requires exact K2/G/E2/F input
files and the separately validated Q2/I3 entrypoint; prior structural inputs are
never upgraded. The F fixed-path addition is specified in the independently approved
[synthetic charge input contract](recording-runtime-charge-scenarios.md).
Runtime activation still requires the complete original charge-scenario inventory
and exact SDK/controller ownership; contract approval is not evidence of an
actual vendor payment or a completed numerical evaluation. Every invocation and
every original recording remains present, including20-recording synthetic
feasibility fixtures;
these fixtures establish structural admission, not real recording independence,
model quality, monetary acceptance, or cost/latency savings.


## Running the persistent collection

Run from a repository checkout containing the integrated runtime and validator;
the SDK wheel packages `gamst`, not the `benchmarks` modules. The fixed
parent entry point consumes a prepared input root; it does not construct a dataset, install
models or choose an accounting policy. Actual evaluation and dataset acquisition
must run on GCP under the repository policy. Local synthetic software tests do
not establish measured product savings.

```sh
python -m benchmarks.recording_coordinator \
  --runtime-inputs /absolute/private/frozen-inputs \
  --expected-execution-sha256 ORIGINAL_INVOCATION_ZERO_EXECUTION_SHA256 \
  --output /absolute/private/new-results \
  --retain-evaluation-artifacts
```

Replace the SHA placeholder with the original canonical execution-object digest
for invocation zero, not a digest of a later result or a guessed file hash. The
[input contract](recording-runtime-input-root.md) describes the fixed inventory
and canonical representation. The
output directory must not already exist. Add `--retain-financial-documents` only
when this consent matches the frozen scope policy; the runtime refuses a mismatch.

The parent runs every frozen invocation in order, waits for the owned child to
be reaped before starting another, and retains original executions and run
references in `execution-inventory.json`. A structural result does not establish
numerical eligibility. The matched-source evaluator additionally needs all
original validated runs, financial evidence and separately reviewed assessments;
missing receipts and failed runs cannot be dropped to obtain a passing result.

The Python API requires the same explicit retention consent as the command.
It returns `validated_runs` as an immutable tuple of issued runtime joins and
`artifact_budget` as the collection's shared lifetime budget. Keep these objects
in the owning process when using the prospective U2 financial bridge; JSON
copies of a run inventory cannot replace an issued native/runtime join.

```python
result = await run_recording_collection(
    input_root, new_output_root,
    expected_execution_sha256=original_execution_zero_sha256,
    retain_evaluation_artifacts=True,
    retain_financial_documents=True,  # Must match the frozen U2 scope policy.
)
views = finalize_financial_runs(
    result['validated_runs'], result['manifest'], documents=original_documents,
)
```

Import the two functions from `benchmarks.recording_coordinator` and
`benchmarks.recording_runtime_finance`. `original_documents` must be the same
validated original input set, not reconstructed answers or revised policies.
The views do not settle missing charges or establish finality.
`reconcile_runtime_finance` additionally requires original financial artifacts,
an explicit ledger, a reviewed validity checkpoint and the reviewed assessment.
Financial artifact writes must use `result['artifact_budget']`; the source/runtime
output budget is never reset for later reconciliation. U1 has no financial
finalization path and remains structurally ineligible for numerical evaluation.
