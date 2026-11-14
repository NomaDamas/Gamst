# Explicit on-demand SmolVLM calls

`SmolVLMCalls` connects an existing registered `CallController` small_vlm request
to the real local SmolVLM2-500M `ProcessProvider`, then publishes accepted context
through HumanSense's observation/MCP path. It does not select models, generate
requests, change sensor/delivery cadence, or substitute for strong_vlm/text routes.

```python
from humansense import SmolVLMCalls, smolvlm_frame_sha256
from humansense.schema import timestamp

# human already has a registered controller and a queued small_vlm request.
# record is its original registered observation; frame is its original InputFrame.
# At submission: visual_reference["sha256"] = smolvlm_frame_sha256(frame.data).
# config contains explicit local checkpoint/tokenizer paths and SHA256 values.
async with await SmolVLMCalls.open(
    human, "controller-id", "/existing/environment/bin/python", config
) as calls:
    result = await calls.execute(
        request_id, record, frame,
        maximum_charge=verified_upper_bound_in_controller_units,
    )
    # Actual cost is intentionally unknown. Once independently verified:
    if result["status"] in ("published", "discarded"):
        controller.settle(request_id, components=verified_complete_charges,
                          now_ms=timestamp())
```

The snippet assumes application-provided assets, evidence, quotes and receipts;
it is not a standalone model download/demo. The selected environment must already
have the pinned native dependencies and SDK. `open` accepts the existing native
configuration only and fixes `window_frames=1`. The separate streaming window
provider still supports its original multi-frame configuration. Open/warmup work
is explicit and must be included in whole-pipeline cold/warm resource accounting.
The adapter never estimates cost from token counts or elapsed milliseconds.

## Admission and evidence

Only a pending small_vlm request is accepted. The original record hash must match
the controller's retained decision. The exact subject, timestamp and complete
sensor metadata must match the original frame; retained uint8 BGR pixels are
bounded to the native 1080p limit, copied and checked against the visual reference
SHA256 before any await. Modified pixels, metadata or records fail before dispatch.
Create the visual reference with `smolvlm_frame_sha256(frame.data)`. This versioned
SHA256 binds a uint8/HWC/BGR format tag, three big-endian shape dimensions, and
C-order bytes. Raw-byte SHA256 alone is rejected: identical bytes reshaped to
different image geometry must not authorize the original capture. The native
window still reports raw pixel hashes alongside its separately validated shape.

One adapter permits one active call; a second call fails immediately rather than
building an invisible queue. Do not use its owned worker directly or concurrently
finish the same request while execute is running. Verified billing through
controller.settle may arrive during execution. On each call the window
state is reset, allowing nonconsecutive retained frames. After that async reset,
actual controller reserve/dispatch rechecks host lifecycle, freshness and budget.
Unknown or insufficient quotes block inference. Execution provenance binds the
real profile/version and the native timestamped prompt hash (before image-token
expansion). Prompt text is fixed by the existing native provider, not caller data.

The owned worker's result must match retained capture/pixel/shape/source metadata,
asset hashes and prompt. Only a nonempty described generation can acknowledge
semantic coverage. Empty/malformed output cannot clear an unknown-scene audit.
Generated text remains an untrusted semantic hypothesis with quality_validated
false; successful execution does not establish semantic correctness.

## Context publication and lifecycle

Accepted results are published at the **original capture timestamp**, with the
registered SmolVLM capability. They are then available through normal Python,
observation streaming, store/history and MCP reads. Old captures remain old;
completion time does not make them fresh.

Pause/reset, subject deletion, consent revocation and host close invalidate the
controller source/processing epoch. A result finishing afterward is discarded and
not republished. Detached financial liability survives deletion. A dispatched call without
a verified receipt—including exceptions, cancellation, empty or discarded
output—reports unknown charges. This conservatively retains a budget slot/ceiling
and may block later calls until verified accounting resolves. Earlier verified
charges are preserved without a second unknown receipt. A forgotten request with
an already settled detached bill still discards late output without reconstructing
erased sensory state. `actual_cost=None` means this adapter does not calculate the
price; the controller ledger is authoritative for verified charges.
No bill is silently set to zero just because output was discarded.

HumanSense and this adapter are used on one event loop. The adapter validates the
accepted epoch before publication; the ordinary push path emits its observation
before its first suspension. Caller-owned returned observations/copies and
external evaluation artifacts have their own retention obligations.

The adapter is explicitly application-owned. Use its async context manager or
`await close()` even if HumanSense has already closed. Host close suppresses future
publication but does not implicitly own this separately opened worker. Adapter
close cancels an active execution, reaps its own worker through ProcessProvider,
and preserves unknown accounting. Timeout/cancellation closes that worker; reopen
explicitly. No other process or session is signalled or cleaned up.

## Validation and remaining gates

Real owned-pipe tests exercise dispatch, MCP publication, quote denial, strong-route
refusal, copy/identity mismatches, cancellation/busy handling, empty/malformed
results, lifecycle invalidation and late charges. Native CI additionally executes
the production loader/generator with a generated **untrained tiny architecture**
through this adapter; it is not trained500M evidence. No trained weights are
installed by this feature or its tests.

Actual trained model gate #150, broader controller/provider scope #154/#161 and
cost/quality/latency #57/#135 remain open. This path handles one retained frame and
one configured small model. It does not supply strong/text providers, automatic
escalation, cross-window KV reuse, all-call metering or a quality-preserving savings
claim. Include warmup, other specialist models, retries, media and local resources
in the existing full-pipeline evaluation.

