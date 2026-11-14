# Explicit call controller

`CallController` connects `CallPolicy` recommendations to a bounded pending queue and an exclusively owned `CallBudget`. It never invokes a model, changes capture/delivery cadence, retrieves media, injects Context into an LLM, or interprets model text as authorization. Applications perform calls themselves after explicit reservation and dispatch admission.

```python
from humansense.call_controller import CallController
from humansense.call_policy import CallPolicy
from humansense.call_budget import CallBudget

controller = CallController(
    policy=CallPolicy(mode='sentinel'),
    budget=CallBudget(unit='USD.micro', pricing_revision='prices-v1', ceiling=1000000))
source = controller.open_source(observation, now_ms=now)
queued = controller.submit(source, observation, sequence=1, now_ms=now,
                           visual_reference=verified_frame_reference)
# Check queued['status']; context_only/rejected do not contain a request ID.
request_id = queued['request_id']
reservation = controller.reserve(request_id, provider='provider', model_revision='model-v1',
    prompt_sha256=actual_prompt_hash, maximum_charge=10000, now_ms=now)
# Check reservation status and then dispatch status: either may be blocked.
admission = controller.dispatch(request_id, now_ms=now)
# Application verifies retained media bytes, performs its explicitly opted-in call,
# and validates completion against admission['evidence'] before acknowledging it.
controller.finish(request_id, success=True,
    evidence_sha256=admission['evidence']['record_sha256'],
    result_sha256=verified_result_hash, now_ms=completed_at)
controller.settle(request_id, components={'input':100,'output':300}, now_ms=billed_at)
```

The example omits application transport/error handling. Never execute when admission is blocked. Retain hashes of the actual prompt and response; hashes are provenance bindings, not authentication or a correctness check. `finish(success=True)` is an explicit trusted application acknowledgement that the bound capture was successfully processed, not a model's confidence score. It does not prove task accuracy or change the observation store. Successful late results remain historical observations of their original capture time, not current-world facts.

## Source and event ownership

Open one handle per subject/source/sensor/capability. The handle binds stream and processing epochs, calibration, coordinates, channel names, modality and the full capability contract and registered `models` provenance (including provider/version). Close the old handle before replacing an epoch/configuration. Old handles cannot mutate the new source. Capture times cannot regress within a source. Submission sequences must strictly increase across the entire controller; invalid evidence/capacity rejection consumes a validated sequence, so retries need a new one. Only owned history supplies hysteresis, last dispatch and last successful semantic capture.

A queued threshold event retains its original record and visual reference while cooldown or budget prevents admission. Subsequent low scores do not erase it. Only byte-equivalent record/rule requests coalesce; different captures, revisions or rules do not substitute for one another. Exact matches with an outstanding request return that request ID, while execution is unfinished and not discarded. A finished/failed/discarded attempt does not absorb a new attempt; its independent outstanding bill remains reserved. The most recent retained attempt/receipt may re-admit an exact failed threshold demand even after newer captures, while preserving newer hysteresis. Admitting a newer exact attempt permanently supersedes older retry eligibility, so eviction of the newer receipt cannot resurrect an older failed attempt with unresolved billing. Freshness/cooldown and fresh budget admission still apply. Once its bounded receipt is evicted, the application must express an explicit new query/trigger; old historical evidence is not silently recovered. Unknown/conflict requests have priority 0, explicit queries 1, background/change requests 2. Snapshot order is advisory; the application chooses the next request. At capacity a higher-priority request may preempt a strictly lower-priority **pending** request, with an explicit `preempted_uninspected` receipt. Reserved/dispatched requests and equal/higher priorities are never preempted. Otherwise rejection is explicit and counted. Capacity is not proof that rare events were retained; applications must observe and handle rejection.

Invalid/missing visual evidence, insufficient input quality, and stale/future captures are rejected rather than queued. Cooldown alone can queue. Freshness and cooldown are rechecked at reservation and dispatch. If dispatch is blocked, its undispatched budget reservation is released and the event remains pending. The controller does not silently substitute a newer frame for a stale event. Pending stale evidence remains inspectable until explicit cancellation or `expire()`; it cannot be dispatched as fresh. Expiry creates an `expired_uninspected` receipt, never successful semantic coverage.

## Completion and liability

Recommendation, reservation, dispatch, result acknowledgement and financial settlement are separate stages. Only actual admitted dispatch advances cooldown. Only successful, non-discarded VLM completion for a still-open source advances semantic coverage, using the original capture time and monotonic maximum across out-of-order completions. Text-only completion does not cover visual audits. Failed/ignored calls do not satisfy them.

Dispatch admission returns a copied bounded `observation` for the application to use as Context. The controller then discards its full observation body, retaining evidence hashes and freshness metadata for acknowledgements and accounting; indefinite billing uncertainty does not retain that body. The application owns its returned copy and media lifecycle.

Result and billing may arrive in either order. An active request retires only after both are final. Unknown billing (`components=None`) preserves liability and capacity; known maps assert complete verified charges as specified in [the ledger contract](call-budget.md). Repeated completion/final settlement and retired/foreign IDs are rejected. Each retry or chained model call needs its own request/reservation; no transport retry is automatic.

Cancelling a dispatched request suppresses future output acceptance but preserves liability. The application must still acknowledge its result/failure and settle its charges; no cancellation/timeout fabricates a refund. Closing a source cancels undispatched work and marks dispatched output for discard. Closing the controller additionally closes admission in its ledger, but late result/financial acknowledgements remain valid. Source closure never resets financial totals.

## Bounds and integration status

Default bounds: 32 sources, 128 total active requests (including dispatched and unresolved billing), 128 receipts, 60-second pending retention. Configured maxima: 128 sources, 256 requests/receipts, one-hour retention. Each stored observation is capped at 128 KiB with existing structural limits; media bytes are not retrieved by this class. Receipts retain evidence hashes, not full records. Evicted receipt counts and rejected-capacity counts remain visible. Expiry is explicit, not a background thread. `expire()` releases expired undispatched budget reservations back to pending events before event retention expires; its return value counts changed requests. `snapshot(now_ms=...)` refreshes time-dependent blocker reporting; without a time it reports at the last controller operation. Supply one nondecreasing clock to every operation.

Give the controller a fresh ledger and never mutate/share that ledger externally; there is no multi-controller admission arbitration. Both components are memory-only and trusted Python application surfaces, not authenticated remote financial services. Persist/reconcile outstanding charges before replacing the controller on process restart. There is no restart recovery guarantee.

Python lifecycle API and real registered-observation integration tests are implemented here. Bounded HumanSense/MCP status exposure, consent/pause/delete lifecycle integration and executable end-to-end application wiring remain required under #154 and its lifecycle/MCP integration child. This class is not automatically attached to a HumanSense instance. Full actual A/B/C cost/quality/latency acceptance remains open under #57/#135. Tests with controlled acknowledgements do not establish real VLM accuracy or savings.
