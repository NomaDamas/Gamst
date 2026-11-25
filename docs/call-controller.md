# Explicit call controller

`CallController` connects `CallPolicy` recommendations to a bounded pending queue and explicit reservation/dispatch lifecycle. It never invokes a model, changes capture or delivery cadence, retrieves media, injects Context into an LLM, or interprets model text as authorization. Applications perform calls themselves after dispatch admission.

```python
from gamst.call_controller import CallController
from gamst.call_policy import CallPolicy

controller = CallController(policy=CallPolicy(mode='sentinel'))
source = controller.open_source(observation, now_ms=now)
queued = controller.submit(source, observation, sequence=1, now_ms=now,
                           visual_reference=verified_frame_reference)
# context_only/rejected results do not contain a request ID.
request_id = queued['request_id']
reservation = controller.reserve(request_id, provider='provider',
    model_revision='model-v1', prompt_sha256=actual_prompt_hash, now_ms=now)
admission = controller.dispatch(request_id, now_ms=now)
# The application verifies retained media and performs its explicitly opted-in call.
controller.finish(request_id, success=True,
    evidence_sha256=admission['evidence']['record_sha256'],
    result_sha256=verified_result_hash, now_ms=completed_at)
```

Never execute when reservation or dispatch is blocked. Retain hashes of the actual prompt and response; hashes bind evidence but do not authenticate a provider or establish correctness. `finish(success=True)` is an application acknowledgement that the bound capture was processed, not a model-confidence score. Completion retires the request into bounded recent history. Late results remain observations of their original capture time, not current-world facts.

## Source and event ownership

Open one handle per subject/source/sensor/capability. The handle binds stream and processing epochs, calibration, coordinates, channels, modality, capability contract and registered model provenance. Close the old handle before replacing an epoch or configuration. Capture times cannot regress and submission sequences must strictly increase. Only owned history supplies hysteresis, last dispatch and last successful semantic capture.

A queued threshold event retains its original record and visual reference while cooldown prevents admission. Subsequent low scores do not erase it. Only byte-equivalent record/rule requests coalesce; different captures, revisions or rules do not substitute for one another. A finished, failed or discarded attempt does not absorb a new attempt. Failed threshold demand may be retried without losing newer hysteresis; admitting a newer exact attempt supersedes older retry eligibility.

Invalid or missing visual evidence, insufficient input quality and stale/future captures are rejected. Cooldown alone can queue. Freshness and cooldown are rechecked at reservation and dispatch. The controller never substitutes a newer frame for stale retained evidence. Pending stale evidence remains inspectable until cancellation or `expire()`; expiry creates an `expired_uninspected` history entry, not semantic coverage.

## Completion and lifecycle

Recommendation, reservation, dispatch and result acknowledgement are separate stages. Only actual dispatch advances cooldown. Only successful, non-discarded VLM completion for a still-open source advances semantic coverage, using the original capture time and monotonic maximum across out-of-order completions. Text-only completion does not cover visual audits. Failed or ignored calls do not satisfy them.

Dispatch returns a copied bounded observation for the application, then discards the controller's full observation body while retaining evidence hashes and freshness metadata. Cancelling a dispatched request suppresses future output acceptance; the application must still acknowledge its result or failure. Closing a source cancels undispatched work and marks dispatched output for discard. Subject deletion erases associated requests and history. Closing the controller prevents further admission.

Default bounds are 32 sources, 128 active requests, 128 recent entries and 60-second pending retention. Configured maxima are 128 sources, 256 requests/recent entries and one-hour retention. Stored observations are capped at 128 KiB; media bytes are not retrieved by this class. Expiry is explicit, not a background thread. `snapshot(now_ms=...)` refreshes time-dependent blockers.

Controllers are memory-only trusted Python application surfaces. Registration with a `Gamst` instance is explicit. Optional [MCP integration](call-integration.md) exposes bounded status and applies runtime consent, pause, reset and deletion lifecycle. See [the executable example](../examples/call_controller_demo.py) for application-owned invocation.
