# Actual SDK client attempts in evaluation

`benchmarks.controller_attempts.RecordedContextClient` connects the repository's
frozen `AttemptRecorder` to actual `TextContextCalls` and `VisualContextCalls`
client invocations. It is opt-in evaluation instrumentation, not a new model
provider or a production telemetry service. It does not install models, discover
providers or enable network calls on construction.

```python
from benchmarks.controller_attempts import RecordedContextClient
from humansense import TextContextCalls

# controller is registered with human; recorder owns one frozen task/system.
# request_id refers to the pending evidence-bound decision being executed.
client = RecordedContextClient(controller, request_id, recorder, completion,
    phase='cold')
async with TextContextCalls(human, 'main', client,
        provider='text', model_revision='frozen-v1') as calls:
    result = await calls.execute(request_id, original_record, 'What changed?',
        maximum_charge=verified_ceiling)

# After independently verified complete billing becomes available:
if client.attempt_id is not None:
    recorder.report(client.attempt_id,
        quantities=complete_frozen_rate_quantities, receipt_sha256=receipt_digest)
    controller.settle(request_id, components=verified_charge_components, now_ms=now)
```

For visual calls, use the same wrapper as the completion callable passed to
VisualContextCalls and supply the original frame in execute. See
[visual client contract](visual-context-calls.md),
[text client contract](text-context-calls.md), and
[attempt recording](call-attempts.md).

Each binding permits one invocation. A blocked SDK budget decision never reaches
the wrapper and has no fabricated attempt. At invocation, the wrapper checks a
live dispatched controller request, exact outgoing request hash, provider/model
aliases and visual/text route. It records evidence, prompt and execution hashes
from that actual ledger entry. A changed request, wrong ID, finished/invalidated
request or reused wrapper fails before invoking the application client. The
wrapper permits the current dispatched request during its next-admission cooldown,
while still rejecting runtime/epoch/source/freshness invalidation. The wrapper
itself does not authorize a model call or hold the controller lock across
asynchronous client work. Like the SDK adapters, use one event loop for runtime
lifecycle operations.

A wrapper invocation consumes the one-shot binding even if binding validation or
recorder admission fails. If the recorder is closed, full, has a mismatched rate
card or rejects lineage, the client is not invoked; the SDK already-dispatched
budget entry conservatively remains unknown until externally verified settlement.
Do not invent a zero receipt. Inspect controller outcomes and recorder admission
rejections when evaluating coverage. An empty recorder only describes its
instrumented scope and is not proof of zero whole-pipeline work.

`attempt_id` is null until the recorder admits an actual client invocation, then
remains available after failure, cancellation, host deletion or adapter close for
late receipts. The recorder never stores request/image/response bodies or error
messages. These explicit evaluation artifacts belong to the application, retain
hashes and identifiers, and have separate retention/deletion obligations; deleting
HumanSense observations does not erase the caller's evaluation report.

`status='success'` in the attempt report means the client coroutine returned.
It does not mean the SDK accepted the answer, that response validation passed, or
that the answer was correct. A malformed response can produce a successful client
attempt and a failed SDK result. Similarly, a late reply may be discarded after
delete/pause despite a completed billable invocation. Conversely a callback that
suppresses cancellation may return successfully even though its surrounding SDK
call is cancelled; the attempt remains billable. Compare controller outcomes and
task-level traces separately; never filter failed/discarded tasks from costs.

Phase cold/warm/unknown is caller-declared. For an explicit retry or escalation,
construct a new binding with predecessor=the_previous_attempt_id and the appropriate
relationship. The recorder verifies completed same-evidence lineage and retry
provider/role identity. The caller/controller still decides whether a new request
is eligible. This wrapper does not schedule retries, fallback, or model changes.

Evaluation quantities and controller financial charges are separate contracts:
report() needs every frozen rate ID for the provider/revision, while settle()
needs verified charge components in the controller's declared unit. Neither call
implicitly performs the other; no unverified token-to-price conversion occurs.
Unknown quantities remain null until verified receipts and prevent completed-cost
reduction. Use cost_quantities() with the independently reviewed report digest to
join the frozen itemized comparison protocol.

Timing covers the actual instrumented client callback, not task answer-ready
latency, image preparation, sensing, local worker startup, queues or all computer
work. Hidden retries inside the application client and calls bypassing the wrapper
are not automatically detected. Every separate backend attempt must be explicitly
instrumented, and whole-pipeline evaluation must include perception/local-worker
costs, preparation and all external attempts. Those complete coverage/real-model
gates remain open in #161/#57/#135. Fixture clients validate the integration and
accounting paths, not model quality or real endpoint execution.
