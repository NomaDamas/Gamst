# Optional call recommendations

`humansense.call_policy.CallPolicy` evaluates one registered observation without network calls, storage mutation, model execution, or changes to sensor/Context delivery cadence. Default `context_only` returns no call. This is the pure rules component of #133, not a completed model router or controller.

```python
from humansense.call_policy import CallPolicy

policy = CallPolicy(mode="sentinel", semantic_interval_ms=5000)
decision = policy.recommend(
    observation_record,
    now_ms=now_ms,
    last_semantic_capture_ms=last_successful_capture_ms,
    query=True,
)
# Inspect decision; this function never dispatches or authorizes a model call.
```

Modes are `context_only`, `periodic`, `sentinel`, and `escalation`. Periodic proposes a small VLM when successful semantic coverage is overdue. Sentinel checks explicit novelty, queries, externally validated task-rule conflicts, threshold entry, and overdue video coverage. Escalation allows a strong VLM for an explicitly supplied task-rule conflict. It never interprets observed text or a model's self-confidence as an escalation instruction. Caller-supplied rules and lifecycle history are trusted inputs, not independently verified facts.

Visual routes require a reference with exactly `source_id`, `sensor_id`, `stream_epoch`, `capture_ms`, and a lowercase 64-character `sha256`, matching this video observation. The caller owns media retention and hash verification; matching metadata is not proof that bytes exist. Audio-to-video association requires separate source alignment and is not inferred. A periodic or visual request on audio alone is blocked for missing visual evidence.

All recommendations bind the subject, source, sensor, stream and processing epochs, calibration, coordinate frame, record ID, original capture timestamp, and canonical record hash. Decisions carry a deterministic hash. Input records are bounded to 128 KiB and use registered observation metadata; unsupported or malformed metadata is rejected. These hashes provide reproducibility, not authentication.

Future or stale capture times, insufficient declared input quality, missing visual evidence, and cooldown block dispatch recommendations while preserving the desired route and unresolved coverage. Input-quality status is not evidence of model accuracy. Semantic coverage is measured from the last successfully inspected capture, never from recommendation or dispatch time. The caller must provide history from the exact source/epoch and a comparable clock.

Hysteresis enters at `change_enter` and exits at or below `change_exit`. `proposed_change_active` is advisory: do not commit it for a blocked request and lose the pending threshold event. Repeated evaluation has no side effects. In particular, recommendation, dispatch, and successful semantic inspection are separate events.

## Required controller and acceptance

#154 remains required: bounded per-source lifecycle state and priority queues; pending rare-event preservation; atomic budget reservations; dispatch/settlement acknowledgements; duplicate/replay protection; unknown-price accounting; and API/MCP observability. The evaluator's `budget_authorized`, `dispatch_recorded`, and `semantic_success_recorded` fields are always false. No consumer should interpret a nonempty recommendation as permission to spend or proof of semantic coverage.

#57/#135 require frozen, paired A/B/C evaluation of 1 fps VLM, thin-model-only, and hybrid pipelines. This rules component proves neither maintained quality nor cost/latency savings. Actual SmolVLM acceptance #150 and the original modality acceptance gates remain open. No trained weights are downloaded for these tests.
