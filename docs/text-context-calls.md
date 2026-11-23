# Explicit text Context calls

`TextContextCalls` connects an explicitly admitted `text_llm` route to an application-owned asynchronous completion client. It serializes the full derived observation so a text LLM can reason over detector, trajectory and audio metadata without receiving pixels or sound. Construction does not load a model, open a network connection or invoke the client. No queue, routing timer or retry is created.

```python
from gamst import TextContextCalls

async with TextContextCalls(human, 'main', completion,
        provider='my.client', model_revision='pinned.model.revision', timeout=30) as calls:
    result = await calls.execute(request_id, original_registered_record,
        'What changed in the observed scene?')
    if result.get('accepted'):
        render_answer(result['text'])
```

The controller must already be registered as `main`. Obtain a request by submitting a registered observation with `query=True` and `needs_visual=False` under an enabled `CallPolicy`; a context-only policy never authorizes a call. The record must exactly match retained evidence and current runtime/source provenance. See [call controller](call-controller.md). Choosing text is an application decision about evidence sufficiency, not proof that vision is unnecessary.

Only pending text_llm requests are accepted. A changed record or wrong route raises. One operation may be in flight; additional execute calls raise immediately. Questions must be nonempty and at most 4096 UTF-8 bytes. Records are bounded to 131072 ASCII-escaped JSON bytes and the complete request to 262144 bytes. Untrusted observation text is serialized as data in a user message, never promoted to a system role.

The completion coroutine receives a private copied dictionary:

```python
{
    'provider': 'my.client', 'model_revision': 'pinned.model.revision',
    'messages': [{'role': 'system', 'content': '...'},
                 {'role': 'user', 'content': '{"question":...,"observation_record":...}'}],
}
```

Provider/revision are caller-declared bounded aliases, not independently verified endpoint identity. The recorded prompt SHA256 covers the canonical request handed to the client. The SDK cannot prove that a callback transmitted it unchanged or prevent client-side transformations, retries or logging. Endpoint authentication, remote-data permission and client lifetime belong to the application.

Return exactly `{'text': nonempty_answer, 'usage': None}` or declared token usage:

```python
{'input_tokens': 100, 'output_tokens': 20, 'cached_input_tokens': 10}
```

Counts are bounded, unverified telemetry, not payment receipts. A completed response includes source record ID/hash, prompt hash, provider/revision, `interpretation='untrusted_text_response'`, `quality_validated=false` and `usage_verified=false`. It is returned only to the explicit caller and never inserted into sensory memory. Controller/MCP diagnostics retain bounded hashes, not question or response bodies. Text completion never satisfies outstanding visual semantic-audit coverage.

The application owns adapter lifetime. Host pause/reset, consent revocation, deletion or close invalidates late replies but does not close the application's remote client. Adapter close cancels its owned operation, not the caller's surrounding task, and cannot be invoked by its own completion coroutine. Timeout is 0.01..300 seconds; late responses are rejected even if a callback catches cancellation and returns. Configure cooperative transport deadlines: arbitrary blocking or cancellation-suppressing Python cannot be forcibly stopped.

Tests use actual async fixture clients over structured observations for evidence rejection, prompt isolation, timeout, cancellation, lifecycle discard, self-close protection and MCP redaction. They do not contact a remote provider or establish model quality.
