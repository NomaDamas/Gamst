# Explicit visual Context calls

`VisualContextCalls` completes an optional small_vlm/strong_vlm execution path using an application-owned async client. It does not own a camera, model, remote endpoint, retry loop or user delivery cadence. This is single-frame on-demand inference, not native stateful model streaming.

```python
from gamst import VisualContextCalls, smolvlm_frame_sha256

async with VisualContextCalls(human, 'main', completion,
        provider='chosen.client', model_revision='pinned.revision') as calls:
    result = await calls.execute(request_id, original_record, original_frame,
        'Explain the conflicting observation.')
    if result.get('accepted'):
        render_answer(result['text'])
```

Register the controller and submit the exact registered observation with an eligible visual decision and matching `visual_reference` first; see [call controller](call-controller.md). Use `smolvlm_frame_sha256(frame.data)` for that reference. The adapter cannot upgrade a text-only decision or infer remote model capability from a caller-declared alias.

Input must be a video `InputFrame` with numpy uint8 HWC BGR pixels, each dimension 32..2160 and at most 1920×1080 pixels, matching the retained sensor, subject and capture timestamp. Pixel copying and shape/format-bound SHA256 validation happen before asynchronous client work. Matching raw bytes with a different shape is insufficient. Optional numpy/OpenCV dependencies must already be installed.

The adapter encodes the exact pixels as lossless PNG, capped at 8 MiB; the ASCII-escaped JSON request is capped at 12 MiB. Record/question bounds match [TextContextCalls](text-context-calls.md). PNG encoding runs in an owned background thread which cancellation/close joins. Python cannot forcibly stop the encoder, so there is no hard realtime guarantee. Preparation occurs before admission; a denied request never invokes the client.

The completion coroutine receives a private dictionary with provider, model revision and the same text messages as TextContextCalls, plus:

```python
'images': [{
    'mime_type': 'image/png',
    'data_base64': '...',
    'sha256': 'SHA256 of encoded PNG bytes',
    'source_frame_sha256': 'versioned shape-bound source pixel digest',
    'capture_ms': 1234,
    'shape': [480, 640, 3],
}]
```

PNG stores normal image channels; shape refers to the original BGR array. The request hash covers exact encoded bytes, source digest, metadata, question, observation and declared aliases. It does not prove unchanged transmission or invocation of the claimed model. Endpoint credentials, remote-image consent, deadlines, retries and client lifetime belong to the application.

Return exactly `{'text': nonempty_text, 'usage': None}` or the bounded declared usage documented for TextContextCalls. The timeout, one-operation/no-queue, cancellation cleanup and self-close contracts are shared. Host pause/reset/revoke/delete/close invalidates late replies but does not own the application client.

Accepted output includes `interpretation='untrusted_visual_response'`, original capture timestamp, image/source hashes and `quality_validated=false`. It is returned only to the caller and never inserted as a sensory observation. Successful visual completion acknowledges visual semantic coverage at the original capture time, not recognition correctness. Empty/malformed output, timeout, failure and invalidated replies cannot satisfy it. MCP status retains bounded hashes rather than PNG or answer bodies.

Tests exercise actual PNG round trips and async fixture clients, not cloud deployment or trained-model quality.
