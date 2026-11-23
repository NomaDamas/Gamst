# Explicit on-demand SmolVLM calls

`SmolVLMCalls` connects a registered `CallController` small_vlm request to the local SmolVLM2 `ProcessProvider`, then publishes accepted context through Gamst's observation/MCP path. It does not select models, generate requests, change sensor cadence or substitute for strong_vlm/text routes.

```python
from gamst import SmolVLMCalls, smolvlm_frame_sha256

# The controller request, original registered record and original InputFrame
# must already exist. config contains explicit local assets and SHA256 values.
async with await SmolVLMCalls.open(
    human, "controller-id", "/existing/environment/bin/python", config
) as calls:
    result = await calls.execute(request_id, record, frame)
```

The environment must already contain the pinned native dependencies and SDK. `open` accepts the native configuration and fixes `window_frames=1`; the separate streaming provider retains multi-frame configuration. Model download, application transport and accounting are outside this adapter.

## Admission and evidence

Only a pending small_vlm request is accepted. The original record hash, subject, timestamp and complete sensor metadata must match. Retained uint8 BGR pixels are bounded to the native 1080p limit, copied and checked against the visual-reference SHA256 before any await. Create that reference with `smolvlm_frame_sha256(frame.data)`: this versioned digest binds uint8/HWC/BGR format, shape and C-order bytes, so identical bytes with different geometry do not authorize the capture.

One adapter permits one active call; a second call fails immediately rather than building an invisible queue. Each call resets window state so nonconsecutive retained frames are valid. Reservation and dispatch recheck host lifecycle and freshness. Optional `dispatch_deadline_ns` is a bounded absolute reading of this host's `time.monotonic_ns()` checked after reset and before dispatch; expiry returns `blocked` / `dispatch_deadline_exceeded` without entering the backend. Execution provenance binds the real profile/version and native timestamped prompt hash.

The worker result must match retained capture, pixels, shape, source metadata, assets and prompt. Only nonempty described output acknowledges semantic coverage. Generated text remains an untrusted semantic hypothesis with `quality_validated=false`; execution does not establish recognition accuracy.

## Publication and lifecycle

Accepted results are published at the original capture timestamp under the registered SmolVLM capability and become available through normal Python, observation streaming, store/history and MCP reads. Completion time does not make an old capture fresh.

Pause/reset, subject deletion, consent revocation and host close invalidate the source or processing epoch. A result finishing afterward is discarded and not republished. Gamst and the adapter are used on one event loop. Caller-owned returned copies have their own retention obligations.

The application owns the adapter lifetime. Use its async context manager or `await close()` even if Gamst has closed. Adapter close cancels active execution and reaps its owned worker; timeout or cancellation closes that worker, so reopen explicitly. No other process or session is signalled.

Tests use owned pipes and fixture workers for admission, publication, evidence mismatches, cancellation, busy handling and lifecycle invalidation. Native CI can execute the production loader with a generated untrained tiny architecture; this is not trained-model quality evidence.
