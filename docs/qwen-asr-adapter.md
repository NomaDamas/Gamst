# Qwen streaming lifecycle adapter

`humansense.providers.qwen_asr.QwenStreamingASR` implements the concrete Qwen vLLM streaming API described in the [source-verified contract](asr-streaming-contract.md). It consumes fixed mono hops, emits replacement transcript revisions and flushes a residual tail. This is an adapter implementation with test-double lifecycle evidence; actual Qwen checkpoint inference is not yet validated (#68).

## Explicit engine ownership and limits

Pass an already explicitly loaded `Qwen3ASRModel` with `backend='vllm'`, the three official streaming methods and `sampling_params.max_tokens` in 1–512. `model_version` must identify the caller's immutable model/runtime manifest. The adapter checks the API boundary, not artifact hashes or class authenticity. A local-only verified loader, isolated worker profile and recorded runtime compatibility remain implementation requirements in #67. Construction imports NumPy only; it does not import Qwen/vLLM, load weights, download files or start a sensor. The caller owns engine disposal and must not share a mutable engine concurrently between lanes.

Input must be one-dimensional mono float32/float64 normalized PCM or int16 at 16 kHz. Declare exactly one sensor channel and pass a complete fixed hop of 10–100 integer milliseconds (default 20). Copies protect the caller's PCM. Chunk size is 100–2000 ms (default 2000), divisible by hop size. Segment limit is at least a chunk, at most 30000 ms, also divisible by hop size. These are resource policies, not verified recognition-quality settings.

Capture-end timestamps must advance by exactly one hop with consecutive input sequence and stable subject/source/sensor/epoch/channel ownership. A mismatch rejects the current hop, clears adapter state and requires the runtime to interrupt the prior partial before retrying the retained hop. SensorStreams already interrupts affected transcript lanes on provider errors/gaps. Direct callers must perform that scoped interruption themselves. Do not join audio across discontinuities or label discarded state final.

At the segment limit, the adapter calls the official finish API, emits a final revision and releases accumulated state. The next hop starts a new utterance ID. This deterministic cap is a forced boundary, not a detected speech endpoint; its configuration must be retained with the run manifest. No overlap stitching or quality-preserving segmentation claim exists. Empty output before decode emits nothing. Unchanged partial text/language is suppressed. Finals are emitted even if empty so an earlier partial can be closed.

## Publication and end-of-input

`process(frame)` returns one transcript dictionary or None. It can run as a SensorStreams provider for automatic capped segments. `finish()` flushes a pending tail once without new audio and returns a final dictionary; repeated finish returns None. `close()` discards pending state, so finish explicitly when a successful end-of-input is intended. Final publication support through SensorStreams control operations is still pending in #67; do not assume clearing/closing a lane finalizes it.

For an explicit driver, keep publication sequence separate from capture-input sequence, including all partials. Call process/finish/reset/close serially in the selected worker, not on an event loop while inference can block. A simplified publication fragment after a successful serialized worker operation is:

```python
# Register adapter.spec once. update is process(frame) or finish() output.
# output_sequence is a per-output-stream counter, NOT the input frame sequence.
from dataclasses import replace

if update is not None:
    await human.push_observation(adapter.spec.id, update,
        sensor=replace(last_capture.sensor, sequence=output_sequence),
        subject_id=last_capture.subject_id,
        timestamp_ms=last_capture.timestamp_ms)
    output_sequence += 1
```

Use a stable explicit stream epoch and scope interruption to subject/source/sensor/capability/epoch on error, clear or cancellation. After publication failure, reset and interrupt the affected partial; the adapter does not retry or roll back already advanced revisions. Do not let late worker completions publish after a reset. The managed runtime must enforce this; the adapter itself is synchronous and is not a queue or cancellation boundary.

## Meaning of transcript fields

Each segment has a unique ID, revision zero on first publication, increasing replacement revisions and immutable final status once published. Start/end describe captured evidence, not word timestamps. Partial end excludes the not-yet-decoded residual buffer; final end includes the flushed tail. Only seven explicit language labels currently map to ISO codes (English, Korean, Chinese, Japanese, German, French, Spanish); other auto-detected labels become `und`. This is not a claim that Qwen supports only seven languages. More mappings require explicit tests. Text beyond 4096 characters or malformed backend output fails rather than silently truncating.

The capability's native streaming mode describes the stateful input API. Qwen still reprocesses accumulated audio; no fixed-cost encoder cache, bounded GPU utilization or real-time performance is inferred from this label. Segment and token caps bound submitted work, not a hung engine's wall time. Runtime watchdog/isolation and bilingual cost/latency evidence remain required.

## Evidence

`python3 -m unittest discover -s tests -p test_qwen_asr.py -v` uses an explicitly labeled API test double, not a recognizer. It verifies chunk arguments, revision replacement, residual finalization, caps/rollover, input ownership, gap/error/overflow resets and real TranscriptMemory publication with a final at the last capture timestamp. No new model, inference runtime, microphone capture or claimed recognition benchmark was installed/executed. #67 and #68 remain open.
