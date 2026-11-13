# Local VLM call diagnostics

The existing `LocalScenePlugin` now records operational call metrics around its synchronous description path. Python `scene_status(human)`, MCP `get_vlm_status` and the existing dashboard status API expose `call_metrics`. Reading status never activates a sensor or calls the model. Metrics are separate from `settings()` so sampling windows do not recursively embed historical telemetry.

`started` counts description-processing attempts, including local preflight failures. `http_attempts` counts entries into the HTTP opener, including connection failures; it cannot prove that a server executed/billed a request. `completed`, `errors` and `in_flight` describe the wrapper lifecycle. A malformed answer is an error even if HTTP succeeded. If the answer includes usage, those reported counts remain observable even when the answer is unusable.

Each recent completion reports a local `call_id`, success/error outcome, whether HTTP was attempted, `call_processing_ms`, whitelisted usage and usage status. Timing uses a monotonic clock from description preparation through HTTP and response parsing. It excludes prior JPEG preparation, frame-window collection, queueing, ingestion and consumer answer generation. Accordingly `task_answer_ready_latency_ms` and `cost` remain null. These diagnostics must not be fed into the evaluator as end-to-end latency or complete charged usage without the remaining instrumentation in #57.

## Usage and accounting boundaries

The supported response shape follows the [Chat Completions usage fields](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/list): prompt/completion token counts and optional prompt details. The SDK preserves `prompt_tokens`, `completion_tokens`, `cached_prompt_tokens` (from `cached_tokens`) and `cache_write_tokens`. Missing fields stay null; absent, partial, fully reported and structurally invalid usage are counted separately. Present counts must be nonnegative integers at most 10^12, and cached read count cannot exceed a reported prompt count.

Unknown response fields are not retained. A malformed usage object does not prevent an otherwise usable scene description; it makes accounting invalid/unknown. “Reported” means the four fields were supplied and passed these structural checks, not that a bill, tokenizer or model execution was independently verified. Provider-specific cache-write accounting can differ; this collector does not subtract cache fields into exclusive billing buckets or infer charges. `reported_token_totals` sums known fields only, and `token_observation_counts` reports how many completed calls supplied each field. A zero sum with zero observations means unknown, not zero usage. Errors/timeouts without usage do not establish zero cost.

## Lifetime, boundedness and cancellation

Updates and snapshots are lock-protected because HTTP inference runs in a worker thread. Keep at most 128 completed records internally; status returns the newest 16 in completion order and `omitted_calls` identifies all earlier completions. Counters/totals cover this plugin instance's lifetime and persist across sampling resets, subject changes and configuration pauses. No subject IDs, prompts, images, descriptions, URLs, response bodies or exception strings are stored in these metrics. Snapshots are detached copies. This is bounded diagnostic sampling, not a complete evaluation trace archive.

Cancelling the async waiter does not necessarily stop an underlying synchronous HTTP request. That request stays in flight in metrics until its wrapper exits. A successful HTTP description may then be discarded by runtime generation/consent checks; metrics success must not be interpreted as delivered context or task success. The existing request timeout and runtime invalidation rules remain unchanged.

## Verification and remaining work

`python3 -m unittest discover -s tests -p test_vlm_metrics.py -v` covers missing/invalid/cache usage, concurrent updates, bounded snapshots, errors with usage, preflight-versus-HTTP counts, no media/error leakage, a blocked request and an actual loopback HTTP exchange. The loopback server returns generated JSON; it is transport/collector evidence, not model inference. No model weights or sensor devices are installed/opened by these tests.

Whole-pipeline trigger-to-answer-ready timing, provider/model provenance binding, exclusive tariff normalization, complete retained traces, independently scored held-out runs and real VLM efficiency acceptance remain open in #57/#10. This increment does not complete those gates.
