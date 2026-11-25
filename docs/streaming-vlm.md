# StreamingVLM provider

GAMST's selected small video-semantic profile is the upstream MIT-Han-Lab
**StreamingVLM** implementation, not SmolVLM. Upstream source:

- repository: `https://github.com/mit-han-lab/streaming-vlm`
- license: MIT
- paper revision reviewed: `arXiv:2510.09608v2`
- model family: StreamingVLM fine-tuned from Qwen2.5-VL-7B-Instruct

The upstream implementation maintains attention-sink tokens, a recent text
window, a recent vision window and contiguous RoPE/KV state. It must be served
by an operator in its own compatible environment. GAMST does not download
weights, install the upstream environment, or claim that a generic endpoint has
StreamingVLM KV-cache semantics.

## GAMST contract

Configure the explicit provider profile `streamingvlm.local`:

```json
{
  "profile": "streamingvlm.local",
  "config": {
    "endpoint": "http://127.0.0.1:9100/v1",
    "model": "streamingvlm-qwen2.5-vl-7b",
    "revision": "PIN_THE_UPSTREAM_CHECKPOINT_AND_SERVER_REVISION",
    "window_frames": 16,
    "max_gap_ms": 2000
  }
}
```

The local OpenAI-compatible bridge sends ordered video frames and the previous
video semantic state. The server returns bounded JSON:

```json
{
  "state": {"phase": "..."},
  "events": [{"type": "...", "at_ms": 123}],
  "summary": "untrusted video-only hypothesis",
  "evidence_ids": ["frame-123"]
}
```

GAMST publishes this as `vision.semantic_state`, then the host fusion layer
combines it with typed audio, ASR, physical and other plugin signals. Strong
VLM/LLM calls remain application-owned and are not triggered by this provider.

## Acceptance gate

The provider contract tests validate chronology, window bounds, previous-state
handoff and evidence IDs. They do **not** establish StreamingVLM quality or
latency. Before claiming SOTA or realtime acceptance, freeze the exact upstream
source/checkpoint hash, serve it behind the endpoint, and measure state quality,
latency, dropped windows, GPU memory and long-stream stability on the GAMST
semantic benchmark.
