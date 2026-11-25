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

## Text context

When the semantic lane is on, what the same run transcribed and counted goes with the frames of a segment as text data
(`config.context`, recipe default `["transcript", "counts"]`). Off: `--vlm-context none` (or one of the two), `"context": []` in the
recipe item, or MCP `configure_pipeline_module` `{"transcript": false}`. Nothing is waited for: utterances and events that overlap the
segment and are in the Store when the last frame is sent, current revisions only. At most 16 utterances, 500 characters each, 4000
together, 32 events, 8 counts.

It travels beside the frame through the existing worker operation (`process_regions`) and reaches the model as the JSON field
`untrusted_data`, next to an instruction that nothing inside it is an instruction, not to count and not to change the counts. The record's
`context` lists the record IDs, utterance IDs, revisions and values given and `as_of_ms`; `depends_on` makes `get_timeline` show
`depends_on_changed` when one was revised later. A recipe cannot list `vision.streaming_semantic` in `records`.

Tests check the request's structure. Whether a given VLM obeys text in a transcript is not tested. With a hosted endpoint
(`allow_remote`) the text leaves the machine like the frames; the run prints a note.

## Acceptance gate

The provider contract tests validate chronology, window bounds, previous-state
handoff and evidence IDs. They do **not** establish StreamingVLM quality or
latency. Before claiming SOTA or realtime acceptance, freeze the exact upstream
source/checkpoint hash, serve it behind the endpoint, and measure state quality,
latency, dropped windows, GPU memory and long-stream stability on the GAMST
semantic benchmark.
