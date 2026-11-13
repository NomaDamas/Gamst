# Streaming ASR execution contract

This is an implementation contract, not a claim that a new ASR backend is installed or validated. Required family 7 remains open in #23. Silero VAD and transcript revision storage already exist; actual streaming ASR inference, integration and bilingual validation remain required.

## Source-verified Qwen behavior

Audit target: [Qwen3-ASR commit 7c6daf77a2421100f5fb066495372c00129d39ff](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py). Its `init_streaming_state`, `streaming_transcribe` and `finish_streaming_transcribe` APIs maintain one mutable stream. Input is mono 16 kHz PCM. New chunks are appended to accumulated audio and the full prefix is submitted again, with text-prefix rollback. Finalization flushes residual audio. Streaming requires vLLM, has no batch support and does not produce word timestamps. State persistence therefore does not establish bounded-cost incremental encoder execution.

The same revision's [dependency manifest](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/pyproject.toml) declares qwen-asr 0.0.6, transformers 4.57.6 and optional vLLM 0.14.0. These are source-declared candidates, not a gamst-tested runtime profile. The repository code declares Apache-2.0; checkpoint identity and license must be checked separately before distribution.

## Runtime responsibilities

The adapter must distinguish continuous capture, input chunk buffering, model decode, publication and finalization. User context delivery cadence must not force repeated model calls or become an assumed recognition cadence. A worker must own its model state and serialize operations for one stream; separate streams must not share mutable state. Blocking decode belongs in an isolated, explicitly selected runtime so sensing and playback lanes continue.

For Qwen's accumulating path, impose explicit segment/audio/token limits before model invocation. A starting implementation target is a configurable maximum segment of 30 seconds, with a smaller limit allowed; this is a resource policy, not a proven accuracy setting. Define and test rollover boundaries, final tail flush, empty segments, VAD endpoints, long speech and continuity across adjacent segments. Never silently discard excess audio or call a truncated prefix a full utterance. Record forced segmentation as provenance. A practical implementation may need overlap, but overlap must not duplicate transcript text or be invented as verified quality preservation.

Do not label the accumulated-prefix implementation as cached-encoder or constant-work streaming. The approved native state-continuing inference requirement remains: demonstrate the intended runtime behavior with actual measurements, and provide a separately validated backend if this path does not meet it. Qwen can still be a supported bounded-segment alternative; this does not waive the native requirement.

## Transcript and timing semantics

Use existing registered `speech.transcript_update` observations and full-replacement revisions. Each segment starts at revision zero with a unique utterance ID. Partial revisions may replace earlier text; final text is immutable. A flush must still be publishable when no new input arrives. Output revision sequence and capture-input sequence are separate counters; do not reuse an already-published sensor sequence for a final update.

Capture start/end are the segment evidence interval, not word alignment or acoustic emission timing. A backend without alignment cannot manufacture token timestamps. Normalize language labels through an explicit mapping with an unknown value, rather than guessing identity or language from empty output. Reject or visibly report text beyond the existing 4096-character transcript bound; silently cutting recognized text is not acceptable.

On gap, owner change, model error, cancellation or queue drop, discard affected model state and close or mark the prior partial as interrupted through an explicit runtime operation. Current transcript interruption behavior must be checked for per-stream isolation before reusing it; an unrelated speaker's stream must not be interrupted. Never finalize incomplete audio as a successful full utterance. Reset/reconnect must not reuse closed utterance IDs or publish stale worker completions.

## Loading and validation gates

Loading must accept only an explicitly configured local model artifact with immutable identity and a declared runtime profile. Reject model IDs/URLs in a local-only loader. Validate required files and checkpoint hashes before loading, configure offline dependency behavior and test that missing assets fail without network access. Use the existing isolated-provider approach where compatible; its current profiles do not establish a Qwen runtime. No model weights or vLLM installation are performed by this document.

Contract tests may use an instrumented test double to verify lifecycle and exact API arguments, but cannot establish recognition support. Required real execution must include Korean and English labeled audio, silence, speech/noise, revisions, non-aligned tails, interrupted utterances, concurrent playback and a stalled worker. Report capture-to-partial and capture-to-final p50/p95 separately, model time, growing-prefix cost, segment-boundary errors, drops and process/GPU memory. Record exact model/runtime hashes, fixture provenance and reference scoring. Add the exact inference reproduction command with the implementation; the upstream demo's remote downloads are not an approved local-only validation command.

This contract preserves #23's full implementation and actual-inference gate. Halo remains deferred; cheaper/faster VLM and preserved quality remain separate measured acceptance requirements.

## Finalization provenance

`transcript_capability` declares optional `finalization_reason`. When present it
is accepted only on a `final` update and must be `input_end` (the caller explicitly
ended the submitted input) or `segment_limit` (the provider forced a bounded
segment rollover). It is retained unchanged in observations, transcript memory,
context and MCP queries, and becomes immutable with the final text. Absence means
the provider did not declare why the segment ended; do not infer silence or a
natural sentence boundary. Existing seven-field transcript producers remain valid.
Older SDK consumers that require exactly seven fields must upgrade to receive
this extension; provider workers and parent SDK versions must match.

Qwen emits `segment_limit` when its configured maximum accumulated duration is
reached, then begins a new utterance ID on the next hop. Explicit `finish()` emits
`input_end` for a remaining segment. Partial updates have no finalization reason.
These describe software control boundaries, not detected human turn completion,
semantic sentence completeness, VAD accuracy or recognition correctness. No audio
sample or capture timestamp is invented for either operation. A final exactly at
the cap keeps `segment_limit`; a subsequent empty finish produces no second final.
VAD/manual utterance endpoint integration remains required in #67; this field
alone does not implement it.
