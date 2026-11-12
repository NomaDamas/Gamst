# Measured latency, with explicit scope

## Runtime instrumentation

`await human.get_metrics()` (MCP `get_metrics`) reports monotonic processing durations for SDK inputs, state/context queries, reasoning, and each registered plugin. Capture status includes camera read time, video/audio processing and audio queue wait. Percentiles use the latest 10,000 observations per stage; count/mean/max/total cover measurements since reset. Stages with no samples are absent, not reported as zero. A failed processing attempt is still timed. No payloads or participant IDs are retained by telemetry.

These are wall processing times, not model accuracy or sensor clock synchronization. Nested stage totals overlap and must not be summed. Plugin timings include inference/preprocessing and any await time. Built-in telemetry does not separate neural-network kernel execution from host scheduling.

## Actual MCP pipe measurement

Measured 2026-11-12 on this development environment, Python 3.12.3, with **100 sequential synthetic text inputs** through a real MCP subprocess:

| Operation | p50 | p95 | p99 |
| --- | --- | --- | --- |
| push_text round trip | 2.07ms | 3.56ms | 3.76ms |
| get_state round trip | 0.44ms | 0.72ms | 0.84ms |

Initialize/process-start round trip: 85.40ms. Includes JSON, pipe transport, storage and rules. Excludes camera, microphone, video/VAD/ASR/image models, GPU and network transport. The single get_metrics sample is not a meaningful percentile population. See the committed [JSON report](../benchmarks/results/stdio-20261112.json) for environment and measured stages. These measurements are not a claim that model event latency meets 200ms.

```bash
python -m benchmarks.stdio --samples 100 --output /tmp/new-stdio-report.json
python -m benchmarks.runtime --samples 200 --json-output /tmp/new-core-report.json
```

## Actual model replay

After installing optional dependencies and licensed local weights, use consented explicit media:

```bash
python -m benchmarks.models --consent \
  --image /path/to/consented-image.jpg \
  --yolo-model /path/to/trusted-yolo.pt \
  --repeats 100 --output /tmp/new-yolo-report.json

python -m benchmarks.models --consent \
  --wav /path/to/consented-16khz-mono-pcm16.wav \
  --vad-model /path/to/silero.jit --asr-model /path/to/whisper-directory \
  --repeats 1 --output /tmp/new-audio-report.json
```

Image replay is not tracking/FPS on moving video. Repeated audio is not independent validation data. The report stores only timing/counters/environment/asset hashes, not images, waveform or transcript contents. Missing files/models/dependencies cause failure, never an invented success report. Runtime inference may retain transcripts in its in-memory store during the test, discarded on close.

## Live capture measurement

```bash
python scripts/verify_live.py --consent --camera 0 --audio \
  --yolo-model /path/to/yolo.pt --vad-model /path/to/silero.jit \
  --asr-model /path/to/whisper-directory --seconds 60 \
  --output /tmp/new-live-report.json
```

This reports actual frame count/FPS, plugin p50/p95/p99, callback queue wait, drops and state-query processing time. Per-operation target comparisons are explicitly named; unmeasured stages are `not_measured`. Fixed-window ASR processing is **not** measured end-of-speech-to-text delay. The v0.1 Whisper default buffers 3 seconds, so a <=1 second speech-end update goal is not established by these timings. Ground-truth speech-end labels and streaming behavior need separate implementation/evaluation.

No real model replay/live hardware report has been generated in this environment: devices, models and optional dependencies are absent and network access is blocked. The core/MCP measurements above are actual results within that narrower scope. Eight-hour camera/model soak, real ASR speech-end latency and predictive validity remain incomplete.

## Speech-finalization path

`await human.flush_audio(subject_id=...)` finalizes residual audio for plugins supporting flush. The Whisper adapter consumes the selected subject's remaining >=100ms audio using its actual duration; shorter residuals are discarded. Empty/already-consumed buffers produce no duplicate transcript. The default transcript timestamp is the last captured audio-block end, not the later wall-clock time at flush.

With a registered VAD emitting `speech.stopped`, `push_audio` processes the complete block through all plugins first, then flushes residual ASR buffers. This permits shorter-than-window utterances to be transcribed at an observed activity transition. The measured stage `pipeline.speech_stop_decision_to_flushed_transcript` uses a monotonic clock from the runtime VAD decision to flush transcript publication. It excludes acoustic speech-end detection delay and hardware callback buffering, and therefore is not the full <=1s user-perceived latency target. A new speech-start clears stale timing anchors; pause/reset/delete invalidates pending finalization.

A mock-model regression proves ordering, residual consumption, subject isolation and no duplicate finalization. Actual Whisper/Silero replay and ground-truth end-of-speech latency remain unverified until the host can supply dependencies, assets and consented audio.
