# Streaming refactor QA

Validation on the local Apple M2 Pro checkout, 2026-11-13:

- 138 Python tests: existing runtime contracts plus independent media lanes,
  bounded video queues, audio gaps, lifecycle cleanup, consent, WebSocket ingress,
  routed Pipecat bridge, classification provenance, entry-point discovery and
  Ollama NDJSON token/cancellation contracts.
- JavaScript renderer checks: scalar gauges, context table, stale readings,
  dominant category, relative score portions, 60-sample history cap, and identical
  chart assets in the local dashboard and public site.
- Real local WebSocket/model fixture test: 12 video packets and 12 audio blocks
  over approximately 6 seconds; 0 audio-router drops; median context cadence
  255 ms; maximum paired video/audio acceptance response 51 ms. The retained VLM
  result included a 6-frame observation window. Inputs were an official portrait
  fixture and generated Korean speech, not newly captured camera/microphone data.
- Actual HTTP transport test with a fixture NDJSON server verified incremental
  delivery through httpx. This is not a claim of testing an Ollama model daemon.
- Wheel build succeeded and includes media routing, both optional provider modules
  and the shared chart asset.

The prior temporal-model fixture checked 8 frames over 3.5 seconds at 2 FPS, with
numerical observations routed into MCP; its measured inference time was 5.674 s.
Model latency is separate from the stream's update cadence. Static portrait
fixtures verify transport and integration, not recognition of changing actions.

Physical devices were not activated for this refactor. Browser automation access
was denied, so visual layout and actual browser-click interaction remain manual
checks. Hugging Face integration was tested with an injected classifier fixture;
no new classification model was downloaded. The existing local Whisper, Silero,
MediaPipe and Qwen installation is unchanged.
