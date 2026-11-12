# Delivery audit — 2026-11-13

This audit separates working code, measured execution and remaining product
capabilities. It is not a goal-state declaration. The source targets v0.3.0; actual downloadable assets and publication state are
listed in GitHub Releases.

## Verified implementation and execution

- **Installable SDK / MCP:** wheel build and clean installation, actual MCP stdio
  communication, CLI and Python/JavaScript contract tests.
- **Real-time context:** separate bounded audio/video lanes, model background
  workers, input sequence-gap handling, live HTTP NDJSON/WebSocket snapshots and
  a shared-runtime MCP bridge. Delivery cadence and inference latency are distinct.
- **Local models:** real fixture execution for face landmarks/blendshapes, Whisper
  English/Korean ASR, Silero VAD, Qwen scene descriptions and Companion replies.
  SUPERB voice scores also passed the real fixture → context → MCP path.
- **Physical inputs:** one explicitly consented 15-second native camera/microphone
  smoke test processed 148 frames/windows with no audio drops or device/plugin
  errors. It was not an accuracy benchmark or long-duration soak; no live speech
  transcription was emitted in that run.
- **Dashboard:** tabbed context, observability, category histories, Companion and
  VLM panels; draggable camera sizing; face/person overlays; language selection
  for ASR; model toggles shared with MCP; Advanced capabilities; an actual VLM
  endpoint diagnostic with a synthetic image. UI-language localization and English
  promotional captures are being finalized in the current source work.
- **Local lifecycle:** consent, pause, stale-result rejection, deletion and retention
  tests; project-owned installer/removal checks; launcher-owned process shutdown.
- **GitHub / hosting:** real pushes, CI and public Pages deployment have succeeded.
  Prior DNS/empty-Git-mount restrictions are historical and no longer block this
  checkout.

Evidence: [v0.2 QA](qa-v0.2.0.md), [streaming QA](qa-streaming.md),
[voice analysis](voice-analysis.md), [settings QA](qa-settings.md).

## Implemented contracts with narrower validation

- Conversational observations and structured consistency are rule-based baselines.
  Semantic reasoning adapters validate cited evidence; they do not establish
  external truth, deception, hidden intent or medical conditions.
- Numeric baselines are subject/model-version separated. Feedback and held-out
  calibration infrastructure exists; no deployment-calibrated personal prediction
  model or real labeled validation dataset is claimed.
- Hugging Face supports installed image/audio classification pipelines. Ollama
  token streaming has HTTP/fixture checks, but no Ollama daemon model run is claimed.
  Dashboard Companion replies still arrive as complete replies.
- LiveKit/Pipecat bridges accept decoded media, explicit ownership and clocks.
  Native SDK interoperability requires separate validation.
- One source/subject owns a modality lane per runtime. Simultaneous independent
  audio subjects require separate runtimes; this is not a diarizing multi-user server.

## Person, body and emotion boundaries

HOG person boxes are associated by geometry, with uncertainty and expiry. Faces
can be detected when the body is off-screen, but face indexes are not persistent
identity IDs. Voice emotion remains a microphone-stream observation. Automatic
face/body association, speaker separation, speaker-to-face matching and verified
per-person emotion histories are missing.

Body-pose integration has produced 33 landmarks for one body in a real local
fixture check (29 ms); geometric wrist-above-shoulder indicators are not a trained
gesture classifier. SAM3 has
been reviewed as a possible segmentation/tracking provider, but its weights,
execution adapter and performance are not installed/verified release features.
See [tracking](tracking.md) for the explicit extension contract.

## Public versus local experience

The public browser preview measures microphone level and face movements. It does
not run the full local Whisper/SUPERB/Qwen stack. The local dashboard, its actual
VLM test and MCP service require local installation. Promotional material must
keep this boundary visible and mark fixture results as examples.

Remaining release gates are the new version's package/artifact verification and
publication and final localization/presentation checks. Long-duration soak, perception accuracy, person identity
association and broader application outcomes remain separate evaluation work.
