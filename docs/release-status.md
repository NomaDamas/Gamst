# Release and local delivery status

This checkout targets **v0.3.0** (2026-11-13), including streaming, voice emotion,
body pose, multilingual settings and the English presentation. The previous
release was v0.2.0. Downloadable tags, wheels and publication state are listed on
[GitHub releases](https://github.com/NomaDamas/Gamst/releases).

## Implemented and exercised

The Python SDK provides timestamped observations, events, state, context and
provenance, with bounded audio/video routing, background inference, MCP stdio,
a shared-runtime WebSocket bridge and a local browser dashboard. The dashboard
includes category histories, tabbed panels, camera-panel resizing and model
switches. A context-aware text Companion and temporal VLM are optional consumers
and producers of context rather than requirements for using the SDK.

Project-local installation has been exercised with MediaPipe face landmarks,
Whisper small, Silero VAD, Qwen3-VL and SUPERB voice-emotion classification.
Real fixture inference, live HTTP/WebSocket/MCP communication, wheel installation,
model removal safeguards and a consented 15-second native camera/microphone
smoke test have been recorded. Voice-emotion scores describe a microphone stream,
not a verified visible person's feelings. English training does not establish
Korean or Chinese emotion accuracy.

Current evidence is in [v0.2 QA](qa-v0.2.0.md),
[streaming QA](qa-streaming.md), [voice analysis](voice-analysis.md) and
[settings QA](qa-settings.md). Each report identifies its tested revision/scope;
older test totals and browser-access limitations are historical, not statements
that present model assets or network access are absent.

GitHub access, source pushes, model downloads and Pages deployment have succeeded
in the current host environment. The earlier restricted-sandbox network and Git
mount limitations no longer describe this checkout.

## Public preview versus local application

The public website is a lightweight browser preview with microphone level and
MediaPipe face measurements. It is not the Python model stack: it does not run
local Whisper, SUPERB voice emotion or the local Qwen VLM. The full application
runs at `http://127.0.0.1:8766` after local installation; its VLM diagnostic uses
a synthetic image without opening camera or microphone. Public presentation
screenshots and examples must identify their input and must not imply that the
public preview has every local feature.

## Remaining boundaries

- Person boxes have stream-local geometric track IDs. Biometric re-identification,
  diarization and verified speaker-to-face/body association are not implemented.
- Face indexes are frame-local; their charts are not stable person-level emotional
  histories. See [tracking](tracking.md).
- MediaPipe body pose is implemented and exercised with a real local fixture:
  33 landmarks for one body, with a measured 29 ms call. This does not establish
  a trained gesture classifier or person identity. SAM3 weights and a tested
  SAM3 execution adapter are not bundled.
- Hugging Face classification is tested with real local voice weights. Ollama
  token streaming and LiveKit/Pipecat bridges have contract/fixture tests; a tested
  Ollama model daemon or complete native-framework deployment is not claimed.
- Long-duration hardware soak, task-success studies and calibrated emotion,
  prediction or personal-adaptation accuracy remain unverified. A passing software
  test suite is not a perception-accuracy evaluation.
