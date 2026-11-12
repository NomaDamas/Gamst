# Third-party source and model policy

This checkout contains original integration code. Model weights are downloaded
separately into the ignored project-owned `.local-models` directory and are not
included in GitHub source or Python wheels. Optional dependencies retain their
own licenses. A code license and a model-weight license are separate obligations.

Network access is available in the current host environment: official model
artifacts have been downloaded, upstream documentation consulted, and repository
changes pushed. Earlier documentation saying no remote access was possible
referred to a previous sandbox. No inspected or reused REA implementation is
claimed here.

## Installed integrations and artifact provenance

- **OpenCV:** built-in HOG person detector, normalized boxes and local IoU tracking.
- **MediaPipe Tasks:** explicit face-landmarker task file, landmarks and blendshapes.
  Body pose is also implemented with explicit local task weights and a real
  33-landmark fixture check; gesture interpretation and identity remain separate.
- **Silero VAD / PyTorch:** local JIT voice-activity model.
- **faster-whisper / CTranslate2:** local Whisper small weights, windowed ASR.
- **Qwen3-VL / llama.cpp:** local GGUF vision-language model and projection weights,
  served by an explicitly downloaded local runner.
- **SUPERB / Transformers:** optional `superb/wav2vec2-base-superb-er` weights for
  four voice-emotion scores, installed by `scripts/setup_voice.sh`.
- **sounddevice / PortAudio:** optional native microphone I/O.
- **websockets / httpx:** optional local streaming transports and provider clients.

`scripts/local_models.py` records download URLs and SHA-256 hashes in
`.local-models/manifest.json`; model revisions are pinned where supported. Review
upstream asset licenses, notices and permitted redistribution terms for the exact
artifacts. The installer downloads into the user's checkout; it does not grant a
new license to redistribute those weights. The public browser preview separately
loads MediaPipe code and its face-model asset from the documented external URLs.

## Extension boundaries

SAM3, YOLO, Pipecat and LiveKit implementations are not vendored. SAM3's upstream
segmentation/tracking capability has been reviewed, but a working SAM3 execution
adapter, weights and local Mac performance are not validated. The existing plugin
interface is an integration boundary, not proof that every external model is
plug-and-play. The LiveKit/Pipecat decoded-media bridges likewise do not establish
compatibility with every framework release.

Hugging Face and Ollama interfaces and Pipecat pipeline concepts are credited in
[streaming architecture](streaming-and-plugins.md). These are protocol and design
references, not claims of copied code or endorsement. Remote provider connectors
require explicit configuration and credentials; use of a paid provider is not
included in a local installation. Never include credentials in logs, examples or
release assets.
