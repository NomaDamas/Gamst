# Third-party source and model policy

This checkout contains original integration code and no bundled model weights or copied third-party implementations. No remote repository was accessible during implementation. In particular, REA (https://github.com/morluto/rea) has not been inspected or reverse engineered, and no claim about its license or usability is made.

Optional dependencies are installed separately. Package license and model-weight license are distinct. Review the exact versions, weights, permitted uses and redistribution terms before distribution:

| Backend | Integration | Assets |
| --- | --- | --- |
| OpenCV | HOG person detection | Built-in detector in user's OpenCV distribution |
| MediaPipe Tasks | Face landmarks and blendshapes | User-supplied local task file, SHA-256 provenance |
| Silero VAD/PyTorch | JIT speech activity | User-supplied local licensed JIT file, SHA-256 provenance |
| faster-whisper/CTranslate2 | Windowed transcript | User-supplied local model directory; local-files-only |
| sounddevice/PortAudio | Native microphone capture | OS audio device and native library |
| websockets | Optional local event transport | No model |

SAM3, YOLO, Pipecat and LiveKit are not vendored. The plugin/push interfaces are extension points, not a claim of a tested native integration. Model names described as 'latest' must be checked against upstream API and licenses before integration; this environment had no network access to do so.

Closed-source implementation cannot be copied without permission. Future provider connectors should use authorized public APIs, explicit credentials, informed data-transfer consent and separately documented retention. This release is local-only.
