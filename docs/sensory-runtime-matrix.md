# Sensory model and runtime matrix

This matrix is the implementation inventory for the approved GAMST runtime
plan. It describes the provider and environment selected by configuration; a
provider entry or passing fixture does not claim that weights, hardware,
quality, latency, or production acceptance has been measured.

The current cross-repository probe heads are GAMST `fc1ab04`, GamstCrew
`e864817`, and Gamst-bench `576494b`. The probe proves packet/checksum
compatibility only.

| Sensory area | Current model/provider | Isolated runtime and profile | Configuration and artifact rules | Current evidence and remaining gate |
|---|---|---|---|---|
| Camera, face | MediaPipe Face Landmarker; landmarks/blendshapes; optional head orientation | Core browser path uses pinned MediaPipe CDN; local worker `mediapipe.face` or `mediapipe.face.video` | Explicit `.task` path plus SHA-256 for local workers; browser uses pinned CDN/model URLs | UI and provider contracts are tested; real camera permission, device behavior, and quality acceptance remain open |
| Body, hands, gestures | MediaPipe Pose/Hand/Gesture | Vision worker `mediapipe.body`, `mediapipe.hand`, `mediapipe.gesture.video` | Separate local task files and hashes; bounded entity/continuity settings | Fixtures and schema pass; real multi-person/device validation remains open |
| Objects and tracking | RF-DETR Nano; IoU/OC-SORT-style stream tracking; optional YOLOE | Detection worker `rfdetr.nano`; tracking state remains in bounded Core stream; YOLOE is a separate worker | RF-DETR 1.12.1 checkpoint path/hash; detection extra keeps Transformers 5 separate from Core/Transformers 4 | Isolated local RF-DETR evidence exists; target-host quality/latency and long-run acceptance remain open |
| Segmentation and masks | EfficientTAM tiny512; optional SAM3/SAM3.1 video | `efficienttam.tiny512` worker; `sam3.1.video` isolated Transformers/Torch runtime | EfficientTAM pins Torch 2.8; SAM3.1 pins Torch 2.10 and requires explicit gated local artifacts | Contracts/fixtures exist; gated weights, license, native runtime, and real video acceptance remain open |
| Depth and geometry | Depth Anything V2 Small; optional Video Depth Anything Small | `depth_anything_v2.small` and `video_depth_anything.small` isolated workers | Explicit local path/hash; VDA has its own Torch environment and is never reused implicitly | Relative/uncalibrated depth context is implemented; metric 3-D quality needs calibration and target-device evidence |
| OCR | PP-OCR mobile | `ppocr.mobile` vision worker with ONNX Runtime | Detector and recognizer paths plus independent SHA-256 values are required | Bounded text-region contract is tested; multilingual accuracy and sustained runtime acceptance remain open |
| Audio speech | Qwen3-ASR 0.6B; Faster-Whisper compatibility path | `qwen3.asr` or `qwen3.asr.endpoint` in dedicated audio runtime; endpoint may pair with Silero | Qwen local bundle requires explicit file inventory/hashes; endpoint requires ASR + VAD artifacts and timing bounds | Streaming/revision contracts and fixtures pass; real weights/device and measured streaming acceptance remain open |
| Voice activity | Silero VAD | `silero.activity` audio worker or paired endpoint profile | Explicit local JIT path/hash; no implicit download | Local adapter is implemented; microphone/device and accuracy acceptance remain open |
| Audio events and prosody | YAMNet, CED-Tiny (optional GPL), prosody/emotion2vec adapters | `yamnet.events`, `audio.prosody`, `emotion2vec.seed.onnx`; each in its selected audio environment | YAMNet artifact/hash; CED is opt-in and isolated; emotion model/head paths and hashes | Contracts and synthetic/replay checks exist; license, weights, and held-out quality acceptance remain open |
| Audio alternatives | Moonshine v2, TEN-VAD, diarization/TalkNet | Separate audio workers; diarization may require Transformers 5 | Each profile has explicit local source/checkpoint and license gates | Optional integrations are documented; no default or production-quality claim is made |
| Text and LLM/VLM consumer | OpenAI-compatible endpoint, MyProxy Terra, local Qwen3-VL 2B, SmolVLM2 | Core sends bounded context text; `gamst-vlm` uses llama.cpp/MLX/vLLM; Playground BYOK is separate | Local MyProxy shortcut is ignored by Git, session-only for keys, and defaults to `gpt-6.1-terra` with low reasoning | UI/request transport and local Qwen smoke exist; user endpoint/key and matched quality/cost runs remain open |
| Decision routing | Deterministic rules; optional Intern-Decision 2B/4B | Core/router runtime; optional model worker with its own Torch/Transformers pins | Typed actions are bounded; model artifacts and calibration are explicit | Decision contract is implemented; calibration and task-quality acceptance remain open |
| Tactile | DIGIT/GelSight capture; Sparsh slip/task head | `gamst-tactile`, profile `sparsh.slip.local` | Sparsh extra pins Torch 2.8; capture metadata and source/license must be explicit | Packet/replay and adapter contracts exist; physical hardware, checkpoint, and task-head acceptance remain open |
| Radar/RF | TI xWR6843 radar; ESP32 CSI/RF variation | Sensor workers `ti.radar` and `esp32.csi`, CPU-side decoding | Explicit sensor clock/calibration and bounded raw packet schema | Parser/provider tests pass; real hardware and calibration acceptance remain open |
| Thermal/environment | Radiometric `TLinearThermal`; BME688/BSEC2 environment context | CPU sensor runtime, `bosch.bsec2` and thermal providers | Raw units and calibration ID are mandatory; BSEC/license/source constraints remain explicit | Synthetic/replay validation exists; device and trained-pattern acceptance remain open |
| Physics context | Deterministic kinematics, velocity/acceleration/jerk, covariance, trajectory envelope, contact/collision candidates | `gamst-physics` deterministic CPU provider; optional physical-process bridge | Uses capture timestamps, common clock, calibration and evidence lineage; with 2-D uncalibrated input it must abstain from force/mass/precise 3-D claims | Focused physics tests pass; held-out calibrated tracks and runtime wiring/acceptance remain open |
| Learned world models | Cosmos/V-JEPA/Genesis-style future prediction | Optional isolated research runtime only; not a default provider | Synthetic futures are labeled and excluded from real evidence and benchmark ground truth | Intentionally not wired into the current product acceptance; research issue remains open |

## Runtime boundary

`gamst-core` owns packet schemas, timestamps, evidence, bounded queues and
memory. GamstCrew starts each selected provider in its own interpreter and
performs readiness, artifact, host and Python-version preflight before launch.
The approved environment split is:

```text
gamst-core       Python 3.11; context, time, evidence, routing
gamst-vision     PyTorch/ONNX/MPS/CUDA vision workers
gamst-audio      PyTorch/ONNX audio workers
gamst-sam31      separate Transformers/Torch SAM runtime
gamst-vlm        llama.cpp, MLX or vLLM
gamst-tactile    Sparsh/DIGIT/GelSight runtime
gamst-physics    deterministic CPU physics and optional research bridge
gamst-bench      frozen manifests, adapters and scoring
```

No provider is selected as the Mac default from a CUDA result. Default
promotion requires matched B/C/D/E measurements on the target hardware,
including quality, on-time grounding, p95 latency, TTFT, memory, drops,
energy/resource cost, evidence correctness and temporal error.
