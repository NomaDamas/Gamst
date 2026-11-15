# Approved candidate execution and compatibility matrix

Inventory date: 2026-11-15. Scope: every named model, algorithm, sensor,
deployment backend and dataset in the [approved plan](perception-expansion-plan.md).
This is the P11 [#13](https://github.com/NomaDamas/Gamst/issues/13) index;
the [24-family acceptance map](perception-acceptance.md) retains the full product gates.
Historical installation statements in older documents are not an inventory of
assets currently present on a machine.

## Reading and executing an entry

- **Implemented** means a concrete path exists, with the stated software tests.
  It does not mean a trained model, native package or physical device passed
  acceptance. **Recorded execution** points to historical evidence with its limits.
- **Not integrated** means there is no supported candidate-specific invocation,
  selected dependency profile or validated runtime/license combination. There is
  deliberately no validation command for that candidate. The linked open issue
  owns that limitation; a different selected model is not evidence for it.
- **Dataset** is not an executable provider. Training, redistribution and split
  provenance require their own review before any resulting checkpoint is used.
- Every `test_...` name below is an existing module. Run the named modules from
  the checkout with `PYTHONPATH=tests:. python3 -m unittest MODULE [MODULE ...] -v`.
  These are software checks, not instructions to install assets or run live
  experiments. Optional dependency skips must be reported, never counted as
  execution. Linked provider documents contain the separate local-asset replay
  commands where available.
- SDK integration code is Apache-2.0 except retained third-party notices described
  in the linked documents. Each entry separately identifies upstream code/runtime
  and model/device assets. **Unreviewed** means no compatibility or redistribution
  permission is asserted. Even a declared license requires verification against
  the exact supplied revision, binary, checkpoint and training data.

## Runtime selection and isolation

The core package has no mandatory third-party dependencies and requires Python
3.10+. [pyproject.toml](../pyproject.toml) is authoritative for optional extras;
[provider_profiles.py](../gamst/provider_profiles.py) is authoritative for
worker profile names. A generic Hugging Face callback is not candidate support.

Use an explicitly selected, already provisioned interpreter for each worker as
described in [provider processes](provider-processes.md). Do not combine all
extras into one environment. In particular, the Transformers 4.57.6 family
(`depth`, `retrieval`, `smolvlm`, `food`) differs from `diarization`'s Python 3.11+
and Transformers 5.23.0; SAM3 has its own Transformers 5 installation. Qwen ASR
uses a separately declared qwen-asr 0.0.6/vLLM 0.14.0 stack. SAM-Audio and PG
require explicit complete runtime manifests, not an inferred compatible extra.
Current pins select a reproducible target, not a proven trained-model deployment.

Common optional profiles include `vision` (OpenCV >=4.8,<5 and NumPy), `detection`
(rfdetr 1.12.1), `environment-audio` (MediaPipe 0.10.35), `ocr` (RapidOCR 3.10.0,
ONNX Runtime 1.23.2), and `duplex` (pywebrtc-audio 0.2.0). Model extras pin their
Torch/Transformers/native versions separately. The `mediapipe` extra's <0.11
range and explicit 0.10.35 consumers must be resolved in a suitable environment;
do not assume every older MediaPipe Tasks API is interchangeable. An installed
OpenCV 5 development run does not satisfy the declared <5 dependency profile.

Offline core checks: `test_core`, `test_process_provider`, `test_perception_api`.
Discovery starts no model, device or child. Explicit open reports unsupported
profiles, missing assets/dependencies or incompatible versions rather than
downloading a replacement. Optional native binaries remain trusted deployment
inputs; process isolation is not an OS sandbox. No installer is run by this matrix.

## Vision candidates

1. **RF-DETR Nano — implemented; recorded fixture execution.** Worker
   `rfdetr.nano`, `rfdetr.trajectories` and `rfdetr.compensated_trajectories`;
   `detection` plus `vision`, explicit CPU/CUDA/MPS. [Detection](vision-detection.md)
   records limited CPU/MPS inference; crossings, occlusion, live performance and
   calibration remain [#17](https://github.com/NomaDamas/Gamst/issues/17).
   Code/Nano reference weights: Apache-2.0 declared; other RF-DETR variants are
   not included. Checks: `test_detection_provider`, `test_detection_trajectories`,
   `test_compensated_trajectories`.
2. **YOLO26n — not integrated.** No YOLO26n profile, pinned runtime or weight
   compatibility evidence. Code and weights unreviewed for this candidate;
   [#17](https://github.com/NomaDamas/Gamst/issues/17).
3. **YOLOX Nano and YOLOX Tiny — not integrated (both variants).** No selected
   checkpoints/runtime or candidate commands; code/weights unreviewed;
   [#17](https://github.com/NomaDamas/Gamst/issues/17).
4. **ByteTrack — not integrated.** The implemented IoU `PersonTracker` is not
   ByteTrack. Algorithm/runtime license unreviewed; appearance assets, if selected,
   need separate review. [#17](https://github.com/NomaDamas/Gamst/issues/17).
5. **BoT-SORT — not integrated.** No camera-motion/ReID BoT-SORT runtime;
   code and any ReID weights unreviewed.
   [#17](https://github.com/NomaDamas/Gamst/issues/17).
6. **YOLOE — implemented text/exemplar adapter, real acceptance pending.**
   `yoloe.local`, `yoloe` extra, Ultralytics 8.4.212/Torch 2.8.0/TorchVision
   0.23.0, explicit CPU/CUDA FP32. [Contract](yoloe-context.md), `test_yoloe`.
   Ultralytics: AGPL-3.0 or commercial; checkpoint/CLIP/MobileCLIP terms separate.
   [#125](https://github.com/NomaDamas/Gamst/issues/125).
7. **SAM3 sampled images — implemented legacy plugin, not video tracking.**
   `LocalSAM3Plugin`, separate local Transformers 5 worker; CPU/MPS/CUDA explicit.
   [Contract and gated-asset setup](sam3.md), `test_sam3`. Model access and exact
   model terms must be accepted separately; no trained inference certification
   is inferred from `ready.json`. [#19](https://github.com/NomaDamas/Gamst/issues/19).
8. **SAM3.1 video — not integrated.** SAM3 image calls do not supply its native
   tracker. Runtime and model terms unreviewed for this path;
   [#19](https://github.com/NomaDamas/Gamst/issues/19).
9. **EfficientTAM — implemented tiny512 video path.** `efficienttam.tiny512`,
   `efficienttam` extra and pinned upstream source; CPU/CUDA explicit.
   [Contract](efficienttam-stream.md), `test_efficienttam_stream`,
   `test_efficienttam_context`. Code Apache-2.0; checkpoint terms separate.
   Generated untrained graph execution is wiring evidence only; trained parity,
   quality and resource acceptance [#127](https://github.com/NomaDamas/Gamst/issues/127).
10. **EfficientViT-SAM — not integrated.** No selected image-mask runtime or
    checkpoint; code/weights unreviewed.
    [#19](https://github.com/NomaDamas/Gamst/issues/19).
11. **MediaPipe body — implemented IMAGE/VIDEO.** `mediapipe.body` and
    `mediapipe.body.video`; explicit local Tasks assets. [Body](body-context.md),
    [VIDEO](landmarker-video.md), `test_body_context`, `test_landmarker_video`.
    Code Apache-2.0; selected task weight/data terms separate. Prior pose fixture
    execution is narrower than current multi-person/device acceptance
    [#81](https://github.com/NomaDamas/Gamst/issues/81),
    [#87](https://github.com/NomaDamas/Gamst/issues/87).
12. **MediaPipe hands/gestures — geometry and selected gesture classifier implemented.**
    `mediapipe.hand`/`.video` and separate `mediapipe.gesture.video`;
    [hand contract](hand-context.md), [gesture contract](gesture-context.md),
    `test_hand_context`, `test_landmarker_video`, `test_gesture_context`.
    MediaPipe0.10.35 CPU/NumPy, explicit verified local task bytes;
    code Apache-2.0, task weights separate. Eight task-label candidates are not
    intent or commands; real task compatibility/accuracy remains pending.
    [#83](https://github.com/NomaDamas/Gamst/issues/83),
    [#20](https://github.com/NomaDamas/Gamst/issues/20).
13. **MediaPipe face — implemented IMAGE/VIDEO landmarks, blendshapes and explicit-geometry head orientation.**
    `mediapipe.face`/`.video`, `mediapipe.head_orientation.video`;
    [face contract](face-context.md), [orientation contract](head-orientation.md),
    `test_face_context`, `test_landmarker_video`, `test_head_orientation`.
    Code Apache-2.0; task weights separate. Head rotation uses declared camera
    calibration/3D geometry and real OpenCV PnP numerical tests; no default
    anthropometry, inferred emotion or verified human pose-accuracy claim;
    [#85](https://github.com/NomaDamas/Gamst/issues/85),
    [#20](https://github.com/NomaDamas/Gamst/issues/20).
14. **RTMPose — not integrated.** No RTMPose checkpoint, runtime or validation
    command; code/weights unreviewed.
    [#20](https://github.com/NomaDamas/Gamst/issues/20).
15. **L2CS-Net — implemented CPU gaze candidate.** `l2cs.gaze`, `gaze` extra;
    [contract](gaze-context.md), `test_gaze`. L2CS code MIT and MediaPipe
    Apache-2.0; gaze/face weights and data terms separate. Gaze geometry does not
    prove camera-calibrated head pose; the separate explicit-geometry PnP profile
    is documented above. Real crop/parity/convention gate
    [#93](https://github.com/NomaDamas/Gamst/issues/93).
16. **OpenCV Lucas-Kanade and Farneback — both implemented numerical paths.**
    `opencv.flow`, `vision` extra, CPU; [contract](vision-motion.md),
    `test_motion_provider`. OpenCV code/runtime license is version-specific
    (supported releases declare Apache-2.0); no weights. Synthetic translation
    tests execute OpenCV, not real-camera validation;
    [#22](https://github.com/NomaDamas/Gamst/issues/22).
17. **Kalman filters — no standalone SDK Kalman provider.** IoU continuity and
    finite-difference kinematics are not Kalman filtering. TI firmware-produced
    tracks below do not prove a host Kalman implementation. Runtime/license not
    selected; [#17](https://github.com/NomaDamas/Gamst/issues/17),
    [#31](https://github.com/NomaDamas/Gamst/issues/31).
18. **RANSAC/homography — partial algorithm coverage.** OpenCV RANSAC similarity
    fitting in `BackgroundSimilarity` and explicitly supplied projective
    homography projection in `geometry.ground_plane` are implemented. Arbitrary
    homography estimation and verified calibration are not.
    [Compensation](compensated-trajectories.md), [plane projection](ground-plane.md),
    `test_camera_similarity`, `test_compensated_runtime`, `test_ground_plane`; `vision` extra, no
    weights, OpenCV terms above. [#134](https://github.com/NomaDamas/Gamst/issues/134).
19. **CoTracker3 online — not integrated.** No pinned online state/runtime or
    checkpoint validation; code/weights unreviewed.
    [#22](https://github.com/NomaDamas/Gamst/issues/22).
20. **Depth Anything V2 Small — implemented relative-depth path.**
    `depth_anything_v2.small`, `depth` extra, CPU float32;
    [contract](depth-context.md), `test_depth`. Reference Small weights/code
    declare Apache-2.0; supplied artifact provenance remains separate. Actual
    inference/ordinal quality [#95](https://github.com/NomaDamas/Gamst/issues/95).
21. **Video Depth Anything — not integrated.** No temporal model/runtime;
    code/checkpoint terms unreviewed. [#22](https://github.com/NomaDamas/Gamst/issues/22).
22. **Depth Anything 3 streaming — not integrated.** No selected streaming state,
    resource profile or assets; code/weights unreviewed.
    [#22](https://github.com/NomaDamas/Gamst/issues/22).
23. **ORB-SLAM3 — implemented monocular native bridge; inertial/stereo modes
    not integrated by that profile.** `orbslam3.monocular`, explicitly supplied
    C++ bridge/library, vocabulary/settings; [native](orbslam-native.md),
    [Context](orbslam-context.md), `test_orbslam_native`, `test_orbslam_context`.
    Upstream GPL-3.0 and separate dependency/vocabulary terms; SDK licensing
    does not relicense a linked binary. Tests use header/API doubles; actual
    native build/motion evidence [#186](https://github.com/NomaDamas/Gamst/issues/186).
24. **PP-OCRv5 mobile — implemented selected local export interface.**
    `ppocr.mobile`, `ocr` extra, CPU ONNX; [contract](ocr-context.md), `test_ocr`.
    Paddle/RapidOCR code Apache-2.0; selected detector, recognizer and language
    exports require separate provenance. Actual export/language compatibility
    [#97](https://github.com/NomaDamas/Gamst/issues/97).
25. **DINOv3 — implemented selected retrieval/region paths.**
    `dinov3.retrieval`, `dinov3.regions`, `retrieval` extra, CPU float32;
    [retrieval](visual-retrieval.md), [memory join](object-appearance-memory.md),
    `test_visual_retrieval`, `test_region_appearance`. DINOv3 model-specific
    license/access terms require verification; Transformers code is separate.
    Actual state compatibility/quality [#99](https://github.com/NomaDamas/Gamst/issues/99),
    [#101](https://github.com/NomaDamas/Gamst/issues/101).
26. **Qwen3-VL 2B — implemented local GGUF/llama.cpp path.** Explicit existing
    model server and launcher; [setup/provenance](local-setup.md), `test_ollama`,
    `test_temporal_scene`. Sampled images are not native streaming memory.
    llama.cpp and selected GGUF/projection/model terms are separate; verify
    manifest and upstream licenses. Historical local use is not matched
    quality/cost acceptance [#40](https://github.com/NomaDamas/Gamst/issues/40).
27. **SmolVLM2 500M Video — implemented bounded image-window path.**
    `smolvlm2.500m.local`, `smolvlm` extra, CPU float32/CUDA explicit;
    [native](smolvlm-native.md), [Context](smolvlm-context.md),
    `test_smolvlm_native`, `test_smolvlm_context`, `test_smolvlm_call`.
    Reference model/code Apache-2.0 declared; substituted weights separate.
    Windows do not establish persistent video KV state; trained acceptance
    [#150](https://github.com/NomaDamas/Gamst/issues/150).
28. **FastVLM 0.5B and FastVLM 1.5B — neither integrated.** No selected asset or
    candidate runtime; code/weight terms unreviewed for both sizes.
    [#40](https://github.com/NomaDamas/Gamst/issues/40).
29. **Cosmos Reason2 2B — not integrated.** No selected native runtime or model
    license/compatibility review. [#40](https://github.com/NomaDamas/Gamst/issues/40).
30. **StreamingVLM — not integrated.** Repeated Qwen/Smol windows are not its
    native stateful runtime. Code/weights unreviewed;
    [#40](https://github.com/NomaDamas/Gamst/issues/40).
31. **NVIDIA TAO PeopleNet, TAO body pose and TAO action — none integrated.**
    Each requires a selected model/version, NVIDIA runtime and separate model
    terms; no candidate validation command is available.
    [#5](https://github.com/NomaDamas/Gamst/issues/5).
32. **TensorRT and DeepStream — neither integrated as a deployment backend.**
    CUDA device/runtime/version and NVIDIA distribution terms unreviewed;
    these names are not additional perception capabilities.
    [#13](https://github.com/NomaDamas/Gamst/issues/13).

## Audio and audiovisual candidates

33. **Silero VAD — implemented recurrent path; recorded fixture execution.**
    `silero.activity`, `vad` extra, explicit local JIT and 16 kHz mono;
    [contract/evidence](speech-activity.md), `test_vad_provider`.
    Reference code/model MIT; Torch terms separate. Recorded continuity does not
    establish live accuracy or ASR acceptance [#23](https://github.com/NomaDamas/Gamst/issues/23).
34. **TEN-VAD — not integrated.** No selected native runtime/assets; code/model
    terms unreviewed. [#23](https://github.com/NomaDamas/Gamst/issues/23).
35. **Nemotron 3.5 ASR Streaming 0.6B — not integrated.** The diarization model
    below is not ASR. No selected ASR runtime/license compatibility;
    [#68](https://github.com/NomaDamas/Gamst/issues/68).
36. **Qwen3-ASR 0.6B — implemented bounded streaming adapter and endpoint path.**
    `qwen3.asr`, `qwen3.asr.endpoint`, separate qwen-asr/vLLM environment above;
    [runtime](qwen-local-runtime.md), [endpoints](managed-endpoints.md),
    `test_qwen_artifact`, `test_qwen_asr`, `test_managed_endpoints`.
    Reference model Apache-2.0 declared; vLLM/runtime distributions separate.
    Accumulated audio is re-fed; constant-work cached inference is not proven.
    Actual bilingual/runtime gate [#68](https://github.com/NomaDamas/Gamst/issues/68).
37. **FasterWhisper — implemented legacy windowed ASR.** `FasterWhisperPlugin`,
    `asr` extra (faster-whisper >=1,<2), local Whisper small/CTranslate2;
    [setup](local-setup.md), `test_plugins`, `test_transcripts`. Runtime, converted
    weights and upstream model terms must be verified separately. No native
    partial/final cache claim; [#23](https://github.com/NomaDamas/Gamst/issues/23).
38. **SenseVoice Small — not integrated.** No selected native or pseudo-streaming
    profile, dependency or code/weight license review;
    [#23](https://github.com/NomaDamas/Gamst/issues/23),
    [#25](https://github.com/NomaDamas/Gamst/issues/25).
39. **Streaming Sortformer — original four-speaker model not integrated;
    selected successor implemented.** `nemotron3.diarization` specifically targets
    Nemotron-3-Diarization eight-slot weights, `diarization` extra/Python 3.11+.
    [Contract](diarization.md), `test_diarization`. Successor weights declare
    OpenMDW-1.1; adapter/Transformers Apache-2.0. This is not binary compatibility
    with Sortformer checkpoints; actual successor acceptance
    [#116](https://github.com/NomaDamas/Gamst/issues/116).
40. **TalkNet — implemented bounded audiovisual windows.** `talknet.local`,
    `talknet` plus `vision`, CPU/CUDA float32; [contract](talknet-context.md),
    `test_talknet`, `test_audiovisual`. Code and retained MFCC reference MIT;
    trained checkpoint terms separate. Real synchronization/quality
    [#118](https://github.com/NomaDamas/Gamst/issues/118).
41. **YAMNet — implemented rolling window; recorded fixture execution.**
    `yamnet.events`, `environment-audio` extra, MediaPipe 0.10.35 CPU;
    [contract/evidence](environment-audio.md), `test_environment_audio`.
    Reference code/model Apache-2.0 declared; four labeled recordings are not
    general accuracy or live hardware acceptance [#24](https://github.com/NomaDamas/Gamst/issues/24).
42. **PANNs — not integrated.** No selected checkpoint/window profile/runtime;
    code/weights unreviewed. [#24](https://github.com/NomaDamas/Gamst/issues/24).
43. **RMS/dBFS, autocorrelation pitch, silence/voicing — implemented DSP or
    VAD paths.** `audio.prosody` uses NumPy (`audio` extra); [prosody](audio-prosody.md),
    `test_prosody_provider`, plus Silero above. Version2 provides bounded STFT/mel
    summaries and eligible-periodic-window HNR/local jitter/local shimmer with
    explicit abstention. YIN remains an unintegrated alternative to autocorrelation.
    No weights for DSP; NumPy and SDK licenses separate.
    [#25](https://github.com/NomaDamas/Gamst/issues/25).
44. **MFCC — implemented as bounded `audio.prosody` mean/std output and separately
    inside TalkNet preprocessing.** `test_prosody_provider`, `test_talknet`;
    [TalkNet retained MIT notice](licenses/python-speech-features-MIT.txt).
    Frontend conventions differ and are explicit in [prosody](audio-prosody.md).
    Real audio validation remains under
    [#25](https://github.com/NomaDamas/Gamst/issues/25).
45. **ASR-derived speaking rate — implemented transcript timing join.**
    [Contract](speaking-rate.md), `test_speaking_rate`; dependency-light SDK code,
    no additional weights. Actual ASR timestamps/real speech acceptance
    [#114](https://github.com/NomaDamas/Gamst/issues/114).
46. **emotion2vec — implemented selected plus-seed export/ONNX path.**
    `emotion2vec.seed.onnx`, `vocal-expression` and separate preparation
    `vocal-expression-export` extras; [contract](vocal-expression.md),
    `test_vocal_expression`, `test_emotion2vec_export`. FunASR code MIT,
    ONNX Runtime MIT, selected weights `other/model-license` unresolved;
    no claim for every emotion2vec variant. Trained export/parity/license
    [#111](https://github.com/NomaDamas/Gamst/issues/111).
47. **openSMILE — not integrated.** No chosen feature set/runtime; code and
    commercial-use terms unreviewed. [#25](https://github.com/NomaDamas/Gamst/issues/25).
48. **GCC-PHAT — implemented two-channel DSP.** `GCCPHATDirection`, NumPy;
    [contract](audio-direction.md), `test_direction_provider`, `test_direction_stream`.
    SDK algorithm code Apache-2.0, no weights; physical array calibration
    [#60](https://github.com/NomaDamas/Gamst/issues/60).
49. **SRP-PHAT — implemented planar array DSP.** `SRPPHATDirection`, NumPy;
    [contract](audio-srp.md), `test_srp_direction`. Independent Apache-2.0 code,
    no pyroomacoustics runtime or weights. Actual array
    [#60](https://github.com/NomaDamas/Gamst/issues/60).
50. **MUSIC and MVDR — neither integrated.** The implemented NumPy
    `DelaySumBeamformer` is a different algorithm (`test_beamforming`,
    [contract](audio-beamforming.md)). No MUSIC/MVDR runtime selected; no license
    inference from that baseline. [#27](https://github.com/NomaDamas/Gamst/issues/27).
51. **pyroomacoustics — reference only, not wrapped or required.** The DSP
    paths above are independent implementations. No package compatibility or
    license combination validated; [#27](https://github.com/NomaDamas/Gamst/issues/27).
52. **WebRTC AEC/NS/AGC — implemented optional native pipeline bridge.**
    `webrtc.aec3`, `duplex` extra; [contract](duplex-audio.md), `test_duplex`.
    AEC-only uses EchoCanceller; explicit NS/AGC options use AudioProcessor.
    Preserve the ambient lane. Wrapper Apache-2.0, WebRTC vendored notices separate, no
    learned weights. Native acoustic acceptance
    [#103](https://github.com/NomaDamas/Gamst/issues/103).
53. **RNNoise — not integrated.** No selected recurrent runtime/weights;
    code/model terms unreviewed. [#28](https://github.com/NomaDamas/Gamst/issues/28).
54. **DeepFilterNet — not integrated.** No selected model/runtime; code/weights
    unreviewed. [#28](https://github.com/NomaDamas/Gamst/issues/28).
55. **SAM-Audio — implemented local native text/temporal and visual-window
    integration.** `sam_audio.text_temporal.local` and owned visual path;
    Python 3.11+, exact complete runtime/artifact manifests;
    [native](sam-audio-native.md), [visual](sam-audio-visual.md),
    `test_sam_audio_native`, `test_sam_audio_context`, `test_sam_audio_visual`.
    SAM/PE/DACVAE/T5 sources and weights retain distinct terms; manifest declarations
    are not license certification. Whole-window lookahead is not causal realtime.
    Trained compatibility/resources/quality [#203](https://github.com/NomaDamas/Gamst/issues/203),
    [#234](https://github.com/NomaDamas/Gamst/issues/234).

## Physical observations, sensors and research candidates

56. **Task-trained lightweight food segmentation — implemented SegFormer-B0
    graph/Context path.** `food.segformer_b0`, `food` extra;
    [contract](food-context.md), `test_food`. Transformers code Apache-2.0;
    task-trained head/data/weight license explicitly supplied and unverified.
    Generated graph tests do not establish trained food quality
    [#148](https://github.com/NomaDamas/Gamst/issues/148).
57. **FoodSAM — not integrated.** No selected pipeline/checkpoints; code/model
    terms unreviewed. [#36](https://github.com/NomaDamas/Gamst/issues/36).
58. **FoodLMM — not integrated.** No selected runtime/checkpoint; code/model
    terms unreviewed. [#36](https://github.com/NomaDamas/Gamst/issues/36).
59. **FoodSeg103 — dataset, not provider.** No training/import/split or dataset
    license audit in this delivery. [#148](https://github.com/NomaDamas/Gamst/issues/148).
60. **Nutrition5k — dataset, not provider.** No nutrition model integration or
    training/data-license audit; visible labels do not measure calories.
    [#36](https://github.com/NomaDamas/Gamst/issues/36).
61. **Lepton/FLIR radiometric frames — implemented assembled TLinear decoder.**
    `TLinearThermal`, standard library; [contract](thermal-context.md),
    `test_thermal_provider`. No automatic VoSPI/USB/radiometric-JPEG driver or
    arbitrary FLIR compatibility. SDK Apache-2.0, firmware/driver terms separate,
    no weights. Capture, RGB association and known-target calibration
    [#53](https://github.com/NomaDamas/Gamst/issues/53).
62. **Timestamped tracks/depth/calibration — implemented kinematics and stereo
    bridge.** `MetricKinematics`, `opencv.rfdetr.stereo`;
    [kinematics](metric-kinematics.md), [stereo](stereo-metric.md),
    `test_kinematics`, `test_stereo_context`. NumPy/OpenCV plus selected detector;
    SDK/OpenCV and detector terms separate. Calibration declarations do not
    establish measured world velocity/force [#105](https://github.com/NomaDamas/Gamst/issues/105).
63. **Force sensors / IMU — IMU BNO055 packet path implemented; generic force
    acquisition not integrated.** `BNO055IMU`, standard library;
    [contract](imu-context.md), `test_imu_provider`. SDK Apache-2.0,
    hardware/firmware terms separate, no weights. Actual acquisition/reference
    [#63](https://github.com/NomaDamas/Gamst/issues/63); unsupported force path
    [#38](https://github.com/NomaDamas/Gamst/issues/38).
64. **PhysObjects — selected PG-InstructBLIP physical-concept path implemented.**
    `pg.physical.local`, explicit source/graph/canonical/runtime manifests;
    [Context](pg-physical-context.md), `test_pg_artifact`, `test_pg_graph`,
    `test_pg_canonical`, `test_pg_physical`. LAVIS/source notices and every
    model/data asset remain separately verified; SDK license is not a grant
    over them. Actual graph/checkpoint compatibility and learned quality
    [#242](https://github.com/NomaDamas/Gamst/issues/242).
65. **Container masks/fill/volume — implemented calibrated color-fill path.**
    `opencv.fill`, `vision`; [contract](calibrated-fill.md), `test_fill`.
    No learned weights; OpenCV and SDK terms separate. Known geometry and
    colored fixed-view assumptions are required; real liquid/reference
    [#109](https://github.com/NomaDamas/Gamst/issues/109).
66. **OpenPIV — implemented selected SciPy backend.** `openpiv.flow`, `piv`
    extra OpenPIV 0.26.1/Python 3.10+/NumPy >=2/SciPy >=1.11, no optional Rust;
    [contract](piv-flow.md), `test_piv`. OpenPIV GPL-3.0-or-later; SDK Apache-2.0,
    no weights. Installed package and calibrated particle/fluid evidence
    [#107](https://github.com/NomaDamas/Gamst/issues/107).
67. **PourIt — not integrated.** No reproducible selected task/runtime/assets;
    code/checkpoint/data terms unreviewed. [#37](https://github.com/NomaDamas/Gamst/issues/37).
68. **ESP-CSI — implemented classic ESP32 channel-motion decoder/DSP.**
    `esp32.csi`, standard library, pinned 20 MHz firmware profile;
    [contract](esp32-csi.md), `test_csi`. Espressif firmware/adapter Apache-2.0,
    no weights. Other chip/firmware profiles unsupported; live motion
    [#121](https://github.com/NomaDamas/Gamst/issues/121).
69. **TI mmWave — implemented xWR6843 selected UART points/tracks decoder.**
    `ti.radar`, standard library; [contract](ti-radar.md), `test_radar`.
    FFT/CFAR/clustering/Kalman occur in selected device firmware, not host SDK
    implementations. TI firmware/binary terms separate from Apache-2.0 adapter;
    configuration/clock/live validation [#123](https://github.com/NomaDamas/Gamst/issues/123).
70. **RF-Pose — not integrated; reproducible runtime/assets/hardware missing.**
    No executable profile or command; research description is not an SDK.
    Code/checkpoint/hardware terms unreviewed;
    [#189](https://github.com/NomaDamas/Gamst/issues/189).
71. **RF-Action — not integrated; reproducible runtime/assets/hardware missing.**
    CSI/radar parsing is not its inference path. No selected runtime or license
    compatibility; [#189](https://github.com/NomaDamas/Gamst/issues/189).
72. **DIGIT — implemented explicit serial-selected capture/packet/replay path.**
    `TactileCapture.open_digit`, pinned digit-interface wrappers;
    [contract](tactile-capture.md), `test_tactile`. Driver/device terms separate
    from SDK; no task weights implied. Actual device/calibration
    [#188](https://github.com/NomaDamas/Gamst/issues/188).
73. **GelSight — implemented explicit GelSight Mini/OpenCV capture path.**
    `TactileCapture.open_gelsight`, `vision` extra; same contract and `test_tactile`.
    Device identity cannot be authenticated from a generic webcam index; SDK,
    OpenCV and sensor terms separate. No general GelSight device guarantee;
    [#188](https://github.com/NomaDamas/Gamst/issues/188).
74. **Sparsh — implemented selected encoder/task-head slip path.**
    `sparsh.slip.local`, `sparsh` extra, CPU Torch attention;
    [native](sparsh-native.md), [Context](sparsh-context.md),
    `test_sparsh_native`, `test_sparsh_context`. Root source CC-BY-NC-4.0 with
    per-file notices; encoder/head terms separate. No generic contact/force head
    or trained sensor quality claim [#226](https://github.com/NomaDamas/Gamst/issues/226).
75. **BME688 — implemented pinned BSEC2 callback/measurement/classifier bridge.**
    `bosch.bsec2`, standard-library worker plus explicit device firmware;
    [contract](bsec2-context.md), `test_bsec2`. Wrapper 1.10.2610/BSEC 2.6.1.0;
    Bosch binary/config terms separate from SDK. Pressure scale is explicitly
    unverified, no guessed SI conversion. Firmware/trained gas/reference
    [#130](https://github.com/NomaDamas/Gamst/issues/130).
76. **BME690 — not integrated.** BME688's driver profile does not support it.
    No selected driver/config or license/runtime compatibility;
    [#130](https://github.com/NomaDamas/Gamst/issues/130).
77. **OpenEB — implemented EventCD capture supervision and Context.**
    `openeb.eventcd` plus explicit `OpenEBProcess`, native OpenEB 5.2.0
    pinned wrappers and NumPy; [capture](openeb-process.md),
    [Context](openeb-context.md), `test_openeb`, `test_openeb_process`.
    OpenEB source Apache-2.0; plugins/device/recording terms separate, no weights.
    Native SDK/camera acceptance [#140](https://github.com/NomaDamas/Gamst/issues/140).
78. **EfficientPhys — implemented bounded face-window graph path.**
    `efficientphys.local`, `efficientphys` extra, explicit CPU/CUDA;
    [native](efficientphys-native.md), [Context](efficientphys-context.md),
    `test_efficientphys_native`, `test_efficientphys_context`. Upstream
    Responsible AI Source Code License 1.1; checkpoint terms separate.
    Generated graph execution is not pulse accuracy; trained reference-pulse
    acceptance [#137](https://github.com/NomaDamas/Gamst/issues/137).
79. **Visual Microphone — implemented offline numerical reconstruction and
    timed-window Context.** `visual_microphone.phase`, `visual-microphone` extra
    NumPy/SciPy; [algorithm](visual-microphone.md),
    [Context](visual-microphone-context.md), `test_visual_microphone`,
    `test_visual_microphone_context`. MIT reference notice retained; no weights.
    Actual high-speed synchronized video/audio and upstream parity
    [#144](https://github.com/NomaDamas/Gamst/issues/144).

## Completion and remaining evidence

The user's 2026-11-15 direction separates finishing approved **software work**
from later actual experiments and model/sensor validation. This inventory makes
unsupported alternatives, external research prerequisites and deferred evidence
visible; it does not mark those capabilities validated or close their issues.
The radio research runtime prerequisite (#189), authentic request-level billing
source (#257/#277), and unresolved “multimodal Jev/cheap Jev” identity (#136)
cannot be satisfied by inventing a model, receipt or compatibility claim.

Before deployment, record exact code/revision, asset hashes, runtime lock, device,
input provenance, command, output and independent review for each selected path.
Actual held-out A/B/C quality, whole-system cost and task-ready latency remain
separate gates. No 20% saving, 10% p95 improvement, trained quality or physical
calibration is established by this matrix or a passing synthetic test.
