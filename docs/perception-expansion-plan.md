# Approved perception SDK expansion plan

Approved 2026-11-13. HumanSense supplies real-time sensory context to an existing LLM. Product priorities are broader perception, lower VLM cost and latency, and preserved task quality. Halo demonstrations are out of scope for this delivery. This plan records the approved research scope; naming a candidate does not assert an implemented or validated provider.

## Architecture and responsibilities

```text
[Sensors and capture] -> [Timestamped sensor envelopes]
 -> [DSP / lightweight stateful providers / optional semantic VLM]
 -> [Typed observations + track state + observation memory]
 -> [Python API / live transport / MCP / observability] -> [Consumer LLM]
```

Capture, inference and context delivery have independent clocks. User-defined delivery cadence is authoritative. Delta compression and event summarization are not prerequisites. Sensing continues during consumer playback; interruption and mid-generation consumption require host support. Playback-reference AEC is separate from synthesis. MCP discovery/control/history does not imply unsolicited context injection into every host.

Streaming modes must distinguish native stateful, rolling window, independent frame and offline clip. Providers declare required window, lookahead, reset-on-gap behavior, device, version, license, input requirements, output units and provenance. Video support alone does not prove causal streaming. No downloads or sensor activation on discovery. Only selected providers load; bounded queues and memory budgets prevent unlimited backlog.

## Approved work packages

- P01: Capability registry, sensor envelope, typed observation validation and extensible context serialization; retain existing SDK compatibility.
- P02: Independent bounded per-sensor streams, provider state isolation, gaps/reconnect/reset semantics, multichannel preservation and continuous duplex capture.
- P03: Vision adapters for detection/tracking, masks, hands/pose/gaze, OCR, embeddings, depth, motion and geometry.
- P04: Audio DSP, VAD, native ASR revisions, sound events, diarization, prosody, emotion candidates, source direction and playback AEC integration.
- P05: Radiometric thermal, radar/CSI, IMU, tactile and environmental sensor adapters with calibrated units and missing-hardware reporting.
- P06: Food, liquid and physical observation pipelines; separate visible evidence, measured quantities and semantic hypotheses.
- P07: Cross-frame object observation memory, last seen versus current visibility, evidence, expiry and optional spatial coordinate mapping.
- P08: Low-rate semantic VLM providers and explicit routing/resource policy; matched quality/cost/latency evaluation against a VLM-only baseline.
- P09: Python/live API/MCP configuration, discovery, latest/history/memory queries and observability for all registered capabilities.
- P10: Optional connection/runtime profiling and user-controlled cadence calibration; sensor and confidence calibration remain distinct.
- P11: Optional model/backend installation profiles, runtime/license compatibility and executable provider validation matrix.
- P12: End-to-end acceptance, independent GPT-6.1 Sol High reviews, plan compliance audit and PR merges.

## Candidate inventory and integration expectations

Candidates are alternatives and research extension targets, not a requirement to run all models simultaneously. Each inventory row must ultimately map to executable provider coverage or an explicitly open limitation issue. A generic callable or a catalog entry alone is not a validated model integration.

### Vision

- Detection/tracking: RF-DETR Nano, YOLO26n, YOLOX Nano/Tiny; ByteTrack and BoT-SORT for track continuity. Frame inference plus stateful tracking. Sources: https://github.com/roboflow/rf-detr ; https://docs.ultralytics.com/modes/track/ ; https://github.com/Megvii-BaseDetection/YOLOX
- Open-vocabulary detection: YOLOE, text and exemplar prompts. https://github.com/THU-MIG/yoloe
- Segmentation: existing SAM3 sampled-image path, SAM3.1 video path, EfficientTAM video tracking, EfficientViT-SAM image masks. Do not conflate image masks with causal video tracking. https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md ; https://github.com/yformer/EfficientTAM ; https://github.com/mit-han-lab/efficientvit
- Body, hands, face: existing MediaPipe face/body plus hands and gestures; RTMPose alternative. Gaze: L2CS-Net; head orientation with camera geometry. Movement is not verified emotion or intent. https://developers.google.cn/edge/mediapipe/solutions/agents ; https://arxiv.org/abs/2303.07399 ; https://github.com/Ahmednull/L2CS-Net
- Motion: OpenCV Lucas-Kanade/Farneback, Kalman filters, RANSAC/homography, CoTracker3 online tracking. https://github.com/facebookresearch/co-tracker
- Depth: Depth Anything V2 Small, Video Depth Anything; Depth Anything 3 streaming is a higher-resource candidate. Monocular relative depth is not calibrated metres. https://github.com/DepthAnything/Depth-Anything-V2 ; https://github.com/DepthAnything/Video-Depth-Anything ; https://github.com/ByteDance-Seed/Depth-Anything-3
- Space: ORB-SLAM3 visual/inertial mapping; metric coordinates require appropriate calibration and sensors. https://github.com/UZ-SLAMLab/ORB_SLAM3
- OCR: PP-OCRv5 mobile, language-specific checkpoints. Retrieval: DINOv3 embeddings; similarity is evidence, not guaranteed identity. https://github.com/PaddlePaddle/PaddleOCR ; https://github.com/facebookresearch/dinov3
- Semantic providers: existing Qwen3-VL 2B, SmolVLM2 500M Video, FastVLM 0.5B/1.5B, Cosmos Reason2 2B, StreamingVLM. Native stream state is distinct from repeated frame prompting. https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct ; https://github.com/apple-aiml-research/ml-fastvlm ; https://huggingface.co/nvidia/Cosmos-Reason2-2B ; https://github.com/mit-han-lab/streaming-vlm
- NVIDIA deployment: TAO PeopleNet, body pose and action models; TensorRT/DeepStream are deployment backends, not perception capabilities. https://docs.nvidia.com/tao/tao-toolkit/latest/text/model_zoo/overview.html

### Audio and audiovisual

- VAD: existing Silero; TEN-VAD alternative. https://github.com/snakers4/silero-vad ; https://github.com/TEN-framework/ten-vad
- Native ASR: Nemotron 3.5 ASR Streaming 0.6B and Qwen3-ASR 0.6B; partial/final/revision contract, declared language and timestamp support. Existing FasterWhisper remains windowed. https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b ; https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- SenseVoice Small: transcription/emotion/events candidate; pseudo-streaming implementations are explicitly labeled. https://github.com/QwenAudio/SenseVoice
- Diarization: Streaming Sortformer. Active speaker: TalkNet with explicit temporal/lookahead constraints; speaker labels are not real-world identity. https://github.com/NVIDIA-NeMo/Speech ; https://github.com/TaoRuijie/TalkNet-ASD
- Environmental sound: YAMNet and PANNs, rolling-window outputs. https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md ; https://github.com/qiuqiangkong/audioset_tagging_cnn
- Prosody: RMS/dBFS, pitch via YIN/autocorrelation, STFT/MFCC, silence/voicing and ASR-derived speaking rate. HNR/jitter/shimmer require suitable voiced windows. dBFS is not dBSPL without calibration.
- Emotion: emotion2vec, optional openSMILE feature extraction with license checks. Vocal features and model labels do not establish internal feelings. https://arxiv.org/abs/2312.15185 ; https://github.com/audeering/opensmile
- Array audio: GCC-PHAT/SRP-PHAT/MUSIC and MVDR; retain channels and array geometry. https://github.com/LCAV/pyroomacoustics
- Duplex: WebRTC AEC/NS/AGC with playback reference; RNNoise/DeepFilterNet alternatives. Preserve an environmental lane rather than erasing relevant sounds through speech denoising. https://github.com/xiph/rnnoise ; https://github.com/Rikorose/DeepFilterNet
- Cross-modal source separation: SAM-Audio optional high-cost provider; native realtime support is not assumed. https://github.com/facebookresearch/sam-audio

### Physical and additional sensors

- Food: task-trained lightweight detector/classifier/segmenter; FoodSAM and FoodLMM research alternatives; FoodSeg103 and Nutrition5k are datasets. Visible food labels do not establish mass, calories or internal temperature. https://github.com/jamesjg/FoodSAM ; https://github.com/YuehaoYin/FoodLMM ; https://github.com/google-research-datasets/Nutrition5k
- Thermal: radiometric Lepton/FLIR frames, ROI min/max/mean, RGB association and calibration metadata; RGB alone does not measure Celsius. https://www.oem.flir.com/en-150/developer/lepton-family/project-lepton-spot-meter-code/
- Physical: tracks plus timestamps/depth/calibration for velocity, acceleration, approach and geometric relations; force/IMU sensors for direct measurements. PhysObjects supplies learned physical-concept evidence, not measured mass/friction. https://arxiv.org/abs/2309.02561
- Liquid: masks and known container geometry for fill level/volume; flow observations; OpenPIV calibrated particle setups; PourIt research adapter. Arbitrary video does not establish pressure or viscosity. https://www.openpiv.net/ ; https://github.com/hetolin/PourIt
- CSI: ESP-CSI channel amplitude/phase, presence/motion with environment-dependent validation. https://github.com/espressif/esp-csi
- Radar: TI mmWave range/Doppler/angle/points/tracks; FFT/CFAR/clustering/Kalman processing. https://www.ti.com/tool/TIDEP-01000
- Through-wall research: RF-Pose/RF-Action with required specialized hardware; no generic through-wall object classification claim. https://rfpose.csail.mit.edu/ ; https://rf-action.csail.mit.edu/
- Touch: DIGIT/GelSight streams, Sparsh representations and task-specific contact/slip heads. https://github.com/facebookresearch/sparsh
- Gas: BME688/BME690 measured environmental channels and trained gas-pattern classifiers; sensor response times and training environment matter. https://www.bosch-sensortec.com/en/software-tools/software/bme688-and-bme690-software
- Frontier extension contracts: OpenEB event cameras; EfficientPhys rPPG sustained windows; Visual Microphone high-speed vibration/audio reconstruction. These remain experimental until hardware/model validation succeeds. https://github.com/prophesee-ai/openeb ; https://arxiv.org/abs/2110.04447 ; https://people.csail.mit.edu/mrub/VisualMic/

## Object memory

Remember observed entity appearance, OCR, track provenance, surrounding relations, last-seen time and coordinate frame. Distinguish currently visible, previously seen, inferred occlusion and unknown current location. Never invent an observation for an unseen object. Bound history and embedding storage; expose clear/expiry controls. Camera-relative positions must not become world positions without mapping. Memory must survive short latest-observation retention while respecting user lifecycle clear/reset.

## Runtime and calibration

Track requested, recommended and applied delivery intervals separately. Profile actual provider latency, memory, gaps, queueing, window/lookahead and consumer capabilities. Optional synthetic endpoint probes must be bounded and explicitly enabled; discovery does not send private media. ACK is not proof of comprehension and decoder token speed is not observation capacity. Cache profiles by hardware/model/backend/precision. Sensor calibration and statistical confidence calibration require their own evidence.

## Performance and quality acceptance

Compare identical recorded inputs and tasks against a declared VLM-only baseline. Report hardware, model/version/precision, sample rate, window/lookahead, warm/cold state, p50/p95 end-to-end observation latency, VLM calls/tokens, measured cost when a price is configured, memory, drops and task quality. No claim of quality preservation from synthetic timing tests alone. Report unknown pricing as unknown. Establish task-specific non-inferiority tolerance before an evaluation; preserve rare events, OCR facts, object identity and ASR revisions. Policy remains user-selectable; savings must not silently suppress required modalities.

## Review and completion gates

Every implementation PR links its issue and acceptance requirements, includes focused failure-path tests and relevant existing regressions, and receives an independent GPT-6.1 Sol High review. Record reviewed commit SHA and findings; resolve findings and re-review changed scope before merging. CI and a plan-compliance checklist must pass. No implementation agent self-approval.

Keep implementation, mocked contract tests, actual model execution, and physical-device validation as separate evidence levels. Missing sensor hardware, unavailable checkpoints, incompatible licenses or unverified research runtime become detailed linked open subissues, not completed boxes. The overall goal stays incomplete while required acceptance remains unmet. Preserve the user's pre-existing uncommitted workspace changes by developing in an isolated worktree.
