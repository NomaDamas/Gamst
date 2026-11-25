# GAMST research basis, model choices and roadmap

Research date: 2026-11-15. "SOTA" is always scoped to a named task, benchmark and metric; no model wins every dataset. Server-GPU numbers below are **not** Apple Silicon measurements. `[UNVERIFIED]` marks rows whose primary source was not fully checked. Defaults stay lightweight and local; heavier models are opt-in extras.

## 1. Default vs optional SOTA per modality

| Modality | Default (lightweight) | Verdict on default | Optional higher-accuracy provider | Why (evidence) |
|---|---|---|---|---|
| Real-time detection | RF-DETR Nano | SOTA family, speed point | RF-DETR-L (Apache-2.0) / 2XL (PML license) | RF-DETR-2XL 60.1 COCO AP50:95 at 17.2 ms on T4 ([repo](https://github.com/roboflow/rf-detr), ICLR 2026) |
| Open-vocab detection | YOLOE | Lightweight-best; AGPL-3.0 | MM-Grounding-DINO | YOLOE real-time text/visual prompts ([ICCV 2025](https://github.com/THU-MIG/yoloe)); Grounding DINO 1.5 Pro is API-only |
| Multi-object tracking | IoU tracker | Neither | **OC-SORT** (lightweight upgrade), MOTIP (accuracy) | OC-SORT CVPR 2023, MIT ([repo](https://github.com/noahcao/OC_SORT)); MOTIP 65.9 HOTA DanceTrack, ~16 FPS RTX 4090 ([CVPR 2025](https://github.com/MCG-NJU/MOTIP)) |
| Promptable image segmentation | SAM3 sampled images | SOTA for concept prompts | — | SAM3 ICLR 2026, LVIS zero-shot mask AP 48.8 vs 38.5 prior ([paper](https://arxiv.org/abs/2511.16719)); custom SAM License, HF-gated |
| Video masks / tracking | EfficientTAM tiny512 | Lightweight-best | **SAM3.1 video** | EfficientTAM-Ti/2 18M, 70.8 SA-V test ([ICCV 2025](https://arxiv.org/abs/2411.18933)); SAM3.1 Object Multiplex ~7× faster at 128 objects on H100 ([release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)) |
| Monocular depth | Depth Anything V2 Small | Lightweight-best | Depth Anything 3 | DA3 δ1 97.1 KITTI / 98.0 NYU vs DA2 94.6 / 97.9 ([ICLR 2026 Oral](https://arxiv.org/abs/2511.10647)) |
| Streaming depth | per-frame DAV2 | Neither | **Video Depth Anything Small** | VDA-S 9.1 ms/frame A100, Apache-2.0; VDA-L KITTI δ1 0.944 ([CVPR 2025](https://github.com/DepthAnything/Video-Depth-Anything)) |
| SLAM / VO | ORB-SLAM3 | Lightweight-best classical | MASt3R-SLAM | TUM RGB-D ATE 0.060 m vs DROID-SLAM 0.163 m ([DA3-Streaming benchmark](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)) |
| Streaming VLM | Qwen3-VL 2B + SmolVLM2 500M | Lightweight-best local pair | StreamingVLM (arm E) | StreamingVLM 66.18% win vs GPT-4o mini on Inf-Streams-Eval, 8 FPS on one H100 ([ICLR 2026](https://arxiv.org/abs/2510.09608)) |
| Face / pose / hands | MediaPipe | Lightweight default `[UNVERIFIED]` | RTMPose, HaMeR `[UNVERIFIED]` | No same-device benchmark yet; keep MediaPipe until one exists |
| Gaze | L2CS-Net | Lightweight baseline | — | Recent candidates use incompatible Gaze360 protocols ([L2CS-Net](https://github.com/Ahmednull/L2CS-Net), MIT) |
| Streaming ASR | Qwen3-ASR 0.6B (+ faster-whisper) | SOTA-class multilingual streaming | **Moonshine v2** (English edge) | Qwen3-ASR 0.6B 4.40 avg streaming WER; Moonshine v2 Tiny 34M, 50 ms median response on M3 |
| VAD | Silero | Lightweight-best | **TEN-VAD** | TEN-VAD 0.016 RTF on M1, better PR curve on its noisy test set ([repo](https://github.com/TEN-framework/ten-vad)) |
| Diarization | Streaming Sortformer | SOTA-class online | DiariZen (offline refine) | DiariZen 8.02% AMI-SDM DER vs pyannote 12.45 |
| Active speaker | TalkNet | Neither | LoCoNet | LoCoNet 95.2 mAP AVA-ActiveSpeaker, 68.4 Ego4D vs TalkNet 51.7 ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_LoCoNet_Long-Short_Context_Network_for_Active_Speaker_Detection_CVPR_2024_paper.pdf)) |
| Audio tagging | YAMNet | Neither | **CED-Tiny / CED-Mini** (GPL-3.0, opt-in) | CED-Tiny 5.5M 48.1 mAP AudioSet-2M ([ICASSP 2024](https://github.com/RicherMans/CED)) |
| Speech emotion | emotion2vec plus | Lightweight-best representation | emotion2vec plus large | 71.79% IEMOCAP LOSO ([ACL Findings 2024](https://aclanthology.org/2024.findings-acl.931/)); no cross-corpus winner |
| Audio separation | SAM-Audio (prompted jobs) | SOTA for promptable separation | TF-MLPNet (continuous speech) | TF-MLPNet 493K params, 14.12 dB SI-SDRi, 10 ms latency |
| Sound localization | GCC-PHAT / SRP-PHAT | Lightweight geometry baseline | Learned SELD (domain-matched only) | DCASE 2025 Task 3 winner 58.47M params, DOA error 12.2° |
| Tactile | Sparsh over DIGIT/GelSight | Lightweight-best practical | AnyTouch (ICLR 2025), UniTouch (CVPR 2024) | Different tasks; Sparsh matches GAMST hardware ([CoRL 2024](https://sparsh-ssl.github.io/)) |
| RF sensing | ESP32-CSI DSP, TI mmWave decoder | Acquisition front ends, not models | mmBody-style learned pose | mmBody 67.22 mm MPJPE (MobiCom 2022) |
| Gas / air | BME688 + BSEC2 | Lightweight deployable pipeline | Trained BME AI-Studio profile | Accuracy belongs to the trained profile, not the sensor |
| Physical concepts | PG-InstructBLIP | Neither | Cosmos-Reason2 / Cosmos 3 | PhysBench (ICLR 2025) for evaluation |
| Physics computation | (none, capability gap) | Neither | **`physical_state` deterministic precompute**; Genesis / Cosmos optional | Kinematic state, trajectory envelopes, contact candidates, cross-modal impact events |
| Fluid flow | OpenPIV | Lightweight classical | RAFT-PIV | [Nature Machine Intelligence 2021](https://www.nature.com/articles/s42256-021-00369-0) |
| Visual microphone | Davis et al. phase method | Lightweight only | EvMic (ICCV 2025) | EvMic SNR 1.214 / STOI 0.481 vs RGB phase −2.837 / 0.322 |
| Multimodal decision | Deterministic rules | Zero-dependency fallback | **Intern-Decision-2B** (learned light) / **4B** (optional SOTA) | 84.68% / 90.02% seven-suite mean, 33–44 ms RTX 4090 ([arXiv 2610.22131](https://arxiv.org/abs/2610.22131), Apache-2.0) |

**Bold** optional providers are implemented as adapters in this repository (real-weight acceptance pending). The others are candidates; see [candidate execution matrix](candidate-execution-matrix.md).

### Why SAM3 runs on sampled images

SAM3 video is a stateful tracker whose cost grows linearly with tracked objects; SAM3.1 adds Object Multiplex buckets to share memory. A live integration needs per-stream predictor state, track lifecycle/eviction, bounded memory, causal masklets and backpressure. GAMST therefore keeps EfficientTAM as the always-on mask tracker and SAM3 for sampled concept segmentation; SAM3.1 video is an opt-in provider. Both SAM3 and SAM3.1 use Meta's custom SAM License and gated Hugging Face weights, so installation requires accepting terms (`scripts/setup_sam3.py install`).

## 2. Jev-style (System One) multimodal decision models

"Jev" is not an acronym and is not JEPA. TypeSafe AI named its typed decision model after William Stanley **Jevons**; "System One" refers to Kahneman's fast thinking ([launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). A Jev-style model takes runtime state plus a caller-defined schema or candidate set and returns a schema-valid choice, binary decision, score or interval with probabilities, usually in one parallel scoring pass instead of open-ended decoding.

| Paper | Contribution | Fit for GAMST |
|---|---|---|
| [Intern-Decision](https://arxiv.org/abs/2610.22131) (2026) | 0.8B/2B/4B; fills typed JSON fields in one pass; Apache-2.0 weights | **Implemented** as optional `intern-decision-2b` / `-4b` provider |
| [VideoJev](https://arxiv.org/abs/2610.18615) (2026) | 4B / 35B-A3B; scores temporal intervals and choices; cached visual prefixes | Video specialist; 4B is the realistic local option |
| [UniJev](https://arxiv.org/abs/2610.25784) (2026) | Trimodal text/image/video; Cora calibration | Calibration research; release incomplete at inspection |
| [LLM-as-Jev](https://arxiv.org/abs/2610.02076) (2026) | Candidate-logit readout on existing LLMs | Low-cost retrofit experiment on GAMST's Qwen model |
| [PixelJev](https://arxiv.org/abs/2609.29283) (2026) | Image + candidates → choice | No official checkpoint |
| [OmniMed-Jev](https://arxiv.org/abs/2610.00381) (2026) | Medical Choice/Noul/Score | Domain-specific; not a default |

The model only selects among fixed GAMST candidates (`fuse_multimodal_context`, `escalate_vlm`, `request_fresh_sensor`, `request_more_audio`, `request_more_video`, `insufficient_evidence`). Evidence IDs, freshness and watermarks stay in host code; timeouts, schema failures and low confidence abstain. See [multimodal decision](multimodal-decision.md).

## 3. Late fusion and where GAMST sits

| Fusion type | Definition | Representative work |
|---|---|---|
| Early | Concatenate or jointly encode low-level features before task processing | Taxonomy in [Baltrušaitis et al., TPAMI 2019](https://ieeexplore.ieee.org/document/8269806) |
| Intermediate | Modality encoders exchange hidden states via attention or bottleneck tokens | [MBT, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html) |
| **Late** | Independent modality models produce final scores/decisions; a combiner aggregates them | Defined in MBT; modular and tolerant of missing modalities |
| Hybrid | Combines stages, e.g. latent bottleneck plus decision combiner | [Perceiver, ICML 2021](https://proceedings.mlr.press/v139/jaegle21a.html), [Perceiver IO, ICLR 2022](https://iclr.cc/virtual/2022/poster/6269) |

GAMST today: `sensors → independent specialists → timestamped typed observations → bounded freshness selection → LLM context or decision provider`. That is **decision-level late fusion mediated by language**. Strengths: provider replacement, missing-modality handling, readable context, auditable evidence IDs. Weakness: a semantic bottleneck that can drop cross-modal timing, geometry and fine detail before fusion.

### Upgrade order

1. **Learned calibrated late fusion** — Intern-Decision behind the existing contract (done as an adapter; calibration on GAMST traces pending).
2. **Event-gated hybrid streaming** — cheap deltas trigger VLM calls with periodic heartbeat, VideoJev-style prefix caching ([StreamMind, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_StreamMind_Unlocking_Full_Frame_Rate_Streaming_Video_Dialogue_through_Event-Gated_ICCV_2025_paper.html)).
3. **Shared retrieval embeddings** — LanguageBind / ImageBind for cross-modal episode retrieval while keeping typed observations as the payload.
4. **Bounded intermediate-fusion specialists** — MBT-style audio-visual modules emitting another typed observation with all source IDs.
5. **Latent bottleneck compressor** — Perceiver IO queries producing bounded decision fields.
6. **Full streaming VLM fusion** — measured only as Gamst-bench arm E, never the silent default.

## 4. Top-venue papers and how GAMST applies them

Gamst-bench arms: **A** full VLM, **B** perception + text LLM, **C** GAMST hybrid with selective VLM, **D** periodic hybrid, **E** small always-on streaming VLM.

| Paper | Venue | GAMST application |
|---|---|---|
| Baltrušaitis et al., *Multimodal Machine Learning: A Survey and Taxonomy* | TPAMI 2019 | Document B/C as language-mediated late fusion; report alignment/freshness errors separately |
| Nagrani et al., *Attention Bottlenecks for Multimodal Fusion* | NeurIPS 2021 | Compare typed late fusion with an MBT AV specialist on synchronized events |
| Jaegle et al., *Perceiver* / *Perceiver IO* | ICML 2021 / ICLR 2022 | Fixed-size latent compressor and capability queries for bounded decision fields |
| Girdhar et al., *ImageBind* | CVPR 2023 | Embedding retrieval ablation in C for related audio/visual/depth episodes |
| Zhu et al., *LanguageBind* | ICLR 2024 | Retrieval for non-RGB sensors with JSON evidence as final interface |
| Chen et al., *VideoLLM-online* | CVPR 2024 | Score response timing and silence, not only final accuracy |
| Qian et al., *Dispider* | CVPR 2025 | Map sensors → decision provider → VLM escalation onto perception/decision/reaction stages |
| Niu et al., *OVO-Bench* | CVPR 2025 | Backward / real-time / forward modes as C/D/E metrics |
| Zhang et al., *Flash-VStream* | ICCV 2025 | Compare streaming visual memory with GAMST observation retention in E |
| Ding et al., *StreamMind* | ICCV 2025 | Event-triggered escalation in C vs fixed-interval D |
| Fu et al., *FrameFusion* | ICCV 2025 | VLM-internal token reduction vs GAMST external compression |
| Chen et al., *FastV* | ECCV 2024 | Prune visual tokens inside escalated VLM calls; check small-object evidence survives |
| Xiao et al., *StreamingLLM (attention sinks)* | ICLR 2024 | Bound LLM/VLM KV memory while retaining structured evidence separately |
| Xu et al., *StreamingVLM* | ICLR 2026 | Arm E baseline for long-run context stability |
| Chatterjee et al., *SPOT-Bench* | NeurIPS 2026 E&D | Timeliness-F1 and false-interruption metrics for GAMST abstention |
| Meta AI, *SAM 3* ([paper](https://arxiv.org/abs/2511.16719)) | ICLR 2026 | Concept segmentation provider; SAM3.1 video as optional tracker |
| *Depth Anything 3* / *Video Depth Anything* ([DA3](https://arxiv.org/abs/2511.10647), [VDA](https://arxiv.org/abs/2501.12375)) | ICLR 2026 / CVPR 2025 | Optional depth upgrades with temporal consistency |
| Garrido et al., *V-JEPA* / Assran et al., *V-JEPA 2* | ICLR 2025 / arXiv 2025 | Latent prediction error as a cheap novelty/escalation gate in C |
| Bruce et al., *Genie* | ICML 2024 | Separately labeled ablation for proactive timing; never real evidence |
| NVIDIA, *Cosmos World Foundation Models* | arXiv 2025 | Synthetic stress tests only; never mixed into evidence or scores |
| Brohan et al., *RT-2* | Science Robotics 2023 | Confidence/abstention gating of high-level action choices |
| Kim et al., *OpenVLA* | CoRL 2024 | GAMST selects goals for a VLA policy; native results reported separately |
| Black et al., *π0* | RSS 2025 | GAMST emits high-level goals; does not compete at motor frequency |
| Davis et al., *The Visual Microphone* | SIGGRAPH 2014 | Classical baseline for vibration-to-audio |
| *RAFT-PIV* ([article](https://www.nature.com/articles/s42256-021-00369-0)) | Nature Machine Intelligence 2021 | Optional dense flow on sampled pairs |

Links for every row are in the research tables above, the [fusion and decision notes](multimodal-decision.md) and [Gamst-bench comparisons](https://github.com/NomaDamas/Gamst-bench/blob/main/docs/COMPARISONS.md).
