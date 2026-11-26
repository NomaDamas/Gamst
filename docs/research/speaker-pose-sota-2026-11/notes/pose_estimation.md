# Open-source human pose estimation for video (SOTA as of Nov 2026), as a front end for counting repeated poses/gestures

Scope: single webcam/phone camera, one main person, counting repeated gestures (e.g. a "shotgun" arm+hand gesture, a passing motion). Star counts, last-push dates and repo SPDX license IDs were pulled from the GitHub API on 2026-11-25 (`gh api repos/<owner>/<repo>`). Every number below has a source. Items I could not verify are in the Gaps sections.

## Q1. Ranked shortlist for 2D body / whole-body keypoints: best accuracy vs. best real-time

### Takeaway
**Accuracy.** Meta's **Sapiens2** (Apr 2026) is the strongest open whole-body 2D estimator, ahead of Sapiens v1, RTMW, DWPose and ViTPose+ on Meta's own 308-keypoint in-the-wild test. Its weights are ViTs from 0.4B to 5B parameters at 1024×768 input, so it is a GPU-server model. **SDPose** (diffusion prior) is on par with Sapiens v1 on COCO-WholeBody and leads on out-of-domain robustness.
**Real-time with a permissive license.** **RTMPose/RTMW** (mmpose/rtmlib, Apache-2.0) and **MediaPipe Pose Landmarker** (Apache-2.0) are the leaders. **YOLO26-pose** has the best COCO-17 speed/accuracy among one-stage models but is AGPL-3.0.

### Cited Findings

**Sapiens2 (Meta, ICLR 2026, released 2026-04-24)**
- 308-keypoint whole-body skeleton (243 face, 40 hand). Top-down at 1024×768 input. Sizes are 0.4B, 0.8B, 1B and 5B. — [Sapiens2 paper](https://arxiv.org/html/2604.21681v1); [repo](https://github.com/facebookresearch/sapiens2)
- Table 3 of the paper (whole-body, 308 kp, flip test, same detections), mAP:
  - Baselines: ViTPose+-H 48.3, DWPose-L (384×288) 66.5, RTMW-L (384×288) 70.1, RTMW-X (384×288) 70.2, Sapiens-1B 76.8, Sapiens-2B 78.3.
  - Sapiens2: 0.4B 76.9, 0.8B 79.4, 1B 80.4, 5B 82.3.
  - — [Sapiens2 paper](https://arxiv.org/html/2604.21681v1)
- Caveats on these numbers:
  - This is Meta's own in-the-wild test set, not COCO-WholeBody.
  - The paper is internally inconsistent: Table 8 gives Sapiens-1B/2B as 74.1/75.3, not 76.8/78.3.
  - The test-set size appears as both "1K/11K" and "10K".
  - Baselines ran at their own lower input resolutions.
  - — [Sapiens2 paper](https://arxiv.org/html/2604.21681v1)
- Backbone cost at 1K resolution: 0.4B is 0.398B params / 1.26 TFLOPs; 1B is 1.462B / 4.715 TFLOPs; 5B is 5.071B / 15.72 TFLOPs. The paper reports no latency or FPS. — [Sapiens2 paper](https://arxiv.org/html/2604.21681v1); [repo](https://github.com/facebookresearch/sapiens2)
- The repo requires Python ≥3.12 and PyTorch ≥2.7, and uses an external person detector (DETR-R101 in the repo). 1,049 stars, last push 2026-05-24. — [repo](https://github.com/facebookresearch/sapiens2); GitHub API
- **Sapiens v1** (ECCV 2024): Sapiens-2B scores 74.4 AP on COCO-WholeBody val (cited by SDPose). The license is CC BY-NC 4.0 (non-commercial). 5,440 stars. — [SDPose paper](https://arxiv.org/html/2509.24980v2); [Sapiens v1 LICENSE](https://github.com/facebookresearch/sapiens/blob/main/LICENSE)

**SDPose (diffusion-prior pose; arXiv 2509.24980, v2 Dec 2025)**
- 72.8 AP / 79.4 AR on COCO-WholeBody val, against Sapiens-2B at 74.4 AP. The largest gap is on hands (67.3 vs 70.4). It claims +6 to +14 AP over HRNet, RTMPose and DWPose, and "on par" with Sapiens in-domain. Its main contribution is out-of-domain robustness. — [SDPose paper](https://arxiv.org/html/2509.24980v2)
- It uses a Stable Diffusion v2 U-Net backbone with 133 whole-body keypoints. Code is MIT; the repo has 181 stars (below the 5k bar) and was last pushed 2026-10-21. — [HF model card](https://huggingface.co/HENGER/SDPose-Wholebody); [repo](https://github.com/T-S-Liang/SDPose-OOD); GitHub API
- Measured at 0.84 fps on a V100, the slowest of the 8 systems evaluated. — [O'Brien et al. 2026, arXiv 2604.24609](https://arxiv.org/html/2604.24609)

**ViTPose / ViTPose++ (2022–2023)**
- COCO val at 256×192: ViTPose-H 79.1 (single-task) and 79.5 (multi-task); ViTPose++-H 79.4. ViTPose-G at 576×432 scores 81.0 val and 81.1 AP on COCO test-dev. — [ViTPose repo](https://github.com/ViTAE-Transformer/ViTPose)
- COCO-WholeBody AP for ViTPose++: S 54.4, B 57.4, L 60.6, H 61.2. — [ViTPose repo](https://github.com/ViTAE-Transformer/ViTPose)
- The README's last news item is from 2023-01-10, and the ViTPose-G weights entry is blank. Apache-2.0, 2,225 stars. These results are older (2022–23) and are now surpassed on whole-body. — [ViTPose repo](https://github.com/ViTAE-Transformer/ViTPose); GitHub API

**RTMPose / RTMW / DWPose / RTMO (OpenMMLab, mmpose and rtmlib)**
- RTMPose body (COCO val, AIC+COCO training):

  | Model | Input | AP | Params | CPU, ORT i7-11700 | GPU, TRT-FP16 GTX 1660Ti | Phone, ncnn Snapdragon 865 |
  |---|---|---|---|---|---|---|
  | RTMPose-t | 256×192 | 68.5 | 3.34M | 3.20 ms | 1.06 ms | 9.02 ms |
  | RTMPose-s | 256×192 | 72.2 | 5.47M | 4.48 ms | 1.39 ms | 13.89 ms |
  | RTMPose-m | 256×192 | 75.8 | 13.59M | 11.06 ms | 2.29 ms | 26.44 ms |
  | RTMPose-l | 384×288 | 77.3 | — | — | 6.05 ms | — |

  The headline claim is "RTMPose-m 75.8 AP at 90+ FPS CPU / 430+ FPS GPU". — [RTMPose project page](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- RTMW (Cocktail14 training), COCO-WholeBody Whole AP: RTMW-l 66.0 and RTMW-x 67.2 at 256×192; RTMW-l 70.1 and RTMW-x 70.2 at 384×288 (17.7 / 29.3 GFLOPs). — [RTMPose project page](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- DWPose (distilled 133-kp), COCO-WholeBody Whole AP: m 60.6, l 63.1 (256×192), l 66.5 (384×288). — [RTMPose project page](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- RTMO-l (one-stage): 74.8 AP on COCO val2017 at 141 FPS on a V100 (CVPR 2024). — [RTMO paper](https://arxiv.org/abs/2312.07526)
- The project page says its multi-dataset results are "not suitable for academic comparison". — [RTMPose project page](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- Maintenance: mmpose is Apache-2.0 with 8,080 stars, but was last pushed 2025-08-04, so it is slowing down. rtmlib (a lightweight ONNX runtime for RTMPose/RTMW/RTMO/DWPose) is Apache-2.0 with 705 stars, last pushed 2026-11-07. DWPose's repo was last pushed 2023-12. — GitHub API: [mmpose](https://github.com/open-mmlab/mmpose), [rtmlib](https://github.com/Tau-J/rtmlib), [DWPose](https://github.com/IDEA-Research/DWPose)

**YOLO26-pose (Ultralytics), COCO-17 keypoints, val2017, 640 px, end-to-end**

| Model | mAP 50-95 | CPU, ONNX | GPU, T4 TRT10 | Params |
|---|---|---|---|---|
| YOLO26n-pose | 57.2 | 40.3 ms | 1.8 ms | 2.9M |
| YOLO26s-pose | 63.0 | — | 2.7 ms | 10.4M |
| YOLO26m-pose | 68.8 | — | 5.0 ms | 21.5M |
| YOLO26l-pose | 70.4 | — | 6.5 ms | 25.9M |
| YOLO26x-pose | 71.6 | 565 ms | 12.2 ms | 57.6M |

- Source for the table: [Ultralytics pose docs](https://docs.ultralytics.com/tasks/pose/)
- Body only (17 kp): no hands or face. The repo is AGPL-3.0 with 63,723 stars and is very active (pushed 2026-11-25). — GitHub API, [ultralytics](https://github.com/ultralytics/ultralytics)

**MediaPipe Pose Landmarker (Google)**
- Lite, Full and Heavy bundles (float16). A 224×224 detector feeds a 256×256 landmarker. It outputs 33 landmarks with relative z, metric "world landmarks" (meters, hip-centred) and an optional segmentation mask. — [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- The doc page gives no latency figures. — [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- I found no new Pose Landmarker model released in 2025–2026. The releases page shows framework work instead, such as moving the iOS PoseLandmarker to the Tasks C API. The docs still list the same three bundles as "Latest". — [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker); [MediaPipe releases](https://github.com/google-ai-edge/mediapipe/releases)
- Apache-2.0, 37,840 stars, pushed 2026-11-25 (active). — GitHub API, [mediapipe](https://github.com/google-ai-edge/mediapipe)

**OpenPose (CMU)**
- 34,614 stars, but last pushed 2024-08. Its license is "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY", and derivatives become the licensor's property. — [OpenPose LICENSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE); GitHub API
- Measured at 4.40 fps on a V100 in the 2026 sign-language study. — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)

**Independent 2026 comparison: O'Brien et al., arXiv 2604.24609 (LREC 2026 SignLang workshop)**
- Eight estimators were run on RWTH-PHOENIX-Weather 2014 and scored on downstream sign-language translation. — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)
- BLEU: SDPose 11.68, Sapiens 11.53, AlphaPose 11.25, MMPose-Wholebody 10.90, OpenPose 10.61, MediaPipe Holistic 10.33, SMPLest-X 9.71, OpenPifPaf 9.37. The authors say "SDPose and Sapiens achieve the highest BLEU/BLEURT scores". — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)
- Share of signing frames with a missing hand (left / right / both): MediaPipe Holistic 20.22 / 23.61 / 8.84 %; OpenPose 8.43 / 3.88 / 0.22 %; OpenPifPaf 67.65 / 59.19 / 40.63 %. Top-down whole-body models never output zero-confidence hands. — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)
- Speed on a V100: AlphaPose 22.89 fps, SMPLest-X 8.36, OpenPifPaf 4.42, OpenPose 4.40, MMPose-WB 3.81, Sapiens 3.29, MediaPipe Holistic 3.15, SDPose 0.84. These are the study's offline-pipeline numbers, not tuned deployments. — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)

### Inferences
- **Best accuracy (2D whole-body):** 1. Sapiens2-1B or 5B; 2. Sapiens v1-2B, or SDPose for robustness to unusual appearance; 3. RTMW-x at 384×288. Sapiens2 leads but only on Meta's own 308-kp benchmark, and it has no COCO-WholeBody number. Treat its lead as likely but not independently confirmed.
- **Best real-time (2D):**
  1. RTMW-l/x or RTMPose-m via rtmlib: Apache-2.0, has hands and face, runs at hundreds of FPS on a GPU and real time on a CPU.
  2. MediaPipe Pose Landmarker (+ Hands/Holistic): Apache-2.0, runs on-device and in the browser, has built-in video tracking.
  3. YOLO26-pose: fastest one-stage model and best COCO-17 AP for its speed, but AGPL and body only.
  4. RTMO-l: one-stage, Apache-2.0.
- For a "shotgun" gesture with hands, a 17-kp body model (YOLO26, RTMO, RTMPose body) is not enough. Use whole-body (RTMW / DWPose / Sapiens2) or body plus a hand model.
- The 2604.24609 fps numbers (e.g. MediaPipe at 3.15 fps on a V100) reflect that study's batch pipeline. They contradict the usual real-time use of MediaPipe, so do not use them as a speed ranking.

### Gaps
- No independent COCO-WholeBody (val or test-dev) number for Sapiens2. The paper reports only its own 308-kp test set.
- No measured Sapiens2 latency or FPS anywhere, and no Apple-silicon benchmarks for any 2D model except what vendors claim.
- MediaPipe Pose Landmarker model-card accuracy (PCK) and latency were not retrieved. The model cards are on HF (litert-community) but I did not fetch them.
- YOLO11-pose figures were not on the current Ultralytics page. YOLO26 COCO test-dev was not found.
- RTMW3D / RTMPose3D H3WB numbers were not retrieved. No H3WB leaderboard was checked.
- Which mmpose model "MMPose Wholebody" means in 2604.24609 (RTMW? DWPose?) was not confirmed.

## Q2. 3D body (mesh / 3D keypoints) and hands: accuracy vs. real-time

### Takeaway
Single-image 3D accuracy is now led by Meta's **SAM 3D Body** and its accelerated **Fast SAM 3D Body**, with NLF-L close behind. A matched-protocol re-evaluation in Nov 2026 (InstantHMR) puts them at about 35–40 mm PA-MPJPE on 3DPW/EMDB, but at 130–416 ms per person on a desktop GPU. For real-time 3D, NLF-S (MIT) and the new **InstantHMR** (Apache-2.0 code, non-commercial weights except an "M-C" variant) are the options. For hands, **WiLoR** beats HaMeR on both accuracy and jitter but is CC BY-NC-ND.

### Cited Findings

**SAM 3D Body (Meta, released 2025-11-19; CVPR 2026, arXiv 2602.15989)**
- Single-image, promptable full-body HMR on the MHR rig, with body, feet and hands plus a hand-decoder refinement. Backbones are DINOv3-H+ (840M) and ViT-H (631M). — [repo](https://github.com/facebookresearch/sam-3d-body)
- Repo-reported numbers (DINOv3 / ViT-H): 3DPW MPJPE 54.8 / 54.8; EMDB MPJPE 61.7 / 62.9; RICH PVE 60.3 / 61.7; FreiHAND PA-MPJPE 5.5. — [repo](https://github.com/facebookresearch/sam-3d-body)
- No video or temporal mode. License is the "SAM License" for code and checkpoints. 3,766 stars, last push 2026-02-19. — [repo](https://github.com/facebookresearch/sam-3d-body); GitHub API

**Matched-protocol re-evaluation: InstantHMR, arXiv 2611.09195 (12 Nov 2026)**
- PA-MPJPE in mm (EMDB-1, 24 joints / 3DPW, 14 joints / RICH) and desktop GPU latency per person:

  | Method | EMDB-1 | 3DPW | RICH | GPU latency |
  |---|---|---|---|---|
  | Fast SAM 3D Body | 38.6 | 34.1 | 32.1 | 130 ms (PyTorch) |
  | SAM 3D Body | 40.3 | 35.1 | 33.5 | 416 ms (PyTorch) |
  | NLF-L | 39.3 | 35.1 | 27.3 | 4.32 ms (RTX 4070, TRT fp16) |
  | NLF-S | 45.0 | 39.4 | 33.7 | 1.52 ms |
  | CameraHMR | 46.7 | 38.8 | 33.6 | 5.31 ms |
  | TokenHMR | 54.7 | 44.5 | 42.1 | — |
  | HMR2.0b | 78.4 | 54.1 | 58.0 | 5.51 ms |
  | Multi-HMR-S | 68.5 | 54.8 | 69.4 | 3.94 ms |
  | SMPLer-X-S32 | 71.0 | 50.4 | 61.2 | 1.38 ms |
  | InstantHMR-M | 48.8 | 40.0 | 41.7 | 0.98 ms (TRT) |
  | InstantHMR-M + 2-crop fit | 44.4 | 39.3 | 41.0 | 1.53 ms |

  — [InstantHMR](https://arxiv.org/abs/2611.09195)
- On a Galaxy S23 NPU: InstantHMR-M takes 5.49 ms, NLF-S 67.97 ms and NLF-L 125 ms per person. SAM 3D Body, Fast SAM 3D Body, CameraHMR, TokenHMR and HMR2.0 do not run on phone NPUs. — [InstantHMR](https://arxiv.org/abs/2611.09195)
- Temporal jitter, as acceleration error on EMDB-1: InstantHMR-M 27.1 mm/frame² against NLF-S 21.2. A causal One-Euro filter cuts it to 15.2 at no added latency. — [InstantHMR](https://arxiv.org/abs/2611.09195)
- InstantHMR licensing: code Apache-2.0; models are "for non-commercial use", plus an M-C variant trained only on commercially usable images that is about 23 mm worse on EMDB-1 (72.2 mm). There is no dedicated hand branch. — [InstantHMR](https://arxiv.org/abs/2611.09195)
- Fast SAM 3D Body accelerates SAM 3D Body by up to 10.9× without retraining. — [InstantHMR related work](https://arxiv.org/abs/2611.09195), citing [arXiv 2603.15603](https://arxiv.org/abs/2603.15603)

**Video 3D (WHAM, GVHMR) and other HMR repos**
- WHAM runs online from past frames. Its reported 200 FPS excludes the keypoint detector and image encoder. GVHMR processes whole sequences, so it is offline. — [InstantHMR related work](https://arxiv.org/abs/2611.09195)
- GVHMR: TPAMI 2026; last news 2025-03-08, when SimpleVO replaced DPVO. Its LICENSE says "prohibited for commercial use", and modifications must be open-source; commercial licensing goes by email. 2,318 stars. — [GVHMR repo](https://github.com/zju3dv/GVHMR); GitHub API
- WHAM: MIT per the GitHub API, 1,184 stars, last pushed 2024-04, so it is stale. — [WHAM](https://github.com/yohanshin/WHAM)
- PromptHMR's LICENSE reads "Non-Commercial Scientific Research Use Only" (Meshcapade). 520 stars. — [PromptHMR LICENSE](https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE)
- Multi-HMR's NAVER "Non-Commercial License" says "Materials are used solely for non-commercial purposes". 431 stars. — [Multi-HMR](https://github.com/naver/multi-hmr)
- 4D-Humans (HMR2.0) is MIT with 1,743 stars. NLF is MIT with 471 stars, last pushed 2026-10-15. CameraHMR has no LICENSE detected by the GitHub API and 264 stars. — GitHub API: [4D-Humans](https://github.com/shubham-goel/4D-Humans), [NLF](https://github.com/isarandi/nlf), [CameraHMR](https://github.com/pixelite1201/CameraHMR)

**Hands**
- FreiHAND PA-MPJPE / PA-MPVPE: HaMeR 6.0 / 5.7, WiLoR 5.5 / 5.1. HO3D v2: HaMeR 7.7 / 7.9, WiLoR 7.5 / 7.7. — [WiLoR paper (CVPR 2025)](https://arxiv.org/html/2409.12259v2)
- Temporal jitter on HO3D: HaMeR 20.43, WiLoR 5.92, achieved without a temporal module. — [WiLoR paper](https://arxiv.org/html/2409.12259v2)
- WiLoR's detector runs at 138–175 FPS on an RTX 4090. The paper gives no end-to-end fps. — [WiLoR paper](https://arxiv.org/html/2409.12259v2)
- WiLoR's license is CC BY-NC-ND 4.0 (750 stars). HaMeR is MIT (1,199 stars) and uses a ViT-H backbone. SAM 3D Body's hand decoder reaches FreiHAND PA-MPJPE 5.5, equal to WiLoR. — [WiLoR repo](https://github.com/rolpotamias/WiLoR); GitHub API; [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)

### Inferences
- **3D, best accuracy:** SAM 3D Body or Fast SAM 3D Body (single image, needs a server GPU), with NLF-L close behind (MIT code). **3D, real-time:** NLF-S or NLF-L on a GPU, InstantHMR-M on phones and edges, and MediaPipe world landmarks as a no-cost baseline.
- For counting gestures, 3D is optional. View-invariant joint angles (elbow and shoulder angles, wrist-to-shoulder vectors) can come from MediaPipe world landmarks or NLF. A full mesh adds little, and SAM 3D Body's latency (130–416 ms) rules it out for live counting.
- Video-native 3D (WHAM, GVHMR) smooths offline but carries a non-commercial license (GVHMR) or is stale (WHAM). An online per-frame model plus a One-Euro filter is the more practical pattern.

### Gaps
- No verified H3WB numbers for any model.
- Published (not re-evaluated) 3DPW/EMDB numbers for PromptHMR, CameraHMR, WHAM and GVHMR were not fetched. The InstantHMR table excludes PromptHMR, WHAM and GVHMR.
- MediaPipe Hands accuracy and latency were not retrieved.
- NLF weight licensing: the code is MIT, but the weights output SMPL-family parameters, whose own licenses are unverified here.
- SMPL/SMPL-X body-model licensing (commonly non-commercial via MPI/Meshcapade) was not fetched in this session and should be checked. MHR was reported as Apache-2.0 only by a secondary search snippet.

## Q3. Temporally stable video keypoints for counting

### Takeaway
Only MediaPipe has a built-in video and tracking mode. The 2026 independent study found MediaPipe (and SMPLest-X) had the lowest jitter, but MediaPipe drops hands most often. Top-down research models (Sapiens, SDPose, RTMW) are per-frame and noticeably jerkier, so add a tracker and a causal filter (e.g. One-Euro) before counting.

### Cited Findings
- MediaPipe running modes:
  - VIDEO: synchronous tracking across frames, with monotonic timestamps.
  - LIVE_STREAM: asynchronous, drops frames while the pipeline is busy.
  - IMAGE: stateless; tracking resets on every call.
  - The tracking-confidence threshold defaults to 0.5.
  - — [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- Jerk-based jitter for all keypoints (Signsuisse / Phoenix): MediaPipe 2.46 / 6.51, SMPLest-X 2.74 / 3.97, AlphaPose 7.36 / 10.32, Sapiens 7.77 / 10.86, OpenPose 6.84 / 26.36, MMPose-WB 8.58 / 14.47, SDPose 14.97 / 13.24. The authors: "MediaPipe and SMPLest-X yield the lowest jitter scores", and "higher translation scores do not necessarily come with smoother trajectories". — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)
- Sapiens was correct in all 15 occlusion instances examined. — [arXiv 2604.24609](https://arxiv.org/html/2604.24609)
- Per-frame HMR jitters on video. A causal One-Euro filter cut InstantHMR-M's acceleration error from 27.1 to 15.2 mm/frame² with no added latency and 0.1–0.3 mm better PA-MPJPE. — [InstantHMR](https://arxiv.org/abs/2611.09195)
- WiLoR's hand jitter is 5.92 against HaMeR's 20.43 on HO3D. — [WiLoR paper](https://arxiv.org/html/2409.12259v2)

### Inferences
- **Pipeline for counting.**
  1. Pose: MediaPipe in VIDEO/LIVE_STREAM mode, or RTMW/RTMPose via rtmlib with a simple tracker.
  2. Smoothing: a One-Euro or Savitzky–Golay filter on joint angles.
  3. Counting: a hysteresis state machine, or template or DTW matching, over owner-declared pose exemplars.
- **Hand dropouts matter for hand-shape gestures.** MediaPipe Holistic's 20–24 % per-hand dropout in signing video means a hand-shape-dependent "shotgun" gesture may need RTMW (which never drops hands) or a dedicated hand model.

### Gaps
- No standard jitter benchmark exists for 2D whole-body models apart from the single 2026 sign-language study. Its fps and jitter came from offline pipelines on signing video, so they may not carry over to webcam gesture counting.
- No source found for rtmlib's or RTMW's built-in smoothing or tracking behaviour.

## Q4. License compatibility with an Apache-2.0 SDK, and self-hosting on a GPU server

### Takeaway
**Clean (Apache-2.0 or MIT):** MediaPipe, mmpose/rtmlib (RTMPose/RTMW/RTMO/DWPose), ViTPose, HaMeR, 4D-Humans, NLF code, WHAM code, SDPose code.
**Custom Meta licenses (commercial use allowed with use restrictions):** Sapiens2 and SAM 3D Body. Sapiens2 explicitly bans "biometric processing" and surveillance.
**Copyleft:** Ultralytics YOLO26 (AGPL-3.0).
**Non-commercial:** Sapiens v1, OpenPose, GVHMR, PromptHMR, Multi-HMR, WiLoR, InstantHMR's default weights.
Everything listed can be self-hosted on a GPU server. The heavy models (Sapiens2 1B/5B, SAM 3D Body, SDPose) effectively require one.

### Cited Findings
- Apache-2.0: mediapipe, mmpose, rtmlib, ViTPose, DWPose. MIT: SDPose-OOD, 4D-Humans, NLF, WHAM, HaMeR. AGPL-3.0: ultralytics. — GitHub API license fields: [mediapipe](https://github.com/google-ai-edge/mediapipe), [mmpose](https://github.com/open-mmlab/mmpose), [rtmlib](https://github.com/Tau-J/rtmlib), [ultralytics](https://github.com/ultralytics/ultralytics)
- **Sapiens2 License** covers code and weights ("models, software… trained model weights"):
  - It is a royalty-free licence to use, modify and distribute. Redistribution must carry the Agreement.
  - It prohibits use "for purposes of surveillance", "for biometric processing", inferring health or sensitive information without consent, and deepfakes.
  - Meta may audit use and require deletion.
  - — [Sapiens2 LICENSE.md](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
- **SAM License** (SAM 3D Body code and checkpoints) is a royalty-free licence to use, modify and distribute, with trade-control, ITAR and military restrictions. — [SAM 3D Body LICENSE](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)
- **Non-commercial or restrictive:**
  - Sapiens v1: CC BY-NC 4.0. — [LICENSE](https://github.com/facebookresearch/sapiens/blob/main/LICENSE)
  - OpenPose: academic and non-profit research only. — [LICENSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
  - GVHMR: commercial use prohibited. — [repo](https://github.com/zju3dv/GVHMR)
  - PromptHMR: non-commercial scientific research only. — [LICENSE](https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE)
  - Multi-HMR: NAVER non-commercial. — [repo](https://github.com/naver/multi-hmr)
  - WiLoR: CC BY-NC-ND 4.0. — [repo](https://github.com/rolpotamias/WiLoR)
  - InstantHMR: default weights non-commercial; M-C weights commercially usable. — [InstantHMR](https://arxiv.org/abs/2611.09195)

### Inferences
- **Licence compatibility for an Apache-2.0 SDK:**
  - MediaPipe and rtmlib/RTMW can be default dependencies.
  - YOLO26 (AGPL) should be an optional extra that users opt into, never bundled.
  - Sapiens2 and SAM 3D Body can be optional server-side adapters, with their license and Acceptable Use Policy passed through to users. The "biometric processing" ban needs legal review if GAMST tracks a specific person. Gesture counting is arguably not biometric identification, but this is not settled.
  - Non-commercial models (OpenPose, Sapiens v1, GVHMR, PromptHMR, Multi-HMR, WiLoR) should not be defaults.
- **Self-hosting:**
  - All weights listed are downloadable (HF, GitHub or OneDrive) and run on PyTorch or ONNX, so server hosting is feasible.
  - Sapiens2-5B (~15.7 TFLOPs per 1K frame) and SAM 3D Body (130–416 ms per person) are batch or near-real-time at best.
  - RTMW, RTMPose and NLF run at hundreds of FPS on a TensorRT GPU.

### Gaps
- Whether the weights of SDPose (SD2 U-Net, potentially CreativeML OpenRAIL++-M), ViTPose (OneDrive), and DWPose/RTMW carry the same license as their code was not individually verified.
- No legal interpretation of the Sapiens2 "biometric processing" clause for gesture counting was found.
- The SMPL/SMPL-X model-file licensing that affects most HMR outputs (HMR2.0, CameraHMR, NLF, WHAM, GVHMR) was not verified in this session.
