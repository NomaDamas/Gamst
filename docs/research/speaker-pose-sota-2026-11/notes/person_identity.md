# Enrolled-person identity in video: face recognition, body re-ID, MOT and promptable tracking (as of 2026-11-25)

Stars were read via `gh api repos/OWNER/REPO` on 2026-11-25. "README" means the repo README fetched via `gh api .../readme` on the same date. All benchmark numbers are as self-reported by the authors unless noted.

## Q1. Face recognition: which model should do the enrollment match? (IJB-B / IJB-C TAR@FAR=1e-4)

### Takeaway
At IJB-C 1e-4, the open models with public weights sit close together, about 97.3-97.8. TopoFR R200 (Glint360K) has the highest self-reported number at 97.84. LVFace-B is 97.70 and has the cleanest license (MIT code, MIT weights on HF). AdaFace R100 (WebFace12M) is 97.66 and is the most battle-tested on low-quality faces. InsightFace code is MIT, but **all InsightFace pretrained models, including buffalo_l and antelopev2, are non-commercial research only (verified)**.

### Cited Findings
- **InsightFace license (verified):** the README says "The code of InsightFace is released under the MIT License". It also says the training data "(and the models trained with these data) are available for non-commercial research purposes only". This applies to manual downloads and to auto-downloads through the python library. Commercial use of buffalo_l requires a licence from recognition-oss-pack@insightface.ai. The model_zoo README says "ALL models are available for non-commercial research purposes only." — [insightface README](https://github.com/deepinsight/insightface#license); [model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo). Stars: 30,299. The GitHub license field is empty.
- **InsightFace buffalo_l** (SCRFD-10GF + R50@WebFace600K): IJB-C(E4) 97.25 and MR-ALL 91.25. buffalo_s (MBF): IJB-C(E4) 95.02. antelopev2 is R100@Glint360K. Packs are 159-407 MB. — [model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- **InsightFace ArcFace ONNX recognition models:** IJB-C(E4) 97.13 for Glint360k R100 and 97.12 for WebFace600K. — [model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- **AdaFace (CVPR'22), TAR@FAR=0.01% IJB-B / IJB-C:**

  | Backbone | Training data | IJB-B | IJB-C |
  |---|---|---|---|
  | R100 | WebFace12M | 96.41 | 97.66 |
  | R100 | WebFace4M | 96.03 | 97.39 |
  | R100 | MS1MV3 | 95.84 | 97.09 |
  | R50 | WebFace4M | 95.44 | 96.98 |
  | R18 | WebFace4M | 93.03 | 94.99 |

  Code is MIT and weights are on GDrive. Stars: 994. Last push 2024-07. — [AdaFace README](https://github.com/mk-minchul/AdaFace). The follow-up KP-RPE / CVLface claims SoTA on TinyFace and IJB-S. — [same README](https://github.com/mk-minchul/AdaFace), [CVLface](https://github.com/mk-minchul/CVLface) (numbers not pulled)
- **TopoFR (NeurIPS'24), IJB-C 1e-5 / 1e-4:**
  - Glint360K: R200 96.71 / 97.84, R100 96.57 / 97.60, R50 95.99 / 97.27.
  - MS1MV2: R200 95.15 / 97.08.
  - Weights are on GDrive. Stars: 44. The GitHub API reports no license (unverified, treat as unlicensed). — [TopoFR README](https://github.com/DanJun6737/TopoFR); [arXiv 2410.10587](https://arxiv.org/abs/2410.10587)
- **LVFace (ByteDance, ICCV 2025 Highlight, ViT + Progressive Cluster Optimization), Glint360K-trained:**

  | Model | IJB-C 1e-6 | IJB-C 1e-5 | IJB-C 1e-4 | IJB-B 1e-4 |
  |---|---|---|---|---|
  | LVFace-T | 88.53 | 95.63 | 96.67 | 95.41 |
  | LVFace-S | 90.06 | 96.52 | 97.31 | 96.14 |
  | LVFace-B | 90.06 | 97.00 | 97.70 | 96.51 |
  | LVFace-L | 89.51 | 97.02 | 97.66 | 96.51 |

  - Weights are on HF in ONNX and .pt (released Aug 2025). Code is MIT; the HF card tag is `license:mit`. Stars: 149.
  - The paper claims SoTA on IJB-B/C and the MFR challenge.
  - Sources: [LVFace README](https://github.com/bytedance/LVFace); [HF](https://huggingface.co/bytedance-research/LVFace); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.html)
- **Older reference points:** InterFace reports IJB-B 95.23 / IJB-C 96.49 at 1e-4. — [arXiv 2210.02018](https://arxiv.org/pdf/2210.02018)
- **FaceLiVTv2-L on an aggregator:** listed at 96.59 TPR@1e-4 on IJB-C (Apr 2026). Unverified; the aggregator conflicts with itself. — [wizwand](https://www.wizwand.com/sota/face-recognition-on-ijb-c-tar-at-far-1e-5-1e-4-1e-3)
- **deepface:** a wrapper (MIT, 23,734 stars). It inherits the licenses of the weights it wraps. — gh api ([repo](https://github.com/serengil/deepface))

### Inferences
- Ranking, by category:
  - **Best accuracy:** TopoFR R200-Glint360K at 97.84. Its license is unclear.
  - **Best usable:** LVFace-B at 97.70 (MIT code and weights, ONNX) ≈ AdaFace R100-WebFace12M at 97.66 (MIT code).
  - **Best real-time:** InsightFace buffalo_s/sc (MBF, 16 MB) for research only, or AdaFace R18 / LVFace-T if a commercial-safe option is needed.
- The weight licenses do not settle commercial use, because the training datasets carry their own terms. Glint360K, WebFace4M/12M and MS1M carry research-oriented dataset terms, so whether weights trained on them can be used commercially is **unverified** for LVFace and AdaFace as well. Legal review is needed before any product use.
- Differences of about 0.1-0.5 pt at IJB-C 1e-4 are smaller than the domain shift to livestream or sports faces (blur, profile, small faces). Each candidate should be gated on a local enrollment set (Owner rule: measured-only claims).

### Gaps
- No per-image latency or FPS was published for LVFace, AdaFace or TopoFR. I found no 2026 face-recognition model with public weights that beats about 97.8 at IJB-C 1e-4.
- I could not confirm a license file for TopoFR, or the commercial terms of the Glint360K and WebFace datasets.

## Q2. Body re-ID when the face is not visible (Market-1501 / MSMT17)

### Takeaway
Among methods with public code and weights, SOLIDER Swin-B (Apache-2.0) and CLIP-ReID ViT-B/16 (MIT) lead: about 76-77 mAP on MSMT17 without re-ranking and about 86.5-86.7 with re-ranking. OSNet (MIT, 2.2M params) is the real-time choice. Cross-domain generalization is poor for all supervised models, so the person's own enrollment clip matters more than leaderboard rank.

### Cited Findings
- **SOLIDER Swin-B, mAP / R1 without re-ranking:** Market 93.9 / 96.9 and MSMT17 77.1 / 90.7. With re-ranking: Market 95.6 / 96.7 and MSMT17 86.5 / 91.7.
  - Swin-T: Market 91.6 / 96.1 and MSMT17 67.4 / 85.9.
  - Code is Apache-2.0 and weights are on GDrive. Stars: 1,510. Last push 2023-07. Re-ID code is in tinyvision/SOLIDER-REID.
  - Source: [SOLIDER README](https://github.com/tinyvision/SOLIDER)
- **CLIP-ReID (AAAI'23), Table 2 without re-ranking:**
  - ViT-B/16 + SIE + OLP: MSMT17 75.8 mAP / 89.7 R1 and Market 90.5 / 95.4.
  - ViT-B/16: MSMT17 73.4 / 88.7.
  - R50: MSMT17 63.0 / 84.4 and Market 89.8 / 95.7.
  - With re-ranking: MSMT17 86.7 mAP / 91.1 R1.
  - Code is MIT, weights are on GDrive, and CLIP weights are OpenAI's (MIT). Stars: 532.
  - Sources: [arXiv 2211.13977](https://arxiv.org/html/2211.13977); [README](https://github.com/Syliz517/CLIP-ReID)
- **FastReID:**
  - Market1501 SBS(R101-ibn): R1 96.3 / mAP 90.3 / mINP 70.0.
  - MSMT17 BoT(R50-ibn): R1 77.0 / mAP 54.4.
  - Apache-2.0 with weights in GitHub releases. Stars: 4,011.
  - Source: [MODEL_ZOO](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
- **OSNet x1.0** (2.2M params, 0.98 GFLOPs), same-domain R1 (mAP): Market 94.2 (82.6), Duke 87.0 (70.2) and MSMT17 74.9 (43.8).
  - osnet_x0_25 (0.2M params, 0.08 GFLOPs): Market 91.2 (75.0).
  - MIT. Stars: 4,950.
  - My column reading is market / duke / msmt order. Source: [deep-person-reid MODEL_ZOO](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md)
- **2026 survey (arXiv 2601.20598):**
  - Supervised models "dominate their training domain but collapse elsewhere". It reports OSNet at 83.57 mAP on Market vs 3.37 on MSMT17 (cross-domain).
  - Off-the-shelf CLIP, DINOv2, SigLIP2 and PE-Core features score under 6 mAP for re-ID, so generic VLM embeddings are not a substitute.
  - The authors call hybrids like CLIP-ReID "the best for ReID tasks currently".
  - Source: [arXiv 2601.20598](https://arxiv.org/html/2601.20598v1)
- **CION pretraining:** reported at 93.3 / 74.3 mAP on Market / MSMT17 with R50-IBN. Venue and weights are unverified. — [HF papers listing](https://huggingface.co/papers?q=person+ReId)
- **BoxMOT** ships ReID weights (OSNet, LMBN, CLIP-ReID) for trackers, but it is AGPL-3.0. — [BoxMOT README](https://github.com/mikel-brostrom/boxmot)

### Inferences
- Ranking, by category:
  - **Best accuracy:** SOLIDER Swin-B, or CLIP-ReID-SIE-OLP with re-ranking (MSMT17 about 86.5-86.7 mAP). Re-ranking is offline and gallery-based, so it does not fit streaming.
  - **Best usable:** CLIP-ReID ViT-B/16 (MIT) or SOLIDER Swin-B (Apache-2.0).
  - **Best real-time:** OSNet x1.0 or x0.25 (MIT).
- Body re-ID is clothing-dependent. In livestreams and sports, teammates often wear identical kits. It should serve as short-horizon glue between face confirmations, not as primary identity.

### Gaps
- I found no verified CVPR / ICCV 2025 supervised re-ID SoTA with public weights. Searches surfaced only older or unverifiable claims.
- No FPS numbers were published for SOLIDER or CLIP-ReID.

## Q3. MOT with re-ID and promptable video tracking (MOT17 / DanceTrack / SportsMOT HOTA)

### Takeaway
- **Tracking-by-detection (MIT-licensed, real-time):** BoostTrack++ is 66.6 HOTA on MOT17 test and BoT-SORT-ReID is 65.0.
- **End-to-end:** MOTIP (Apache-2.0, CVPR'25) is strongest on DanceTrack (67.5-73.7 HOTA) and SportsMOT (71.9-75.2), the benchmarks most like sports.
- **Promptable mask trackers:** SAM 3 / 3.1 (SAM License) and SAMURAI (Apache-2.0, on SAM 2.1) keep a clicked or prompted person but do not themselves re-identify after long exits. GAMST already has SAM 3, SAM 3.1 video, EfficientTAM and OC-SORT adapters.

### Cited Findings
- **BoostTrack / ++, MOT17 test:**
  - BoostTrack++: HOTA 66.6, MOTA 80.7, IDF1 82.2, IDSW 1062.
  - BoostTrack (no re-ID): HOTA 65.4.
  - MOT20 test: BoostTrack++ 66.4.
  - The authors say it ranks first among online methods in HOTA and retains real-time execution speed. MIT. Stars: 235.
  - Source: [README](https://github.com/vukasin-stanojevic/BoostTrack)
- **BoT-SORT-ReID:** MOT17 test MOTA 80.5, IDF1 80.2, HOTA 65.0. MOT20 test HOTA 63.3. MIT. Stars: 1,586. — [README](https://github.com/NirAharon/BoT-SORT)
- **Deep OC-SORT:** MOT17 HOTA 64.9 / IDF1 80.6, MOT20 63.9 and DanceTrack HOTA 61.3 (AssA 45.8). The authors report about 6 HOTA over OC-SORT on DanceTrack. MIT. Stars: 288. OC-SORT is MIT with 1,165 stars. — [README](https://github.com/GerardMaggiolino/Deep-OC-SORT)
- **MOTIP (CVPR 2025, ID-prediction formulation):**
  - DanceTrack HOTA: 67.5 base, 71.4 with CrowdHuman, 73.7 trained on train+val.
  - SportsMOT HOTA: 71.9, and 75.2 trained on train+val.
  - Apache-2.0. Stars: 587. Some checkpoints are marked TBD.
  - Sources: [MOTIP repo](https://github.com/MCG-NJU/MOTIP); [arXiv 2403.16848](https://arxiv.org/abs/2403.16848)
- **BoxMOT unified benchmark (same detector / ReID per its setup; config unverified):**

  | Tracker | MOT17-ablation HOTA / IDF1 | SportsMOT-val HOTA / IDF1 |
  |---|---|---|
  | occluboost (new) | 71.10 / 85.28 | 83.17 / 89.36 |
  | botsort | 69.68 / 82.33 | 76.93 / 78.30 |
  | boosttrack | 69.25 / 83.20 | 76.32 / 77.82 |
  | deepocsort | 67.95 / 80.54 | 79.51 / 79.59 |
  | bytetrack | 67.68 / 79.16 | — |

  AGPL-3.0. Stars: 8,330. — [BoxMOT README](https://github.com/mikel-brostrom/boxmot)
- **SAMURAI** (SAM 2.1 + Kalman motion-aware memory, zero-shot): "7.1% AUC gain on LaSOT_ext and a 3.5% AO gain on GOT-10k". It "operates in real-time" (no FPS stated). Apache-2.0 code on SAM 2.1 weights (Apache-2.0). Stars: 7,125. — [arXiv 2411.11922](https://arxiv.org/abs/2411.11922); [README](https://github.com/yangchris11/samurai)
- **SAM 2:** Apache-2.0, 20,223 stars. — gh api ([repo](https://github.com/facebookresearch/sam2))
- **SAM 3:**
  - Results: SA-V test cgF1 30.3 / pHOTA 58.0, YT-Temporal-1B 50.8 / 69.9, SmartGlasses 36.4 / 63.6, LVVIS mAP 36.3 and BURST HOTA 44.5.
  - License: the "SAM License" (custom, updated 2025-11-19). It is royalty-free with derivatives allowed, but prohibits ITAR, military and weapons uses. Stars: 12,401.
  - Sources: [SAM 3 README](https://github.com/facebookresearch/sam3); [LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- **SAM 3.1 Object Multiplex** (2026-03-27) uses shared memory for joint multi-object tracking. Reported results:
  - About 7x speedup at 128 objects on a single H100 vs SAM 3 (Nov 2025).
  - +2.0 on MOSEv2 and VOS gains on 6 of 7 benchmarks.
  - Mixed SA-Co/VEval results.
  - Weights: [HF facebook/sam3.1](https://huggingface.co/facebook/sam3.1). Source: [RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

### Inferences
- **Recommended pipeline for one enrolled person:**
  1. Run a detector plus a MIT tracker (BoT-SORT or BoostTrack++; GAMST already has OC-SORT) for every person.
  2. Run face embedding (LVFace or AdaFace) on tracklets whenever a face is visible, and confirm the target ID against the enrollment gallery.
  3. Use body re-ID (CLIP-ReID or OSNet) embeddings, kept per session, to re-attach after occlusion or re-entry when no face is visible.
  4. Optionally hand the confirmed box to SAM 3.1 or EfficientTAM for a mask track.
- SAM-family trackers lack an identity gallery. After an exit they need a re-prompt, so the face and re-ID gate should issue that re-prompt.
- Ranking, by category:
  - **Best accuracy:** MOTIP on DanceTrack and SportsMOT; BoostTrack++ on MOT17.
  - **Best usable:** BoT-SORT-ReID or BoostTrack++ (MIT, weights public). Avoid BoxMOT in non-AGPL products.
  - **Best real-time:** BoostTrack without re-ID, or OC-SORT. Use SAM 3.1 Multiplex when masks are needed (GPU).

### Gaps
- No HOTA was found for SAM 3 / 3.1 or SAMURAI on MOT17, DanceTrack or SportsMOT. They report VOS and SA-Co metrics only, so the two families cannot be compared directly.
- Missing FPS: BoostTrack++, MOTIP and SAMURAI have no figures on comparable hardware. Only the SAM 3.1 relative speedup is published.
- The BoxMOT "occluboost" tracker's provenance and paper are unverified. Its numbers come from BoxMOT's own evaluation protocol.
