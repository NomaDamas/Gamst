# Training-free / zero-shot / exemplar repetitive action counting and pose/action matching (state as of 2026-11-25)

Scope note: numbers below are copied from paper tables I read in full text (pdftotext of the PDFs) unless marked otherwise. GitHub star counts were read live via the GitHub API on 2026-11-25. Older results (pre-2025) are flagged with their year. Most RepCount numbers are *quoted by later papers from earlier papers*, not re-run; where two papers report different numbers for the same method, it is noted.

## Q1. Method landscape and benchmark numbers (RepCount-A, Countix, UCFRep)

### Takeaway
On paper, the best published RepCount-A number in Nov 2026 is "Count What Repeats" (PAMS, CVPR 2026 Findings), a skeleton-based self-supervised counter (MAE 0.129 / OBO 0.842). Next come CountLLM (CVPR 2025, MAE 0.162 / OBO 0.639) and the exemplar-trained ESCounts (ACCV 2024, MAE 0.213 / OBO 0.563). D2-STX (CVPR 2026 Findings) leads among supervised non-LLM video+pose models only on matched backbones. No independent re-run of PAMS exists, and I found no public code for it. On an independent Oct 2026 benchmark (RepProbe), ESCounts is by far the strongest *specialist* counter, and zero-shot VLMs (Qwen3-VL-8B, InternVL3.5-8B, etc.) do much worse, collapsing at high counts.

### Cited Findings

**Consolidated RepCount-A / Countix / UCFRep (in-domain, MAE lower is better, OBO higher is better)**, quoted from the PAMS paper's Table 1 (CVPR 2026 Findings) — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf):

| Method (venue) | Input | RepCount MAE / OBO | Countix MAE / OBO | UCFRep MAE / OBO |
|---|---|---|---|---|
| RepNet (CVPR'20) | RGB | 0.995 / 0.013 | 0.432 / 0.393 | 0.987 / 0.018 |
| TransRAC (CVPR'22) | RGB | 0.443 / 0.291 | 0.478 / 0.283 | 0.581 / 0.329 |
| ESCount(s) (ACCV'24) | RGB | 0.213 / 0.563 | 0.276 / 0.673 | 0.216 / 0.704 |
| IVAC-P2L (TMM'25) | RGB | 0.402 / 0.344 | 0.507 / 0.431 | 0.503 / 0.420 |
| PoseRAC (ICONIP'24) | skeleton | 0.236 / 0.560 | 0.387 / 0.497 | 0.312 / 0.452 |
| GMFL (arXiv'24) | skeleton | 0.216 / 0.560 | 0.269 / 0.594 | 0.259 / 0.650 |
| SPKDB-Net (CVIU'25) | skeleton | 0.210 / 0.605 | 0.320 / 0.612 | 0.320 / 0.612 |
| BIGC-Net (EAAI'25) | skeleton | 0.215 / 0.579 | – | 0.287 / 0.598 |
| CountLLM (CVPR'25) | RGB + LLM | 0.162 / 0.639 | 0.215 / 0.801 | 0.116 / 0.839 |
| **PAMS / Count What Repeats (CVPR'26 F)** | skeleton, self-supervised | **0.129 / 0.842** | **0.098 / 0.879** | 0.208 / 0.686 |

- PAMS zero-shot cross-dataset transfer: RepCount→Countix 0.099 / 0.861; RepCount→UCFRep 0.156 / 0.762; Countix→RepCount 0.129 / 0.842, the same as its in-domain number; Countix→UCFRep 0.174 / 0.723 — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
- PAMS method: a Transformer encoder (about 12M params; 4 layers, d=512) trained on *unlabeled skeleton sequences* with a Period-Adaptive Multi-Scale Temporal Cycle Consistency loss (scales 0.5/1.0/1.5). At inference a period head plus a "Multi-Expert Consensus" algorithm (FFT plus adaptive peak detection at several smoothing scales, with voting) produce the count. Trained on one A100. The paper claims the fastest inference of the compared methods (bubble plot in its Fig. 1) but gives no FPS number in the text I extracted — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf); [CVF abstract page](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html)
- PAMS reports its own UCFRep result (0.208 MAE) losing to CountLLM (0.116), and blames the small size of UCFRep for self-supervised learning — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)

**D2-STX (CVPR 2026 Findings; Wang, Wang, Liu, Hu, Yu; Zhejiang U / XJTU / PolyU)** — [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)
- Dual-branch video (Swin-T / RX3D101 / R(2+1)D18) plus pose (BlazePose 2D keypoints into CTR-GCN pre-trained on NTU-RGB+D), fused by decoupled spatial-temporal cross-attention. Outputs a density map. Trained fully supervised in three stages (50+35+25 epochs) on one RTX 4090.
- RepCount-A (Swin-T): RMSE 6.616, MAE 0.255, OBZ 0.200, OBO 0.500, against ESCounts (Swin-T) 6.905 / 0.298 / 0.183 / 0.403.
- UCFRep (RX3D101): MAE 0.236 / OBO 0.777, against ESCounts 0.247 / 0.731. Countix (R(2+1)D18): MAE 0.278 / OBO 0.734, against ESCounts 0.293 / 0.701.
- Cross-dataset (train RepCount-A, no fine-tune): →UCFRep 0.297 / 0.643; →Countix 0.359 / 0.605. ESCounts scores 0.317 / 0.571 and 0.374 / 0.521; RepNet scores 0.998 / 0.009 and 0.729 / 0.183.
- Pose-only branch on RepCount-A is much weaker: MAE 0.843 / OBO 0.173, against video-only 0.298 / 0.403. The paper attributes this to distracting people and occlusions degrading pose estimation.
- Limitations section admits CountLLM outperforms D2-STX on the compared metrics. It claims D2-STX's advantages are inference of about 10–20 ms, training on 1×4090 against CountLLM's 8×A100, and being prompt-free.
- Code: "https://github.com/ZJU-Emerging-AI-Lab/D2-STX" (footnote in paper).
- Conflict: D2-STX compares against ESCounts' **Swin-T** variant (0.298 / 0.403 on RepCount-A). ESCounts' best **ViT-B (VideoMAE)** variant reports 0.213 / 0.563, which beats D2-STX's 0.255 / 0.500 on RepCount-A — [ESCounts arXiv](https://arxiv.org/abs/2403.18074). So "SOTA" in D2-STX holds only under the matched backbone.

**ESCounts / Every Shot Counts (ACCV 2024; Sinha, Stergiou, Damen; Bristol)** — [ESCounts arXiv](https://arxiv.org/abs/2403.18074); [ACCV page](https://openaccess.thecvf.com/content/ACCV2024/html/Sinha_Every_Shot_Counts_Using_Exemplars_for_Repetition_Counting_in_Videos_ACCV_2024_paper.html); [project page](https://sinhasaptarshi.github.io/escounts)
- Encoder-decoder that cross-attends video tokens to 0–2 exemplar clips during training and learns a latent z0 for exemplar-free inference. Encoder is a **frozen** VideoMAE-pretrained ViT-B (Kinetics-400). Only the decoder and z0 are trained (300 epochs, one V100). It uses 64-frame sliding windows (16×224×224 input), producing 1568 tokens per window, and ensembles 4 sequences at inference.
- RepCount (ViT-B): RMSE 4.455, MAE 0.213, OBZ 0.245, OBO 0.563. TransRAC scores 0.443 / 0.291 and RepNet 0.995 / 0.013. RepCount "open set" split: MAE 0.436 / OBO 0.519, against TransRAC 0.625 / 0.204.
- Countix: R(2+1)D18 MAE 0.293 / OBO 0.701; ViT-B 0.276 / 0.673; RepNet 0.364 / 0.697 (2020 number).
- UCFRep (ViT-B): MAE 0.216 / OBO 0.704. Methods that fine-tune their encoder do better: Context 0.147 / 0.790, Sight & Sound 0.143 / 0.800.
- **Exemplar use at inference (Table 4).** With 1 exemplar from the same video, RepCount scores MAE 0.210 / OBO 0.589 and UCFRep 0.203 / 0.714. With 3 exemplars from the same video, RepCount scores 0.206 / 0.596 and UCFRep 0.198 / 0.723. Exemplars from a *different* video of the same class work comparably (1-shot RepCount 0.207 / 0.563). Gains over exemplar-free (0.213 / 0.563) are small.

**CountLLM (CVPR 2025; Yao, Cheng, Huang, Li)** — [CountLLM arXiv](https://arxiv.org/abs/2503.17690); [CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html)
- Video encoder (VideoMAE or Video Swin, Kinetics-pretrained, LoRA-tuned), a periodicity transformer, a projector and Vicuna-7B. It is prompted with periodicity text and instruction-tuned.
- RepCount: VideoMAE 0.162 / 0.639; VSwinT 0.209 / 0.552. UCFRep (VideoMAE): 0.116 / 0.839. Countix (VideoMAE): 0.215 / 0.801.
- Cross-dataset: RepCount→UCFRep 0.139 / 0.816; RepCount→Countix 0.281 / 0.730.
- Training cost is 8×A100, according to the D2-STX paper's comparison — [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)

**PoseRAC (arXiv 2023 / ICONIP 2024; Yao, Cheng, Zou)** — [PoseRAC arXiv](https://arxiv.org/abs/2303.08450)
- Pose Saliency Transformer over per-frame keypoints. It needs "Pose Saliency Annotation": the authors *pre-define two salient poses per action class* (for example squat = upright / squatting) and annotate their frame indices in training videos. Counting is done by classifying each frame against the class's salient poses. It reports OBO 0.56 on RepCount (vs TransRAC 0.29) and plots a speed advantage in FPS (Fig. 3; exact FPS not extracted).
- D2-STX states PoseRAC "does not satisfy the action-agnostic setting" — [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)

**RepNet (CVPR 2020; Dwibedi et al., Google)** — [RepNet arXiv](https://arxiv.org/abs/2006.15418)
- Class-agnostic: a temporal self-similarity matrix (TSM) over frame embeddings feeds a learned period predictor, trained on synthetic repetitions plus Countix. Countix MAE 0.364 / OBO 0.697 (2020 number, as quoted in [ESCounts](https://arxiv.org/abs/2403.18074)). On RepCount it transfers very poorly: MAE 0.995 / OBO 0.013 — [ESCounts](https://arxiv.org/abs/2403.18074).

**Other 2024–2026 open-set / text-conditioned work**
- OVR dataset + OVRCounter (arXiv 2407.17085, 2024): open-vocabulary, text- or video-conditioned counting on Kinetics/Ego4D. On OVR-Kinetics, OVRCounter scores MAE 0.39 and OBO-*error* 0.41, against RepNet 0.86 / 0.65. On OVR-Ego4D it scores MAE 0.35 / OBOE 0.34, against RepNet 0.74 / 0.57. With mismatched text, its density output drops to near zero, so text conditioning does select *which* repetition is counted — [OVR arXiv](https://arxiv.org/abs/2407.17085)
- MultiCounter (arXiv 2409.04035): multi-instance action-agnostic counting in untrimmed videos. Found in search, not read in detail — [MultiCounter arXiv](https://arxiv.org/abs/2409.04035)

**VLM zero-shot counting (independent benchmark, Oct 2026)** — RepProbe (Khoshnoodi and Bargal, Georgetown; 42,233 clips; RepProbe-Real is built from RepCount) — [RepProbe arXiv](https://arxiv.org/abs/2610.18844)
- Overall MAE / exact / OBO / OBO in the k=14–20 band: ESCounts 1.07 / 41.9% / 77.5% / 65.6%; RepNet 2.20 / 21.7% / 53.0% / 21.3%; Cosmos-Reason2-8B 2.89 / 17.5% / 47.5% / 22.2%; Qwen3-VL-8B 5.02 / 12.9% / 35.0% / 4.9%; InternVL3.5-8B 5.98 / 7.3% / 18.7% / 7.3%. Video-R1-7B, VideoChat-R1.5-7B, LLaVA-OneVision-7B and LongVILA-R1-7B score worse still.
- All generalist VLMs reach 0% OBO in the high-count band on the synthetic split. A linear probe on the VLM's final-layer features recovers counts much better than its generated answers do.
- Note: RepProbe uses MAE in absolute count units, which is not comparable to the normalized MAE in the table above.

### Inferences
- For a "count *this specific* user-defined gesture" product, raw leaderboard rank matters less than **conditioning**. PAMS, D2-STX, RepNet, TransRAC and CountLLM (without a custom prompt) count *the dominant repetition* in a clip and cannot be told which movement to count. Only ESCounts (video exemplar), OVRCounter (text or video conditioning) and pose-template matching let the user specify the target.
- PAMS's numbers are the most interesting but are self-reported, without public code. Treat them as unverified until reproduced.
- Exemplars help ESCounts only slightly (MAE 0.213 to 0.206). The model was trained on the RepCount fitness classes, so exemplar conditioning on truly novel gestures (for example a streamer's "shotgun" motion) is untested in the literature.

### Gaps
- No paper evaluates these counters on non-fitness, short, sparse gestures (streamer signature gestures, sports passes). All benchmarks are dense, periodic and mostly fitness or Kinetics.
- No exact FPS numbers were extracted for ESCounts, PoseRAC or PAMS. D2-STX's "10–20 ms" is its own claim, and GPU/resolution details for that timing are not given.
- I could not confirm whether TransRAC's own reported RepCount numbers differ from the 0.443 / 0.291 quoted by later papers. They are consistent across the ESCounts, D2-STX and PAMS tables.

## Q2. Which are truly training-free for an unseen action, and which need per-action labels?

### Takeaway
None of the learned counters needs per-action training to *run* on an unseen action: RepNet, TransRAC, ESCounts, D2-STX, PAMS and CountLLM are all class-agnostic at inference. But all of them were trained once on a counting dataset, and only PAMS avoids count labels altogether (self-supervised). PoseRAC is the one that needs **per-action-class** salient-pose definitions and labels. The only truly training-free pipelines are signal-processing ones: pose keypoints plus self-similarity or FFT peaks, template matching, or DTW against one exemplar.

### Cited Findings
- **RepNet** is class-agnostic at inference, but its period predictor is trained with count/period supervision on synthetic data and Countix — [RepNet arXiv](https://arxiv.org/abs/2006.15418). PAMS describes RepNet as able to "infer counts without labels" but still relying on synthetic supervision — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
- **ESCounts** is zero-shot or exemplar at inference. The encoder is frozen, but the decoder and zero-shot latent are trained on count density maps, and exemplars are annotated repetition segments. PAMS notes ESCounts "still requires a template annotation and is vulnerable to tempo" variation — [ESCounts arXiv](https://arxiv.org/abs/2403.18074); [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
- **D2-STX** is fully supervised with density and count losses, and action-agnostic at test time — [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)
- **PAMS** is trained without count labels on unlabeled skeletons. Its inference is FFT plus peak-detection consensus, and it transfers across datasets without fine-tuning — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
- **CountLLM** is instruction-tuned with count supervision. It is prompt-dependent: D2-STX notes it "rel[ies] on precise periodicity descriptions" — [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf); [CountLLM arXiv](https://arxiv.org/abs/2503.17690)
- **PoseRAC** pre-defines two salient poses for every action class and annotates them in the training set, so it is per-action supervised — [PoseRAC arXiv](https://arxiv.org/abs/2303.08450)
- **Generalist VLMs** (Qwen3-VL, InternVL3.5, etc.) are fully zero-shot via prompting, but their counting is unreliable, especially above about 14 reps — [RepProbe arXiv](https://arxiv.org/abs/2610.18844)
- **Skeleton one-shot action recognition** (NTU RGB+D 120 one-shot protocol) recognizes but does not count. These methods train on base classes and then classify novel classes from one exemplar.
  - SkelHCC (arXiv 2606.03610, June 2026) top-1 with 20/40/60/80/100 base classes: 52.0 / 60.9 / 67.4 / 67.9 / 71.6%. It does not update the backbone and adapts only 0.5M parameters. Its "LMV-Cache" is training-free.
  - Prior methods at 100 base classes: SkeletonX (2025) 69.1%, MC-Scale (2024) 68.7%, CrossGLG (2024) 62.6%, ALCA-GCN 57.6%, SL-DML 50.9%, uDTW 49.0%, APSR (2020) 45.3%.
  - SkelHCC states its code "will be available" (not confirmed public).
  - Source: [SkelHCC arXiv](https://arxiv.org/abs/2606.03610)

### Inferences
- For "at most one demonstration, no training on that movement", two kinds of option meet the constraint without any GAMST-side training:
  - (a) A frozen pretrained class-agnostic counter. ESCounts with a 1-shot exemplar is the only one with exemplar input and public weights. PAMS has no exemplar input and no public code.
  - (b) A training-free pose pipeline: keypoints, a normalized pose feature, similarity to the exemplar, then peak or DTW counting.
- One-shot skeleton recognizers (SkelHCC, SkeletonX) answer "is this the demonstrated action?" at about 70% top-1 on NTU-120 novel classes. That is a useful gate, but it does not count and it depends on the skeleton format of the base-class training set.

### Gaps
- No paper benchmarks one-shot *pose hold* detection (static pose + duration). This is a pose-similarity thresholding problem, and I found no standard benchmark or SOTA for it.

## Q3. Code, weights, license, GitHub stars, and weight (real-time?)

### Takeaway
All of the main video counters have public code, but their repos are tiny (at most 120 stars) and none is a maintained product. ESCounts (MIT, 32 stars) is the best-documented exemplar-capable option. D2-STX (MIT, 1 star) is very new. I found no public repo for PAMS/Count What Repeats or CountLLM. PAMS (12M-param skeleton Transformer) and D2-STX (10–20 ms claimed) are the light ones. ESCounts (ViT-B on 64-frame windows) and CountLLM (7B LLM) are heavier.

### Cited Findings
GitHub stars, license and last push date, read via the GitHub API on 2026-11-25:

| Repo | Stars | License | Last push | Source |
|---|---|---|---|---|
| RepNet (in the google-research monorepo, so stars are not RepNet-specific) | 39,285 | Apache-2.0 | 2026-11 | [google-research/google-research](https://github.com/google-research/google-research) |
| materight/RepNet-pytorch (PyTorch port with pretrained weights) | 42 | none listed | 2024-11 | [materight/RepNet-pytorch](https://github.com/materight/RepNet-pytorch) |
| SvipRepetitionCounting/TransRAC | 120 | Apache-2.0 | 2025-02 | [TransRAC repo](https://github.com/SvipRepetitionCounting/TransRAC) |
| sinhasaptarshi/EveryShotCounts (ESCounts) | 32 | MIT | 2024-12 | [EveryShotCounts repo](https://github.com/sinhasaptarshi/EveryShotCounts) |
| MiracleDance/PoseRAC | 22 | MIT | 2023-04 | [PoseRAC repo](https://github.com/MiracleDance/PoseRAC) |
| ZJU-Emerging-AI-Lab/D2-STX | 1 | MIT | 2026-04 | [D2-STX repo](https://github.com/ZJU-Emerging-AI-Lab/D2-STX) |
| raphaelmemmesheimer/skeleton-dml (one-shot skeleton) | 24 | none listed | 2023-02 | [skeleton-dml repo](https://github.com/raphaelmemmesheimer/skeleton-dml) |

- CountLLM: no GitHub repo found by search (`gh search repos CountLLM` returned nothing), and no code URL appears in the arXiv PDF text — [CountLLM arXiv](https://arxiv.org/abs/2503.17690). **Code not confirmed public.**
- Count What Repeats / PAMS: no code URL appears in the CVF PDF text. **Code not confirmed public.** — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
- Weight and complexity as stated by the papers:
  - PAMS: encoder of about 12M parameters with O(T²·d) attention plus an O(T log T) FFT consensus; the authors call it practical for deployment — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
  - D2-STX: inference of about 10–20 ms; trained on 1×RTX 4090 — [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)
  - CountLLM: Vicuna-7B backbone; trained on 8×A100 — [CountLLM arXiv](https://arxiv.org/abs/2503.17690); [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)
  - ESCounts: frozen ViT-B over 64-frame windows (1568 tokens each), with predictions aggregated over 4 sequences — [ESCounts arXiv](https://arxiv.org/abs/2403.18074)
- Video foundation backbones that could serve as exemplar-similarity embedders:

  | Repo | Stars | License | Note | Source |
  |---|---|---|---|---|
  | facebookresearch/vjepa2 | 4,915 | MIT | | [vjepa2](https://github.com/facebookresearch/vjepa2) |
  | OpenGVLab/VideoMAEv2 | 836 | MIT | | [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) |
  | OpenGVLab/InternVideo (includes InternVideo2) | 2,422 | Apache-2.0 | | [InternVideo](https://github.com/OpenGVLab/InternVideo) |
  | microsoft/VideoX (includes X-CLIP) | 1,072 | NOASSERTION | license not clear | [VideoX](https://github.com/microsoft/VideoX) |

  The counting papers already use this family: ESCounts and CountLLM rely on VideoMAE (v1) ViT-B as their encoder — [ESCounts arXiv](https://arxiv.org/abs/2403.18074); [CountLLM arXiv](https://arxiv.org/abs/2503.17690)
- Pose and point-tracking front-ends:

  | Repo | Stars | License | Source |
  |---|---|---|---|
  | google-ai-edge/mediapipe (BlazePose; D2-STX uses BlazePose for its pose stream) | 37,840 | Apache-2.0 | [mediapipe](https://github.com/google-ai-edge/mediapipe); [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf) |
  | open-mmlab/mmpose | 8,080 | Apache-2.0 | [mmpose](https://github.com/open-mmlab/mmpose) |
  | google-deepmind/tapnet | 2,274 | Apache-2.0 | [tapnet](https://github.com/google-deepmind/tapnet) |

### Inferences
- Pose-only pipelines (BlazePose or RTMPose keypoints, then a 12M-param or smaller model or pure math) are easily real-time on CPU or edge. ESCounts-class ViT-B video counters need a GPU and run in windows, so they suit offline or near-real-time use.
- Low star counts mean these research repos would have to be vendored or wrapped, not depended on as maintained libraries.

### Gaps
- I did not check whether ESCounts or D2-STX publish trained checkpoints, as opposed to code only. The repos exist, but weight availability was not opened. Verify before relying on them.
- No published speed benchmarks for V-JEPA 2, InternVideo2 or X-CLIP used as counting embedders were found.

## Q4. Pose-keypoint periodicity / DTW baselines vs learned counters

### Takeaway
The evidence is thin and mostly old. A training-free skeleton self-similarity plus spectrogram method (Hsu et al., 2021) reports very high accuracy on clean single-person exercise data (MAE 0.06, OBO 0.89–0.95), but it was never run on RepCount, so it cannot be compared head-to-head. On in-the-wild RepCount, the 2026 evidence says the *pose-only* learned branch is much worse than video (MAE 0.843 vs 0.298) because of pose-estimation failures. The best 2026 result (PAMS) nevertheless *is* skeleton-based, with an FFT and peak-detection back end: a learned periodic embedding followed by classical periodicity math.

### Cited Findings
- Hsu et al., "Viewpoint-Invariant Exercise Repetition Counting" (arXiv 2107.13760, 2021, older): computes a skeleton self-similarity, then analyzes its spectrogram to count, with no counting model trained. MM-Fit overall MAE 0.06 / OBOA 0.94; UI-PRMD MAE 0.06 / OBOA 0.95. With window 128 and step 1/2 it scores MAE 0.08 / OBOA 0.89. On MM-Fit it matched or beat the Strömbäck et al. baseline per exercise, which the authors say needed motion-specific hyperparameters — [Hsu et al. arXiv](https://arxiv.org/abs/2107.13760)
- D2-STX single-branch ablation:
  - RepCount-A: pose-only (BlazePose + CTR-GCN) MAE 0.843 / OBO 0.173; video-only 0.298 / 0.403; fused 0.255 / 0.500.
  - UCFRep: pose-only 0.464 / 0.394; video-only 0.324 / 0.614.
  - The authors blame distracting people, occlusions, and a domain gap from NTU 3D pretraining.
  - Source: [D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)
- Skeleton-only *learned* counters (PoseRAC, GMFL, SPKDB-Net, BIGC-Net) reach RepCount MAE 0.21–0.24, equal to or better than RGB ESCounts (0.213). PAMS (skeleton, self-supervised, FFT/peak back end) reaches 0.129 — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)
- RepNet's core is itself a self-similarity matrix: a learned TSM followed by a period classifier. It is the learned analogue of autocorrelation, yet it transfers poorly to RepCount (MAE 0.995) — [RepNet arXiv](https://arxiv.org/abs/2006.15418); [ESCounts arXiv](https://arxiv.org/abs/2403.18074)
- DTW for one-shot skeleton matching: uDTW (a learned uncertainty-DTW, Wang & Koniusz 2022) gets 49.0% on NTU-120 one-shot at 100 base classes, against 71.6% for SkelHCC (2026). DTW-style matching works but trails recent learned matchers — [SkelHCC arXiv](https://arxiv.org/abs/2606.03610)
- ESCounts frames its approach as matching repetitions to exemplars, a learned analogue of template matching. PAMS argues ESCounts is "vulnerable to tempo" mismatch, while its multi-scale consensus is tempo-robust: they report stable MAE across 0.1×–2.0× playback speed, where ESCounts and PoseRAC degrade — [Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)

### Inferences
- For single-person, front-facing streamer video (clean pose), a training-free pose pipeline is likely competitive. The Hsu et al. results on clean exercise data and the strength of skeleton counters on RepCount both point that way. The failure modes are multi-person scenes, occlusion and out-of-frame limbs, the conditions where D2-STX's pose branch collapsed.
- A practical hybrid follows the PAMS recipe in spirit:
  1. Normalize keypoints (root-centred, scale-normalized, optionally view-normalized).
  2. Compute a 1D similarity-to-exemplar signal, using cosine or OKS against the exemplar's key pose(s), or subsequence DTW against the exemplar trajectory.
  3. Run multi-scale peak detection with hysteresis or minimum-period constraints, then vote.
  This needs no training and gives *target selectivity*, which PAMS and D2-STX lack. It is an inference, not a benchmarked result.
- Sports passes (soccer, basketball) are sparse, aperiodic events. Periodicity and FFT counters (RepNet, PAMS) assume quasi-periodic dense repetition and are a poor fit. Exemplar or template detection (DTW or one-shot recognition over sliding windows) is the more appropriate formulation. This is also an inference.

### Gaps
- No 2024–2026 paper reports a training-free pose plus DTW or autocorrelation baseline on RepCount-A. A direct "math vs learned" number on the standard benchmark does not exist in what I found.
- Hsu et al.'s datasets (MM-Fit, UI-PRMD) are lab or exercise datasets. Their numbers are not comparable to RepCount's normalized MAE in difficulty.

## Q5. Is there a ≥5k-star open-source project that does exemplar pose counting out of the box?

### Takeaway
I found none. The ≥5k-star projects in this space are pose estimators or general CV frameworks: Ultralytics 63.7k (AGPL-3.0), MediaPipe 37.8k, MMPose 8.1k. Their built-in rep counters are rule- or angle-threshold-based for predefined exercises, not exemplar-based. Every exemplar or class-agnostic counting repo found has at most 120 stars.

### Cited Findings
- ultralytics/ultralytics: 63,723 stars, AGPL-3.0 — [ultralytics repo](https://github.com/ultralytics/ultralytics). The AIGym workout solution counts reps from joint-angle up/down thresholds. This is per the task brief and prior knowledge of the Ultralytics docs; **I did not re-verify the AIGym source in this session**.
- google-ai-edge/mediapipe: 37,840 stars, Apache-2.0. It provides pose landmarks, not a counting product — [mediapipe repo](https://github.com/google-ai-edge/mediapipe)
- open-mmlab/mmpose: 8,080 stars, Apache-2.0, last push 2025-08 — [mmpose repo](https://github.com/open-mmlab/mmpose)
- facebookresearch/vjepa2: 4,915 stars, MIT. Just under 5k; a video embedding model, not a counter — [vjepa2 repo](https://github.com/facebookresearch/vjepa2)
- GitHub search "repetition counting" sorted by stars returned at most 54 stars for any repo (Xiaodomgdomg/Deep-Temporal-Repetition-Counting, CVPR 2020 code). Nothing near 5k — [GitHub search](https://github.com/search?q=repetition+counting&type=repositories&s=stars)

### Inferences
- An "exemplar pose counting out of the box" capability would be new for an open-source project of significant size. GAMST would be assembling it from a pose front-end (MediaPipe or RTMPose under Apache-2.0, avoiding Ultralytics' AGPL) plus its own exemplar-similarity and peak/DTW logic, optionally benchmarked against ESCounts (MIT).

### Gaps
- My GitHub search used only the query "repetition counting". Repos named differently (for example "rep counter", "exercise counter", "fitness AI") could exist with more stars. A broader search was not run within the tool budget.
