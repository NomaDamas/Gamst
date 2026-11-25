# Verification of open-source SOTA facts (checked 2026-11-25)

Method: `gh api` repo metadata/READMEs/LICENSE files, Hugging Face `/api/models` JSON, paper PDFs via pdftotext. "Unverified" = not found in a primary source.

## Q1. w2v-BERT 2.0 SV (arXiv 2510.04213): code, checkpoints, licenses, WeSpeaker

### Takeaway
The code and many trained checkpoints are public. The repo is CC BY-NC-SA 4.0. The HF checkpoint repo states no license, and its training data includes VoxBlink2, which is NC-SA, so treat it as non-commercial. WeSpeaker ships the same LoRA-Adapter-MFA model; its code is Apache-2.0, and its docs say model weights follow the dataset license. The base facebook/w2v-bert-2.0 is MIT.

### Cited Findings
- Repo ZXHY-82/w2v-BERT-2.0_SV: created 2025-09-15, last push 2026-03-28, 61 stars. GitHub reports license `NOASSERTION`. `License.md` is the full **CC BY-NC-SA 4.0** text — [repo](https://github.com/ZXHY-82/w2v-BERT-2.0_SV), [License.md](https://github.com/ZXHY-82/w2v-BERT-2.0_SV/blob/main/License.md)
- The README has training recipes in 3 stages (frozen+LoRA, then joint FT, then large-margin FT) plus KD-guided structured pruning. It uses torch 2.7.1/cu118 and was trained on 8 GPUs. Model size is "580+6.2M" params — [README](https://github.com/ZXHY-82/w2v-BERT-2.0_SV)
- Checkpoints on HF `zl389/w2v-bert-2.0_SV` (not gated, created 2025-10-05, modified 2026-04-27). There is **no license tag and no README/model card** — [HF API](https://huggingface.co/api/models/zl389/w2v-bert-2.0_SV), [HF repo](https://huggingface.co/zl389/w2v-bert-2.0_SV)
  - Trained on Vox2+VoxBlink2: `model_base_0.23.pth` (Vox1-O/E/H EER 0.23/0.38/0.81%) and `model_lmft_0.14.pth` (0.14/0.31/0.73%) — [README](https://github.com/ZXHY-82/w2v-BERT-2.0_SV)
  - Trained on Vox2 only, per stage: `Adapter_MFA_voxceleb2/s1..s3` (best 0.18/0.37/0.81%) and `Lora_Adapter_MFA_voxceleb2/s1..s3` (best 0.23/0.46/1.03%) — same
  - Not in the README: `Lora_Adapter_MFA_VoxCleleb2_VoxBlink2_CnCeleb1&2_KeSpeech_3dSpeaker/s1,s2` and `TidyVoice2026_GRL/ckpt_0044.pth` — [HF API file list](https://huggingface.co/api/models/zl389/w2v-bert-2.0_SV)
- VoxBlink2 is CC BY-NC-SA 4.0, and the project says models trained on it are also CC BY-NC-SA 4.0, with "no commercial application" — [VoxBlink2 LICENSE](https://raw.githubusercontent.com/VoxBlink2/ScriptsForVoxBlink2/main/LICENSE), [voxblink2.github.io](https://voxblink2.github.io)
- WeSpeaker (Apache-2.0 code, 1477 stars) merged PR #439 "Integrate w2v-bert2-LoRA-adapter-MFA model" on 2025-12-02 and PR #466 (VoxCeleb reproduction) on 2026-07-01 — [PR 439](https://github.com/wenet-e2e/wespeaker/pull/439), [PR 466](https://github.com/wenet-e2e/wespeaker/pull/466)
- WeSpeaker `docs/pretrained.md` lists the following checkpoints — [pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
  - VoxCeleb: `W2V-BERT2.0` / `W2V-BERT2.0_LM` (ModelScope `shangguanqituan/wespeaker-w2v-bert2`)
  - VoxCeleb+VoxBlink2: `voxceleb_voxblink2_w2v_bert2_lora_adapterMFA(_lm).tar.gz` (wenet.org.cn)
- The CLI flag `--w2vbert2_mfa` maps to `voxceleb_voxblink2_w2v_bert2_lora_adapterMFA_lm.tar.gz` — [cli/utils.py](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/utils.py), [cli/hub.py](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/hub.py)
- WeSpeaker model license, quoted: "The pretrained model in WeNet follows the license of it's corresponding dataset". It gives VoxCeleb → CC BY 4.0 as the example — [pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- WeSpeaker's VoxCeleb reproduction of LoRA-MFA reaches EER 0.250/0.398/0.838% (Vox1-O/E/H) after the frozen(LoRA) → joint FT → LMFT stages — [examples/voxceleb/v2/README.md](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md)
- facebook/w2v-bert-2.0 is **MIT** (HF tag `license:mit`), not gated, about 1.0M downloads — [HF](https://huggingface.co/facebook/w2v-bert-2.0)

### Inferences
- Under WeSpeaker's dataset-license rule, the VoxCeleb-only `W2V-BERT2.0(_LM)` is effectively CC BY 4.0 (commercial use OK with attribution). The `voxblink2` LoRA-MFA tarballs, which are also the `--w2vbert2_mfa` CLI default, inherit CC BY-NC-SA 4.0.
- For the zl389 HF weights, the only explicit license is the NC-SA repo license, so assume non-commercial.

### Gaps
- No per-file license text inside the WeSpeaker tarballs or the ModelScope card was checked.
- Whether WeSpeaker ships ONNX for the w2v-BERT models: unverified.

## Q2. Meta Sapiens2 (facebookresearch/sapiens2, arXiv 2604.21681)

### Takeaway
Pose models come in 0.4B/0.8B/1B/5B, predict 308 keypoints, and are public on HF without gating, under a custom "Sapiens2 License". There is no 0.1B pose model. No latency/fps figure is reported. A TorchScript export script exists, but there is no ONNX export or prebuilt TorchScript checkpoint. The license explicitly bans "biometric processing".

### Cited Findings
- Repo: created 2026-03-01, 1049 stars. GitHub license `NOASSERTION` (custom `LICENSE.md`). The paper is ICLR 2026, Meta Reality Labs — [repo](https://github.com/facebookresearch/sapiens2), [arXiv PDF](https://arxiv.org/pdf/2604.21681)
- Backbones (params / FLOPs at 1024×768): 0.1B 0.114B/0.342T; 0.4B 0.398B/1.260T; 0.8B 0.818B/2.592T; 1B 1.462B/4.715T; 1B-4K 1.607B; 5B 5.071B/15.722T — [README](https://github.com/facebookresearch/sapiens2)
- Pose checkpoints (308 keypoints, Sociopticon format, 274 of them on the face; top-down, using DETR-R101-DC5 person boxes): `facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}` — [MODEL_ZOO.md](https://github.com/facebookresearch/sapiens2/blob/main/docs/MODEL_ZOO.md), [POSE.md](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md)
- HF `facebook/sapiens2-pose-0.4b` and `-1b`: `gated: false`, tag `license:other`, files are `model.safetensors` plus `sapiens2_*_pose.safetensors` — [HF 0.4b](https://huggingface.co/facebook/sapiens2-pose-0.4b), [HF 1b](https://huggingface.co/facebook/sapiens2-pose-1b)
- Latency/fps: none found in the README, POSE.md, or the arXiv PDF. The only related lines are about A100 bf16 training and the demo default "2 jobs/GPU" — [POSE.md](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md), [arXiv PDF](https://arxiv.org/pdf/2604.21681)
- Export: `sapiens/pose/tools/deployment/pytorch2torchscript.py` and `sapiens/dense/tools/deployment/pytorch2torchscript.py` exist. There is no ONNX script in the repo tree, and the HF pose repos have no `.pt`/`.onnx` files — [repo tree](https://github.com/facebookresearch/sapiens2/tree/main/sapiens/pose/tools/deployment)
- Community port: `mnmly/mlx-swift-sapiens2`, an MLX-Swift inference-only port of the backbone and 4 task heads (not verified in depth) — [repo](https://github.com/mnmly/mlx-swift-sapiens2)
- License §b.vi, exact text: "Your use of the Sapiens Materials will not involve or encourage others to use for: (i) for purposes of surveillance, including any research or development relating to surveillance; (ii) for biometric processing; … (iv) for attempting to identify or re-identify any individual included in the Sapiens Materials; … (vi) to collect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws" — [LICENSE.md](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
- Other clauses: §b.iii requires compliance with "any and all laws governing the processing of biometric information"; §b.iv bans reverse engineering; §c gives Meta audit and deletion rights — same

### Inferences
- "Biometric processing" is not defined in the license. Using pose keypoints (especially the 274 face keypoints) in a per-person product is a legal-risk area. Treat Sapiens2 as research/offline only unless counsel clears it.

### Gaps
- Measured fps on any GPU or on MPS: not reported by Meta.

## Q3. ESCounts (ACCV 2024, Sinha/Stergiou/Damen)

### Takeaway
The code is public under MIT at sinhasaptarshi/EveryShotCounts, and a trained RepCount checkpoint is public on Google Drive. Inference needs an old stack: torch 1.10/cu111, detectron2, mmcv-full, pytorchvideo, and a VideoMAE ViT-B encoder.

### Cited Findings
- Repo `sinhasaptarshi/EveryShotCounts`: MIT, 32 stars, last push 2024-12-18. `sinhasaptarshi/escounts` is only the project webpage — [code](https://github.com/sinhasaptarshi/EveryShotCounts), [webpage repo](https://github.com/sinhasaptarshi/escounts)
- README: "You can download our trained model from [here]", linking to a Google Drive file — [README](https://github.com/sinhasaptarshi/EveryShotCounts), [Drive](https://drive.google.com/file/d/1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48/view?usp=drive_link)
- Environment: python 3.8, torch==1.10.0+cu111, detectron2 cu111 wheel, mmcv-full via openmim, a pinned pytorchvideo commit, av 10.0.0 — [README](https://github.com/sinhasaptarshi/EveryShotCounts)
- Encoder: VideoMAE ViT-B/16x4 MAE (Kinetics) from `dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth` — [README](https://github.com/sinhasaptarshi/EveryShotCounts)
- Exemplar tokens have shape N×3×8×14×14. `token_pool_ratio 0.4` spatially pools tokens to save memory — same
- `demo.py --video_name data/x.mp4 --resource cpu` defaults to CPU and loads `repcount_trained.pyth` — [demo.py](https://github.com/sinhasaptarshi/EveryShotCounts/blob/main/demo.py)
- Reported RepCount results: OBO 0.39→0.56, MAE 0.38→0.21 — [README/abstract](https://arxiv.org/abs/2403.18074)

### Gaps
- Whether the demo uses user-supplied exemplars or the zero-shot latent: unverified (the CLI takes only a video).
- Runtime per clip: not reported.

## Q4. Count What Repeats / PAMS (CVPR 2026 Findings, Gao et al.)

### Takeaway
No public code or weights were found.

### Cited Findings
- The CVF supplement exists. Title: "Count What Repeats: Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive Action Counting". It reports OBO 84.2% and held-out-category tests. Its text contains no GitHub or code link (pdftotext grep) — [supplement PDF](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf)
- GitHub search for "Count What Repeats" / "PAMS repetition counting" finds no official repo. `jklongxing/counting-what-repeats` (MIT, created 2026-11-13) is a different project: a VLM action-conditioned counting benchmark — [repo](https://github.com/jklongxing/counting-what-repeats)

### Gaps
- The main-paper PDF URL guess on CVF returned HTML, so the main paper was not checked for a code link.

## Q5. SkelHCC (arXiv 2606.03610)

### Takeaway
Code is public at lya19971103/SkelHCC (ICML 2026). The README claims Apache-2.0 but no LICENSE file exists. No released weights were found. Note that SkelHCC is one-shot skeleton action *recognition*, not counting.

### Cited Findings
- The paper's title is "Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition". It evaluates on NTU-60/120 and PKU-MMD, and says "Code will be available at github.com/SkelHCC" — [arXiv PDF](https://arxiv.org/pdf/2606.03610)
- Repo `lya19971103/SkelHCC`: created 2026-05-22, last push 2026-06-29, 6 stars. GitHub license is `null`. The README badge and §License say Apache 2.0, but no LICENSE file is in the root listing — [repo](https://github.com/lya19971103/SkelHCC)
- The README shows `main.py --weights WEIGHTS_DIR` but gives no download link for weights — same

## Q6. 2025-2026 open (code AND weights) exemplar-conditioned / open-set repetition counting or one-shot action spotting

### Takeaway
None found that satisfies code + weights + video + exemplar-conditioned and is newer than ESCounts. ESCounts (2024) remains the only verified open exemplar-video counter with weights.

### Cited Findings
- CountLLM (CVPR 2025) is generalizable counting via an LLM. No code repo was found by GitHub search — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html)
- ActRep-R1 (Apache-2.0, 2026-05) is an RL recipe for Qwen-VL counting. Its README covers training and eval but has no released checkpoint link — [repo](https://github.com/yicheng-2019/ActRep-R1)
- CaRaCount (TPAMI'25 / AAAI) is class-agnostic few-shot exemplar counting, but on **wearable IMU** data, not video. License `NOASSERTION` — [repo](https://github.com/bbvisual/CaRaCount)
- TempoRAC is an Apache-2.0 multi-person RAC research log. Its README says "No formal response checkpoint exists" — [repo](https://github.com/appleweiping/TempoRAC)
- Few-shot action localization: only the older WFSAL (ICMR'21) turned up — [repo](https://github.com/June01/WFSAL-icmr21)

### Gaps
- One-shot action spotting with 2025-26 weights: none found. The search was GitHub plus one web search, so it is not exhaustive.

## Q7. Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B

### Takeaway
The models are `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`, plus `-hf` Transformers-native variants. All are Apache-2.0 and not gated. Official docs show only CUDA. CPU fallback is mentioned by a maintainer, and MPS is not documented. Weights are about 4.7 GB (ASR) and 1.84 GB (aligner) in BF16.

### Cited Findings
- HF IDs: `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ForcedAligner-0.6B` (released 2026-01-29), and `-hf` variants (2026-06-26, "Native Transformers support … torch.compile"). All are tagged `license:apache-2.0` with `gated: false`. Repo QwenLM/Qwen3-ASR is Apache-2.0 — [HF ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [HF aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), [HF aligner-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf), [GitHub](https://github.com/QwenLM/Qwen3-ASR)
- Actual parameter counts (HF safetensors metadata): ASR-1.7B has 2,349,217,408 BF16 params (4.70 GB in 2 shards). ForcedAligner-0.6B has 917,728,896 BF16 params (1.84 GB) — [HF API ASR](https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B), [HF API aligner](https://huggingface.co/api/models/Qwen/Qwen3-ForcedAligner-0.6B)
- The aligner is NAR (non-autoregressive) and timestamps "arbitrary units within up to 5 minutes of speech in 11 languages" (zh, en, yue, fr, de, it, ja, ko, pt, ru, es) — [README](https://github.com/QwenLM/Qwen3-ASR)
- Every official example uses `dtype=torch.bfloat16, device_map="cuda:0"`. Optional FlashAttention2 works only in fp16/bf16. Streaming is vLLM-only and returns no timestamps — [README](https://github.com/QwenLM/Qwen3-ASR)
- CPU: issue #5 "allow cpu if no gpu detected" was closed with the reply "transformers backend support cpu if gpu not detect" — [issue 5](https://github.com/QwenLM/Qwen3-ASR/issues/5)
- Open issue #220: qwen-asr 0.0.6 pins transformers==4.57.6, but qwen3_asr needs 5.13+ — [issue 220](https://github.com/QwenLM/Qwen3-ASR/issues/220)
- Open issue #213: on sdpa/eager, the audio-encoder windowing is not applied, so clips longer than 8 s get full attention, with memory implications — [issue 213](https://github.com/QwenLM/Qwen3-ASR/issues/213)
- Apple Silicon (community conversions, not official): CoreML FP16 aligner, MLX 4-bit aligner, `mlx-qwen3-asr` — [CoreML](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16), [MLX 4bit](https://hf-cn0.pages.dev/Alkd/Qwen3-ForcedAligner-0.6B-4bit), [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/)

### Inferences
- Approximate VRAM from weight size: aligner ≥ ~2 GB and ASR-1.7B ≥ ~5 GB in BF16, plus activations. Together, ~7-8 GB is a plausible floor. This is an estimate, not measured.

### Gaps
- PyTorch MPS for either model: no official statement and no issue found. Must be measured locally.
- Official VRAM figures: not published in the README.
