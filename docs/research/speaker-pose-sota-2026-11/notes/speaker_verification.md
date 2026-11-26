# Open-source speaker verification / target-speaker enrollment front ends (state as of 2026-11-25)

Scope: enroll one target voice from a few utterances, then score each streaming speech segment (often 1-3 s, sometimes shorter) against it, in noisy streamer-style Korean audio. Every number below carries its source. Items marked **[unverified]** come from background knowledge that I could not confirm with a fetched source in this session.

GitHub star counts were read with `gh api repos/<owner>/<repo>` on 2026-11-25 (the "pushed" date is the last push to any branch):

| Repo | Stars | Code license | Last push |
|---|---|---|---|
| NVIDIA NeMo (now redirects to `NVIDIA-NeMo/Speech`) | 18,787 | Apache-2.0 | 2026-11-25 |
| k2-fsa/sherpa-onnx | 16,215 | Apache-2.0 | 2026-11-24 |
| speechbrain/speechbrain | 11,951 | Apache-2.0 | 2026-10-14 |
| pyannote/pyannote-audio | 10,841 | MIT | 2026-11-24 |
| modelscope/3D-Speaker | 3,226 | Apache-2.0 | 2025-12-08 (about 1 year without a push) |
| wenet-e2e/wespeaker | 1,477 | Apache-2.0 | 2026-09-22 |
| IDRnD/redimnet | 226 | MIT | 2026-08-28 |
| PalabraAI/redimnet2 | 125 | MIT | 2026-08-28 |
| ZXHY-82/w2v-BERT-2.0_SV | 61 | NOASSERTION (no license detected) | 2026-03-28 |
| Padawan101/ZP-KWS | 10 | none detected | 2026-06-17 |
| microsoft/UniSpeech (WavLM SV) | 492 | NOASSERTION | 2024-04-05, **archived** |

## Q1. Which models are SOTA on VoxCeleb1-O/E/H (EER, minDCF) and on CN-Celeb or out-of-domain sets?

### Takeaway
The best published numbers come from large SSL-backbone systems: w2v-BERT 2.0 with a LoRA/MFA head reaches 0.12 / 0.27 / 0.55 % EER on Vox1-O/E/H with 580M+ parameters. Fbank CNNs trained with extra VoxBlink2 data come close: ResNet293 at 0.17 % Vox1-O, and ReDimNet2-B6 at 0.23 / 0.35 / 0.67 % with only 12.3M params. Among models trained on VoxCeleb2 only, ReDimNet2-B6 (0.29 / 0.52 / 0.99) and WeSpeaker ResNet293-LM (0.425 / 0.641 / 1.146 with AS-Norm+QMF) are the strongest light-to-mid models. On CN-Celeb, the best number is ReDimNet2-B6 trained with CN-Celeb2 data (3.59 % EER).

### Cited Findings

**SSL-backbone systems (heavy)**
- w2v-BERT 2.0 + LoRA_Adapter_MFA (580M + 6.2M params), trained on VoxCeleb2 + VoxBlink2, with LMFT and calibration: Vox1-O 0.12 % / minDCF 0.025, Vox1-E 0.27 / 0.028, Vox1-H 0.55 / 0.051. The authors call this SOTA. Without LMFT and calibration it scores 0.23 / 0.38 / 0.81. On the CN-Celeb test set, trained on CN-Celeb1&2 only, it scores 4.67 % / 0.297. Paper submitted 5 Oct 2025, v3 dated 9 Mar 2026. — [arXiv 2510.04213 v2 HTML](https://arxiv.org/html/2510.04213v2); [arXiv abs](https://arxiv.org/abs/2510.04213)
- Same paper: knowledge-distillation-guided structured pruning at about 80 % sparsity shrinks the backbone to 124M + 6.2M params and scores 0.18 % Vox1-O EER after LMFT. The paper reports no RTF or latency. — [arXiv 2510.04213 v2](https://arxiv.org/html/2510.04213v2)
- Comparison numbers quoted in the same table (EER % / minDCF):
  - WavLM-Large + ECAPA (317 + 8.8M): 0.38 / 0.48 / 0.99
  - WavLM-Large + CA-MHFA: 0.42 / 0.48 / 0.96
  - WavLM-Large + LAP+ASTP: 0.37 / 0.50 / 1.01
  - "NeMo Large" + MFA (131M): 0.43 / 0.66 / 1.35
  - ReDimNet-B6 (15.0M): 0.37 / 0.53 / 1.00 (minDCF 0.030 / 0.051 / 0.097)
  - ERes2NetV2 (17.8M): 0.61 / 0.76 / 1.45, CN-Celeb 6.04
  - ResNet221 (23.8M): 0.51 / 0.68 / 1.21, CN-Celeb 5.66
  - ResNet293 with VoxBlink2 data (98.9M as listed): 0.17 / 0.37 / 0.68
  - CAM++ (7.2M): 0.73 / 0.89 / 1.76, CN-Celeb 6.78
  - ECAPA-TDNN C=1024 (14.7M): 0.87 / 1.12 / 2.12

  — [arXiv 2510.04213 v2](https://arxiv.org/html/2510.04213v2)
- WeSpeaker's own w2v-BERT 2.0 recipe (frozen LoRA, then joint fine-tuning, then large-margin fine-tuning): 0.250 / 0.398 / 0.838. WeSpeaker WavLM + ECAPA c512 (best variant): 0.415 / 0.551 / 1.118. — [WeSpeaker voxceleb/v2 README](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)
- ResNet293 trained on VoxBlink2: 0.170 % EER on Vox1-O, described as a new SOTA in the VoxBlink2 paper (2024, older result). — [VoxBlink2 arXiv 2407.11510](https://arxiv.org/html/2407.11510v1) (via search snippet)

**Fbank CNN/TDNN systems, WeSpeaker table (VoxCeleb2-dev training, EER % on O / E / H; no minDCF on that page)**

Source for all rows: [WeSpeaker voxceleb/v2 README](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)

| Model | Params | Config | O / E / H |
|---|---|---|---|
| ECAPA c512 | 6.19M | plain | 1.069 / 1.209 / 2.310 |
| ECAPA c512 | 6.19M | LM + AS-Norm | 0.782 / 1.005 / 1.824 |
| ECAPA c1024 | 14.65M | LM + AS-Norm + QMF | 0.707 / 0.894 / 1.615 |
| ResNet34 | 6.63M | plain | 0.867 / 1.049 / 1.959 |
| ResNet34 | 6.63M | LM + AS-Norm + QMF | 0.659 / 0.821 / 1.437 |
| CAM++ | 7.18M | plain | 0.803 / 0.932 / 1.860 |
| CAM++ | 7.18M | LM + AS-Norm | 0.659 / 0.803 / 1.569 |
| Gemini DF-ResNet114 | 6.53M | LM + AS-Norm | 0.638 / 0.839 / 1.427 |
| ERes2Net34_Base | 7.88M | LM + AS-Norm | 0.744 / 0.896 / 1.603 |
| ResNet221 | 23.79M | LM + AS-Norm | 0.505 / 0.676 / 1.213 |
| ResNet293 | 28.62M | LM + AS-Norm + QMF | 0.425 / 0.641 / 1.146 |
| SimAM-ResNet34 (VoxBlink2 pretrain) | 25.2M | LM + AS-Norm + QMF | 0.372 / 0.559 / 0.997 |
| SimAM-ResNet100 (VoxBlink2 pretrain) | 50.2M | LM + AS-Norm + QMF | 0.202 / 0.421 / 0.795 |
| ReDimNet-B2 baseline | 4.89M | plain | 0.782 / 0.907 / 1.667 |
| U³-xi ReDimNet-B2 | 5.46M | uncertainty score + LM + AS-Norm + QMF | 0.399 / 0.638 / 1.170 |
| ReDimNet2-B6 | n/a | official ckpt, reproduced in WeSpeaker | 0.276 / 0.518 / 1.003 |

**ReDimNet (IDRnD, Interspeech 2024; older but still competitive), vox2 ft_lm, EER % on O / E / H**

Source for all rows: [IDRnD/redimnet EVALUATION.md](https://github.com/IDRnD/redimnet/blob/master/EVALUATION.md)

| Model | Params | GMACs | O / E / H |
|---|---|---|---|
| b0 | 1.0M | 0.43 | 1.16 / 1.25 / 2.20 |
| b2 | 4.7M | 0.90 | 0.57 / 0.76 / 1.32 |
| b3 | 3.0M | 3.00 | 0.50 / 0.73 / 1.33 |
| b5 | 9.2M | 9.87 | 0.43 / 0.61 / 1.08 |
| b6 | 15.0M | 20.27 | 0.40 / 0.55 / 1.05 |

- ReDimNet model "M" trained on vb2+vox2+cnc (ft_mix): 0.835 / 0.745 / 1.284, CN-Celeb 7.474. — [IDRnD/redimnet EVALUATION.md](https://github.com/IDRnD/redimnet/blob/master/EVALUATION.md)

**ReDimNet2 (PalabraAI; arXiv 2603.11841, Interspeech 2026, newest family found), EER % on O / E / H**

Source for all rows: [PalabraAI/redimnet2 README](https://github.com/PalabraAI/redimnet2)

| Model | Params | GMACs | Training data | O / E / H | CN-Celeb (multi-enroll) |
|---|---|---|---|---|---|
| B0 | 1.1M | 0.33 | VoxCeleb2-dev | 1.04 / 1.16 / 1.97 | n/a |
| B2 | 3.6M | 0.95 | VoxCeleb2-dev | 0.57 / 0.76 / 1.41 | n/a |
| B3 | 4.1M | 2.70 | VoxCeleb2-dev | 0.42 / 0.66 / 1.22 | n/a |
| B4 | 6.6M | 4.62 | VoxCeleb2-dev | 0.37 / 0.58 / 1.07 | n/a |
| B6 | 12.3M | 13.05 | VoxCeleb2-dev | 0.29 / 0.52 / 0.99 | n/a |
| B6 | 12.3M | 13.05 | vb2+vox2, lm | 0.23 / 0.35 / 0.67 | 7.49 |
| B6 | 12.3M | 13.05 | vb2+vox2+cnc2, lm | 0.26 / 0.41 / 0.79 | 3.59 |
| B3 | 4.1M | 2.70 | vb2+vox2+cnc2, lm | 0.51 / 0.60 / 1.14 | 4.78 |

- The README lists the VoxCeleb2 training pipeline and the VoxBlink2/CN-Celeb2 recipes as still on the roadmap. — [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)

**3D-Speaker (Alibaba Tongyi) table: VoxCeleb1-O EER, CN-Celeb EER, 3D-Speaker EER**

Source for all rows: [3D-Speaker README](https://github.com/modelscope/3D-Speaker)

| Model | Params | Vox1-O | CN-Celeb | 3D-Speaker |
|---|---|---|---|---|
| ECAPA-TDNN | 20.8M | 0.86 % | 8.01 % | 8.87 % |
| CAM++ | 7.2M | 0.65 % | 6.78 % | 7.75 % |
| ERes2NetV2 | 17.8M | 0.61 % | 6.14 % | 6.52 % |
| ERes2Net-large | 22.46M | 0.52 % | 6.17 % | 6.34 % |
| ResNet34 | 6.34M | 1.05 % | 6.92 % | 7.29 % |

- The README's newest news item is dated 2024.12. — [3D-Speaker README](https://github.com/modelscope/3D-Speaker)
- CAM++ paper (2023, older), EER / minDCF on Vox1-O, Vox1-E, Vox1-H, CN-Celeb:
  - CAM++ (7.18M): 0.73 / 0.0911, 0.89 / 0.0995, 1.76 / 0.1729, CN-Celeb 6.78 / 0.3830
  - ECAPA-TDNN (14.66M): 0.89 / 1.07 / 1.98, CN-Celeb 7.45
  - ResNet34 (6.70M): 0.97 / 1.03 / 1.88, CN-Celeb 6.97

  — [CAM++ arXiv 2303.00332](https://arxiv.org/abs/2303.00332) (PDF text extracted locally)

**Other single-model numbers**
- NVIDIA TitaNet-Large (about 23M params): 0.66 % EER on the cleaned VoxCeleb1 trials (NeMo 1.10.0, 2022-era result). Trained on VoxCeleb1/2, Fisher, Switchboard, LibriSpeech, and SRE 2004-2010. The model card recommends no newer NeMo SV model. — [HF nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- SpeechBrain ECAPA (`spkrec-ecapa-voxceleb`): 0.80 % EER on Vox1-test cleaned (release dated 05-03-21, older). Trained on VoxCeleb1+2. — [HF speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- pyannote `wespeaker-voxceleb-resnet34-LM` is a wrapper around WeSpeaker ResNet34-LM. Its card gives no benchmark numbers. It points to "Legacy" (pyannote.audio 3.4.x) and "Latest" (4.x) collections, and to the paid pyannoteAI for "better and faster options". — [HF pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

### Inferences
- For a light front end, ReDimNet2-B2/B3 (3.6-4.1M params, about 1-2.7 GMACs) matches or beats CAM++ (7.2M) and ResNet34 on VoxCeleb, based on the tables above. Note that not every row uses the same scoring setup: the ReDimNet2 numbers have no AS-Norm noted.
- An "SOTA" gap of 0.1-0.3 % EER on VoxCeleb (clean, English celebrity interviews, mostly utterances longer than 4 s) probably says little about 1-3 s Korean game-stream segments. CN-Celeb (multi-genre, including entertainment and live broadcast) is the more relevant OOD proxy. There, every model sits at 3.6-8 % EER, so domain mismatch dominates the architecture choice.
- Models trained with VoxBlink2 (multilingual YouTube) or CN-Celeb data improve CN-Celeb sharply: ReDimNet2-B6 goes from 7.49 to 3.59 % when cnc2 is added. A Korean-heavy fine-tune set would likely matter more than the backbone choice.

### Gaps
- minDCF is missing for most WeSpeaker, ReDimNet2, and 3D-Speaker rows: their pages publish only EER.
- I could not confirm 2025-2026 VoxSRC-style challenge results. VoxSRC's last edition was 2023 **[unverified]**.
- I did not find newer NeMo SV checkpoints than TitaNet (for example, a 2025-2026 NeMo SV release). The "NeMo Large + MFA, 131M" row in arXiv 2510.04213 cites a research system, not a confirmed public checkpoint.
- I did not check ECAPA2 (arXiv 2401.08342, HF weights). The search surfaced it, but I collected no numbers.

## Q2. Which are robust on short utterances (<2 s) and cross-lingual / Korean data? Any Korean SV benchmarks?

### Takeaway
EER roughly triples from full-length to 1 s trials, even for good models: ECAPA goes from 1.03 % to 3.04 % on Vox1-O. On harder trial lists it reaches double digits at 1 s (Vox1-H about 14-20 %). No public Korean SV benchmark with published EERs turned up. The only Korean-specific study found reports results qualitatively in its abstract. Plan to measure on in-house Korean streamer data.

### Cited Findings
- arXiv 2509.19721 (Sep 2025, revised Jan 2026), Vox1-O EER without AS-Norm, full / 2 s / 1.5 s / 1 s:
  - ECAPA-TDNN: 1.03 / 1.76 / 2.12 / 3.04
  - ECAPA + multi-resolution encoder: 1.01 / 1.32 / 1.60 / 2.33
  - Proposed SSL-PTM + FBank + MRE: 0.78 / 1.24 / 1.46 / 2.29

  The paper gives no code link. — [arXiv 2509.19721v1](https://arxiv.org/html/2509.19721v1)
- ERes2NetV2 was designed for short-duration trials. Vox1-O EER is 1.48 % at 2 s, 0.98 % at 3 s, and 0.61 % at full length (2024, older). — [arXiv 2406.02167](https://arxiv.org/abs/2406.02167)
- "Beyond Short Segments" (arXiv 2609.25007) reports that a baseline reaches 20.449 % EER on Vox1-H at 1 s, and that its method lowers this to 14.571 %. On Vox1-E at 1 s, EER drops from 18.059 % to 8.511 %. These come from a search snippet only; I did not fetch the full table. — [arXiv 2609.25007](https://arxiv.org/pdf/2609.25007)
- Other short-utterance work found but not read in full:
  - DAME (Duration-Aware Matryoshka Embedding): nested sub-embeddings aligned to duration, model-agnostic. — [arXiv 2601.13999](https://www.arxiv.org/pdf/2601.13999)
  - FineSpeaker: evaluated on 1 s and 0.5 s segments. — [arXiv 2611.15288](https://arxiv.org/pdf/2611.15288)
- WeSpeaker notes that its "LM" (large-margin fine-tuned) checkpoints help mainly on audio longer than about 3 s. — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- Korean: Yoon & Park (Pusan National Univ.) trained ECAPA-TDNN on VoxCeleb combined with Korean speech, in the Korean Journal of Applied Statistics 37(2), pp. 209-224, April 2024. The abstract says adding Korean data improved performance "for all test sets". The page names no datasets and gives no EER numbers. — [KoreaScience JAKO202414143303118](https://koreascience.or.kr/article/JAKO202414143303118.pub)
- Multilingual: TidyVoice (arXiv 2601.16358), a cleaned multilingual Common Voice SV set. Results (EER % / minDCF):
  - ResNet-293 pretrained on VoxBlink2: Tidy-M 2.74 / 0.82, Tidy-X 3.89 / 0.93
  - After fine-tuning on Tidy-M: Tidy-M 0.35 / 0.19, Tidy-X 1.65 / 0.71

  The hardest trial type is target speaker in a different language versus non-target in the same language. The extracted text does not say whether Korean is included. The paper says the data, trials, and models are released under WeSpeaker examples. — [arXiv 2601.16358](https://arxiv.org/html/2601.16358)
- VoxBlink2: about 9.9M utterances from 111,284 YouTube users across multiple languages (2024). — [VoxBlink2 arXiv 2407.11510](https://arxiv.org/html/2407.11510v1) (via search snippet)
- Multilingual checkpoints available off the shelf:
  - WeSpeaker SimAM-ResNet34/100 trained on VoxBlink2 (ModelScope only, with ONNX). — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
  - ReDimNet/ReDimNet2 vb2+vox2(+cnc) variants. — [IDRnD/redimnet](https://github.com/IDRnD/redimnet); [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)
  - 3D-Speaker CAM++/ERes2Net trained on 200k Chinese speakers ("zh-cn common"). — [3D-Speaker README](https://github.com/modelscope/3D-Speaker)

### Inferences
- For 1-3 s windows, expect an EER several times the full-utterance figure. Enrolling with multiple utterances (averaging several 3-10 s clean enrollment clips), adding temporal smoothing or hysteresis across consecutive segments, and calibrating thresholds on in-domain data will probably matter more than picking a 0.1 % better model.
- Prefer checkpoints trained on multilingual data (VoxBlink2) or on Chinese multi-genre data (CN-Celeb, 3D-Speaker 200k) over VoxCeleb-only English checkpoints for Korean. Language-mismatch effects are documented (TidyVoice), but none of the sources quantify them for Korean.
- Candidate Korean evaluation and fine-tuning data includes AIHub Korean speaker-recognition corpora **[unverified: not checked in this session]**. The team should build a small in-domain trial list: the streamer plus other voices, game SFX, and BGM, cut into 1-3 s.

### Gaps
- I found no public Korean SV benchmark with reported EERs, and no per-language Korean result in TidyVoice or VoxBlink2.
- I found no published robustness numbers for game-sound or BGM interference specifically. The closest proxy is CN-Celeb's entertainment and live-broadcast genres **[genre composition unverified in this session]**.
- I found no apples-to-apples 1 s comparison of ReDimNet2, CAM++, ResNet293, and w2v-BERT.

## Q3. Which toolkits package them for easy inference, and what licenses apply to the weights?

### Takeaway
The easiest streaming-friendly path is sherpa-onnx (16.2k stars, Apache-2.0, actively maintained). It ships ready ONNX exports of 3D-Speaker CAM++/ERes2Net/ERes2NetV2, WeSpeaker ResNet34-293/CAM++, and NeMo TitaNet. WeSpeaker gives the widest range of SOTA checkpoints, but its weights "follow the dataset license". ReDimNet/ReDimNet2 are MIT-licensed code loaded via torch.hub, with small repos. The license on VoxBlink2-trained weights needs checking before commercial use.

### Cited Findings
- **sherpa-onnx `speaker-recongition-models` release assets (file sizes in MB):**
  - 3D-Speaker exports:
    - `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`: 26
    - `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`: 26
    - `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`: 28
    - `3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx`: 68
    - `3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx`: 110
  - WeSpeaker exports:
    - `wespeaker_en_voxceleb_CAM++(_LM).onnx`: 27
    - `wespeaker_en_voxceleb_resnet34(_LM).onnx`: 25
    - `wespeaker_en_voxceleb_resnet221_LM.onnx`: 90
    - `wespeaker_en_voxceleb_resnet293_LM.onnx`: 109
    - `wespeaker_zh_cnceleb_resnet34(_LM).onnx`: 25
  - NeMo exports:
    - `nemo_en_titanet_small.onnx`: 38
    - `nemo_en_titanet_large.onnx`: 96
    - `nemo_en_speakerverification_speakernet.onnx`: 22

  The release page says each model keeps its upstream license. — `gh api repos/k2-fsa/sherpa-onnx/releases/tags/speaker-recongition-models`; [release page](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
- **WeSpeaker** (1.5k stars, Apache-2.0 code):
  - Weights: "The pretrained model in WeNet follows the license of it's corresponding dataset." VoxCeleb is listed as CC BY 4.0.
  - Formats: `.pt` and `.onnx`, with ONNX inference through `wespeaker/bin/infer_onnx.py`.
  - The w2v-BERT 2.0 and ReDimNet2-B6 checkpoints have no ONNX export.
  - The VoxBlink2 SimAM models are hosted on ModelScope only.

  — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- **3D-Speaker** (3.2k stars, Apache-2.0 code):
  - Includes an `runtime/onnxruntime` directory (added 2024.4).
  - Weights are on ModelScope, for example `iic/speech_campplus_sv_zh-cn_16k-common` and `iic/speech_eres2netv2_sv_zh-cn_16k-common`.
  - The repo has had no push since 2025-12-08.

  — [3D-Speaker README](https://github.com/modelscope/3D-Speaker); gh api
- **ReDimNet** (IDRnD, 226 stars, MIT):
  - Loads via `torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', model_name=..., train_type=..., dataset=...)`.
  - Needs PyTorch 2.0 or later; outputs a 192-dim embedding from 16 kHz input.

  — [IDRnD/redimnet](https://github.com/IDRnD/redimnet)
- **ReDimNet2** (PalabraAI, 125 stars, MIT): loads via `torch.hub.load("PalabraAI/redimnet2","redimnet2",model_name="b0".."b6",train_type="lm")`; weights are tagged v1.0.0. — [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)
- **SpeechBrain** (12.0k stars): `spkrec-ecapa-voxceleb` is licensed Apache-2.0 on its HF card. Use `SpeakerRecognition.from_hparams(...).verify_files()`. — [HF speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **NeMo** (18.8k stars): TitaNet-Large weights are CC-BY-4.0. Load with `EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")`. — [HF titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- **pyannote** (10.8k stars, MIT): `pyannote/wespeaker-voxceleb-resnet34-LM` is listed as cc-by-4.0 and needs pyannote.audio 3.1 or later. Its gating status is not stated on the card. — [HF card](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- **w2v-BERT 2.0 SV** code: ZXHY-82/w2v-BERT-2.0_SV (61 stars, no license detected by GitHub). — gh api; [arXiv 2510.04213](https://arxiv.org/abs/2510.04213)
- **Kiwano**: I could not locate a public repo. `Orange-OpenSource/kiwano` returns 404 through gh api.

### Inferences
- License flags for commercial use:
  - Weights trained on VoxBlink2 (WeSpeaker SimAM models, ReDimNet/ReDimNet2 "vb2" variants, ResNet293-VB2) may inherit VoxBlink2's data license. I recall that license as non-commercial (CC BY-NC-SA) **[unverified]**.
  - CN-Celeb-trained weights similarly follow CN-Celeb's license **[unverified]**.
  - The code license (MIT/Apache) does not settle the weight license, because WeSpeaker explicitly ties weights to the dataset license.
  - The w2v-BERT-2.0_SV repo has no detected license, so treat it as all-rights-reserved until clarified.
  - The safest permissive combinations found: SpeechBrain ECAPA (Apache-2.0 card), TitaNet (CC-BY-4.0), and WeSpeaker VoxCeleb models (CC-BY-4.0 per WeSpeaker).
  - The license of 3D-Speaker's ModelScope weights was not checked on each model card **[unverified]**.
- For a GAMST-style CPU streaming front end, sherpa-onnx gives one runtime with C/C++/Python bindings plus VAD. Swapping models becomes a file change.

### Gaps
- I did not read the ModelScope model-card licenses for 3D-Speaker weights.
- I did not confirm the VoxBlink2 and CN-Celeb dataset licenses from primary pages.
- I did not confirm Kiwano's status or location.

## Q4. Target-speaker keyword spotting: are there open-source systems that detect "keyword X spoken by enrolled speaker Y"?

### Takeaway
Yes, but only as small research code. The best documented one is ZP-KWS (Sep 2026, 10 stars). It fuses an open-vocabulary KWS branch with a 0.9M EfficientTDNN speaker branch through a multiplicative AND, enrolls from one utterance, and works with English benchmarks only. Other personalized-KWS papers (PCOV-KWS, ProKWS, PK-MTL) have no confirmed code. In practice, a cascade (VAD, then KWS/ASR, then an SV gate on the same segment) built from mature components is the usable route.

### Cited Findings
- **ZP-KWS**, "Personalized Keyword Spotting for User-Defined Keywords Leveraging Text-Independent Speaker Verification" (Hu, Huang, Wang, Lee, Chen; NTNU / E.SUN / United Link), arXiv 2606.20106 v2, 17 Sep 2026:
  - Architecture: a KWS branch (0.65M params, phoneme-supervised, G2P text encoder) plus an EfficientTDNN-Small SV branch (about 0.9M params, 192-dim, pretrained on VoxCeleb2 and fine-tuned with GE2E). The two are combined as p_final = p_utt · p_spk.
  - Modes: conventional, target-biased, and target-only, with no retraining needed.
  - Enrollment: a single enrollment utterance, which may contain a different keyword (text-independent).
  - Code: https://github.com/Padawan101/ZP-KWS

  — [arXiv 2606.20106](https://arxiv.org/html/2606.20106)
- ZP-KWS target-only results, FRR at 1 % FAR / EER:

  | Dataset | ZP-KWS | PK-MTL | PhonMatchNet |
  |---|---|---|---|
  | LibriPhrase-Easy | 29.47 / 8.16 | 72.79 / 17.74 | 97.00 / 25.16 |
  | LibriPhrase-Hard | 78.20 / 13.87 | n/a | n/a |
  | Qualcomm | 33.12 / 8.81 | n/a | n/a |
  | Google Speech Commands | 49.76 / 11.05 | n/a | n/a |

  The authors list calibration under noisy and mismatched conditions as future work. Latency is not measured. — [arXiv 2606.20106](https://arxiv.org/html/2606.20106)
- ZP-KWS repo: 10 stars, no license detected, last push 2026-06-17. — `gh api repos/Padawan101/ZP-KWS`
- **PCOV-KWS** (arXiv 2603.18023): a multi-task KWS+SV network (TDResNeXt; 211k-376k params) trained on English MSWC. It reports 3.85 % EER for personalized open-vocabulary KWS. I found no code link. — [HF papers 2603.18023](https://huggingface.co/papers/2603.18023)
- **ProKWS** (arXiv 2603.18024, 5 Mar 2026): a phoneme stream plus a prosody stream for personalized KWS. The abstract gives no numbers and no code link. — [arXiv 2603.18024](https://arxiv.org/abs/2603.18024)
- **PK-MTL** (Qualcomm, 2022, older): the foundational joint KWS+SV paper that defines the target-biased and target-only settings. Code is not confirmed. — [arXiv 2206.13708](https://arxiv.org/pdf/2206.13708)
- Google holds patents and patent applications on personalized and target-speaker KWS. These are not open source. — [US12347439B2](https://patents.google.com/patent/US12347439); [US 2025/0078840](https://patents.justia.com/patent/20250078840)

### Inferences
- No production-grade, Korean-capable, open-source "keyword X by speaker Y" system turned up. ZP-KWS's late-fusion design (an independent veto from SV) is the same pattern a cascade gives. So a cascade with a stronger SV model (CAM++/ReDimNet2) and a Korean KWS or ASR (for example, sherpa-onnx Korean KWS/ASR models **[availability unverified]**) is likely more accurate and more maintainable than adopting ZP-KWS directly.
- The fusion rule p_kws · p_spk, with grid-searched weights and biases, is a useful, documented baseline for the gating logic.

### Gaps
- I did not search for target-speaker ASR systems (for example, TS-ASR in NeMo/ESPnet, or speaker-conditioned Whisper variants), so their open-source status is unchecked.
- I found no Korean personalized-KWS benchmark.

## Q5. Trade-off: best accuracy choice vs best light streaming choice

### Takeaway
For best accuracy without regard to cost, use w2v-BERT 2.0 SV (0.12 / 0.27 / 0.55 % EER, 580M+ params, GPU-class, unclear code license) or ResNet293-VB2. For a practical accuracy/cost balance, use ReDimNet2-B6 (12.3M, 13 GMACs, MIT, 0.29-0.23 % Vox1-O). For the best light streaming front end, use CAM++ via sherpa-onnx (7.2M, 1.72 GFLOPs, CPU RTF 0.013 single-thread, about 26 MB ONNX, a zh-en variant available) or ReDimNet2-B2/B3 (3.6-4.1M, about 1-2.7 GMACs, but no ONNX/sherpa package yet).

### Cited Findings
- CAM++ paper, Table 3, measured on CPU, single thread:

  | Model | Params | FLOPs | RTF |
  |---|---|---|---|
  | CAM++ | 7.18M | 1.72G | 0.013 |
  | ECAPA-TDNN | 14.66M | 3.96G | 0.033 |
  | ResNet34 | 6.70M | 6.84G | 0.032 |

  The paper says CAM++ achieves "more than twice the inference speed" of both baselines (2023 result; the CPU model is not named in the extracted text). — [CAM++ arXiv 2303.00332](https://arxiv.org/abs/2303.00332)
- ReDimNet2 compute: B0 1.1M / 0.33 GMACs, B2 3.6M / 0.95, B3 4.1M / 2.70, B6 12.3M / 13.05. — [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)
- w2v-BERT 2.0 SV: 580M + 6.2M params. The pruned version is 124M + 6.2M at 0.18 % Vox1-O. No RTF is reported. — [arXiv 2510.04213](https://arxiv.org/html/2510.04213v2)
- ONNX sizes in sherpa-onnx: CAM++ 26-28 MB, ResNet34 25 MB, ERes2NetV2 68 MB, ResNet293-LM 109 MB, TitaNet-Large 96 MB. — sherpa-onnx release assets via gh api

### Ranked shortlist (synthesis; every number is traceable to the sections above)

| Rank | Model | Params / compute | Best reported EER (source above) | Packaging / license | Repo stars, maintenance | Role |
|---|---|---|---|---|---|---|
| 1 (light default) | CAM++ (3D-Speaker; zh-cn 200k or zh_en "advanced" ckpt) | 7.18M, 1.72 GFLOPs, CPU RTF 0.013 | Vox1-O 0.65 % (3D-Speaker), CN-Celeb 6.78 % | sherpa-onnx ONNX 26 MB; Apache-2.0 code, weight license per ModelScope card [unverified] | 3.2k stars; no push since Dec 2025 (stale); sherpa-onnx 16.2k stars, active | Streaming CPU front end; Chinese-trained, which may transfer better to East Asian speech than VoxCeleb-only [inference] |
| 2 (light, higher accuracy) | ReDimNet2-B2/B3 (or B6) | 3.6-4.1M, 0.95-2.7 GMACs (B6: 12.3M, 13 GMACs) | B3 vox2: 0.42 / 0.66 / 1.22; B6 vb2+vox2+cnc2: 0.26 / 0.41 / 0.79, CN-Celeb 3.59 | torch.hub, MIT code; vb2 weights may carry the VoxBlink2 license [unverified]; ONNX export not provided | 125 stars, active 2026 (Interspeech 2026) | Best accuracy per FLOP; needs own ONNX export |
| 3 (mid, permissive) | WeSpeaker ResNet34-LM / ResNet221-LM / ResNet293-LM | 6.6M / 23.8M / 28.6M | 0.659 / 0.821 / 1.437 (R34); 0.425 / 0.641 / 1.146 (R293) | ONNX in WeSpeaker and sherpa-onnx; weights CC-BY-4.0 (VoxCeleb) | 1.5k stars, active Sep 2026 | Commercially clean baseline; English-only training |
| 4 (multilingual) | WeSpeaker SimAM-ResNet34/100 (VoxBlink2) | 25.2M / 50.2M | 0.372 / 0.559 / 0.997; 0.202 / 0.421 / 0.795 | ModelScope ONNX; VoxBlink2 license [unverified] | 1.5k stars | Multilingual YouTube data, closest to streamer domain [inference] |
| 5 (legacy, easy) | SpeechBrain ECAPA / NeMo TitaNet-L | about 20M / about 23M | 0.80 % / 0.66 % Vox1-O (2021-22) | Apache-2.0 / CC-BY-4.0; TitaNet in sherpa-onnx | 12k / 18.8k stars, active toolkits | Familiar baselines, now outperformed |
| 6 (max accuracy, offline/GPU) | w2v-BERT 2.0 + LoRA-MFA | 580M + 6.2M (pruned 124M) | 0.12 / 0.27 / 0.55; CN-Celeb 4.67 | Research code, no license detected; WeSpeaker recipe (0.25 / 0.398 / 0.838) | 61 stars | Teacher / offline re-scoring only |
| n/a (TS-KWS) | ZP-KWS | 1.55M total | TO-KWS EER 8.16-13.87 % (English) | No license detected | 10 stars | Reference design only |

### Inferences
- Recommended pipeline for 1-3 s Korean streamer segments:
  1. VAD, then a CAM++ (sherpa-onnx) embedding per segment.
  2. Cosine score against a centroid of several enrollment clips.
  3. Threshold calibrated on an in-domain Korean dev set, with hysteresis across segments.
  4. Optionally re-score borderline segments with ReDimNet2-B6 or ResNet293.
- Every candidate should be benchmarked locally on real Korean stream audio before committing. None of the sources provides Korean or game-audio numbers.

### Gaps
- I found no apples-to-apples CPU RTF numbers for ReDimNet2, ResNet293, or TitaNet on the same hardware.
- I found no published streaming-latency measurements for any SV model on 1 s windows.
- I did not run a measured benchmark in this session. All numbers come from the cited sources.
