# Target Speaker Extraction (TSE), enrollment-conditioned: SOTA as of Nov 2026

Stars were fetched with `gh api repos/...` on 2026-11-25. "NOASSERTION" means GitHub could not detect the license, so the README or LICENSE text was read instead.

## Q1. Best-accuracy SOTA (benchmarks)

### Takeaway
Discriminative: on standard SI-SDRi benchmarks, the strongest verified numbers among models with released weights come from USEF-TFGridNet. Generative: the flow-matching models (MeanFlow-TSE, AD-FlowTSE) and SoloSpeech lead on Libri2Mix-noisy SI-SDR and quality. Generative papers report SI-SDR, not SI-SDRi, so the two families cannot be compared directly.

### Cited Findings
- USEF-TFGridNet SI-SDRi: WSJ0-2mix 23.3, Libri2Mix (max, 16k) 16.8, WHAM! 17.6, WHAMR! 16.1 dB. It has 15.2M params, and the paper does not say whether it is causal (it uses BLSTM and full attention). — [USEF-TSE paper](https://arxiv.org/html/2409.02615)
- Baselines in the same table: X-TF-GridNet WSJ0-2mix 20.7 / WHAM! 15.3 / WHAMR! 14.6 (12.7M params). SpEx+ 17.4 / Libri2Mix 9.0 / WHAM! 13.1 / WHAMR! 10.9. CIENet-mDPTNet 21.4 / 16.6 / 15.7 (2.9M). The paper's prose and tables disagree slightly (23.2 vs 23.3). — [USEF-TSE paper](https://arxiv.org/html/2409.02615)
- USEF real-world test: DNS-2023 blind set, DNSMOS OVRL 3.12 headset / 2.96 speakerphone, vs SpEx+ 2.45 / 2.22. — [USEF-TSE paper](https://arxiv.org/html/2409.02615)
- Libri2Mix-noisy SI-SNR (SoloSpeech table): TD-SpeakerBeam 9.21, X-TF-GridNet 9.85, USEF-TSE 10.17, SSL-MHFA 10.60, SoloSpeech 11.12 (WER 0.15, SIM 0.96). — [SoloSpeech paper](https://arxiv.org/html/2505.19314)
- Out-of-domain results, models trained on Libri2Mix (SI-SNR): MUSAN noise, which includes music: SoloSpeech 10.40 vs USEF 8.96. WHAM!: 10.68 vs 9.15. DEMAND: 11.41 vs 9.70. Real recordings (MOS, 12 raters): CHiME-5 2.93 vs 2.10, RealSEP 2.70 vs 2.04. — [SoloSpeech paper](https://arxiv.org/html/2505.19314)
- SoloSpeech size: compressor 49.3M + extractor 200.8M. MeanFlow-TSE's table lists it at 589M total, 50 NFE, RTF 0.75. — [SoloSpeech](https://arxiv.org/html/2505.19314); [MeanFlow-TSE](https://arxiv.org/html/2512.18572)
- MeanFlow-TSE (Dec 2025) SI-SDR: Libri2Mix-noisy 12.85 (PESQ 2.21), clean 18.80 (PESQ 3.26). AD-FlowTSE scores 12.69 / 17.49. Both are about 359M params, 1 NFE, RTF 0.018. MeanFlow-TSE's speaker similarity is lower than AD-FlowTSE's on noisy (0.73 vs 0.87). — [MeanFlow-TSE](https://arxiv.org/html/2512.18572)
- EvoTSE (Apr 2026) reports SI-SDRi gains from 16.65 to 17.91 dB, but the snippet did not make the dataset clear. UNVERIFIED. — [EvoTSE](https://arxiv.org/pdf/2604.06810)
- Generative LM-based TSE (TSELM-L, AnyEnhance, LLaSE-G1, LauraTSE, GenTSE) is compared mostly on DNSMOS, NISQA, WER and speaker similarity, not SI-SDRi. The comparison tables reached me without column labels, so no numbers are quoted here. GenTSE (Dec 2025) claims to beat earlier LM-based TSE on Libri2Mix. — [LauraTSE](https://arxiv.org/html/2504.07402v3); [GenTSE](https://arxiv.org/html/2512.20978v1)
- LLaSE-G1 evaluated its TSE task on the DNS-2023 blind set, not on Libri2Mix. — [LLaSE-G1](https://arxiv.org/pdf/2503.00493)

### Inferences
- For keeping one person's voice against music or game sounds, the MUSAN and real-recording results are the most relevant evidence. There, the generative SoloSpeech beats discriminative USEF-TSE. Generative outputs can hallucinate, though, and a counting pipeline needs faithful timing, so this has to be measured on our own data.

### Gaps
- No public X-TF-GridNet code or weights were found (an ESPnet or Hugging Face search found nothing).
- AnyEnhance weights and license, and Metis-TSE weights, were not verified.
- No Libri2Mix-to-real gap figure is published (the REAL-TSE paper gives none).

## Q2. Best usable SOTA (code and weights public), with licenses

### Takeaway
Under a permissive license, the only ready-to-run audio-enrollment TSE options are WeSep (Apache-2.0, pretrained English BSRNN+ECAPA, CLI) and ClearerVoice SpEx+ (Apache-2.0 repo, 8 kHz WSJ0 checkpoint). The stronger models, USEF-TSE, SoloSpeech, TSELM and LauraTSE, release weights but are non-commercial (CC BY-NC or BY-NC-SA).

### Cited Findings
- **USEF-TSE** (75★): weights on HF `ZBang/USEF-TSE`, CC BY-NC 4.0. — [GitHub](https://github.com/ZBang/USEF-TSE)
- **SoloSpeech** (329★): code, checkpoints (HF `OpenSound/SoloSpeech-models`) and eval toolkit are all CC BY-NC 4.0. The README lists "improve efficiency" as future work. — [GitHub](https://github.com/WangHelin1997/SoloSpeech)
- **WeSep** (336★, Apache-2.0, pushed 2026-11-19): the v0.1 master branch has recipes for speaker, spatial, visual and text cues (BSRNN, SpEx+, DPCCN, TF-GridNet). Its pretrained models, Python API and ONNX export are listed as "coming soon". — [GitHub](https://github.com/wenet-e2e/wesep). The `wesep_2024` branch ships a pretrained English BSRNN+ECAPA model, runnable as `wesep --audio_file mix --audio_file2 enroll` with 16 kHz input, a VAD option and `load_model("english")`. — [wesep_2024 README](https://github.com/wenet-e2e/wesep/tree/wesep_2024)
- The WeSep Interspeech-2026 paper (cue-composable: enrollment, spatial, visual, text) says the toolkit "will be publicly available". — [arXiv 2607.27436](https://arxiv.org/abs/2607.27436)
- **ClearerVoice-Studio** (4,655★, Apache-2.0): its only audio-enrollment TSE is SpEx+ on WSJ0-2mix at 8 kHz, non-causal, at 17.1 SI-SDRi (16.9 in the paper). Weights: HF `alibabasglab/log_wsj0-2mix_speech_SpEx-plus_2spk`. The other TSE models are conditioned on face/lip, gesture or EEG, not voice. AV_MossFormer2_TSE_16K is visual-conditioned. — [ClearerVoice TSE](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction); [repo](https://github.com/modelscope/ClearerVoice-Studio)
- **SpeakerBeam** (BUT/NTT, 158★, last push 2021): released under an "evaluation-only" software license agreement. — [GitHub](https://github.com/BUTSpeechFIT/speakerbeam)
- **TSELM** (64★): CC BY-NC-SA 4.0. Built on WavLM-L, k-means and a HiFiGAN; TSELM-L checkpoint on HF `Beilong/TSELM`. — [GitHub](https://github.com/Beilong-Tang/TSELM)
- **LauraTSE** (code repo 40★): CC BY-NC-SA 4.0, checkpoint on HF `Beilong/LauraTSE`, uses the FunCodec `zh_en` 16k codec. — [GitHub](https://github.com/Beilong-Tang/lauraTSE_code)
- **LLaSE-G1** (107★): code is Apache-2.0, checkpoints on HF `ASLP-lab/LLaSE-G1`. Covers SE, TSE and more in one LLaMA-based model. The license of the weights was not checked. — [GitHub](https://github.com/Kevin-naticl/LLaSE-G1)
- **MeanFlow-TSE** (34★): code is MIT, but the paper lists no weights link. — [GitHub](https://github.com/rikishimizu/MeanFlow-TSE); [paper](https://arxiv.org/html/2512.18572)
- **Meta SAM-Audio** (3,765★): see Q4. It cannot take an enrollment clip.

### Inferences
- Commercial-safe path: start from the WeSep BSRNN+ECAPA checkpoint, which is Apache-2.0 and ready to run, then fine-tune it on our own mixes (game audio, music, Korean speech).
- Research or prototype path, non-commercial: USEF-TSE gives the best SI-SDRi. SoloSpeech gives the best robustness to music and real recordings.

### Gaps
- Benchmark numbers for the wesep_2024 English model were not retrieved (UNVERIFIED).
- RTF for USEF-TSE and TSELM was not found.

## Q3. Best streaming / causal SOTA

### Takeaway
I found no verified public checkpoint of a strong causal audio-enrollment TSE. The proven recipe for real-world streaming is a causal BSRNN or TF-GridNet trained in WeSep. That is how the REAL-TSE 2026 online track was won, under a ≤100 ms algorithmic latency budget.

### Cited Findings
- SLT 2026 REAL-TSE Challenge: real conversational recordings in Mandarin and English (AISHELL-4, AliMeeting, AMI, DipCo, CHiME6), 6,991 trials, 11.3 h. The online track has a ≤100 ms latency budget, and online systems landed at about 20–100 ms. Winners were mostly causal BSRNN or TF-GridNet with causal norms, unidirectional RNNs and limited look-ahead. The top three teams trained on real recordings. Gains came from data adaptation, pseudo-labels and loss design more than from architecture. The baselines (BSRNN_EMB, BSRNN_EMB_CAUSAL, ...) are trained on Libri2Mix-100 and described as weak; their code is public at `REAL-TSE/wesep-real-tse` (47★, no license file) with no weights link. — [REAL-TSE paper](https://arxiv.org/html/2607.15198v1)
- StarTSE (Apr 2026): streaming autoregressive LM TSE, about 89M params, best at 560 ms chunks. RTF 0.248 on RTX 4090 and 0.182 on L40S. Libri2Mix WER 0.152, though the paper also prints 0.138. No code or weights found. — [StarTSE](https://arxiv.org/html/2604.19635v2)
- Dual-mode training (APSIPA 2025): one model for streaming and offline use; causal TF-GridNet at about 20 ms latency. — [APSIPA 2025](https://apsipa.org/proceedings/2025/papers/APSIPA2025_P184.pdf)
- SpeakerBeam-SS (2024): causal Conv-TasNet with state-space modeling, RTF −78% vs a causal Conv-TasNet at the same quality. — [arXiv 2407.01857](https://arxiv.org/pdf/2407.01857)
- Falcon (Oct 2026): causal audio-visual TSE (needs video of the speaker), −91.9% params and 11.1× lower GPU latency vs causal AV-TFGridNet. — [arXiv 2610.32749](https://arxiv.org/abs/2610.32749)

### Inferences
- StarTSE's 560 ms chunks are too slow for per-frame counting. A causal BSRNN at about 20–100 ms fits. No streaming generative model has public weights.

### Gaps
- No published SI-SDRi for causal BSRNN in WeSep on Libri2Mix was retrieved. Code availability for SpeakerBeam-SS and dual-mode TF-GridNet was not checked.

## Q4. Meta SAM-Audio: prompt types; can it take an enrollment voice clip?

### Takeaway
No. SAM-Audio accepts text, visual (clicks or masks on video) and span prompts. A span is a set of time intervals inside the mixture itself, "without requiring external reference audio". It is not voice enrollment.

### Cited Findings
- Three prompt types: text, visual and span. The README mentions no reference-audio prompt. Sizes: small, base and large, plus "-tv" variants. Weights are gated on HF (`facebook/sam-audio-large`). Subjective "Speaker" scores: 3.12, 3.57 and 3.60 (scale not described). — [GitHub](https://github.com/facebookresearch/sam-audio)
- Span = S∈ℝ^{k×2}, the start and end times of the target in the mix. Span prompting "can be inherently ambiguous when multiple sound events occur simultaneously". Params: 500M / 1B / 3B, excluding the PE, T5 and DAC-VAE encoders. Inference uses a 16-step ODE solver plus beam-8 re-ranking. — [SAM Audio paper](https://arxiv.org/html/2512.18099v1)
- Text such as "male speech" cannot pick between two male speakers; visual masks can. Span-only prompts degrade on continuous sounds like speech. — [Meta blog / paper via search](https://ai.meta.com/blog/sam-audio/)
- License: the custom "SAM License" (updated 2025-11-19) grants a royalty-free right to use, modify and distribute the code and the weights. Full restrictions were not reviewed. — [LICENSE](https://github.com/facebookresearch/sam-audio)

### Inferences
- A possible workaround: run diarization plus speaker verification against the enrollment clip to find the spans where the target speaks, then feed those spans with a "speech" text prompt into SAM-Audio. Untested, slow, and the span docs warn about overlapping speech.

### Gaps
- SAM-Audio RTF (section 7.6) and its speaker-separation tables were in a truncated part of the paper and were not read.

## Q5. Multilingual / Korean evidence

### Takeaway
None of the reviewed TSE systems has a Korean evaluation. Training and evaluation data are English (LibriMix, WSJ0) or Mandarin plus English (REAL-TSE).

### Cited Findings
- REAL-TSE covers Mandarin and English only. — [REAL-TSE](https://arxiv.org/html/2607.15198v1)
- The WeSep pretrained model is labeled "English". — [wesep_2024 README](https://github.com/wenet-e2e/wesep/tree/wesep_2024)
- LauraTSE uses a `zh_en` codec. — [lauraTSE_code](https://github.com/Beilong-Tang/lauraTSE_code)

### Gaps
- Korean TSE performance needs an in-house eval set, for example Korean enrollment plus game audio mixes, scored with SI-SDRi and downstream keyword or ASR WER.
