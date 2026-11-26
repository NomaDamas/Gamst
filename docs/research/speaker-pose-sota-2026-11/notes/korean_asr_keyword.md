# Open-source Korean ASR with word timestamps and keyword/hotword support, for counting a spoken word in a live stream (as of 2026-11-25)

Scope note: about 22 tool calls. GitHub star counts were pulled live with `gh api repos/<repo>` on 2026-11-25 (stargazers_count, license). "Unverified" means I found no primary source in this session.

## Ranked shortlist and Korean accuracy numbers

### Takeaway
Measured on Korean, the strongest open model is **Qwen3-ASR-1.7B** (Apache-2.0): FLEURS-ko CER 2.57, Common Voice-ko CER 5.88. Pairing it with **Qwen3-ForcedAligner-0.6B** gives the best documented Korean word timestamps (37.2 ms AAS). For low-latency true streaming with Korean, the current candidates are **NVIDIA nemotron-3.5-asr-streaming-0.6b** (FLEURS-ko CER about 7.1–8.3 at 80 ms–1.12 s chunks) and the older, lighter **sherpa-onnx Korean streaming zipformer** (KsponSpeech CER about 10%). Only sherpa-onnx has hotword biasing that works on transducers, and even it has no documented Korean tokenization mode.

### Cited Findings

**Proposed ranking (my synthesis of the numbers below):**
1. Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B: best accuracy and best timestamps.
2. Qwen3-ASR-0.6B + ForcedAligner: the same stack at lower cost.
3. NVIDIA Nemotron 3.5 Transcribe 30B-A4B: high accuracy, offline only, heavy.
4. nemotron-3.5-asr-streaming-0.6b: true low-latency streaming with Korean.
5. faster-whisper / WhisperX with Whisper large-v3 or turbo: mature ecosystem, weaker Korean.
6. sherpa-onnx Korean streaming zipformer: lightest, on-device, lowest accuracy.
7. SenseVoice-Small / Fun-ASR-MLT-Nano: Korean supported, but no public Korean CER.

**Qwen3-ASR (Alibaba Qwen, released 2026-01-29)**
- Released 2026-01-29 as Qwen3-ASR 0.6B/1.7B plus Qwen3-ForcedAligner-0.6B. Native Transformers support was added 2026-06-26. The README lists no newer ASR version (no "Qwen3.5-ASR") — [GitHub QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- Covers 30 languages including Korean, plus 22 Chinese dialects (52 in total). Apache-2.0. 3,941 stars on 2026-11-25 (gh api) — [GitHub](https://github.com/QwenLM/Qwen3-ASR); [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)
- Korean CER (tech report, Tables A.2a/A.2b) — [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337):

  | Benchmark | 0.6B | 1.7B | Qwen3-ASR-Flash (closed API, reference only) |
  |---|---|---|---|
  | FLEURS-ko | 3.72 | 2.57 | 2.07 |
  | Common Voice-ko | 8.48 | 5.88 | 3.82 |
  | MLC-SLM-ko (conversational) | 10.31 | 8.61 | 8.09 |

- The report gives no per-language Korean scores for Whisper-large-v3, GPT-4o, Gemini or Fun-ASR-MLT-Nano. It gives only multi-language averages. On FLEURS (12 languages): Whisper-large-v3 5.27, Qwen3-ASR-1.7B 4.90, Qwen3-ASR-0.6B 7.57, Fun-ASR-MLT-Nano 10.03. On Common Voice (13 languages): Whisper-large-v3 10.77, Qwen3-ASR-1.7B 9.18 — [Tech report Table 5](https://arxiv.org/html/2601.21337)
- KsponSpeech is not evaluated in the report — [Tech report](https://arxiv.org/html/2601.21337)
- A community Korean fine-tune reports a **base** Qwen3-ASR-1.7B fleurs_ko CER of 0.0142 (fine-tune: 0.0138) and a Korean total CER of 0.0686 → 0.0537. This **conflicts** with the official 2.57, probably because of different text normalization. It is a third-party, unverified number — [TeamUNIVA/qwen3_asr_1.7b_ko_beta](https://huggingface.co/TeamUNIVA/qwen3_asr_1.7b_ko_beta)
- Speed: TTFT as low as 92 ms (0.6B, concurrency 1). At concurrency 128 the 0.6B transcribes 2000 s of audio in 1 s — [arXiv abstract](https://arxiv.org/abs/2601.21337)

**NVIDIA Nemotron 3.5 Transcribe (released 2026-10-28)**
- Size: 30B-A4B (card: 34B total, 4.3B active). License OpenMDW-1.1, commercial use allowed. Korean is in the "leading tier" (ASR, multi-speaker, code-switching). **FLEURS-ko CER 2.78**. Word timestamps supported (stream=false, n=1 only). Built for recorded audio, not live streaming. All of this comes from a secondary summary of the model card; I did not fetch the card itself — [braindetox.kr summary](https://braindetox.kr/en/posts/nvidia_nemotron_3_5_transcribe_korean_asr.html); [HF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- NeMo docs list it with 40 languages, word timestamps and "contextual" transcription — [NeMo ASR checkpoints](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html)

**NVIDIA nemotron-3.5-asr-streaming-0.6b (released 2026-06-04)**
- Korean (ko-KR) is "transcription-ready". OpenMDW-1.1, commercial use allowed. FLEURS-ko CER by chunk size (language given / auto-detect): 80 ms 7.59/8.31; 160 ms 7.70/8.18; 320 ms 7.27/7.81; 560 ms 7.18/7.49; 1.12 s 7.12/7.30. The card does not mention word timestamps or word boosting — [HF model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- No Canary or Parakeet model supports Korean. Canary covers EU4/EU25, and Parakeet is English plus single languages (ja, vi, da) — [NeMo ASR checkpoints](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html)
- NVIDIA/NeMo repo: 18,787 stars, Apache-2.0 (gh api, 2026-11-25)

**Whisper large-v3 / turbo (faster-whisper, WhisperX)**
- Zero-shot Whisper-large-v3 Korean CER of **5.31% FLEURS-ko** and **6.01% Common Voice 25-ko** comes from a third-party Korean model card. It is unverified, and the snippet did not make clear which card it came from — [BuzzASR/korean (likely source)](https://huggingface.co/BuzzASR/korean)
- Raw Whisper large-v3 on KsponSpeech eval_clean is about 12% CER, also from a self-reported third-party card and unverified — search snippet, same set of cards as above ([ghost613/whisper-large-v3-turbo-korean](https://huggingface.co/ghost613/whisper-large-v3-turbo-korean), [batiai/batisay-ko-large-GGUF](https://huggingface.co/batiai/batisay-ko-large-GGUF))
- Korean fine-tunes do much better. One turbo fine-tune reports Zeroth test CER about 2.06% ([o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2](https://huggingface.co/o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2)). A Whisper-medium KsponSpeech fine-tune reports 1.71% (search snippet, unverified).
- Stars on 2026-11-25 (gh api): openai/whisper 113,462 (MIT); SYSTRAN/faster-whisper 26,998 (MIT); m-bain/whisperX 25,658 (BSD-2-Clause); ggml-org/whisper.cpp 55,502 (MIT)

**sherpa-onnx streaming zipformer (k2-fsa)**
- `sherpa-onnx-streaming-zipformer-korean-2024-06-16` is Korean-only. Its int8 encoder is about 121 MB (fp32 about 279 MB) — [sherpa-onnx zipformer models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html). **Older (2024) result.**
- The underlying icefall model has 79.02M parameters and was trained on KsponSpeech plus MUSAN. Streaming CER at a 640 ms chunk with modified beam search: 9.91 eval_clean / 10.72 eval_other. At 320 ms: 10.13 / 10.88 — [icefall KsponSpeech streaming checkpoint](https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12)
- k2-fsa/sherpa-onnx: 16,215 stars, Apache-2.0 (gh api, 2026-11-25)

**FunASR / SenseVoice / Fun-ASR**
- The Fun-ASR-MLT-Nano card lists Korean among 31 languages, but says per-language MLT results "will be added" later. No Korean CER is published — [Fun-ASR-MLT-Nano-2512 card (mirror)](https://huggingface.co/aoiandroid/Fun-ASR-MLT-Nano-2512)
- The FunASR Korean-language benchmark page reports only an overall CER (SenseVoice-Small 8.92%, Fun-ASR-Nano 10.56%) and does not say which languages it covers. It is not a Korean-specific number — [FunASR benchmark (ko)](https://modelscope.github.io/FunASR/ko/benchmark.html)
- In the Qwen3 report's multilingual averages, Fun-ASR-MLT-Nano trails badly: FLEURS 10.03 vs 4.90 for Qwen3-ASR-1.7B — [Tech report Table 5](https://arxiv.org/html/2601.21337)
- Stars on 2026-11-25 (gh api): modelscope/FunASR 21,120 (MIT); FunAudioLLM/SenseVoice 9,812 (MIT); FunAudioLLM/Fun-ASR 1,616 (Apache-2.0)

**Other candidates**
- Moonshine (moonshine-ai/moonshine): 11,595 stars; GitHub reports the license as NOASSERTION (gh api). I could not verify Korean support or a Korean CER in this session (search returned irrelevant results).
- Kimi-Audio (MoonshotAI/Kimi-Audio): 4,752 stars; no license detected; last push 2025-06-21 (gh api). Step-Audio2 (stepfun-ai): 1,531 stars, Apache-2.0 (gh api). I found no Korean ASR numbers for either.

### Inferences
- **Best accuracy:** Qwen3-ASR-1.7B (FLEURS-ko 2.57) is slightly ahead of Nemotron 3.5 Transcribe (2.78) on the one benchmark where both report Korean. It is also roughly 20x smaller in active parameters and Apache-licensed. The 2.57 vs 2.78 comparison mixes the two vendors' own evaluations, so normalization may differ.
- **Best light streaming option:** nemotron-3.5-asr-streaming-0.6b. It is the only Korean-capable model found with official sub-second chunk streaming (80 ms–1.12 s) and a published Korean CER (about 7–8%). sherpa-onnx's Korean zipformer is lighter (79M parameters, ONNX, CPU and mobile) but sits around 10% CER on KsponSpeech, which is a different test set, so the two numbers are not directly comparable.
- A practical "near-streaming" design: run Qwen3-ASR on rolling windows of about 2 s (the report's streaming setting), then run ForcedAligner on finalized segments to get word times. Streaming itself cannot emit timestamps (see the next section).
- A direct Korean comparison of Whisper vs Qwen3 is not available from primary sources. The third-party zero-shot Whisper-large-v3 FLEURS-ko CER of about 5.3% would make it roughly 2x worse than Qwen3-ASR-1.7B, but that number is unverified.

### Gaps
- I found no official or primary Korean CER on KsponSpeech or Zeroth-Korean for Qwen3-ASR, Nemotron or zero-shot Whisper.
- Moonshine, Kimi-Audio and Step-Audio Korean support and CER remain unverified.
- I did not open the Hugging Face Open ASR Leaderboard (it is English-centric, and its multilingual track was not checked for Korean).
- I did not fetch the Nemotron 3.5 Transcribe HF card directly; its numbers come from a secondary summary.

## Streaming, latency and word-level timestamps

### Takeaway
Qwen3-ForcedAligner-0.6B is the only aligner with a published **Korean** timestamp error: 37.2 ms AAS on MFA-labeled data. Across languages it is about 3x better than NeMo NFA and WhisperX. WhisperX's Korean alignment uses a community wav2vec2 model with no published Korean timestamp error. Qwen3-ASR streaming does **not** emit timestamps, so an offline alignment pass is needed.

### Cited Findings
- Qwen3-ForcedAligner-0.6B supports 11 languages, including Korean (zh, en, yue, fr, de, it, ja, ko, pt, ru, es). It is an LLM-based non-autoregressive timestamp predictor — [Tech report §3](https://arxiv.org/html/2601.21337); [GitHub](https://github.com/QwenLM/Qwen3-ASR)
- Accumulated average shift (AAS, ms, lower is better), from Table 9 — [Tech report](https://arxiv.org/html/2601.21337):
  - Korean, MFA-labeled raw: **37.2 ms**. Korean, MFA-labeled concatenated 300 s: **42.2 ms**. The baselines have no Korean entries.
  - Average across languages, MFA-labeled raw: Qwen3-FA 42.9 / NFA 129.8 / WhisperX 133.2 / Monotonic-Aligner 161.1.
  - Average across languages, concatenated 300 s: Qwen3-FA 52.9 / NFA 246.7 / WhisperX 2708.4 / Monotonic-Aligner 1742.4. Baseline accuracy collapses on long audio.
  - Human-labeled average: Qwen3-FA 32.4 / NFA 101.2 / Monotonic-Aligner 141.3. Not broken down by language.
- Qwen3-ASR streaming runs only on the vLLM backend and supports neither batch inference nor timestamps — [GitHub README](https://github.com/QwenLM/Qwen3-ASR)
- Report streaming settings: 2 s chunks, 5-token fallback, last 4 chunks left unfixed. Streaming WER degrades (average offline → streaming): 1.7B 2.69 → 3.33, 0.6B 3.48 → 4.40. Evaluated only on LibriSpeech, FLEURS-en and FLEURS-zh, **not Korean** — [Tech report §4.5, Table 8](https://arxiv.org/html/2601.21337)
- WhisperX's default Korean alignment model is `kresnik/wav2vec2-large-xlsr-korean`. Characters missing from its dictionary are kept as wildcards, mapped to each frame's highest non-blank score — [whisperX alignment.py](https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py)
- nemotron-3.5-asr-streaming-0.6b chunk options: 80/160/320/560/1120 ms, set via `att_context_size` [56, {0,1,3,6,13}]. Word timestamps are not mentioned — [HF card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- Nemotron 3.5 Transcribe supports word timestamps only with stream=false and n=1. No Korean timestamp accuracy is reported — [braindetox.kr summary of card](https://braindetox.kr/en/posts/nvidia_nemotron_3_5_transcribe_korean_asr.html)
- sherpa-onnx Korean zipformer: native streaming with 320 or 640 ms chunks (see above). Transducer token timestamps are a general sherpa-onnx feature, but I did not verify a Korean timestamp accuracy number.

### Inferences
- For counting keyword occurrences with times, the Qwen3 pipeline (rolling-window ASR → ForcedAligner on finalized text) gives word times within about 40 ms on Korean, as measured on MFA-labeled data. That is far tighter than counting needs, so latency comes from ASR windowing, not from alignment.
- WhisperX on long or concatenated audio degrades sharply (2708 ms average AAS at 300 s in Qwen's test). If WhisperX is used, align short VAD segments.

### Gaps
- No Korean AAS is published for NFA, WhisperX/wav2vec2-ko, MMS, or Whisper's native word timestamps. The Qwen comparison table has "–" for the Korean baselines.
- No end-to-end streaming latency (emit time minus speech time) is published for Korean for any model.
- The FLEURS-ko CER of Qwen3-ASR in streaming mode is unknown.

## Contextual biasing / hotwords and open-vocabulary keyword spotting for Korean

### Takeaway
I found no SOTA open-source system that offers ready-to-use, text-defined (no-training) open-vocabulary keyword spotting for Korean. sherpa-onnx KWS models exist only for Chinese and English. Biasing support for Korean is weak or unmeasured everywhere: Qwen3-ASR and Nemotron accept a text "context" prompt, FunASR advertises hotwords, and sherpa-onnx hotwords officially document only cjkchar/bpe units. The practical route is to count keywords on the ASR transcript (with normalization or fuzzy matching), using a context prompt to bias toward slang.

### Cited Findings
- sherpa-onnx pretrained KWS models: `kws-zipformer-zh-en-3M-2025-12-20` (Chinese & English), `kws-zipformer-wenetspeech-3.3M-2024-01-01` (Chinese only), `kws-zipformer-gigaspeech-3.3M-2024-01-01` (English only). **There is no Korean KWS model** — [sherpa-onnx KWS pretrained models](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html)
- sherpa-onnx open-vocabulary KWS works "just like a tiny ASR system" that only decodes the given keywords. Each keyword has a boosting score and a trigger threshold, and keywords are tokenized with `sherpa-onnx-cli text2token`. No retraining is needed, but a KWS checkpoint in the target language is — [sherpa-onnx KWS](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)
- sherpa-onnx ASR hotwords: transducer models only, with `modified_beam_search` (not greedy). They are implemented as an Aho-Corasick automaton with partial-match boosting. Documented modeling units are cjkchar, bpe and cjkchar+bpe (examples in Chinese and English). The doc reports no aggregate WER gain, only anecdotal before-and-after examples — [sherpa-onnx hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- The sherpa-onnx Korean zipformer config dumps contain `hotwords_file` and `hotwords_score` fields, but the examples use greedy search. Korean hotword use is not documented — [zipformer models page](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)
- Qwen3-ASR training includes context-biasing data, and context tokens in the system prompt can customize output. The report gives no hotword metrics and no Korean numbers — [Tech report §2.2](https://arxiv.org/html/2601.21337). The README does not describe a hotword API — [GitHub](https://github.com/QwenLM/Qwen3-ASR)
- Nemotron 3.5 Transcribe has a "Context terms:" prompt for domain terms. The card warns that context can pull output toward wrong terms. The effect on Korean is not measured — [braindetox.kr summary of card](https://braindetox.kr/en/posts/nvidia_nemotron_3_5_transcribe_korean_asr.html)
- Fun-ASR advertises "customizable hotword recognition" for domain terms — [Fun-ASR Technical Report](https://arxiv.org/html/2509.12508v4). SenseVoice documentation mentions neural hotword enhancement — [SenseVoice docs](https://docsearch.algolia.com/mcp/docs/repo/funaudiollm/sensevoice). No Korean hotword numbers were found.
- A community SenseVoiceSmall fine-tune for a single Korean wake word ("제리") reports a baseline CER of 0.096 before fine-tuning. It is a single-speaker experiment, not a benchmark — [plzsay/sensevoice-ko-jerry](https://huggingface.co/plzsay/sensevoice-ko-jerry)
- Research on text-enrolled open-vocabulary KWS (much of it from Samsung and NCSOFT) is evaluated on English (LibriPhrase, WSJ), not Korean: MATE (arXiv 2601.14012, Jan 2026), adversarial deep metric learning (arXiv 2505.16735), CTC-aligned audio-text embedding for streaming OV-KWS (arXiv 2406.07923), PhonMatchNet (arXiv 2308.16511) — [2601.14012](https://www.arxiv.org/pdf/2601.14012); [2505.16735](https://arxiv.org/pdf/2505.16735v1); [2406.07923](https://arxiv.org/pdf/2406.07923); [2308.16511](https://arxiv.org/pdf/2308.16511v1.pdf)

### Inferences
- For owner-declared Korean keywords, including slang and neologisms, the most defensible design with released tools is:
  1. Qwen3-ASR (or nemotron-streaming for low latency) with the keyword list injected as context.
  2. Count matches on normalized transcript text. This needs Korean-specific handling: spacing variants, and particles attached to the keyword (e.g., 키워드+가/를).
  3. Take times from Qwen3-ForcedAligner.
  This needs a measured gate on a small owner-recorded Korean keyword set (precision/recall of counts), because no published Korean biasing numbers exist.
- sherpa-onnx Korean hotwords may work by passing the model's own tokens (a bpe vocab), but that is undocumented and must be tested before relying on it.
- Neologisms absent from training data are the main risk. Context prompting can also cause false insertions (the Nemotron card warning), so the gate should measure false positives on speech that does not contain the keyword.

### Gaps
- I found no measured Korean contextual-biasing gain (B-WER or keyword recall) for any open model.
- I found no Korean open-vocabulary KWS checkpoint or paper with Korean results.
- I did not check whether WhisperX or faster-whisper `initial_prompt` / `hotwords` (faster-whisper has a `hotwords` argument) help on Korean slang, and found no number for it.
