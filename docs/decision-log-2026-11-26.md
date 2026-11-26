# Decision log: counting one person's words in real time (2026-11-26)

Everything the owner decided or agreed on 2026-11-26, the technical decisions taken with their evidence, and the answers
given to the owner's questions. Design text: [realtime-attribution-decisions.md](realtime-attribution-decisions.md),
[catch-up-verification.md](catch-up-verification.md), [speaker-tracks.md](speaker-tracks.md). Measurements:
`docs/measurements/*-2026-11-26.md`. Test clip: 54 s, several people in a noisy shop, word 뽀또, 10 words by the person
(owner's labels by ear).

## 1. Owner decisions

| # | decision |
|---|---|
| 1 | GAMST is a realtime system that can also analyse recordings, not a video-analysis tool. Every product path runs on a live stream. |
| 2 | Realtime first, correction after: provisional counts at once; a concurrent verification lane revises only what is wrong. |
| 3 | Transcription must be streaming. Whisper is dropped as default (its local weights were deleted); it stays a selectable candidate. |
| 4 | Who said a word is decided from the owner's few-shot voice sample of the person. Faces and lips are an optional fallback, off by default. |
| 5 | State of the art by default, measured on the owner's clips; non-commercial state of the art only as opt-in. |
| 6 | No tuning on one clip; defaults are chosen on a set of clips with word truth. |
| 7 | Everything inside GAMST: registry candidates, one environment per model, Pipeline modules and lanes, recipes. No parallel framework. |
| 8 | Nothing is dropped when processing falls behind: the live lane jumps to "now", the skipped range is caught up by parallel workers (pool, capped by resources) from a copy of the live state, listening from the last speech pause. |
| 9 | Verification runs serially in capture order, only over ranges that catch-up has finished; catch-up can never overwrite a final. |
| 10 | Live diarization (Nemotron-3) does not decide who said a word; pyannoteAI Precision-3 is used in the verification lane. |
| 11 | Overlapped speech: extract the person's voice with state-of-the-art target speaker extraction (TSE), in real time, parallel if needed; non-commercial allowed as opt-in. |
| 12 | Words that remain uncertain are shown to the LLM as "uncertain", never as "not said". |
| 13 | Reduce misses: a word only the live recogniser heard is kept when the voice evidence for the person is clear. |
| 14 | Separate consent for sending audio to pyannoteAI (diarization) and to the transcription providers. |
| 15 | Sound-word terms: "달" counts only as a whole word made of that syllable ("달", "달달달"), once per repetition, never inside other words. |
| 16 | Keys live only in `~/.config/gamst/keys.env` (mode 600); hosted calls print their price first. |
| 17 | Never touch anything outside GAMST (other projects, other agents' files). |
| 19 | Clip "감스트 피파 억까": count 시발, 씨발, 씨팔, 십팔 as one term (owner); "18" written as digits is not counted. Risk noted: 십팔 is also the number eighteen read aloud. |
| 20 | Clips and terms: 뽀또 (감스트 뽀또), 몇기야 (해병대식 내리갈굼), 시발/씨발/씨팔/십팔 (피파 억까), 달 as a sound word (애교쟁이 인디깅). |
| 18 | Heavy local runs go one at a time through `try-it/guard.sh` (memory guard) after an out-of-memory crash. |

## 2. Technical decisions and their evidence

| decision | why | evidence |
|---|---|---|
| Per-word voice score uses ~1.5 s around a word | speaker embeddings need about a second of sound | scores of 0.6 s snippets were unstable; 1.5 s mixed neighbours (testbed doc) |
| The voice reference should come from the same kind of recording | channel and excitement change the voice more than the model size | another recording: 8/6; owner-marked 1.5 s of the same clip: 12/9 (leaky); w2v-BERT did not separate (testbed doc) |
| Live slots not used for attribution | they mixed people and drifted | Nemotron-3: person's words in all slots, 5 person + 2 other in its best slot (diarization doc) |
| Precision-3 in verification | it kept the person in one slot; it is not streaming but fast | 10 person words, only 2 overlapped other words in its slot; 54 s in 12.7 s (diarization doc) |
| Scribe v3 Live as live recogniser | best published error rate; best recall here | 9/10 heard, 2.7 s to final with client commits (streaming-asr doc) |
| gpt-live-transcribe as verification recogniser | invented nothing where Scribe did | owner confirmed nothing is said at 23.8-24.2 s; Scribe wrote the word there, gpt did not (streaming-asr doc) |
| Qwen3-ForcedAligner for gpt word times | gpt-live returns no word timestamps; the aligner is among the best published | text never changed (12/12), 0.3 s per segment, 0.04 s median from Scribe's times |
| Client commits instead of server VAD | server VAD hardly commits in shop noise | 23 s vs 2.7 s median to final |
| Catch-up by worker pool with state copy | keeps "now" and loses nothing | forced slowdown: 0 audio dropped, final words identical, gaps filled in ~2.1 s (catch-up doc) |
| Verification after catch-up only | prevents an earlier lane overwriting a final | completeness watermarks; test in tests/test_catch_up.py |
| Combination result | precision first, recall next | final 6 counted, 6 right (precision 1.00 in 5 runs), recall 0.6 (catch-up doc) |
| Dropped approaches | measured no gain here | PixIT separation (unstable), lips (0 fixes 0 breaks), Qwen3-ASR (worse), larger voice model, Sortformer (slower than real time, merged everyone) |

## 3. Answers to the owner's questions

- **Forced alignment vs transcription.** Transcription finds the words; forced alignment takes known text and finds where each word is
  in the audio. gpt-live gives text only, so an aligner adds times without changing the text.
- **Why a 1-1.5 s limit?** It belongs to scoring a voice from a short snippet, not to voice technology: deciding identity on a whole
  speaker segment (diarization) or extracting the person's voice (TSE) avoids it.
- **Diarization vs separation vs TSE.** Diarization labels who speaks when (overlap = several labels); separation splits one
  microphone's mixture into voices; TSE extracts one known voice. Attributing an overlapped word needs separation or TSE.
- **Does Precision-3 handle more than two people?** Yes, and it marks overlap. Speaker caps (Sortformer 4, Live-1 8) belong to live
  models that no longer decide attribution.
- **Why is pyannoteAI in verification and not the transcription check?** Verification checks two things: was the word said
  (second recogniser, gpt-live) and who said it (Precision-3 + the few-shot voice). pyannoteAI answers "who"; it is not a recogniser.
- **Two recognisers?** Live = Scribe v3 (fast, hears most); verification = gpt-live with keyword hints (the words to count, not
  Scribe's transcript). Both heard → final; only gpt heard → added; only Scribe heard → kept if the voice evidence is clear (decision 13).
- **Jumping to "now"?** The LLM answers about the present at once; the skipped seconds show as pending and are filled within
  seconds; spoken words are history, so a later correction is acceptable.
- **How many catch-up workers?** At least one; more only shorten waiting. Cap = resources (1 on a laptop, provider concurrency
  for hosted, GPU memory on a server).
- **Where does catch-up live?** In the Pipeline; modules only need snapshot/restore (hosted recognisers get the preceding text).
- **What does GAMST count?** Two few-shots: the words (text or example clips, also given to recognisers as hints) and the
  person's voice; it counts the given words said by the given person.
- **Why did the out-of-memory crash happen?** Several agents ran large models at once (Voxtral, Sortformer, DiariZen with 10.9 GB
  peak); not the catch-up design, which was not built yet. Swap filled the disk.

## 4. Open (to decide on the owner's clip set)

Thresholds and margins; verification region length (latency vs pyannoteAI cost, 20 s minimum per job); third-recogniser vote;
which TSE model; whether the live recogniser should change after the AssemblyAI / Deepgram / Speechmatics comparison.
