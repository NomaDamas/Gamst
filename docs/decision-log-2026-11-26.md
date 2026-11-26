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
| 18 | Heavy local runs go one at a time through `try-it/guard.sh` (memory guard) after an out-of-memory crash. |
| 19 | Clip "감스트 피파 억까": count 시발, 씨발, 씨팔, 십팔 and "18" (as some recognisers write it) as one term (owner). "18" is matched only as a whole word (never inside 180 or 2018), like the sound-word terms. Risk noted: 십팔 / 18 are also the number eighteen; such matches are flagged in the result. |
| 20 | Clips and terms: 뽀또 (감스트 뽀또), 몇기야 (해병대식 내리갈굼), 시발/씨발/씨팔/십팔/18 (피파 억까), 달 as a sound word (애교쟁이 인디깅). |
| 21 | Verification regions follow speech: a region closes only at a speech pause after a minimum length, with a bounded maximum for continuous speech; no utterance is ever split between regions (replaces the fixed 15 s regions). |
| 22 | Three recognisers may vote (live Scribe v3, gpt-live, Speechmatics): a word counts as said when at least two heard it; measured before it becomes the default. |
| 23 | The recogniser defaults (live and verification) are chosen on the four owner clips, not on one: Speechmatics is the measured candidate for live (no invented word in 6 runs, ~1.2 s to final), Scribe v3 Live is today's default. |
| 24 | Uncertain words keep the event `term.spoken` in state `uncertain` (shown, not counted, settled later by TSE or verification). |
| 25 | Counting words uses sound only: no face, lip or other visual evidence (owner). The `--with lips` part stays in the code, off, and is not used for word counting. |
| 26 | The HUD shows the live (L1) count at once; corrections from L2/L3/L4 appear later on the HUD, at the moment they are decided (visible correction), never retroactively. Realtime must be visible (promotion). |
| 27 | L1 uses a voice sample of the person from the same recording right away (owner-marked range; automatic "main speaker" as an option when none is marked). |
| 28 | Voice sample UX, because microphones and sound quality differ: realtime → a short sample recording on the same input before counting starts (countdown, level meter, consent, re-record); recordings → a sample file or a time range of the same recording; otherwise an automatic same-recording "main speaker" reference. The owner does not label voice ranges for the test clips. |
| 29 | Fix the attribution bottlenecks found on the owner labels (docs/measurements/attribution-bottleneck-2026-11-26.md): same-recording reference; relative comparison with the other speakers present (automatic negatives); several references (owner sample + words confirmed final); diarization context anchored to each word; validation leave-one-clip-out. |

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
| Clear-voice keep for words one recogniser heard (decision of the owner, measured) | recall without a third hearing | final 8 counted, 7 right in 4 runs: 21.46 recovered, the invented 24.16 kept too (verify-recall doc) |
| Majority of three recognisers (Speechmatics voter, decision 22, measured) | two independent hearings before a word counts | fixed regions: final 7 counted, 7 right, recall 0.70 in 6 runs (keep_single on and off the same); 24.16 refused, 20.14/21.46/22.30 confirmed by Speechmatics (verify-vote doc) |
| Speech-aligned regions (decision 21, measured) | no utterance split between regions | boundaries cut 0 utterances (fixed 15 s cut 5 per run); regions 10.0 / 13.6 / 11.7 s; same 4 pyannoteAI jobs; provisional -> final p50 21-24 s, p95 37-41 s (fixed: 15-20 / 22-24 s); the moved window gave 27.44 s (another voice) to the person in 2 of 3 runs (verify-vote doc) |
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

- **What does verification check?** Two things per word. (1) Was it said: a second (and optionally third) recogniser listens
  again; nobody else heard it → revoked (e.g. the invented word at 24.16 s). (2) Who said it: Precision-3 splits the region into
  anonymous speakers A, B…; CAM++ compares each speaker's whole speech with the person's few-shot voice and names the person;
  the word takes the speaker active at its time. Overlapped and unclear → uncertain, to be settled by TSE (L3).
- **Does the LLM wait for verification?** No. A provisional word is in the Context as soon as L1 commits it; verification only
  changes its state later (final / revoked / uncertain). Pending catch-up ranges are shown as pending.
- **"Sending" to pyannoteAI.** pyannoteAI is a cloud API: GAMST uploads the region's audio (with margins) and gets speaker
  segments back. That is why it has its own consent flag (`--allow-remote-diarization`).
- **Precision-3 vs CAM++.** Precision-3 answers *when* each (anonymous) speaker talked and where speech overlaps; it does not
  know the person. CAM++ answers *is this sound the person* by comparing with the few-shot voice. In L1 CAM++ sees ~1.5 s around a
  word (fast, may include a neighbour); in L4 it sees the whole speech Precision-3 gave one speaker (long, clean). Precision-3
  does not check CAM++; it gives CAM++ better material. It sits in the verification lane because it is not a streaming model.
- **Why ~1.5 s around a word?** A word is 0.3-0.5 s; a voiceprint needs about a second to be stable. Longer windows can include a
  neighbour: the other voice at 20.74 s scored 0.27 with 0.6 s and 0.41 with 1.5 s.
- **Diarization vs TSE.** Diarization slices *time* (who, when; overlap = two labels on the same seconds, sound still mixed);
  TSE slices *sound* (the person's voice extracted from the mixture). L3 runs TSE only where speech overlaps.
- **Where did 15 s come from?** An implementation default, not tuned; it is replaced by speech-aligned regions (decision 21).
  The extra context sent to Precision-3 (15 s before, 5 s after) keeps speaker labels stable; it does not decide the boundary.
- **CAM++ and few-shot.** CAM++ turns sound into a 192-number voiceprint; the few-shot sample gives the reference voiceprint;
  the score is their similarity (0-1).

## 4. Open (to decide on the owner's clip set)

Thresholds and margins; verification region length (latency vs pyannoteAI cost, 20 s minimum per job); third-recogniser vote;
which TSE model; whether the live recogniser should change after the AssemblyAI / Deepgram / Speechmatics comparison.
