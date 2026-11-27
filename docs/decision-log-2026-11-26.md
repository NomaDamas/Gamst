# Decision log: counting one person's words in real time (2026-11-26)

Everything the owner decided or agreed on 2026-11-26, 2026-11-27 (35-50: [integrated perception design](design-2026-11-27-integrated-perception.md)) and 2026-11-28 (58-66: a given person's given action), the technical decisions taken with their evidence, and the answers
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
| 30 | The owner's labels saved by the labelling tool are the ground truth for the four clips (뽀또 10, 몇기야 6, 시발 7, 달 23; files `try-it/testbed/<clip>/truth_owner.json`, `truth_score.json`). |
| 31 | State of the art first (owner, 2026-11-27): the best recogniser by published results (the live `asr`, today Scribe v3 Live) is never outvoted by weaker ones. Others may add words it missed; they may veto one of its words only with strong contrary evidence (its own word confidence low and the others heard speech there and wrote something else); a disputed word is `uncertain` (shown, not counted, never "not said"), not revoked. The majority of three (decision 22) stays an option (`decide: "majority"`). |
| 32 | A word only the primary (state-of-the-art) recogniser heard is counted (`contested: "stand"`, default), even when the others heard other speech there: SOTA first. Measured from three runs: +7 right, +3 wrong (the invented 24.16 s in 뽀또). Scribe's own word confidence cannot separate invented from true words (24.16 s: 0.33; true 41.76 s: 0.02). |
| 33 | Two modes: count (default, today's behaviour) and search (recall first: never miss the person; uncertain words become candidates with their reason). |
| 34 | Wrap up, document all experiments and lessons, publish demo videos in a GitHub issue table, then merge PR #366 (owner, 2026-11-27). |
| 35 | Demo videos are not uploaded anywhere (copyright: broadcast clips); the video column of issue #371 was removed. Videos stay local. |
| 36 | Free local disk: models and environments GAMST no longer uses are deleted (voxtral, sortformer, nemotron3, diarizen, ps4 TSE, sherpa zipformer ko, Light-ASD, MediaPipe pose_lite, their venvs, merged worktrees; multilingual-e5 1.1 GB). Other sessions' worktrees are untouched. |
| 37 | Movement counting uses ONE model, the most accurate, counting every frame directly: no verification lane, no weaker fast lane, no "uncertain" state for movements, no skipped frames (owner: "검증레인은 없애고 성능이 더 좋은걸로 바로 세면 되지", "건너뛴 프레임이 너무 많은데?", "애매함도 없애"). Evidence: the verifier lowered the few-shot path to 14/16; the fast lane skipped 101-483 of 714 frames. |
| 38 | Default pose model RTMW-l 384x288 (CoreML); MediaPipe full selectable with `--pose`. At start a short benchmark decides: if the SOTA model (live + L2 within the memory budget) keeps up, it is used; if it is too heavy, MediaPipe full, said in the report, HUD and `get_pipeline`. Never two models in one run. |
| 39 | Sapiens2 1B is an opt-in pose model the user selects (`--pose sapiens2`); the owner approved its download. Its licence (no biometric processing or surveillance; Meta audit rights) is stated; GAMST never bundles or auto-downloads it. |
| 40 | Two timings for movement counting: realtime (L1 in order + L2 correction by the same model over backlog intervals) and post (a finished file: every frame, batches, chunk parallelism). Late counts are corrections (보정) and exist only in realtime. |
| 41 | L2 continuity: intervals start with a pre-roll that builds state only (rule windows + rate window + warm-up), an event belongs to the lane whose owned range contains its end time (no doubles, no gaps), the L1 resume is a cut too, the person lock is handed over (box overlap, else face; otherwise `who: unknown`, never a guess); duplicates within ±0.1 s are merged at storage. |
| 42 | No out-of-memory: every run has a memory budget (video memory, spill bound, number of L2 replicas from free memory, re-checked); heavy jobs only through `guard.sh`. |
| 43 | Who / what / words are separate and composable: "a given person's given action" is person + action. Actions are counted for every track and identity is bound afterwards (retroactive), like words heard first and attributed by voice. |
| 44 | Identity few-shot is REQUIRED and enforced in the user-defined references layer: an instruction with `who` and no person reference (or one below the quality check) is refused with guidance (CLI, MCP, HUD). Inputs: photo, video range, camera seconds, voice; consent record. Identity uses face + whole-body appearance (+ voice), not the face only; appearance is trusted within one video. The appearance model is chosen by SOTA criteria (official release, permissive licence, digest, <= 500 MB), replacing SOLIDER (Google Drive only, no digest, CUDA, data terms); LVFace-B (455 MB) is downloaded (owner: follow the recommendations). |
| 45 | Inputs are camera, video file, microphone and sensors, each entering on its own; user-defined inputs are allowed. |
| 46 | A request without a recipe is composed by the agent at once (new MCP `compose_recipe`); existing recipes are used as they are or customised (`set_instruction`, `--with/--without`, `configure_pipeline_module`, `--set`). New MCP `enroll_person` and `who` in `set_instruction`. |
| 47 | The semantic VLM module stays OFF (owner: not needed). When used, it must be a causal state chain (each state links the previous state, what changed and the causing events), not independent 4 s snapshots. |
| 48 | Word counting keeps its current structure (lanes, verification, uncertain/revoked states). |
| 49 | Sports: build what Ultralytics claims for sports (ball/player tracking, heat maps, form vs reference) as a recipe case, and bring other SOTA or other-sport open source (e.g. Roboflow Sports) in as recipes, in an isolated worktree, branch `feat/sports-recipes` and its own PR (tmux window `gamst-sports`); AGPL models (Ultralytics) only as opt-in extras. Generic pieces (position heat map, per-occurrence form/intensity score, object lane) are built as general modules. |
| 50 | Philosophy (owner): high-frequency modalities are not inferred one by one by an LLM/VLM; light models plus physical, mathematical and engineering computation hand context to the LLM, so it shows it watches in real time: GAMST attaches perception to the LLM. Design: [design-2026-11-27-integrated-perception.md](design-2026-11-27-integrated-perception.md). Merges only with the owner's explicit OK (PR #373 pending). |
| 51 | (Technical, PR #373, [design](design-2026-11-27-integrated-perception.md) §4) The pose lane batches: up to 8 waiting frames per call (`batch`), the next call starts while the last one is committed, one CoreML session per fixed batch size (8, 4, 2) warmed up at load. Measured: 14.0 ms per frame live, real-time factor 0.93 at 60 fps, 0 skipped (testbed "One model, every frame"). |
| 52 | (Technical) L2 workers run the same model on the GPU, in float32. The Neural Engine beside the GPU adds 15 crops/s (+25 %), a second GPU worker 12.7 crops/s (+19 %), but the Neural Engine computes in float16 and its counts are not measured as equal; it stays an option (`l2_settings`), not the default. |
| 53 | (Technical, design §4.2 safety net) An L2 count is `state: final` with `correction: true` and `decided_by: "catch_up"` (not a new revision state, so word counting's state tables stay as they are); an event the other lane stored with the same label and person within 100 ms (`SAME_EVENT_MS`) is not stored twice. |
| 54 | (Technical, design §4) The start benchmark (`choose_pose`) assumes one person, measures synthetic frames through the real shared-memory transfer, live and one L2 worker at the same time, the detector at its interval; RTMW stays if live + L2 >= input fps. Measured at the start of each run; the numbers are in the report, stderr, the HUD and `get_pipeline` notes. |
| 55 | (Technical) Identity at a cut: `TargetPerson.snapshot/fork/restore` hold the locked person by box overlap (IoU >= 0.3) during the pre-roll, else only a face inside the interval locks it; forks share the face/body models. Face points and joints are read by each pose's `keypoint_set` (RTMW `coco_wholebody133` face points 2, 1, 0, 71, 77), never by MediaPipe indices. Not run on people: LVFace and SOLIDER weights are not on this Mac; identity thresholds are unmeasured on real people. |
| 56 | (Owner, 2026-11-27) The whole-body appearance (re-ID) model must be the state of the art ("무조건 SOTA"): GAMST is not used commercially, so non-commercial terms (LUPerson pretraining, MSMT17 research terms) and official Google Drive releases without a published digest are acceptable; datasets serve tests and demos only. Chosen by published same-domain MSMT17 results: PersonViT ViT-B/16 (mAP 80.8 / R1 92.0 without re-ranking), default; OSNet-AIN stays an option. |
| 57 | (Owner, 2026-11-27) Identity thresholds stay provisional until the owner brings labelled multi-person clips. Answers to the identity follow-up: the binding runs with nobody enrolled (MCP enrollment mid-run); LVFace-B stays (accuracy first) and its scoring runs beside the pose lane, never in it. |
| 58 | (Owner, 2026-11-28) An action is specified by text only, few-shot clips only, or **text + clips together**. Text + clips: the text becomes the rule skeleton (which joints, which comparisons, in which order); the clips (and negatives) calibrate the rule's numbers; counting uses both. Count mode counts what the rule AND an example say; search mode what either says (OR). Every event carries its evidence: the rule satisfied (time of each step) and the nearest example with its distance. `gamst recipe enroll --rule` now takes clips (it refused them before). Built: `gamst/providers/pose_exemplar.py` (`rule+sequence`), `movement_rule.calibrate`, `compose_recipe` / `set_instruction` `what.movement: {text, rule, examples, negatives, combine}`. Measured: [measurements/action-spec-2026-11-28.md](measurements/action-spec-2026-11-28.md). |
| 59 | (Owner, 2026-11-28) A specific person REQUIRES both a text (a name or a description) AND a photo few-shot; either alone is refused with guidance, in the CLI, MCP and `compose_recipe` (`who: {name, text, photos, consent}`). The description is kept with the person reference; it identifies nobody (no model reads it). |
| 60 | (Owner, 2026-11-28) Count each described unit once: one 관제탑 cycle (fold → spread sideways → fold) or one punch = 1, each with its own start and end; repetitions are separate and never merged. |
| 61 | (Owner, 2026-11-28) Other people are recognised continuously too (the binding follows everyone, named or not), with one row per person in the persons view. |
| 62 | (Owner, 2026-11-28) 관제탑 = both arms folded in front → spread out to BOTH SIDES → folded again = 1; each repetition counts separately. It is NOT "arms straight up": the earlier `rule_gwan*.json` (arms up) was wrong. Rule: `docs/movements/gwanjetap.rule.json`. |
| 63 | (Owner, 2026-11-28) 맨까송 punch = 감스트 punched = 1 (one strike; on this clip mostly a fist brought down onto the chair). Rule: `docs/movements/punch.rule.json` (fist by `hand_open` on the raw-score hand gate, plus a fast wrist). |
| 64 | (Owner, 2026-11-28) Alpamayo2 (the edge provider of the acceptance matrix, P1-2) is kept. |
| 65 | (Owner, 2026-11-28) A plan for a small VLM is needed (what it would do beside the counting lanes, on which hardware); not built here. |
| 66 | (Owner, 2026-11-28) Non-commercial state-of-the-art models are allowed: GAMST is not used commercially (extends decision 56 beyond re-ID). |

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
| Primary first (decision 31, measured) | the best recogniser is not outvoted | 4 clips x 3 runs (s1, p1, p2): no word revoked by the recognisers; 41.76 and 28.10 (person) and 24.16 (other) contested -> uncertain every run (majority: revoked); final right/counted as v7 (6/7, 3/4, 6/6, 13/13) except 몇기야 p1 2/3 (27.10 contested -> uncertain); Scribe's own confidence 0.02 at 41.76 vs 0.33 at 24.16, so no veto by confidence (docs/handoff-feat-sota-first-rule.md) |
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
- Lessons learned (kept up to date): [lessons-2026-11-26.md](lessons-2026-11-26.md)
