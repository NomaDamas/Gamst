# Handoff 2026-11-26: realtime word counting by one person (temporary, keep updated)

Read this first when resuming (also `docs/lessons-2026-11-26.md`). Then `docs/decision-log-2026-11-26.md` (every owner decision, technical decision with
evidence, and answer), `docs/realtime-attribution-decisions.md` (design and lanes), `docs/catch-up-verification.md`
(catch-up and verification contracts), `docs/speaker-tracks.md`, `docs/measurements/*-2026-11-26.md`.

## 1. Goal (owner)

GAMST counts how many times a given person says given words (and makes given movements), in real time, from
few-shot references (the words as text or example clips; the person's voice as a short sample), inside the GAMST
structure: input layer → Recipe → Pipeline (lanes) → Perception modules (registry candidates) → Context (capture-time
timeline with states) → MCP → LLM. State of the art by default, measured on the owner's clips; non-commercial
state of the art as opt-in only. Realtime is required; accuracy is defended by later lanes.

## 2. Branches

- Integration branch: `feat/target-speaker-keyword-and-pose-count` (worktree `.local-run/wt-speaker-pose`), Draft PR
  NomaDamas/Gamst#366 against Main (`feat/humansense-20261112-154922`). Pushed. Do not merge to Main, release 0.7.0 or
  archive GamstCrew without asking the owner again.
- Merged today (in order): runtime-instructions, av-speaker, sota-asr, overlap-speech, realtime-attribution (core),
  sota-diarization, streaming-asr, default-streaming-asr, catch-up, asr-more-providers, verify-recall.
- In progress (not merged): `feat/verify-vote` (`.local-run/wt-vote`: 3-recogniser vote + speech-aligned verification
  regions), `feat/realtime-tse` (`.local-run/wt-tse`: target speaker extraction lane L3).
- Full test suite: exactly 15 pre-existing errors (test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2).
  Run it only through the memory guard (section 5).

## 3. Design as built

```text
input (microphone, camera)        all audio kept (gamst/providers/audio_memory.py), nothing dropped
L1 live          streaming ASR (default elevenlabs.scribe_v3_live.hosted; candidates speechmatics.enhanced.hosted,
                 openai.gpt_live_transcribe.hosted, deepgram.nova_3.hosted, assemblyai.stt_3_7_realtime.hosted,
                 local sherpa/voxtral) + CAM++ voice score on ~1.5 s around each word → provisional counts
L2 catch-up      on lag > 2 s the live module jumps to now; skipped range = to-do; worker pool (cap by resources)
                 forks the module (state copy; hosted = new session + preceding text), listens from the last pause
L3 TSE           (building) overlap detected → extract the person's voice → re-recognise + CAM++ → decide
L4 verification  serial, only over complete ranges: second recogniser gpt-live-transcribe (+ keyword hints,
                 Qwen3-ForcedAligner times) [+ optional third voter Speechmatics]; pyannoteAI Precision-3 splits
                 speakers (cloud, --allow-remote-diarization); CAM++ names the person's speaker from whole segments
                 → final / revoked / uncertain (shown, not counted)
Context          states provisional / final / revoked / uncertain / pending; MCP count_events / get_timeline
```

Recipe: `person-term-count` (and `scene`, `person-voice-style-count`); options `--allow-remote-asr`,
`--allow-remote-diarization`, `--asr CANDIDATE`, `--with verification`, `--with tracks` (live slots, opt-in, not used),
`--with lips` (off), `--with overlap` (PixIT, off), `--realtime`/`--as-live` (paced replay), `--debug-slowdown`,
`gamst recipe bench --clips DIR` (multi-clip scoring). Video: `python -m gamst.count_video` (states coloured).

## 4. Measured so far (one 54 s clip, 뽀또, owner truth 10 words)

| setup | final counted / right | precision | recall |
|---|---|---|---|
| Whisper turbo + voice from another recording (start) | 8 / 6 | 0.75 | 0.6 |
| Scribe v3 + gpt verification + Precision-3 (catch-up B2) | 6 / 6 | 1.00 | 0.6 |
| + keep clear-voice words (verify-recall) | 8 / 7 (4 uncertain) | 0.875 | 0.7 |

The one false count left is a word Scribe invented at 24.16 s (owner confirmed nothing is said 23.8-24.2 s); Speechmatics
never wrote it (6/6 runs) → the 3-recogniser vote is being measured. Catch-up forced-overload test: 0 audio dropped,
final words identical. Shotgun movement clip: 16/16 (text rule, collision = sudden stop, sound shown only).

## 5. Environment and rules that bite

- Try environment: `.local-run/try-it/` (gamst.sh sources `~/.config/gamst/keys.env`; profiles; `set-keys.command`
  opens a Terminal to paste keys). Keys present: PYANNOTEAI, OPENAI, ELEVENLABS, ASSEMBLYAI, DEEPGRAM, SPEECHMATICS.
- Durable test inputs: `.local-run/try-it/testbed/` (a.mp4/a.wav 뽀또 clip; b.* shotgun clip; g.wav owner-marked 1.5 s
  voice cut of a.wav — leaks 12.04/12.64; e2s.wav + voice.json = the person's voice from another recording;
  voice_owner.json; truth_bbo.json; labels_bbo.md; clips.md; asr/ recorded word streams of every provider;
  diarization/, tse/). The session scratchpad under /private/tmp is wiped on restart — never keep inputs there.
- Memory: the Mac (32 GB) hit OOM when several agents ran large models. Every local model run and every full test
  suite goes through `.local-run/try-it/guard.sh NAME command...` (one heavy job at a time, waits for ≥35 % free memory,
  kills only its own job under 12 %). At most two or three agents at once; only one heavy local job.
- Disk ~14 GB free. Delete only GAMST things this work created and can regenerate; never anything of other projects
  (absolutely not `/private/tmp/claude-501/-Users-jinminseong-Desktop-tacticall`) or other agents' files.
- zsh: an unquoted `$VAR` holding several paths is ONE word (a `rm -rf $P` silently did nothing once).
- Main checkout `~/Desktop/catch-your-mind` has uncommitted owner files: never touch; work in `.local-run/` worktrees.
- Spend so far today on hosted APIs: under $1 in total (each report lists its part). Report before larger spend.

## 6. Next steps (in order)

1. Merge `feat/verify-vote` (3-recogniser majority vote; speech-aligned verification regions with 0 split utterances).
2. Merge `feat/realtime-tse` (SOTA TSE incl. non-commercial opt-in; live overlap detection; L3 forkable; real-time
   factor on this Mac; GPU needs).
3. Whole-word term matching (needed for "달" as a sound word with each repetition counted, and "18" never inside other
   numbers; number-like matches flagged).
4. Run the SOTA combination on the four owner clips (`try-it/testbed/clips.md`): 감스트 뽀또 (뽀또), 해병대식 내리갈굼
   (몇기야), 피파 억까 (시발/씨발/씨팔/십팔/18), 애교쟁이 인디깅 (달). Produce candidate lists for the owner to label
   (person / other / nothing said), then choose defaults (live recogniser: Speechmatics vs Scribe; vote rule; region
   length; thresholds) on all four, never on one.
5. Render count videos with original sound into `~/Desktop/홍명보/gamst/GAMST 카운트 결과/` (뽀또 v4 and one per new clip).

## 7. Open owner questions

- Uncertain words as `term.spoken` in state uncertain (built) — confirm.
- Licences: Nemotron-3 (OpenMDW-1.1, redistribution review), Sortformer (NVIDIA Open Model License), DiariZen
  (CC BY-NC, opt-in).
- Voice reference for the new clips: another recording of the person (current) or owner-marked ranges per clip.


## 8. Update (end of 2026-11-26) — read this section first

- Integration HEAD e346ede (pushed). Decisions now 1-29 in `docs/decision-log-2026-11-26.md`.
- Owner labelled all four clips (labelling tool `.local-run/try-it/label_tool`, run `../host/bin/python server.py 8777`;
  labels in `~/Desktop/홍명보/gamst/GAMST 라벨링/<clip> - 정답.json`, truth files `try-it/testbed/<bbo|marine|fifa|indi>/truth_owner.json`, `truth_score.json`).
  Owner counts: 뽀또 10 (owner earlier said 9 — confirm), 몇기야 6, 시발 7, 달 23.
- Scores vs labels (`docs/measurements/owner-labels-2026-11-26.md`): words heard 45/46; attributed right only 8/45 with the
  other-recording voice. Bottleneck (`docs/measurements/attribution-bottleneck-2026-11-26.md`): (1) voice reference from another
  recording; (2) deciding one short word at a time. Not the lane structure; live diarization is weak on this audio, Precision-3 is
  fine; overlap is a real limit (TSE PS4 no gain).
- Running (agent in `.local-run/wt-hud`, branch `feat/instant-hud`, handoff `docs/handoff-feat-instant-hud.md` there):
  instant L1 HUD with later visible corrections (decision 26); same-recording voice (automatic main speaker for the test clips; owner
  labels no ranges); voice sample UX (`--record-voice-first`, `--voice-wav`, `--voice-range`; decision 28); speaker decision v2
  (relative comparison with automatic negatives, several references incl. L4-final words, word-anchored diarization context;
  decision 29); leave-one-clip-out validation; per-error-type table; rerun four clips; videos `<term>_카운트_v5_즉시.mp4`;
  per-repetition times for "달달달" via Qwen3-ForcedAligner.
- Next after it: merge feat/instant-hud, suite via guard.sh, push, report to the owner in Korean with the per-error table and
  video paths; then continue toward "더 완벽" (owner's words): remaining errors by type, voice sample UX demo, promotion video.


## 9. Update (after feat/instant-hud, integration 1058e67)

- Merged: instant HUD (L1 at once, corrections shown when decided), voice sample options (`--record-voice-first`,
  `--voice-range`, `--voice-wav`, automatic `--main-speaker` via Precision-3), owner-label scoring (`scripts/score_owner_clips.py`,
  bench `error_types`), speaker decision v2 (opt-in only; it did not help in leave-one-clip-out). Suite: 15 known errors.
- Same-recording voice (automatic main speaker), scored on owner labels (words in the enrolment segments excluded; scored person
  words 9 / 4 / 7 / 13): final right/counted 7/12, 3/4, 6/6, 12/12 (total 28/34); L1 instant 24/30; person→other misses 37 → 1;
  other→person 6 (5 in 뽀또, whose automatic reference was mixed: two segments score 0.42 against each other); not heard 4.
- Videos v5 (instant HUD): `~/Desktop/홍명보/gamst/GAMST 카운트 결과/<clip>/<term>_카운트_v5_즉시.mp4`.
- Left: MCP/HUD hook for the voice-recording step; running main speaker for live input without a sample; remaining errors are
  other voices that CAM++ and Precision-3 confuse with the person (better sample = owner sample on the same input; stronger
  speaker model or TSE did not help so far).


## 10. Update (v6 merged)

L1 counts every heard word at once (`l1.policy instant`, default), excluded only below `clear_other` 0.1. Voice reference
quality check (`gamst/voice_quality.py`, `--voice-min-quality` 0.5, fallback order: owner sample → clean pieces → `--voice-fallback FILE`
replace → flag). v6 vs owner labels (L1 / final, right/counted): 뽀또 7/14 / 6/7; 몇기야 4/5 / 3/4; 시발 7/7 / 6/6; 달 13/13 / 8/8
(another run 12/12 — hosted recognisers vary). Latency word→HUD p50 1.8-6.9 s (turn commits); short-turn variant
`try-it/hud/l1/person-term-count-short-turns.json` is faster but lost words — owner decision. Videos `<term>_카운트_v6.mp4`.


## 11. Update (wrap-up before merge, 2026-11-27)

- Decision 32: SOTA first (`contested: stand`). v8 vs owner labels: 30/33 (뽀또 7/9, 몇기야 3/4, 시발 7/7, 달 13/13), word → HUD ~0.7 s.
- Decision 33: `--mode count|search` (search: recall first, candidates with reasons; MCP `set_count_mode`). Live sound-only HUD window.
- Wrap-up: `docs/wrapup-2026-11-26-person-words-and-movements.md`; HUD demo: `docs/hud-demo.md`; demo issue NomaDamas/Gamst#371.
- Decision 34: merge PR #366 into Main after the wrap-up (owner). Next improvements are listed in the wrap-up.

## 12. Update (owner discussion, 2026-11-27) — read this section first

- **Read first:** [design-2026-11-27-integrated-perception.md](design-2026-11-27-integrated-perception.md). It covers the philosophy, the owner-approved
  integrated structure (who / what / words), how each is specified (identity few-shot required and enforced), realtime
  vs post modes, L2 continuity, the answers to the owner's questions, and Ultralytics and the sports track.
- **Decisions 35-50** are in the decision log; **lessons 33-37** are in the lessons file.
- **Branches and worktrees:**
  - `feat/realtime-pose` (`.local-run/wt-rt-pose`, Draft PR #373):
    - one model (RTMW-l) counts every frame;
    - verification lane removed;
    - realtime/post timing;
    - benchmark-based model choice;
    - L2 person-lock handover;
    - CI numpy fix.
    - Not merged: owner OK needed.
  - `feat/sapiens2-pose` (`.local-run/wt-sapiens2`): Sapiens2 1B opt-in provider, weights downloaded (5.7 GB) under
    `try-it/models/sapiens2.pose_1b`; measurements in progress.
  - `feat/sports-recipes` (`.local-run/wt-sports`, tmux window `gamst-sports`): sports recipes. It is a separate session the
    owner works with directly.
- **Next, in order:**
  1. identity: few-shot enforcement, face + whole-body appearance (pick a replacement for SOLIDER), photo enrollment,
     LVFace-B download, `person_binding`, MCP `enroll_person`, `who` in `set_instruction`, keypoint-set-aware face
     alignment (RTMW);
  2. MCP `compose_recipe`, and the semantic module as a causal state chain (kept off);
  3. a conversational demo over MCP.
- **Needed from the owner:** multi-person clips with labels of who did what, and when, to measure identity.
- **Removed today:** multilingual-e5 (1.1 GB, unused); the labelling tool server was stopped.
