# Handoff 2026-11-26: realtime word counting by one person (temporary, keep updated)

Read this first when resuming. Then `docs/decision-log-2026-11-26.md` (every owner decision, technical decision with
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
