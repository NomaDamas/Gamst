# Handoff: feat/instant-hud (worktree .local-run/wt-hud, from integration 70595ab, merged 9ca4545). Not pushed.

Read this alone to finish the task. Reply to the coordinator in English; owner-facing text in Korean.

## Owner decisions (2026-11-26)
- 26 HUD: the big number is L1's count the moment L1 commits a word; corrections of L2 catch-up / L3 TSE / L4
  verification appear later, at the moment they are decided ("−1 취소 #9 (다른 사람)", "+1 추가 #12", "확정"), never
  retroactively. Words only (no face/visual evidence).
- 27 L1 uses a voice sample of the person from the same recording right away.
- 28 The owner will NOT mark voice ranges for the four clips: use the automatic same-recording reference ("main
  speaker"), exclude words inside the enrolment SEGMENTS from scoring, and say so.
- 29 Voice-sample UX (mics differ): live -> recording step on the same input before counting
  (`--record-voice-first SECONDS`: countdown "세려는 사람만 N초 동안 말해 주세요", level meter, consent line, re-record);
  recordings -> `--voice-wav FILE` or `--voice-range FROM-TO`, else automatic main speaker. Documented, tested with
  stand-ins (MCP/HUD hook for it: not built, optional "if simple").
- Coordinator: after that, "speaker decision v2", all generic: (a) relative decision vs the person's references AND
  the other speakers present (negatives automatic from diarization speakers not matching the person); (b) several
  references (enrolment + words confirmed final by L4, never provisional), max/centroid parameter; (c) diarization
  context anchored to each word (fixed s before/after) instead of the region; (d) L1 uses best live evidence (per-word
  score vs references and negatives known at commit), L4 decides on speaker segments. Leave-one-clip-out (choose on 3,
  test on the 4th, all 4 folds). Report a per-error-type table (heard?, person→other, other→person, uncertain) per
  clip for: current, same-clip reference only, v2. Spend cap $3 total; guard.sh; commit often.

## Rules
Every local model run / full suite through `/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it/guard.sh NAME
cmd...`. Full suite (demo venv `~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests`): exactly the 15
known errors (test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2). Never touch the main checkout, the
demo venv, anything outside GAMST, others' files. Commits end with `Co-Authored-By: Claude Opus 5.5
<noreply@anthropic.com>`. No transcripts stored. Don't push.

## Done (committed)
- A. `event_states[*].history` (commit + every revision, `capture_ms`, `decided_ms` = knowledge time when paced, else
  record time; `reason`, `basis`, `by` = settling lane via verdicts' `settled_by`), `report.decision_clock`.
  `gamst/count_video.py`: `hud_timeline()` + `_draw_hud()`; correction boxes in Korean; dropping number struck out;
  `--hold-settled` = last decision + 2.4 s (max 30 s). Tests `tests/test_instant_hud.py`.
- B. `gamst recipe enroll ... --kind voice --main-speaker [--start 0 --seconds 15] --allow-remote-diarization`
  (`gamst/recipes/enrollment.py`: `choose_main_speaker`, `_main_speaker`); every voice file has `enrolled_from`.
  Voice-sample UX: `gamst/recipes/voice_sample.py` (`--record-voice-first`, `--voice-range`, `--voice-wav`,
  `--main-speaker-seconds`, `--voice-consent`, `--voice-out`; only active when one of these is given), report
  `voice_sample` with `excluded_s`; tests `tests/test_voice_sample.py`; skill doc `skills/gamst-person-terms/SKILL.md`.
- Scoring: `gamst/recipes/bench.py` (truth `count` repetitions, `truth.exclude`, `word_end_ms` / inferred whole-word
  end, `error_types()`, `live_events()`); `scripts/score_owner_clips.py REPORT TRUTH --voice V [--exclude
  segments|window]` (default segments, decision 28).
- C. Four clips rerun (run name m1) + videos v5 (see below). Frames checked (Korean readable, corrections shown).
- Started (uncommitted at writing -> check `git status`): `gamst/providers/voice_bank.py` (VoiceBank: references,
  negatives, `decide(vector, floor, margin)`, `shared(pipeline, reference, embeddings, settings)`), not wired yet.

## Measured: per clip vs owner truth (`try-it/testbed/<clip>/truth_owner.json`), words inside the enrolled
main-speaker segments excluded (decision 28)
Same-clip reference = automatic main speaker of the first 15 s (no owner range exists): 뽀또 SPEAKER_00 of 2 (6.1 s
alone), 몇기야 SPEAKER_00 of 2 (8.9 s), 시발 1 speaker (7.1 s), 달 1 speaker (9.5 s). Run m1, 21:49-21:56.
| clip | person scored | L1 instant right/counted | final right/counted | person→other | other→person | uncertain | not heard |
|---|---|---|---|---|---|---|---|
| 뽀또 | 9 | 4/9 | 7/12 | 0 | 5 | 0 | 2 (14.05 owner-added, 28.1) |
| 몇기야 | 4 | 3/4 | 3/4 | 0 | 1 | 0 | 1 (39.24) |
| 시발 | 7 | 5/5 | 6/6 | 1 | 0 | 0 | 0 |
| 달 (each repetition) | 13 | 12/12 | 12/12 | 0 | 0 | 0 | 1 |
| total | 33 | 24/30 | 28/34 | 1 | 6 | 0 | 4 |
Stricter (whole 0-15 s window excluded): 뽀또 8: L1 4/8, final 7/11; 몇기야 4: 3/4, 3/4; 시발 7: 5/5, 6/6; 달 7: 7/7, 7/7.
"Current" (old runs r1/r2: voice from another recording), 0-15 s excluded: 뽀또 8: L1 4/4, final 6/7 (p→o 1, o→p 1,
unc 1); 몇기야 4: 0/0, 0/0 (p→o 4); 시발 7: 0/0, 0/0 (p→o 7); 달 7: 0/0, 0/0 (p→o 7).
Recompute: `cd /private/tmp && PYTHONPATH=<wt-hud> <try-it>/host/bin/python <wt-hud>/scripts/score_owner_clips.py
<try-it>/testbed/<clip>/hud/m1.json <try-it>/testbed/<clip>/truth_owner.json --voice <...>/hud/m1.voice.json`
(old runs: testbed/<clip>/runs/r1.json, bbo r2.json).
Cost so far (list prices): 4 runs ≈ $0.12 recognisers + pyannoteAI 0.017 EUR; v2 prepare ≈ 80 pyannote jobs ≈ 0.05 EUR.

Videos: `~/Desktop/홍명보/gamst/GAMST 카운트 결과/{뽀또/뽀또, 감스트의 해병대식 내리갈굼/몇기야, 감스트 피파 억까/시발,
애교쟁이 인디깅/달}_카운트_v5_즉시.mp4` (`try-it/hud/video.sh SHORT m1 "이 영상 첫 15초의 주 화자(자동)"`).

## Commands
- run a clip: `T=.local-run/try-it; $T/guard.sh hud $T/hud/run.sh SHORT NAME [extra run options]` (bbo marine fifa
  indi; enrols main speaker unless truth_owner.json has "voice"; `VOICE=file` env reuses a voice file).
- D (per-repetition times): `$T/guard.sh hud-align $T/venvs/qwen-asr-transformers/bin/python
  <wt-hud>/scripts/qa_repetition_times.py "<TestBed>/애교쟁이 인디깅.mp4" $T/models/qwen3.forced_aligner_0_6b
  7.30-7.94:5 8.60-9.30:5 11.58-12.28:5 31.08-32.06:7` (queued at 22:00; result in the task output only -> rerun if
  lost). Reference = syllable energy nuclei (a proxy, not the owner's ear). If the aligner does not beat the even split,
  say so and keep the even split.
- v2 study: `cd /private/tmp && set -a && . ~/.config/gamst/keys.env && set +a && PYTHONPATH=<wt-hud> $T/guard.sh
  hud-v2prep ~/.local/share/gamst/venvs/sherpa-onnx/bin/python <wt-hud>/scripts/qa_speaker_decision_v2.py prepare`
  (caches `try-it/testbed/v2/<clip>.json`, skips clips already done), then `... qa_speaker_decision_v2.py evaluate`
  (local, cheap: rules current vs v2, LOCO over a grid, all folds printed).

## Left, in order
1. Read v2 `evaluate` output; pick what helps (relative margin, grow/combine, context 10/5 vs 4/2, L1 floor/margin).
2. Wire v2 into GAMST generically: `voice_bank.shared` in `SpeakerVerify` (option `relative={floor, margin}`: word
   embedding vs bank at commit) and in `WordVerification` (voice settings `decision: relative`, `floor`, `margin`,
   `word_margin`, `anchor_ms: [before, after]`; in `_identify` name the person speaker as the closest one with margin,
   add the other speakers as negatives; in `_voice_of` grow references from person_speaker decisions; word-level
   fallback = relative). CLI: `--speaker-decision JSON|FILE` -> context -> perception_parts `_speaker_verify` /
   `_word_verify`. Defaults unchanged (absolute). Tests.
3. Live runs of the four clips with v2, each clip with the parameters chosen on the other three (LOCO), score, the
   per-error-type table (current / same-clip only / v2). Optionally v6 videos.
4. Full suite through guard (15 known errors), handoff, final report (what changed, per-clip table L1/final, voice
   reference per clip, cost, video paths, D result, v2 folds).
