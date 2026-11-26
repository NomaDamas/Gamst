# Handoff: feat/instant-hud (worktree .local-run/wt-hud, from integration 70595ab, merged 9ca4545)

Owner decisions 26-27 (2026-11-26): the HUD shows L1's count at once and every later correction at the moment it was
decided; L1 uses a voice sample of the person from the same clip. Then (coordinator, same day): "speaker decision v2"
(relative decision against other speakers, several references, word-anchored diarization context, best live evidence
at L1), leave-one-clip-out on the four clips. Not pushed.

## Done (committed)
- A. Report: every `event_states` entry has `history` (commit + every revision, oldest first) with `capture_ms` and
  `decided_ms` (paced run: knowledge time = when it appears live; else the record's capture time), `reason`, `basis`,
  `by` (settling lane: verdicts' `settled_by` now carried on `event.revision`). `report.decision_clock` says which.
  `gamst/count_video.py`: `hud_timeline()` (pure, tested) + new panel: big number = count known at t; L1 words +1 at
  commit; corrections at their decided time as boxes "−1 취소 #9 / 다른 사람 목소리", "+1 추가 #12 / 감스트 목소리로
  정정", "−1 보류", "확정 #5"; dropping number shown struck out then the new one; `--hold-settled` holds until the last
  decision + 2.4 s (max 30 s). Old reports (no history) fall back to committed/settled fields.
- B. `gamst recipe enroll person-term-count --kind voice --main-speaker [--start 0 --seconds 15]
  --allow-remote-diarization`: pyannoteAI Precision-3 on the window, the speaker alone longest is enrolled from their
  own segments. Every voice file now has `enrolled_from` (range or main_speaker + segments). Range enrolment is sliced
  exactly to [start, start+seconds].
- Scoring: `gamst.recipes.bench` truth words with `count` (repetitions), `truth.exclude` ranges, whole-word end for
  repetitions (`word_end_ms` on new events; inferred for old reports), `error_types()`, `live_events()` (L1 instant);
  `scripts/score_owner_clips.py REPORT TRUTH --voice V [--exclude window|segments]`.
- Tests: `tests/test_instant_hud.py`.

## Runs
- `try-it/hud/run.sh SHORT NAME` (through `try-it/guard.sh hud ...`): owner range from `truth_owner.json` "voice" if
  present, else main speaker of the first 15 s; outputs `try-it/testbed/SHORT/hud/NAME.{voice.json,enroll.log,json,log}`.
- Owner voice ranges: none marked at 21:5x (check again before each run).

## Baseline (old runs, voice from another recording; this scorer)
| clip | person | heard | L1 right/counted | final right/counted | person→other | uncertain | other→person |
|---|---|---|---|---|---|---|---|
| 뽀또 (r2) | 10 | 9 | 4/4 | 7/8 | 1 | 1 | 1 |
| 몇기야 | 6 | 6 | 0/0 | 1/1 | 4 | 1 | 0 |
| 시발 | 7 | 7 | 0/0 | 0/0 | 7 | 0 | 0 |
| 달 | 23 | 23 | 0/0 | 0/0 | 23 | 0 | 0 |

## Next
C runs + videos (`<term>_카운트_v5_즉시.mp4`), D aligner per repetition, then v2.
