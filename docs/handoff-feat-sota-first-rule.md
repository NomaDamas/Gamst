# Handoff: feat/sota-first-rule (worktree .local-run/wt-sota-first; from integration 6298a8a, merged 587729b = v7)

Owner decision 31: "가장 SOTA가 우선이어야지, 더 약한 모델들이 투표로 이기면 안 된다." The three-recogniser majority of the
verification lane is replaced by a primary-first rule (`decide: "primary"`, the recipe default when voters are read);
`majority` stays an option. Not pushed.

## Done (committed)
- `gamst/providers/word_verification.py`: `decide: "primary"` with `contested` (`uncertain` default | `stand`),
  `contest_by` (`all` default | `any`), `low_confidence` (None default = no veto | probability), `add` (`agreed` default |
  `any`); verdict reasons `contested` (uncertain) / `vetoed`; `primary_confidence` on verdicts when the recogniser gives
  word confidence. Rule text: docs/catch-up-verification.md ("Primary first", "Primary confidence").
- `gamst/recipes/perception_parts.py`: voters -> `vote: true, decide: "primary"` (was `majority`).
- Tests: tests/test_catch_up.py (4 new: never outvoted, contested -> uncertain / stand, veto only with low confidence,
  settings checked).
- Docs: catch-up-verification, decision-log (31), lessons (27).

## Not done / for the coordinator
- Scribe v3 Live's `logprob` is not passed on by the live adapter (`streaming_hosted.ElevenLabsSession.translate` keeps
  (text, start, end) only; `streaming_asr._answer` builds the word dicts). A 4th tuple item with `{logprob}` plus
  `**extra` in `_answer` would carry it; I did not edit those files (outside this branch's scope). With the measured
  numbers the default (no veto) would not change anyway.
- Owner question: `contested: "uncertain"` (default, as briefed) leaves 41.76 / 28.10 / 27.1 shown but not counted;
  `contested: "stand"` would count them and 24.16 too (as the person: the voice says person_speaker there).

## Scribe v3 Live confidence (raw WebSocket probe, one pass per clip, VAD commits, keyterm)
Words carry `logprob`. 24.16 (뽀또, invented / other voice) 0.33; 41.76 (시발, person, true) 0.02; person words (23)
0.02-0.63 median 0.47; other voices (7) 0.33-0.67 median 0.57. Lower confidence goes with the true words: (a) cannot
separate. Probe script and raw JSON: scratchpad only (regenerable: `probe.py` streams the clip PCM).

## Runs (paced as live, `try-it/hud/sotafirst/run.sh` = hud/v7/run.sh with PYTHONPATH on this worktree)
`VOICE=$T/testbed/<clip>/hud/m1.voice.json $T/guard.sh sotafirst $T/hud/sotafirst/run.sh <clip> RUN --voice-fallback
$T/testbed/voice.json`; s1 = before the v7 merge (v6 base), p1 / p2 = v7 base + primary rule. Scored with
`scripts/score_owner_clips.py REPORT truth_owner.json --voice m1.voice.json`.

| clip | v7 final a / b / c | p1 final | p2 final | s1 final | uncertain (p1 / p2) | errors at final (p1 / p2) vs v7a |
|---|---|---|---|---|---|---|
| 뽀또 | 6/7, 6/7, 6/7 | 6/7 | 6/7 | 6/7 | 12.62, 19.74, 24.16, 28.10 / same | unc 2, not heard 1, o→p 1 / same; v7a: p→o 1 (28.10), unc 1, not heard 1, o→p 1 |
| 몇기야 | 3/4, 3/4, 3/4 | 2/3 | 3/4 | 3/4 | 27.10 / - | unc 1, not heard 1, o→p 1 / not heard 1, o→p 1; v7a: not heard 1, o→p 1 |
| 시발 | 6/6, 6/6, 6/6 | 6/6 | 6/6 | 6/6 | 41.76 / 41.76 | unc 1 / unc 1; v7a: p→o 1 (41.76 revoked) |
| 달 | 13/13, 12/12, 12/12 | 13/13 | 13/13 | 13/13 | - / - | 0 / 0 (p2 L1 12/12, verification added 9.34) |

The two words: 24.16 uncertain (`primary+contested(second+third)+person_speaker`) in s1, p1, p2 (majority: revoked
heard_by_one); 41.76 uncertain, same basis, in s1, p1, p2 (majority: revoked heard_by_one in v6, v6a, v7a-c). No word
was revoked by the recognisers any more; contested words in 3 runs: 24.16 x3 (other), 28.10 x3, 41.76 x3, 27.10 x1
(person). With `contested: "stand"` (offline from the bases): +7 right, +3 wrong (24.16) over the 3 runs.

## Cost (list prices)
Recognisers (Scribe $0.27 + Speechmatics $0.40 + gpt-live $1.02 per hour) on ~695 s (s1, p1, p2 = 3 x 213.7 s; s2 stopped
during 뽀또 for the v7 merge) ≈ $0.33; Scribe probe 214 s ≈ $0.02; pyannoteAI ~51 jobs ≈ $0.05. Total ≈ $0.40 (cap $1.5).
