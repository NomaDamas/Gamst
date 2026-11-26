# Handoff: feat/l1-instant-policy (worktree .local-run/wt-l1, from integration 3c63b1c). Not pushed.

Read this alone to finish the task. Reply to the coordinator in English; owner-facing text in Korean.

## Task (coordinator, 2026-11-26, after the owner looked at the v5 run m1)
1. L1 marked many of the person's words "다른 사람" provisionally (뽀또 19.74, 21.40, 22.30: CAM++ 0.46 / 0.50 / 0.34 <
   0.5), so they appeared only when verification added them at 44.6 s. Owner (decision 26): every heard word counted
   at L1 at once, corrected later; L1 must not suppress a heard word unless the voice is clearly someone else.
2. 뽀또's automatic main-speaker reference (first 15 s) mixes two people (its two pieces 0.42 apart): the other man's
   words at 49-54 s were made final as the person (5-6 false counts).
Do: A. L1 instant policy as the person-term-count default (old behaviour an option). B. generic voice reference
quality check with fallback, reported in the run report and on the HUD. C. rerun the four clips, score against
`try-it/testbed/<clip>/truth_owner.json`, v6 videos, frames checked. Spend cap $1.5.

## Rules
Local runs / full suites only through `/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it/guard.sh l1 cmd...`.
Full suite: demo venv `~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests` (15 known errors:
test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2). Never touch the main checkout, the demo venv,
anything outside GAMST or others' files. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
Don't push.

## Done (committed)
- A. `SpeakerVerify(l1={policy, clear_other, clear_margin})` (`gamst/providers/speaker_verify.py`, module doc "L1
  policy"): `instant` = a provisional word counts at once (also without a score) unless its score < `clear_other`
  (default 0.2) or, with the v2 bank, a known other voice is closer by `clear_margin` (default off); `threshold` = old.
  Words carry `l1: "instant"`. Only for provisional words (a verification lane settles them): otherwise refused.
  `person-term-count.json` speaker_check `"l1": {"policy": "instant", "clear_other": 0.2}`; passed by
  `perception_parts._speaker_verify` only when verification reads the words. CLI `--l1 instant|threshold`,
  `--l1-clear-other`, `--l1-clear-margin`. Why 0.2: m1 L1 scores of the person's words 0.26-0.82, other voices
  0.36-0.79 (no threshold separates them at L1).
- B. `gamst/voice_quality.py`: `self_consistency` (mean / min pairwise cosine of a reference's embeddings; one piece =
  unmeasured), `choose` (below `min_quality` 0.5: first `--voice-fallback FILE` of the same model whose own quality
  passes, `combined` (default; centroid of each, equal weight) or `replace`; else kept and flagged `reference_low`),
  `hud_line` ("목소리 기준 품질 낮음 (0.42 < 0.50) · 다른 녹음 샘플과 합쳐 씀"). Wired in `_reference_value` (runs only),
  report `voice_quality`, stderr note, live CountView line (ASCII), video HUD line (`count_video.py`, y 1036). Enrolment
  writes `quality` into every voice file. CLI `--voice-fallback` (repeatable), `--voice-min-quality`,
  `--voice-fallback-mode combined|replace`.
- Tests `tests/test_l1_instant_policy.py`; `test_perception_recipes` voice keys + `quality`.

## Measured before the runs (offline, v2 study cache embeddings, owner labels, enrolment segments excluded)
Reference quality (mean pairwise): 뽀또 0.417 (2 pieces), 몇기야 0.625, 시발 0.447 (pairs 0.665/0.266/0.41), 달 0.68;
`testbed/voice.json` (e2s.wav, other recording, 40 pieces) 0.519. "current" rule (word's diarized speaker >= 0.5):
| clip | same-clip p→o / o→p | other recording alone | combined (equal weight) | union of all pieces |
|---|---|---|---|---|
| 뽀또 (14 words) | 0 / 6 | 0 / 2 | 0 / 2 | 0 / 2 |
| 몇기야 (5) | 0 / 1 | 3 / 0 | 0 / 1 | 3 / 0 |
| 시발 (7) | 0 / 0 | 7 / 0 | 0 / 0 | 7 / 0 |
| 달 (1) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
-> fallback default `combined`. With min_quality 0.5, 뽀또 and 시발 fall back (combined), 몇기야 and 달 keep their own.

## Runs (v6)
Command per clip: `VOICE=$T/testbed/<clip>/hud/m1.voice.json $T/guard.sh l1 $T/hud/l1/run.sh <clip> v6 --voice-fallback
$T/testbed/voice.json` (`hud/l1/run.sh` = `hud/run.sh` with PYTHONPATH on this worktree; same m1 reference so only the
policy and the check change). Started 23:0x. Results: see below when filled.
