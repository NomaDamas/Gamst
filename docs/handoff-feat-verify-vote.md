# Handoff: feat/verify-vote (worktree .local-run/wt-vote)

## Done (committed)
- Three-recogniser majority in the verification lane (`gamst/providers/word_verification.py`, `decide: "majority"`,
  `keep_single`); voter item `vote_asr` (Speechmatics Enhanced, terms bias, capability `speech.vote_words`), `--with vote`;
  the builder turns on `vote` + `majority` when the item reads a voter; `with_asr` and setup `roles` skip a hosted voter.
- Speech-aligned verification regions (owner decision 21): `WINDOWS` min_ms 10000, max_ms 20000, cap_ms 30000,
  pause_ms 500 (`step_ms` = old fixed regions); status `regions`, `straddling`, `diarize_jobs`; window records carry
  `region`. Unit tests in `tests/test_catch_up.py` (RegionTest, majority test).
- Docs: `docs/catch-up-verification.md` (majority, regions).

## Measured (a.wav, B2 config + vote, fixed 15 s regions)
- keep_single on (E0, E2, E3) and off (F1-F3), 6 runs, all identical: final 7 counted / 7 right, precision 1.00, recall 0.70,
  uncertain 3 (12.64, 19.74, 27.44); provisional 4/4; provisional->final p50 15.2-20.0 s, p95 21.8-23.6 s.
- 24.16 revoked (heard_by_one, others heard speech); 20.14, 21.46, 22.30 counted (Scribe + Speechmatics);
  19.74 uncertain (overlap); 28.10 revoked (heard_by_one); 14.0 not transcribed.
- Fixed 15 s boundaries cut 5 utterances per run (2 Scribe, 1 Speechmatics, 2 gpt); 4 pyannoteAI jobs per run.

## Left
- Speech-aligned runs G1-G3 (running at the time of writing): region lengths min/median/max, straddling (must be 0),
  latency, jobs vs fixed; then `docs/measurements/verify-vote-2026-11-26.md` and the results line in
  `docs/decision-log-2026-11-26.md` (add results/links only; rows 21-24 are the coordinator's).

## Rerun
```
S=<scratchpad>/cu   # run3.sh, score.py, summary.py, straddle.py live there
try-it/guard.sh catchup $S/run3.sh G1 --with verification --with vote
python $S/summary.py G1; python $S/straddle.py $S G1
# keep_single off: RECIPE=$S/ptc-nosingle.json try-it/guard.sh catchup $S/run3.sh F1 --with verification --with vote
```
run3.sh = `gamst recipe run person-term-count --audio a.wav --voice voice.json --terms 뽀또 --language ko --as-live
--headless --allow-remote-asr --allow-remote-diarization --profile try-it/profile-stream-openai.json --report ... --store ...`.
