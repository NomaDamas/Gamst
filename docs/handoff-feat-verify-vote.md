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

## Measured, speech-aligned regions v1 (G1-G3; pause = 500 ms gap in all recognisers)
- Identical in 3 runs: regions 10.0 / 13.6 / 11.7 s + end (reasons pause, cap_short_pause x2), 0 utterances straddled
  (fixed 15 s: 5 per run), 4 pyannoteAI jobs (as fixed). Final 8 counted / 7 right (27.44, another voice, counted:
  its window's Precision-3 labels gave the person's speaker), uncertain 2. Provisional->final p50 46-50 s, p95 63-68 s
  (fixed: 15-20 / 22-24 s): with no 500 ms gap shared by all recognisers the region waited for the 30 s cap.
- Fix committed: an utterance end is a pause too (cap_short_pause removed).

## Measured, speech-aligned regions (H1-H3, committed version)
- Same boundaries 10.0 / 23.64 / 35.39 s, all `pause`; 0 utterances straddled; 4 jobs; final 8/7 (H1, H3: 27.44 counted)
  and 7/7 (H2); uncertain 2; provisional->final p50 21.0-23.9 s, p95 36.8-41.0 s.
- Written up: docs/measurements/verify-vote-2026-11-26.md; decision-log technical rows added.

## Left
- Nothing of this task; open for the owner: region latency (last region waits for the end), 27.44 attribution with the
  moved window, defaults on the 4-clip set.

## Rerun
```
S=<scratchpad>/cu   # run3.sh, score.py, summary.py, straddle.py live there
try-it/guard.sh catchup $S/run3.sh G1 --with verification --with vote
python $S/summary.py G1; python $S/straddle.py $S G1
# keep_single off: RECIPE=$S/ptc-nosingle.json try-it/guard.sh catchup $S/run3.sh F1 --with verification --with vote
```
run3.sh = `gamst recipe run person-term-count --audio a.wav --voice voice.json --terms 뽀또 --language ko --as-live
--headless --allow-remote-asr --allow-remote-diarization --profile try-it/profile-stream-openai.json --report ... --store ...`.
