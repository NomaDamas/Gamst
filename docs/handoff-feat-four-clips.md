# Handoff: feat/four-clips (worktree .local-run/wt-clips, from feat/target-speaker-keyword-and-pose-count 72c4e93)

Task: whole-word terms (owner decisions 15, 19), the SOTA combination on the owner's four clips, candidate lists for
the owner to label, count videos. Not pushed.

## Done (committed)
- Whole-word terms in `gamst/providers/term_events.py` (`word:W`, `word:W*`, `#flag`, `may_be_number`), verification
  per repetition (`part_id`, `WordVerification._repetitions`), recogniser hints without the syntax (`bias_terms`);
  docs in the module docstring and `skills/gamst-person-terms/SKILL.md`; tests `tests/test_whole_word_terms.py`.
  Full suite: 1685 tests, exactly the 15 known errors.
- Verification status keeps start, term and spelling of second-hearing / voter occurrences (no transcript), 256 kept.
- `python -m gamst.term_candidates --report R --out candidates.md`: one row per place any recogniser wrote the word.

## Inputs (durable)
- `try-it/clips-run.sh SHORT NAME`: one paced run (bbo, marine, fifa, indi), report + log in `try-it/testbed/SHORT/runs/`.
- `try-it/testbed/fifa/terms.json` (one term 시발: 시발, 씨발, 씨팔, 십팔#may_be_number, word:18); 달: `word:달*`.
- `try-it/testbed/bbo/truth.json` (bench format of labels_bbo.md).

## Measured
(filled in as runs finish)

## Left
- Runs, candidate lists, videos.

## Rerun
```
T=.local-run/try-it
$T/guard.sh clips $T/clips-run.sh bbo r1
python -m gamst.term_candidates --report $T/testbed/bbo/runs/r1.json --out $T/testbed/bbo/candidates.md --title "..."
```
