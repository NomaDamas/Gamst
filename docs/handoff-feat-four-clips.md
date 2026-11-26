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

## Measured (2026-11-26, one run per clip, 뽀또 twice)
| clip | term | final | provisional | uncertain | rows in candidates.md |
|---|---|---|---|---|---|
| 뽀또 r1 / r2 | 뽀또 | 8 / 8 (7 right: precision 0.875, recall 0.70) | 0 | 2 | 16 |
| 해병대식 내리갈굼 | 몇기야 | 1 | 0 | 1 | 8 |
| 피파 억까 | 시발 (5 spellings) | 0 | 0 | 0 | 7 |
| 애교쟁이 인디깅 | 달 (word:달*) | 0 | 0 | 0 | 29 |
- 뽀또: same decisions as recorded H1/H3 (27.42 s, another voice, counted via Precision-3 window; 24.16 refused).
- 피파 / 인디깅: every occurrence was given to another voice (per-word scores 0.17-0.40 against voice.json, diarized
  speakers not the person): the voice from another recording does not match these clips; nothing tuned.
- r1 of 뽀또 ran the integration worktree's code (python -m put that cwd first on sys.path); clips-run.sh now cd's
  to try-it. Same combination, same decisions.
- Hosted cost of the 5 runs (list prices): Scribe ~$0.020, gpt-live ~$0.076, Speechmatics ~$0.030, pyannoteAI 20 jobs
  567.8 s billed ~0.018 EUR; total ~$0.15.
- Candidate lists: try-it/testbed/{bbo,marine,fifa,indi}/candidates.md.

## Left
- Videos.

## Rerun
```
T=.local-run/try-it
$T/guard.sh clips $T/clips-run.sh bbo r1
python -m gamst.term_candidates --report $T/testbed/bbo/runs/r1.json --out $T/testbed/bbo/candidates.md --title "..."
```
