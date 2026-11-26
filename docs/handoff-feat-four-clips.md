# Handoff: feat/four-clips (worktree .local-run/wt-clips, from feat/target-speaker-keyword-and-pose-count 72c4e93)

Task: whole-word terms (owner decisions 15, 19), the SOTA combination on the owner's four clips, candidate lists for
the owner to label, count videos. Not pushed.

## Done (committed)
- Whole-word terms in `gamst/providers/term_events.py` (`word:W`, `word:W*`, `#flag`, `may_be_number`), verification
  per repetition (`part_id`, `WordVerification._repetitions`), recogniser hints without the syntax (`bias_terms`);
  docs in the module docstring and `skills/gamst-person-terms/SKILL.md`; tests `tests/test_whole_word_terms.py`.
  Full suite: 1685 tests, then 1694 after the b4e99bb merge: exactly the 15 known errors.
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

- Videos (original sound, provisional orange / final green / revoked red / uncertain violet, last picture held silent
  until the last stretch settles, `--hold-settled`), in `~/Desktop/홍명보/gamst/GAMST 카운트 결과/`:
  `뽀또/뽀또_카운트_v4_SOTA.mp4` (r2), `감스트의 해병대식 내리갈굼/몇기야_카운트_SOTA.mp4`,
  `감스트 피파 억까/시발_카운트_SOTA.mp4`, `애교쟁이 인디깅/달_카운트_SOTA.mp4`. Frames checked: Korean readable.
- Not seen in these runs: no recogniser wrote 십팔 or 18 in 피파 (the flag path is only unit-tested); no Scribe session
  closed early (no recogniser errors, every run complete).

## Left (owner)
- Label the four candidates.md (감스트 / 다른 사람 / 말 안 함).
- Voice reference for 피파 / 인디깅 / 내리갈굼: voice.json (another recording) scored the person's own speech under
  0.5 (인디깅 31 s: 감스트 alone on a bicycle, all 7 repetitions "another voice"). Owner-marked ranges per clip, or a
  reference from the same kind of recording, before choosing defaults on the four clips.

## Rerun
```
T=.local-run/try-it
$T/guard.sh clips $T/clips-run.sh bbo r1        # bbo | marine | fifa | indi
(cd $T && PYTHONPATH=../wt-clips host/bin/python clips-score.py testbed/bbo/runs/r1.json testbed/bbo/truth.json)
python -m gamst.term_candidates --report $T/testbed/bbo/runs/r1.json --out $T/testbed/bbo/candidates.md --title "..."
python -m gamst.count_video --report R --video CLIP --out OUT --title T --count speech.term_events:term.spoken \
  --other speech.term_events:term.other_voice --count-label "감스트가 말한 X" --other-label "다른 사람" --note "..." --hold-settled
```
Run python from the wt-clips folder (or with cwd elsewhere): `python -m` puts the working directory first on sys.path.
```
```
