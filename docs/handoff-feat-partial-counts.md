# Handoff: feat/partial-counts (worktree .local-run/wt-partial, from integration 57cc8ff). Not pushed.

Read this alone to finish the task. Reply to the coordinator in English; owner-facing text in Korean.

## Task (coordinator, 2026-11-27)
The owner compares v6 with the first video (`뽀또_카운트.mp4`), which looked instant only because the old `count_video`
drew each +1 at the word's own capture time (after offline decoding), not when GAMST knew it. v6 draws counts when
decided; L1 waits for the recogniser's turn commit (word -> HUD p50 1.8-6.9 s, max 13 s). Goal: the count rises within
about a second of the word, honestly (decided time), from the live recogniser's partial hypotheses; owner decision 26
(count at once at L1, correct later visibly).
1. Term spotting on partials (generic): stable term -> provisional count at once in a new lightest state `heard`, with
   the per-word voice check on the audio so far (L1 instant). Final commit reconciles by capture time: confirmed ->
   `provisional` (same event id), vanished -> `revoked` `partial_withdrawn`. Verification (L4) unchanged.
2. Debounce (N revisions or T ms); measure flicker.
3. HUD: `heard` distinct; the big number = heard + provisional + final - revoked; corrections visible.
4. Rerun the four clips (several runs where cheap), per clip: word end -> first HUD count p50/p95/max (target ~1 s),
   withdrawn partial counts, L1 and final right/counted vs v6; `<term>_카운트_v7.mp4`, frames checked.
5. Lesson in docs/lessons-2026-11-26.md. Spend cap $1.5; no transcripts stored.

## Rules
Local runs / full suites only through `/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it/guard.sh partial cmd...`.
Full suite: demo venv `~/.cache/gamst/demo/venv/bin/python -m unittest discover -s tests` (15 known errors:
test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2). Never touch the main checkout, the demo venv,
anything outside GAMST or others' files. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
Don't push.

## Measured before building
Scribe v3 Live partials (probe, 12 s of a.wav, 2026-11-27): `partial_transcript` has `text` and `turn_probability`
only, no word times (34 partials in 12 s, about one per 0.35 s). So a heard word's time is estimated (below).

## Done (committed)
- `gamst/providers/partial_terms.py`: `PartialSpotter` (stable = the k-th occurrence of a term in an utterance's
  partial text held in `min_revisions` partials in a row (default 2) or for `min_ms`; once per utterance; times from the
  partial's word times where it has them, else end = heard_to_ms - lag_ms (400) - ms_per_char (150) x characters after
  it; trace of counts per partial, no text) and `reconcile` (monotone pairing in time, nearest first, `max_ms`).
- `SpeakerVerify(partials={terms, ...})`: partial records -> stable terms scored on the sound so far (1.5 s window ending
  300 ms after the estimated end), L1 policy as for words; out as `status: partial` records whose words are only the
  heard terms (`state: heard`, `word_id heard:<utt>:<term>:<k>`). `status()` = spotter stats + trace.
- `TermEvents`: heard words -> `term.spoken` `state: heard` (counted; a clear other voice is not heard-counted); the
  final's occurrences pair with the heard events of the same turn (same utterance, or same recogniser within
  `match_ms` 4 s): `event.revision` of the heard event to the final's state (provisional / uncertain / final) with the
  word's own times, `by: l1`; the `_open` entry moves to the heard id so verdicts settle it; unpaired heard ->
  `revoked partial_withdrawn`; a final other voice -> heard revoked `other_voice`; a turn whose final never came ->
  withdrawn by the next final of that recogniser (or at `finish`). Raw partials (no voice module, `--without voice`):
  `TermEvents(partials=...)` spots them itself.
- `counting.py`: STATES heard, provisional, final, revoked, uncertain; COUNTED heard+provisional+final; a revision may
  make an event provisional; `with_state` gives a heard event the revision's times (`heard_start_ms/heard_end_ms` keep
  the estimate). `states` lists `heard` only when there is one.
- `count_video.py`: `heard` pale (BGR 200,222,236) "들림", dashed frame + "+1 들림" on the picture, dotted underline in
  the feed, dashed tick on the bar; a heard step drawn at its decided time; heard -> provisional is no change (no second
  +1); withdrawal "−1 취소 #n 부분 인식 철회". Subtitle "들리는 즉시(들림) · L1 확인(잠정) · 정정은 결정된 순간에".
- `bench.py`: `l1_state`, `live_events` (a heard word counts at L1 when its `by: l1` revision does), `heard_events`
  (first count), `withdrawn`, `first_count_ms`; `error_types` leaves out withdrawn heard words.
  `scripts/score_owner_clips.py`: views heard / l1 / final, `withdrawn`, `latency_s` (person / all: p50 p95 max).
  `scripts/partial_flicker.py`: replays the debounce over a run's trace for other N / T.
- Recipe `person-term-count.json`: asr `"partials": true`, speaker_check `"partials": {"terms": "terms"}`, terms
  `"partials": {}` (raw path only). CLI `--partials on|off`, `--partial-revisions N`, `--partial-ms MS`.
- Tests `tests/test_partial_counts.py` (13).
- Run scripts `try-it/hud/v7/run.sh` / `video.sh` (= hud/l1 with PYTHONPATH on this worktree; `VER` names the video).

## Runs (2026-11-27, paced as live, `try-it/hud/v7/run.sh`, voice per v6 defaults)
Per clip and run: `VOICE=$T/testbed/<clip>/hud/m1.voice.json $T/guard.sh partial $T/hud/v7/run.sh <clip> v7a|v7b|v7c
--voice-fallback $T/testbed/voice.json` (same reference and fallback as v6/v6r; 뽀또 uses the replace fallback like v6r).
Scored with `scripts/score_owner_clips.py REPORT truth_owner.json --voice m1.voice.json` (enrolment segments excluded;
person words 9 / 4 / 7 / 13 as in the v6 handoff). v6 rows rescored with the same script (뽀또: v6r, the v6 video's run).

### Word end (owner label) -> first count on the HUD (s, p50 / p95 / max, person words)
| clip | v6 | v7a | v7b | v7c |
|---|---|---|---|---|
| 뽀또 | 4.29 / 12.83 / 12.83 | 1.10 / 1.44 / 1.44 | 1.10 / 1.43 / 1.43 | 1.06 / 3.43 / 3.43 |
| 몇기야 | 7.79 / 11.59 / 11.59 | 0.89 / 3.22 / 3.22 | 2.17 / 3.21 / 3.21 | 0.89 / 2.50 / 2.50 |
| 시발 | 4.55 / 6.20 / 6.20 | 1.26 / 1.77 / 1.77 | 1.20 / 1.77 / 1.77 | 0.87 / 1.25 / 1.25 |
| 달 | 1.65 / 1.65 / 1.65 | 0.47 / 1.10 / 1.10 | 0.26 / 1.52 / 1.52 | 0.29 / 1.57 / 1.57 |
Pooled v7 (3 runs x 4 clips, 94 person words): p50 0.73, p95 2.17, max 3.43 s (v6: p50 ~4.3, max 12.8).
Slow ones (> 1.6 s): 몇기야 17.34 (all 3 runs 2.5-3.2 s: the partial held the word only ~3 s later), 뽀또 28.1 (v7c, 3.4 s)
and 몇기야 36.4 (v7b, 2.2 s): never stable in a partial before the commit, counted at L1's commit; 시발 23.18 / 24.78
(1.7-1.8 s). No scored word was counted before its end; one excluded word (몇기야 2.72, inside the voice range) was
heard at 1.91 s, before the owner's end label: the partial completed it from its start (keyterm bias).

### Accuracy (right / counted) vs v6
| clip | v6 L1 | v6 final | v7 heard (first count) a / b / c | v7 L1 a / b / c | v7 final a / b / c | withdrawn a / b / c |
|---|---|---|---|---|---|---|
| 뽀또 | 7/14 | 6/7 | 8/15, 8/15, 8/15 | 8/15, 8/15, 8/15 | 6/7, 6/7, 6/7 | 0 / 0 / 0 (heard 15/14/13) |
| 몇기야 | 4/5 | 3/4 | 4/5, 4/5, 4/5 | 3/4, 4/5, 3/4 | 3/4, 3/4, 3/4 | 1 / 0 / 1 (heard 8/6/7) |
| 시발 | 7/7 | 6/6 | 7/8, 7/7, 7/7 | 7/7, 7/7, 7/7 | 6/6, 6/6, 6/6 | 1 / 0 / 0 (heard 8/7/7) |
| 달 | 13/13 | 8/8 | 13/16, 12/14, 12/14 | 13/13, 12/12, 12/12 | 13/13, 12/12, 12/12 | 3 / 2 / 2 (heard 23/22/19) |
Withdrawn in total: 10 of 149 heard counts (6.7%), each shown as "-1 취소 #n 부분 인식 철회" when the final came: 몇기야 39.3
(a person word the final did not write, run variance: v7a/v7c), 시발 58.6 (no word there; withdrawn at the end), 달 15.2 /
21.3 / 24.5 (no word there). Clear other voice at the heard check (not counted): 뽀또 v7a 1.
Errors at final (v7a): 뽀또 p->o 1, uncertain 1, not heard 1, o->p 1 (same class as v6r); 몇기야 o->p 1, not heard 1
(39.24: Scribe's final did not write it in v7a/v7c; v7b wrote it); 시발 p->o 1 (41.76 heard_by_one, as v6); 달 0 (v6 had 5
heard_by_one at 12.28 x5: run variance of the voters, lesson 22).
v7c 달: verification lost its connection (RemoteDisconnected) -> words stayed provisional (final column = provisional).

### Debounce (replayed on the v7a traces, `scripts/partial_flicker.py`, counts only)
| N partials in a row | heard | withdrawn | flickers | delay after final word end p50 / p95 / max (s) |
|---|---|---|---|---|
| 1 | 67 | 15 (22%) | 0 | 0.28-0.64 / up to 2.76 |
| **2 (default)** | 55 | 6 (11%) | 17 | 0.58-1.1 / up to 3.06 |
| 3 | 41 | 0 | 24 | 0.9-1.4 / up to 3.36 |
(per clip ranges; N=3 also loses heard words, which then wait for the commit.) Partials: 129-158 per clip (~2.5/s);
flicker = a term that appeared in a partial and left before it was stable: 10-27 % of appearances at N=2 (v7a).
With the voice check the live withdrawals at N=2 were 6 of 55 in v7a (one heard candidate was a clear other voice).

### Videos v7 (run v7a; `try-it/hud/v7/video.sh SHORT v7a "목소리 메모"`, through guard)
`~/Desktop/홍명보/gamst/GAMST 카운트 결과/뽀또/뽀또_카운트_v7.mp4`, `감스트의 해병대식 내리갈굼/몇기야_카운트_v7.mp4`,
`감스트 피파 억까/시발_카운트_v7.mp4`, `애교쟁이 인디깅/달_카운트_v7.mp4`. Frames checked: 뽀또 12.3 (pale dashed frame, "1",
"+1 들림", feed "들림 11.78초 #1" dotted, tally 들림 1), 15.0 (2, both 잠정 after L1), 48.0 ("12 -> 9", 확정 #1, 보류 #2/#3); 몇기야 41.4 ("8 -> 7", "-1 취소 #8 부분 인식 철회", 39.30초 struck out); 달 9.0 (3 들림), 16.3 (17: 잠정 16, 들림 1).
Fixed after the first render: the U+2212 minus drew a box (the Korean font has none): ASCII "-"; the subtitle was
too wide for the panel.

### Cost
Recognisers (Scribe + Speechmatics + gpt-live, list $1.69/h with keyterms ~$1.75/h) on 3 x 213.7 s + 12 s probe ~ $0.32;
pyannoteAI ~51 jobs ~0.05 EUR (~$0.05). Total ~ $0.37 (cap $1.5).

## Full suite (through guard, after the HUD fix)
1735 tests: exactly the 15 known errors (test_stereo_context 7, test_stereo_metric 6, test_sam_audio_native 2).

## Open / owner decisions
- `heard` is on by default in person-term-count (owner decision 26); `--partials off` restores v6 timing.
- N=2 chosen from the replay (N=1 withdraws twice as often; N=3 withdraws none but leaves more words to the commit).
- Raw-partials path (`--without voice`) is unit-tested only, not run live.
- Not merged into integration; not pushed.
