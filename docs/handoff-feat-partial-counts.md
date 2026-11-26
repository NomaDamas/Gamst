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

## Runs
(filled in below as they finish)
