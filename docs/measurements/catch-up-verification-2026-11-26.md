# Catch-up lanes and the verification lane on the 54 s shop clip, 2026-11-26 (Mac M2 Pro)

Design and contracts: [../catch-up-verification.md](../catch-up-verification.md). One clip (`a.wav`, several people in a
shop), so nothing here is a held-out accuracy.

Run: `gamst recipe run person-term-count --with verification --allow-remote-asr --as-live --audio a.wav --voice
voice.json --terms 뽀또` (voice: the person's 120 s from another recording; `voice_owner.json` was not used). L1 =
Scribe v3 Live with the term as keyterms (client commits), per-word CAM++ score, threshold 0.5; L4 = gpt-live-transcribe
+ keyword + Qwen3-ForcedAligner (MPS), pyannoteAI Precision-3 windows of 35 s per 15 s, CAM++ on each speaker's own
sound. Truth: the owner's word ends (person 12.04, 14.0, 19.74, 20.14, 21.46, 22.30, 28.10, 38.20, 46.64, 47.74; others
12.64, 20.34, 20.74, 27.44, 49-54 s; nothing said 23.8-24.7), ±0.6 s, scored by `gamst.recipes.bench`.

## 1. Forced overload: L1 slowed on purpose (`--debug-slowdown asr=300@5-9,20-24,36-40`)

300 ms of sleep before every 100 ms audio frame the live recogniser got in 5-9 s, 20-24 s and 36-40 s of the run.

| run | audio given up (asr gaps / memory dropped) | complete | skips (TODOs) | TODO ranges (skipped, heard from) | max TODO backlog | TODO -> done p50 / p95 | provisional: counted / right | final: counted / right | provisional -> final p50 / p95 | L1 busy / audio |
|---|---|---|---|---|---|---|---|---|---|---|
| B2, no slowdown | 0 / 0 | yes | 0 | - | 0 | - | 4 / 4 | 6 / 6 | 18.1 / 21.2 s | 0.78 s / 54.4 s = 0.014 |
| C2, slowdown | 0 / 0 | yes | 3 (all done) | 6.0-8.1 s from 3.66; 21.0-23.1 s from 18.14; 37.0-39.1 s from 35.30 | 1 TODO, 4.96 s of audio | 2.11 / 2.15 s | 4 / 4 | 6 / 6 | 19.7 / 27.5 s | 17.9 s / 54.4 s = 0.33 |

- **Final counts identical**: the same 6 words (11.78, 20.06, 22.30, 38.26, 46.74, 47.80 s) were final in both runs, and the
  same 4 provisional. The only difference: 19.74 s (the person, not counted in either) is `not_heard` in B2 and
  `uncertain_overlap` in C2, because gpt-live-transcribe wrote that word in C2 and not in B2 (two runs of the hosted
  model; the slowdown does not touch L4).
- Each TODO's worker (a new Scribe session with the last 50 characters committed before as `previous_text`) started at the
  live session's last pause (1.7-2.9 s before the skipped range, within `max_lookback_ms` 10 s) and heard 17.7 s in all
  (lookback + range + 1.5 s tail) in 6.3 s of work. The words at 19-23 s and 35-39 s that L1 counted came from the
  workers, at their capture times.
- Audio memory: at most 2.9 MB in memory (30 s kept plus holds), nothing spilled (budget 64 MB), 0 dropped.
- L1 "busy" is the time the live lane spent in the module, including the injected sleeps of the frames it did hear.
- Run wall time from start to report: 85.2 s (B2), 82.6 s (C2) for 54.4 s of audio; the last ~25 s are the verification
  lane finishing behind (and model start-up).
- An earlier pair (B1/C1) found a bug, fixed before B2/C2: a worker's turns had the live session's utterance IDs, so
  `speaker_check` dropped them as already seen (C1 lost the words of the 2nd and 3rd TODO at L1; its final count was
  still 4 = B1's 4, because L4 decides finals from its own hearing).

## 2. Full combination without slowdown (B2), per labelled word

| truth end | who (truth) | L1 word end | L1 voice score | L1 provisional | L4 decision | reason / basis |
|---|---|---|---|---|---|---|
| 12.04 | person | 11.78 | 0.35 | not counted | **counted** (revised) | heard twice, Precision-3 speaker of the person |
| 12.64 | other | 12.62 | 0.41 | not counted | not counted | `uncertain_overlap` (both speakers over it, per-word score < 0.5) |
| 14.0 | person | - | - | not transcribed | not transcribed | neither recogniser wrote it |
| 19.74 | person | 19.74 | 0.18 | not counted | not counted | `not_heard` (gpt did not write it here; in C2 it did: `uncertain_overlap`) |
| 20.14 | person | 20.06 | 0.27 | not counted | **counted** (revised) | heard twice, person's speaker |
| 20.34 | other | - | - | not transcribed | not transcribed | |
| 20.74 | other | 20.72 | 0.30 | not counted | not counted | `other_speaker` |
| 21.46 | person | 21.40 | 0.39 | not counted | not counted | `not_heard` |
| 22.30 | person | 22.30 | 0.51 | counted | counted (final) | heard twice, person's speaker |
| 23.8-24.7 (nothing said) | none | 24.16 | 0.36 | not counted | not counted | `not_heard` (gpt wrote no such word: Scribe's invented word is refused) |
| 27.44 | other | 27.42 | 0.14 | not counted | not counted | `uncertain_overlap` |
| 28.10 | person | 28.10 | 0.36 | not counted | not counted | `not_heard` |
| 38.20 | person | 38.26 | 0.54 | counted | counted (final) | heard twice, person's speaker |
| 46.64 | person | 46.74 | 0.55 | counted | counted (final) | heard twice, person's speaker |
| 47.74 | person | 47.82 | 0.51 | counted | counted (final) | heard twice, person's speaker |
| 49-54 | other | 49.88, 50.40, 52.20 | 0.17-0.22 | not counted | not counted | `other_speaker` (2), `no_speaker` + per-word score (1) |

| view (B2) | counted | right | precision | recall (of 10) | invented | other voice |
|---|---|---|---|---|---|---|
| provisional (L1, at once) | 4 | 4 | 1.00 | 0.40 | 0 | 0 |
| final (after L4) | 6 | 6 | 1.00 | 0.60 | 0 | 0 |

Word end -> provisional: p50 3.9 s, p95 13.5 s (Scribe's commits). Provisional -> final: p50 18.1 s, p95 21.2 s.

Run to run (all with the verification lane, no slowdown): final counted / right 5/5 (A1), 4/4 (A2), 5/5 (A3), 4/4 (B1),
6/6 (B2); provisional 4/4 in all. A1-A3 ran before the L1 keyterms were set; A1/A2 fed gpt in bursts, A3 at 1.0x
real time. Final precision was 1.00 in every run; final recall moved with what gpt-live-transcribe wrote in 13-30 s
(it missed 28.10 in every run, as in the streaming-ASR measurement, and 19-22 s in some runs).

## What was chosen, and on what

- Not chosen on this clip: `lag_ms` 2000, `max_lookback_ms` 10000, `tail_ms` 1500, workers 2 (capped by the hosted
  session's concurrency 2), thresholds 0.5, `near_ms` 1000, `overlap_share` 0.3, windows 15 / 15 / 5 s, both revoke
  flags on (the owner's rule). The slowdown spans were picked to fall inside speech.
- **Chosen after seeing this clip**: the verification recogniser's pace `feed_speed` 1.25 (default). In one run each, gpt
  fed in bursts (A2) wrote nothing at 13-26 s and fed at 1.0x (A3) wrote three words there; A1 (bursts) also wrote them.
  At 1.25x (B1, B2, C2) it wrote nothing there in B1 and wrote them in B2 and C2. The evidence does not separate pace
  from run-to-run variation; the default follows the API being a realtime API, not this result.
- L1 keyterms (`"bias": "terms"` on `asr`) follow the agreed text ("Scribe v3 Live (keyterms = the words to count)").

## Cost (list prices)

7 runs (A1-A3, B1, C1, B2, C2): Scribe v3 Live 7 x 54.4 s (+ 2 x 17.7 s of catch-up sessions) about $0.029;
gpt-live-transcribe 7 x 54.3 s about $0.108; pyannoteAI Precision-3 28 jobs, 800 s billed, about 0.025 EUR. Total about
$0.16. Unit tests use local stand-ins only.
