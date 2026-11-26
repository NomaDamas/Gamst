# Three-recogniser majority and speech-aligned verification regions, a.wav (2026-11-26)

Follows [verify-recall-2026-11-26.md](verify-recall-2026-11-26.md) (runs D1-D4). Owner decisions 21 (regions follow
speech) and 22 (three recognisers vote), [../decision-log-2026-11-26.md](../decision-log-2026-11-26.md). Contracts:
[../catch-up-verification.md](../catch-up-verification.md).

Run as B2/D plus the voter: `gamst recipe run person-term-count --with verification --with vote --allow-remote-asr
--allow-remote-diarization --as-live --audio a.wav --voice voice.json --terms 뽀또`.
- L1: Scribe v3 Live with keyterms, per-word CAM++.
- L4: gpt-live-transcribe with the keyword and Qwen3-ForcedAligner times; Precision-3 windows (15 s before, 5 s after
  each region); CAM++ per speaker.
- Voter: Speechmatics Enhanced with additional_vocab (`vote_asr`, capability `speech.vote_words`).
- Rule: `decide: majority` (two of three), `clear: either`, `clear_margin` 0.1, `near_ms` 1000.

Truth as before (±0.6 s).

## Totals

| runs | regions | keep_single | final counted / right | precision | recall | uncertain | provisional | provisional -> final p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| D1-D4 (before: no voter) | fixed 15 s | - | 8 / 7 | 0.875 | 0.70 | 4 | 4 / 4 | 14.7-17.6 / 21.3-21.8 s |
| E0, E2, E3 | fixed 15 s | on | 7 / 7 | 1.00 | 0.70 | 3 | 4 / 4 | 15.2-16.9 / 21.8-22.5 s |
| F1-F3 | fixed 15 s | off | 7 / 7 | 1.00 | 0.70 | 3 | 4 / 4 | 16.3-20.0 / 22.3-23.6 s |
| G1-G3 (first version of regions) | speech-aligned | on | 8 / 7 | 0.875 | 0.70 | 2 | 4 / 4 | 45.7-50.4 / 62.8-67.6 s |
| H1-H3 | speech-aligned | on | 8 / 7 (H1, H3), 7 / 7 (H2) | 0.875 / 1.00 | 0.70 | 2 | 4 / 4 | 21.0-23.9 / 36.8-41.0 s |

- **keep_single on and off decided every word the same** in E and F: no word here was written by one recogniser
  with nobody else writing anything near it.
- **The voter removed the false count of D** (24.16 s): Speechmatics did not write that word in any run, so Scribe was
  alone. It also confirmed 20.14, 21.46 and 22.30 in every run, which gpt-live wrote only in some runs.

## Per word (fixed 15 s: E/F, six runs, identical; speech-aligned: H1-H3)

| truth end | who | provisional | final (E/F) | why (E/F) | final (H) |
|---|---|---|---|---|---|
| 12.04 | person | no | counted | Scribe + gpt + Speechmatics, person's speaker | counted |
| 12.64 | other | no | uncertain | Scribe + gpt, overlap, score < 0.5 | uncertain |
| 14.0 | person | - | not transcribed | no recogniser wrote it in the first hearing | not transcribed |
| 19.74 | person | no | uncertain | Scribe + Speechmatics, overlap | uncertain |
| 20.14 | person | no | counted | Scribe + Speechmatics, person's speaker | counted |
| 20.74 | other | no | no | other_speaker | no |
| 21.46 | person | no | counted | Scribe + Speechmatics, person's speaker | counted |
| 22.30 | person | counted | counted | Scribe + Speechmatics, person's speaker | counted |
| 23.8-24.7 (nothing said) | none | no | **no** | heard_by_one (only Scribe; the others heard speech there) | no |
| 27.44 | other | no | uncertain | all three, overlap | **counted** in H1/H3 (person's speaker), not counted in H2 (no_speaker) |
| 28.10 | person | no | no | heard_by_one (only Scribe) | no |
| 38.20 / 46.64 / 47.74 | person | counted | counted | all three | counted |
| 49-54 (3 words) | other | no | no | other_speaker ×2, no_speaker ×1 | no |

28.10 was `uncertain` in D (heard once, overlapped voice); under the majority rule it is `heard_by_one` before the
voice is looked at.

## Speech-aligned regions (owner decision 21) vs fixed 15 s

| | fixed 15 s (E/F) | speech-aligned v1 (G) | speech-aligned (H, committed) |
|---|---|---|---|
| region lengths | 15, 15, 15, 9.4 s | 10.0, 13.6, 11.7 s + 19.0 s to the end | 10.0, 13.6, 11.7 s + 19.0 s to the end |
| min / median / max of closed regions | 15 / 15 / 15 s | 10.0 / 11.7 / 13.6 s | 10.0 / 11.7 / 13.6 s |
| reasons | fixed | pause, cap_short_pause ×2, end | pause ×3, end |
| utterances a boundary cut (Scribe / Speechmatics / gpt) | 2 / 1 / 2 every run | 0 / 0 / 0 | 0 / 0 / 0 |
| pyannoteAI jobs per run | 4 | 4 | 4 |
| provisional -> final p50 / p95 | 15.2-20.0 / 21.8-23.6 s | 45.7-50.4 / 62.8-67.6 s | 21.0-23.9 / 36.8-41.0 s |

- **G (first version)** closed a region only at a gap of 500 ms in which none of the three recognisers wrote anything.
  This clip has no such gap in continuous shop talk, so regions 2 and 3 waited for the 30 s cap before they closed, and
  latency tripled. The committed version (H) also counts the end of an utterance as a pause, which is the owner's
  wording ("end of a committed utterance / silence ≥ a pause"). The boundaries fell at the same places (10.0, 23.64,
  35.39 s), now each closed at a pause.
- **H is still slower than fixed** (p50 +5 s, p95 +15 s). The last region (35.4 s to the end, 19 s) is decided only at
  the end of the input, and region 2 closes at 23.6 s once all three recognisers are complete past it.
- **Windows moved, and attribution moved with them.** Region 3 ([23.6, 35.4] s, window [8.6, 40.4] s) gave 27.44 s
  (another voice, overlapped) to the person's speaker in H1/H3 and G1-G3. With fixed regions (window [0, 35] s) the
  same word was overlapped and `uncertain`. The voice rule is unchanged; Precision-3's labels depend on the window it
  is given.

## Chosen, and on what

Nothing was chosen on this clip:
- `min_ms` 10 s, `max_ms` 20 s and `cap_ms` 30 s bracket the earlier fixed 15 s.
- `pause_ms` 500 ms is GAMST's endpointer pause.
- `decide: majority` and `keep_single` are the owner's rule.
- The change from G to H follows the owner's wording, not the numbers: it was made after seeing G's latency, and both
  versions are reported.

## Cost (list prices)

12 runs (E0, E2, E3, F1-F3, G1-G3, H1-H3):
- Scribe with keyterms: about $0.049
- gpt-live-transcribe: about $0.185
- Speechmatics: 12 × 54.4 s at $0.40/h, about $0.073
- pyannoteAI Precision-3: 48 jobs, about 0.043 EUR

Total about $0.35.
