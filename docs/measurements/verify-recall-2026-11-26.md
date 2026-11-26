# Verification lane: keeping words heard once when the voice is clear, and the uncertain state (2026-11-26)

Follows [catch-up-verification-2026-11-26.md](catch-up-verification-2026-11-26.md) (run B2 there, before this change).
Owner decisions: a word only the first hearing has is not revoked when the voice evidence for the person is clear;
overlapped words whose voice is unclear are `uncertain` (not counted, not revoked); pyannoteAI has its own consent
(`--allow-remote-diarization`); regions stay 15 s. Contracts: [../catch-up-verification.md](../catch-up-verification.md).

Run exactly as B2, three times (D1-D3): `gamst recipe run person-term-count --with verification --allow-remote-asr
--allow-remote-diarization --as-live --audio a.wav --voice voice.json --terms 뽀또`. L1 Scribe v3 Live + keyterms,
per-word CAM++; L4 gpt-live-transcribe + keyword + Qwen3-ForcedAligner, Precision-3 windows, CAM++ per speaker. Rule
defaults: `keep_clear_voice` on, `clear: either`, `clear_margin` 0.1, threshold 0.5, `near_ms` 1000, `overlap_share` 0.3;
vote off. Truth as before (±0.6 s).

## Totals

| run | final counted / right | precision | recall | uncertain (not counted) | provisional counted / right | provisional -> final p50 / p95 |
|---|---|---|---|---|---|---|
| B2 (before) | 6 / 6 | 1.00 | 0.60 | - | 4 / 4 | 18.1 / 21.2 s |
| D1 | 8 / 7 | 0.875 | 0.70 | 4 | 4 / 4 | 17.6 / 21.8 s |
| D2 | 8 / 7 | 0.875 | 0.70 | 4 | 4 / 4 | 16.0 / 21.3 s |
| D3 | 8 / 7 | 0.875 | 0.70 | 4 | 4 / 4 | 14.8 / 21.4 s |
| D4 (after the representation change below) | 8 / 7 | 0.875 | 0.70 | 4 | 4 / 4 | 14.7 / 21.4 s |

After D1-D3 one representation was changed and D4 run to check it: a `term.other_voice` event that becomes uncertain is
revoked (`uncertain_overlap`) and a `term.spoken` in state `uncertain` replaces it, so every uncertain word is a
`term.spoken` (`count_events(event="term.spoken")` gives `states.uncertain` 4 in D4; in D1-D3 the 4 were
`term.other_voice` events in state `uncertain`). D4 decided every word as D3 did.

The false count is the same in all three runs: the word Scribe invented at 24.16 s (nothing said at 23.8-24.7 s). The
second recogniser did not write it, but the person was speaking there: Precision-3 has the person's speaker (score
0.598) alone over it, so the clear-voice rule keeps it (`heard_once+clear_voice+person_speaker`; its per-word score is
0.36).

## Per labelled word (D1 / D2 / D3)

| truth end | who | L1 score | provisional | final D1 | final D2 | final D3 | basis (D3) |
|---|---|---|---|---|---|---|---|
| 12.04 | person | 0.35 | no | counted | counted | counted | heard twice, person's speaker |
| 12.64 | other | 0.41 | no | uncertain | uncertain | uncertain | heard twice, overlap, score below 0.5 |
| 14.0 | person | - | not transcribed | - | - | - | neither recogniser wrote it |
| 19.74 | person | 0.18 | no | uncertain | uncertain | uncertain | heard once (twice in D1/D2), overlap |
| 20.14 | person | 0.24 | no | counted | counted | counted | heard once, clear (person's speaker alone); twice in D1/D2 |
| 20.34 | other | - | not transcribed | - | - | - | |
| 20.74 | other | 0.30 | no | no | no | no | other_speaker (D3: heard once, not_heard + other_speaker) |
| 21.46 | person | 0.39 | no | counted | counted | counted | heard once, clear (person's speaker alone); twice in D1/D2 |
| 22.30 | person | 0.51 | counted | counted | counted | counted | heard once, clear (D3); twice in D1/D2 |
| 23.8-24.7 (nothing) | none | 0.36 | no | **counted** | **counted** | **counted** | heard once, clear (person's speaker alone): false count |
| 27.44 | other | 0.14 | no | uncertain | uncertain | uncertain | heard twice, overlap |
| 28.10 | person | 0.36 | no | uncertain | uncertain | uncertain | heard once, overlap, score below the clear margin |
| 38.20 / 46.64 / 47.74 | person | 0.54 / 0.55 / 0.51 | counted | counted | counted | counted | heard twice |
| 49-54 (3 words) | other | 0.17-0.22 | no | no | no | no | other_speaker ×2, no_speaker ×1 |

Of the previous misses: **12.04 and 21.46 are counted in all three runs; 19.74 and 28.10 are `uncertain`** (shown, not
counted, not revoked: a later lane may settle them). In D1/D2 the second recogniser wrote 21.46 itself (heard twice); in
D3 it did not, and the clear-voice rule kept it. **This is not without a false count**: the invented 24.16 s word is
counted in all three runs (it was refused in B2 as `not_heard`).

## What the other `clear` settings would have done (computed from the recorded verdicts, not run)

Every word heard only once had a per-word score below `threshold + clear_margin` = 0.6 (0.18-0.51), so `clear: score`
and `clear: both` revoke all of them; `clear: speaker` decides as `either` did.

| clear | D1 / D2 final counted / right | D3 final counted / right |
|---|---|---|
| either (default), speaker | 8 / 7 | 8 / 7 |
| score, both | 7 / 7 (24.16 revoked) | 4 / 4 (20.14, 21.46, 22.30 and 24.16 revoked) |

On this clip the per-speaker identification cannot tell the invented word from the person's real words: it was said
(or rather, written) while the person spoke. A third hearing (`vote`, off; ready for the AssemblyAI / Deepgram /
Speechmatics adapters) or `clear: both` are the generic ways to require more than the voice for a word only one
recogniser wrote. Not chosen here.

## Chosen, and on what

Nothing was set on this clip: `clear: either` and `clear_margin` 0.1 follow the owner's wording ("the person's speaker
is the one active at the word, or the per-word score is at least threshold + margin") and were set before these runs;
the rest are the earlier defaults.

## Cost (list prices)

4 runs: Scribe v3 Live with keyterms 4 × 54.4 s about $0.016; gpt-live-transcribe 4 × 54.3 s about $0.062; pyannoteAI
Precision-3 16 jobs, 457 s billed, about 0.014 EUR. Total about $0.09.
