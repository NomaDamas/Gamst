# Four clips scored against the owner's labels (2026-11-26)

The owner labelled every place a recogniser wrote the term (labelling tool `try-it/label_tool`), and added words no recogniser
wrote. GAMST run: Scribe v3 Live + verification + vote (gpt-live, Speechmatics) + Precision-3, voice reference from another
recording of the person (no per-clip few-shot), speech-aligned regions.

| clip | person's words (owner) | heard by GAMST's recognisers | counted right | counted wrong | heard but given to someone else | other-voice places | nothing-said places |
|---|---|---|---|---|---|---|---|
| 뽀또 | 10 | 9 | 7 | 1 | 2 | 6 | 1 |
| 몇기야 | 6 | 6 | 1 | 0 | 5 | 2 | 0 |
| 시발 (5 spellings) | 7 | 7 | 0 | 0 | 7 | 0 | 0 |
| 달 (each repetition) | 23 | 23 | 0 | 0 | 23 | 0 | 2 |
| total | 46 | 45 (98 %) | 8 | 1 | 37 | 8 | 3 |

Hearing the words works (45 of 46). Deciding that the person said them fails with a voice reference from another recording
(8 of 45). In two clips nobody else says the term, so counting every heard word would have been right there.

## v8 (SOTA first: a word only the primary recogniser heard counts; partial-hypothesis counts; same-recording voice with quality fallback)

Two runs each, identical. Final right / counted (words inside the voice enrolment excluded): 뽀또 7/9, 몇기야 3/4, 시발 7/7,
달 13/13 — total 30/33 (recall 91 %, precision 91 %; v7: 28/30). Remaining: 뽀또 24.16 (invented, counted), 27.42 (other voice
given to the person by diarization), 14.05 (no recogniser heard it); 몇기야 39.24 (left out of Scribe's final). Word → HUD ~0.7 s.

## Search mode (owner decision 33: never miss the person, over-count rather than miss), 2026-11-27

`--mode search`, otherwise v8a's settings and voice file. 뽀또, one run (`search1`): final 8 right of 15 counted, the
person said 9 (recall 89 %, precision 53 %; count mode v8a/v8b: 7 of 9 counted, recall 78 %, precision 78 %). Counted as
candidates: 12.48 and 19.58 (uncertain_overlap: other / person), 20.56, 49.58, 50.22 (other_speaker) and 52.02
(no_speaker), all other voices but 19.58; none had every voice score 0.3 under the threshold, so none was revoked.
Still missed: 14.05 (no recogniser heard it). Word → HUD p50 1.1 s.

The other three clips, re-scored offline from the v8a/v8b reports (bounds: what search mode counts that count mode left
uncertain, and also what it revoked except `partial_withdrawn`): 몇기야 3/4, 시발 7/7, 달 13/13 in both bounds (search mode
changes nothing there: their only revocations were `partial_withdrawn`). All four clips: recall 31/33 (94 %), precision
31/35 (89 %) to 31/39 (79 %); count mode 30/33 (91 %), 30/33 (91 %). The 뽀또 run reached the upper bound (8/15).
