# Handoff: feat/realtime-tse (realtime target speaker extraction on overlapped speech)

Worktree `.local-run/wt-tse`, branch `feat/realtime-tse` from `feat/target-speaker-keyword-and-pose-count` (f26e7e4).
Design: [realtime-tse.md](realtime-tse.md). Owner decision 2026-11-26: on overlapped speech, extract the enrolled voice
with a SOTA TSE model (few-shot clip; NC weights opt-in), in real time (catch-up workers if needed), decide the
overlapped words from the extracted audio.

## SOTA research (2026-11-26, web; details in the coordinator report)

- SLT 2026 REAL-TSE challenge (arXiv 2607.15198, https://real-tse.github.io/challenge/): real mixtures (zh/en,
  meetings, cafes, homes, cars, a phone channel). Offline track: 1 MERL (TER 0.462; no weights), 2 YiJiaHe = PS4
  (TER 0.509; **released, Apache-2.0**), 3 CARTSE. Online track (<=100 ms) top ~0.59 TER, no weights released.
  Baseline WeSep BSRNN (Libri2Mix) 0.74. The PS4 = YiJiaHe identification is inferred (affiliation + matching numbers).
- REAL-T (Interspeech 2025): VoxCeleb-trained BSRNN best on real data (zh 57.6 %, en 66.5 % TER); USEF-TFGridNet
  68.0 / 87.3 %; TSELM collapsed (331 / 192 %).
- PS4: HF `TaurenMountain/PS4` @ 8feb7a1b, `checkpoint_epoch037.pt` 282746518 B sha256 3d5621c8...,
  `inference.py` sha256 172f2233...; BSRNN + ECAPA-TDNN, 16 kHz, not causal; trained on AISHELL-4, AliMeeting, AMI,
  CHiME-6 (CHiME-6 data is research-licensed: owner should read).
- Others: USEF-TSE / USEF-Laura-TSE CC BY-NC 4.0; TSELM, SoloSpeech CC BY-NC-SA 4.0 (generative, no real-data WER);
  WeSep code Apache-2.0; REAL-TSE causal baselines Apache-2.0 (Google Drive, no pinned URL); MossFormer2_SS_16K
  Apache-2.0; TIGER MIT/Apache; SepReformer Apache-2.0 (WSJ0 only). No TSE model trained on Korean.

## Built (all inside the Pipeline / registry / recipe)

- `gamst/providers/overlap_tse.py`: `MultiVoice` (where voices meet, from live diarization slot activity; rules
  near/overlap), `TargetSpeechWords` (TSE lane, own lane, forkable for catch-up, PS4 per stretch, fork of the live
  recogniser per stretch, CAM++ scores), `TSEDecision` (person / tse_not_heard / uncertain_overlap; heard-only adds).
- `gamst/providers/target_speech.py`: `PS4Extractor`. `model_workers.py`: long enrollment clips sent in rows.
- `word_verification.py`: L4 uses the TSE decision for overlapped words.
- Registry `audio.target_speech` candidate `ps4.bsrnn_real` (env `ps4-tse`); setup mapping in `crew/setup.py`.
- Recipe `person-term-count`: items `tse`, `tse_decision` (part `tse`), `consumes_with` keys "a+b" (generic, in
  `recipes/__init__.py`). Parts `tse-words`, `tse-decide` in `recipes/perception_parts.py`.
- Tests: `tests/test_overlap_tse.py` (stand-ins). Scripts: `scripts/qa_tse.py` (per-word table), `scripts/qa_diarization.py overlap`.

## Downloads / environments made (outside the repo)

- `.local-run/try-it/models/ps4.tse/` (checkpoint 283 MB, inference.py, README.md).
- `.local-run/try-it/venvs/ps4-tse` (uv, Python 3.11, torch 2.8.0, torchaudio 2.8.0, numpy 2.2.6; ~430 MB).
- `.local-run/try-it/profile-tse.json` (stream-openai profile + nemotron3 + ps4, threads 4).
- Run helpers: `try-it/tse-run.sh`, `try-it/tse-measure.sh`, `try-it/rss-sample.sh`. Runs in `try-it/testbed/tse/runs/`.

## Measured so far (a.wav, 54 s, paced live)

- PS4 standalone CPU (10 threads): 2 s window 1.17 s, 4 s 1.95 s, 10 s 5.76 s; whole clip RTF 0.60, RSS 5.6 GB
  (whole clip). MPS fails in the publisher's code.
- Overlap detection recall on the 8 overlapped labelled words: Nemotron-3 low_latency live, rule near (1.5 s): 7/8
  (misses 12.04), 21.8 s of 54 flagged, also flags person-alone 14.0, 21.46, 47.74; rule overlap: 3/8, 6.9 s.
  Precision-3 (offline) near: 8/8; Sortformer near 5/8.
- Run d3 (`--with tse`, no verification): 9 stretches; TSE recogniser heard the person's 19.74 and 20.14 but also the
  other voices' 12.64, 20.74, 27.44; not 28.10. CAM++ scores on extracted audio 0.21-0.46: every decision at
  threshold 0.5 was `uncertain` (7 words); final count unchanged 4/4. Lane: extraction 17.5 s for 28.9 s of
  extracted audio, hearing ~4 s per stretch (serial then; now concurrent), 3 catch-up TODOs all done, 0 gaps;
  provisional latency added p50 12.1 s / p95 17 s (fixed since: words with no possible stretch pass at once).

- Run d4 (hearing concurrent, early release, workers 2): 9 stretches, 8 words decided, all `uncertain`; extraction
  in the live module 27.2 s for 23.3 s of audio (CPU contention: live + fork PS4 workers at 4 threads + Nemotron);
  3 TODOs, TODO->done p50 18.8 s / p95 30.4 s, 0 gaps; up to 4 PS4 worker processes alive at once, 6.2 GB together,
  1.7 GB each at most; decision wait p50 9.4 s / p95 14.3 s. Recipe default set back to workers 1.

- Final series (details: docs/measurements/tse-2026-11-26.md): v0 (verification only) and v1, v2 (tse +
  verification): provisional 4/4, final 4/4 in all three, identical words. TSE: 8 words decided, all uncertain; PS4
  leaks the other voices' words (12.64, 20.74, 27.44) into the extracted sound; g.wav enrollment (d5g) the same.
  v2 lane: extraction 0.63 x real time with 1 worker, 4 catch-up TODOs (p50 9.2 s, p95 34.2 s), 0 gaps, worker
  1.7 GB; overlapped words decided 10-21 s after being said. PS4 alone RTF 0.53-1.05 by threads/window.
- Open issue: with --with tse the live Scribe session was closed by the server at the end of the input (51.84 s) in
  4 of 5 runs; not isolated (4 concurrent sessions in a probe were fine).
- Unit tests (try-it host interpreter, numpy + websockets): 1677 run, 3 errors, the same 3 as the base commit
  (test_watch, test_streaming_semantic, test_yoloe bridge); the demo venv (15 errors) was not used.

## Left

- Nothing required. Possible next steps: the owner's clip set for thresholds/rule; a GPU worker; the Scribe close issue.

## Rerun

```text
cd .local-run/try-it
./guard.sh tse ./tse-measure.sh NAME --with tse [--with verification]
../try-it/host/bin/python ../wt-tse/scripts/qa_tse.py testbed/tse/runs/NAME.db testbed/tse/runs/NAME.json
```
