# Overlapped speech, live: target speaker extraction (owner decision, 2026-11-26)

When voices overlap, GAMST extracts the enrolled person's voice from the mixture with a state-of-the-art
target-speaker-extraction (TSE) model conditioned on the owner's few-shot clip, in real time, and decides the overlapped
words from the extracted sound. If one instance cannot keep up, the Pipeline's catch-up (parallel workers) works the
backlog off; it is not a recording-only feature. Builds on [realtime-attribution-decisions.md](realtime-attribution-decisions.md)
and [catch-up-verification.md](catch-up-verification.md). Code: `gamst/providers/overlap_tse.py`; recipe part `--with tse`.
Measured: [measurements/tse-2026-11-26.md](measurements/tse-2026-11-26.md).

```text
microphone ─┬─▶ L1 asr (Scribe v3 Live) ─▶ speaker_check (per-word CAM++ on the room) ─▶ tse_decision ─▶ terms (provisional)
            │                                                                              ▲          └─▶ L4 verification
            ├─▶ diarization (Nemotron-3 slots, live) ─▶ where voices meet ─┐               │ records of the lane
            │                                                              ▼               │
            └──────────────────────────────▶ tse lane (own lane, forkable): per stretch ───┘
                                                 extract the person (PS4, few-shot clip)
                                                 └▶ fork of the live recogniser hears the extracted sound
                                                 └▶ CAM++ scores the extracted sound
```

## Parts

- **Where voices meet** (`MultiVoice`, math). From the live diarization's slot activity (`audio.speaker_activity`, the
  `diarization` item of `--with tracks`; any `audio.diarization` candidate). Slot identities are not trusted (owner
  decision: live slots mixed people on the test clip), only how many slots are active. Rule `near` (default): two or
  more slots each active for at least `min_slot_ms` (100) within `context_ms` (1500) of a frame -- the per-word voice
  score's own context, so exactly where that score embeds two voices. Rule `overlap`: two or more slots active in the
  same 10 ms frame. A stretch is closed once the diarization is `context_ms` past its end.
- **The TSE lane** (`TargetSpeechWords`, item `tse`, part `tse-words`, role `audio.target_speech`). Its own lane; reads
  the microphone and the slot activity. For each closed stretch (in pieces of at most `max_window_ms`, 8 s) it takes
  the kept audio from `margin_ms` (1500) before to `margin_ms` after it, extracts the person with the extractor worker
  (the enrollment clip, `--voice-wav`, is sent to the worker once and stays in memory), feeds the extracted sound to its
  fork of the live recogniser (`asr.fork()`: a new session of the same recogniser with the same key terms and
  language; a session the server closed is opened again and the stretch heard once more), and scores the extracted
  sound with the speaker model of `speaker_check` (CAM++): each counted word heard there (`word_context_ms` 1500
  around it) and the whole extracted span. Output `speech.target_words`, one record per stretch: `start_ms`/`end_ms`
  (the stretch), `extract_ms`, `words` (heard in the extracted sound, with times), `terms` (counted words heard there,
  each with its score), `score` (whole span), `level_db` / `mixture_db`, `took_ms` (extract, asr, score). No audio, no
  embedding.
- **Catch-up.** The lane has `fork()`, `snapshot()` (`pause_ms`: `margin_ms` before the first stretch not done or still
  open) and `max_forks` = `workers` (item setting; each fork opens its own extractor worker when `workers` > 1, else
  shares the one). When the lane falls more than `lag_ms` behind, the Pipeline skips it to now and gives the skipped
  capture range to a fork (docs/catch-up-verification.md); forks share the slot activity the live module reads and the
  list of records. The live module leaves stretches whose middle lies in a range the Pipeline claimed for a worker;
  a fork does the stretches whose middle lies in what it heard. Ownership by capture time removes duplicates at edges.
- **The decision** (`TSEDecision`, item `tse_decision`, part `tse-decide`, math). Between `speaker_check` and `terms`
  / the verification lane. Each utterance waits until `Pipeline.complete_until('tse')` has passed its last word (the
  lane holds its completeness at the start of a stretch it still owes, an open stretch, and the pending catch-up
  ranges), so no record can be missed. A counted word inside a stretch is decided from that stretch's record; every
  other word passes unchanged (`mixture_score` keeps the room score of a decided word):

| extracted sound | decision | `match` | `reason` |
|---|---|---|---|
| the same term heard within `near_ms` (1000) and its score >= `threshold` (0.5) | person | true | |
| heard, score below `threshold` | uncertain | false | `uncertain_overlap` |
| not heard, whole span scored >= `threshold` (the extraction is the person) | not the person | false | `tse_not_heard` |
| not heard, span not identified as the person | uncertain | false | `uncertain_overlap` |

  A counted word heard only in the extracted sound, scored >= `threshold` and not written by the live recogniser
  there, is added as a provisional utterance (`basis: tse_heard_only`; `add_heard`, on). The words carry
  `tse: {decision, heard, score, window_score, window}` and `decided_by: "tse"`.
- **L4.** With `--with verification`, the verification lane reads `tse_decision` instead of `speaker_check`. For a
  word its diarization windows call overlapped (person and another speaker each over `overlap_share`), the TSE
  decision replaces the per-word score on the room audio: person -> final; not the person -> revoked `other_speaker`;
  uncertain -> revoked `uncertain_overlap`. Its other rules (heard twice, `not_heard`, `no_speech`, the person's or
  another speaker dominant) are unchanged.

## Run

```text
gamst recipe run person-term-count --with tse [--with verification] --allow-remote-asr \
    --voice voice.json --voice-wav OWNER_CLIP.wav --terms ...
```

`--with tse` brings the live diarization (`diarization` item) with it. The `audio.target_speech` role must be set up
(`ps4.bsrnn_real`, its own environment `ps4-tse`: torch 2.8.0, torchaudio 2.8.0). The extracted sound goes to the
recogniser's provider like the live sound (Scribe v3 Live is billed per second: overlapped stretches with margins are
billed once more).

## Models (state of the art, 2026-11; sources in the measurement)

- Default `ps4.bsrnn_real`: PS4 (TaurenMountain/PS4, Apache-2.0), BSRNN + ECAPA-TDNN fine-tuned on real meeting
  recordings (AISHELL-4, AliMeeting, AMI, CHiME-6), 2nd in the SLT 2026 REAL-TSE offline track and the best system
  with released weights (arXiv 2607.15198, 2607.08111). Runs on this Mac's CPU (not MPS: the publisher's code fails
  there). Not causal: it runs on windows, which is why the lane works per stretch.
- `usef.tfgridnet` (CC BY-NC 4.0, opt-in `allow_noncommercial`), `wesep.bsrnn_ecapa` (Apache-2.0 code): registered
  earlier; both did worse than VoxCeleb-trained BSRNN on real conversations in REAL-T (Interspeech 2025).
- Best overall on real data: MERL's system (REAL-TSE offline track 1st, TER 0.462 vs PS4's 0.509) has no released
  weights; it cannot be registered. No released TSE model was trained on Korean.

## Measured (one clip; [measurements/tse-2026-11-26.md](measurements/tse-2026-11-26.md))

On the 54 s shop clip the lane ran live with one worker (extraction 0.63 x real time, the backlog of the overlap
bursts caught up by the Pipeline, nothing given up) and decided 8 words, all `uncertain`: PS4 kept the other voices'
words next to the person in the extracted sound and the extracted sound's voice score stayed under the threshold.
Counts with and without `--with tse` were the same (final 4 / 4 of 10). Overlapped words are decided 10-21 s after
they are said on this Mac's CPU (about 4 s without the lane). A second worker process on the same CPU did not help.

## Not decided here

- `rule`, `context_ms`, `margin_ms`, `threshold` and `near_ms` are module defaults, not chosen on a clip; they need the
  owner's set of clips.
- Whether `workers` should be 1 on a laptop (each worker is another model process) or follow free memory.
