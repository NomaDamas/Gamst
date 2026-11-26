# Catch-up lanes and the verification lane (owner design, 2026-11-26)

Implements [realtime-attribution-decisions.md](realtime-attribution-decisions.md) (decision 1, and "Details agreed with
the owner"). Everything here is inside the existing Pipeline, records and revision states; nothing is specific to a
word or a clip, and every number below is a parameter.

```text
microphone ─┬─▶ L1 live lane (asr: any forkable module) ── words, provisional ──▶ speaker_check (per-word voice) ─▶ terms
            │        │ falls behind > lag_ms: skips to now; the skipped capture range is a TODO
            │        ▼
            │   L2 catch-up workers (pool) ── fork of the module, its state at the TODO's start ──▶ same records,
            │                                                                       at their capture times
            ├─▶ audio memory (every frame kept until nobody holds it; memory, then a spill file)
            │
            └─▶ L4 verification lane (serial, in capture order, behind; only complete stretches)
                   second recogniser + diarization windows + few-shot voice ─▶ final / revoked (reason) / added
```

## L1 and L2: a live lane that falls behind (`gamst/pipeline.py`)

- A module in a lane of its own that has `fork()` catches up (`add_module(..., catch_up=None | False | {...})`; recipe
  item `"catch_up"`). Settings (`CATCH_UP`): `lag_ms` 2000, `max_lookback_ms` 10000, `tail_ms` 1500, `workers` 2,
  `speed` None (as fast as the module takes it). Only live runs; a recorded input still waits for the lanes.
- When the oldest audio frame waiting in the lane is more than `lag_ms` older than now, the lane **skips to now**: the
  waiting audio frames are taken out of the lane, and their capture range `[start, end]` becomes a TODO. Nothing is
  dropped: every audio frame of the input is in the input's `AudioMemory` (`gamst/providers/audio_memory.py`:
  64 MB in memory, older frames spill to an unnamed temporary file; a frame is let go only when it is older than
  `keep_ms` (30 s) and than every hold). Video lanes keep thinning as before.
- Before skipping, the Pipeline takes `snapshot()` of the live module (the state at the TODO's start) and calls its
  `skipped(start, end)`. A worker gets `fork()`, then `restore(snapshot)`, and hears the kept audio from
  `from_ms = max(start - max_lookback_ms, snapshot["pause_ms"])` (the module's last pause: the end of its last
  committed turn; a module that copies its whole state gives no `pause_ms` and starts at `start`) to `end + tail_ms`.
- **Ownership by capture time.** The worker owns `[from_ms, end]`: its results whose middle lies there are committed
  (at their own capture times), its others dropped; the live module's results whose middle lies in a claimed range
  are dropped (`module.trim(value, keep)`, else by the record's `start_ms`/`end_ms`). Duplicates at the edges are
  removed this way; the verification lane pairs words one to one, so a duplicate left at an edge cannot be counted
  twice after verification.
- Pool: `min(catch_up.workers, module.max_forks)`. A hosted streaming session says its provider concurrency
  (`concurrency` in its config, default 2); a local model module says 1 (one model per process on this Mac, under the
  memory guard).
- Completeness: a pending TODO holds `complete_until_ms` of the module (and of everything that reads it) at
  `from_ms - 1`; a record waiting in a lane holds it at the earliest capture time it speaks about (`start_ms`); a
  streaming recogniser holds it at the first sound whose words are not out (`pending_since_ms`: the open turn or an
  unanswered commit). `Pipeline.inputs_complete_until(name)`: what a module reads is complete up to there.
- `status()['modules'][name]['catch_up']`: skips, skipped_ms, done, failed, pending, max_pending, max_pending_ms,
  heard_ms, busy_ms, todo_to_done_ms (p50/p95) and the last TODOs; `status()['audio_memory']`: bytes in memory and
  spilled, `dropped` (0 unless a disk failed). `busy_ms` of every module is the time it spent processing.
- **Hosted recognisers** (`gamst/providers/streaming_asr.py`, `streaming_hosted.py`): `StreamingWords.fork()` opens a
  new session of the same recogniser (its audio is billed too), `snapshot()` = `{pause_ms, context}` (the last
  committed text), `restore()` gives the context as the session's prompt (OpenAI `prompt`, ElevenLabs
  `previous_text`, the last 50 characters), `skipped()` commits the open turn, `trim()` keeps words by capture time.
- Test and debug: `gamst recipe run ... --debug-slowdown MODULE=MS@FROM-TO[,FROM-TO]` sleeps MS before each audio frame
  the module gets in those seconds of the run.

## Provisional words (L1/L2)

L1 is the recipe's `asr` (Scribe v3 Live, the counted words as `keyterms`: `"bias": "terms"`). With `--with verification`, `speaker_check` (the per-word CAM++ score against the owner's few-shot voice) gives each
word a `word_id` and `state: "provisional"`; `terms` counts it at once (`provisional`). Live diarization slots are not
used to decide who said a word (owner decision). Late words of a catch-up worker are scored on the kept audio.

## L4: the verification lane (`gamst/providers/word_verification.py`, part `word-verify`, item `verification`)

Consumes the microphone (as ticks) and `speaker_check`. Its own work runs in capture order, in regions of `step_ms`
(15 s), each only when `inputs_complete_until` has passed it (no TODO, no unanswered turn, no queued record there) by
`near_ms`, so a catch-up worker never changes a word after it was settled.

- **Second recogniser**: `gpt-live-transcribe`, the counted words as `keywords`, Qwen3-ForcedAligner word times
  (`"recogniser"`, `"aligner"`), fed the kept audio as far as it is complete, at most `feed_speed` (1.25) times real
  time: it is a realtime API, and a little faster lets it catch up after a stall (`null`: as soon as complete, in
  bursts).
- **Diarization windows**: pyannoteAI Precision-3 (`"diarizer"`) on `[region - before_ms (15 s), region + after_ms
  (5 s)]` (windows overlap), at most `concurrency` (2) jobs at once. Each speaker's own sound (where it alone is
  active) is cut in `piece_ms` pieces, embedded with CAM++ and its duration-weighted mean scored against the voice;
  speakers at or above `threshold` (0.5, the module default everywhere) are the person; under `min_speaker_ms` of own
  sound a speaker is unidentified.
- **Word rule** (`RULE`, a parameter set): an occurrence of a counted word in the first hearing is paired one to one,
  nearest first, with an occurrence of the same word in the second hearing whose end is within `near_ms` (1000);
  paired -> the voice rule decides. Unpaired, and the second recogniser or a speaker was active there:
  - kept (final) when the voice evidence for the person is **clear** (`keep_clear_voice`, on): `clear` = `speaker` (the
    person's diarized speaker is active at the word and no other speaker over `overlap_share`), `score` (the per-word
    CAM++ score is at least `threshold + clear_margin`, margin 0.1), `either` (default) or `both`;
  - `uncertain` (`uncertain_overlap`) when the person's speaker overlaps another one and the score is not clear;
  - revoked `not_heard` when the voice points elsewhere or is absent.

  Unpaired and nobody spoke there -> revoked `no_speech` (`revoke_no_speech`). An unpaired occurrence of the second
  hearing -> added, final (a `kind: "utterance"` value with `decided_by: "verification"`).
- **Vote** (off: `vote: false`): any `speech.transcript_words` source the item also reads (another recogniser's module,
  e.g. AssemblyAI / Deepgram / Speechmatics adapters once registered) votes. A first-hearing occurrence that a voter also
  heard within `near_ms` counts as heard twice (`heard_by_vote+...`); `vote_adds_once: false` adds a second-hearing-only
  occurrence only when a voter heard it too. Recipe: add the module, list it in the item's `consumes` and set
  `"rule": {"vote": true}`.
- **Voice rule** (`VOICE`) on the word's span (the second hearing's): person speaker over `overlap_share` (0.3) and
  another one too -> overlapped: the per-word CAM++ score (1.5 s around the word) decides, below threshold it is
  `uncertain_overlap`; person dominant -> the person; another identified speaker dominant ->
  `other_speaker`; no speaker or an unidentified one -> the per-word score (`no_speaker`, `speaker_unidentified`).
  `active_share` 0.1: a speaker over less of the word is not active there.
- Outputs (`speech.word_verdicts`): `verdicts` (every provisional word: `match`, `provisional`, `changed`, `basis`,
  `reason`, `heard`, `speaker`, `speaker_score`, `word_score`), added `utterance`s, and one `window` record per window
  (speakers, their own speech in ms and score; no audio, no embedding). `terms` turns verdicts into `event.revision`
  records (`final`, or `revoked` with the reason; `basis` is kept on the revision), only for provisional events.
- **Uncertain** (`uncertain_overlap`): the verdict says `uncertain: true`; `terms` gives the event the revision state
  `uncertain`: not counted, not revoked (never "not said"), kept open, so a later lane's verdict for its `word_id`s
  settles it (`final` / `revoked`; the newest revision decides). An added word that is uncertain is committed as
  `term.spoken` with `state: "uncertain"` and a `word_id`. `count_events` gives `states.uncertain` beside the count and
  `where={"state": "uncertain"}` lists them; `get_timeline` shows the state; the report's `states` has it; bench and
  scoring leave uncertain words out of what is counted.
- Opt-in: `--with verification --allow-remote-asr --allow-remote-diarization` (the first for the second recogniser, the
  second for pyannoteAI; each prints its price before audio is sent).

## Not decided here

- The thresholds and windows above are module defaults, not chosen on a clip; they need the owner's set of clips.
