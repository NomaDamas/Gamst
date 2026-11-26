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

Consumes the microphone (as ticks), `speaker_check` and any voter. Its own work runs in capture order, in **regions
that follow speech** (owner decision 21), each decided only when `inputs_complete_until` has passed it (no TODO, no
unanswered turn, no queued record there) by `near_ms`, so a catch-up worker never changes a word after it was settled.

- **Regions** (`WINDOWS`): a region closes at a speech pause once it is `min_ms` (10 s) long and at most `max_ms`
  (20 s): the middle of a gap between two utterances (an utterance of some recogniser ended there and none was going
  on), or, after the last utterance, `pause_ms / 2` into a silence of at least `pause_ms` (500 ms, the endpointer's
  pause). Without one by `max_ms` it closes at the next one up to `cap_ms` (30 s, reason `max_extended`); only when
  speech runs on past `cap_ms` is it cut there (`cap_split`, the only case that splits an utterance, and the region
  record says so). A pause counts only once every recogniser (first hearing, second, voters) is complete past it. The
  last region runs to the end of the input (`end`). Each region's reason and length are in the `window` record
  (`region`) and in the lane's status (`regions`, `straddling`: utterances a boundary cuts, `diarize_jobs`).
  `step_ms` gives the old fixed regions.
- **Second recogniser**: `gpt-live-transcribe`, the counted words as `keywords`, Qwen3-ForcedAligner word times
  (`"recogniser"`, `"aligner"`), fed the kept audio as far as it is complete, at most `feed_speed` (1.25) times real
  time: it is a realtime API, and a little faster lets it catch up after a stall (`null`: as soon as complete, in
  bursts).
- **Diarization windows**: pyannoteAI Precision-3 (`"diarizer"`) on `[region - before_ms (15 s), region + after_ms
  (5 s)]` (windows overlap, so speaker labels have context; speakers are re-identified in each window by the voice), at most `concurrency` (2) jobs at once. Each speaker's own sound (where it alone is
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
- **Primary first** (`decide: "primary"`, owner decision 31: "the state of the art comes first; weaker models must not
  win by a vote"; the recipe sets it, with `vote`, when the item reads another recogniser: `--with vote` adds
  `vote_asr`, Speechmatics Enhanced with the words as additional vocabulary). The first hearing is the **primary**
  recogniser: the recipe's live `asr` item, i.e. the registry's `audio.asr` default, chosen by published results (today
  ElevenLabs Scribe v3 Live). A primary word another recogniser also wrote within `near_ms` is said (the voice rule
  decides who). A word only the primary wrote **stands** (the voice rule decides who) unless it is **contested**: the
  other recognisers (`contest_by: "all"` of those the item reads, default, or `"any"`) wrote other words within
  `near_ms` of it, i.e. heard speech there and wrote something else. A contested word is **vetoed** (revoked, reason
  `vetoed`) only when the primary's own confidence in it is low too (`low_confidence`, a probability: the lowest
  `confidence` or `exp(logprob)` of its words; default `null` = no veto, because on the owner clips the primary's
  confidence does not separate its invented word from true ones, see "Primary confidence" below); otherwise it is
  `uncertain` (`contested: "uncertain"`, default: reason `contested`, shown, not counted, never "not said") or stands
  (`contested: "stand"`). A contested word the voice gives to another speaker stays that speaker's (`other_speaker`).
  Words only the other recognisers wrote are added as under majority (`add: "agreed"`: two of them, or one alone with a
  clear voice) or whenever one wrote it (`add: "any"`). Verdict bases: `primary+uncontested+<voice>`,
  `primary+contested(second+third)+<voice>`, `...+low_confidence`; the verdict carries `primary_confidence` when the
  recogniser gave one.
- **Majority** (`decide: "majority"`, an option since decision 31; before, the recipe default with voters): an occurrence is **said** when at
  least two of the recognisers (the first hearing, the second one, the voters) wrote the same word within `near_ms`
  (paired one to one, nearest first); the voice rule then decides who. An occurrence only one of them wrote is revoked
  (`heard_by_one`), or not added, unless `keep_single` (on) and no other recogniser wrote anything within `near_ms` of it
  and the voice is clear (`clear`, as above); where nobody spoke it is `no_speech`; unclear overlap stays `uncertain`.
- **Voice rule** (`VOICE`) on the word's span (the second hearing's): person speaker over `overlap_share` (0.3) and
  another one too -> overlapped: the per-word CAM++ score (1.5 s around the word) decides, below threshold it is
  `uncertain_overlap`; with `--with tse` the TSE lane's decision on the extracted voice replaces that per-word
  score for such a word ([realtime-tse.md](realtime-tse.md)); person dominant -> the person; another identified speaker dominant ->
  `other_speaker`; no speaker or an unidentified one -> the per-word score (`no_speaker`, `speaker_unidentified`).
  `active_share` 0.1: a speaker over less of the word is not active there.
- Outputs (`speech.word_verdicts`): `verdicts` (every provisional word: `match`, `provisional`, `changed`, `basis`,
  `reason`, `heard`, `speaker`, `speaker_score`, `word_score`), added `utterance`s, and one `window` record per window
  (speakers, their own speech in ms and score; no audio, no embedding). `terms` turns verdicts into `event.revision`
  records (`final`, or `revoked` with the reason; `basis` is kept on the revision), only for provisional events.
- **Uncertain** (`uncertain_overlap`): the verdict says `uncertain: true`; `terms` gives the event the revision state
  `uncertain`: not counted, not revoked (never "not said"), kept open, so a later lane's verdict for its `word_id`s
  settles it (`final` / `revoked`; the newest revision decides). An uncertain word is always a `term.spoken` in state
  `uncertain`: a provisional `term.spoken` gets that revision; a provisional `term.other_voice` is revoked
  (`uncertain_overlap`) and an uncertain `term.spoken` replaces it; an added word is committed so, with a `word_id`. `count_events` gives `states.uncertain` beside the count and
  `where={"state": "uncertain"}` lists them; `get_timeline` shows the state; the report's `states` has it; bench and
  scoring leave uncertain words out of what is counted.
- Opt-in: `--with verification --allow-remote-asr --allow-remote-diarization` (the first for the second recogniser, the
  second for pyannoteAI; each prints its price before audio is sent).

## Primary confidence (measured 2026-11-27, decides `low_confidence`)

Scribe v3 Live gives every committed word a `logprob` (raw WebSocket, `committed_transcript_with_timestamps.words[]` =
`{text, start, end, type, speaker_id, logprob, characters, channel_index}`; checked with the real API on the four owner
clips, one pass each, VAD commits, the counted word as keyterm). The live adapter (`streaming_hosted.ElevenLabsSession`)
does not pass it on yet (out of this branch's scope); the rule reads `confidence` / `logprob` on the primary's words
whenever an adapter gives them, so with today's adapter `low_confidence` has nothing to read and no word is vetoed.

The counted word's confidence (p = exp(logprob)) against the owner's labels:

| | n | min | median | max |
|---|---|---|---|---|
| the person's words (true) | 23 | 0.02 | 0.47 | 0.63 |
| another voice's words | 7 | 0.33 | 0.57 | 0.67 |

The two words the vote was about: 뽀또 24.16 s (Scribe's word where the owner hears nothing / now labelled another
voice) **0.33** (logprob -1.11); 피파 clip 41.76 s (the person said it) **0.02** (logprob -3.84). The other true words
the majority revoked: 뽀또 28.10 0.11, 달 12.28 0.35. Lower confidence goes with the *true* words here: any threshold
that vetoes 24.16 (> 0.33) also vetoes 41.76, 28.10, 몇기야 17.34 (0.15), 시발 23.18 / 29.50 (0.30) and 달 7.94 (0.18).
So condition (a) cannot separate them on these clips and `low_confidence` defaults to `null` (no veto).

Measured live (4 clips x 3 runs, v7 base; table in docs/handoff-feat-sota-first-rule.md): no word is revoked by the
recognisers any more; the contested words were 24.16 (another voice) and 28.10, 41.76 (the person) in every run and
27.10 (the person) in one: all `uncertain`. Final right/counted is the same as the majority (v7) on every clip and run except 몇기야 p1 (2/3: 27.10 contested this time,
uncertain; v7 had it confirmed by the second recogniser); the
person words the majority revoked are now shown as uncertain instead of "not said". `contested: "stand"` would have
counted 7 more right words and 3 wrong ones (24.16) over the three runs: the owner's choice.

## Not decided here

- The thresholds and windows above are module defaults, not chosen on a clip; they need the owner's set of clips.

## Video: a live lane that samples, a verification lane over every frame (2026-11-27)

The same design for a movement counter (`person-action-count --with verification`). Measured on the owner's shotgun clip:
[testbed-2026-11-26.md](measurements/testbed-2026-11-26.md), "Real time".

```text
camera ─┬─▶ L1 pose (MediaPipe, sample="newest") ── exemplar: counts at once, provisional ──┐
        │        busy: only the newest frame waits; the others are skipped by choice        │
        │                                                                                    ▼
        └─▶ video memory ─▶ verify_pose (RTMW, backlog="spill", every frame, behind) ─▶ verify_exemplar ─▶ verification
             (256 MB in memory, then an unnamed spill file, 8 GB bound)          final / revoked / uncertain / added
```

- `add_module(..., sample="newest")` (recipe item `"sample"`): the lane holds one waiting frame; while the module is busy a
  newer frame replaces it. Skipped frames are `status()['modules'][name]['sampled']['skipped']` (`choice: "newest"`), never a
  gap, so the count stays `complete`. Rates in rules use capture times (`movement_rule.RATE_MS`), so fewer frames per second
  give the same units. The person-action-count recipe's `pose` item samples by default; a recording that is not paced
  still gets every frame.
- `add_module(..., backlog="spill")` (recipe item `"backlog"`): the lane never thins or gives up frames. Its waiting video
  frames live in `Pipeline.video_memory()` (`gamst/providers/video_memory.py`): the camera's arrays up to `memory_bytes`, the
  rest in an unnamed temporary file, read back when the lane gets there; the file starts over whenever nothing is spilled.
  Only beyond `disk_bytes` is a frame not kept, and that is declared (`backlog` gap). `status()['video_memory']` has the
  peaks. A lane that is slower than the camera on average grows this backlog without end; the bound is what stops it.
- `--realtime` (a recording at its own pace) waits for neither kind of lane while frames arrive, as a camera would not.
- `verification` (`gamst/providers/count_verification.py`, part `count-verify`): reads the live counter (`verifies`) and the
  verification counter (`verifier`) in capture order. Once the verifier has seen `near_ms` (150) past a live count it commits,
  under the live counter's capability, `final` (the verifier counted it too, ends within `near_ms`, one to one),
  `revoked` (`not_seen_by_verifier`: it had the person and did not count), `uncertain` (`verifier_saw_no_pose`), or adds a
  `final` count of its own (`decided_by: "verification"`). The live counter commits every count as `provisional` with an
  `event_id` when a `count-verify` item verifies it.
- The verifier's movement: `--verify-exemplar FILE` (enrolled with `--pose` of the verifier's model), default `--exemplar`. A
  text rule needs no second file.
- `--pose CANDIDATE` replaces the live pose model (and how its joints are read); `--set MODULE.KEY=VALUE` changes one item
  setting (`pose.delegate=gpu`, `verify_pose.det_every_ms=0`, `verify_pose.device=cpu`). The verifier's model is set up
  beside the role's default under the profile's `"candidates"`.
