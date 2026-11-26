# Word by one person in real time: speaker tracks, provisional counts, revisions

Implements [realtime-attribution-decisions.md](realtime-attribution-decisions.md) inside the existing Pipeline,
recipe and record rules. `gamst recipe run person-term-count --with tracks ...` (opt-in; the default is unchanged).

```text
microphone ─┬─▶ asr (any streaming ASR) ─────────── speech.transcript_words ─┐
            ├─▶ diarization (speaker-slots) ──────── audio.speaker_activity ──┤
            └──────────────────────────────────────────── audio (ring) ──────┴─▶ tracks (speaker-tracks, lane: CAM++)
                                                                                    │ speech.speaker_tracks
                         terms (TermEvents) ◀── provisional words ──────────────────┤
                               ▲                                                     ▼
                               └──── speech.word_verdicts ◀── verify (track-verify, own lane)
```

## Contracts

**Words in** (`speech.transcript_words`, any ASR): `{utterance_id, status: "final", text, start_ms, end_ms,
words: [{text, start_ms, end_ms}], language?}`, capture times in ms. Each `utterance_id` once; utterances do not
repeat words of earlier ones (a streaming ASR commits each stable stretch once). Partial hypotheses are not read.

**Slot activity in** (`audio.speaker_activity`, the shape of `NemotronDiarization`): a record value
`{"diarization": {"session_id": str, "chunks": [{"start_ms", "end_ms", "activity": [{"slot": int,
"intervals_ms": [[start, end], ...]}]}]}}`. Chunks of a session come in capture order and do not overlap; activity
is known for the whole chunk span. A new `session_id` means new people. A candidate's worker gets 16 kHz mono audio in
whole 10 ms blocks on a gap-free sample clock (`WholeBlocks`); `"mode"` and `"threshold"` of the recipe item go to its
config. Any `audio.diarization` registry candidate whose worker answers this shape plugs in with
`"candidate": "<id>"` on the `diarization` item.

**Tracks out** (`speech.speaker_tracks`): `kind: "slots"` (each slot's `speech_ms`, `score`, `enrolled`, `stable`,
`revision`; no audio, no embedding) and `kind: "utterance"` (the words, each with `word_id`, `slot`, `slot_score`,
`slots_active`, `match`, `decided_by`, `state: "provisional"`).

**Verdicts** (`speech.word_verdicts`): `{kind: "verdicts", verdicts: [{word_id, match, provisional, changed,
settled_by: stable | settled | window | final, reason?, slot, slot_score}]}`.

**Revision states** (`gamst/counting.py`, any capability): an event may carry `event_id` and
`state: "provisional"`; a later record of the same capability holds `{"event": "event.revision", "revises":
<event_id>, "state": "final" | "revoked", "reason"?, start_ms, end_ms}`. No state = final. The newest revision
decides. `count_events`: count = provisional + final, `states` / `group_states` give each, `where={"state": ...}`
chooses; revoked events are never counted by default. `get_timeline` shows each event with its current state
(`revised_by`), not the revision beside it. `recipe run --report` has `states`, `event_states` (with
`committed_ms`, `settled_ms`), `score` (after revisions) and `score_provisional` (as counted at once).
A `term.other_voice` that a verdict gives to the person is revoked and a final `term.spoken` with `replaces` is added.

## Settings (defaults; not chosen on a clip)

| what | value | why |
|---|---|---|
| diarization mode | `low_latency` (720 ms step + 320 ms right context) | Nemotron real-time factor on the Mac CPU alone: 0.25 vs 0.48 (`very_low_latency`) and 0.87 (`ultra_low_latency`) |
| slot sound | only where that slot alone is active (`overlap: false`) | a mixture of two voices embeds as neither |
| piece | 2 s of a slot's own sound per embedding; identity = duration-weighted centroid of its pieces, last 60 s | CAM++ costs ~17 ms per second of sound: re-embedding a growing slot would not stay real-time |
| before the first piece | the slot's sound if ≥ 0.8 s, else `slot_unknown` (not counted at once) | |
| threshold | 0.5 (`--speaker-threshold`) | the module default; nothing set on a clip |
| stable | 8 s of a slot's own sound | |
| settle | 10 s of slot activity after the word, at most 60 s | |

## Tools

- `--as-live` gives a recording at its own pace, as a live capture (nothing waits for slow modules).
- `gamst recipe bench person-term-count --clips DIR [run options]`: the folder format and columns are in
  `gamst/recipes/bench.py`; scoring by end time; `--words` prints each word's slot, score, provisional and final decision.
- `transcript-replay` (`"file"`) gives a recorded ASR's words when the ASR gave them; `python -m
  gamst.providers.transcript_replay` exports them from a store, keeping only the counted terms' text.
- `python -m gamst.count_video --report ...` draws provisional (orange), final (green), revoked (red, struck out).
