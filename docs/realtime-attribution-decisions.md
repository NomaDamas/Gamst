# Realtime word counting by one person: owner decisions (2026-11-26)

These decisions extend [perception-context-decisions.md](perception-context-decisions.md). GAMST is a realtime
system that can also analyse recordings, not a video-analysis tool: every product path must run on a live stream.
Recorded files are only how the test bed is replayed.

## Decisions

1. **Realtime first, then correction.** A count is committed as soon as it is known (`provisional`). A verification
   lane runs concurrently on the last seconds of the same capture and later revises only the records it finds wrong
   (`final`, or `revoked` with the reason). Nothing waits for verification; nothing is recomputed from the start.
2. **Streaming transcription.** The words come from a streaming ASR with word times (local or hosted). Whisper in
   phrase mode is a baseline only: it waits for a phrase to end and is not a streaming recogniser.
3. **Who said a word = speaker tracks, not word snippets.** A streaming diarization module labels who is active every
   frame (tens of milliseconds). Each speaker slot accumulates its own sound over time, and the slot's identity is
   decided from that long audio against the owner's few-shot voice reference. A word belongs to the slot active at its
   capture time. This replaces scoring 1-1.5 s around each word, which mixes neighbours when people alternate in
   under a second (measured: `docs/measurements/testbed-2026-11-26.md`).
4. **The voice is given few-shot by the owner** (a time range or a short clip of the person), with a consent record.
   The voice score stays the primary evidence; faces and lips are an optional fallback, off by default
   (`docs/measurements/av-speaker-2026-11-26.md`).
5. **State of the art, measured here.** Diarization and ASR candidates are chosen by published multilingual results and
   then measured on several owner clips. Non-commercial weights that are state of the art are registered as opt-in
   candidates only (`allow_noncommercial`). Hosted models need an explicit opt-in that audio leaves the device, read
   their key from the environment only, and print a cost line before use.
6. **No tuning on one clip.** Defaults (thresholds, delays) are chosen on the owner's set of clips with word truth,
   never on one clip; any setting chosen on a clip is stated with its result.
7. **Inside GAMST.** Every part is a registry candidate with its own environment, a Pipeline module in a lane, and a
   recipe part; records follow the capability contracts and the private-payload rule. No parallel framework.

## Shape

```text
microphone ─┬─▶ streaming ASR ────────────────────────────▶ words + capture times
            ├─▶ streaming diarization (slots, ~80 ms) ────▶ who is active, per frame
            │        └─▶ slot identity: each slot's accumulated sound vs the few-shot voice (updated over time)
            └─▶ audio ring (last N s, in memory)
words + slots ─▶ attribution by capture time ─▶ term events: provisional, at once
                                                    │
audio ring + later context ─▶ verification lane (concurrent) ─▶ revisions: final / revoked (only what changed)
                                                    │
                                     Context (timeline, counts with revision state) ─▶ MCP ─▶ LLM
```

## Measured before these decisions (one 54 s clip, several people in a shop)

| approach | counted / right (truth 10) |
|---|---|
| Whisper turbo + voice from another recording | 8 / 6 |
| + owner-marked 1.5 s of the same clip | 12 / 9 (that 1.5 s holds one of the counted words) |
| larger voice model, PixIT separation, pyannoteAI, lips, Qwen3-ASR | no reliable gain |

Remaining errors were a word the ASR invented while the person spoke (long-context bias) and words of other people
said within a second of the person's.

## Catch-up and verification (owner decisions, 2026-11-26)

- **Nothing is dropped.** When a live lane falls behind by more than `lag_ms`, it skips to "now"; the skipped capture range is
  kept (memory, spilled to disk past a budget) and becomes a to-do item.
- **Catch-up workers** (a pool, capped by resources: one on a laptop, more for hosted streams or a GPU server) take to-do
  items. A worker starts from a copy of the live module's state at the range start and listens from the last speech pause
  before it (bounded), so the word in progress is not cut. Hosted recognisers get the preceding text instead of a state copy.
- **Verification lane** runs in capture order behind both, and only over ranges that are complete (no to-do left): a second
  recogniser (gpt-live-transcribe + Qwen3-ForcedAligner times) checks the words, pyannoteAI Precision-3 on windows tells who
  spoke, the few-shot voice identifies the person; it emits final / revoked only where something changes. Catch-up can never
  overwrite a final.
- **Live diarization slots do not decide who said a word** (Nemotron-3 mixed people on the test clip). Live uses the per-word
  voice score for provisional counts.
- **Recognisers:** live = ElevenLabs Scribe v3 Live (best published error rate; best recall here), verification =
  gpt-live-transcribe with keyword hints (wrote no invented word here; the owner confirmed by ear that nothing is said at
  23.8-24.2 s where Scribe wrote the word). Non-commercial state-of-the-art models are registered as opt-in only.
