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

## Details agreed with the owner (2026-11-26)

**Where it lives.** Catch-up and the verification lane are part of the GAMST **Pipeline** (lanes, a to-do queue of capture
ranges, completeness watermarks). Models stay **swappable**: each one is a registry candidate chosen per recipe item
(`"candidate"`), on the command line (`--asr CANDIDATE`, `--with-role audio.diarization=CANDIDATE`) or in a profile. The
default of each role is whatever measured best on the owner's clips (state of the art first); today: ASR live =
`elevenlabs.scribe_v3_live.hosted` (Whisper is no longer the default; it stays a selectable candidate).

**Lanes.**

```text
capture  ─────────────────────────────────────────────▶  (all audio kept; nothing dropped)
L1 live          recogniser + per-word voice score ─▶ provisional counts; on lag > lag_ms jump to "now"
L2 catch-up      worker pool takes skipped ranges; state copied from L1 at the range start; listens from the last
                 speech pause before it (bounded look-back); hosted recognisers get the preceding text instead
L4 verification  serial, in capture order, only over complete ranges: second recogniser, Precision-3 windows,
                 few-shot voice on each speaker's segments ─▶ final / revoked only where something changes
Context / MCP    ranges still being caught up show as pending, then final
```

**Order and conflicts.** A range is verified only after L1/L2 have finished it (watermark), so catch-up can never overwrite
a final; every result is a versioned record and the newest revision wins.

**Worker count.** L2 needs at least one worker to catch up; more only shorten how long a range stays pending (a waiting item is
queued, never dropped). The cap follows resources: 1 on a laptop (memory guard), the provider's concurrency for hosted
recognisers, free GPU memory on a server. Processes running at once: L1 + L2 workers + L4.

**Voice decision.** A speaker-embedding model (CAM++) turns sound into a vector; the decision compares it with the vector of
the owner's few-shot sample of the person. L1/L2 use the sound around each word (provisional); L4 uses the whole segments
Precision-3 attributes to one speaker (longer, cleaner: final).

**Two recognisers.** L1 = Scribe v3 Live (keyterms = the words to count); L4 = gpt-live-transcribe with keyword hints (the
words to count, passed to OpenAI as hints — not Scribe's transcript) and Qwen3-ForcedAligner timestamps. Rule (to be chosen on
the clip set): both heard → final; only L1 heard → revoked; only L4 heard → final added.

**Diarization.** Precision-3 handles more than two speakers and marks overlapped speech; what no model here does reliably is
splitting one microphone's overlapped sound into separate voices. Live diarization models have speaker caps (Sortformer 4,
pyannoteAI Live-1 8); they do not decide who said a word, so the caps do not limit counting. Separate microphones per person
would make attribution direct (each microphone is its own input); not configured now.

**Terms that are sound words.** A term can be an onomatopoeic syllable such as "달" said alone or repeated ("달달달"): it is
matched only as a whole word made of that syllable, never inside other words ("달라", "달걀"), and each repetition counts.

How it is built (contracts, settings, status fields): [catch-up-verification.md](catch-up-verification.md).
