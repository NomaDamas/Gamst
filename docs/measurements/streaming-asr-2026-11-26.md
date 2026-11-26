# Streaming ASR with capture-time word times (2026-11-26)

This follows the owner decision that the product path is a streaming recogniser
([realtime-attribution-decisions.md](../realtime-attribution-decisions.md), item 2). Whisper phrase mode is kept only
as the recorded baseline.

## What was built (branch `feat/streaming-asr`)

- `gamst/providers/streaming_asr.py` has `StreamingWords`, a Pipeline module. Audio goes in at 16 kHz. Out come
  `speech.transcript_words` records, the same ones the phrase parts emit: final utterances with capture-time word times,
  plus optional partials. `terms`, the speaker modules and scoring read them unchanged.
  - `CaptureClock` maps a recogniser's seconds (of the audio it was sent) to capture time, so network delay cannot
    move a word.
  - `Endpointer` is an energy-pause commit for recognisers that wait for the client to end a turn.
  - Finals leave in commit order.
  - Two optional fields are added: `final_after_ms` (wall time from the sound to the final) and `heard_to_ms`
    (partials).
- `gamst/providers/streaming_hosted.py` holds the WebSocket sessions.
  - OpenAI `gpt-live-transcribe`: Realtime API, `?intent=transcription`.
  - ElevenLabs Scribe v2 Realtime and v3 Live.
  - Opt-in as for the other hosted modules: `allow_remote` or `--allow-remote-asr`, `wss://`, and the key from the
    environment only.
  - A cost line is printed before any audio is sent. It includes the keyterm surcharge.
- Aligner lane. OpenAI's docs say `gpt-live-transcribe` "doesn't return word-level timestamps".
  - The text of each committed turn is force-aligned on the same audio by `audio.word_alignment` in its own worker.
    Qwen3-ForcedAligner-0.6B is the default (MPS bf16).
  - The record's text is always the API's. If the aligned words do not spell exactly that text, the record keeps the
    text without word times (`align_error`).
  - `faster_whisper.align_large_v3_turbo` (ctranslate2 alignment) stays as a non-default candidate.
- `gamst/providers/streaming_local.py` holds the local streaming workers (`streaming.asr.words`):
  - sherpa-onnx Korean streaming Zipformer;
  - Voxtral-Mini-4B-Realtime-2602, MLX 4-bit (`mlx-audio` 0.6.8).
- Registry, `audio.asr`:
  - `openai.gpt_live_transcribe.hosted`, `elevenlabs.scribe_v2_realtime.hosted`, `elevenlabs.scribe_v3_live.hosted`;
  - `sherpa.zipformer_ko_streaming`, `voxtral.mini_4b_realtime_2602.mlx`.
- Registry, `audio.word_alignment`: `faster_whisper.align_large_v3_turbo`.
- Recipe:
  - The `stream-words` part.
  - `--asr CANDIDATE` swaps the transcribing module of any recipe.
  - `--realtime [SPEED]` replays a recording at capture pace, so latency can be measured.
- Tests: `tests/test_streaming_asr.py`, with stand-in WebSocket servers. They check:
  - late answers keep capture times;
  - out-of-order completions;
  - the aligner cannot change text;
  - the key stays in the header;
  - the opt-in;
  - long segments reach the aligner.

ElevenLabs word times are counted from the start of the session's audio. This was checked with the real API: manual
commits at known stream positions, and the words fell inside their own segments only on the session clock.

## Run (one 54 s Korean shop clip `a.wav`, several speakers; M2 Pro)

How it was run:
- 1x real time (`--realtime`), `--without voice` (hearing only).
- Term `뽀또`. Only the spelling `뽀또` is counted; another spelling is said where it occurs.
- Truth: the owner's word end times, scored at ±0.6 s, one heard word per label.
  - The person: 12.04, 14.0, 19.74, 20.14, 21.46, 22.30, 28.10, 38.20, 46.64, 47.74.
  - Others: 12.64, 20.34, 20.74, 27.44, and three or more in 49–52 s.
  - Nothing said: 24.70.
- Latency is measured per word: the wall time from when the word's end reached the module to when its final record
  arrived.
- Time error is the word end minus Whisper turbo's word end at the same moment.

| run | person heard (of 10) | others heard | 24.70 (nothing said) | other unlabelled | latency median (min–max) | word end − turbo end | cost / 54 s |
|---|---|---|---|---|---|---|---|
| Whisper turbo phrase mode (baseline, earlier run) | 9 (not 14.0) | 12.64 20.34 20.74 27.44; 49–52: 0 | **invented** (24.70) | – | not measured (waits hop 4 s + margin 3 s) | 0 | local |
| Scribe v2 RT, server VAD (run 1 / run 2) | 10 / 10 | 12.64 20.74 27.44 / 12.64 20.74; 49–52: 3 | a word at 23.84–24.16 | 15.5 s | 23.3 s (2.5–33.7) | −0.22…+0.10 | $0.0033 |
| Scribe v2 RT, client commits (2 runs, same) | 8, plus 12.04 written 포도 (not 14.0) | 12.64 (포도) 20.74; 49–52: 3 | 23.84–24.16 | 26.1 s | 2.7 s (0.4–12.0) | −0.18…+0.10 | $0.0033 |
| Scribe v2 RT, client commits + keyterms | 9 (not 14.0) | 12.64 20.74; 49–52: 3 | 23.84–24.16 | 26.1 s | 2.8 s (0.4–12.0) | −0.18…+0.10 | $0.0041 |
| Scribe v3 Live, server VAD | 9 (not 14.0) | 12.64 20.74 27.44; 49–52: 3 | 23.84–24.16 | – | 21.8 s (2.4–34.0) | −0.14…+0.10 | $0.0033 |
| Scribe v3 Live, client commits | 9 (not 14.0) | 12.64 20.74 27.44; 49–52: 3 | 23.84–24.16 | – | 2.7 s (0.4–12.0) | −0.26…+0.10 | $0.0033 |
| Scribe v3 Live, client commits + keyterms | 9 (not 14.0) | 12.64 20.74 27.44; 49–52: 3 | 23.84–24.16 | – | 2.8 s (0.4–12.0) | −0.26…+0.10 | $0.0041 |
| gpt-live-transcribe + Qwen3-FA (2 runs) | 3 / 4 | 49–52: 1 (run 1 also 12.64) | not written | – | 4.7 s (1.6–13.2) | −0.26…+0.10 | $0.0154 |
| gpt-live-transcribe + keywords + Qwen3-FA (2 runs) | 8 / 8 (not 14.0, 28.10) | 12.64 20.74 27.44 (run 2); 49–52: 3 | **not written** | 26.6 s (run 1) | 3.6 s (1.5–13.1) | −0.32…+0.10 | $0.0154 |
| sherpa Zipformer ko (local) | 1 | – | – | – | 3.7 s | +0.40 | local |
| Voxtral 4B Realtime MLX, 480 ms delay (local) | 0 (one word in 54 s) | – | – | – | – | – | local |
| Voxtral, 960 ms delay (local) | 3 (nothing before 35 s) | 49–52: 2 | – | 52.6 s | 4.1 s (2.5–6.5) | +0.16…+0.28 | local |

Notes on the table:

- **24.70.** Every Scribe run writes the word at 23.84–24.16 s. That end is 0.54 s before the 24.70 "nothing said"
  label, so it falls inside the ±0.6 s window. Whisper's invented word ends at 24.70 itself. Whether 23.8–24.2 holds a
  real word is for the owner to check by ear. gpt-live-transcribe did not write a word there in any run.
- **14.0.** Only Scribe v2 with server VAD heard it, in both runs. With client commits, a commit fell at about 13.9 s,
  inside the word (final spans end at 13.7 s and the next one starts at 14.0 s). The energy endpointer cut the word.
- **Latency.** Server VAD (1.5 s silence) almost never commits in continuous shop noise: 4 finals in 54 s. Client
  commits make 12 finals. The 12 s maximum is the 38.2 word: no pause was found until the endpointer's longest turn.
- **Partial → final.**
  - A partial first shows the word a median of about 0.5 s (Scribe v3), 0.8 s (Scribe v2) and 1.0 s (gpt) after the
    word ends.
  - The final arrives 2.7–3.6 s after.
  - Partials are revised up to 15 (Scribe v2), 45 (Scribe v3, client commits) and 82 (gpt) times per utterance.
  - A partial sometimes showed the word earlier than its final did, at another place in the same utterance (it was
    revised).
- **gpt-live-transcribe without keywords** wrote 3 words for 18.9–23.6 s, where five of the person's words are.
  With `keywords: ["뽀또"]` it wrote them.
- **The aligner never changed the API's text.**
  - Both keyword runs: 12/12 finals aligned, 0 align errors. The aligned words' characters equal the API text's in
    12/12.
  - The aligner splits more finely than spaces: 112 space-separated words became 129 aligned words.
  - Qwen3-ForcedAligner time per committed segment: median 306–322 ms, max 546 ms, for segments with a median of
    3.7 s and a maximum of 14.7 s (MPS bf16).
- **Aligner versus the recogniser's own times.** Compared on 13 shared target words: Qwen3-ForcedAligner on
  gpt-live-transcribe's text against Scribe v2's native word times.
  - Word ends: median |Δ| 0.04 s, max 0.28 s (at 52.2 s).
  - Word starts: Qwen's are 0–0.22 s earlier.
- **Voxtral.** The streaming session stays in padding over this clip's opening. At 480 ms delay it wrote one word in
  54 s. Started 10 s into the clip, it wrote 6 counted words, and at 960 / 2400 ms it wrote them too.
  - On clean speech (macOS TTS) it streams correctly at any chunk size.
  - Its word times come from the 80 ms token clock and were 0.16–0.28 s late against turbo.
- **Run to run.** The same hosted settings twice moved single words: 27.44 heard or not, 28.10 matched at 27.88 or
  28.10. One clip cannot rank close settings.

## Cost

List prices, read 2026-11-26. Silence is billed too.

| provider | price |
|---|---|
| OpenAI `gpt-live-transcribe` | $0.017 per minute ($1.02 per hour) |
| ElevenLabs Scribe v2 Realtime / v3 Live | $0.22 per hour, keyterms +$0.05 per hour |

This work's hosted calls came to about $0.11 at list price: 7 OpenAI and 10 ElevenLabs sessions, including smoke tests.

## Not done

- One clip only. The defaults (endpointer rules, commit strategy) were not tuned on it. The commit strategy default
  (client commits) was chosen by the structural latency gap above.
- AssemblyAI and Deepgram: no keys.
- Voxtral was not made to recover from the stuck-in-padding state. A session restart on long silence would be a
  heuristic and was not added.
- The MMS-FA aligner was not compared.
- Transcripts were not stored. Exports keep only the target word's text: `try-it/testbed/asr/*.words.jsonl`.
