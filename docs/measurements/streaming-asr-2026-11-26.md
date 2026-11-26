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

Added later (branch `feat/asr-more-providers`):
- Sessions for AssemblyAI STT-3.7-Realtime, Deepgram Nova-3 Korean and Speechmatics Realtime Enhanced: binary PCM16
  in, word times from the stream start.
- The service ends turns by default. `commit_strategy: manual` sends GAMST's commits instead (`ForceEndpoint` /
  `Finalize` / `ForceEndOfUtterance`).
- Keyterms: `keyterms_prompt` / `keyterm` / `additional_vocab` (`"bias": "terms"`). Profanity filters are sent as off.
- Registry: `assemblyai.stt_3_7_realtime.hosted`, `deepgram.nova_3.hosted`, `speechmatics.enhanced.hosted`. All are
  opt-in and never automatic.

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
| AssemblyAI STT-3.7-Realtime, server turns | 1 as 뽀또 (+2 as 포또) | 49–52: 3 (포또) | – | 11.1, 38.9 s | 8.4 s (7.5–8.4) | −0.27…−0.10 | $0.0045 |
| AssemblyAI, server turns + keyterms | 1 as 뽀또 (+5 as 포또) | 20.74 (포또); 49–52: 3 (포또) | – | 11.1, 22.9, 38.9 s | 14.6 s (7.2–16.8) | −0.27…+0.30 | $0.0045 |
| AssemblyAI, client commits (ForceEndpoint) | 1 | – | – | 10.9 s | 3.6 s | −0.36 | $0.0045 |
| AssemblyAI, client commits + keyterms | 2 as 뽀또 (+3 as 포또) | 20.74 (포또); 49–52: 1 | – | 10.9, 23.0, 38.8, 46.0 s | 3.2 s (0.5–11.7) | −0.47…+0.20 | $0.0045 |
| Deepgram Nova-3 ko, server endpointing (300 ms) | 4 | 20.74; 49–52: 1 | – | – | 2.3 s (1.0–3.5) | +0.03…+0.10 | $0.0043 |
| Deepgram, server + keyterm | 6 | 12.64 20.74; 49–52: 1 | – | – | 2.3 s (1.1–3.5) | −0.19…+0.10 | $0.0055 |
| Deepgram, client commits (Finalize) | 3 | 20.34 20.74 | – | 21.2 s | 2.4 s (1.2–3.3) | −0.06…+0.06 | $0.0043 |
| Deepgram, client commits + keyterm (2 runs) | 6 / 7 as 뽀또 (+12.04 as 포또 both) | 3 / 4 of 4; 49–52: 3 / 2 | **written once** (23.66–24.38, run 1) | – / 21.2 s | 2.3 / 2.2 s | −0.26…+0.09 | $0.0055 |
| **Speechmatics Enhanced, server (max_delay 0.7 s)** | 7 | 20.74 27.44; 49–52: 2 | not written | – | **1.2 s (0.95–1.76)** | −0.35…0 | $0.0060 |
| **Speechmatics, server + additional_vocab (2 runs)** | **8 / 8** (not 14.0, 28.10) | 20.74 27.44; 49–52: 2 | **not written** | – | **1.2 / 1.1 s** (0.9–1.4) | −0.32…0 | $0.0060 |
| Speechmatics, client commits (ForceEndOfUtterance), ± vocab | 8 / 8 | 20.74 27.44; 49–52: 2 | not written | – | 1.2 / 1.1 s | −0.35…0 | $0.0060 |
| Speechmatics, max_delay 2.0 s + vocab | 8 | 20.74 27.44; 49–52: 2 | not written | – | 2.5 s (2.2–2.6) | −0.32…0 | $0.0060 |
| sherpa Zipformer ko (local) | 1 | – | – | – | 3.7 s | +0.40 | local |
| Voxtral 4B Realtime MLX, 480 ms delay (local) | 0 (one word in 54 s) | – | – | – | – | – | local |
| Voxtral, 960 ms delay (local) | 3 (nothing before 35 s) | 49–52: 2 | – | 52.6 s | 4.1 s (2.5–6.5) | +0.16…+0.28 | local |

Notes on the table:

- **24.70.** Every Scribe run writes the word at 23.84–24.16 s. That end is 0.54 s before the 24.70 "nothing said"
  label, so it falls inside the ±0.6 s window. Whisper's invented word ends at 24.70 itself.
  - The owner confirmed by ear that nothing is said at 23.8–24.2 s, so the Scribe word there is invented.
  - Not written there: gpt-live-transcribe (any run), Speechmatics (6 runs) and AssemblyAI (4 runs).
  - Deepgram wrote it in 1 of 2 runs with client commits and keyterms.
- **The three later providers (added 2026-11-26).**
  - Spellings: AssemblyAI mostly writes 포또, even with `keyterms_prompt`. Its word ends run 0.1–0.47 s early, and it
    writes the word at 10.9–11.1 s and 38.8–38.9 s, where no moment is labelled. Its server turns in this noise lasted
    up to 17 s.
  - Speechmatics finalises word by word (about 90 finals in 54 s). A partial shows the word about 0.85 s after it ends,
    with at most 3 revisions.
  - Speechmatics with `max_delay` 2.0 s heard nothing more than with 0.7 s.
  - Deepgram's partials show the word about 0.5 s after it ends, with at most 5 revisions.
  - No provider heard 14.0. Only Deepgram (run 2) and Scribe heard 28.10.
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
| AssemblyAI STT-3.7-Realtime | $0.30 per hour of session time, keyterms included (Agent model $0.45) |
| Deepgram Nova-3 monolingual streaming | $0.0048 per minute ($0.29 per hour), keyterm prompting +$0.0013 per minute |
| Speechmatics Realtime Enhanced | $0.40 per hour |

This work's hosted calls came to about $0.11 at list price: 7 OpenAI and 10 ElevenLabs sessions, including smoke tests.
The three later providers came to about $0.09 at list price: 15 runs of 54 s, plus smoke and profanity tests.

## Profanity

Measured on a macOS TTS clip, «아 시발 진짜 짜증나. 씨발 이게 뭐야. 시발, 또 늦었어.», with each adapter's settings. No
provider masked anything.

| provider | default | how to keep it off | what it wrote for the three swear words |
|---|---|---|---|
| ElevenLabs Scribe v3 Live | no masking | do not set `entity_redaction`; `offensive_language` would redact into a separate `redacted_transcript` | 씨발, 씨발, 씨팔 |
| OpenAI gpt-live-transcribe | no filter documented | – | 씨발, 씨발, 십팔 |
| AssemblyAI | `filter_profanity` default false; masking would keep the first letter plus `*` | GAMST sends `filter_profanity=false` | 시발, 시발, 18 |
| Deepgram | `profanity_filter` default false; Korean is in its filter's language list | GAMST sends `profanity_filter=false` | 씨발, 씨발, 씨발 |
| Speechmatics | no masking; it tags profanity in English, Spanish and Italian only | – | 씨발, 시발, 18 |

To count the word, the terms need the spellings 시발, 씨발, 씨팔 and 십팔. Two providers wrote the third one as "18",
a digit string that cannot be counted as the word without false matches.

## Default

The owner chose Scribe v3 Live with client commits (`elevenlabs.scribe_v3_live.hosted`) as the `audio.asr` default on
every tier, and the `stream-words` part for `person-term-count`, `person-voice-style-count` and `scene` (Whisper's
weights were removed; its candidates stay registered as choices). The choice rests on this one clip.
- Setup installs only the WebSocket client.
- A run without `--allow-remote-asr` stops before any input opens. The message names the flag and the local choices
  (`--asr sherpa.zipformer_ko_streaming`, `--asr voxtral.mini_4b_realtime_2602.mlx`).
- The recipe tests drive the real part against a local stand-in server (`ScribeStandIn` in
  `tests/test_streaming_asr.py`).

## Not done

- One clip only. The defaults (endpointer rules, commit strategy) were not tuned on it. The commit strategy default
  (client commits) was chosen by the structural latency gap above.
- AssemblyAI and Deepgram: no keys.
- Voxtral was not made to recover from the stuck-in-padding state. A session restart on long silence would be a
  heuristic and was not added.
- The MMS-FA aligner was not compared.
- Transcripts were not stored. Exports keep only the target word's text: `try-it/testbed/asr/*.words.jsonl`.
