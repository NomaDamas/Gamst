---
name: gamst-person-terms
description: Count how often chosen words (typed, or taken from example clips), a text pattern or a demonstrated tone of voice occur, said by one enrolled person or by anyone, and read the timestamped count from GAMST context. Use when someone asks how many times something was said or someone spoke in a certain way.
---

# Counting what is said, and how

By default a count is tied to one enrolled person. `--without voice` counts whoever speaks; then every event and the report
say `speaker: not_bound`.

Recipes: `person-term-count` (words) and `person-voice-style-count` (tone; text patterns optional).

The person must agree. Enrollment needs `--consent`, stores embeddings only (file mode 0600), and `run`
refuses an enrollment file without the consent record.

```bash
gamst recipe setup person-term-count
gamst recipe enroll person-term-count --kind voice --microphone --seconds 20 \
    --consent "who agreed, to what, when" --out me.json
gamst recipe run person-term-count --microphone --voice me.json --terms 대박,진짜 --allow-remote-asr
gamst recipe run person-term-count --audio room.wav --voice me.json --terms 're:(거든요|잖아요)$' --headless \
    --allow-remote-asr --truth truth.json --report out.json
gamst recipe enroll person-voice-style-count --audio demo.wav@2-3 --audio more.wav@4-5.2 --negative calm.wav@3-4 --out style.json
gamst recipe run person-voice-style-count --microphone --voice me.json --exemplar style.json
```

`--video FILE` reads a video's own sound (needs ffmpeg and ffprobe on PATH).

## The recognizer

The words come from a streaming recognizer with word times: by default ElevenLabs Scribe v3 Live
(`elevenlabs.scribe_v3_live.hosted`, part `stream-words`). It is hosted: the sound leaves the device as it is captured and
is billed ($0.22 per hour of sound, silence too; +$0.05 per hour with key terms; list price 2026-11-26). Nothing is sent
without `--allow-remote-asr`, and the key is read from `ELEVENLABS_API_KEY` only. Without the flag the run stops before it
starts and says so. A cost line is printed before any sound is sent.

On this device instead (set up the same way, `gamst recipe setup person-term-count --asr CANDIDATE`):
`--asr sherpa.zipformer_ko_streaming` (CPU) or `--asr voxtral.mini_4b_realtime_2602.mlx` (Apple silicon). Both heard far
fewer of the counted words on the one shop clip measured (docs/measurements/streaming-asr-2026-11-26.md). Whisper
(`faster_whisper.*`, part `whisper-words`) and the others in `gamst recipe roles` stay registered choices.

## Words by example

```bash
gamst recipe enroll person-term-count --kind terms --video a.mp4@3.1-3.8 --audio b.wav@0-1 --proposals p.json --allow-remote-asr
gamst recipe proposals list --proposals p.json
gamst recipe proposals confirm 3fa9c1d2e0b7 --proposals p.json                  # or 3fa9c1d2e0b7=대박,대애박 to edit
gamst recipe enroll person-term-count --kind terms --proposals p.json --out words.json
gamst recipe run person-term-count --microphone --without voice --terms words.json --allow-remote-asr
```

The recipe's recognizer writes down each clip; the different spellings become one proposed term. Only the owner confirms or
edits it (an agent may propose with MCP `propose_movement`, never confirm). Name the part of each clip that is the word. No audio is
kept; `words.json` holds text. `--no-confirm` saves undecided words; their events say `confirmed: false`. Which spellings a
recognizer will produce for a word is not predicted: check a recording you counted yourself.

While it runs (`--serve`), an agent may replace the words: MCP `set_instruction {"kind": "terms", "actor": "jev", "terms": [...]}`
(no `re:` patterns), `search_retained` looks for them in earlier transcripts. See `gamst-scene`.

Pipeline: microphone -> speaker verification (segments) -> speaker gate (other voices silenced) -> ASR with
word times -> speaker verification again on each utterance's own words -> term events. Records:
`speech.term_events` (`count_events`, event `term.spoken`), `audio.exemplar_events`, `audio.speaker_match`.

## Limits

- The speaker threshold (`--speaker-threshold`, default 0.5) has not been measured on people. Count a take
  yourself and score it with `--truth` before quoting a number.
- The gate does not separate voices that overlap. A separation model does (`target-speech` part, in a copy
  of the recipe): set `"use": "target-speech"` on `target_voice`, pass `--voice-wav`, and list local files in
  `"config"`. USEF-TFGridNet is CC BY-NC 4.0 and needs `allow_noncommercial`.
- `--with overlap` (off by default) splits the sound into one source per speaker with pyannote PixIT and scores
  each word on the sources active at it; `speaker_score` is the best of them, `mixture_score` the room sound.
  Gated weights (accept `pyannote/speech-separation-ami-1.0` and `pyannote/separation-ami-1.0`), role
  `audio.speaker_sources`; slower than real time on a Mac. On the one clip measured it did not count better
  (docs/measurements/overlap-2026-11-26.md).
- `--with hosted` (off by default) sends the sound to pyannoteAI (voiceprint + identification; billed, the audio
  leaves the device; `allow_remote: true` and `PYANNOTEAI_API_KEY`). Each word gets `turn_match` beside its
  voice score; it does not decide. A cost line is printed before each call.
- An utterance too short or too quiet to score is not counted.
- The recognizer ends a turn after a pause (0.5 s of quieter sound; 0.16 s once a turn is 12 s long) or at 20 s;
  a word is counted when its turn is final. On the one 54 s shop clip measured, Scribe v3 Live heard 9 of the person's 10 words and its finals came 2.7 s
  (median; 0.4-12.0 s) after the word (docs/measurements/streaming-asr-2026-11-26.md).
- Tone channels compare with the speaker's own recent median: give each example as a part of a longer recording
  (`FILE@start-end`, a few seconds of usual speech before it), not as a trimmed clip.
- Tone channels are pitch, level, voicing, periodicity and brightness; no emotion is inferred.
