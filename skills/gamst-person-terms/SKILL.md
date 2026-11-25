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
gamst recipe run person-term-count --microphone --voice me.json --terms 대박,진짜
gamst recipe run person-term-count --audio room.wav --voice me.json --terms 're:(거든요|잖아요)$' --headless \
    --truth truth.json --report out.json
gamst recipe enroll person-voice-style-count --audio demo.wav@2-3 --audio more.wav@4-5.2 --negative calm.wav@3-4 --out style.json
gamst recipe run person-voice-style-count --microphone --voice me.json --exemplar style.json
```

`--video FILE` reads a video's own sound (needs ffmpeg and ffprobe on PATH).

## Words by example

```bash
gamst recipe enroll person-term-count --kind terms --video a.mp4@3.1-3.8 --audio b.wav@0-1 --proposals p.json
gamst recipe proposals list --proposals p.json
gamst recipe proposals confirm 3fa9c1d2e0b7 --proposals p.json                  # or 3fa9c1d2e0b7=대박,대애박 to edit
gamst recipe enroll person-term-count --kind terms --proposals p.json --out words.json
gamst recipe run person-term-count --microphone --without voice --terms words.json
```

The recipe's recognizer writes down each clip; the different spellings become one proposed term. Only the owner confirms or
edits it (an agent may propose with MCP `propose_movement`, never confirm). Name the part of each clip that is the word. No audio is
kept; `words.json` holds text. `--no-confirm` saves undecided words; their events say `confirmed: false`. Which spellings a
recognizer will produce for a word is not predicted: check a recording you counted yourself.

Pipeline: microphone -> speaker verification (segments) -> speaker gate (other voices silenced) -> ASR with
word times -> speaker verification again on each utterance's own words -> term events. Records:
`speech.term_events` (`count_events`, event `term.spoken`), `audio.exemplar_events`, `audio.speaker_match`.

## Limits

- The speaker threshold (`--speaker-threshold`, default 0.5) has not been measured on people. Count a take
  yourself and score it with `--truth` before quoting a number.
- The gate does not separate voices that overlap. A separation model does (`target-speech` part, in a copy
  of the recipe): set `"use": "target-speech"` on `target_voice`, pass `--voice-wav`, and list local files in
  `"config"`. USEF-TFGridNet is CC BY-NC 4.0 and needs `allow_noncommercial`.
- An utterance too short or too quiet to score is not counted.
- Whisper works in 3 s windows: a word cut by an edge can be missed.
- Tone channels compare with the speaker's own recent median: give each example as a part of a longer recording
  (`FILE@start-end`, a few seconds of usual speech before it), not as a trimmed clip.
- Tone channels are pitch, level, voicing, periodicity and brightness; no emotion is inferred.
