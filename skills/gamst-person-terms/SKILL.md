---
name: gamst-person-terms
description: Count how often one enrolled person says chosen words, a text pattern, or uses a demonstrated tone of voice, and read the timestamped count from GAMST context. Use when someone asks how many times a specific person said something or spoke in a certain way.
---

# Counting what one enrolled person says, and how

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
gamst recipe enroll person-voice-style-count --audio demo.wav --start 2 --seconds 1 --out style.json
gamst recipe run person-voice-style-count --microphone --voice me.json --exemplar style.json
```

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
- Tone channels are pitch, level, voicing, periodicity and brightness; no emotion is inferred.
