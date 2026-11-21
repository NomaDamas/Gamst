# Semantic VLM and audio Context scope

## Decisions (2026-11-22)

- RelateAnything is **not** part of the current GAMST implementation. Existing RF-DETR/YOLOE boxes, tracks, geometry and GAMST time state already cover zone/near/entered-style conditions. RelateAnything remains a future optional relation provider only for `holding`, `riding`, `on`, and similar object-pair relations.
- Qwen3.5-9B-VL remains the open-weight semantic consumer. It is not a per-frame sensor: it receives a bounded frame window and/or structured GAMST Context when a question or Watch requires semantic interpretation.
- Full-duplex conversation is **out of current scope**. GAMST currently extracts audio Context; it does not open a realtime voice session, start TTS, or implement playback cancellation.
- A commercial realtime voice API is an optional, disabled-by-default consumer configuration. Enabling it only stores validated endpoint/model metadata; it does not start a network call or spend credits.

## Audio Context contract

The `audio.context` provider combines caller-owned outputs from the audio workers. It never accepts or stores raw samples.

| Field | Meaning | Availability |
|---|---|---|
| `speech_active` | speech is currently detected | TEN-VAD |
| `silence_ms` | duration since the last detected speech | derived from VAD timestamps |
| `barge_in` | speech overlaps assistant playback | only when playback state is supplied; otherwise null |
| `double_talk` | microphone speech overlaps playback/reference | only with reference; otherwise null |
| `aec_residual_dbfs`, `aec_erle_db` | residual echo and reduction estimate | only with AEC reference |
| `sound_events` | bounded labels and uncalibrated scores | CED |
| `speaker_turn`, `speaker_slots` | speaker activity/turn changes | Nemotron diarization |
| `doa_deg`, `doa_confidence` | sound direction | multi-microphone input only; single mic is unavailable |
| `prosody` | RMS, F0, voicing, pauses, speech rate, spectral features | DSP feature provider |
| `transcript` | partial/final/interrupted speech text | Qwen3-ASR |

Unavailable measurements remain `null`; GAMST must not infer them.

## Optional realtime voice backend

MCP tools:

- `get_voice_backend_status`
- `configure_voice_backend`

Default:

```json
{"enabled": false, "provider": "none", "context_only": true}
```

Supported provider labels: `openai`, `gemini`, `custom`. The configuration is a transport boundary only. A product/application must supply the actual realtime client, playback reference, consent, billing guard, and privacy policy before enabling it.

## Semantic flow

```text
RF-DETR / YOLOE / pose / audio workers
  -> typed GAMST observations
  -> temporal Context and Watch
  -> Qwen3.5-VL only for semantic question/Watch interpretation
```

A relation such as “person entered the doorway” is temporal state, not a one-frame VLM claim. A future relation provider may produce “person holding cup” observations, but it must remain inside the same perception boundary and include model provenance, score semantics, evidence IDs and realtime measurements.

## General object rotation and toss events

`vision.object_rotation` (`AngularMotionCounter`) accepts an object track,
ROI center and corresponding feature points. It emits phase, full-turn count,
angular velocity, direction and feature agreement confidence. It is not
coin-specific: the same contract serves coins, wheels, tires, fans and tools.

`vision.object_events` (`ObjectEventReducer`) consumes track support/contact,
vertical velocity and optional semantic face labels:

```text
held -> released -> airborne -> caught = one completed toss
heads -> tails / tails -> heads = one face flip
```

It emits timestamped toss/face-change events and counts. Qwen3.5-VL may supply
the optional object/face semantic label; it does not count every frame. Optical
flow/feature phase and the reducer do. A full coin demo still needs a capture
UI and real-video smoke run on the local Mac.
