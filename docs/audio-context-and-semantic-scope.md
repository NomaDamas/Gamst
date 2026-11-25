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

## General semantic perception flow

Qwen3.5-9B-VL is a **general GAMST perception-layer model**. Its purpose is
not to be a coin detector or a final answer pipeline. At a low cadence or at
event boundaries it understands the semantic flow of a scene:

- what objects and participants are present
- what actions are unfolding
- which state changed between windows
- whether an event appears to start, continue or end
- what the current scene means in natural language

It can provide semantic hints for object labels, rotation/toss state and
heads/tails, but GAMST does not treat those hints as calibrated measurements.
The high-rate numeric and temporal work stays in the additive perception
capabilities and existing tracking/flow components.

```text
camera
  -> Qwen3.5-VL general scene semantic perception (low cadence)
  -> light detector/tracker and existing OpenCV/timeseries components
  -> additive GAMST perception capabilities:
       vision.rotation (in-plane turns and out-of-plane flips)
  -> GAMST Context / Watch / overlay
```

## Generic object rotation

Rotation is a fast-lane capability, not an object-specific application. Coins,
wheels, fans and cards are optional detector labels on the same timeline. The
full description, hardware limits and measured results are in
[docs/halo-coin.md](halo-coin.md).

```text
any detector observation / explicit ROI / detector-free motion region
  -> ShapeTracks re-finds each region every capture (ellipse + features)
  -> vision.rotation: in-plane turns + out-of-plane flips, rev/s and status
  -> rotation.started / rotation.ended events
  -> GAMST Context / user Watch / overlay
```

`RotationPlugin` consumes the latest detector boxes already in the Store via
`human.register(plugin)` and `push_video`. `RotationProvider` also runs in a
SensorStreams/ProcessProvider lane with ROI or motion regions. Semantic labels
are optional; unknown objects remain measurable and the low-cadence semantic
VLM may describe them later. The providers are additive to the existing GAMST
boundary and do not restructure the engine, MCP, store or clock.
