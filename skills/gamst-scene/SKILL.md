---
name: gamst-scene
description: Run several GAMST perception parts together on one capture clock, from a camera and a microphone or from a video file and its own sound - movements of a person, words that are said, any registry role (depth, hands, sound events, speech activity, ...) and optionally a VLM's interpretation - and read them aligned in time. Use when one recipe is not enough, or when someone asks what happened and what was said at the same moment.
---

# Several things at once, on one clock

Recipe `scene` composes the parts of `person-action-count` and `person-term-count`; it has no module of its own.

| Part | `--without` | Needs |
|---|---|---|
| `movement` (pose, channels, exemplar) | `--without movement` | `--exemplar move.json` |
| `person` (only the enrolled person) | `--without person` | `--person me.json` |
| `speech` (recognizer, terms) | `--without speech` | `--terms 대박` or a terms file |
| `voice` (only the enrolled voice) | `--without voice` | `--voice me.json` |
| `repetition`, `semantic` (VLM) | off; `--with NAME` | a VLM endpoint for `semantic` |

A part whose enrollment is missing is refused with the option that leaves it out. An input nothing reads is not opened.

```bash
gamst recipe setup scene --without person --without voice
gamst recipe enroll scene --video a.mp4 --video b.mp4 --negative c.mp4 --mode sequence --out move.json
gamst recipe run scene --video film.mp4 --exemplar move.json --terms 대박 --without person --without voice
gamst recipe run scene --camera 0 --microphone --without movement --without voice --terms 대박
gamst recipe run scene --video film.mp4 ... --with semantic            # VLM: segments, with this run's transcript and counts
gamst recipe run scene --video film.mp4 ... --with-role audio.events --with-role vision.hand
gamst recipe roles                                                     # what can be added, what cannot and why
```

- `--video FILE` reads the pictures and the file's own sound (ffmpeg and ffprobe on PATH; local files only, nothing written to disk).
- `--camera 0 --microphone` uses the host's clock for both. A camera frame is late by the camera's own latency, which GAMST does
  not know. Measure it (a clap in view), then `--offset-ms camera=-80`. The offset is in the report.
- Recordings of different files need `--offset-ms ID=MS` (0: they start together). Live and recorded inputs do not mix.
- `--input ID=SOURCE[:VALUE]` names a source for one input (two cameras, a separate sound file).
- An optional role pauses while a needed part is behind. If an input fails, the run stops and names it.

Read the result with MCP/SDK `get_timeline` (one row per moment, all modules), `count_events`, or `gamst recipe export`.

## A sensor (IMU and similar)

In a recipe file, declare what the numbers are; nothing is guessed:

```json
{"id": "imu", "use": "sensor", "sources": ["sensor"], "modality": "imu",
 "time": {"column": "t", "unit": "s"},
 "columns": [{"name": "ax", "unit": "m/s2", "scale": 10}, {"name": "ay", "unit": "m/s2", "scale": 10}]}
```

`--sensor file.csv` (header row) or `file.jsonl`. `repetition` (imu), `exemplar` and `rhythm` (`"channel": "ax"`) consume it; `enroll ...
--sensor file.csv@4-5` learns a movement, tied to these columns, units and scales. `scale` divides a channel: exemplar distances are
compared across channels and the default threshold assumes values of about one. Live: the host calls `Pipeline.push`. No hardware drivers.

## Limits

Not measured: accuracy of any part here, the VLM, and whether one machine keeps up with several real models at once. Run together with
real models: nothing except pose + YOLOE (throughput only).
