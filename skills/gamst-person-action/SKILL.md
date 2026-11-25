---
name: gamst-person-action
description: Count how many times a person makes a pose or movement they demonstrated once (a signature gesture, a pass motion), with no training, and read the timestamped count from GAMST context. Use when someone asks how many times a person did a specific move on camera.
---

# Demonstrated pose or movement counting with GAMST

The pipeline is the built-in recipe `person-action-count` (`gamst/recipes/person-action-count.json`):

- input: a camera, a video file or a GAMST 240 take;
- user-defined input: an **exemplar**, the pose or movement demonstrated once (`gamst recipe enroll`);
- modules: MediaPipe Pose Landmarker (VIDEO mode, in its own process and environment) → pose channels (joint angles and wrist positions in
  shoulder widths, independent of where the person stands) → exemplar matching (hold or sequence) and
  the periodic repetition counter.

No model is trained. The exemplar is a feature vector (hold) or a short sequence of them (movement).

## Enroll the exemplar

```bash
gamst recipe setup person-action-count
gamst recipe enroll person-action-count --video demo.mp4 --start 2.0 --seconds 1.0 --out pose.json
gamst recipe enroll person-action-count --camera 0 --start 3 --seconds 1.5 --mode sequence --out pass.json
```

Exemplars enrolled before 2026-11-25 are refused: pose channels now correct for the image's aspect ratio. Enroll them
again.

`hold` takes the median pose of the window; `sequence` keeps the movement as up to 30 frames.

## Run and read the count

```bash
gamst recipe check person-action-count --fps 30 --max-reps-s 2
gamst recipe run person-action-count --camera 0 --exemplar pose.json
gamst recipe run person-action-count --video clip.mp4 --exemplar pose.json --headless --report out.json
```

Over MCP (`gamst mcp`), an agent reads the same records:

- `count_events` with `capability_id: "temporal.exemplar_events"`, `event: "exemplar.matched"`:
  the count and the evidence record IDs;
- `get_pipeline` / `configure_pipeline_module` to see the modules and switch one off or on.

## Limits

- One person: the most visible pose. Binding the count to one enrolled person (face and body
  matching) is the next module.
- `--enter` / `--exit` are mean joint-angle differences (units of pi radians); defaults 0.12 / 0.18.
- No human-counted take has been measured yet. Measure a take you counted yourself before quoting
  a number.
