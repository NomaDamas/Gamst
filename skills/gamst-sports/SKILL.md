---
name: gamst-sports
description: Analyse a sport from a camera or a recording with GAMST — hits, bounces and rallies of racket sports, possession, passes, positions and heat maps of team sports, repetitions and form against a demonstration, swings, body turns, billiards shots — by choosing a sports recipe and a model stack (ultralytics, roboflow, sota), checking the frame rate first, and reading the timestamped events over MCP. Use when someone asks how many hits, rallies, passes, reps, swings or turns happened, who did them, or when.
---

# Sports with GAMST

GAMST counts and times; the LLM explains. Never estimate a count from frames yourself: run a recipe and read its records.
The map of parts, stacks and limits is `docs/sports.md`.

## 1. Pick the recipe

| the request is about | recipe | `--sport` |
|---|---|---|
| hits, bounces, rallies, who hit (tennis, badminton, table tennis, padel, squash) | `racket-rally` | tennis, badminton, table-tennis, padel, squash |
| possession, passes, turnovers, positions, distance, heat map (football, basketball, futsal, handball, hockey, volleyball) | `team-field` | football, futsal, basketball, handball, ice-hockey, volleyball |
| repetitions of an exercise, punches, strokes, and how each compares with a demonstration | `body-reps` (needs `--exemplar`) | – |
| golf / baseball / tennis swings, with or without an impact sound | `swing` (needs `--exemplar`) | – |
| turns and airtime of a skater, diver, gymnast (experimental) | `body-rotation` | – |
| billiards shots and ball contacts | `billiards` | – |

Other sports: write a recipe file with the same parts (`docs/sports.md` section 2) and run it by path. Say that it is
unmeasured.

## 2. Check the camera before anything else

```bash
gamst recipe check racket-rally --sport table-tennis --fps 25  # exits 1: table tennis needs 33 fps
```

If it fails, tell the person which frame rate the sport needs and why (the numbers in `problems`), instead of running.
Pass on any `warnings` (the fastest shots may be lost at this frame rate).

## 3. Choose the stack and the timing

```bash
gamst recipe stacks racket-rally          # ultralytics | roboflow | sota (default), each realtime and post
gamst recipe setup racket-rally --stack sota --dry-run    # show what would be installed; ask before installing
```

- `ultralytics`: YOLO26 detector with Ultralytics' tracker and YOLO26-pose (AGPL-3.0 models).
- `roboflow`: RF-DETR, trackers ByteTrack, SigLIP team split.
- `sota`: RF-DETR, trackers BoT-SORT, TrackNetV3 ball model, RTMW pose.
- `--timing realtime` for a camera (models measured at start, a fallback if too slow); `--timing post` for a finished
  file (largest models, the ball model's temporal ensemble).

Setup downloads weights: state the files and sizes from the dry run and get the person's OK first.

## 4. Mark the court (positions in metres, bounces in or out)

Ask for the four corners of the court as the camera sees them (far left, far right, near right, near left), write
`court.json` `{"court": "tennis.singles", "points": [[x, y], ...]}` and pass `--court court.json`.

## 5. Run and read

```bash
gamst recipe run racket-rally --stack sota --sport tennis --video match.mp4 --court court.json --timing post --serve
```

Over MCP: `count_events` with capability `sports.contacts` (event `contact.hit`, `contact.bounce`), `sports.rally`
(`rally.ended`: hits, bounces, who, duration, `ended_by`), `sports.possession` (`possession.pass`, `.turnover`),
`sports.occupancy` (tracks: distance, speed, grid), `sports.form` (`form.compared`: angles vs the demonstration),
`sports.spin` (`spin.jump`), `sports.motion` (`shot.started`, `contact.objects`). `create_watch` on `rally.ended` or
`contact.hit` tells you when to speak.

## 6. What to say and not to say

- Passes and possession are **estimated** (every event says `estimated: true`); never call them official statistics.
- A sound alone is not a hit; a ball the model does not see is not counted. Say so when the ball was lost often
  (`ball.lost` events).
- Ball speed is in image units per second, not km/h.
- These recipes are not yet measured on real footage (`measured` in each recipe file). Say that with any number.
