---
name: gamst-coin-flip
description: Count how many times a tossed coin flips, with an iPhone at 240 fps and GAMST, and read the timestamped result from GAMST context. Use when someone asks to count coin flips, referee a flip trick shot, or compare what a 1 fps video model sees with what GAMST measures.
---

# Coin flip with GAMST

A hand-tossed coin is in the air for about 0.3–0.5 s and flips 10–30 times per second.
A video model that samples 1 frame per second sees at most one frame of the flight.
GAMST records every frame at 240 fps, follows one named object through the throw, counts its flips, and stores the
count as a timestamped observation an agent can read.

## What the agent gets

One `vision.rotation.refined` observation per throw. Its object looks like this (toss 7 of a real take):

```json
{"track_id": "metal-coin-7", "label": "metal coin",
 "attributes": {"flips": 10.2, "turns": 5.1, "rev_s": 13.55, "status": "counting",
                "coverage": 1.0, "duration_ms": 375, "trajectory": "free_fall",
                "started_ms": 1795496643254, "ended_ms": 1795496643629}}
```

| Field | Meaning |
|---|---|
| `flips` | half turns (heads → tails is one flip) |
| `turns` | full revolutions (`flips / 2`) |
| `rev_s` | turns per second over the counted flight |
| `status` | `counting`; `partial` = the coin was lost for part of the flight, so the count is a lower bound; `insufficient` / `undersampled` = no count |
| `coverage` | share of flight frames where the coin was seen |
| `trajectory` | `free_fall` for a real throw |
| `started_ms`, `ended_ms` | flight start and end on the GAMST runtime clock |

## The recipe

The pipeline is the built-in recipe `coin-flip` (`gamst/recipes/coin-flip.json`):

- capture: GAMST 240 iPhone app, live or a recorded take;
- find: YOLOE, label `"metal coin"`;
- follow: EfficientTAM;
- count: flips, up to 40 rev/s, started by ANALYZE.

`gamst recipe show coin-flip` prints it, what setup installs, and the frame rate it needs.

## Setup (once)

```bash
pip install gamst-sdk
gamst recipe setup coin-flip      # --dry-run prints the plan first
```

Setup installs only what this pipeline uses:

- the `rotation`, `yoloe` and `efficienttam` extras and the pinned CLIP;
- the YOLOE and EfficientTAM tiny weights, SHA256-checked;
- EfficientTAM at the pinned revision in `~/.cache/gamst/sources`.

Then build `apps/ios-highspeed-camera` (`xcodegen`, iOS 15+) onto an iPhone that records 240 fps. Put the phone and the
Mac on the same Wi-Fi, and set the Mac's address and port 9123 in the app.

Check a camera before relying on it. The command exits 1 if the camera's frame rate cannot count the recipe's speed:

```bash
gamst recipe check coin-flip --fps 240
```

A coin's outline repeats every half turn, so a camera counts up to `fps / 4` rev/s. Above that, a count would be an
alias: `run` refuses a take that is too slow, and the runtime reports `undersampled` instead of a number.

## Run

```bash
gamst recipe run coin-flip --iphone 9123 --store gamst-coin.db --refine-show
```

1. Fix the phone on a stand. Keep the whole flight in frame and use even light.
2. Optional: drag a box around the coin in the HUD. The phone then streams a native-resolution strip there, which
   keeps 240 fps when Wi-Fi cannot carry full frames.
3. Press REC (phone or space) just before the toss. Press STOP right after the catch. Shorter takes count faster.
4. Press ANALYZE (or key `a`).
   - The panel shows "GAMST is counting", the number of frames, the current stage and a progress bar.
   - The flips appear as soon as the count exists, about 30 s per toss on an M2 Pro MacBook Pro.
   - `--refine-show` opens `refined.mp4` (8× slow motion with the running count) when it is rendered.

Count a recorded take again with `gamst recipe run coin-flip --take TAKE_DIR --headless`.

Other `gamst demo rotation` options pass through and win. For example:

- `--refine-cached TAKE_DIR` makes ANALYZE show an already counted take at once. Keep it as the stage backup if the
  room's Wi-Fi or light ruins a live toss.
- `--record-dir DIR` sets where takes are saved.

Keep the label `"metal coin"`; with plain `"coin"` the detector missed the coin entirely. To count another flat object,
pass `--label`. The label is what lands in context and object memory.

## Read the result as an agent

Point the MCP server at the same store and read the throws:

```bash
gamst-mcp --store gamst-coin.db
```

- `get_observations` with `capability_id: "vision.rotation.refined"` returns the throws, newest first.
- Answer with `flips` and its `status`. When `status` is `partial`, say "at least N flips".

The demo process also keeps the named coin in its own `ObjectMemory`. A separate MCP process does not share that
memory, so read the label from the observation.

## Show it

- `refined.mp4` in the take folder replays each throw at 8× slow motion with the outline GAMST followed and the running
  count.
- `python -m gamst.demos.rotation_live render TAKE_DIR` re-renders it from `refined.json` without running a model.

## Limits (measured, not guaranteed)

- On a 7-toss take, 6 tosses were within one flip of a person's frame-by-frame count. The seventh was 6.4 flips low
  because the counted flight ended early.
- GAMST tended to count slightly low.
- Only coins were tested.
  - Another flat object needs only a different `--refine` label.
  - A bottle is not flat, so flip counting by outline thinning does not apply to it as is.
