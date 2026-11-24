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

## Setup (once)

1. Install the extras, then fetch the pinned weights (YOLOE and EfficientTAM tiny, SHA256-checked).
2. Check out EfficientTAM at the pinned revision.
3. Build `apps/ios-highspeed-camera` (`xcodegen`, iOS 15+) onto an iPhone that records 240 fps. Put the phone and the Mac
   on the same Wi-Fi, and set the Mac's address and port 9123 in the app.

```bash
pip install "gamst-sdk[rotation,yoloe,efficienttam]" \
  "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"
gamst demo rotation --fetch-models
git clone https://github.com/yformer/EfficientTAM && git -C EfficientTAM checkout abcd061
```

## Run a take

```bash
gamst demo rotation --iphone 9123 --device mps \
  --refine "metal coin" --efficienttam-source EfficientTAM \
  --record-dir gamst-takes --store gamst-coin.db
```

1. Fix the phone on a stand. Keep the whole flight in frame and use even light.
2. Optional: drag a box around the coin in the HUD. The phone then streams a native-resolution strip there, which
   keeps 240 fps when Wi-Fi cannot carry full frames.
3. Press REC (phone or space), toss, catch, press STOP.
4. Wait for the HUD to show the counts. On an M2 Pro MacBook Pro this takes about 45 s per toss.

Use `"metal coin"` as the label. With plain `"coin"`, the detector missed the coin entirely.
The label is what lands in context and object memory.

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
