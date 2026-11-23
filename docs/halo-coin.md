# Coin Halo: counting how many times a coin turns

The first GAMST Halo shows the one thing a VLM or LLM cannot do by looking at
frames: follow a fast physical event continuously and measure it. GAMST keeps
the high-frequency perception (tracking, phase, timing) outside the model and
hands the model a bounded, timestamped result only when something happened.

```text
camera (30-240 fps)
  -> YOLOE-26s "coin" in its own ProcessProvider worker, ~10 Hz   (names objects)
  -> ShapeTracks: every capture, re-find each object, fit ellipse + features
  -> ObjectSpinCounter   vision.object_spin       out-of-plane turns (tossed / table-spun coin)
  -> AngularMotionCounter vision.object_rotations in-plane turns (wheel facing the camera)
  -> Store + user Watch "spin ended" -> optional VLM call, once per episode
```

All three capabilities are additive providers. Engine, Store, Watch, MCP,
clock and ProcessProvider contracts are unchanged.

## Run it

```bash
pip install "gamst-sdk[coin]" "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"
gamst demo coin --fetch-models            # pinned YOLOE-26s + MobileCLIP2, sha256-checked
gamst demo coin --camera 0                # live
gamst demo coin --video toss.mov          # recorded, e.g. iPhone slow motion
gamst demo coin --detector round          # no model download; shapes are "round candidate"
```

`--vlm-endpoint URL --vlm-model NAME --vlm-key-env VAR` adds one
OpenAI-compatible VLM call per finished spin (what object, which face is up).
The VLM never estimates the count. `--store PATH` keeps derived records in
SQLite; raw frames are never stored. `--report out.json` writes the episodes.

## How a turn is counted

A coin turning about an axis in its own plane shows an ellipse whose
minor/major ratio is |cos(angle)|; its square is a sinusoid at the half-turn
rate. `ObjectSpinCounter` fits that sinusoid in 250 ms windows and integrates
the rate over the spin, so a coin that slows down is still followed. In-plane
rotation uses tracked feature points; `points_epoch` marks reseeded points so
the accumulated phase survives feature loss instead of resetting.

## Hardware limit

Uniform samples cannot distinguish a spin from its alias. The count is only
reported when the camera can resolve the fastest spin the application expects
(`expected_max_rev_s`, default 40 rev/s): turns per second must stay below
capture_fps / 4.

| Camera | Countable up to | Hand-tossed coin (36-40 rev/s) |
|---|---|---|
| Mac FaceTime HD, 30 fps (measured modes) | 7.5 rev/s | withheld as `undersampled` |
| iPhone Continuity Camera, 60 fps (measured modes) | 15 rev/s | withheld |
| iPhone slow motion, 240 fps (Apple specification) | 60 rev/s | counted |

The toss rate is from Diaconis, Holmes and Montgomery, *Dynamical bias in the
coin toss* (SIAM Review 2007): natural tosses spin at 36-40 rev/s for about
0.5 s. Below the limit GAMST reports that the spin was too fast for the camera
and never prints an aliased number.

## Measured on real footage (2026-11-23, Apple Silicon CPU)

| Clip (real, Pexels licence) | Before (round contours) | Now |
|---|---|---|
| Coin spun on a table, 25 fps slow motion, 375 frames | longest track 1 frame, 0 turns | coin observed in 344/375 frames, one track ID; 2 spin episodes, both `undersampled` (limit 6.2 rev/s) |
| Hand toss, 30 fps slow motion, 54 airborne frames | not tracked | coin observed in 54/54 airborne frames, one track ID; episodes `undersampled` (limit 7.5 rev/s) |
| Live Mac camera, 90 frames | — | runs at 30 fps with the detector in its worker |

Costs: YOLOE-26s through ProcessProvider 123-126 ms per detection (CPU,
≤1 MP input); per-capture tracking 3.6-5.7 ms p50 on 1080p frames. A Watch alarm
fired for every spin episode.

**Not yet validated: count accuracy.** No public clip has a known capture rate
fast enough for a hand toss, and stock "slow motion" clips change the coin's
face every 2-3 frames, so even a person cannot count them. Accuracy needs
real 240 fps recordings with frame-by-frame human counts, and the gate is
8 of 10 tosses exact. A 240 fps on-device iPhone capture app is the follow-up
for live tossed-coin counting; at 30-60 fps the live demo counts slower spins
only when the application declares a lower `expected_max_rev_s`.

The single VLM call is implemented; the measurement run returned HTTP 503 and a
timeout from `gemini-3.8-flash` (provider high demand), and the demo recorded
the failure instead of an answer.
