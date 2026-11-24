---
name: gamst-rotation
description: Count turns and rev/s of something spinning in the image plane (a fan, a wheel, a turntable) with GAMST, check first whether the camera is fast enough, and read the timestamped result from GAMST context. Use when someone asks how fast or how many times something rotates on camera.
---

# In-plane rotation with GAMST

The pipeline is the built-in recipe `rotation` (`gamst/recipes/rotation.json`):

- capture: a camera, a video file or a GAMST 240 take;
- find: a fixed box around the rotating part (`--roi`);
- follow: the box itself, since the axis does not move;
- count: in-plane turns by polar registration.

Polar registration compares what moved between frames, in angle around the axis. A static grill, frame or background
cancels out. No feature has to survive from frame to frame, so textured or repeating blades are measured too.

## Check the camera first

An object that looks the same `N` times per turn is countable only below `fps / (2 × N)` rev/s. For example, a 3-blade
fan has `N = 3`. Above that limit, the blades look as if they turn slowly or backwards, the wagon-wheel effect.

```bash
gamst recipe check rotation --symmetry 3 --max-rpm 1500            # which frame rate is needed?
gamst recipe check rotation --symmetry 3 --max-rpm 1500 --fps 240  # exits 1 if this camera cannot count it
```

Set `--symmetry` to the number of identical blades or spokes. Use 1 for an object with one mark.

| Camera | 3-blade fan | 5-spoke wheel | Marked turntable (N = 1) |
|---|---|---|---|
| 30 fps webcam | 5 rev/s (300 rpm) | 3 rev/s | 15 rev/s |
| 240 fps GAMST 240 | 40 rev/s (2400 rpm) | 24 rev/s | 120 rev/s |

`run` refuses a video or take whose frame rate is too slow. With a live camera, the frame rate is known only while it
runs: a turn above the limit is then reported `undersampled`, with no count.

## Setup and run

```bash
gamst recipe setup rotation        # only the rotation extra: OpenCV, NumPy, Pillow; no model weights
gamst recipe run rotation --camera 0 --roi .3,.2,.4,.6 --symmetry 3 --max-rpm 1500
gamst recipe run rotation --video fan.mp4 --capture-fps 240 --roi .3,.2,.4,.6 --symmetry 3 --max-rpm 1500
gamst recipe run rotation --take gamst-takes/take-20261125-101500 --roi .3,.2,.4,.6 --symmetry 3 --max-rpm 1500
```

- `--roi x,y,width,height` is normalized to the frame. Put the box around the whole rotating part, centered on the
  axis. The box itself is the object: nothing is segmented, so a light fan on a light ceiling still counts.
- Fix the camera, face the axis, and keep the box still.
- Declare a realistic top speed (`--max-rpm`). The check can only protect against speeds you declare.
- For a fast fan, record a take with the GAMST 240 iPhone app (`gamst demo rotation --iphone 9123`, REC, STOP). Then
  count it with `--take`.
- Add `--store gamst-rotation.db` to keep the records, and `--headless --report out.json` for a JSON summary.

## Read the result as an agent

`get_observations` with `capability_id: "vision.rotation"` returns the records. A spin ends with a
`rotation.ended` event:

```json
{"type": "rotation.ended", "axis": "in_plane", "status": "counting", "turns": 23.9, "rev_s": 12.0,
 "duration_ms": 1992, "capture_fps": 240.0, "max_countable_rev_s": 40.0, "expected_max_rev_s": 20.0}
```

- `turns` and `rev_s` are the count. With `status: "undersampled"` both are null: the frame rate could not vouch for
  the declared speed.
- `max_countable_rev_s` is `capture_fps / (2 × symmetry)`.
- A file or take that ends while the object still turns ends with `reason: "input_ended"` and the turns so far.

## Limits (measured, not guaranteed)

Measured on 2026-11-25. No real fan or wheel has been compared with a human count yet.

| Input | Reference | GAMST |
|---|---|---|
| Synthetic 240 fps 3-blade fan, 5 rev/s | 10 turns | 9.96 |
| Same, 12 rev/s behind a static grill | 24 turns | 23.90 |
| Same, 30 rev/s | 60 turns | 59.75 |
| Same, 50 rev/s (limit 40) | 100 turns | refused by `run`; 59.75 (an alias, 30 rev/s) if forced |
| Real 24 fps stock clip of a record label, 40 s (Pexels 5983043) | 22.65 turns (each checkpoint's frame rotated to match frame 0) | 22.49 (−0.7%) |
| Real 30 fps stock clip of a 5-blade ceiling fan (Pexels 4432095) | blades move about 36° per frame: the limit | refused at a realistic 250 rpm (needs 42 fps); 0.25 rev/s, an alias, when forced with `--max-rev-s 2.9` |

- The ceiling fan shows why the check matters. At 30 fps the forced count looked plausible and was wrong.
- Without `--roi`, regions from frame motion start late: the rate was right, but 6–51% of the turns were missed.
  That is why the recipe needs a box.
- The axis must stay still in the frame. A moving wheel, for example on a car, is not covered.
