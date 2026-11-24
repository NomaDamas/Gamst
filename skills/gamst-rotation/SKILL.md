---
name: gamst-rotation
description: Count turns and rev/s of something spinning in the image plane (a fan, a wheel, a turntable) with GAMST, check first whether the camera is fast enough, and read the timestamped result from GAMST context. Use when someone asks how fast or how many times something rotates on camera.
---

# In-plane rotation with GAMST

The pipeline is the built-in recipe `rotation` (`gamst/recipes/rotation.json`):

- capture: a camera, a video file or a GAMST 240 take;
- find: a fixed box around the rotating part (`--roi`);
- follow: GAMST shape tracks;
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

- `--roi x,y,width,height` is normalized to the frame. Put the box around the whole rotating part, centered on the axis.
- Fix the camera, face the axis, and keep the box still.
- For a fast fan, record a take with the GAMST 240 iPhone app (`gamst demo rotation --iphone 9123`, REC, STOP). Then
  count it with `--take`.
- Add `--store gamst-rotation.db` to keep the records, and `--headless --report out.json` for a JSON summary.

## Read the result as an agent

`get_observations` with `capability_id: "vision.rotation"` returns the records. A spin ends with a
`rotation.ended` event:

```json
{"type": "rotation.ended", "axis": "in_plane", "status": "counting", "turns": 23.72, "rev_s": 11.91,
 "duration_ms": 1992, "capture_fps": 240.0, "max_countable_rev_s": 40.0, "expected_max_rev_s": 20.0}
```

- `turns` and `rev_s` are the count. With `status: "undersampled"` both are null: the frame rate could not vouch for
  the declared speed.
- `max_countable_rev_s` is `capture_fps / (2 × symmetry)`.

## Limits (measured, not guaranteed)

Measured only on synthetic 240 fps videos of a 3-blade fan (2026-11-25). No real fan or wheel has been compared with
ground truth yet.

| Synthetic fan | True turns | GAMST |
|---|---|---|
| 5 rev/s | 10 | 9.96 |
| 12 rev/s behind a static grill | 24 | 23.72 |
- Without `--roi`, regions from frame motion start late: the rate was right, but 6–51% of the turns were missed.
  That is why the recipe needs a box.
- The axis must stay still in the frame. A moving wheel, for example on a car, is not covered.
