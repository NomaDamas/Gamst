# Generic object rotation (the coin Halo example)

GAMST follows fast physical motion continuously and hands a semantic model a
bounded, timestamped result instead of asking it to inspect every frame.
`vision.rotation` is the capability; a coin, wheel, fan, disc or card is only
an optional detector label. There is no object-class-specific counting path.

```text
camera / recorded video
  -> standard GAMST intake
       human.register(RotationPlugin(...)) + push_video(frame)
       or SensorStreams / ProcessProvider("vision.rotation")
  -> regions from the latest detector observation in the Store
       (RF-DETR, YOLOE, a fine-tuned detector, any compatible provider)
       OR an explicit normalized ROI OR detector-free frame motion
  -> ShapeTracks every capture: ellipse + corresponding feature points
  -> vision.rotation observation every capture
       in-plane turns and rev/s; flip turns and rev/s; per-object status
       rotation.started / rotation.ended events
  -> Store / get_observations / MCP observation tools / user Watch
```

Detector identity is optional. A region without a detector label has
`label: null`; the low-fps semantic VLM can name the same timeline later.
`targets=["coin", "wheel"]` is an optional label filter, not a detector
requirement. An isolated ProcessProvider cannot read the Store, so its region
source is `roi`, `motion`, or `auto` (motion fallback); Store-backed detector
composition uses `RotationPlugin`.

## Run the viewer

The demo owns only capture and presentation. Every displayed number comes
from a committed GAMST record; alarms come from a user Watch on
`rotation.ended`.

```bash
pip install "gamst-sdk[rotation,yoloe]" \
  "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"

gamst demo rotation --fetch-models  # optional, pinned + sha256-checked assets
gamst demo rotation --camera 0 --targets coin,wheel,fan --expected-max-rev-s 8
gamst demo rotation --video toss.mov --device mps
gamst demo rotation --video motion.mp4 --detector motion
gamst demo rotation --video crop.mp4 --roi 0.2,0.2,0.4,0.4
```

`--detector auto` uses YOLOE when its pinned local assets exist and otherwise
uses motion regions. `--detector yoloe` requires those assets explicitly.
`--device cpu|mps` selects the YOLOE worker device. `--store PATH` retains
derived records only; raw frames are not stored. `--report out.json` writes
observed-frame coverage, statuses, record-derived events and Watch alarm count.

## Python and Watch contract

```python
from gamst.providers.rotation import RotationPlugin

plugin = RotationPlugin(
    human,
    region_source="auto",             # detections, roi or motion also accepted
    targets=["coin", "wheel"],       # optional
    expected_max_rev_s=8,
)
human.register(plugin)
await human.push_video(frame, timestamp_ms=capture_ms)

records = await human.get_observations(
    capability_id="vision.rotation", window="5s", limit=20,
)

watch = human.watches.create(
    owner="user",
    description="Tell me when an object stops rotating.",
    rule={
        "conditions": [{
            "capability_id": "vision.rotation",
            "path": "events",
            "op": "contains",
            "value": "rotation.ended",
            "within_ms": 1000,
        }],
        "cooldown_ms": 0,
        "max_per_minute": 60,
        "context_window_ms": 3000,
    },
)
```

Each per-object value contains `track_id`, optional `label`, `observed`,
`status`, `in_plane_turns`, `in_plane_rev_s`, `in_plane_direction`,
`flip_turns`, `flip_rev_s`, `capture_fps`, `max_countable_rev_s` (flips) and
`in_plane_max_countable_rev_s`.
`counting` means motion is active within the declared sampling limit; it is
not a claim that count accuracy has been validated. `idle` includes stationary
or not-yet-measurable tracks. `undersampled` withholds the affected flip or
in-plane count.

For a capture-timed or high-fps source, register `RotationProvider` in a
SensorStreams lane, or open the `vision.rotation` ProcessProvider profile and
register that worker in the lane. The same `vision.rotation` records and Watch
rules apply.

## How turns are measured

An object turning about an axis in its own plane shows an ellipse whose
minor/major ratio is `|cos(angle)|`; its square is a sinusoid at the half-turn
rate. `ObjectSpinCounter` fits that sinusoid in short windows and integrates
the rate, allowing the flip to slow down.

In-plane rotation has two estimators (`in_plane` option, `--in-plane`):

- `points` (default): corresponding feature points around the tracked center. `points_epoch` marks reseeded points,
  so phase is not differentiated across unrelated features. On a synthetic textured 3-blade fan the points did not
  survive from frame to frame and no turn was counted.
- `polar`: what moved between frames, correlated in angle around the axis (`PolarRotationCounter`). Static grills
  cancel, and no feature has to survive. The `rotation` recipe uses it. With an ROI, the box is the object and is not
  segmented: a light fan on a light ceiling was observed in 10 of 529 frames by segmentation, and in all frames as a box.

`RotationPlugin.finish()` (called by the demo when a file or take ends) reports a rotation still running at the end
as `rotation.ended` with reason `input_ended`.

Uniform samples cannot distinguish fast motion from its alias. Flip counts are
reported only when `capture_fps / 4 >= expected_max_rev_s` (default 40 rev/s).
Otherwise the event is `undersampled` and `turns`/`rev_s` are null. In-plane
counts are withheld the same way unless `capture_fps / (2 × symmetry) >= expected_max_rev_s`.
`symmetry` is how many times the object looks the same per turn, for example 3 for a 3-blade fan.

| Camera | Flip-countable up to | Hand-tossed coin (36-40 rev/s) |
|---|---|---|
| 30 fps | 7.5 rev/s | withheld as `undersampled` |
| 60 fps | 15 rev/s | withheld |
| 240 fps | 60 rev/s | within the sampling limit |

The toss rate is from Diaconis, Holmes and Montgomery, *Dynamical bias in the
coin toss* (SIAM Review 2007). A slowed-down stock export is not proof of its
original capture rate; use real capture timestamps or an explicit, justified
`--capture-fps`.

## Measured on real footage (2026-11-23, this Apple Silicon Mac)

End-to-end runs used `push_video` for rotation, YOLOE through a SensorStreams /
ProcessProvider detector lane when enabled, Store `get_observations`, and a
user Watch. No private detector/counting loop produced the records.

| Real clip and region source | Frames | Frames with an observed object | Statuses (object-frame counts) | `rotation.ended` / Watch alarms |
|---|---:|---:|---|---:|
| `853779.mp4`, YOLOE `coin,wheel,fan`, MPS | 375 | 343 (91.47%) | 133 idle, 238 undersampled | 3 / 3 |
| `853779.mp4`, detector-free motion | 375 | 319 (85.07%) | 147 idle, 235 undersampled | 5 / 5 |
| `8369956.mp4`, detector-free motion | 332 | 290 (87.35%) | 499 idle, 30 counting, 57 undersampled | 18 / 18 |

The first run's three ended flip episodes were all `undersampled` at the
25 fps clip rate with `expected_max_rev_s=8`; counts remained null. Coverage is
for the whole clip, not a claim about a manually selected airborne interval.

A separate 25-frame sample of the same real clip (every 15th frame, ≤1 MP
input) measured YOLOE detection latency through ProcessProvider:

| Device | p50 | Minimum | Maximum | Detections returned |
|---|---:|---:|---:|---:|
| CPU | 157.07 ms | 144.36 ms | 289.53 ms | 20 |
| MPS | 45.65 ms | 29.63 ms | 6311.56 ms | 20 |

The MPS maximum includes first-call warmup. MPS is now accepted by both
`yoloe_native.configuration` and the capability device contract.

**Counting accuracy is not claimed.** These runs establish intake, coverage,
status/event/Watch behavior and detection latency, not exact turn counts.
Accuracy requires real high-fps recordings with frame-by-frame human counts.

## Take refinement against frame counts (2026-11-24, iPhone SE 2, 240 fps)

`--refine "metal coin"` was run on two real takes:

- Capture: native 554×1080 strip, hand-tossed coin.
- Detector: YOLOE-26s, text prompt, every 4th frame.
- Tracker: EfficientTAM tiny 512 on MPS, at about 20–23 frames/s.

Ground truth comes from counting edge-on frames by eye in coin-centred crops (the assistant's
count), except white shirt #1, which the person who threw it counted in the rendered video: 11 flips
(half turns). The assistant's count for that throw was ≈4.2 turns, so the other assistant counts
may also be low. `turns` are full revolutions; one flip is half a turn.

| Toss | Frame-counted | GAMST refined | Note |
|---|---:|---:|---|
| Black shirt #1 | ≈3.4 turns, ~10 rev/s | 3.5 turns, 9.9 rev/s, `free_fall` | one object followed through the whole flight |
| Black shirt #2 | not countable by eye (coin in front of the face) | `partial` (coverage 52%) | reported as a lower bound, not a count |
| Black shirt #3 | — | not found | YOLOE never named the coin |
| White shirt #1 | 11 flips = 5.5 turns (thrower's count) | 5.1 turns, 15.6 rev/s | |
| White shirt #3 | ≈4.0 turns, ~15 rev/s (assistant count) | 4.5 turns, 16.8 rev/s | |
| White shirt, other throws | not counted | 0.5–2.0 turns over 90–120 ms flights | short fragments; not trustworthy |

Findings:

- With a correct seed, the segmenter keeps one identity through face, edge and glint. The live
  motion tracker could not do this.
- Error is −0.4 turns (white #1, thrower's count) and +0.1 turns (black #1, assistant count).
- Recall is limited by naming. YOLOE scores the coin at 0.2–0.9 in only a few frames per throw,
  and sometimes names a static shirt detail instead. Following each of the best 3 hits on its
  own, and keeping the best-observed throw, stops a wrong hit from corrupting the right one.
- A wrong hit can still produce a short fragment, which is reported with its own short duration.

Counting accuracy is still not claimed.

## The coin-flip recipe on the Pipeline

`gamst recipe run coin-flip` and `gamst demo rotation --iphone/--iphone-replay` build the same
`gamst.pipeline.Pipeline`. There is one implementation; the demo only keeps its own defaults
(count on Stop, every throw).

```text
camera (gamst240)  IPhone240Input      gamst/inputs.py   the app's TCP stream or a raw.gmst replayed;
                                                         phone commands record, camera, zoom, crop
  -> record        TakeRecorder        gamst/takes.py    raw.gmst, summary.json (+ hud.mp4 from the view)
  --take-->        derived input: a finished take folder (media, never in context)
  -> count         TakeFlips           gamst/takes.py    refine_take in its own process (find: YOLOE,
                                                         follow: EfficientTAM), then vision.rotation.refined
view halo-coin     HaloCoinView        gamst/halo.py     keys and mouse -> commands; throws read from records
```

While it runs, `get_pipeline` lists `record` and `count`; `configure_pipeline_module` can switch
`count` off, and takes are still recorded. `python -m gamst.takes refine TAKE_DIR ...` counts a
saved take without a runtime, and `python -m gamst.takes render TAKE_DIR` re-draws `refined.mp4`.

The move did not change the count. Take `140345` (below) was replayed through the code before the
move (commit `30b705f`) and through the Pipeline, headless, same weights, MPS, on 2026-11-25:

| Toss | Before (flips) | Pipeline (flips) | Flight frames (both) |
|---:|---:|---:|---|
| 1 | 8.8 | 8.8 | 424–536 |
| 2 | 5.2 | 5.2 | 1157–1243 |
| 3 | 14.4 | 14.4 | 1786–1899 |
| 4 | 8.6 | 8.6 | 2435–2503 |
| 5 | 11.6 | 11.6 | 3109–3219 |
| 6 | 2.4 | 2.4 | 3736–3820 |
| 7 | 10.2 | 10.2 | 4384–4475 |

- Every throw: `refined.json` results are equal field for field, including the per-frame
  ellipses. That holds for `gamst recipe run coin-flip --all-throws` and for
  `gamst demo rotation --iphone-replay ... --refine "metal coin"`. The 7 stored `vision.rotation.refined` records carry the same flips. Both match the
  table below, which was counted on 2026-11-24.
- The recipe's default (one throw per take, models kept loaded) counted toss 5 at 11.4 flips
  both before and after. It found 55 detections in both runs.
- The take each run saved again has the same `raw.gmst` bytes (sha256) as the original.
- Wall time for the whole 21.5 s take on an M2 Pro, one run each: every throw 6 min 50 s before
  and 7 min 37 s after; the default 1 min 28 s before and 1 min 39 s after. The 90 s test suite
  ran on the same machine during the Pipeline runs, so these times do not show a slowdown or rule
  one out.

## Seven tosses against a person's flip count (2026-11-24, take `140345`)

White shirt, iPhone SE 2, 240.2 fps, native 554×1080 strip, 21.5 s, 5,173 frames.
The thrower counted flips (half turns) frame by frame in `refined.mp4`.

| Toss | Person (flips) | GAMST (flips) | GAMST turns · rev/s | Flight GAMST counted |
|---:|---:|---:|---|---:|
| 1 | 9 | 8.8 | 4.4 · 9.6 | 462 ms |
| 2 | 6 | 5.2 | 2.6 · 7.4 | 354 ms |
| 3 | 15 | 14.4 | 7.2 · 15.5 | 466 ms |
| 4 | 15 | 8.6 | 4.3 · 15.5 | 279 ms |
| 5 | 12 | 11.6 | 5.8 · 12.7 | 454 ms |
| 6 | 3 | 2.4 | 1.2 · 3.6 | 346 ms |
| 7 | 10 | 10.2 | 5.1 · 13.6 | 375 ms |

Findings:

- 6 of 7 tosses are within one flip of the person's count.
- GAMST is slightly low on 6 of 7 tosses.
- Toss 4 is low by about 6 flips because the counted flight (279 ms) is much shorter than the
  rate implies (15 flips at 15.5 rev/s ≈ 480 ms). The parabola window stopped early; the count
  itself is not a sampling error.

Time on an Apple M2 Pro MacBook Pro (MPS), for toss 7's 169-frame (0.7 s) window:

| Step | Time |
|---|---:|
| Decode 169 frames | 0.5 s |
| YOLOE on every 4th frame (43 frames, 2 tiles each) | 4.7 s |
| EfficientTAM + count, per seed | 13–14 s |
| One toss, 3 seeds (current default) | ≈ 45 s |

The whole take, 7 tosses including YOLOE over all 21.5 s, took about 8 minutes.

Each throw is handed to agents as GAMST context. This is toss 7's object in the stored
`vision.rotation.refined` observation; times are on the runtime clock, and the capture clock is
kept in the event as `capture_timestamp_ms`:

```json
{"track_id": "metal-coin-7", "label": "metal coin",
 "attributes": {"flips": 10.2, "turns": 5.1, "rev_s": 13.55, "status": "counting",
                "coverage": 1.0, "duration_ms": 375, "trajectory": "free_fall",
                "started_ms": 1795496643254, "ended_ms": 1795496643629}}
```
