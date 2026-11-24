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
`flip_turns`, `flip_rev_s`, `capture_fps` and `max_countable_rev_s`.
`counting` means motion is active within the declared sampling limit; it is
not a claim that count accuracy has been validated. `idle` includes stationary
or not-yet-measurable tracks. `undersampled` withholds the affected flip count.

For a capture-timed or high-fps source, register `RotationProvider` in a
SensorStreams lane, or open the `vision.rotation` ProcessProvider profile and
register that worker in the lane. The same `vision.rotation` records and Watch
rules apply.

## How turns are measured

An object turning about an axis in its own plane shows an ellipse whose
minor/major ratio is `|cos(angle)|`; its square is a sinusoid at the half-turn
rate. `ObjectSpinCounter` fits that sinusoid in short windows and integrates
the rate, allowing the flip to slow down. In-plane rotation uses corresponding
feature points around the tracked center. `points_epoch` marks reseeded points
so phase is not differentiated across unrelated features.

Uniform samples cannot distinguish fast motion from its alias. Flip counts are
reported only when `capture_fps / 4 >= expected_max_rev_s` (default 40 rev/s).
Otherwise the event is `undersampled` and `turns`/`rev_s` are null. In-plane
episode status uses its separate `capture_fps / 2` phase-wrap limit.

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

Ground truth comes from counting edge-on frames by eye in coin-centred crops.

| Toss | Frame-counted | GAMST refined | Note |
|---|---:|---:|---|
| Black shirt #1 | ≈3.4 turns, ~10 rev/s | 3.5 turns, 9.9 rev/s, `free_fall` | one object followed through the whole flight |
| Black shirt #2 | not countable by eye (coin in front of the face) | `partial` (coverage 52%) | reported as a lower bound, not a count |
| Black shirt #3 | — | not found | YOLOE never named the coin |
| White shirt #1 | ≈4.2 turns, ~15 rev/s | 5.1 turns, 15.6 rev/s | flight start taken ~14 frames early |
| White shirt #3 | ≈4.0 turns, ~15 rev/s | 4.5 turns, 16.8 rev/s | |
| White shirt, other throws | not counted | 0.5–2.0 turns over 90–120 ms flights | short fragments; not trustworthy |

Findings:

- With a correct seed, the segmenter keeps one identity through face, edge and glint. The live
  motion tracker could not do this.
- Error is about +0.1 to +0.9 turns.
- Recall is limited by naming. YOLOE scores the coin at 0.2–0.9 in only a few frames per throw,
  and sometimes names a static shirt detail instead. Following each of the best 3 hits on its
  own, and keeping the best-observed throw, stops a wrong hit from corrupting the right one.
- A wrong hit can still produce a short fragment, which is reported with its own short duration.

Counting accuracy is still not claimed.
