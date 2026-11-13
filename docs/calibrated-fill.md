# Calibrated colored-liquid fill context

```mermaid
flowchart LR
  A[Fixed camera and calibrated container strip] --> B[OpenCV HSV color mask]
  B --> C[Dry prefix and wet suffix boundary check]
  C --> D[Declared height-to-volume interpolation]
  D --> E[Estimate or unknown in SDK / MCP context]
```

The capture owner supplies the stationary camera/container geometry, a clear
interior measurement strip, selected liquid color range and independently
measured height-to-volume table. `CalibratedColorFill` owns native image processing
and bounded context. This is a specific fixed-view colored-liquid observation
path, not general liquid segmentation, identity recognition, or a claim that
transparent water can be measured from color. Actual-scene validation remains
[#109](https://github.com/NomaDamas/Gamst/issues/109) under required family #37.

## Runtime and configuration

Use existing optional `humansense-sdk[vision]` (OpenCV4.8..<5, NumPy>=1.24).
No additional model, dependency install, camera activation or download occurs.
Core/profile discovery remains standard-library-only. Native runtime versions
are included in every output; development execution used OpenCV4.13.0 and
NumPy2.2.6 on Python3.10.11/macOS15.6.1arm64. Adapter/OpenCV use Apache-2.0.

```python
config = dict(
    width=128, height=128,
    roi=[16, 16, 112, 112],  # left, top, right, bottom; upper bounds exclusive
    hsv_lower=[100, 100, 50], hsv_upper=[130, 255, 255],
    volume_curve_ml=[[0, 0], [0.5, 100], [1, 400]],
    calibration_id='beaker-v1', container_id='beaker', coordinate_frame='image',
)
```

The ROI is a fixed interior vertical strip from the calibrated empty reference
at its bottom to the calibrated full reference at its top. It must remain inside
the container at all observed heights; walls/background must be excluded.
Image dimensions are32..2048 per axis and at most2Mi pixels. ROI must be at
least8px wide and16px tall. No resizing, cropping beyond that ROI, stabilization,
container tracking or perspective correction is implicit. Moving camera/container
or an altered crop requires a newly validated configuration.

HSV uses OpenCV uint8 BGR conversion with hue0..179 and saturation/value0..255;
see [official colorspace documentation](https://docs.opencv.org/4.11.0/df/d9d/tutorial_py_colorspaces.html)
and [inRange thresholding](https://docs.opencv.org/4.9.0/da/d97/tutorial_threshold_inRange.html).
Bounds are inclusive and ordered; minimum saturation32/value16 excludes a
configuration that deliberately treats achromatic/dark background as liquid.
There is one range, no circular-hue wrap. Select a suitable range/calibration for
the target scene; these limits do not guarantee actual fluid discrimination.

The 2..32 calibration points must increase strictly in both image-height fraction
and mL, start at(0,0), end at(1,positive volume), and stay within1,000,000mL.
Intermediate volumes use linear interpolation **between calibration points**;
a tapered vessel is not assumed cylindrical. Height fraction is measured in this
fixed image strip, not a 3D height measurement or volume fraction. The complete
configuration is hashed into the capability version and copied into owned tuples.
Do not mutate the provider's configuration after construction.

## Decision and output

`process(InputFrame(bgr, 'video', capture_ms, sensor=metadata))` requires a plain
uint8 NumPy image of exactly the configured size, integer capture timestamp, and
matching calibration ID/coordinate frame. Masked arrays are rejected. Only the
owned ROI copy is passed through `cvtColor` and `inRange`; images/masks are never
returned or retained between calls.

A row is dry at<=20% selected-color coverage and wet at>=80%. The algorithm counts
consecutive dry rows from the top and wet rows from the bottom. It requires at
least2 of each and at most2 unresolved transition rows between them. The estimated
boundary lies midway through that transition. It rejects missing bottom support,
missing headspace and fragmented/wide sloped profiles. This deliberately narrow
shape check can reject genuine foam/waves and can be fooled by suitable colored
backgrounds/reflections. It does not prove liquid identity or detect every occlusion.

`vision.container_fill` contains `height_fraction`, `volume_ml`, meniscus image
coordinate, ROI/calibration/container/color/curve provenance and native versions.
Unknown results set quantities to `None` and give a reason:

- `no_color_evidence`: cannot distinguish empty, clear liquid, wrong color or occlusion.
- `headspace_not_visible`: cannot distinguish full/overflow, painted object or obscured boundary.
- `bottom_color_support_missing`: no qualifying colored region reaches the bottom reference.
- `fragmented_or_sloped_profile`: selected rows do not form the permitted single boundary.

Endpoint values are not silently assigned0%/100%. `calibration_verified`,
`liquid_identity_verified` and `empty_or_full_verified` remain false. Unknown
observations replace earlier estimates in current context so a past volume is not
reused as a current estimate. No mass, density, calories or temperature is inferred.

Resolution intervals span transition-width/2 plus half a pixel on either side
of the inferred boundary, transformed through the same calibration curve.
These are **image-grid resolution bounds only**, not statistical confidence or
bounds on real accuracy, lighting effects, calibration error or false color matches.

## SDK / worker usage

```python
from humansense import ProcessProvider, SensorStreams

spec = ProcessProvider.capability('opencv.fill', config)
async def create():
    return await ProcessProvider.open('/explicit/vision-env/bin/python',
                                      'opencv.fill', config, inference_timeout=5)
streams = SensorStreams(human, max_packet_bytes=8*1024*1024)
streams.register(spec, create)
await streams.open('camera', 'fill-device', [spec.id])
# Submit matching InputFrame/SensorMetadata to the managed lane.
```

The explicit8MiB packet budget accommodates the full2Mi-pixel BGR limit
(6MiB of pixels plus framing); the default4MiB budget only fits smaller frames.
Choose budgets and queue capacity for the actual camera size.

For direct calls, import `CalibratedColorFill` from `humansense.providers.fill`,
construct it with the same config, register its `spec` and publish `process(frame)`
via `human.push_observation`. Preserve input subject, capture timestamp and sensor
metadata. Existing MCP `get_observations` and context API expose the result.

Frame mode is independent, with zero lookahead and no smoothing/history. Gaps
need no prior-image reset; `reset()` is a no-op. `process` and idempotent `close`
are serialized; processing after close fails. Direct native calls have no hard
timeout; use the owned worker for inference deadline/process cleanup. Processing
memory is bounded by the configured ROI and output by the capability JSON limits.
The user controls capture and context delivery cadence separately.

## Evidence and remaining acceptance

```sh
python3 -m unittest discover -s tests -p test_fill.py -v
PYTHONPATH=. python3 examples/validate_fill.py --output new-fill-report.json
```

Tests execute real OpenCV color processing and an actual selected worker through
SensorStreams/MCP:25/50/75% image levels map to50/100/250mL on the nonlinear test
curve; empty/full/gray/occluded/sloped controls return unknown. The standalone
smoke records deterministic image hashes, runtime versions, five expected outcomes
and per-call processing observations. This is native-runtime synthetic-image
evidence, not measured liquid/container calibration, live end-to-end latency or
sustained-resource acceptance. Required#109 includes actual held-out container
levels, misleading backgrounds, transparent-liquid failures and live metrics.
OpenPIV's separate flow acceptance remains#107; neither path alone closes#37.
