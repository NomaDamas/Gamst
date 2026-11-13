# Planar SRP-PHAT direction context

`humansense.providers.srp.SRPPHATDirection` scans a full-circle azimuth grid using the average PHAT-weighted cross-correlation of every microphone pair at its predicted far-field delay. This is executable NumPy DSP with no weights or automatic capture. It complements the two-microphone cone estimate and selected mono [beamforming bridge](audio-beamforming.md).

The method and coordinate conventions follow the standard [SRP-PHAT array-localization formulation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.doa.srp.html): source direction is defined relative to array axes. This implementation is independent and does not install or wrap pyroomacoustics.

## Supported geometry and input

Declare 3–8 synchronized microphones with unique ordered channel names and XY positions in metres. Coordinates must be within ±1 m, pair spacing at least 0.02 m, and the centered geometry's smaller singular value at least 0.01 m; collinear or nearly degenerate arrays are rejected. The source is assumed far away and within the array plane. Azimuth is counterclockwise from +X, 0–359 degrees. No elevation, distance, world location, speaker identity or calibrated confidence is inferred.

Sound speed is explicitly configurable within 300–380 m/s. A matching calibration ID, channel order and coordinate-frame identifier are mandatory; ID consistency does not validate hardware calibration, microphone polarity/gain or synchronization. Positions are copied at construction. Changing hardware settings requires a new verified profile.

Each input is one complete 20–200 ms window, at 16–96 kHz, shaped samples × channels, with normalized float32/float64 or signed int16 PCM. The capture timestamp denotes window end. No assembly or history is retained; the capture layer must ensure a window contains contiguous synchronized samples and no gap. SensorStreams supplies bounded queues and context lifecycle, but does not certify within-window continuity.

## Processing and ambiguity

The provider removes per-channel DC, applies a Hann window, computes zero-padded FFTs, PHAT-normalizes usable bins for every pair and evaluates the inverse-transform correlations at nearest-sample predicted delays. For a direction vector u, predicted t_i − t_j is dot(position_j − position_i, u)/sound_speed. Grid spacing is one of 1, 2, 5 (default) or 10 degrees; output grid spacing is not an accuracy guarantee. Integer-delay quantization can produce plateaus and low spatial resolution.

Silence/weak channels and inadequate broadband pair support yield unavailable results. The mean correlation peak must reach 0.05 and exceed the strongest candidate more than 10 degrees away by 0.02; otherwise the output has `available=false`, an explicit reason and null azimuth. These thresholds are uncalibrated rejection heuristics, not proof of correctness. Competing sources, reflections, near-field or elevated sources, synchronized electrical interference and duplicated mono can violate assumptions; no unique physical direction is guaranteed even for an accepted peak. No multi-source separation or tracking is claimed.

Work is bounded by 28 pairs, 360 grid directions and 65,536-point FFTs at the supported maxima. Pair correlations are processed sequentially; no recordings or raw spectra enter context. Input arrays are copied and never modified. Windows are independent; reset is a no-op and closed instances reject work.

```python
from humansense.providers.srp import SRPPHATDirection

spec = SRPPHATDirection.capability(window_ms=80)
streams.register(spec, lambda: SRPPHATDirection(
    positions_m=((0, 0), (0.2, 0), (0, 0.2), (0.2, 0.2)),
    channels=('front-left', 'front-right', 'rear-left', 'rear-right'),
    calibration_id='array-profile', coordinate_frame='array',
    sound_speed_m_s=343, window_ms=80, grid_degrees=5))
# Open the matching sensor lane and submit contiguous calibrated array windows.
```

Output includes azimuth or unavailable reason, geometry/calibration provenance, score/competitor, grid and delay resolution, capture interval, sample rate and explicit assumptions. `physical_calibration_verified` and `confidence_calibrated` remain false. To steer a selected microphone pair, compute its predicted delay from the accepted direction and that pair's ordered geometry; use a deliberate selection policy, not an unchecked cast of azimuth to delay.

## Evidence and open gates

`python3 -m unittest discover -s tests -p test_srp_direction.py -v` checks eight synthetic far-field directions, both PCM encodings, ownership, silence/narrowband rejection, invalid geometry/input, reset/close and actual SensorStreams context delivery. These are integer-delayed synthetic broadband signals, not real room or microphone evidence. No actual array accuracy or end-to-end latency claim is made.

The required hardware recording, reference geometry, room/competing-source trials and timing/resource evidence remain in #60 under family #27. This adds the SRP processing path but does not close those gates or the overall SDK goal. Model installations and Halo demonstrations remain deferred.
