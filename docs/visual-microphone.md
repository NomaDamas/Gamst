# Phase-based Visual Microphone: offline reconstruction

This numerical path reconstructs an **unvalidated, arbitrary-unit waveform** from
small visible surface vibrations in an explicitly supplied high-speed grayscale
ROI sequence. It does not interpret speech, measure acoustic pressure or turn
normal 30fps video into a microphone. It requires neither learned weights nor
MATLAB. No cameras, downloads or installations start implicitly.

```mermaid
flowchart LR
  A[Declared contiguous high-speed ROI window] --> B[Complex steerable bands]
  B --> C[Weighted phase traces]
  C --> D[Whole-window alignment and high-pass]
  D --> E[Local normalized waveform and numeric metadata]
```

This is the offline numerical engine, not yet a SensorStreams provider. Capture
provenance, source/clock/gap contracts, bounded owned-worker execution and MCP
integration remain required under [#144](https://github.com/NomaDamas/Gamst/issues/144).
The caller must keep raw waveform arrays outside LLM Context. A downstream
provider will expose bounded numerical summaries only.

## Algorithm and provenance

Based on the [official Visual Microphone basic release](https://people.csail.mit.edu/mrub/VisualMic/)
and its `vmSoundFromVideo.m`, `vmAlignAToB.m`, `buildSCFpyr.m`,
`buildSCFpyrLevs.m` and `rcosFn.m`. Inspected official `VMSlim.zip` SHA256:
`443988c19051b8dbdc18d14e153b95ea16e91f70b5bd8e35f035ecb7b35a2b97`.
The source archive's MIT notice and original author attribution are preserved in
[licenses/visual-microphone.txt](../licenses/visual-microphone.txt).
This reference dates to SIGGRAPH 2014; it is not claimed to be a current SOTA
benchmark winner or the full paper implementation.

The concrete Python adaptation computes the first complex steerable pyramid
level with two orientations, raised-cosine radial LUT and analytic angular LUT.
Each frame's phase relative to the first frame is wrapped to [-pi,pi), weighted
by squared amplitude, averaged spatially and divided by total amplitude. Traces
are aligned by full-window cross-correlation and circularly shifted, summed,
then passed through a third-order Butterworth high-pass at 0.05 of Nyquist.
Following the basic release, the first ten filtered samples are replaced with
the filtered mean, and the waveform is scaled to [-1,1].

Explicit departures/limitations:

- Only one scale/two orientations; no pyramid downsampling, other levels or
  residuals. Only fine texture in that spatial band contributes. Texture outside
  that band can leave quantization/noise as the dominant signal.
- Input is already grayscale uint8 ROI; no automatic ROI detection, color-channel
  conversion, spatial resizing, decoding or camera-motion compensation.
- Float64 FFT/interpolation replaces MATLAB's numerical/runtime path. Original
  MATLAB parity has **not** been executed; `official_matlab_parity_verified=False`.
- Empty initial orientation bands are excluded; the strongest initial band is
  the alignment reference. The original always uses its first band. Texture
  loss later in a selected band causes abstention instead of division by zero.
- No spectral subtraction, rolling-shutter reconstruction, pressure calibration,
  exposure correction or learned denoising. Circular alignment can wrap samples;
  complete-window lookahead means independently reconstructed adjacent windows
  cannot simply be concatenated into continuous audio.

## API, bounds and time semantics

```python
from humansense.providers.visual_microphone import recover

result = recover(grayscale_roi_frames, capture_fps=2200)
waveform = result['waveform']  # Owned local ndarray, or None on abstention.
metadata = result['metadata']
```

`grayscale_roi_frames` is a plain NumPy uint8 array with shape `(N,H,W)`:
128..4096 frames, each dimension 32..128 pixels, at most16MiB input. Bounds and
plain-array type are checked before an owned copy. Masked/subclass arrays reject.
Caller must synchronize input mutation while snapshotting. Computation holds
one bounded sequence and small per-frame FFT/band workspaces; it does not store
a pyramid for every frame. Input hash identifies the owned contiguous pixel bytes;
it is not proof of the original video file, timestamps or frame completeness.

`capture_fps` is an explicitly declared 100..100000Hz actual acquisition rate,
not the container playback rate. It sets waveform sample rate, Nyquist bound,
frequency bins and high-pass cutoff. Sample span is `(N-1)/fps`; analysis duration
is `N/fps`. The function cannot verify timing regularity, shutter exposure,
dropped frames or clock calibration. Never fill dropped frames and claim an
unchanged sample rate. Split at source/ROI/capture-clock discontinuities.

The output peak is the largest Hann-windowed, detrended waveform power bin
between the high-pass cutoff and Nyquist. `peak_fraction` is a fraction of power
in those searched bins, **not confidence, acoustic SNR or validated quality**.
Camera motion, aliasing, lighting changes, compression and noise may produce a
peak with no sound. Finite high-pass attenuation also remains below the declared
cutoff. Static/no-texture windows return no waveform; the numerical thresholds
are not a calibrated speech/activity detector. `quality='not_assessed'`,
`hardware_validated=False`, `intelligibility_validated=False` always remain.

The function is synchronous and bounded by input size, not a wall-clock deadline.
Use an owned worker for an application requiring cancellation/full-duplex
isolation; that integration is a remaining deliverable in #144. The source
window must be complete before reconstruction starts. Do not report CPU processing
time alone as capture-to-output latency.

## Dependencies and evidence

Optional extra `visual-microphone`: NumPy>=1.24 and SciPy>=1.10. Tested locally on
Python3.10.11, NumPy2.2.6 and SciPy1.15.3. No packages or trained weights were
installed for this implementation. Discovery imports neither library.

```sh
python3 -m unittest discover -s tests -p test_visual_microphone.py -v
```

Tests exercise actual numerical reconstruction of known subpixel vibration,
analytic complex-band translation, high-pass/normalization, static/flat windows,
texture loss, odd geometry, single-orientation texture, bounds and repeated-call
isolation. A generated 125Hz displacement sampled at2000Hz yields the expected
125Hz bin and >0.999 absolute correlation between pre-filter phase trace and
known displacement. This is a mathematical fixture, not a physical microphone
or actual high-speed camera result. MATLAB/runtime parity, synchronized real
video/microphone evidence, quality, sustained processing and live latency remain
mandatory in #144/#35; this engine alone does not close that capability family.
