# Two-microphone direction context

`humansense.providers.direction.GCCPHATDirection` is an executable NumPy GCC-PHAT DSP provider for two synchronized microphones. The [SpeechBrain GCC-PHAT API](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.multi_mic.html) and [Pyroomacoustics TDOA API](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.sync.html) describe this method family. This SDK implementation uses its own bounded NumPy calculations; it does not load either toolkit or model weights. SDK code is Apache-2.0; NumPy has its own license. Tested locally with NumPy 2.2.6/Python 3.10.11; the existing optional audio extra supplies NumPy, while core discovery remains dependency-free.

## Input and geometry

Construct with explicit `spacing_m` (.02–1 m), `sound_speed_m_s` (300–380), `calibration_id`, two distinct ordered channel names and a coordinate-frame name. Speed is not inferred from temperature. `SensorMetadata` must match that profile, channel order and coordinate frame exactly. Calibration IDs identify application-owned geometry/channel/synchronization records; matching a string is not a physical calibration test. Supply independent synchronized microphone channels, not mono duplicated into stereo, and correct channel latency/skew before calling the provider.

Input is a NumPy samples-by-two float32/float64 array normalized to [-1,1], or signed native-endian int16 PCM. Bytes, mono, unsupported dtypes, missing/mismatched metadata, nonfinite samples and incorrect window lengths are rejected. Sample rate is 16–96 kHz, configured window 20–200 ms (default80). Each input contains exactly one contiguous complete window. Its timestamp is the capture end; the start is end minus window. Aggregate smaller capture chunks upstream without joining across gaps. Independent windows retain no PCM state, have no lookahead beyond their collection window, and can be submitted continuously through `SensorStreams`. Reset never joins old/new audio.

```python
from humansense import SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.direction import GCCPHATDirection

# Within a consented async HumanSense application:
async def register_array(human):
    streams = SensorStreams(human)
    spec = GCCPHATDirection.capability(window_ms=80)
    streams.register(spec, lambda: GCCPHATDirection(
        spacing_m=.2, sound_speed_m_s=343, calibration_id='array-profile-1',
        channels=('mic0', 'mic1'), coordinate_frame='array', window_ms=80))
    await streams.open('device', 'array', [spec.id])
    return streams

# For each already assembled contiguous 80ms window:
# await streams.submit(InputFrame(pcm_matrix, 'audio', capture_end_ms,
#     sample_rate=16000, sensor=SensorMetadata('device', 'array', sequence,
#     channels=('mic0', 'mic1'), coordinate_frame='array',
#     calibration_id='array-profile-1')))
```

## Meaning and ambiguity

After mean removal and Hann windowing, the provider phase-normalizes cross spectra and searches integer-sample delay peaks within the physical baseline delay. Positive `delay_seconds` is t(mic0) − t(mic1): under a far-field plane-wave assumption, positive source projection points from mic0 toward mic1. `source_projection_on_baseline = delay × sound_speed / spacing`. `cone_angle_degrees` is acos(projection) relative to that baseline. Two microphones leave an entire cone of directions unresolved; it is not unique azimuth, elevation, 3D location, real-world identity or distance. Swapping channels reverses the sign.

Delay resolution is one sample, without interpolation. Near endfire, rounding can put an otherwise valid peak outside the conservative integer geometry bound; the provider rejects it rather than clamps it into a confident direction. Near-field sound, reverberation, multiple speakers, unsynchronized clocks, narrowband signals and channel processing can invalidate the plane-wave interpretation.

Unavailable windows report a reason and null delay/angle/projection. Checks cover energy, inadequate geometry resolution, too little spectral support, a dominant delay outside geometry and weak/ambiguous correlation peaks. These are fixed heuristics (not calibrated source-detection confidence); they can reject useful audio or accept a spurious peak in adverse conditions. `peak_strength` and `peak_ratio` are diagnostic scores, and `confidence_calibrated` stays false. No sound is classified as speech and no speaker identity is inferred. Source separation and SRP/beamforming are not implemented by this class.

## Bounds and verification status

At most19,200 stereo samples are accepted (200ms at96kHz); FFT length is at most65,536. Only one window is processed at a time per SensorStreams lane and no raw samples are emitted into context. Window collection latency is separate from compute/queue/delivery latency.

`python3 -m unittest discover -s tests -p test_direction_provider.py -v` tests actual NumPy execution on generated broadband signals with known signed shifts, both normalized/int16 PCM, unobservable inputs, malformed input/profile and SensorStreams-to-context output. CI explicitly runs this suite with NumPy installed. Synthetic shifted signals establish algorithm/contract behavior, not measured microphone-array direction accuracy. Actual calibrated-array recordings, live capture/reconnect/skew/reverberation/multiple-source measurements, resource/timing evidence and SRP/beamforming integration remain required under #27. No model/device installation is performed by these tests.

## Stateful fixed-hop microphone input

`GCCPHATStream` wraps the concrete window DSP with bounded overlapping assembly. Its default input is one20ms stereo chunk at a time; after four contiguous chunks it analyzes80ms, then produces one updated direction per20ms hop. This is stateful window processing, not a neural streaming model or a claim of zero collection delay. Configure `hop_ms` in5..window_ms and make it divide the window exactly. Sample rate × hop duration must give an integer sample count; unsupported fractional chunk lengths are rejected rather than resampled.

Register `GCCPHATStream.capability(window_ms=80, hop_ms=20)` and a factory for `GCCPHATStream(hop_ms=20, spacing_m=..., sound_speed_m_s=..., calibration_id=..., window_ms=80)`. The remaining channel/coordinate options are the same as `GCCPHATDirection`. The capability ID is `audio.gcc_phat.streaming_direction`; it declares window mode, state and reset-on-gap. Each input timestamp remains the capture **end of that hop**, not callback time.

Each lane retains at most one window's normalized stereo PCM (at most19,200 stereo sample frames). Input is copied; mean removal/windowing is performed only on the assembled analysis window. Sequence gaps, source/sensor/subject/epoch/sample-rate changes, or capture-time discontinuity beyond1ms discard partial audio and refill before producing another estimate. Timestamps are checked against a cumulative sample clock to prevent incremental timestamp drift from joining disjoint audio. Duplicate/backward active-stream sequence/time and rejected input clear partial state and raise. Explicit runtime resets also clear pending samples; no gap is padded with fabricated silence. The runtime's queue/gap handling remains authoritative.

Output adds `hop_ms`, `first_sequence`, `last_sequence` and a lifetime `continuity_resets` counter for discontinuities detected by this wrapper. Runtime-triggered resets are counted separately by SensorStreams; this counter is not a sum of all runtime resets. Before a full contiguous window exists, the provider returns no new observation. Existing context freshness/invalidation rules apply; a previous retained observation is not a new measurement during refill.

`python3 -m unittest discover -s tests -p 'test_direction*.py' -v` also checks exact equivalence to independent full-window DSP, nonmutated input, fixed memory, sequence/clock/identity/epoch gaps, rejected chunks, reset/close and runtime gap handling. This provides an executable20ms-input-to-context path while retaining the same unresolved real-array accuracy and SRP/beamforming requirements in #60/#27.
