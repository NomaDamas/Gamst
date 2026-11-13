# Streaming source selection

`DelaySumBeamformer` is a concrete causal, fixed-steering two-channel delay-and-sum processor. NumPy performs the DSP; no checkpoint, microphone activation or implicit download is involved. The method aligns earlier-arriving audio to the later channel and averages it. It is a basic spatial filter, not speaker identification, echo cancellation, blind source separation or a quality guarantee. Background audio remains available in the untouched input frame.

Delay-and-sum is a standard array processing baseline; see the [LCAV pyroomacoustics implementation and methods](https://github.com/LCAV/pyroomacoustics). This implementation uses integer time-domain delays with no FFT wraparound or fractional interpolation. It does not wrap or install pyroomacoustics.

## Input and causal behavior

Declare spacing (0.02–1 m), sound speed (300–380 m/s), calibration ID, ordered channel names and coordinate frame, as for [GCC-PHAT direction](audio-direction.md). Supply synchronized normalized float32/float64 or int16 PCM shaped samples × 2, at 16–96 kHz, in exact 5–100 ms hops with integer sample counts. Capture timestamps identify hop ends. Calibration, channel order and frame must match the profile. Device synchronization/gain matching and physical calibration remain acquisition responsibilities.

`steering_delay_seconds` uses t(mic0) − t(mic1), matching GCC-PHAT. A positive value delays mic1; a negative value delays mic0. The requested delay must fit spacing/sound-speed geometry. Quantization rounds to the nearest sample (ties to even), and the context includes requested/applied delay and error. Maximum quantization error is half a sample; high-frequency and reverberant performance may be poor. Steering is fixed for the instance. Recreate the instance and reset downstream state when changing selection; do not silently steer from an unavailable direction estimate.

History is bounded by the absolute integer delay, at most 320 stereo samples at the supported extremes. The first nonzero-delay hop after initialization/reset is withheld to fill history; no artificial zero padding enters downstream audio. Zero steering needs no warmup. Sequence/epoch/source/subject/rate changes and clock gaps reset history. Duplicate/backward sequence or timestamps reject and reset; malformed input also resets. Capture-clock drift beyond 1 ms from the accumulated expected clock is a gap. Call sequentially once per input hop and give each lane its own instance.

The returned mono frame keeps the acquisition interval, sequence and epoch, but has an explicit distinct sensor ID and channel `beam`. These timestamps describe the output clock, not a delay-compensated acoustic emission time. `channel_delays_samples` provides the per-channel delay; `alignment_latency_ms` reports the maximum causal delay, separate from hop acquisition, warmup, processing and downstream inference latency. There is zero future lookahead. No dBSPL or physical calibration claim is made.

## PCM stays in the processing lane

```python
from humansense.providers.beamforming import DelaySumBeamformer

beam = DelaySumBeamformer(
    spacing_m=0.2, sound_speed_m_s=343, calibration_id='array-profile',
    steering_delay_seconds=0.00025, output_sensor_id='selected-speech')
human.register_capability(beam.spec)

async def on_array_hop(frame):
    result = beam.transform(frame)
    if result is None:
        return
    await human.push_observation(beam.spec.id, result.context,
        sensor=frame.sensor, subject_id=frame.subject_id,
        timestamp_ms=frame.timestamp_ms)
    # Pass result.frame to your explicitly configured VAD/ASR or mono DSP lane.
    # Preserve frame separately for ambient sound recognition.
```

`transform` returns `BeamformedBlock(frame, context)`. Never put the frame/PCM in context. Context contains steering provenance, delay, RMS/peak, input clipping flag and validity limitations. The output is a convex average with no adaptive gain; input clipping can still destroy information. `process` implements the SensorStreams provider contract and publishes only these metrics, discarding derived PCM. Use either transform mode or provider mode for a given hop, not both on one instance. No callback/network execution or persistent PCM archive occurs.

## Validation and remaining scope

`python3 -m unittest discover -s tests -p test_beamforming.py -v` runs actual NumPy DSP: known positive/negative/zero-delay broadband signals, signed PCM16 boundary impulses without circular wrap, quantization, malformed/gap/reset behavior, a real AudioProsody downstream path and context publication. These are constructed signals, not real microphone recordings. Noise reduction, intelligibility, spatial selectivity and physical calibration are unverified; corresponding output flags stay false.

Required real-array recordings, measured direction/selection performance, timing/resources and SRP coverage remain tracked in #27 and hardware child #60. This bridge adds executable source-selection processing; it does not complete that family or the full-duplex AEC requirement in #28. No model weights were installed.
