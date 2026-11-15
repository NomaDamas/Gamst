# Streaming audio signal features

`AudioProsody` provides local windowed signal measurements without ASR, a neural checkpoint or a VLM call. It explicitly selects a microphone channel rather than silently mixing array input. Its bounded rolling buffer survives contiguous chunks and resets after stream gaps, sample-rate changes or incompatible capture timing.

```python
from gamst import SensorStreams
from gamst.providers.prosody import AudioProsody

streams = SensorStreams(human)
streams.register(AudioProsody.capability(window_ms=80),
                 lambda: AudioProsody(window_ms=80, hop_ms=20))
await streams.open("microphone", "device1", ["audio.prosody"])
```

Submit normalized mono PCM, little-endian mono PCM16 bytes, or a samples-by-channels numeric matrix with explicit `channel=N`. Sample rate must be 8–96 kHz. `timestamp_ms` denotes the end of the submitted audio block. Channel names must match a supplied matrix layout. Preserve the original array stream for direction processing; this provider never changes other lanes' input.

Output `features` includes RMS, dBFS (null for exact silence), AC RMS, autocorrelation pitch candidate in Hz, periodicity score, spectral centroid in Hz, clipping fraction, channel, rate and actual window duration. Periodicity is a signal score, not a calibrated probability or voice detector. A periodic machine tone can satisfy it. dBFS is not calibrated dBSPL; these fields do not establish emotion, identity or physiological state.

The default window is 80 ms and the minimum output hop is 20 ms of newly submitted samples. Each process call emits at most one result for the latest full window, so a large block does not reconstruct all intermediate historical feature windows. Delivery cadence remains controlled by the consuming SDK application. Pitch uses normalized FFT autocorrelation with local peak selection; real speech, harmonic ambiguity, reverberation and noise need separate quality evaluation. This does not claim state-of-the-art speech pitch accuracy.

The provider requires NumPy; it is imported on explicit construction. Metadata discovery performs no download or capture. Tests execute actual numerical processing on known tones including range endpoints, silence, DC, seeded noise, clipping, stereo selection, rate changes, gaps and public context integration. These prove signal processing behavior, not real microphone or emotion-model validation. Related open work: #25 and #6.

Version2 also returns bounded `spectral` and `cycles` objects. Spectral features
use25ms Hann STFT windows with10ms hop, per-frame mean subtraction,
next-power-of-two FFT,40 triangular HTK-spaced mel power bands and13
orthonormal DCT-II coefficients. Returned values are mean mel
power, MFCC mean/std and normalized spectral flux, not the full spectrogram.
Natural log uses a1e-12 floor, with no preemphasis or cepstral liftering; these
explicit conventions are not interchangeable with another model's frontend.
Even at96kHz/250ms the window contains at most23 STFT frames.

Cycle estimates require no clipped samples, autocorrelation periodicity>=.9,
at least5 complete positive-zero-crossing cycles and every period within25% of
the pitch-derived period. Otherwise HNR, jitter and shimmer are null with a
specific abstention reason. Local jitter is mean adjacent period difference
divided by mean period; local shimmer uses cycle peak-to-peak amplitudes in the
same ratio. HNR is10log10(r/(1-r)) using autocorrelation periodicity, capped at
approximately60dB to keep finite values. Extra crossings/complex voiced waveforms
may abstain; these are restricted periodic-signal estimates, not clinical voice
measurements. Real speech/reference validation remains deferred.
