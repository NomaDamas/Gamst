# Streaming audio signal features

`AudioProsody` provides local windowed signal measurements without ASR, a neural checkpoint or a VLM call. It explicitly selects a microphone channel rather than silently mixing array input. Its bounded rolling buffer survives contiguous chunks and resets after stream gaps, sample-rate changes or incompatible capture timing.

```python
from humansense import SensorStreams
from humansense.providers.prosody import AudioProsody

streams = SensorStreams(human)
streams.register(AudioProsody.capability(window_ms=80),
                 lambda: AudioProsody(window_ms=80, hop_ms=20))
await streams.open("microphone", "device1", ["audio.prosody"])
```

Submit normalized mono PCM, little-endian mono PCM16 bytes, or a samples-by-channels numeric matrix with explicit `channel=N`. Sample rate must be 8–96 kHz. `timestamp_ms` denotes the end of the submitted audio block. Channel names must match a supplied matrix layout. Preserve the original array stream for direction processing; this provider never changes other lanes' input.

Output `features` includes RMS, dBFS (null for exact silence), AC RMS, autocorrelation pitch candidate in Hz, periodicity score, spectral centroid in Hz, clipping fraction, channel, rate and actual window duration. Periodicity is a signal score, not a calibrated probability or voice detector. A periodic machine tone can satisfy it. dBFS is not calibrated dBSPL; these fields do not establish emotion, identity or physiological state.

The default window is 80 ms and the minimum output hop is 20 ms of newly submitted samples. Each process call emits at most one result for the latest full window, so a large block does not reconstruct all intermediate historical feature windows. Delivery cadence remains controlled by the consuming SDK application. Pitch uses normalized FFT autocorrelation with local peak selection; real speech, harmonic ambiguity, reverberation and noise need separate quality evaluation. This does not claim state-of-the-art speech pitch accuracy.

The provider requires NumPy; it is imported on explicit construction. Metadata discovery performs no download or capture. Tests execute actual numerical processing on known tones including range endpoints, silence, DC, seeded noise, clipping, stereo selection, rate changes, gaps and public context integration. These prove signal processing behavior, not real microphone or emotion-model validation. Related open work: #25 and #6.
