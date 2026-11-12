# External media bridges

HumanSense accepts existing media streams without opening another camera. The decoded-media bridges bind inputs to a known participant and normalize audio/timestamps. They do not automatically identify people or associate a camera face with a microphone speaker.

```python
from humansense.integrations import InputBridge, ClockMapper
bridge = InputBridge(human, subject_id="verified-participant",
    audio_clock=ClockMapper(units_per_second=48000,
        source_anchor=0, wall_anchor_ms=session_start_unix_ms))
await bridge.push_audio(stereo_pcm16_bytes, sample_rate=48000, channels=2,
                        source_timestamp=sample_counter)
await bridge.push_text("The participant's actual transcript")
await bridge.push_video(decoded_bgr_frame, timestamp_ms=capture_unix_ms)
```

Audio conversion supports interleaved PCM16 or normalized floats, downmixes channels and maintains fractional resampling state across chunks. The simple linear interpolator is not a high-quality anti-aliasing resampler; use your own converter for spectral fidelity. Reset the bridge and clock explicitly on track restart or audio format change. Bridges register a buffer-cleanup hook with HumanSense so pause/delete/close clears held interpolation data and invalidates in-flight decoders. Standalone mock/custom runtimes without plugin lifecycle support must call bridge.reset() themselves.

Epoch timestamps and source timestamps are mutually exclusive. Independent devices need independent `ClockMapper`s with anchors referring to the same known instant. Receipt time can be used when capture timestamps are unknown, but that does not establish sensor synchronization. Clock regression is rejected instead of silently relabeling old media as current.

## LiveKit boundary

```python
from humansense.integrations import LiveKitBridge, ClockMapper
bridge = LiveKitBridge(human, subject_id=participant_id,
    video_decoder=your_native_frame_to_bgr,
    video_clock=ClockMapper(units_per_second=1_000_000,
        source_anchor=source_start_us, wall_anchor_ms=session_start_unix_ms))
# In your existing video stream loop:
await bridge.on_video(video_event.frame, timestamp_us=video_event.timestamp_us)
# PCM16 AudioFrame boundary: data, sample_rate, num_channels attributes.
await bridge.on_audio(audio_frame, timestamp_ms=known_capture_unix_ms)
```

The video decoder is deliberately injected: verify your installed LiveKit SDK's native pixel conversion API. The adapter is duck typed and contains no copied LiveKit source. Actual native version compatibility has not been tested here.

## Pipecat boundary

`PipecatBridge` exposes `on_video(decoded_pixels, ...)`, `on_audio(pcm, sample_rate=..., channels=...)` and `on_transcript(text, ...)` to invoke from your application's processor. It is not an unverified subclass of Pipecat's evolving processor API. Your processor must still forward its original frames. Decoded-media/runtime tests pass; upstream SDK/runtime validation remains required.
