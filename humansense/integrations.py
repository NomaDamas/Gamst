"""Decoded-media bridges, without importing a transport framework.

Adapters deliberately do not open devices, identify speakers, or download code.
Native framework objects are duck typed at an explicit, small boundary: callers
must supply a video decoder and confirm source timestamps' units and meaning.
"""
from __future__ import annotations

import inspect
import math
import struct
import time


def _wall_timestamp(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("timestamp_ms must be a nonnegative integer")
    return value


class ClockMapper:
    """Map one monotonic media clock to epoch milliseconds using an anchor.

    Use independent mappers for independent audio/video clocks. For synchronized
    streams, supply anchors referring to the same wall-clock instant. Transport
    receipt time alone does not establish A/V synchronization. Regressing PTS
    raises before changing state; call reset() explicitly when a source restarts.
    """

    def __init__(self, units_per_second=1_000_000, *, source_anchor=None,
                 wall_anchor_ms=None):
        if isinstance(units_per_second, bool) or not math.isfinite(units_per_second) or units_per_second <= 0:
            raise ValueError("units_per_second must be finite and positive")
        self.units_per_second = units_per_second
        self.reset(source_anchor=source_anchor, wall_anchor_ms=wall_anchor_ms)

    @staticmethod
    def _source(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("source timestamp must be finite and nonnegative")
        return value

    def reset(self, *, source_anchor=None, wall_anchor_ms=None):
        if source_anchor is not None:
            source_anchor = self._source(source_anchor)
            if wall_anchor_ms is None:
                raise ValueError("An explicit source anchor requires wall_anchor_ms")
        wall = None if wall_anchor_ms is None else _wall_timestamp(wall_anchor_ms)
        self.source_anchor = source_anchor
        self.wall_anchor_ms = wall
        self.last_source_timestamp = source_anchor

    def map(self, source_timestamp, *, wall_timestamp_ms=None):
        source = self._source(source_timestamp)
        if self.last_source_timestamp is not None and source < self.last_source_timestamp:
            raise ValueError("Source clock regressed; reset the mapper for a restarted stream")
        if wall_timestamp_ms is not None:
            wall_timestamp_ms = _wall_timestamp(wall_timestamp_ms)
        anchor = self.source_anchor if self.source_anchor is not None else source
        wall = self.wall_anchor_ms
        if wall is None:
            wall = wall_timestamp_ms if wall_timestamp_ms is not None else time.time_ns() // 1_000_000
        mapped = wall + round((source - anchor) * 1000 / self.units_per_second)
        if mapped < 0:
            raise ValueError("Mapped timestamp precedes the Unix epoch")
        self.source_anchor, self.wall_anchor_ms = anchor, wall
        self.last_source_timestamp = source
        return mapped


class PCMConverter:
    """Streaming mono downmix and linear sample-rate conversion to 16kHz.

    Input bytes are signed little-endian PCM16, or normalized finite floats.
    Channels are interleaved and averaged. Linear interpolation preserves clock
    continuity across blocks but is not an anti-aliasing mastering resampler;
    applications needing spectral fidelity should inject their own converter.
    """

    def __init__(self, output_rate=16000):
        if isinstance(output_rate, bool) or not isinstance(output_rate, int) or output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        self.output_rate = output_rate
        self.reset()

    def reset(self):
        self._format = None
        self._buffer = []
        self._position = 0

    def convert(self, samples, *, sample_rate, channels=1):
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            raise ValueError("sample_rate must be an integer from 8000 to 192000")
        if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 32:
            raise ValueError("channels must be an integer from 1 to 32")
        if self._format is not None and self._format != (sample_rate, channels):
            raise ValueError("Audio format changed; reset converter explicitly")
        if isinstance(samples, (bytes, bytearray, memoryview)):
            raw = bytes(samples)
            if len(raw) % 2:
                raise ValueError("PCM16 byte count must be even")
            values = [item[0] / 32768 for item in struct.iter_unpack("<h", raw)]
        else:
            if hasattr(samples, "reshape"):
                samples = samples.reshape(-1)
            try:
                values = [float(value) for value in samples]
            except (TypeError, ValueError) as exc:
                raise ValueError("Audio samples must be interleaved normalized floats") from exc
        if len(values) > sample_rate * channels * 5:
            raise ValueError("At most five seconds of source audio per block")
        if len(values) % channels:
            raise ValueError("Interleaved sample count must be divisible by channels")
        if any(not math.isfinite(value) or abs(value) > 1 for value in values):
            raise ValueError("Audio samples must be finite and normalized to [-1, 1]")
        if not values:
            return []
        mono = [sum(values[index:index + channels]) / channels for index in range(0, len(values), channels)]
        self._format = (sample_rate, channels)
        self._buffer.extend(mono)
        output = []
        while True:
            index, remainder = divmod(self._position, self.output_rate)
            if index >= len(self._buffer) or (remainder and index + 1 >= len(self._buffer)):
                break
            value = self._buffer[index]
            if remainder:
                fraction = remainder / self.output_rate
                value += (self._buffer[index + 1] - value) * fraction
            output.append(value)
            self._position += sample_rate
        consumed = min(self._position // self.output_rate, len(self._buffer) - 1)
        if consumed:
            del self._buffer[:consumed]
            self._position -= consumed * self.output_rate
        return output


class _BridgeLifecycle:
    name = "input_bridge_lifecycle"
    modalities = ()

    def __init__(self, bridge):
        self.bridge = bridge

    async def process(self, frame):
        return []

    def reset(self):
        self.bridge._clear_state()

    def close(self):
        self.reset()


class InputBridge:
    """Bind explicitly owned decoded inputs to a shared HumanSense runtime.

    ``subject_id`` is supplied by the application, never inferred from a face or
    a voice. For group calls, create one bridge per known participant/track.
    Timestamp inputs are mutually exclusive: epoch ``timestamp_ms``, or media
    ``source_timestamp`` with a dedicated ClockMapper. Omitting both uses receipt
    time and does not claim sensor-level synchronization.
    """

    def __init__(self, human, *, subject_id, video_decoder=None,
                 video_clock=None, audio_clock=None, audio_converter=None):
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("An explicit non-empty subject_id is required")
        self.human, self.subject_id = human, subject_id
        self.video_decoder = video_decoder
        self.video_clock, self.audio_clock = video_clock, audio_clock
        self.audio_converter = audio_converter or PCMConverter()
        self._generation = 0
        register = getattr(human, "register", None)
        if register is not None:
            register(_BridgeLifecycle(self))

    def _clear_state(self):
        self._generation += 1
        self.audio_converter.reset()

    @staticmethod
    def _timestamp(timestamp_ms, source_timestamp, clock):
        if timestamp_ms is not None and source_timestamp is not None:
            raise ValueError("Use epoch timestamp_ms or source_timestamp, not both")
        if source_timestamp is not None:
            if clock is None:
                raise ValueError("Source timestamps require an explicitly configured ClockMapper")
            return clock.map(source_timestamp)
        return time.time_ns() // 1_000_000 if timestamp_ms is None else _wall_timestamp(timestamp_ms)

    async def push_video(self, frame, *, timestamp_ms=None, source_timestamp=None):
        generation = self._generation
        timestamp = self._timestamp(timestamp_ms, source_timestamp, self.video_clock)
        if self.video_decoder is not None:
            frame = self.video_decoder(frame)
            if inspect.isawaitable(frame):
                frame = await frame
        if generation != self._generation:
            return []
        return await self.human.push_video(frame, subject_id=self.subject_id, timestamp_ms=timestamp)

    async def push_audio(self, samples, *, sample_rate, channels=1,
                         timestamp_ms=None, source_timestamp=None):
        timestamp = self._timestamp(timestamp_ms, source_timestamp, self.audio_clock)
        converted = self.audio_converter.convert(samples, sample_rate=sample_rate, channels=channels)
        if not converted:
            return []
        return await self.human.push_audio(converted, sample_rate=self.audio_converter.output_rate,
                                          subject_id=self.subject_id, timestamp_ms=timestamp)

    async def push_text(self, text, *, timestamp_ms=None, role="user"):
        return await self.human.push_text(text, subject_id=self.subject_id,
                                         timestamp_ms=self._timestamp(timestamp_ms, None, None), role=role)

    async def reset(self):
        """Discard audio interpolation state; caller resets source clocks explicitly."""
        self._clear_state()
        reset = getattr(self.human, "reset_processing", None)
        if reset is not None:
            result = reset()
            if inspect.isawaitable(result):
                await result


class LiveKitBridge(InputBridge):
    """Duck-typed LiveKit frame boundary; no version compatibility claim.

    Call ``on_video(event.frame, timestamp_us=event.timestamp_us)`` with an
    explicitly injected decoder producing BGR pixels. ``on_audio(frame)`` reads
    PCM16 frame.data, frame.sample_rate, frame.num_channels. A caller-provided
    audio timestamp is required to claim capture-clock alignment; receipt time
    is used otherwise. Install and verify the chosen livekit SDK separately.
    """

    def __init__(self, human, *, subject_id, video_decoder, **kwargs):
        if not callable(video_decoder):
            raise ValueError("LiveKit video_decoder must explicitly convert native frames to BGR")
        super().__init__(human, subject_id=subject_id, video_decoder=video_decoder, **kwargs)

    async def on_video(self, frame, *, timestamp_us=None, timestamp_ms=None):
        return await self.push_video(frame, source_timestamp=timestamp_us, timestamp_ms=timestamp_ms)

    async def on_audio(self, frame, *, timestamp_us=None, timestamp_ms=None):
        return await self.push_audio(frame.data, sample_rate=frame.sample_rate,
            channels=frame.num_channels, source_timestamp=timestamp_us, timestamp_ms=timestamp_ms)


class PipecatBridge(InputBridge):
    """Decoded payload callbacks for use inside an application's processor.

    This is not a FrameProcessor subclass and does not consume undocumented
    Pipecat frame classes. Pass decoded PCM/video/transcript values explicitly.
    The enclosing processor retains responsibility for frame forwarding.
    """

    async def on_video(self, pixels, **timing):
        return await self.push_video(pixels, **timing)

    async def on_audio(self, pcm, *, sample_rate, channels=1, **timing):
        return await self.push_audio(pcm, sample_rate=sample_rate, channels=channels, **timing)

    async def on_transcript(self, text, **kwargs):
        return await self.push_text(text, **kwargs)
