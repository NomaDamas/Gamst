import struct
import asyncio
from types import SimpleNamespace
import unittest

from humansense.integrations import ClockMapper, PCMConverter, InputBridge, LiveKitBridge, PipecatBridge


class ClockTests(unittest.TestCase):
    def test_microsecond_anchor_alignment_and_no_reanchoring_drift(self):
        clock = ClockMapper(source_anchor=2_000_000, wall_anchor_ms=10000)
        self.assertEqual(clock.map(2_500_000), 10500)
        self.assertEqual(clock.map(3_000_000, wall_timestamp_ms=99999), 11000)

    def test_auto_anchor_and_explicit_restart(self):
        clock = ClockMapper(units_per_second=48000)
        self.assertEqual(clock.map(96000, wall_timestamp_ms=5000), 5000)
        self.assertEqual(clock.map(144000), 6000)
        with self.assertRaisesRegex(ValueError, "regressed"):
            clock.map(0)
        self.assertEqual(clock.map(192000), 7000)
        clock.reset(source_anchor=0, wall_anchor_ms=8000)
        self.assertEqual(clock.map(48000), 9000)

    def test_independent_clocks_can_share_wall_anchor(self):
        video = ClockMapper(units_per_second=90000, source_anchor=900000, wall_anchor_ms=10000)
        audio = ClockMapper(units_per_second=48000, source_anchor=0, wall_anchor_ms=10000)
        self.assertEqual(video.map(945000), audio.map(24000))

    def test_invalid_clock_inputs_do_not_mutate_anchor(self):
        clock = ClockMapper()
        for value in [-1, float("nan"), float("inf"), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                clock.map(value)
        self.assertIsNone(clock.source_anchor)
        with self.assertRaises(ValueError):
            clock.reset(source_anchor=0)


class PCMTests(unittest.TestCase):
    def test_stereo_pcm16_downmix(self):
        converter = PCMConverter()
        pcm = struct.pack("<hhhh", 16384, -16384, 16384, 0)
        self.assertEqual(converter.convert(pcm, sample_rate=16000, channels=2), [0., .25])

    def test_rate_conversion_exact_across_block_boundaries(self):
        self.assertEqual(PCMConverter().convert([0., .3, .6, .9], sample_rate=48000), [0., .9])
        converter = PCMConverter()
        output = []
        for block in [[0.], [.3], [.6], [.9]]:
            output += converter.convert(block, sample_rate=48000)
        self.assertEqual(output, [0., .9])
        self.assertEqual(PCMConverter().convert([0., 1.], sample_rate=8000), [0., .5, 1.])

    def test_long_stream_has_no_per_block_rounding_drift(self):
        whole = PCMConverter().convert([.5] * 44100, sample_rate=44100)
        converter = PCMConverter()
        blocked = []
        for _ in range(100):
            blocked += converter.convert([.5] * 441, sample_rate=44100)
        self.assertEqual(len(blocked), 16000)
        self.assertEqual(blocked, whole)

    def test_format_change_requires_reset_and_invalid_input_does_not_change_state(self):
        converter = PCMConverter()
        self.assertEqual(converter.convert([.2], sample_rate=16000), [.2])
        for samples, kwargs in [([float("nan")], {"sample_rate": 16000}),
                                ([1.1], {"sample_rate": 16000}),
                                (b"x", {"sample_rate": 16000}),
                                ([0.], {"sample_rate": 16000, "channels": 2}),
                                ([0.], {"sample_rate": 48000})]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                converter.convert(samples, **kwargs)
        self.assertEqual(converter.convert([.3], sample_rate=16000), [.3])
        converter.reset()
        self.assertEqual(converter.convert([.4], sample_rate=48000), [.4])


class FakeHuman:
    def __init__(self): self.calls = []
    async def push_video(self, frame, **kwargs):
        self.calls.append(("video", frame, kwargs)); return kwargs
    async def push_audio(self, samples, **kwargs):
        self.calls.append(("audio", samples, kwargs)); return kwargs
    async def push_text(self, text, **kwargs):
        self.calls.append(("text", text, kwargs)); return kwargs
    async def reset_processing(self): self.calls.append(("reset",))


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_subject_and_two_clock_domains_are_preserved(self):
        human = FakeHuman()
        bridge = InputBridge(human, subject_id="participant-a",
            video_clock=ClockMapper(source_anchor=1000000, wall_anchor_ms=5000),
            audio_clock=ClockMapper(source_anchor=0, wall_anchor_ms=5000))
        await bridge.push_video("pixels", source_timestamp=1500000)
        await bridge.push_audio([0., 1.], sample_rate=8000, source_timestamp=500000)
        self.assertEqual(human.calls[0][2]["timestamp_ms"], human.calls[1][2]["timestamp_ms"])
        self.assertEqual(human.calls[1][1], [0., .5, 1.])
        for call in human.calls:
            self.assertEqual(call[2]["subject_id"], "participant-a")
        self.assertEqual(human.calls[1][2]["sample_rate"], 16000)

    async def test_source_timestamp_requires_explicit_clock(self):
        bridge = InputBridge(FakeHuman(), subject_id="known-participant")
        with self.assertRaisesRegex(ValueError, "ClockMapper"):
            await bridge.push_video("pixels", source_timestamp=100)
        with self.assertRaisesRegex(ValueError, "not both"):
            await bridge.push_video("pixels", source_timestamp=100, timestamp_ms=100)
        with self.assertRaises(ValueError):
            InputBridge(FakeHuman(), subject_id="")

    async def test_livekit_duck_boundary_uses_injected_decoder_and_explicit_pcm_format(self):
        human = FakeHuman()
        async def decoder(frame): return frame.decoded
        bridge = LiveKitBridge(human, subject_id="track-owner", video_decoder=decoder,
            video_clock=ClockMapper(source_anchor=0, wall_anchor_ms=10000))
        await bridge.on_video(SimpleNamespace(decoded="bgr pixels"), timestamp_us=100000)
        await bridge.on_audio(SimpleNamespace(data=struct.pack("<hh", 16384, 0), sample_rate=16000, num_channels=2), timestamp_ms=10100)
        self.assertEqual(human.calls[0][1], "bgr pixels")
        self.assertEqual(human.calls[0][2]["timestamp_ms"], 10100)
        self.assertEqual(human.calls[1][1], [.25])
        with self.assertRaises(ValueError):
            LiveKitBridge(human, subject_id="track-owner", video_decoder=None)

    async def test_pipecat_decoded_callbacks_and_reset(self):
        human = FakeHuman()
        bridge = PipecatBridge(human, subject_id="owner")
        await bridge.on_transcript("clarify please", timestamp_ms=100)
        await bridge.on_audio([0., .5], sample_rate=8000, timestamp_ms=100)
        await bridge.reset()
        await bridge.on_audio([.2], sample_rate=16000, timestamp_ms=101)
        self.assertEqual(human.calls[0], ("text", "clarify please", {"subject_id": "owner", "timestamp_ms": 100, "role": "user"}))
        self.assertEqual(human.calls[2], ("reset",))

    async def test_reset_discards_frame_from_inflight_native_decoder(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def decoder(frame):
            entered.set()
            await release.wait()
            return frame
        human = FakeHuman()
        bridge = InputBridge(human, subject_id="known", video_decoder=decoder)
        task = asyncio.create_task(bridge.push_video("pixels", timestamp_ms=100))
        await entered.wait()
        await bridge.reset()
        release.set()
        self.assertEqual(await task, [])
        self.assertEqual(human.calls, [("reset",)])

    async def test_decoded_media_reaches_real_sdk_with_grounded_outputs(self):
        from humansense import HumanSense
        from humansense.plugins import AudioLevelPlugin
        async with HumanSense(consent=True) as human:
            human.register(AudioLevelPlugin())
            bridge = PipecatBridge(human, subject_id="participant")
            await bridge.on_audio(struct.pack("<hhhh", 16384, -16384, 16384, 0),
                                  sample_rate=16000, channels=2)
            result = await bridge.on_transcript("다시 설명해 주세요")
            self.assertEqual(result["subject_id"], "participant")
            self.assertEqual(result["modalities"]["audio"], "available")
            self.assertEqual(result["modalities"]["video"], "unavailable")
            self.assertIsNone(result["predictions"]["conversation"]["needs_clarification"]["probability"])
            self.assertEqual(result["recommended_action"]["type"], "ask_clarifying_question")
            self.assertEqual(set(result["predictions"]), {"conversation", "behavior", "consistency"})


if __name__ == "__main__":
    unittest.main()


class RuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_deletion_clears_bridge_audio_and_inflight_decoder(self):
        import asyncio
        from humansense import HumanSense
        entered, release = asyncio.Event(), asyncio.Event()
        async def decoder(frame):
            entered.set(); await release.wait(); return frame
        async with HumanSense(consent=True) as human:
            bridge = InputBridge(human, subject_id='a', video_decoder=decoder)
            await bridge.push_audio([.5] * 4, sample_rate=48000)
            self.assertTrue(bridge.audio_converter._buffer)
            task = asyncio.create_task(bridge.push_video('frame'))
            await entered.wait()
            await human.delete_data()
            self.assertEqual(bridge.audio_converter._buffer, [])
            release.set()
            self.assertEqual(await task, [])
