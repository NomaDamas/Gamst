import unittest
from dataclasses import replace
try:
    import numpy as np
except ImportError:
    np = None
from humansense.providers.direction import GCCPHATDirection, GCCPHATStream
from humansense.plugins import InputFrame
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.schema import timestamp


def provider():
    return GCCPHATStream(spacing_m=.2, sound_speed_m_s=343, calibration_id='fixture')


def chunks(count=8, start=1000):
    base = np.random.default_rng(7).normal(0, .1, 320*count+50)
    data = np.column_stack((base[15:15+320*count], base[20:20+320*count])).astype(np.float32)
    return [InputFrame(data[i*320:(i+1)*320], 'audio', start+(i+1)*20, sample_rate=16000,
        sensor=SensorMetadata('fixture', 'array', i, channels=('mic0', 'mic1'),
                              coordinate_frame='array', calibration_id='fixture')) for i in range(count)]


@unittest.skipIf(np is None, 'numpy not installed')
class StreamingDirection(unittest.TestCase):
    def test_fixed_hops_match_direct_window_without_mutating_input(self):
        p, inputs = provider(), chunks()
        for i, frame in enumerate(inputs):
            copy = frame.data.copy()
            result = p.process(frame)
            np.testing.assert_array_equal(copy, frame.data)
            self.assertLessEqual(len(p.chunks), 4)
            if i < 3:
                self.assertIsNone(result)
                continue
            direct = p.backend.process(replace(frame, data=np.concatenate([f.data for f in inputs[i-3:i+1]])))
            for key, value in direct['direction'].items():
                self.assertEqual(result['direction'][key], value)
            self.assertEqual((result['direction']['first_sequence'], result['direction']['last_sequence']), (i-3, i))
            self.assertEqual(result['direction']['hop_ms'], 20)

    def test_gaps_owner_epoch_and_clock_changes_refill_window(self):
        for change in ('sequence', 'clock', 'epoch', 'source', 'subject'):
            p, inputs = provider(), chunks()
            for f in inputs[:3]:
                self.assertIsNone(p.process(f))
            f = inputs[3]
            if change == 'sequence':
                f = replace(f, sensor=replace(f.sensor, sequence=8))
            elif change == 'clock':
                f = replace(f, timestamp_ms=f.timestamp_ms+100)
            elif change == 'epoch':
                f = replace(f, sensor=replace(f.sensor, stream_epoch='new'))
            elif change == 'source':
                f = replace(f, sensor=replace(f.sensor, source_id='new'))
            else:
                f = replace(f, subject_id='other')
            self.assertIsNone(p.process(f), change)
            self.assertEqual(len(p.chunks), 1)
            self.assertEqual(p.continuity_resets, 1)

    def test_rejected_chunk_and_manual_reset_discard_partial_audio(self):
        p, inputs = provider(), chunks()
        p.process(inputs[0])
        with self.assertRaises(ValueError):
            p.process(replace(inputs[1], data=inputs[1].data[:100]))
        self.assertEqual(len(p.chunks), 0)
        p.process(inputs[1])
        with self.assertRaises(ValueError):
            p.process(inputs[1])
        self.assertEqual(len(p.chunks), 0)
        p.process(inputs[2]); p.reset()
        self.assertEqual(len(p.chunks), 0)
        p.close()
        with self.assertRaises(RuntimeError):
            p.process(inputs[3])
        for hop in (True, 3, 30, 81):
            with self.assertRaises(ValueError):
                GCCPHATStream.capability(hop_ms=hop)


@unittest.skipIf(np is None, 'numpy not installed')
class StreamRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_real_dsp_hops_publish_after_window_and_reset_on_gap(self):
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = GCCPHATStream.capability()
            streams.register(spec, provider)
            await streams.open('fixture', 'array', [spec.id])
            inputs = chunks(start=timestamp()-500)
            for f in inputs[:4]:
                await streams.submit(f)
                await streams.drain()
            observation = next(iter((await human.get_context())['observations'].values()))
            self.assertAlmostEqual(observation['value']['direction']['delay_seconds'], 5/16000)
            await streams.submit(inputs[6])
            await streams.drain()
            status = streams.status()['streams'][0]['providers'][0]
            self.assertEqual(status['resets'], 1)
            self.assertEqual(status['errors'], 0)
