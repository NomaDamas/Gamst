import unittest
try:
    import numpy as np
except ImportError:
    np = None
from humansense.providers.direction import GCCPHATDirection
from humansense.plugins import InputFrame
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.schema import timestamp


def provider(**kwargs):
    return GCCPHATDirection(**dict(dict(spacing_m=.2, sound_speed_m_s=343,
        calibration_id='fixture'), **kwargs))


def frame(data, **kwargs):
    return InputFrame(data, 'audio', timestamp(), sample_rate=16000,
        sensor=SensorMetadata('fixture', 'array', 0, channels=('mic0', 'mic1'),
                              coordinate_frame='array', calibration_id='fixture'), **kwargs)


def shifted(delay):
    rng = np.random.default_rng(42)
    base = rng.normal(0, .1, 1500)
    # mic0 receives the signal later for positive delay.
    return np.column_stack((base[100-delay:1380-delay], base[100:1380])).astype(np.float32)


@unittest.skipIf(np is None, 'numpy not installed')
class Direction(unittest.TestCase):
    def test_signed_delays_and_far_field_cone(self):
        p = provider()
        for lag in (-5, 0, 5):
            result = p.process(frame(shifted(lag)))['direction']
            self.assertTrue(result['available'], result)
            self.assertAlmostEqual(result['delay_seconds'], lag/16000)
            self.assertAlmostEqual(result['source_projection_on_baseline'], lag/16000*343/.2)
            self.assertFalse(result['unique_3d_direction'])
            self.assertFalse(result['confidence_calibrated'])
        quantized = (shifted(5)*32768).astype(np.int16)
        self.assertAlmostEqual(p.process(frame(quantized))['direction']['delay_seconds'], 5/16000)
        p.reset()
        p.close()
        with self.assertRaises(RuntimeError):
            p.process(frame(shifted(0)))

    def test_unobservable_inputs_do_not_invent_direction(self):
        p = provider()
        cases = [np.zeros((1280, 2)), np.ones((1280, 2))*.5,
                 np.repeat(np.sin(np.arange(1280)*2*np.pi*1000/16000)[:, None], 2, axis=1), shifted(30)]
        for data in cases:
            result = p.process(frame(data))['direction']
            self.assertFalse(result['available'], result)
            self.assertIsNone(result['cone_angle_degrees'])
        self.assertEqual(provider(spacing_m=.02).process(frame(shifted(0)))['direction']['reason'],
                         'insufficient_delay_resolution')

    def test_reject_mono_bad_profile_nonfinite_and_wrong_windows(self):
        from dataclasses import replace
        p = provider()
        f = frame(shifted(0))
        for bad in (replace(f, data=f.data[:, 0]), replace(f, data=f.data[:500]),
                    replace(f, sensor=SensorMetadata('x', 'y', 0)), replace(f, timestamp_ms=1),
                    replace(f, data=np.full((1280, 2), np.nan)), replace(f, data=f.data*100)):
            with self.assertRaises(ValueError):
                p.process(bad)
        for kwargs in ({'spacing_m': 0}, {'sound_speed_m_s': float('inf')},
                       {'channels': ('x', 'x')}, {'window_ms': True}):
            with self.assertRaises(ValueError):
                provider(**kwargs)


@unittest.skipIf(np is None, 'numpy not installed')
class DirectionRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_window_direction_reaches_context(self):
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = GCCPHATDirection.capability()
            streams.register(spec, provider)
            await streams.open('fixture', 'array', [spec.id])
            await streams.submit(frame(shifted(5)))
            await streams.drain()
            observation = next(iter((await human.get_context())['observations'].values()))
            self.assertAlmostEqual(observation['value']['direction']['delay_seconds'], 5/16000)
            self.assertEqual(observation['sensor']['calibration_id'], 'fixture')
