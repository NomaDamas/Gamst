import unittest
from dataclasses import replace
try:
    import numpy as np
except ImportError:
    np = None
from humansense.providers.srp import SRPPHATDirection
from humansense.plugins import InputFrame
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.schema import timestamp

POSITIONS = ((0, 0), (.2, 0), (0, .2), (.2, .2))
CHANNELS = ('a', 'b', 'c', 'd')


def provider(**kwargs):
    return SRPPHATDirection(**dict(dict(positions_m=POSITIONS, channels=CHANNELS,
        calibration_id='fixture'), **kwargs))


def frame(angle=0):
    rate, length = 48000, 3840
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    advance = np.rint(np.array(POSITIONS) @ direction*rate/343).astype(int)
    base = np.random.default_rng(8).uniform(-.5, .5, length+300)
    data = np.column_stack([base[100+a:100+a+length] for a in advance]).astype(np.float32)
    return InputFrame(data, 'audio', timestamp(), sample_rate=rate,
        sensor=SensorMetadata('fixture', 'array', 0, channels=CHANNELS,
            calibration_id='fixture', coordinate_frame='array'))


@unittest.skipIf(np is None, 'numpy not installed')
class SRP(unittest.TestCase):
    def test_broadband_directions_int16_and_input_ownership(self):
        p = provider()
        for angle in (0, 45, 90, 135, 180, 225, 270, 315):
            f = frame(angle); copy = f.data.copy()
            result = p.process(f)['direction']
            self.assertTrue(result['available'], result)
            self.assertLessEqual(abs((result['azimuth_degrees']-angle+180)%360-180), 5)
            np.testing.assert_array_equal(f.data, copy)
            p.spec.validate({'direction': result})
            pcm = replace(f, data=(f.data*32767).astype(np.int16))
            self.assertEqual(p.process(pcm)['direction']['azimuth_degrees'], result['azimuth_degrees'])

    def test_unavailable_silence_narrowband_and_invalid_inputs(self):
        p, f = provider(), frame()
        result = p.process(replace(f, data=np.zeros_like(f.data)))['direction']
        self.assertFalse(result['available']); self.assertIsNone(result['azimuth_degrees'])
        tone = np.sin(np.arange(len(f.data))*2*np.pi*1000/48000)*.3
        result = p.process(replace(f, data=np.tile(tone[:, None], (1,4))))['direction']
        self.assertFalse(result['available'])
        for bad in (replace(f, data=f.data[:-1]), replace(f, timestamp_ms=0),
                    replace(f, data=np.full_like(f.data, np.nan)),
                    replace(f, sensor=replace(f.sensor, channels=tuple(reversed(CHANNELS))))):
            with self.assertRaises(ValueError): p.process(bad)
        before = p.process(f); p.reset(); self.assertEqual(p.process(f), before)
        p.close()
        with self.assertRaises(RuntimeError): p.process(f)

    def test_degenerate_geometry_and_bad_configuration(self):
        for kwargs in ({'positions_m':((0,0),(.1,0),(.2,0),(.3,0))},
            {'positions_m':((0,0),(0,0),(.2,0),(0,.2))}, {'grid_degrees':3},
            {'sound_speed_m_s':float('nan')}, {'window_ms':True}):
            with self.assertRaises(ValueError): provider(**kwargs)


@unittest.skipIf(np is None, 'numpy not installed')
class SRPRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_array_estimate_reaches_context(self):
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = SRPPHATDirection.capability()
            streams.register(spec, provider)
            await streams.open('fixture', 'array', [spec.id])
            await streams.submit(frame(90)); await streams.drain()
            observation = next(iter((await human.get_context())['observations'].values()))
            self.assertEqual(observation['value']['direction']['azimuth_degrees'], 90)
