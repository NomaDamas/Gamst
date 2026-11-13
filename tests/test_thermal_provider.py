import json
import struct
import unittest

from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.thermal import TLinearThermal
from humansense.schema import timestamp


def provider(**kwargs):
    return TLinearThermal(**dict(dict(width=2, height=2, kelvin_per_count=.01,
        byteorder='little', calibration_id='fixture', emissivity=.95), **kwargs))


def frame(values=(30000, 30100, 0, 65535), byteorder='<', **kwargs):
    return InputFrame(struct.pack(byteorder+'4H', *values), 'thermal', timestamp(),
        sensor=SensorMetadata('test', 'thermal', 0, calibration_id='fixture'), **kwargs)


class Thermal(unittest.TestCase):
    def test_known_kelvin_conversion_and_missing_pixels(self):
        result = provider().process(frame())['temperature']
        roi = result['regions'][0]
        self.assertAlmostEqual(roi['min_celsius'], 26.85)
        self.assertAlmostEqual(roi['max_celsius'], 27.85)
        self.assertAlmostEqual(roi['mean_celsius'], 27.35)
        self.assertEqual((roi['valid_pixels'], roi['invalid_pixels']), (2, 2))
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertNotIn('30000', json.dumps(result))
        empty = provider().process(frame((0, 65535, 0, 0)))['temperature']['regions'][0]
        self.assertFalse(empty['available'])
        self.assertIsNone(empty['mean_celsius'])

    def test_endianness_resolution_roi_and_copy(self):
        rois = {'one': [1, 0, 2, 1]}
        p = provider(byteorder='big', kelvin_per_count=.1, rois=rois)
        rois['one'][0] = 0
        result = p.process(frame((3000, 3100, 3200, 3300), '>'))['temperature']['regions'][0]
        self.assertEqual(result['valid_pixels'], 1)
        self.assertAlmostEqual(result['mean_celsius'], 36.85)
        p.reset()
        self.assertEqual(p.process(frame((3000, 3100, 3200, 3300), '>'))['temperature']['regions'][0], result)
        p.close()
        with self.assertRaises(RuntimeError):
            p.process(frame())

    def test_invalid_config_and_payloads(self):
        for args in ({'width': True}, {'height': 513}, {'kelvin_per_count': 1},
                     {'byteorder': 'native'}, {'emissivity': float('nan')}, {'emissivity': 0},
                     {'calibration_id': ''}, {'rois': {}}, {'rois': {'x': (0, 0, 3, 2)}}):
            with self.assertRaises(ValueError):
                provider(**args)
        p = provider()
        for bad in (InputFrame(b'123', 'thermal', 0), InputFrame(b'12345678', 'video', 0),
                    InputFrame(b'12345678', 'thermal', 0, sensor=SensorMetadata('a', 'b', 0))):
            with self.assertRaises(ValueError):
                p.process(bad)
        bad = frame()
        from dataclasses import replace
        for value in (b'', b'0'*10, [30000]*4):
            with self.assertRaises(ValueError):
                p.process(replace(bad, data=value))


class ThermalRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_thermal_reaches_context_without_pixels(self):
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = TLinearThermal.capability()
            streams.register(spec, provider)
            await streams.open('test', 'thermal', [spec.id])
            await streams.submit(frame())
            await streams.drain()
            context = await human.get_context()
            observation = next(iter(context['observations'].values()))
            self.assertEqual(observation['sensor']['calibration_id'], 'fixture')
            self.assertAlmostEqual(observation['value']['temperature']['regions'][0]['mean_celsius'], 27.35)
            self.assertNotIn('30000', json.dumps(observation['value']))
            self.assertEqual(streams.status()['streams'][0]['providers'][0]['errors'], 0)
