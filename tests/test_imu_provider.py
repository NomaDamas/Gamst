import math
import struct
import unittest
from dataclasses import replace
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.imu import BNO055IMU
from humansense.schema import timestamp


def provider(**kwargs):
    return BNO055IMU(**dict(dict(unit_sel=0, operation_mode=8, acceleration_range_g=4,
        gyroscope_range_dps=2000, calibration_id='fixture'), **kwargs))


def frame(words=(100, -200, 981, 30000, -30000, 1234, 160, -320, 0)):
    return InputFrame(struct.pack('<9h', *words), 'imu', timestamp(),
        sensor=SensorMetadata('fixture', 'imu', 0, calibration_id='fixture', coordinate_frame='imu'))


class IMU(unittest.TestCase):
    def test_register_offsets_signed_units_and_ignored_magnetometer(self):
        motion = provider().process(frame())['motion']
        self.assertEqual(motion['acceleration_m_s2'], [1, -2, 9.81])
        self.assertAlmostEqual(motion['angular_velocity_rad_s'][0], math.radians(10))
        self.assertAlmostEqual(motion['angular_velocity_rad_s'][1], math.radians(-20))
        motion = provider(unit_sel=3).process(frame((1000, -1000, 0, 0, 0, 0, 900, -900, 0)))['motion']
        self.assertEqual(motion['acceleration_m_s2'], [9.80665, -9.80665, 0])
        self.assertEqual(motion['angular_velocity_rad_s'], [1, -1, 0])
        self.assertFalse(motion['gravity_removed'])
        self.assertFalse(motion['physical_calibration_verified'])

    def test_supported_fusion_modes_and_effective_range_thresholds(self):
        for mode in (8, 11, 12):
            for units, acc, gyr in ((0, 3923, 32000), (3, 4000, 31416)):
                motion = provider(operation_mode=mode, unit_sel=units).process(
                    frame((acc, 0, 0, 0, 0, 0, gyr, 0, 0)))['motion']
                self.assertEqual(motion['input_axes_near_acceleration_range'], [True, False, False])
                self.assertEqual(motion['input_axes_near_gyroscope_range'], [True, False, False])
                self.assertFalse(motion['wire_endpoint_detected'])

    def test_bias_before_rotation_and_range_flags_before_correction(self):
        p = provider(output_frame='body', acceleration_bias=(1, 0, 0),
            rotation_to_output=((0, -1, 0), (1, 0, 0), (0, 0, 1)))
        motion = p.process(frame())['motion']
        self.assertEqual(motion['acceleration_m_s2'], [2, 0, 9.81])
        self.assertEqual(motion['coordinate_frame'], 'body')
        motion = provider(acceleration_bias=(100, 0, 0)).process(
            frame((4000, 0, 0, 0, 0, 0, 32767, 0, 0)))['motion']
        self.assertTrue(motion['possible_saturation'])
        self.assertTrue(motion['wire_endpoint_detected'])
        self.assertEqual(motion['input_axes_near_acceleration_range'], [True, False, False])

    def test_reject_bad_config_frame_and_lifecycle(self):
        for kwargs in ({'unit_sel': True}, {'operation_mode': 0}, {'operation_mode': 5}, {'operation_mode': 7}, {'acceleration_range_g': 3}, {'acceleration_range_g': 2}, {'acceleration_range_g': 8}, {'acceleration_range_g': 16},
            {'gyroscope_range_dps': 300}, {'gyroscope_range_dps': 125}, {'gyroscope_range_dps': 250}, {'gyroscope_range_dps': 500}, {'gyroscope_range_dps': 1000}, {'acceleration_bias': (math.nan, 0, 0)},
            {'rotation_to_output': ((-1, 0, 0), (0, 1, 0), (0, 0, 1)), 'output_frame': 'body'},
            {'rotation_to_output': ((1, 0, 0), (1, 0, 0), (0, 0, 1))}):
            with self.assertRaises(ValueError):
                provider(**kwargs)
        p, f = provider(), frame()
        for bad in (replace(f, data=b''), replace(f, modality='video'), replace(f, timestamp_ms=math.inf),
                    replace(f, sensor=SensorMetadata('a', 'b', 0))):
            with self.assertRaises(ValueError):
                p.process(bad)
        before = p.process(f)
        p.reset()
        self.assertEqual(p.process(f), before)
        p.close()
        with self.assertRaises(RuntimeError):
            p.process(f)


class IMURuntime(unittest.IsolatedAsyncioTestCase):
    async def test_decoded_si_measurements_reach_context(self):
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = BNO055IMU.capability()
            streams.register(spec, provider)
            await streams.open('fixture', 'imu', [spec.id])
            await streams.submit(frame())
            await streams.drain()
            observation = next(iter((await human.get_context())['observations'].values()))
            self.assertEqual(observation['value']['motion']['acceleration_m_s2'], [1, -2, 9.81])
            self.assertEqual(observation['sensor']['calibration_id'], 'fixture')
