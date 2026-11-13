import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from humansense.providers.motion import OpenCVFlow
from humansense.plugins import InputFrame
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.schema import timestamp


@unittest.skipIf(cv2 is None, 'optional OpenCV/numpy not installed')
class Motion(unittest.TestCase):
    def image(self):
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, (160, 240), dtype=np.uint8)
        return cv2.GaussianBlur(image, (5, 5), 0)

    def test_known_translation_with_both_actual_algorithms(self):
        image = self.image()
        moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (240, 160), borderMode=cv2.BORDER_REFLECT)
        for method in ('lucas_kanade', 'farneback'):
            provider = OpenCVFlow(method=method)
            self.assertIsNone(provider.process(InputFrame(image, 'video', 1000)))
            result = provider.process(InputFrame(moved, 'video', 1100))['motion']
            self.assertAlmostEqual(result['dx_pixels'], 3, delta=.5)
            self.assertAlmostEqual(result['dy_pixels'], 2, delta=.5)
            self.assertAlmostEqual(result['median_speed_pixels_per_second'], 36.06, delta=6)
            self.assertFalse(result['camera_motion_separated'])
            provider.close()

    def test_no_texture_is_insufficient_not_false_zero_motion(self):
        for method in ('lucas_kanade', 'farneback'):
            provider = OpenCVFlow(method=method)
            blank = np.zeros((64, 64), dtype=np.uint8)
            provider.process(InputFrame(blank, 'video', 1000))
            self.assertIsNone(provider.process(InputFrame(blank, 'video', 1100)))

    def test_gap_shape_change_reset_and_invalid_time(self):
        provider = OpenCVFlow()
        image = self.image()
        provider.process(InputFrame(image, 'video', 1000))
        with self.assertRaises(ValueError):
            provider.process(InputFrame(image, 'video', 1000))
        self.assertIsNone(provider.process(InputFrame(image, 'video', 3000)))
        self.assertIsNone(provider.process(InputFrame(image[:100], 'video', 3100)))
        provider.reset()
        self.assertIsNone(provider.process(InputFrame(image, 'video', 3200)))
        provider.close()
        with self.assertRaises(RuntimeError):
            provider.process(InputFrame(image, 'video', 3300))

    def test_invalid_input_and_original_pixel_scaling(self):
        provider = OpenCVFlow(max_side=120)
        with self.assertRaises(ValueError):
            provider.process(InputFrame(np.zeros((30, 30), dtype=np.float32), 'video', 0))
        image = self.image()
        moved = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 0]]), (240, 160), borderMode=cv2.BORDER_REFLECT)
        provider.process(InputFrame(image, 'video', 0))
        result = provider.process(InputFrame(moved, 'video', 100))['motion']
        self.assertAlmostEqual(result['dx_pixels'], 4, delta=.8)


class MotionMetadata(unittest.TestCase):
    def test_declaration_does_not_load_backend(self):
        spec = OpenCVFlow.capability()
        self.assertTrue(spec.stateful)
        self.assertEqual(spec.streaming_mode, 'native')
        with self.assertRaises(ValueError):
            OpenCVFlow.capability('invented')


@unittest.skipIf(cv2 is None, 'optional OpenCV/numpy not installed')
class MotionRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_real_flow_reaches_public_context(self):
        image = Motion().image()
        moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (240, 160), borderMode=cv2.BORDER_REFLECT)
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = OpenCVFlow.capability(version=cv2.__version__)
            streams.register(spec, OpenCVFlow)
            await streams.open('fixture', 'camera', [spec.id])
            now = timestamp()
            for seq, data in enumerate((image, moved)):
                await streams.submit(InputFrame(data, 'video', now-100+seq*100,
                    sensor=SensorMetadata('fixture', 'camera', seq)))
                await streams.drain()
            observation = next(iter((await human.get_context())['observations'].values()))
            self.assertAlmostEqual(observation['value']['motion']['dx_pixels'], 3, delta=.5)
            self.assertEqual(observation['models'][0]['version'], cv2.__version__)
            self.assertEqual(observation['sensor']['source_id'], 'fixture')
