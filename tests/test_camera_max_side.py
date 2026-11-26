import importlib.util
import unittest
from unittest.mock import patch

HAS_CV = all(importlib.util.find_spec(name) for name in ('cv2', 'numpy'))


@unittest.skipUnless(HAS_CV, 'needs opencv')
class CameraMaxSideTest(unittest.IsolatedAsyncioTestCase):
    async def shapes(self, size, **options):
        import numpy as np
        from gamst.inputs import CameraInput

        class Device:
            def __init__(self, *_):
                self.left = 3

            def isOpened(self):
                return True

            def read(self):
                self.left -= 1
                return self.left >= 0, np.zeros(size, np.uint8)

            def release(self):
                pass

        with patch('cv2.VideoCapture', Device):
            return [frame.data.shape async for frame in CameraInput(0, max_frames=3, **options).frames()]

    async def test_larger_frames_shrink_and_keep_their_proportions(self):
        self.assertEqual(await self.shapes((1080, 1920, 3), max_side=640), [(360, 640, 3)] * 3)
        self.assertEqual(await self.shapes((1920, 1080, 3), max_side=640), [(640, 360, 3)] * 3)

    async def test_smaller_frames_and_the_default_are_untouched(self):
        self.assertEqual(await self.shapes((360, 480, 3), max_side=640), [(360, 480, 3)] * 3)
        self.assertEqual(await self.shapes((1080, 1920, 3)), [(1080, 1920, 3)] * 3)


if __name__ == '__main__':
    unittest.main()
