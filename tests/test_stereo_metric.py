"""Generated stereo fixtures prove software geometry, not device calibration."""
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from humansense.providers.stereo_metric import StereoGeometry


def calibration():
    return dict(calibration_id='fixture', width=320, height=160,
                left_intrinsics=[[200,0,160],[0,200,80],[0,0,1]],
                right_intrinsics=[[200,0,160],[0,200,80],[0,0,1]],
                left_distortion=[0]*5, right_distortion=[0]*5,
                right_from_left_rotation=[[1,0,0],[0,1,0],[0,0,1]],
                right_from_left_translation_m=[-.1,0,0])


@unittest.skipIf(cv2 is None, 'explicit OpenCV optional runtime unavailable')
class StereoMetricTests(unittest.TestCase):
    def test_actual_matcher_known_disparity(self):
        model = StereoGeometry(calibration(), disparities=32)
        self.addCleanup(model.close)
        left = np.random.default_rng(49).integers(0,256,(160,320), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        saved = left.copy()
        xyz, valid = model.compute(left, right)
        roi = valid[20:140,60:240]
        self.assertGreater(roi.mean(), .95)
        np.testing.assert_allclose(np.median(xyz[20:140,60:240][roi], axis=0)[2], 2.5, atol=.03)
        self.assertTrue(np.array_equal(saved, left))
        # Positive x follows image-right, positive y image-down in this frame.
        self.assertGreater(np.nanmedian(xyz[90:120,180:220,0]), 0)
        self.assertGreater(np.nanmedian(xyz[90:120,180:220,1]), 0)
        self.assertTrue(np.isnan(xyz[~valid]).all())

    def test_no_texture_abstains_and_close(self):
        model = StereoGeometry(calibration(), disparities=32)
        a = np.zeros((160,320), np.uint8)
        xyz, valid = model.compute(a,a)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(xyz).all())
        model.close()
        with self.assertRaises(RuntimeError): model.compute(a,a)
        with self.assertRaises(RuntimeError): model.rectified_mask(a)

    def test_original_mask_remaps_and_owns_input(self):
        model = StereoGeometry(calibration(), disparities=32)
        self.addCleanup(model.close)
        mask = np.zeros((160,320), np.uint8)
        mask[30:80,80:130] = 255
        got = model.rectified_mask(mask)
        mask.fill(0)
        self.assertEqual(int(got.sum()),2500)

    def test_invalid_geometry_and_resources(self):
        c = calibration()
        for field, value in [('right_from_left_translation_m',[0,0,0]),
                             ('right_from_left_translation_m',[.1,0,0]),
                             ('right_from_left_translation_m',[0,-.1,0]),
                             ('right_from_left_rotation',[[1,0,0],[0,1,0],[0,0,-1]]),
                             ('left_intrinsics',[[200,0,160],[0,float('nan'),80],[0,0,1]]),
                             ('width',100000)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                StereoGeometry(dict(c, **{field:value}), disparities=32)
        for d in (0,15,256):
            with self.assertRaises(ValueError): StereoGeometry(c, disparities=d)
        model = StereoGeometry(c, disparities=32)
        self.addCleanup(model.close)
        for image in (np.zeros((160,320), np.float32), np.zeros((160,320,3),np.uint8)):
            with self.assertRaises(ValueError): model.compute(image,image)


if __name__ == '__main__':
    unittest.main()
