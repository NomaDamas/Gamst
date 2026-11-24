import importlib.util
import unittest

HAS_CV = all(importlib.util.find_spec(name) for name in ('cv2', 'numpy'))


def fan_frames(rev_s, fps=240., blades=3, spin_s=1., still_s=.3, size=200, grill=False):
    """A textured fan whose blades are identical (true N-fold symmetry): still, spinning, still."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(3)
    radius, center = size // 2 - 10, (size / 2, size / 2)
    texture = cv2.GaussianBlur(rng.integers(70, 230, (size, size), dtype=np.uint8), (5, 5), 0)
    blade = np.zeros((size, size), np.uint8)
    cv2.ellipse(blade, (size // 2, size // 2), (radius, radius), 0, -25, 25, 255, -1)
    one = np.where(blade > 0, texture, 0).astype(np.uint8)
    fan = np.zeros_like(one)
    for k in range(blades):
        turn = cv2.getRotationMatrix2D(center, 360 * k / blades, 1)
        fan = np.maximum(fan, cv2.warpAffine(one, turn, (size, size)))
    frames, angle = [], 0.
    for index in range(int((spin_s + 2 * still_s) * fps)):
        if still_s <= index / fps < still_s + spin_s:
            angle += 360 * rev_s / fps
        image = np.full((size, size), 30, np.uint8)
        turned = cv2.warpAffine(fan, cv2.getRotationMatrix2D(center, angle, 1), (size, size))
        image[turned > 0] = turned[turned > 0]
        cv2.circle(image, (size // 2, size // 2), 10, 200, -1)
        if grill:  # a static guard in front of the blades
            for ring in range(25, radius + 10, 14):
                cv2.circle(image, (size // 2, size // 2), ring, 255, 1)
        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return frames


@unittest.skipUnless(HAS_CV, 'needs opencv and numpy (the rotation extra)')
class PolarRotationTest(unittest.TestCase):
    def count(self, frames, symmetry, fps=240.):
        import cv2
        import numpy as np
        from gamst.providers.polar_rotation import PolarRotationCounter
        counter = PolarRotationCounter(symmetry=symmetry)
        for index, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            value = counter.process(gray, (100, 100), 95, 1000 + round(index * 1000 / fps))
        return value['rotation_count']

    def test_counts_a_symmetric_fan_below_the_limit(self):
        # 3 blades at 240 fps: countable up to 40 rev/s
        self.assertAlmostEqual(self.count(fan_frames(4), 3), 4, delta=.1)
        self.assertAlmostEqual(self.count(fan_frames(30), 3), 30, delta=.6)

    def test_a_static_grill_does_not_hold_the_count_at_zero(self):
        self.assertAlmostEqual(self.count(fan_frames(12, grill=True), 3), 12, delta=.3)

    def test_above_the_limit_the_fan_aliases(self):
        # 50 rev/s with 3 blades at 240 fps turns 75 deg per frame: it looks like 45 deg backwards
        self.assertAlmostEqual(self.count(fan_frames(50), 3), 30, delta=.6)

    def test_vision_rotation_withholds_an_in_plane_count_the_camera_cannot_vouch_for(self):
        from gamst.capabilities import SensorMetadata
        from gamst.plugins import InputFrame
        from gamst.providers.rotation import RotationProvider
        frames = fan_frames(4)
        results = {}
        for expected_max in (30, 50):
            provider = RotationProvider(region_source='roi', roi=dict(x=.02, y=.02, width=.96, height=.96),
                                        in_plane='polar', symmetry=3, expected_max_rev_s=expected_max)
            ended = []
            for index, frame in enumerate(frames + frames[:1] * 120):
                sensor = SensorMetadata(source_id='camera', sensor_id='fan', sequence=index,
                                        coordinate_frame='image', stream_epoch='test')
                value = provider.process(InputFrame(frame, 'video', 1000 + round(index * 1000 / 240), sensor=sensor))
                ended += [event for event in value['events']
                          if event['type'] == 'rotation.ended' and event['axis'] == 'in_plane']
            provider.close()
            results[expected_max] = ended
        counted = max(results[30], key=lambda event: event['turns'] or 0)
        self.assertEqual(counted['status'], 'counting')
        self.assertAlmostEqual(counted['turns'], 4, delta=.2)
        self.assertAlmostEqual(counted['max_countable_rev_s'], 40, delta=.2)  # 240 fps on millisecond stamps
        self.assertTrue(results[50])
        self.assertTrue(all(event['status'] == 'undersampled' and event['turns'] is None for event in results[50]))

    def test_a_spin_still_running_when_the_input_ends_is_reported(self):
        from gamst.capabilities import SensorMetadata
        from gamst.plugins import InputFrame
        from gamst.providers.rotation import RotationProvider
        provider = RotationProvider(region_source='roi', roi=dict(x=.02, y=.02, width=.96, height=.96),
                                    in_plane='polar', symmetry=3, expected_max_rev_s=10)
        frames = fan_frames(3, still_s=.2)
        spinning = frames[:len(frames) - round(.2 * 240)]  # the file ends mid-spin
        for index, frame in enumerate(spinning):
            sensor = SensorMetadata(source_id='camera', sensor_id='fan', sequence=index,
                                    coordinate_frame='image', stream_epoch='test')
            value = provider.process(InputFrame(frame, 'video', 1000 + round(index * 1000 / 240), sensor=sensor))
            self.assertFalse([event for event in value['events'] if event['type'] == 'rotation.ended'])
        ended = provider.finish()['events']
        provider.close()
        self.assertEqual([(event['axis'], event['reason']) for event in ended], [('in_plane', 'input_ended')])
        self.assertAlmostEqual(ended[0]['turns'], 3, delta=.15)


if __name__ == '__main__':
    unittest.main()
