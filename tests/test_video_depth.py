import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.provider_profiles import capability
from gamst.providers.video_depth import VideoDepthAnythingContext, configuration, _read

CONFIG = dict(source_dir='/source', path='/model.pth', sha256='a'*64)


class Predictor:
    def __init__(self):
        self.calls = []
        self.resets = 0
        self.closed = False
        self.output = None

    @property
    def cache_size(self):
        return min(len(self.calls), 32)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True

    def __call__(self, rgb):
        self.calls.append(rgb.copy())
        if self.output is not None:
            return self.output
        height, width = rgb.shape[:2]
        return np.tile(np.arange(width, dtype=np.float32), (height, 1))


def frame(index=0, *, timestamp=None, epoch='one'):
    data = np.zeros((36, 48, 3), dtype=np.uint8)
    data[:, :, 0] = 10
    data[:, :, 2] = 90
    return InputFrame(data, 'video', 1000+40*index if timestamp is None else timestamp,
        sensor=SensorMetadata('camera', 'video', index, coordinate_frame='image', stream_epoch=epoch))


def context(**options):
    predictor = Predictor()
    provider = VideoDepthAnythingContext(predictor, checkpoint_sha256='a'*64, **options)
    return provider, predictor


class Discovery(unittest.TestCase):
    def test_profile_is_dependency_light_and_bounded(self):
        code = ('import sys;from gamst.provider_profiles import capability;'
                's=capability("video_depth_anything.small",'+repr(CONFIG)+');'
                'assert s.stateful and s.reset_on_gap;'
                'assert not {"torch","cv2","video_depth_anything"}&sys.modules.keys()')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for bad in ({'source_dir': 'relative'}, {'device': 'mps'}, {'grid_size': True},
                    {'max_gap_ms': 0}, {'sha256': 'A'*64}, {'extra': 1}):
            with self.assertRaises(ValueError):
                configuration(dict(CONFIG, **bad))

    def test_checkpoint_verification(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            path = Path(root).resolve()/'weights.pth'
            path.write_bytes(b'fixture')
            digest = hashlib.sha256(b'fixture').hexdigest()
            self.assertEqual(_read(path, digest, 7), b'fixture')
            with self.assertRaises(ValueError):
                _read(path, digest, 6)
            with self.assertRaises(ValueError):
                _read(path, 'a'*64, 7)
            with self.assertRaises(ValueError):
                _read(Path(root)/'missing', digest)


@unittest.skipIf(np is None, 'optional numpy unavailable')
class Adapter(unittest.TestCase):
    def test_rgb_copy_temporal_cache_and_spatial_grid(self):
        provider, predictor = context(grid_size=2)
        first = provider.process(frame())
        second = provider.process(frame(1))
        self.assertEqual(first['summary']['status'], 'estimated')
        self.assertFalse(first['summary']['stream_reset'])
        self.assertFalse(second['summary']['stream_reset'])
        self.assertEqual(second['summary']['processed_frames'], 2)
        self.assertEqual(second['summary']['cache_frames'], 2)
        self.assertEqual(second['summary']['capture_timestamp_ms'], 1040)
        self.assertTrue(second['summary']['temporal_tracking'])
        self.assertFalse(second['summary']['temporal_comparability'])
        self.assertFalse(second['summary']['metric_depth'])
        self.assertEqual(predictor.calls[0][0, 0].tolist(), [90, 0, 10])
        self.assertEqual(len(second['grid']['relative_nearness']), 4)
        provider.close()
        self.assertTrue(predictor.closed)
        with self.assertRaises(RuntimeError):
            provider.process(frame(2))

    def test_gap_epoch_sequence_and_session_limit_reset_native_cache(self):
        for changed in (frame(2), frame(1, timestamp=5000), frame(1, epoch='two')):
            provider, predictor = context()
            provider.process(frame())
            result = provider.process(changed)
            self.assertTrue(result['summary']['stream_reset'])
            self.assertEqual(result['summary']['processed_frames'], 1)
            self.assertEqual(predictor.resets, 1)
        provider, predictor = context(max_frames=1)
        provider.process(frame())
        result = provider.process(frame(1))
        self.assertTrue(result['summary']['stream_reset'])
        self.assertEqual(predictor.resets, 1)

    def test_flat_abstention_and_invalid_native_output(self):
        provider, predictor = context()
        predictor.output = np.ones((36, 48), dtype=np.float32)
        result = provider.process(frame())
        self.assertEqual(result['summary']['status'], 'unavailable_low_spread')
        self.assertEqual(result['grid']['relative_nearness'], [None]*9)
        for output in (np.zeros((48, 36)), np.zeros((36, 48, 1)),
                       np.full((36, 48), np.nan), np.full((36, 48), -1.),
                       np.zeros((36, 48), dtype=np.uint8)):
            provider, predictor = context()
            predictor.output = output
            with self.assertRaises(ValueError):
                provider.process(frame())
            self.assertEqual(predictor.resets, 1)
            self.assertIsNone(provider.last)

    def test_stale_capture_rejected_without_reset(self):
        provider, predictor = context()
        provider.process(frame(1))
        with self.assertRaises(ValueError):
            provider.process(frame(0))
        self.assertEqual(predictor.resets, 0)
        self.assertEqual(len(predictor.calls), 1)
        provider.reset()
        self.assertEqual(predictor.resets, 1)


if __name__ == '__main__':
    unittest.main()
