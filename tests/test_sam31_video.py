import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gamst.capabilities import SensorMetadata, bounded_json
from gamst.plugins import InputFrame
from gamst.providers.sam31_video import SAM31VideoContext, configuration, _verify_checkpoint

CONFIG = dict(source_dir='/source', path='/model.pt', sha256='b'*64)


class Backend:
    def __init__(self):
        self.calls = []
        self.resets = 0
        self.closed = False
        self.failure = None

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True

    def infer(self, images, *, prompt, max_objects):
        self.calls.append((len(images), prompt, max_objects))
        if self.failure is not None:
            return self.failure
        result = []
        for index, image in enumerate(images):
            ids = [0] if index % 4 >= 2 else [0, 1]
            masks = np.zeros((len(ids), *image.shape[:2]), dtype=bool)
            masks[0, 2:10, 3:12] = True
            if len(ids) == 2:
                masks[1, 12:20, 15:24] = True
            result.append(dict(obj_ids=ids, masks=masks))
        return result


def frame(index=0, *, timestamp=None, epoch='one'):
    data = np.zeros((32, 40, 3), dtype=np.uint8)
    return InputFrame(data, 'video', 1000+40*index if timestamp is None else timestamp,
        sensor=SensorMetadata('camera', 'video', index, coordinate_frame='image', stream_epoch=epoch))


def context(**options):
    backend = Backend()
    provider = SAM31VideoContext(backend, checkpoint_sha256='b'*64, **options)
    return provider, backend


class Discovery(unittest.TestCase):
    def test_profile_is_dependency_light_and_explicitly_bounded(self):
        code = ('import sys;from gamst.provider_profiles import capability;'
                's=capability("sam3.1.video",'+repr(CONFIG)+');'
                'assert s.streaming_mode=="window" and s.stateful;'
                'assert not {"torch","sam3","PIL"}&sys.modules.keys()')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for bad in ({'source_dir': 'relative'}, {'device': 'cpu'}, {'max_frames': 17},
                    {'max_objects': 5}, {'max_track_gap': 8}, {'prompt': ''}, {'extra': 1}):
            with self.assertRaises(ValueError):
                configuration(dict(CONFIG, **bad))

    def test_checkpoint_hash_is_streamed_and_required(self):
        with tempfile.TemporaryDirectory(dir='.') as root:
            path = Path(root).resolve()/'sam31.pt'
            path.write_bytes(b'fixture')
            digest = hashlib.sha256(b'fixture').hexdigest()
            self.assertEqual(_verify_checkpoint(path, digest), path)
            with self.assertRaises(ValueError):
                _verify_checkpoint(path, 'a'*64)
            with self.assertRaises(ValueError):
                _verify_checkpoint(Path(root)/'missing', digest)


@unittest.skipIf(np is None, 'optional numpy unavailable')
class Adapter(unittest.TestCase):
    def test_bounded_chunk_timestamped_masklets_and_eviction(self):
        provider, backend = context(max_frames=4, max_track_gap=1)
        for index in range(3):
            result = provider.process(frame(index))
            self.assertEqual(result['summary']['status'], 'buffering')
            self.assertEqual(result['summary']['buffered_frames'], index+1)
            self.assertEqual(backend.calls, [])
        result = provider.process(frame(3))
        self.assertEqual(result['summary']['status'], 'estimated')
        self.assertEqual(result['summary']['lookahead_ms'], 120)
        self.assertFalse(result['summary']['causal'])
        self.assertEqual(result['summary']['backpressure'], 'synchronous_bounded_chunk_inference')
        self.assertTrue(result['summary']['native_session_evicted'])
        self.assertEqual(backend.calls, [(4, 'person', 4)])
        self.assertEqual(len(result['objects']), 2)
        first, second = result['objects']
        self.assertEqual(len(first['frames']), 4)
        self.assertEqual([row['timestamp_ms'] for row in first['frames']], [1000, 1040, 1080, 1120])
        self.assertEqual(first['frames'][0]['mask']['foreground_pixels'], 72)
        self.assertEqual(second['status'], 'expired')
        self.assertIn(second['track_id'], result['summary']['expired_track_ids'])
        self.assertEqual(second['eviction_timestamp_ms'], 1120)
        self.assertLessEqual(len(json.dumps(result)), 65536)
        bounded_json(result, forbid_raw=True)
        next_chunk = provider.finish()
        self.assertIsNone(next_chunk)

    def test_chunk_local_ids_and_finish_partial_chunk(self):
        provider, backend = context(max_frames=2)
        provider.process(frame())
        first = provider.process(frame(1))
        provider.process(frame(2))
        second = provider.process(frame(3))
        self.assertNotEqual(first['objects'][0]['track_id'], second['objects'][0]['track_id'])
        provider.process(frame(4))
        final = provider.finish()
        self.assertEqual(final['summary']['status'], 'estimated')
        self.assertEqual(final['summary']['capture_timestamp_ms'], 1160)
        self.assertEqual(backend.calls[-1][0], 1)

    def test_gap_drops_buffered_frames_and_applies_no_fake_output(self):
        provider, backend = context(max_frames=2)
        provider.process(frame())
        buffering = provider.process(frame(2, timestamp=5000))
        self.assertTrue(buffering['summary']['stream_reset'])
        self.assertEqual(buffering['summary']['dropped_buffered_frames'], 1)
        self.assertEqual(backend.calls, [])
        result = provider.process(frame(3, timestamp=5040))
        self.assertEqual(result['summary']['status'], 'estimated')
        self.assertEqual(result['summary']['chunk_start_timestamp_ms'], 5000)

    def test_malformed_native_output_invalidates_chunk(self):
        provider, backend = context(max_frames=1)
        for failure in ({}, [], [dict(obj_ids=[0], masks=np.zeros((32, 40), dtype=bool))],
                        [dict(obj_ids=[0], masks=np.zeros((1, 32, 40), dtype=np.uint8))],
                        [dict(obj_ids=[True], masks=np.zeros((1, 32, 40), dtype=bool))]):
            backend.failure = failure
            with self.assertRaises(ValueError):
                provider.process(frame())
            self.assertEqual(provider.buffer, [])
            self.assertGreaterEqual(backend.resets, 1)

    def test_stale_input_and_terminal_close(self):
        provider, backend = context(max_frames=2)
        provider.process(frame(1))
        with self.assertRaises(ValueError):
            provider.process(frame(0))
        self.assertEqual(len(provider.buffer), 1)
        provider.close()
        self.assertTrue(backend.closed)
        with self.assertRaises(RuntimeError):
            provider.process(frame(2))
        with self.assertRaises(RuntimeError):
            provider.reset()


if __name__ == '__main__':
    unittest.main()
