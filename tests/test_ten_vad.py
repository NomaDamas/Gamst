import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gamst import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.ten_vad import TENActivity, TEN_VAD_SOURCE_REVISION


class RuntimeDouble:
    def __init__(self):
        self.scores = []
        self.inputs = []
        self.resets = self.closes = 0

    def process(self, samples):
        self.inputs.append(samples.copy())
        score = self.scores.pop(0) if self.scores else .1
        return score, int(score >= .5)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closes += 1


def provider(runtime=None, **kwargs):
    return TENActivity(runtime or RuntimeDouble(), version='fixture', **kwargs)


def frame(values, ts=2000, sequence=0):
    return InputFrame(values, 'audio', ts, sample_rate=16000,
        sensor=SensorMetadata('ambient', 'mic', sequence))


@unittest.skipIf(np is None, 'numpy not installed')
class TENVADProvider(unittest.TestCase):
    def test_hop_buffering_timestamps_hysteresis_and_pcm_conversion(self):
        runtime = RuntimeDouble()
        vad = provider(runtime)
        self.assertIsNone(vad.process(frame([.1]*160)))
        runtime.scores = [.8, .4, .2, .9]
        result = vad.process(frame([.2]*928, 2058))
        self.assertEqual([item['end_ms'] for item in result['decisions']],
                         [2006, 2022, 2038, 2054])
        self.assertEqual([item['transition'] for item in result['decisions']],
                         ['start', 'none', 'stop', 'start'])
        self.assertEqual(result['state']['pending_samples'], 64)
        self.assertEqual(runtime.inputs[0].dtype, np.int16)
        self.assertEqual(runtime.inputs[0].shape, (256,))
        self.assertEqual(int(runtime.inputs[0][0]), 3276)
        self.assertEqual(result['state']['hop_ms'], 16)

    def test_ten_millisecond_hop(self):
        runtime = RuntimeDouble()
        runtime.scores = [.7]
        vad = provider(runtime, hop_ms=10)
        result = vad.process(frame(np.zeros(160, dtype=np.float32), 1010))
        self.assertEqual(result['decisions'][0]['start_ms'], 1000)
        self.assertEqual(result['decisions'][0]['end_ms'], 1010)
        self.assertEqual(vad.spec.window_ms, 10)

    def test_gap_resets_native_state_and_discards_remainder(self):
        runtime = RuntimeDouble()
        vad = provider(runtime)
        vad.process(frame([.1]*160))
        vad.process(frame([.2]*160, 2100))
        self.assertEqual(runtime.resets, 1)
        runtime.scores = [.9]
        result = vad.process(frame([.3]*96, 2106))
        self.assertEqual(result['decisions'][0]['start_ms'], 2090)
        self.assertEqual(len(runtime.inputs[0]), 256)

    def test_bounded_empty_and_malformed_audio(self):
        vad = provider()
        self.assertIsNone(vad.process(frame([])))
        for values in ([0.]*16001, bytes(32002), b'\x00', [float('nan')], [2.], [[0.]]):
            with self.assertRaises(ValueError):
                vad.process(frame(values))
        result = vad.process(frame(bytes(32000)))
        self.assertEqual(len(result['decisions']), 62)
        self.assertEqual(len(vad.buffer), 128)

    def test_source_rate_timestamp_and_layout_validation(self):
        vad = provider()
        vad.process(frame([0.]*256))
        for bad in (frame([0.]*256),
                    replace(frame([0.]*256, 2020), sample_rate=8000),
                    replace(frame([0.]*256, 2020), sensor=None),
                    replace(frame([0.]*256, 2020), sensor=SensorMetadata('other', 'mic', 1)),
                    replace(frame([0.]*256, 2020), sensor=SensorMetadata('ambient', 'mic', 1, channels=('a', 'b')))):
            with self.assertRaises(ValueError):
                vad.process(bad)
        vad.reset()
        self.assertIsNotNone(vad.process(replace(frame([0.]*256), sensor=SensorMetadata('other', 'mic', 1))))

    def test_invalid_runtime_results_reset_and_close_releases(self):
        for bad in (float('nan'), -.1, 1.1):
            runtime = RuntimeDouble()
            runtime.scores = [bad]
            vad = provider(runtime)
            with self.assertRaises(ValueError):
                vad.process(frame([0.]*256))
            self.assertEqual(vad.buffer, [])
            self.assertEqual(runtime.resets, 1)
        runtime = RuntimeDouble()
        vad = provider(runtime)
        vad.close()
        vad.close()
        self.assertEqual(runtime.closes, 1)
        with self.assertRaises(RuntimeError):
            vad.process(frame([0.]*256))

    def test_local_hash_and_configuration_bounds(self):
        digest = hashlib.sha256(b'native fixture').hexdigest()
        self.assertIn(TEN_VAD_SOURCE_REVISION[:12], TENActivity.artifact_version(digest))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'libten_vad.so'
            path.write_bytes(b'wrong native library')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                TENActivity.from_local(path, digest)
        with self.assertRaises(ValueError):
            provider(hop_ms=20)
        with self.assertRaises(ValueError):
            provider(start_threshold=.2, stop_threshold=.3)


if __name__ == '__main__':
    unittest.main()
