import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

try:
    import numpy as np
except ImportError:
    np = None

from gamst import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.moonshine_asr import (
    MODEL_FILES, MOONSHINE_SOURCE_REVISION, MoonshineStreamingASR)


def line(text, *, line_id=7, start=0., duration=.04, complete=False):
    return NS(text=text, line_id=line_id, start_time=start, duration=duration,
              is_complete=complete)


class RuntimeDouble:
    """Lifecycle/API double only; it is not recognition evidence."""
    def __init__(self):
        self.started = False
        self.starts = self.resets = self.stops = self.closes = 0
        self.inputs = []
        self.transcript = NS(lines=[])

    def start(self):
        self.started = True
        self.starts += 1

    def reset(self):
        self.started = False
        self.resets += 1

    def add_audio(self, values):
        if not self.started:
            raise RuntimeError('not started')
        self.inputs.append(values.copy())

    def update_transcription(self):
        return self.transcript

    def stop(self):
        self.started = False
        self.stops += 1
        return self.transcript

    def close(self):
        self.started = False
        self.closes += 1


def provider(runtime=None, **kwargs):
    options = dict(update_ms=40, max_segment_ms=200)
    options.update(kwargs)
    return MoonshineStreamingASR(runtime or RuntimeDouble(),
        model_version='fixture.not_inference', **options)


def frame(index, start=1000):
    return InputFrame(np.zeros(320, dtype=np.float32), 'audio', start+(index+1)*20,
        sample_rate=16000, sensor=SensorMetadata('fixture', 'mic', index, channels=('mono',)))


@unittest.skipIf(np is None, 'numpy not installed')
class MoonshineLifecycle(unittest.TestCase):
    def test_partial_revisions_vad_final_and_native_timestamps(self):
        runtime = RuntimeDouble()
        speech = provider(runtime)
        self.assertIsNone(speech.process(frame(0)))
        runtime.transcript = NS(lines=[line('hello')])
        first = speech.process(frame(1))
        self.assertEqual((first['text'], first['status'], first['revision']),
                         ('hello', 'partial', 0))
        self.assertEqual((first['start_ms'], first['end_ms'], first['language']),
                         (1000, 1040, 'en'))
        self.assertIsNone(speech.process(frame(2)))
        runtime.transcript = NS(lines=[line('hello world', duration=.08, complete=True)])
        final = speech.process(frame(3))
        self.assertEqual((final['text'], final['status'], final['revision']),
                         ('hello world', 'final', 1))
        self.assertEqual(final['finalization_reason'], 'vad_endpoint')
        self.assertEqual(final['end_ms'], 1080)
        self.assertIsNone(speech.process(frame(4)))
        self.assertEqual(runtime.starts, 1)
        self.assertEqual(len(runtime.inputs), 5)

    def test_empty_partial_abstains_then_retracts_prior_text(self):
        runtime = RuntimeDouble()
        speech = provider(runtime)
        speech.process(frame(0))
        runtime.transcript = NS(lines=[line('', duration=.02)])
        speech.process(frame(1))  # The first empty line is intentionally not published.
        runtime.transcript = NS(lines=[line('wrong')])
        speech.process(frame(2))
        self.assertEqual(speech.process(frame(3))['text'], 'wrong')
        runtime.transcript = NS(lines=[line('', duration=.08)])
        speech.process(frame(4))
        update = speech.process(frame(5))
        self.assertEqual((update['text'], update['revision'], update['status']), ('', 1, 'partial'))

    def test_finish_manual_endpoint_and_segment_rollover(self):
        runtime = RuntimeDouble()
        speech = provider(runtime, update_ms=20, max_segment_ms=60)
        runtime.transcript = NS(lines=[line('one', duration=.02)])
        self.assertEqual(speech.process(frame(0))['status'], 'partial')
        runtime.transcript = NS(lines=[line('one!', duration=.02, complete=True)])
        final = speech.finish()
        self.assertEqual(final['finalization_reason'], 'input_end')
        self.assertIsNone(speech.finish())
        self.assertEqual(runtime.stops, 1)

        runtime.transcript = NS(lines=[line('two', duration=.02)])
        partial = speech.process(frame(1))
        runtime.transcript = NS(lines=[line('two', duration=.02, complete=True)])
        endpoint = speech.endpoint()
        self.assertEqual(endpoint['finalization_reason'], 'manual_endpoint')
        self.assertNotEqual(endpoint['utterance_id'], final['utterance_id'])

        runtime.transcript = NS(lines=[line('three', duration=.02)])
        speech.process(frame(2))
        runtime.transcript = NS(lines=[line('three', duration=.04)])
        speech.process(frame(3))
        runtime.transcript = NS(lines=[line('three', duration=.06, complete=True)])
        rollover = speech.process(frame(4))
        self.assertEqual(rollover['status'], 'final')
        self.assertEqual(rollover['finalization_reason'], 'segment_limit')
        self.assertEqual(runtime.stops, 3)

    def test_invalid_input_gap_and_runtime_output_reset_session(self):
        bad_frames = (replace(frame(1), data=np.zeros(10)),
                      replace(frame(1), sample_rate=8000),
                      replace(frame(1), timestamp_ms=5000), frame(3))
        for bad in bad_frames:
            runtime = RuntimeDouble()
            speech = provider(runtime)
            speech.process(frame(0))
            with self.assertRaises(ValueError):
                speech.process(bad)
            self.assertIsNone(speech.session_start)
            self.assertGreaterEqual(runtime.resets, 1)
        runtime = RuntimeDouble()
        speech = provider(runtime)
        speech.process(frame(0))
        runtime.transcript = NS(lines=[line('x'*4097)])
        with self.assertRaises(ValueError):
            speech.process(frame(1))
        self.assertIsNone(speech.session_start)
        runtime.transcript = NS(lines=[line('late', duration=2.)])
        speech.process(frame(0))
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            speech.process(frame(1))

    def test_close_is_idempotent_and_constructor_bounds(self):
        runtime = RuntimeDouble()
        speech = provider(runtime)
        speech.close()
        speech.close()
        self.assertEqual(runtime.closes, 1)
        with self.assertRaises(RuntimeError):
            speech.process(frame(0))
        with self.assertRaises(ValueError):
            provider(update_ms=45)
        with self.assertRaises(ValueError):
            provider(max_segment_ms=40000)

    def test_local_bundle_requires_every_hash_and_matching_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in MODEL_FILES:
                payload = ('fixture-'+name).encode()
                (root/name).write_bytes(payload)
                files[name] = hashlib.sha256(payload).hexdigest()
            loaded, version = MoonshineStreamingASR.verify_local(root, files)
            self.assertEqual(loaded, root.resolve())
            self.assertIn(MOONSHINE_SOURCE_REVISION[:12], version)
            (root/'tokenizer.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                MoonshineStreamingASR.verify_local(root, files)
            with self.assertRaises(ValueError):
                MoonshineStreamingASR.verify_local(root/'missing', files)
            with self.assertRaises(ValueError):
                MoonshineStreamingASR.artifact_version({})


if __name__ == '__main__':
    unittest.main()
