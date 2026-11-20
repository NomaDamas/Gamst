import hashlib
import json
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
from gamst.providers.ced_events import CEDEvents, CED_SOURCE_REVISION


class ClassifierDouble:
    def __init__(self):
        self.inputs = []
        self.closed = False
        self.bad = None

    def classify(self, values):
        self.inputs.append(values.copy())
        if self.bad == 'shape':
            return np.zeros(526, dtype=np.float32)
        scores = np.zeros(527, dtype=np.float32)
        scores[1], scores[2] = .9, .8
        if self.bad == 'nan':
            scores[1] = float('nan')
        return scores

    def close(self):
        self.closed = True


def labels():
    return tuple('label-'+str(index) for index in range(527))


def provider(classifier=None, **kwargs):
    options = dict(window_ms=1000, hop_ms=200)
    options.update(kwargs)
    return CEDEvents(classifier or ClassifierDouble(), labels(), model='tiny',
        version='fixture.not_inference', **options)


def frame(data, ts=1000, sequence=0, channels=()):
    return InputFrame(data, 'audio', ts, sample_rate=16000,
        sensor=SensorMetadata('ambient', 'mic', sequence, channels=channels))


@unittest.skipIf(np is None, 'optional numpy not installed')
class CEDAudio(unittest.TestCase):
    def test_window_hop_nonexclusive_scores_and_abstention(self):
        classifier = ClassifierDouble()
        events = provider(classifier)
        for index in range(49):
            result = events.process(frame(np.zeros(320, dtype=np.float32), ts=1000+20*index))
            if index < 48:
                self.assertIsNone(result)
        self.assertIsNone(result)  # Full window, but only 980 ms accumulated.
        result = events.process(frame(np.zeros(320, dtype=np.float32), ts=1980))
        self.assertEqual(result['window']['sample_count'], 16000)
        self.assertEqual(result['window']['end_ms']-result['window']['start_ms'], 1000)
        self.assertEqual([event['index'] for event in result['events']], [1, 2])
        self.assertGreater(sum(event['score'] for event in result['events']), 1)
        self.assertEqual(result['window']['ontology'], 'AudioSet-527')
        self.assertEqual(provider(score_threshold=1).process(
            frame(np.zeros(16000, dtype=np.float32)))['events'], [])

    def test_small_accumulating_gaps_and_overlaps_do_not_mix_audio(self):
        for step in (9, 11):
            classifier = ClassifierDouble()
            events = provider(classifier)
            for index in range(300):
                self.assertIsNone(events.process(frame(np.zeros(160, dtype=np.float32), ts=1000+step*index)))
            self.assertEqual(classifier.inputs, [])

    def test_channel_selection_preserves_input_and_requires_metadata(self):
        stereo = np.column_stack((np.full(16000, .1), np.full(16000, .5)))
        classifier = ClassifierDouble()
        result = provider(classifier, channel=1).process(frame(stereo, channels=('ambient', 'reference')))
        self.assertEqual(result['window']['channel'], 1)
        np.testing.assert_array_equal(classifier.inputs[0], np.full(16000, .5, dtype=np.float32))
        np.testing.assert_array_equal(stereo[:, 0], np.full(16000, .1))
        with self.assertRaises(ValueError):
            provider().process(frame(stereo))
        with self.assertRaises(ValueError):
            provider(channel=1).process(frame(stereo, channels=('one',)))

    def test_bounded_topk_pcm_bytes_and_source_reset(self):
        result = provider(top_k=1).process(frame(bytes(32000)))
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['window']['above_threshold_count'], 2)
        events = provider()
        events.process(frame(bytes(32000)))
        other = replace(frame(bytes(32000), ts=2000), sensor=SensorMetadata('other', 'mic', 0))
        with self.assertRaises(ValueError):
            events.process(other)
        events.reset()
        self.assertIsNotNone(events.process(other))

    def test_invalid_model_output_and_audio_reset_and_close(self):
        for bad in ('shape', 'nan'):
            classifier = ClassifierDouble()
            classifier.bad = bad
            events = provider(classifier)
            with self.assertRaises(ValueError):
                events.process(frame(np.zeros(16000, dtype=np.float32)))
            self.assertEqual(events.buffer, [])
        events = provider()
        for data in ([float('nan')], [1.5], b'\x00', np.zeros(16001)):
            with self.assertRaises(ValueError):
                events.process(frame(data, ts=2020))
        with self.assertRaises(ValueError):
            events.process(replace(frame(np.zeros(320), ts=2020), sample_rate=48000))
        classifier = events.classifier
        events.close()
        events.close()
        self.assertTrue(classifier.closed)
        with self.assertRaises(RuntimeError):
            events.process(frame(np.zeros(320), ts=2020))

    def test_local_hashes_labels_and_mini_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'ced_tiny.onnx'
            label_file = root/'config.json'
            model.write_bytes(b'ONNX fixture')
            label_file.write_text(json.dumps({'id2label': {str(i): label for i, label in enumerate(labels())}}))
            model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
            label_hash = hashlib.sha256(label_file.read_bytes()).hexdigest()
            _, _, loaded_labels, version = CEDEvents.verify_local(
                model, label_file, model_hash, label_hash, 'tiny')
            self.assertEqual(loaded_labels, labels())
            self.assertIn(CED_SOURCE_REVISION[:12], version)
            spec = CEDEvents.capability(model='mini', version=version)
            self.assertEqual(spec.id, 'audio.ced_mini.events')
            self.assertIn('GPL-3.0', spec.license)
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                CEDEvents.verify_local(model, label_file, '0'*64, label_hash, 'tiny')
            label_file.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                CEDEvents.verify_local(model, label_file, model_hash, label_hash, 'tiny')
            with self.assertRaises(ValueError):
                CEDEvents.artifact_version('huge', model_hash, label_hash)


if __name__ == '__main__':
    unittest.main()
