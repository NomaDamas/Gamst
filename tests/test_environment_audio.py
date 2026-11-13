from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from humansense import HumanSense, SensorStreams, SensorMetadata, CapabilitySpec, OutputField, ProcessProvider
from humansense.plugins import InputFrame
from humansense.providers.environment_audio import YAMNetEvents, MODEL_VERSION
from humansense.schema import timestamp


class Classifier:
    def __init__(self):
        self.inputs, self.closed = [], False
        self.bad = False

    def classify(self, data):
        self.inputs.append(data.copy())
        categories = [NS(index=i, category_name='label-'+str(i), score=0.) for i in range(521)]
        categories[1].score, categories[2].score = .9, .8
        if self.bad:
            categories[1].score = float('nan')
        return [NS(classifications=[NS(categories=categories)])]

    def close(self):
        self.closed = True


def frame(data, ts=1000, sequence=0, channels=()):
    return InputFrame(data, 'audio', ts, sample_rate=16000,
        sensor=SensorMetadata('ambient', 'mic', sequence, channels=channels))


@unittest.skipIf(np is None, 'optional numpy not installed')
class EnvironmentAudio(unittest.TestCase):
    def provider(self, **kwargs):
        return YAMNetEvents(Classifier(), lambda array: array, version='test', **kwargs)

    def test_contiguous_chunks_window_hop_and_nonexclusive_scores(self):
        provider = self.provider()
        for index in range(49):
            result = provider.process(frame(np.zeros(320), ts=1000+20*index))
            if index < 48:
                self.assertIsNone(result)
        self.assertEqual(len(provider.classifier.inputs), 1)
        self.assertEqual(result['window']['sample_count'], 15600)
        self.assertEqual(result['window']['end_ms']-result['window']['start_ms'], 975)
        self.assertFalse(result['window']['exclusive'])
        self.assertGreater(sum(e['score'] for e in result['events']), 1)
        for index in range(49, 73):
            result = provider.process(frame(np.zeros(320), ts=1000+20*index))
            if index < 72:
                self.assertIsNone(result)
        self.assertIsNotNone(result)
        self.assertEqual(len(provider.buffer), 15600)

    def test_small_accumulating_gaps_and_overlaps_do_not_mix_audio(self):
        for step in (9, 11):
            provider = self.provider()
            for index in range(300):
                self.assertIsNone(provider.process(frame(np.zeros(160), ts=1000+step*index)))
            self.assertEqual(provider.classifier.inputs, [])

    def test_channel_selection_preserves_original_and_requires_metadata(self):
        stereo = np.column_stack((np.full(15600, .1), np.full(15600, .5)))
        provider = self.provider(channel=1)
        result = provider.process(frame(stereo, channels=('ambient', 'reference')))
        self.assertEqual(result['window']['channel'], 1)
        np.testing.assert_array_equal(provider.classifier.inputs[0], np.full(15600, .5, dtype=np.float32))
        np.testing.assert_array_equal(stereo[:, 0], np.full(15600, .1))
        with self.assertRaises(ValueError):
            self.provider().process(frame(stereo))
        with self.assertRaises(ValueError):
            self.provider(channel=1).process(frame(stereo, channels=('one',)))

    def test_bounded_topk_and_empty_filtered_result(self):
        result = self.provider(top_k=1).process(frame(np.zeros(15600)))
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['window']['above_threshold_count'], 2)
        self.assertEqual(self.provider(score_threshold=1).process(frame(np.zeros(15600)))['events'], [])

    def test_errors_reset_partial_state_and_close_is_idempotent(self):
        provider = self.provider()
        provider.classifier.bad = True
        with self.assertRaises(ValueError):
            provider.process(frame(np.zeros(15600)))
        self.assertEqual(provider.buffer, [])
        self.assertIsNone(provider.process(frame(np.zeros(320), ts=2000)))
        with self.assertRaises(ValueError):
            provider.process(frame(np.zeros(320), ts=2000))
        with self.assertRaises(ValueError):
            provider.process(replace(frame(np.zeros(320), ts=2020), sample_rate=48000))
        with self.assertRaises(ValueError):
            provider.process(frame(np.zeros(16001), ts=2020))
        for data in ([float('nan')], [1.5], b'\x00'):
            with self.assertRaises(ValueError):
                provider.process(frame(data, ts=2020))
        provider.close()
        provider.close()
        self.assertTrue(provider.classifier.closed)
        with self.assertRaises(RuntimeError):
            provider.process(frame(np.zeros(320), ts=2020))

    def test_reset_required_between_sources_and_pcm_bytes_work(self):
        provider = self.provider()
        self.assertIsNotNone(provider.process(frame(bytes(31200))))
        other = replace(frame(bytes(31200), ts=2000), sensor=SensorMetadata('other', 'mic', 0))
        with self.assertRaises(ValueError):
            provider.process(other)
        provider.reset()
        self.assertIsNotNone(provider.process(other))

    def test_pinned_file_validation_and_worker_discovery(self):
        self.assertEqual(ProcessProvider.capability('yamnet.events', {}).version, MODEL_VERSION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.tflite'
            path.write_bytes(b'wrong model')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                YAMNetEvents.from_local(path)


@unittest.skipIf(np is None, 'optional numpy not installed')
class AmbientRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_speech_lane_mutation_cannot_remove_ambient_observation(self):
        class SpeechCleaner:
            def process(self, frame):
                frame.data[:] = 0
                return {'cleaned': True}
        classifier = Classifier()
        provider = YAMNetEvents(classifier, lambda array: array, version='test')
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            cleaning = CapabilitySpec('clean', 'clean', 'audio', 'fixture', '1',
                (OutputField('cleaned', 'boolean'),), 'frame')
            streams.register(cleaning, SpeechCleaner)
            streams.register(provider.spec, lambda: provider)
            await streams.open('ambient', 'mic', ['clean', provider.spec.id])
            values = np.full(15600, .2)
            await streams.submit(frame(values, ts=timestamp()))
            await streams.drain()
            np.testing.assert_allclose(classifier.inputs[0], .2)
            np.testing.assert_array_equal(values, np.full(15600, .2))
            context = await human.get_context()
            found = [v for v in context['observations'].values() if v['capability']['id'] == provider.spec.id]
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]['value']['events'][0]['index'], 1)
