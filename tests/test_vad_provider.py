from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from humansense import ProcessProvider, SensorMetadata, SensorStreams, HumanSense
from humansense.plugins import InputFrame
from humansense.providers.vad import SileroActivity
from humansense.schema import timestamp


class Model:
    def __init__(self):
        self.resets = 0
        self.scores = []
        self.inputs = []

    def reset_states(self):
        self.resets += 1

    def score(self, values):
        self.inputs.append(list(values))
        return self.scores.pop(0) if self.scores else .1


def frame(values, ts=2000, sequence=0):
    return InputFrame(values, 'audio', ts, sample_rate=16000,
        sensor=SensorMetadata('ambient', 'mic', sequence))


class VADProvider(unittest.TestCase):
    def provider(self):
        model = Model()
        return SileroActivity(model, model.score, version='fixture')

    def test_remainder_timestamps_and_all_transitions(self):
        provider = self.provider()
        self.assertIsNone(provider.process(frame([.1]*320, 2000)))
        provider.model.scores = [.8, .5, .2, .9]
        result = provider.process(frame([.2]*1760, 2110))
        decisions = result['decisions']
        self.assertEqual([d['end_ms'] for d in decisions], [2012, 2044, 2076, 2108])
        self.assertEqual([d['transition'] for d in decisions], ['start', 'none', 'stop', 'start'])
        self.assertEqual([d['active'] for d in decisions], [True, True, False, True])
        self.assertEqual(result['state']['pending_samples'], 32)
        self.assertEqual(provider.model.inputs[0], [.1]*320+[.2]*192)
        self.assertEqual(provider.model.resets, 1)

    def test_capture_rounding_does_not_accumulate(self):
        for step in (9, 11):
            provider = self.provider()
            for index in range(100):
                self.assertIsNone(provider.process(frame([0.]*160, 2000+step*index)))
            self.assertEqual(provider.model.inputs, [])

    def test_gap_discards_buffer_and_neural_state(self):
        provider = self.provider()
        provider.process(frame([.1]*320, 2000))
        self.assertIsNone(provider.process(frame([.2]*320, 2100)))
        self.assertEqual(provider.model.resets, 2)
        result = provider.process(frame([.3]*192, 2112))
        self.assertEqual(provider.model.inputs[0], [.2]*320+[.3]*192)
        self.assertEqual(result['decisions'][0]['start_ms'], 2080)

    def test_bounded_block_and_empty_malformed_pcm(self):
        provider = self.provider()
        self.assertIsNone(provider.process(frame([])))
        for values in ([0.]*16001, bytes(32002), b'\x00', [float('nan')], [2.], [[0.]]):
            with self.assertRaises(ValueError):
                provider.process(frame(values))
        result = provider.process(frame(bytes(32000)))
        self.assertEqual(len(result['decisions']), 31)
        self.assertEqual(len(provider.buffer), 128)
        result = provider.process(frame(bytes(32000), 3000))
        self.assertLessEqual(len(result['decisions']), 32)
        self.assertLess(len(provider.buffer), 512)

    def test_source_timestamp_rate_and_layout(self):
        provider = self.provider()
        provider.process(frame([0.]*320))
        for bad in (frame([0.]*320), replace(frame([0.]*320, 2020), sample_rate=8000),
                    replace(frame([0.]*320, 2020), sensor=None),
                    replace(frame([0.]*320, 2020), sensor=SensorMetadata('other', 'mic', 1)),
                    replace(frame([0.]*320, 2020), sensor=SensorMetadata('ambient', 'mic', 1, channels=('a', 'b')))):
            with self.assertRaises(ValueError):
                provider.process(bad)
        provider.reset()
        self.assertIsNone(provider.process(replace(frame([0.]*320), sensor=SensorMetadata('other', 'mic', 1))))

    def test_invalid_model_result_resets_and_close_releases(self):
        provider = self.provider()
        model = provider.model
        model.scores = [.9, float('nan')]
        with self.assertRaises(ValueError):
            provider.process(frame([0.]*1024))
        self.assertFalse(provider.active)
        self.assertEqual(provider.buffer, [])
        self.assertEqual(model.resets, 2)
        provider.close()
        provider.close()
        self.assertIsNone(provider.model)
        self.assertIsNone(provider.score)
        with self.assertRaises(RuntimeError):
            provider.process(frame([0.]*512))

    def test_model_hash_and_discovery_without_torch(self):
        digest = 'a'*64
        self.assertEqual(ProcessProvider.capability('silero.activity', {'sha256': digest}).streaming_mode, 'native')
        with self.assertRaises(ValueError):
            ProcessProvider.capability('silero.activity', {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrong.jit'
            path.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                SileroActivity.from_local(path, digest)


class VADRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_sequence_gap_resets_native_state_and_retains_decisions(self):
        model = Model()
        provider = SileroActivity(model, model.score, version='fixture')
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            streams.register(provider.spec, lambda: provider)
            await streams.open('ambient', 'mic', [provider.spec.id])
            now = timestamp()-100
            await streams.submit(frame([0.]*320, now))
            await streams.drain()
            await streams.submit(frame([0.]*320, now+20, sequence=2))
            await streams.drain()
            self.assertEqual(model.inputs, [])
            model.scores = [.9, .1]
            await streams.submit(frame([0.]*704, now+64, sequence=3))
            await streams.drain()
            context = await human.get_context()
            value = next(iter(context['observations'].values()))
            self.assertEqual([d['transition'] for d in value['value']['decisions']], ['start', 'stop'])
            self.assertTrue(value['fresh'])
