import math
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.prosody import AudioProsody
from humansense.schema import timestamp


@unittest.skipIf(np is None, 'optional numpy not installed')
class Prosody(unittest.TestCase):
    def tone(self, hz=220, duration_ms=80, rate=16000, amplitude=.2, start=0):
        t = (np.arange(round(rate*duration_ms/1000))+start)/rate
        return amplitude*np.sin(2*np.pi*hz*t)

    def frame(self, data, ts=1000, rate=16000, channels=()):
        return InputFrame(data, 'audio', ts, sample_rate=rate,
                          sensor=SensorMetadata('mic', 'device', 0, channels=channels))

    def test_pitch_and_energy_on_known_signals(self):
        for hz in (60, 110, 220, 440, 500):
            provider = AudioProsody()
            features = provider.process(self.frame(self.tone(hz)))['features']
            self.assertAlmostEqual(features['pitch_hz'], hz, delta=1.)
            self.assertAlmostEqual(features['rms'], .2/math.sqrt(2), delta=.005)
            self.assertGreater(features['periodicity_score'], .95)

    def test_silence_dc_noise_and_clipping_semantics(self):
        for samples in (np.zeros(1280), np.full(1280, .5), np.random.default_rng(9).normal(0, .1, 1280)):
            features = AudioProsody().process(self.frame(samples))['features']
            self.assertIsNone(features['pitch_hz'])
            self.assertFalse(features['periodic_signal_candidate'])
        self.assertIsNone(AudioProsody().process(self.frame(np.zeros(1280)))['features']['dbfs'])
        clipped = AudioProsody().process(self.frame(np.ones(1280)))['features']
        self.assertEqual(clipped['clipping_fraction'], 1.)

    def test_chunk_state_and_gap_reset(self):
        provider = AudioProsody()
        for index in range(4):
            result = provider.process(self.frame(self.tone(duration_ms=20, start=index*320), ts=1000+20*index))
            if index < 3:
                self.assertIsNone(result)
        self.assertAlmostEqual(result['features']['pitch_hz'], 220, delta=1.)
        self.assertEqual(len(provider.buffer), 1280)
        self.assertIsNone(provider.process(self.frame(self.tone(duration_ms=20), ts=2000)))
        self.assertEqual(len(provider.buffer), 320)
        provider.reset()
        self.assertEqual(provider.buffer, [])

    def test_explicit_channel_and_rate_change(self):
        stereo = np.column_stack((self.tone(220), self.tone(330)))
        with self.assertRaises(ValueError):
            AudioProsody().process(self.frame(stereo, channels=('left', 'right')))
        provider = AudioProsody(channel=1)
        self.assertAlmostEqual(provider.process(self.frame(stereo, channels=('left', 'right')))['features']['pitch_hz'], 330, delta=1)
        with self.assertRaises(ValueError):
            provider.process(self.frame(stereo, ts=1080, channels=('only-one',)))
        mono = AudioProsody()
        mono.process(self.frame(self.tone()))
        result = mono.process(self.frame(self.tone(rate=8000), ts=1080, rate=8000))
        self.assertEqual(result['features']['sample_rate'], 8000)
        self.assertEqual(result['features']['window_samples'], 640)

    def test_short_overlap_and_gap_do_not_form_false_contiguous_window(self):
        for timestamps in ((1000, 1001, 1002, 1003), (1000, 1020, 1043, 1063)):
            provider = AudioProsody()
            for ts in timestamps:
                self.assertIsNone(provider.process(self.frame(self.tone(duration_ms=20), ts=ts)))
            self.assertLess(len(provider.buffer), 1280)

    def test_integer_timestamp_rounding_is_supported(self):
        provider = AudioProsody()
        rate, count = 44100, 500
        result = None
        for index in range(8):
            samples = .2*np.sin(2*np.pi*220*(np.arange(count)+index*count)/rate)
            ts = round(1000+(index+1)*count*1000/rate)
            result = provider.process(self.frame(samples, ts=ts, rate=rate))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['features']['pitch_hz'], 220, delta=1.)

    def test_rejects_invalid_pcm_and_lifecycle(self):
        provider = AudioProsody()
        for data in ([float('nan')], [1.1], [float('inf')]):
            with self.assertRaises(ValueError):
                provider.process(self.frame(data))
        provider.process(self.frame(self.tone()))
        with self.assertRaises(ValueError):
            provider.process(self.frame(self.tone()))
        provider.close()
        with self.assertRaises(RuntimeError):
            provider.process(self.frame(self.tone(), ts=1080))


@unittest.skipIf(np is None, 'optional numpy not installed')
class ProsodyRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_feature_window_reaches_context(self):
        async with HumanSense(consent=True) as human:
            streams = SensorStreams(human)
            spec = AudioProsody.capability()
            streams.register(spec, AudioProsody)
            await streams.open('mic', 'device', [spec.id])
            tone = Prosody().tone()
            await streams.submit(InputFrame(tone, 'audio', timestamp(), sample_rate=16000,
                sensor=SensorMetadata('mic', 'device', 0)))
            await streams.drain()
            observation = next(iter((await human.get_context())['observations'].values()))
            self.assertAlmostEqual(observation['value']['features']['pitch_hz'], 220, delta=1)
            self.assertEqual(observation['capability']['window_ms'], 80)
            self.assertEqual(observation['interpretation'], 'estimate')
