import unittest
from dataclasses import replace
try:
    import numpy as np
except ImportError:
    np = None
from humansense.providers.beamforming import DelaySumBeamformer
from humansense import HumanSense, SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.prosody import AudioProsody
from humansense.schema import timestamp


def provider(delay=5/16000):
    return DelaySumBeamformer(steering_delay_seconds=delay, output_sensor_id='selected',
        spacing_m=.2, sound_speed_m_s=343, calibration_id='fixture')


def frames(lag=5, count=8):
    rng = np.random.default_rng(5)
    clean = rng.uniform(-.5, .5, 320*count+40)
    data = np.column_stack((clean[20:20+320*count], clean[20+lag:20+lag+320*count]))
    start = timestamp()-count*20
    return [InputFrame(data[i*320:(i+1)*320], 'audio', start+(i+1)*20, sample_rate=16000,
        sensor=SensorMetadata('array', 'ambient', i, channels=('mic0', 'mic1'),
            coordinate_frame='array', calibration_id='fixture')) for i in range(count)]


@unittest.skipIf(np is None, 'numpy not installed')
class Beamforming(unittest.TestCase):
    def test_alignment_both_signs_and_zero_with_bounded_history(self):
        for lag in (-5, 0, 5):
            p = provider(lag/16000)
            inputs = frames(lag)
            all_data = np.concatenate([f.data for f in inputs])
            for i, f in enumerate(inputs):
                before = f.data.copy()
                result = p.transform(f)
                np.testing.assert_array_equal(before, f.data)
                if i == 0 and lag:
                    self.assertIsNone(result)
                    continue
                delay0 = max(0, -lag)
                expected = all_data[i*320-delay0:(i+1)*320-delay0, 0]
                np.testing.assert_allclose(result.frame.data, expected, atol=3e-8)
                self.assertEqual(result.frame.sensor.sensor_id, 'selected')
                self.assertLessEqual(0 if p.history is None else len(p.history), 5)
                p.spec.validate(result.context)

    def test_gaps_and_malformed_input_clear_history(self):
        for change in ('sequence', 'clock', 'epoch', 'subject'):
            p, inputs = provider(), frames()
            p.transform(inputs[0])
            f = inputs[1]
            if change == 'sequence': f = replace(f, sensor=replace(f.sensor, sequence=8))
            if change == 'clock': f = replace(f, timestamp_ms=f.timestamp_ms+5)
            if change == 'epoch': f = replace(f, sensor=replace(f.sensor, stream_epoch='new'))
            if change == 'subject': f = replace(f, subject_id='new')
            self.assertIsNone(p.transform(f))
            self.assertEqual(p.continuity_resets, 1)
        with self.assertRaises(ValueError): p.transform(replace(f, data=f.data[:2]))
        self.assertIsNone(p.history)
        p.transform(inputs[0])
        with self.assertRaises(ValueError): p.transform(inputs[0])
        self.assertIsNone(p.history)
        p.close()
        with self.assertRaises(RuntimeError): p.transform(inputs[1])
        for delay in (True, float('nan'), 1):
            with self.assertRaises(ValueError): provider(delay)

    def test_quantization_int16_and_no_gain_or_circular_wrap(self):
        p = provider(2.4/16000)
        f = frames()[0]
        data = np.zeros((320, 2), dtype=np.int16)
        data[-1, 1] = 32767
        self.assertIsNone(p.transform(replace(f, data=data)))
        result = p.transform(replace(f, timestamp_ms=f.timestamp_ms+20,
            sensor=replace(f.sensor, sequence=1), data=np.zeros_like(data)))
        self.assertAlmostEqual(result.frame.data[1], 32767/65536)
        self.assertEqual(np.count_nonzero(result.frame.data), 1)
        self.assertEqual(result.context['selection']['channel_delays_samples'], [0, 2])


@unittest.skipIf(np is None, 'numpy not installed')
class BeamformingContext(unittest.IsolatedAsyncioTestCase):
    async def test_selected_audio_feeds_real_prosody_and_metadata_reaches_context(self):
        async with HumanSense(consent=True) as human:
            p, prosody = provider(), AudioProsody()
            human.register_capability(p.spec)
            features = None
            for f in frames():
                result = p.transform(f)
                if result is None: continue
                await human.push_observation(p.spec.id, result.context, sensor=f.sensor,
                    timestamp_ms=f.timestamp_ms)
                features = prosody.process(result.frame) or features
            self.assertIsNotNone(features)
            obs = next(iter((await human.get_context())['observations'].values()))
            self.assertEqual(obs['value']['selection']['output_sensor_id'], 'selected')
            self.assertNotIn('data', obs['value']['selection'])
