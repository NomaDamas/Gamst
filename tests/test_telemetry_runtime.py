import unittest
from humansense import HumanSense
from humansense.plugins import AudioLevelPlugin


class RuntimeTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_observed_stages_from_real_pcm_and_state_operations(self):
        async with HumanSense(consent=True) as human:
            human.register(AudioLevelPlugin())
            await human.push_audio([.1]*1600)
            await human.push_text('Please clarify')
            await human.get_state()
            metrics=await human.get_metrics()
            stages=metrics['latency']['stages']
            for name in ('sdk.push_audio','plugin.audio_level','sdk.push_signal','sdk.push_text','sdk.get_state'):
                self.assertGreater(stages[name]['count'],0)
                self.assertGreaterEqual(stages[name]['p95_ms'],0)
            self.assertNotIn('plugin.silero_vad',stages)
            await human.delete_data()
            self.assertEqual((await human.get_metrics())['latency']['stages'],{})

    async def test_failed_input_counts_as_a_measured_attempt(self):
        async with HumanSense() as human:
            with self.assertRaises(PermissionError): await human.push_text('blocked')
            stage=(await human.get_metrics())['latency']['stages']['sdk.push_text']
            self.assertEqual(stage['count'],1)

    async def test_vision_interpretation_remains_uncalibrated_and_not_baseline(self):
        async with HumanSense(consent=True) as human:
            await human.push_signal('scene.model_description',
                {'interpretation':'unverified_model_hypotheses',
                 'expressions':[{'name':'interest','model_score':.6}],
                 'observations':[{'name':'posture_description','value':'seated','evidence_description':'upper body visible'}]},
                'video',model_id='test-image-model',model_version='1',quality='insufficient')
            result=(await human.get_history(type='human.inference'))[-1]
            self.assertFalse(result['predictions']['conversation']['expressions']['verified'])
            self.assertIsNone(result['predictions']['conversation']['expressions']['probability'])
            self.assertEqual(result['predictions']['behavior']['metrics'],{})
            self.assertFalse(result['predictions']['behavior']['visual_hypotheses']['verified'])
