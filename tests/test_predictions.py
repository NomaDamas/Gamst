import unittest
from humansense import HumanSense


class PredictionTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_score_and_explicit_feedback_preserve_evidence(self):
        async with HumanSense(consent=True) as human:
            human.define_prediction('needs_help', target='user_requests_assistance', score_name='interaction.help_model_score',
                                    scorer_model_id='custom', scorer_model_version='1')
            await human.push_signal('interaction.help_model_score', .7, 'text', model_id='custom', model_version='1')
            result = (await human.get_history(type='human.inference'))[-1]
            value = result['predictions']['custom']['needs_help']
            self.assertEqual(value['model_score'], .7)
            self.assertIsNone(value['probability'])
            self.assertFalse(value['calibrated'])
            self.assertTrue(value['evidence_ids'])
            feedback = await human.record_feedback(result['id'], True, prediction='needs_help')
            self.assertEqual(feedback['observation']['prediction_snapshot']['model_score'], .7)
            with self.assertRaises(ValueError):
                await human.record_feedback(result['id'], False, subject_id='other')
            await human.delete_data()
            self.assertEqual(await human.get_history(), [])

    async def test_wrong_scorer_and_invalid_score_abstain(self):
        async with HumanSense(consent=True) as human:
            human.define_prediction('x', target='help', score_name='score', scorer_model_id='expected')
            await human.push_signal('score', .7, 'text', model_id='other')
            result = (await human.get_history(type='human.inference'))[-1]
            self.assertEqual(result['predictions']['custom']['x']['status'], 'unavailable')
            await human.push_signal('score', 4, 'text', model_id='expected')
            result = (await human.get_history(type='human.inference'))[-1]
            self.assertIsNone(result['predictions']['custom']['x']['probability'])

    async def test_calibration_matches_scorer_and_deletion_drops_artifact(self):
        from humansense import LabeledOutcome, fit_calibration
        artifact = fit_calibration(target='help',
            training=[LabeledOutcome(f't-{i}', .85, i < 16) for i in range(20)],
            validation=[LabeledOutcome(f'v-{i}', .85, i < 16) for i in range(20)],
            training_dataset='train', validation_dataset='heldout', scorer_model_id='help-model', scorer_model_version='1')
        async with HumanSense(consent=True) as human:
            human.define_prediction('help', target='help', score_name='help.model_score', calibration=artifact)
            await human.push_signal('help.model_score', .85, 'text', model_id='help-model', model_version='1')
            result = (await human.get_history(type='human.inference'))[-1]
            self.assertAlmostEqual(result['predictions']['custom']['help']['probability'], .8)
            self.assertTrue(result['predictions']['custom']['help']['calibrated'])
            await human.delete_data()
            await human.push_signal('help.model_score', .85, 'text', model_id='help-model', model_version='1')
            result = (await human.get_history(type='human.inference'))[-1]
            self.assertIsNone(result['predictions']['custom']['help']['probability'])
