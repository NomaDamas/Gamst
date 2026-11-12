import asyncio
import unittest
from humansense import HumanSense
from humansense.reasoning import _normalize, parse_analysis, prepare_records


class AnalysisRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_backend_references_records_and_preserves_measured_behavior(self):
        class Backend:
            model_id, model_version = 'test-model', '1'
            async def analyze(self, records):
                prepared = prepare_records(records)
                ids = [r['id'] for r in prepared if r.get('text')]
                return _normalize(parse_analysis({'expressions': [], 'interaction': [
                    {'name': 'needs_clarification', 'hypothesis': 'May need clarification', 'evidence_ids': ids}],
                    'consistency': []}, prepared))
        async with HumanSense(consent=True) as human:
            human.set_analysis_backend(Backend())
            await human.push_text('I do not understand this step')
            await human.push_signal('head.yaw', 4., 'video', model_id='vision', model_version='1')
            result = await human.analyze()
            self.assertEqual(result['recommended_action']['type'], 'ask_clarifying_question')
            self.assertIsNone(result['predictions']['conversation']['hypotheses'][0]['probability'])
            self.assertIn('head.yaw', result['predictions']['behavior']['metrics'])
            self.assertIn('vision', [m['id'] for m in result['models']])

    async def test_analysis_is_discarded_when_data_deleted_during_provider_call(self):
        entered, release = asyncio.Event(), asyncio.Event()
        class Backend:
            model_id, model_version = 'test-model', '1'
            async def analyze(self, records):
                entered.set()
                await release.wait()
                return _normalize(parse_analysis({'expressions': [], 'interaction': [], 'consistency': []}, prepare_records(records)))
        async with HumanSense(consent=True) as human:
            human.set_analysis_backend(Backend())
            await human.push_text('private')
            task = asyncio.create_task(human.analyze())
            await entered.wait()
            await human.delete_data()
            release.set()
            with self.assertRaises(RuntimeError):
                await task
            self.assertEqual(await human.get_history(), [])

    async def test_semantic_analysis_survives_media_updates_but_marks_new_turn(self):
        class Backend:
            model_id, model_version = 'semantic', '1'
            async def analyze(self, records):
                prepared = prepare_records(records)
                return _normalize(parse_analysis({'expressions': [], 'interaction': [
                    {'name': 'may_want_shorter_answer', 'hypothesis': 'May prefer brevity',
                     'evidence_ids': [prepared[0]['id']]}], 'consistency': []}, prepared))
        async with HumanSense(consent=True) as human:
            human.set_analysis_backend(Backend())
            await human.push_text('This explanation is very long')
            analyzed = await human.analyze()
            await human.push_signal('audio.rms', .02, 'audio')
            result = (await human.get_history(type='human.inference'))[-1]
            cached = result['predictions']['conversation']['model_analysis']
            self.assertEqual(cached['record_id'], analyzed['id'])
            self.assertFalse(cached['based_on_prior_turn'])
            self.assertEqual(result['recommended_action']['type'], 'offer_shorter_answer')
            result = await human.push_text('Thanks; now let us discuss a different task')
            self.assertTrue(result['predictions']['conversation']['model_analysis']['based_on_prior_turn'])
            self.assertIsNone(result['recommended_action'])
            await human.delete_data()
            result = await human.push_text('New topic')
            self.assertNotIn('model_analysis', result['predictions']['conversation'])
