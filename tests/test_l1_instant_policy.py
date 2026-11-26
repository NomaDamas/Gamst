"""L1 counts every heard word at once (excluded only when clearly another voice); voice reference quality check."""
import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamst.providers.speaker_verify import SpeakerVerify, centroid, cosine
from gamst.voice_quality import choose, combine, hud_line, self_consistency


class Embedder:
    model_id = 'stand-in'

    def __init__(self, vector):
        self.vector = vector

    def embed(self, samples, rate):
        return self.vector


def module(vector, covered=True, **options):
    found = SpeakerVerify(Embedder(vector), words=True, context_ms=1500, segments=False, threshold=.5, **options)
    pipeline = type('P', (), dict(references={'voice': dict(kind='voice', value=dict(
        model='stand-in', embeddings=[[1., 0., 0.]]))}))()
    found.bind(pipeline)
    found._covers = lambda first, last: covered
    found._span = lambda first, last: np.full(16000, .1, dtype=np.float32)
    return found


def word(found):
    return asyncio.run(found._word(dict(text='뽀또', start_ms=500, end_ms=800)))


class L1PolicyTest(unittest.TestCase):
    def test_threshold_policy_is_the_old_behaviour(self):
        self.assertFalse(word(module([.45, .89, 0.]))['match'])       # 0.45 < 0.5: another voice's at once
        self.assertFalse(word(module([1., 0., 0.], covered=False))['match'])     # no score: not counted

    def test_instant_counts_every_heard_word_unless_clearly_another_voice(self):
        instant = dict(provisional=True, l1=dict(policy='instant', clear_other=.2))
        low = word(module([.45, .89, 0.], **instant))
        self.assertEqual((low['match'], low['l1'], round(low['speaker_score'], 2)), (True, 'instant', .45))
        self.assertTrue(word(module([1., 0., 0.], covered=False, **instant))['match'])    # heard, no score yet
        self.assertFalse(word(module([.1, .99, 0.], **instant))['match'])      # 0.1 < clear_other: clearly another

    def test_instant_with_a_bank_excludes_only_by_a_clear_margin(self):
        options = dict(provisional=True, relative=dict(floor=.5, margin=.1),
                       l1=dict(policy='instant', clear_other=.2, clear_margin=.3))
        found = module([.8, .6, 0.], **options)
        found.bank.add_negative([.6, .8, 0.], 'B')       # 0.96 vs the person's 0.8: closer, but not by 0.3
        self.assertTrue(word(found)['match'])
        found = module([.3, .95, 0.], **options)
        found.bank.add_negative([0., 1., 0.], 'B')       # 0.95 vs 0.3
        self.assertFalse(word(found)['match'])

    def test_instant_needs_a_lane_that_settles_the_words_later(self):
        with self.assertRaises(ValueError):
            SpeakerVerify(Embedder([1., 0., 0.]), words=True, context_ms=1500, l1=dict(policy='instant'))
        with self.assertRaises(ValueError):
            SpeakerVerify(Embedder([1., 0., 0.]), words=True, context_ms=1500, provisional=True, l1=dict(policy='x'))

    def test_person_term_count_is_instant_by_default_and_the_command_line_can_restore_the_threshold(self):
        from gamst.recipes import _l1
        recipe = json.loads(Path(__file__).parents[1].joinpath('gamst/recipes/person-term-count.json').read_text())
        check = next(item for item in recipe['modules'] if item['name'] == 'speaker_check')
        self.assertEqual(check['l1']['policy'], 'instant')
        self.assertEqual(_l1(argparse.Namespace(l1='threshold')), dict(policy='threshold'))
        self.assertEqual(_l1(argparse.Namespace(l1=None, l1_clear_other=.3)), dict(policy='instant', clear_other=.3))
        self.assertIsNone(_l1(argparse.Namespace()))

    def test_the_part_passes_the_policy_only_when_verification_settles_the_words(self):
        from gamst.recipes import perception_parts

        captured = {}

        class Stand(SpeakerVerify):
            def __init__(self, embedder, **options):
                captured.update(options)
                super().__init__(embedder, **options)

        async def embedder(options, context):
            return Embedder([1., 0., 0.])
        original_embedder = perception_parts.speaker_embedder
        import gamst.providers.speaker_verify as sv
        original_class = sv.SpeakerVerify
        perception_parts.speaker_embedder, sv.SpeakerVerify = embedder, Stand
        try:
            item = dict(name='speaker_check', use='speaker-verify', words=True, context_ms=1500, segments=False,
                        consumes=['microphone', 'asr:speech.transcript_words'], l1=dict(policy='instant', clear_other=.2))
            verify = dict(name='verification', use='word-verify', consumes=['microphone', 'speaker_check'])
            asyncio.run(perception_parts._speaker_verify(item, dict(modules=[item, verify], inputs={'microphone': 1},
                                                                    l1=dict(policy='threshold'))))
            self.assertEqual(captured['l1'], dict(policy='threshold', clear_other=.2))
            captured.clear()
            asyncio.run(perception_parts._speaker_verify(item, dict(modules=[item], inputs={'microphone': 1})))
            self.assertNotIn('l1', captured)       # nothing settles them later: the threshold decides
        finally:
            perception_parts.speaker_embedder, sv.SpeakerVerify = original_embedder, original_class


class VoiceQualityTest(unittest.TestCase):
    person = [[1., .1, 0.], [.95, .2, 0.], [1., 0., .1]]
    mixed = [[1., 0., 0.], [.4, .92, 0.]]

    def value(self, embeddings, kind='main_speaker', model='stand-in'):
        return dict(kind='voice', model=model, embeddings=embeddings, enrolled_from=dict(kind=kind))

    def test_self_consistency_is_the_mean_pairwise_similarity(self):
        quality = self_consistency(self.mixed)
        self.assertEqual((quality['pieces'], quality['mean'], quality['min']), (2, .399, .399))
        self.assertGreater(self_consistency(self.person)['mean'], .9)
        self.assertIsNone(self_consistency([[1., 0., 0.]])['mean'])      # one piece: unmeasured, never low

    def test_a_good_reference_is_used_alone(self):
        value, consent, report = choose(self.value(self.person), 'c', [('other', self.value(self.mixed), 'd')])
        self.assertEqual((report['used'], report['low'], consent), ('reference', False, 'c'))
        self.assertIsNone(hud_line(report))

    def test_a_mixed_reference_is_combined_with_another_provided_reference_by_default(self):
        other = self.value([[.9, .3, 0.], [.95, .25, 0.]], kind='file')
        value, consent, report = choose(self.value(self.mixed), dict(statement='a', recorded_ms=1),
                                        [('owner.json', other, dict(statement='b', recorded_ms=2))])
        self.assertEqual((report['used'], report['fallback']['name']), ('combined', 'owner.json'))
        self.assertEqual(len(value['embeddings']), 2)       # equal weight: one centroid each
        expected = centroid([centroid(self.mixed), centroid(other['embeddings'])])
        self.assertAlmostEqual(cosine(centroid(value['embeddings']), expected), 1., places=6)
        self.assertEqual(consent, dict(statement='a + b', recorded_ms=2))
        self.assertTrue(hud_line(report).startswith('목소리 기준 품질 낮음 (0.40 < 0.50)'))

    def test_replace_mode_and_unusable_fallbacks(self):
        other = self.value(self.person, kind='file')
        value, consent, report = choose(self.value(self.mixed), 'a', [('o', other, 'b')], mode='replace')
        self.assertEqual((report['used'], value['embeddings'], consent), ('replaced', self.person, 'b'))
        value, _, report = choose(self.value(self.mixed), 'a', [('m', self.value(self.person, model='x'), 'b'),
                                                               ('low', self.value(self.mixed), 'b')])
        self.assertEqual((report['used'], value['embeddings']), ('reference_low', self.mixed))
        self.assertEqual([item['usable'] for item in report['tried']], [False, False])
        with self.assertRaises(ValueError):
            combine(self.value(self.mixed), self.value(self.person, model='x'))

    def test_the_run_checks_its_voice_file_and_the_report_says_so(self):
        from gamst.recipes import _reference_value
        with tempfile.TemporaryDirectory() as folder:
            main, owner = Path(folder, 'main.json'), Path(folder, 'owner.json')
            main.write_text(json.dumps(dict(self.value(self.mixed), consent=dict(statement='a'))))
            owner.write_text(json.dumps(dict(self.value(self.person, kind='file'), consent=dict(statement='b'))))
            args = argparse.Namespace(voice=str(main), voice_fallback=[str(owner)], voice_min_quality=None,
                                      voice_fallback_mode=None, voice_wav=None)
            value, consent = _reference_value(dict(kind='voice', name='voice'), args)
        self.assertEqual(args.voice_quality['used'], 'combined')
        self.assertEqual(args.voice_quality['origin'], 'main_speaker')
        self.assertEqual(len(value['embeddings']), 2)
        self.assertEqual(consent['statement'], 'a + b')


if __name__ == '__main__':
    unittest.main()
