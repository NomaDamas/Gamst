import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gamst import recipes


class RecipeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def take(self, fps):
        (self.tmp / 'raw.gmst').write_bytes(b'')
        (self.tmp / 'summary.json').write_text(json.dumps(dict(frames=900, capture_fps=fps)))
        return str(self.tmp)

    def test_flip_sampling_limit_is_a_quarter_of_the_frame_rate(self):
        recipe = recipes.load('coin-flip')
        self.assertTrue(recipes.check(recipe, fps=240)['ok'])
        verdict = recipes.check(recipe, fps=150, max_rev_s=40)
        self.assertFalse(verdict['ok'])
        self.assertEqual(verdict['max_countable_rev_s'], 37.5)
        # a coin's outline repeats every half turn whatever symmetry the caller passes
        self.assertEqual(recipes.check(recipe, fps=240, symmetry=5)['max_countable_rev_s'], 60)

    def test_check_without_fps_names_the_camera_to_get(self):
        self.assertEqual(recipes.check(recipes.load('coin-flip'), max_rev_s=100)['required_fps'], 400)

    def test_setup_installs_only_the_pipeline_parts(self):
        steps = recipes.plan(recipes.load('coin-flip'), self.tmp)
        self.assertEqual(steps['extras'], ['rotation', 'yoloe', 'efficienttam'])
        self.assertEqual(set(steps['models']['files']),
                         {'yoloe-26s-seg.pt', 'mobileclip2_b.ts', 'efficienttam_ti_512x512.pt'})
        self.assertTrue(any(line.startswith('git+https://github.com/ultralytics/CLIP.git@') for line in steps['pip']))
        self.assertEqual(list(steps['sources']), ['EfficientTAM'])

    def test_a_pipeline_the_runtime_cannot_run_is_rejected(self):
        recipe = json.loads((recipes.RECIPES / 'coin-flip.json').read_text())
        recipe['pipeline'][2]['use'] = 'optical-flow'
        path = self.tmp / 'bad.json'
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'unknown part'):
            recipes.load(str(path))

    def test_run_refuses_a_take_too_slow_for_the_recipe(self):
        with patch.object(recipes.demo, 'main') as start, \
                self.assertRaisesRegex(SystemExit, '60 fps counts at most 15.0'):
            recipes.main(['run', 'coin-flip', '--take', self.take(60.0)])
        start.assert_not_called()

    def test_run_passes_the_pipeline_and_lets_extra_options_win(self):
        take = self.take(239.6)
        with patch.object(recipes.demo, 'main') as start:
            recipes.main(['run', 'coin-flip', '--take', take, '--label', 'gold coin', '--device', 'cpu'])
        argv = start.call_args.args[0]
        self.assertEqual(argv[argv.index('--iphone-replay') + 1], str(Path(take) / 'raw.gmst'))
        self.assertEqual(argv[argv.index('--refine') + 1], 'gold coin')
        self.assertEqual(argv[-2:], ['--device', 'cpu'])

    def test_blade_symmetry_lowers_the_countable_speed(self):
        recipe = recipes.load('rotation')
        verdict = recipes.check(recipe, fps=240, symmetry=3, max_rev_s=25)  # a 3-blade fan at 1500 rpm
        self.assertTrue(verdict['ok'])
        self.assertEqual(verdict['max_countable_rev_s'], 40)
        self.assertEqual(recipes.check(recipe, symmetry=3, max_rev_s=25)['required_fps'], 150)
        self.assertFalse(recipes.check(recipe, fps=60, symmetry=3, max_rev_s=25)['ok'])

    def test_rotation_setup_needs_no_models(self):
        steps = recipes.plan(recipes.load('rotation'), self.tmp)
        self.assertEqual(steps['extras'], ['rotation'])
        self.assertEqual(steps['models']['files'], [])
        self.assertEqual(steps['sources'], {})

    def test_rotation_run_counts_inside_the_given_box_with_the_blade_count(self):
        take = self.take(240.)
        with patch.object(recipes.demo, 'main') as start:
            with self.assertRaisesRegex(SystemExit, '--roi'):
                recipes.main(['run', 'rotation', '--take', take])
            recipes.main(['run', 'rotation', '--take', take, '--symmetry', '3', '--max-rpm', '1500',
                          '--roi', '.2,.2,.6,.6'])
        argv = start.call_args.args[0]
        self.assertEqual(argv[argv.index('--take') + 1], str(Path(take) / 'raw.gmst'))
        self.assertEqual(argv[argv.index('--in-plane') + 1], 'polar')
        self.assertEqual(argv[argv.index('--symmetry') + 1], '3')
        self.assertEqual(argv[argv.index('--expected-max-rev-s') + 1], '25')


if __name__ == '__main__':
    unittest.main()
