import json
from unittest.mock import patch

import pytest

from gamst import recipes


def test_flip_sampling_limit_is_a_quarter_of_the_frame_rate():
    recipe = recipes.load('coin-flip')
    assert recipes.check(recipe, fps=240)['ok']
    verdict = recipes.check(recipe, fps=150, max_rev_s=40)
    assert not verdict['ok']
    assert verdict['max_countable_rev_s'] == 37.5
    # a coin's outline repeats every half turn whatever symmetry the caller passes
    assert recipes.check(recipe, fps=240, symmetry=5)['max_countable_rev_s'] == 60


def test_check_without_fps_names_the_camera_to_get():
    assert recipes.check(recipes.load('coin-flip'), max_rev_s=100)['required_fps'] == 400


def test_setup_installs_only_the_pipeline_parts(tmp_path):
    steps = recipes.plan(recipes.load('coin-flip'), tmp_path)
    assert steps['extras'] == ['rotation', 'yoloe', 'efficienttam']
    assert set(steps['models']['files']) == {'yoloe-26s-seg.pt', 'mobileclip2_b.ts', 'efficienttam_ti_512x512.pt'}
    assert any(line.startswith('git+https://github.com/ultralytics/CLIP.git@') for line in steps['pip'])
    assert list(steps['sources']) == ['EfficientTAM']


def test_a_pipeline_the_runtime_cannot_run_is_rejected(tmp_path):
    recipe = json.loads((recipes.RECIPES / 'coin-flip.json').read_text())
    recipe['pipeline'][2]['use'] = 'optical-flow'
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='unknown part'):
        recipes.load(str(path))


def test_run_refuses_a_take_too_slow_for_the_recipe(tmp_path):
    (tmp_path / 'raw.gmst').write_bytes(b'')
    (tmp_path / 'summary.json').write_text(json.dumps(dict(frames=60, capture_fps=60.0)))
    with patch.object(recipes.demo, 'main') as start, pytest.raises(SystemExit, match='60 fps counts at most 15.0'):
        recipes.main(['run', 'coin-flip', '--take', str(tmp_path)])
    start.assert_not_called()


def test_run_passes_the_pipeline_and_lets_extra_options_win(tmp_path):
    (tmp_path / 'raw.gmst').write_bytes(b'')
    (tmp_path / 'summary.json').write_text(json.dumps(dict(frames=900, capture_fps=239.6)))
    with patch.object(recipes.demo, 'main') as start:
        recipes.main(['run', 'coin-flip', '--take', str(tmp_path), '--label', 'gold coin', '--device', 'cpu'])
    argv = start.call_args.args[0]
    assert argv[argv.index('--iphone-replay') + 1] == str(tmp_path / 'raw.gmst')
    assert argv[argv.index('--refine') + 1] == 'gold coin'
    assert argv[-2:] == ['--device', 'cpu']
