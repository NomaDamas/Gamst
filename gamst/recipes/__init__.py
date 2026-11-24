"""Executable recipes: one JSON file names the GAMST parts a task needs and how they connect.

A recipe composes parts GAMST already has; it adds no runtime of its own.

  gamst recipe list
  gamst recipe show coin-flip
  gamst recipe setup coin-flip [--dry-run]   install only what the pipeline uses
  gamst recipe check coin-flip --fps 240     test a camera against the recipe
  gamst recipe run coin-flip [--iphone 9123 | --take DIR] [gamst demo rotation options]

``run`` starts ``gamst demo rotation`` with the pipeline as its options; other options pass
through and win.

Sampling limit. An object whose look repeats ``symmetry`` times per turn can be counted only while
it turns less than half a repeat between two frames, so

  max countable rev/s = fps / (2 * symmetry)

A flipping disc's outline repeats every half turn (symmetry 2): the ``fps / 4`` rule of
:mod:`gamst.providers.object_spin`. Above the limit the count would be an alias, so ``check`` and
``run`` refuse a known frame rate that cannot reach the recipe's speed, and the runtime marks such
counts ``undersampled`` instead of reporting them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from ..demos import rotation as demo

RECIPES = Path(__file__).resolve().parent
SOURCES = Path(os.environ.get('GAMST_SOURCES', Path.home() / '.cache' / 'gamst' / 'sources'))
STEPS = ('capture', 'find', 'follow', 'count')
SOURCE_REPOSITORIES = {
    'EfficientTAM': ('https://github.com/yformer/EfficientTAM', 'gamst.providers.efficienttam_stream',
                     'UPSTREAM_REVISION'),
}
# What each part needs installed; setup installs the union over one recipe's pipeline, nothing else.
PARTS = {
    ('capture', 'gamst240'): dict(extras=('rotation',)),
    ('find', 'yoloe'): dict(extras=('yoloe',), pip=('clip',), models=tuple(demo.MODEL_ASSETS)),
    ('follow', 'efficienttam'): dict(extras=('efficienttam',), models=tuple(demo.REFINE_ASSETS),
                                     sources=('EfficientTAM',)),
    ('count', 'flip'): dict(extras=('rotation',)),
}
# The pipelines the runtime runs: count part -> (find parts, follow parts) it works with.
FLOWS = {'flip': (('yoloe',), ('efficienttam',))}
SYMMETRY = {'flip': 2}


def names():
    return sorted(path.stem for path in RECIPES.glob('*.json'))


def load(name):
    """A built-in recipe by name, or a recipe file path; validated."""
    path = Path(name)
    if not (path.suffix == '.json' and path.is_file()):
        path = RECIPES / f'{name}.json'
        if not path.is_file():
            raise SystemExit(f'unknown recipe {name!r}; built-in: {", ".join(names())}')
    recipe = json.loads(path.read_text())
    steps = {}
    for item in recipe.get('pipeline', ()):
        step, use = item.get('step'), item.get('use')
        if step not in STEPS or step in steps:
            raise ValueError(f'{path}: each of {STEPS} must appear once in the pipeline')
        if (step, use) not in PARTS:
            raise ValueError(f'{path}: unknown part {use!r} for step {step!r}')
        steps[step] = item
    if set(steps) != set(STEPS):
        raise ValueError(f'{path}: pipeline needs the steps {STEPS}')
    finds, follows = FLOWS[steps['count']['use']]
    if steps['find']['use'] not in finds or steps['follow']['use'] not in follows:
        raise ValueError(f"{path}: {steps['count']['use']!r} counting runs with find {finds} and follow {follows}")
    if not steps['capture'].get('sources'):
        raise ValueError(f'{path}: capture needs its sources')
    if steps['find']['use'] == 'yoloe' and not steps['find'].get('label'):
        raise ValueError(f'{path}: find with yoloe needs a label')
    if not 0 < float(steps['count'].get('max_rev_s', 0)) <= 1000:
        raise ValueError(f'{path}: count needs max_rev_s in (0, 1000]')
    return dict(recipe, steps=steps)


def check(recipe, *, fps=None, max_rev_s=None, symmetry=None):
    """Hardware and sampling verdict for a camera frame rate (``fps=None``: what to buy)."""
    count = recipe['steps']['count']
    symmetry = SYMMETRY.get(count['use']) or symmetry or count.get('symmetry', 1)
    max_rev_s = float(max_rev_s or count['max_rev_s'])
    min_fps = (recipe.get('hardware') or {}).get('min_fps') or 0
    sampling_fps = 2 * symmetry * max_rev_s
    result = dict(recipe=recipe['name'], count=count['use'], symmetry=symmetry, max_rev_s=max_rev_s,
                  required_fps=round(max(min_fps, sampling_fps), 1))
    if fps is None:
        return result
    limit = fps / (2 * symmetry)
    # min_fps is the camera mode the recipe was measured with (a nominal 240 fps camera measures
    # ~239 fps); the sampling limit is physics and has no tolerance.
    hardware = fps >= .95 * min_fps
    sampling = limit >= max_rev_s
    problems = []
    if not hardware:
        problems.append(f'the recipe was measured at {min_fps:g} fps; this camera gives {fps:g} fps')
    if not sampling:
        problems.append(f'{fps:g} fps counts at most {limit:.1f} rev/s ({60 * limit:.0f} rpm) with symmetry '
                        f'{symmetry}; {max_rev_s:g} rev/s needs {sampling_fps:.0f} fps, or a lower --max-rev-s')
    return dict(result, fps=fps, max_countable_rev_s=round(limit, 2), ok=hardware and sampling,
                problems=problems)


def _extra_requirements(extras):
    """Requirements of the named ``gamst-sdk`` extras: the checkout's pyproject, else installed metadata."""
    pyproject = RECIPES.parents[1] / 'pyproject.toml'
    if pyproject.is_file() and sys.version_info >= (3, 11):
        import tomllib
        table = tomllib.loads(pyproject.read_text())['project']['optional-dependencies']
        found = [line for extra in extras for line in table[extra]]
    else:
        from importlib import metadata
        found = []
        for line in metadata.requires('gamst-sdk') or ():
            requirement, _, marker = line.partition(';')
            match = re.search(r'extra\s*==\s*["\']([^"\']+)["\']', marker)
            if match and match.group(1) in extras:
                found.append(requirement.strip())
    return list(dict.fromkeys(found))


def plan(recipe, models=None):
    """Everything ``setup`` installs for this recipe."""
    extras, pip, assets, sources = [], [], [], []
    for step in STEPS:
        part = PARTS[(step, recipe['steps'][step]['use'])]
        extras += part.get('extras', ())
        pip += part.get('pip', ())
        assets += part.get('models', ())
        sources += part.get('sources', ())
    extras = list(dict.fromkeys(extras))
    requirements = _extra_requirements(extras)
    if 'clip' in pip:
        from ..providers.yoloe_native import CLIP_REVISION
        requirements.append(f'git+https://github.com/ultralytics/CLIP.git@{CLIP_REVISION}')
    return dict(recipe=recipe['name'], extras=extras, pip=requirements,
                models=dict(directory=str(models or demo.DEFAULT_MODELS), files=list(dict.fromkeys(assets))),
                sources={name: str(SOURCES / name) for name in dict.fromkeys(sources)})


def _checkout(name, path):
    import importlib
    url, module, attribute = SOURCE_REPOSITORIES[name]
    revision = getattr(importlib.import_module(module), attribute)
    if not (path / '.git').is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', url, str(path)], check=True)
    head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != revision:
        subprocess.run(['git', '-C', str(path), 'fetch', '--quiet', 'origin'], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', revision], check=True)
    print(f'{name}: {path} at {revision[:7]}')


def setup(recipe, *, models=None, dry_run=False):
    steps = plan(recipe, models)
    print(json.dumps(steps, indent=2))
    if dry_run:
        return steps
    if steps['pip']:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *steps['pip']], check=True)
    if steps['models']['files']:
        demo.fetch_models(Path(steps['models']['directory']), steps['models']['files'])
    for name, path in steps['sources'].items():
        _checkout(name, Path(path))
    return steps


def _take_recording(path):
    path = Path(path)
    raw = path / 'raw.gmst' if path.is_dir() else path
    if not raw.is_file():
        raise SystemExit(f'no GAMST 240 recording at {raw}')
    return raw


def _take_fps(raw):
    summary = raw.parent / 'summary.json'
    if summary.is_file():
        fps = json.loads(summary.read_text()).get('capture_fps')
        if fps:
            return float(fps)
    from ..demos.rotation_live import read_recording
    frames = read_recording(raw)
    span = frames[-1][0].seconds - frames[0][0].seconds if len(frames) > 1 else 0
    return (len(frames) - 1) / span if span else None


def _device():
    return 'mps' if sys.platform == 'darwin' and platform.machine() == 'arm64' else 'cpu'


def demo_argv(recipe, args):
    """``gamst demo rotation`` options for this recipe and the chosen source."""
    steps = recipe['steps']
    source = next((name for name in ('iphone', 'take') if getattr(args, name, None) is not None),
                  steps['capture']['sources'][0])
    if source not in steps['capture']['sources']:
        raise SystemExit(f"{recipe['name']} captures from {', '.join(steps['capture']['sources'])}, not {source}")
    argv = ['--iphone', str(args.iphone or 9123)] if source == 'iphone' else [
        '--iphone-replay', str(_take_recording(args.take))]
    count = steps['count']
    max_rev_s = args.max_rev_s or count['max_rev_s']
    argv += ['--refine', args.label or steps['find']['label'],
             '--efficienttam-source', str(SOURCES / 'EfficientTAM'),
             '--expected-max-rev-s', f'{max_rev_s:g}', '--device', _device()]
    argv += ['--refine-on', 'button', '--refine-fast'] if count.get('on', 'analyze') == 'analyze' else [
        '--refine-on', 'stop']
    return argv


def _speed_options(parser):
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument('--max-rev-s', type=float, help="fastest turn rate to count (default: the recipe's)")
    speed.add_argument('--max-rpm', type=float, help='the same in revolutions per minute')


def _max_rev_s(args):
    return args.max_rpm / 60 if args.max_rpm else args.max_rev_s


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gamst recipe', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('list', help='built-in recipes')
    show = actions.add_parser('show', help='the recipe, what setup installs, and the frame rate it needs')
    show.add_argument('recipe')
    install = actions.add_parser('setup', help="install only the recipe's parts")
    install.add_argument('recipe')
    install.add_argument('--models', help=f'weights directory (default {demo.DEFAULT_MODELS})')
    install.add_argument('--dry-run', action='store_true', help='print the plan, install nothing')
    verify = actions.add_parser('check', help='can this camera run the recipe without aliasing?')
    verify.add_argument('recipe')
    verify.add_argument('--fps', type=float, help='camera frames per second (omit to get the required rate)')
    _speed_options(verify)
    start = actions.add_parser('run', help='run the pipeline (extra options go to gamst demo rotation)')
    start.add_argument('recipe')
    source = start.add_mutually_exclusive_group()
    source.add_argument('--iphone', type=int, metavar='PORT', help='live GAMST 240 app (default port 9123)')
    source.add_argument('--take', metavar='DIR', help='a recorded GAMST 240 take (folder or raw.gmst)')
    start.add_argument('--label', help="what to find (default: the recipe's label)")
    _speed_options(start)
    args, extra = parser.parse_known_args(argv)
    if extra and args.action != 'run':
        parser.error(f'unrecognized arguments: {" ".join(extra)}')
    if args.action == 'list':
        for name in names():
            print(f"{name:12} {load(name)['summary']}")
        return
    recipe = load(args.recipe)
    if args.action == 'show':
        print(json.dumps(dict({key: value for key, value in recipe.items() if key != 'steps'},
                              setup=plan(recipe), sampling=check(recipe)), indent=2, ensure_ascii=False))
    elif args.action == 'setup':
        setup(recipe, models=args.models, dry_run=args.dry_run)
    elif args.action == 'check':
        verdict = check(recipe, fps=args.fps, max_rev_s=_max_rev_s(args))
        print(json.dumps(verdict, indent=2))
        if verdict.get('ok') is False:
            raise SystemExit(1)
    else:
        args.max_rev_s = _max_rev_s(args)
        options = demo_argv(recipe, args) + extra
        fps = _take_fps(_take_recording(args.take)) if args.take else None
        verdict = check(recipe, fps=fps, max_rev_s=args.max_rev_s)
        print(json.dumps(dict(recipe=recipe['name'], sampling=verdict, demo=options)), flush=True)
        if verdict.get('ok') is False:
            raise SystemExit('; '.join(verdict['problems']))
        demo.main(options)
