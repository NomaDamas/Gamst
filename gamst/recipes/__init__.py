"""Executable recipes: one JSON file names the inputs, perception modules and view a task needs.

A recipe composes parts GAMST already has (``gamst/recipes/parts.py``) into a
:class:`gamst.pipeline.Pipeline`; it adds no runtime of its own.

  gamst recipe list
  gamst recipe show person-action-count
  gamst recipe setup person-action-count [--dry-run]     install only what the recipe uses (gamst setup's apply)
  gamst recipe check rotation --fps 240 --symmetry 3     test a camera against the sampling limit
  gamst recipe enroll person-action-count --video demo.mp4 --start 2 --seconds 1.5 --out shotgun.json
  gamst recipe run person-action-count --camera 0 --exemplar shotgun.json
  gamst recipe run rotation --take DIR --roi .3,.2,.4,.6 --symmetry 3
  gamst recipe enroll person-term-count --kind voice --microphone --seconds 20 --consent '...' --out me.json
  gamst recipe run person-term-count --microphone --voice me.json --terms 대박 [--truth truth.json]
  gamst recipe run coin-flip --iphone 9123 | --take DIR

Recipe file::

  {"name", "summary", "skill", "hardware": {"min_fps", "camera", "compute"},
   "inputs": [{"id": "camera", "use": "camera", "sources": ["camera", "video", "take"]}],
   "references": [{"name": "exemplar", "kind": "exemplar"}],        user-defined inputs
   "modules": [{"name": "pose", "use": "mediapipe-pose", "consumes": ["camera"], ...options},
               {"name": "find", "use": "yoloe", "in": "count"}],     a stage that runs inside "count"
   (a module may be "optional": "on" | "off"; --with / --without choose it)
   "view": "count" | "rotation-view" | "halo-coin",
   "records": ["temporal.exemplar_events"], "measured": "..."}

``run`` builds the Pipeline, feeds it the chosen source and shows the view; agents read the same
records over MCP (``get_observations``, ``count_events``) and switch modules with
``configure_pipeline_module``. ``coin-flip`` is the GAMST 240 app: the ``gamst240`` input (the
iPhone, or a take replayed), a take recorder, the after-take flip counter (YOLOE and EfficientTAM
are its stages, in its own process) and the ``halo-coin`` HUD.

Sampling limit. A part's ``sampling`` rule says which frame rate counts its target without
aliasing, e.g. an object whose look repeats ``symmetry`` times per turn is countable only below
``fps / (2 * symmetry)`` rev/s, and a movement repeated r times per second needs more than 2r
frames per second. ``check`` and ``run`` refuse a known frame rate below the limit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import platform
import re
import sys

from ..demos import rotation as demo
from .parts import PARTS

RECIPES = Path(__file__).resolve().parent
SOURCES_BY_INPUT = {'camera': ('camera', 'video', 'take'), 'gamst240': ('iphone', 'take'),
                    'microphone': ('microphone', 'audio')}
REFERENCE_KINDS = ('voice', 'person', 'face', 'body', 'exemplar', 'terms', 'object')


def names():
    return sorted(path.stem for path in RECIPES.glob('*.json'))


def choose(recipe, include=(), exclude=()):
    """Resolve ``"optional": "on" | "off"`` modules (``--with`` / ``--without``).

    A module may name what it reads when an optional one is absent (``consumes_without``), and a
    reference what it is for (``for``); both follow the choice.
    """
    optional = {item['name']: item['optional'] for item in recipe.get('modules', ()) if item.get('optional')}
    unknown = (set(include) | set(exclude)) - set(optional)
    if unknown:
        raise SystemExit(f"{recipe['name']} has no optional module {', '.join(sorted(unknown))}; "
                         f"optional: {', '.join(sorted(optional)) or 'none'}")
    absent = {name for name, state in optional.items()
              if name in exclude or (state == 'off' and name not in include)}
    modules = []
    for item in recipe.get('modules', ()):
        if item['name'] in absent:
            continue
        for name in absent & set(item.get('consumes_without', {})):
            item = dict(item, consumes=item['consumes_without'][name])
        modules.append(item)
    references = [item for item in recipe.get('references', ())
                  if item.get('for') is None or item['for'] not in absent]
    return dict(recipe, modules=modules, references=references) if absent else recipe


def load(name, include=(), exclude=()):
    """A built-in recipe by name, or a recipe file path; validated, optional modules resolved."""
    path = Path(name)
    if not (path.suffix == '.json' and path.is_file()):
        path = RECIPES / f'{name}.json'
        if not path.is_file():
            raise SystemExit(f'unknown recipe {name!r}; built-in: {", ".join(names())}')
    recipe = choose(json.loads(path.read_text()), include, exclude)
    inputs = recipe.get('inputs') or []
    if len(inputs) != 1:
        raise ValueError(f'{path}: a recipe has one primary input (derived inputs come from modules)')
    for item in inputs:
        part = PARTS.get(item.get('use'))
        if part is None or part.kind != 'input':
            raise ValueError(f'{path}: unknown input part {item.get("use")!r}')
        supported = SOURCES_BY_INPUT.get(item['use'], (item['use'],))
        if not item.get('sources') or set(item['sources']) - set(supported):
            raise ValueError(f"{path}: input {item['use']!r} reads from {supported}")
    known = {item['id'] for item in inputs}
    for reference in recipe.get('references', ()):
        if reference.get('kind') not in REFERENCE_KINDS:
            raise ValueError(f'{path}: reference kind must be one of {REFERENCE_KINDS}')
    for item in recipe.get('modules', ()):
        part = PARTS.get(item.get('use'))
        if part is None or part.kind != 'module':
            raise ValueError(f'{path}: unknown part {item.get("use")!r}')
        if item.get('in') and item['in'] not in {other['name'] for other in recipe['modules']}:
            raise ValueError(f"{path}: module {item['name']!r} runs in {item['in']!r}, which is not a module")
        for source in item.get('consumes', ()):
            if source not in known and source.split(':', 1)[0] not in known:
                raise ValueError(f"{path}: module {item['name']!r} consumes {source!r}, which no input or "
                                 'earlier module provides')
        known.add(item['name'])
        known.update(item.get('emits', ()))
    view = PARTS.get(recipe.get('view'))
    if view is None or view.kind != 'view':
        raise ValueError(f'{path}: unknown view {recipe.get("view")!r}')
    return recipe


def _sampled(recipe):
    return [item for item in recipe.get('modules', ()) if PARTS[item['use']].sampling is not None]


def check(recipe, *, fps=None, max_rev_s=None, symmetry=None, max_reps_s=None):
    """Hardware and sampling verdict for a camera frame rate (``fps=None``: what to buy)."""
    min_fps = (recipe.get('hardware') or {}).get('min_fps') or 0
    result = dict(recipe=recipe['name'])
    required, problems, ok = min_fps, [], True
    for item in _sampled(recipe):
        verdict = PARTS[item['use']].sampling(item, fps, symmetry=symmetry, max_rev_s=max_rev_s,
                                              max_reps_s=max_reps_s)
        result.update({key: value for key, value in verdict.items() if key not in ('ok', 'problem')},
                      count=item['use'])
        required = max(required, verdict['required_fps'])
        if verdict.get('ok') is False:
            ok = False
            problems.append(verdict['problem'])
    result['required_fps'] = round(required, 1)
    if fps is None:
        return result
    # min_fps is the camera mode the recipe was measured with (a nominal 240 fps camera measures
    # ~239 fps); the sampling limit is physics and has no tolerance.
    if fps < .95 * min_fps:
        ok = False
        problems.insert(0, f'the recipe was measured at {min_fps:g} fps; this camera gives {fps:g} fps')
    return dict(result, fps=fps, ok=ok, problems=problems)


def _items(recipe):
    """(item, part) for the recipe's input, modules and view, in order."""
    items = list(recipe['inputs']) + list(recipe.get('modules', ())) + [dict(use=recipe['view'])]
    return [(item, PARTS[item['use']]) for item in items]


def _parts(recipe):
    return [part for _, part in _items(recipe)]


def _role(item):
    part = PARTS[item['use']]
    return part.role, item.get('candidate', part.candidate)


def roles(recipe):
    """``{role: candidate}`` of the role registry that the recipe's parts need."""
    selected = {}
    for item, part in _items(recipe):
        if part.role is None:
            continue
        role, model = _role(item)
        if selected.setdefault(role, model) != model:
            raise ValueError(f"{recipe['name']}: {role} is both {selected[role]} and {model}")
    return selected


def plan(recipe, models=None, hardware=None):
    """The setup plan for exactly the recipe's parts (``gamst.crew.setup.make_plan``; installs nothing).

    Packages go into this interpreter (recipe modules run in-process or as workers of it), and
    every planned weight goes into one directory, ``models`` (default: the one ``gamst demo
    rotation`` reads), so the coin-flip view finds the detector and the refiner side by side.
    """
    from ..crew.setup import make_plan
    steps = make_plan(roles=roles(recipe), dependencies=False, isolation='current', hardware=hardware)
    steps['recipe'] = recipe['name']
    steps['models_dir'] = str(Path(models or demo.DEFAULT_MODELS).expanduser().absolute())
    return steps


def weight_files(steps):
    return [weight['name'] for selection in steps['roles'].values() for weight in selection['weights']]


def setup(recipe, *, models=None, dry_run=False, prefix=None, profile=None):
    """Install only the recipe's parts with the same apply as ``gamst setup``; writes the profile."""
    from ..crew.setup import SetupError, apply_plan
    steps = plan(recipe, models)
    print(json.dumps(steps, indent=2))
    if dry_run:
        return steps
    try:
        result = apply_plan(steps, prefix=prefix, profile_path=profile, models_dir=steps['models_dir'])
    except SetupError as error:
        raise SystemExit(f'setup: {error}') from error
    print(json.dumps(result, indent=2))
    return result


def installed(recipe, item, profile=None):
    """The profile entry setup wrote for this module's role and candidate."""
    from ..profile import read_profile
    role, model = _role(item)
    try:
        entry = read_profile(profile)['roles'].get(role)
    except FileNotFoundError:
        entry = None
    if entry is None or entry.get('model') != model:
        raise SystemExit(f"{role} {model} is not set up; run: gamst recipe setup {recipe['name']}")
    return entry


# ---------------------------------------------------------------- run

def _take_recording(path):
    path = Path(path)
    raw = path / 'raw.gmst' if path.is_dir() else path
    if not raw.is_file():
        raise SystemExit(f'no GAMST 240 recording at {raw}')
    return raw


def _take_fps(raw):
    from ..inputs import TakeInput
    return TakeInput(raw).fps()


def _source_fps(args):
    """The input's frame rate when it is known before the run (a take or a file), else None."""
    if getattr(args, 'take', None):
        return _take_fps(_take_recording(args.take))
    if getattr(args, 'video', None):
        if args.capture_fps:
            return args.capture_fps
        import cv2
        video = cv2.VideoCapture(args.video)
        fps = video.get(cv2.CAP_PROP_FPS) if video.isOpened() else 0
        video.release()
        return fps or None
    return None


def _device():
    return 'mps' if sys.platform == 'darwin' and platform.machine() == 'arm64' else 'cpu'


def _source(recipe, args):
    item = recipe['inputs'][0]
    chosen = next((name for name in ('iphone', 'take', 'camera', 'video', 'audio', 'microphone')
                   if getattr(args, name, None) not in (None, False)), item['sources'][0])
    if chosen not in item['sources']:
        raise SystemExit(f"{recipe['name']} reads from {', '.join(item['sources'])}, not {chosen}")
    return item, chosen


def from_rotation_demo(args):
    """``gamst demo rotation --iphone/--iphone-replay`` options -> (the coin-flip recipe, its run options).

    The demo keeps its own defaults: count each take on Stop (``--refine-on button``: on ANALYZE),
    one process per take that counts every throw unless ``--refine-fast``, and no counting at all
    without ``--refine`` (a camera monitor that records takes).
    """
    recipe = load('coin-flip')
    if not args.refine:
        recipe = dict(recipe, modules=[item for item in recipe['modules'] if item['use'] == 'take-recorder'],
                      records=[])
    options = argparse.Namespace(
        iphone=args.iphone, take=args.iphone_replay, label=args.refine, max_rev_s=args.expected_max_rev_s,
        device=args.device, models=args.models, worker_python=args.worker_python, store=args.store,
        headless=args.headless, report=args.report, record_dir=args.record_dir, refine_show=args.refine_show,
        refine_cached=args.refine_cached, efficienttam_source=args.efficienttam_source,
        count_on='analyze' if args.refine_on == 'button' else 'stop', fast=args.refine_fast)
    return recipe, options


def _reference_value(reference, args):
    """(value, consent record or None) of one user-defined input from its command-line option."""
    kind, name = reference['kind'], reference['name']
    if kind in ('voice', 'person'):
        from .enrollment import saved_reference
        path = getattr(args, kind, None)
        if not path:
            raise SystemExit(f"this recipe needs an enrolled {kind}: run 'gamst recipe enroll --kind {kind}' and "
                             f'pass --{kind} FILE ({name})')
        value, consent = saved_reference(path, kind)
        if kind == 'voice' and getattr(args, 'voice_wav', None):
            value['wav'] = args.voice_wav   # extraction models condition on the clip itself
        return value, consent
    if kind == 'exemplar':
        if not args.exemplar:
            raise SystemExit(f"this recipe needs a demonstrated exemplar: run 'gamst recipe enroll' and pass "
                             f'--exemplar FILE ({name})')
        saved = json.loads(Path(args.exemplar).read_text())
        return (saved['value'] if isinstance(saved, dict) and 'value' in saved else saved), None
    if kind == 'terms':
        if not args.terms:
            raise SystemExit(f'this recipe needs --terms word1,word2 ({name})')
        return [item.strip() for item in args.terms.split(',') if item.strip()], None
    raise SystemExit(f'reference kind {kind!r} is not enrollable from the command line yet')


async def build_pipeline(recipe, args, human, *, source=None, modules=None):
    """The recipe's Pipeline on ``human``: input, user-defined references, modules (no view)."""
    from ..inputs import open_source
    from ..pipeline import Pipeline
    pipeline = Pipeline(human, re.sub(r'[^A-Za-z0-9_.:-]', '-', recipe['name']))
    item, chosen = _source(recipe, args)
    if source is None:
        value = dict(camera=getattr(args, 'camera', None), video=getattr(args, 'video', None),
                     take=getattr(args, 'take', None), audio=getattr(args, 'audio', None),
                     iphone=getattr(args, 'iphone', None)).get(chosen)
        options = dict(capture_fps=args.capture_fps) if chosen == 'video' and args.capture_fps else {}
        if getattr(args, 'frames', None) and chosen in ('camera', 'video', 'take'):
            options['max_frames'] = args.frames
        # A GAMST 240 take replays through the phone's input, so its Start/Stop marks stay.
        kind = 'gamst240-take' if item['use'] == 'gamst240' and chosen == 'take' else chosen
        source = open_source(kind, value, **options)
    pipeline.add_input(item['id'], source)
    for reference in recipe.get('references', ()):
        value, consent = _reference_value(reference, args)
        pipeline.add_reference(reference['name'], reference['kind'], value, label=reference.get('label'),
                               consent=consent)
    context = dict(human=human, labels=args.label.split(',') if getattr(args, 'label', None) else None,
                   roi=getattr(args, 'roi', None), symmetry=getattr(args, 'symmetry', None),
                   max_rev_s=getattr(args, 'max_rev_s', None), max_reps_s=getattr(args, 'max_reps_s', None),
                   device=getattr(args, 'device', None) or _device(),
                   worker_python=getattr(args, 'worker_python', None),
                   installed=lambda item: installed(recipe, item, getattr(args, 'profile', None)),
                   models=getattr(args, 'models', None), enter=getattr(args, 'enter', None),
                   exit=getattr(args, 'exit', None), language=getattr(args, 'language', None),
                   speaker_threshold=getattr(args, 'speaker_threshold', None), inputs={item['id']: source},
                   record_dir=getattr(args, 'record_dir', None), replay=chosen == 'take',
                   efficienttam_source=getattr(args, 'efficienttam_source', None),
                   count_on=getattr(args, 'count_on', None), fast=getattr(args, 'fast', None),
                   show=getattr(args, 'refine_show', False), cached=getattr(args, 'refine_cached', None))
    modules = modules or recipe['modules']
    context['stages'] = {item['name']: item for item in modules if item.get('in')}
    for item in modules:
        if item.get('in'):
            continue  # runs inside the module it names (its build reads context['stages'])
        build = PARTS[item['use']].build
        if build is None:
            raise SystemExit(f"part {item['use']!r} runs inside the {recipe['view']} view")
        try:
            built = await build(item, context)
            pipeline.add_module(item['name'], built['module'], consumes=item['consumes'], spec=built.get('spec'),
                                every_ms=built.get('every_ms'), queue=built.get('queue'),
                                emits=tuple(item.get('emits', ())), enabled=item.get('enabled', True))
        except BaseException:
            await pipeline.close()   # workers of the modules already built
            raise
    return pipeline


def _view_class(name):
    if name == 'halo-coin':
        from ..halo import HaloCoinView
        return HaloCoinView
    from ..views import CountView, RotationView
    return RotationView if name == 'rotation-view' else CountView


async def run_pipeline(recipe, args):
    from .. import Gamst
    report = dict(recipe=recipe['name'], frames=0)
    async with Gamst(consent=True, store=args.store) as human:
        pipeline = await build_pipeline(recipe, args, human)
        view, idle = None, None
        try:
            view = _view_class(recipe['view'])(pipeline, headless=args.headless,
                                               capabilities=recipe.get('records', ()))
        except BaseException:
            await pipeline.close()
            raise

        async def on_frame(input_id, frame, records):
            report['frames'] += 1
            report.setdefault('start_ms', frame.timestamp_ms - (
                round(1000 * len(frame.data) / frame.sample_rate) if frame.sample_rate else 0))
            return await view.update(input_id, frame, records)

        if callable(getattr(view, 'idle', None)):
            idle = asyncio.create_task(view.idle())  # an interactive view stays drawn between frames
        try:
            await pipeline.run(on_frame=on_frame)
            report['counts'], report['events'] = {}, {}
            for capability in recipe.get('records', ()):
                counted = await human.count_events(capability, window='24h', group_by='@event', max_evidence=100)
                report['counts'][capability] = counted['groups']
                report['events'][capability] = [item['event'] for item in counted['evidence']]
            if getattr(args, 'truth', None):
                from ..engine import timestamp
                from ..scoring import score
                truth, start_ms = json.loads(Path(args.truth).read_text()), report.get('start_ms', 0)
                rows = human.store.query('local-user', start_ms, timestamp(), 'observation.signal', limit=1_000_000,
                                         registered_only=True, capability_id=truth['capability'])
                found = [event for row in rows for event in row['observation']['value'].get('events') or ()]
                report['score'] = score(truth, found, start_ms=start_ms)
            report['pipeline'] = pipeline.status()
        finally:
            if idle is not None:
                idle.cancel()
            view.close()
            await pipeline.close()
    print(json.dumps({key: report[key] for key in ('recipe', 'frames', 'counts', 'score') if key in report},
                     ensure_ascii=False), flush=True)
    if getattr(args, 'report', None):
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


async def enroll(recipe, args):
    """Record the owner's exemplar: feature channels over [start, start+seconds] of the chosen input."""
    from .. import Gamst
    kind = getattr(args, 'kind', None) or 'exemplar'
    if kind != 'exemplar':
        return await _enroll_identity(recipe, args, kind)
    exemplar = next((item for item in recipe['modules'] if item['use'] == 'exemplar'), None)
    if exemplar is None:
        raise SystemExit(f"{recipe['name']} has no exemplar module to enroll for")
    # The demonstration itself is the only person or voice in it, so identity modules are not used here.
    upstream = [dict(item, consumes=[recipe['inputs'][0]['id']]) if item['use'] == 'prosody' else item
                for item in choose(json.loads(json.dumps(recipe)), exclude=[
                    item['name'] for item in recipe['modules'] if item.get('optional')])['modules']
                if item['use'] in ('mediapipe-pose', 'pose-channels', 'prosody', 'voice-channels')]
    mode = args.mode or exemplar.get('mode', 'hold')
    samples = []
    async with Gamst(consent=True) as human:
        pipeline = await build_pipeline(dict(recipe, references=[]), args, human, modules=upstream)
        first = {}

        async def on_frame(input_id, frame, records):
            first.setdefault('ms', frame.timestamp_ms)
            seconds = (frame.timestamp_ms - first['ms']) / 1000
            if seconds > args.start + args.seconds:
                return False
            if seconds >= args.start:
                for record in records:
                    capability = (record['observation'].get('capability') or {}).get('id')
                    if capability in ('vision.pose_channels', 'audio.voice_channels'):
                        samples.append(record['observation']['value'])
            return True

        try:
            await pipeline.run(on_frame=on_frame)
        finally:
            await pipeline.close()
    if not samples:
        raise SystemExit('nothing to learn in the enrollment window (no visible person, or no audio)')
    if mode == 'hold':
        size = len(samples[0]['channels'])
        mask = [sum(item['mask'][i] for item in samples) * 2 > len(samples) for i in range(size)]
        channels = []
        for i in range(size):
            values = sorted(item['channels'][i] for item in samples if item['mask'][i])
            channels.append(values[len(values) // 2] if values and mask[i] else 0.)
        value = dict(channels=channels, mask=mask)
    else:
        step = max(1, len(samples) // 30)
        value = [dict(channels=item['channels'], mask=item['mask']) for item in samples[::step]]
    saved = dict(kind='exemplar', recipe=recipe['name'], mode=mode, frames=len(samples),
                 window_s=[args.start, args.start + args.seconds], value=value)
    Path(args.out).write_text(json.dumps(saved))
    print(json.dumps(dict(out=args.out, mode=mode, frames=len(samples))), flush=True)
    return saved


async def _enroll_identity(recipe, args, kind):
    from . import enrollment
    context = dict(installed=lambda item: installed(recipe, item, getattr(args, 'profile', None)))
    if kind == 'person':
        return await enrollment.enroll_person(recipe, args, build_pipeline=build_pipeline, context=context)
    from ..inputs import open_source
    _, chosen = _source(recipe, args)
    return await enrollment.enroll_voice(recipe, args, context=context,
                                         source=open_source(chosen, getattr(args, 'audio', None)))


# ---------------------------------------------------------------- command line

def _optional_options(parser):
    parser.add_argument('--with', dest='include', action='append', default=[], metavar='MODULE',
                        help='add an optional module of the recipe')
    parser.add_argument('--without', dest='exclude', action='append', default=[], metavar='MODULE',
                        help='leave an optional module out')



def _speed_options(parser):
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument('--max-rev-s', type=float, help="fastest turn rate to count (default: the recipe's)")
    speed.add_argument('--max-rpm', type=float, help='the same in revolutions per minute')
    parser.add_argument('--symmetry', type=int, choices=range(1, 65), metavar='N',
                        help='times the object looks the same per in-plane turn, e.g. blades of a fan '
                             "(default: the recipe's)")
    parser.add_argument('--max-reps-s', type=float, help="fastest repeated movement to count, per second")


def _source_options(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--iphone', type=int, metavar='PORT', help='live GAMST 240 app (default port 9123)')
    source.add_argument('--take', metavar='DIR', help='a recorded GAMST 240 take (folder or raw.gmst)')
    source.add_argument('--camera', type=int, metavar='INDEX', help='a local camera')
    source.add_argument('--video', metavar='FILE', help='a video file (add --capture-fps for high-speed files)')
    source.add_argument('--audio', metavar='FILE', help='a PCM WAV file')
    source.add_argument('--microphone', action='store_true', help='the default microphone')
    parser.add_argument('--capture-fps', type=float, help='override file timing for high-fps captures')
    parser.add_argument('--frames', type=int, help='stop after this many frames')
    parser.add_argument('--models', help='YOLOE/EfficientTAM weights directory')
    parser.add_argument('--worker-python', help='python for model worker processes (default: this one)')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--label', help="what to find (default: the recipe's label)")
    parser.add_argument('--roi', type=lambda value: [float(item) for item in value.split(',')],
                        help='normalized x,y,width,height')
    parser.add_argument('--profile', help='setup profile to read (default ~/.config/gamst/profile.json)')


def _max_rev_s(args):
    return args.max_rpm / 60 if getattr(args, 'max_rpm', None) else getattr(args, 'max_rev_s', None)


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
    install.add_argument('--prefix', help='setup prefix for source checkouts (default ~/.local/share/gamst)')
    install.add_argument('--profile', help='profile to write (default ~/.config/gamst/profile.json)')
    verify = actions.add_parser('check', help='can this camera run the recipe without aliasing?')
    verify.add_argument('recipe')
    verify.add_argument('--fps', type=float, help='camera frames per second (omit to get the required rate)')
    _speed_options(verify)
    learn = actions.add_parser('enroll', help="record a user-defined exemplar (a demonstrated pose or movement)")
    learn.add_argument('recipe')
    _source_options(learn)
    learn.add_argument('--start', type=float, default=0., help='seconds into the input where the demonstration starts')
    learn.add_argument('--seconds', type=float, default=1.5, help='length of the demonstration')
    learn.add_argument('--mode', choices=['hold', 'sequence'], help="hold: one pose; sequence: a movement")
    learn.add_argument('--kind', choices=['exemplar', 'voice', 'person'], default='exemplar',
                       help='voice / person: save embeddings only (needs --consent)')
    learn.add_argument('--consent', help='who consented to being enrolled, and to what')
    learn.add_argument('--out', required=True, help='enrollment JSON to write')
    start = actions.add_parser('run', help='run the recipe')
    start.add_argument('recipe')
    _source_options(start)
    _speed_options(start)
    start.add_argument('--exemplar', help='exemplar JSON from gamst recipe enroll')
    start.add_argument('--terms', help='comma-separated words to count (re:PATTERN for a pattern)')
    start.add_argument('--voice', help='voice JSON from gamst recipe enroll --kind voice')
    start.add_argument('--voice-wav', help="the enrolled person's clip, for separation models only")
    start.add_argument('--person', help='person JSON from gamst recipe enroll --kind person')
    start.add_argument('--speaker-threshold', type=float, help='cosine score that counts as the enrolled voice')
    start.add_argument('--language', help="spoken language, e.g. ko (default: the recipe's)")
    start.add_argument('--truth', help='ground-truth JSON to score the run against (gamst/scoring.py)')
    start.add_argument('--enter', type=float, help='exemplar match threshold (mean channel difference)')
    start.add_argument('--exit', type=float, help='exemplar release threshold')
    start.add_argument('--store', default=':memory:', help='SQLite store for the records (default: memory)')
    start.add_argument('--headless', action='store_true', help='no window; events as JSON lines')
    start.add_argument('--report', help='write frames, counts and pipeline status as JSON')
    for command in (show, install, learn, start):
        _optional_options(command)
    takes = start.add_argument_group('GAMST 240 takes (coin-flip)')
    takes.add_argument('--record-dir', help='one folder per take: raw.gmst, hud.mp4, refined.json/mp4 '
                                            '(default gamst-takes)')
    takes.add_argument('--refine-show', action='store_true',
                       help='open refined.mp4 (8x slow motion with the running count) when it is ready')
    takes.add_argument('--refine-cached', metavar='TAKE_DIR',
                       help='stage backup: ANALYZE shows this already counted take at once')
    takes.add_argument('--all-throws', action='store_true',
                       help='count every throw of a take, one process per take (default: the best-named throw, '
                            'models kept loaded)')
    args = parser.parse_args(argv)
    if getattr(args, 'all_throws', False):
        args.fast = False
    if args.action == 'list':
        for name in names():
            print(f"{name:20} {load(name)['summary']}")
        return
    recipe = load(args.recipe, getattr(args, 'include', ()), getattr(args, 'exclude', ()))
    if args.action == 'show':
        print(json.dumps(dict(recipe, setup=plan(recipe), sampling=check(recipe)), indent=2, ensure_ascii=False))
    elif args.action == 'setup':
        setup(recipe, models=args.models, dry_run=args.dry_run, prefix=args.prefix, profile=args.profile)
    elif args.action == 'check':
        verdict = check(recipe, fps=args.fps, max_rev_s=_max_rev_s(args), symmetry=args.symmetry,
                        max_reps_s=args.max_reps_s)
        print(json.dumps(verdict, indent=2))
        if verdict.get('ok') is False:
            raise SystemExit(1)
    elif args.action == 'enroll':
        asyncio.run(enroll(recipe, args))
    else:
        args.max_rev_s = _max_rev_s(args)
        verdict = check(recipe, fps=_source_fps(args), max_rev_s=args.max_rev_s, symmetry=args.symmetry,
                        max_reps_s=args.max_reps_s)
        if verdict.get('ok') is False:
            raise SystemExit('; '.join(verdict['problems']))
        print(json.dumps(dict(recipe=recipe['name'], sampling=verdict)), flush=True)
        return asyncio.run(run_pipeline(recipe, args))
