"""The parts a recipe can name: input modules, perception modules and views.

Each part declares how to build its Pipeline module, its sampling rule (what frame rate the input
needs so the count is not an alias) and the setup role it needs: ``role`` and ``candidate`` name an
entry of the one GAMST role registry (:mod:`gamst.crew.registry`), which owns the packages, the
pinned weight URLs and their SHA256. ``gamst recipe setup`` plans exactly those roles and runs the
same :func:`gamst.crew.setup.apply_plan` as ``gamst setup``; a module item may pick another
candidate of its part's role with ``"candidate"``. Adding a part here is how a new capability
becomes usable from recipes; the Pipeline, the records and the MCP context stay the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable


@dataclass
class Part:
    kind: str                      # 'input' | 'module' | 'view' | 'plugin' (an input or a module)
    role: str | None = None        # registry role whose packages/weights setup installs
    candidate: str | None = None   # default registry candidate of that role
    build: Callable[..., Any] | None = None       # async (options, context) -> dict(module, spec?, every_ms?)
    sampling: Callable[..., Any] | None = None    # (options, fps) -> dict(required_fps, ...) or None
    transcribes: bool = False                     # commits utterances with text and times
    # Further roles the builder reads: (item key naming the candidate, role, default candidate).
    extra_roles: tuple = ()


# ---------------------------------------------------------------- sampling rules (physics, no tuning)

def rotation_sampling(options, fps, *, symmetry=None, max_rev_s=None, **_):
    """Countable only below fps / (2 x symmetry) rev/s; a flipping disc repeats every half turn."""
    symmetry = 2 if options.get('use') == 'flip' else symmetry or options.get('symmetry', 1)
    max_rev_s = float(max_rev_s or options['max_rev_s'])
    result = dict(symmetry=symmetry, max_rev_s=max_rev_s, required_fps=2 * symmetry * max_rev_s)
    if fps:
        limit = fps / (2 * symmetry)
        result.update(max_countable_rev_s=round(limit, 2), ok=limit >= max_rev_s,
                      problem=None if limit >= max_rev_s else
                      f'{fps:g} fps counts at most {limit:.1f} rev/s ({60 * limit:.0f} rpm) with symmetry '
                      f'{symmetry}; {max_rev_s:g} rev/s needs {2 * symmetry * max_rev_s:.0f} fps, or a lower --max-rev-s')
    return result


def movement_sampling(options, fps, *, max_reps_s=None, **_):
    """A movement repeated r times per second is resolved only above 2r samples per second (Nyquist)."""
    rate = float(max_reps_s or options.get('max_reps_s', 2))
    result = dict(max_reps_s=rate, required_fps=2 * rate)
    if fps:
        ok = fps / 2 >= rate
        result.update(max_countable_reps_s=round(fps / 2, 2), ok=ok,
                      problem=None if ok else f'{fps:g} fps resolves at most {fps / 2:.1f} movements/s; '
                                              f'{rate:g}/s needs {2 * rate:.0f} fps')
    return result


# ---------------------------------------------------------------- module builders

async def _rotation(options, context):
    from ..providers.rotation import RotationPlugin
    config = dict(expected_max_rev_s=float(context.get('max_rev_s') or options['max_rev_s']),
                  symmetry=int(context.get('symmetry') or options.get('symmetry', 1)),
                  in_plane=options.get('in_plane', 'polar'), region_source=options.get('regions', 'auto'))
    roi = context.get('roi')
    if config['region_source'] == 'roi':
        if roi is None:
            raise SystemExit("this recipe counts inside a fixed box: pass --roi x,y,width,height")
    if roi is not None:
        x, y, width, height = roi
        config.update(region_source='roi', roi=dict(x=x, y=y, width=width, height=height))
    elif config['region_source'] == 'detections' and not context.get('detector'):
        config['region_source'] = 'auto'
    if context.get('labels'):
        config['targets'] = context['labels']
    return dict(module=RotationPlugin(context['human'], **config))


async def _yoloe(options, context):
    from ..crew.modules import open_worker
    from ..demos.rotation import _check_worker_dependencies, detector_config
    entry = context['installed'](options)
    python = context.get('worker_python') or entry.get('python') or sys.executable
    _check_worker_dependencies(python)
    labels = context.get('labels') or [options['label']]
    models = context.get('models') or Path(entry['config']['checkpoint_path']).parent
    # One frame's result depends on that frame only, so "replicas": N workers may share the frames.
    config = dict(detector_config(models, labels, context.get('device', 'cpu')), stateless=True)
    module = await open_worker(python, entry['provider_profile'], config, replicas=int(options.get('replicas', 1)))
    context['detector'] = True
    return dict(module=module, spec=module.spec, every_ms=options.get('every_ms'))   # every frame, unless declared


async def _mediapipe_pose(options, context):
    """MediaPipe Pose (VIDEO mode) in its own environment; ``scene.poses`` records, joints on the derived ``joints`` input."""
    from ..crew.modules import open_private_worker
    from ..providers.pose import PoseWorkerPlugin
    entry = context['installed'](options)
    worker = await open_private_worker(context.get('worker_python') or entry.get('python') or sys.executable,
                                       entry['provider_profile'],
                                       dict(entry['config'], max_entities=int(options.get('max_poses', 1)),
                                            allow_skips=True))
    # allow_skips: a frame lost on a loaded machine must not rebuild the detector (docs/landmarker-video.md).
    return dict(module=PoseWorkerPlugin(worker, joints=options.get('joints'),
                                        tracker=options.get('tracker', 'iou')))


async def _mediapipe_graph(options, context):
    from ..mediapipe_graph import MediaPipeGraphModule
    config = context['installed'](options)['config']
    return dict(module=MediaPipeGraphModule.from_manifest(
        config['manifest'], branches=options.get('branches'), side_packets=options.get('side_packets'),
        derived={**({'poses': options['joints']} if options.get('joints') else {}), **(options.get('derived') or {})}))


def _pose_source(options):
    """How to read the joints of the item's pose model (``"source"``; default MediaPipe Pose)."""
    if options.get('source') != 'sapiens2':
        return {}
    from ..providers.sapiens2_pose import GOLIATH_BODY
    return dict(mapping=GOLIATH_BODY, min_visibility=.3)   # the official demo's --kpt-thr


async def _pose_channels(options, context):
    from ..providers.pose_exemplar import PoseChannels
    module = PoseChannels(source=options.get('source', 'mediapipe_pose'), **_pose_source(options))
    return dict(module=module, spec=module.spec)


def _sensor(options, context):
    """The sensor input a module reads directly, or None."""
    from ..inputs import SensorFileInput
    recipe = context.get('recipe') or {}
    item = next((item for item in recipe.get('inputs', ()) if item['use'] == 'sensor' and item['id'] in options['consumes']), None)
    return None if item is None else SensorFileInput('', **{key: item[key] for key in ('columns', 'time', 'modality')
                                                            if key in item})


class _Channels:
    """Strip a channel value to {"channels": [...]}, the RepetitionCounter input contract.

    A sensor sample with a missing channel is not a sample of the movement; a pose's hidden joints are part of it.
    """

    def __init__(self, counter):
        self.counter, self.spec = counter, counter.spec

    def process(self, frame):
        from dataclasses import replace
        value = frame.data if isinstance(frame.data, dict) else {}
        if not isinstance(value.get('channels'), list) or self.spec.modality != 'video' and not all(value.get('mask') or ()):
            return None
        return self.counter.process(replace(frame, data=dict(channels=value['channels'])))

    def finish(self):
        return self.counter.finish()

    def reset(self):
        self.counter.reset()


async def _repetition(options, context):
    from ..providers.pose_exemplar import CHANNELS
    from ..providers.repetition import RepetitionCounter
    rate = float(context.get('max_reps_s') or options.get('max_reps_s', 2))
    sensor = _sensor(options, context)
    if sensor is not None and sensor.modality != 'imu':
        raise SystemExit('the repetition counter reads imu samples')
    counter = RepetitionCounter(channels=len(CHANNELS if sensor is None else sensor.columns),
                                modality='video' if sensor is None else 'imu', min_period_s=max(.05, 1 / (2 * rate)),
                                max_period_s=float(options.get('max_period_s', 6.7)))
    return dict(module=_Channels(counter), spec=counter.spec)


async def _exemplar(options, context):
    from ..providers.pose_exemplar import ExemplarEvents
    given = [context.get('enter'), context.get('exit')]     # the command line wins over the movement file
    from . import speaker_bound
    sensor = _sensor(options, context)
    modality = options.get('modality', 'video') if sensor is None else sensor.modality
    enter = float(given[0] or options.get('enter', .12))
    exit_ = float(given[1] or max(enter, options.get('exit', .18)))
    # hold, sequence or rule comes from the movement file; the item's "mode" is only enroll's default.
    module = ExemplarEvents(reference=options.get('reference', 'exemplar'), enter=enter, exit=exit_,
                            min_hold_ms=int(options.get('min_hold_ms', 200)),
                            fixed_thresholds=any(value is not None for value in given),
                            modality=modality, contract=None if sensor is None else sensor.contract,
                            speaker_bound=modality != 'audio' or speaker_bound(context['modules'], options['name']),
                            capability_id=options.get('capability_id', 'temporal.exemplar_events'),
                            **_pose_source(options))
    return dict(module=module, spec=module.spec)


async def _terms(options, context):
    from ..providers.term_events import TermEvents
    from . import speaker_bound
    module = TermEvents(reference=options.get('reference', 'terms'), report_others=bool(options.get('report_others')),
                        speaker_bound=speaker_bound(context['modules'], options['name']))
    return dict(module=module, spec=module.spec)


async def _role(options, context):
    """Any role of the registry, exactly as ``ProfileRuntime`` runs it (:func:`gamst.crew.modules.role_module`)."""
    from ..crew.modules import RoleUnavailable, refusal, role_module
    from ..crew.registry import ROLES, candidate
    role = ROLES[options['role']]
    if isinstance(options.get('config'), dict):    # prepared by hand: the item says where
        if not options.get('candidate'):
            raise SystemExit(f"module {options['name']!r} with its own config names its candidate")
        chosen = candidate(role.id, options['candidate'])
        entry = dict(model=chosen.id, provider_profile=chosen.provider_profile, config=options['config'],
                     python=options.get('python') or sys.executable)
    else:
        entry = context['installed'](options)
    reason = refusal(role.id, entry['model'])
    if reason:
        raise SystemExit(f"{role.id} {entry['model']}: {reason}")
    config = entry.get('config') or {}
    if config.get('allow_remote') is True and 'eur_per_hour' in config:
        # A worker's own output is closed once it starts, so the cost line of a hosted candidate is said here.
        print(f"note: {role.id} {entry['model']} is hosted: the audio leaves this device and is billed at "
              f"{config['eur_per_hour']} EUR/h (list price in the registry), 20 s minimum per stream or job",
              file=sys.stderr)
    try:
        built = await role_module(context['human'], role.id, entry, python=context.get('worker_python'))
    except RoleUnavailable as error:
        raise SystemExit(f"{role.id} {entry['model']}: {error}") from error
    if built['modality'] != role.modality:
        close = getattr(built['module'], 'close', None)
        value = close() if callable(close) else None
        if hasattr(value, '__await__'):
            await value
        raise SystemExit(f"{role.id} {entry['model']} reads {built['modality']}, not {role.modality}")
    return dict(module=built['module'], spec=built['spec'], every_ms=options.get('every_ms'))


async def _plugin(options, context):
    """A user-made module: an installed ``gamst.plugins`` entry point the owner selected (trusted code in this process)."""
    from ..extensions import load_plugin
    from . import input_modality, selected_plugin
    name = selected_plugin(options, context.get('plugins'))
    try:
        module = load_plugin(name, **(options.get('options') or {}))
    except (ValueError, TypeError, ImportError, RuntimeError) as error:
        raise SystemExit(f"module {options['name']!r}: {error}") from error
    inputs = {item['id']: input_modality(item) for item in (context.get('recipe') or {}).get('inputs', ())}
    wrong = [source for source in options['consumes'] if source in inputs and inputs[source] not in module.modalities]
    if wrong:
        raise SystemExit(f"module {options['name']!r}: plugin {name} reads {', '.join(module.modalities)}; {wrong[0]} is {inputs[wrong[0]]}")
    return dict(module=module, spec=getattr(module, 'spec', None), every_ms=options.get('every_ms'))


VIDEO = dict(role='input.video', candidate='opencv')
ROTATION = dict(role='vision.rotation', candidate='gamst.rotation')


async def _take_recorder(options, context):
    from ..takes import TakeRecorder
    source = context['inputs'][options['consumes'][0]]
    emits = options.get('emits') or ['take']
    return dict(module=TakeRecorder(context.get('record_dir') or 'gamst-takes', source=source, emits=emits[0]))


def _shared_python(*entries):
    """The one interpreter of candidates that must load in one process (they share an environment)."""
    pythons = {entry.get('python') for entry in entries}
    if len(pythons) != 1:
        raise SystemExit(f"{' and '.join(entry['model'] for entry in entries)} run in one process but were set up in "
                         'different environments; run: gamst recipe setup <recipe>')
    python = pythons.pop()
    if python is not None and not Path(python).is_file():
        raise SystemExit(f'{python} does not exist any more; run: gamst recipe setup <recipe>')
    return python


async def _flip(options, context):
    """After-take flip count of one named object; its ``yoloe`` and ``efficienttam`` stages run inside it."""
    from ..profile import role_python
    from ..takes import TakeFlips
    stages = list(context.get('stages', {}).values())
    find = next((item for item in stages if item['use'] == 'yoloe'), None)
    follow = next((item for item in stages if item['use'] == 'efficienttam'), None)
    if find is None or follow is None:
        raise SystemExit('the flip count runs a yoloe and an efficienttam stage ("in": "<this module>")')
    models, source = context.get('models'), context.get('efficienttam_source')
    python, checkpoint = context.get('worker_python'), None
    if source is None:  # a recipe run: weights, sources and the environment are where setup put them
        detector, refiner = context['installed'](find), context['installed'](follow)
        if models is None:
            models = str(Path(detector['config']['checkpoint_path']).parent)
            checkpoint = refiner['config'].get('checkpoint_path')
        source = refiner['config']['source_dir']
        python = python or _shared_python(detector, refiner)
    elif python is None:  # gamst demo rotation: the environment setup made for both, else this one
        pythons = {role_python(PARTS[item['use']].role, item.get('candidate', PARTS[item['use']].candidate))
                   for item in (find, follow)}
        python = pythons.pop() if len(pythons) == 1 else None
    else:  # gamst demo rotation --worker-python: only the detector worker runs there
        python = sys.executable
    on = context.get('count_on') or options.get('on', 'analyze')
    fast = context['fast'] if context.get('fast') is not None else options.get('fast', on == 'analyze')
    counter = TakeFlips(
        context['human'], label=','.join(context['labels']) if context.get('labels') else find['label'],
        models=models, device=context.get('device', 'cpu'), efficienttam_source=source,
        efficienttam_checkpoint=checkpoint, python=python, worker_python=context.get('worker_python'),
        expected_max_rev_s=float(context.get('max_rev_s') or options['max_rev_s']), on=on, fast=fast,
        count_at_end=bool(context.get('replay')), show=bool(context.get('show')), cached=context.get('cached'),
        log_dir=context.get('record_dir') or 'gamst-takes')
    return dict(module=counter.open())

PARTS = {
    # inputs
    'camera': Part('input', **VIDEO),
    'video': Part('input', **VIDEO),
    'take': Part('input', **VIDEO),
    'audio': Part('input'),
    'microphone': Part('input', role='input.microphone', candidate='sounddevice'),
    'gamst240': Part('input', **VIDEO),
    'sensor': Part('input'),
    'plugin': Part('plugin', build=_plugin),
    # perception modules
    'role': Part('module', build=_role),
    'rotation': Part('module', **ROTATION, build=_rotation, sampling=rotation_sampling),
    'take-recorder': Part('module', **VIDEO, build=_take_recorder),
    'flip': Part('module', **ROTATION, build=_flip, sampling=rotation_sampling),
    'yoloe': Part('module', role='vision.detect', candidate='yoloe.26s', build=_yoloe),
    'efficienttam': Part('module', role='vision.video_segmentation', candidate='efficienttam.ti_512'),
    'mediapipe-pose': Part('module', role='vision.pose', candidate='mediapipe.pose_full', build=_mediapipe_pose),
    'mediapipe-graph': Part('module', role='vision.mediapipe_graph', candidate='graph.holistic_cpu',
                            build=_mediapipe_graph),
    'pose-channels': Part('module', build=_pose_channels),
    'repetition': Part('module', build=_repetition, sampling=movement_sampling),
    'exemplar': Part('module', build=_exemplar, sampling=movement_sampling),
    'terms': Part('module', build=_terms),
    # views
    'count': Part('view'),
    'rotation-view': Part('view'),
    'halo-coin': Part('view', role='view.hud', candidate='pillow'),
}

from .perception_parts import parts as _perception_parts  # noqa: E402  (needs Part)

PARTS.update(_perception_parts(Part))
