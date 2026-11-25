"""The parts a recipe can name: input modules, perception modules and views.

Each perception part declares what ``setup`` installs (extras, pinned weights, pinned source
checkouts), how to build its Pipeline module, and its sampling rule (what frame rate the input
needs so the count is not an alias). Adding a part here is how a new capability becomes usable
from recipes; the Pipeline, the records and the MCP context stay the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

MODELS_ROOT = Path(os.environ.get('GAMST_MODELS', Path.home() / '.cache' / 'gamst' / 'models'))

# Pinned MediaPipe Pose Landmarker task files (Google's versioned model bucket, float16 v1).
POSE_ASSETS = {
    'pose_landmarker_full.task': (
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/'
        'pose_landmarker_full.task', '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1'),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/'
        'pose_landmarker_heavy.task', '64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b'),
}


@dataclass
class Part:
    kind: str                      # 'input' | 'module' | 'view'
    extras: tuple = ()
    pip: tuple = ()
    models: dict = field(default_factory=dict)   # file name -> (url, sha256), fetched into models/<group>
    group: str = ''
    sources: tuple = ()
    build: Callable[..., Any] | None = None       # async (options, context) -> dict(module, spec?, emits?)
    sampling: Callable[..., Any] | None = None    # (options, fps) -> dict(required_fps, ...) or None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def model_path(part, name, models=None):
    directory = Path(models or MODELS_ROOT) / part.group
    path = directory / name
    if not path.is_file():
        raise SystemExit(f'missing {path}; run: gamst recipe setup <recipe>')
    expected = part.models[name][1]
    if sha256(path) != expected:
        raise SystemExit(f'{path} does not match its pinned sha256 {expected}')
    return path


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


class _Detector:
    """YOLOE open-vocabulary detector worker as a provider module (its own process, pinned weights)."""

    def __init__(self, worker, spec):
        self.worker, self.spec = worker, spec

    async def process(self, frame):
        from ..demos.rotation import _detector_image
        from ..plugins import InputFrame
        result = self.worker.process(InputFrame(_detector_image(frame.data), 'video', frame.timestamp_ms,
                                                subject_id=frame.subject_id, sensor=frame.sensor))
        return await result if hasattr(result, '__await__') else result

    async def close(self):
        await self.worker.close()


async def _yoloe(options, context):
    from .. import ProcessProvider
    from ..demos.rotation import DEFAULT_MODELS, _check_worker_dependencies, detector_config
    python = context.get('worker_python')
    _check_worker_dependencies(python)
    labels = context.get('labels') or [options['label']]
    config = detector_config(Path(context.get('models') or DEFAULT_MODELS), labels, context.get('device', 'cpu'))
    spec = ProcessProvider.capability('yoloe.local', config)
    worker = await ProcessProvider.open(python, 'yoloe.local', config)
    context['detector'] = True
    return dict(module=_Detector(worker, spec), spec=spec, every_ms=int(options.get('every_ms', 100)))


async def _mediapipe_pose(options, context):
    from ..providers.pose import MediaPipePosePlugin
    name = options.get('model', 'pose_landmarker_full.task')
    path = model_path(PARTS['mediapipe-pose'], name, context.get('models_root'))
    return dict(module=MediaPipePosePlugin(str(path), max_poses=int(options.get('max_poses', 1))))


async def _pose_channels(options, context):
    from ..providers.pose_exemplar import PoseChannels
    module = PoseChannels(source=options.get('source', 'mediapipe_pose'))
    return dict(module=module, spec=module.spec)


class _Channels:
    """Strip a pose-channel value to {"channels": [...]}, the RepetitionCounter input contract."""

    def __init__(self, counter):
        self.counter, self.spec = counter, counter.spec

    def process(self, frame):
        from dataclasses import replace
        value = frame.data if isinstance(frame.data, dict) else {}
        if not isinstance(value.get('channels'), list):
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
    counter = RepetitionCounter(channels=len(CHANNELS), min_period_s=max(.05, 1 / (2 * rate)),
                                max_period_s=float(options.get('max_period_s', 6.7)))
    return dict(module=_Channels(counter), spec=counter.spec)


async def _exemplar(options, context):
    from ..providers.pose_exemplar import ExemplarEvents
    module = ExemplarEvents(reference=options.get('reference', 'exemplar'), mode=options.get('mode', 'hold'),
                            enter=float(context.get('enter') or options.get('enter', .12)),
                            exit=float(context.get('exit') or options.get('exit', .18)),
                            min_hold_ms=int(options.get('min_hold_ms', 200)))
    return dict(module=module, spec=module.spec)


async def _terms(options, context):
    from ..providers.term_events import TermEvents
    module = TermEvents(reference=options.get('reference', 'terms'))
    return dict(module=module, spec=module.spec)


PARTS = {
    # inputs
    'camera': Part('input', extras=('rotation',)),
    'video': Part('input', extras=('rotation',)),
    'take': Part('input', extras=('rotation',)),
    'audio': Part('input'),
    'microphone': Part('input', pip=('sounddevice',)),
    'gamst240': Part('input', extras=('rotation',)),
    # perception modules
    'rotation': Part('module', extras=('rotation',), build=_rotation, sampling=rotation_sampling),
    'flip': Part('module', extras=('rotation',), sampling=rotation_sampling),
    'yoloe': Part('module', extras=('yoloe',), pip=('clip',), models='yoloe', build=_yoloe),
    'efficienttam': Part('module', extras=('efficienttam',), models='efficienttam', sources=('EfficientTAM',)),
    'mediapipe-pose': Part('module', extras=('mediapipe',), models=POSE_ASSETS, group='mediapipe',
                           build=_mediapipe_pose),
    'pose-channels': Part('module', build=_pose_channels),
    'repetition': Part('module', build=_repetition, sampling=movement_sampling),
    'exemplar': Part('module', build=_exemplar, sampling=movement_sampling),
    'terms': Part('module', build=_terms),
    # views
    'count': Part('view'),
    'rotation-view': Part('view', extras=('rotation',)),
    'halo-coin': Part('view', extras=('rotation',)),
}
