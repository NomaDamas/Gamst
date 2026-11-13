"""Explicit local EfficientTAM construction in an owned worker only."""
import hashlib
from importlib.metadata import version, PackageNotFoundError
import io
import os
from pathlib import Path
import sys
import tempfile

from .efficienttam_sources import SOURCES
from .efficienttam_stream import EfficientTAMStream
from .yoloe import _hash


def configuration(config):
    required = {'source_dir', 'checkpoint_path', 'checkpoint_sha256', 'runtime_root'}
    if (type(config) is not dict or not required <= set(config) or
            set(config)-required-{'device', 'max_gap_ms', 'max_frames'} or
            any(type(config[k]) is not str or not Path(config[k]).is_absolute()
                for k in ('source_dir', 'checkpoint_path', 'runtime_root'))):
        raise ValueError('EfficientTAM requires absolute local source/checkpoint/runtime paths and checkpoint hash')
    _hash(config['checkpoint_sha256'])
    device = config.get('device', 'cpu')
    if type(device) is not str or device not in ('cpu', 'cuda'):
        raise ValueError('EfficientTAM device must be cpu or cuda')
    return dict(checkpoint_sha256=config['checkpoint_sha256'], device=device,
                max_gap_ms=config.get('max_gap_ms', 1000), max_frames=config.get('max_frames', 1000000))


def _read(path, digest, limit):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('absolute existing local EfficientTAM artifact required')
    with path.open('rb') as source:
        data = source.read(limit+1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('EfficientTAM artifact size/hash mismatch: '+path.name)
    return data


def _weights(model, payload, torch):
    weights = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if type(weights) is not dict or set(weights) != {'model'} or not isinstance(weights['model'], dict):
        raise ValueError('EfficientTAM checkpoint must contain exactly the model state dictionary')
    actual, expected = weights['model'], model.state_dict()
    if set(actual) != set(expected):
        raise ValueError('EfficientTAM checkpoint must contain every exact model tensor')
    for name, tensor in actual.items():
        if (type(tensor) is not torch.Tensor or tensor.shape != expected[name].shape or
                tensor.dtype != expected[name].dtype or tensor.device.type != 'cpu' or
                not torch.isfinite(tensor).all().item()):
            raise ValueError('EfficientTAM tensor shape/dtype/finite mismatch: '+name)
    model.load_state_dict(actual, strict=True)


class NativeEfficientTAM:
    def __init__(self, stream, directory):
        self.stream, self.directory = stream, directory

    def step(self, image, *, boxes=None):
        if self.stream is None: raise RuntimeError('EfficientTAM native runtime closed')
        return self.stream.step(image, boxes=boxes)

    def reset(self):
        if self.stream is None: raise RuntimeError('EfficientTAM native runtime closed')
        self.stream.reset()

    def close(self):
        stream, self.stream = self.stream, None
        if stream is not None: stream.close()
        directory, self.directory = self.directory, None
        if directory is not None: directory.cleanup()


def load_worker(config):
    from .efficienttam_context import EfficientTAMContext
    args = configuration(config)
    EfficientTAMContext.capability(**args)
    if (os.environ.get('GAMST_PROVIDER_WORKER') != '1' or
            any(k == 'efficient_track_anything' or k.startswith('efficient_track_anything.') for k in sys.modules)):
        raise RuntimeError('EfficientTAM requires a fresh owned ProcessProvider worker')
    # Snapshot verified source and checkpoint bytes before executing source. No
    # installed upstream package or caller Hydra configuration can replace them.
    blobs = {name: _read(Path(config['source_dir'])/name, digest, 2*1024*1024)
             for name, digest in SOURCES.items()}
    payload = _read(config['checkpoint_path'], args['checkpoint_sha256'], 256*1024*1024)
    for package, wanted in (('torch', '2.8.0'), ('torchvision', '0.23.0'),
                            ('hydra-core', '1.3.2'), ('omegaconf', '2.3.0')):
        try: found = version(package).split('+')[0]
        except PackageNotFoundError as exc:
            raise RuntimeError('selected EfficientTAM runtime requires '+package+'=='+wanted) from exc
        if found != wanted: raise RuntimeError('selected EfficientTAM runtime requires '+package+'=='+wanted)
    root = Path(config['runtime_root']).resolve(strict=True)
    if not root.is_dir(): raise ValueError('runtime_root must be an existing directory')
    directory = tempfile.TemporaryDirectory(prefix='gamst-efficienttam-', dir=root)
    stream = None
    try:
        for name, data in blobs.items():
            path = Path(directory.name)/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        del blobs
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, directory.name)
        import torch
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            raise RuntimeError('EfficientTAM worker requires fresh Hydra state')
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        if args['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('selected CUDA device unavailable')
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            model = build_efficienttam_video_predictor(
                'configs/efficienttam/efficienttam_ti_512x512.yaml', ckpt_path=None,
                device='cpu', mode='eval', hydra_overrides_extra=['++model.compile_image_encoder=False'],
                apply_postprocessing=False, vos_optimized=False).float().eval()
        _weights(model, payload, torch)
        del payload
        model = model.to(args['device'])
        stream = EfficientTAMStream(model, max_frames=args['max_frames'])
        return EfficientTAMContext(NativeEfficientTAM(stream, directory), artifact_verified=True, **args)
    except BaseException:
        if stream is not None: stream.close()
        directory.cleanup()
        raise
