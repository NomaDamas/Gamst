"""Bounded local EfficientPhys graph bridge; not a pulse-rate/context provider.

Caller supplies already preprocessed RGB windows. No face detection, resizing,
normalization, capture-time assignment or physiological interpretation here.
"""
from importlib.metadata import version
import io
from pathlib import Path
import threading
import types

from .vocal_expression import _digest, _read

REVISION = 'b7500b848f84ad7f86e277b4612563b69f4f88f9'
SOURCE_SHA256 = '384b01b5c99d6d280b5ddb9e2358631f89747e40e3181b6a174e26d6e74798c9'


def model_class(source):
    """Execute only the exact pinned user-supplied upstream source bytes."""
    import hashlib
    if type(source) is not bytes or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('EfficientPhys source SHA256 mismatch')
    module = types.ModuleType('_gamst_verified_efficientphys')
    exec(compile(source, 'verified/EfficientPhys.py', 'exec'), module.__dict__)
    return module.EfficientPhys


def _geometry(image_size, frame_depth):
    if type(image_size) is not int or image_size not in (36, 72, 96):
        raise ValueError('EfficientPhys image_size must be36,72or96')
    if type(frame_depth) is not int or frame_depth != 10:
        raise ValueError('only pinned frame_depth10 is supported')


def _state(model, payload, torch):
    actual = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    expected = model.state_dict()
    if not isinstance(actual, dict) or not actual or any(type(k) is not str for k in actual):
        raise ValueError('EfficientPhys requires a bare state dictionary')
    # Official trainer saves DataParallel's module-prefixed bare state dict.
    if all(k.startswith('module.') for k in actual):
        actual = {k[7:]: v for k, v in actual.items()}
    if set(actual) != set(expected):
        raise ValueError('EfficientPhys checkpoint keys do not match the exact graph')
    for name, tensor in actual.items():
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or
                tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype or
                not torch.isfinite(tensor).all().item()):
            raise ValueError('EfficientPhys checkpoint tensor shape/dtype/finite mismatch')
    model.load_state_dict(actual, strict=True)


class EfficientPhysNative:
    """One independent window per call, official terminal-frame duplication.

    The graph uses bidirectional temporal shifts inside ten-frame groups. It
    cannot emit causal per-frame outputs. Outputs are arbitrary model units,
    not BPM or calibrated physiological measurements. Max300inputframes.
    """
    def __init__(self, model, *, image_size=72, frame_depth=10, device='cpu'):
        _geometry(image_size, frame_depth)
        if device not in ('cpu', 'cuda') or type(device) is not str:
            raise ValueError('select cpu or cuda')
        import torch
        if not isinstance(model, torch.nn.Module) or model.training:
            raise ValueError('explicit evaluated Torch model required')
        self.model, self.image_size, self.frame_depth = model, image_size, frame_depth
        self.device = device
        self.artifacts_verified = False
        self._lock = threading.RLock()

    @classmethod
    def from_local(cls, source_path, checkpoint_path, *, checkpoint_sha256,
                   image_size=72, frame_depth=10, device='cpu'):
        _geometry(image_size, frame_depth)
        _digest(checkpoint_sha256)
        if type(device) is not str or device not in ('cpu', 'cuda'):
            raise ValueError('select cpu or cuda')
        if any(type(p) is not str or not Path(p).is_absolute() for p in (source_path, checkpoint_path)):
            raise ValueError('absolute local EfficientPhys source/checkpoint paths required')
        source = _read(source_path, 16384, SOURCE_SHA256)
        payload = _read(checkpoint_path, 32*1024*1024, checkpoint_sha256)
        if version('torch').split('+')[0] != '2.8.0':
            raise RuntimeError('selected EfficientPhys runtime requires torch==2.8.0')
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('selected CUDA device unavailable')
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            model = model_class(source)(img_size=image_size, frame_depth=frame_depth).float().cpu().eval()
        _state(model, payload, torch)
        result = cls(model.to(device), image_size=image_size, frame_depth=frame_depth, device=device)
        result.artifacts_verified = True
        return result

    def predict(self, rgb):
        import numpy as np
        import torch
        with self._lock:
            if self.model is None:
                raise RuntimeError('EfficientPhys runtime closed')
            if (type(rgb) is not np.ndarray or rgb.dtype != np.dtype('float32') or
                    rgb.ndim != 4 or rgb.shape[1:] != (self.image_size, self.image_size, 3) or
                    not 10 <= len(rgb) <= 300 or len(rgb) % self.frame_depth or
                    not np.isfinite(rgb).all()):
                raise ValueError('finite float32RGB NHWC window of10..300frames, multipleof10 required')
            # No implicit standardization: upstream does it over the complete
            # source clip BEFORE chunking. Per-window normalization is different.
            frames = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).to(self.device)
            with torch.inference_mode():
                data = torch.cat((frames, frames[-1:]), dim=0)
                output = self.model(data)
                if (type(output) is not torch.Tensor or output.dtype != torch.float32 or
                        tuple(output.shape) != (len(rgb), 1) or not torch.isfinite(output).all().item()):
                    raise ValueError('invalid EfficientPhys output')
                return output[:, 0].cpu().numpy().copy()

    def close(self):
        with self._lock:
            self.model = None
