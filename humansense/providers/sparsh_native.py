"""Local pinned Sparsh graph. Inputs are preprocessed windows, not raw touch.

No trained asset acquisition, contact classification or physical force calibration.
The private import graph never registers tactile_ssl in sys.modules.
"""
import builtins
import importlib.util
import json
from pathlib import Path
import threading
import types

from .sparsh_sources import REVISION, SOURCES, STATE_SHAPES, LICENSE_SHA256
from .vocal_expression import _read

GRAPH = 'sparsh-dino-base-slipforce-v1'


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def source_graph(root):
    """Hash all sources before executing any; exact exports, private imports."""
    root = Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('absolute supplied source directory required')
    root = root.resolve()
    license_path = (root / "LICENSE.md").resolve()
    if not license_path.is_relative_to(root):
        raise ValueError("license escapes supplied directory")
    _read(license_path, 65536, LICENSE_SHA256)
    payloads = {}
    for relative, digest in SOURCES.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('source escapes supplied directory')
        payloads[relative] = _read(path, 65536, digest)
    modules = {}
    for name in ('tactile_ssl', 'tactile_ssl.model', 'tactile_ssl.model.layers',
                 'tactile_ssl.downstream_task'):
        modules[name] = types.ModuleType(name)
    exports = {
        'tactile_ssl.model': {'VIT_EMBED_DIMS': ('vision_transformer', 'VIT_EMBED_DIMS')},
        'tactile_ssl.model.layers': {
            'Mlp': ('mlp', 'Mlp'), 'MemEffAttention': ('attention', 'MemEffAttention'),
            'NestedTensorBlock': ('block', 'NestedTensorBlock'),
            'PatchEmbed': ('patch_embed', 'PatchEmbed'),
            'PatchEmbed3D': ('patch_embed', 'PatchEmbed3D'),
            'SinusoidalEmbed': ('patch_embed', 'SinusoidalEmbed'),
            'SwiGLUFFNFused': ('swiglu_ffn', 'SwiGLUFFNFused'),
            'CrossAttention': ('attention', 'CrossAttention'),
            'CrossAttentionBlock': ('attention', 'CrossAttentionBlock'),
        },
    }

    def load(name):
        if name in modules:
            return modules[name]
        relative = name.replace('.', '/') + '.py'
        if name == 'tactile_ssl.utils':
            relative = 'tactile_ssl/utils/__init__.py'
        if relative not in payloads:
            raise ImportError('source module outside pinned graph: ' + name)
        module = types.ModuleType(name)
        module.__package__ = name if relative.endswith('__init__.py') else name.rpartition('.')[0]
        module.__builtins__ = dict(vars(builtins), __import__=private_import)
        modules[name] = module
        exec(compile(payloads[relative], 'verified/' + relative, 'exec'), module.__dict__)
        return module

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            name = importlib.util.resolve_name('.' * level + name, globals['__package__'])
        if name == 'xformers' or name.startswith('xformers.'):
            # Explicit CPU torch fallback; no mutation of process environment.
            raise ImportError('gamst selects upstream torch attention fallback')
        if name == 'tactile_ssl' or name.startswith('tactile_ssl.'):
            module = load(name)
            for symbol in fromlist:
                if not hasattr(module, symbol) and symbol in exports.get(name, {}):
                    suffix, attr = exports[name][symbol]
                    setattr(module, symbol, getattr(load(name + '.' + suffix), attr))
            return module if fromlist else modules['tactile_ssl']
        return builtins.__import__(name, globals, locals, fromlist, level)

    encoder = load('tactile_ssl.model.vision_transformer')
    head = load('tactile_ssl.downstream_task.slip_decoders')
    return encoder, head


def construct(encoder_module, head_module):
    """Exact official base graph, no configurable architecture or random fallback."""
    encoder = encoder_module.vit_base(img_size=(320, 240), patch_size=16,
        in_chans=6, pos_embed_fn='sinusoidal', num_register_tokens=1)
    head = head_module.SlipForceProbe(embed_dim='base', num_heads=12, depth=1,
        attn_pool=True, num_classes=2, dropout=0.0)
    return encoder.float().cpu().eval(), head.float().cpu().eval()


def _header(payload):
    if len(payload) < 10:
        raise ValueError('invalid safetensors file')
    size = int.from_bytes(payload[:8], 'little')
    if not 2 <= size <= 131072 or 8 + size > len(payload):
        raise ValueError('invalid safetensors header length')
    header = json.loads(payload[8:8+size], object_pairs_hook=_pairs)
    if type(header) is not dict or not 1 <= len(header) <= 512:
        raise ValueError('bounded tensor mapping required')
    for name, tensor in header.items():
        if type(name) is not str or not name or len(name) > 256 or type(tensor) is not dict or set(tensor) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('canonical tensor entry required; no metadata')
        shape, offsets = tensor['shape'], tensor['data_offsets']
        if tensor['dtype'] != 'F32' or type(shape) is not list or len(shape) > 4 or any(type(n) is not int or not 1 <= n <= 10000 for n in shape):
            raise ValueError('bounded float32 shape required')
        if type(offsets) is not list or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError('integer tensor offsets required')
    end = 0
    for tensor in sorted(header.values(), key=lambda item: item['data_offsets'][0]):
        shape, offsets = tensor['shape'], tensor['data_offsets']
        count = 1
        for n in shape:
            count *= n
        if type(offsets) is not list or len(offsets) != 2 or any(type(n) is not int for n in offsets) or offsets != [end, end + count*4]:
            raise ValueError('contiguous bounded tensor offsets required')
        end += count*4
        if end > 400*1024*1024:
            raise ValueError('tensor bytes exceed limit')
    if end != len(payload)-8-size:
        raise ValueError('tensor data length mismatch')
    return header


def strict_state(model, payload):
    import torch
    from safetensors.torch import load
    header = _header(payload)
    expected = model.state_dict()
    if set(header) != set(expected) or any(header[k]['shape'] != list(v.shape) or v.dtype != torch.float32 for k,v in expected.items()):
        raise ValueError('canonical keys/shapes do not match exact graph')
    state = load(payload)
    if any(v.layout != torch.strided or v.device.type != 'cpu' or v.dtype != torch.float32 or not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError('nonfinite or invalid checkpoint tensor')
    model.load_state_dict(state, strict=True)


class SparshSlipNative:
    """Synchronous CPU B1 window inference; lock owns inference through close.

    Async callers must retain/drain their thread operation on cancellation.
    There is no hard time limit or realtime/quality claim at this layer.
    """
    def __init__(self):
        raise TypeError("use SparshSlipNative.from_local with verified assets")

    @classmethod
    def from_local(cls, source_root, encoder_path, head_path, *, encoder_sha256, head_sha256):
        for path in (encoder_path, head_path):
            if type(path) is not str or not Path(path).is_absolute():
                raise ValueError('absolute local canonical encoder and head paths required')
        encoder_bytes = _read(encoder_path, 400*1024*1024, encoder_sha256)
        head_bytes = _read(head_path, 32*1024*1024, head_sha256)
        import numpy as np
        for payload, expected in zip((encoder_bytes, head_bytes), STATE_SHAPES):
            header = _header(payload)
            if {k: v['shape'] for k,v in header.items()} != expected:
                raise ValueError('canonical keys/shapes do not match pinned graph')
            start = 8 + int.from_bytes(payload[:8], 'little')
            if not np.isfinite(np.frombuffer(payload, dtype='<f4', offset=start)).all():
                raise ValueError('nonfinite checkpoint tensor')
        from importlib.metadata import version
        if version('torch').split('+')[0] != '2.8.0':
            raise RuntimeError('selected Sparsh runtime requires torch==2.8.0')
        import torch
        if torch.get_default_dtype() != torch.float32:
            raise RuntimeError("float32 Torch default dtype required")
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            encoder, head = construct(*source_graph(source_root))
        strict_state(encoder, encoder_bytes)
        strict_state(head, head_bytes)
        result = object.__new__(cls)
        result._encoder, result._head = encoder, head
        result._lock = threading.RLock()
        result.identity = {'graph': GRAPH, 'source_revision': REVISION,
                           'encoder_sha256': encoder_sha256, 'head_sha256': head_sha256,
                           'attention': 'torch-cpu', 'trained_quality_verified': False}
        return result

    def predict(self, pair):
        import numpy as np
        import torch
        with self._lock:
            if self._encoder is None:
                raise RuntimeError('Sparsh runtime closed')
            if type(pair) is not np.ndarray or pair.dtype != np.dtype('float32') or pair.shape != (1,6,320,240) or not np.isfinite(pair).all() or pair.min() < 0 or pair.max() > 1:
                raise ValueError('finite float32 B1x6x320x240 input in [0,1] required')
            with torch.inference_mode():
                output = self._head({'latent': self._encoder(torch.from_numpy(pair.copy()))})
            if type(output) is not dict or set(output) != {'slip', 'force'}:
                raise ValueError('invalid native output')
            for name, shape in (('slip', (1,2)), ('force', (1,3))):
                v = output[name]
                if type(v) is not torch.Tensor or v.dtype != torch.float32 or tuple(v.shape) != shape or not torch.isfinite(v).all().item():
                    raise ValueError('invalid native output tensor')
            if output['force'].abs().max().item() > 1:
                raise ValueError('invalid normalized delta force')
            return {'slip_logits': output['slip'][0].numpy().copy(),
                    'normalized_delta_force': output['force'][0].numpy().copy()}

    def close(self):
        with self._lock:
            self._encoder = self._head = None
