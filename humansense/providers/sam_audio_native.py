"""Pinned upstream SAM graph with local-only assets and exact tensor coverage.

No model is installed or downloaded here. File/source/tensor verification is
not a trained-checkpoint compatibility, quality or performance verdict.
"""
from contextlib import contextmanager, ExitStack
from collections import OrderedDict
from copy import deepcopy
import importlib.abc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
import threading
import time
from unittest.mock import patch
import zipfile

from ..sam_audio_artifact import read_manifest, verify_manifest, verify_runtime, _file, _regular_path, SOURCE_PINS
from ..sam_audio_sources import VerifiedSources
from ..sam_audio_windows import validate_outputs

MAX_STORAGE = 16*1024**3
MAX_TENSORS = 100000
MAX_SERIALIZED_ASSETS = 24*1024**3+512*1024**2
REQUIRED_PACKAGES = frozenset(('torch', 'torchaudio', 'torchvision', 'transformers',
    'safetensors', 'huggingface-hub', 'numpy', 'torchdiffeq', 'torchcodec', 'timm',
    'einops', 'descript-audiotools'))


def _denied(*args, **kwargs):
    raise RuntimeError('network/model download disabled in owned SAM worker')


class _NoAuxiliaryImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('imagebind', 'laion_clap', 'audiobox_aesthetics'):
            raise ModuleNotFoundError('unsupported auxiliary model disabled')


@contextmanager
def offline():
    """Deny Python sockets/known Hub loaders; not an OS security sandbox."""
    finder = _NoAuxiliaryImports()
    if any(name.split('.')[0] in ('imagebind','laion_clap','audiobox_aesthetics') for name in sys.modules):
        raise RuntimeError('auxiliary model package already imported; fresh worker required')
    sys.meta_path.insert(0, finder)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                          HF_HUB_DISABLE_TELEMETRY='1'))
            for name in ('connect', 'connect_ex'):
                stack.enter_context(patch.object(socket.socket, name, _denied))
            stack.enter_context(patch.object(socket, 'create_connection', _denied))
            import huggingface_hub
            for name in ('hf_hub_download', 'snapshot_download'):
                stack.enter_context(patch.object(huggingface_hub, name, _denied))
            import torch
            for name in ('download_url_to_file', 'load_state_dict_from_url'):
                stack.enter_context(patch.object(torch.hub, name, _denied))
            yield
    finally:
        sys.meta_path.remove(finder)


def _runtime_closure(manifest):
    """Require declared exact versions for the worker's loaded dependency closure."""
    owners = importlib.metadata.packages_distributions()
    declared = manifest['runtime']['packages']
    seen=set()
    for name in set(sys.modules):
        root = name.split('.')[0]
        if root in sys.stdlib_module_names or root in ('sam_audio', 'core', 'dacvae', 'humansense', '__main__'):
            continue
        module = sys.modules[name]
        filename = getattr(module, '__file__', None)
        if filename is None or root in seen:
            continue
        seen.add(root)
        distributions = owners.get(root, [])
        if not distributions:
            raise RuntimeError('imported dependency has no declared distribution identity')
        for distribution in distributions:
            key = distribution.lower().replace('_', '-').replace('.', '-')
            if key not in declared or importlib.metadata.version(distribution) != declared[key]:
                raise RuntimeError('imported dependency absent from frozen runtime')


def _config_bounds(value, depth=0):
    if depth > 12:
        raise ValueError('configuration nesting exceeds bound')
    if type(value) is dict:
        if len(value) > 128:
            raise ValueError('configuration object exceeds bound')
        for key,item in value.items():
            if key in ('n_layers','num_layers','num_decoder_layers','n_codebooks','depth','depths','encoder_layers','decoder_layers') and (type(item) is not int or not 1 <= item <= 128):
                raise ValueError('configuration layer/codebook count exceeds SDK bound')
            if key in ('n_heads','num_heads') and (type(item) is not int or not 1 <= item <= 256):
                raise ValueError('configuration attention heads exceed SDK bound')
            _config_bounds(item, depth+1)
    elif type(value) is list:
        if len(value) > 64: raise ValueError('configuration list exceeds bound')
        for item in value: _config_bounds(item, depth+1)
    elif type(value) in (int, float) and abs(value) > 10000000:
        raise ValueError('configuration scalar exceeds allocation planning bound')


def _plan(model, torch):
    state = model.state_dict()
    if not state or len(state) > MAX_TENSORS:
        raise ValueError('model tensor count exceeds bound')
    total = sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
    if total > MAX_STORAGE:
        raise ValueError('prospective model storage exceeds SDK16GiB bound')
    return {key:(tuple(t.shape), t.dtype) for key,t in state.items()}


def _state(actual, expected, torch):
    if type(actual) not in (dict,OrderedDict) or set(actual) != set(expected):
        raise ValueError('checkpoint tensor keys differ from required graph')
    total = 0
    for name,tensor in actual.items():
        shape,dtype = expected[name]
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or
                tuple(tensor.shape) != shape or tensor.dtype != dtype or
                not torch.isfinite(tensor).all().item()):
            raise ValueError('checkpoint shape/dtype/finite mismatch')
        total += tensor.numel()*tensor.element_size()
        if total > MAX_STORAGE: raise ValueError('checkpoint storage exceeds bound')
    return actual


def _geometry(actual, expected, torch, device):
    if type(actual) not in (dict,OrderedDict) or set(actual) != set(expected):
        raise ValueError('checkpoint tensor keys differ from required graph')
    for key,tensor in actual.items():
        shape,dtype=expected[key]
        if (type(tensor) is not torch.Tensor or tensor.device.type != device or
                tuple(tensor.shape) != shape or tensor.dtype != dtype):
            raise ValueError('checkpoint shape/dtype mismatch')


def _torch_checkpoint(path, digest, expected, torch):
    """Deserialize one immutable verified ZIP snapshot, meta geometry first.

    Snapshot RAM and CPU graph/checkpoint RAM are additional bounded costs;
    this is not a peak-RAM measurement or a trained compatibility claim.
    """
    blob = _file(Path(path), digest, MAX_STORAGE, retain=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        if (not 1 <= len(infos) <= MAX_TENSORS+32 or
                any(i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1 for i in infos) or
                len({i.filename for i in infos}) != len(infos) or
                sum(i.file_size for i in infos) > MAX_STORAGE or
                any(i.filename.endswith('/data.pkl') and i.file_size > 32*1024*1024 for i in infos)):
            raise ValueError('unsupported or excessive Torch storage archive')
    planned = torch.load(io.BytesIO(blob), map_location='meta', weights_only=True)
    _geometry(planned, expected, torch, 'meta')
    del planned
    return _state(torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True), expected, torch)


def _t5_state(path, digest, expected, torch):
    """Encoder-only immutable safetensors; explicit shared embedding alias.

    Full T5 decoder/head checkpoints are deliberately unsupported until their
    exact omission policy is verified against supplied actual assets.
    """
    from safetensors.torch import load
    blob = _file(Path(path), digest, 8*1024**3, retain=True)
    if len(blob) < 8:
        raise ValueError('short safetensors header')
    length = int.from_bytes(blob[:8], 'little')
    if not 0 < length <= min(32*1024*1024, len(blob)-8):
        raise ValueError('safetensors header exceeds bound')
    header = json.loads(blob[8:8+length])
    if type(header) is not dict or len(header) > MAX_TENSORS+1:
        raise ValueError('safetensors tensor count exceeds bound')
    keys = set(header)-{'__metadata__'}
    aliases = {'shared.weight', 'encoder.embed_tokens.weight'}
    missing = set(expected)-keys
    if keys-set(expected) or missing-aliases or not keys & aliases:
        raise ValueError('unsupported T5 encoder tensor coverage; full decoder layouts rejected')
    total = 0
    for key in keys:
        entry = header[key]
        if (type(entry) is not dict or entry.get('dtype') != 'F32' or
                type(entry.get('shape')) is not list or tuple(entry['shape']) != expected[key][0]):
            raise ValueError('T5 tensor shape/dtype mismatch')
        count = 1
        for n in entry['shape']: count *= n
        total += count*4
        if total > MAX_STORAGE: raise ValueError('T5 tensor storage exceeds bound')
    # The safe parser validates offsets/overlap/storage on the exact snapshot.
    actual = load(blob)
    if missing:
        other = next(iter(keys & aliases))
        for key in missing: actual[key] = actual[other]
    if all(key in actual for key in aliases) and not torch.equal(actual['shared.weight'],actual['encoder.embed_tokens.weight']):
        raise ValueError('T5 tied embedding aliases differ')
    return _state(actual, expected, torch)


def _asset_budget(declaration):
    paths=[Path(source['root'])/name for source in declaration['sources'].values() for name in source['files']]
    paths += [Path(declaration['sam'][key]) for key in ('config_path','checkpoint_path')]
    paths += [Path(declaration['t5']['root'])/name for name in declaration['t5']['files']]
    if sum(_regular_path(path).st_size for path in paths) > MAX_SERIALIZED_ASSETS:
        raise ValueError('aggregate serialized assets exceed SDK24.5GiB bound')


def _local_tokenizer(transformers, declaration):
    # The owned parent creates this mode0700 directory and removes it after
    # startup completes/fails, including cancellation followed by worker kill.
    directory=os.environ.get('GAMST_SAM_TOKENIZER_DIR')
    if not directory or not Path(directory).is_absolute():
        raise RuntimeError('parent-owned SAM tokenizer staging directory required')
    directory=Path(directory)
    info=_regular_path(directory,directory=True)
    if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid() or list(directory.iterdir()):
        raise ValueError('empty private owned tokenizer directory required')
    root=Path(declaration['t5']['root'])
    snapshots={name:_file(root/name,digest,128*1024 if name.endswith('config.json') else 32*1024*1024,retain=True)
               for name,digest in declaration['t5']['files'].items() if name != 'model.safetensors'}
    for name,data in snapshots.items():
        descriptor=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb') as stream: stream.write(data)
    return transformers.AutoTokenizer.from_pretrained(str(directory),
        local_files_only=True,trust_remote_code=False,use_fast=True)


class _Tokenizer:
    """Record the exact upstream forward tokenization; publish counts only."""
    def __init__(self, tokenizer):
        self.tokenizer, self.stats = tokenizer, None

    def __call__(self, texts, **kwargs):
        full = self.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
        result = self.tokenizer(texts, **kwargs)
        before = len(full['input_ids'][0])
        after = int(result['attention_mask'][0].sum().item())
        if not 0 <= before <= 8192 or not 0 <= after <= kwargs['max_length']:
            raise ValueError('tokenizer lengths exceed SDK bounds')
        self.stats = dict(token_count_before=before, token_count_after=after,
                          max_tokens=kwargs['max_length'], truncated=before > after)
        return result


class SAMAudioNative:
    @classmethod
    def from_local(cls, config):
        from .sam_audio_context import configuration
        config = configuration(config)
        if sys.version_info < (3,11):
            raise RuntimeError('SAM native worker requires Python>=3.11')
        declaration = read_manifest(config['manifest_path'], expected_sha256=config['manifest_sha256'])
        if not REQUIRED_PACKAGES <= set(declaration['runtime']['packages']):
            raise ValueError('complete required native runtime declarations missing')
        verify_runtime(declaration, expected_sha256=config['manifest_sha256'])
        report = verify_manifest(declaration, expected_sha256=config['manifest_sha256'])
        _asset_budget(declaration)
        _config_bounds(report['effective_config'])
        finder = VerifiedSources(declaration)
        finder.install()
        try:
            with offline():
                import torch
                import transformers
                from sam_audio.model import model as upstream
                from sam_audio.model.config import SAMAudioConfig
                from sam_audio.processor import SAMAudioProcessor
                from sam_audio.model.text_encoder import T5TextEncoder
                t5root = declaration['t5']['root']
                # Construct config from verified immutable bytes, never Hub.
                t5config = json.loads(_file(Path(t5root)/'config.json', declaration['t5']['files']['config.json'],128*1024,retain=True))
                _config_bounds(t5config)
                t5config = transformers.T5Config(**t5config)
                class LocalT5(T5TextEncoder):
                    def __init__(self, cfg):
                        torch.nn.Module.__init__(self)
                        self.model = transformers.T5EncoderModel(deepcopy(t5config))
                        self.pad_mode, self.max_length = cfg.pad_mode, cfg.max_length
                        self.tokenizer = None
                effective = report['effective_config']
                with patch.object(upstream, 'T5TextEncoder', LocalT5):
                    with torch.device('meta'):
                        planned = upstream.SAMAudio(SAMAudioConfig(**deepcopy(effective)))
                        expected = _plan(planned, torch)
                    del planned
                    # Reconstruct instead of to_empty: nonpersistent positional
                    # buffers must retain official initialized values.
                    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(config['random_seed'])
                        model = upstream.SAMAudio(SAMAudioConfig(**deepcopy(effective))).float().cpu().eval()
                if _plan(model, torch) != expected or model.sample_rate != 48000 or model.audio_codec.hop_length != report['hop_length']:
                    raise ValueError('constructed graph differs from verified plan')
                core = {k:v for k,v in expected.items() if not k.startswith('text_encoder.')}
                state = _torch_checkpoint(declaration['sam']['checkpoint_path'], declaration['sam']['checkpoint_sha256'],core,torch)
                text = {k.removeprefix('text_encoder.model.'):v for k,v in expected.items() if k.startswith('text_encoder.model.')}
                t5 = _t5_state(str(Path(t5root)/'model.safetensors'),declaration['t5']['files']['model.safetensors'],text,torch)
                state.update({'text_encoder.model.'+k:v for k,v in t5.items()})
                _geometry(state, expected,torch,'cpu')
                torch.nn.Module.load_state_dict(model,state,strict=True)
                del state,t5
                tokenizer = _local_tokenizer(transformers,declaration)
                # Catch tokenizer/config changes during actual constructor IO.
                verify_manifest(declaration, expected_sha256=config['manifest_sha256'])
                model.text_encoder.tokenizer = _Tokenizer(tokenizer)
                if config['device']=='cuda' and not torch.cuda.is_available():
                    raise RuntimeError('selected CUDA device unavailable')
                model = model.to(config['device']).eval()
                _runtime_closure(declaration)
                self = cls()
                self.model, self.processor = model, SAMAudioProcessor(report['hop_length'],48000)
                self.hop_length, self.config, self.lock = report['hop_length'],dict(config),threading.RLock()
                self.finder = finder
                self.provenance = dict(source_revision=SOURCE_PINS['sam_audio'],
                    runtime_manifest_sha256=report['runtime_manifest_sha256'],sam_manifest_sha256=report['manifest_sha256'],
                    t5_manifest_sha256=report['t5_manifest_sha256'],checkpoint_sha256=declaration['sam']['checkpoint_sha256'],
                    original_config_sha256=report['original_config_sha256'],effective_config_sha256=report['effective_config_sha256'],
                    device=config['device'],precision='float32',predict_spans=False,reranking_candidates=1,
                    ode_steps=config['ode_steps'],ode_method='midpoint',random_seed=config['random_seed'])
                return self
        except BaseException:
            finder.remove()
            raise

    def predict(self, prepared):
        import torch
        with self.lock, offline():
            if self.model is None: raise RuntimeError('SAM native backend closed')
            started = time.monotonic()
            features = prepared.padded_samples//prepared.hop_length
            if features > 512:
                raise ValueError('native inference exceeds SDK512 feature-frame bound')
            batch = self.processor(descriptions=[prepared.prompt],
                audios=[torch.from_numpy(prepared.array()).reshape(1,-1)],
                anchors=[list(prepared.anchors)] if prepared.anchors else None)
            if (tuple(batch.audios.shape)!=(1,1,prepared.sample_count) or batch.sizes.tolist()!=[features] or
                    batch.wav_sizes.tolist()!=[prepared.sample_count] or tuple(batch.audio_pad_mask.shape)!=(1,features) or
                    not batch.audio_pad_mask.all().item()):
                raise ValueError('official processor window geometry mismatch')
            alignment = torch.zeros((1,features),dtype=torch.long)
            ids = [0,3]
            for index,(token,first,last) in enumerate(prepared.feature_spans):
                ids.append(1 if token=='+' else 2);alignment[0,first:last]=index+2
            if batch.anchor_ids.tolist()!=[ids] or not torch.equal(batch.anchor_alignment.cpu(),alignment):
                raise ValueError('official processor anchor quantization mismatch')
            devices = [torch.cuda.current_device()] if self.config['device']=='cuda' else []
            self.model.text_encoder.tokenizer.stats = None
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.random.default_generator.manual_seed(self.config['random_seed'])
                if devices: torch.cuda.manual_seed(self.config['random_seed'])
                result = self.model.separate(batch.to(self.config['device']),predict_spans=False,
                    reranking_candidates=1,ode_opt={'method':'midpoint','options':{'step_size':1/self.config['ode_steps']}})
            converted = []
            for values in (result.target,result.residual):
                if type(values) is not list or len(values)!=1 or type(values[0]) is not torch.Tensor or values[0].dtype!=torch.float32:
                    raise ValueError('official separation must return one float32 tensor per component')
                converted.append([values[0].detach().cpu().numpy().copy()])
            target,residual = validate_outputs(*converted,sample_count=prepared.sample_count,hop_length=prepared.hop_length)
            tokens = self.model.text_encoder.tokenizer.stats
            if tokens is None: raise ValueError('actual upstream tokenizer evidence missing')
            return target.tobytes(),residual.tobytes(),dict(tokens),(time.monotonic()-started)*1000

    def close(self):
        with self.lock:
            self.model = None
            self.finder.remove()
