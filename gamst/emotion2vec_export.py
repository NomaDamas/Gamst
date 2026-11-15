"""Explicit, local-only preparation for the seed vocal-expression consumer.

Importing this module does not import any optional native runtime. Run preparation
in a separate selected environment; it is not part of capture or inference.
"""
import argparse
from collections import OrderedDict
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform

from .providers.vocal_expression import LABELS, SOURCE_SHA256, _read

REVISION = '7be9a30d52bdc793cc24e6f7d25358d6648927c1'
SOURCE_SIZE = 1118245358
CONFIG_SHA256 = '50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e'
TOKENS_SHA256 = '654625c18ef0f0583441a3b811c20904f2cf30febb716f32e14810b8e6bb715c'
RUNTIME = {'torch': '2.8.0', 'funasr': '1.4.26', 'onnx': '1.18.0',
           'onnxruntime': '1.23.2', 'omegaconf': '2.3.0', 'PyYAML': '6.0.3'}
GRAPH_LIMIT = 512 * 1024 * 1024
FEATURE_ATOL = 2e-4
FEATURE_RTOL = 2e-4
SCORE_ATOL = 2e-5


def read_sources(checkpoint, config, tokens):
    """Snapshot verified bytes before deserialization, including eval-bearing YAML."""
    config_bytes = _read(config, 3170, CONFIG_SHA256)
    token_bytes = _read(tokens, 120, TOKENS_SHA256)
    source_bytes = _read(checkpoint, SOURCE_SIZE, SOURCE_SHA256)
    if len(source_bytes) != SOURCE_SIZE:
        raise ValueError('source checkpoint size mismatch')
    return source_bytes, config_bytes, token_bytes


def checked_runtime():
    actual = {name: version(name) for name in RUNTIME}
    for name, expected in RUNTIME.items():
        if actual[name].split('+')[0] != expected:
            raise RuntimeError(f'preparation requires {name}=={expected}; found {actual[name]}')
    actual['numpy'] = version('numpy')
    return actual


def select_state(checkpoint):
    """Allow conventional state wrappers, rejecting ambiguous alternatives."""
    wrappers = []
    state = checkpoint
    for _ in range(4):
        if type(state) not in (dict, OrderedDict):
            raise ValueError('checkpoint state must be a plain mapping')
        present = [key for key in ('state_dict', 'model_state_dict', 'model') if key in state]
        if not present:
            if not state or len(state) > 10000 or any(type(k) is not str for k in state):
                raise ValueError('invalid or oversized checkpoint state')
            return state, wrappers
        if len(present) != 1:
            raise ValueError('ambiguous checkpoint state wrappers')
        wrappers.append(present[0])
        state = state[present[0]]
    raise ValueError('too many checkpoint wrappers')


def load_complete_state(model, checkpoint, torch):
    """Every destination must have exactly one finite, same-shape/dtype source.

    Training-only entries may remain, but their names are recorded. No missing
    tensor is filled from the randomly initialized destination model.
    """
    state, wrappers = select_state(checkpoint)
    destination = model.state_dict()
    mapped, mapping = {}, {}
    prefixes = ('', 'd2v_model.', 'module.', 'module.d2v_model.')
    for name, expected in destination.items():
        candidates = [prefix + name for prefix in prefixes if prefix + name in state]
        if len(candidates) != 1:
            raise ValueError(f'missing or ambiguous checkpoint tensor: {name}')
        key = candidates[0]
        value = state[key]
        if (type(value) is not torch.Tensor or value.layout != torch.strided or
                value.device.type != 'cpu' or value.shape != expected.shape or
                value.dtype != expected.dtype or not torch.isfinite(value).all().item()):
            raise ValueError(f'incompatible or nonfinite checkpoint tensor: {key}')
        mapped[name], mapping[name] = value, key
    if not mapped or len(set(mapping.values())) != len(mapping):
        raise ValueError('empty or reused checkpoint tensor mapping')
    model.load_state_dict(mapped, strict=True)
    return {'wrappers': wrappers, 'destination_to_source': mapping,
            'unused_source_keys': sorted(set(state) - set(mapping.values()))}


def extract_head(model, torch):
    state = model.state_dict()
    weight, bias = state.get('proj.weight'), state.get('proj.bias')
    for tensor, shape in ((weight, (9, 768)), (bias, (9,))):
        if (type(tensor) is not torch.Tensor or tuple(tensor.shape) != shape or
                tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item() or
                torch.max(torch.abs(tensor)).item() > 1e6):
            raise ValueError('expected bounded float32 seed projection9x768 and bias9')
    return {'source_checkpoint_sha256': SOURCE_SHA256, 'labels': list(LABELS),
            'weight': weight.tolist(), 'bias': bias.tolist()}


def feature_model(model, torch):
    """Pinned export semantics without replacing the source model's forward."""
    class Features(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source = model

        def forward(self, samples):
            mean = samples.mean(dim=1, keepdim=True)
            variance = samples.var(dim=1, keepdim=True, unbiased=False)
            normalized = (samples - mean) / torch.sqrt(variance + 1e-5)
            return self.source(source=normalized, padding_mask=None, mask=False,
                               features_only=True, remove_extra_tokens=True)['x']
    return Features().eval()


class LimitedBuffer(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > GRAPH_LIMIT:
            raise ValueError('feature graph exceeds512MiB')
        return super().write(data)


def serialize_graph(features, torch, onnx):
    """Fixed batch1; dynamic sample/frame axes. Keep every tensor in the blob."""
    buffer = LimitedBuffer()
    with torch.no_grad():
        torch.onnx.export(features, (torch.zeros(1, 32000),), buffer,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {1: 'samples'}, 'output': {1: 'frames'}},
                          opset_version=17, dynamo=False, external_data=False)
    graph = onnx.load_model_from_string(buffer.getvalue())
    # Inspect nested TensorProto fields too, not only top-level initializers.
    def check_embedded(message):
        if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
            if message.external_data or message.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError('external tensor data is not supported')
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                if field.label == field.LABEL_REPEATED:
                    for child in value:
                        check_embedded(child)
                else:
                    check_embedded(value)
    check_embedded(graph)
    onnx.helper.set_model_props(graph, {
        'gamst.export_contract': 'emotion2vec_seed_features_v1',
        'gamst.source_checkpoint_sha256': SOURCE_SHA256,
    })
    onnx.checker.check_model(graph, full_check=True)
    payload = graph.SerializeToString()
    if not 0 < len(payload) <= GRAPH_LIMIT:
        raise ValueError('feature graph exceeds512MiB')
    return payload


def parity_cases(np):
    rng = np.random.default_rng(20261114)
    for count in (16000, 31984, 32000, 64000):
        yield f'noise-{count}', rng.uniform(-.7, .7, (1, count)).astype(np.float32)
    yield 'silence-16000', np.zeros((1, 16000), np.float32)
    yield 'constant-32000', np.full((1, 32000), .125, np.float32)
    yield 'tone-64000', (.3 * np.sin(np.arange(64000) * (2 * np.pi * 440 / 16000))).astype(np.float32)[None]


def verify_parity(model, features, session, head, torch, np):
    """Fail closed on dynamic-shape, normalization, feature or head differences."""
    weight = np.asarray(head['weight'], dtype=np.float64)
    bias = np.asarray(head['bias'], dtype=np.float64)
    rows = []
    for name, samples in parity_cases(np):
        with torch.no_grad():
            x = torch.from_numpy(samples.copy())
            # Native inference uses layer_norm, unlike the export's mean/variance formula.
            normalized = torch.nn.functional.layer_norm(x, x.shape)
            native = model(source=normalized, padding_mask=None, mask=False,
                           features_only=True, remove_extra_tokens=True)['x']
            reference = features(x)
            native_scores = torch.softmax(model.proj(native.mean(dim=1)), dim=-1).numpy()[0]
        actual = session.run(['output'], {'input': samples.copy()})[0]
        expected, upstream = reference.numpy(), native.numpy()
        for value in (actual, expected, upstream):
            if (type(value) is not np.ndarray or value.dtype != np.float32 or value.ndim != 3 or
                    value.shape[0] != 1 or not 1 <= value.shape[1] <= 256 or value.shape[2] != 768 or
                    not np.isfinite(value).all() or np.max(np.abs(value)) > 1e6):
                raise ValueError(f'{name}: invalid feature grid')
        if actual.shape != expected.shape or expected.shape != upstream.shape:
            raise ValueError(f'{name}: dynamic feature shape mismatch')
        if (not np.allclose(actual, expected, atol=FEATURE_ATOL, rtol=FEATURE_RTOL) or
                not np.allclose(actual, upstream, atol=FEATURE_ATOL, rtol=FEATURE_RTOL)):
            raise ValueError(f'{name}: native/export/ORT feature parity failed')
        logits = weight @ actual[0].mean(axis=0, dtype=np.float64) + bias
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        if (native_scores.shape != (9,) or not np.isfinite(native_scores).all() or
                not np.allclose(probabilities, native_scores, atol=SCORE_ATOL, rtol=0)):
            raise ValueError(f'{name}: native/consumer probability parity failed')
        rows.append({'case': name, 'samples': samples.shape[1], 'frames': actual.shape[1],
                     'input_sha256': hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest(),
                     'export_feature_max_abs_error': float(np.max(np.abs(actual - expected))),
                     'native_feature_max_abs_error': float(np.max(np.abs(actual - upstream))),
                     'probability_max_abs_error': float(np.max(np.abs(probabilities - native_scores)))})
    return rows


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def _write_new(path, payload):
    with path.open('xb') as output:
        output.write(payload)


def prepare(checkpoint, config, tokens, output_dir):
    """Never download, overwrite, delete, or relax failed native/parity checks."""
    output = Path(output_dir).expanduser()
    output.mkdir(parents=False, exist_ok=False)
    try:
        runtime = checked_runtime()
        source, config_bytes, token_bytes = read_sources(checkpoint, config, tokens)
        import torch
        import numpy as np
        import yaml
        import onnx
        import onnxruntime as ort
        # This explicit preparation process imports FunASR, whose root imports
        # many modules. The live SDK/ORT consumer never imports it.
        from funasr.models.emotion2vec.model import Emotion2vec
        parsed = yaml.safe_load(config_bytes)
        labels = [line.split('/')[-1] for line in token_bytes.decode('utf-8').splitlines()]
        labels = ['unknown' if label == '<unk>' else label for label in labels]
        if labels != list(LABELS) or parsed['model'] != 'Emotion2vec' or parsed['model_conf']['normalize'] is not True:
            raise ValueError('unexpected seed labels or normalization configuration')
        model = Emotion2vec(model_conf=parsed['model_conf'], vocab_size=9).cpu().eval()
        # No unsafe pickle fallback, allowlisting or upstream partial-load helper.
        state = torch.load(io.BytesIO(source), map_location='cpu', weights_only=True)
        del source
        mapping = load_complete_state(model, state, torch)
        del state
        head = extract_head(model, torch)
        head_bytes = _json_bytes(head)
        if len(head_bytes) > 512 * 1024:
            raise ValueError('projection head exceeds512KiB')
        features = feature_model(model, torch)
        blob = serialize_graph(features, torch, onnx)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(blob, sess_options=options, providers=['CPUExecutionProvider'])
        rows = verify_parity(model, features, session, head, torch, np)
        report = {'schema': 'gamst.emotion2vec-export.v1', 'status': 'passed',
                  'source_revision': REVISION, 'source_checkpoint_sha256': SOURCE_SHA256,
                  'source_config_sha256': CONFIG_SHA256, 'source_tokens_sha256': TOKENS_SHA256,
                  'source_size': SOURCE_SIZE, 'runtime': runtime, 'python': platform.python_version(),
                  'platform': platform.platform(), 'opset': 17, 'dynamo': False,
                  'mapping': mapping, 'synthetic_parity': rows,
                  'tolerances': {'feature_atol': FEATURE_ATOL, 'feature_rtol': FEATURE_RTOL,
                                 'probability_atol': SCORE_ATOL, 'probability_rtol': 0},
                  'model_sha256': hashlib.sha256(blob).hexdigest(),
                  'head_sha256': hashlib.sha256(head_bytes).hexdigest(),
                  'real_speech_quality_verified': False, 'live_latency_verified': False,
                  'weight_license_verified': False, 'acceptance_issue': 111}
        _write_new(output / 'seed.onnx', blob)
        _write_new(output / 'head.json', head_bytes)
        # A completed success report is the commit marker; partial artifacts are not ready.
        _write_new(output / 'report.json', _json_bytes(report))
        return report
    except Exception as exc:
        _write_new(output / 'failure.json', _json_bytes({'status': 'failed',
                   'error_type': type(exc).__name__, 'error': str(exc), 'acceptance_issue': 111}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='explicit local model.pt')
    parser.add_argument('--config', required=True, help='pinned local config.yaml')
    parser.add_argument('--tokens', required=True, help='pinned local tokens.txt')
    parser.add_argument('--output-dir', required=True, help='new directory; parent must already exist')
    args = parser.parse_args()
    report = prepare(args.checkpoint, args.config, args.tokens, args.output_dir)
    print(json.dumps({key: report[key] for key in ('status', 'model_sha256', 'head_sha256')}))


if __name__ == '__main__':
    main()
