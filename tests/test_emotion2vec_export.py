"""Preparation contracts and arithmetic; no trained checkpoint/export claimed."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from humansense import emotion2vec_export as export

try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


class PreparationContractTests(unittest.TestCase):
    def test_stdlib_help_and_discovery(self):
        code = "import humansense.emotion2vec_export; import sys; assert not {'torch','numpy','funasr','onnx','onnxruntime'} & sys.modules.keys()"
        subprocess.run([sys.executable, '-S', '-c', code], check=True)
        result = subprocess.run([sys.executable, '-S', '-m', 'humansense.emotion2vec_export', '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--checkpoint', result.stdout)

    def test_verified_snapshots_and_reject_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('model', 'config', 'tokens')]
            payloads = [b'checkpoint', b'config', b'tokens']
            for path, payload in zip(paths, payloads):
                path.write_bytes(payload)
            with patch.multiple(export, SOURCE_SIZE=len(payloads[0]),
                                SOURCE_SHA256=hashlib.sha256(payloads[0]).hexdigest(),
                                CONFIG_SHA256=hashlib.sha256(payloads[1]).hexdigest(),
                                TOKENS_SHA256=hashlib.sha256(payloads[2]).hexdigest()):
                snapshots = export.read_sources(*paths)
                self.assertEqual(snapshots, tuple(payloads))
                paths[0].write_bytes(b'changed')
                self.assertEqual(snapshots[0], payloads[0])
                with self.assertRaises(ValueError):
                    export.read_sources(*paths)
                # YAML is checked before trying to open the much larger checkpoint.
                paths[1].write_bytes(b'eval injection')
                with self.assertRaises(ValueError):
                    export.read_sources(Path(directory) / 'absent', *paths[1:])

    def test_versions_wrapper_ambiguity_and_bounded_blob(self):
        with patch.object(export, 'version', side_effect=lambda key: export.RUNTIME.get(key, '2.2.6')):
            self.assertEqual(export.checked_runtime()['torch'], '2.8.0')
        with patch.object(export, 'version', return_value='unknown'):
            with self.assertRaises(RuntimeError):
                export.checked_runtime()
        self.assertEqual(export.select_state({'model': OrderedDict(weight=1)})[1], ['model'])
        for state in ({}, [], {'model': {}, 'state_dict': {}}, {1: 1},
                      {'model': {'model': {'model': {'model': {'weight': 1}}}}}):
            with self.assertRaises(ValueError):
                export.select_state(state)
        with patch.object(export, 'GRAPH_LIMIT', 4):
            buffer = export.LimitedBuffer()
            buffer.write(b'1234')
            with self.assertRaises(ValueError):
                buffer.write(b'5')

    def test_existing_outputs_untouched_and_failure_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            existing = parent / 'existing'
            existing.mkdir()
            (existing / 'report.json').write_text('user data')
            with patch.object(export, 'checked_runtime', side_effect=RuntimeError('missing optional runtime')) as runtime:
                with self.assertRaises(FileExistsError):
                    export.prepare('x', 'y', 'z', existing)
                runtime.assert_not_called()
                self.assertEqual((existing / 'report.json').read_text(), 'user data')
                fresh = parent / 'fresh'
                with self.assertRaisesRegex(RuntimeError, 'missing optional'):
                    export.prepare('x', 'y', 'z', fresh)
                self.assertEqual(sorted(p.name for p in fresh.iterdir()), ['failure.json'])
                self.assertEqual(json.loads((fresh / 'failure.json').read_text())['status'], 'failed')


def tiny_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(768, 9)
            # Deterministic fixture, unrelated to trained emotion2vec features.
            with torch.no_grad():
                self.proj.weight.copy_(torch.arange(6912).reshape(9, 768) / 1000000)
                self.proj.bias.copy_(torch.arange(9) / 100)
            self.register_buffer('offset', torch.tensor(.25))

        def forward(self, source, **kwargs):
            if kwargs != dict(padding_mask=None, mask=False, features_only=True, remove_extra_tokens=True):
                raise AssertionError(kwargs)
            return {'x': (source[:, ::320, None] + self.offset).expand(-1, -1, 768)}
    return Model().eval()


@unittest.skipIf(torch is None, 'optional torch/NumPy absent')
class TensorPreparationTests(unittest.TestCase):
    def test_complete_mapping_and_missing_ambiguous_shape_dtype_nan(self):
        model = tiny_model()
        state = {'module.d2v_model.' + key: value.clone() for key, value in model.state_dict().items()}
        state['ema_training_only'] = {'not_loaded': True}
        report = export.load_complete_state(model, {'model': state}, torch)
        self.assertEqual(report['unused_source_keys'], ['ema_training_only'])
        self.assertEqual(len(report['destination_to_source']), 3)
        for change in ('missing', 'ambiguous', 'shape', 'dtype', 'nan'):
            invalid = dict(state)
            key = 'module.d2v_model.proj.weight'
            if change == 'missing':
                del invalid[key]
            elif change == 'ambiguous':
                invalid['proj.weight'] = invalid[key]
            elif change == 'shape':
                invalid[key] = torch.zeros(1)
            elif change == 'dtype':
                invalid[key] = invalid[key].double()
            else:
                invalid[key] = invalid[key].clone()
                invalid[key][0, 0] = float('nan')
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with self.subTest(change=change), self.assertRaises(ValueError):
                export.load_complete_state(model, invalid, torch)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[key], value))

    def test_head_and_normalization_match_native_torch(self):
        model = tiny_model()
        head = export.extract_head(model, torch)
        self.assertEqual(head['labels'], list(export.LABELS))
        self.assertEqual(np.asarray(head['weight']).shape, (9, 768))
        features = export.feature_model(model, torch)
        for _, samples in export.parity_cases(np):
            x = torch.from_numpy(samples)
            normalized = torch.nn.functional.layer_norm(x, x.shape)
            expected = (normalized[:, ::320, None] + .25).expand(-1, -1, 768)
            self.assertTrue(torch.allclose(features(x), expected, atol=1e-6, rtol=1e-6))
        with torch.no_grad():
            model.proj.bias[0] = float('inf')
        with self.assertRaises(ValueError):
            export.extract_head(model, torch)

    def test_multilength_parity_catches_shape_feature_and_head_regressions(self):
        model = tiny_model()
        features = export.feature_model(model, torch)
        head = export.extract_head(model, torch)
        class Session:
            fault = None
            def run(self, outputs, inputs):
                actual = features(torch.from_numpy(inputs['input'])).detach().numpy().copy()
                if self.fault == 'shape' and inputs['input'].shape[1] == 64000:
                    actual = actual[:, :-1]
                elif self.fault == 'feature':
                    actual += .01
                elif self.fault == 'nan':
                    actual[0, 0, 0] = float('nan')
                return [actual]
        session = Session()
        rows = export.verify_parity(model, features, session, head, torch, np)
        self.assertEqual(len(rows), 7)
        self.assertEqual([row['samples'] for row in rows[:4]], [16000, 31984, 32000, 64000])
        for fault in ('shape', 'feature', 'nan'):
            session.fault = fault
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                export.verify_parity(model, features, session, head, torch, np)
        session.fault = None
        head['bias'][0] += .1
        with self.assertRaisesRegex(ValueError, 'probability'):
            export.verify_parity(model, features, session, head, torch, np)

    def test_prepare_uses_weights_only_and_publishes_matching_hashes_after_parity(self):
        model = tiny_model()
        buffer = io.BytesIO()
        torch.save({'model': model.state_dict()}, buffer)
        tokens = '\n'.join(export.LABELS[:-1] + ('<unk>',)).encode()
        source = (buffer.getvalue(), b'model: Emotion2vec\nmodel_conf: {normalize: true}', tokens)
        options = types.SimpleNamespace
        features = export.feature_model(model, torch)
        session = types.SimpleNamespace(run=lambda _, inputs: [features(torch.from_numpy(inputs['input'])).detach().numpy()])
        ort = types.SimpleNamespace(SessionOptions=options, InferenceSession=lambda *args, **kw: session)
        modules = {'onnx': types.ModuleType('onnx'), 'onnxruntime': ort,
                   'funasr.models.emotion2vec.model': types.SimpleNamespace(Emotion2vec=lambda **kw: model),
                   'yaml': types.SimpleNamespace(safe_load=lambda _: {'model': 'Emotion2vec', 'model_conf': {'normalize': True}})}
        original_load = torch.load
        def load(stream, **kw):
            self.assertEqual(kw, {'weights_only': True, 'map_location': 'cpu'})
            self.assertIsInstance(stream, io.BytesIO)
            return original_load(stream, **kw)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
                patch.object(export, 'checked_runtime', return_value={'fixture': 'not-native-export'}), \
                patch.object(export, 'read_sources', return_value=source), \
                patch.object(export, 'serialize_graph', return_value=b'graph-double'), \
                patch.object(torch, 'load', side_effect=load):
            output = Path(directory) / 'prepared'
            report = export.prepare('checkpoint', 'config', 'tokens', output)
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['model_sha256'], hashlib.sha256((output / 'seed.onnx').read_bytes()).hexdigest())
            self.assertEqual(report['head_sha256'], hashlib.sha256((output / 'head.json').read_bytes()).hexdigest())
            self.assertEqual(json.loads((output / 'report.json').read_text()), report)
            self.assertFalse(report['real_speech_quality_verified'])
            failed = Path(directory) / 'failed'
            with patch.object(export, 'verify_parity', side_effect=ValueError('parity rejected')):
                with self.assertRaisesRegex(ValueError, 'parity rejected'):
                    export.prepare('checkpoint', 'config', 'tokens', failed)
            self.assertEqual([p.name for p in failed.iterdir()], ['failure.json'])


if __name__ == '__main__':
    unittest.main()
