import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers import diarization as module
from humansense.providers.diarization import NemotronDiarization, MODES, _NativeBackend

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


def frame(seq=0, end=1000, count=16000):
    return InputFrame(np.full(count, .1, np.float32), 'audio', end, sample_rate=16000,
                      sensor=SensorMetadata('fixture', 'mic', seq, channels=('mono',), stream_epoch='e1'))


class Backend:
    def __init__(self, mode='ultra_low_latency'):
        self.mode, self.calls = mode, []
    def infer(self, values, *, first, last, cache):
        self.calls.append((values.copy(), first, last, cache))
        frames = (len(values)//160 if first else (len(values)-256)//160) if last else MODES[self.mode][0]*8
        scores = np.full((frames, 8), .1, np.float32)
        scores[:, 0] = .9
        scores[::2, 1] = .8
        values[:] = 0  # Adapter must not expose its retained overlap to mutation.
        return scores, {'generation': len(self.calls)}


class ContractTests(unittest.TestCase):
    def test_stdlib_discovery_and_config(self):
        code = "from humansense.provider_profiles import capability; import sys; capability('nemotron3.diarization',{'path':'/local'}); assert not {'torch','numpy','transformers'} & sys.modules.keys()"
        subprocess.run([sys.executable, '-S', '-c', code], check=True)
        for options in ({'mode': 'offline'}, {'threshold': True}, {'threshold': float('nan')},
                        {'threshold': 0}, {'device': 'auto'}):
            with self.assertRaises(ValueError):
                NemotronDiarization.capability(**options)
        self.assertEqual(NemotronDiarization.capability().lookahead_ms, 80)
        with patch.object(module, 'version', return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError, '5.23.0'):
                NemotronDiarization.from_local('/absent')


@unittest.skipIf(np is None, 'optional NumPy absent')
class StreamTests(unittest.TestCase):
    def test_all_modes_absolute_chunk_slices_cache_and_final_coverage(self):
        for mode, (chunk, right) in MODES.items():
            backend = Backend(mode)
            provider = NemotronDiarization(backend, mode=mode)
            # Irregular block sizes, ten whole seconds, distinct waveform enables
            # comparison against independent whole-recording absolute slices.
            wave = np.random.default_rng(7).uniform(-.9, .9, 160000).astype(np.float32)
            position, chunks, sequence = 0, [], 0
            sizes = (160, 320, 4960, 16000, 1760)
            while position < len(wave):
                count = min(sizes[sequence % len(sizes)], len(wave)-position)
                f = dataclasses.replace(frame(sequence, (position+count)//16, count), data=wave[position:position+count])
                value = provider.process(f)
                if value:
                    chunks.extend(value['diarization']['chunks'])
                self.assertLess(len(provider._buffer), provider.later_samples)
                position += count
                sequence += 1
            final = provider.finish()['diarization']
            chunks.extend(final['chunks'])
            self.assertTrue(final['final'])
            self.assertIsNone(provider.finish())
            self.assertIsNone(provider._cache)
            self.assertEqual(len(provider._buffer), 0)
            cursor = 0
            for values, first, last, cache in backend.calls:
                start = 0 if first else cursor*160-256
                stop = len(wave) if last else start+(((chunk+right)*8-1)*160+200 if first else (chunk+right)*8*160+400)
                np.testing.assert_array_equal(values, wave[start:stop])
                self.assertEqual(cache is None, first)
                cursor += (len(wave)//160-cursor) if last else chunk*8
            self.assertEqual(chunks[0]['start_ms'], 0)
            self.assertEqual(chunks[-1]['end_ms'], 10000)
            self.assertTrue(all(a['end_ms'] == b['start_ms'] for a, b in zip(chunks, chunks[1:])))
            for item in chunks:
                self.assertEqual(item['activity'][0]['intervals_ms'], [[item['start_ms'], item['end_ms']]])
                self.assertTrue(item['activity'][1]['intervals_ms'])  # Independent overlap, not softmax/argmax.
            with self.assertRaises(RuntimeError):
                provider.process(frame())
            provider.reset()
            self.assertIsNone(provider.process(frame(0, 10, 160)))
            provider.close()

    def test_short_final_whole_frames_empty_finish_and_pcm_forms(self):
        for count in (160, 320, 5120):
            for form in ('float', 'int16', 'bytes'):
                backend = Backend()
                provider = NemotronDiarization(backend)
                f = frame(0, count//16, count)
                data = np.full(count, 16384, np.int16)
                if form == 'bytes':
                    f = dataclasses.replace(f, data=data.astype('<i2').tobytes())
                elif form == 'int16':
                    f = dataclasses.replace(f, data=data[:, None])
                else:
                    f = dataclasses.replace(f, data=np.full(count, .5, np.float32))
                self.assertIsNone(provider.process(f))
                result = provider.finish()['diarization']
                self.assertEqual(result['chunks'][0]['end_ms'], count//16)
                np.testing.assert_array_equal(backend.calls[0][0], np.full(count, .5, np.float32))
                self.assertEqual(backend.calls[0][1:3], (True, True))
        empty = NemotronDiarization(Backend())
        self.assertIsNone(empty.finish())
        self.assertIsNone(empty.finish())

    def test_gaps_owner_changes_and_invalid_input_clear_all_state(self):
        for change in ('gap', 'epoch', 'subject', 'sequence_mode'):
            backend = Backend()
            provider = NemotronDiarization(backend)
            first = provider.process(frame())['diarization']['session_id']
            next_frame = frame(1, 2000)
            if change == 'gap':
                next_frame = frame(2, 2010)
            elif change == 'epoch':
                next_frame = dataclasses.replace(next_frame, sensor=dataclasses.replace(next_frame.sensor, stream_epoch='e2'))
            elif change == 'subject':
                next_frame = dataclasses.replace(next_frame, subject_id='other')
            else:
                next_frame = dataclasses.replace(next_frame, sensor=dataclasses.replace(next_frame.sensor, capture_sequence=99))
            old_calls = len(backend.calls)
            value = provider.process(next_frame)['diarization']
            self.assertNotEqual(first, value['session_id'])
            self.assertIsNone(backend.calls[old_calls][3])
            self.assertEqual(value['chunks'][0]['start_ms'], next_frame.timestamp_ms-1000)
        provider = NemotronDiarization(Backend())
        for bad in (frame(0, 2000), frame(1, 1000), frame(1, 2000, 159),
                    dataclasses.replace(frame(1, 2000), sample_rate=8000),
                    dataclasses.replace(frame(1, 2000), data=np.full(16000, float('nan'), np.float32)),
                    dataclasses.replace(frame(1, 2000), data=np.ma.array(np.zeros(16000, np.float32)))):
            provider.reset()
            provider.process(frame())
            with self.assertRaises(ValueError):
                provider.process(bad)
            self.assertIsNone(provider._cache)
            self.assertEqual(len(provider._buffer), 0)
        provider.close()
        self.assertIsNone(provider.backend)
        with self.assertRaises(RuntimeError):
            provider.process(frame())

    def test_invalid_grid_and_worst_fragmentation_bound(self):
        class Bad(Backend):
            def infer(self, *args, **kw):
                scores, cache = super().infer(*args, **kw)
                scores[0, 0] = float('nan')
                return scores, cache
        provider = NemotronDiarization(Bad())
        with self.assertRaises(ValueError):
            provider.process(frame())
        self.assertIsNone(provider._cache)
        class Alternating(Backend):
            def infer(self, *args, **kw):
                scores, cache = super().infer(*args, **kw)
                scores[:] = .1
                scores[::2] = .9
                return scores, cache
        provider = NemotronDiarization(Alternating())
        end = 2**53-1-2000
        provider.process(frame(0, end, 16000))
        output = provider.process(frame(1, end+1000))
        self.assertLess(len(json.dumps(output).encode()), 65536)
        self.assertEqual(output, provider.spec.validate(output))


@unittest.skipIf(torch is None or np is None, 'optional torch/NumPy absent')
class NativeBoundaryTests(unittest.TestCase):
    def test_local_verified_snapshots_strict_weights_and_processor_framing(self):
        native = types.ModuleType('transformers')
        native.Nemotron3DiarizationConfig = lambda **kw: types.SimpleNamespace(**kw)
        native.Nemotron3DiarizationForAudioFrameClassification = lambda cfg: torch.nn.Linear(2, 2)
        native.NemotronAsrStreamingFeatureExtractor = lambda **kw: kw
        class Processor:
            def __init__(self, extractor, **kw):
                pass
            def set_streaming_mode(self, mode):
                chunk, right = MODES[mode]
                self.num_samples_first_audio_chunk = ((chunk+right)*8-1)*160+200
                self.num_samples_per_audio_chunk = (chunk+right)*8*160+400
                self.num_mel_frames_per_step = chunk*8
            def audio_chunk_start(self, cursor):
                return cursor*160-256
        native.Nemotron3DiarizationProcessor = Processor
        state = {'weight': torch.ones(2, 2), 'bias': torch.ones(2)}
        def load(blob):
            self.assertEqual(blob, b'fixture-weights')
            return state
        safe = types.SimpleNamespace(load=load)
        contents = {'config.json': b'{}', 'processor_config.json': json.dumps(dict(
            feature_extractor={'feature_extractor_type': 'fixture'}, processor_class='fixture')).encode(),
            'model.safetensors': b'fixture-weights'}
        artifacts = {name: (len(blob), hashlib.sha256(blob).hexdigest()) for name, blob in contents.items()}
        versions = {'transformers': '5.23.0', 'torch': '2.8.0', 'safetensors': '0.8.0'}
        with tempfile.TemporaryDirectory(dir='.') as directory, patch.dict(sys.modules, {'transformers': native, 'safetensors.torch': safe}), \
                patch.object(module, 'ARTIFACTS', artifacts), patch.object(module, 'version', side_effect=versions.__getitem__):
            for name, blob in contents.items():
                (Path(directory)/name).write_bytes(blob)
            provider = NemotronDiarization.from_local(directory)
            self.assertTrue(provider._artifacts_verified)
            self.assertTrue(torch.equal(provider.backend.model.weight, torch.ones(2, 2)))
            provider.close()
            del state['bias']
            with self.assertRaisesRegex(ValueError, 'missing/extra'):
                NemotronDiarization.from_local(directory)
            state['bias'] = torch.ones(2, dtype=torch.float64)
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                NemotronDiarization.from_local(directory)
            (Path(directory)/'config.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                NemotronDiarization.from_local(directory)

    def test_native_final_streaming_switch_sigmoid_and_cache_contract(self):
        class Inputs(dict):
            def to(self, device):
                return self
        seen = []
        state = types.SimpleNamespace(embeds=torch.zeros(1, 264, 512), fifo=torch.zeros(1, 264, 512),
            probs=torch.zeros(1, 264, 8), num_cache_frames=0, num_fifo_frames=1)
        def model(**kwargs):
            seen.append(kwargs)
            return types.SimpleNamespace(logits=torch.full((1, 1, 8), 2.), speaker_cache=state)
        backend = _NativeBackend(model, lambda *a, **k: Inputs(input_features=torch.zeros(1, 1, 128)), torch, 'cpu')
        probabilities, cache = backend.infer(np.zeros(160, np.float32), first=True, last=True, cache=None)
        self.assertEqual(seen[0]['num_lookahead_frames'], 0)
        self.assertIs(cache, state)
        np.testing.assert_allclose(probabilities, 1/(1+np.exp(-2)), rtol=1e-6)
        self.assertGreater(float(probabilities.sum()), 1)  # Multi-speaker activity scores.
        state.fifo = torch.zeros(1, 265, 512)
        with self.assertRaises(ValueError):
            backend.infer(np.zeros(160, np.float32), first=True, last=True, cache=None)


@unittest.skipIf(np is None, 'optional NumPy absent')
class ManagedTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_finish_to_mcp_and_worker_reaping(self):
        children = []
        real_spawn = asyncio.create_subprocess_exec
        script = '''
import sys
sys.path.insert(0,'tests')
from test_diarization import Backend
from humansense.providers.diarization import NemotronDiarization
from humansense.provider_worker import main
NemotronDiarization.from_local=classmethod(lambda cls,path,**options:cls(Backend(options.get('mode','ultra_low_latency')),**options))
main()
'''
        async def spawn(*args, **kwargs):
            child = await real_spawn(sys.executable, '-u', '-c', script, **kwargs)
            children.append(child)
            return child
        try:
            with patch('asyncio.create_subprocess_exec', spawn):
                async with HumanSense(consent=True) as human:
                    streams = SensorStreams(human)
                    spec = ProcessProvider.capability('nemotron3.diarization', {'path': '/fixture'})
                    async def create():
                        return await ProcessProvider.open(sys.executable, 'nemotron3.diarization', {'path': '/fixture'})
                    streams.register(spec, create)
                    await streams.open('fixture', 'mic', [spec.id])
                    now = int(time.time()*1000)-100
                    f = dataclasses.replace(frame(0, now, 320), sensor=SensorMetadata('fixture', 'mic', 0, channels=('mono',)))
                    await streams.submit(f)
                    await streams.drain()
                    await streams.finish('fixture', 'mic')
                    output = await MCPServer(human).call_tool('get_observations', {'capability_id': spec.id})
                    result = output['items'][0]['observation']['value']['diarization']
                    self.assertTrue(result['final'])
                    self.assertEqual(result['chunks'][0]['end_ms'], now)
                    self.assertFalse(result['identity_verified'])
            self.assertTrue(children and all(child.returncode is not None for child in children))
        finally:
            for child in children:
                if child.returncode is None:
                    child.kill()
                    await child.wait()


if __name__ == '__main__':
    unittest.main()
