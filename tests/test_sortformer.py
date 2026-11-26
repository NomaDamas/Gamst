"""Streaming Sortformer adapter: framing against NeMo's own streaming loop, with model doubles (no NeMo, no weights)."""
import math
import tempfile
import unittest
from pathlib import Path

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers import sortformer as module
from gamst.providers.diarization import NemotronDiarization, SLOTS
from gamst.providers.sortformer import MODES, SortformerDiarization

try:
    import numpy as np
except ImportError:
    np = None


def features(samples):
    """A local feature like log-mel: frame j summarises samples [160j-256, 160j+256), zero outside; L//160 frames."""
    padded = np.concatenate((np.zeros(256, np.float32), samples, np.zeros(256, np.float32)))
    count = len(samples) // 160
    return np.stack([np.array([padded[160 * j:160 * j + 512].mean(), np.abs(padded[160 * j:160 * j + 512]).max()],
                              np.float32) for j in range(count)], axis=1) if count else np.zeros((2, 0), np.float32)


class Backend:
    """Records every chunk the provider feeds; slot scores are a pure function of the chunk."""

    def __init__(self):
        self.calls = []

    def features(self, samples):
        return features(samples)

    def initial_state(self):
        return {'steps': 0}

    def step(self, feats, state, *, left, right):
        self.calls.append((feats.copy(), left, right))
        frames = feats.shape[1] - left - right
        count = math.ceil(frames / 8)
        body = feats[:, left:left + frames]
        values = np.array([[1 / (1 + math.exp(-50 * body[0, 8 * k:8 * k + 8].mean())), .9, .1, .0]
                           for k in range(count)], np.float32)
        return values, dict(steps=state['steps'] + 1)


def reference(samples, chunk, right):
    """NeMo's streaming_feat_loader over the whole capture's features (sortformer_modules.py, NeMo 3.0.0)."""
    feats, calls, stt, end = features(samples), [], 0, 0
    total = feats.shape[1]
    while end < total:
        left = min(8, stt)
        end = min(stt + chunk * 8, total)
        r = min(right * 8, total - end)
        calls.append((feats[:, stt - left:end + r], left, r))
        stt = end
    return calls


def frame(seq, end_ms, data):
    return InputFrame(data, 'audio', end_ms, 'local-user', 16000,
                      SensorMetadata(source_id='capture', sensor_id='mic', sequence=seq, channels=('mono',)))


@unittest.skipIf(np is None, 'optional NumPy absent')
class SortformerTests(unittest.TestCase):
    def test_every_chunk_is_the_one_nemo_feeds_from_the_whole_capture(self):
        wave = np.random.default_rng(3).uniform(-.8, .8, 16000 * 7 + 480).astype(np.float32)
        for mode in ('ultra_low_latency', 'low_latency'):
            for block in (160, 1600, 4960):
                backend = Backend()
                provider = SortformerDiarization(backend, mode=mode)
                chunks, seq = [], 0
                for position in range(0, len(wave), block):
                    data = wave[position:position + block]
                    value = provider.process(frame(seq, (position + len(data)) // 16, data))
                    seq += 1
                    if value:
                        chunks += value['diarization']['chunks']
                    self.assertLess(len(provider._buffer), 16000 * 2)      # bounded: context + margin only
                final = provider.finish()['diarization']
                chunks += final['chunks']
                expected = reference(wave, *MODES[mode][:2])
                self.assertEqual(len(backend.calls), len(expected))
                for (got, left, right), (want, want_left, want_right) in zip(backend.calls, expected):
                    self.assertEqual((left, right), (want_left, want_right))
                    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
                self.assertEqual(chunks[0]['start_ms'], 0)
                self.assertTrue(all(a['end_ms'] == b['start_ms'] for a, b in zip(chunks, chunks[1:])))
                self.assertTrue(final['final'])

    def test_the_record_is_the_one_every_diarization_candidate_emits(self):
        backend = Backend()
        provider = SortformerDiarization(backend, mode='ultra_low_latency')
        wave = np.full(16000, .5, np.float32)
        value = provider.process(frame(0, 1000, wave))['diarization']
        chunk = value['chunks'][0]
        self.assertEqual(chunk['frame_ms'], 80)
        self.assertEqual([item['slot'] for item in chunk['activity']], list(range(SLOTS)))
        self.assertTrue(chunk['activity'][1]['intervals_ms'])          # independent sigmoid slots, overlap allowed
        self.assertEqual([item['intervals_ms'] for item in chunk['activity'][4:]], [[]] * 4)
        self.assertEqual(value['lookahead_ms'], 80)
        nemotron = NemotronDiarization.capability()
        self.assertEqual(provider.spec.output_name, nemotron.output_name)
        keys = {'session_id', 'chunks', 'capture_end_ms', 'final', 'mode', 'threshold', 'slot_count', 'identity_verified',
                'slot_continuity_verified', 'lookahead_ms', 'source_revision', 'checkpoint_sha256',
                'artifact_hashes_verified', 'quality_verified', 'turn_boundaries_final', 'overlap_supported'}
        self.assertEqual(set(value), keys)
        self.assertFalse(value['artifact_hashes_verified'])            # a double never claims the real weights

    def test_a_gap_starts_a_new_session(self):
        provider = SortformerDiarization(Backend(), mode='ultra_low_latency')
        provider.process(frame(0, 1000, np.zeros(16000, np.float32)))
        session = provider._session
        provider.process(frame(1, 3000, np.zeros(16000, np.float32)))     # 1 s of capture is missing
        self.assertNotEqual(provider._session, session)
        self.assertEqual(provider._origin, 2000)

    def test_capability_and_local_load_checks(self):
        with self.assertRaises(ValueError):
            SortformerDiarization.capability(mode='offline')
        self.assertEqual(SortformerDiarization.capability(mode='low_latency').lookahead_ms, 560)
        from gamst.provider_profiles import capability
        self.assertEqual(capability('sortformer.diarization', {'path': '/m'}).id, 'audio.sortformer.diarization')
        with self.assertRaises(ValueError):
            capability('sortformer.diarization', {'path': '/m', 'chunk': 3})
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, module.FILE).write_bytes(b'not the model')
            with self.assertRaises((RuntimeError, ValueError)):     # wrong packages here, or the hash
                SortformerDiarization.from_local(directory)


if __name__ == '__main__':
    unittest.main()
