"""Turn-based audio.diarization candidates (pyannoteAI Live-1, offline references) with stand-ins: no network, no key."""
import io
import json
import queue
import time
import unittest
from unittest.mock import patch

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers import slot_sources
from gamst.providers.diarization import SLOTS
from gamst.providers.pyannoteai import PyannoteAI
from gamst.providers.slot_sources import CHUNK, HostedLiveDiarization, OfflineDiarization, _Turns

try:
    import numpy as np
except ImportError:
    np = None

KEYS = {'session_id', 'chunks', 'capture_end_ms', 'final', 'mode', 'threshold', 'slot_count', 'identity_verified',
        'slot_continuity_verified', 'lookahead_ms', 'source_revision', 'checkpoint_sha256', 'artifact_hashes_verified',
        'quality_verified', 'turn_boundaries_final', 'overlap_supported'}


def frame(seq, end_ms, data):
    return InputFrame(data, 'audio', end_ms, 'local-user', 16000,
                      SensorMetadata(source_id='capture', sensor_id='mic', sequence=seq, channels=('mono',)))


class Socket:
    """A Live-1 stand-in: after each 100 ms message it may answer with scripted events (seconds of stream time)."""

    def __init__(self, script):
        self.script, self.sent, self.inbox, self.closed = script, [], queue.Queue(), False

    def send(self, message):
        self.sent.append(message)
        if isinstance(message, bytes):
            seconds = round(sum(isinstance(item, bytes) for item in self.sent) * .1, 1)
            for event in self.script.get(seconds, ()):
                self.inbox.put(json.dumps(event))
        elif json.loads(message) == {'type': 'end_of_stream'}:
            for event in self.script.get('end', ()):
                self.inbox.put(json.dumps(event))
            self.inbox.put(None)

    def recv(self):
        item = self.inbox.get()
        if item is None:
            raise type('ConnectionClosedOK', (Exception,), {})()
        return item

    def close(self):
        self.closed = True


class Client:
    endpoint, model = 'https://api.example.com', 'live-1'

    def __init__(self):
        self.streams, self.booked = 0, []

    def live_stream(self, *, heartbeat_ms):
        self.streams += 1
        return f'wss://stream.example.com/{self.streams}?heartbeat={heartbeat_ms}'

    def streamed(self, seconds):
        self.booked.append(round(seconds, 1))


def settle(socket):
    """Let the provider's reader thread take what the stand-in answered (it reads on its own thread)."""
    for _ in range(1000):
        if socket.inbox.empty():
            break
        time.sleep(.001)
    time.sleep(.005)


def event(kind, at, speaker=None):
    if kind == 'heartbeat':
        return dict(type='heartbeat', data=dict(stream_duration=at, last_predicted_timestamp=at))
    return dict(type=f'diarization_speaker_{kind}', data=dict(timestamp=at, speaker=speaker))


@unittest.skipIf(np is None, 'optional NumPy absent')
class LiveTests(unittest.TestCase):
    def run_stream(self, script, *, seconds=2.0, block=800):
        sockets = []

        def connect(url):
            sockets.append(Socket(script))
            return sockets[-1]

        client = Client()
        provider = HostedLiveDiarization(client, connect=connect)
        records, wave = [], np.zeros(int(16000 * seconds), np.float32)
        for seq, position in enumerate(range(0, len(wave), block)):
            data = wave[position:position + block]
            records.append(provider.process(frame(seq, 5000 + (position + len(data)) // 16, data)))
            settle(sockets[0])
        final = provider.finish()
        return provider, client, sockets, [r for r in records if r], final

    def test_slots_follow_speaker_events_and_commit_only_up_to_the_predicted_time(self):
        script = {.5: [event('start', .2, 'SPEAKER_00')], .8: [event('heartbeat', .6)],
                  1.2: [event('start', .9, 'SPEAKER_01'), event('end', 1.0, 'SPEAKER_00'), event('heartbeat', 1.1)],
                  'end': [event('end', 1.85, 'SPEAKER_01')]}
        provider, client, sockets, records, final = self.run_stream(script, seconds=1.95)
        chunks = [c for r in records for c in r['diarization']['chunks']] + final['diarization']['chunks']
        self.assertEqual(chunks[0]['start_ms'], 5000)
        self.assertTrue(all(a['end_ms'] == b['start_ms'] for a, b in zip(chunks, chunks[1:])))
        self.assertEqual(chunks[-1]['end_ms'], 5000 + 1950)
        spans = {slot: [] for slot in range(SLOTS)}
        for chunk in chunks:
            self.assertEqual(len(chunk['activity']), SLOTS)
            for item in chunk['activity']:
                spans[item['slot']] += item['intervals_ms']

        def merged(items):
            out = []
            for a, b in items:
                if out and a <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], b)
                else:
                    out.append([a, b])
            return out
        self.assertEqual(merged(spans[0]), [[5200, 6000]])
        self.assertEqual(merged(spans[1]), [[5900, 6850]])
        self.assertTrue(all(chunk['end_ms'] <= 5000 + 1100 for r in records for chunk in r['diarization']['chunks']))
        self.assertEqual(set(final['diarization']), KEYS)
        self.assertTrue(final['diarization']['final'])
        socket = sockets[0]
        audio = [item for item in socket.sent if isinstance(item, bytes)]
        self.assertTrue(all(len(item) == CHUNK * 4 for item in audio))      # 100 ms of float32 per message
        self.assertEqual(len(audio), 20)                                     # 1.95 s: the last message is padded
        self.assertEqual(json.loads(socket.sent[-1]), {'type': 'end_of_stream'})
        self.assertTrue(socket.closed)
        self.assertEqual(client.booked, [2.0])

    def test_a_server_error_ends_the_session(self):
        script = {.2: [dict(type='error', message='Invalid chunk size')]}
        sockets = []
        provider = HostedLiveDiarization(Client(), connect=lambda url: sockets.append(Socket(script)) or sockets[-1])
        with self.assertRaisesRegex(RuntimeError, 'Invalid chunk size'):
            for seq in range(20):
                provider.process(frame(seq, 100 * (seq + 1), np.zeros(1600, np.float32)))
                settle(sockets[0])
        self.assertTrue(sockets[0].closed)

    def test_streams_only_by_opt_in_with_a_cost_line(self):
        with self.assertRaisesRegex(PermissionError, 'leaves this device'):
            slot_sources.hosted_live({'eur_per_hour': .198})
        out = io.StringIO()
        client = PyannoteAI(endpoint='https://api.example.com', api_key_env='GAMST_TEST_LIVE_KEY', model='live-1',
                            eur_per_hour=.198, allow_remote=True, out=out)
        with patch.object(PyannoteAI, '_request', return_value={'id': 's', 'url': 'wss://x'}) as request:
            self.assertEqual(client.live_stream(heartbeat_ms=200), 'wss://x')
        self.assertEqual(request.call_args.args, ('POST', '/v1/live', {'heartbeat': {'enabled': True, 'interval': 200}}))
        self.assertIn('0.0011 EUR', out.getvalue())               # the 20 s minimum at 0.198 EUR/h
        client.streamed(54.4)
        self.assertIn('0.0030 EUR', out.getvalue())
        from gamst.provider_profiles import capability
        spec = capability('pyannoteai.live.diarization', {'eur_per_hour': .198, 'model': 'live-1', 'allow_remote': True})
        self.assertEqual(spec.output_name, 'audio.speaker_activity')
        with self.assertRaises(ValueError):
            capability('pyannoteai.live.diarization', {'model': 'live-1'})     # no price, no cost line


class Batch:
    revision, sha256, verified = 'r1', 'a' * 64, True

    def __init__(self):
        self.seen = None

    def diarize(self, samples):
        self.seen = len(samples)
        return [(0.5, 1.5, 'A'), (1.2, 2.0, 'B'), (2.5, 3.0, 'A')]


@unittest.skipIf(np is None, 'optional NumPy absent')
class OfflineTests(unittest.TestCase):
    def test_one_final_record_for_the_whole_session(self):
        backend = Batch()
        provider = OfflineDiarization(backend, name='test.offline', version=['v'], license='test')
        for seq in range(30):
            self.assertIsNone(provider.process(frame(seq, 1000 + 100 * (seq + 1), np.zeros(1600, np.float32))))
        value = provider.finish()['diarization']
        self.assertEqual(backend.seen, 48000)
        self.assertEqual(set(value), KEYS)
        self.assertTrue(value['final'])
        (chunk,) = value['chunks']
        self.assertEqual((chunk['start_ms'], chunk['end_ms']), (1000, 4000))
        self.assertEqual(chunk['activity'][0]['intervals_ms'], [[1500, 2500], [3500, 4000]])
        self.assertEqual(chunk['activity'][1]['intervals_ms'], [[2200, 3000]])
        self.assertEqual(chunk['activity'][0]['max_probability'], 1.)
        self.assertEqual(value['checkpoint_sha256'], 'a' * 64)

    def test_hosted_batch_and_noncommercial_opt_in(self):
        out = io.StringIO()
        client = PyannoteAI(endpoint='https://api.example.com', model='precision-3', eur_per_hour=.112,
                            allow_remote=True, out=out)
        with patch.object(PyannoteAI, 'upload', return_value='media://x'), \
                patch.object(PyannoteAI, '_request', return_value={'jobId': 'j'}) as request, \
                patch.object(PyannoteAI, '_job', return_value={'diarization': [dict(speaker='S', start=0, end=1)]}):
            turns = slot_sources.HostedBatch(client).diarize(np.zeros(16000, np.float32))
        self.assertEqual(turns, [(0., 1., 'S')])
        self.assertEqual(request.call_args.args[1:], ('/v1/diarize', {'url': 'media://x', 'model': 'precision-3'}))
        self.assertIn('billed as 20.0 s', out.getvalue())
        from gamst.provider_profiles import capability
        with self.assertRaisesRegex(PermissionError, 'CC BY-NC'):
            capability('offline.diarization', {'backend': 'diarizen', 'path': '/m'})
        self.assertEqual(capability('offline.diarization', {'backend': 'diarizen', 'path': '/m',
                                                            'allow_noncommercial': True}).id, 'audio.diarizen.diarization')
        with self.assertRaisesRegex(PermissionError, 'CC BY-NC'):
            slot_sources.DiariZenBackend.from_local('/m')

    def test_turns_are_cut_into_chunks_once(self):
        turns = _Turns()
        turns.start('A', 100)
        self.assertEqual(turns.chunk(0, 300)['activity'][0]['intervals_ms'], [[100, 300]])
        turns.end('A', 450)
        chunk = turns.chunk(300, 600)
        self.assertEqual(chunk['activity'][0]['intervals_ms'], [[300, 450]])
        self.assertEqual(chunk['activity'][0]['mean_probability'], .5)
        self.assertEqual(turns.chunk(600, 900)['activity'][0]['intervals_ms'], [])
        for index in range(SLOTS):           # 'A' holds slot 0 already
            turns.start(f'S{index}', 900)
        self.assertEqual(turns.dropped, 1)                    # a ninth label has no slot


if __name__ == '__main__':
    unittest.main()
