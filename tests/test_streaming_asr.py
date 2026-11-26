"""Streaming recognisers: capture-time word times, commits, the aligner lane, and stand-in WebSocket servers.

No paid call: the hosted sessions talk to a local stand-in that speaks the documented protocol (OpenAI Realtime
transcription, ElevenLabs Scribe Realtime) and answers late, so a word's capture time must not move with the delay.
"""
import asyncio
import base64
import json
import time
import unittest

from gamst.plugins import InputFrame

try:
    import numpy as np
except ImportError:      # pragma: no cover
    np = None
try:
    import websockets  # noqa: F401
except ImportError:      # pragma: no cover
    websockets = None

BASE = 50_000_000      # capture ms of the stream's first sample in these tests


def frames(seconds, *, loud=(), chunk=1600, gap_at=None, gap_ms=0):
    """100 ms frames of quiet noise with louder spans ``loud`` [(start_s, end_s)]; an optional capture gap."""
    rng = np.random.default_rng(0)
    out, at = [], BASE
    for index in range(int(seconds * 16000 / chunk)):
        start = index * chunk / 16000
        level = .3 if any(a <= start < b for a, b in loud) else .003
        if gap_at is not None and abs(start - gap_at) < 1e-6:
            at += gap_ms
        at += round(1000 * chunk / 16000)
        out.append(InputFrame((rng.standard_normal(chunk) * level).astype(np.float32), 'audio', at, sample_rate=16000))
    return out


@unittest.skipIf(np is None, 'numpy')
class ClockTest(unittest.TestCase):
    def test_stream_seconds_become_capture_times_across_a_gap(self):
        from gamst.providers.streaming_asr import CaptureClock
        clock = CaptureClock(16000)
        clock.push(16000, 10_000, wall_ms=1)        # 9.0-10.0 s captured
        clock.push(16000, 15_000, wall_ms=2)        # 14.0-15.0 s (a 4 s gap was glued out of the stream)
        self.assertEqual(clock.seconds_ms(.5), 9_500)
        self.assertEqual(clock.seconds_ms(1.25), 14_250)
        self.assertEqual(clock.seconds_ms(.25, origin=16000), 14_250)
        self.assertEqual(clock.arrival_ms(9_800), 1)
        self.assertEqual(clock.arrival_ms(14_900), 2)

    def test_the_chunked_resampler_matches_one_pass(self):
        from gamst.providers.audio_ring import resample
        from gamst.providers.streaming_asr import StreamResampler
        signal = np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000).astype(np.float32)
        streamed = StreamResampler(16000, 24000)
        pieces = [streamed.push(signal[i:i + 1600]) for i in range(0, 16000, 1600)] + [streamed.push([], final=True)]
        joined = np.concatenate(pieces)
        whole = resample(signal, 16000, 24000)
        self.assertEqual(len(joined), len(whole))
        self.assertLess(float(np.max(np.abs(joined[200:-200] - whole[200:-200]))), 2e-3)

    def test_the_endpointer_commits_after_a_pause_and_at_the_longest_turn(self):
        from gamst.providers.streaming_asr import Endpointer
        point = Endpointer(min_seconds=.5, soft_max_seconds=5, max_seconds=6)
        cuts, sent = [], 0
        for frame in frames(5, loud=[(1, 2.5)]):
            cuts += [sent + cut for cut in point.push(frame.data)]
            sent += len(frame.data)
        self.assertEqual(len(cuts), 1)
        self.assertTrue(2.9 <= cuts[0] / 16000 <= 3.2, cuts)
        point, cuts, sent = Endpointer(min_seconds=.5, soft_max_seconds=5, max_seconds=6), [], 0
        for frame in frames(13, loud=[(0, 13)]):
            cuts += [sent + cut for cut in point.push(frame.data)]
            sent += len(frame.data)
        self.assertEqual([round(cut / 16000) for cut in cuts], [6, 12])


class _Stand:
    """A local WebSocket server; ``handler(ws)`` speaks one protocol."""

    def __init__(self, handler):
        self.handler, self.headers = handler, []

    async def __aenter__(self):
        from websockets.asyncio.server import serve

        async def handle(ws):
            self.headers.append(dict(ws.request.headers))
            await self.handler(ws)
        self.server = await serve(handle, '127.0.0.1', 0)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *_):
        self.server.close()
        await self.server.wait_closed()


class ScribeStandIn:
    """The default ``audio.asr`` candidate (Scribe v3 Live) answered by a local stand-in for ElevenLabs.

    A WebSocket server in its own thread speaks the documented realtime protocol; the registry endpoint of the
    candidate points at it for the test (a local endpoint needs no opt-in and no key), so a recipe run uses its real
    ``stream-words`` part: the session, the endpointer, the capture clock. Only the recogniser is a stand-in.

    ``hear()`` is called once per session and returns ``word(index, samples)``: what is said in the index-th block of
    ``block_s`` seconds of the session's audio, as ``(text, start_s, end_s)`` within the block, or None. Each commit
    is answered with the words of the blocks that begin before it (word times in seconds of the session's audio);
    ``turns`` keeps the words of each answer that had any.
    """
    CANDIDATE = 'elevenlabs.scribe_v3_live.hosted'

    def __init__(self, hear, block_s=1.):
        self.hear, self.block_s, self.queries, self.turns = hear, float(block_s), [], []

    def start(self, test):
        import threading
        from unittest import mock
        from websockets.asyncio.server import serve
        from gamst.crew.registry import ROLES
        self.loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        thread.start()

        async def opened():
            return await serve(self._session, '127.0.0.1', 0)
        self.server = asyncio.run_coroutine_threadsafe(opened(), self.loop).result(10)
        url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1/speech-to-text/realtime"
        patcher = mock.patch.dict(ROLES['audio.asr'].candidates[self.CANDIDATE].config, endpoint=url)
        patcher.start()
        test.addCleanup(patcher.stop)

        def stop():
            self.server.close()
            asyncio.run_coroutine_threadsafe(self.server.wait_closed(), self.loop).result(10)
            self.loop.call_soon_threadsafe(self.loop.stop)
            thread.join(10)
            self.loop.close()
        test.addCleanup(stop)
        return self

    async def _session(self, ws):
        self.queries.append(ws.request.path)
        word, audio, told = self.hear(), bytearray(), 0
        size = round(16000 * self.block_s)
        await ws.send(json.dumps(dict(message_type='session_started', session_id=f's{len(self.queries)}', config={})))
        async for raw in ws:
            message = json.loads(raw)
            audio += base64.b64decode(message.get('audio_base_64') or '')
            if not message.get('commit'):
                continue
            samples = np.frombuffer(bytes(audio), '<i2').astype(np.float32) / 32768
            words = []
            while told * size < len(samples):
                said = word(told, samples[told * size:(told + 1) * size])
                if said:
                    text, start, end = said
                    words.append(dict(text=text, start=told * self.block_s + start, end=told * self.block_s + end,
                                      type='word', logprob=0.))
                told += 1
            text = ' '.join(item['text'] for item in words)
            if words:
                self.turns.append([item['text'] for item in words])
            await ws.send(json.dumps(dict(message_type='committed_transcript', text=text)))
            await ws.send(json.dumps(dict(message_type='committed_transcript_with_timestamps', text=text, words=words)))


async def feed(module, items, *, finish=True):
    out = []
    for frame in items:
        out += await module.process(frame) or []
    if finish:
        out += await module.finish() or []
    await module.close()
    return out


@unittest.skipIf(np is None or websockets is None, 'numpy and websockets')
class ElevenLabsStandInTest(unittest.TestCase):
    def test_late_word_times_are_capture_times_and_the_key_stays_in_the_header(self):
        from gamst.providers.streaming_asr import StreamingWords
        from gamst.providers.streaming_hosted import session
        seen = dict(query=None, samples=0, chunks=0)

        async def scribe(ws):
            seen['query'] = ws.request.path
            await ws.send(json.dumps(dict(message_type='session_started', session_id='s', config={})))
            async for raw in ws:
                message = json.loads(raw)
                seen['samples'] += len(base64.b64decode(message['audio_base_64'])) // 2
                seen['chunks'] += 1
                if message['commit'] and seen.get('answered'):
                    await ws.send(json.dumps(dict(message_type='committed_transcript_with_timestamps', text='', words=[])))
                elif message['commit']:
                    seen['answered'] = True
                    await asyncio.sleep(.3)    # a slow answer must not move the words
                    words = [dict(text='뽀또', start=1.2, end=1.6, type='word', logprob=0.),
                             dict(text=' ', start=1.6, end=1.7, type='spacing', logprob=0.),
                             dict(text='주세요', start=1.7, end=2.1, type='word', logprob=0.)]
                    await ws.send(json.dumps(dict(message_type='committed_transcript', text='뽀또 주세요')))
                    await ws.send(json.dumps(dict(message_type='committed_transcript_with_timestamps',
                                                  text='뽀또 주세요', words=words)))

        async def run():
            async with _Stand(scribe) as stand:
                import os
                os.environ['GAMST_TEST_EL_KEY'] = 'k-123'
                chosen = session(dict(api='elevenlabs', endpoint=stand.url + '/v1/speech-to-text/realtime',
                                      commit_strategy='manual', api_key_env='GAMST_TEST_EL_KEY', terms=['뽀또']))
                module = StreamingWords(chosen, language='ko')
                out = await feed(module, frames(4, loud=[(1, 2.2)]))
                os.environ.pop('GAMST_TEST_EL_KEY')
                return out, stand.headers
        out, headers = asyncio.run(run())
        finals = [value for value in out if value['status'] == 'final']
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]['text'], '뽀또 주세요')
        self.assertEqual([(w['text'], w['start_ms'], w['end_ms']) for w in finals[0]['words']],
                         [('뽀또', BASE + 1200, BASE + 1600), ('주세요', BASE + 1700, BASE + 2100)])
        self.assertGreaterEqual(finals[0]['final_after_ms'], 250)
        self.assertEqual(headers[0]['xi-api-key'], 'k-123')
        self.assertNotIn('k-123', seen['query'])
        self.assertIn('include_timestamps=true', seen['query'])
        self.assertIn('keyterms=', seen['query'])
        self.assertEqual(seen['samples'], 4 * 16000)

    def test_a_hosted_endpoint_needs_the_owners_opt_in_and_wss(self):
        from gamst.providers.streaming_hosted import check_config, cost_line
        with self.assertRaises(PermissionError):
            check_config(dict(api='openai'))
        with self.assertRaisesRegex(ValueError, 'wss'):
            check_config(dict(api='openai', allow_remote=True, endpoint='ws://api.example.com/x'))
        with self.assertRaisesRegex(ValueError, 'environment'):
            check_config(dict(api='elevenlabs', allow_remote=True, api_key='secret'))
        with self.assertRaisesRegex(ValueError, 'line break'):
            check_config(dict(api='openai', allow_remote=True, terms=['a\nb']))
        line = cost_line(dict(api='elevenlabs', terms=['뽀또']), 3600)
        self.assertIn('api.elevenlabs.io', line)
        self.assertIn('$0.27 per hour', line)
        self.assertIn('silence too', line)


class _Aligner:
    model_id = 'stand-in-aligner'

    def __init__(self, items):
        self.items, self.calls = items, []

    async def align(self, samples, rate, text, language):
        self.calls.append((len(samples), text, language))
        return self.items(text)


@unittest.skipIf(np is None or websockets is None, 'numpy and websockets')
class OpenAIStandInTest(unittest.TestCase):
    async def _run(self, aligner):
        from gamst.providers.streaming_asr import Endpointer, StreamingWords
        from gamst.providers.streaming_hosted import session
        got = dict(update=None, samples=0, commits=0)

        async def realtime(ws):
            items = []
            async for raw in ws:
                message = json.loads(raw)
                if message['type'] == 'session.update':
                    got['update'] = message
                    await ws.send(json.dumps(dict(type='session.updated', session={})))
                elif message['type'] == 'input_audio_buffer.append':
                    got['samples'] += len(base64.b64decode(message['audio'])) // 2
                elif message['type'] == 'input_audio_buffer.commit':
                    got['commits'] += 1
                    item = f'item_{got["commits"]}'
                    items.append(item)
                    await ws.send(json.dumps(dict(type='input_audio_buffer.committed', item_id=item)))
                    await ws.send(json.dumps(dict(type='conversation.item.input_audio_transcription.delta',
                                                  item_id=item, content_index=0, delta='뽀또')))
                    answers = (((items[1], '두 번째 뽀또'), (items[0], '뽀또 하나')) if len(items) == 2 else
                               ((item, ''),) if len(items) > 2 else ())   # out of order: matched by item_id
                    for done, text in answers:
                        await ws.send(json.dumps(dict(
                            type='conversation.item.input_audio_transcription.completed', item_id=done,
                            content_index=0, transcript=text)))

        async with _Stand(realtime) as stand:
            chosen = session(dict(api='openai', endpoint=stand.url + '/v1/realtime?intent=transcription',
                                  terms=['뽀또'], delay='low'))
            module = StreamingWords(chosen, aligner=aligner, language='ko', partials=True,
                                    endpointer=Endpointer(min_seconds=.5, soft_max_seconds=5, max_seconds=6))
            out = await feed(module, frames(6, loud=[(1, 2), (3.5, 4.6)]))
        return out, got

    def test_the_text_is_the_apis_and_the_aligner_only_adds_times(self):
        def items(text):
            return [dict(text=word, start_s=.1 + .3 * index, end_s=.3 + .3 * index)
                    for index, word in enumerate(text.split())]
        aligner = _Aligner(items)
        out, got = asyncio.run(self._run(aligner))
        transcription = got['update']['session']['audio']['input']['transcription']
        self.assertEqual(transcription, dict(model='gpt-live-transcribe', languages=['ko'], keywords=['뽀또'], delay='low'))
        self.assertIsNone(got['update']['session']['audio']['input']['turn_detection'])
        self.assertEqual(got['samples'], 6 * 24000)
        finals = [value for value in out if value['status'] == 'final']
        self.assertEqual([value['text'] for value in finals], ['뽀또 하나', '두 번째 뽀또'])   # commit order
        self.assertTrue(all(value['aligned'] for value in finals))
        self.assertEqual(sorted(call[1] for call in aligner.calls), ['두 번째 뽀또', '뽀또 하나'])
        self.assertEqual(finals[0]['words'][0]['start_ms'], BASE + 100)   # the turn starts at the first sample
        self.assertEqual(finals[0]['start_ms'], BASE + 100)
        self.assertGreater(finals[1]['words'][0]['start_ms'], BASE + 2500)   # the second turn's own audio
        self.assertTrue(any(value['status'] == 'partial' and value['text'] == '뽀또' for value in out))

    def test_an_aligner_that_changes_a_word_leaves_the_text_without_times(self):
        aligner = _Aligner(lambda text: [dict(text='보또', start_s=.1, end_s=.4)])
        out, _ = asyncio.run(self._run(aligner))
        finals = [value for value in out if value['status'] == 'final']
        self.assertEqual([value['text'] for value in finals], ['뽀또 하나', '두 번째 뽀또'])
        self.assertTrue(all(value['words'] == [] and not value['aligned'] and 'align_error' in value
                            for value in finals))


@unittest.skipIf(np is None, 'numpy')
class LocalStreamingTest(unittest.TestCase):
    def test_sherpa_tokens_become_words_on_the_stream_clock(self):
        from gamst.providers.streaming_local import SherpaStreaming

        class Result:
            start_time, tokens, timestamps = 10., ['▁뽀', '또', '▁주', '세요'], [.4, .52, .9, 1.04]
        words = SherpaStreaming.words(Result, 12.)
        self.assertEqual([word for word, _, _ in words], ['뽀또', '주세요'])
        self.assertAlmostEqual(words[0][1], 10.4)
        self.assertAlmostEqual(words[0][2], 10.9)
        self.assertAlmostEqual(words[1][1], 10.9)

    def test_voxtral_tokens_are_80_ms_apart_and_a_pause_ends_the_turn(self):
        from gamst.providers.streaming_local import VoxtralStreaming
        script = {5: 33, 6: 1001, 7: 1002, 9: 33, 10: 1003}     # token index -> id; everything else is padding

        class Tokenizer:
            @staticmethod
            def token_bytes(token):
                return {1001: '뽀'.encode(), 1002: '또'.encode(), 1003: ' 네'.encode()}.get(token, b'')

        class Session:
            def __init__(self):
                self.generated, self.fed, self.done = [], 0, False

            def feed(self, samples):
                self.fed += len(samples)

            def close(self):
                self.closing = True

            def step(self, max_decode_tokens=4):
                while len(self.generated) < self.fed // 1280:
                    self.generated.append(script.get(len(self.generated), 32))
                if getattr(self, 'closing', False):
                    self.done = True

        class Model:
            _tokenizer = Tokenizer()

            def create_streaming_session(self, transcription_delay_ms=None):
                return Session()

        module = VoxtralStreaming(Model(), 'stand-in', pause_seconds=.6)

        async def run():
            out = []
            for frame in frames(3):
                out += await module.process(frame) or []
            return out + (await module.finish() or [])
        values = [item['value'] for item in asyncio.run(run())]
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['text'], '뽀또 네')
        self.assertEqual([(w['text'], w['start_ms'] - BASE, w['end_ms'] - BASE) for w in values[0]['words']],
                         [('뽀또', 400, 640), ('네', 720, 880)])
        self.assertEqual(values[0]['finalization_reason'], 'pause')


class RecipeTest(unittest.TestCase):
    def test_asr_picks_the_part_that_builds_the_candidate(self):
        from gamst.recipes import asr_part, load, with_asr
        self.assertEqual(asr_part('openai.gpt_live_transcribe.hosted'), 'stream-words')
        self.assertEqual(asr_part('elevenlabs.scribe_v2_realtime.hosted'), 'stream-words')
        self.assertEqual(asr_part('sherpa.zipformer_ko_streaming'), 'stream-words')
        self.assertEqual(asr_part('voxtral.mini_4b_realtime_2602.mlx'), 'stream-words')
        self.assertEqual(asr_part('faster_whisper.large_v3_turbo'), 'whisper-words')
        self.assertEqual(asr_part('elevenlabs.scribe_v2.hosted'), 'hosted-words')
        recipe = with_asr(load('person-term-count', exclude=['voice']), 'elevenlabs.scribe_v2_realtime.hosted')
        asr = next(item for item in recipe['modules'] if item['name'] == 'asr')
        self.assertEqual((asr['use'], asr['candidate']), ('stream-words', 'elevenlabs.scribe_v2_realtime.hosted'))
        with self.assertRaises(SystemExit):
            asr_part('nothing.like.this')

    def test_hosted_streaming_candidates_never_send_by_default(self):
        from gamst.crew.registry import ROLES
        for name in ('openai.gpt_live_transcribe.hosted', 'elevenlabs.scribe_v2_realtime.hosted',
                     'elevenlabs.scribe_v3_live.hosted'):
            candidate = ROLES['audio.asr'].candidates[name]
            # the default is set up like any default (only its WebSocket client; no weights); the others by hand
            self.assertEqual(candidate.automatic, name == 'elevenlabs.scribe_v3_live.hosted')
            self.assertEqual(candidate.weights, ())
            self.assertIs(candidate.config['allow_remote'], False)
            self.assertNotIn('api_key', candidate.config)
        for name in ('sherpa.zipformer_ko_streaming', 'voxtral.mini_4b_realtime_2602.mlx'):
            candidate = ROLES['audio.asr'].candidates[name]
            self.assertEqual(candidate.provider_profile, 'streaming.asr.words')
            self.assertTrue(all(weight.sha256 and '/resolve/' in weight.url for weight in candidate.weights))

    def test_the_default_recogniser_is_streaming_and_hosted_only_with_the_owners_consent(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock
        from gamst import recipes
        from gamst.crew.registry import ROLES, TIERS
        default = 'elevenlabs.scribe_v3_live.hosted'
        self.assertEqual(set(ROLES['audio.asr'].defaults.values()), {default})
        self.assertEqual(set(ROLES['audio.asr'].defaults), set(TIERS))
        for name in ('person-term-count', 'person-voice-style-count', 'scene'):
            asr = next(item for item in json.loads((recipes.RECIPES / f'{name}.json').read_text())['modules']
                       if item['name'] == 'asr')
            self.assertEqual((asr['use'], asr['candidate']), ('stream-words', default), name)
        profile = Path(tempfile.mkdtemp()) / 'none.json'
        run = ['run', 'person-term-count', '--audio', str(profile.parent / 'missing.wav'), '--without', 'voice',
               '--terms', '대박', '--headless', '--profile', str(profile)]
        with mock.patch.dict(os.environ), self.assertRaises(SystemExit) as caught:
            os.environ.pop('ELEVENLABS_API_KEY', None)
            recipes.main(run)
        text = str(caught.exception)       # said before any input opens (the file does not exist)
        for part in (default, 'leaves this device', '--allow-remote-asr', '--asr sherpa.zipformer_ko_streaming',
                     '--asr voxtral.mini_4b_realtime_2602.mlx', 'gamst recipe setup person-term-count --asr'):
            self.assertIn(part, text)
        with mock.patch.dict(os.environ), self.assertRaises(SystemExit) as caught:
            os.environ.pop('ELEVENLABS_API_KEY', None)
            recipes.main(run + ['--allow-remote-asr'])
        self.assertNotIn('--allow-remote-asr', str(caught.exception))       # consented: on to the next problem
        with self.assertRaisesRegex(SystemExit, '--allow-remote-asr'):     # words by example are transcribed too
            recipes.main(['enroll', 'person-term-count', '--kind', 'terms', '--audio', str(profile.parent / 'a.wav'),
                          '--proposals', str(profile.parent / 'p.json'), '--profile', str(profile)])

    def test_a_replayed_recording_arrives_at_its_own_speed(self):
        from gamst.inputs import Paced

        class Source:
            modality, live = 'audio', False

            async def frames(self):
                for index in range(5):
                    yield InputFrame(None, 'audio', 1000 + 100 * index)

        async def run():
            start, seen = time.monotonic(), []
            async for _ in Paced(Source(), speed=4).frames():
                seen.append(time.monotonic() - start)
            return seen
        seen = asyncio.run(run())
        self.assertAlmostEqual(seen[-1], .1, delta=.05)    # 400 ms of capture at 4x
        self.assertFalse(Paced(Source()).live)


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(np is None, 'numpy')
class AlignerTransportTest(unittest.TestCase):
    def test_a_long_segment_reaches_the_aligner_whole(self):
        from gamst.providers.model_workers import RemoteAligner, _Aligner
        seen = {}

        class Aligner:
            def align(self, samples, rate, text, language):
                seen['samples'] = samples
                return [dict(text=text, start_s=0., end_s=len(samples) / rate)]

        worker = _Aligner(Aligner())

        class Worker:
            async def process_regions(self, frame, request):
                from gamst.provider_wire import array_size
                array_size(frame.data.dtype.str, list(frame.data.shape))     # what the wire would refuse
                return worker.process_regions(frame, request)

        client = RemoteAligner.__new__(RemoteAligner)
        client.worker, client._sequence = Worker(), 0
        samples = np.arange(16000 * 20 + 7, dtype=np.float32)
        items = asyncio.run(client.align(samples, 16000, '뽀또', 'ko'))
        self.assertTrue(np.array_equal(seen['samples'], samples))
        self.assertAlmostEqual(items[0]['end_s'], 20 + 7 / 16000)
