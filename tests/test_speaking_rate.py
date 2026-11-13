import json
import subprocess
import sys
import unicodedata
import unittest

from humansense import HumanSense, SensorMetadata, SensorStreams, transcript_capability
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.schema import timestamp
from humansense.speaking_rate import estimate_rate
from humansense.transcripts import TranscriptMemory, TRANSCRIPT_OUTPUT

try:
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
except ImportError:
    serve = connect = None


def update(text='We speak clearly.', language='en', start=1000, end=3000, revision=0, status='partial'):
    return dict(utterance_id='u', text=text, language=language, start_ms=start, end_ms=end,
                revision=revision, status=status)


def record(value, seq=0, subject='user', epoch='e1'):
    return dict(id=f'r{seq}', type='observation.signal', subject_id=subject, timestamp_ms=value['end_ms'],
                quality={'status': 'sufficient'}, models=[{'id': 'asr', 'version': 'fixture'}],
                observation=dict(name=TRANSCRIPT_OUTPUT, value=value, modality='audio',
                                 sensor=dict(source_id='mic', sensor_id='mono', sequence=seq, stream_epoch=epoch),
                                 capability=dict(id='asr', freshness_ms=1000), processing_epoch='runtime'))


class RateTests(unittest.TestCase):
    def test_no_optional_imports_and_english_units(self):
        code = "from humansense.speaking_rate import estimate_rate; import sys; assert not {'torch','numpy','scipy','onnxruntime'} & sys.modules.keys()"
        subprocess.run([sys.executable, '-S', '-c', code], check=True)
        result = estimate_rate(update("We're well-known—really. Café!", language='EN-us'), 'partial')
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['value'], 120.)
        self.assertEqual(result['unit'], 'orthographic_words_per_minute')
        self.assertEqual(result['status'], 'provisional')
        self.assertTrue(result['pause_time_included'])
        self.assertFalse(result['acoustic_rate_verified'])

    def test_korean_normalization_and_integer_time_differences(self):
        phrase = '안녕하세요, 반가워요!'
        for text in (phrase, unicodedata.normalize('NFD', phrase)):
            result = estimate_rate(update(text, 'ko-KR', start=2**80, end=2**80+2000, status='final'), 'final')
            self.assertEqual(result['count'], 9)
            self.assertEqual(result['value'], 4.5)
            self.assertEqual(result['unit'], 'hangul_blocks_per_second')
            self.assertEqual(result['duration_ms'], 2000)
            self.assertEqual(result['status'], 'estimated')

    def test_ambiguity_empty_short_and_interruption_are_not_zero_speech(self):
        cases = [(update('120 dollars'), 'numeric_text'),
                 (update('Ⅲ things'), 'numeric_text'),
                 (update('三個', 'ko'), 'unsupported_script_or_symbol'),
                 (update('Hello 안녕'), 'unsupported_script_or_symbol'),
                 (update('CPU가 좋아요', 'ko'), 'unsupported_script_or_symbol'),
                 (update('좋아요🙂', 'ko'), 'unsupported_script_or_symbol'),
                 (update('...'), 'no_lexical_text'), (update(''), 'no_lexical_text'),
                 (update('你好', 'zh'), 'unsupported_language'),
                 (update('Hello', 'enough'), 'unsupported_language'),
                 (update('Hello', 'en-UnitedStates'), 'unsupported_language'),
                 (update(end=1999), 'interval_under_1000ms'),
                 (update(end=1000), 'interval_under_1000ms')]
        for value, reason in cases:
            with self.subTest(value=value):
                result = estimate_rate(value, 'partial')
                self.assertEqual(result['reason'], reason)
                self.assertIsNone(result['value'])
                self.assertEqual(result['status'], 'unavailable')
        result = estimate_rate(update(), 'interrupted')
        self.assertEqual(result['reason'], 'interrupted_utterance')
        self.assertIsNone(result['value'])
        self.assertEqual(estimate_rate(update(end=2000), 'partial')['value'], 180.)

    def test_bounds_and_punctuation_conventions(self):
        for changes in ({'end_ms': True}, {'start_ms': -1}, {'end_ms': 999},
                        {'end_ms': 301001}, {'revision': -1}, {'text': 'a'*4097},
                        {'language': 'e'*33}, {'status': 'invented'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                estimate_rate(dict(update(), **changes), 'partial')
        # Apostrophes and ASCII hyphens join runs only when surrounded by letters.
        for text, count in (("can't won't", 2), ("'a' b--c d-", 4), ('hi/there', 2),
                            ('first_name', 2), ('   \t\n', 0), ('a'*4096, 1)):
            self.assertEqual(estimate_rate(update(text), 'partial')['count'], count)

    def test_revisions_retract_retry_and_snapshot_budget_include_rates(self):
        memory = TranscriptMemory()
        def put(value, seq):
            memory.commit(memory.prepare(record(value, seq), now_ms=value['end_ms']))
            return memory.snapshot('user', now_ms=value['end_ms'])['utterances'][0]
        first = put(update(), 0)
        corrected = put(update('We speak', revision=1), 1)
        self.assertEqual((first['speaking_rate']['count'], corrected['speaking_rate']['count']), (3, 2))
        self.assertEqual(corrected['speaking_rate']['value'], 60.)
        corrected['speaking_rate']['value'] = -1
        self.assertEqual(memory.snapshot('user', now_ms=3000)['utterances'][0]['speaking_rate']['value'], 60.)
        final = put(update('', revision=2, status='final'), 2)
        retry = put(update('', revision=2, status='final'), 3)
        self.assertEqual(final, retry)
        self.assertIsNone(final['speaking_rate']['value'])
        self.assertEqual(final['evidence_id'], 'r2')
        self.assertNotIn('speaking_rate', next(iter(memory.entries.values())))
        # Construct a large entry to test the exact encoded entry budget boundary.
        memory = TranscriptMemory()
        put(update('word '*500), 0)
        item = memory.snapshot('user', now_ms=3000)['utterances'][0]
        size = len(json.dumps(item, ensure_ascii=True).encode())
        self.assertEqual(len(memory.snapshot('user', now_ms=3000, max_bytes=size)['utterances']), 1)
        result = memory.snapshot('user', now_ms=3000, max_bytes=size-1)
        self.assertEqual(result['utterances'], [])
        self.assertTrue(result['truncated'])

    def test_owner_isolation_stale_final_and_clear(self):
        memory = TranscriptMemory()
        for seq, subject, epoch in ((0, 'user', 'a'), (1, 'other', 'a'), (2, 'user', 'b')):
            value = update(status='final' if seq == 0 else 'partial')
            memory.commit(memory.prepare(record(value, seq, subject, epoch), now_ms=3000))
        memory.interrupt(subject_id='user', epoch='b')
        result = memory.snapshot('user', now_ms=5000)['utterances']
        self.assertEqual([r['speaking_rate']['value'] for r in result], [90., None])
        self.assertTrue(all(not r['fresh'] for r in result))
        self.assertEqual(memory.snapshot('other', now_ms=3000)['utterances'][0]['speaking_rate']['value'], 90.)
        memory.clear('user')
        self.assertEqual(memory.snapshot('user', now_ms=3000)['utterances'], [])


class RateRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sensor_asr_revisions_to_python_context_and_mcp(self):
        # Deterministic ASR updates exercise the real stream/reducer/MCP path,
        # not actual recognition or acoustic-rate quality.
        start = timestamp()-5000
        class Backend:
            def __init__(self):
                self.revision = -1
            def process(self, frame):
                self.revision += 1
                return update('안녕하세요' if self.revision == 0 else '안녕하세요 반가워요',
                              'ko', start=start, end=frame.timestamp_ms, revision=self.revision)
            def reset(self):
                self.revision = -1
        async with HumanSense(consent=True) as human:
            spec = transcript_capability('asr.rate', 'fixture', '1', streaming_mode='native', freshness_ms=10000)
            streams = SensorStreams(human)
            streams.register(spec, Backend)
            await streams.open('mic', 'mono', [spec.id])
            server = MCPServer(human)
            for seq in range(2):
                await streams.submit(InputFrame(bytes(320), 'audio', start+2000+seq*1000,
                    sample_rate=16000, sensor=SensorMetadata('mic', 'mono', seq, channels=('mono',))))
                await streams.drain()
            entries = (await human.get_transcripts())['utterances']
            self.assertEqual(len(entries), 1)
            rate = entries[0]['speaking_rate']
            self.assertEqual((rate['count'], rate['value'], rate['revision']), (9, 3., 1))
            self.assertEqual((await human.get_context())['transcripts']['utterances'][0]['speaking_rate'], rate)
            response = await server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': 'get_transcripts', 'arguments': {}}})
            wire = json.loads(response['result']['content'][0]['text'])
            self.assertEqual(wire['utterances'][0]['speaking_rate'], rate)
            await streams.clear()
            self.assertIsNone((await human.get_transcripts())['utterances'][0]['speaking_rate']['value'])
            await human.delete_data()
            self.assertEqual((await human.get_transcripts())['utterances'], [])

    @unittest.skipIf(serve is None, 'optional websockets absent')
    async def test_actual_context_subscription_recomputes_without_model_calls(self):
        from humansense.server import handle_connection
        async with HumanSense(consent=True) as human:
            spec = transcript_capability('asr.rate', 'fixture', '1', streaming_mode='native', freshness_ms=10000)
            human.register_capability(spec)
            start = timestamp()-5000
            async def push(text, revision, status='partial'):
                await human.push_observation(spec.id, update(text, start=start, end=start+2000,
                    revision=revision, status=status), sensor=SensorMetadata('mic', 'mono', revision),
                    timestamp_ms=start+2000)
            await push('one two three four', 0)
            server = MCPServer(human)
            async with serve(lambda socket: handle_connection(server, socket), '127.0.0.1', 0,
                             origins=[None], max_size=1048576, max_queue=16) as listener:
                async with connect('ws://127.0.0.1:'+str(listener.sockets[0].getsockname()[1])) as socket:
                    await socket.send(json.dumps(dict(jsonrpc='2.0', id=1, method='humansense/subscribe',
                        params=dict(types=['context'], interval_ms=50))))
                    async def next_revision(revision):
                        for _ in range(20):
                            message = json.loads(await asyncio.wait_for(socket.recv(), 2))
                            if message.get('method') == 'notifications/humansense/context':
                                entries = message['params']['transcripts']['utterances']
                                if entries and entries[0]['update']['revision'] == revision:
                                    return entries[0]
                        self.fail('expected revised context notification')
                    first = await next_revision(0)
                    self.assertEqual(first['speaking_rate']['value'], 120.)
                    await push('one two', 1, 'final')
                    final = await next_revision(1)
                    self.assertEqual(final['speaking_rate']['value'], 60.)
                    self.assertEqual(final['speaking_rate']['status'], 'estimated')
                    self.assertNotEqual(first['evidence_id'], final['evidence_id'])
            self.assertEqual(human._subscribers, set())


if __name__ == '__main__':
    unittest.main()
import asyncio
