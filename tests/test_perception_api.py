import asyncio
import json
import unittest

from humansense import HumanSense, CapabilitySpec, OutputField, SensorMetadata, SensorStreams, transcript_capability
from humansense.mcp import MCPServer
from humansense.schema import timestamp
from humansense.server import handle_connection

try:
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
except ImportError:
    serve = connect = None


class PerceptionAPI(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_and_status_do_not_activate_provider(self):
        async with HumanSense(consent=False) as human:
            server = MCPServer(human)
            spec = CapabilitySpec('sensor.test', 'sensor.test', 'imu', 'fixture', '1',
                (OutputField('value', 'number'),), 'native')
            calls = []
            streams = SensorStreams(human)
            streams.register(spec, lambda: calls.append('loaded'))
            sdk = await human.list_capabilities()
            wire = await server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': 'list_capabilities', 'arguments': {}}})
            self.assertFalse(wire['result']['isError'])
            self.assertEqual(json.loads(wire['result']['content'][0]['text']), json.loads(json.dumps(sdk)))
            status = await server.call_tool('get_sensor_status')
            self.assertEqual(status['managers'][0]['registered_capabilities'], [spec.id])
            self.assertEqual(status['managers'][0]['streams'], [])
            self.assertEqual(calls, [])
            self.assertFalse(human.consent)
            self.assertFalse(hasattr(human, '_media_router'))

    async def test_capability_pages_are_bounded_and_revision_checked(self):
        async with HumanSense(consent=False) as human:
            for name in ('z', 'a', 'm'):
                human.register_capability(CapabilitySpec(name, name, 'imu', 'fixture', '1',
                    (OutputField('value', 'number'),), 'native'))
            server = MCPServer(human)
            first = await server.call_tool('list_capabilities', {'limit': 1})
            self.assertEqual([x['id'] for x in first['capabilities']], ['a'])
            second = await human.list_capabilities(offset=first['next_offset'],
                revision=first['revision'], limit=1)
            self.assertEqual([x['id'] for x in second['capabilities']], ['m'])
            last = await human.list_capabilities(offset=second['next_offset'], revision=first['revision'])
            self.assertEqual([x['id'] for x in last['capabilities']], ['z'])
            self.assertIsNone(last['next_offset'])
            bounded = await human.list_capabilities(max_bytes=1024)
            self.assertLessEqual(sum(len(json.dumps(x, ensure_ascii=True).encode())
                for x in bounded['capabilities']), 1024)
            for args in ({'offset': True}, {'offset': 1}, {'limit': 0}, {'max_bytes': 100},
                         {'revision': 'bad'}, {'offset': 4, 'revision': first['revision']}):
                with self.assertRaises(ValueError):
                    await human.list_capabilities(**args)
            human.register_capability(CapabilitySpec('b', 'b', 'imu', 'fixture', '1',
                (OutputField('value', 'number'),), 'native'))
            with self.assertRaisesRegex(ValueError, 'registry changed'):
                await server.call_tool('list_capabilities', {'offset': 1, 'revision': first['revision']})

    async def test_filtered_history_bounds_and_no_legacy_raw_payloads(self):
        async with HumanSense(consent=True) as human:
            server = MCPServer(human)
            spec = CapabilitySpec('a', 'a', 'environment', 'fixture', '1',
                (OutputField('value', 'string'),), 'frame')
            human.register_capability(spec)
            now = timestamp()-100
            for sequence in range(4):
                await human.push_observation('a', {'value': str(sequence)*1200},
                    sensor=SensorMetadata('src'+str(sequence % 2), 's', sequence), timestamp_ms=now+sequence)
            await human.push_signal('legacy', {'raw': 'not permitted in registered context'}, 'video')
            result = await server.call_tool('get_observations', {'source_id': 'src1', 'limit': 1})
            self.assertEqual(result['returned_count'], 1)
            self.assertTrue(result['truncated'])
            self.assertEqual(result['items'][0]['observation']['sensor']['sequence'], 3)
            self.assertEqual(result['items'][0]['observation']['capability']['id'], 'a')
            result = await human.get_observations(max_bytes=1024)
            self.assertEqual(result['items'], [])
            self.assertTrue(result['truncated'])
            self.assertEqual((await human.get_observations(capability_id='absent'))['items'], [])
            for args in ({'limit': True}, {'max_bytes': 1000000}, {'source_id': 'x'*129}, {'unknown': 1}):
                with self.assertRaises(ValueError):
                    await server.call_tool('get_observations', args)

    async def test_transcripts_and_object_evidence_match_python(self):
        async with HumanSense(consent=True) as human:
            server = MCPServer(human)
            spec = transcript_capability('asr', 'fixture', '1', streaming_mode='native')
            human.register_capability(spec)
            now = timestamp()-100
            for revision, text in enumerate(('안녕', '안녕하세요')):
                await human.push_observation('asr', dict(utterance_id='u', revision=revision, text=text,
                    status='final' if revision else 'partial', start_ms=now-100, end_ms=now,
                    language='ko-KR'), sensor=SensorMetadata('mic', 's', revision), timestamp_ms=now)
            result = await server.call_tool('get_transcripts')
            self.assertEqual(result, await human.get_transcripts())
            self.assertEqual([v['update']['text'] for v in result['utterances']], ['안녕하세요'])
            objects = CapabilitySpec('objects', 'objects', 'video', 'fixture', '1',
                (OutputField('objects', 'array'),), 'frame')
            human.register_capability(objects)
            record = await human.push_observation(objects.id, {'objects': [{'track_id': 't1', 'label': 'cup'}]},
                sensor=SensorMetadata('camera', 's', 0), timestamp_ms=now)
            await human.remember_objects(record['id'])
            memory = await server.call_tool('get_object_memory', {'text': 'cup'})
            self.assertEqual(memory, await human.get_object_memory(text='cup'))
            self.assertIsNone(memory['items'][0]['current_location'])
            self.assertEqual(memory['items'][0]['last_observation']['record_id'], record['id'])


class Socket:
    def __init__(self):
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.release = asyncio.Event()
        self.block = False
        self.context_sends = 0
    def __aiter__(self):
        return self
    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)
    async def send(self, raw):
        message = json.loads(raw)
        if message.get('method') == 'notifications/humansense/context':
            self.context_sends += 1
            if self.block:
                await self.release.wait()
        await self.outgoing.put(message)


class ContextSubscriptions(unittest.IsolatedAsyncioTestCase):
    async def receive_id(self, socket, ident):
        while True:
            message = await asyncio.wait_for(socket.outgoing.get(), 2)
            if message.get('id') == ident:
                return message

    async def test_cadence_subject_reconfiguration_and_invalid_update(self):
        async with HumanSense(consent=True) as human:
            server, socket = MCPServer(human), Socket()
            task = asyncio.create_task(handle_connection(server, socket))
            try:
                await socket.incoming.put({'jsonrpc': '2.0', 'id': 1, 'method': 'humansense/subscribe',
                    'params': {'types': ['context'], 'subject_id': 'alice', 'window': '1m', 'interval_ms': 50}})
                ack = await self.receive_id(socket, 1)
                self.assertEqual(ack['result']['context'], {'subject_id': 'alice', 'window': '1m', 'interval_ms': 50})
                messages = [await asyncio.wait_for(socket.outgoing.get(), 2) for _ in range(2)]
                self.assertTrue(all(m['params']['subject_id'] == 'alice' for m in messages))
                self.assertGreaterEqual(messages[1]['params']['timestamp_ms']-messages[0]['params']['timestamp_ms'], 45)
                for ident, params in enumerate(({'interval_ms': 0}, {'interval_ms': True}, {'window': '2d'},
                        {'subject_id': ''}, {'unexpected': 1}), 10):
                    await socket.incoming.put({'jsonrpc': '2.0', 'id': ident, 'method': 'humansense/subscribe',
                        'params': dict(types=['context'], **params)})
                    self.assertEqual((await self.receive_id(socket, ident))['error']['code'], -32602)
                # Failed reconfiguration leaves the old subscription running.
                self.assertEqual((await asyncio.wait_for(socket.outgoing.get(), 2))['params']['subject_id'], 'alice')
                await socket.incoming.put({'jsonrpc': '2.0', 'id': 30, 'method': 'humansense/subscribe',
                    'params': {'types': []}})
                await self.receive_id(socket, 30)
                self.assertEqual(human._subscribers, set())
            finally:
                await socket.incoming.put(None)
                await task

    async def test_slow_context_consumer_has_one_snapshot_in_flight_and_disconnect_cleans(self):
        async with HumanSense(consent=True) as human:
            server, socket = MCPServer(human), Socket()
            socket.block = True
            task = asyncio.create_task(handle_connection(server, socket))
            await socket.incoming.put({'jsonrpc': '2.0', 'id': 1, 'method': 'humansense/subscribe',
                'params': {'types': ['context'], 'interval_ms': 50}})
            await asyncio.sleep(.2)
            self.assertEqual(socket.context_sends, 1)
            # Runtime ingestion continues while this client's send is blocked.
            await asyncio.wait_for(human.push_text('still processing'), 1)
            socket.release.set()
            await self.receive_id(socket, 1)
            await socket.incoming.put(None)
            await asyncio.wait_for(task, 2)
            self.assertEqual(human._subscribers, set())


@unittest.skipIf(serve is None, 'optional websockets not installed')
class LoopbackPerception(unittest.IsolatedAsyncioTestCase):
    async def test_actual_websocket_tool_context_and_reconnect(self):
        async with HumanSense(consent=True) as human:
            adapter = MCPServer(human)
            spec = transcript_capability('asr', 'fixture', '1', streaming_mode='native')
            human.register_capability(spec)
            now = timestamp()-100
            await human.push_observation('asr', dict(utterance_id='u', revision=0, text='hello',
                status='final', start_ms=now-100, end_ms=now, language='en-US'),
                subject_id='alice', sensor=SensorMetadata('mic', 's', 0), timestamp_ms=now)
            async with serve(lambda socket: handle_connection(adapter, socket), '127.0.0.1', 0,
                             origins=[None], max_size=1048576, max_queue=16) as listener:
                url = 'ws://127.0.0.1:'+str(listener.sockets[0].getsockname()[1])
                for connection_number in range(2):
                    async with connect(url) as socket:
                        await socket.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                            'params': {'name': 'get_transcripts', 'arguments': {'subject_id': 'alice'}}}))
                        result = json.loads(await asyncio.wait_for(socket.recv(), 2))['result']
                        self.assertFalse(result['isError'])
                        self.assertEqual(json.loads(result['content'][0]['text'])['utterances'][0]['update']['text'], 'hello')
                        await socket.send(json.dumps({'jsonrpc': '2.0', 'id': 3,
                            'method': 'humansense/context_profile'}))
                        initial = json.loads(await asyncio.wait_for(socket.recv(), 2))['result']
                        self.assertFalse(initial['configured'])
                        self.assertEqual(initial['completed'], 0)
                        await socket.send(json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'humansense/subscribe',
                            'params': {'types': ['context'], 'interval_ms': 50, 'subject_id': 'alice'}}))
                        received = [json.loads(await asyncio.wait_for(socket.recv(), 2)) for _ in range(2)]
                        context = next(m['params'] for m in received if 'method' in m)
                        self.assertEqual(context['subject_id'], 'alice')
                        self.assertEqual(len(context['transcripts']['utterances']), 1)
                        self.assertEqual(context['transcripts']['utterances'][0]['update']['text'], 'hello')
                        await socket.send(json.dumps({'jsonrpc': '2.0', 'id': 4,
                            'method': 'humansense/context_profile'}))
                        while True:
                            status = json.loads(await asyncio.wait_for(socket.recv(), 2))
                            if status.get('id') == 4:
                                break
                        self.assertEqual(status['result']['applied_interval_ms'], 50)
                        self.assertGreaterEqual(status['result']['completed'], 1)
                        self.assertEqual(status['result']['consumer_comprehension'], 'unknown')
            self.assertEqual(human._subscribers, set())
