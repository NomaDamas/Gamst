"""Fixture transport evidence only: no SAM installation or trained inference."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
import struct
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.schema import timestamp
from humansense.separation_media import (PROFILE, PROTOCOL, MAX_PAIR_BYTES,
    SeparationPair, capability, frame_binding, pack_pair, unpack_pair)

try:
    import numpy as np
except ImportError:
    np = None


def observation():
    return dict(separation=dict(status='fixture-estimate', quality_validated=False,
        additive_reconstruction_verified=False, ambient_preserved=True),
        window={}, prompt={}, model={'fixture_engine': True}, media={})


def request(n=48000):
    payload = struct.pack('<f', .25) * n
    sensor = dict(source_id='mic', sensor_id='device', sequence=0,
        channels=['mono'], coordinate_frame=None, calibration_id=None,
        stream_epoch='epoch', capture_sequence=None)
    header = dict(metadata=dict(modality='audio', sample_rate=48000,
        timestamp_ms=10000, subject_id='alice', sensor=sensor),
        layout=dict(kind='ndarray', dtype='<f4', shape=[n]))
    return header, payload


def sample_pair(n=48000):
    return SeparationPair(observation(), struct.pack('<f', .125)*n,
                          struct.pack('<f', -.75)*n, n+1)


class PairWire(unittest.TestCase):
    def setUp(self):
        self.spec = capability({'fixture': True})
        self.binding = frame_binding(*request(), generation=0, config_id=self.spec.version)
        self.pair = sample_pair()

    def test_closed_roundtrip_and_binding(self):
        obs, header, binary = pack_pair(self.pair, self.binding, self.spec)
        pair = unpack_pair(obs, header, binary, self.binding, self.spec)
        self.assertEqual(pair, self.pair)
        self.assertEqual(header['binding']['input_sha256'], hashlib.sha256(request()[1]).hexdigest())
        self.assertEqual(header['binding']['capture_start_ms'], 9000)
        self.assertLessEqual(len(binary), MAX_PAIR_BYTES)

    def test_invalid_layout_hash_padding_and_binding(self):
        obs, header, binary = pack_pair(self.pair, self.binding, self.spec)
        variants = []
        for key, value in [('encoding', '<f8'), ('protocol', 'generic-media'),
                           ('padded_samples', True), ('padded_samples', 47999),
                           ('padded_samples', 96001), ('extra', 1)]:
            changed = deepcopy(header); changed[key] = value; variants.append(changed)
        for key, value in [('offset', 1), ('bytes', True), ('sha256', '0'*64), ('name', 'ambient')]:
            changed = deepcopy(header); changed['components'][0][key] = value; variants.append(changed)
        changed = deepcopy(header); changed['components'].reverse(); variants.append(changed)
        changed = deepcopy(header); changed['components'].append(changed['components'][0]); variants.append(changed)
        changed = deepcopy(header); changed['binding']['generation'] = False; variants.append(changed)
        changed = deepcopy(header); changed['binding']['sensor']['source_id'] = 'other'; variants.append(changed)
        for changed in variants:
            with self.subTest(header=changed), self.assertRaises(ValueError):
                unpack_pair(obs, changed, binary, self.binding, self.spec)
        for changed in (binary[:-1], binary+b'x', b''):
            with self.assertRaises(ValueError):
                unpack_pair(obs, header, changed, self.binding, self.spec)

    def test_malformed_native_pair_and_waveforms(self):
        for value in (float('nan'), float('inf'), -17., 17.):
            bad = replace(self.pair, target=struct.pack('<f', value)*48000)
            with self.assertRaises(ValueError): pack_pair(bad, self.binding, self.spec)
        for bad in (None, replace(self.pair, target=b'bad'),
                    replace(self.pair, target=bytearray(self.pair.target)),
                    replace(self.pair, observation={**observation(), 'media': {'reference': 'forged'}})):
            with self.assertRaises(ValueError): pack_pair(bad, self.binding, self.spec)
        good = replace(self.pair, target=struct.pack('<f', 2.)*48000)
        self.assertEqual(pack_pair(good, self.binding, self.spec)[2][:4], struct.pack('<f', 2.))

    def test_input_bounds_before_model_work(self):
        for n in (47999, 192001):
            with self.assertRaises(ValueError): frame_binding(*request(n), generation=0, config_id='test')
        for key, value in [('sample_rate', True), ('sample_rate', 16000), ('timestamp_ms', 2)]:
            head, pcm = request(); head['metadata'][key] = value
            with self.assertRaises(ValueError): frame_binding(head, pcm, generation=0, config_id='test')
        for value in (float('nan'), float('inf'), 1.1):
            head, pcm = request()
            with self.assertRaises(ValueError):
                frame_binding(head, struct.pack('<f', value)*48000, generation=0, config_id='test')
        head, pcm = request(); head['metadata']['sensor']['stream_epoch'] = None
        with self.assertRaises(ValueError): frame_binding(head, pcm, generation=0, config_id='test')

    def test_closed_metadata_rejects_raw_arrays_and_unknown_nested_fields(self):
        variants = []
        for field in ('separation', 'window', 'prompt', 'model', 'media'):
            value = observation(); value[field]['target_pcm'] = [.25]*100; variants.append(value)
        value = observation(); value['prompt']['token_count_after'] = [.25]*100; variants.append(value)
        value = observation(); value['model']['fixture_engine'] = {'nested': [.25]*100}; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError): self.spec.validate(value)
            with self.assertRaises(ValueError): pack_pair(replace(self.pair, observation=value), self.binding, self.spec)
        obs, header, binary = pack_pair(self.pair, self.binding, self.spec)
        obs['separation']['target_pcm'] = [.25]*100
        with self.assertRaises(ValueError): unpack_pair(obs, header, binary, self.binding, self.spec)


class Retention(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)
        self.owner = self.human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device')
        self.spec = capability({})
        self.binding = frame_binding(*request(), generation=0, config_id=self.spec.version)
        self.pair = sample_pair()

    async def asyncTearDown(self):
        await self.human.close()

    def admit(self, owner=None):
        return (owner or self.owner)._admit(self.pair, self.binding)

    async def test_exact_owner_epoch_and_immutable_read(self):
        meta = self.admit(); ref = meta['reference']
        self.assertEqual(self.owner.retrieve(ref, 'target', stream_epoch='epoch'), self.pair.target)
        self.assertIs(type(self.owner.retrieve(ref, 'target', stream_epoch='epoch')), bytes)
        for epoch in ('other', None):
            with self.assertRaises(PermissionError): self.owner.retrieve(ref, 'target', stream_epoch=epoch)
        for subject, source in [('bob', 'mic'), ('alice', 'other'), ('alice', 'mic')]:
            other = self.human.enable_separation_media(subject_id=subject, source_id=source, sensor_id='device')
            with self.assertRaises(PermissionError): other.retrieve(ref, 'target', stream_epoch='epoch')
        copied = self.owner.lookup(ref, stream_epoch='epoch'); copied['components']['target']['sha256'] = 'bad'
        self.assertNotEqual(self.owner.lookup(ref, stream_epoch='epoch'), copied)

    async def test_global_capacity_evicts_atomic_pairs(self):
        refs = [self.admit()['reference'] for _ in range(9)]
        self.assertEqual(len(self.owner._store.entries), 8)
        self.assertEqual(self.owner._store.bytes, len(self.pair.target)*16)
        with self.assertRaises(PermissionError): self.owner.retrieve(refs[0], 'residual', stream_epoch='epoch')
        self.assertEqual(self.owner.retrieve(refs[-1], 'residual', stream_epoch='epoch'), self.pair.residual)

    async def test_global_byte_budget_evicts_pairs_before_count_limit(self):
        async with HumanSense(consent=True) as human:
            owner = human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device',
                capacity=32, max_bytes=MAX_PAIR_BYTES)
            refs = [owner._admit(self.pair, self.binding)['reference'] for _ in range(5)]
            self.assertEqual((len(owner._store.entries), owner._store.bytes), (4, MAX_PAIR_BYTES))
            with self.assertRaises(PermissionError): owner.retrieve(refs[0], 'target', stream_epoch='epoch')
            self.assertEqual(owner.retrieve(refs[-1], 'residual', stream_epoch='epoch'), self.pair.residual)

    async def test_ttl_timer_releases_bytes_without_reads(self):
        with patch('humansense.separation_media.time.monotonic', return_value=100): meta = self.admit()
        store = self.owner._store
        with patch('humansense.separation_media.time.monotonic', return_value=130): store._prune()
        self.assertEqual((store.bytes, len(store.entries), store.timer), (0, 0, None))
        with self.assertRaises(PermissionError): self.owner.retrieve(meta['reference'], 'target', stream_epoch='epoch')

    async def test_actual_expiry_callback_runs_without_lookup(self):
        async with HumanSense(consent=True) as human:
            owner = human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device', ttl_ms=10)
            owner._admit(self.pair, self.binding)
            for _ in range(100):
                if not owner._store.entries: break
                await asyncio.sleep(.005)
            self.assertEqual((owner._store.bytes, owner._store.timer), (0, None))

    async def test_inflight_budget_is_shared_and_kept_until_release(self):
        store = self.owner._store
        other = self.human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device')
        self.owner._reserve(192000)
        other._reserve(192000)
        third = self.human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device')
        with self.assertRaises(ValueError): third._reserve(192000)
        self.owner.close()
        self.assertIn(self.owner, store.reservations)
        self.owner._release()
        third._reserve(192000)
        other._release(); third._release()
        self.assertEqual(store.reservations, {})

    async def test_generation_consent_pause_delete_reset_and_close_revoke(self):
        for action in ('pause', 'delete_data', 'reset_processing', 'consent', 'close'):
            human = HumanSense(consent=True)
            owner = human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device')
            meta = self.admit(owner)
            if action == 'consent': human.consent = False
            else: await getattr(human, action)()
            self.assertEqual((owner._store.bytes, len(owner._store.entries)), (0, 0))
            with self.assertRaises((PermissionError, RuntimeError)):
                owner.retrieve(meta['reference'], 'target', stream_epoch='epoch')
            await human.close()

    async def test_reset_and_owner_close_revoke_prior_references(self):
        meta = self.admit(); self.owner._reset()
        with self.assertRaises(PermissionError): self.owner.retrieve(meta['reference'], 'target', stream_epoch='epoch')
        self.owner._reset_complete()
        changed = deepcopy(self.binding); changed['sensor']['stream_epoch'] = 'new'; changed['owner_revision'] = 1
        meta = self.owner._admit(self.pair, changed)
        self.assertEqual(self.owner.retrieve(meta['reference'], 'target', stream_epoch='new'), self.pair.target)
        self.owner.close()
        with self.assertRaises(PermissionError): self.owner.retrieve(meta['reference'], 'target', stream_epoch='new')

    async def test_opt_in_limits_and_subject_binding(self):
        async with HumanSense() as human:
            self.assertIsNone(human._separation_media)
            with self.assertRaises(PermissionError): human.enable_separation_media(source_id='mic', sensor_id='device')
        for options in ({'ttl_ms': True}, {'ttl_ms': 60001}, {'capacity': 33}, {'max_bytes': MAX_PAIR_BYTES-1}):
            async with HumanSense(consent=True) as human:
                with self.assertRaises(ValueError): human.enable_separation_media(source_id='mic', sensor_id='device', **options)
        with self.assertRaises(ValueError):
            self.human.enable_separation_media(source_id='mic', sensor_id='device', ttl_ms=1000)
        changed = deepcopy(self.binding); changed['subject_id'] = 'bob'
        with self.assertRaises(PermissionError): self.owner._admit(self.pair, changed)


FIXTURE = '''
import time
from humansense import provider_worker as w
from humansense.separation_media import SeparationPair
class Fixture:
    def __init__(self, config): self.config=config
    def process(self, frame):
        time.sleep(self.config.get('delay', 0))
        n=len(frame.data)
        observation=dict(separation=dict(status='fixture-estimate',quality_validated=False),
            window=dict(sample_count=n),prompt={},model=dict(fixture_engine=True),media={})
        return SeparationPair(observation,(frame.data*.5).astype('<f4').tobytes(),
            (frame.data*-.25).astype('<f4').tobytes(),n)
    def reset(self): pass
    def finish(self): return None
    def close(self): pass
w._construct=lambda profile,config:(Fixture(config),w.capability(profile,config))
w.main()
'''


@unittest.skipIf(np is None, 'optional numpy unavailable')
class OwnedPipes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)
        self.children = []
        self.real_spawn = asyncio.create_subprocess_exec
        self.providers = []

    async def asyncTearDown(self):
        for provider in self.providers: await provider.close()
        await self.human.close()
        leaked = [child for child in self.children if child.returncode is None]
        for child in leaked: child.kill(); await child.wait()
        self.assertEqual(leaked, [])

    def fixture(self, code=FIXTURE):
        async def spawn(*args, **kwargs):
            child = await self.real_spawn(sys.executable, '-u', '-c', code, **kwargs)
            self.children.append(child)
            return child
        return patch('asyncio.create_subprocess_exec', spawn)

    async def open(self, config=None, **kwargs):
        owner = self.human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device')
        provider = await ProcessProvider.open(sys.executable, PROFILE, config or {}, separation_owner=owner, **kwargs)
        self.providers.append(provider)
        return provider, owner

    def frame(self, sequence=0, epoch='epoch'):
        return InputFrame(np.full(48000, .25, dtype='<f4'), 'audio', timestamp(), 'alice', 48000,
            SensorMetadata('mic', 'device', sequence, channels=('mono',), stream_epoch=epoch))

    async def test_actual_child_pair_retrieval_and_ambient_preservation(self):
        with self.fixture():
            provider, owner = await self.open()
            frame = self.frame(); original = frame.data.copy()
            result = await provider.process(frame)
            np.testing.assert_array_equal(frame.data, original)
            ref = result['media']['reference']
            self.assertEqual(owner.retrieve(ref, 'target', stream_epoch='epoch'), (original*.5).tobytes())
            self.assertEqual(owner.retrieve(ref, 'residual', stream_epoch='epoch'), (original*-.25).tobytes())
            self.assertIsNone(await provider.finish())
            await provider.reset()
            with self.assertRaises(PermissionError): owner.retrieve(ref, 'target', stream_epoch='epoch')
            fresh = await provider.process(self.frame(1, 'new'))
            self.assertEqual(owner.lookup(fresh['media']['reference'], stream_epoch='new')['sample_count'], 48000)

    async def test_offline_flags_are_set_before_worker_imports_without_parent_changes(self):
        code = FIXTURE.replace('import time', "import time, os\nassert os.environ['HF_HUB_OFFLINE']=='1'\nassert os.environ['TRANSFORMERS_OFFLINE']=='1'")
        before = dict(os.environ)
        with self.fixture(code):
            provider, owner = await self.open()
            await provider.close()
        self.assertEqual(dict(os.environ), before)

    async def test_public_context_and_mcp_contain_metadata_only(self):
        from humansense.mcp import MCPServer
        with self.fixture():
            provider, owner = await self.open()
            streams = SensorStreams(self.human)
            streams.register(provider.spec, lambda: provider)
            await streams.open('mic', 'device', [provider.spec.id], subject_id='alice')
            frame = self.frame(epoch=None)
            await streams.submit(frame); await streams.drain()
            context = await self.human.get_context(subject_id='alice')
            value = next(iter(context['observations'].values()))['value']
            self.assertTrue(value['model']['fixture_engine'])
            epoch = streams.status()['streams'][0]['providers'][0]['stream_epoch']
            ref = value['media']['reference']
            self.assertEqual(len(owner.retrieve(ref, 'target', stream_epoch=epoch)), 48000*4)
            server = MCPServer(self.human)
            response = await server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': 'get_context', 'arguments': {'subject_id': 'alice'}}})
            serialized = json.dumps([context, response, await self.human.get_history(subject_id='alice')])
            self.assertIn(ref, serialized)
            self.assertNotIn('base64', serialized)
            self.assertNotIn('waveform', serialized)
            self.assertNotIn('"target": [', serialized)
            await streams.clear()
            with self.assertRaises(PermissionError): owner.retrieve(ref, 'target', stream_epoch=epoch)

    async def test_native_and_malicious_pipe_pcm_metadata_never_reaches_context_or_mcp(self):
        from humansense.mcp import MCPServer
        code_paths = [FIXTURE.replace("window=dict(sample_count=n)",
            "window=dict(sample_count=n,target_pcm=frame.data[:100].tolist())"),
            FIXTURE.replace('w.main()', '''
original=w.envelope
def corrupt(h,p=b''):
    if 'separation' in h: h['result']['separation']['target_pcm']=[.25]*100
    return original(h,p)
w.envelope=corrupt
w.main()
''')]
        for code in code_paths:
            with self.fixture(code):
                provider, owner = await self.open()
                streams = SensorStreams(self.human)
                # Prior iteration registered the same canonical contract. Use a
                # new manager; its factory never starts native inference itself.
                streams.register(provider.spec, lambda: provider)
                await streams.open('mic','device',[provider.spec.id],subject_id='alice')
                await streams.submit(self.frame(epoch=None)); await streams.drain()
                self.assertEqual((await self.human.get_context(subject_id='alice'))['observations'], {})
                self.assertEqual(owner._store.bytes, 0)
                response = await MCPServer(self.human).handle({'jsonrpc':'2.0','id':1,'method':'tools/call',
                    'params': {'name':'get_context','arguments':{'subject_id':'alice'}}})
                self.assertNotIn('target_pcm', json.dumps(response))
                await streams.close()

    async def test_direct_registered_observation_cannot_bypass_closed_metadata(self):
        from humansense.mcp import MCPServer
        spec = capability({})
        self.human.register_capability(spec)
        value = observation(); value['separation']['target_pcm'] = self.frame().data[:100].tolist()
        with self.assertRaises(ValueError):
            await self.human.push_observation(spec.id, value, sensor=self.frame().sensor, subject_id='alice')
        self.assertEqual((await self.human.get_context(subject_id='alice'))['observations'], {})
        response = await MCPServer(self.human).handle({'jsonrpc':'2.0','id':1,'method':'tools/call',
            'params': {'name':'get_observations','arguments':{'subject_id':'alice'}}})
        self.assertNotIn('target_pcm', json.dumps(response))

    async def test_network_context_subscription_is_metadata_only(self):
        from humansense.mcp import MCPServer
        from humansense.server import handle_connection
        from test_server import Socket
        with self.fixture():
            provider, owner = await self.open()
            frame = self.frame()
            result = await provider.process(frame)
            self.human.register_capability(provider.spec)
            await self.human.push_observation(provider.spec.id, result, sensor=frame.sensor,
                subject_id='alice', timestamp_ms=frame.timestamp_ms)
            server, socket = MCPServer(self.human), Socket()
            task = asyncio.create_task(handle_connection(server, socket))
            try:
                await socket.incoming.put({'jsonrpc': '2.0', 'id': 1,
                    'method': 'humansense/subscribe', 'params': dict(types=['context'], interval_ms=50, subject_id='alice')})
                messages = [await asyncio.wait_for(socket.outgoing.get(), 1) for _ in range(2)]
                context = next(x['params'] for x in messages if x.get('method', '').endswith('/context'))
                ack = next(x for x in messages if x.get('id') == 1)
                self.assertEqual(ack['result']['context']['interval_ms'], 50)
                value = next(iter(context['observations'].values()))['value']
                self.assertEqual(value, result)
                self.assertNotIn('base64', json.dumps(context))
                # No public method is installed for reference-based waveform access.
                await socket.incoming.put({'jsonrpc':'2.0','id':2,'method':'humansense/separation/retrieve',
                    'params':{'reference': result['media']['reference']}})
                while True:
                    response = await asyncio.wait_for(socket.outgoing.get(), 1)
                    if response.get('id') == 2: break
                self.assertIn('error', response)
            finally:
                await socket.incoming.put(None); await task

    async def test_slow_separation_does_not_block_independent_sensor_or_context(self):
        from humansense import CapabilitySpec, OutputField
        done = asyncio.Event()
        class Fast:
            async def process(self, frame): done.set(); return {'value': 1}
            def close(self): pass
        with self.fixture():
            provider, owner = await self.open({'delay': .2})
            streams = SensorStreams(self.human)
            streams.register(provider.spec, lambda: provider)
            spec = CapabilitySpec('imu.fixture', 'imu.fixture', 'imu', 'fixture', '1',
                (OutputField('value', 'number'),), 'frame')
            streams.register(spec, Fast)
            await streams.open('mic', 'device', [provider.spec.id], subject_id='alice')
            await streams.open('imu', 'imu', [spec.id], subject_id='alice')
            await streams.submit(self.frame(epoch=None))
            await streams.submit(InputFrame({}, 'imu', timestamp(), 'alice',
                sensor=SensorMetadata('imu', 'imu', 0)))
            await asyncio.wait_for(done.wait(), .1)
            await asyncio.wait_for(self.human.get_context(subject_id='alice'), .1)
            await streams.drain()
            self.assertEqual(len((await self.human.get_context(subject_id='alice'))['observations']), 2)

    async def test_gap_reset_revokes_prior_pair_and_rebinds_new_epoch(self):
        with self.fixture():
            provider, owner = await self.open()
            streams = SensorStreams(self.human)
            streams.register(provider.spec, lambda: provider)
            await streams.open('mic', 'device', [provider.spec.id], subject_id='alice')
            await streams.submit(self.frame(epoch=None)); await streams.drain()
            first = next(iter((await self.human.get_context(subject_id='alice'))['observations'].values()))
            await streams.submit(self.frame(sequence=3, epoch=None)); await streams.drain()
            latest = next(iter((await self.human.get_context(subject_id='alice'))['observations'].values()))
            self.assertNotEqual(first['sensor']['stream_epoch'], latest['sensor']['stream_epoch'])
            with self.assertRaises(PermissionError):
                owner.retrieve(first['value']['media']['reference'], 'target', stream_epoch=first['sensor']['stream_epoch'])
            self.assertEqual(len(owner.retrieve(latest['value']['media']['reference'], 'target',
                stream_epoch=latest['sensor']['stream_epoch'])), 48000*4)

    async def test_timeout_and_cancellation_reap_and_remove_retention(self):
        for cancel in (False, True):
            with self.fixture():
                provider, owner = await self.open({'delay': .2}, inference_timeout=.02 if not cancel else 5)
                task = asyncio.create_task(provider.process(self.frame()))
                if cancel:
                    await asyncio.sleep(.01); task.cancel()
                with self.assertRaises(asyncio.CancelledError if cancel else asyncio.TimeoutError): await task
                self.assertTrue(provider.closed)
                self.assertEqual((owner._store.bytes, owner._store.reservations), (0, {}))
                self.assertIsNotNone(self.children[-1].returncode)

    async def test_clear_racing_completion_prevents_late_admission(self):
        with self.fixture():
            provider, owner = await self.open({'delay': .05})
            task = asyncio.create_task(provider.process(self.frame()))
            await asyncio.sleep(.01)
            await self.human.reset_processing()
            with self.assertRaises((PermissionError, RuntimeError)): await task
            self.assertEqual((owner._store.bytes, owner._store.reservations), (0, {}))
            self.assertTrue(provider.closed)

    async def test_close_interrupts_stalled_media_request_without_late_admission(self):
        with self.fixture():
            provider, owner = await self.open({'delay': 60}, inference_timeout=600)
            task = asyncio.create_task(provider.process(self.frame()))
            while provider._sequence < 2: await asyncio.sleep(0)
            await asyncio.wait_for(provider.close(), 3)
            with self.assertRaises((RuntimeError, asyncio.IncompleteReadError, ConnectionError)):
                await task
            self.assertIsNotNone(self.children[-1].returncode)
            self.assertEqual((owner._store.bytes, owner._store.reservations), (0, {}))

    async def test_reset_intent_suppresses_old_response_and_allows_new_epoch(self):
        with self.fixture():
            provider, owner = await self.open({'delay': .05})
            process = asyncio.create_task(provider.process(self.frame()))
            while provider._sequence < 2: await asyncio.sleep(0)
            reset = asyncio.create_task(provider.reset())
            while not owner.reset_pending: await asyncio.sleep(0)
            self.assertIsNone(await process)
            await reset
            self.assertEqual((owner._store.bytes, len(owner._store.entries)), (0, 0))
            fresh = await provider.process(self.frame(1, 'new'))
            self.assertEqual(owner.lookup(fresh['media']['reference'], stream_epoch='new')['sample_count'], 48000)

    async def test_process_queued_before_reset_intent_cannot_rebind_old_epoch(self):
        with self.fixture():
            provider, owner = await self.open({'delay': .05})
            active = asyncio.create_task(provider.process(self.frame()))
            while provider._sequence < 2: await asyncio.sleep(0)
            queued = asyncio.create_task(provider.process(self.frame(1)))
            await asyncio.sleep(0)
            reset = asyncio.create_task(provider.reset())
            while not owner.reset_pending: await asyncio.sleep(0)
            self.assertIsNone(await active)
            with self.assertRaises(PermissionError): await queued
            await reset
            self.assertEqual(owner._store.bytes, 0)
            fresh = await provider.process(self.frame(2, 'new'))
            self.assertEqual(len(owner.retrieve(fresh['media']['reference'], 'target', stream_epoch='new')), 192000)

    async def test_process_started_during_reset_cannot_queue_and_rebind_old_epoch(self):
        with self.fixture():
            provider, owner = await self.open({'delay': .05})
            active = asyncio.create_task(provider.process(self.frame()))
            while provider._sequence < 2: await asyncio.sleep(0)
            reset = asyncio.create_task(provider.reset())
            while not owner.reset_pending: await asyncio.sleep(0)
            # This starts after reset intent, while the active exchange still
            # owns the pipe. It must fail admission before awaiting that lock.
            with self.assertRaises(PermissionError):
                await asyncio.wait_for(provider.process(self.frame(1)), .02)
            self.assertIsNone(await active)
            await reset
            self.assertEqual((owner._store.bytes, owner.epoch), (0, None))
            fresh = await provider.process(self.frame(2, 'new'))
            self.assertEqual(len(owner.retrieve(fresh['media']['reference'], 'target', stream_epoch='new')), 192000)

    async def test_cancel_reset_while_waiting_reaps_inflight_worker(self):
        with self.fixture():
            provider, owner = await self.open({'delay': 60}, inference_timeout=600)
            process = asyncio.create_task(provider.process(self.frame()))
            while provider._sequence < 2: await asyncio.sleep(0)
            reset = asyncio.create_task(provider.reset())
            while not owner.reset_pending: await asyncio.sleep(0)
            reset.cancel()
            with self.assertRaises(asyncio.CancelledError): await reset
            with self.assertRaises((RuntimeError, asyncio.IncompleteReadError, ConnectionError)): await process
            self.assertTrue(provider.closed)
            self.assertIsNotNone(self.children[-1].returncode)
            self.assertEqual((owner._store.bytes, owner._store.reservations), (0, {}))

    async def test_malicious_child_response_fails_before_admission(self):
        corruption = ["h['separation']['binding']['subject_id']='bob'",
                      "h['separation']['components'][0]['offset']=1",
                      "p=p+b'x'", "p=p[:-4]", "h.pop('separation')", "h['id']+=1"]
        for mutation in corruption:
            code = FIXTURE.replace('w.main()', '''
original=w.envelope
def corrupt(h,p=b''):
    if 'separation' in h:
        MUTATION
    return original(h,p)
w.envelope=corrupt
w.main()
'''.replace('MUTATION', mutation))
            with self.fixture(code):
                provider, owner = await self.open()
                with self.assertRaises((ValueError, RuntimeError)): await provider.process(self.frame())
                self.assertTrue(provider.closed)
                self.assertEqual(owner._store.bytes, 0)

    async def test_excessive_declared_response_rejected_before_body_read(self):
        code = FIXTURE.replace('w.main()', '''
import struct
original=w.envelope
def corrupt(h,p=b''):
    if 'separation' in h: return struct.pack('!II',20,1536001)
    return original(h,p)
w.envelope=corrupt
w.main()
''')
        with self.fixture(code):
            provider, owner = await self.open(inference_timeout=1)
            with self.assertRaisesRegex(ValueError, 'exceeds private wire bounds'):
                await provider.process(self.frame())
            self.assertTrue(provider.closed)

    async def test_nonmedia_profiles_cannot_opt_in_and_native_is_unavailable(self):
        owner = self.human.enable_separation_media(subject_id='alice', source_id='mic', sensor_id='device')
        with self.assertRaises(ValueError):
            await ProcessProvider.open(sys.executable, 'audio.prosody', {}, separation_owner=owner)
        with self.assertRaises(PermissionError): await ProcessProvider.open(sys.executable, PROFILE, {})
        # Production path has no fake/native substitution and must fail closed.
        with self.assertRaises(RuntimeError):
            await ProcessProvider.open(sys.executable, PROFILE, {}, separation_owner=owner)
        self.assertTrue(owner.closed)

    async def test_legacy_json_worker_still_rejects_binary_reply(self):
        from test_process_provider import FAKE_WORKER, frame
        code = FAKE_WORKER.replace('w.main()', '''
original=w.envelope
def corrupt(h,p=b''):
    if 'result' in h and isinstance(h['result'],dict) and 'features' in h['result']:
        p=b'not-allowed'
    return original(h,p)
w.envelope=corrupt
w.main()
''')
        with self.fixture(code):
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            self.providers.append(provider)
            with self.assertRaisesRegex(ValueError, 'cannot contain raw'): await provider.process(frame())
            self.assertTrue(provider.closed)

