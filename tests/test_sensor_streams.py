import asyncio
from dataclasses import replace
import threading
import unittest

from humansense import HumanSense, SensorStreams, CapabilitySpec, OutputField, SensorMetadata
from humansense.plugins import InputFrame
from humansense.schema import timestamp


def spec(name='imu.motion', modality='imu'):
    return CapabilitySpec(name, name, modality, 'test.provider', '1',
        (OutputField('count', 'integer'), OutputField('value', 'object')),
        'native', stateful=True, reset_on_gap=True)


class Counter:
    def __init__(self):
        self.count = self.resets = 0
        self.closed = False

    def process(self, frame):
        self.count += 1
        return {'count': self.count, 'value': frame.data}

    def reset(self):
        self.count = 0
        self.resets += 1

    def close(self):
        self.closed = True


class Streams(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)
        self.runtime = SensorStreams(self.human, capacity=1)
        self.instances = []
        def factory():
            instance = Counter()
            self.instances.append(instance)
            return instance
        self.runtime.register(spec(), factory)

    async def asyncTearDown(self):
        await self.human.close()

    def frame(self, source='s1', sequence=0, data=None):
        return InputFrame(data if data is not None else {'x': 1}, 'imu', timestamp(),
                          sensor=SensorMetadata(source, 'imu', sequence))

    async def test_explicit_loading_and_independent_state(self):
        self.assertEqual(self.instances, [])
        for source in ('s1', 's2'):
            await self.runtime.open(source, 'imu', ['imu.motion'])
        await self.runtime.submit(self.frame('s1', 0))
        await self.runtime.drain()
        await self.runtime.submit(self.frame('s1', 1))
        await self.runtime.submit(self.frame('s2', 0))
        await self.runtime.drain()
        values = (await self.human.get_context())['observations'].values()
        self.assertEqual(sorted(v['value']['count'] for v in values), [1, 2])
        self.assertEqual(self.runtime.buffer_bytes, 0)

    async def test_gap_resets_only_affected_provider(self):
        for source in ('s1', 's2'):
            await self.runtime.open(source, 'imu', ['imu.motion'])
            await self.runtime.submit(self.frame(source, 0))
        await self.runtime.drain()
        await self.runtime.submit(self.frame('s1', 5))
        await self.runtime.drain()
        self.assertEqual([i.resets for i in self.instances], [1, 0])

    async def test_slow_sync_provider_does_not_block_other_sensor(self):
        entered, release = threading.Event(), threading.Event()
        class Slow(Counter):
            def process(self, frame):
                entered.set()
                release.wait(3)
                return super().process(frame)
        self.runtime.register(spec('imu.slow'), Slow)
        await self.runtime.open('slow', 'imu', ['imu.slow'])
        await self.runtime.open('fast', 'imu', ['imu.motion'])
        try:
            await self.runtime.submit(self.frame('slow'))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await self.runtime.submit(self.frame('fast'))
            for _ in range(100):
                if self.instances[0].count:
                    break
                await asyncio.sleep(.005)
            self.assertEqual(self.instances[0].count, 1)
        finally:
            release.set()
            await self.runtime.drain()

    async def test_overflow_drops_oldest_and_resets(self):
        await self.runtime.open('s1', 'imu', ['imu.motion'])
        await self.runtime.submit(self.frame(sequence=0))
        await self.runtime.submit(self.frame(sequence=1))
        await self.runtime.submit(self.frame(sequence=2))
        await self.runtime.drain()
        stats = self.runtime.status()['streams'][0]['providers'][0]
        self.assertEqual(stats['dropped'], 2)
        self.assertEqual(stats['processed'], 1)
        self.assertEqual(stats['resets'], 1)

    async def test_clear_reconnect_and_pause_invalidate_input(self):
        await self.runtime.open('s1', 'imu', ['imu.motion'])
        await self.runtime.submit(self.frame())
        await self.runtime.clear('s1')
        self.assertEqual(self.runtime.status()['streams'], [])
        self.assertTrue(self.instances[0].closed)
        self.assertEqual(self.runtime.buffer_bytes, 0)
        await self.runtime.open('s1', 'imu', ['imu.motion'])
        await self.runtime.submit(self.frame())
        await self.human.pause()
        with self.assertRaises(PermissionError if not self.human.consent else RuntimeError):
            await self.runtime.submit(self.frame(sequence=1))
        self.assertEqual(self.runtime.buffer_bytes, 0)

    async def test_rejects_shared_provider_instance(self):
        shared = Counter()
        self.runtime.register(spec('imu.shared'), lambda: shared)
        await self.runtime.open('s1', 'imu', ['imu.shared'])
        with self.assertRaises(ValueError):
            await self.runtime.open('s2', 'imu', ['imu.shared'])
        self.assertFalse(shared.closed)

    async def test_input_copy_and_sequence_validation(self):
        await self.runtime.open('s1', 'imu', ['imu.motion'])
        data = {'x': 1}
        await self.runtime.submit(self.frame(data=data))
        data['x'] = 100
        await self.runtime.drain()
        output = next(iter((await self.human.get_context())['observations'].values()))
        self.assertEqual(output['value']['value'], {'x': 1})
        with self.assertRaises(ValueError):
            await self.runtime.submit(self.frame())
        with self.assertRaises(ValueError):
            await self.runtime.submit(replace(self.frame(sequence=1), subject_id='someone-else'))

    async def test_audio_channels_reach_provider_unchanged(self):
        observed = []
        class Audio(Counter):
            def process(self, frame):
                observed.append((frame.data, frame.sensor.channels))
                return {'count': 1, 'value': {}}
        self.runtime.register(spec('audio.test', 'audio'), Audio)
        await self.runtime.open('array', 'mic', ['audio.test'])
        await self.runtime.submit(InputFrame([[.1, .2], [.3, .4]], 'audio', timestamp(),
            sample_rate=16000, sensor=SensorMetadata('array', 'mic', 0, channels=('left', 'right'))))
        await self.runtime.drain()
        self.assertEqual(observed, [([[.1, .2], [.3, .4]], ('left', 'right'))])

    async def test_clear_waits_for_sync_inference_and_discards_its_result(self):
        entered, release = threading.Event(), threading.Event()
        instance = Counter()
        original = instance.process
        def process(frame):
            entered.set()
            release.wait(3)
            self.assertFalse(instance.closed)
            return original(frame)
        instance.process = process
        self.runtime.register(spec('imu.blocking'), lambda: instance)
        await self.runtime.open('s1', 'imu', ['imu.blocking'])
        await self.runtime.submit(self.frame())
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        clearing = asyncio.create_task(self.runtime.clear())
        try:
            await asyncio.sleep(.01)
            self.assertFalse(clearing.done())
        finally:
            release.set()
            await clearing
        self.assertTrue(instance.closed)
        self.assertEqual((await self.human.get_context())['observations'], {})
        self.assertEqual(self.runtime.buffer_bytes, 0)

    async def test_cancelled_sync_factory_closes_constructed_provider(self):
        entered, release = threading.Event(), threading.Event()
        instance = Counter()
        def factory():
            entered.set()
            release.wait(3)
            return instance
        self.runtime.register(spec('imu.loading'), factory)
        opening = asyncio.create_task(self.runtime.open('s1', 'imu', ['imu.loading']))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        opening.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await opening
        self.assertTrue(instance.closed)
        self.assertEqual(self.runtime.status()['streams'], [])

    async def test_budget_rejection_does_not_consume_sequence(self):
        runtime = SensorStreams(self.human, max_packet_bytes=10, max_buffer_bytes=10)
        runtime.register(spec('imu.small'), Counter)
        await runtime.open('s1', 'imu', ['imu.small'])
        with self.assertRaises(ValueError):
            await runtime.submit(self.frame(data={'long': 'value'}))
        await runtime.submit(self.frame(data={}))
        await runtime.drain()
        self.assertEqual(runtime.buffer_bytes, 0)
