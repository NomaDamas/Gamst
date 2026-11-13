import asyncio
from dataclasses import replace
import struct
import sys
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.provider_wire import encode_frame, decode_frame, envelope, sizes, PAYLOAD_LIMIT
from humansense.schema import timestamp

try:
    import numpy as np
except ImportError:
    np = None


FAKE_WORKER = '''
import os, time
from humansense import provider_worker as w
class Provider:
    def process(self, frame):
        print('native backend diagnostic')
        os.write(1, b'low-level native stdout\\n')
        time.sleep(frame.data.get('delay', 0))
        return {'features': {'marker': frame.data.get('marker', 0)}}
    def reset(self): pass
    def close(self): pass
w._construct = lambda profile, config: (Provider(), w.capability(profile, config))
w.main()
'''


def frame(data=None, ts=1000, source='mic', sequence=0):
    return InputFrame({} if data is None else data, 'audio', ts, sample_rate=16000,
                      sensor=SensorMetadata(source, 'device', sequence))


class Wire(unittest.TestCase):
    def test_bytes_and_json_roundtrip_and_limits(self):
        for data in (b'\x00\x01'*32, {'sensor_value': 3}):
            encoded, payload = encode_frame(frame(data))
            restored = decode_frame(encoded, payload)
            self.assertEqual(restored, frame(data))
        with self.assertRaises(ValueError):
            sizes(struct.pack('!II', 20, PAYLOAD_LIMIT+1))
        with self.assertRaises(ValueError):
            envelope({'bad': float('nan')})
        with self.assertRaises(ValueError):
            decode_frame({'metadata': {}, 'layout': {'kind': 'ndarray', 'shape': [2], 'dtype': '<f8'}}, b'')

    @unittest.skipIf(np is None, 'optional numpy not installed')
    def test_array_channel_dtype_endianness_and_strides(self):
        for dtype in ('<f4', '>f8', '<i2', '|u1'):
            original = np.arange(12).reshape(6, 2).astype(dtype)[::2]
            f = replace(frame(original), sensor=SensorMetadata('mic', 'device', 0, channels=('left', 'right')))
            encoded, payload = encode_frame(f)
            restored = decode_frame(encoded, payload)
            np.testing.assert_array_equal(restored.data, original)
            self.assertEqual(restored.data.dtype.str, original.dtype.str)
            self.assertEqual(restored.sensor.channels, ('left', 'right'))
        with self.assertRaises(ValueError):
            encode_frame(frame(np.array([object()], dtype=object)))


class Processes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.children = []
        self.real_spawn = asyncio.create_subprocess_exec

    async def asyncTearDown(self):
        # Fail on leaked owned children, but never leave a failed test's worker.
        leaked = [p for p in self.children if p.returncode is None]
        for p in leaked:
            p.kill()
            await p.wait()
        self.assertEqual(leaked, [], 'provider leaked child processes')

    def fixture(self, code=FAKE_WORKER):
        async def spawn(*args, **kwargs):
            process = await self.real_spawn(sys.executable, '-u', '-c', code, **kwargs)
            self.children.append(process)
            return process
        return patch('asyncio.create_subprocess_exec', spawn)

    async def test_qwen_profile_offline_environment_is_child_local(self):
        import os
        from humansense.qwen_artifact import REQUIRED
        before = dict(os.environ)
        code = FAKE_WORKER.replace('w.main()',
            "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
            "assert os.environ['TRANSFORMERS_OFFLINE'] == '1'\n"
            "assert os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] == '0'\n"
            "w.main()")
        config = dict(path='/fixture-not-installed', files={name:'0'*64 for name in REQUIRED})
        with self.fixture(code):
            provider = await ProcessProvider.open(sys.executable, 'qwen3.asr', config)
            await provider.close()
        self.assertEqual(dict(os.environ), before)

    async def test_finish_roundtrip_and_absent_finalizer(self):
        code = FAKE_WORKER.replace('    def reset(self): pass',
            "    def finish(self): return {'features': {'final': True}}\n    def reset(self): pass")
        with self.fixture(code):
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            await provider.process(frame())
            self.assertEqual(await provider.finish(), {'features': {'final': True}})
            await provider.close()
        with self.fixture():
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            self.assertIsNone(await provider.finish())
            await provider.close()

    async def test_invalid_finish_output_and_timeout_stop_owned_child(self):
        for body, timeout, expected in (
            ("return {'features': {'raw': 'forbidden'}}", 2, RuntimeError),
            ("time.sleep(1); return None", .05, asyncio.TimeoutError)):
            code = FAKE_WORKER.replace('    def reset(self): pass',
                '    def finish(self): '+body+'\n    def reset(self): pass')
            with self.fixture(code):
                provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {}, inference_timeout=timeout)
                with self.assertRaises(expected): await provider.finish()
                self.assertTrue(provider.closed)
                self.assertIsNotNone(self.children[-1].returncode)
                await provider.close()

    async def test_parent_revalidates_successful_finish_payload(self):
        code = FAKE_WORKER.replace('    def reset(self): pass',
            "    def finish(self): return {'features': {'raw': 'forbidden'}}\n    def reset(self): pass")
        code = code.replace('w.main()',
            'from humansense.capabilities import CapabilitySpec\nCapabilitySpec.validate = lambda self, value: value\nw.main()')
        with self.fixture(code):
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            with self.assertRaises(ValueError): await provider.finish()
            self.assertTrue(provider.closed)
            self.assertIsNotNone(self.children[-1].returncode)

    async def test_finish_integrates_with_managed_stream_close(self):
        code = FAKE_WORKER.replace('    def reset(self): pass',
            "    def finish(self): return {'features': {'final': True}}\n    def reset(self): pass")
        with self.fixture(code):
            async with HumanSense(consent=True) as human:
                streams = SensorStreams(human)
                contract = ProcessProvider.capability('audio.prosody', {})
                streams.register(contract, lambda: ProcessProvider.open(sys.executable, 'audio.prosody', {}))
                await streams.open('mic','device',[contract.id])
                await streams.submit(frame(ts=timestamp()-100))
                result = await streams.finish('mic','device')
                self.assertEqual(result['published'],[contract.id])
                observation = next(iter((await human.get_context())['observations'].values()))
                self.assertTrue(observation['value']['features']['final'])
                self.assertIsNotNone(self.children[-1].returncode)

    async def test_native_stdout_is_not_protocol_and_reopen_is_explicit(self):
        with self.fixture():
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            self.assertEqual(await provider.process(frame({'marker': 7})), {'features': {'marker': 7}})
            await provider.reset()
            await provider.close()
            await provider.close()
            with self.assertRaises(RuntimeError):
                await provider.process(frame())

    async def test_timeout_reaps_child_and_never_accepts_late_output(self):
        with self.fixture():
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {}, inference_timeout=.05)
            with self.assertRaises(asyncio.TimeoutError):
                await provider.process(frame({'delay': 2}))
            self.assertTrue(provider.closed)
            self.assertIsNotNone(self.children[-1].returncode)
            with self.assertRaises(RuntimeError):
                await provider.reset()
            await provider.close()

    async def test_repeated_cancellation_collects_owned_child(self):
        with self.fixture():
            provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            request = asyncio.create_task(provider.process(frame({'delay': 5})))
            await asyncio.sleep(.03)
            request.cancel()
            await asyncio.sleep(0)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            await provider.close()
            self.assertIsNotNone(self.children[-1].returncode)

    async def test_start_timeout_and_bad_response_are_closed(self):
        for code in ('import time; time.sleep(5)', "import os; os.write(1, b'badwire!')"):
            with self.fixture(code):
                with self.assertRaises((asyncio.TimeoutError, ValueError)):
                    await ProcessProvider.open(sys.executable, 'audio.prosody', {}, startup_timeout=.1)

    async def test_cancel_during_spawn_does_not_orphan_child(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def slow_spawn(*args, **kwargs):
            child = await self.real_spawn(sys.executable, '-c', 'import time; time.sleep(5)', **kwargs)
            self.children.append(child)
            started.set()
            await release.wait()
            return child
        with patch('asyncio.create_subprocess_exec', slow_spawn):
            opening = asyncio.create_task(ProcessProvider.open(sys.executable, 'audio.prosody', {}))
            await started.wait()
            opening.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await opening

    async def test_excessive_malformed_output_closes_paused_pipe_without_gc(self):
        code = "import os; os.write(1, b'x'*10000000)"
        for _ in range(3):
            with self.fixture(code):
                with self.assertRaises(ValueError):
                    await ProcessProvider.open(sys.executable, 'audio.prosody', {})
            child = self.children[-1]
            self.assertIsNotNone(child.returncode)
            self.assertTrue(child._transport.is_closing())
            self.assertTrue(child.stdout._transport.is_closing())

    async def test_wrong_contract_and_raw_response_are_rejected(self):
        for suffix in (
            "w.capability=lambda p,c: __import__('humansense').CapabilitySpec('wrong','wrong','audio','wrong','1',(__import__('humansense').OutputField('features','object'),),'frame')\n",
            "w.envelope=lambda value: __import__('humansense.provider_wire',fromlist=['envelope']).envelope(value,b'raw')\n",
        ):
            code = FAKE_WORKER.replace('w.main()', suffix+'w.main()')
            with self.fixture(code):
                with self.assertRaises((RuntimeError, ValueError)):
                    await ProcessProvider.open(sys.executable, 'audio.prosody', {})

    async def test_stalled_sensor_does_not_block_other_worker(self):
        with self.fixture():
            async with HumanSense(consent=True) as human:
                streams = SensorStreams(human)
                spec = ProcessProvider.capability('audio.prosody', {})
                async def factory():
                    return await ProcessProvider.open(sys.executable, 'audio.prosody', {}, inference_timeout=.5)
                streams.register(spec, factory)
                await streams.open('slow', 'device', [spec.id])
                await streams.open('fast', 'device', [spec.id])
                now = timestamp()
                await streams.submit(frame({'delay': 5}, ts=now, source='slow'))
                await streams.submit(frame({'marker': 42}, ts=now, source='fast'))
                await streams.drain()
                records = human.store.latest_signals('local-user', 0, timestamp())
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['observation']['sensor']['source_id'], 'fast')
                self.assertEqual(records[0]['observation']['value']['features']['marker'], 42)

    async def test_factory_contract_mismatch_closes_worker_before_stream_opens(self):
        with self.fixture():
            async with HumanSense(consent=True) as human:
                streams = SensorStreams(human)
                spec = ProcessProvider.capability('audio.prosody', {})
                async def factory():
                    return await ProcessProvider.open(sys.executable, 'audio.prosody', {'window_ms': 40})
                streams.register(spec, factory)
                with self.assertRaisesRegex(ValueError, 'contract differs'):
                    await streams.open('mic', 'device', [spec.id])
                self.assertEqual(streams.streams, {})

    @unittest.skipIf(np is None, 'optional numpy not installed')
    async def test_real_audio_worker_preserves_samples_and_reset_state(self):
        # Actual NumPy DSP in a separate process; no model-output fixture.
        provider = await ProcessProvider.open(sys.executable, 'audio.prosody', {})
        self.children.append(provider._process)
        try:
            wave = .2*np.sin(2*np.pi*220*np.arange(1280)/16000)
            output = await provider.process(frame(wave))
            self.assertAlmostEqual(output['features']['pitch_hz'], 220, delta=1)
            await provider.reset()
            self.assertIsNone(await provider.process(frame(wave[:320])))
        finally:
            await provider.close()
