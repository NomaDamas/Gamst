"""Actual SDK threads/coroutines with synthetic frozen native4 source evidence."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest

from humansense import HumanSense, SensorStreams, CapabilitySpec, OutputField, SensorMetadata
from humansense.plugins import InputFrame
from benchmarks.recording_evaluation import RecordingEvaluationSession
from benchmarks.recording_native_schema import digest, run_context
from benchmarks.recording_native_writer import RecordingNativeWriter
from benchmarks.replay_clock import ReplayClock
from recording_native_fixtures import declarations, rehash


class Provider:
    config = {}
    def __init__(self):
        self.calls=[];self.resets=0;self.closed=False
    def process(self, frame):
        self.calls.append((threading.get_ident(),frame.data))
        return dict(count=len(self.calls),value={'marker':'PRIVATE provider value'})
    def reset(self):self.resets+=1
    def endpoint(self):return dict(count=20,value={})
    def finish(self):return dict(count=30,value={})
    def close(self):self.closed=True


class AsyncProvider(Provider):
    async def process(self,frame):return super().process(frame)
    async def reset(self):super().reset()
    async def endpoint(self):return super().endpoint()
    async def finish(self):return super().finish()
    async def close(self):super().close()


class Streams(unittest.IsolatedAsyncioTestCase):
    async def setup(self,factory,*,capacity=2):
        directory=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(directory.cleanup);self.root=Path(directory.name).resolve()
        d=declarations(self.root,provider='fixture.stream',revision='1',role='signal',boundary='in_process_backend')
        for arm in d['collection']['arms']:
            arm['provider_bindings'][0].update(operation_kinds=['process','endpoint','finish'],semantic_lane=None)
        rehash(d)
        self.writer=await RecordingNativeWriter.open(self.root/'native',run_context(d,'a'*64,0))
        self.observer=RecordingEvaluationSession.from_frozen(d,self.writer,run_nonce='a'*64,
            execution_ordinal=0,retain_evaluation_hashes=True)
        self.addAsyncCleanup(self.observer.close)
        self.clock=ReplayClock(d['source_manifest']);self.scope=self.observer.stream_scope(0)
        self.human=HumanSense(consent=True);self.addAsyncCleanup(self.human.close)
        self.runtime=SensorStreams(self.human,capacity=capacity)
        self.spec=CapabilitySpec('fixture.audio','fixture.audio','audio','fixture.stream','1',
            (OutputField('count','integer'),OutputField('value','object')),'native',
            stateful=True,reset_on_gap=True)
        self.runtime.register(self.spec,factory)

    async def open(self):
        await self.runtime.open('mic','mic-1',[self.spec.id],evaluation_bindings={self.spec.id:self.scope},
            evaluation_configurations={self.spec.id:{}})
        self.observer.bind_mapping(self.clock.mapping,self.clock.sha256)

    def capture(self,index=0):
        record=self.observer.declarations['source_manifest']['records'][index]
        frame=InputFrame((self.root/'pcm.bin').read_bytes(),'audio',
            self.clock.capture(record['offset_ns'])['sdk_timestamp_ms'],sample_rate=8000,
            sensor=SensorMetadata('mic','mic-1',record['sequence']))
        reference=dict(kind='capture',source_index=0,capture_ordinal=index,
            record_sha256=digest(record),mapping_sha256=self.clock.sha256)
        return frame,reference

    def rows(self):
        return [json.loads(line) for path in sorted(self.writer.output.glob('native-*.jsonl'))
            for line in path.read_bytes().splitlines() if '"event":' in line.decode()]

    async def close_native(self):
        await self.human.close();await self.observer.close()
        from benchmarks.recording_native_validation import validate_recording_native_inventory
        validate_recording_native_inventory(self.observer.declarations,self.writer.output,
            source_inventory=self.observer.declarations['source_manifest'])
        return json.loads((self.writer.output/'terminal.json').read_bytes())

    async def test_actual_sync_and_async_entry_source_refs_and_lifecycle_are_separate(self):
        main_thread=threading.get_ident()
        for asynchronous in (False,True):
            with self.subTest(asynchronous=asynchronous):
                provider=AsyncProvider() if asynchronous else Provider()
                if asynchronous:
                    async def factory():return provider
                else:
                    def factory():return provider
                await self.setup(factory);await self.open()
                for index in (0,2):
                    frame,ref=self.capture(index)
                    await self.runtime.submit(frame,evaluation_capture_ref=ref);await self.runtime.drain()
                epoch=self.runtime.status()['streams'][0]['providers'][0]['stream_epoch']
                await self.runtime.endpoint('mic','mic-1',self.spec.id,stream_epoch=epoch,after_sequence=2)
                await self.runtime.finish('mic','mic-1')
                terminal=await self.close_native();rows=self.rows()
                self.assertEqual(terminal['state'],'closed');self.assertEqual(terminal['entry_count'],4)
                self.assertEqual(terminal['unsettled_entries'],4);self.assertEqual(provider.resets,1)
                self.assertTrue(provider.closed);self.assertEqual(self.runtime.buffer_bytes,0)
                self.assertEqual([r['fields']['operation_kind'] for r in rows if r['event']=='reserved'],
                    ['process','process','endpoint','finish'])
                self.assertTrue(all(r['fields']['scope']==0 for r in rows if r['event']=='reserved'))
                kinds={r['fields']['kind'] for r in rows if r['event']=='work_started'}
                self.assertTrue({'startup','reset','residency','idle','close'}<=kinds)
                milestones=[r['fields']['milestone'] for r in rows if r['event']=='worker_milestone']
                self.assertEqual(milestones.count('thread_joined'),0 if asynchronous else 1)
                self.assertNotIn('reap_confirmed',milestones)
                self.assertTrue(all((thread==main_thread)==asynchronous for thread,_ in provider.calls))
                self.assertEqual([r['fields']['reference']['capture_ordinal'] for r in rows if r['event']=='evidence_ref'],[0,2,2,2])
                self.assertNotIn('PRIVATE',json.dumps(rows))

    async def test_configuration_and_payload_mismatch_fail_before_factory_or_queue_admission(self):
        provider=Provider();constructed=[]
        def factory():constructed.append(True);return provider
        await self.setup(factory)
        with self.assertRaises(ValueError):
            await self.runtime.open('mic','mic-1',[self.spec.id],evaluation_bindings={self.spec.id:self.scope},
                evaluation_configurations={self.spec.id:{'different':True}})
        with self.assertRaises(ValueError):
            await self.runtime.open('mic','mic-1',[self.spec.id],evaluation_bindings={self.spec.id:
                self.observer.task_scope(self.observer.declarations['tasks'][0]['task_id'],0)},
                evaluation_configurations={self.spec.id:{}})
        self.assertFalse(constructed);await self.open()
        frame,ref=self.capture()
        for candidate,locator in ((frame,None),(replace(frame,data=b'\x00'*len(frame.data)),ref),
                (frame,dict(ref,capture_ordinal=1))):
            with self.assertRaises(ValueError):await self.runtime.submit(candidate,evaluation_capture_ref=locator)
        self.assertEqual(self.runtime.buffer_bytes,0);self.assertEqual(self.observer._entries,0)
        data=bytearray(frame.data);frame=replace(frame,data=data)
        await self.runtime.submit(frame,evaluation_capture_ref=ref)
        data[:]=b'\x00'*len(data);ref['capture_ordinal']=3
        await self.runtime.drain();terminal=await self.close_native()
        self.assertEqual(provider.calls[0][1],(self.root/'pcm.bin').read_bytes())
        self.assertEqual(terminal['entry_count'],1)
        self.assertEqual([r['fields']['reference']['capture_ordinal'] for r in self.rows() if r['event']=='evidence_ref'],[0])

    async def test_pause_waits_actual_thread_discards_late_result_and_preserves_unknown_entry(self):
        entered=threading.Event();release=threading.Event()
        class Held(Provider):
            def process(self,frame):
                entered.set();release.wait(3)
                if self.closed:raise AssertionError('provider closed during actual thread')
                return super().process(frame)
        provider=Held();await self.setup(lambda:provider);await self.open()
        frame,ref=self.capture();await self.runtime.submit(frame,evaluation_capture_ref=ref)
        self.assertTrue(await asyncio.to_thread(entered.wait,1))
        paused=asyncio.create_task(self.human.pause())
        try:
            await asyncio.sleep(.02);self.assertFalse(paused.done());self.assertFalse(provider.closed)
        finally:release.set()
        await asyncio.wait_for(paused,2)
        self.assertFalse(self.human.store.query('local-user',kind='observation.signal',name=self.spec.output_name))
        terminal=await self.close_native();rows=self.rows()
        self.assertTrue(provider.closed);self.assertEqual(terminal['entry_count'],1)
        self.assertEqual(terminal['unsettled_entries'],1);self.assertEqual(terminal['state'],'closed')
        self.assertEqual([r['fields']['outcome'] for r in rows if r['event']=='published'],['discarded'])
        self.assertEqual(self.runtime.buffer_bytes,0)

    async def test_cancel_while_sync_callback_queued_releases_without_native_entry(self):
        loop=asyncio.get_running_loop();loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        provider=Provider();await self.setup(lambda:provider);await self.open();await self.writer.drain()
        entered=threading.Event();release=threading.Event()
        def hold():entered.set();release.wait(3)
        blocker=asyncio.create_task(asyncio.to_thread(hold))
        while not entered.is_set():await asyncio.sleep(.001)
        frame,ref=self.capture();await self.runtime.submit(frame,evaluation_capture_ref=ref)
        for _ in range(100):
            if self.observer._reservations==1:break
            await asyncio.sleep(.001)
        self.assertEqual(self.observer._entries,0)
        paused=asyncio.create_task(self.human.pause())
        try:await asyncio.sleep(.02)
        finally:release.set();await blocker
        await asyncio.wait_for(paused,2);terminal=await self.close_native()
        self.assertFalse(provider.calls);self.assertTrue(provider.closed)
        self.assertEqual(terminal['entry_count'],0);self.assertEqual(terminal['released_count'],1)
        self.assertEqual(terminal['unsettled_entries'],0);self.assertEqual(terminal['state'],'closed')

    async def test_overwritten_pending_capture_has_no_native_entry_or_false_coverage(self):
        provider=Provider();await self.setup(lambda:provider,capacity=1);await self.open()
        for index in range(3):
            frame,ref=self.capture(index);await self.runtime.submit(frame,evaluation_capture_ref=ref)
        await self.runtime.drain()
        stats=self.runtime.status()['streams'][0]['providers'][0]
        self.assertEqual(stats['dropped'],2);self.assertEqual(stats['processed'],1)
        terminal=await self.close_native();self.assertEqual(terminal['entry_count'],1)
        self.assertEqual([r['fields']['reference']['capture_ordinal'] for r in self.rows() if r['event']=='evidence_ref'],[2])

    async def test_repeated_factory_waiter_cancel_joins_returned_provider_and_its_close(self):
        factory_entered=threading.Event();factory_release=threading.Event()
        close_entered=threading.Event();close_release=threading.Event()
        class HeldClose(Provider):
            def close(self):close_entered.set();close_release.wait(3);super().close()
        provider=HeldClose()
        def factory():factory_entered.set();factory_release.wait(3);return provider
        await self.setup(factory)
        opened=asyncio.create_task(self.runtime.open('mic','mic-1',[self.spec.id],
            evaluation_bindings={self.spec.id:self.scope},evaluation_configurations={self.spec.id:{}}))
        try:
            self.assertTrue(await asyncio.to_thread(factory_entered.wait,1))
            opened.cancel();await asyncio.sleep(0);opened.cancel();factory_release.set()
            self.assertTrue(await asyncio.to_thread(close_entered.wait,1))
            opened.cancel();await asyncio.sleep(.01);self.assertFalse(opened.done())
        finally:factory_release.set();close_release.set()
        with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(opened,2)
        self.assertTrue(provider.closed);self.assertFalse(self.runtime.streams)
        self.assertFalse(self.human._sensor_provider_owners)
        self.observer.bind_mapping(self.clock.mapping,self.clock.sha256)
        terminal=await self.close_native();self.assertEqual(terminal['state'],'closed')
        self.assertEqual(terminal['entry_count'],0);self.assertEqual(terminal['unclosed_workers'],0)

    async def test_factory_failure_has_real_startup_work_and_no_native_or_provider_close_claim(self):
        def factory():raise ValueError('PRIVATE factory failure')
        await self.setup(factory)
        with self.assertRaises(ValueError):
            await self.runtime.open('mic','mic-1',[self.spec.id],evaluation_bindings={self.spec.id:self.scope},
                evaluation_configurations={self.spec.id:{}})
        self.observer.bind_mapping(self.clock.mapping,self.clock.sha256)
        terminal=await self.close_native();rows=self.rows()
        self.assertEqual(terminal['entry_count'],0);self.assertEqual(terminal['unclosed_workers'],0)
        marks=[r['fields']['milestone'] for r in rows if r['event']=='worker_milestone']
        self.assertIn('startup_failed',marks);self.assertIn('thread_joined',marks)
        self.assertNotIn('provider_closed',marks);self.assertNotIn('PRIVATE',json.dumps(rows))

    async def test_async_factory_suppressing_cancel_cannot_publish_open_stream_or_leak_provider(self):
        provider=AsyncProvider();entered=asyncio.Event()
        async def factory():
            entered.set()
            try:await asyncio.Event().wait()
            except asyncio.CancelledError:return provider
        await self.setup(factory)
        opened=asyncio.create_task(self.runtime.open('mic','mic-1',[self.spec.id],
            evaluation_bindings={self.spec.id:self.scope},evaluation_configurations={self.spec.id:{}}))
        await entered.wait();opened.cancel()
        with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(opened,2)
        self.assertTrue(provider.closed);self.assertFalse(self.runtime.streams)
        self.assertFalse(self.human._sensor_provider_owners)
        self.observer.bind_mapping(self.clock.mapping,self.clock.sha256)
        terminal=await self.close_native();self.assertEqual(terminal['state'],'closed')
        self.assertEqual(terminal['entry_count'],0);self.assertEqual(terminal['unclosed_workers'],0)


if __name__=='__main__':unittest.main()
