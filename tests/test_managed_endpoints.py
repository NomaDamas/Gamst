import asyncio
from dataclasses import replace
import threading
import unittest

from humansense import HumanSense, SensorStreams, SensorMetadata, transcript_capability
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.schema import timestamp


class Provider:
    def __init__(self):
        self.segment=0; self.revision=-1; self.events=[]; self.closed=False
        self.enter=threading.Event(); self.release=None; self.block_process=False
    def process(self,frame):
        self.events.append(('process',frame.sensor.sequence))
        if self.block_process:
            self.block_process=False; self.enter.set(); self.release.wait(5)
        self.end=frame.timestamp_ms
        if self.revision<0: self.start=self.end
        return self.update('partial')
    def update(self,status):
        self.revision+=1
        return dict(utterance_id=str(self.segment),revision=self.revision,text='fixture',status=status,
                    start_ms=self.start,end_ms=self.end,language='en')
    def endpoint(self):
        self.events.append(('endpoint',))
        if self.release is not None and not self.release.is_set():
            self.enter.set(); self.release.wait(5)
        if self.revision<0: return None
        value=self.update('final'); value['finalization_reason']='manual_endpoint'
        self.segment+=1; self.revision=-1
        return value
    def finish(self):
        self.events.append(('finish',))
        if self.revision<0: return None
        value=self.update('final'); value['finalization_reason']='input_end'
        self.segment+=1; self.revision=-1
        return value
    def reset(self): self.segment+=1; self.revision=-1; self.events.append(('reset',))
    def close(self): self.closed=True


class ManagedEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human=HumanSense(consent=True); self.streams=SensorStreams(self.human,capacity=1)
        self.spec=transcript_capability('fixture.asr','fixture','1',streaming_mode='native')
        self.providers=[]
        def factory():
            p=Provider(); self.providers.append(p); return p
        self.streams.register(self.spec,factory)
        await self.streams.open('source','mic',[self.spec.id])
        self.start=timestamp()-1000
    async def asyncTearDown(self):
        for p in self.providers:
            if p.release: p.release.set()
        await self.human.close()
    def frame(self,i,sensor='mic'):
        return InputFrame(b'fixture','audio',self.start+i*20,sample_rate=16000,
                          sensor=SensorMetadata('source',sensor,i))
    def args(self):
        status=self.streams.status()['streams'][0]
        return dict(source_id='source',sensor_id='mic',capability_id=self.spec.id,
                    stream_epoch=status['providers'][0]['stream_epoch'],after_sequence=status['last_sequence'])
    async def start_endpoint(self):
        task=asyncio.create_task(self.streams.endpoint(**self.args()))
        await asyncio.sleep(0)
        return task

    async def test_barrier_order_independent_output_sequence_and_mcp(self):
        p=self.providers[0]; p.release=threading.Event(); p.block_process=True
        await self.streams.submit(self.frame(0))
        self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
        task=await self.start_endpoint()
        await self.streams.submit(self.frame(1))
        p.release.set()
        self.assertTrue((await task)['published'])
        await self.streams.drain()
        self.assertEqual(p.events[:3],[('process',0),('endpoint',),('process',1)])
        entries=(await self.human.get_transcripts())['utterances']
        self.assertEqual(sorted(e['sequence'] for e in entries),[1,2])
        self.assertEqual({e['status'] for e in entries},{'partial','final'})
        args=dict(manager_id=self.streams.manager_id,**self.args())
        result=await MCPServer(self.human).call_tool('endpoint_sensor_stream',args)
        self.assertFalse(result['closed']); self.assertTrue(result['published'])
        self.assertFalse((await self.streams.endpoint(**self.args()))['published'])
        await self.streams.submit(self.frame(2)); await self.streams.drain()
        await self.streams.finish('source','mic')
        entries=(await self.human.get_transcripts())['utterances']
        self.assertEqual(sorted(e['sequence'] for e in entries),[1,3,5])
        self.assertEqual(sorted(e['capture_sequence'] for e in entries),[0,1,2])
        self.assertTrue(all(e['status']=='final' for e in entries))

    async def test_stale_controls_and_pending_bound(self):
        await self.streams.submit(self.frame(0)); await self.streams.drain()
        args=self.args()
        for bad in (dict(args,stream_epoch='old'),dict(args,after_sequence=True),
                    dict(args,after_sequence=1),dict(args,capability_id='other')):
            with self.assertRaises(ValueError): await self.streams.endpoint(**bad)
        p=self.providers[0]; p.release=threading.Event()
        task=await self.start_endpoint()
        self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
        with self.assertRaises(ValueError): await self.streams.endpoint(**args)
        p.release.set(); await task
        await self.streams.clear(); await self.streams.open('source','mic',[self.spec.id])
        await self.streams.submit(self.frame(0))
        with self.assertRaises(ValueError): await self.streams.endpoint(**args)

    async def test_cancel_suppresses_final_and_other_stream_keeps_processing(self):
        await self.streams.open('source','other',[self.spec.id])
        await self.streams.submit(self.frame(0)); await self.streams.drain()
        p=self.providers[0]; p.release=threading.Event()
        task=await self.start_endpoint()
        self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
        await self.streams.submit(self.frame(0,'other'))
        other=self.streams.streams[('source','other')]['lanes'][0]
        await asyncio.wait_for(other['queue'].join(),1)
        self.assertEqual(other['processed'],1)
        task.cancel(); await asyncio.sleep(.01)
        self.assertFalse(task.done())
        p.release.set()
        with self.assertRaises(asyncio.CancelledError): await task
        entries=(await self.human.get_transcripts())['utterances']
        self.assertEqual({e['sensor_id']:e['status'] for e in entries},{'mic':'interrupted','other':'partial'})
        await self.streams.submit(self.frame(1)); await self.streams.drain()
        self.assertIn(('reset',),p.events)

    async def test_drop_before_barrier_rejects_without_finalizing_wrong_audio(self):
        p=self.providers[0]; p.release=threading.Event(); p.block_process=True
        await self.streams.submit(self.frame(0))
        self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
        await self.streams.submit(self.frame(1))
        task=await self.start_endpoint()
        await self.streams.submit(self.frame(2))
        p.release.set()
        with self.assertRaises(RuntimeError): await task
        await self.streams.drain()
        self.assertNotIn(('endpoint',),p.events)
        self.assertEqual(self.streams.buffer_bytes,0)

    async def test_clear_and_finish_serialize_with_active_endpoint(self):
        for terminal in ('clear','finish'):
            if terminal=='finish': await self.streams.open('source','mic',[self.spec.id])
            await self.streams.submit(self.frame(0)); await self.streams.drain()
            p=self.providers[-1]; p.release=threading.Event()
            task=await self.start_endpoint()
            self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
            close=asyncio.create_task(getattr(self.streams,terminal)('source','mic'))
            await asyncio.sleep(.01)
            self.assertFalse(close.done()); self.assertNotIn(('finish',),p.events)
            p.release.set()
            if terminal=='clear':
                with self.assertRaises(RuntimeError): await task
            else: self.assertTrue((await task)['published'])
            await close
            self.assertTrue(p.closed)

    async def test_queued_cancel_never_calls_provider_and_drain_waits_control(self):
        p=self.providers[0]; p.release=threading.Event(); p.block_process=True
        await self.streams.submit(self.frame(0))
        self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
        task=await self.start_endpoint()
        task.cancel(); await asyncio.sleep(0)
        drain=asyncio.create_task(self.streams.drain())
        await asyncio.sleep(.01); self.assertFalse(drain.done())
        p.release.set()
        with self.assertRaises(asyncio.CancelledError): await task
        await drain
        self.assertNotIn(('endpoint',),p.events)
        self.assertEqual((await self.human.get_transcripts())['utterances'][0]['status'],'partial')

    async def test_delete_during_mcp_endpoint_does_not_wait_for_global_tool_lock(self):
        server=MCPServer(self.human)
        await self.streams.submit(self.frame(0)); await self.streams.drain()
        p=self.providers[0]; p.release=threading.Event()
        task=asyncio.create_task(server.call_tool('endpoint_sensor_stream',
            dict(manager_id=self.streams.manager_id,**self.args())))
        self.assertTrue(await asyncio.to_thread(p.enter.wait,2))
        generation=self.human._generation
        deletion=asyncio.create_task(server.call_tool('delete_data'))
        for _ in range(100):
            if self.human._generation != generation: break
            await asyncio.sleep(.01)
        self.assertNotEqual(self.human._generation,generation)
        p.release.set()
        with self.assertRaises(RuntimeError): await task
        await deletion
        self.assertEqual((await self.human.get_transcripts())['utterances'],[])

    async def test_output_sequence_overflow_fails_without_wrapping(self):
        f=replace(self.frame(0),sensor=SensorMetadata('source','mic',2**63-1))
        await self.streams.submit(f); await self.streams.drain()
        with self.assertRaises(RuntimeError): await self.streams.endpoint(**self.args())
        entries=(await self.human.get_transcripts())['utterances']
        self.assertEqual(entries[0]['status'],'interrupted')
        self.assertEqual(entries[0]['sequence'],2**63-1)
