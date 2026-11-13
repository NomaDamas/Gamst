import asyncio
import threading
import unittest
from dataclasses import replace
from humansense import HumanSense, SensorStreams, SensorMetadata
from humansense.plugins import InputFrame
from humansense.transcripts import transcript_capability
from humansense.schema import timestamp


def spec():
    return transcript_capability('test.asr.finish','test.asr','1',streaming_mode='native')


class ASR:
    def __init__(self):
        self.count=0; self.closed=False; self.started=threading.Event(); self.release=None
        self.fail=False; self.flushes=0
    def process(self, frame):
        self.count+=1; self.end=frame.timestamp_ms
        if self.count==1: self.start=self.end
        return self.update('partial')
    def update(self,status):
        return dict(utterance_id='fixture',revision=self.count-1,text='hello' if status=='partial' else 'hello!',
                    status=status,start_ms=self.start,end_ms=self.end,language='en')
    def finish(self):
        self.flushes+=1; self.started.set()
        if self.release is not None: self.release.wait(5)
        if self.fail: raise ValueError('test finalization failure')
        self.count+=1
        return self.update('final')
    def reset(self): self.count=0
    def close(self): self.closed=True


class Finalization(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human=HumanSense(consent=True); self.streams=SensorStreams(self.human)
        self.instances=[]
        def factory():
            p=ASR(); self.instances.append(p); return p
        self.streams.register(spec(),factory)
        await self.streams.open('source','mic',[spec().id])
    async def asyncTearDown(self):
        for p in self.instances:
            if p.release is not None: p.release.set()
        await self.human.close()
    def frame(self,i=0,sensor='mic'):
        return InputFrame(b'input','audio',timestamp()-100, sample_rate=16000,
            sensor=SensorMetadata('source',sensor,i,channels=('mono',)))
    async def test_drains_final_without_new_capture_and_closes(self):
        f=self.frame(); await self.streams.submit(f)
        result=await self.streams.finish('source','mic')
        self.assertEqual(result['published'],[spec().id]); self.assertTrue(self.instances[0].closed)
        entry=(await self.human.get_transcripts())['utterances'][0]
        self.assertEqual(entry['status'],'final'); self.assertEqual(entry['sequence'],1)
        self.assertEqual(entry['updated_ms'],f.timestamp_ms)
        self.assertEqual(self.streams.buffer_bytes,0)
        self.assertEqual(self.streams.streams,{})
        with self.assertRaises(ValueError): await self.streams.submit(self.frame(1))
        await self.streams.open('source','mic',[spec().id])
        self.assertEqual(len(self.instances),2)
    async def test_cancel_waits_for_worker_and_other_stream_progresses(self):
        await self.streams.open('source','other',[spec().id])
        await self.streams.submit(self.frame()); await self.streams.drain()
        p=self.instances[0]; p.release=threading.Event()
        task=asyncio.create_task(self.streams.finish('source','mic'))
        self.assertTrue(await asyncio.to_thread(p.started.wait,2))
        with self.assertRaises(ValueError): await self.streams.submit(self.frame(1))
        await self.streams.submit(self.frame(sensor='other')); await self.streams.drain()
        task.cancel(); await asyncio.sleep(.01)
        self.assertFalse(p.closed); self.assertFalse(task.done())
        p.release.set()
        with self.assertRaises(asyncio.CancelledError): await task
        entries=(await self.human.get_transcripts())['utterances']
        self.assertEqual({e['sensor_id']:e['status'] for e in entries},{'mic':'interrupted','other':'partial'})
        self.assertTrue(p.closed)
    async def test_generation_change_rejects_late_final(self):
        await self.streams.submit(self.frame()); await self.streams.drain()
        p=self.instances[0]; p.release=threading.Event()
        task=asyncio.create_task(self.streams.finish('source','mic'))
        self.assertTrue(await asyncio.to_thread(p.started.wait,2))
        pause=asyncio.create_task(self.human.pause()); await asyncio.sleep(.01)
        p.release.set(); result=await task; await pause
        self.assertEqual(result['published'],[])
        self.assertEqual((await self.human.get_transcripts())['utterances'][0]['status'],'interrupted')
    async def test_failure_and_empty_stream_still_release_ownership(self):
        result=await self.streams.finish('source','mic')
        self.assertEqual(result['published'],[]); self.assertEqual(self.instances[0].flushes,0)
        await self.streams.open('source','mic',[spec().id])
        await self.streams.submit(self.frame()); await self.streams.drain()
        self.instances[-1].fail=True
        with self.assertRaises(RuntimeError): await self.streams.finish('source','mic')
        self.assertEqual(self.human._sensor_provider_owners,{})
        self.assertEqual((await self.human.get_transcripts())['utterances'][0]['status'],'interrupted')
